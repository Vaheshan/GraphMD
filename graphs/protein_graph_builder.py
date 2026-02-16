from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data

from utils.geometry import compute_backbone_orientation, safe_normalize


@dataclass
class ProteinGraphInputs:
    """
    Container for per-complex protein inputs.

    Attributes:
        backbone_coords: Tensor of shape (R, 3, 3) with backbone atoms (N, CA, C).
        md_residue_coords: Optional tensor of shape (T, R, 3) with residue centroids
            across T MD frames for correlation edges.
        residue_indices_seq: Optional tensor of shape (R,) with sequential residue
            indices (0..R-1). If None, indices are assumed to be ordered 0..R-1.
    """

    backbone_coords: Tensor
    md_residue_coords: Optional[Tensor] = None
    residue_indices_seq: Optional[Tensor] = None


class ProteinGraphBuilder:
    """
    Build static residue-level protein graphs.

    Static edges include:
        - Sequential edges between neighboring residues along the chain.
        - Spatial edges between residues whose centroids are within a cutoff.

    Edge attributes encode:
        - Edge type as a 3D one-hot: [sequential, spatial, dynamic]
        - A scalar weight: distance for static edges.
    """

    def __init__(
        self,
        cutoff_residue: float = 8.0,
        max_neighbors: int = 32,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cutoff_residue = float(cutoff_residue)
        self.max_neighbors = int(max_neighbors)
        self.device = device

    def _compute_residue_centroids(self, backbone_coords: Tensor) -> Tensor:
        # backbone_coords: (R, 3, 3)
        return backbone_coords.mean(dim=1)

    def _build_sequential_edges(self, num_residues: int) -> Tuple[Tensor, Tensor]:
        if num_residues <= 1:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0, 4), dtype=torch.float32, device=self.device)
            return edge_index, edge_attr

        idx = torch.arange(num_residues - 1, device=self.device, dtype=torch.long)
        src = idx
        dst = idx + 1

        # Undirected edges: i <-> i+1
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
        )

        # No meaningful geometric weight beyond unit distance for sequential edges.
        # Use weight = 1.0; type one-hot = [1, 0, 0].
        num_edges = edge_index.size(1)
        edge_type = torch.tensor(
            [1.0, 0.0, 0.0], device=self.device, dtype=torch.float32
        ).repeat(num_edges, 1)
        weight = torch.ones((num_edges, 1), device=self.device, dtype=torch.float32)
        edge_attr = torch.cat([edge_type, weight], dim=-1)
        return edge_index, edge_attr

    def _build_spatial_edges(
        self, centroids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Build spatial edges based on centroid distances with per-node neighbor cap.
        """
        R = centroids.size(0)
        if R <= 1:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0, 4), dtype=torch.float32, device=self.device)
            return edge_index, edge_attr

        # Pairwise distances (R, R)
        dist = torch.cdist(centroids, centroids, p=2)
        # Mask out self-distances
        dist.fill_diagonal_(float("inf"))

        # For each residue, find neighbors within cutoff and limit to max_neighbors
        all_src = []
        all_dst = []
        all_weight = []

        for i in range(R):
            row = dist[i]  # (R,)
            # Indices of residues within cutoff
            mask = row < self.cutoff_residue
            neighbor_idx = torch.nonzero(mask, as_tuple=False).view(-1)
            if neighbor_idx.numel() == 0:
                continue
            # Sort neighbors by distance
            dists_i = row[neighbor_idx]
            sorted_dists, order = torch.sort(dists_i)
            if self.max_neighbors is not None:
                order = order[: self.max_neighbors]
                sorted_dists = sorted_dists[: self.max_neighbors]
                neighbor_idx = neighbor_idx[order]

            src_i = torch.full_like(neighbor_idx, i)
            all_src.append(src_i)
            all_dst.append(neighbor_idx)
            all_weight.append(sorted_dists)

        if not all_src:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0, 4), dtype=torch.float32, device=self.device)
            return edge_index, edge_attr

        src = torch.cat(all_src)
        dst = torch.cat(all_dst)
        weight = torch.cat(all_weight)

        # Build undirected edges
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
        )
        weight_all = torch.cat([weight, weight], dim=0).unsqueeze(-1)

        num_edges = edge_index.size(1)
        edge_type = torch.tensor(
            [0.0, 1.0, 0.0], device=self.device, dtype=torch.float32
        ).repeat(num_edges, 1)
        edge_attr = torch.cat([edge_type, weight_all], dim=-1)
        return edge_index, edge_attr

    def __call__(self, inputs: ProteinGraphInputs) -> Data:
        """
        Build a static protein residue graph for a single complex.

        Args:
            inputs: ProteinGraphInputs with backbone_coords of shape (R, 3, 3).

        Returns:
            torch_geometric.data.Data with:
                x: (R, F_res) residue features (orientation).
                pos: (R, 3) residue centroids.
                edge_index: (2, E_static).
                edge_attr: (E_static, 4) [one-hot edge type (3), weight].
        """
        backbone_coords = inputs.backbone_coords.to(self.device)
        R = backbone_coords.size(0)

        # Residue features from backbone orientation
        x = compute_backbone_orientation(backbone_coords)  # (R, 9)

        # Residue centroids (used for spatial edges)
        centroids = self._compute_residue_centroids(backbone_coords)

        # Sequential edges
        seq_edge_index, seq_edge_attr = self._build_sequential_edges(R)

        # Spatial edges
        spatial_edge_index, spatial_edge_attr = self._build_spatial_edges(centroids)

        # Concatenate edge sets
        if seq_edge_index.numel() == 0 and spatial_edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0, 4), dtype=torch.float32, device=self.device)
        elif seq_edge_index.numel() == 0:
            edge_index, edge_attr = spatial_edge_index, spatial_edge_attr
        elif spatial_edge_index.numel() == 0:
            edge_index, edge_attr = seq_edge_index, seq_edge_attr
        else:
            edge_index = torch.cat([seq_edge_index, spatial_edge_index], dim=1)
            edge_attr = torch.cat([seq_edge_attr, spatial_edge_attr], dim=0)

        data = Data(
            x=x,
            pos=centroids,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=R,
        )
        return data


class CorrelationEdgeBuilder:
    """
    Build dynamic motion-correlation edges from MD residue trajectories.

    Correlation is computed across frames over residue displacement vectors
    relative to their mean position.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.5,
        max_corr_neighbors: int = 16,
        device: Optional[torch.device] = None,
    ) -> None:
        self.correlation_threshold = float(correlation_threshold)
        self.max_corr_neighbors = int(max_corr_neighbors)
        self.device = device

    def _compute_residue_displacements(self, md_residue_coords: Tensor) -> Tensor:
        """
        md_residue_coords: (T, R, 3)
        Returns:
            displacements: (T, R, 3) zero-mean per residue.
        """
        mean_pos = md_residue_coords.mean(dim=0, keepdim=True)
        return md_residue_coords - mean_pos

    def _compute_correlation_matrix(self, displacements: Tensor) -> Tensor:
        """
        Compute a residue-residue correlation matrix from displacements.

        We flatten displacement vectors over time and xyz, z-score them per residue,
        and compute Pearson-like correlations in a vectorized way.
        """
        T, R, _ = displacements.shape
        # Flatten (T, R, 3) -> (R, T*3)
        disp_flat = displacements.permute(1, 0, 2).reshape(R, T * 3)
        # Standardize per residue
        mean = disp_flat.mean(dim=1, keepdim=True)
        std = disp_flat.std(dim=1, unbiased=False, keepdim=True)
        std_clamped = torch.clamp(std, min=1e-6)
        z = (disp_flat - mean) / std_clamped

        # Correlation matrix: (R, R)
        L = z.size(1)
        corr = (z @ z.t()) / max(L - 1, 1)
        corr = torch.clamp(corr, -1.0, 1.0)
        corr.fill_diagonal_(0.0)
        return corr

    def add_correlation_edges(
        self,
        data: Data,
        md_residue_coords: Tensor,
    ) -> Data:
        """
        Augment a static residue graph with dynamic correlation edges.

        Args:
            data: Protein residue Data with fields x, pos, edge_index, edge_attr.
            md_residue_coords: Tensor of shape (T, R, 3) with residue centroids
                across T MD frames.

        Returns:
            New Data with additional dynamic edges appended to edge_index/edge_attr.
        """
        if md_residue_coords is None:
            return data

        md_residue_coords = md_residue_coords.to(self.device)
        T, R, _ = md_residue_coords.shape
        if R != data.num_nodes:
            raise ValueError(
                f"md_residue_coords has R={R} residues but graph has num_nodes={data.num_nodes}"
            )

        displacements = self._compute_residue_displacements(md_residue_coords)
        corr = self._compute_correlation_matrix(displacements)  # (R, R)

        all_src = []
        all_dst = []
        all_weight = []

        for i in range(R):
            row = corr[i]  # (R,)
            mask = row > self.correlation_threshold
            neighbor_idx = torch.nonzero(mask, as_tuple=False).view(-1)
            if neighbor_idx.numel() == 0:
                continue

            values = row[neighbor_idx]
            # Sort by descending correlation magnitude
            values_abs = values.abs()
            sorted_vals, order = torch.sort(values_abs, descending=True)
            if self.max_corr_neighbors is not None:
                order = order[: self.max_corr_neighbors]
                sorted_vals = sorted_vals[: self.max_corr_neighbors]
                neighbor_idx = neighbor_idx[order]

            src_i = torch.full_like(neighbor_idx, i)
            all_src.append(src_i)
            all_dst.append(neighbor_idx)
            all_weight.append(sorted_vals)

        if not all_src:
            return data

        src = torch.cat(all_src)
        dst = torch.cat(all_dst)
        weight = torch.cat(all_weight)

        edge_index_corr = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
        )
        weight_all = torch.cat([weight, weight], dim=0).unsqueeze(-1)

        num_edges_corr = edge_index_corr.size(1)
        edge_type_corr = torch.tensor(
            [0.0, 0.0, 1.0], device=self.device, dtype=torch.float32
        ).repeat(num_edges_corr, 1)
        edge_attr_corr = torch.cat([edge_type_corr, weight_all], dim=-1)

        if data.edge_index is None or data.edge_index.numel() == 0:
            new_edge_index = edge_index_corr
            new_edge_attr = edge_attr_corr
        else:
            new_edge_index = torch.cat([data.edge_index.to(self.device), edge_index_corr], dim=1)
            new_edge_attr = torch.cat([data.edge_attr.to(self.device), edge_attr_corr], dim=0)

        return Data(
            x=data.x.to(self.device),
            pos=data.pos.to(self.device),
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            num_nodes=data.num_nodes,
        )

