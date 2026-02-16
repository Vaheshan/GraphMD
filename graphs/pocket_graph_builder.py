from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data

from utils.geometry import compute_atom_pair_geometric_features


@dataclass
class PocketGraphInputs:
    """
    Container for per-complex pocket/ligand graph inputs.

    Attributes:
        atom_coords: Tensor of shape (A, 3) with 3D coordinates of all atoms
            (protein pocket atoms + ligand atoms) for a given frame.
        atom_features: Tensor of shape (A, F_atom) with precomputed atom features
            (e.g., one-hot types, charges, etc.).
        atom_is_ligand: Boolean tensor of shape (A,) indicating ligand atoms.
        atom_to_residue: Optional LongTensor of shape (A,) mapping each protein
            atom to a residue index (0..R-1) or -1 for ligand/unmapped atoms.
    """

    atom_coords: Tensor
    atom_features: Tensor
    atom_is_ligand: Tensor
    atom_to_residue: Optional[Tensor] = None


class PocketGraphBuilder:
    """
    Build atom-level pocket graphs that include the ligand and nearby protein atoms.

    Edges are created between atoms whose pairwise distance is below a cutoff.
    Edge attributes are the geometric atom-pair features F_atom_pair.
    """

    def __init__(
        self,
        cutoff_atom: float = 4.5,
        max_neighbors: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cutoff_atom = float(cutoff_atom)
        self.max_neighbors = int(max_neighbors)
        self.device = device

    def _build_edges(
        self, coords: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Build undirected edges between atoms within the cutoff distance.

        Returns:
            edge_index: LongTensor of shape (2, E).
            edge_attr: FloatTensor of shape (E, F_edge) with geometric features.
        """
        A = coords.size(0)
        if A <= 1:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0, 9), dtype=torch.float32, device=self.device)
            return edge_index, edge_attr

        dist = torch.cdist(coords, coords, p=2)
        dist.fill_diagonal_(float("inf"))

        all_src = []
        all_dst = []

        for i in range(A):
            row = dist[i]
            mask = row < self.cutoff_atom
            neighbor_idx = torch.nonzero(mask, as_tuple=False).view(-1)
            if neighbor_idx.numel() == 0:
                continue

            dists_i = row[neighbor_idx]
            sorted_dists, order = torch.sort(dists_i)
            if self.max_neighbors is not None:
                order = order[: self.max_neighbors]
                neighbor_idx = neighbor_idx[order]

            src_i = torch.full_like(neighbor_idx, i)
            all_src.append(src_i)
            all_dst.append(neighbor_idx)

        if not all_src:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0, 9), dtype=torch.float32, device=self.device)
            return edge_index, edge_attr

        src = torch.cat(all_src)
        dst = torch.cat(all_dst)

        # Undirected edges
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
        )

        # Geometric features for each directed edge
        pos_i = coords[edge_index[0]]
        pos_j = coords[edge_index[1]]
        edge_attr = compute_atom_pair_geometric_features(pos_i, pos_j)
        return edge_index, edge_attr

    def __call__(self, inputs: PocketGraphInputs) -> Data:
        """
        Build a pocket atom graph for a single complex/frame.

        Args:
            inputs: PocketGraphInputs instance.

        Returns:
            torch_geometric.data.Data with:
                x: (A, F_atom) atom features.
                pos: (A, 3) atom coordinates.
                edge_index: (2, E) atom pair edges.
                edge_attr: (E, 9) geometric edge features.
                is_ligand: (A,) boolean mask.
                atom_to_residue: Optional (A,) LongTensor if provided in inputs.
        """
        coords = inputs.atom_coords.to(self.device)
        x = inputs.atom_features.to(self.device)
        is_ligand = inputs.atom_is_ligand.to(self.device)
        atom_to_residue = (
            None if inputs.atom_to_residue is None else inputs.atom_to_residue.to(self.device)
        )

        edge_index, edge_attr = self._build_edges(coords)

        data = Data(
            x=x,
            pos=coords,
            edge_index=edge_index,
            edge_attr=edge_attr,
            is_ligand=is_ligand,
            atom_to_residue=atom_to_residue,
            num_nodes=coords.size(0),
        )
        return data

