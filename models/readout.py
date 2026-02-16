from typing import Tuple

import torch
from torch import nn, Tensor


class AttentionPoolingReadout(nn.Module):
    """
    Top-K + attention pooling over selected pocket atoms.

    For each complex:
        - Select K protein atoms closest to the ligand centroid.
        - Select K ligand atoms closest to the protein pocket centroid.
        - Form set S of unique selected atoms.
        - Compute attention scores:
              s_i = w^T tanh(W h_i)
              beta_i = softmax_i(s_i)
              Z = Σ_i beta_i h_i
    """

    def __init__(
        self,
        hidden_dim: int,
        top_k: int = 16,
    ) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.vector = nn.Parameter(torch.randn(hidden_dim))

    def _select_topk_indices(
        self,
        coords: Tensor,
        batch: Tensor,
        is_ligand: Tensor,
    ) -> Tensor:
        """
        Compute indices of atoms to attend over for all complexes in the batch.
        """
        device = coords.device
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        selected_indices = []

        for b in range(B):
            mask_b = batch == b
            if not mask_b.any():
                continue

            idx_b = torch.nonzero(mask_b, as_tuple=False).view(-1)
            coords_b = coords[idx_b]
            is_ligand_b = is_ligand[idx_b]

            ligand_idx_b = idx_b[is_ligand_b]
            protein_idx_b = idx_b[~is_ligand_b]

            if ligand_idx_b.numel() == 0 or protein_idx_b.numel() == 0:
                # Fallback: use all atoms in this batch
                selected_indices.append(idx_b)
                continue

            # Ligand centroid
            ligand_coords = coords[ligand_idx_b]
            ligand_centroid = ligand_coords.mean(dim=0, keepdim=True)  # (1, 3)

            # Protein centroid (proxy for pocket centroid)
            protein_coords = coords[protein_idx_b]
            protein_centroid = protein_coords.mean(dim=0, keepdim=True)

            # Protein atoms closest to ligand centroid
            dist_protein = torch.cdist(protein_coords, ligand_centroid).squeeze(-1)
            k_protein = min(self.top_k, protein_idx_b.numel())
            topk_protein = torch.topk(
                dist_protein, k=k_protein, largest=False
            ).indices
            selected_protein = protein_idx_b[topk_protein]

            # Ligand atoms closest to protein centroid
            dist_ligand = torch.cdist(ligand_coords, protein_centroid).squeeze(-1)
            k_ligand = min(self.top_k, ligand_idx_b.numel())
            topk_ligand = torch.topk(
                dist_ligand, k=k_ligand, largest=False
            ).indices
            selected_ligand = ligand_idx_b[topk_ligand]

            selected_b = torch.unique(torch.cat([selected_protein, selected_ligand]))
            selected_indices.append(selected_b)

        if not selected_indices:
            return torch.zeros(0, dtype=torch.long, device=device)

        return torch.cat(selected_indices, dim=0)

    def forward(
        self,
        h: Tensor,
        coords: Tensor,
        batch: Tensor,
        is_ligand: Tensor,
    ) -> Tensor:
        """
        Args:
            h: (A, D) atom embeddings.
            coords: (A, 3) atom coordinates.
            batch: (A,) batch indices.
            is_ligand: (A,) boolean mask indicating ligand atoms.

        Returns:
            Z: (B, D) pooled embedding per complex.
        """
        if h.numel() == 0:
            return h

        device = h.device
        selected_idx = self._select_topk_indices(coords, batch, is_ligand)
        if selected_idx.numel() == 0:
            # Fallback: simple mean pooling per batch
            B = int(batch.max().item()) + 1
            Z = torch.zeros((B, h.size(-1)), device=device, dtype=h.dtype)
            for b in range(B):
                mask_b = batch == b
                if mask_b.any():
                    Z[b] = h[mask_b].mean(dim=0)
            return Z

        h_sel = h[selected_idx]  # (S, D)
        batch_sel = batch[selected_idx]

        # Attention scores
        proj = torch.tanh(self.proj(h_sel))  # (S, D)
        scores = (proj * self.vector).sum(dim=-1)  # (S,)

        # Softmax per batch
        B = int(batch.max().item()) + 1
        Z = torch.zeros((B, h.size(-1)), device=device, dtype=h.dtype)
        for b in range(B):
            mask_b = batch_sel == b
            if not mask_b.any():
                continue
            idx_b = torch.nonzero(mask_b, as_tuple=False).view(-1)
            scores_b = scores[idx_b]
            h_b = h_sel[idx_b]
            alpha_b = torch.softmax(scores_b, dim=0).unsqueeze(-1)
            Z[b] = (alpha_b * h_b).sum(dim=0)

        return Z

