from typing import Optional

import torch
from torch import nn, Tensor


class CrossGraphAttentionModule(nn.Module):
    """
    Residue → atom cross-graph attention.

    For each pocket atom a and nearby residues r:
        Q_a = W_Q h_a
        K_r = W_K h_r
        V_r = W_V h_r
        score_ar = (Q_a · K_r) / sqrt(d)
        alpha_ar = softmax_r(score_ar)
        h_a' = h_a + Σ_r alpha_ar V_r

    Batched operation is supported via batch index tensors.
    """

    def __init__(
        self,
        atom_dim: int,
        residue_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or atom_dim

        self.q_proj = nn.Linear(atom_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(residue_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(residue_dim, atom_dim, bias=False)

        self.scale = hidden_dim ** 0.5

    def forward(
        self,
        atom_h: Tensor,
        residue_h: Tensor,
        atom_batch: Tensor,
        residue_batch: Tensor,
        atom_to_residue: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            atom_h: (A, D_a) atom embeddings.
            residue_h: (R, D_r) residue embeddings.
            atom_batch: (A,) batch indices for atoms.
            residue_batch: (R,) batch indices for residues.
            atom_to_residue: Optional (A,) LongTensor mapping each atom to a residue
                index (0..R-1) or -1 if unmapped. When provided, attention is
                restricted to residues that actually appear in the pocket atoms
                for each complex.

        Returns:
            Updated atom embeddings of shape (A, D_a).
        """
        device = atom_h.device
        A = atom_h.size(0)

        Q = self.q_proj(atom_h)  # (A, H)
        K = self.k_proj(residue_h)  # (R, H)
        V = self.v_proj(residue_h)  # (R, D_a)

        out = atom_h.clone()

        batch_ids = torch.unique(atom_batch)
        for b in batch_ids:
            atom_mask_b = atom_batch == b
            res_mask_b = residue_batch == b

            if not atom_mask_b.any() or not res_mask_b.any():
                continue

            atom_idx_b = torch.nonzero(atom_mask_b, as_tuple=False).view(-1)
            res_idx_b = torch.nonzero(res_mask_b, as_tuple=False).view(-1)

            # Optionally restrict residues to those with at least one pocket atom.
            if atom_to_residue is not None:
                atom_to_residue_b = atom_to_residue[atom_idx_b]
                valid_res = atom_to_residue_b[atom_to_residue_b >= 0]
                if valid_res.numel() > 0:
                    res_from_atoms = torch.unique(valid_res)
                    # Intersect with residues in this batch
                    res_idx_b = torch.unique(
                        torch.cat([res_idx_b, res_from_atoms.to(res_idx_b.device)])
                    )

            Q_b = Q[atom_idx_b]  # (A_b, H)
            K_b = K[res_idx_b]  # (R_b, H)
            V_b = V[res_idx_b]  # (R_b, D_a)

            # Attention scores: (A_b, R_b)
            scores = Q_b @ K_b.t() / self.scale
            alpha = torch.softmax(scores, dim=-1)  # (A_b, R_b)
            ctx = alpha @ V_b  # (A_b, D_a)

            out[atom_idx_b] = out[atom_idx_b] + ctx

        return out

