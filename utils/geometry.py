import torch
from torch import Tensor


def safe_normalize(v: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    """
    Safely normalize vectors along a given dimension.

    Args:
        v: Input tensor of shape (..., D).
        dim: Dimension along which to compute the norm.
        eps: Small value to avoid division by zero.

    Returns:
        Tensor with the same shape as v, where vectors along `dim` have unit norm
        when their original norm is >= eps, and are left as zeros otherwise.
    """
    norm = torch.linalg.norm(v, dim=dim, keepdim=True)
    norm_clamped = torch.clamp(norm, min=eps)
    v_norm = v / norm_clamped
    # For near-zero vectors, return zeros instead of arbitrary directions
    v_norm = torch.where(norm > eps, v_norm, torch.zeros_like(v_norm))
    return v_norm


def compute_backbone_orientation(backbone_coords: Tensor) -> Tensor:
    """
    Compute residue-level orientation features from backbone atom coordinates.

    Each residue is represented by backbone atoms (N, CA, C) with shape (R, 3, 3).
    We construct three local axes:
        u1 = normalize(N - CA)
        u2 = normalize(CA - C)
        u3 = normalize(cross(u1, u2))

    The residue feature is then F_residue = concat(u1, u2, u3) with dim = 9.

    Args:
        backbone_coords: Tensor of shape (R, 3, 3) for R residues.

    Returns:
        Tensor of shape (R, 9) containing residue orientation features.
    """
    if backbone_coords.ndim != 3 or backbone_coords.shape[1] != 3 or backbone_coords.shape[2] != 3:
        raise ValueError(
            "backbone_coords must have shape (R, 3, 3) with atoms ordered as (N, CA, C). "
            f"Got shape {tuple(backbone_coords.shape)}"
        )

    n = backbone_coords[:, 0, :]
    ca = backbone_coords[:, 1, :]
    c = backbone_coords[:, 2, :]

    u1 = safe_normalize(n - ca, dim=-1)
    u2 = safe_normalize(ca - c, dim=-1)
    # Cross product to build a third orthogonal axis
    u3_raw = torch.cross(u1, u2, dim=-1)
    u3 = safe_normalize(u3_raw, dim=-1)

    return torch.cat([u1, u2, u3], dim=-1)


def compute_atom_pair_geometric_features(
    pos_i: Tensor,
    pos_j: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute atom-pair geometric features for edges between atoms i and j.

    For each pair (i, j):
        d_hat = normalize(x_i - x_j)
        n_hat = normalize(cross(x_i, x_j))
        b_hat = cross(d_hat, n_hat)

    The feature is F_atom_pair = concat(d_hat, n_hat, b_hat) with dim = 9.

    Args:
        pos_i: Tensor of shape (E, 3), positions of source atoms.
        pos_j: Tensor of shape (E, 3), positions of target atoms.
        eps: Small numerical stability constant.

    Returns:
        Tensor of shape (E, 9) containing per-edge geometric features.
    """
    d_vec = pos_i - pos_j
    d_hat = safe_normalize(d_vec, dim=-1, eps=eps)

    n_raw = torch.cross(pos_i, pos_j, dim=-1)
    n_hat = safe_normalize(n_raw, dim=-1, eps=eps)

    b_raw = torch.cross(d_hat, n_hat, dim=-1)
    b_hat = safe_normalize(b_raw, dim=-1, eps=eps)

    return torch.cat([d_hat, n_hat, b_hat], dim=-1)

