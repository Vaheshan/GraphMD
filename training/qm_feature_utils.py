from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch


def _decode_name(raw) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _safe_scalar(arr) -> float:
    value = np.asarray(arr)
    if value.size == 0:
        return 0.0
    return float(value.reshape(-1)[0])


def _extract_qm_features_from_group(mol_group) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    mol_props = mol_group["mol_properties"]
    for key in mol_props.keys():
        feats[f"mol_{key}"] = _safe_scalar(mol_props[key][()])

    atom_props = mol_group["atom_properties"]
    prop_names = [_decode_name(x) for x in atom_props["atom_properties_names"][()]]
    prop_vals = np.asarray(atom_props["atom_properties_values"][()])
    if prop_vals.ndim == 2 and prop_vals.shape[1] == len(prop_names):
        for i, name in enumerate(prop_names):
            col = prop_vals[:, i]
            feats[f"atom_mean_{name}"] = float(np.nanmean(col))
            feats[f"atom_std_{name}"] = float(np.nanstd(col))

    return feats


def build_qm_feature_table(
    qmh5_file: str,
    selected_pdb_ids: List[str],
    selected_feature_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Build fixed-order QM feature vectors per pdb-id.

    Returns:
      - mapping: pdb_id -> np.ndarray of shape (F,)
      - feature_names: ordered names used in the vector
    """
    raw_rows: Dict[str, Dict[str, float]] = {}
    all_names = set()

    with h5py.File(qmh5_file, "r") as qm_h5:
        for pdb_id in selected_pdb_ids:
            if pdb_id not in qm_h5:
                continue
            row = _extract_qm_features_from_group(qm_h5[pdb_id])
            raw_rows[pdb_id] = row
            all_names.update(row.keys())

    feature_names = sorted(all_names)
    if selected_feature_names is not None:
        requested = set(selected_feature_names)
        feature_names = [name for name in feature_names if name in requested]
    table: Dict[str, np.ndarray] = {}
    for pdb_id, row in raw_rows.items():
        vec = np.array([row.get(name, 0.0) for name in feature_names], dtype=np.float32)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        table[pdb_id] = vec

    return table, feature_names


def load_selected_features_from_correlation_csv(
    correlation_csv_path: str,
    min_abs_corr: Optional[float] = None,
    top_k: Optional[int] = None,
) -> List[str]:
    """
    Load selected QM feature names from correlation CSV.

    The CSV is expected to contain columns: 'feature', 'pearson_r'.
    """
    df = pd.read_csv(correlation_csv_path)
    if "feature" not in df.columns or "pearson_r" not in df.columns:
        raise ValueError(
            "Correlation CSV must contain 'feature' and 'pearson_r' columns."
        )

    df = df.copy()
    df["abs_corr"] = df["pearson_r"].abs()
    if min_abs_corr is not None:
        df = df[df["abs_corr"] >= float(min_abs_corr)]
    df = df.sort_values("abs_corr", ascending=False)
    if top_k is not None:
        df = df.head(int(top_k))
    return [str(x) for x in df["feature"].tolist()]


def make_qm_tensor_for_batch(
    pdb_ids: List[str],
    qm_feature_table: Dict[str, np.ndarray],
    feature_dim: int,
) -> torch.Tensor:
    """
    Convert pdb-id sequence into batched tensor for labels['qm_features'].
    """
    rows = []
    zeros = np.zeros((feature_dim,), dtype=np.float32)
    for pdb_id in pdb_ids:
        rows.append(qm_feature_table.get(pdb_id, zeros))
    return torch.tensor(np.stack(rows, axis=0), dtype=torch.float32)
