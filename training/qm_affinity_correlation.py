import argparse
import csv
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


def _decode_name(raw: np.ndarray) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _safe_scalar(value) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return np.nan
    return float(arr.reshape(-1)[0])


def _build_feature_row(mol_group) -> Dict[str, float]:
    row: Dict[str, float] = {}

    mol_props = mol_group["mol_properties"]
    for key in mol_props.keys():
        row[f"mol_{key}"] = _safe_scalar(mol_props[key][()])

    atom_props = mol_group["atom_properties"]
    names = [_decode_name(x) for x in atom_props["atom_properties_names"][()]]
    values = np.asarray(atom_props["atom_properties_values"][()])
    if values.ndim == 2 and values.shape[1] == len(names):
        for idx, name in enumerate(names):
            col = values[:, idx]
            row[f"atom_mean_{name}"] = float(np.nanmean(col))
            row[f"atom_std_{name}"] = float(np.nanstd(col))
            row[f"atom_min_{name}"] = float(np.nanmin(col))
            row[f"atom_max_{name}"] = float(np.nanmax(col))

    return row


def _try_extract_norm(norm_h5: h5py.File, target_key: str) -> Tuple[float, float]:
    if target_key in norm_h5:
        node = norm_h5[target_key]
        if isinstance(node, h5py.Group):
            mean = _safe_scalar(node.get("mean", np.array([0.0]))[()])
            std = _safe_scalar(node.get("std", np.array([1.0]))[()])
            return mean, std if std != 0 else 1.0

    for mean_key in ("mean", "target_mean", f"{target_key}_mean"):
        if mean_key in norm_h5:
            mean = _safe_scalar(norm_h5[mean_key][()])
            break
    else:
        mean = 0.0

    for std_key in ("std", "target_std", f"{target_key}_std"):
        if std_key in norm_h5:
            std = _safe_scalar(norm_h5[std_key][()])
            break
    else:
        std = 1.0

    if std == 0:
        std = 1.0
    return mean, std


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom == 0:
        return np.nan
    return float(np.sum(x_centered * y_centered) / denom)


def _normalize_pdb_id(pid: str) -> str:
    return str(pid).strip().upper()


def _load_xlsx_targets(
    xlsx_path: str,
    pdb_column: str,
    target_column: str,
) -> Dict[str, float]:
    df = pd.read_excel(xlsx_path)
    if pdb_column not in df.columns:
        raise ValueError(f"PDB column '{pdb_column}' not found in {xlsx_path}.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {xlsx_path}.")

    out: Dict[str, float] = {}
    for _, row in df[[pdb_column, target_column]].iterrows():
        pid = _normalize_pdb_id(row[pdb_column])
        try:
            y = float(row[target_column])
        except Exception:
            continue
        if np.isfinite(y):
            out[pid] = y
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute feature-wise correlation with affinity values from QM H5."
    )
    parser.add_argument("--qm-h5", required=True, help="Path to QM H5 file.")
    parser.add_argument(
        "--norm-h5",
        default=None,
        help="Optional path to normalization H5 file for targets.",
    )
    parser.add_argument(
        "--target-key",
        default="Electron_Affinity",
        help="Key under mol_properties used as affinity target.",
    )
    parser.add_argument(
        "--use-denormalized-target",
        action="store_true",
        help="Assume target in qm-h5 is normalized and reverse with norm-h5 stats.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Print top-k absolute correlation features.",
    )
    parser.add_argument(
        "--output-csv",
        default="qm_affinity_correlations.csv",
        help="Output CSV path for feature correlations.",
    )
    parser.add_argument(
        "--target-xlsx",
        default=None,
        help="Optional XLSX file containing external target values (e.g., logKa.xlsx).",
    )
    parser.add_argument(
        "--target-pdb-column",
        default="PDBID",
        help="PDB ID column name in --target-xlsx.",
    )
    parser.add_argument(
        "--target-value-column",
        default="logKa",
        help="Target value column name in --target-xlsx.",
    )
    args = parser.parse_args()

    rows: List[Dict[str, float]] = []
    targets: List[float] = []
    xlsx_targets: Optional[Dict[str, float]] = None
    if args.target_xlsx is not None:
        xlsx_targets = _load_xlsx_targets(
            args.target_xlsx,
            pdb_column=args.target_pdb_column,
            target_column=args.target_value_column,
        )
        print(
            f"Loaded {len(xlsx_targets)} targets from {args.target_xlsx} "
            f"using columns ({args.target_pdb_column}, {args.target_value_column})."
        )
    target_mean: float = 0.0
    target_std: float = 1.0
    norm_h5: Optional[h5py.File] = None
    if args.use_denormalized_target:
        if args.norm_h5 is None:
            raise ValueError(
                "--norm-h5 is required when --use-denormalized-target is set."
            )
        norm_h5 = h5py.File(args.norm_h5, "r")
        target_mean, target_std = _try_extract_norm(norm_h5, args.target_key)

    with h5py.File(args.qm_h5, "r") as qm_h5:
        for pdb_id in qm_h5.keys():
            mol = qm_h5[pdb_id]
            pdb_id_norm = _normalize_pdb_id(pdb_id)
            if xlsx_targets is not None:
                if pdb_id_norm not in xlsx_targets:
                    continue
                y = xlsx_targets[pdb_id_norm]
            else:
                if "mol_properties" not in mol or args.target_key not in mol["mol_properties"]:
                    continue
                y = _safe_scalar(mol["mol_properties"][args.target_key][()])
                if args.use_denormalized_target:
                    y = y * target_std + target_mean

            feature_row = _build_feature_row(mol)
            if len(feature_row) == 0 or np.isnan(y):
                continue

            rows.append(feature_row)
            targets.append(y)
    if norm_h5 is not None:
        norm_h5.close()

    if len(rows) < 2:
        raise RuntimeError("Not enough valid molecules to compute correlations.")

    all_features = sorted(set().union(*[set(r.keys()) for r in rows]))
    y_arr = np.asarray(targets, dtype=np.float64)
    correlations: List[Tuple[str, float]] = []

    for feature_name in all_features:
        x_vals = np.asarray(
            [r.get(feature_name, np.nan) for r in rows], dtype=np.float64
        )
        valid = np.isfinite(x_vals) & np.isfinite(y_arr)
        corr = _pearson(x_vals[valid], y_arr[valid])
        correlations.append((feature_name, corr))

    correlations.sort(
        key=lambda x: abs(x[1]) if np.isfinite(x[1]) else -1.0, reverse=True
    )

    print(f"Computed correlations for {len(correlations)} features.")
    print(f"Top {min(args.top_k, len(correlations))} by absolute Pearson r:")
    for name, corr in correlations[: args.top_k]:
        print(f"{name:45s} r={corr:.6f}")

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "pearson_r"])
        for name, corr in correlations:
            writer.writerow([name, corr])

    print(f"Saved correlation table to {args.output_csv}")


if __name__ == "__main__":
    main()
