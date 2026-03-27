from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src_v2.rul.evaluation.metrics import mae, nasa_score, rmse


@dataclass
class Config:
    group: str = "G4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round 3: evaluate whether MCD predictive uncertainty is informative"
    )
    parser.add_argument("--group", type=str, default="G4", choices=["G2", "G4"])
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.group = args.group
    return cfg


def group_root(group: str) -> Path:
    mapping = {
        "G2": _PROJECT_ROOT / "outputs_v2" / "G2_CNN_BiLSTM_MSE_MCD",
        "G4": _PROJECT_ROOT / "outputs_v2" / "G4_CNN_BiLSTM_LinEx_MCD",
    }
    if group not in mapping:
        raise ValueError(f"Unsupported group: {group}")
    return mapping[group]


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_average(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    xr = rankdata_average(np.asarray(x, dtype=float))
    yr = rankdata_average(np.asarray(y, dtype=float))
    return safe_corr(xr, yr)


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> float | None:
    if len(y_true) == 0:
        return None

    order = np.argsort(y_std)
    y_true = y_true[order]
    y_pred = y_pred[order]
    y_std = y_std[order]

    n_bins = min(10, len(y_true))
    bins = np.array_split(np.arange(len(y_true)), n_bins)
    total = 0.0
    count = 0
    for idx in bins:
        if len(idx) == 0:
            continue
        avg_std = float(np.mean(y_std[idx]))
        avg_abs_err = float(np.mean(np.abs(y_pred[idx] - y_true[idx])))
        total += abs(avg_std - avg_abs_err) * len(idx)
        count += len(idx)

    if count == 0:
        return None
    return float(total / count)


def top_bottom_error_ratio(abs_err: np.ndarray, unc: np.ndarray) -> float | None:
    if len(abs_err) < 4:
        return None
    order = np.argsort(unc)
    k = max(1, len(abs_err) // 4)
    low = abs_err[order[:k]]
    high = abs_err[order[-k:]]
    denom = float(np.mean(low))
    if denom == 0:
        return None
    return float(np.mean(high) / denom)


def coverage_at_sigma(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    z: float,
) -> float | None:
    if len(y_true) == 0:
        return None
    lo = y_pred - z * y_std
    hi = y_pred + z * y_std
    inside = (y_true >= lo) & (y_true <= hi)
    return float(np.mean(inside))


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["y_true", "pred_mean", "pred_std"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required prediction columns in {path.name}: {missing}"
        )
    return df


def evaluate_run(run_dir: Path) -> Dict[str, Any]:
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions.csv in {run_dir.name}")

    df = load_predictions(pred_path)
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["pred_mean"].to_numpy(dtype=float)
    y_std = df["pred_std"].to_numpy(dtype=float)
    abs_err = np.abs(y_pred - y_true)

    return {
        "run_dir": run_dir.name,
        "num_samples": int(len(df)),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa": nasa_score(y_true, y_pred),
        "pred_std_mean": float(np.mean(y_std)),
        "pred_std_std": float(np.std(y_std, ddof=0)),
        "abs_err_mean": float(np.mean(abs_err)),
        "pearson_unc_abs_err": safe_corr(y_std, abs_err),
        "spearman_unc_abs_err": spearman_corr(y_std, abs_err),
        "ece_abs_err_vs_unc": expected_calibration_error(y_true, y_pred, y_std),
        "top_bottom_error_ratio": top_bottom_error_ratio(abs_err, y_std),
        "coverage_1sigma": coverage_at_sigma(y_true, y_pred, y_std, z=1.0),
        "coverage_2sigma": coverage_at_sigma(y_true, y_pred, y_std, z=2.0),
    }


def aggregate_results(df: pd.DataFrame, group: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "group": group,
        "num_runs": int(len(df)),
    }
    metric_cols = [
        "rmse",
        "mae",
        "nasa",
        "pred_std_mean",
        "pred_std_std",
        "abs_err_mean",
        "pearson_unc_abs_err",
        "spearman_unc_abs_err",
        "ece_abs_err_vs_unc",
        "top_bottom_error_ratio",
        "coverage_1sigma",
        "coverage_2sigma",
    ]
    for col in metric_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            summary[f"{col}_mean"] = (
                float(series.mean()) if series.notna().any() else None
            )
            summary[f"{col}_std"] = (
                float(series.std(ddof=0)) if series.notna().any() else None
            )
    return summary


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(Config(), args)

    root = group_root(cfg.group)
    if not root.exists():
        raise FileNotFoundError(f"Group output dir not found: {root}")

    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(root.iterdir()):
        if run_dir.is_dir() and (run_dir / "predictions.csv").exists():
            rows.append(evaluate_run(run_dir))

    if not rows:
        raise RuntimeError(f"No runs with predictions.csv found under {root}")

    df = pd.DataFrame(rows)
    summary = aggregate_results(df, cfg.group)

    out_dir = _PROJECT_ROOT / "outputs_v2" / f"{cfg.group}_round3_mcd_validity"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = out_dir / "round3_detail.csv"
    summary_json = out_dir / "round3_summary.json"
    manifest_json = out_dir / "round3_manifest.json"

    df.to_csv(detail_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg)}, f, indent=2)

    print(f"=== Round 3 MCD Validity: {cfg.group} ===")
    print(df)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nSaved files:")
    print(f"outputs_v2/{cfg.group}_round3_mcd_validity/{detail_csv.name}")
    print(f"outputs_v2/{cfg.group}_round3_mcd_validity/{summary_json.name}")
    print(f"outputs_v2/{cfg.group}_round3_mcd_validity/{manifest_json.name}")


if __name__ == "__main__":
    main()
