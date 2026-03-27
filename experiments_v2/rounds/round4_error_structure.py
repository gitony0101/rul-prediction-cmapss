from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src_v2.rul.evaluation.metrics import mae, nasa_score, rmse


@dataclass
class Config:
    groups: tuple[str, ...] = ("G1", "G2", "G3", "G4")
    danger_threshold: float = 20.0
    severe_error_threshold: float = 10.0
    moderate_error_threshold: float = 5.0
    rul_bins: tuple[float, ...] = (0.0, 20.0, 50.0, 125.0)


GROUP_OUTPUT_DIRS = {
    "G1": "G1_CNN_BiLSTM_MSE",
    "G2": "G2_CNN_BiLSTM_MSE_MCD",
    "G3": "G3_CNN_BiLSTM_LinEx",
    "G4": "G4_CNN_BiLSTM_LinEx_MCD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round 4: error structure analysis across experiment groups"
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=["G1", "G2", "G3", "G4"],
        choices=["G1", "G2", "G3", "G4"],
    )
    parser.add_argument("--danger-threshold", type=float, default=20.0)
    parser.add_argument("--severe-error-threshold", type=float, default=10.0)
    parser.add_argument("--moderate-error-threshold", type=float, default=5.0)
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.groups = tuple(args.groups)
    cfg.danger_threshold = float(args.danger_threshold)
    cfg.severe_error_threshold = float(args.severe_error_threshold)
    cfg.moderate_error_threshold = float(args.moderate_error_threshold)
    return cfg


def group_root(group: str) -> Path:
    if group not in GROUP_OUTPUT_DIRS:
        raise ValueError(f"Unsupported group: {group}")
    return _PROJECT_ROOT / "outputs_v2" / GROUP_OUTPUT_DIRS[group]


def load_predictions(pred_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    if "y_true" not in df.columns:
        raise ValueError(f"Missing y_true in {pred_path}")

    if "y_pred" in df.columns:
        pred_col = "y_pred"
    elif "pred_mean" in df.columns:
        pred_col = "pred_mean"
    else:
        raise ValueError(f"Missing y_pred or pred_mean in {pred_path}")

    out = df.copy()
    if pred_col != "y_pred":
        out["y_pred"] = out[pred_col]
    return out


def build_error_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["y_true"] = out["y_true"].astype(float)
    out["y_pred"] = out["y_pred"].astype(float)
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["squared_error"] = out["error"] ** 2
    out["is_overestimate"] = (out["error"] > 0).astype(int)
    out["is_underestimate"] = (out["error"] < 0).astype(int)
    return out


def compute_basic_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    err = df["error"].to_numpy(dtype=float)
    abs_err = np.abs(err)

    return {
        "num_samples": int(len(df)),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa": nasa_score(y_true, y_pred),
        "mean_error": float(np.mean(err)),
        "mean_abs_error": float(np.mean(abs_err)),
        "median_abs_error": float(np.median(abs_err)),
        "overestimate_rate": float(np.mean(err > 0)),
        "underestimate_rate": float(np.mean(err < 0)),
    }


def compute_threshold_metrics(
    df: pd.DataFrame,
    moderate_error_threshold: float,
    severe_error_threshold: float,
) -> Dict[str, Any]:
    err = df["error"].to_numpy(dtype=float)
    abs_err = np.abs(err)

    return {
        "p_abs_error_gt_moderate": float(np.mean(abs_err > moderate_error_threshold)),
        "p_abs_error_gt_severe": float(np.mean(abs_err > severe_error_threshold)),
        "p_overestimate_gt_moderate": float(np.mean(err > moderate_error_threshold)),
        "p_underestimate_lt_neg_moderate": float(
            np.mean(err < -moderate_error_threshold)
        ),
        "p_overestimate_gt_severe": float(np.mean(err > severe_error_threshold)),
        "p_underestimate_lt_neg_severe": float(np.mean(err < -severe_error_threshold)),
    }


def rul_bin_labels(bins: Tuple[float, ...]) -> List[str]:
    labels: List[str] = []
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        if i == 0:
            labels.append(f"[{int(left)},{int(right)}]")
        else:
            labels.append(f"({int(left)},{int(right)}]")
    return labels


def compute_binned_metrics(
    df: pd.DataFrame,
    bins: Tuple[float, ...],
    moderate_error_threshold: float,
    severe_error_threshold: float,
) -> List[Dict[str, Any]]:
    labels = rul_bin_labels(bins)
    work = df.copy()
    work["rul_bin"] = pd.cut(
        work["y_true"],
        bins=list(bins),
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows: List[Dict[str, Any]] = []
    for label in labels:
        sub = work[work["rul_bin"] == label].copy()
        row: Dict[str, Any] = {
            "rul_bin": label,
            "num_samples": int(len(sub)),
        }
        if len(sub) == 0:
            row.update(
                {
                    "rmse": None,
                    "mae": None,
                    "nasa": None,
                    "mean_error": None,
                    "mean_abs_error": None,
                    "overestimate_rate": None,
                    "underestimate_rate": None,
                    "p_abs_error_gt_moderate": None,
                    "p_abs_error_gt_severe": None,
                    "p_overestimate_gt_moderate": None,
                    "p_overestimate_gt_severe": None,
                }
            )
        else:
            row.update(compute_basic_metrics(sub))
            row.update(
                compute_threshold_metrics(
                    sub,
                    moderate_error_threshold=moderate_error_threshold,
                    severe_error_threshold=severe_error_threshold,
                )
            )
        rows.append(row)
    return rows


def compute_danger_zone_metrics(
    df: pd.DataFrame,
    danger_threshold: float,
    moderate_error_threshold: float,
    severe_error_threshold: float,
) -> Dict[str, Any]:
    danger = df[df["y_true"] <= danger_threshold].copy()
    safe = df[df["y_true"] > danger_threshold].copy()

    result: Dict[str, Any] = {
        "danger_threshold": float(danger_threshold),
        "danger_num_samples": int(len(danger)),
        "safe_num_samples": int(len(safe)),
    }

    if len(danger) > 0:
        danger_metrics = compute_basic_metrics(danger)
        danger_thresholds = compute_threshold_metrics(
            danger,
            moderate_error_threshold=moderate_error_threshold,
            severe_error_threshold=severe_error_threshold,
        )
        for k, v in {**danger_metrics, **danger_thresholds}.items():
            result[f"danger_{k}"] = v
    else:
        for key in [
            "rmse",
            "mae",
            "nasa",
            "mean_error",
            "mean_abs_error",
            "median_abs_error",
            "overestimate_rate",
            "underestimate_rate",
            "p_abs_error_gt_moderate",
            "p_abs_error_gt_severe",
            "p_overestimate_gt_moderate",
            "p_underestimate_lt_neg_moderate",
            "p_overestimate_gt_severe",
            "p_underestimate_lt_neg_severe",
        ]:
            result[f"danger_{key}"] = None

    if len(safe) > 0:
        safe_metrics = compute_basic_metrics(safe)
        safe_thresholds = compute_threshold_metrics(
            safe,
            moderate_error_threshold=moderate_error_threshold,
            severe_error_threshold=severe_error_threshold,
        )
        for k, v in {**safe_metrics, **safe_thresholds}.items():
            result[f"safe_{k}"] = v
    else:
        for key in [
            "rmse",
            "mae",
            "nasa",
            "mean_error",
            "mean_abs_error",
            "median_abs_error",
            "overestimate_rate",
            "underestimate_rate",
            "p_abs_error_gt_moderate",
            "p_abs_error_gt_severe",
            "p_overestimate_gt_moderate",
            "p_underestimate_lt_neg_moderate",
            "p_overestimate_gt_severe",
            "p_underestimate_lt_neg_severe",
        ]:
            result[f"safe_{key}"] = None

    return result


def evaluate_run(
    run_dir: Path, cfg: Config
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions.csv in {run_dir}")

    pred_df = load_predictions(pred_path)
    work = build_error_frame(pred_df)

    run_summary: Dict[str, Any] = {"run_dir": run_dir.name}
    run_summary.update(compute_basic_metrics(work))
    run_summary.update(
        compute_threshold_metrics(
            work,
            moderate_error_threshold=cfg.moderate_error_threshold,
            severe_error_threshold=cfg.severe_error_threshold,
        )
    )
    run_summary.update(
        compute_danger_zone_metrics(
            work,
            danger_threshold=cfg.danger_threshold,
            moderate_error_threshold=cfg.moderate_error_threshold,
            severe_error_threshold=cfg.severe_error_threshold,
        )
    )

    binned_rows = compute_binned_metrics(
        work,
        bins=cfg.rul_bins,
        moderate_error_threshold=cfg.moderate_error_threshold,
        severe_error_threshold=cfg.severe_error_threshold,
    )
    for row in binned_rows:
        row["run_dir"] = run_dir.name

    return run_summary, binned_rows


def aggregate_group_summary(df: pd.DataFrame, group: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "group": group,
        "num_runs": int(len(df)),
    }

    metric_cols = [
        c for c in df.columns if c != "run_dir" and pd.api.types.is_numeric_dtype(df[c])
    ]
    for col in metric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            summary[f"{col}_mean"] = float(series.mean())
            summary[f"{col}_std"] = float(series.std(ddof=0))
        else:
            summary[f"{col}_mean"] = None
            summary[f"{col}_std"] = None

    if "rmse" in df.columns and len(df) > 0:
        best_idx = pd.to_numeric(df["rmse"], errors="coerce").idxmin()
        best_row = df.loc[best_idx]
        summary["best_run_dir"] = str(best_row["run_dir"])
        summary["best_rmse"] = float(best_row["rmse"])

    return summary


def aggregate_group_bins(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in df.columns
        if c not in {"run_dir", "rul_bin"} and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows: List[Dict[str, Any]] = []
    for rul_bin, sub in df.groupby("rul_bin", sort=False):
        row: Dict[str, Any] = {
            "rul_bin": rul_bin,
            "num_runs": int(sub["run_dir"].nunique()),
        }
        for col in numeric_cols:
            series = pd.to_numeric(sub[col], errors="coerce")
            if series.notna().any():
                row[f"{col}_mean"] = float(series.mean())
                row[f"{col}_std"] = float(series.std(ddof=0))
            else:
                row[f"{col}_mean"] = None
                row[f"{col}_std"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(Config(), args)

    out_root = _PROJECT_ROOT / "outputs_v2" / "round4_error_structure"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "round4_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg)}, f, indent=2)

    for group in cfg.groups:
        root = group_root(group)
        if not root.exists():
            print(f"[WARN] Missing group output dir for {group}: {root}")
            continue

        run_rows: List[Dict[str, Any]] = []
        bin_rows: List[Dict[str, Any]] = []

        for run_dir in sorted(root.iterdir()):
            if run_dir.is_dir() and (run_dir / "predictions.csv").exists():
                run_summary, run_bins = evaluate_run(run_dir, cfg)
                run_rows.append(run_summary)
                bin_rows.extend(run_bins)

        if not run_rows:
            print(f"[WARN] No valid predictions found for {group}")
            continue

        run_df = pd.DataFrame(run_rows)
        bins_df = pd.DataFrame(bin_rows)
        summary = aggregate_group_summary(run_df, group)
        bins_summary_df = aggregate_group_bins(bins_df)

        group_dir = out_root / group
        group_dir.mkdir(parents=True, exist_ok=True)

        detail_csv = group_dir / "round4_detail.csv"
        summary_json = group_dir / "round4_summary.json"
        bins_detail_csv = group_dir / "round4_bins_detail.csv"
        bins_summary_csv = group_dir / "round4_bins_summary.csv"

        run_df.to_csv(detail_csv, index=False)
        bins_df.to_csv(bins_detail_csv, index=False)
        bins_summary_df.to_csv(bins_summary_csv, index=False)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"=== Round 4 Error Structure: {group} ===")
        print(run_df)
        print("\nSummary:")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print("\nSaved files:")
        print(f"outputs_v2/round4_error_structure/{group}/{detail_csv.name}")
        print(f"outputs_v2/round4_error_structure/{group}/{summary_json.name}")
        print(f"outputs_v2/round4_error_structure/{group}/{bins_detail_csv.name}")
        print(f"outputs_v2/round4_error_structure/{group}/{bins_summary_csv.name}")
        print()

    print("Saved files:")
    print("outputs_v2/round4_error_structure/round4_manifest.json")


if __name__ == "__main__":
    main()
