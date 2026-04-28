from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


@dataclass
class Config:
    group: str = "G1"


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize one experiment group across multiple seeds"
    )
    parser.add_argument(
        "--group", type=str, default="G1", choices=["G1", "G2", "G3", "G4"]
    )
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.group = args.group
    return cfg


def get_group_output_dir(group: str) -> Path:
    mapping = {
        "G1": _PROJECT_ROOT / "outputs" / "G1_CNN_BiLSTM_MSE",
        "G2": _PROJECT_ROOT / "outputs" / "G2_CNN_BiLSTM_MSE_MCD",
        "G3": _PROJECT_ROOT / "outputs" / "G3_CNN_BiLSTM_LinEx",
        "G4": _PROJECT_ROOT / "outputs" / "G4_CNN_BiLSTM_LinEx_MCD",
    }
    if group not in mapping:
        raise ValueError(f"Unsupported group: {group}")
    return mapping[group]


def safe_read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_run_info(run_dir: Path) -> Dict[str, Any] | None:
    metrics_path = run_dir / "test_metrics.json"
    config_path = run_dir / "config.json"
    best_summary_path = run_dir / "best_summary.json"
    val_metrics_path = run_dir / "val_metrics.json"

    if not metrics_path.exists():
        return None

    row: Dict[str, Any] = {"run_dir": run_dir.name}
    row.update(safe_read_json(metrics_path))

    if config_path.exists():
        cfg_json = safe_read_json(config_path)
        row["dataset_name"] = cfg_json.get("dataset_name")
        row["seq_len"] = cfg_json.get("seq_len")
        row["validation_ratio"] = cfg_json.get("validation_ratio")
        row["batch_size"] = cfg_json.get("batch_size")
        row["seed"] = cfg_json.get("seed")
        row["linex_a"] = cfg_json.get("linex_a")
        row["dropout_rate"] = cfg_json.get("dropout_rate")
        row["mc_samples_val"] = cfg_json.get("mc_samples_val")
        row["mc_samples_test"] = cfg_json.get("mc_samples_test")

    if best_summary_path.exists():
        best_summary = safe_read_json(best_summary_path)
        row["best_epoch"] = best_summary.get("best_epoch")
        row["best_val_rmse"] = best_summary.get("best_val_rmse")

    if val_metrics_path.exists():
        val_metrics = safe_read_json(val_metrics_path)
        for k, v in val_metrics.items():
            row[f"val_{k}"] = v

    return row


def unique_non_null(df: pd.DataFrame, col: str) -> List[Any]:
    if col not in df.columns:
        return []
    values = df[col].dropna().tolist()
    unique_values: List[Any] = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def check_protocol_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "num_runs": int(len(df)),
        "unique_dataset_name": unique_non_null(df, "dataset_name"),
        "unique_seq_len": unique_non_null(df, "seq_len"),
        "unique_validation_ratio": unique_non_null(df, "validation_ratio"),
        "unique_batch_size": unique_non_null(df, "batch_size"),
        "unique_linex_a": unique_non_null(df, "linex_a"),
        "unique_dropout_rate": unique_non_null(df, "dropout_rate"),
        "unique_mc_samples_val": unique_non_null(df, "mc_samples_val"),
        "unique_mc_samples_test": unique_non_null(df, "mc_samples_test"),
    }

    required_test_metrics = ["rmse", "mae", "nasa"]
    required_val_metrics = ["val_rmse", "val_mae", "val_nasa"]

    report["missing_required_test_metrics"] = [
        col for col in required_test_metrics if col not in df.columns
    ]
    report["missing_required_val_metrics"] = [
        col for col in required_val_metrics if col not in df.columns
    ]
    report["has_uncertainty_metrics"] = "pred_std_mean" in df.columns
    return report


def build_summary(df: pd.DataFrame, group: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "group": group,
        "num_runs": int(len(df)),
        "rmse_mean": float(df["rmse"].mean()) if "rmse" in df.columns else None,
        "rmse_std": float(df["rmse"].std(ddof=0)) if "rmse" in df.columns else None,
        "mae_mean": float(df["mae"].mean()) if "mae" in df.columns else None,
        "mae_std": float(df["mae"].std(ddof=0)) if "mae" in df.columns else None,
        "nasa_mean": float(df["nasa"].mean()) if "nasa" in df.columns else None,
        "nasa_std": float(df["nasa"].std(ddof=0)) if "nasa" in df.columns else None,
        "val_rmse_mean": (
            float(df["val_rmse"].mean()) if "val_rmse" in df.columns else None
        ),
        "val_rmse_std": (
            float(df["val_rmse"].std(ddof=0)) if "val_rmse" in df.columns else None
        ),
        "val_mae_mean": (
            float(df["val_mae"].mean()) if "val_mae" in df.columns else None
        ),
        "val_mae_std": (
            float(df["val_mae"].std(ddof=0)) if "val_mae" in df.columns else None
        ),
        "val_nasa_mean": (
            float(df["val_nasa"].mean()) if "val_nasa" in df.columns else None
        ),
        "val_nasa_std": (
            float(df["val_nasa"].std(ddof=0)) if "val_nasa" in df.columns else None
        ),
    }

    if "pred_std_mean" in df.columns:
        summary["pred_std_mean_mean"] = float(df["pred_std_mean"].mean())
        summary["pred_std_mean_std"] = float(df["pred_std_mean"].std(ddof=0))

    if "rmse" in df.columns and len(df) > 0:
        best_idx = df["rmse"].astype(float).idxmin()
        best_row = df.loc[best_idx]
        summary["best_seed"] = (
            int(best_row["seed"])
            if "seed" in df.columns and pd.notna(best_row.get("seed"))
            else None
        )
        summary["best_run_dir"] = str(best_row["run_dir"])
        summary["best_rmse"] = float(best_row["rmse"])
        if "linex_a" in df.columns and pd.notna(best_row.get("linex_a")):
            summary["best_linex_a"] = float(best_row["linex_a"])
    else:
        summary["best_seed"] = None
        summary["best_run_dir"] = None
        summary["best_rmse"] = None

    return summary


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(Config(), args)
    root = get_group_output_dir(cfg.group)

    if not root.exists():
        raise FileNotFoundError(f"Group output dir not found: {root.name}")

    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        row = collect_run_info(run_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No test_metrics.json files found.")
        return

    df = pd.DataFrame(rows)
    protocol = check_protocol_consistency(df)
    summary = build_summary(df, cfg.group)

    out_dir = _PROJECT_ROOT / "outputs" / f"{cfg.group}_multiseed_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = out_dir / f"{cfg.group}_multiseed_detail.csv"
    summary_json = out_dir / f"{cfg.group}_multiseed_summary.json"
    protocol_json = out_dir / f"{cfg.group}_multiseed_protocol_check.json"
    manifest_json = out_dir / f"{cfg.group}_multiseed_manifest.json"

    sort_cols = [c for c in ["seed", "linex_a", "run_dir"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    df.to_csv(detail_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(protocol_json, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg)}, f, indent=2)

    print(f"Detail saved to: {detail_csv}")
    print(f"Summary saved to: {summary_json}")
    print(f"Protocol check saved to: {protocol_json}")
    print(f"Manifest saved to: {manifest_json}")

    print("\nProtocol check:")
    for k, v in protocol.items():
        print(f"{k}: {v}")

    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
