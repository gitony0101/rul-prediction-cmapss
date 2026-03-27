from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


@dataclass
class Config:
    groups: tuple[str, ...] = ("G1", "G2", "G3", "G4")


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check output protocol consistency for one or more experiment groups"
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=["G1", "G2", "G3", "G4"],
        choices=["G1", "G2", "G3", "G4"],
    )
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.groups = tuple(args.groups)
    return cfg


def group_output_dir(group: str) -> Path:
    mapping = {
        "G1": _PROJECT_ROOT / "outputs_v2" / "G1_CNN_BiLSTM_MSE",
        "G2": _PROJECT_ROOT / "outputs_v2" / "G2_CNN_BiLSTM_MSE_MCD",
        "G3": _PROJECT_ROOT / "outputs_v2" / "G3_CNN_BiLSTM_LinEx",
        "G4": _PROJECT_ROOT / "outputs_v2" / "G4_CNN_BiLSTM_LinEx_MCD",
    }
    if group not in mapping:
        raise ValueError(f"Unsupported group: {group}")
    return mapping[group]


def safe_read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def expected_required_files() -> List[str]:
    return [
        "best_model.pth",
        "last_model.pth",
        "train_history.csv",
        "predictions.csv",
        "config.json",
        "normalizer.json",
        "split_units.json",
        "val_metrics.json",
        "test_metrics.json",
        "best_summary.json",
    ]


def required_metric_keys_base() -> List[str]:
    return ["rmse", "mae", "nasa"]


def required_metric_keys_uncertainty() -> List[str]:
    return ["pred_std_mean", "pred_std_std", "mc_samples"]


def get_group_protocol(group: str) -> Dict[str, Any]:
    has_uncertainty = group in {"G2", "G4"}
    has_linex = group in {"G3", "G4"}

    return {
        "group": group,
        "required_files": expected_required_files(),
        "required_test_metric_keys": required_metric_keys_base()
        + (required_metric_keys_uncertainty() if has_uncertainty else []),
        "required_val_metric_keys": required_metric_keys_base()
        + (required_metric_keys_uncertainty() if has_uncertainty else []),
        "required_best_summary_keys": ["best_epoch", "best_val_rmse"],
        "prediction_required_columns": ["y_true", "y_pred"]
        + (["pred_mean", "pred_std"] if has_uncertainty else []),
        "config_expected_keys": [
            "experiment_name",
            "dataset_name",
            "seed",
            "data_dir",
            "output_root",
            "seq_len",
            "rul_cap",
            "validation_ratio",
            "hidden_size",
            "num_lstm_layers",
            "dense_size",
            "cnn_out_channels",
            "cnn_kernel_size",
            "cnn_stride",
            "cnn_pool_size",
            "batch_size",
            "lr",
            "weight_decay",
            "num_epochs",
            "early_stopping_patience",
            "grad_clip",
        ]
        + (
            ["dropout_rate", "mc_samples_val", "mc_samples_test"]
            if has_uncertainty
            else []
        )
        + (["linex_a"] if has_linex else []),
    }


def check_file_exists(run_dir: Path, filename: str) -> bool:
    return (run_dir / filename).exists()


def check_json_keys(path: Path, required_keys: List[str]) -> Dict[str, Any]:
    result = {"exists": path.exists(), "missing_keys": [], "extra_keys": []}
    if not path.exists():
        result["missing_keys"] = required_keys
        return result

    payload = safe_read_json(path)
    keys = list(payload.keys())
    result["missing_keys"] = [k for k in required_keys if k not in payload]
    result["extra_keys"] = [k for k in keys if k not in required_keys]
    return result


def check_csv_columns(path: Path, required_columns: List[str]) -> Dict[str, Any]:
    result = {
        "exists": path.exists(),
        "missing_columns": [],
        "extra_columns": [],
        "num_rows": None,
    }
    if not path.exists():
        result["missing_columns"] = required_columns
        return result

    df = pd.read_csv(path)
    cols = list(df.columns)
    result["missing_columns"] = [c for c in required_columns if c not in cols]
    result["extra_columns"] = [c for c in cols if c not in required_columns]
    result["num_rows"] = int(len(df))
    return result


def check_train_history(path: Path) -> Dict[str, Any]:
    required_columns = ["epoch", "train_loss", "val_loss"]
    result = {"exists": path.exists(), "missing_columns": [], "num_rows": None}
    if not path.exists():
        result["missing_columns"] = required_columns
        return result

    df = pd.read_csv(path)
    cols = list(df.columns)
    result["missing_columns"] = [c for c in required_columns if c not in cols]
    result["num_rows"] = int(len(df))
    return result


def check_split_units(path: Path) -> Dict[str, Any]:
    result = {
        "exists": path.exists(),
        "missing_keys": [],
        "train_units_count": None,
        "val_units_count": None,
    }
    required_keys = ["train_units", "val_units"]

    if not path.exists():
        result["missing_keys"] = required_keys
        return result

    payload = safe_read_json(path)
    result["missing_keys"] = [k for k in required_keys if k not in payload]
    if "train_units" in payload and isinstance(payload["train_units"], list):
        result["train_units_count"] = int(len(payload["train_units"]))
    if "val_units" in payload and isinstance(payload["val_units"], list):
        result["val_units_count"] = int(len(payload["val_units"]))
    return result


def check_run_dir(group: str, run_dir: Path) -> Dict[str, Any]:
    protocol = get_group_protocol(group)

    file_presence = {
        filename: check_file_exists(run_dir, filename)
        for filename in protocol["required_files"]
    }

    config_check = check_json_keys(
        run_dir / "config.json", protocol["config_expected_keys"]
    )
    val_metrics_check = check_json_keys(
        run_dir / "val_metrics.json", protocol["required_val_metric_keys"]
    )
    test_metrics_check = check_json_keys(
        run_dir / "test_metrics.json", protocol["required_test_metric_keys"]
    )
    best_summary_check = check_json_keys(
        run_dir / "best_summary.json", protocol["required_best_summary_keys"]
    )
    predictions_check = check_csv_columns(
        run_dir / "predictions.csv", protocol["prediction_required_columns"]
    )
    train_history_check = check_train_history(run_dir / "train_history.csv")
    split_units_check = check_split_units(run_dir / "split_units.json")

    missing_files = [k for k, v in file_presence.items() if not v]

    issues: List[str] = []
    if missing_files:
        issues.append(f"missing_files={missing_files}")
    if config_check["missing_keys"]:
        issues.append(f"config_missing={config_check['missing_keys']}")
    if val_metrics_check["missing_keys"]:
        issues.append(f"val_metrics_missing={val_metrics_check['missing_keys']}")
    if test_metrics_check["missing_keys"]:
        issues.append(f"test_metrics_missing={test_metrics_check['missing_keys']}")
    if best_summary_check["missing_keys"]:
        issues.append(f"best_summary_missing={best_summary_check['missing_keys']}")
    if predictions_check["missing_columns"]:
        issues.append(
            f"predictions_missing_columns={predictions_check['missing_columns']}"
        )
    if train_history_check["missing_columns"]:
        issues.append(
            f"train_history_missing_columns={train_history_check['missing_columns']}"
        )
    if split_units_check["missing_keys"]:
        issues.append(f"split_units_missing={split_units_check['missing_keys']}")

    status = "PASS" if len(issues) == 0 else "FAIL"

    return {
        "group": group,
        "run_dir": run_dir.name,
        "status": status,
        "issues": issues,
        "file_presence": file_presence,
        "config_check": config_check,
        "val_metrics_check": val_metrics_check,
        "test_metrics_check": test_metrics_check,
        "best_summary_check": best_summary_check,
        "predictions_check": predictions_check,
        "train_history_check": train_history_check,
        "split_units_check": split_units_check,
    }


def check_group(group: str) -> Dict[str, Any]:
    root = group_output_dir(group)

    if not root.exists():
        return {
            "group": group,
            "status": "MISSING_GROUP_DIR",
            "group_dir": str(root),
            "num_runs": 0,
            "runs": [],
            "pass_count": 0,
            "fail_count": 0,
        }

    run_results = []
    for run_dir in sorted(root.iterdir()):
        if run_dir.is_dir():
            run_results.append(check_run_dir(group, run_dir))

    pass_count = sum(1 for r in run_results if r["status"] == "PASS")
    fail_count = sum(1 for r in run_results if r["status"] != "PASS")
    group_status = "PASS" if fail_count == 0 else "FAIL"

    return {
        "group": group,
        "status": group_status,
        "group_dir": f"outputs_v2/{root.name}",
        "num_runs": int(len(run_results)),
        "pass_count": int(pass_count),
        "fail_count": int(fail_count),
        "runs": run_results,
    }


def build_flat_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group_result in results:
        group = group_result["group"]
        if group_result["status"] == "MISSING_GROUP_DIR":
            rows.append(
                {
                    "group": group,
                    "run_dir": "",
                    "status": "MISSING_GROUP_DIR",
                    "num_issues": None,
                    "issues": "",
                }
            )
            continue

        for run in group_result["runs"]:
            rows.append(
                {
                    "group": group,
                    "run_dir": run["run_dir"],
                    "status": run["status"],
                    "num_issues": int(len(run["issues"])),
                    "issues": " | ".join(run["issues"]),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(Config(), args)

    results = [check_group(group) for group in cfg.groups]

    out_dir = _PROJECT_ROOT / "outputs_v2" / "protocol_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_json = out_dir / "protocol_check_detail.json"
    summary_csv = out_dir / "protocol_check_summary.csv"
    manifest_json = out_dir / "protocol_check_manifest.json"

    with open(detail_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    flat_df = build_flat_summary(results)
    flat_df.to_csv(summary_csv, index=False)

    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg)}, f, indent=2)

    print("=== Protocol Check ===")
    for group_result in results:
        print(
            f"{group_result['group']}: status={group_result['status']} "
            f"num_runs={group_result['num_runs']} "
            f"pass_count={group_result.get('pass_count', 0)} "
            f"fail_count={group_result.get('fail_count', 0)}"
        )

    print("\nSaved files:")
    print("outputs_v2/protocol_checks/protocol_check_detail.json")
    print("outputs_v2/protocol_checks/protocol_check_summary.csv")
    print("outputs_v2/protocol_checks/protocol_check_manifest.json")


if __name__ == "__main__":
    main()
