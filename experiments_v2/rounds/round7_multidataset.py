from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


_THIS_FILE = Path(__file__).resolve()
_EXPERIMENTS_ROOT = _THIS_FILE.parents[1]
_PROJECT_ROOT = _THIS_FILE.parents[3]
_OUTPUTS_ROOT = _PROJECT_ROOT / "outputs_v2" / "round7_multidataset"


@dataclass
class Config:
    groups: tuple[str, ...] = ("G1", "G2", "G3", "G4")
    datasets: tuple[str, ...] = ("FD003", "FD002", "FD004")
    seeds: tuple[int, ...] = (42,)
    linex_a: float = 0.04
    mc_samples_val: int = 10
    mc_samples_test: int = 20
    skip_existing: bool = True
    dry_run: bool = False


RUN_SCRIPT_MAP = {
    "G1": "G1_run.py",
    "G2": "G2_run.py",
    "G3": "G3_run.py",
    "G4": "G4_run.py",
}

GROUP_OUTPUT_NAME_MAP = {
    "G1": "G1_CNN_BiLSTM_MSE",
    "G2": "G2_CNN_BiLSTM_MSE_MCD",
    "G3": "G3_CNN_BiLSTM_LinEx",
    "G4": "G4_CNN_BiLSTM_LinEx_MCD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round 7 multi-dataset experiment runner for G1/G2/G3/G4"
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["G1", "G2", "G3", "G4"],
        choices=["G1", "G2", "G3", "G4"],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["FD003", "FD002", "FD004"],
        choices=["FD001", "FD002", "FD003", "FD004"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
    )
    parser.add_argument("--linex-a", type=float, default=0.04)
    parser.add_argument("--mc-samples-val", type=int, default=10)
    parser.add_argument("--mc-samples-test", type=int, default=20)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false"
    )
    parser.set_defaults(skip_existing=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.groups = tuple(args.groups)
    cfg.datasets = tuple(args.datasets)
    cfg.seeds = tuple(args.seeds)
    cfg.linex_a = float(args.linex_a)
    cfg.mc_samples_val = int(args.mc_samples_val)
    cfg.mc_samples_test = int(args.mc_samples_test)
    cfg.skip_existing = bool(args.skip_existing)
    cfg.dry_run = bool(args.dry_run)
    return cfg


def round7_group_output_root(dataset_name: str, group: str) -> Path:
    return _OUTPUTS_ROOT / dataset_name / GROUP_OUTPUT_NAME_MAP[group]


def round7_run_dir(dataset_name: str, group: str, seed: int, linex_a: float) -> Path:
    base = round7_group_output_root(dataset_name, group)
    if group in {"G3", "G4"}:
        return base / f"seed_{seed}_a_{linex_a:.2f}"
    return base / f"seed_{seed}"


def build_command(
    group: str,
    dataset_name: str,
    seed: int,
    cfg: Config,
) -> List[str]:
    script_name = RUN_SCRIPT_MAP[group]
    script_path = _EXPERIMENTS_ROOT / script_name

    cmd = [
        sys.executable,
        str(script_path),
        "--seed",
        str(seed),
        "--dataset",
        dataset_name,
        "--output-root",
        str(round7_group_output_root(dataset_name, group)),
    ]

    if group in {"G3", "G4"}:
        cmd.extend(["--linex-a", str(cfg.linex_a)])

    if group in {"G2", "G4"}:
        cmd.extend(
            [
                "--mc-samples-val",
                str(cfg.mc_samples_val),
                "--mc-samples-test",
                str(cfg.mc_samples_test),
            ]
        )

    return cmd


def find_metrics_path(dataset_name: str, group: str, seed: int, linex_a: float) -> Path:
    return round7_run_dir(dataset_name, group, seed, linex_a) / "test_metrics.json"


def run_one(
    group: str,
    dataset_name: str,
    seed: int,
    cfg: Config,
) -> Dict[str, Any]:
    metrics_path = find_metrics_path(dataset_name, group, seed, cfg.linex_a)

    result: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "group": group,
        "seed": seed,
        "metrics_path": str(metrics_path),
        "status": None,
        "command": None,
    }

    if cfg.skip_existing and metrics_path.exists():
        result["status"] = "skipped_existing"
        return result

    cmd = build_command(group, dataset_name, seed, cfg)
    result["command"] = cmd

    if cfg.dry_run:
        result["status"] = "dry_run"
        return result

    print(f"[INFO] Running {group} | {dataset_name} | seed={seed}")
    completed = subprocess.run(cmd, cwd=str(_EXPERIMENTS_ROOT))
    if completed.returncode != 0:
        result["status"] = f"failed_returncode_{completed.returncode}"
    else:
        result["status"] = "done"

    return result


def safe_read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_results(cfg: Config) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for dataset_name in cfg.datasets:
        for group in cfg.groups:
            for seed in cfg.seeds:
                run_dir = round7_run_dir(dataset_name, group, seed, cfg.linex_a)
                test_metrics_path = run_dir / "test_metrics.json"
                val_metrics_path = run_dir / "val_metrics.json"
                config_path = run_dir / "config.json"

                if not test_metrics_path.exists():
                    continue

                row: Dict[str, Any] = {
                    "dataset_name": dataset_name,
                    "group": group,
                    "seed": seed,
                    "run_dir": run_dir.name,
                }

                row.update(safe_read_json(test_metrics_path))

                if val_metrics_path.exists():
                    val_metrics = safe_read_json(val_metrics_path)
                    for k, v in val_metrics.items():
                        row[f"val_{k}"] = v

                if config_path.exists():
                    cfg_json = safe_read_json(config_path)
                    row["seq_len"] = cfg_json.get("seq_len")
                    row["batch_size"] = cfg_json.get("batch_size")
                    row["linex_a"] = cfg_json.get("linex_a")
                    row["mc_samples_val"] = cfg_json.get("mc_samples_val")
                    row["mc_samples_test"] = cfg_json.get("mc_samples_test")

                rows.append(row)

    return pd.DataFrame(rows)


def summarize_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame()

    group_cols = ["dataset_name", "group"]
    summary_rows: List[Dict[str, Any]] = []

    for (dataset_name, group), sub in detail_df.groupby(group_cols, sort=True):
        row: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "group": group,
            "num_runs": int(len(sub)),
        }

        for metric in ["rmse", "mae", "nasa", "val_rmse", "val_mae", "val_nasa"]:
            if metric in sub.columns:
                s = pd.to_numeric(sub[metric], errors="coerce")
                row[f"{metric}_mean"] = float(s.mean()) if s.notna().any() else None
                row[f"{metric}_std"] = float(s.std(ddof=0)) if s.notna().any() else None

        if "pred_std_mean" in sub.columns:
            s = pd.to_numeric(sub["pred_std_mean"], errors="coerce")
            row["pred_std_mean_mean"] = float(s.mean()) if s.notna().any() else None
            row["pred_std_mean_std"] = float(s.std(ddof=0)) if s.notna().any() else None

        if "rmse" in sub.columns:
            s = pd.to_numeric(sub["rmse"], errors="coerce")
            best_idx = s.idxmin()
            best_row = sub.loc[best_idx]
            row["best_seed"] = int(best_row["seed"])
            row["best_rmse"] = float(best_row["rmse"])
            row["best_run_dir"] = str(best_row["run_dir"])

        summary_rows.append(row)

    out_df = pd.DataFrame(summary_rows)
    out_df = out_df.sort_values(["dataset_name", "rmse_mean", "group"]).reset_index(
        drop=True
    )
    return out_df


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(Config(), args)

    _OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

    execution_rows: List[Dict[str, Any]] = []
    for dataset_name in cfg.datasets:
        for group in cfg.groups:
            for seed in cfg.seeds:
                execution_rows.append(run_one(group, dataset_name, seed, cfg))

    execution_df = pd.DataFrame(execution_rows)
    execution_log_path = _OUTPUTS_ROOT / "round7_execution_log.csv"
    execution_df.to_csv(execution_log_path, index=False)

    detail_df = collect_results(cfg)
    detail_path = _OUTPUTS_ROOT / "round7_detail.csv"
    detail_df.to_csv(detail_path, index=False)

    summary_df = summarize_results(detail_df)
    summary_path = _OUTPUTS_ROOT / "round7_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    manifest_path = _OUTPUTS_ROOT / "round7_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg)}, f, indent=2)

    print("\n=== Round 7 Multi-dataset Summary ===")
    if summary_df.empty:
        print("No completed runs found.")
    else:
        print(summary_df)

    print("\nSaved files:")
    print("outputs_v2/round7_multidataset/round7_execution_log.csv")
    print("outputs_v2/round7_multidataset/round7_detail.csv")
    print("outputs_v2/round7_multidataset/round7_summary.csv")
    print("outputs_v2/round7_multidataset/round7_manifest.json")


if __name__ == "__main__":
    main()
