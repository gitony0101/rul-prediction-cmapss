from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

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
    smoke_test: bool = False


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


def make_timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def make_history_run_dir(
    dataset_name: str, group: str, seed: int, linex_a: float
) -> Path:
    history_root = round7_group_output_root(dataset_name, group) / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    if group in {"G3", "G4"}:
        name = f"seed_{seed}_a_{linex_a:.2f}__{make_timestamp_id()}"
    else:
        name = f"seed_{seed}__{make_timestamp_id()}"
    run_dir = history_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def round7_latest_run_dir(
    dataset_name: str, group: str, seed: int, linex_a: float
) -> Path:
    base = round7_group_output_root(dataset_name, group)
    if group in {"G3", "G4"}:
        return base / f"seed_{seed}_a_{linex_a:.2f}"
    return base / f"seed_{seed}"


def parse_seed_and_linex_from_run_dir(
    group: str, run_dir_name: str
) -> Tuple[Optional[int], Optional[float]]:
    if not run_dir_name.startswith("seed_"):
        return None, None

    if group in {"G3", "G4"}:
        parts = run_dir_name.split("_a_")
        if len(parts) != 2:
            return None, None
        seed_part, a_part = parts
        try:
            seed = int(seed_part.replace("seed_", ""))
            linex_a = float(a_part)
        except ValueError:
            return None, None
        return seed, linex_a

    try:
        seed = int(run_dir_name.replace("seed_", ""))
    except ValueError:
        return None, None
    return seed, None


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
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only the first dataset/group/seed combination and strictly validate that outputs were written to the expected Round 7 directory.",
    )
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
    cfg.smoke_test = bool(args.smoke_test)
    return cfg


def round7_group_output_root(dataset_name: str, group: str) -> Path:
    return _OUTPUTS_ROOT / dataset_name / GROUP_OUTPUT_NAME_MAP[group]


def round7_run_dir(dataset_name: str, group: str, seed: int, linex_a: float) -> Path:
    return round7_latest_run_dir(dataset_name, group, seed, linex_a)


def snapshot_latest_run_to_history(
    dataset_name: str,
    group: str,
    seed: int,
    linex_a: float,
) -> Optional[Path]:
    latest_dir = round7_latest_run_dir(dataset_name, group, seed, linex_a)
    if not latest_dir.exists() or not latest_dir.is_dir():
        return None

    history_dir = make_history_run_dir(dataset_name, group, seed, linex_a)
    for child in latest_dir.iterdir():
        target = history_dir / child.name
        if child.is_dir():
            import shutil

            shutil.copytree(child, target)
        else:
            import shutil

            shutil.copy2(child, target)
    return history_dir


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


# === Added output validation helpers ===
def expected_required_output_paths(
    dataset_name: str,
    group: str,
    seed: int,
    linex_a: float,
) -> Dict[str, Path]:
    run_dir = round7_run_dir(dataset_name, group, seed, linex_a)
    return {
        "run_dir": run_dir,
        "metrics": run_dir / "test_metrics.json",
        "val_metrics": run_dir / "val_metrics.json",
        "config": run_dir / "config.json",
        "predictions": run_dir / "predictions.csv",
        "best_summary": run_dir / "best_summary.json",
    }


def validate_round7_outputs(
    dataset_name: str,
    group: str,
    seed: int,
    linex_a: float,
) -> Tuple[bool, str, Dict[str, str]]:
    required = expected_required_output_paths(dataset_name, group, seed, linex_a)
    run_dir = required["run_dir"]

    try:
        run_dir.resolve().relative_to(_OUTPUTS_ROOT.resolve())
    except ValueError:
        return (
            False,
            f"run_dir_outside_round7_root:{run_dir}",
            {k: str(v) for k, v in required.items()},
        )

    missing = [
        name
        for name, path in required.items()
        if name != "run_dir" and not path.exists()
    ]
    if missing:
        return (
            False,
            f"missing_expected_outputs:{','.join(missing)}",
            {k: str(v) for k, v in required.items()},
        )

    return True, "ok", {k: str(v) for k, v in required.items()}


def run_one(
    group: str,
    dataset_name: str,
    seed: int,
    cfg: Config,
) -> Dict[str, Any]:
    metrics_path = find_metrics_path(dataset_name, group, seed, cfg.linex_a)

    expected_paths = expected_required_output_paths(
        dataset_name, group, seed, cfg.linex_a
    )
    result: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "group": group,
        "seed": seed,
        "metrics_path": str(metrics_path),
        "expected_run_dir": str(expected_paths["run_dir"]),
        "status": None,
        "command": None,
        "history_run_dir": None,
        "output_validation": None,
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
        return result

    is_valid, validation_msg, validated_paths = validate_round7_outputs(
        dataset_name=dataset_name,
        group=group,
        seed=seed,
        linex_a=cfg.linex_a,
    )
    result["output_validation"] = validation_msg
    result.update({f"validated_{k}": v for k, v in validated_paths.items()})

    if not is_valid:
        result["status"] = "failed_output_validation"
        return result

    result["status"] = "done"
    history_dir = snapshot_latest_run_to_history(
        dataset_name=dataset_name,
        group=group,
        seed=seed,
        linex_a=cfg.linex_a,
    )
    if history_dir is not None:
        result["history_run_dir"] = str(history_dir)

    return result


def safe_read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_results() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if not _OUTPUTS_ROOT.exists():
        return pd.DataFrame()

    for dataset_dir in sorted(_OUTPUTS_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        if dataset_name == "round7_multidataset":
            continue

        for group, output_dir_name in GROUP_OUTPUT_NAME_MAP.items():
            group_root = dataset_dir / output_dir_name
            if not group_root.exists() or not group_root.is_dir():
                continue

            candidate_run_dirs: List[Path] = []
            for run_dir in sorted(group_root.iterdir()):
                if not run_dir.is_dir():
                    continue
                if run_dir.name == "history":
                    history_dirs = [
                        child for child in sorted(run_dir.iterdir()) if child.is_dir()
                    ]
                    candidate_run_dirs.extend(history_dirs)
                    continue
                candidate_run_dirs.append(run_dir)

            for run_dir in candidate_run_dirs:

                seed, linex_a = parse_seed_and_linex_from_run_dir(group, run_dir.name)
                if seed is None:
                    continue

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
                    "linex_a": linex_a,
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
                    row["linex_a"] = cfg_json.get("linex_a", row.get("linex_a"))
                    row["mc_samples_val"] = cfg_json.get("mc_samples_val")
                    row["mc_samples_test"] = cfg_json.get("mc_samples_test")

                rows.append(row)

    if not rows:
        return pd.DataFrame()

    detail_df = pd.DataFrame(rows)
    sort_cols = [
        c
        for c in ["dataset_name", "group", "seed", "linex_a", "run_dir"]
        if c in detail_df.columns
    ]
    detail_df = detail_df.sort_values(sort_cols).reset_index(drop=True)
    return detail_df


def append_execution_log(new_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        old_df = pd.read_csv(output_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    if not combined.empty:
        dedup_cols = [
            c
            for c in [
                "dataset_name",
                "group",
                "seed",
                "status",
                "metrics_path",
                "history_run_dir",
            ]
            if c in combined.columns
        ]
        if dedup_cols:
            combined = combined.drop_duplicates(subset=dedup_cols, keep="last")

    combined.to_csv(output_path, index=False)
    return combined


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
    stop_after_first = False
    for dataset_name in cfg.datasets:
        if stop_after_first:
            break
        for group in cfg.groups:
            if stop_after_first:
                break
            for seed in cfg.seeds:
                execution_rows.append(run_one(group, dataset_name, seed, cfg))
                if cfg.smoke_test:
                    stop_after_first = True
                    break

    execution_df = pd.DataFrame(execution_rows)
    execution_log_path = _OUTPUTS_ROOT / "round7_execution_log.csv"
    execution_df = append_execution_log(execution_df, execution_log_path)

    if cfg.smoke_test and not execution_df.empty:
        latest_row = execution_df.iloc[-1].to_dict()
        print("\n=== Round 7 Smoke Check ===")
        for key in [
            "dataset_name",
            "group",
            "seed",
            "status",
            "output_validation",
            "expected_run_dir",
            "validated_run_dir",
            "validated_metrics",
            "validated_val_metrics",
            "validated_config",
            "validated_predictions",
            "validated_best_summary",
            "history_run_dir",
        ]:
            if key in latest_row and pd.notna(latest_row[key]):
                print(f"{key}: {latest_row[key]}")

    detail_df = collect_results()
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
    print(
        "\nNote: detail/summary are rebuilt from all discovered run directories under outputs_v2/round7_multidataset, including history snapshots. Round 7 now also performs strict post-run output validation so a child script returning success without writing into the expected Round 7 tree will be marked as failed_output_validation instead of being treated as a valid run."
    )


if __name__ == "__main__":
    main()
