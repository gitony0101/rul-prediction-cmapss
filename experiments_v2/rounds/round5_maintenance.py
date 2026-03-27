from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src_v2.rul.constants import get_feature_columns
from src_v2.rul.dataset import (
    SequenceDataset,
    add_train_rul,
    apply_normalizer,
    build_windows,
    fit_normalizer,
    read_cmapss_split,
)
from src_v2.rul.inference.mcd import mc_dropout_predict
from src_v2.rul.models.cnn_bilstm import CNNBiLSTM
from src_v2.rul.models.cnn_bilstm_dropout import CNNBiLSTMDropout
from src_v2.rul.utils import select_device


@dataclass
class Config:
    groups: tuple[str, ...] = ("G1", "G2", "G3", "G4")
    thresholds: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)
    preventive_cost: float = 1.0
    failure_cost: float = 10.0
    early_cost_per_cycle: float = 0.05
    batch_size_override: int | None = None
    mc_samples_override: int | None = None


GROUP_OUTPUT_DIRS = {
    "G1": "G1_CNN_BiLSTM_MSE",
    "G2": "G2_CNN_BiLSTM_MSE_MCD",
    "G3": "G3_CNN_BiLSTM_LinEx",
    "G4": "G4_CNN_BiLSTM_LinEx_MCD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round 5: validation full-trajectory maintenance simulation across experiment groups"
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=["G1", "G2", "G3", "G4"],
        choices=["G1", "G2", "G3", "G4"],
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
    )
    parser.add_argument("--preventive-cost", type=float, default=1.0)
    parser.add_argument("--failure-cost", type=float, default=10.0)
    parser.add_argument("--early-cost-per-cycle", type=float, default=0.05)
    parser.add_argument("--batch-size-override", type=int, default=None)
    parser.add_argument("--mc-samples-override", type=int, default=None)
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.groups = tuple(args.groups)
    cfg.thresholds = tuple(float(x) for x in args.thresholds)
    cfg.preventive_cost = float(args.preventive_cost)
    cfg.failure_cost = float(args.failure_cost)
    cfg.early_cost_per_cycle = float(args.early_cost_per_cycle)
    cfg.batch_size_override = args.batch_size_override
    cfg.mc_samples_override = args.mc_samples_override
    return cfg


def group_root(group: str) -> Path:
    if group not in GROUP_OUTPUT_DIRS:
        raise ValueError(f"Unsupported group: {group}")
    return _PROJECT_ROOT / "outputs_v2" / GROUP_OUTPUT_DIRS[group]


def safe_read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path_from_project(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((_PROJECT_ROOT / path).resolve())


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []

    for X, y, meta in loader:
        X = X.to(device)
        out = model(X).detach().cpu().numpy()
        preds.append(out)
        trues.append(y.detach().cpu().numpy())

        batch_size = len(y)
        for i in range(batch_size):
            row: Dict[str, Any] = {}
            for k, v in meta.items():
                item = v[i]
                row[k] = _to_python_scalar(item)
            metas.append(row)

    y_pred = np.concatenate(preds) if preds else np.array([], dtype=float)
    y_true = np.concatenate(trues) if trues else np.array([], dtype=float)
    return y_true, y_pred, metas


def build_model(group: str, cfg_json: Dict[str, Any], input_size: int):
    common_kwargs = {
        "input_size": input_size,
        "cnn_out_channels": int(cfg_json["cnn_out_channels"]),
        "cnn_kernel_size": int(cfg_json["cnn_kernel_size"]),
        "cnn_stride": int(cfg_json["cnn_stride"]),
        "cnn_pool_size": int(cfg_json["cnn_pool_size"]),
        "hidden_size": int(cfg_json["hidden_size"]),
        "num_lstm_layers": int(cfg_json["num_lstm_layers"]),
        "dense_size": int(cfg_json["dense_size"]),
    }
    if group in {"G2", "G4"}:
        return CNNBiLSTMDropout(
            **common_kwargs,
            dropout_rate=float(cfg_json["dropout_rate"]),
        )
    return CNNBiLSTM(**common_kwargs)


def infer_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "unit_id" not in out.columns:
        raise ValueError("Predicted trajectory meta is missing unit_id")

    if "cycle" not in out.columns:
        cycle_candidates = [c for c in out.columns if c.lower().endswith("cycle")]
        if not cycle_candidates:
            raise ValueError("Predicted trajectory meta is missing cycle")
        out = out.rename(columns={cycle_candidates[0]: "cycle"})

    out["unit_id"] = out["unit_id"].astype(int)
    out["cycle"] = out["cycle"].astype(int)
    return out


def build_validation_trajectory_predictions(
    run_dir: Path,
    group: str,
    cfg: Config,
    device: str,
) -> pd.DataFrame:
    cfg_json = safe_read_json(run_dir / "config.json")
    split_json = safe_read_json(run_dir / "split_units.json")

    data_dir = resolve_path_from_project(str(cfg_json["data_dir"]))
    dataset_name = str(cfg_json["dataset_name"])
    seq_len = int(cfg_json["seq_len"])
    rul_cap = float(cfg_json["rul_cap"])
    batch_size = (
        int(cfg.batch_size_override)
        if cfg.batch_size_override is not None
        else int(cfg_json["batch_size"])
    )

    feats = get_feature_columns()
    train_df, _, _ = read_cmapss_split(data_dir, dataset_name)
    train_df = add_train_rul(train_df, rul_cap)

    train_units = [int(x) for x in split_json["train_units"]]
    val_units = [int(x) for x in split_json["val_units"]]

    tr_split = train_df[train_df["unit_id"].isin(train_units)].copy()
    val_split = train_df[train_df["unit_id"].isin(val_units)].copy()

    mean, std = fit_normalizer(tr_split, feats)
    val_split = apply_normalizer(val_split, feats, mean, std)

    X_val, y_val, meta_val = build_windows(val_split, feats, seq_len, mode="all")
    val_ds = SequenceDataset(X_val, y_val, meta_val)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(group, cfg_json, input_size=len(feats)).to(device)
    state = torch.load(run_dir / "best_model.pth", map_location=device)
    model.load_state_dict(state)

    if group in {"G2", "G4"}:
        mc_samples = (
            int(cfg.mc_samples_override)
            if cfg.mc_samples_override is not None
            else int(cfg_json.get("mc_samples_val", 30))
        )
        y_true, pred_mean, pred_std, metas = mc_dropout_predict(
            model=model,
            loader=val_loader,
            device=device,
            n_samples=mc_samples,
        )
        pred_df = pd.DataFrame(metas)
        pred_df["y_true"] = pd.Series(y_true).astype(float)
        pred_df["decision_rul"] = pd.Series(pred_mean).astype(float)
        pred_df["pred_mean"] = pd.Series(pred_mean).astype(float)
        pred_df["pred_std"] = pd.Series(pred_std).astype(float)
    else:
        y_true, y_pred, metas = predict_loader(model, val_loader, device)
        pred_df = pd.DataFrame(metas)
        pred_df["y_true"] = pd.Series(y_true).astype(float)
        pred_df["decision_rul"] = pd.Series(y_pred).astype(float)
        pred_df["y_pred"] = pd.Series(y_pred).astype(float)

    pred_df = infer_meta_columns(pred_df)
    pred_df = pred_df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    return pred_df


def simulate_unit_trajectory(
    unit_df: pd.DataFrame,
    threshold: float,
    preventive_cost: float,
    failure_cost: float,
    early_cost_per_cycle: float,
) -> Dict[str, Any]:
    work = unit_df.sort_values("cycle").reset_index(drop=True)
    true_cross = work[work["y_true"] <= threshold]
    pred_cross = work[work["decision_rul"] <= threshold]

    true_cross_exists = len(true_cross) > 0
    pred_cross_exists = len(pred_cross) > 0

    true_cross_cycle = int(true_cross.iloc[0]["cycle"]) if true_cross_exists else None
    true_cross_rul = float(true_cross.iloc[0]["y_true"]) if true_cross_exists else None

    pred_cross_cycle = int(pred_cross.iloc[0]["cycle"]) if pred_cross_exists else None
    pred_cross_rul = (
        float(pred_cross.iloc[0]["decision_rul"]) if pred_cross_exists else None
    )
    pred_cross_true_rul = (
        float(pred_cross.iloc[0]["y_true"]) if pred_cross_exists else None
    )

    if pred_cross_exists and (
        not true_cross_exists or pred_cross_cycle <= true_cross_cycle
    ):
        action = "preventive"
        preventive_action = 1
        failure_event = 0
        unnecessary_maintenance = int(
            (not true_cross_exists) or (pred_cross_cycle < true_cross_cycle)
        )
        protected_danger = int(
            true_cross_exists and pred_cross_cycle <= true_cross_cycle
        )
        action_cycle = pred_cross_cycle
        action_true_rul = pred_cross_true_rul
        action_pred_rul = pred_cross_rul
        early_lead = max(float(action_true_rul) - threshold, 0.0)
        maintenance_cost = preventive_cost + early_cost_per_cycle * early_lead
        failure_penalty = 0.0
    elif true_cross_exists:
        action = "failure"
        preventive_action = 0
        failure_event = 1
        unnecessary_maintenance = 0
        protected_danger = 0
        action_cycle = true_cross_cycle
        action_true_rul = true_cross_rul
        action_pred_rul = None
        maintenance_cost = 0.0
        failure_penalty = failure_cost
        early_lead = 0.0
    else:
        action = "no_action"
        preventive_action = 0
        failure_event = 0
        unnecessary_maintenance = 0
        protected_danger = 0
        action_cycle = None
        action_true_rul = None
        action_pred_rul = None
        maintenance_cost = 0.0
        failure_penalty = 0.0
        early_lead = 0.0

    total_cost = maintenance_cost + failure_penalty

    return {
        "unit_id": int(work.iloc[0]["unit_id"]),
        "threshold": float(threshold),
        "num_windows": int(len(work)),
        "first_cycle": int(work["cycle"].min()),
        "last_cycle": int(work["cycle"].max()),
        "last_true_rul": float(work.iloc[-1]["y_true"]),
        "last_decision_rul": float(work.iloc[-1]["decision_rul"]),
        "true_cross_exists": int(true_cross_exists),
        "pred_cross_exists": int(pred_cross_exists),
        "true_cross_cycle": true_cross_cycle,
        "pred_cross_cycle": pred_cross_cycle,
        "true_cross_rul": true_cross_rul,
        "pred_cross_rul": pred_cross_rul,
        "pred_cross_true_rul": pred_cross_true_rul,
        "action": action,
        "action_cycle": action_cycle,
        "action_true_rul": action_true_rul,
        "action_pred_rul": action_pred_rul,
        "preventive_action": preventive_action,
        "protected_danger": protected_danger,
        "failure_event": failure_event,
        "unnecessary_maintenance": unnecessary_maintenance,
        "early_lead": float(early_lead),
        "maintenance_cost": float(maintenance_cost),
        "failure_cost": float(failure_penalty),
        "total_cost": float(total_cost),
    }


def evaluate_run(
    run_dir: Path,
    group: str,
    cfg: Config,
    device: str,
) -> List[Dict[str, Any]]:
    traj_df = build_validation_trajectory_predictions(run_dir, group, cfg, device)
    rows: List[Dict[str, Any]] = []
    for threshold in cfg.thresholds:
        for _, unit_df in traj_df.groupby("unit_id", sort=True):
            row = simulate_unit_trajectory(
                unit_df,
                threshold=threshold,
                preventive_cost=cfg.preventive_cost,
                failure_cost=cfg.failure_cost,
                early_cost_per_cycle=cfg.early_cost_per_cycle,
            )
            row["run_dir"] = run_dir.name
            rows.append(row)
    return rows


def aggregate_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in df.columns
        if c not in {"run_dir", "action"} and pd.api.types.is_numeric_dtype(df[c])
    ]

    rows: List[Dict[str, Any]] = []
    for threshold, sub in df.groupby("threshold", sort=True):
        row: Dict[str, Any] = {
            "threshold": float(threshold),
            "num_runs": int(sub["run_dir"].nunique()),
            "num_units": int(len(sub) / max(sub["run_dir"].nunique(), 1)),
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
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def build_group_summary(agg_df: pd.DataFrame, group: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "group": group,
        "num_thresholds": int(len(agg_df)),
    }

    if len(agg_df) == 0:
        summary["best_threshold"] = None
        return summary

    best_idx = pd.to_numeric(agg_df["total_cost_mean"], errors="coerce").idxmin()
    best_row = agg_df.loc[best_idx]

    summary["best_threshold"] = float(best_row["threshold"])
    summary["best_total_cost_mean"] = float(best_row["total_cost_mean"])
    summary["best_total_cost_std"] = float(best_row["total_cost_std"])
    summary["best_failure_event_mean"] = float(best_row["failure_event_mean"])
    summary["best_failure_event_std"] = float(best_row["failure_event_std"])
    summary["best_unnecessary_maintenance_mean"] = float(
        best_row["unnecessary_maintenance_mean"]
    )
    summary["best_unnecessary_maintenance_std"] = float(
        best_row["unnecessary_maintenance_std"]
    )
    summary["best_preventive_action_mean"] = float(best_row["preventive_action_mean"])
    summary["best_preventive_action_std"] = float(best_row["preventive_action_std"])
    summary["best_protected_danger_mean"] = float(best_row["protected_danger_mean"])
    summary["best_protected_danger_std"] = float(best_row["protected_danger_std"])
    summary["best_early_lead_mean"] = float(best_row["early_lead_mean"])
    summary["best_early_lead_std"] = float(best_row["early_lead_std"])

    return summary


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(Config(), args)
    device = select_device()

    print("[INFO] Round 5 uses validation full trajectories from split_units.json")
    print("[INFO] Maintenance simulation is run on complete run-to-failure validation units")
    print("[INFO] Test split is not used in this version")

    out_root = _PROJECT_ROOT / "outputs_v2" / "round5_maintenance"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "round5_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg)}, f, indent=2)

    for group in cfg.groups:
        root = group_root(group)
        if not root.exists():
            print(f"[WARN] Missing group output dir for {group}: {root}")
            continue

        run_rows: List[Dict[str, Any]] = []
        for run_dir in sorted(root.iterdir()):
            if run_dir.is_dir() and (run_dir / "best_model.pth").exists():
                print(
                    f"[INFO] Building validation full-trajectory policy evaluation for {group} | {run_dir.name}"
                )
                run_rows.extend(evaluate_run(run_dir, group, cfg, device))

        if not run_rows:
            print(f"[WARN] No valid runs found for {group}")
            continue

        run_df = pd.DataFrame(run_rows)
        agg_df = aggregate_thresholds(run_df)
        summary = build_group_summary(agg_df, group)

        group_dir = out_root / group
        group_dir.mkdir(parents=True, exist_ok=True)

        detail_csv = group_dir / "round5_detail.csv"
        threshold_summary_csv = group_dir / "round5_threshold_summary.csv"
        summary_json = group_dir / "round5_summary.json"

        run_df.to_csv(detail_csv, index=False)
        agg_df.to_csv(threshold_summary_csv, index=False)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"=== Round 5 Maintenance: {group} ===")
        print(agg_df)
        print("\nSummary:")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print("\nSaved files:")
        print(f"outputs_v2/round5_maintenance/{group}/{detail_csv.name}")
        print(f"outputs_v2/round5_maintenance/{group}/{threshold_summary_csv.name}")
        print(f"outputs_v2/round5_maintenance/{group}/{summary_json.name}")
        print()

    print("Saved files:")
    print("outputs_v2/round5_maintenance/round5_manifest.json")


if __name__ == "__main__":
    main()
