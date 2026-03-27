from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src_v2.rul.dataset import (
    SequenceDataset,
    add_test_rul,
    add_train_rul,
    apply_normalizer,
    build_windows,
    fit_normalizer,
    read_cmapss_split,
    split_train_val_by_unit,
)
from src_v2.rul.evaluation.metrics import mae, nasa_score, rmse
from src_v2.rul.utils import select_device


@dataclass
class Config:
    data_dir: str = "CMAPSSData"
    dataset: str = "FD001"
    output_root: str = "outputs_v2/round6_ablation"

    seeds: tuple[int, ...] = (7, 21, 42, 87, 123)

    seq_len: int = 50
    rul_cap: float = 125.0
    validation_ratio: float = 0.2

    hidden_size: int = 64
    num_lstm_layers: int = 2
    dense_size: int = 64

    cnn_out_channels: int = 64
    cnn_kernel_size: int = 5
    cnn_stride: int = 1
    cnn_pool_size: int = 1

    dropout_rate: float = 0.3

    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 40
    early_stopping_patience: int = 8
    grad_clip: float = 5.0

    linex_a: float = 0.04
    mc_samples_val: int = 10
    mc_samples_test: int = 20


ABLATION_SETTINGS = {
    "bilstm_mse": {
        "use_cnn": False,
        "use_mcd": False,
        "loss_name": "mse",
    },
    "cnn_bilstm_mse": {
        "use_cnn": True,
        "use_mcd": False,
        "loss_name": "mse",
    },
    "bilstm_linex_mcd": {
        "use_cnn": False,
        "use_mcd": True,
        "loss_name": "linex",
    },
    "cnn_bilstm_linex_mcd": {
        "use_cnn": True,
        "use_mcd": True,
        "loss_name": "linex",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round 6: minimal CNN ablation for BiLSTM-based RUL models"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(ABLATION_SETTINGS.keys()),
        choices=list(ABLATION_SETTINGS.keys()),
    )
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--data-dir", type=str, default="CMAPSSData")
    parser.add_argument("--output-root", type=str, default="outputs_v2/round6_ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42, 87, 123])
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--rul-cap", type=float, default=125.0)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-lstm-layers", type=int, default=2)
    parser.add_argument("--dense-size", type=int, default=64)
    parser.add_argument("--cnn-out-channels", type=int, default=64)
    parser.add_argument("--cnn-kernel-size", type=int, default=5)
    parser.add_argument("--cnn-stride", type=int, default=1)
    parser.add_argument("--cnn-pool-size", type=int, default=1)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--linex-a", type=float, default=0.04)
    parser.add_argument("--mc-samples-val", type=int, default=10)
    parser.add_argument("--mc-samples-test", type=int, default=20)
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.data_dir = args.data_dir
    cfg.dataset = args.dataset
    cfg.output_root = args.output_root
    cfg.seeds = tuple(args.seeds)
    cfg.seq_len = args.seq_len
    cfg.rul_cap = args.rul_cap
    cfg.validation_ratio = args.validation_ratio
    cfg.hidden_size = args.hidden_size
    cfg.num_lstm_layers = args.num_lstm_layers
    cfg.dense_size = args.dense_size
    cfg.cnn_out_channels = args.cnn_out_channels
    cfg.cnn_kernel_size = args.cnn_kernel_size
    cfg.cnn_stride = args.cnn_stride
    cfg.cnn_pool_size = args.cnn_pool_size
    cfg.dropout_rate = args.dropout_rate
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.num_epochs = args.num_epochs
    cfg.early_stopping_patience = args.early_stopping_patience
    cfg.grad_clip = args.grad_clip
    cfg.linex_a = args.linex_a
    cfg.mc_samples_val = args.mc_samples_val
    cfg.mc_samples_test = args.mc_samples_test
    return cfg


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


class BiLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_lstm_layers: int,
        dense_size: int,
        dropout_rate: float = 0.0,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout_rate if num_lstm_layers > 1 and use_dropout else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, dense_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity(),
            nn.Linear(dense_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = self.dropout(h)
        y = self.head(h).squeeze(-1)
        return y


class CNNBiLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_lstm_layers: int,
        dense_size: int,
        cnn_out_channels: int,
        cnn_kernel_size: int,
        cnn_stride: int,
        cnn_pool_size: int,
        dropout_rate: float = 0.0,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
            stride=cnn_stride,
            padding=cnn_kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.pool = (
            nn.Identity()
            if cnn_pool_size <= 1
            else nn.MaxPool1d(
                kernel_size=cnn_pool_size,
                stride=cnn_pool_size,
            )
        )

        lstm_dropout = dropout_rate if num_lstm_layers > 1 and use_dropout else 0.0
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, dense_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity(),
            nn.Linear(dense_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = self.dropout(h)
        y = self.head(h).squeeze(-1)
        return y


def build_model(input_size: int, cfg: Config, setting: Dict[str, Any]) -> nn.Module:
    if setting["use_cnn"]:
        return CNNBiLSTMRegressor(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_lstm_layers=cfg.num_lstm_layers,
            dense_size=cfg.dense_size,
            cnn_out_channels=cfg.cnn_out_channels,
            cnn_kernel_size=cfg.cnn_kernel_size,
            cnn_stride=cfg.cnn_stride,
            cnn_pool_size=cfg.cnn_pool_size,
            dropout_rate=cfg.dropout_rate,
            use_dropout=setting["use_mcd"],
        )
    return BiLSTMRegressor(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_lstm_layers=cfg.num_lstm_layers,
        dense_size=cfg.dense_size,
        dropout_rate=cfg.dropout_rate,
        use_dropout=setting["use_mcd"],
    )


def linex_loss(y_pred: torch.Tensor, y_true: torch.Tensor, a: float) -> torch.Tensor:
    diff = y_pred - y_true
    z = torch.clamp(a * diff, min=-50.0, max=50.0)
    return torch.mean(torch.exp(z) - z - 1.0)


def get_loss(setting: Dict[str, Any], linex_a: float):
    if setting["loss_name"] == "mse":
        return nn.MSELoss()
    if setting["loss_name"] == "linex":
        return lambda y_pred, y_true: linex_loss(y_pred, y_true, linex_a)
    raise ValueError(f"Unsupported loss_name: {setting['loss_name']}")


@torch.no_grad()
def predict_deterministic(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    force_eval: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    if force_eval:
        model.eval()
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []

    for x, y, meta in loader:
        x = x.to(device)
        out = model(x).detach().cpu().numpy()
        preds.append(out)
        trues.append(y.numpy())

        batch_size = len(y)
        for i in range(batch_size):
            row = {}
            for k, v in meta.items():
                item = v[i]
                if isinstance(item, torch.Tensor):
                    row[k] = item.item() if item.numel() == 1 else item.tolist()
                elif isinstance(item, np.generic):
                    row[k] = item.item()
                else:
                    row[k] = item
            metas.append(row)

    y_true = np.concatenate(trues) if trues else np.array([], dtype=float)
    y_pred = np.concatenate(preds) if preds else np.array([], dtype=float)
    return y_true, y_pred, metas


def enable_dropout_only(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
        elif isinstance(m, nn.LSTM):
            m.train()


@torch.no_grad()
def predict_mc_dropout(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    all_preds: List[np.ndarray] = []
    ref_true: np.ndarray | None = None
    ref_meta: List[Dict[str, Any]] | None = None

    for i in range(n_samples):
        model.eval()
        enable_dropout_only(model)
        y_true, y_pred, metas = predict_deterministic(
            model,
            loader,
            device,
            force_eval=False,
        )
        all_preds.append(y_pred)
        if i == 0:
            ref_true = y_true
            ref_meta = metas

    pred_stack = np.stack(all_preds, axis=0)
    pred_mean = pred_stack.mean(axis=0)
    pred_std = pred_stack.std(axis=0)
    assert ref_true is not None
    assert ref_meta is not None
    return ref_true, pred_mean, pred_std, ref_meta


def resolve_output_root(output_root: str) -> Path:
    p = Path(output_root)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / p).resolve()


def resolve_data_dir(data_dir: str) -> str:
    p = Path(data_dir)
    if p.is_absolute():
        return str(p)
    return str((_PROJECT_ROOT / p).resolve())


def prepare_data(
    cfg: Config,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, Any]]:
    data_dir = resolve_data_dir(cfg.data_dir)
    feats = [f"op_setting_{i}" for i in range(1, 4)] + [
        f"sensor_{i}" for i in range(1, 22)
    ]

    train_df, test_df, rul_df = read_cmapss_split(data_dir, cfg.dataset)
    train_df = add_train_rul(train_df, cfg.rul_cap)
    test_df = add_test_rul(test_df, rul_df, cfg.rul_cap)

    tr_split, val_split, train_units, val_units = split_train_val_by_unit(
        train_df,
        validation_ratio=cfg.validation_ratio,
        seed=seed,
    )

    mean, std = fit_normalizer(tr_split, feats)
    tr_split = apply_normalizer(tr_split, feats, mean, std)
    val_split = apply_normalizer(val_split, feats, mean, std)
    test_df = apply_normalizer(test_df, feats, mean, std)

    X_train, y_train, meta_train = build_windows(
        tr_split, feats, cfg.seq_len, mode="all"
    )
    X_val, y_val, meta_val = build_windows(val_split, feats, cfg.seq_len, mode="last")
    X_test, y_test, meta_test = build_windows(test_df, feats, cfg.seq_len, mode="last")

    train_ds = SequenceDataset(X_train, y_train, meta_train)
    val_ds = SequenceDataset(X_val, y_val, meta_val)
    test_ds = SequenceDataset(X_test, y_test, meta_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    extras = {
        "train_units": [int(x) for x in train_units],
        "val_units": [int(x) for x in val_units],
        "feature_columns": feats,
        "normalizer_mean": {k: float(v) for k, v in mean.items()},
        "normalizer_std": {k: float(v) for k, v in std.items()},
        "data_dir_resolved": data_dir,
    }
    return train_loader, val_loader, test_loader, feats, extras


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    use_mcd: bool,
    mc_samples: int,
) -> Dict[str, Any]:
    if use_mcd:
        y_true, pred_mean, pred_std, metas = predict_mc_dropout(
            model, loader, device, mc_samples
        )
        return {
            "y_true": y_true,
            "y_pred": pred_mean,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "metas": metas,
            "metrics": {
                "rmse": rmse(y_true, pred_mean),
                "mae": mae(y_true, pred_mean),
                "nasa": nasa_score(y_true, pred_mean),
                "pred_std_mean": float(np.mean(pred_std)),
                "pred_std_std": float(np.std(pred_std)),
                "mc_samples": int(mc_samples),
            },
        }

    y_true, y_pred, metas = predict_deterministic(model, loader, device)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "metas": metas,
        "metrics": {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "nasa": nasa_score(y_true, y_pred),
        },
    }


def train_one_run(
    experiment_name: str,
    setting: Dict[str, Any],
    cfg: Config,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    set_seed(seed)

    train_loader, val_loader, test_loader, feats, extras = prepare_data(cfg, seed)
    model = build_model(len(feats), cfg, setting).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = get_loss(setting, cfg.linex_a)

    run_dir = resolve_output_root(cfg.output_root) / experiment_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    history_rows: List[Dict[str, Any]] = []
    best_state = None
    best_epoch = None
    best_val_rmse = math.inf
    patience = 0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        epoch_losses: List[float] = []

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else math.nan

        val_eval = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            use_mcd=setting["use_mcd"],
            mc_samples=cfg.mc_samples_val,
        )
        val_metrics = val_eval["metrics"]

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss_proxy": train_loss,
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_nasa": val_metrics["nasa"],
        }
        if setting["use_mcd"]:
            history_row["val_pred_std_mean"] = val_metrics["pred_std_mean"]
            history_row["val_pred_std_std"] = val_metrics["pred_std_std"]
        history_rows.append(history_row)

        print(
            f"[{experiment_name}][seed={seed}] "
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_nasa={val_metrics['nasa']:.4f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if patience >= cfg.early_stopping_patience:
            print(f"[{experiment_name}][seed={seed}] early stopping at epoch {epoch}")
            break

    assert best_state is not None
    assert best_epoch is not None

    torch.save(best_state, run_dir / "best_model.pth")
    torch.save(model.state_dict(), run_dir / "last_model.pth")

    model.load_state_dict(best_state)

    val_eval = evaluate_model(
        model=model,
        loader=val_loader,
        device=device,
        use_mcd=setting["use_mcd"],
        mc_samples=cfg.mc_samples_val,
    )
    test_eval = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        use_mcd=setting["use_mcd"],
        mc_samples=cfg.mc_samples_test,
    )

    val_metrics = val_eval["metrics"]
    test_metrics = test_eval["metrics"]

    pred_df = pd.DataFrame(test_eval["metas"])
    pred_df["y_true"] = pd.Series(test_eval["y_true"]).astype(float)
    pred_df["y_pred"] = pd.Series(test_eval["y_pred"]).astype(float)
    if setting["use_mcd"]:
        pred_df["pred_mean"] = pd.Series(test_eval["pred_mean"]).astype(float)
        pred_df["pred_std"] = pd.Series(test_eval["pred_std"]).astype(float)
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(run_dir / "train_history.csv", index=False)

    config_payload = {
        "experiment_name": experiment_name,
        "dataset_name": cfg.dataset,
        "seed": seed,
        "data_dir": cfg.data_dir,
        "output_root": cfg.output_root,
        "seq_len": cfg.seq_len,
        "rul_cap": cfg.rul_cap,
        "validation_ratio": cfg.validation_ratio,
        "hidden_size": cfg.hidden_size,
        "num_lstm_layers": cfg.num_lstm_layers,
        "dense_size": cfg.dense_size,
        "cnn_out_channels": cfg.cnn_out_channels,
        "cnn_kernel_size": cfg.cnn_kernel_size,
        "cnn_stride": cfg.cnn_stride,
        "cnn_pool_size": cfg.cnn_pool_size,
        "dropout_rate": cfg.dropout_rate,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "num_epochs": cfg.num_epochs,
        "early_stopping_patience": cfg.early_stopping_patience,
        "grad_clip": cfg.grad_clip,
        "linex_a": cfg.linex_a,
        "mc_samples_val": cfg.mc_samples_val,
        "mc_samples_test": cfg.mc_samples_test,
        "use_cnn": bool(setting["use_cnn"]),
        "use_mcd": bool(setting["use_mcd"]),
        "loss_name": setting["loss_name"],
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    with open(run_dir / "split_units.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_units": extras["train_units"],
                "val_units": extras["val_units"],
            },
            f,
            indent=2,
        )

    with open(run_dir / "normalizer.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": extras["feature_columns"],
                "mean": extras["normalizer_mean"],
                "std": extras["normalizer_std"],
            },
            f,
            indent=2,
        )

    with open(run_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    with open(run_dir / "best_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_rmse": best_val_rmse,
            },
            f,
            indent=2,
        )

    return {
        "experiment_name": experiment_name,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_nasa": test_metrics["nasa"],
    }


def summarize_experiment(
    output_root: Path,
    experiment_name: str,
) -> Dict[str, Any]:
    exp_dir = output_root / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        test_metrics_path = run_dir / "test_metrics.json"
        best_summary_path = run_dir / "best_summary.json"
        if not test_metrics_path.exists() or not best_summary_path.exists():
            continue

        with open(test_metrics_path, "r", encoding="utf-8") as f:
            test_metrics = json.load(f)
        with open(best_summary_path, "r", encoding="utf-8") as f:
            best_summary = json.load(f)

        row = {
            "run_dir": run_dir.name,
            "rmse": test_metrics["rmse"],
            "mae": test_metrics["mae"],
            "nasa": test_metrics["nasa"],
            "best_epoch": best_summary["best_epoch"],
            "best_val_rmse": best_summary["best_val_rmse"],
        }
        if "pred_std_mean" in test_metrics:
            row["pred_std_mean"] = test_metrics["pred_std_mean"]
            row["pred_std_std"] = test_metrics["pred_std_std"]
            row["mc_samples"] = test_metrics["mc_samples"]
        rows.append(row)

    df = pd.DataFrame(rows)
    detail_path = exp_dir / "round6_detail.csv"
    df.to_csv(detail_path, index=False)

    summary: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "num_runs": int(len(df)),
    }

    if df.empty:
        summary["status"] = "no_completed_runs_found"
        with open(exp_dir / "round6_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    for col in ["rmse", "mae", "nasa", "best_val_rmse"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            summary[f"{col}_mean"] = float(s.mean()) if s.notna().any() else None
            summary[f"{col}_std"] = float(s.std(ddof=0)) if s.notna().any() else None

    if "pred_std_mean" in df.columns:
        s = pd.to_numeric(df["pred_std_mean"], errors="coerce")
        summary["pred_std_mean_mean"] = float(s.mean()) if s.notna().any() else None
        summary["pred_std_mean_std"] = float(s.std(ddof=0)) if s.notna().any() else None

    with open(exp_dir / "round6_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def compare_experiments(output_root: Path, experiments: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name in experiments:
        summary_path = output_root / name / "round6_summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    df = pd.DataFrame(rows)
    if not df.empty and "rmse_mean" in df.columns:
        df = df.sort_values(["rmse_mean", "experiment_name"]).reset_index(drop=True)

    df.to_csv(output_root / "round6_comparison.csv", index=False)
    return df


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(Config(), args)
    device = select_device()
    output_root = resolve_output_root(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(output_root / "round6_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {"config": asdict(cfg), "experiments": list(args.experiments)}, f, indent=2
        )

    for experiment_name in args.experiments:
        setting = ABLATION_SETTINGS[experiment_name]
        print(f"\n=== Running {experiment_name} ===")
        for seed in cfg.seeds:
            train_one_run(
                experiment_name=experiment_name,
                setting=setting,
                cfg=cfg,
                seed=seed,
                device=device,
            )
        summary = summarize_experiment(output_root, experiment_name)
        print(f"\nSummary for {experiment_name}:")
        for k, v in summary.items():
            print(f"{k}: {v}")
        if summary.get("status") == "no_completed_runs_found":
            print(
                f"[WARN] No completed runs were found under {output_root / experiment_name}. "
                "Please check whether training wrote test_metrics.json and best_summary.json."
            )

    compare_df = compare_experiments(output_root, list(args.experiments))

    print("\n=== Round 6 Comparison ===")
    if compare_df.empty:
        print("No completed experiment summaries found.")
    else:
        print(compare_df)

    print("\nSaved files:")
    print(f"{cfg.output_root}/round6_manifest.json")
    print(f"{cfg.output_root}/round6_comparison.csv")
    for experiment_name in args.experiments:
        print(f"{cfg.output_root}/{experiment_name}/round6_detail.csv")
        print(f"{cfg.output_root}/{experiment_name}/round6_summary.json")


if __name__ == "__main__":
    main()
