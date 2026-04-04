from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

import pandas as pd
import torch
from torch.utils.data import DataLoader


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src_v2.rul.constants import (
    DEFAULT_RUL_CAP,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_VALIDATION_RATIO,
    get_feature_columns,
)
from src_v2.rul.dataset import (
    SequenceDataset,
    read_cmapss_split,
    add_train_rul,
    add_test_rul,
    split_train_val_by_unit,
    fit_normalizer,
    apply_normalizer,
    normalizer_to_dict,
    build_windows,
)
from src_v2.rul.models.cnn_bilstm import CNNBiLSTM
from src_v2.rul.training.loss import LinExLoss
from src_v2.rul.training.trainer import Trainer
from src_v2.rul.evaluation.metrics import rmse, mae, nasa_score
from src_v2.rul.utils import ensure_dir, set_seed, select_device


@dataclass
class Config:
    experiment_name: str = "G3_CNN_BiLSTM_LinEx_v2"
    dataset_name: str = "FD001"
    seed: int = 42

    data_dir: str = "CMAPSSData"
    output_root: str = "outputs/G3_CNN_BiLSTM_LinEx"

    seq_len: int = DEFAULT_SEQUENCE_LENGTH
    rul_cap: float = DEFAULT_RUL_CAP
    validation_ratio: float = DEFAULT_VALIDATION_RATIO

    hidden_size: int = 20
    num_lstm_layers: int = 2
    dense_size: int = 100
    cnn_out_channels: int = 128
    cnn_kernel_size: int = 5
    cnn_stride: int = 1
    cnn_pool_size: int = 1

    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-3
    num_epochs: int = 100
    early_stopping_patience: int = 15
    grad_clip: float = 5.0

    linex_a: float = 0.04


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def resolve_path_from_project(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((_PROJECT_ROOT / path).resolve())


def to_serializable_float(value) -> float:
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run G3 experiment")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--rul-cap", type=float, default=None)
    parser.add_argument("--validation-ratio", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-lstm-layers", type=int, default=None)
    parser.add_argument("--dense-size", type=int, default=None)
    parser.add_argument("--cnn-out-channels", type=int, default=None)
    parser.add_argument("--cnn-kernel-size", type=int, default=None)
    parser.add_argument("--cnn-stride", type=int, default=None)
    parser.add_argument("--cnn-pool-size", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--linex-a", type=float, default=None)
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.seed is not None:
        cfg.seed = args.seed
    if args.dataset is not None:
        cfg.dataset_name = args.dataset
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.output_root is not None:
        cfg.output_root = args.output_root
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.rul_cap is not None:
        cfg.rul_cap = args.rul_cap
    if args.validation_ratio is not None:
        cfg.validation_ratio = args.validation_ratio
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.early_stopping_patience is not None:
        cfg.early_stopping_patience = args.early_stopping_patience
    if args.hidden_size is not None:
        cfg.hidden_size = args.hidden_size
    if args.num_lstm_layers is not None:
        cfg.num_lstm_layers = args.num_lstm_layers
    if args.dense_size is not None:
        cfg.dense_size = args.dense_size
    if args.cnn_out_channels is not None:
        cfg.cnn_out_channels = args.cnn_out_channels
    if args.cnn_kernel_size is not None:
        cfg.cnn_kernel_size = args.cnn_kernel_size
    if args.cnn_stride is not None:
        cfg.cnn_stride = args.cnn_stride
    if args.cnn_pool_size is not None:
        cfg.cnn_pool_size = args.cnn_pool_size
    if args.grad_clip is not None:
        cfg.grad_clip = args.grad_clip
    if args.linex_a is not None:
        cfg.linex_a = args.linex_a
    return cfg


def main():
    args = parse_args()
    cfg = apply_overrides(Config(), args)
    raw_data_dir = cfg.data_dir
    raw_output_root = cfg.output_root
    cfg.data_dir = resolve_path_from_project(cfg.data_dir)
    cfg.output_root = resolve_path_from_project(cfg.output_root)

    run_dir = os.path.join(cfg.output_root, f"seed_{cfg.seed}_a_{cfg.linex_a:.2f}")
    ensure_dir(run_dir)

    set_seed(cfg.seed)
    device = select_device()

    feats = get_feature_columns()

    train_df, test_df, rul_df = read_cmapss_split(cfg.data_dir, cfg.dataset_name)
    train_df = add_train_rul(train_df, cfg.rul_cap)
    test_df = add_test_rul(test_df, rul_df, cfg.rul_cap)

    tr_split, val_split, train_units, val_units = split_train_val_by_unit(
        train_df,
        validation_ratio=cfg.validation_ratio,
        seed=cfg.seed,
    )

    mean, std = fit_normalizer(tr_split, feats)
    tr_split = apply_normalizer(tr_split, feats, mean, std)
    val_split = apply_normalizer(val_split, feats, mean, std)
    test_df = apply_normalizer(test_df, feats, mean, std)

    X_train, y_train, meta_train = build_windows(
        tr_split, feats, cfg.seq_len, mode="all"
    )
    X_val, y_val, meta_val = build_windows(val_split, feats, cfg.seq_len, mode="all")
    X_test, y_test, meta_test = build_windows(test_df, feats, cfg.seq_len, mode="last")

    train_ds = SequenceDataset(X_train, y_train, meta_train)
    val_ds = SequenceDataset(X_val, y_val, meta_val)
    test_ds = SequenceDataset(X_test, y_test, meta_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = CNNBiLSTM(
        input_size=len(feats),
        cnn_out_channels=cfg.cnn_out_channels,
        cnn_kernel_size=cfg.cnn_kernel_size,
        cnn_stride=cfg.cnn_stride,
        cnn_pool_size=cfg.cnn_pool_size,
        hidden_size=cfg.hidden_size,
        num_lstm_layers=cfg.num_lstm_layers,
        dense_size=cfg.dense_size,
    ).to(device)

    criterion = LinExLoss(a=cfg.linex_a)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        grad_clip=cfg.grad_clip,
    )

    best_val_rmse = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate_loss(val_loader)
        val_true, val_pred, _ = trainer.predict(val_loader)

        val_rmse = rmse(val_true, val_pred)
        val_mae = mae(val_true, val_pred)
        val_nasa = nasa_score(val_true, val_pred)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_nasa": val_nasa,
            "linex_a": cfg.linex_a,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_rmse={val_rmse:.4f} "
            f"val_mae={val_mae:.4f} "
            f"val_nasa={val_nasa:.4f} "
            f"linex_a={cfg.linex_a:.2f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pth"))

    if best_state is None:
        raise RuntimeError(
            "best_state is None. Training did not produce a valid checkpoint."
        )

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))

    val_true, val_pred, _ = trainer.predict(val_loader)
    test_true, test_pred, test_meta = trainer.predict(test_loader)

    val_metrics = {
        "rmse": to_serializable_float(rmse(val_true, val_pred)),
        "mae": to_serializable_float(mae(val_true, val_pred)),
        "nasa": to_serializable_float(nasa_score(val_true, val_pred)),
    }

    test_metrics = {
        "rmse": to_serializable_float(rmse(test_true, test_pred)),
        "mae": to_serializable_float(mae(test_true, test_pred)),
        "nasa": to_serializable_float(nasa_score(test_true, test_pred)),
    }

    pred_df = pd.DataFrame(test_meta)
    pred_df["y_true"] = pd.Series(test_true).astype(float)
    pred_df["y_pred"] = pd.Series(test_pred).astype(float)
    pred_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

    pd.DataFrame(history).to_csv(
        os.path.join(run_dir, "train_history.csv"), index=False
    )

    config_to_save = asdict(cfg)
    config_to_save["data_dir"] = raw_data_dir
    config_to_save["output_root"] = raw_output_root
    save_json(os.path.join(run_dir, "config.json"), config_to_save)
    save_json(
        os.path.join(run_dir, "normalizer.json"), normalizer_to_dict(feats, mean, std)
    )
    save_json(
        os.path.join(run_dir, "split_units.json"),
        {
            "train_units": sorted(int(x) for x in train_units),
            "val_units": sorted(int(x) for x in val_units),
        },
    )
    save_json(os.path.join(run_dir, "val_metrics.json"), val_metrics)
    save_json(os.path.join(run_dir, "test_metrics.json"), test_metrics)
    save_json(
        os.path.join(run_dir, "best_summary.json"),
        {
            "best_epoch": int(best_epoch),
            "best_val_rmse": to_serializable_float(best_val_rmse),
        },
    )

    print("\nRun finished.")
    print(f"Device: {device}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val RMSE: {best_val_rmse:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"LinEx a: {cfg.linex_a:.2f}")
    print(f"Outputs saved to: {Path(run_dir)}")


if __name__ == "__main__":
    main()
