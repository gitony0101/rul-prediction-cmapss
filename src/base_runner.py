from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.constants import (
    DEFAULT_RUL_CAP,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_VALIDATION_RATIO,
    get_feature_columns,
)
from src.dataset import (
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
from src.models.cnn_bilstm import CNNBiLSTM
from src.models.cnn_bilstm_dropout import CNNBiLSTMDropout
from src.training.loss import MSELoss, LinExLoss
from src.training.trainer import Trainer
from src.evaluation.metrics import rmse, mae, nasa_score
from src.utils import ensure_dir, set_seed, select_device


@dataclass
class Config:
    experiment_name: str = "Base_Experiment"
    dataset_name: str = "FD001"
    seed: int = 42

    data_dir: str = "CMAPSSData"
    output_root: str = "outputs/Base"

    # Loss and Uncertainty
    loss_type: str = "mse"  # "mse" or "linex"
    linex_a: float = 0.04
    use_mcd: bool = False
    dropout_rate: float = 0.5
    mc_samples_val: int = 10
    mc_samples_test: int = 20

    # Hyperparameters
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


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def to_serializable_float(value: Any) -> float:
    return float(value)


def run_experiment(cfg: Config, project_root: Path):
    raw_data_dir = cfg.data_dir
    raw_output_root = cfg.output_root
    
    # Resolve paths
    cfg.data_dir = str((project_root / cfg.data_dir).resolve())
    cfg.output_root = str((project_root / cfg.output_root).resolve())

    run_dir = os.path.join(cfg.output_root, f"seed_{cfg.seed}")
    if cfg.loss_type == "linex":
        run_dir = os.path.join(cfg.output_root, f"seed_{cfg.seed}_a_{cfg.linex_a}")
    
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

    X_train, y_train, meta_train = build_windows(tr_split, feats, cfg.seq_len, mode="all")
    X_val, y_val, meta_val = build_windows(val_split, feats, cfg.seq_len, mode="all")
    X_test, y_test, meta_test = build_windows(test_df, feats, cfg.seq_len, mode="last")

    train_ds = SequenceDataset(X_train, y_train, meta_train)
    val_ds = SequenceDataset(X_val, y_val, meta_val)
    test_ds = SequenceDataset(X_test, y_test, meta_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # Model Selection
    if cfg.use_mcd:
        model = CNNBiLSTMDropout(
            input_size=len(feats),
            cnn_out_channels=cfg.cnn_out_channels,
            cnn_kernel_size=cfg.cnn_kernel_size,
            cnn_stride=cfg.cnn_stride,
            cnn_pool_size=cfg.cnn_pool_size,
            hidden_size=cfg.hidden_size,
            num_lstm_layers=cfg.num_lstm_layers,
            dense_size=cfg.dense_size,
            dropout_rate=cfg.dropout_rate,
        ).to(device)
    else:
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

    # Loss Selection
    if cfg.loss_type == "linex":
        criterion = LinExLoss(a=cfg.linex_a)
    else:
        criterion = MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
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
        
        if cfg.use_mcd:
            val_loss = trainer.evaluate_loss_mcd(val_loader, num_samples=cfg.mc_samples_val)
            val_true, val_pred_mean, val_pred_std = trainer.predict_mcd(val_loader, num_samples=cfg.mc_samples_val)
            val_pred = val_pred_mean
        else:
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
        }
        history.append(row)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_rmse={val_rmse:.4f}")

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

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))

    # Final Inference
    if cfg.use_mcd:
        test_true, test_pred_mean, test_pred_std = trainer.predict_mcd(test_loader, num_samples=cfg.mc_samples_test)
        test_pred = test_pred_mean
        pred_std_mean = float(test_pred_std.mean())
    else:
        test_true, test_pred, _ = trainer.predict(test_loader)
        pred_std_mean = None

    test_metrics = {
        "rmse": to_serializable_float(rmse(test_true, test_pred)),
        "mae": to_serializable_float(mae(test_true, test_pred)),
        "nasa": to_serializable_float(nasa_score(test_true, test_pred)),
    }
    if pred_std_mean is not None:
        test_metrics["pred_std_mean"] = pred_std_mean

    # Save results
    save_json(os.path.join(run_dir, "test_metrics.json"), test_metrics)
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "train_history.csv"), index=False)
    
    config_to_save = asdict(cfg)
    config_to_save["data_dir"] = raw_data_dir
    config_to_save["output_root"] = raw_output_root
    save_json(os.path.join(run_dir, "config.json"), config_to_save)

    print(f"\nRun finished. Best epoch: {best_epoch}, Test RMSE: {test_metrics['rmse']:.4f}")
