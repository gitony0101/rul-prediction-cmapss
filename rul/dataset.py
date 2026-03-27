from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src_v2.rul.constants import get_all_columns


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.meta: List[Dict] = meta.to_dict("records")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.meta[idx]


def _read_cmapss_table(path: Path) -> pd.DataFrame:
    cols = get_all_columns()
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < len(cols):
        raise ValueError(
            f"File {path} has only {df.shape[1]} columns, expected at least {len(cols)}"
        )
    df = df.iloc[:, : len(cols)].copy()
    df.columns = cols
    return df


def read_cmapss_split(data_dir: str, dataset_name: str):
    root = Path(data_dir)
    train_path = root / f"train_{dataset_name}.txt"
    test_path = root / f"test_{dataset_name}.txt"
    rul_path = root / f"RUL_{dataset_name}.txt"

    missing = [str(p) for p in [train_path, test_path, rul_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing C-MAPSS files: {missing}")

    train_df = _read_cmapss_table(train_path)
    test_df = _read_cmapss_table(test_path)

    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, engine="python")
    rul_df = rul_df.iloc[:, :1].copy()
    rul_df.columns = ["RUL"]

    return train_df, test_df, rul_df


def add_train_rul(train_df: pd.DataFrame, rul_cap: float) -> pd.DataFrame:
    df = train_df.copy()
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL"] = (max_cycle - df["cycle"]).clip(lower=0, upper=rul_cap).astype(float)
    return df


def add_test_rul(
    test_df: pd.DataFrame, rul_df: pd.DataFrame, rul_cap: float
) -> pd.DataFrame:
    df = test_df.copy()
    unit_max = df.groupby("unit_id")["cycle"].max().sort_index()
    unit_ids = unit_max.index.tolist()

    if len(rul_df) != len(unit_ids):
        raise ValueError(
            f"RUL file length {len(rul_df)} does not match number of test units {len(unit_ids)}"
        )

    final_rul_map = {
        unit_id: float(rul_df.iloc[i, 0]) for i, unit_id in enumerate(unit_ids)
    }
    true_end_cycle = df["unit_id"].map(lambda u: unit_max.loc[u] + final_rul_map[u])
    df["RUL"] = (
        (true_end_cycle - df["cycle"]).clip(lower=0, upper=rul_cap).astype(float)
    )
    return df


def split_train_val_by_unit(
    train_df: pd.DataFrame,
    validation_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    units = sorted(train_df["unit_id"].unique().tolist())
    if len(units) < 2:
        raise ValueError("Need at least 2 training units to create a validation split")

    rng = np.random.default_rng(seed)
    shuffled = units.copy()
    rng.shuffle(shuffled)

    val_count = int(round(len(shuffled) * validation_ratio))
    val_count = max(1, min(len(shuffled) - 1, val_count))

    val_units = sorted(int(x) for x in shuffled[:val_count])
    train_units = sorted(int(x) for x in shuffled[val_count:])

    tr_split = train_df[train_df["unit_id"].isin(train_units)].copy()
    val_split = train_df[train_df["unit_id"].isin(val_units)].copy()

    return tr_split, val_split, train_units, val_units


def fit_normalizer(train_df: pd.DataFrame, feature_cols: List[str]):
    mean = train_df[feature_cols].mean(axis=0)
    std = train_df[feature_cols].std(axis=0, ddof=0).replace(0.0, 1.0)
    return mean, std


def apply_normalizer(
    df: pd.DataFrame,
    feature_cols: List[str],
    mean: pd.Series,
    std: pd.Series,
) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = (out[feature_cols] - mean) / std
    return out


def normalizer_to_dict(
    feature_cols: List[str], mean: pd.Series, std: pd.Series
) -> Dict:
    return {
        "features": list(feature_cols),
        "mean": {c: float(mean[c]) for c in feature_cols},
        "std": {c: float(std[c]) for c in feature_cols},
    }


def _window_with_left_padding(
    values: np.ndarray, end_idx: int, seq_len: int
) -> np.ndarray:
    start_idx = end_idx - seq_len + 1
    if start_idx >= 0:
        return values[start_idx : end_idx + 1]

    pad_len = -start_idx
    pad_block = np.repeat(values[[0]], pad_len, axis=0)
    return np.concatenate([pad_block, values[0 : end_idx + 1]], axis=0)


def build_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    mode: str = "all",
):
    if mode not in {"all", "last"}:
        raise ValueError(f"Unsupported window mode: {mode}")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    meta_rows: List[Dict] = []

    for unit_id, unit_df in df.groupby("unit_id", sort=True):
        unit_df = unit_df.sort_values("cycle").reset_index(drop=True)
        feat_values = unit_df[feature_cols].to_numpy(dtype=np.float32)
        rul_values = unit_df["RUL"].to_numpy(dtype=np.float32)
        cycle_values = unit_df["cycle"].to_numpy()

        end_indices = range(len(unit_df)) if mode == "all" else [len(unit_df) - 1]

        for end_idx in end_indices:
            X_list.append(_window_with_left_padding(feat_values, end_idx, seq_len))
            y_list.append(float(rul_values[end_idx]))
            meta_rows.append(
                {
                    "unit_id": int(unit_id),
                    "cycle": int(cycle_values[end_idx]),
                    "window_end_cycle": int(cycle_values[end_idx]),
                    "window_mode": mode,
                }
            )

    X = (
        np.stack(X_list, axis=0)
        if X_list
        else np.empty((0, seq_len, len(feature_cols)), dtype=np.float32)
    )
    y = np.asarray(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)

    return X, y, meta
