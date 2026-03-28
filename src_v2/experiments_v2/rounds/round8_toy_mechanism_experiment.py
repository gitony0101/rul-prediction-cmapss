from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("round8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT_ROOT = str(PROJECT_ROOT / "outputs_v2" / "round8_toy_mechanism")
LEGACY_OUTPUT_ROOT = (
    Path(__file__).resolve().parent / "outputs_v2" / "round8_toy_mechanism"
)


# =========================================================
# Config
# =========================================================
@dataclass
class GeneratorConfig:
    train_units: int = 200
    val_units: int = 40
    test_units: int = 40
    min_cycles: int = 120
    max_cycles: int = 320
    rul_cap: float = 125.0
    n_sensors: int = 6
    base_seed: int = 42

    # latent health process
    h0_mean: float = 1.0
    h0_std: float = 0.02
    h0_min: float = 0.94
    h0_max: float = 1.06
    lambda_min: float = 0.0015
    lambda_max: float = 0.0045
    gamma_min: float = 1.05
    gamma_max: float = 1.35

    # operating conditions
    omega1: float = 0.03
    omega2: float = 0.015
    baseline_std: float = 0.3
    drift_std: float = 0.05

    # sensor coefficients
    s3_a1: float = 0.4
    s3_c1: float = 0.8
    s4_c2: float = 0.5
    s4_c3: float = 0.7
    s5_a2: float = 0.3
    s5_c4: float = 0.6

    # scenario sigma levels are relative multipliers on clean sensor std
    sigma_scale_map: Dict[str, float] = field(
        default_factory=lambda: {
            "sigma_1": 0.10,
            "sigma_3": 0.30,
            "sigma_5": 0.50,
        }
    )


@dataclass
class TrainConfig:
    output_root: str = DEFAULT_OUTPUT_ROOT
    sequence_length: int = 50
    batch_size: int = 256
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_size: int = 64
    lstm_layers: int = 2
    dropout: float = 0.20
    cnn_channels: int = 64
    cnn_kernel_size: int = 5
    dense_size: int = 64
    grad_clip: float = 5.0
    mcd_samples: int = 50
    linex_a: float = 0.04
    seed_list: Tuple[int, ...] = (0, 1, 2)
    num_workers: int = 0


@dataclass
class MaintenanceConfig:
    output_root: str = DEFAULT_OUTPUT_ROOT
    fleet_size: int = 20
    maintenance_capacity: int = 4
    mission_horizon: int = 30
    decision_rounds: int = 40
    maintenance_cost: float = 1.0
    early_cost: float = 2.0
    failure_cost: float = 20.0
    maintenance_reset_to_start: bool = True
    base_seed: int = 123


DEFAULT_SCENARIOS: Tuple[str, ...] = (
    "A_sigma_1",
    "A_sigma_3",
    "A_sigma_5",
    "B_sigma_1_3",
    "B_sigma_1_5",
    "B_sigma_3_5",
)
MODEL_GROUPS: Tuple[str, ...] = ("G1", "G2", "G3", "G4")
GROUP_TO_LOSS = {"G1": "mse", "G2": "mse", "G3": "linex", "G4": "linex"}
GROUP_TO_MCD = {"G1": False, "G2": True, "G3": False, "G4": True}

DEFAULT_LINEX_A_LIST: Tuple[float, ...] = (0.01, 0.02, 0.04, 0.08)

SCENARIO_TO_NOISE_LEVEL = {
    "A_sigma_1": "low",
    "A_sigma_3": "mid",
    "A_sigma_5": "high",
    "B_sigma_1_3": "mixed_low_mid",
    "B_sigma_1_5": "mixed_low_high",
    "B_sigma_3_5": "mixed_mid_high",
}


# =========================================================
# Utilities
# =========================================================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        LOGGER.info("Selected device: mps")
        return torch.device("mps")
    if torch.cuda.is_available():
        LOGGER.info("Selected device: cuda")
        return torch.device("cuda")
    LOGGER.info("Selected device: cpu")
    return torch.device("cpu")


DEVICE = select_device()
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# =======================
# History helpers
# =======================


def make_timestamp_id() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")


def make_history_run_dir(group_dir: Path, stage: str) -> Path:
    history_root = group_dir / f"{stage}_history"
    ensure_dir(history_root)
    run_dir = history_root / make_timestamp_id()
    ensure_dir(run_dir)
    return run_dir


def write_csv_with_history(
    df: pd.DataFrame, latest_path: Path, history_dir: Path
) -> None:
    ensure_dir(latest_path.parent)
    df.to_csv(latest_path, index=False)
    df.to_csv(history_dir / latest_path.name, index=False)


def write_json_with_history(obj: dict, latest_path: Path, history_dir: Path) -> None:
    save_json(obj, latest_path)
    save_json(obj, history_dir / latest_path.name)


def write_torch_with_history(obj: dict, latest_path: Path, history_dir: Path) -> None:
    ensure_dir(latest_path.parent)
    torch.save(obj, latest_path)
    torch.save(obj, history_dir / latest_path.name)


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_a_tag(a: Optional[float]) -> str:
    if a is None:
        return "a_none"
    a_str = f"{a:.6f}".rstrip("0").rstrip(".")
    return f"a_{a_str.replace('.', 'p')}"


def parse_a_tag(a_tag: str) -> Optional[float]:
    if a_tag == "a_none":
        return None
    if not a_tag.startswith("a_"):
        raise ValueError(f"Unsupported a tag: {a_tag}")
    return float(a_tag[2:].replace("p", "."))


def get_group_a_values(
    group: str, linex_a: float, linex_a_list: Optional[Sequence[float]]
) -> List[Optional[float]]:
    if GROUP_TO_LOSS[group] != "linex":
        return [None]
    if linex_a_list:
        return [float(a) for a in linex_a_list]
    return [float(linex_a)]


def discover_a_tags_for_group_scenario(
    output_root: Path, group: str, scenario: str
) -> List[str]:
    scenario_dir = output_root / group / scenario
    if not scenario_dir.exists():
        return []
    discovered = sorted(
        child.name
        for child in scenario_dir.iterdir()
        if child.is_dir() and child.name.startswith("a_")
    )
    if discovered:
        return discovered
    legacy_seed_dirs = [
        child
        for child in scenario_dir.iterdir()
        if child.is_dir() and child.name.startswith("seed_")
    ]
    if legacy_seed_dirs:
        return ["a_none"]
    return []


# ---------------------------------------------------------
# Scenario/Noise utilities and file logging helpers
# ---------------------------------------------------------
def get_noise_level_label(scenario: str) -> str:
    return SCENARIO_TO_NOISE_LEVEL.get(scenario, "unknown")


def has_completed_training_outputs(group_dir: Path) -> bool:
    return (
        (group_dir / "metrics.json").exists()
        and (group_dir / "test_predictions.csv").exists()
        and (group_dir / "best_model.pt").exists()
    )


def has_completed_maintenance_outputs(group_dir: Path) -> bool:
    return (group_dir / "maintenance_metrics.json").exists() and (
        group_dir / "maintenance_round_logs.csv"
    ).exists()


def setup_file_logging(output_root: Path, action: str) -> Path:
    logs_dir = output_root / "logs"
    ensure_dir(logs_dir)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"round8_{action}_{timestamp}.log"

    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    root_logger.addHandler(file_handler)
    return log_path


# ---------------------------------------------------------
# Round 8 legacy output helpers
# ---------------------------------------------------------
def has_round8_outputs(output_root: Path) -> bool:
    if not output_root.exists():
        return False
    if (output_root / "synthetic_data").exists():
        return True
    return any((output_root / group).exists() for group in MODEL_GROUPS)


def resolve_output_root(path_str: str, action: str) -> Path:
    requested_root = Path(path_str)
    if requested_root.exists():
        return requested_root

    if str(requested_root) == DEFAULT_OUTPUT_ROOT and has_round8_outputs(
        LEGACY_OUTPUT_ROOT
    ):
        LOGGER.warning(
            "Requested output root %s does not exist. Falling back to legacy Round 8 output root %s for action=%s.",
            requested_root,
            LEGACY_OUTPUT_ROOT,
            action,
        )
        return LEGACY_OUTPUT_ROOT

    return requested_root


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = y_pred - y_true
    score = np.where(
        err < 0,
        np.exp(-err / 13.0) - 1.0,
        np.exp(err / 10.0) - 1.0,
    )
    return float(np.sum(score))


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted.")
        return (x - self.mean_) / self.std_

    def state_dict(self) -> dict:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler is not fitted.")
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict) -> "StandardScaler":
        scaler = cls()
        scaler.mean_ = np.asarray(state["mean"], dtype=np.float32)
        scaler.std_ = np.asarray(state["std"], dtype=np.float32)
        return scaler


# =========================================================
# Synthetic data generation
# =========================================================
def scenario_to_sigma_values(name: str, cfg: GeneratorConfig) -> Tuple[float, ...]:
    if name.startswith("A_"):
        key = name.replace("A_", "")
        return (cfg.sigma_scale_map[key],)
    if name.startswith("B_"):
        _, key1, key2 = name.split("_")
        return (
            cfg.sigma_scale_map[f"sigma_{key1}"],
            cfg.sigma_scale_map[f"sigma_{key2}"],
        )
    raise ValueError(f"Unsupported scenario: {name}")


class SyntheticScenarioGenerator:
    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg

    def _sample_engine_meta(
        self,
        unit_id: int,
        split: str,
        sigma_values: Tuple[float, ...],
        rng: np.random.Generator,
    ) -> dict:
        h0 = float(
            np.clip(
                rng.normal(self.cfg.h0_mean, self.cfg.h0_std),
                self.cfg.h0_min,
                self.cfg.h0_max,
            )
        )
        lam = float(rng.uniform(self.cfg.lambda_min, self.cfg.lambda_max))
        gamma = float(rng.uniform(self.cfg.gamma_min, self.cfg.gamma_max))
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        baseline = float(rng.normal(0.0, self.cfg.baseline_std))

        if len(sigma_values) == 1:
            noise_group = "single"
            sigma_scale = float(sigma_values[0])
        else:
            idx = int(rng.integers(0, len(sigma_values)))
            noise_group = f"group_{idx}"
            sigma_scale = float(sigma_values[idx])

        return {
            "unit_id": unit_id,
            "split": split,
            "h0": h0,
            "lambda": lam,
            "gamma": gamma,
            "phi": phi,
            "baseline_b": baseline,
            "noise_group": noise_group,
            "sigma_scale": sigma_scale,
        }

    def _build_clean_engine(
        self, meta: dict, rng: np.random.Generator
    ) -> Optional[pd.DataFrame]:
        h0 = meta["h0"]
        lam = meta["lambda"]
        gamma = meta["gamma"]
        phi = meta["phi"]
        baseline = meta["baseline_b"]

        rows = []
        t = 0
        h_prev = h0
        failure_time: Optional[int] = None
        while True:
            h = h0 - lam * (t**gamma)
            if h <= 0.0:
                failure_time = t
                break

            o1 = math.sin(self.cfg.omega1 * t + phi)
            o2 = math.cos(self.cfg.omega2 * t + phi)
            o3 = baseline + float(rng.normal(0.0, self.cfg.drift_std))
            dh = h - h_prev if t > 0 else 0.0

            x1 = h
            x2 = dh
            x3 = self.cfg.s3_a1 * o1 + self.cfg.s3_c1 * h
            x4 = self.cfg.s4_c2 * h + self.cfg.s4_c3 * (h**2)
            x5 = self.cfg.s5_a2 * o2 + self.cfg.s5_c4 * (h**1.5)
            x6 = 0.15 * o3

            rows.append(
                {
                    "cycle": t + 1,
                    "health": h,
                    "op_1": o1,
                    "op_2": o2,
                    "op_3": o3,
                    "sensor_1": x1,
                    "sensor_2": x2,
                    "sensor_3": x3,
                    "sensor_4": x4,
                    "sensor_5": x5,
                    "sensor_6": x6,
                }
            )
            h_prev = h
            t += 1
            if t > self.cfg.max_cycles + 5:
                break

        if failure_time is None:
            return None

        if not (self.cfg.min_cycles <= failure_time <= self.cfg.max_cycles):
            return None

        df = pd.DataFrame(rows)
        df["true_rul"] = failure_time - df["cycle"].to_numpy() + 1
        df["true_rul"] = df["true_rul"].clip(lower=0.0)
        df["true_rul_capped"] = df["true_rul"].clip(upper=self.cfg.rul_cap)
        df["failure_time"] = failure_time
        return df

    def _apply_noise(
        self, clean_df: pd.DataFrame, sigma_scale: float, rng: np.random.Generator
    ) -> pd.DataFrame:
        sensor_cols = [f"sensor_{i}" for i in range(1, self.cfg.n_sensors + 1)]
        noisy_df = clean_df.copy()
        clean_values = noisy_df[sensor_cols].to_numpy(dtype=np.float32)
        sensor_std = clean_values.std(axis=0)
        sensor_std[sensor_std < 1e-6] = 1.0
        noise = rng.normal(0.0, sigma_scale, size=clean_values.shape).astype(
            np.float32
        ) * sensor_std.reshape(1, -1)
        noisy_df[sensor_cols] = clean_values + noise
        noisy_df["noise_std_reference"] = sigma_scale
        return noisy_df

    def generate_scenario(self, scenario: str, output_root: Path, seed: int) -> None:
        scenario_dir = output_root / "synthetic_data" / scenario
        ensure_dir(scenario_dir)
        cfg = self.cfg
        rng = np.random.default_rng(
            cfg.base_seed + seed + sum(ord(c) for c in scenario)
        )
        sigma_values = scenario_to_sigma_values(scenario, cfg)

        split_sizes = {
            "train": cfg.train_units,
            "val": cfg.val_units,
            "test": cfg.test_units,
        }

        all_frames: Dict[str, List[pd.DataFrame]] = {k: [] for k in split_sizes}
        engine_meta_rows: List[dict] = []
        global_unit_id = 1

        for split, n_units in split_sizes.items():
            created = 0
            trials = 0
            while created < n_units:
                meta = self._sample_engine_meta(
                    global_unit_id, split, sigma_values, rng
                )
                clean_df = self._build_clean_engine(meta, rng)
                trials += 1
                if clean_df is None:
                    if trials > 20000:
                        raise RuntimeError(
                            f"Too many retries while generating {scenario} {split}"
                        )
                    continue

                noisy_df = self._apply_noise(clean_df, meta["sigma_scale"], rng)
                noisy_df["unit_id"] = global_unit_id
                noisy_df["split"] = split
                noisy_df["noise_group"] = meta["noise_group"]
                noisy_df["scenario_id"] = scenario
                noisy_df["seed"] = seed
                all_frames[split].append(noisy_df)

                meta_row = dict(meta)
                meta_row["failure_time"] = int(clean_df["failure_time"].iloc[0])
                meta_row["scenario_id"] = scenario
                meta_row["seed"] = seed
                engine_meta_rows.append(meta_row)

                global_unit_id += 1
                created += 1

        for split, frames in all_frames.items():
            split_df = pd.concat(frames, ignore_index=True)
            split_df.to_csv(scenario_dir / f"{split}.csv", index=False)

        engine_meta_df = pd.DataFrame(engine_meta_rows)
        engine_meta_df.to_csv(scenario_dir / "engine_meta.csv", index=False)
        config_payload = {
            "scenario": scenario,
            "seed": seed,
            "generator_config": asdict(cfg),
            "sigma_values": list(sigma_values),
        }
        save_json(config_payload, scenario_dir / "scenario_config.json")
        LOGGER.info("Generated scenario %s at %s", scenario, scenario_dir)


# =========================================================
# Window dataset
# =========================================================
class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, meta: pd.DataFrame):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], idx


def build_windows(
    df: pd.DataFrame,
    sequence_length: int,
    sensor_cols: Sequence[str],
    target_col: str = "true_rul_capped",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    windows: List[np.ndarray] = []
    targets: List[float] = []
    meta_rows: List[dict] = []
    for unit_id, unit_df in df.groupby("unit_id"):
        unit_df = unit_df.sort_values("cycle").reset_index(drop=True)
        values = unit_df[sensor_cols].to_numpy(dtype=np.float32)
        targets_arr = unit_df[target_col].to_numpy(dtype=np.float32)
        for end_idx in range(sequence_length - 1, len(unit_df)):
            start_idx = end_idx - sequence_length + 1
            windows.append(values[start_idx : end_idx + 1])
            targets.append(float(targets_arr[end_idx]))
            row = unit_df.iloc[end_idx]
            meta_rows.append(
                {
                    "unit_id": int(unit_id),
                    "cycle": int(row["cycle"]),
                    "true_rul": float(row["true_rul"]),
                    "true_rul_capped": float(row["true_rul_capped"]),
                    "health": float(row["health"]),
                    "noise_group": row["noise_group"],
                    "scenario_id": row["scenario_id"],
                    "seed": int(row["seed"]),
                }
            )
    x = np.stack(windows, axis=0)
    y = np.asarray(targets, dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)
    return x, y, meta_df


def fit_scaler_from_train(
    train_df: pd.DataFrame, sensor_cols: Sequence[str]
) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[list(sensor_cols)].to_numpy(dtype=np.float32))
    return scaler


def apply_scaler(
    df: pd.DataFrame, scaler: StandardScaler, sensor_cols: Sequence[str]
) -> pd.DataFrame:
    out = df.copy()
    out[list(sensor_cols)] = scaler.transform(
        out[list(sensor_cols)].to_numpy(dtype=np.float32)
    )
    return out


# =========================================================
# Model
# =========================================================
class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cnn_channels: int,
        cnn_kernel_size: int,
        hidden_size: int,
        lstm_layers: int,
        dropout: float,
        dense_size: int,
    ) -> None:
        super().__init__()
        padding = cnn_kernel_size // 2
        self.conv1 = nn.Conv1d(
            input_dim, cnn_channels, kernel_size=cnn_kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, dense_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B, T, C]
        seq_out, _ = self.lstm(x)
        last = seq_out[:, -1, :]
        pred = self.head(last).squeeze(-1)
        return pred


class LinExLoss(nn.Module):
    def __init__(self, a: float = 0.04):
        super().__init__()
        self.a = a

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = pred - target
        val = torch.exp(self.a * err) - self.a * err - 1.0
        return val.mean()


# =========================================================
# Train and inference
# =========================================================
def build_dataloader(
    dataset: SequenceDataset, batch_size: int, shuffle: bool, num_workers: int
) -> DataLoader:
    pin_memory = DEVICE.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    grad_clip: float,
) -> float:
    model.train()
    losses = []
    for x, y, _ in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.detach().item())
    return float(np.mean(losses))


@torch.no_grad()
def predict_deterministic(
    model: nn.Module, loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    model.eval()
    preds, targets, indices = [], [], []
    for x, y, idx in loader:
        x = x.to(DEVICE)
        pred = model(x).detach().cpu().numpy()
        preds.append(pred)
        targets.append(y.numpy())
        indices.extend(idx.tolist())
    return np.concatenate(preds), np.concatenate(targets), indices


@torch.no_grad()
def predict_with_mcd(
    model: nn.Module, loader: DataLoader, samples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    # keep dropout active, batchnorm frozen in eval mode
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    mean_preds, std_preds, targets, indices = [], [], [], []
    for x, y, idx in loader:
        x = x.to(DEVICE)
        mc_preds = []
        for _ in range(samples):
            pred = model(x).detach().cpu().numpy()
            mc_preds.append(pred)
        arr = np.stack(mc_preds, axis=0)
        mean_preds.append(arr.mean(axis=0))
        std_preds.append(arr.std(axis=0))
        targets.append(y.numpy())
        indices.extend(idx.tolist())
    return (
        np.concatenate(mean_preds),
        np.concatenate(std_preds),
        np.concatenate(targets),
        indices,
    )


@torch.no_grad()
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nasa": nasa_score(y_true, y_pred),
        "bias": float(np.mean(err)),
        "over_rate": float(np.mean(err > 0.0)),
        "danger_over_5": float(np.mean(err > 5.0)),
        "danger_over_10": float(np.mean(err > 10.0)),
    }


def create_model(input_dim: int, cfg: TrainConfig) -> CNNBiLSTM:
    model = CNNBiLSTM(
        input_dim=input_dim,
        cnn_channels=cfg.cnn_channels,
        cnn_kernel_size=cfg.cnn_kernel_size,
        hidden_size=cfg.hidden_size,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
        dense_size=cfg.dense_size,
    )
    return model.to(DEVICE)


def load_scenario_frames(output_root: Path, scenario: str) -> Dict[str, pd.DataFrame]:
    scenario_dir = output_root / "synthetic_data" / scenario
    return {
        "train": pd.read_csv(scenario_dir / "train.csv"),
        "val": pd.read_csv(scenario_dir / "val.csv"),
        "test": pd.read_csv(scenario_dir / "test.csv"),
        "engine_meta": pd.read_csv(scenario_dir / "engine_meta.csv"),
    }


def run_train_for_group_scenario_seed(
    group: str,
    scenario: str,
    seed: int,
    cfg: TrainConfig,
    linex_a_override: Optional[float] = None,
) -> None:
    set_all_seeds(seed)
    output_root = Path(cfg.output_root)
    effective_linex_a: Optional[float]
    if linex_a_override is not None:
        effective_linex_a = float(linex_a_override)
    elif GROUP_TO_LOSS[group] == "linex":
        effective_linex_a = float(cfg.linex_a)
    else:
        effective_linex_a = None
    a_tag = format_a_tag(effective_linex_a)
    group_dir = output_root / group / scenario / a_tag / f"seed_{seed}"
    ensure_dir(group_dir)
    train_run_dir = make_history_run_dir(group_dir, "train")

    frames = load_scenario_frames(output_root, scenario)
    sensor_cols = [f"sensor_{i}" for i in range(1, 7)]
    scaler = fit_scaler_from_train(frames["train"], sensor_cols)
    train_df = apply_scaler(frames["train"], scaler, sensor_cols)
    val_df = apply_scaler(frames["val"], scaler, sensor_cols)
    test_df = apply_scaler(frames["test"], scaler, sensor_cols)

    x_train, y_train, meta_train = build_windows(
        train_df, cfg.sequence_length, sensor_cols
    )
    x_val, y_val, meta_val = build_windows(val_df, cfg.sequence_length, sensor_cols)
    x_test, y_test, meta_test = build_windows(test_df, cfg.sequence_length, sensor_cols)

    train_ds = SequenceDataset(x_train, y_train, meta_train)
    val_ds = SequenceDataset(x_val, y_val, meta_val)
    test_ds = SequenceDataset(x_test, y_test, meta_test)

    train_loader = build_dataloader(train_ds, cfg.batch_size, True, cfg.num_workers)
    val_loader = build_dataloader(val_ds, cfg.batch_size, False, cfg.num_workers)
    test_loader = build_dataloader(test_ds, cfg.batch_size, False, cfg.num_workers)

    model = create_model(input_dim=len(sensor_cols), cfg=cfg)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    if GROUP_TO_LOSS[group] == "mse":
        loss_fn: nn.Module = nn.MSELoss()
    else:
        if effective_linex_a is None:
            raise RuntimeError("LinEx group requires a valid linex_a value.")
        loss_fn = LinExLoss(a=effective_linex_a)

    best_state = None
    best_val = float("inf")
    history = []

    LOGGER.info(
        "Training %s | %s | %s | seed=%s",
        group,
        scenario,
        a_tag,
        seed,
    )
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, cfg.grad_clip
        )
        val_pred, val_true, _ = predict_deterministic(model, val_loader)
        val_metrics = evaluate_predictions(val_true, val_pred)
        metric_to_select = val_metrics["rmse"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        LOGGER.info(
            "[%s|%s|%s|seed=%s] epoch=%d train_loss=%.4f val_rmse=%.4f val_bias=%.4f",
            group,
            scenario,
            a_tag,
            seed,
            epoch,
            train_loss,
            val_metrics["rmse"],
            val_metrics["bias"],
        )
        if metric_to_select < best_val:
            best_val = metric_to_select
            best_state = copy.deepcopy(model.state_dict())

    best_epoch = int(min(history, key=lambda row: row["val_rmse"])["epoch"])
    if best_state is None:
        raise RuntimeError("No best model state captured during training.")
    model.load_state_dict(best_state)

    if GROUP_TO_MCD[group]:
        test_pred_mean, test_pred_std, test_true, idx = predict_with_mcd(
            model, test_loader, cfg.mcd_samples
        )
    else:
        test_pred_mean, test_true, idx = predict_deterministic(model, test_loader)
        test_pred_std = np.zeros_like(test_pred_mean)

    test_metrics = evaluate_predictions(test_true, test_pred_mean)
    test_metrics["pred_std_mean"] = float(np.mean(test_pred_std))
    test_metrics["pred_std_p90"] = float(np.quantile(test_pred_std, 0.90))

    order = np.argsort(np.asarray(idx))
    pred_df = meta_test.iloc[order].reset_index(drop=True).copy()
    pred_df["pred_mean"] = test_pred_mean[order]
    pred_df["pred_std"] = test_pred_std[order]
    pred_df["error"] = pred_df["pred_mean"] - pred_df["true_rul_capped"]
    pred_df["group"] = group
    pred_df["loss_name"] = GROUP_TO_LOSS[group]
    pred_df["use_mcd"] = GROUP_TO_MCD[group]

    train_history_df = pd.DataFrame(history)
    metrics_payload = {
        "group": group,
        "scenario": scenario,
        "seed": seed,
        "a_tag": a_tag,
        "linex_a": effective_linex_a,
        "best_epoch": best_epoch,
        "best_val_rmse": float(best_val),
        "train_run_id": train_run_dir.name,
        "train_config": asdict(cfg),
        "metrics": test_metrics,
        "group_settings": {
            "loss": GROUP_TO_LOSS[group],
            "use_mcd": GROUP_TO_MCD[group],
        },
        "scaler": scaler.state_dict(),
    }

    write_csv_with_history(
        train_history_df, group_dir / "train_history.csv", train_run_dir
    )
    write_csv_with_history(pred_df, group_dir / "test_predictions.csv", train_run_dir)
    write_json_with_history(metrics_payload, group_dir / "metrics.json", train_run_dir)
    write_torch_with_history(best_state, group_dir / "best_model.pt", train_run_dir)

    latest_train_pointer = {
        "latest_train_run_id": train_run_dir.name,
        "latest_train_run_dir": str(train_run_dir),
    }
    write_json_with_history(
        latest_train_pointer,
        group_dir / "latest_train_run.json",
        train_run_dir,
    )
    LOGGER.info("Saved outputs to %s (history run: %s)", group_dir, train_run_dir)


# =========================================================
# Maintenance simulator
# =========================================================
def survival_probability(mean: float, std: float, mission_horizon: int) -> float:
    if std <= 1e-8:
        return 1.0 if mean > mission_horizon else 0.0
    z = (mission_horizon - mean) / std
    return float(1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def run_maintenance_for_prediction_table(
    pred_df: pd.DataFrame, cfg: MaintenanceConfig, rng: np.random.Generator
) -> dict:
    units = sorted(pred_df["unit_id"].unique().tolist())
    if len(units) < cfg.fleet_size:
        raise ValueError("Not enough units in prediction table to form a fleet.")

    fleet_units = rng.choice(units, size=cfg.fleet_size, replace=False).tolist()
    fail_count = 0
    early_count = 0
    maint_count = 0
    utilization = []
    round_logs = []

    # Each unit state is represented by a current cycle pointer.
    max_cycle_by_unit = pred_df.groupby("unit_id")["cycle"].max().to_dict()
    min_cycle_by_unit = pred_df.groupby("unit_id")["cycle"].min().to_dict()
    current_cycle = {
        unit: int(
            rng.integers(
                min_cycle_by_unit[unit],
                max(
                    min_cycle_by_unit[unit] + 1,
                    max_cycle_by_unit[unit] - cfg.mission_horizon,
                ),
            )
        )
        for unit in fleet_units
    }

    for round_idx in range(1, cfg.decision_rounds + 1):
        candidates = []
        for unit in fleet_units:
            sub = pred_df[
                (pred_df["unit_id"] == unit) & (pred_df["cycle"] == current_cycle[unit])
            ]
            if sub.empty:
                nearest_idx = (
                    (pred_df[pred_df["unit_id"] == unit]["cycle"] - current_cycle[unit])
                    .abs()
                    .idxmin()
                )
                row = pred_df.loc[nearest_idx]
            else:
                row = sub.iloc[0]
            mean = float(row["pred_mean"])
            std = float(row["pred_std"])
            p_survive = survival_probability(mean, std, cfg.mission_horizon)
            risk = 1.0 - p_survive
            candidates.append(
                {
                    "unit_id": unit,
                    "cycle": int(current_cycle[unit]),
                    "true_rul": float(row["true_rul"]),
                    "pred_mean": mean,
                    "pred_std": std,
                    "risk": risk,
                    "survival_prob": p_survive,
                }
            )

        candidates_df = (
            pd.DataFrame(candidates)
            .sort_values("risk", ascending=False)
            .reset_index(drop=True)
        )
        to_maintain = candidates_df.head(cfg.maintenance_capacity).copy()
        maintain_ids = set(to_maintain["unit_id"].tolist())
        utilization.append(len(maintain_ids) / cfg.maintenance_capacity)
        maint_count += len(maintain_ids)
        early_count += int((to_maintain["true_rul"] > cfg.mission_horizon).sum())

        for unit in fleet_units:
            if unit in maintain_ids and cfg.maintenance_reset_to_start:
                current_cycle[unit] = min_cycle_by_unit[unit]
                continue
            current_cycle[unit] += cfg.mission_horizon
            if current_cycle[unit] > max_cycle_by_unit[unit]:
                fail_count += 1
                current_cycle[unit] = min_cycle_by_unit[unit]

        round_logs.append(
            {
                "round": round_idx,
                "maintained_units": sorted(maintain_ids),
                "avg_risk": float(candidates_df["risk"].mean()),
                "avg_survival_prob": float(candidates_df["survival_prob"].mean()),
            }
        )

    total_cost = (
        cfg.maintenance_cost * maint_count
        + cfg.early_cost * early_count
        + cfg.failure_cost * fail_count
    )

    return {
        "fleet_size": cfg.fleet_size,
        "maintenance_capacity": cfg.maintenance_capacity,
        "mission_horizon": cfg.mission_horizon,
        "decision_rounds": cfg.decision_rounds,
        "maint_count": int(maint_count),
        "early_count": int(early_count),
        "fail_count": int(fail_count),
        "total_cost": float(total_cost),
        "maintenance_utilization": float(np.mean(utilization) if utilization else 0.0),
        "round_logs": round_logs,
    }


def run_maintenance_for_group_scenario_seed(
    group: str,
    scenario: str,
    seed: int,
    cfg: MaintenanceConfig,
    a_tag: str = "a_none",
) -> None:
    output_root = Path(cfg.output_root)
    if a_tag == "a_none":
        legacy_group_dir = output_root / group / scenario / f"seed_{seed}"
        group_dir = (
            legacy_group_dir
            if legacy_group_dir.exists()
            else output_root / group / scenario / a_tag / f"seed_{seed}"
        )
    else:
        group_dir = output_root / group / scenario / a_tag / f"seed_{seed}"
    maintenance_run_dir = make_history_run_dir(group_dir, "maintenance")
    pred_path = group_dir / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    pred_df = pd.read_csv(pred_path)
    rng = np.random.default_rng(
        cfg.base_seed + seed + sum(ord(c) for c in f"{group}_{scenario}_{a_tag}")
    )
    result = run_maintenance_for_prediction_table(pred_df, cfg, rng)
    result.update(
        {
            "group": group,
            "scenario": scenario,
            "seed": seed,
            "a_tag": a_tag,
            "linex_a": parse_a_tag(a_tag),
            "maintenance_run_id": maintenance_run_dir.name,
        }
    )
    maintenance_logs_df = pd.DataFrame(result["round_logs"])
    write_json_with_history(
        result, group_dir / "maintenance_metrics.json", maintenance_run_dir
    )
    write_csv_with_history(
        maintenance_logs_df,
        group_dir / "maintenance_round_logs.csv",
        maintenance_run_dir,
    )
    latest_maintenance_pointer = {
        "latest_maintenance_run_id": maintenance_run_dir.name,
        "latest_maintenance_run_dir": str(maintenance_run_dir),
    }
    write_json_with_history(
        latest_maintenance_pointer,
        group_dir / "latest_maintenance_run.json",
        maintenance_run_dir,
    )
    LOGGER.info(
        "Maintenance saved to %s (history run: %s)",
        group_dir,
        maintenance_run_dir,
    )


# =========================================================
# Summary
# =========================================================
def summarize_round8(output_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    for group in MODEL_GROUPS:
        group_dir = output_root / group
        if not group_dir.exists():
            continue
        for scenario_dir in sorted(group_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            a_tag_dirs = [
                child
                for child in sorted(scenario_dir.iterdir())
                if child.is_dir() and child.name.startswith("a_")
            ]
            if not a_tag_dirs:
                legacy_seed_dirs = [
                    child
                    for child in sorted(scenario_dir.iterdir())
                    if child.is_dir() and child.name.startswith("seed_")
                ]
                if legacy_seed_dirs:
                    a_tag_dirs = [scenario_dir]
                else:
                    continue

            for a_dir in a_tag_dirs:
                if a_dir == scenario_dir:
                    a_tag = "a_none"
                    seed_dirs = [
                        child
                        for child in sorted(scenario_dir.iterdir())
                        if child.is_dir() and child.name.startswith("seed_")
                    ]
                else:
                    a_tag = a_dir.name
                    seed_dirs = [
                        child
                        for child in sorted(a_dir.iterdir())
                        if child.is_dir() and child.name.startswith("seed_")
                    ]

                for seed_dir in seed_dirs:
                    metrics_path = seed_dir / "metrics.json"
                    maintenance_path = seed_dir / "maintenance_metrics.json"
                    if not metrics_path.exists():
                        continue
                    metrics = load_json(metrics_path)
                    row = {
                        "group": group,
                        "scenario": scenario_dir.name,
                        "a_tag": metrics.get("a_tag", a_tag),
                        "linex_a": metrics.get("linex_a"),
                        "seed": seed_dir.name.replace("seed_", ""),
                        "train_run_id": metrics.get("train_run_id"),
                        **metrics["metrics"],
                    }
                    if maintenance_path.exists():
                        maint = load_json(maintenance_path)
                        row.update(
                            {
                                "maintenance_run_id": maint.get("maintenance_run_id"),
                                "maint_count": maint["maint_count"],
                                "early_count": maint["early_count"],
                                "fail_count": maint["fail_count"],
                                "total_cost": maint["total_cost"],
                                "maintenance_utilization": maint[
                                    "maintenance_utilization"
                                ],
                            }
                        )
                    detail_rows.append(row)

    if not detail_rows:
        raise RuntimeError("No Round 8 outputs found to summarize.")

    detail_df = pd.DataFrame(detail_rows)
    detail_df["seed"] = detail_df["seed"].astype(str)
    detail_df = detail_df.sort_values(
        ["scenario", "group", "a_tag", "seed"]
    ).reset_index(drop=True)

    group_cols = ["scenario", "group", "a_tag", "linex_a"]
    non_numeric_cols = set(
        group_cols
        + [
            "seed",
            "train_run_id",
            "maintenance_run_id",
        ]
    )
    numeric_cols = [c for c in detail_df.columns if c not in non_numeric_cols]
    if not numeric_cols:
        raise RuntimeError("No numeric columns found for Round 8 summary aggregation.")

    summary_mean_df = detail_df.groupby(group_cols, dropna=False, as_index=False)[
        numeric_cols
    ].mean()
    seed_count_df = (
        detail_df.groupby(group_cols, dropna=False, as_index=False)["seed"]
        .nunique()
        .rename(columns={"seed": "n_seeds"})
    )
    summary_df = summary_mean_df.merge(seed_count_df, on=group_cols, how="left")

    if int(seed_count_df["n_seeds"].max()) > 1:
        summary_std_df = (
            detail_df.groupby(group_cols, dropna=False, as_index=False)[numeric_cols]
            .std()
            .add_suffix("_std")
        )
        summary_std_df = summary_std_df.rename(
            columns={
                "scenario_std": "scenario",
                "group_std": "group",
                "a_tag_std": "a_tag",
                "linex_a_std": "linex_a",
            }
        )
        summary_df = summary_df.merge(summary_std_df, on=group_cols, how="left")

    return detail_df, summary_df


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round 8 synthetic mechanism experiment"
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["generate", "train", "maintenance", "summary"],
    )
    parser.add_argument("--scenario", nargs="*", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--group", nargs="*", default=list(MODEL_GROUPS))
    parser.add_argument("--seed", nargs="*", type=int, default=None)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--linex_a", type=float, default=0.04)
    parser.add_argument(
        "--linex_a_list",
        nargs="*",
        type=float,
        default=None,
        help="Optional list of LinEx a values for scanning. Only applies to G3/G4.",
    )
    parser.add_argument("--mcd_samples", type=int, default=50)
    parser.add_argument("--fleet_size", type=int, default=20)
    parser.add_argument("--maintenance_capacity", type=int, default=4)
    parser.add_argument("--mission_horizon", type=int, default=30)
    parser.add_argument("--decision_rounds", type=int, default=40)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run actions even if the expected output files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = resolve_output_root(args.output_root, args.action)
    if args.action != "summary":
        ensure_dir(output_root)
    log_path = setup_file_logging(output_root, args.action)
    LOGGER.info("File logging enabled at %s", log_path)

    if args.action == "generate":
        gen_cfg = GeneratorConfig()
        generator = SyntheticScenarioGenerator(gen_cfg)
        for scenario in args.scenario:
            generator.generate_scenario(scenario, output_root, seed=gen_cfg.base_seed)
        manifest = {
            "action": "generate",
            "scenarios": args.scenario,
            "generator_config": asdict(gen_cfg),
            "default_linex_a_list": list(DEFAULT_LINEX_A_LIST),
        }
        save_json(manifest, output_root / "round8_manifest.json")
        return

    if args.action == "train":
        train_cfg = TrainConfig(
            output_root=str(output_root),
            epochs=args.epochs,
            batch_size=args.batch_size,
            linex_a=args.linex_a,
            mcd_samples=args.mcd_samples,
        )
        seed_list = list(args.seed) if args.seed else list(train_cfg.seed_list)
        for scenario in args.scenario:
            for group in args.group:
                a_values = get_group_a_values(group, args.linex_a, args.linex_a_list)
                for a_value in a_values:
                    for seed in seed_list:
                        run_train_for_group_scenario_seed(
                            group,
                            scenario,
                            seed,
                            train_cfg,
                            linex_a_override=a_value,
                        )
        return

    if args.action == "maintenance":
        maintenance_cfg = MaintenanceConfig(
            output_root=str(output_root),
            fleet_size=args.fleet_size,
            maintenance_capacity=args.maintenance_capacity,
            mission_horizon=args.mission_horizon,
            decision_rounds=args.decision_rounds,
        )
        seed_list = list(args.seed) if args.seed else [0, 1, 2]
        for scenario in args.scenario:
            for group in args.group:
                if args.linex_a_list is not None:
                    a_tags = [
                        format_a_tag(a)
                        for a in get_group_a_values(
                            group, args.linex_a, args.linex_a_list
                        )
                    ]
                else:
                    discovered_a_tags = discover_a_tags_for_group_scenario(
                        output_root, group, scenario
                    )
                    if discovered_a_tags:
                        a_tags = discovered_a_tags
                    else:
                        a_tags = [
                            format_a_tag(a)
                            for a in get_group_a_values(group, args.linex_a, None)
                        ]
                for a_tag in a_tags:
                    for seed in seed_list:
                        run_maintenance_for_group_scenario_seed(
                            group,
                            scenario,
                            seed,
                            maintenance_cfg,
                            a_tag=a_tag,
                        )
        return

    if args.action == "summary":
        if not has_round8_outputs(output_root):
            raise RuntimeError(
                f"No Round 8 outputs found to summarize under: {output_root}"
            )
        detail_df, summary_df = summarize_round8(output_root)
        cost_comparison_df = build_round8_cost_comparison_table(summary_df)

        detail_csv_path = output_root / "round8_detail.csv"
        summary_csv_path = output_root / "round8_summary.csv"
        cost_csv_path = output_root / "round8_cost_comparison.csv"
        summary_xlsx_path = output_root / "round8_summary.xlsx"

        detail_df.to_csv(detail_csv_path, index=False)
        summary_df.to_csv(summary_csv_path, index=False)
        cost_comparison_df.to_csv(cost_csv_path, index=False)

        excel_saved = False
        try:
            with pd.ExcelWriter(summary_xlsx_path, engine="openpyxl") as writer:
                detail_df.to_excel(writer, sheet_name="detail", index=False)
                summary_df.to_excel(writer, sheet_name="summary", index=False)
                cost_comparison_df.to_excel(
                    writer, sheet_name="cost_comparison", index=False
                )
            excel_saved = True
        except ModuleNotFoundError as exc:
            if exc.name != "openpyxl":
                raise
            LOGGER.warning(
                "openpyxl is not installed. Skipping Excel export and saving CSV files only."
            )

        if excel_saved:
            LOGGER.info(
                "Saved %s, %s, %s, and %s",
                detail_csv_path,
                summary_csv_path,
                cost_csv_path,
                summary_xlsx_path,
            )
        else:
            LOGGER.info(
                "Saved %s, %s, and %s",
                detail_csv_path,
                summary_csv_path,
                cost_csv_path,
            )
        print(summary_df.fillna("").to_string(index=False))
        return

    raise ValueError(f"Unsupported action: {args.action}")


def build_round8_cost_comparison_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "total_cost" not in summary_df.columns:
        return pd.DataFrame(
            columns=["scenario", "noise_level", "G1", "G2", "G3_best", "G4_best"]
        )

    rows = []
    for scenario in sorted(summary_df["scenario"].dropna().unique().tolist()):
        scenario_df = summary_df[summary_df["scenario"] == scenario].copy()
        row = {
            "scenario": scenario,
            "noise_level": get_noise_level_label(str(scenario)),
            "G1": "",
            "G2": "",
            "G3_best": "",
            "G4_best": "",
        }

        for group, column_name in [
            ("G1", "G1"),
            ("G2", "G2"),
            ("G3", "G3_best"),
            ("G4", "G4_best"),
        ]:
            group_df = scenario_df[scenario_df["group"] == group].copy()
            if group_df.empty:
                continue

            sort_cols = [
                c for c in ["total_cost", "fail_count", "rmse"] if c in group_df.columns
            ]
            best_row = group_df.sort_values(sort_cols, ascending=True).iloc[0]
            total_cost = float(best_row["total_cost"])
            linex_a = best_row.get("linex_a")

            if group in {"G3", "G4"} and pd.notna(linex_a):
                row[column_name] = f"{total_cost:.2f} (a={float(linex_a):.2f})"
            else:
                row[column_name] = f"{total_cost:.2f}"

        rows.append(row)

    return pd.DataFrame(rows)


# ================================
# Output root resolution utilities
# ================================
LEGACY_OUTPUT_ROOT = (
    Path(DEFAULT_OUTPUT_ROOT).parent / "outputs" / "round8_toy_mechanism"
)


def has_round8_outputs(output_root: Path) -> bool:
    if not output_root.exists():
        return False

    if (output_root / "round8_detail.csv").exists() or (
        output_root / "round8_summary.csv"
    ).exists():
        return True

    if (output_root / "synthetic_data").exists():
        synthetic_dir = output_root / "synthetic_data"
        if any(child.is_dir() for child in synthetic_dir.iterdir()):
            return True

    for group in MODEL_GROUPS:
        group_dir = output_root / group
        if not group_dir.exists():
            continue
        for scenario_dir in group_dir.iterdir():
            if not scenario_dir.is_dir():
                continue
            if any(
                child.is_dir() and child.name.startswith("seed_")
                for child in scenario_dir.iterdir()
            ):
                return True
            if any(
                child.is_dir() and child.name.startswith("a_")
                for child in scenario_dir.iterdir()
            ):
                return True

    return False


def resolve_output_root(path_str: str, action: str) -> Path:
    requested_root = Path(path_str)
    requested_has_outputs = has_round8_outputs(requested_root)

    if requested_has_outputs:
        return requested_root

    if str(requested_root) == DEFAULT_OUTPUT_ROOT and has_round8_outputs(
        LEGACY_OUTPUT_ROOT
    ):
        LOGGER.warning(
            "Requested output root %s has no Round 8 outputs. Falling back to legacy Round 8 output root %s for action=%s.",
            requested_root,
            LEGACY_OUTPUT_ROOT,
            action,
        )
        return LEGACY_OUTPUT_ROOT

    return requested_root


if __name__ == "__main__":
    main()
