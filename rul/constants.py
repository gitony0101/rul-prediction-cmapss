from typing import List, Tuple

DEFAULT_RUL_CAP: float = 125.0
DEFAULT_SEQUENCE_LENGTH: int = 50
DEFAULT_VALIDATION_RATIO: float = 0.2

CMAPSS_DATASETS: Tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")
MODEL_GROUPS: Tuple[str, ...] = ("G1", "G2", "G3", "G4")


def get_feature_columns() -> List[str]:
    return [f"op_setting_{i}" for i in range(1, 4)] + [
        f"sensor_{i}" for i in range(1, 22)
    ]


def get_all_columns() -> List[str]:
    return ["unit_id", "cycle"] + get_feature_columns()
