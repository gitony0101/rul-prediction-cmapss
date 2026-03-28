from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]

GROUPS = ["G1", "G2", "G3", "G4"]


def get_summary_path(group: str) -> Path:
    return (
        _PROJECT_ROOT
        / "outputs_v2"
        / f"{group}_multiseed_logs"
        / f"{group}_multiseed_summary.json"
    )


def load_summary(group: str) -> dict | None:
    path = get_summary_path(group)
    if not path.exists():
        print(f"[WARN] Missing summary for {group}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    rows = []

    for g in GROUPS:
        summary = load_summary(g)
        if summary is None:
            continue

        rows.append(
            {
                "group": g,
                "num_runs": summary.get("num_runs"),
                "rmse_mean": summary.get("rmse_mean"),
                "rmse_std": summary.get("rmse_std"),
                "mae_mean": summary.get("mae_mean"),
                "mae_std": summary.get("mae_std"),
                "nasa_mean": summary.get("nasa_mean"),
                "nasa_std": summary.get("nasa_std"),
                "best_rmse": summary.get("best_rmse"),
                "best_seed": summary.get("best_seed"),
                "best_run_dir": summary.get("best_run_dir"),
                "best_linex_a": summary.get("best_linex_a"),
                "val_rmse_mean": summary.get("val_rmse_mean"),
                "val_rmse_std": summary.get("val_rmse_std"),
                "pred_std_mean_mean": summary.get("pred_std_mean_mean"),
                "pred_std_mean_std": summary.get("pred_std_mean_std"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty and "rmse_mean" in df.columns:
        df = df.sort_values(["rmse_mean", "group"], ascending=[True, True]).reset_index(
            drop=True
        )

    out_path = _PROJECT_ROOT / "outputs_v2" / "group_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n=== Group Comparison ===")
    print(df)
    print("\nSaved to: outputs_v2/group_comparison.csv")


if __name__ == "__main__":
    main()
