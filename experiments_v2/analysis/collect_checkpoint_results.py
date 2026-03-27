from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


@dataclass
class Config:
    output_dir_name: str = "checkpoint_bundle"


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
_OUTPUTS_ROOT = _PROJECT_ROOT / "outputs_v2"


def safe_read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_dict(v, prefix=f"{key}."))
        else:
            flat[key] = v
    return flat


def collect_group_comparison(bundle_dir: Path) -> Dict[str, Any]:
    src = _OUTPUTS_ROOT / "group_comparison.csv"
    dst = bundle_dir / "group_comparison.csv"
    ok = copy_if_exists(src, dst)
    result: Dict[str, Any] = {
        "source": str(src.relative_to(_PROJECT_ROOT)),
        "copied": ok,
        "target": str(dst.relative_to(_PROJECT_ROOT)) if ok else None,
    }
    if ok:
        df = pd.read_csv(src)
        result["num_rows"] = int(len(df))
        result["columns"] = list(df.columns)
        result["table"] = df.to_dict(orient="records")
    return result


def collect_round3(bundle_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for group in ["G2", "G4"]:
        src = _OUTPUTS_ROOT / f"{group}_round3_mcd_validity" / "round3_summary.json"
        dst = bundle_dir / "round3" / group / "round3_summary.json"
        ok = copy_if_exists(src, dst)
        entry: Dict[str, Any] = {
            "source": str(src.relative_to(_PROJECT_ROOT)),
            "copied": ok,
            "target": str(dst.relative_to(_PROJECT_ROOT)) if ok else None,
        }
        if ok:
            entry["summary"] = safe_read_json(src)
        result[group] = entry
    return result


def collect_round4(bundle_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for group in ["G1", "G2", "G3", "G4"]:
        src = _OUTPUTS_ROOT / "round4_error_structure" / group / "round4_summary.json"
        dst = bundle_dir / "round4" / group / "round4_summary.json"
        ok = copy_if_exists(src, dst)
        entry: Dict[str, Any] = {
            "source": str(src.relative_to(_PROJECT_ROOT)),
            "copied": ok,
            "target": str(dst.relative_to(_PROJECT_ROOT)) if ok else None,
        }
        if ok:
            entry["summary"] = safe_read_json(src)
        result[group] = entry
    return result


def collect_round5(bundle_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for group in ["G1", "G2", "G3", "G4"]:
        summary_src = (
            _OUTPUTS_ROOT / "round5_maintenance" / group / "round5_summary.json"
        )
        summary_dst = bundle_dir / "round5" / group / "round5_summary.json"
        threshold_src = (
            _OUTPUTS_ROOT
            / "round5_maintenance"
            / group
            / "round5_threshold_summary.csv"
        )
        threshold_dst = bundle_dir / "round5" / group / "round5_threshold_summary.csv"

        summary_ok = copy_if_exists(summary_src, summary_dst)
        threshold_ok = copy_if_exists(threshold_src, threshold_dst)

        entry: Dict[str, Any] = {
            "summary_source": str(summary_src.relative_to(_PROJECT_ROOT)),
            "summary_copied": summary_ok,
            "summary_target": (
                str(summary_dst.relative_to(_PROJECT_ROOT)) if summary_ok else None
            ),
            "threshold_source": str(threshold_src.relative_to(_PROJECT_ROOT)),
            "threshold_copied": threshold_ok,
            "threshold_target": (
                str(threshold_dst.relative_to(_PROJECT_ROOT)) if threshold_ok else None
            ),
        }

        if summary_ok:
            entry["summary"] = safe_read_json(summary_src)

        if threshold_ok:
            df = pd.read_csv(threshold_src)
            entry["threshold_table"] = df.to_dict(orient="records")

        result[group] = entry
    return result


def build_flat_rows(collected: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if "group_comparison" in collected and "table" in collected["group_comparison"]:
        for row in collected["group_comparison"]["table"]:
            out = {"section": "group_comparison"}
            out.update(row)
            rows.append(out)

    for group, entry in collected.get("round3", {}).items():
        if "summary" in entry:
            out = {"section": "round3_summary", "group": group}
            out.update(flatten_dict(entry["summary"]))
            rows.append(out)

    for group, entry in collected.get("round4", {}).items():
        if "summary" in entry:
            out = {"section": "round4_summary", "group": group}
            out.update(flatten_dict(entry["summary"]))
            rows.append(out)

    for group, entry in collected.get("round5", {}).items():
        if "summary" in entry:
            out = {"section": "round5_summary", "group": group}
            out.update(flatten_dict(entry["summary"]))
            rows.append(out)

    return pd.DataFrame(rows)


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_checkpoint_markdown(collected: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Checkpoint Results Bundle")
    lines.append("")
    lines.append("## 1. Main group comparison")
    lines.append("")

    gc = collected.get("group_comparison", {})
    table = gc.get("table", [])
    if table:
        cols = [
            "group",
            "num_runs",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "nasa_mean",
            "val_rmse_mean",
            "pred_std_mean_mean",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for row in table:
            lines.append("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")
    else:
        lines.append("No group comparison found.")
    lines.append("")

    lines.append("## 2. Round 3 summaries")
    lines.append("")
    for group in ["G2", "G4"]:
        entry = collected.get("round3", {}).get(group, {})
        summary = entry.get("summary")
        lines.append(f"### {group}")
        if summary is None:
            lines.append("Missing Round 3 summary.")
            lines.append("")
            continue

        fields = [
            "rmse_mean",
            "mae_mean",
            "nasa_mean",
            "pearson_unc_abs_err_mean",
            "spearman_unc_abs_err_mean",
            "ece_abs_err_vs_unc_mean",
            "top_bottom_error_ratio_mean",
            "coverage_1sigma_mean",
            "coverage_2sigma_mean",
        ]
        for f in fields:
            lines.append(f"- {f}: {fmt(summary.get(f))}")
        lines.append("")

    lines.append("## 3. Round 4 summaries")
    lines.append("")
    for group in ["G1", "G2", "G3", "G4"]:
        entry = collected.get("round4", {}).get(group, {})
        summary = entry.get("summary")
        lines.append(f"### {group}")
        if summary is None:
            lines.append("Missing Round 4 summary.")
            lines.append("")
            continue

        fields = [
            "rmse_mean",
            "mae_mean",
            "nasa_mean",
            "mean_error_mean",
            "danger_rmse_mean",
            "danger_nasa_mean",
            "safe_rmse_mean",
            "safe_nasa_mean",
        ]
        for f in fields:
            lines.append(f"- {f}: {fmt(summary.get(f))}")
        lines.append("")

    lines.append("## 4. Round 5 summaries")
    lines.append("")
    for group in ["G1", "G2", "G3", "G4"]:
        entry = collected.get("round5", {}).get(group, {})
        summary = entry.get("summary")
        lines.append(f"### {group}")
        if summary is None:
            lines.append("Missing Round 5 summary.")
            lines.append("")
            continue

        fields = [
            "best_threshold",
            "best_total_cost_mean",
            "best_failure_event_mean",
            "best_unnecessary_maintenance_mean",
            "best_preventive_action_mean",
            "best_protected_danger_mean",
            "best_early_lead_mean",
        ]
        for f in fields:
            lines.append(f"- {f}: {fmt(summary.get(f))}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    cfg = Config()
    bundle_dir = _OUTPUTS_ROOT / cfg.output_dir_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    collected: Dict[str, Any] = {
        "config": asdict(cfg),
        "group_comparison": collect_group_comparison(bundle_dir),
        "round3": collect_round3(bundle_dir),
        "round4": collect_round4(bundle_dir),
        "round5": collect_round5(bundle_dir),
    }

    collected_json = bundle_dir / "collected_results.json"
    with open(collected_json, "w", encoding="utf-8") as f:
        json.dump(collected, f, indent=2)

    flat_df = build_flat_rows(collected)
    flat_csv = bundle_dir / "collected_results.csv"
    flat_df.to_csv(flat_csv, index=False)

    md_text = build_checkpoint_markdown(collected)
    md_path = bundle_dir / "checkpoint_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print("=== Checkpoint bundle created ===")
    print(f"Bundle directory: outputs_v2/{cfg.output_dir_name}")
    print(f"Saved: outputs_v2/{cfg.output_dir_name}/collected_results.json")
    print(f"Saved: outputs_v2/{cfg.output_dir_name}/collected_results.csv")
    print(f"Saved: outputs_v2/{cfg.output_dir_name}/checkpoint_summary.md")


if __name__ == "__main__":
    main()
