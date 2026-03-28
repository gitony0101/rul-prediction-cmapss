from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Config:
    group: str = "G1"
    seeds: tuple[int, ...] = (7, 21, 42, 87, 123)
    dataset: str | None = None
    data_dir: str | None = None
    output_root: str | None = None
    linex_a: float | None = None
    mc_samples_val: int | None = None
    mc_samples_test: int | None = None


_THIS_FILE = Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent
_PROJECT_ROOT = _THIS_FILE.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one experiment group across multiple seeds"
    )
    parser.add_argument(
        "--group", type=str, default="G1", choices=["G1", "G2", "G3", "G4"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 42, 87, 123])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--linex-a", type=float, default=None)
    parser.add_argument("--mc-samples-val", type=int, default=None)
    parser.add_argument("--mc-samples-test", type=int, default=None)
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg.group = args.group
    cfg.seeds = tuple(args.seeds)
    cfg.dataset = args.dataset
    cfg.data_dir = args.data_dir
    cfg.output_root = args.output_root
    cfg.linex_a = args.linex_a
    cfg.mc_samples_val = args.mc_samples_val
    cfg.mc_samples_test = args.mc_samples_test
    return cfg


def get_script_name(group: str) -> str:
    return f"{group}_run.py"


def build_command(cfg: Config, seed: int) -> List[str]:
    script_name = get_script_name(cfg.group)
    cmd: List[str] = [sys.executable, script_name, "--seed", str(seed)]

    if cfg.dataset is not None:
        cmd.extend(["--dataset", cfg.dataset])
    if cfg.data_dir is not None:
        cmd.extend(["--data-dir", cfg.data_dir])
    if cfg.output_root is not None:
        cmd.extend(["--output-root", cfg.output_root])

    if cfg.group in {"G3", "G4"} and cfg.linex_a is not None:
        cmd.extend(["--linex-a", str(cfg.linex_a)])

    if cfg.group in {"G2", "G4"}:
        if cfg.mc_samples_val is not None:
            cmd.extend(["--mc-samples-val", str(cfg.mc_samples_val)])
        if cfg.mc_samples_test is not None:
            cmd.extend(["--mc-samples-test", str(cfg.mc_samples_test)])

    return cmd


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(Config(), args)

    script_name = get_script_name(cfg.group)
    script_path = _THIS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path.name}")

    run_log = []

    for seed in cfg.seeds:
        cmd = build_command(cfg, seed)
        print(f"\n========== Running {cfg.group} with seed={seed} ==========")
        print("Command:", " ".join(cmd))

        result = subprocess.run(
            cmd,
            cwd=str(_THIS_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        run_item = {
            "group": cfg.group,
            "seed": seed,
            "command": cmd,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        run_log.append(run_item)

        if result.stdout.strip():
            print(result.stdout)
        if result.stderr.strip():
            print(result.stderr)

        if result.returncode != 0:
            print(f"Run failed for {cfg.group}, seed={seed}")
            break

    out_dir = _PROJECT_ROOT / "outputs_v2" / f"{cfg.group}_multiseed_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(
        out_dir / f"{cfg.group}_multiseed_run_log.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "config": asdict(cfg),
                "runs": run_log,
            },
            f,
            indent=2,
        )

    print(f"\nMulti-seed run log saved to: outputs_v2/{cfg.group}_multiseed_logs")


if __name__ == "__main__":
    main()
