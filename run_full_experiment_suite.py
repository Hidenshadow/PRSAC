"""Run the default robustness experiment and strong-attack recovery workflow."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/full_experiment_suite.json")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--clean", action="store_true", help="Delete configured output roots before running.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-default-robustness", action="store_true")
    parser.add_argument("--skip-strong-recovery", action="store_true")
    return parser.parse_args()


def load_config(path_text: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / path_text).read_text(encoding="utf-8"))


def quote_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)


def safe_remove_tree(path: Path, dry_run: bool) -> None:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    allowed_names = {"robustness", "strong_attack_recovery", "experiment_suite_logs"}
    if root not in resolved.parents:
        raise ValueError(f"refusing to remove path outside project: {resolved}")
    if resolved.name not in allowed_names:
        raise ValueError(f"refusing to remove unexpected output directory: {resolved}")
    print(f"Removing {resolved}")
    if not dry_run and resolved.exists():
        shutil.rmtree(resolved)


def run_stage(command: list[str], log_path: Path, dry_run: bool) -> None:
    print(quote_command(command))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"# Started {datetime.now().isoformat(timespec='seconds')}\n")
        log_file.write(f"# Command: {quote_command(command)}\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_file.write(f"\n# Finished {datetime.now().isoformat(timespec='seconds')}\n")
        log_file.write(f"# Exit code: {completed.returncode}\n")
    if completed.returncode != 0:
        raise RuntimeError(f"stage failed with exit code {completed.returncode}; see {log_path}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    default_cfg = dict(config.get("default_robustness", {}))
    strong_cfg = dict(config.get("strong_attack_recovery", {}))
    log_dir = PROJECT_ROOT / str(config.get("logs", {}).get("output_dir", "runs/experiment_suite_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    default_output = PROJECT_ROOT / str(default_cfg.get("output_root", "runs/robustness"))
    strong_output = PROJECT_ROOT / str(strong_cfg.get("output_root", "runs/strong_attack_recovery"))

    if args.clean:
        if default_cfg.get("enabled", True) and not args.skip_default_robustness:
            safe_remove_tree(default_output, args.dry_run)
        if strong_cfg.get("enabled", True) and not args.skip_strong_recovery:
            safe_remove_tree(strong_output, args.dry_run)

    if default_cfg.get("enabled", True) and not args.skip_default_robustness:
        seeds = args.seeds or str(default_cfg.get("seeds", "0-4"))
        default_command = [
            args.python,
            "run_ppo_robustness_experiment.py",
            "--config",
            str(default_cfg.get("config", "configs/ppo_robustness_experiment.json")),
            "--seeds",
            seeds,
        ]
        run_stage(default_command, log_dir / "default_robustness.log", args.dry_run)

    if strong_cfg.get("enabled", True) and not args.skip_strong_recovery:
        strong_command = [
            args.python,
            "run_strong_attack_recovery_experiment.py",
            "--config",
            str(strong_cfg.get("config", "configs/strong_attack_recovery_experiment.json")),
            "--checkpoint",
            str(strong_cfg.get("checkpoint", "runs/robustness/seed0/nominal_ppo/checkpoint.pt")),
            "--output-root",
            str(strong_cfg.get("output_root", "runs/strong_attack_recovery")),
            "--python",
            args.python,
        ]
        run_stage(strong_command, log_dir / "strong_attack_recovery.log", args.dry_run)

    print(f"Experiment suite logs: {log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
