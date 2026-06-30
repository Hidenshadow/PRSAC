#!/usr/bin/env python
"""Wait for game-recovery runs and build the protocol performance figure."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOTS = (
    PROJECT_ROOT / "runs" / "game_recovery_level1",
    PROJECT_ROOT / "runs" / "game_recovery_level2_level3",
)
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "runs" / "game_recovery_protocol_analysis"
LEVELS = ("level1", "level2", "level3")
DIFFICULTIES = ("easy", "medium", "hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roots", nargs="*", type=Path, default=list(DEFAULT_SOURCE_ROOTS))
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--algorithm-name", type=str, default="game_ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--watch", action="store_true", help="Poll until every expected run is complete.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=0.0, help="Zero means no timeout.")
    parser.add_argument("--smooth-window", type=int, default=3)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def source_seed_dir(source_roots: list[Path], level: str, difficulty: str, seed: int) -> Path | None:
    rel = Path(f"{level}_{difficulty}_shock_recovery_1seed") / f"seed{seed}"
    for root in source_roots:
        candidate = root / rel
        if candidate.exists():
            return candidate
    return None


def expected_runs(source_roots: list[Path], seed: int) -> list[tuple[str, str, Path | None]]:
    runs = []
    for level in LEVELS:
        for difficulty in DIFFICULTIES:
            runs.append((level, difficulty, source_seed_dir(source_roots, level, difficulty, seed)))
    return runs


def incomplete_runs(source_roots: list[Path], seed: int) -> list[str]:
    missing = []
    for level, difficulty, seed_dir in expected_runs(source_roots, seed):
        label = f"{level}/{difficulty}/seed{seed}"
        if seed_dir is None:
            missing.append(f"{label}: missing seed directory")
            continue
        if not (seed_dir / "shock_recovery_curve.csv").exists():
            missing.append(f"{label}: missing shock_recovery_curve.csv")
        if not (seed_dir / "nominal_training_eval.csv").exists():
            missing.append(f"{label}: missing nominal_training_eval.csv")
    return missing


def wait_for_completion(source_roots: list[Path], seed: int, poll_seconds: int, timeout_hours: float) -> None:
    started = time.time()
    poll_seconds = max(int(poll_seconds), 5)
    while True:
        missing = incomplete_runs(source_roots, seed)
        if not missing:
            print("All expected game-recovery outputs are complete.", flush=True)
            return
        elapsed_hours = (time.time() - started) / 3600.0
        if timeout_hours > 0.0 and elapsed_hours >= timeout_hours:
            details = "\n".join(f"- {item}" for item in missing)
            raise TimeoutError(f"Timed out waiting for game-recovery outputs:\n{details}")
        print(
            f"Waiting for game-recovery outputs: {len(missing)} missing "
            f"(elapsed {elapsed_hours:.2f} h). Next check in {poll_seconds}s.",
            flush=True,
        )
        for item in missing[:9]:
            print(f"  - {item}", flush=True)
        time.sleep(poll_seconds)


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def stage_outputs(source_roots: list[Path], analysis_root: Path, algorithm_name: str, seed: int) -> Path:
    algorithm_root = analysis_root / algorithm_name
    for level, difficulty, source_dir in expected_runs(source_roots, seed):
        if source_dir is None:
            raise FileNotFoundError(f"missing source run for {level}/{difficulty}/seed{seed}")
        if not (source_dir / "shock_recovery_curve.csv").exists():
            raise FileNotFoundError(f"missing shock_recovery_curve.csv: {source_dir}")
        if not (source_dir / "nominal_training_eval.csv").exists():
            raise FileNotFoundError(f"missing nominal_training_eval.csv: {source_dir}")

        experiment_dir = algorithm_root / f"{level}_{difficulty}_shock_recovery_1seeds"
        target_seed_dir = experiment_dir / f"seed{seed}"
        target_seed_dir.mkdir(parents=True, exist_ok=True)
        copy_if_exists(source_dir / "shock_recovery_curve.csv", target_seed_dir / "shock_recovery_curve.csv")
        copy_if_exists(source_dir / "nominal_training_eval.csv", target_seed_dir / "nominal_training_eval.csv")
        copy_if_exists(source_dir / "shock_recovery_summary.csv", target_seed_dir / "shock_recovery_summary.csv")
        copy_if_exists(source_dir / "run_config.json", target_seed_dir / "run_config.json")
    return algorithm_root


def aggregate_staged_outputs(analysis_root: Path, algorithm_name: str) -> None:
    aggregate_script = PROJECT_ROOT / "scripts" / "aggregate_shock_recovery_3seeds.py"
    for level in LEVELS:
        for difficulty in DIFFICULTIES:
            experiment_dir = analysis_root / algorithm_name / f"{level}_{difficulty}_shock_recovery_1seeds"
            subprocess.run(
                [sys.executable, str(aggregate_script), str(experiment_dir)],
                cwd=str(PROJECT_ROOT),
                check=True,
            )


def build_figure(analysis_root: Path, algorithm_name: str, smooth_window: int) -> Path:
    os.environ.setdefault("MPLBACKEND", "Agg")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.plot_ppo_difficulty_performance_story import plot_protocol

    output_dir = analysis_root / "paper_story_1seed"
    summary = plot_protocol(
        analysis_root,
        output_dir,
        seed_count=1,
        algorithms=[algorithm_name],
        smooth_window=smooth_window,
    )
    summary.to_csv(output_dir / "game_recovery_performance_story.csv", index=False)
    figure_path = output_dir / "fig_rl_baseline_protocol_performance.png"
    alias_path = output_dir / "fig_game_recovery_protocol_performance.png"
    if figure_path.exists():
        shutil.copy2(figure_path, alias_path)
    print(f"Saved figure: {figure_path}", flush=True)
    print(f"Saved alias: {alias_path}", flush=True)
    print(f"Saved summary: {output_dir / 'game_recovery_performance_story.csv'}", flush=True)
    print(summary.to_string(index=False), flush=True)
    return figure_path


def main() -> int:
    args = parse_args()
    source_roots = [resolve_path(path) for path in args.source_roots]
    analysis_root = resolve_path(args.analysis_root)
    if args.watch:
        wait_for_completion(source_roots, args.seed, args.poll_seconds, args.timeout_hours)
    else:
        missing = incomplete_runs(source_roots, args.seed)
        if missing:
            details = "\n".join(f"- {item}" for item in missing)
            raise SystemExit(f"Cannot build figure yet; incomplete outputs:\n{details}")

    analysis_root.mkdir(parents=True, exist_ok=True)
    stage_outputs(source_roots, analysis_root, args.algorithm_name, args.seed)
    aggregate_staged_outputs(analysis_root, args.algorithm_name)
    build_figure(analysis_root, args.algorithm_name, args.smooth_window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
