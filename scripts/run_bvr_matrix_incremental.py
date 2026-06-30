#!/usr/bin/env python
"""Run BVR recovery tasks incrementally with bounded parallelism.

The runner treats a task as complete if either ``BVR_DONE.txt`` exists or the
task stdout contains the normal ``Saved BVR outputs`` completion line. This
lets the matrix resume cleanly after a reboot or after a launcher process exits
while child training processes have already finished.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BVRTask:
    level: str
    difficulty: str
    seed: int
    source_run_dir: Path
    output_dir: Path

    @property
    def label(self) -> str:
        return f"{self.level}_{self.difficulty}_seed{self.seed}"


@dataclass
class RunningTask:
    task: BVRTask
    process: subprocess.Popen[bytes]
    stdout_handle: object
    stderr_handle: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("runs/rl_baselines/ppo"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/bvr/full_5seed_after_nonlearning_20260619"))
    parser.add_argument("--levels", nargs="+", default=["level1", "level2", "level3"])
    parser.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--anchor-checkpoint-role",
        choices=("nominal", "final_recovery"),
        default="nominal",
        help="Checkpoint used as the frozen BVR anchor policy.",
    )
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="Evaluation episodes per domain. Use 0 to match each source PPO run.",
    )
    parser.add_argument("--rollout-episodes-per-iteration", type=int, default=64)
    parser.add_argument("--max-candidate-sets-per-iteration", type=int, default=128)
    parser.add_argument("--verifier-epochs", type=int, default=5)
    parser.add_argument("--verifier-batch-size", type=int, default=32)
    parser.add_argument("--selection-margin", type=float, default=5e-4)
    parser.add_argument("--belief-safety-penalty", type=float, default=1.0)
    parser.add_argument("--belief-cost-margin", type=float, default=0.02)
    parser.add_argument("--belief-constraint-margin", type=float, default=0.02)
    parser.add_argument("--advantage-loss-weight", type=float, default=1.0)
    parser.add_argument("--advantage-beta", type=float, default=2.0)
    parser.add_argument("--benefit-loss-weight", type=float, default=2.0)
    parser.add_argument("--benefit-epsilon", type=float, default=0.25)
    parser.add_argument("--benefit-positive-weight", type=float, default=3.0)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def log(message: str, log_path: Path) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def task_complete(task: BVRTask) -> bool:
    done_marker = task.output_dir / "BVR_DONE.txt"
    if done_marker.exists():
        return True
    stdout_path = task.output_dir / "stdout.log"
    metrics_path = task.output_dir / "bvr_metrics.csv"
    if not stdout_path.exists() or not metrics_path.exists():
        return False
    try:
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="ignore")
        line_count = sum(1 for _ in metrics_path.open("r", encoding="utf-8", errors="ignore"))
    except OSError:
        return False
    if "Saved BVR outputs" in stdout_text and line_count >= 2:
        done_marker.write_text(f"completed before runner resume {datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")
        return True
    return False


def task_failed(task: BVRTask) -> bool:
    return (task.output_dir / "BVR_FAILED.txt").exists()


def build_tasks(args: argparse.Namespace) -> list[BVRTask]:
    source_root = resolve(args.source_root)
    output_root = resolve(args.output_root)
    tasks: list[BVRTask] = []
    for level in args.levels:
        for difficulty in args.difficulties:
            scenario = f"{level}_{difficulty}"
            for seed in args.seeds:
                tasks.append(
                    BVRTask(
                        level=level,
                        difficulty=difficulty,
                        seed=int(seed),
                        source_run_dir=source_root / f"{scenario}_shock_recovery_5seeds" / f"seed{seed}",
                        output_dir=output_root / scenario / f"seed{seed}",
                    )
                )
    return tasks


def command_for_task(args: argparse.Namespace, task: BVRTask) -> list[str]:
    command = [
        str(resolve(args.python)),
        str(PROJECT_ROOT / "scripts" / "train_bvr.py"),
        "--source-run-dir",
        str(task.source_run_dir),
        "--output-dir",
        str(task.output_dir),
        "--device",
        "cpu",
        "--num-iterations",
        str(args.num_iterations),
        "--eval-episodes",
        str(args.eval_episodes),
        "--rollout-episodes-per-iteration",
        str(args.rollout_episodes_per_iteration),
        "--max-candidate-sets-per-iteration",
        str(args.max_candidate_sets_per_iteration),
        "--verifier-epochs",
        str(args.verifier_epochs),
        "--verifier-batch-size",
        str(args.verifier_batch_size),
        "--selection-margin",
        str(args.selection_margin),
        "--belief-safety-penalty",
        str(args.belief_safety_penalty),
        "--belief-cost-margin",
        str(args.belief_cost_margin),
        "--belief-constraint-margin",
        str(args.belief_constraint_margin),
        "--advantage-loss-weight",
        str(args.advantage_loss_weight),
        "--advantage-beta",
        str(args.advantage_beta),
        "--benefit-loss-weight",
        str(args.benefit_loss_weight),
        "--benefit-epsilon",
        str(args.benefit_epsilon),
        "--benefit-positive-weight",
        str(args.benefit_positive_weight),
    ]
    anchor_checkpoint = anchor_checkpoint_for_task(task, str(args.anchor_checkpoint_role))
    if anchor_checkpoint is not None:
        command.extend(["--clean-checkpoint", str(anchor_checkpoint)])
    return command


def anchor_checkpoint_for_task(task: BVRTask, role: str) -> Path | None:
    if role == "nominal":
        return None
    if role != "final_recovery":
        raise ValueError(f"unsupported anchor checkpoint role: {role}")
    checkpoints_dir = task.source_run_dir / "checkpoints"
    candidates = sorted(checkpoints_dir.glob("checkpoint_recovery_step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no recovery checkpoints found under {checkpoints_dir}")
    return candidates[-1]


def start_task(args: argparse.Namespace, task: BVRTask, log_path: Path) -> RunningTask:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    failed_marker = task.output_dir / "BVR_FAILED.txt"
    if failed_marker.exists():
        failed_marker.unlink()
    stdout_handle = (task.output_dir / "stdout.log").open("w", encoding="utf-8")
    stderr_handle = (task.output_dir / "stderr.log").open("w", encoding="utf-8")
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "PYTORCH_NUM_THREADS"):
        env[key] = "1"
    cmd = command_for_task(args, task)
    log(f"starting {task.label}", log_path)
    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=stdout_handle,
        stderr=stderr_handle,
        env=env,
    )
    return RunningTask(task=task, process=process, stdout_handle=stdout_handle, stderr_handle=stderr_handle)


def close_running(entry: RunningTask) -> None:
    entry.stdout_handle.close()
    entry.stderr_handle.close()


def main() -> int:
    args = parse_args()
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "bvr_matrix_runner.log"
    cpu_count = os.cpu_count() or 1
    max_parallel = int(args.max_parallel) if int(args.max_parallel) > 0 else max(1, math.floor(cpu_count * 0.8))
    tasks = build_tasks(args)

    log(f"BVR matrix runner started tasks={len(tasks)} max_parallel={max_parallel}", log_path)
    ready: list[BVRTask] = []
    skipped_done = 0
    skipped_failed = 0
    missing_source = 0
    for task in tasks:
        if task_complete(task):
            skipped_done += 1
            continue
        if task_failed(task) and not bool(args.retry_failed):
            skipped_failed += 1
            continue
        if not task.source_run_dir.exists():
            missing_source += 1
            task.output_dir.mkdir(parents=True, exist_ok=True)
            (task.output_dir / "BVR_FAILED.txt").write_text(f"missing source: {task.source_run_dir}\n", encoding="utf-8")
            log(f"missing source {task.label}: {task.source_run_dir}", log_path)
            continue
        ready.append(task)

    log(
        f"initial status skipped_done={skipped_done} skipped_failed={skipped_failed} "
        f"missing_source={missing_source} queued={len(ready)}",
        log_path,
    )
    if args.dry_run:
        for task in ready[:10]:
            log(f"dry-run queued {task.label}", log_path)
        return 0

    running: list[RunningTask] = []
    completed = 0
    failed = 0
    next_index = 0
    while next_index < len(ready) or running:
        while next_index < len(ready) and len(running) < max_parallel:
            running.append(start_task(args, ready[next_index], log_path))
            next_index += 1

        time.sleep(float(args.poll_seconds))
        still_running: list[RunningTask] = []
        for entry in running:
            return_code = entry.process.poll()
            if return_code is None:
                still_running.append(entry)
                continue
            close_running(entry)
            if return_code == 0:
                completed += 1
                (entry.task.output_dir / "BVR_DONE.txt").write_text(
                    f"completed {datetime.now().isoformat(timespec='seconds')}\n",
                    encoding="utf-8",
                )
                log(f"completed {entry.task.label}", log_path)
            else:
                failed += 1
                (entry.task.output_dir / "BVR_FAILED.txt").write_text(
                    f"failed exit_code={return_code} {datetime.now().isoformat(timespec='seconds')}\n",
                    encoding="utf-8",
                )
                log(f"failed {entry.task.label} exit_code={return_code}", log_path)
        running = still_running
        log(
            f"progress completed={completed} failed={failed} running={len(running)} "
            f"queued={len(ready) - next_index} skipped_done={skipped_done}",
            log_path,
        )

    total_done = sum(1 for task in tasks if task_complete(task))
    total_failed = sum(1 for task in tasks if task_failed(task))
    log(f"BVR matrix runner finished total_done={total_done}/{len(tasks)} total_failed={total_failed}", log_path)
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
