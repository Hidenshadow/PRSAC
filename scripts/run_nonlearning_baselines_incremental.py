#!/usr/bin/env python
"""Incremental runner for non-learning planner baselines.

The original evaluator writes summaries after all requested source runs finish.
This wrapper saves one completed level/difficulty/seed chunk at a time, then
refreshes the aggregate tables and figure. It is meant for long runs on a
machine that may reboot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.evaluate_nonlearning_planner_baselines as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=base.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-seed-count", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--levels", nargs="+", choices=base.LEVELS, default=list(base.LEVELS))
    parser.add_argument("--difficulties", nargs="+", choices=base.DIFFICULTIES, default=list(base.DIFFICULTIES))
    parser.add_argument("--num-eval-episodes", type=int, default=300)
    parser.add_argument("--validation-episodes", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=base.DEFAULT_OUTPUT_DIR / "5seeds_incremental")
    parser.add_argument("--rl-summary", type=Path, default=base.DEFAULT_RL_SUMMARY)
    parser.add_argument("--game-summary", type=Path, default=base.DEFAULT_GAME_SUMMARY)
    parser.add_argument("--include-all-presets", action="store_true")
    parser.add_argument("--disable-validation-best", action="store_true")
    parser.add_argument("--disable-model-minimax", action="store_true")
    parser.add_argument("--minimax-variant-mode", choices=("component", "scale", "scale_component"), default="component")
    parser.add_argument("--minimax-mixture-size", type=int, default=5)
    parser.add_argument("--minimax-jitter", type=float, default=0.18)
    parser.add_argument("--disable-risk-inflated", action="store_true")
    parser.add_argument("--disable-belief-cvar", action="store_true")
    parser.add_argument("--risk-inflation-scale", type=float, default=0.42)
    parser.add_argument("--risk-inflation-radius", type=int, default=2)
    parser.add_argument("--belief-cvar-samples", type=int, default=5)
    parser.add_argument("--belief-cvar-alpha", type=float, default=0.35)
    parser.add_argument("--belief-cvar-noise-scale", type=float, default=0.55)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--save-episode-details", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute chunks even if their done marker exists.")
    return parser.parse_args()


def chunk_id(run_dir: Path) -> str:
    level, difficulty, seed = base.run_label_from_dir(run_dir)
    return f"{level}_{difficulty}_seed{seed}"


def chunk_paths(chunks_dir: Path, run_dir: Path) -> dict[str, Path]:
    cid = chunk_id(run_dir)
    return {
        "domain": chunks_dir / f"{cid}_domain.csv",
        "detail": chunks_dir / f"{cid}_details.csv",
        "meta": chunks_dir / f"{cid}_meta.json",
        "done": chunks_dir / f"{cid}.done",
    }


def write_chunk(paths: dict[str, Path], domain: pd.DataFrame, detail: pd.DataFrame, meta: dict[str, Any]) -> None:
    paths["domain"].parent.mkdir(parents=True, exist_ok=True)
    domain.to_csv(paths["domain"], index=False)
    if not detail.empty:
        detail.to_csv(paths["detail"], index=False)
    paths["meta"].write_text(json.dumps(base.safe_json(meta), indent=2) if hasattr(base, "safe_json") else json.dumps(meta, indent=2), encoding="utf-8")
    paths["done"].write_text("ok\n", encoding="utf-8")


def read_chunks(chunks_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    domain_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    metas: list[dict[str, Any]] = []
    for done in sorted(chunks_dir.glob("*.done")):
        stem = done.stem
        domain_path = chunks_dir / f"{stem}_domain.csv"
        detail_path = chunks_dir / f"{stem}_details.csv"
        meta_path = chunks_dir / f"{stem}_meta.json"
        if domain_path.exists():
            domain_frames.append(pd.read_csv(domain_path))
        if detail_path.exists():
            detail_frames.append(pd.read_csv(detail_path))
        if meta_path.exists():
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
    domain = pd.concat(domain_frames, ignore_index=True) if domain_frames else pd.DataFrame()
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return domain, detail, metas


def refresh_aggregate(output_dir: Path, args: argparse.Namespace) -> None:
    chunks_dir = output_dir / "chunks"
    domain, detail, metas = read_chunks(chunks_dir)
    if domain.empty:
        return
    panel = base.panel_summary(domain)
    rl = base.load_rl_rows(args.rl_summary, args.game_summary)
    delta = base.compare_with_rl(panel, rl)
    domain.to_csv(output_dir / "planner_baseline_domain_summary.csv", index=False)
    panel.to_csv(output_dir / "planner_baseline_panel_summary.csv", index=False)
    delta.to_csv(output_dir / "planner_baseline_vs_rl_deltas.csv", index=False)
    if not detail.empty:
        detail.to_csv(output_dir / "planner_baseline_episode_details.csv", index=False)
    pd.DataFrame(metas).to_csv(output_dir / "planner_baseline_run_metadata.csv", index=False)
    base.plot_bars(panel, rl, output_dir)
    base.write_report(panel, delta, output_dir, metas)


def main() -> int:
    args = parse_args()
    if args.quick:
        args.num_eval_episodes = min(int(args.num_eval_episodes), 8)
        args.validation_episodes = min(int(args.validation_episodes), 6)
    output_dir = base.resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = base.source_run_dirs(args)
    if not run_dirs:
        raise FileNotFoundError("no source runs found")

    completed = 0
    skipped = 0
    total = len(run_dirs)
    for index, run_dir in enumerate(run_dirs, start=1):
        paths = chunk_paths(chunks_dir, run_dir)
        cid = chunk_id(run_dir)
        if paths["done"].exists() and not bool(args.force):
            skipped += 1
            print(f"[{index}/{total}] skip completed {cid}", flush=True)
            continue
        print(f"[{index}/{total}] evaluating {cid}", flush=True)
        domain, detail, meta = base.evaluate_run(run_dir, args)
        meta = dict(meta)
        meta["chunk_id"] = cid
        write_chunk(paths, domain, detail, meta)
        completed += 1
        refresh_aggregate(output_dir, args)
        print(f"[{index}/{total}] completed {cid}", flush=True)

    refresh_aggregate(output_dir, args)
    print(f"Incremental non-learning baselines finished: completed={completed}, skipped={skipped}, total={total}")
    print(f"Output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
