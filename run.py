"""Run named experiment configs with short commands.

Examples:

    python run.py ppo_lunar_map_pool_relative_reward
    python run.py robustness_ppo_nominal --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="list",
        help="Config name such as ppo_lunar_map_pool_relative_reward or 'list'.",
    )
    parser.add_argument("--config-dir", type=str, default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help=(
            "Run a comma-separated training-seed sweep, for example 0,1,2. "
            "For evaluation configs, model paths containing seed0 are rewritten."
        ),
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override one config value, for example --set seed=2 or --set args.seed=2.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    return parser.parse_args()


def parse_seed_list(text: str | None) -> list[int]:
    if text is None:
        return []

    seeds: list[int] = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "-" in item and not item.startswith("-"):
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(item))

    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds contains duplicates: {seeds}")
    return seeds


def list_configs(config_dir: Path) -> None:
    config_paths = sorted(config_dir.glob("*.json"))
    if not config_paths:
        print(f"No configs found under {config_dir}")
        return

    print("Available configs:")
    for path in config_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            description = data.get("description", "")
        except json.JSONDecodeError:
            description = "invalid JSON"
        print(f"  {path.stem:<24} {description}")


def infer_value(text: str, existing: Any | None = None) -> Any:
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None

    if isinstance(existing, bool):
        return lowered in {"1", "yes", "y", "true", "on"}
    if isinstance(existing, int) and not isinstance(existing, bool):
        return int(text)
    if isinstance(existing, float):
        return float(text)
    if isinstance(existing, list):
        return [item.strip() for item in text.split(",") if item.strip()]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"override must be key=value, got {override!r}")
    raw_key, raw_value = override.split("=", 1)
    key = raw_key.strip()
    if not key:
        raise ValueError("override key must not be empty")

    if key in {"script", "description"}:
        config[key] = infer_value(raw_value, config.get(key))
        return

    args = config.setdefault("args", {})
    if key.startswith("args."):
        key = key.split(".", 1)[1]
    existing = args.get(key)
    args[key] = infer_value(raw_value, existing)


def load_config(config_name: str, config_dir: Path) -> tuple[Path, dict[str, Any]]:
    return load_config_with_seen(config_name, config_dir, seen=set())


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge a small config override into a base config."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if key == "args":
            merged_args = merged.setdefault("args", {})
            merged_args.update(copy.deepcopy(value))
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config_with_seen(
    config_name: str,
    config_dir: Path,
    seen: set[Path],
) -> tuple[Path, dict[str, Any]]:
    path = Path(config_name)
    if path.suffix != ".json":
        path = config_dir / f"{config_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"config does not exist: {path}")
    resolved_path = path.resolve()
    if resolved_path in seen:
        raise ValueError(f"cyclic config inheritance involving {path}")
    seen.add(resolved_path)

    config = json.loads(path.read_text(encoding="utf-8"))
    if "extends" in config:
        _, base_config = load_config_with_seen(str(config["extends"]), config_dir, seen)
        config = merge_config(base_config, config)

    if "script" not in config:
        raise ValueError(f"config must define 'script': {path}")
    if "args" in config and not isinstance(config["args"], dict):
        raise ValueError(f"config 'args' must be an object: {path}")
    config.setdefault("args", {})
    return path, config


def append_arg(command: list[str], key: str, value: Any) -> None:
    flag = f"--{key}"
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, list):
        for item in value:
            append_arg(command, key, item)
        return
    command.extend([flag, str(value)])


def build_command(config: dict[str, Any], python_executable: str) -> list[str]:
    script = PROJECT_ROOT / str(config["script"])
    if not script.exists():
        raise FileNotFoundError(f"script does not exist: {script}")

    command = [python_executable, str(script)]
    for key, value in config.get("args", {}).items():
        append_arg(command, key, value)
    return command


MODEL_SEED_TOKEN_PATTERN = re.compile(
    r"(cleanrl_ppo_costmap_seed)\d+"
)


def replace_seed_tokens(value: Any, seed: int) -> Any:
    """Rewrite model checkpoint seed fragments without changing map-pool seeds."""

    if isinstance(value, str):
        return MODEL_SEED_TOKEN_PATTERN.sub(lambda match: f"{match.group(1)}{seed}", value)
    if isinstance(value, list):
        return [replace_seed_tokens(item, seed) for item in value]
    if isinstance(value, dict):
        return {key: replace_seed_tokens(item, seed) for key, item in value.items()}
    return value


def config_for_seed(config: dict[str, Any], seed: int, multi_seed: bool) -> dict[str, Any]:
    """Create a seed-specific config while keeping evaluation fair.

    Training scripts use ``--seed`` as the training seed. Evaluation scripts
    keep their configured evaluation seed, but model paths are rewritten
    from ``seed0`` to ``seedN`` so trained checkpoints from each seed are tested
    on the same shared episode set.
    """

    seeded = copy.deepcopy(config)
    script_name = Path(str(seeded["script"])).name
    args = seeded.setdefault("args", {})

    if script_name.startswith("train_"):
        args["seed"] = seed
        return seeded

    seeded["args"] = replace_seed_tokens(args, seed)

    return seeded


def main() -> int:
    args = parse_args()
    config_dir = Path(args.config_dir)

    if args.config == "list":
        list_configs(config_dir)
        return 0

    config_path, config = load_config(args.config, config_dir)
    for override in args.overrides:
        apply_override(config, override)

    description = config.get("description", "")
    print(f"Config: {config_path}")
    if description:
        print(f"Description: {description}")

    seeds = parse_seed_list(args.seeds)
    configs_to_run = [(None, config)] if not seeds else [
        (seed, config_for_seed(config, seed, multi_seed=len(seeds) > 1))
        for seed in seeds
    ]

    for seed, run_config in configs_to_run:
        command = build_command(run_config, args.python)
        prefix = f"Seed {seed}" if seed is not None else "Command"
        print(f"{prefix}:")
        print(" ".join(shlex.quote(part) for part in command))

        if args.dry_run:
            continue

        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        if completed.returncode != 0:
            return int(completed.returncode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
