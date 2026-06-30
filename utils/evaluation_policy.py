"""Shared policy loading helpers for evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from utils.cleanrl_policy import load_cleanrl_agent, predict_cleanrl_action


def _checkpoint_config(model_config: dict[str, Any]) -> dict[str, Any]:
    nested = model_config.get("config")
    if isinstance(nested, dict):
        return nested
    return model_config


def _config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    hyphen_key = key.replace("_", "-")
    if key in config:
        return config[key]
    if hyphen_key in config:
        return config[hyphen_key]
    return default


def load_model(path: str | Path, requested_type: str = "auto") -> tuple[str | None, Any, dict[str, Any]]:
    """Load a CleanRL-style policy checkpoint."""

    path = Path(path)
    model_type = requested_type
    if requested_type == "auto":
        model_type = "cleanrl"

    if model_type == "cleanrl":
        model, model_config = load_cleanrl_agent(path)
        return "cleanrl", model, model_config

    raise ValueError(f"Unsupported model type: {requested_type}")


def predict_action(
    model_type: str,
    model: Any,
    obs: np.ndarray,
    residual_features: np.ndarray | None = None,
) -> np.ndarray:
    """Predict one continuous planner-parameter action."""

    if model_type == "cleanrl":
        return predict_cleanrl_action(model, obs, residual_features=residual_features)
    raise ValueError(f"Unsupported model type: {model_type}")


def resolve_observation_mode(requested: str, model_config: dict[str, Any]) -> str:
    """Resolve auto observation mode from checkpoint config."""

    if requested != "auto":
        return requested
    config = _checkpoint_config(model_config)
    return str(_config_value(config, "observation_mode", "terrain"))


def resolve_action_config(
    requested_action_mode: str,
    requested_action_gain: float | None,
    requested_max_uncertainty_lambda: float | None,
    model_config: dict[str, Any],
) -> tuple[str, float, float]:
    """Resolve planner action decoding parameters from checkpoint config."""

    config = _checkpoint_config(model_config)
    if requested_action_mode != "auto":
        action_mode = requested_action_mode
    else:
        action_mode = str(_config_value(config, "action_mode", "direct"))

    if requested_action_gain is not None:
        action_gain = float(requested_action_gain)
    else:
        action_gain = float(_config_value(config, "action_gain", 1.0))

    if requested_max_uncertainty_lambda is not None:
        max_uncertainty_lambda = float(requested_max_uncertainty_lambda)
    else:
        max_uncertainty_lambda = float(_config_value(config, "max_uncertainty_lambda", 1.0))

    return action_mode, action_gain, max_uncertainty_lambda
