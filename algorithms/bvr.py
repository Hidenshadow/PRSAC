"""Belief-Route Verifier Recovery.

BVR keeps the nominal policy frozen, generates a small deterministic set of
planner-parameter candidates, and uses a learned verifier to pick among routes
planned on the attacked belief map. True-map costs are used only for training
the verifier labels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from algorithms.local_repair import CandidateAction
from utils.metrics import (
    OBJECTIVE_NAMES,
    action_to_planning_weights,
    action_to_uncertainty_lambda,
    plan_with_weights,
)


CONSTRAINT_FEATURE_KEYS = (
    "max_hazard",
    "min_communication_quality",
    "mean_illumination_quality",
    "max_slope_ratio",
    "near_slope_limit_fraction",
    "slope_limit_violation_fraction",
)

ROUTE_SCALAR_FEATURES = (
    "success",
    "path_length_norm",
    "belief_scalar_cost_scaled",
    "belief_constraint_penalty_scaled",
    "belief_hazard_exposure",
    "belief_illumination_exposure",
    "belief_communication_exposure",
    "belief_uncertainty_exposure",
    "mean_path_confidence",
)


@dataclass(frozen=True)
class BVRCandidateBatch:
    candidates: list[CandidateAction]
    features: np.ndarray
    true_scores: np.ndarray
    raw_results: list[dict[str, Any]]


@dataclass(frozen=True)
class BVRSelection:
    action: np.ndarray
    clean_action: np.ndarray
    selected_index: int
    selected_candidate_type: str
    verifier_scores: np.ndarray
    selection_scores: np.ndarray
    candidates: list[CandidateAction]


@dataclass(frozen=True)
class BVRTrainerConfig:
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 3e-4
    target_temperature: float = 1.5
    set_weight_beta: float = 2.0
    set_weight_max: float = 5.0
    max_replay_sets: int = 2048
    advantage_loss_weight: float = 1.0
    advantage_beta: float = 2.0
    advantage_clip: float = 5.0
    benefit_loss_weight: float = 2.0
    benefit_epsilon: float = 0.25
    benefit_positive_weight: float = 3.0


def _as_action_vector(value: float | Iterable[float] | np.ndarray, action_dim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 1:
        array = np.repeat(array, int(action_dim)).astype(np.float32)
    if array.size != int(action_dim):
        raise ValueError(f"{name} must be scalar or length {action_dim}, got {array.size}")
    return array.astype(np.float32)


def action_range_fraction(
    fraction: float | Iterable[float] | np.ndarray,
    action_low: float | Iterable[float] | np.ndarray,
    action_high: float | Iterable[float] | np.ndarray,
    action_dim: int,
) -> np.ndarray:
    low = _as_action_vector(action_low, action_dim, "action_low")
    high = _as_action_vector(action_high, action_dim, "action_high")
    frac = _as_action_vector(fraction, action_dim, "fraction")
    return (frac * (high - low)).astype(np.float32)


def generate_bvr_candidates(
    clean_action: np.ndarray,
    action_low: float | Iterable[float] | np.ndarray,
    action_high: float | Iterable[float] | np.ndarray,
    search_radius: float | Iterable[float] | np.ndarray,
    pairwise_candidate_mode: str = "adjacent",
) -> list[CandidateAction]:
    """Generate a compact local action set around the clean-policy anchor."""

    clean = np.asarray(clean_action, dtype=np.float32).reshape(-1)
    dim = clean.size
    low = _as_action_vector(action_low, dim, "action_low")
    high = _as_action_vector(action_high, dim, "action_high")
    radius = _as_action_vector(search_radius, dim, "search_radius")
    candidates = [CandidateAction(np.clip(clean, low, high).astype(np.float32), "clean_anchor")]
    for index in range(dim):
        plus = clean.copy()
        plus[index] += radius[index]
        minus = clean.copy()
        minus[index] -= radius[index]
        candidates.append(CandidateAction(np.clip(plus, low, high).astype(np.float32), f"plus_dim_{index}", index))
        candidates.append(CandidateAction(np.clip(minus, low, high).astype(np.float32), f"minus_dim_{index}", index))

    pairwise_mode = str(pairwise_candidate_mode).lower()
    if pairwise_mode not in {"none", "adjacent", "all"}:
        raise ValueError("pairwise_candidate_mode must be 'none', 'adjacent', or 'all'")
    if pairwise_mode != "none":
        if pairwise_mode == "adjacent":
            pairs = [(index, index + 1) for index in range(max(dim - 1, 0))]
        else:
            pairs = [(left, right) for left in range(dim) for right in range(left + 1, dim)]
        signs = ((1.0, 1.0, "pp"), (1.0, -1.0, "pm"), (-1.0, 1.0, "mp"), (-1.0, -1.0, "mm"))
        for left, right in pairs:
            for left_sign, right_sign, sign_name in signs:
                action = clean.copy()
                action[left] += left_sign * radius[left]
                action[right] += right_sign * radius[right]
                candidates.append(
                    CandidateAction(
                        np.clip(action, low, high).astype(np.float32),
                        f"pair_{sign_name}_dim_{left}_{right}",
                        None,
                    )
                )
    return candidates


def route_verifier_feature_names(obs_dim: int, action_dim: int) -> list[str]:
    names: list[str] = []
    names.extend([f"obs_{index}" for index in range(int(obs_dim))])
    names.extend([f"action_{index}" for index in range(int(action_dim))])
    names.extend([f"clean_action_{index}" for index in range(int(action_dim))])
    names.extend([f"action_delta_{index}" for index in range(int(action_dim))])
    names.extend([f"planner_weight_{name}" for name in OBJECTIVE_NAMES])
    names.extend([f"mission_minus_weight_{name}" for name in OBJECTIVE_NAMES])
    names.append("lambda_uncertainty_norm")
    names.extend(ROUTE_SCALAR_FEATURES)
    names.extend([f"belief_objective_{name}" for name in OBJECTIVE_NAMES])
    names.extend([f"path_uncertainty_{name}" for name in OBJECTIVE_NAMES])
    names.extend([f"belief_constraint_{name}" for name in CONSTRAINT_FEATURE_KEYS])
    return names


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return float(number)


def _dict_feature(source: dict[str, Any], key: str, default: float = 0.0) -> float:
    return _safe_float(source.get(key, default), default)


def _clip_feature(value: float, low: float = -5.0, high: float = 5.0) -> float:
    return float(np.clip(_safe_float(value), low, high))


def extract_route_verifier_features(
    observation: np.ndarray,
    clean_action: np.ndarray,
    action: np.ndarray,
    result: dict[str, Any],
    *,
    map_size: int,
    max_uncertainty_lambda: float,
) -> np.ndarray:
    """Build verifier features using only attacked-belief route information.

    This intentionally ignores true-map fields such as ``scalar_cost``,
    ``objectives``, ``map_mismatch_penalty``, and true-layer exposures.
    """

    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    clean = np.asarray(clean_action, dtype=np.float32).reshape(-1)
    act = np.asarray(action, dtype=np.float32).reshape(-1)
    if clean.shape != act.shape:
        raise ValueError("clean_action and action must share shape")

    weights = np.asarray(result.get("weights", np.zeros(len(OBJECTIVE_NAMES))), dtype=np.float32).reshape(-1)
    if weights.size < len(OBJECTIVE_NAMES):
        weights = np.pad(weights, (0, len(OBJECTIVE_NAMES) - weights.size))
    weights = weights[: len(OBJECTIVE_NAMES)]
    mission = np.asarray(result.get("mission_priority", np.zeros(len(OBJECTIVE_NAMES))), dtype=np.float32).reshape(-1)
    if mission.size < len(OBJECTIVE_NAMES):
        mission = np.zeros(len(OBJECTIVE_NAMES), dtype=np.float32)
    mission = mission[: len(OBJECTIVE_NAMES)]
    lambda_norm = _safe_float(result.get("lambda_uncertainty", 0.0)) / max(float(max_uncertainty_lambda), 1e-6)

    path_length_norm = _safe_float(result.get("path_length", 0.0)) / max(float(map_size) * 2.0, 1.0)
    scalar_values = [
        1.0 if bool(result.get("success", False)) else 0.0,
        _clip_feature(path_length_norm, 0.0, 3.0),
        _clip_feature(_safe_float(result.get("belief_scalar_cost", 10.0)) / 10.0, 0.0, 10.0),
        _clip_feature(_safe_float(result.get("belief_constraint_penalty", 10.0)) / 10.0, 0.0, 10.0),
        _clip_feature(_safe_float(result.get("belief_hazard_exposure", 0.0)), 0.0, 2.0),
        _clip_feature(_safe_float(result.get("belief_illumination_exposure", 0.0)), 0.0, 2.0),
        _clip_feature(_safe_float(result.get("belief_communication_exposure", 0.0)), 0.0, 2.0),
        _clip_feature(_safe_float(result.get("belief_uncertainty_exposure", 0.0)), 0.0, 2.0),
        _clip_feature(_safe_float(result.get("mean_path_confidence", 0.0)), 0.0, 1.0),
    ]

    belief_objectives = dict(result.get("belief_objectives", {}) or {})
    path_uncertainty = dict(result.get("path_uncertainty", {}) or {})
    belief_constraints = dict(result.get("belief_constraint_metrics", {}) or {})

    features: list[float] = []
    features.extend([_clip_feature(value, 0.0, 1.0) for value in obs])
    features.extend([_clip_feature(value, 0.0, 1.0) for value in act])
    features.extend([_clip_feature(value, 0.0, 1.0) for value in clean])
    features.extend([_clip_feature(value, -1.0, 1.0) for value in (act - clean)])
    features.extend([_clip_feature(value, 0.0, 1.0) for value in weights])
    features.extend([_clip_feature(value, -1.0, 1.0) for value in (mission - weights)])
    features.append(_clip_feature(lambda_norm, 0.0, 1.0))
    features.extend(scalar_values)
    features.extend([_clip_feature(_dict_feature(belief_objectives, name), 0.0, 5.0) for name in OBJECTIVE_NAMES])
    features.extend([_clip_feature(_dict_feature(path_uncertainty, name), 0.0, 5.0) for name in OBJECTIVE_NAMES])
    features.extend([_clip_feature(_dict_feature(belief_constraints, name), 0.0, 5.0) for name in CONSTRAINT_FEATURE_KEYS])
    return np.asarray(features, dtype=np.float32)


def belief_only_episode(episode: Any) -> Any:
    """Return an episode whose planner result cannot access true-map costs."""

    if getattr(episode, "true_costmap", None) is None:
        return episode
    try:
        return replace(episode, true_costmap=None)
    except TypeError:
        episode_copy = episode
        try:
            episode_copy.true_costmap = None
        except Exception:
            pass
        return episode_copy


def _attack_value(env_attack: dict[str, Any] | None, key: str, default: Any) -> Any:
    if not isinstance(env_attack, dict):
        return default
    return env_attack.get(key, default)


def plan_belief_route_action(
    episode: Any,
    action: np.ndarray,
    runtime: dict[str, Any],
    env_attack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan a candidate route on attacked belief only.

    The returned dictionary may contain a ``scalar_cost`` key, but after
    stripping ``true_costmap`` this cost is belief cost, not true-terrain cost.
    """

    belief_episode = belief_only_episode(episode)
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    weights = action_to_planning_weights(
        belief_episode,
        action_array,
        action_mode=str(runtime.get("action_mode", "preference_delta")),
        action_gain=float(runtime.get("action_gain", 2.0)),
    )
    lambda_uncertainty = action_to_uncertainty_lambda(
        action_array,
        max_uncertainty_lambda=float(runtime.get("max_uncertainty_lambda", 1.0)),
    )
    result = plan_with_weights(
        belief_episode,
        weights,
        lambda_uncertainty=lambda_uncertainty,
        allow_diagonal=True,
        attack_budget_fraction=float(_attack_value(env_attack, "attack_budget_fraction", 0.18)),
        attack_strength=float(_attack_value(env_attack, "attack_strength", 1.0)),
        attacker_temperature=float(_attack_value(env_attack, "attacker_temperature", 0.5)),
        attacker_response=str(_attack_value(env_attack, "attacker_response", "zscore_topk")),
        attacker_top_fraction=float(_attack_value(env_attack, "attacker_top_fraction", 0.15)),
        attacker_sharpness=float(_attack_value(env_attack, "attacker_sharpness", 3.0)),
    )
    result = dict(result)
    result["mission_priority"] = np.asarray(belief_episode.mission_priority, dtype=np.float32)
    return result


class RouteVerifier(nn.Module):
    """Small MLP that scores candidate route features for listwise reranking."""

    policy_class = "belief_route_verifier"

    def __init__(
        self,
        feature_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        activation: str = "tanh",
        final_init_std: float = 1e-4,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.activation = str(activation)
        if activation == "tanh":
            activation_layer: type[nn.Module] = nn.Tanh
        elif activation == "relu":
            activation_layer = nn.ReLU
        else:
            raise ValueError("activation must be 'tanh' or 'relu'")
        layers: list[nn.Module] = []
        in_dim = self.feature_dim
        for hidden in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, int(hidden)))
            layers.append(activation_layer())
            in_dim = int(hidden)
        final = nn.Linear(in_dim, 1)
        nn.init.normal_(final.weight, mean=0.0, std=float(final_init_std))
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class BVRCandidateDataset(Dataset):
    def __init__(self, feature_sets: list[np.ndarray], score_sets: list[np.ndarray]) -> None:
        if len(feature_sets) != len(score_sets):
            raise ValueError("feature_sets and score_sets must have the same length")
        if not feature_sets:
            self.features = torch.zeros((0, 0, 0), dtype=torch.float32)
            self.scores = torch.zeros((0, 0), dtype=torch.float32)
            return
        candidate_count = int(feature_sets[0].shape[0])
        feature_dim = int(feature_sets[0].shape[1])
        for features, scores in zip(feature_sets, score_sets):
            if features.shape != (candidate_count, feature_dim):
                raise ValueError("all BVR feature sets must share candidate count and feature dimension")
            if np.asarray(scores).reshape(-1).shape[0] != candidate_count:
                raise ValueError("score count must match candidate count")
        self.features = torch.as_tensor(np.stack(feature_sets, axis=0).astype(np.float32))
        self.scores = torch.as_tensor(np.stack(score_sets, axis=0).astype(np.float32))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.scores[index]


class BVRTrainer:
    def __init__(self, verifier: RouteVerifier, config: BVRTrainerConfig | None = None) -> None:
        self.verifier = verifier
        self.config = config or BVRTrainerConfig()
        self.optimizer = torch.optim.Adam(self.verifier.parameters(), lr=float(self.config.learning_rate))

    @property
    def device(self) -> torch.device:
        return next(self.verifier.parameters()).device

    def update(self, feature_sets: list[np.ndarray], score_sets: list[np.ndarray]) -> dict[str, float]:
        if not feature_sets:
            return {
                "bvr/loss": 0.0,
                "bvr/advantage_loss": 0.0,
                "bvr/benefit_loss": 0.0,
                "bvr/top1_accuracy": 0.0,
                "bvr/positive_benefit_label_rate": 0.0,
                "bvr/predicted_benefit_rate": 0.0,
                "bvr/mean_selected_regret": 0.0,
                "bvr/mean_score_spread": 0.0,
                "bvr/num_candidate_sets": 0,
            }
        if len(feature_sets) > int(self.config.max_replay_sets):
            feature_sets = feature_sets[-int(self.config.max_replay_sets) :]
            score_sets = score_sets[-int(self.config.max_replay_sets) :]
        dataset = BVRCandidateDataset(feature_sets, score_sets)
        loader = DataLoader(
            dataset,
            batch_size=max(int(self.config.batch_size), 1),
            shuffle=True,
            drop_last=False,
        )
        losses: list[float] = []
        advantage_losses: list[float] = []
        benefit_losses: list[float] = []
        accuracies: list[float] = []
        positive_benefit_rates: list[float] = []
        predicted_benefit_rates: list[float] = []
        regrets: list[float] = []
        spreads: list[float] = []
        temperature = max(float(self.config.target_temperature), 1e-6)
        for _ in range(max(int(self.config.epochs), 1)):
            for feature_batch, score_batch in loader:
                feature_batch = feature_batch.to(self.device)
                score_batch = score_batch.to(self.device)
                batch_size, candidate_count, feature_dim = feature_batch.shape
                finite_scores = torch.where(torch.isfinite(score_batch), score_batch, torch.full_like(score_batch, -1e6))
                centered = finite_scores - finite_scores.max(dim=1, keepdim=True).values
                target = torch.softmax(centered / temperature, dim=1)
                logits = self.verifier(feature_batch.reshape(batch_size * candidate_count, feature_dim)).reshape(
                    batch_size,
                    candidate_count,
                )
                log_probs = torch.log_softmax(logits, dim=1)
                per_set_loss = -(target * log_probs).sum(dim=1)
                spread = finite_scores.max(dim=1).values - finite_scores.min(dim=1).values
                set_weight = torch.clamp(
                    spread / max(float(self.config.set_weight_beta), 1e-8),
                    min=1.0,
                    max=float(self.config.set_weight_max),
                )
                true_advantage = finite_scores - finite_scores[:, :1]
                advantage_beta = max(float(self.config.advantage_beta), 1e-8)
                target_advantage = torch.clamp(
                    true_advantage / advantage_beta,
                    min=-float(self.config.advantage_clip),
                    max=float(self.config.advantage_clip),
                )
                predicted_advantage = logits - logits[:, :1]
                candidate_weight = 1.0 + torch.clamp(
                    torch.abs(true_advantage) / advantage_beta,
                    min=0.0,
                    max=max(float(self.config.set_weight_max) - 1.0, 0.0),
                )
                advantage_loss = ((predicted_advantage - target_advantage) ** 2 * candidate_weight).sum(dim=1) / (
                    torch.clamp(candidate_weight.sum(dim=1), min=1.0)
                )
                benefit_target = (true_advantage > float(self.config.benefit_epsilon)).float()
                benefit_mask = torch.ones_like(benefit_target)
                benefit_mask[:, 0] = 0.0
                benefit_weight = torch.where(
                    benefit_target > 0.0,
                    torch.full_like(benefit_target, float(self.config.benefit_positive_weight)),
                    torch.ones_like(benefit_target),
                )
                benefit_weight = benefit_weight * benefit_mask
                benefit_terms = torch.nn.functional.binary_cross_entropy_with_logits(
                    predicted_advantage,
                    benefit_target,
                    reduction="none",
                )
                benefit_loss = (benefit_terms * benefit_weight).sum(dim=1) / torch.clamp(
                    benefit_weight.sum(dim=1),
                    min=1.0,
                )
                per_set_total_loss = (
                    per_set_loss
                    + float(self.config.advantage_loss_weight) * advantage_loss
                    + float(self.config.benefit_loss_weight) * benefit_loss
                )
                loss = (per_set_total_loss * set_weight).sum() / torch.clamp(set_weight.sum(), min=1.0)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    true_best = torch.argmax(finite_scores, dim=1)
                    predicted_best = torch.argmax(logits, dim=1)
                    chosen_scores = finite_scores.gather(1, predicted_best[:, None]).squeeze(1)
                    best_scores = finite_scores.gather(1, true_best[:, None]).squeeze(1)
                    losses.append(float(loss.detach().cpu()))
                    advantage_losses.append(float(advantage_loss.mean().detach().cpu()))
                    accuracies.append(float((predicted_best == true_best).float().mean().detach().cpu()))
                    regrets.append(float((best_scores - chosen_scores).mean().detach().cpu()))
                    spreads.append(float(spread.mean().detach().cpu()))
                    benefit_losses.append(float(benefit_loss.mean().detach().cpu()))
                    positive_benefit_rates.append(float(benefit_target[:, 1:].mean().detach().cpu()))
                    predicted_benefit_rates.append(
                        float((torch.sigmoid(predicted_advantage[:, 1:]) > 0.5).float().mean().detach().cpu())
                    )
        return {
            "bvr/loss": float(np.mean(losses)) if losses else 0.0,
            "bvr/advantage_loss": float(np.mean(advantage_losses)) if advantage_losses else 0.0,
            "bvr/benefit_loss": float(np.mean(benefit_losses)) if benefit_losses else 0.0,
            "bvr/top1_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "bvr/positive_benefit_label_rate": float(np.mean(positive_benefit_rates))
            if positive_benefit_rates
            else 0.0,
            "bvr/predicted_benefit_rate": float(np.mean(predicted_benefit_rates)) if predicted_benefit_rates else 0.0,
            "bvr/mean_selected_regret": float(np.mean(regrets)) if regrets else 0.0,
            "bvr/mean_score_spread": float(np.mean(spreads)) if spreads else 0.0,
            "bvr/num_candidate_sets": int(len(feature_sets)),
        }


@torch.no_grad()
def select_bvr_action_from_features(
    verifier: RouteVerifier,
    candidates: list[CandidateAction],
    features: np.ndarray,
    clean_action: np.ndarray,
    selection_margin: float = 0.0,
    belief_safety_penalty: float = 1.0,
    belief_cost_margin: float = 0.02,
    belief_constraint_margin: float = 0.02,
) -> BVRSelection:
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim != 2:
        raise ValueError("features must have shape [candidate_count, feature_dim]")
    if len(candidates) != feature_array.shape[0]:
        raise ValueError("candidate count does not match feature rows")
    device = next(verifier.parameters()).device
    tensor = torch.as_tensor(feature_array, dtype=torch.float32, device=device)
    scores = verifier(tensor).detach().cpu().numpy().astype(np.float32)
    selection_scores = _belief_safe_selection_scores(
        scores,
        feature_array,
        action_dim=np.asarray(clean_action, dtype=np.float32).reshape(-1).size,
        belief_safety_penalty=float(belief_safety_penalty),
        belief_cost_margin=float(belief_cost_margin),
        belief_constraint_margin=float(belief_constraint_margin),
    )
    selected_index = int(np.argmax(selection_scores))
    if selected_index != 0:
        predicted_advantage = float(selection_scores[selected_index] - selection_scores[0])
        if predicted_advantage < float(selection_margin):
            selected_index = 0
    return BVRSelection(
        action=np.asarray(candidates[selected_index].action, dtype=np.float32),
        clean_action=np.asarray(clean_action, dtype=np.float32),
        selected_index=selected_index,
        selected_candidate_type=candidates[selected_index].candidate_type,
        verifier_scores=scores,
        selection_scores=selection_scores.astype(np.float32),
        candidates=candidates,
    )


def _belief_safe_selection_scores(
    verifier_scores: np.ndarray,
    features: np.ndarray,
    *,
    action_dim: int,
    belief_safety_penalty: float,
    belief_cost_margin: float,
    belief_constraint_margin: float,
) -> np.ndarray:
    """Penalize candidates that look worse than the anchor on belief-route signals.

    The gate uses only deployment-available attacked-belief route features. If
    a caller supplies synthetic test features that do not follow the full BVR
    layout, this function leaves scores unchanged.
    """

    scores = np.asarray(verifier_scores, dtype=np.float32).reshape(-1).copy()
    feature_array = np.asarray(features, dtype=np.float32)
    if float(belief_safety_penalty) <= 0.0 or feature_array.ndim != 2 or feature_array.shape[0] != scores.size:
        return scores
    fixed_tail = 3 * int(action_dim) + 2 * len(OBJECTIVE_NAMES) + 1 + len(ROUTE_SCALAR_FEATURES) + len(OBJECTIVE_NAMES) * 2 + len(CONSTRAINT_FEATURE_KEYS)
    obs_dim = int(feature_array.shape[1]) - int(fixed_tail)
    if obs_dim < 0:
        return scores
    scalar_start = obs_dim + 3 * int(action_dim) + 2 * len(OBJECTIVE_NAMES) + 1
    success_idx = scalar_start
    belief_cost_idx = scalar_start + 2
    belief_constraint_idx = scalar_start + 3
    if belief_constraint_idx >= feature_array.shape[1]:
        return scores

    clean_success = float(feature_array[0, success_idx])
    clean_cost = float(feature_array[0, belief_cost_idx])
    clean_constraint = float(feature_array[0, belief_constraint_idx])
    candidate_success = feature_array[:, success_idx]
    candidate_cost = feature_array[:, belief_cost_idx]
    candidate_constraint = feature_array[:, belief_constraint_idx]
    cost_excess = np.maximum(candidate_cost - clean_cost - float(belief_cost_margin), 0.0)
    constraint_excess = np.maximum(candidate_constraint - clean_constraint - float(belief_constraint_margin), 0.0)
    success_drop = np.maximum(clean_success - candidate_success, 0.0)
    penalty = float(belief_safety_penalty) * (cost_excess + constraint_excess + success_drop)
    penalty[0] = 0.0
    return (scores - penalty.astype(np.float32)).astype(np.float32)


def save_bvr_checkpoint(path: str | Path, verifier: RouteVerifier, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_class": RouteVerifier.policy_class,
            "feature_dim": int(verifier.feature_dim),
            "hidden_sizes": list(verifier.hidden_sizes),
            "activation": verifier.activation,
            "verifier_state_dict": verifier.state_dict(),
            "metadata": dict(metadata),
        },
        path,
    )


def load_bvr_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[RouteVerifier, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    verifier = RouteVerifier(
        int(payload["feature_dim"]),
        hidden_sizes=tuple(int(size) for size in payload.get("hidden_sizes", (128, 128))),
        activation=str(payload.get("activation", "tanh")),
    )
    verifier.load_state_dict(payload["verifier_state_dict"])
    verifier.to(device)
    verifier.eval()
    return verifier, dict(payload.get("metadata", {}))
