# Research Position

This project now focuses on PPO-only robustness evaluation for
planner-in-the-loop rover routing.

The policy does not directly output rover movements. It outputs weighted A*
planner parameters:

```text
[w_distance, w_energy, w_hazard, w_communication, w_illumination, lambda_uncertainty]
```

Weighted A* plans the complete path, and evaluation measures scalar path cost
under clean and attacked settings.

## Main Question

How robust is a PPO planner-parameter policy under:

- no attack,
- observation attack,
- environmental attack,
- combined observation and environmental attack?

The active comparison is among PPO variants only:

```text
nominal PPO
env-attack fine-tuned PPO
obs-attack fine-tuned PPO
```

## Reward Handling

Nominal PPO uses the true scalar cost with `reward_mode = relative_heuristic`:

```text
reward = reward_scale * (heuristic_cost - policy_cost) / abs(heuristic_cost)
```

Environmental fine-tuning can switch the reward cost to attacked cost when
`reward_uses_attacked_cost = true`.

Observation fine-tuning corrupts only the policy input; reward is still computed
from the path on the true underlying map.

## Evaluation

Final comparison should use scalar cost and degradation metrics, not PPO reward:

```text
absolute_degradation = attacked_cost - nominal_cost
relative_degradation = absolute_degradation / abs(nominal_cost)
```

Evaluation is run on shared episode seeds so all PPO variants see the same map,
start, and goal distributions.
