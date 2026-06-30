# Experiment Direction

## Current Main Direction

PRB-PPO: Planner-Residual Belief PPO.

The current paper protocol is official-shock-only recovery:

1. Train a clean nominal PPO policy on clean maps.
2. Introduce the official structured shock.
3. Continue recovery training under the same official attacked environment.
4. Plot clean nominal training -> shock/drop -> recovery under attack.

PRB-PPO preserves the PPO recovery optimizer, reward path, and GAE path. It adds a residual-belief latent inferred from planner-belief versus true-attacked-cost discrepancies under the same official shock. Auxiliary losses train residual/cost prediction heads; they do not rewrite PPO advantages or shape rewards.

Main metric: `residual_degradation`, lower is better.
Secondary metric: `recovery_closure`, higher is better.

## Main Algorithms

- `ppo`: standard same-shock PPO recovery baseline.
- `sac`: SAC recovery baseline when available.
- `prb_ppo`: current proposed same-shock residual-belief recovery method.

## Archived Exploratory Directions

- `mirror_ppo`: attack-variant robust game. Useful for robustness analysis, but not aligned with the current same-shock-only paper direction.
- `pr_ppo` / `rp_ppo`: direct planner-action guidance, including MSE/NLL/CPA/pairwise preference losses. These showed planner counterfactual signal exists but did not beat PPO reliably.
- `mra_ppo`: trajectory-tail risk weighting. Did not stably beat PPO in the current protocol.
- post-hoc repair / conformal repair / policy libraries: useful diagnostics, not part of the current main algorithm.

## First PRB Sanity Runs

Run seed0 quick tests before any 20480-step or 5-seed sweep:

```bash
./.venv/bin/python scripts/run_prb_ppo_suite.py \
  --suite custom \
  --experiments level3_easy_shock_recovery_5seeds \
  --seeds 0 \
  --preset prb_mlp_aux005 \
  --output-root runs/prb_ppo_quick \
  --device auto \
  --max-jobs 1 \
  --no-only-missing
```

Compare against synchronized PPO at fixed recovery steps `1024`, `2048`, and `3072`.

Proceed to `level3_hard_shock_recovery_5seeds seed0` and `level2_medium_shock_recovery_5seeds seed0` only if level3_easy improves without the old PR/RP plateau around cost `4.135`.
