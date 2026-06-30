# PPO Robustness Commands

All commands assume they are run from the project root.

## Check

```bash
python run_tests.py
python run.py
```

## One Command: Single Seed

Formal run. All settings except seed are in `configs/ppo_robustness_experiment.json`:

```bash
python run_ppo_robustness_experiment.py --seeds 0
```

Dry-run the exact commands without launching training:

```bash
python run_ppo_robustness_experiment.py --seeds 0 --dry-run
```

## Multi-Seed

```bash
python run_ppo_robustness_experiment.py --seeds 0,1,2
python run_ppo_robustness_experiment.py --seeds 0-4
```

## Individual Steps

Train nominal PPO:

```bash
python run_robustness_workflow.py --base-config configs/ppo_lunar_map_pool_relative_reward.json --mode train_nominal
```

Fine-tune env-attack PPO:

```bash
python run_robustness_workflow.py --base-config configs/ppo_lunar_map_pool_relative_reward.json --mode finetune_env_attack --checkpoint runs/robustness/nominal_ppo/checkpoint.pt
```

Fine-tune obs-attack PPO:

```bash
python run_robustness_workflow.py --base-config configs/ppo_lunar_map_pool_relative_reward.json --mode finetune_obs_attack --checkpoint runs/robustness/nominal_ppo/checkpoint.pt
```

Evaluate:

```bash
python evaluate_robustness.py --config configs/evaluate_robustness_lunar_in_domain.json --output runs/robustness/results/in_domain_summary.csv
python evaluate_robustness.py --config configs/evaluate_robustness_lunar_heldout.json --output runs/robustness/results/heldout_summary.csv
```

Visualize:

```bash
python visualize_robustness_results.py --input runs/robustness/results/robustness_summary.csv --output-dir runs/robustness/results/figures
```
