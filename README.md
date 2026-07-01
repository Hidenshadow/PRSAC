# PR-SAC Reproducibility Code

This anonymous repository contains the code used for the paper on robust global
path recovery for planetary rovers under corrupted multi-layer terrain beliefs.
It is a source-code release only: trained checkpoints, raw experiment outputs,
generated figures, logs, and large DEM/DTM files are intentionally excluded.

PR-SAC is the paper name of the proposed method. Some implementation files keep
the legacy internal name `ldac`; those entries correspond to PR-SAC.

## Repository Layout

- `maps/`, `envs/`, `planners/`, `utils/`: map generation/loading, corrupted-belief
  wrappers, weighted A* planning, metrics, policy loading, and shared recovery
  helpers.
- `configs/levels/ppo_difficulty/`: the nine Easy/Medium/Hard by Level 1/2/3
  scenario definitions used by the paper.
- `configs/rovers/`: rover parameter files for the lunar and Mars levels.
- `scripts/`: only the non-learning baseline evaluator, robust-SAC recovery
  preparation scripts, and Mars terrain preprocessing helpers.
- `train_cleanrl_ppo.py`, `train_cleanrl_sac.py`: PPO and SAC trainers for
  planner-preference policies.
- `run_shock_recovery_experiment.py`: clean training, corrupted-belief drop
  evaluation, and recovery training protocol.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Public Terrain Data

Level 1 uses synthetic controlled maps generated from fixed seeds and requires
no external download.

Level 2 uses NASA PGDA LOLA lunar south-pole terrain:
https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/NPD/NPD_final_adj_5mpp_surf.tif

Level 3 uses the HiRISE/PDS Mars DTM product:
https://www.uahirise.org/dtm/ESP_076968_1475

The raw public rasters and derived numpy terrain tiles are not committed. See
`docs/DATA_SOURCES.md`, `extract_real_dem_tile.py`,
`scripts/prepare_level3_mars_assets.py`, and
`scripts/prepare_level3_mars_dteed_difficulty_tiles.py` for recreation steps.

## Reproduction Workflow

The main experiment protocol is:

1. Train a clean planner-preference policy.
2. Evaluate the clean policy after the corrupted terrain belief is introduced.
3. Fine-tune under the corrupted planner-visible belief.
4. Evaluate held-out performance with the true/reference terrain cost used only
   for reward and evaluation.

Representative commands:

```bash
python run_shock_recovery_experiment.py --help
python train_cleanrl_ppo.py --help
python train_cleanrl_sac.py --help
python scripts/evaluate_nonlearning_planner_baselines.py --help
```

Non-learning baselines are evaluated with:

```bash
python scripts/evaluate_nonlearning_planner_baselines.py --help
```

The release deliberately omits plotting, table-generation, front-page figure,
appendix figure, and local launch/watch scripts. Reproduction is driven by the
Python training/evaluation entry points above; new outputs are written under
`runs/`, which is ignored by Git.

## Excluded Artifacts

This release excludes:

- `runs/`, `exports/`, result tables, generated figures, logs, and TensorBoard
  files;
- trained checkpoints and model weights;
- raw DEM/DTM rasters and derived `.npy`/`.npz` terrain arrays;
- local IDE, virtual environment, and machine-specific files.
