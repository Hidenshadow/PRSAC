# Anonymous Reproducibility Release

This repository contains the code needed to reproduce the experiments for the submitted paper on robust global path recovery under corrupted multi-layer terrain beliefs. It intentionally excludes trained checkpoints, raw experiment outputs, generated figures, local logs, and large terrain rasters.

## Contents

- `algorithms/`, `envs/`, `planners/`, and `utils/`: planner, environment, and learning components.
- `configs/`: scenario, rover, and training configuration files.
- `scripts/`: experiment launch, preprocessing, evaluation, and plotting utilities.
- top-level `run_*.py`, `train_cleanrl_*.py`, and `evaluate_*.py`: main experiment entry points.

Some internal filenames retain legacy implementation names such as `ldac`; these correspond to the submitted method name PR-SAC.

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

The release does not include large DEM/DTM files. The real-terrain levels can be regenerated from public sources:

- Lunar Level 2: NASA PGDA LOLA NPD 5 m/pixel surface DEM, `NPD_final_adj_5mpp_surf.tif`.
  Download: https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/NPD/NPD_final_adj_5mpp_surf.tif
- Mars Level 3: HiRISE/PDS DTM product `DTEED_076968_1475_076823_1475_A01`.
  Product page: https://www.uahirise.org/dtm/ESP_076968_1475
  PDS directory: https://hirise.lpl.arizona.edu/PDS/DTM/ESP/ORB_076900_076999/ESP_076968_1475_ESP_076823_1475/

Use the preprocessing scripts in `scripts/` and `extract_real_dem_tile.py` to recreate project-format terrain tiles from these public products.

## Reproducing Experiments

The experiment protocol is:

1. clean-policy training;
2. corrupted-belief drop evaluation;
3. recovery training under corrupted planner-visible terrain beliefs;
4. held-out evaluation using true terrain-cost maps for reward/evaluation only.

Representative entry points:

```bash
python run_shock_recovery_experiment.py --help
python train_cleanrl_ppo.py --help
python train_cleanrl_sac.py --help
python scripts/evaluate_nonlearning_planner_baselines.py --help
```

For full reproduction, run the scenario-specific launch scripts under `scripts/` after preparing terrain tiles. New outputs are written to `runs/`, which is ignored by Git.

## Excluded Files

This anonymous release intentionally excludes:

- `runs/`, `exports/`, result tables, generated figures, logs, and TensorBoard files;
- trained checkpoints and model weights;
- raw DEM/DTM rasters and derived numpy terrain arrays;
- local IDE, virtual environment, and machine-specific files.
