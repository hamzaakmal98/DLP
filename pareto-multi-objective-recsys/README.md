# KuaiRand Pareto MMoE

Research project for multi-objective recommendation on the KuaiRand-Pure dataset.

This repository implements a reusable, configuration-driven pipeline for:

- training MMoE-inspired multi-task models,
- Pareto-frontier reranking to study tradeoffs between objectives,
- evaluating ranking quality using NDCG@K, and
- producing presentation-ready EDA, figures, and tables.

Table of contents
- Problem description
- Motivation
- Dataset
- Repo structure
- Setup
- Data placement
- Main commands
- Data Leakage Rules
- Outputs
- Future Extensions

See `docs/` for detailed design, data engineering, and experiment notes.

## Problem description

- Task: multi-objective recommendation on the KuaiRand-Pure dataset.
- Goal: rank candidate videos for individual users while balancing multiple engagement objectives.

Primary objectives:

- `like`
- `long_view`
- `creator_interest`

Final evaluation is performed with NDCG@K. We also use Pareto frontier analysis and reranking to study and produce non-dominated sets of recommendations that balance the objectives above.

## Motivation

- Optimizing a single objective (e.g., only likes) can degrade other user and platform goals.
- Multi-objective recommendation provides fine-grained control and visibility into tradeoffs between engagement metrics.
- Pareto frontier reranking identifies non-dominated candidate lists, enabling principled selection of balanced recommendations.

## Dataset

- Source: KuaiRand-Pure.
- We use the random-exposure log where available to reduce exposure bias in evaluation and model analysis.
- Important: current-row engagement outcomes (e.g., whether the user liked this specific impression) are treated as targets and must NOT be used as predictor inputs during training or inference (see Data Leakage Rules).

## Repo structure

- `artifacts/` — experiment outputs (models, checkpoints, figures, tables)
- `configs/` — YAML configurations for experiments and training runs
- `data/` — dataset storage
	- `data/raw/` — original KuaiRand-Pure files (do not commit)
	- `data/interim/` — intermediate preprocessing artifacts
	- `data/processed/` — cleaned and split data ready for modeling
- `docs/` — design, data engineering, and experiment notes
- `notebooks/` — exploratory notebooks (EDA, modeling, reranking)
- `reports/` — saved metrics, predictions, and analysis outputs
- `scripts/` — convenience scripts and run wrappers (e.g., `run_train.py`)
- `src/` — primary source package
	- `src/data/` — data loaders and preprocessing
	- `src/features/` — feature engineering
	- `src/models/` — model definitions (MMoE-inspired architectures)
	- `src/rerank/` — Pareto reranking utilities
	- `src/evaluation/` — NDCG@K and other metrics
	- `src/visualization/` — plotting and presentation helpers
	- `src/utils/` — config, logging, and shared utilities
- `tests/` — unit/integration tests
- `requirements.txt`, `pyproject.toml`, `Makefile` — environment and run helpers

## Setup

1. Create a virtual environment and activate it (example using venv):

```bash
python -m venv .venv
.
# Windows PowerShell
& .venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run tests to verify the scaffold:

```bash
make test
```

## Data placement

Place the KuaiRand-Pure dataset inside `data/raw/KuaiRand-Pure/`. Keep raw data out of version control. After placing raw files, run preprocessing to produce `data/processed/` artifacts used by training and evaluation.

Recommended layout:

```
data/
	raw/KuaiRand-Pure/...
	interim/...
	processed/train.csv
	processed/val.csv
	processed/test.csv
```

## Main commands

Use the `Makefile` targets to run common workflows from the repository root. Each command writes detailed logs to `logs/` and high-level progress to the terminal.

- Inspect raw dataset:

```bash
make inspect
```

- Prepare data:

```bash
make prepare-data
```

- Run EDA (scripted):

```bash
make eda
```

- Train model (uses `configs/model.yaml` by default):

```bash
make train
```

- Generate predictions on holdout (example):

```bash
make predict
```

- Run baseline comparisons:

```bash
make baselines
```

- Run Pareto reranking pipeline:

```bash
make pareto
```

- Evaluate predictions and compute NDCG@K:

```bash
make evaluate
```

- Generate the final analysis report (markdown):

```bash
make full-report
```

Notes: most `make` targets call the corresponding `scripts/*.py` entrypoint and accept the same CLI flags (e.g., `--out-root`, `--config`) when run manually. See `Makefile` for defaults.


## Outputs

Key outputs and their locations:

- Trained models: `artifacts/models/`
- Checkpoints and optimizer state: `artifacts/checkpoints/`
- Figures and plots: `artifacts/figures/`
- Result tables and CSVs: `artifacts/tables/` and `reports/predictions/`
- Metrics and evaluation reports: `reports/metrics/`


## Future Extensions

- Implement a production-like inference pipeline and batch scoring utilities.
- Add hyperparameter sweeps and experiment tracking (e.g., MLflow, Weights & Biases).
- Extend reranking module to support constrained optimization (e.g., fairness or exposure constraints).
- Implement more robust model baselines (e.g., LightGBM, simple neural baselines) and automated comparison dashboards.

---
