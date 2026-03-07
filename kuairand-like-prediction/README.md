# kuairand-like-prediction

This repository is a compact, research-oriented prototype for predicting the `is_like` label
in KuaiRand-style user × video interaction logs. 

Key goals
- Provide tooling to load KuaiRand-format CSVs.
- Produce a leakage-safe dataset for predicting `is_like`.
- Train and evaluate three baseline families: Logistic Regression, LightGBM, and a PyTorch MLP.
- Evaluate using ROC-AUC, PR-AUC, log loss, and top-k ranking metrics (precision@k by user).

Important modeling constraint
- Only pre-exposure features may be used when training the main `is_like` predictor.
	Pre-exposure features are attributes available before the user sees the candidate video
	(e.g., user profile, video metadata, historical summary features computed only from
	events that occur before the exposure time).
- Post-exposure signals (click, watch duration, later engagement counts, or any feature
	that depends on the user's behaviour after the exposure) MUST NOT be used as predictors
	for the main `is_like` model. See the `Data leakage rules` section for concrete examples.

Project structure
- `configs/` — YAML config files (default hyperparameters and paths). See `configs/default.yaml`.
- `data/` — place KuaiRand CSVs here (ignored by git). Default path used by the example config:
	`data/KuaiRand-1K.csv`.
- `src/` — source code: data loading, dataset building, features, models, training and eval.
- `src/models/` — model implementations: `logreg`, `lgbm`, and `torch_mlp`.
- `notebooks/` — experiment notebooks and EDA.
- `reports/` — figures and run outputs.
- `models/` — model artifacts (created at runtime; ignored by git).
- `tests/` — lightweight unit tests.

Quick setup (Python virtual environment)
1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r kuairand-like-prediction/requirements.txt
```

Running training
- By default, the training entrypoint is `src/train.py`, which reads `configs/default.yaml`.
- Basic command (uses the config file to decide which models to run):

```powershell
python -m kuairand-like-prediction.src.train --config kuairand-like-prediction/configs/default.yaml
```

To run a single model, edit `configs/default.yaml` and set `train.models` to `['logreg']`,
`['lgbm']`, or `['mlp']`, or create a copy of `default.yaml` and point `--config` at it.

Examples (change `models` in the config or create a short config snippet):
- Baseline logistic regression (set `train.models: [logreg]` in the config and run the command above).
- LightGBM (set `train.models: [lgbm]` and tune `models.lgbm` hyperparameters in the config).
- PyTorch MLP (set `train.models: [mlp]` and tune `models.mlp` hyperparameters in the config).

Evaluation
- The training script saves models to the directory specified by `output.dir` in the config
	and prints evaluation metrics to stdout (ROC-AUC, PR-AUC, log loss). It also computes
	precision@1 and precision@5 aggregated by user when `user_id` is available in the dataset.
- For reproducible analysis and plotting, use the notebooks under `notebooks/` to load
	saved models and produce ROC/PR curves and per-user ranking analyses.

Data leakage rules (short, concrete guidance)
- Allowed (pre-exposure):
	- User profile fields (age bucket, country, device type) as recorded before exposure.
	- Video metadata (duration, category, tags) from the time of the exposure.
	- Aggregated historical summaries computed strictly from events that occurred before the
		exposure timestamp (e.g., user's past like-rate on similar videos computed up to exposure time).
- Forbidden (post-exposure / leaking features):
	- Click indicators for the candidate impression (did the user click this video?).
	- Watch duration or watch-fraction for the candidate impression or any subsequent watches.
	- Aggregates that include events after the exposure (e.g., future engagement counts, or
		including the current impression in the computation of historical rates).
	- Features that directly encode the target or are computed from a label proxy.
- Practical checks:
	- Every feature should answer: "Would this be available to the model before we showed the item?"
	- If the answer is no, remove or re-compute the feature so it is only based on pre-exposure data.

Planned extension toward MMoE
- The current repo provides a single-task `is_like` baseline. A natural extension is a Multi-gate
	Mixture-of-Experts (MMoE) model to jointly predict related engagement signals (e.g., click, watch,
	subscribe) while sharing low-level representations. Steps for that extension:
	1. Define multiple targets and a cleaned, leakage-safe multi-task dataset.
 2. Implement shared experts and task-specific gating layers in PyTorch.
 3. Add multi-task loss weighting and evaluation (per-task metrics + combined ranking metrics).
 4. Compare single-task vs multi-task performance to validate transfer gains.



