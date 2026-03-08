# Design Document — KuaiRand Pareto MMoE

This document describes the design and rationale for the multi-objective recommendation pipeline implemented in this repository.

---

## 1. Problem definition

We rank candidate videos for individual users to maximize multiple engagement objectives simultaneously. Primary objectives in this project are:
- `like` — whether the user liked the video
- `long_view` — whether the impression resulted in a long watch
- `creator_interest` — creator-level engagement (follow or proxy)

The core problem is a multi-objective ranking problem: generate a candidate set for a user and produce an ordered recommendation list that balances these (sometimes conflicting) objectives.

## 2. Why a multi-objective recommender is appropriate

- Platform goals are multi-faceted: maximizing a single metric (e.g., likes) can harm watchtime or creator discovery.
- Multi-objective models surface tradeoffs and allow stakeholders to pick operating points rather than relying on a single collapsed metric.
- In research and production, explicitly modelling multiple objectives reduces the need for ad-hoc reweighting in downstream systems.

## 3. Why we use a custom MMoE-inspired predictor

- MMoE (Multi-gate Mixture-of-Experts) provides a principled way to share low-level representation learning while allowing task-specific routing and specialization via experts and gates.
- A custom MMoE-inspired design in this repository supports:
  - A shared encoder that produces dense, fused feature embeddings suitable for all tasks.
  - Multiple experts that learn diverse subspace projections useful across tasks.
  - Task-specific gates that produce task-conditioned mixtures of expert outputs, enabling specialization without complete parameter isolation.
- This balances statistical efficiency (shared signal) with task specialization (separate gates/towers), which is ideal for related engagement objectives.

## 4. Why we use Pareto frontier reranking

- Scalarization (weighted sums) requires committing to weights; Pareto reranking exposes the non-dominated set of tradeoff-optimal candidates.
- Pareto frontiers let decision makers inspect alternatives (each good on different objectives) rather than forcing a single operating point.
- For research and deployment, Pareto-based outputs are especially useful when different stakeholders (engagement, creator health, retention teams) prefer different tradeoffs.

## 5. End-to-end system flow

1. Raw data ingestion: collect impression logs and feature tables (user, video statistics, video basic attributes) from `KuaiRand-Pure` CSV files.
2. Preprocessing and feature engineering: cleaning, join interactions, target engineering (including `creator_interest` proxy), leakage-safe feature selection; outputs saved to `data/processed` and `artifacts/feature_metadata`.
3. EDA & reports: generation of presentation-ready figures and summaries under `artifacts/figures/eda` and `reports/analysis`.
4. Model training: `SharedEncoder` -> `CustomMMoE` experts & gates -> task-specific heads; trainer saves checkpoints in `artifacts/checkpoints` and training history under `artifacts`.
5. Prediction: model scores per-objective (`like_score`, `longview_score`, `creator_score`) are produced for candidate sets or full holdout.
6. Candidate generation: union of top-N candidates per objective (scripted in `src/rerank/candidate_generation.py`).
7. Two reranking branches applied to candidate pool:
   - Scalarization branch: min-max normalization + weighted sum to produce single `scalar_score` for ranking and sweep experiments.
   - Pareto frontier branch: non-dominated sorting to extract frontier candidates per user and present frontier-ordered recommendations.
8. Evaluation: ranking metrics (NDCG@K, Precision@K, Recall@K) computed per-objective and aggregated; classification metrics (ROC-AUC, PR-AUC) reported per-task.
9. Final analysis report: `reports/analysis/final_experiment_narrative.md` summarizes comparisons and tradeoffs.

## 6. Training-time architecture

- Input features: categorical embeddings (user_id, video_id, creator_id, etc.), numeric features (video statistics, user aggregates), and engineered features from the registry.
- `SharedEncoder`: maps embeddings and numeric features into a shared projection vector (configurable projection_dim).
- `CustomMMoE`: consists of N experts (MLP blocks) producing expert outputs; each task has a learned gate producing weights over experts; gates use the shared encoder output as input.
- Task towers / heads: task-specific small MLPs consuming the gated expert mixture to produce logits for each objective.
- Loss: task-wise binary cross-entropy (or other appropriate losses) combined via configurable loss weights.
- Optimization and training harness: early stopping, checkpointing, and training history saved for reproducibility.

## 7. Inference-time architecture

- Given a user and a candidate set of items, inference proceeds:
  1. Compute feature vectors for each user-item candidate (same featurization as training).
  2. Run `SharedEncoder` to produce shared embeddings.
  3. Run experts and task gates in `CustomMMoE` to produce per-task scores.
  4. Produce per-objective scores (`like_score`, `longview_score`, `creator_score`) from task heads.
- Downstream options:
  - scalarize scores into a single ranking (fast, single-pass) for production use; or
  - compute Pareto frontier on the candidate pool (slower, preserves tradeoffs) and render frontier to stakeholders.

## 8. Evaluation methodology with NDCG@K

- Primary ranking metric: NDCG@K computed per-user and averaged across users. K choices (e.g., 5, 10, 20) are configurable.
- For each method (single-objective, scalarized, Pareto), we compute:
  - NDCG@K per objective (using the appropriate target for that objective)
  - Precision@K and Recall@K as auxiliary checks
  - Classification metrics (ROC-AUC, PR-AUC, log loss) on held-out examples where applicable
- Aggregation: average across users after computing per-user NDCG to avoid domination by highly active users; optionally report median and standard error.
- Statistical checks: weight-sweep comparisons and Pareto metrics are reported alongside absolute NDCG values; significance testing is recommended for production claims.

## 9. Assumptions and limitations

- Dataset representativeness: evaluation relies on logged impressions; selection bias exists where logs reflect non-random exposures. We prefer random-exposure logs where available to mitigate this.
- Creator interest proxy: when an explicit follow column is unavailable, we use a heuristic proxy (repeat interactions); proxies introduce measurement noise and may bias toward popular creators.
- Candidate generation: results depend heavily on the candidate pool. Our pipeline uses top-N per-objective union; alternative candidate generators may yield different frontiers.
- Scalability: Pareto extraction per user is more expensive than scalarized ranking; production deployment requires batching or approximations for large candidate pools.
- Categorical encodings: current prototype factorizes categories per-file; production requires consistent, persisted encoders across splits.

## 10. Planned future work

- Persist categorical encoders and ensure identical mappings across train/val/test and production.
- Add experiment tracking integration (MLflow, W&B) for reproducible sweeps and logging.
- Implement constrained reranking (e.g., constraints for fairness, exposure limits) built on top of Pareto frontiers.
- Add faster approximate Pareto algorithms and GPU-accelerated frontier extraction for large-scale inference.
- Add unit tests and CI for critical components (rerank, evaluation, preprocess) and synthetic-data integration tests.

---
