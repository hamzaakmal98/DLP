MMoE Integration Plan (sketch)
================================

Goal
-----
Provide a short, actionable plan for integrating the current single-task `is_like` predictor
into a future multi-task MMoE scorer used by the recommender stack.

How the current single-task model plugs in
-----------------------------------------
- The existing `is_like` model (MLP or LightGBM pipeline) becomes one task head in the MMoE.
- Keep the pre-processing and feature registry unchanged: MMoE consumes the same leakage-safe
  pre-exposure features produced by `src/preprocess.py`.
- Replace the single-task trunk with a shared expert layer: several small experts (each a small MLP)
  compute intermediate representations from the same preprocessed inputs.
- A gating network (task-specific) computes a weighted combination of expert outputs for each task.
- The `is_like` head will be a simple binary head (single linear neuron producing logits) fed with
  the gated expert mixture.

Changes needed for multi-task training
-------------------------------------
1. Model changes
   - Implement expert MLPs and per-task gates.
   - Replace the single `MLPModel` with a shared expert module and task-specific heads.
   - Optionally allow some task-specific bottom layers before heads.

2. Loss & optimization
   - Define per-task losses (e.g., BCE for `is_like` and `click`, regression/other for continuous tasks).
   - Use a weighted sum of task losses; provide configurable task weights and optional dynamic
     loss balancing strategies (GradNorm, uncertainty weighting).

3. Data pipeline
   - Ensure the dataset exposes all targets for multi-task examples (some tasks may be missing —
     use masks in loss computation).
   - Train/validation/test splits should be the same across tasks (time-aware splitting still applies).

4. Metrics & monitoring
   - Track per-task metrics (ROC-AUC, PR-AUC, log loss, plus ranking metrics where relevant).
   - Monitor downstream fairness/coverage metrics if tasks influence ranking.

Prediction output format for downstream reranking
------------------------------------------------
- Produce a compact per-impression JSON/CSV record with:
  - `user_id`, `video_id`, `timestamp`
  - Per-task probabilities (e.g., `p_like`, `p_click`, `p_long_view`)
  - Optional per-task logits if downstream needs calibration
- Downstream rerankers can consume `p_*` scores as features; for combined objectives produce
  a scalar utility score (weighted sum of tasks) when a single ranking score is required.

Implementation notes and rollout
-------------------------------
- Start by implementing the MMoE stub (interface) and a minimal expert/gate implementation in PyTorch.
- Validate transfer by training single-task vs multi-task on the `is_like` target and comparing metrics.
- Keep the feature registry and leakage checks unchanged and enforced: multi-task models must not
  use post-exposure signals.

Where to read or extend
-----------------------
- See `src/models/mmoe_stub.py` (this file) for the integration skeleton and comments.
- Add a concrete PyTorch module under `src/models/` when ready to implement experts and gates.

Summary
-------
This plan keeps the current safe preprocessing and single-task pipelines, and outlines a minimal
path to migrate the `is_like` predictor into a shared-expert multi-task scorer used by the recommender.
