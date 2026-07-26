# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI service that predicts food preparation time for a restaurant system. A GradientBoostingRegressor
(scikit-learn) predicts raw prep minutes from order features; the API then applies two adjustments before
returning an estimate: an `equipment_factor` (per-branch kitchen speed) and an "adaptive buffer" (inflates
the time shown to customers so food tends to arrive early). As of the 2026-07 rewrite, **every number in
this system comes from real Supabase order history — there is no synthetic data and no hand-picked business
constant anywhere in the pipeline.** Both `equipment_factor` and the buffer multipliers are statistics
estimated from real completed orders, refreshed on every `/retrain` call (see `calibration.py`).

## Commands

```bash
pip install -r requirements.txt

# Train/refresh the global model from real Supabase order history.
# Requires SUPABASE_URL / SUPABASE_SERVICE_KEY. If there isn't enough real data yet
# (fresh deploy, empty DB), this prints a warning and exits 0 WITHOUT writing a model —
# it never falls back to fabricated data.
python train_model.py

# Run the API locally
uvicorn main:app --reload --port 8000
```

There is no test suite and no linter configured in this repo.

Production (`Procfile`) always runs `train_model.py` on boot before starting uvicorn:
```
web: python train_model.py && uvicorn main:app --host 0.0.0.0 --port $PORT
```
Because `train_model.py` now exits 0 even when it trains nothing, this is safe on a fresh deploy — the API
starts and serves the honest `baseline_no_ml` estimate (see below) until real orders accumulate.

`.env` (based on `env.example`) needs `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` (service role key, not anon).
Every code path that touches a model or a learned parameter needs Supabase now — there is no synthetic
fallback left to degrade to. `/predict` still runs without Supabase configured, but only in the honest
`baseline_no_ml` mode described below (raw menu prep time, no ML, no equipment factor, no buffer).

## Architecture

**Modules, and why each exists:**
- `paths.py` — single source of truth for the on-disk model layout. `model/global/` is the fallback model;
  `model/branch_{id}/` is a branch-specific model once that branch has enough real orders. There is
  deliberately no legacy flat `model/*.pkl` path anymore — that dual-layout inconsistency from before the
  2026-07 rewrite is gone; everything reads and writes through this one module.
- `training_data.py` — the ONLY place real training data gets built from Supabase (`build_training_df`).
  Used by both `train_model.py` and `/retrain` in `main.py`, so the global bootstrap model and a live
  branch retrain are guaranteed to compute features identically. Also defines `FEATURES` (the canonical
  6-feature list) and the minimum-sample thresholds.
- `model_fit.py` — `chronological_split` (holds out the most RECENT real orders, never a random shuffle —
  a random split let old code evaluate a model against data from "the future" relative to its training
  set) and `fit_best_model` (GridSearchCV over GBR hyperparameters — nothing is hardcoded anymore).
- `calibration.py` — computes `equipment_factor` and the adaptive buffer table from real data. See below.
- `metrics_log.py` — appends every retrain attempt (accepted or rejected) to `model/retrain_history.jsonl`,
  since retrains used to silently overwrite the model file with no audit trail.

**Two things that used to be hardcoded business constants are now learned statistics, refreshed on every
`/retrain` call for that scope:**
- `equipment_factor` — used to be a number an admin typed into `POST /branches/equipment` (that endpoint is
  gone). Now `calibration.compute_learned_equipment_factor` computes it as the ratio of a branch's real
  `actual_prep_minutes / weighted_prep_time` against the same ratio computed globally, clipped to `[0.5, 2.0]`
  (a branch can be genuinely slower than baseline now, not just faster). Needs
  `MIN_SAMPLES_FOR_EQUIPMENT_FACTOR` (10) real branch orders; below that it's the neutral `1.0` — an
  identity value, not a guess. `GET /branches/equipment` is read-only now; the value is written only by
  `/retrain`.
- Adaptive buffer — used to be a fixed peak-hour set `{12,13,18,19,20}`, a fixed `queue_length > 5`
  threshold, and fixed multipliers (1.05–1.25) based on a psychology paper, never checked against this
  system's own data. Now `calibration.compute_learned_buffer_table` calibrates it empirically: it computes
  out-of-fold predictions (K-fold, so the model is scored against orders it didn't memorize) for every real
  order, buckets residual ratios by `hour_of_day` and by a data-derived `queue_length` quantile split, and
  takes the `BUFFER_QUANTILE` (0.85) of each bucket — i.e. "inflate the estimate enough that the real time
  beats it ~85% of the time," measured, not assumed. Buckets with fewer than `MIN_GROUP_FOR_BUFFER_BUCKET`
  (5) real orders are omitted; `lookup_buffer` falls back from the most specific bucket down to a neutral
  `1.0` when a scope has no calibration data yet at all.

**`/retrain` is the only place either of the above gets written**, and it always:
1. Pulls real orders via `training_data.build_training_df` — refuses with HTTP 422 below
   `MIN_SAMPLES_FOR_MODEL` (40; raised from the old 30 specifically so the chronological holdout below has
   at least ~15 rows instead of ~6 — a 6-row MAE is noise-dominated and was flagged as a real problem).
2. If `branch_id` is given, recomputes and upserts that branch's `equipment_factor` regardless of whether
   there's enough data for a full branch model (the factor only needs the lower `MIN_SAMPLES_FOR_EQUIPMENT_FACTOR`
   bar).
3. Splits chronologically (`model_fit.chronological_split`), grid-searches hyperparameters on the training
   slice, and evaluates the candidate AND the current reference model on the *same* chronological holdout —
   apples-to-apples, and never look-ahead.
4. Only saves the new model (+ refreshed buffer table) if holdout MAE ≤ old holdout MAE + 2.0 minutes — this
   fail-safe guard is unchanged from before and is worth keeping; don't remove it without understanding why.
5. Appends the full attempt (accepted or rejected) to `model/retrain_history.jsonl` via `metrics_log`.

**Prediction pipeline order** (see `predict()` in `main.py`):
1. Compute raw features from request `items`.
2. Resolve which model to use (`_get_model_for_branch`) — `"branch:{id}"` if that branch has its own
   retrained model, else `"global"` if a global model has been trained from real data, else
   `"baseline_no_ml"` if neither exists yet (a normal cold-start state, not an error).
3. `equipment_factor` is applied to `weighted_prep_time` only in `"global"` scope (branch models already
   encode their own kitchen's speed in their training data — applying the factor again would double count
   it). In `"baseline_no_ml"` scope no adjustment is applied at all.
4. If `model_scope == "baseline_no_ml"`: `estimated_minutes` is just the raw summed menu prep time — no
   ML claim is made, and the response is labeled as such. This is the honest alternative to what used to
   be "serve a synthetic-trained model and call it a prediction."
5. Otherwise, run inference to get `model_prediction`, look up the calibrated buffer multiplier for
   `(hour_of_day, queue_length)` via `calibration.lookup_buffer`, and multiply to get `estimated_minutes`.

The `breakdown` dict in every `/predict` response exposes `model_prediction` (pre-buffer),
`estimated_minutes` (post-buffer), `buffer_label` (which calibration bucket fired, or `"no_data_yet"` /
`"no_model_yet"`), and `model_scope` (`"global"`, `"branch:{id}"`, or `"baseline_no_ml"`) — treat these
fields as a stable contract when touching the response shape.

Supabase calls are wrapped defensively in read paths (`_get_equipment_factor`, `GET /branches/equipment`):
a missing/misconfigured Supabase connection falls back to defaults rather than raising. `/retrain` does
raise on Supabase failure, since that's an explicit write/train action where silent fallback would hide a
real failure.

## Known gaps worth knowing about before extending this further

- No auth on `/retrain` or the model-mutating paths — anyone who can reach the API can trigger a retrain.
  Not addressed in the 2026-07 rewrite; flag before exposing this publicly.
- No model versioning beyond the JSONL log — a saved model still overwrites the previous `.pkl` in place.
  `retrain_history.jsonl` gives you the metrics trail but not a way to roll back to a previous model file.
- `model_old_synthetic_backup_kept/` in the repo root is the pre-rewrite synthetic-trained model, renamed
  out of the way (not deleted) so nothing reads it by accident. Safe to delete once you've confirmed you
  don't need to compare against it.
