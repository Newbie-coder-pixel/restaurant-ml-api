# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI service that predicts food preparation time for a restaurant system. A GradientBoostingRegressor
(scikit-learn) predicts raw prep minutes from order features; the API then applies two independent
adjustments before returning an estimate: an `equipment_factor` (per-branch kitchen speed, from Supabase)
and an "adaptive buffer" (deliberately overestimates the time shown to customers so food tends to arrive
early — see the theory comment block at the top of `main.py`).

## Commands

```bash
pip install -r requirements.txt

# Train/regenerate the baseline synthetic model (writes model/prep_time_model.pkl + model/features.pkl)
python train_model.py

# Run the API locally
uvicorn main:app --reload --port 8000
```

There is no test suite and no linter configured in this repo.

Production (`Procfile`) always retrains the synthetic baseline model on boot before starting uvicorn:
```
web: python train_model.py && uvicorn main:app --host 0.0.0.0 --port $PORT
```
Keep this in mind when changing `train_model.py` — a bug there breaks deploys, not just local training.

`.env` (based on `env.example`) needs `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` (service role key, not anon)
for the `/retrain` and `/branches/equipment` endpoints. `/predict` degrades gracefully without Supabase
configured (equipment_factor just falls back to `1.0`).

## Architecture

**Model resolution is per-branch with fallback to global**, and there are two on-disk layouts in play:
- `model/global/{prep_time_model.pkl,features.pkl}` — current global/fallback model
- `model/branch_{branch_id}/{prep_time_model.pkl,features.pkl}` — model trained on one branch's real data
- `model/{prep_time_model.pkl,features.pkl}` (flat, no subfolder) — legacy location from before per-branch
  models existed; `_load_global_model()` in `main.py` reads this only if `model/global/` doesn't exist yet.
  `train_model.py` still writes to this legacy flat path, not `model/global/` — so a from-scratch
  `python train_model.py` run and a live `/retrain`-managed deployment currently produce models in
  different locations. Be aware of this when changing either save path.

**Two data sources feed the models**: `train_model.py` generates 2000 synthetic samples with a hand-tuned
formula (menu prep times + queue/peak-hour heuristics + noise) for the bootstrap/global model.
`/retrain` in `main.py` instead builds real training data from Supabase `order_items`/`orders`/`menu_items`
(`_build_training_df`), aggregating per order and computing actual prep time from
`prepared_at - sent_to_kitchen_at` timestamps. Both paths must produce the exact same feature set
(`total_quantity`, `total_item_types`, `weighted_prep_time`, `special_request_count`, `hour_of_day`,
`queue_length`) since a branch model and the global model are used interchangeably at inference time.

**`/retrain` only overwrites an existing model if it's not worse**: it trains a candidate model, compares
its MAE on a held-out split against the *current* model for that scope (branch model if present, else
global), and only saves if `new_mae <= old_mae + 2.0` minutes. It also refuses to run with fewer than 30
orders for the requested scope. This means a bad retrain request fails safe rather than silently
degrading predictions — don't "fix" a rejected retrain by removing this guard without understanding why
the caller wanted a stricter one.

**Prediction pipeline order** (see `predict()` in `main.py`) matters and is easy to get backwards:
1. Compute raw features from request `items`.
2. Multiply `weighted_prep_time` by `equipment_factor` (Supabase-configured per branch, clamped 0.3–1.0)
   *before* it goes into the model — equipment speed affects the model's input, not its output.
3. Run inference to get `model_prediction`.
4. Multiply `model_prediction` by the adaptive buffer multiplier (based on `hour_of_day` peak windows and
   `queue_length` threshold) to get `estimated_minutes` — the buffer is applied *after* inference and is
   the only thing that inflates the number shown to the customer.

The `breakdown` dict in every `/predict` response intentionally exposes both the pre-buffer
`model_prediction` and the post-buffer `estimated_minutes`, plus `model_scope` (`"global"` or
`"branch:{id}"`), so the Flutter client can show "why this estimate" — treat these fields as a stable
contract when touching the response shape.

Supabase calls are wrapped defensively throughout (`_get_equipment_factor`, `/branches/equipment`): a
missing/misconfigured Supabase connection falls back to defaults rather than raising, except for
`/retrain` and `POST /branches/equipment`, which do raise since those are explicit write/train actions
where silent fallback would hide a real failure.
