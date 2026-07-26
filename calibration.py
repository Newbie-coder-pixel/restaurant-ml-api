"""Data-driven replacements for what used to be hardcoded business constants.

Previously:
  - equipment_factor was a number an admin typed into a form (0.3-1.0).
  - the adaptive buffer used a fixed peak-hour set {12,13,18,19,20}, a fixed
    queue_length > 5 threshold, and fixed multipliers (1.05/1.10/1.15/1.25),
    picked from theory (Oliver, 1980) but never checked against this system's
    own data.

Both are now statistics estimated from real completed orders and refreshed on
every /retrain call. The only constants left below are sanity clips (a factor
or buffer multiplier still has to stay in a physically sane range) and minimum
sample-size floors below which we refuse to trust a statistic — neither of
those encodes a business assumption about *when* a kitchen is busy.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

from training_data import FEATURES, TARGET, MIN_SAMPLES_FOR_EQUIPMENT_FACTOR

EQUIPMENT_FACTOR_CLIP = (0.5, 2.0)  # a branch's kitchen can be faster OR slower than baseline
BUFFER_CLIP = (1.0, 1.5)  # buffer only ever inflates, never shrinks, the model's own prediction
BUFFER_QUANTILE = 0.85  # calibration target: actual time should beat the shown estimate ~85% of the time
MIN_GROUP_FOR_BUFFER_BUCKET = 5


def compute_learned_equipment_factor(df_branch: pd.DataFrame, df_global: pd.DataFrame) -> tuple[float, int]:
    """How much faster/slower this branch's real kitchen is vs the global baseline,
    measured the same way for both: actual_prep_minutes / weighted_prep_time.

    Returns (factor, n_samples_used). Falls back to the neutral factor 1.0 (i.e. no
    adjustment at all) when there isn't enough real branch data yet to trust the
    ratio — 1.0 here is an identity value, not a guessed business constant.
    """
    n = len(df_branch)
    if n < MIN_SAMPLES_FOR_EQUIPMENT_FACTOR or df_global.empty:
        return 1.0, n

    branch_ratio = (df_branch[TARGET] / df_branch["weighted_prep_time"].replace(0, np.nan)).median()
    global_ratio = (df_global[TARGET] / df_global["weighted_prep_time"].replace(0, np.nan)).median()

    if not np.isfinite(branch_ratio) or not np.isfinite(global_ratio) or global_ratio == 0:
        return 1.0, n

    factor = float(np.clip(branch_ratio / global_ratio, *EQUIPMENT_FACTOR_CLIP))
    return factor, n


def _out_of_fold_predictions(df: pd.DataFrame, model_params: dict) -> np.ndarray:
    """Predict every historical order as if it were unseen, so the buffer is
    calibrated against the errors the model actually makes on new orders — not
    against orders it already memorized (which would understate how much buffer
    is really needed)."""
    X = df[FEATURES]
    y = df[TARGET]
    n_splits = max(2, min(5, len(df)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(df))
    for train_idx, test_idx in kf.split(X):
        m = GradientBoostingRegressor(**model_params)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = m.predict(X.iloc[test_idx])
    return np.clip(oof, 1, None)


def compute_learned_buffer_table(df: pd.DataFrame, model_params: dict) -> dict:
    """Empirically calibrate how much to inflate the shown estimate, bucketed by
    hour_of_day and by a data-derived queue_length quantile split, so the real
    finish time beats the shown estimate roughly BUFFER_QUANTILE of the time.

    Buckets with too few real orders are simply omitted (lookup falls back to a
    coarser bucket, see lookup_buffer) rather than guessed.
    """
    n = len(df)
    if n < MIN_GROUP_FOR_BUFFER_BUCKET:
        return {"per_hour": {}, "queue_bin_edges": [], "per_queue_bin": [], "default": 1.0, "samples_used": n}

    oof_pred = _out_of_fold_predictions(df, model_params)
    ratio = df[TARGET].to_numpy() / oof_pred

    global_default = float(np.clip(np.quantile(ratio, BUFFER_QUANTILE), *BUFFER_CLIP))

    per_hour: dict[str, float] = {}
    hours = df["hour_of_day"].to_numpy()
    for h in np.unique(hours):
        mask = hours == h
        if mask.sum() >= MIN_GROUP_FOR_BUFFER_BUCKET:
            per_hour[str(int(h))] = float(np.clip(np.quantile(ratio[mask], BUFFER_QUANTILE), *BUFFER_CLIP))

    queue = df["queue_length"].to_numpy()
    queue_bin_edges: list[float] = []
    per_queue_bin: list[float | None] = []
    n_unique = len(np.unique(queue))
    n_bins = min(4, n_unique)
    if n_bins >= 2:
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = sorted(set(float(e) for e in np.quantile(queue, quantiles)))
        bin_idx = np.searchsorted(edges, queue, side="right")
        for b in range(len(edges) + 1):
            mask = bin_idx == b
            if mask.sum() >= MIN_GROUP_FOR_BUFFER_BUCKET:
                per_queue_bin.append(float(np.clip(np.quantile(ratio[mask], BUFFER_QUANTILE), *BUFFER_CLIP)))
            else:
                per_queue_bin.append(None)
        queue_bin_edges = edges

    return {
        "per_hour": per_hour,
        "queue_bin_edges": queue_bin_edges,
        "per_queue_bin": per_queue_bin,
        "default": global_default,
        "samples_used": n,
    }


def lookup_buffer(hour_of_day: int, queue_length: int, table: dict | None) -> tuple[float, str]:
    """Look up the calibrated buffer multiplier for this context. Falls back from
    the most specific signal available down to a neutral 1.0 when the scope has no
    calibration data yet at all."""
    if not table or table.get("samples_used", 0) < MIN_GROUP_FOR_BUFFER_BUCKET:
        return 1.0, "no_data_yet"

    h_mult = table.get("per_hour", {}).get(str(hour_of_day))

    q_mult = None
    edges = table.get("queue_bin_edges") or []
    values = table.get("per_queue_bin") or []
    if edges and values:
        idx = int(np.searchsorted(edges, queue_length, side="right"))
        if idx < len(values):
            q_mult = values[idx]

    candidates = [m for m in (h_mult, q_mult) if m is not None]
    if not candidates:
        return table.get("default", 1.0), "global_default"
    if h_mult is not None and q_mult is not None:
        return max(candidates), "hour+queue"
    return candidates[0], ("hour" if h_mult is not None else "queue")


def save_table(path: str, table: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(table, f, indent=2)


def load_table(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
