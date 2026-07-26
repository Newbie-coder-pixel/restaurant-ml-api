"""Real-data training pipeline, shared by train_model.py and /retrain.

There is no synthetic data generator anywhere in this project anymore. Every model
(global or per-branch) is trained from real Supabase order history, computed the
same way here as it is at inference time in main.py.
"""
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FEATURES = [
    "total_quantity",
    "total_item_types",
    "weighted_prep_time",
    "special_request_count",
    "hour_of_day",
    "queue_length",
]
TARGET = "actual_prep_minutes"

# Minimum real orders required before a scope (branch or global) can be trained AND
# evaluated with a statistically meaningful chronological holdout (see main.py
# chronological_split). Below this, /retrain refuses rather than reporting an MAE
# computed from a handful of rows.
MIN_SAMPLES_FOR_MODEL = 40

# Minimum real branch orders required before a branch's equipment-speed factor can be
# estimated from data at all. Lower bar than a full model since it's a single ratio
# statistic, not a fitted model — but still enough to not be noise-dominated.
MIN_SAMPLES_FOR_EQUIPMENT_FACTOR = 10


def get_supabase():
    """Lazy import so the module can be imported (e.g. for tests) without the
    supabase package needing a live connection."""
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY belum diset di environment.")
    return create_client(url, key)


def build_training_df(supabase, branch_id: str | None = None) -> pd.DataFrame:
    """Query order_items with completed timestamps and aggregate into the exact
    feature set used at inference. branch_id=None pulls all branches (global scope).

    Returns an empty DataFrame if there isn't any usable real data yet — callers
    must treat that as "not enough data", never fall back to fabricated rows.
    """
    query = (
        supabase.from_("order_items")
        .select(
            "order_id, quantity, special_requests, "
            "sent_to_kitchen_at, prepared_at, "
            "menu_items(preparation_time_minutes), "
            "orders(created_at, branch_id)"
        )
        .not_.is_("sent_to_kitchen_at", "null")
        .not_.is_("prepared_at", "null")
    )
    if branch_id:
        query = query.eq("orders.branch_id", branch_id)

    res = query.execute()
    rows = res.data
    if not rows:
        return pd.DataFrame()

    df_raw = pd.DataFrame(rows)

    df_raw["preparation_time_minutes"] = df_raw["menu_items"].apply(
        lambda x: x.get("preparation_time_minutes", 15) if isinstance(x, dict) else 15
    )
    df_raw["order_created_at"] = df_raw["orders"].apply(
        lambda x: x.get("created_at") if isinstance(x, dict) else None
    )
    df_raw = df_raw.drop(columns=["menu_items", "orders"])

    df_raw["sent_to_kitchen_at"] = pd.to_datetime(df_raw["sent_to_kitchen_at"], utc=True)
    df_raw["prepared_at"] = pd.to_datetime(df_raw["prepared_at"], utc=True)
    df_raw["order_created_at"] = pd.to_datetime(df_raw["order_created_at"], utc=True)

    df_raw["actual_prep_minutes"] = (
        (df_raw["prepared_at"] - df_raw["sent_to_kitchen_at"]).dt.total_seconds() / 60
    ).round(1)

    df_raw = df_raw[(df_raw["actual_prep_minutes"] >= 1) & (df_raw["actual_prep_minutes"] <= 120)]
    if df_raw.empty:
        return pd.DataFrame()

    def agg_order(grp):
        created = grp["order_created_at"].iloc[0]
        # Fall back to the real sent_to_kitchen_at timestamp (still real data) if
        # orders.created_at is missing — never a made-up hour.
        hour_source = created if created is not pd.NaT else grp["sent_to_kitchen_at"].iloc[0]
        return pd.Series({
            "total_quantity": grp["quantity"].sum(),
            "total_item_types": len(grp),
            "weighted_prep_time": (grp["preparation_time_minutes"] * grp["quantity"]).sum(),
            "special_request_count": grp["special_requests"].apply(
                lambda x: 1 if isinstance(x, str) and x.strip() != "" else 0
            ).sum(),
            "hour_of_day": hour_source.hour,
            "order_time": grp["sent_to_kitchen_at"].min(),
            "order_end": grp["prepared_at"].max(),
            "actual_prep_minutes": (
                grp["prepared_at"].max() - grp["sent_to_kitchen_at"].min()
            ).total_seconds() / 60,
        })

    df_orders = df_raw.groupby("order_id").apply(agg_order).reset_index(drop=True)

    # queue_length = how many other real orders were still being prepared at the
    # exact moment this order started. Computed from actual overlapping timestamps,
    # never hardcoded — mirrors the "orders currently preparing" definition used
    # live in /predict.
    starts = df_orders["order_time"].to_numpy()
    ends = df_orders["order_end"].to_numpy()
    still_preparing = (starts[None, :] <= starts[:, None]) & (ends[None, :] > starts[:, None])
    queue_lengths = still_preparing.sum(axis=1) - 1
    df_orders["queue_length"] = np.clip(queue_lengths, 0, None)
    df_orders = df_orders.drop(columns=["order_end"])

    df_orders = df_orders[
        (df_orders["actual_prep_minutes"] >= 1) & (df_orders["actual_prep_minutes"] <= 120)
    ]

    return df_orders.reset_index(drop=True)
