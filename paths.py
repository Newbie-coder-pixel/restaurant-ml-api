"""Single source of truth for on-disk model layout.

Everything under model/global/ is the fallback model; model/branch_{id}/ holds a
branch-specific model once that branch has enough real orders. There is no more
legacy flat model/*.pkl path — train_model.py and /retrain both write here now.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "global", "prep_time_model.pkl")
GLOBAL_FEAT_PATH = os.path.join(MODEL_DIR, "global", "features.pkl")
GLOBAL_BUFFER_TABLE_PATH = os.path.join(MODEL_DIR, "global", "buffer_table.json")

RETRAIN_HISTORY_PATH = os.path.join(MODEL_DIR, "retrain_history.jsonl")


def branch_dir(branch_id: str) -> str:
    return os.path.join(MODEL_DIR, f"branch_{branch_id}")


def branch_model_path(branch_id: str) -> tuple[str, str]:
    d = branch_dir(branch_id)
    return os.path.join(d, "prep_time_model.pkl"), os.path.join(d, "features.pkl")


def branch_buffer_table_path(branch_id: str) -> str:
    return os.path.join(branch_dir(branch_id), "buffer_table.json")
