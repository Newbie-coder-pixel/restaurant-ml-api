"""Append-only audit trail for every retrain attempt (accepted or rejected).

Previously a retrain silently overwrote the model file with no history — if a
retrain looked "barely acceptable", there was no record to look back on. This
just appends one JSON line per attempt; nothing here is ever mutated or deleted.
"""
import json
import os
from datetime import datetime, timezone


def append_retrain_log(path: str, entry: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
