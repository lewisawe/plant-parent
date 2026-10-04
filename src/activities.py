"""Temporal activities for Plant Parent.

Activities are where ALL side effects live: reading plant state, calling TabPFN,
and writing notifications. The workflow (workflow.py) stays deterministic by only
orchestrating these. Each activity is plain-importable and callable outside a
worker (the dashboard's --local mode and the idempotency demo call them directly).

Idempotency
-----------
`notify` is safe to run twice for the same calendar day. It keys each triage
notification by f"{plant_id}:{date}" in data/notified.json and appends to
data/notifications.log only for keys it has not seen. This is what powers the
"kill mid-run, resume without double-notifying" durability demo.
"""

from __future__ import annotations

import json
import os
from datetime import date as _date
from typing import Any

from temporalio import activity

# Import siblings whether or not src/ is a package on sys.path.
try:  # pragma: no cover - import plumbing
    from . import data_gen  # type: ignore
    from .predict import PlantPredictor, TabPFNAuthRequired  # type: ignore
except ImportError:  # pragma: no cover - import plumbing
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import data_gen  # type: ignore
    from predict import PlantPredictor, TabPFNAuthRequired  # type: ignore

# Runtime state dir (gitignored). Resolved relative to the app root (src/..).
_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_APP_ROOT, "data")
_NOTIFIED_JSON = os.path.join(_DATA_DIR, "notified.json")
_NOTIFICATIONS_LOG = os.path.join(_DATA_DIR, "notifications.log")


def _ensure_data_dir() -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)


def _load_notified() -> dict[str, Any]:
    try:
        with open(_NOTIFIED_JSON, encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_notified(state: dict[str, Any]) -> None:
    _ensure_data_dir()
    tmp = _NOTIFIED_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
    os.replace(tmp, _NOTIFIED_JSON)


@activity.defn
async def read_plant_state() -> list[dict[str, Any]]:
    """Read the current (unlabeled) state of Mesh's plants.

    In production this would read a store the friend logs into; here it returns
    the three current plants from the generator. Side-effecting (IO) -> activity.
    """
    return data_gen.current_plants()


@activity.defn
async def run_prediction(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the TabPFN predictor on the given plant rows.

    Returns per-plant triage dicts. If TabPFN has no token available, raises
    TabPFNAuthRequired; the workflow's RetryPolicy treats this as non-retryable
    (see workflow.py) so it surfaces immediately instead of looping.
    """
    predictor = PlantPredictor()
    return predictor.predict(rows)


@activity.defn
async def notify(results: list[dict[str, Any]], day: str | None = None) -> dict[str, Any]:
    """Write triage notifications idempotently for a given day.

    `day` is an ISO date string passed in by the workflow (so time stays out of
    the deterministic workflow body). For each result, the key f"{plant_id}:{day}"
    is checked against data/notified.json; only unseen keys are appended to
    data/notifications.log and recorded. Returns a summary with counts so the
    caller can see how many were newly sent vs skipped (idempotent).
    """
    day = day or _date.today().isoformat()
    _ensure_data_dir()
    notified = _load_notified()

    sent, skipped = [], []
    for r in results:
        key = f"{r['plant_id']}:{day}"
        line = _format_notification(r, day)
        if key in notified:
            skipped.append(key)
            continue
        with open(_NOTIFICATIONS_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        notified[key] = {"line": line, "day": day}
        sent.append(key)

    _save_notified(notified)
    return {"day": day, "sent": sent, "skipped": skipped, "results": results}


def _format_notification(r: dict[str, Any], day: str) -> str:
    return (
        f"[{day}] {r['name']}: {r['next_action']} "
        f"(days_until_trouble={r['days_until_trouble']})"
    )
