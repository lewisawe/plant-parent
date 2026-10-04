"""Plant Parent web UI — a friendly, screenshot-worthy plant triage page.

This is a thin VIEW on top of the existing prediction core. It does NOT
re-implement any triage logic: it imports `PlantPredictor` from `predict.py`
and `current_plants()` from `data_gen.py`, calls
`PlantPredictor().predict(current_plants())`, and renders the returned dicts as
plant cards. No Temporal server is required (same no-infra path as
`app.run_local()`, but without even touching the activities/workflow layer).

Honesty note
------------
The data-source badge is driven directly by the real `source` value on each
result dict ("tabpfn" vs "offline_reference"). We never imply TabPFN ran when it
did not. The source is whatever `predict.py` actually produced given the token
situation (it auto-loads TABPFN_TOKEN from plantParent/.env via predict.py).

Run it:
    .venv/bin/python src/web.py
    # then open http://localhost:5000
"""

from __future__ import annotations

import os
import sys

from flask import Flask, jsonify, render_template

# predict.py / data_gen.py are siblings in src/. Make them importable whether
# this file is launched as `python src/web.py` or imported as a module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_gen  # noqa: E402
from predict import PlantPredictor  # noqa: E402

# Templates live in plantParent/templates (one level up from src/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES = os.path.join(os.path.dirname(_HERE), "templates")

app = Flask(__name__, template_folder=_TEMPLATES)


# --- Friendly presentation helpers (display only; no prediction logic here) ---

# Map raw next_action strings -> (emoji, friendly verb). Urgency (red vs green)
# is decided by _card_view below, not by this table.
_ACTION_LABELS = {
    "water_today": ("💧", "Water today"),
    "move_to_light": ("☀️", "Move to brighter light"),
    "fine": ("✅", "Happy as is"),
}

# Actions that mean "this plant needs you now" -> red/urgent treatment.
_URGENT_ACTIONS = {"water_today", "move_to_light"}


def _days_phrase(action: str, days: int) -> str:
    """Phrase days_until_trouble naturally for a human reader."""
    if action == "water_today" or days <= 0:
        if days < 0:
            n = -days
            return f"overdue by {n} day{'s' if n != 1 else ''}"
        return "water today"
    return f"fine for {days} day{'s' if days != 1 else ''}"


def _card_view(result: dict) -> dict:
    """Turn a raw predict() result dict into a display-ready card dict."""
    action = result["next_action"]
    days = int(result["days_until_trouble"])
    emoji, verb = _ACTION_LABELS.get(action, ("🪴", action.replace("_", " ").title()))
    urgent = action in _URGENT_ACTIONS or days <= 0
    return {
        "name": result["name"],
        "species": result["species"].replace("_", " ").title(),
        "action_emoji": emoji,
        "action_label": verb,
        "days_phrase": _days_phrase(action, days),
        "status_dot": "🔴" if urgent else "🟢",
        "status_word": "Needs you" if urgent else "All good",
        "urgent": urgent,
        # sort key: most urgent (lowest days) first
        "_sort": (0 if urgent else 1, days),
    }


def _source_badge(results: list[dict]) -> dict:
    """Honest data-source badge derived from the actual result source values."""
    live = bool(results) and all(r.get("source") == "tabpfn" for r in results)
    if live:
        return {
            "live": True,
            "text": "Powered by TabPFN (live)",
            "icon": "🟢",
        }
    return {
        "live": False,
        "text": "⚠ Offline reference (set TABPFN_TOKEN for live TabPFN)",
        "icon": "⚠",
    }


def _triage() -> tuple[list[dict], dict]:
    """Run the real predictor and return (cards_sorted, source_badge)."""
    predictor = PlantPredictor()
    results = predictor.predict(data_gen.current_plants())
    cards = sorted((_card_view(r) for r in results), key=lambda c: c["_sort"])
    return cards, _source_badge(results)


@app.route("/")
def index():
    cards, badge = _triage()
    return render_template("triage.html", cards=cards, badge=badge)


@app.route("/api/triage")
def api_triage():
    """JSON view of the same triage (handy for scripts / debugging)."""
    predictor = PlantPredictor()
    results = predictor.predict(data_gen.current_plants())
    return jsonify(
        {
            "source": results[0]["source"] if results else None,
            "plants": results,
        }
    )


def main() -> int:
    port = int(os.environ.get("PORT", "5000"))
    # debug=False so this never hot-reloads/blocks oddly in a demo setting.
    app.run(host="127.0.0.1", port=port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
