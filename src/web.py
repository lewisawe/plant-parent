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

# Map raw next_action strings -> (emoji, friendly verb). The emoji is a tiny
# aria-hidden flourish only; the tier (urgent/soon/happy) is decided by the
# resolver below, not by this table.
_ACTION_LABELS = {
    "water_today": ("💧", "Water today"),
    "move_to_light": ("☀️", "Move to brighter light"),
    "fine": ("✅", "Happy as is"),
}

# A plant counts as "happy" only when it is fine AND comfortably far from
# trouble. Below this horizon a "fine" plant is still a gentle heads-up.
_HAPPY_MIN = 14

# Horizon (days) over which the thirst gauge maps days-until-trouble -> 0..1.
_GAUGE_HORIZON = 14

# SOON plants are capped so the gauge never reads "full / nothing to do" — it
# should say "topping-up needed soon". This is a PRESENTATION choice only; the
# raw days_until_trouble shown in the text stays truthful.
_SOON_GAUGE_CAP = 0.55


def _days_phrase(action: str, days: int) -> str:
    """Phrase days_until_trouble naturally for a human reader."""
    if action == "water_today" or days <= 0:
        if days < 0:
            n = -days
            return f"overdue by {n} day{'s' if n != 1 else ''}"
        return "water today"
    return f"fine for {days} day{'s' if days != 1 else ''}"


def _tier(action: str, days: int) -> str:
    """Resolve the 3-tier urgency class (display-only; no prediction logic).

    urgent : act now (overdue / water today)
    happy  : fine and comfortably far from trouble
    soon   : everything else, e.g. the succulent's move_to_light heads-up
    """
    if action == "water_today" or days <= 0:
        return "urgent"
    if action == "fine" and days >= _HAPPY_MIN:
        return "happy"
    return "soon"


def _gauge(tier: str, days: int) -> tuple[float, str]:
    """Map days-until-trouble -> (fill fraction 0..1, accessible level word).

    Fraction = clamp(days / horizon). SOON tiers are capped (see cap constant)
    so a heads-up never looks perfectly full. The raw days text is untouched.
    """
    fraction = days / _GAUGE_HORIZON
    fraction = max(0.0, min(1.0, fraction))
    if tier == "soon":
        fraction = min(fraction, _SOON_GAUGE_CAP)
    if fraction <= 0.12:
        level = "empty"
    elif fraction < 0.5:
        level = "low"
    elif fraction < 0.85:
        level = "half full"
    else:
        level = "full"
    return round(fraction, 3), level


def _card_view(result: dict) -> dict:
    """Turn a raw predict() result dict into a display-ready card dict."""
    action = result["next_action"]
    days = int(result["days_until_trouble"])
    emoji, verb = _ACTION_LABELS.get(action, ("🪴", action.replace("_", " ").title()))
    tier = _tier(action, days)
    gauge_fraction, gauge_level = _gauge(tier, days)
    status_word = {
        "urgent": "Needs you now",
        "soon": "Heads-up",
        "happy": "All good",
    }[tier]
    # Species adds info only when it meaningfully differs from the display name;
    # otherwise we drop it (the forgetful friend doesn't need a restatement).
    name = result["name"]
    species_pretty = result["species"].replace("_", " ").title()
    species = species_pretty if species_pretty.lower() != name.lower() else ""
    return {
        "name": name,
        "species": species,
        "action": action,  # raw key -> lets the template pick the mood/CTA
        "action_emoji": emoji,
        "action_label": verb,
        "days_phrase": _days_phrase(action, days),
        "tier": tier,
        "status_word": status_word,
        "actionable": tier != "happy",
        "gauge_fraction": gauge_fraction,  # 0..1 display-only
        "gauge_level": gauge_level,  # accessible word for aria-label
        # sort key: urgent -> soon -> happy, then soonest trouble first
        "_sort": ({"urgent": 0, "soon": 1, "happy": 2}[tier], days),
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
    owner = os.environ.get("PLANT_OWNER", "My friend")
    return render_template("triage.html", cards=cards, badge=badge, owner=owner)


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
