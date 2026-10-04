"""Synthetic plant-care data generator for Plant Parent.

HONEST SEED NOTE
----------------
Every row this module produces is SYNTHETIC-FROM-RULES. There is NO real logged
history here. Mesh (our lightly fictional friend who kills every plant) has not
been tracking waterings, so we cannot and do not fake a real history.

Instead we encode well-documented houseplant-care rules in `label_row()` and
sample feature combinations around them. TabPFN then uses these labeled rows as
*in-context training examples* (one forward pass, no gradient training) to score
Mesh's actual current plants. As Mesh logs real waterings over time, those real
rows can be appended and will personalize the predictions — the synthetic seed is
a sensible prior, not a pretend dataset.

Documented care rules used (common houseplant guidance):
  * Snake plant & succulents tolerate long dry spells (water sparingly; ~14-21
    day baseline interval). Overwatering, not underwatering, is their usual killer.
  * Ferns need consistently moist soil and sulk/brown fast when dry (~3-5 day
    baseline interval).
  * More light and a smaller pot dry the soil out faster -> shorter safe interval.
  * Higher room temperature and lower humidity also shorten the safe interval.

Shared schema (single source of truth for generator + predictor)
----------------------------------------------------------------
Features (one row per plant):
  species            : "snake_plant" | "fern" | "succulent"
  pot_size_cm        : int  (diameter, ~8-30)
  light              : "low" | "medium" | "high" (from window direction)
  days_since_watered : int
  room_temp_c        : int
  humidity_pct       : int
Labels (what TabPFN learns to predict):
  next_action        : "water_today" | "fine" | "move_to_light"
  days_until_trouble : int  (days until the plant is likely in trouble)
"""

from __future__ import annotations

import random
from typing import Any

# ---- Shared schema constants (imported by predict.py so encoding matches) ----

SPECIES = ("snake_plant", "fern", "succulent")
LIGHT_LEVELS = ("low", "medium", "high")
NEXT_ACTIONS = ("water_today", "fine", "move_to_light")

FEATURE_KEYS = (
    "species",
    "pot_size_cm",
    "light",
    "days_since_watered",
    "room_temp_c",
    "humidity_pct",
)
LABEL_KEYS = ("next_action", "days_until_trouble")

# Baseline "safe" days between waterings per species under neutral conditions.
_BASE_INTERVAL = {
    "snake_plant": 18,  # tolerates long dry spells
    "succulent": 16,  # also drought-tolerant
    "fern": 4,  # needs consistent moisture
}


def _safe_interval(features: dict[str, Any]) -> float:
    """Compute the species-and-condition-adjusted safe watering interval (days).

    Starts from the species baseline, then shortens for drying conditions
    (bright light, small pot, warm room, low humidity) per the documented rules.
    """
    interval = float(_BASE_INTERVAL[features["species"]])

    # Light: brighter dries soil faster.
    light_factor = {"low": 1.20, "medium": 1.0, "high": 0.80}[features["light"]]
    interval *= light_factor

    # Pot size: small pots hold less water -> dry faster. 15cm is the neutral point.
    pot = features["pot_size_cm"]
    interval *= 0.6 + (pot / 15.0) * 0.4  # ~0.8x at 8cm, ~1.0x at 15cm, ~1.3x at 30cm

    # Temperature: warmer -> faster drying. 22C neutral.
    interval *= 1.0 - (features["room_temp_c"] - 22) * 0.03

    # Humidity: drier air -> faster drying. 50% neutral.
    interval *= 1.0 + (features["humidity_pct"] - 50) * 0.004

    return max(1.0, interval)


def label_row(features: dict[str, Any]) -> tuple[str, int]:
    """Apply documented care rules to a feature row -> (next_action, days_until_trouble).

    This is the honest rule core. TabPFN learns THIS mapping in-context from the
    seeded rows; it is not a separate predictor used at serve time.
    """
    interval = _safe_interval(features)
    days_since = features["days_since_watered"]

    # days_until_trouble: how long until the plant is likely stressed.
    days_until_trouble = int(round(interval - days_since))

    # Ferns in low light that are also dry: the real fix is often light + water,
    # but Mesh's recurring mistake is forgetting to water, so prioritise water.
    if days_until_trouble <= 0:
        next_action = "water_today"
    elif (
        features["light"] == "low"
        and features["species"] in ("succulent", "snake_plant")
        and days_until_trouble >= interval * 0.5
    ):
        # Drought-tolerant species that are fine on water but starved of light:
        # the better next action is to move them somewhere brighter.
        next_action = "move_to_light"
    else:
        next_action = "fine"

    # Clamp days_until_trouble to a sane display range.
    days_until_trouble = max(-5, min(days_until_trouble, 60))
    return next_action, days_until_trouble


def generate(n: int = 200, seed: int = 0) -> list[dict[str, Any]]:
    """Generate ~100-300 labeled synthetic rows across the shared schema.

    Rows are sampled across species and plausible conditions, then labeled by
    `label_row()`. Returns a list of dicts each containing all FEATURE_KEYS and
    all LABEL_KEYS. These serve as TabPFN's in-context training examples.
    """
    if not (50 <= n <= 400):
        # Keep the table tiny — TabPFN is designed for small tables and we only
        # need enough rows to cover the rule surface.
        n = max(100, min(n, 300))

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for _ in range(n):
        features = {
            "species": rng.choice(SPECIES),
            "pot_size_cm": rng.randint(8, 30),
            "light": rng.choice(LIGHT_LEVELS),
            "days_since_watered": rng.randint(0, 30),
            "room_temp_c": rng.randint(16, 30),
            "humidity_pct": rng.randint(20, 80),
        }
        action, days = label_row(features)
        row = dict(features)
        row["next_action"] = action
        row["days_until_trouble"] = days
        rows.append(row)
    return rows


def current_plants() -> list[dict[str, Any]]:
    """Return Mesh's three live, UNLABELED plants to be scored by TabPFN.

    Chosen so the three species diverge in rule-consistent ways:
      * fern        : overdue and in a small warm pot -> should be most urgent
                      (shortest days_until_trouble / water_today).
      * snake_plant : watered recently, drought-tolerant -> fine for a while.
      * succulent   : drought-tolerant but stuck in LOW light -> move_to_light.
    These carry a `plant_id` and the common name for display; they have NO labels.
    """
    return [
        {
            "plant_id": "fern-1",
            "name": "Fern",
            "species": "fern",
            "pot_size_cm": 12,
            "light": "medium",
            "days_since_watered": 6,
            "room_temp_c": 25,
            "humidity_pct": 40,
        },
        {
            "plant_id": "snake-1",
            "name": "Snake plant",
            "species": "snake_plant",
            "pot_size_cm": 20,
            "light": "medium",
            "days_since_watered": 5,
            "room_temp_c": 21,
            "humidity_pct": 55,
        },
        {
            "plant_id": "succulent-1",
            "name": "Succulent",
            "species": "succulent",
            "pot_size_cm": 10,
            "light": "low",
            "days_since_watered": 4,
            "room_temp_c": 22,
            "humidity_pct": 45,
        },
    ]


if __name__ == "__main__":
    rows = generate()
    print(f"generated {len(rows)} synthetic-from-rules rows")
    print("sample row:", rows[0])
    print("\nMesh's current plants (unlabeled, to be scored by TabPFN):")
    for p in current_plants():
        # Show the rule-based reference label so the divergence is visible even
        # before TabPFN runs (TabPFN reproduces this mapping in-context).
        feats = {k: p[k] for k in FEATURE_KEYS}
        action, days = label_row(feats)
        print(f"  {p['name']:12s} -> rule says: {action:14s} days_until_trouble={days}")
