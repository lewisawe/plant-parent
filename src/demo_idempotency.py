"""Durability / idempotency demo for Plant Parent (targeted check (c)).

The Temporal durability story is: if the worker is killed mid-run and resumes,
the friend must NOT be notified twice for the same day. The `notify` activity
enforces this with an idempotency key f"{plant_id}:{date}" persisted in
data/notified.json.

This script is the scripted stand-in that proves the property WITHOUT needing a
running dev server: it calls `notify` twice for the SAME day with the SAME
triage results and asserts that:
  * the second call sends 0 new notifications (all skipped), and
  * data/notifications.log gained each plant's line exactly once.

When a dev server IS running, the README documents the equivalent manual beat:
start the workflow, kill the worker during `notify`, restart it, and confirm in
the Temporal UI that the retried activity does not double-write — same guarantee,
same idempotency key.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import activities  # noqa: E402

# A fixed day and fixed, TabPFN-independent triage results so the demo runs with
# no token and is fully deterministic.
_DAY = "2026-10-04"
_RESULTS = [
    {"plant_id": "fern-1", "name": "Fern", "species": "fern",
     "next_action": "water_today", "days_until_trouble": -3},
    {"plant_id": "snake-1", "name": "Snake plant", "species": "snake_plant",
     "next_action": "fine", "days_until_trouble": 16},
    {"plant_id": "succulent-1", "name": "Succulent", "species": "succulent",
     "next_action": "move_to_light", "days_until_trouble": 12},
]


def _count_log_lines() -> int:
    try:
        with open(activities._NOTIFICATIONS_LOG, encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except FileNotFoundError:
        return 0


async def main() -> int:
    # Start from a clean slate so the counts are unambiguous.
    for path in (activities._NOTIFIED_JSON, activities._NOTIFICATIONS_LOG):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    first = await activities.notify(_RESULTS, _DAY)
    lines_after_first = _count_log_lines()

    # Simulate "worker killed and resumed" -> notify runs AGAIN for the same day.
    second = await activities.notify(_RESULTS, _DAY)
    lines_after_second = _count_log_lines()

    print(f"first run : sent={len(first['sent'])} skipped={len(first['skipped'])}")
    print(f"second run: sent={len(second['sent'])} skipped={len(second['skipped'])}")
    print(f"log lines after first={lines_after_first}, after second={lines_after_second}")

    ok = (
        len(first["sent"]) == len(_RESULTS)
        and len(first["skipped"]) == 0
        and len(second["sent"]) == 0
        and len(second["skipped"]) == len(_RESULTS)
        and lines_after_first == len(_RESULTS)
        and lines_after_second == len(_RESULTS)  # unchanged -> no double-notify
    )
    if ok:
        print("\nPASS: resume did not double-notify (idempotent).")
        return 0
    print("\nFAIL: idempotency violated.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
