"""Deterministic Temporal workflow for Plant Parent's daily check.

DETERMINISM RULE
----------------
The workflow body does NO direct IO, time, or randomness. It only orchestrates
activities (read_plant_state -> run_prediction -> notify). The one piece of
"time" it needs — today's date for the idempotency key — is taken from
`workflow.now()`, which Temporal records and replays deterministically, and is
passed INTO the notify activity. All real side effects live in activities.py.

"Check every morning"
---------------------
Two standard patterns are shown:
  * `DailyPlantCheck` runs the pipeline once and returns the triage result. This
    is what the dashboard starts and what the end-to-end check runs.
  * `DailyPlantCheckLoop` demonstrates the "sleep 24h and repeat" pattern using
    `workflow.sleep(timedelta(hours=24))` (a continuously-running workflow). In
    production you'd more likely attach `DailyPlantCheck` to a Temporal Schedule;
    both approaches are noted below. The loop is provided to SHOW the pattern and
    is not required to actually run for a day during the demo.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

# Activity references must be imported through the workflow sandbox safely.
# The worker and dashboard put src/ on sys.path and import this as a top-level
# module, so `import activities` is the correct form (a relative import has no
# package context inside the Temporal sandbox).
with workflow.unsafe.imports_passed_through():
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import activities  # type: ignore

TASK_QUEUE = "plant-parent"

# TabPFN auth failures are not transient — do not retry them in a loop.
_NON_RETRYABLE = ["TabPFNAuthRequired"]

_ACTIVITY_OPTS = dict(
    start_to_close_timeout=timedelta(minutes=5),
    retry_policy=RetryPolicy(
        initial_interval=timedelta(seconds=1),
        maximum_attempts=3,
        non_retryable_error_types=_NON_RETRYABLE,
    ),
)


@workflow.defn
class DailyPlantCheck:
    """Run the daily triage pipeline once and return the notification summary."""

    @workflow.run
    async def run(self) -> dict[str, Any]:
        # Deterministic timestamp from the workflow clock (replay-safe).
        day = workflow.now().date().isoformat()

        rows = await workflow.execute_activity(
            activities.read_plant_state, **_ACTIVITY_OPTS
        )
        results = await workflow.execute_activity(
            activities.run_prediction, rows, **_ACTIVITY_OPTS
        )
        summary = await workflow.execute_activity(
            activities.notify, args=[results, day], **_ACTIVITY_OPTS
        )
        return summary


@workflow.defn
class DailyPlantCheckLoop:
    """"Check every morning" pattern: run, then sleep 24h, repeat.

    Shown to illustrate the durable long-running pattern. A production deployment
    would instead register `DailyPlantCheck` on a Temporal Schedule (cron-like),
    which is easier to pause/backfill. This loop is NOT started by the demo.
    """

    @workflow.run
    async def run(self, iterations: int = 0) -> None:
        # iterations=0 means run forever; a positive value bounds it (for tests).
        count = 0
        while True:
            day = workflow.now().date().isoformat()
            rows = await workflow.execute_activity(
                activities.read_plant_state, **_ACTIVITY_OPTS
            )
            results = await workflow.execute_activity(
                activities.run_prediction, rows, **_ACTIVITY_OPTS
            )
            await workflow.execute_activity(
                activities.notify, args=[results, day], **_ACTIVITY_OPTS
            )
            count += 1
            if iterations and count >= iterations:
                return
            # Durable sleep — the workflow survives worker restarts across this.
            await workflow.sleep(timedelta(hours=24))
