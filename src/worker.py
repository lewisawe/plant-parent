"""Temporal worker for Plant Parent.

Connects to a local Temporal dev server at localhost:7233, registers the
DailyPlantCheck workflow and the three activities on the `plant-parent` task
queue, and runs until interrupted.

If the dev server is not running, it prints a clear instruction and exits
non-zero — it does NOT retry in a loop or hang silently.
"""

from __future__ import annotations

import asyncio
import os
import sys

from temporalio.client import Client
from temporalio.worker import Worker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import activities  # noqa: E402
import workflow  # noqa: E402

TEMPORAL_TARGET = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")


async def main() -> int:
    try:
        client = await Client.connect(TEMPORAL_TARGET)
    except Exception as exc:  # noqa: BLE001 - connection failure is the point
        print(
            f"\nCould not connect to the Temporal server at {TEMPORAL_TARGET}.\n"
            "Start the local dev server first, e.g.:\n"
            "  temporal server start-dev\n"
            "(install the CLI with: curl -sSf https://temporal.download/cli.sh | sh)\n"
            "Or use the no-infra path instead:\n"
            "  .venv/bin/python src/app.py --local\n"
            f"\nUnderlying error: {exc}\n",
            file=sys.stderr,
        )
        return 1

    worker = Worker(
        client,
        task_queue=workflow.TASK_QUEUE,
        workflows=[workflow.DailyPlantCheck, workflow.DailyPlantCheckLoop],
        activities=[
            activities.read_plant_state,
            activities.run_prediction,
            activities.notify,
        ],
    )
    print(
        f"Plant Parent worker connected to {TEMPORAL_TARGET}, "
        f"task queue '{workflow.TASK_QUEUE}'. Waiting for workflows... (Ctrl-C to stop)"
    )
    await worker.run()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\nWorker stopped.")
        sys.exit(0)
