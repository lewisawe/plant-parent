"""Plant Parent CLI dashboard + workflow starter.

Two modes:
  * default : connect to Temporal, start a DailyPlantCheck workflow, wait for the
              result, and print the triage table. Requires a running dev server
              and a running worker.
  * --local : run the SAME read -> predict -> notify pipeline in-process by
              calling the activity functions directly. Demoable with zero infra
              (no Temporal server, no worker). Still needs a TabPFN token to get
              real predictions; without one it prints clear guidance.

Triage table is sorted by urgency and uses emoji:
  🔴 water_today / overdue   🟡 due soon   🟢 fine for a while
Example line: "🔴 Fern — water today | 🟢 Snake plant — fine for 9 days"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import activities  # noqa: E402
import workflow  # noqa: E402
from predict import TabPFNAuthRequired  # noqa: E402

TEMPORAL_TARGET = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")


def _urgency(result: dict) -> tuple[int, str]:
    """Return (sort_key, emoji). Lower sort_key = more urgent."""
    action = result["next_action"]
    days = result["days_until_trouble"]
    if action == "water_today" or days <= 0:
        return (0, "🔴")
    if days <= 3:
        return (1, "🟡")
    return (2, "🟢")


def _phrase(result: dict) -> str:
    action = result["next_action"]
    days = result["days_until_trouble"]
    if action == "water_today" or days <= 0:
        return "water today"
    if action == "move_to_light":
        return "move to light"
    if days <= 3:
        return f"water in {days} day{'s' if days != 1 else ''}"
    return f"fine for {days} days"


def _offline_banner(results: list[dict]) -> str | None:
    """Return the honest offline-reference banner if results are not TabPFN output."""
    if results and all(r.get("source") == "offline_reference" for r in results):
        return (
            "⚠ OFFLINE REFERENCE (no TABPFN_TOKEN) — showing seed rule labels, "
            "not live TabPFN output.\n"
            "  Set TABPFN_TOKEN for real predictions "
            "(see README / https://ux.priorlabs.ai)."
        )
    return None


def render_table(results: list[dict]) -> str:
    ordered = sorted(results, key=lambda r: (_urgency(r)[0], r["days_until_trouble"]))
    lines = []
    for r in ordered:
        _, emoji = _urgency(r)
        lines.append(f"{emoji} {r['name']} — {_phrase(r)}")
    header = "🪴  Plant Parent — triage for Mesh's plants"
    one_liner = " | ".join(lines)
    out = f"{header}\n{'-' * len(header)}\n" + "\n".join(lines) + f"\n\n{one_liner}"
    banner = _offline_banner(results)
    if banner:
        out = f"{banner}\n\n{out}"
    return out


async def run_local() -> int:
    """Run the pipeline in-process without Temporal (no-infra demo path)."""
    day = date.today().isoformat()
    rows = await activities.read_plant_state()
    try:
        results = await activities.run_prediction(rows)
    except TabPFNAuthRequired as exc:
        # Only reached if NEITHER tabpfn nor tabpfn_client is importable.
        print("\n[No TabPFN backend installed]\n")
        print(str(exc))
        return 0
    summary = await activities.notify(results, day)
    print(render_table(summary["results"]))
    print(
        f"\nnotifications: {len(summary['sent'])} sent, "
        f"{len(summary['skipped'])} skipped (idempotent) for {summary['day']}"
    )
    return 0


async def run_temporal() -> int:
    """Start a DailyPlantCheck workflow on the Temporal server and show results."""
    from temporalio.client import Client

    try:
        client = await Client.connect(TEMPORAL_TARGET)
    except Exception as exc:  # noqa: BLE001
        print(
            f"\nCould not connect to Temporal at {TEMPORAL_TARGET}.\n"
            "Start the dev server and worker, or run with --local for the "
            "no-infra path:\n  .venv/bin/python src/app.py --local\n"
            f"\nUnderlying error: {exc}\n",
            file=sys.stderr,
        )
        return 1

    try:
        summary = await client.execute_workflow(
            workflow.DailyPlantCheck.run,
            id=f"plant-parent-{date.today().isoformat()}",
            task_queue=workflow.TASK_QUEUE,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"\nWorkflow execution failed: {exc}\n"
            "Is the worker running?  .venv/bin/python src/worker.py\n",
            file=sys.stderr,
        )
        return 1

    print(render_table(summary["results"]))
    print(
        f"\nnotifications: {len(summary['sent'])} sent, "
        f"{len(summary['skipped'])} skipped (idempotent) for {summary['day']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Plant Parent triage dashboard")
    parser.add_argument(
        "--local",
        action="store_true",
        help="run the pipeline in-process without a Temporal server/worker",
    )
    args = parser.parse_args()
    if args.local:
        return asyncio.run(run_local())
    return asyncio.run(run_temporal())


if __name__ == "__main__":
    sys.exit(main())
