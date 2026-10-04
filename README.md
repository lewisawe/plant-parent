# Plant Parent — "Will my plant survive my friend?"

A tiny houseplant-care **triage tool** for Mesh, a lightly fictional friend who
kills every plant. It looks at each plant's state (species, pot size, light,
days since watered, room temp, humidity) and tells Mesh, per plant, the next
action and how long until trouble — as a simple triage dashboard:

```
🔴 Fern — water today | 🟢 Succulent — move to light | 🟢 Snake plant — fine for 16 days
```

Three plants with genuinely different needs drive the demo: a **snake plant**
(dry, infrequent), a **fern** (consistently moist), and a **succulent** (dry).

## What's under the hood

- **TabPFN** (Prior Labs' tabular foundation model) is the prediction core. Each
  plant is one feature row; TabPFN predicts the next action (classifier) and
  days-until-trouble (regressor) in a single forward pass — no gradient training,
  no hyperparameter search. A thin abstraction (`src/predict.py`) runs either the
  **local `tabpfn`** package or the **hosted `tabpfn-client`** behind one
  `predict()` interface.
- **Temporal** (`temporalio`) wraps the daily check as a durable workflow:
  `read_plant_state → run_prediction → notify`, each step an activity. The
  workflow body is deterministic (no IO/time/random — the only clock is
  `workflow.now()`); all side effects live in activities. The `notify` activity
  is **idempotent** per `plant_id:date`, so a killed-and-resumed run never
  double-notifies.
- **Dashboard** is a clean CLI table (no SPA, no web framework by design).

## Honest note on the data (important)

There is **no real logged history**. Mesh hasn't been tracking waterings. So the
TabPFN in-context training table is **SYNTHETIC-FROM-RULES**: `src/data_gen.py`
encodes well-documented houseplant-care rules (succulents and snake plants
tolerate long dry spells; ferns need frequent moisture; brighter light and
smaller pots dry soil faster; higher temp / lower humidity shorten the safe
interval) and samples ~200 labeled rows from them. TabPFN uses those as its
in-context examples. As Mesh logs **real** waterings over time, those real rows
can be appended and the predictions personalize. The synthetic seed is a sensible
prior, not a pretend history — we do not fake real data.

## TabPFN backend decision (and the token you need)

- **Chosen backend: local `tabpfn` (CPU).** It installed cleanly in one attempt
  alongside CPU `torch`. The hosted `tabpfn-client` is also installed as the
  documented single-attempt fallback; both sit behind the same `predict()`.
- **A free Prior Labs token is REQUIRED for genuine TabPFN inference.** Local
  inference must download license-gated model weights (a one-time license
  acceptance), and the hosted client calls the Prior Labs API. Without a token,
  the tool runs in an explicitly-labeled **OFFLINE REFERENCE** mode that shows the
  seed rule labels (the same signal TabPFN learns in-context) — clearly marked as
  *not* live TabPFN output. The moment a token is set, the exact same code path
  produces genuine TabPFN predictions with no fallback.

Get and set a token (one time):

```bash
# 1. Open https://ux.priorlabs.ai, log in / register
# 2. Accept the license on the Licenses tab
# 3. Copy your API key from https://ux.priorlabs.ai/account
export TABPFN_TOKEN="<your-api-key>"
```

> Before capturing submission screenshots, a human MUST set `TABPFN_TOKEN` and do
> a real TabPFN run so the dashboard shows genuine TabPFN output (not the offline
> reference). See the TODO in `PROGRESS.md`.

## How to run

All commands use absolute paths from the app root. Let
`APP=/home/sierra/Desktop/projects/devTo/.worktrees/plant-parent/plantParent`.

### 1. Create the venv and install

Python 3.11 is required (system Python 3.14 has no torch wheel). The venv already
exists in this checkout; to recreate it:

```bash
"$(which python3.12 || which python3.11 || which python3)" -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install --upgrade pip
# torch must come from the CPU wheel index, not PyPI:
"$APP/.venv/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
"$APP/.venv/bin/pip" install -r "$APP/requirements.txt"
```

### 2. (Optional) Start the Temporal dev server

The dashboard works without it (`--local`), but to run the real workflow you need
a Temporal server. Two options:

**Option A — standalone CLI** (gives you the Web UI for the manual kill/resume demo):

```bash
# If the temporal CLI is missing, install it (one line):
curl -sSf https://temporal.download/cli.sh | sh
export PATH="$PATH:$HOME/.temporalio/bin"
# Start the dev server (Web UI at http://localhost:8233):
temporal server start-dev
```

**Option B — bundled server via `temporalio` (no CLI install).** The `temporalio`
package can start the same dev server in-process with
`WorkflowEnvironment.start_local()` (it downloads/starts Temporal automatically).
This is what the final end-to-end verification used, so no separate CLI is
required just to prove the workflow runs. See `PROGRESS.md` → "FINAL end-to-end
verification" for the exact snippet.

### 3. Run the worker (needs the dev server from step 2)

```bash
"$APP/.venv/bin/python" "$APP/src/worker.py"
```

If the server is down it prints a clear "start the Temporal dev server first"
message and exits — it does not hang or loop.

### 4. Open the dashboard

```bash
# Through Temporal (starts a DailyPlantCheck workflow, needs server + worker):
"$APP/.venv/bin/python" "$APP/src/app.py"

# OR the no-infra path (runs the same read→predict→notify in-process):
"$APP/.venv/bin/python" "$APP/src/app.py" --local
```

### 4b. Open the web UI (friendly cards, no Temporal needed)

A minimal but polished Flask page shows the same per-plant triage as visual
cards — nice to hand to a non-technical friend. It renders the SAME data by
calling `PlantPredictor.predict(current_plants())`; it does **not** need the
Temporal server or worker.

```bash
"$APP/.venv/bin/python" "$APP/src/web.py"
# then open http://localhost:5000  (set PORT=xxxx to change the port)
```

The page shows an **honest data-source badge**: "Powered by TabPFN (live)" when
a `TABPFN_TOKEN` is set and real inference ran, or "⚠ Offline reference (set
TABPFN_TOKEN for live TabPFN)" when it fell back to the seed-rule reference. The
badge always matches the actual prediction source — it never implies TabPFN ran
when it did not. A JSON view of the same triage is at `/api/triage`.

### 5. Durability / idempotency demo (kill-and-resume without double-notify)

Scripted (no server needed):

```bash
rm -f "$APP/data/notified.json" "$APP/data/notifications.log"
"$APP/.venv/bin/python" "$APP/src/demo_idempotency.py"   # prints PASS
```

Manual beat with the dev server running: start the workflow, kill the worker
while the `notify` activity runs, restart the worker, and watch in the Temporal
Web UI that the retried activity completes without writing duplicate lines to
`data/notifications.log` — the `plant_id:date` idempotency key guarantees it.

## Project layout

```
src/
  data_gen.py          synthetic-from-rules generator + Mesh's 3 current plants
  predict.py           TabPFN local-or-hosted predictor (one predict() interface)
  activities.py        Temporal activities: read_plant_state, run_prediction, notify
  workflow.py          deterministic DailyPlantCheck (+ 24h-loop pattern)
  worker.py            Temporal worker (task queue "plant-parent")
  app.py               CLI dashboard + workflow starter (--local for no-infra)
  web.py               minimal Flask web UI (friendly cards; reuses predict core)
  demo_idempotency.py  scripted kill/resume-without-double-notify proof
templates/
  triage.html          self-contained web UI page (inline CSS, no build step)
requirements.txt       pinned tabpfn + tabpfn-client + temporalio + flask
```

## Scope (intentionally small)

Smallest real thing that demos: synthetic generator + TabPFN predict abstraction +
Temporal workflow/worker + a minimal CLI dashboard, plus a minimal Flask web UI
(same prediction core, no build step / npm / CDN). No broad unit-test suite, no
SPA. Verification is targeted checks (different-advice, workflow-completes,
idempotency, and a web-UI render check), recorded in `PROGRESS.md`.
