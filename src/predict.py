"""TabPFN prediction core for Plant Parent (local-or-hosted behind one interface).

PREDICTION CORE = TabPFN
------------------------
This module is the ONE place that turns plant state into advice, and it does so
with TabPFN (Prior Labs' tabular foundation model), not with a hand-rolled rule
engine. The documented care rules live in `data_gen.label_row()` purely to SEED
in-context training examples; at serve time TabPFN itself maps a plant's feature
row to (next_action, days_until_trouble) in a single forward pass — no gradient
training, no hyperparameter search.

Backend selection (thin abstraction, one `predict()` interface)
---------------------------------------------------------------
  1. LOCAL  `tabpfn`         — primary. CPU inference. Needs a one-time license
                               token (TABPFN_TOKEN) to download model weights.
  2. HOSTED `tabpfn_client`  — single-attempt fallback if local is unavailable.
                               Same Prior Labs account/token.

Both expose `TabPFNClassifier` / `TabPFNRegressor` with the same sklearn-style
`fit`/`predict` API, so the body below is backend-agnostic. The active backend is
chosen at construction time and recorded on the instance (`.backend`).

AUTH BLOCKER + OFFLINE REFERENCE (honest)
-----------------------------------------
Running TabPFN — local OR hosted — requires a Prior Labs account and a token:
local inference must download gated model weights (one-time license acceptance),
and the hosted client calls their API. In an environment with no TABPFN_TOKEN set
and no cached token, we CANNOT run genuine TabPFN inference.

To keep the tool demoable end-to-end without faking TabPFN output, `predict()`
has two clearly separated paths:

  * TOKEN PRESENT  -> genuine TabPFN inference. `backend_mode == "tabpfn"`.
                      Each result carries `source="tabpfn"`.
  * NO TOKEN       -> OFFLINE REFERENCE. Returns the seed rule labels from
                      `data_gen.label_row()` (the SAME signal TabPFN learns
                      in-context), each result flagged `source="offline_reference"`
                      and `backend_mode == "offline_reference"`. This is NOT
                      TabPFN output and callers MUST label it as such to the user.

The offline path exists ONLY as a no-token reference; it never masquerades as
TabPFN. The instant `TABPFN_TOKEN` is exported, `predict()` uses real TabPFN with
no fallback. A free token is obtainable at https://ux.priorlabs.ai (see README).
"""

from __future__ import annotations

import os
import sys
from typing import Any

try:  # numpy ships with both tabpfn backends; needed for the hosted client's API
    import numpy as _np
except Exception:  # noqa: BLE001 - degrade gracefully; lists still work on local
    _np = None


def _load_env_token() -> None:
    """Populate TABPFN_TOKEN from a local .env file if it is not already set.

    Standard-library only (no python-dotenv dependency). Looks for a .env in the
    project root (parent of src/) and the current working directory. Accepts the
    canonical key `TABPFN_TOKEN` as well as a few friendly aliases (`TabPFN`,
    `TABPFN`, `TABPFN_API_KEY`) so a key dropped in .env under a near-miss name
    still works. A real shell env var always wins over the file.
    """
    if os.environ.get("TABPFN_TOKEN", "").strip():
        return
    aliases = ("TABPFN_TOKEN", "TABPFN_API_KEY", "TABPFN", "TabPFN")
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.path.dirname(here), ".env"),  # plantParent/.env
        os.path.join(os.getcwd(), ".env"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key in aliases and val:
                        os.environ["TABPFN_TOKEN"] = val
                        return
        except OSError:
            continue


_load_env_token()

# data_gen is a sibling module; support both `import data_gen` (when src/ is on
# sys.path) and `from src import ...` style imports.
try:  # pragma: no cover - import plumbing
    from . import data_gen  # type: ignore
except ImportError:  # pragma: no cover - import plumbing
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import data_gen  # type: ignore

FEATURE_KEYS = data_gen.FEATURE_KEYS
NEXT_ACTIONS = list(data_gen.NEXT_ACTIONS)

# Stable integer encodings for categorical features (must match between the
# seeded training rows and the current plants scored at predict time).
_SPECIES_CODE = {s: i for i, s in enumerate(data_gen.SPECIES)}
_LIGHT_CODE = {l: i for i, l in enumerate(data_gen.LIGHT_LEVELS)}
_ACTION_CODE = {a: i for i, a in enumerate(NEXT_ACTIONS)}
_ACTION_DECODE = {i: a for a, i in _ACTION_CODE.items()}


class TabPFNAuthRequired(RuntimeError):
    """Raised when no usable TabPFN backend auth (token) is available."""


_AUTH_HELP = (
    "TabPFN needs a one-time Prior Labs token to run (local weights download or\n"
    "hosted API). No TABPFN_TOKEN was found in the environment.\n\n"
    "To enable real predictions:\n"
    "  1. Open https://ux.priorlabs.ai in a browser and log in (or register)\n"
    "  2. Accept the license on the Licenses tab\n"
    "  3. Copy your API key from https://ux.priorlabs.ai/account\n"
    '  4. export TABPFN_TOKEN="<your-api-key>"  then re-run.\n'
)


def _encode_features(row: dict[str, Any]) -> list[float]:
    """Encode one feature row into a numeric vector in a fixed column order."""
    return [
        float(_SPECIES_CODE[row["species"]]),
        float(row["pot_size_cm"]),
        float(_LIGHT_CODE[row["light"]]),
        float(row["days_since_watered"]),
        float(row["room_temp_c"]),
        float(row["humidity_pct"]),
    ]


def _detect_backend() -> tuple[str, Any, Any]:
    """Return (backend_name, TabPFNClassifier, TabPFNRegressor).

    Backend choice:
      * TABPFN_BACKEND=hosted|local forces a backend explicitly.
      * Otherwise: if a TABPFN_TOKEN is set, prefer the HOSTED client (the token
        is a hosted `tabpfn_sk_` key; local inference instead needs an
        interactive browser license acceptance to download weights, which fails
        in a non-interactive terminal). With no token, prefer local.
    Importing does NOT require a token — only fitting/predicting does.
    """
    forced = os.environ.get("TABPFN_BACKEND", "").strip().lower()

    def _try_local():
        from tabpfn import TabPFNClassifier, TabPFNRegressor  # type: ignore
        return "tabpfn (local)", TabPFNClassifier, TabPFNRegressor

    def _try_hosted():
        import tabpfn_client  # type: ignore
        from tabpfn_client import TabPFNClassifier, TabPFNRegressor  # type: ignore
        # Authenticate the hosted client from the token so no interactive login
        # is attempted (which would fail in a non-interactive terminal).
        tok = os.environ.get("TABPFN_TOKEN", "").strip()
        if tok:
            try:
                tabpfn_client.set_access_token(tok)
            except Exception:  # noqa: BLE001 - older/newer client surfaces vary
                pass
        return "tabpfn_client (hosted)", TabPFNClassifier, TabPFNRegressor

    # Build the preference order.
    if forced == "local":
        order = [_try_local]
    elif forced == "hosted":
        order = [_try_hosted]
    elif os.environ.get("TABPFN_TOKEN", "").strip():
        order = [_try_hosted, _try_local]  # token present -> hosted first
    else:
        order = [_try_local, _try_hosted]  # no token -> local first

    last_exc = None
    for fn in order:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - try the next backend
            last_exc = exc
    raise TabPFNAuthRequired(
        "No usable TabPFN backend. Install one:\n"
        "  pip install tabpfn   (local, CPU)\n"
        "  pip install tabpfn-client   (hosted)\n"
        f"Underlying error: {last_exc}"
    ) from last_exc


def _has_token() -> bool:
    """True if a TabPFN token is plausibly available (env var or cached)."""
    if os.environ.get("TABPFN_TOKEN", "").strip():
        return True
    # Local tabpfn caches a token under ~/.cache/tabpfn after first acceptance.
    cache = os.path.expanduser("~/.cache/tabpfn")
    try:
        for fn in os.listdir(cache):
            if "token" in fn.lower():
                return True
    except OSError:
        pass
    return False


class PlantPredictor:
    """TabPFN-backed predictor with a single `predict()` method.

    When a TabPFN token is available it fits a TabPFNClassifier (next_action) +
    TabPFNRegressor (days_until_trouble) once on the seeded in-context table and
    scores the current plants with genuine TabPFN inference. When no token is
    available it returns an explicitly-flagged OFFLINE REFERENCE built from the
    seed rule mapping (never passed off as TabPFN output).
    """

    def __init__(self, seed_rows: list[dict[str, Any]] | None = None, device: str = "cpu"):
        self.device = device
        self.backend, self._Clf, self._Reg = _detect_backend()
        self._seed_rows = seed_rows if seed_rows is not None else data_gen.generate()
        self._clf = None
        self._reg = None
        # "tabpfn" once genuine inference is in use; "offline_reference" otherwise.
        # Resolved on first predict() based on token availability.
        self.backend_mode = "tabpfn" if _has_token() else "offline_reference"

    @property
    def using_tabpfn(self) -> bool:
        """True when genuine TabPFN inference will run (a token is available)."""
        return _has_token()

    # -- internal: build and fit the two TabPFN estimators once ---------------
    def _fit(self) -> None:
        if self._clf is not None and self._reg is not None:
            return

        X = [_encode_features(r) for r in self._seed_rows]
        y_action = [_ACTION_CODE[r["next_action"]] for r in self._seed_rows]
        y_days = [float(r["days_until_trouble"]) for r in self._seed_rows]

        # TabPFNClassifier/Regressor accept device="cpu" on the local backend;
        # the hosted client ignores unknown kwargs, so guard with a fallback.
        try:
            self._clf = self._Clf(device=self.device)
            self._reg = self._Reg(device=self.device)
        except TypeError:
            self._clf = self._Clf()
            self._reg = self._Reg()

        # Single in-context "fit" — no gradient training happens here.
        # The hosted client requires array-like (needs .shape); numpy-wrap when available.
        Xf = _np.asarray(X, dtype=float) if _np is not None else X
        self._clf.fit(Xf, _np.asarray(y_action) if _np is not None else y_action)
        self._reg.fit(Xf, _np.asarray(y_days, dtype=float) if _np is not None else y_days)

    def _offline_reference(self, current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return seed-rule labels, explicitly flagged as NOT TabPFN output.

        Uses `data_gen.label_row()` — the exact mapping TabPFN learns in-context
        from the seeded rows — so it is a defensible reference, not invented data.
        """
        results: list[dict[str, Any]] = []
        for row in current_rows:
            feats = {k: row[k] for k in FEATURE_KEYS}
            action, days = data_gen.label_row(feats)
            results.append(
                {
                    "plant_id": row.get("plant_id", row["species"]),
                    "name": row.get("name", row["species"].replace("_", " ").title()),
                    "species": row["species"],
                    "next_action": action,
                    "days_until_trouble": int(days),
                    "source": "offline_reference",
                }
            )
        return results

    def predict(self, current_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score each current plant.

        If a TabPFN token is available: genuine TabPFN inference (one forward pass
        each), results tagged source="tabpfn". Otherwise: an explicitly-flagged
        OFFLINE REFERENCE from the seed rules, results tagged
        source="offline_reference". Either way returns a list of dicts:
        {plant_id, name, species, next_action, days_until_trouble, source}.
        """
        if not _has_token():
            self.backend_mode = "offline_reference"
            return self._offline_reference(current_rows)

        self.backend_mode = "tabpfn"
        self._fit()
        X = [_encode_features(r) for r in current_rows]
        Xp = _np.asarray(X, dtype=float) if _np is not None else X
        action_codes = self._clf.predict(Xp)
        days_pred = self._reg.predict(Xp)

        results: list[dict[str, Any]] = []
        for row, a_code, days in zip(current_rows, action_codes, days_pred):
            results.append(
                {
                    "plant_id": row.get("plant_id", row["species"]),
                    "name": row.get("name", row["species"].replace("_", " ").title()),
                    "species": row["species"],
                    "next_action": _ACTION_DECODE[int(round(float(a_code)))],
                    "days_until_trouble": int(round(float(days))),
                    "source": "tabpfn",
                }
            )
        return results


if __name__ == "__main__":
    # Targeted check (a): snake/fern/succulent get different, rule-consistent advice.
    predictor = PlantPredictor()
    print(f"TabPFN backend: {predictor.backend}")
    plants = data_gen.current_plants()
    results = predictor.predict(plants)

    if predictor.backend_mode == "offline_reference":
        print(
            "\n⚠ OFFLINE REFERENCE (no TABPFN_TOKEN) — showing seed rule labels, "
            "NOT live TabPFN output.\n  Set TABPFN_TOKEN for real predictions "
            "(see README / https://ux.priorlabs.ai).\n"
        )
    else:
        print("\nGenuine TabPFN predictions for Mesh's plants:")

    for r in results:
        print(
            f"  {r['name']:12s} -> {r['next_action']:14s} "
            f"days_until_trouble={r['days_until_trouble']}  [{r['source']}]"
        )
    sys.exit(0)
