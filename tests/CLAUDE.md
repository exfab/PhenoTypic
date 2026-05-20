# Tests

## Markers

Registered in `[tool.pytest.ini_options].markers` in
[pyproject.toml](../pyproject.toml).

| Marker | Default | Meaning |
|---|---|---|
| `smoke` | runs | Basic operation test. |
| `slow` | **deselected** (`addopts = "-m 'not slow'"`) | Heavy parametrize sweep, excluded from PR runs. Run nightly with `pytest -m slow` or `-m "smoke or slow"`. |
| `ci_flaky` | runs locally, **deselected on CI** | E2E tests that pass reliably locally but flake on GitHub-hosted shared runners. CI passes `-m "not ci_flaky"`. |

## The `ci_flaky` convention

### What it's for

Browser-driven Playwright E2E tests whose timing budgets (Playwright
`wait_for_function` polls, hard-coded disk-poll loops on Dash callback
chains, etc.) pass reliably on developer hardware but stochastically
exceed those budgets on GHA `ubuntu-latest` runners — typically because
a 2-vCPU shared VM has higher tail latency than an M-series laptop or a
beefy workstation. The tests are **correct** and **the SUT works**; the
test harness's timing budget is too tight for the noisier environment.

Add this marker only after confirming with the workflow below that the
test:

1. Passes ≥10 consecutive local runs of the **full containing file**
   (single-test repeats are insufficient — the failure mode is
   per-test variance, but you want to rule out test-ordering flake).
2. Has **failed on at least two separate CI runs** with a Playwright
   timeout, a `_time.monotonic()` deadline, or another wall-clock
   assertion against a Dash callback chain.
3. Is **not** failing due to a real bug (read the diff; reproduce the
   timing locally by spawning CPU hogs and re-running if unsure).

A test that fails once on CI is not flaky — it's a normal failure
until proven otherwise.

### How to mark

**Whole module** (preferred when most tests in the file share the same
DOM-poll pattern):

```python
import pytest
pytestmark = pytest.mark.ci_flaky
```

**Single test** (when only one test in an otherwise-stable file
flakes):

```python
@pytest.mark.ci_flaky
def test_color_picker_lists_measurements_and_qc_severities(...):
    ...
```

Pair the marker with a one-line comment naming the specific timing
budget that flakes, so a later reader can decide whether to widen the
budget or restructure the wait. Example from
[tests/e2e/gui/test_heatmap_tab.py](e2e/gui/test_heatmap_tab.py):

```python
# Module-level marker: skipped on CI via ``-m "not ci_flaky"`` in the
# gui-e2e workflow. Locally these tests pass reliably (verified: 24/24
# across three full-file runs on macOS aarch64); on GHA ubuntu-latest
# shared runners the Dash callback chain + Plotly first-render budget
# stochastically exceeds the 15s ``wait_for_function`` poll.
pytestmark = pytest.mark.ci_flaky
```

### How to run

```bash
# Default: includes ci_flaky (use this when developing locally)
PLAYWRIGHT=1 uv run pytest tests/e2e/gui

# CI behavior: skip ci_flaky
PLAYWRIGHT=1 uv run pytest tests/e2e/gui -m "not ci_flaky"

# Only the flaky ones (e.g. to re-validate after a server-side speedup)
PLAYWRIGHT=1 uv run pytest tests/e2e/gui -m ci_flaky
```

### Re-validation: removing the marker

The marker is a **debt entry**, not a permanent label. Re-validate
periodically (e.g. after every server-side perf improvement to the
Dash callback graph or Plotly render path):

1. Remove `pytestmark = pytest.mark.ci_flaky` (or the per-test
   decorator) locally.
2. Push a draft PR; let `gui-e2e` run twice in a row.
3. If both runs pass, drop the marker. If either run flakes with the
   same timing-budget signature, restore the marker and document why
   in the comment above it.

### What this is NOT

- **Not** a way to silence a real bug. If the test fails for a reason
  other than wall-clock-poll-timeout (assertion mismatch, missing DOM
  element, server error), fix the test — `ci_flaky` is the wrong tool.
- **Not** a substitute for genuinely-fast tests. Anything that can be
  rewritten to wait on a server-side completion signal (Dash callback
  resolved event, file-write watchdog, etc.) instead of polling DOM
  text should be rewritten. The marker is for "we have to wait on a
  fuzzy condition and the budget is tight."
- **Not** for unit or integration tests. Those run on Linux too and
  should be deterministic. `ci_flaky` is exclusively for the
  Playwright/`tests/e2e/gui/` lane.

## Layout

```
tests/
├── CLAUDE.md                 ← this file
├── unit/                     ← deterministic, no I/O beyond tmp_path
├── smoke/                    ← end-to-end sanity, fast
├── integration/              ← multi-component, sandboxed
│   ├── cli/
│   └── gui/                  ← Flask test-client (no browser)
├── e2e/
│   └── gui/                  ← Playwright + Werkzeug subprocess
│       ├── conftest.py       ← live_server, fake_sandbox helpers
│       └── builder/          ← DAG builder sub-suite
└── fixtures/
    └── builder_dag/          ← JSON DAG fixtures (UTF-8; always read with encoding="utf-8")
```

## Gotchas

- **Always read JSON fixtures with `encoding="utf-8"`.** `Path.read_text()`
  and `Path.open()` default to the locale codec, which is `cp1252` on
  Windows CI runners and chokes on UTF-8 characters in fixtures (e.g.
  the leftwards arrow in
  [fixtures/builder_dag/aux_cycle.json](fixtures/builder_dag/aux_cycle.json)).
- **`PLAYWRIGHT=1` is required** for any test under `tests/e2e/gui/`.
  The conftest module-skip enforces this — running without it produces
  "Set PLAYWRIGHT=1 to run browser E2E tests" and the entire module is
  skipped, not failed.
- **Per-test Werkzeug cold start is ~2.5s** on M-series local; ~5–8s
  on GHA runners. Tests that fit ≤10s of work into a fresh function-scoped
  server on local hardware can blow past 30s on CI. This is the
  signature `ci_flaky` exists to absorb.
