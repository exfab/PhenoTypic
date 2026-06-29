---
name: ci-flaky-quarantine
description: How to mark, run, and retire the pytest ci_flaky marker for PhenoTypic's Playwright GUI E2E tests. Use when a tests/e2e/gui/ Playwright test passes locally but times out on GitHub-hosted CI runners, or when deciding whether to add/remove the ci_flaky marker.
---

# The `ci_flaky` convention

`ci_flaky` marks browser-driven Playwright E2E tests (under `tests/e2e/gui/`)
whose timing budgets pass reliably on developer hardware but stochastically
exceed those budgets on GHA `ubuntu-latest` runners (a 2-vCPU shared VM has
higher tail latency than an M-series laptop). The tests are **correct** and the
SUT works; the harness's timing budget is too tight for the noisier
environment. CI passes `-m "not ci_flaky"`; locally the marker runs.

## Add the marker only after confirming all three

1. The test **passes ≥10 consecutive local runs of the full containing file**
   (single-test repeats are insufficient — rule out test-ordering flake).
2. It has **failed on ≥2 separate CI runs** with a Playwright timeout, a
   `time.monotonic()` deadline, or another wall-clock assertion against a Dash
   callback chain.
3. It is **not** failing due to a real bug (read the diff; reproduce the timing
   locally by spawning CPU hogs and re-running if unsure).

A test that fails once on CI is a normal failure until proven otherwise.

## How to mark

Whole module (preferred when most tests share the same DOM-poll pattern):

```python
import pytest
pytestmark = pytest.mark.ci_flaky
```

Single test (when only one test in an otherwise-stable file flakes):

```python
@pytest.mark.ci_flaky
def test_color_picker_lists_measurements_and_qc_severities(...):
    ...
```

Pair the marker with a one-line comment naming the specific timing budget that
flakes, so a later reader can decide whether to widen the budget or restructure
the wait. Canonical example: `tests/e2e/gui/test_heatmap_tab.py`.

## How to run

```bash
# Default: includes ci_flaky (use when developing locally)
PLAYWRIGHT=1 uv run pytest tests/e2e/gui

# CI behavior: skip ci_flaky
PLAYWRIGHT=1 uv run pytest tests/e2e/gui -m "not ci_flaky"

# Only the flaky ones (e.g. to re-validate after a server-side speedup)
PLAYWRIGHT=1 uv run pytest tests/e2e/gui -m ci_flaky
```

## Retiring the marker (it's a debt entry, not a label)

Re-validate periodically — especially after any server-side perf improvement to
the Dash callback graph or Plotly render path:

1. Remove `pytestmark = pytest.mark.ci_flaky` (or the per-test decorator)
   locally.
2. Push a draft PR; let `gui-e2e` run twice in a row.
3. If both runs pass, drop the marker. If either flakes with the same
   timing-budget signature, restore the marker and document why in the comment
   above it.

## What `ci_flaky` is NOT

- **Not** a way to silence a real bug. If the test fails for any reason other
  than wall-clock-poll-timeout (assertion mismatch, missing DOM element, server
  error), fix the test.
- **Not** a substitute for fast tests. Anything that can wait on a server-side
  completion signal (Dash callback-resolved event, file-write watchdog) instead
  of polling DOM text should be rewritten.
- **Not** for unit or integration tests — those run on Linux too and must be
  deterministic. `ci_flaky` is exclusively for the `tests/e2e/gui/` Playwright
  lane.
