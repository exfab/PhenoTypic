# Tests

## Markers

Registered in `[tool.pytest.ini_options].markers` in
[pyproject.toml](../pyproject.toml).

| Marker | Default | Meaning |
|---|---|---|
| `smoke` | runs | Basic operation test. |
| `slow` | **deselected** (`addopts = "-m 'not slow'"`) | Heavy parametrize sweep, excluded from PR runs. Run nightly with `pytest -m slow` or `-m "smoke or slow"`. |
| `ci_flaky` | runs locally, **deselected on CI** | E2E tests that pass reliably locally but flake on GitHub-hosted shared runners. CI passes `-m "not ci_flaky"`. |
| `postgres` | autoskipped | Tune study-DB tests; skipped unless `PHENOTYPIC_TEST_PG_URL` is set. |
| `slurm` | autoskipped | Distributed tune-worker tests; skipped unless `sbatch` is on PATH. |

## The `ci_flaky` convention (summary)

`ci_flaky` marks `tests/e2e/gui/` Playwright tests that pass locally but time out
on GHA shared runners; CI runs `-m "not ci_flaky"`. Add it **only** after ≥10 green
local full-file runs + ≥2 CI timeout failures + ruling out a real bug. It's a debt
entry — re-validate and remove after server-side speedups. The full marking,
running, and retirement procedure is in the **`ci-flaky-quarantine`** skill.

## Layout

```
tests/
├── CLAUDE.md                 ← this file
├── unit/                     ← deterministic, no I/O beyond tmp_path
├── smoke/                    ← end-to-end sanity, fast
├── gui/                      ← top-level Dash/unit GUI suite (in testpaths)
│   ├── builder/  browse/  results_viewer/  _shared/
├── integration/              ← multi-component, sandboxed
│   ├── cli/
│   └── gui/                  ← Flask test-client (no browser)
├── e2e/                      ← NOT in testpaths; PLAYWRIGHT=1-gated
│   └── gui/                  ← Playwright + Werkzeug subprocess
│       ├── conftest.py       ← live_server, fake_sandbox helpers
│       └── builder/          ← DAG builder sub-suite
├── migration/                ← NOT in testpaths; golden-comparison suite
│   └── _goldens/             ← recorded outputs — see the warning below
└── fixtures/
    └── builder_dag/          ← JSON DAG fixtures (UTF-8; always read with encoding="utf-8")
```

## `tests/migration` runs only when you ask for it — and it is currently red

`testpaths` is `["tests/unit", "tests/smoke", "tests/integration", "tests/gui"]`.
**`tests/migration` is not in it**, so a bare `uv run pytest` never collects it. It is
reached only by naming it explicitly, or through the sharded gate harness with
`SCOPE=full`.

**As of 2026-09-05, 57 scenarios in `test_equivalence.py` fail against their goldens**,
across every operation subpackage — `analysis`, `correction`, `detect`, `enhance`, `grid`,
`measure`, `refine`. The differences are **not** rounding: the population is bimodal, with
max absolute differences running from `1.9e-08` up to **`1.5157`**, and mismatched-element
fractions up to **100%**. Something moved the output of these operations and nothing caught
it, because nothing runs this suite.

**Do not regenerate the goldens to get a green run.** Goldens exist to catch exactly this
drift, and re-recording them converts an unexplained behaviour change into a blessed one with
no trace that it happened. At a 100% mismatch that is not a tidy-up — it is the deletion of
the only evidence. Diagnose the drift, or leave the suite red and state the exclusion.

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
