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
