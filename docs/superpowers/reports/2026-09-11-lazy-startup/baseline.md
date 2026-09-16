# Lazy startup — baseline (Task 0)

**Commit:** `BASE_PRE = f0f9a544c`. It carries the plan; its code is identical to `d437ce765`.

**Where measured:** detached worktree `/tmp/pht-lazy-base`, synced with `uv sync --group dev --group test-qt --group docs --all-extras`.

**Machine:** macOS 26.6.2 arm64, Python 3.12.11. Measured 2026-09-11.

## Startup timings

Best of 5 fresh-interpreter runs, after one warm-up. Full samples are in `startup-before.json`. The script is `docs/superpowers/plans/2026-09-11-lazy-startup/measure_startup.py`.

| Path | Best |
|---|---|
| bare interpreter | 0.011 s |
| `import phenotypic` | 1.572 s |
| `from phenotypic import Image` | 1.568 s |
| `python -m phenotypic --help` | 1.612 s |
| `phenotypic-gui --help` | 1.797 s |
| hub to servable (no request) | 2.008 s |
| hub + first `/builder/` request | 2.051 s |

## Static analysis

The finding files for later comparisons are `/tmp/lazy-mypy-before.txt` and `/tmp/lazy-ruff-before.txt`. Compare with `compare_findings.py mypy|ruff`.

- **mypy** (`--cache-dir /tmp/lazy-mypy-cache-before src/phenotypic`): `Found 418 errors in 121 files (checked 770 source files)`.
- **ruff** (`ruff check src/phenotypic`): `Found 25 errors.`
  - The spec and plan quote 65, which is the count from a wider scope used in the private-GUI gates.
  - The comparable scope here is `src/phenotypic`, so its baseline is 25.
  - The gates compare finding sets rather than counts.

## Builder e2e

Command: `PLAYWRIGHT=1 … pytest tests/e2e/gui/builder -n 4`.

Result: `15 passed, 60 skipped in 49.04s`, with no failures.

## Known pre-existing local failures (default lanes)

- `tests/unit/test_ngff_schema_fixtures.py::test_schema_matches_recorded_digest[image|label|ome|_version.schema]`
  - Cause: autocrlf rewrites the vendored schema bytes. The committed LF blobs match `SOURCE.md`.
- `tests/e2e/gui/test_builder_preview_viv.py` under `-n`
  - Cause: the preview cache root is shared across processes. The test passes alone.
