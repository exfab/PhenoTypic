# Baseline — private GUI module

**Commit:** `main @ 0117572d9` (branch `refactor/private-gui`, no source changes yet) · **Date:** 2026-09-10 · **Host:** macOS (darwin), local

## Test surface

Derived with the command in `plan.md` § "Test surface": 262 test files.

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest $(cat /tmp/private-gui-surface.txt) \
  -q --no-header -p no:randomly -p no:cacheprovider -o addopts= -m "not slow" -n 6 \
  --junitxml=/tmp/private-gui-baseline.xml
```

Result: `3009 passed, 16 skipped, 3 xfailed, 29 warnings in 62.83s`, exit 0.
Failing node IDs: none.

## Static checks

| Check | Result |
|---|---|
| `uv run mypy src/phenotypic` | `Found 418 errors in 121 files (checked 771 source files)` |
| `uv run ruff check src/phenotypic scripts tests` | `Found 65 errors.` |

Both are pre-existing on `main`; the gate is "not worse", compared by finding set.
