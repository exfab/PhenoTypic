# Parallel Measure-Only Marker Publication Implementation Record

**Goal:** Remove serial pre-dispatch work from a legacy SLURM remeasurement while using the existing parallel workers to regenerate success markers.

## Delivered changes

- `phenotypicCLI.py`: skips legacy marker migration only for `--mode measure`.
- `_cli_state_management.py`: skips HDF identity calculation when selecting measure-only work.
- `_cli_slurm_array_scripts.py`: omits expected identity and hash flags when rendering measure-only worker scripts.
- Tests cover the mode guard, selection path, and rendered script; normal array-script coverage remains green.

## Verification

- `uv run --offline pytest tests/unit/cli/test_cli_state_management.py -q` - 12 passed.
- `uv run --offline pytest tests/unit/cli/test_cli_slurm_array.py -q` - 37 passed.
- Scoped Ruff, bytecode compilation, and `git diff --check` passed.
- Live launcher job `27737261` completed in 47 seconds and submitted array `27737269`; task 0 remeasured an HDF and exited 0.
