# Medium Dedupe Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate runtime-safe IO helpers for atomic writes, SLURM script rendering, and HDF open recovery without changing user-visible outputs.

**Architecture:** Build on existing private SDK helpers instead of adding parallel abstractions. The first slice extends `phenotypic.sdk_._atomic_io` with callback, JSON, and parquet helpers; later slices add shared SLURM script rendering and HDF open-recovery policy.

**Tech Stack:** Python 3.12, pytest, polars, pandas, h5py, SLURM bash script generation, `uv` runner.

---

## Scope Adjustment

The medium spec listed `src/phenotypic/sdk_/_atomic_writes.py` as a new file. The codebase already has `src/phenotypic/sdk_/_atomic_io.py` with public `atomic_write_text` and `atomic_write_bytes` exports. Phase 2 should extend that existing helper instead of creating a duplicate module.

## Task 1: Atomic Write And Parquet Policy

**Files:**
- Modify: `src/phenotypic/sdk_/_atomic_io.py`
- Modify: `src/phenotypic/sdk_/__init__.py`
- Modify: `src/phenotypic/_cli/_cli_output_manager.py`
- Modify: `src/phenotypic/_cli/_cli_chunk_writer.py`
- Modify: `src/phenotypic/_cli/_cli_recompile_worker.py`
- Modify: `src/phenotypic/_cli/_cli_sidecar.py`
- Modify: `src/phenotypic/_cli/_cli_error_outputs.py`
- Modify: `src/phenotypic/_cli/_dashboard/_analysis_helpers.py`
- Modify: `src/phenotypic/_cli/_dashboard/_manifest_builder.py`
- Modify: `src/phenotypic/gui/analysis/_recipe_state.py`
- Modify: `src/phenotypic/gui/tune/_winner.py`
- Modify: `src/phenotypic/sdk_/_io_constants.py`
- Modify: `src/phenotypic/sdk_/_qc_recipe/_recipe.py`
- Modify: `src/phenotypic/tune/_study_store.py`
- Test: `tests/unit/sdk_/test_atomic_writes.py`
- Modify: `tests/unit/tune/test_atomic_io.py`

- [x] **Step 1: Write failing tests for callback, JSON, and parquet helpers**

Add `tests/unit/sdk_/test_atomic_writes.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from phenotypic.sdk_ import (
    PARQUET_WRITE_OPTIONS,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_with_writer,
)


def _no_tmp_debris(directory: Path) -> bool:
    return not any(p.name.endswith(".tmp") for p in directory.iterdir())


def test_atomic_write_with_writer_removes_temp_after_writer_failure(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")

    def writer(path: str) -> None:
        Path(path).write_text("new", encoding="utf-8")
        raise OSError("writer failed")

    with pytest.raises(OSError, match="writer failed"):
        atomic_write_with_writer(target, writer)

    assert target.read_text(encoding="utf-8") == "old"
    assert _no_tmp_debris(tmp_path)


def test_atomic_write_json_writes_pretty_json_with_trailing_newline(tmp_path):
    target = tmp_path / "state.json"

    atomic_write_json(target, {"b": 2, "a": 1})

    assert target.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": 2}


def test_atomic_write_parquet_uses_shared_default_options(tmp_path, monkeypatch):
    target = tmp_path / "frame.parquet"
    captured: dict[str, object] = {}

    def fake_to_parquet(self, path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        Path(path).write_bytes(b"PARQUET")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    atomic_write_parquet(target, pd.DataFrame({"x": [1]}))

    assert target.read_bytes() == b"PARQUET"
    assert captured["kwargs"] == {"index": False, **PARQUET_WRITE_OPTIONS}
    assert _no_tmp_debris(tmp_path)
```

- [x] **Step 2: Verify the tests fail before implementation**

Run:

```bash
uv run pytest tests/unit/sdk_/test_atomic_writes.py -q
```

Expected: import errors for `PARQUET_WRITE_OPTIONS`, `atomic_write_json`, `atomic_write_parquet`, and `atomic_write_with_writer`.

- [x] **Step 3: Implement SDK atomic helpers**

Update `src/phenotypic/sdk_/_atomic_io.py` with:

```python
import json
from collections.abc import Callable, Mapping
from typing import Any

PARQUET_WRITE_OPTIONS: dict[str, Any] = {
    "compression": "zstd",
    "compression_level": 3,
}


def atomic_write_with_writer(
    path: Union[str, Path],
    writer: Callable[[str], None],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Union[str, None] = None
    try:
        handle = tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        )
        tmp_path = handle.name
        handle.close()
        writer(tmp_path)
        with open(tmp_path, "r+b") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def atomic_write_json(
    path: Union[str, Path],
    payload: Mapping[str, Any] | list[Any],
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=indent, sort_keys=sort_keys) + "\n",
    )


def atomic_write_parquet(
    path: Union[str, Path],
    frame: Any,
    **kwargs: Any,
) -> None:
    write_options = {"index": False, **PARQUET_WRITE_OPTIONS, **kwargs}
    atomic_write_with_writer(
        path,
        lambda tmp_path: frame.to_parquet(tmp_path, **write_options),
    )
```

Export these names from `src/phenotypic/sdk_/__init__.py`.

- [x] **Step 4: Verify SDK tests pass**

Run:

```bash
uv run pytest tests/unit/sdk_/test_atomic_writes.py -q
```

Expected: all tests pass.

- [x] **Step 5: Migrate tune parquet writer**

Update `src/phenotypic/tune/_study_store.py`:

```python
from phenotypic.sdk_ import atomic_write_parquet
```

Replace the body of `JournalStudyStore.to_parquet` with:

```python
atomic_write_parquet(path, self.to_dataframe())
```

Remove the now-unused `os` import.

- [x] **Step 6: Update tune atomic tests**

Modify `tests/unit/tune/test_atomic_io.py` so the parquet failure monkeypatch targets `phenotypic.sdk_._atomic_io.os.replace` instead of `phenotypic.tune._study_store.os.replace`.

- [x] **Step 7: Verify tune atomic tests pass**

Run:

```bash
uv run pytest tests/unit/tune/test_atomic_io.py -q
```

Expected: all tests pass.

- [x] **Step 8: Migrate runtime callback atomic writes**

Update `src/phenotypic/_cli/_cli_output_manager.py`:

```python
from phenotypic.sdk_ import atomic_write_with_writer, atomic_write_parquet
```

Remove the local `_atomic_write` implementation and replace uses:

```python
atomic_write_with_writer(csv_path, fit_pl.write_csv)
atomic_write_parquet(pq_path, fit_pl.to_pandas())
```

For existing polars frames, use callback writes when a pandas conversion would be wasteful:

```python
atomic_write_with_writer(
    pq_path,
    lambda p: frame.write_parquet(p, **PARQUET_WRITE_OPTIONS),
)
```

Update `src/phenotypic/_cli/_cli_chunk_writer.py` to import `atomic_write_with_writer` from `phenotypic.sdk_` instead of `_atomic_write` from `_cli_output_manager`.

Actual migration also included recompile worker status/shard writes, staged GPU
sidecar writes, dashboard analysis sidecar writes, dashboard manifest JSON,
error-analysis deliverable writes, GUI recipe/winner writes, and QC recipe
persistence. Those call sites either imported the removed private helper or
carried their own same-directory temp/replace implementation.

- [x] **Step 9: Verify runtime output tests pass**

Run:

```bash
uv run pytest tests/unit/cli/test_cli_output_manager.py tests/unit/cli/test_cli_analysis_plugins.py tests/unit/cli/test_cli_recompile_slurm.py tests/unit/cli/test_cli_progress_dashboard.py -q
```

Expected: all tests pass.

Verified on 2026-07-07:

```bash
uv run pytest tests/unit/sdk_/test_atomic_writes.py tests/unit/tune/test_atomic_io.py tests/unit/cli/test_cli_output_manager.py tests/unit/cli/test_cli_analysis_plugins.py tests/unit/cli/test_cli_recompile_slurm.py tests/unit/cli/test_cli_progress_dashboard.py -q
# 121 passed
uv run pytest tests/unit/cli/test_cli_sidecar.py tests/unit/cli/test_cli_error_outputs.py tests/unit/qc/test_qc_recipe.py -q
# 19 passed
uv run pytest tests/integration/gui/test_tune_winner.py tests/unit/gui/analysis/test_standalone_bundle.py tests/unit/gui/analysis/test_recipe_state_load_warnings.py -q
# 18 passed
uv run ruff check src/phenotypic/sdk_/_atomic_io.py src/phenotypic/sdk_/__init__.py src/phenotypic/tune/_study_store.py tests/unit/tune/test_atomic_io.py tests/unit/sdk_/test_atomic_writes.py src/phenotypic/_cli/_cli_output_manager.py src/phenotypic/_cli/_cli_chunk_writer.py src/phenotypic/_cli/_cli_recompile_worker.py src/phenotypic/_cli/_dashboard/_analysis_helpers.py src/phenotypic/_cli/_dashboard/_manifest_builder.py src/phenotypic/_cli/_cli_error_outputs.py src/phenotypic/_cli/_cli_sidecar.py src/phenotypic/sdk_/_qc_recipe/_recipe.py src/phenotypic/sdk_/_io_constants.py
# All checks passed
git diff --check
# no whitespace errors
```

- [x] **Step 10: Commit Task 1**

Run:

```bash
git add docs/superpowers/plans/2026-07-07-medium-dedupe-phase-2/plan.md src/phenotypic/sdk_/_atomic_io.py src/phenotypic/sdk_/__init__.py src/phenotypic/_cli/_cli_output_manager.py src/phenotypic/_cli/_cli_chunk_writer.py src/phenotypic/_cli/_cli_recompile_worker.py src/phenotypic/_cli/_cli_sidecar.py src/phenotypic/_cli/_cli_error_outputs.py src/phenotypic/_cli/_dashboard/_analysis_helpers.py src/phenotypic/_cli/_dashboard/_manifest_builder.py src/phenotypic/gui/analysis/_recipe_state.py src/phenotypic/gui/tune/_winner.py src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/_qc_recipe/_recipe.py src/phenotypic/tune/_study_store.py tests/unit/sdk_/test_atomic_writes.py tests/unit/tune/test_atomic_io.py
git commit -m "refactor: centralize atomic parquet writes"
```

## Task 2: SLURM Script Rendering Helper

**Files:**
- Create: `src/phenotypic/sdk_/slurm/_script_rendering.py`
- Modify: `src/phenotypic/_cli/_cli_slurm_array_scripts.py`
- Modify: `src/phenotypic/_cli/_cli_recompile_slurm_scripts.py`
- Modify: `src/phenotypic/_cli/_cli_staged_slurm.py`
- Modify: `src/phenotypic/_execution/_slurm.py`
- Test: `tests/unit/sdk_/test_slurm_script_rendering.py`

- [ ] **Step 1: Write failing script-rendering tests**

Add tests for a `SlurmArrayScriptSpec` that renders:

```python
#!/bin/bash
<SBATCH directives>
#SBATCH --array=0-2

set -e
set -u

<prelude>

TASK_INDICES=(
    5
    8
    13
)

...
```

The test must assert job body insertion, exit-code logging, executable chmod through a write helper, and optional signal/requeue directives.

- [ ] **Step 2: Implement `SlurmArrayScriptSpec`**

The helper should accept job name, slurm args, log path, task indices, body, prelude, and optional signal/requeue settings. It should call existing `generate_slurm_directives` rather than replacing directive policy.

- [ ] **Step 3: Migrate one call site at a time**

Migrate recompile arrays first because they are the smallest. Then migrate forward image arrays, staged GPU arrays, and tune worker arrays.

- [ ] **Step 4: Verify SLURM tests**

Run:

```bash
uv run pytest tests/unit/cli/test_cli_slurm_array.py tests/unit/cli/test_cli_recompile_slurm.py tests/unit/cli/test_staged_slurm_scripts.py tests/unit/tune/test_slurm_executor.py tests/unit/sdk_/test_slurm_dispatcher.py tests/unit/sdk_/test_slurm_script_rendering.py -q
```

- [ ] **Step 5: Commit Task 2**

```bash
git add src/phenotypic/sdk_/slurm/_script_rendering.py src/phenotypic/_cli/_cli_slurm_array_scripts.py src/phenotypic/_cli/_cli_recompile_slurm_scripts.py src/phenotypic/_cli/_cli_staged_slurm.py src/phenotypic/_execution/_slurm.py tests/unit/sdk_/test_slurm_script_rendering.py
git commit -m "refactor: share slurm array script rendering"
```

## Task 3: HDF Open-Recovery Helper

**Files:**
- Modify: `src/phenotypic/sdk_/hdf_.py`
- Test: `tests/unit/sdk_/test_hdf_open_recovery.py`

- [ ] **Step 1: Write failing HDF recovery tests**

Add tests that monkeypatch `h5py.File`, `subprocess.run`, and `time.sleep` to verify:

- lock errors retry with exponential backoff;
- `safe_writer()` runs `h5clear -s` between attempts;
- `swmr_writer()` runs `h5clear -s` and `h5clear -f` between attempts;
- non-lock `OSError` raises immediately;
- final lock failure raises the existing helpful runtime error.

- [ ] **Step 2: Extract private helper**

Add `_open_hdf_with_recovery(...)` in `hdf_.py` with arguments:

```python
def _open_hdf_with_recovery(
    filepath: Path,
    opener: Callable[[], h5py.File],
    *,
    context: str,
    lock_markers: tuple[str, ...],
    clear_status: bool,
    clear_force: bool,
    max_retries: int = 3,
    retry_delay: float = 0.5,
) -> h5py.File:
```

- [ ] **Step 3: Route `safe_writer()` and `swmr_writer()` through it**

`safe_writer()` passes status clearing only. `swmr_writer()` passes status and force clearing and keeps the SWMR-enable fallback local in its opener.

- [ ] **Step 4: Verify HDF tests**

Run:

```bash
uv run pytest tests/unit/sdk_/test_hdf_open_recovery.py tests/integration/cli/test_cli_hdf_output.py -q
```

- [ ] **Step 5: Commit Task 3**

```bash
git add src/phenotypic/sdk_/hdf_.py tests/unit/sdk_/test_hdf_open_recovery.py
git commit -m "refactor: share hdf writer recovery"
```

## Final Phase 2 Verification

Run:

```bash
uv run pytest tests/unit/sdk_/test_atomic_writes.py tests/unit/tune/test_atomic_io.py tests/unit/cli/test_cli_output_manager.py tests/unit/cli/test_cli_analysis_plugins.py
uv run pytest tests/unit/cli/test_cli_slurm_array.py tests/unit/cli/test_cli_recompile_slurm.py tests/unit/cli/test_staged_slurm_scripts.py tests/unit/tune/test_slurm_executor.py tests/unit/sdk_/test_slurm_dispatcher.py tests/unit/sdk_/test_slurm_script_rendering.py
uv run pytest tests/unit/sdk_/test_hdf_open_recovery.py tests/integration/cli/test_cli_hdf_output.py
uv run ruff check src/phenotypic/sdk_/_atomic_io.py src/phenotypic/sdk_/hdf_.py src/phenotypic/sdk_/slurm/_script_rendering.py src/phenotypic/_cli/_cli_output_manager.py src/phenotypic/_cli/_cli_chunk_writer.py src/phenotypic/_cli/_cli_slurm_array_scripts.py src/phenotypic/_cli/_cli_recompile_slurm_scripts.py src/phenotypic/_cli/_cli_staged_slurm.py src/phenotypic/_execution/_slurm.py src/phenotypic/tune/_study_store.py tests/unit/sdk_/test_atomic_writes.py tests/unit/tune/test_atomic_io.py tests/unit/sdk_/test_slurm_script_rendering.py tests/unit/sdk_/test_hdf_open_recovery.py
git diff --check
```
