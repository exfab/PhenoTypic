# Source Timeline View — Phase 5: CLI `deliverables/metadata.csv` Copy

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement spec §8 (decision D6): a **best-effort**, non-blocking copy of the run's `--metadata` source CSV to `<output>/deliverables/metadata.csv`, plus a single-sourced filename constant + path helper. This preserves the full, portable, co-located original metadata mapping alongside the run's other deliverables — the inner-join mirror (`deliverables/measurements.parquet`) drops measurement rows with no matching metadata key, and `progress/job_metadata.json` only records an absolute path that is useless once results move off the cluster (spec §8.2).

**Architecture:** This phase is **pure CLI/IO** — fully independent of the timeline GUI surfaces (Phases 1–4) and of the §16 focus-navigate pivot. It touches exactly two production modules: `phenotypic.sdk_._io_constants` (a new filename constant + `metadata_csv_deliverable_path` helper, matching the existing `deliverables_dir` / `master_measurements_parquet_path` style) and `phenotypic._cli._cli_output_manager.finalize_post_master_outputs` (a guarded `shutil.copy` of the source CSV into the deliverables path).

**Key code finding (deviation from spec §8.1 — simpler than the spec implied):** the spec describes reading the `--metadata` path back out of `progress/job_metadata.json` (`JobMetadataKey.METADATA_CSV`) inside finalize. In the **real code** that is unnecessary: `finalize_post_master_outputs(output_dir, master_df, pipeline, metadata_csv=None, no_qc=False)` **already accepts `metadata_csv` as a parameter** (`_cli_output_manager.py:506-512`), and **both** final-write call sites already thread the path in:

- `aggregate_measurements` (forward CLI) → `finalize_post_master_outputs(..., metadata_csv=metadata_csv, ...)` (`_cli_output_manager.py:904-907`).
- `_run_post_master_steps` (`--recompile` worker) → reads `task.get(JobMetadataKey.METADATA_CSV)` into `metadata_csv` and passes it (`_cli_recompile_worker.py:386-392`).

So the copy is implemented **once**, inside `finalize_post_master_outputs`, using the already-present `metadata_csv` argument. **No call-site changes are required** and finalize does **not** open `job_metadata.json`. `JobMetadataKey.METADATA_CSV` (`_io_constants.py:1415`, value `"metadata_csv"`) is verified to exist and is what the recompile worker reads to *populate* that argument — but finalize itself only sees the resolved `Path`.

**Tech Stack:** Python 3, `pathlib.Path`, `shutil` (already imported at `_cli_output_manager.py:14`), pytest, polars (only in tests, for staging a master frame).

## Global Constraints

- **`uv` is the sole runner.** Every command is `uv run …`; never bare `python`/`pip`.
- **Single-source the filename.** The literal `"metadata.csv"` lives once as `DELIVERABLES_METADATA_CSV: Final[str]` in `phenotypic.sdk_._io_constants`, mirroring `MEASUREMENTS_CSV` / `MASTER_MEASUREMENTS_CSV`. The path is composed by `metadata_csv_deliverable_path(output_dir)` rooted at `deliverables_dir(output_dir)`, exactly like `master_measurements_csv_path` / `measurements_csv_path`. Never hand-join the name.
- **Best-effort, never raises.** The copy is guarded like every other finalize side effect (`_seed_measurements`, `split_master_by_feature`, `reemit_error_deliverables`): wrap in `try/except Exception`, log at WARNING with `exc_info=True`, and continue. A missing/unreadable source CSV, a permission error, or any other failure must **not** abort finalize — the master/mirror outputs stay authoritative.
- **Finalize-only — do NOT touch the chunk writer.** The copy is added to `finalize_post_master_outputs` and nowhere else. `_aggregate_chunks_locked` (`_cli_chunk_writer.py:78`) is a **mid-run intermediate** writer that intentionally bypasses the post pipeline / per-feature splits / analysis chain (it does its own `join_metadata` at `_cli_chunk_writer.py:146-148` but never calls `finalize_post_master_outputs`). Adding the copy there would re-run the work on every checkpoint and violate the documented finalize-only convention (CLAUDE.md "Finalize via finalize_post_master_outputs"). State this explicitly; add nothing to the chunk writer.
- **No call-site changes.** Because `finalize_post_master_outputs` already receives `metadata_csv`, the two call sites (`aggregate_measurements`, `_run_post_master_steps`) are **unchanged**. Do not re-thread the path or add a `job_metadata.json` read.
- **Export discipline.** `_io_constants.py` has **no module-level `__all__`**; its public names are surfaced by being imported into `phenotypic/sdk_/__init__.py` and listed in *that* module's `__all__`. Add the new constant + helper to the `from ._io_constants import (...)` block **and** the `__all__` list in `sdk_/__init__.py`, alongside the existing `deliverables_dir` / `measurements_csv_path` entries. **Do not** re-export through `gui/_config.py`: this filename is consumed only by the CLI finalize path, not by GUI import code, and the GUI re-export list (`gui/_config.py:68-78,122-126`) deliberately carries only the names the GUI reads (`MEASUREMENTS_CSV`, `MASTER_MEASUREMENTS_PARQUET`, …). (Spec §9 mentions "re-exported through `gui/_config.py`"; the **established convention** is to re-export only when a GUI module imports the name — no Phase 5 GUI code does, so it stays out. Note this deviation in the commit body if a reviewer expects the re-export.)
- **Test collection.** Both new test homes are already inside `testpaths = ["tests/unit", "tests/smoke", "tests/integration"]` (`pyproject.toml:190`): the path-helper unit test lands in `tests/unit/sdk_/test_io_constants.py` (extend the existing `TestPathHelpers` class) and the finalize-copy test lands in `tests/unit/cli/test_cli_output_manager.py` (new test class). No `testpaths` change is needed in this phase.

---

### Task 1: `DELIVERABLES_METADATA_CSV` constant + `metadata_csv_deliverable_path` helper

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py` (new `Final[str]` constant next to `MEASUREMENTS_CSV`; new path helper next to `measurements_csv_path`)
- Modify: `src/phenotypic/sdk_/__init__.py` (import + `__all__`)
- Test: `tests/unit/sdk_/test_io_constants.py` (extend `TestPathHelpers`; add a filename assertion to the filenames test class)

**Interfaces:**
- Consumes: `deliverables_dir` (existing), `DIR_DELIVERABLES` (existing).
- Produces:
  - `DELIVERABLES_METADATA_CSV: Final[str] = "metadata.csv"` — the co-located copy of the run's `--metadata` source CSV under `deliverables/`.
  - `metadata_csv_deliverable_path(output_dir: Path) -> Path` → `<output>/deliverables/metadata.csv`.
  - Both re-exported from `phenotypic.sdk_`.

- [ ] **Step 1: Write the failing test**

Extend `tests/unit/sdk_/test_io_constants.py`. Add the import to the existing `from phenotypic.sdk_ import (...)` block (keep alphabetical order — between `measurements_parquet_path` and `pipeline_json_path` for the helper; the constant comes from the same package):

```python
from phenotypic.sdk_ import (
    ...
    measurements_csv_path,
    measurements_parquet_path,
    metadata_csv_deliverable_path,
    ...
)
```

Add one assertion to the **filenames** test class (the class holding `test_measurements_mirror_filenames`, around `test_io_constants.py:190`) — import `DELIVERABLES_METADATA_CSV` at the top alongside the other filename constants and assert its value:

```python
def test_metadata_csv_deliverable_filename(self) -> None:
    assert DELIVERABLES_METADATA_CSV == "metadata.csv"
```

Add one method to `TestPathHelpers` (next to `test_measurements_mirror_paths`, `test_io_constants.py:340`):

```python
def test_metadata_csv_deliverable_path(self, output: Path) -> None:
    deliv = output / "deliverables"
    assert metadata_csv_deliverable_path(output) == deliv / "metadata.csv"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/sdk_/test_io_constants.py -v -k "metadata_csv_deliverable"`
Expected: FAIL with `ImportError: cannot import name 'metadata_csv_deliverable_path' from 'phenotypic.sdk_'` (and/or `DELIVERABLES_METADATA_CSV`).

- [ ] **Step 3: Write minimal implementation**

In `src/phenotypic/sdk_/_io_constants.py`, add the constant immediately after the `MEASUREMENTS_PARQUET` block (around line 170), matching the surrounding `#:`-comment + `Final[str]` style:

```python
#: Best-effort co-located copy of the run's ``--metadata`` source CSV,
#: written into :data:`DIR_DELIVERABLES` by
#: :func:`phenotypic._cli._cli_output_manager.finalize_post_master_outputs`.
#: The post-applied mirror (:data:`MEASUREMENTS_CSV`) carries the metadata
#: columns but the join is *inner* — rows with no matching key are dropped —
#: so this preserves the full, portable original mapping next to the other
#: deliverables (spec §8 / D6). The copy is best-effort and never blocks the
#: run; a missing/unreadable source is logged and skipped.
DELIVERABLES_METADATA_CSV: Final[str] = "metadata.csv"
```

Add the path helper immediately after `measurements_parquet_path` (around line 797), matching that helper's one-line-docstring style:

```python
def metadata_csv_deliverable_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/metadata.csv`` (co-located ``--metadata`` copy)."""
    return deliverables_dir(output_dir) / DELIVERABLES_METADATA_CSV
```

In `src/phenotypic/sdk_/__init__.py`, add both names to the `from ._io_constants import (...)` block and to `__all__`, alphabetically beside the existing deliverables entries:

- import block: add `DELIVERABLES_METADATA_CSV` near the other filename constants and `metadata_csv_deliverable_path` after `measurements_parquet_path` (`__init__.py:153`).
- `__all__`: add `"DELIVERABLES_METADATA_CSV"` and `"metadata_csv_deliverable_path"` beside `"measurements_parquet_path"` (`__init__.py:363`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/sdk_/test_io_constants.py -v -k "metadata_csv_deliverable or filenames"`
Expected: PASS (the new filename assertion + the new path-helper assertion green; existing tests in the file still pass).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/__init__.py tests/unit/sdk_/test_io_constants.py
git commit -m "feat(cli): add DELIVERABLES_METADATA_CSV + metadata_csv_deliverable_path helper"
```

---

### Task 2: Best-effort `--metadata` CSV copy inside `finalize_post_master_outputs`

**Files:**
- Modify: `src/phenotypic/_cli/_cli_output_manager.py` (`finalize_post_master_outputs` — add a guarded `shutil.copy` block; extend the docstring)
- Test: `tests/unit/cli/test_cli_output_manager.py` (new `TestFinalizeCopiesMetadataCsv` class)

**Interfaces:**
- Consumes: the existing `metadata_csv: Optional[Path]` parameter of `finalize_post_master_outputs` (`_cli_output_manager.py:510`); `metadata_csv_deliverable_path` (Task 1); `shutil` (already imported, `_cli_output_manager.py:14`).
- Produces: when `metadata_csv` is a readable file, `<output>/deliverables/metadata.csv` is a byte-for-byte copy of it. When `metadata_csv` is `None`, missing, or unreadable, **no exception escapes** and the rest of finalize runs unchanged.

- [ ] **Step 1: Write the failing test**

Append a new class to `tests/unit/cli/test_cli_output_manager.py` (the file already imports `finalize_post_master_outputs`, `pl`, `Path`, `pytest`, and `ImagePipeline`):

```python
class TestFinalizeCopiesMetadataCsv:
    """``finalize_post_master_outputs`` copies the ``--metadata`` source CSV to
    ``deliverables/metadata.csv`` (best-effort, never raising) — spec §8 / D6."""

    @staticmethod
    def _master_df() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Metadata_Dataset": ["ds1", "ds1"],
                "Metadata_ImageFile": ["plateA", "plateA"],
                "Object_Label": [1, 2],
                "Size_Area": [100.0, 110.0],
            }
        )

    def test_copies_metadata_csv_byte_for_byte(self, tmp_path: Path) -> None:
        import phenotypic.sdk_ as tools_

        output_dir = tmp_path / "out"
        output_dir.mkdir()
        # A real metadata CSV whose key column matches the master so the inner
        # join succeeds (the copy must not depend on join success, but this
        # mirrors a normal run).
        source = tmp_path / "meta.csv"
        # Non-ASCII strain name (accented) so the byte-level read_bytes()
        # comparison also catches any accidental text-mode re-encode in a
        # future refactor (e.g. a copy that round-trips through str / a
        # platform-default codec). A byte-for-byte copy preserves the UTF-8 cell.
        source.write_text(
            "Metadata_ImageFile,Metadata_Strain\nplateA,Säccharomyces\n",
            encoding="utf-8",
        )

        finalize_post_master_outputs(
            output_dir, self._master_df(), ImagePipeline(),
            metadata_csv=source, no_qc=True,
        )

        copied = tools_.metadata_csv_deliverable_path(output_dir)
        assert copied.exists()
        assert copied.read_bytes() == source.read_bytes()

    def test_no_metadata_csv_means_no_copy(self, tmp_path: Path) -> None:
        import phenotypic.sdk_ as tools_

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        finalize_post_master_outputs(
            output_dir, self._master_df(), ImagePipeline(),
            metadata_csv=None, no_qc=True,
        )

        assert not tools_.metadata_csv_deliverable_path(output_dir).exists()
        # finalize still produced the mirror.
        assert tools_.measurements_parquet_path(output_dir).exists()

    def test_missing_source_is_best_effort_no_raise(self, tmp_path: Path) -> None:
        import phenotypic.sdk_ as tools_

        output_dir = tmp_path / "out"
        output_dir.mkdir()
        missing = tmp_path / "does_not_exist.csv"

        # Must NOT raise even though the source is absent (the join itself is
        # also guarded, so finalize completes and seeds the mirror).
        finalize_post_master_outputs(
            output_dir, self._master_df(), ImagePipeline(),
            metadata_csv=missing, no_qc=True,
        )

        assert not tools_.metadata_csv_deliverable_path(output_dir).exists()
        assert tools_.measurements_parquet_path(output_dir).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli/test_cli_output_manager.py::TestFinalizeCopiesMetadataCsv -v`
Expected: **exactly one** of the three tests fails red — `test_copies_metadata_csv_byte_for_byte` fails on `assert copied.exists()` (no copy is written yet). That single red is the gate proving the new behavior is genuinely absent.

The other two — `test_no_metadata_csv_means_no_copy` and `test_missing_source_is_best_effort_no_raise` — are **expected to pass already**, before any implementation: there is no copy logic to break, and finalize already seeds the mirror, so the "no copy file" + "no raise" + "mirror exists" assertions all hold today. They are **regression guards**, not part of the red gate. Do **not** mistake their pre-implementation green for a broken/missing red gate, and do **not** "fix" them to fail — the one red on `test_copies_metadata_csv_byte_for_byte` is the only failure required here. All three must be green after Step 3.

- [ ] **Step 3: Write minimal implementation**

In `finalize_post_master_outputs` (`_cli_output_manager.py`), add a best-effort copy block. Place it **after** the metadata-join block and **after** `_seed_measurements(output_dir, post_df)` (so the deliverables dir is on the natural write path), guarded exactly like the other side effects. The function already has `from pathlib import Path` semantics for `metadata_csv`, and `shutil` is imported at module top (line 14):

```python
    # Best-effort: preserve the FULL original ``--metadata`` mapping next to the
    # other deliverables. The mirror's metadata join is *inner* (rows with no
    # matching key are dropped) and ``job_metadata.json`` only records an absolute
    # path useless once results move off the cluster, so a co-located copy is the
    # only portable, complete artifact (spec §8 / D6). Guarded like every other
    # finalize side effect — a missing/unreadable source is logged and skipped,
    # never aborting the run.
    #
    # Content-only copy by design: use ``shutil.copy`` (data + mode), NOT
    # ``shutil.copy2`` — the source mtime is irrelevant to the deliverable and is
    # deliberately not preserved. Re-running the CLI (forward) or a ``--recompile``
    # finalize overwrites ``deliverables/metadata.csv`` with a fresh copy, mirroring
    # how ``master_measurements.*`` / the mirror are rewritten on every rerun.
    if metadata_csv is not None:
        try:
            dest = metadata_csv_deliverable_path(output_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(metadata_csv, dest)  # overwrites on rerun; mtime not preserved
        except Exception:
            logger.warning(
                "Failed to copy metadata CSV to deliverables/ (master/mirror "
                "still written)",
                exc_info=True,
            )
```

Add the import at the top of `_cli_output_manager.py` — extend whichever `from phenotypic.sdk_ import (...)` / `from phenotypic.sdk_._io_constants import (...)` block already pulls in `measurements_csv_path` / `measurements_parquet_path` (the helpers `_seed_measurements` uses), adding `metadata_csv_deliverable_path` there. (Verify the exact existing import block at authoring time; match its form rather than adding a new bare import line.)

Extend the docstring: add a numbered step (between the current step 3 `_seed_measurements` and step 4 split) noting the best-effort copy, and add a sentence to the `metadata_csv` Args entry: *"When provided, the source file is also best-effort copied to ``deliverables/metadata.csv`` (spec §8 / D6) so the full original mapping survives the inner join; the copy is content-only (``shutil.copy``, mtime not preserved) and is overwritten with a fresh copy on every rerun/``--recompile``, like the master and mirror; a failed copy is logged and never raises."*

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/cli/test_cli_output_manager.py::TestFinalizeCopiesMetadataCsv -v`
Expected: PASS (3 tests). Then run the whole file to confirm no regression:
Run: `uv run pytest tests/unit/cli/test_cli_output_manager.py -v`
Expected: PASS (all existing tests + the 3 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_output_manager.py tests/unit/cli/test_cli_output_manager.py
git commit -m "feat(cli): best-effort copy of --metadata CSV to deliverables/metadata.csv"
```

---

### Task 3: Phase regression + lint + type check

**Files:** none (verification only).

- [ ] **Step 1: Run the two touched test modules together**

Run: `uv run pytest tests/unit/sdk_/test_io_constants.py tests/unit/cli/test_cli_output_manager.py -v`
Expected: PASS (both modules green, including all pre-existing tests).

- [ ] **Step 2: Confirm the chunk writer was NOT modified**

Run: `git diff --stat`
Expected: changes only in `src/phenotypic/sdk_/_io_constants.py`, `src/phenotypic/sdk_/__init__.py`, `src/phenotypic/_cli/_cli_output_manager.py`, and the two test files. **`src/phenotypic/_cli/_cli_chunk_writer.py` must NOT appear** — the copy is finalize-only (Global Constraints). If it appears, STOP and revert that change.

- [ ] **Step 3: Lint + type check the touched modules**

Run: `uv run ruff check src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/__init__.py src/phenotypic/_cli/_cli_output_manager.py`
Run: `uv run mypy src/phenotypic/sdk_/_io_constants.py src/phenotypic/_cli/_cli_output_manager.py`
Expected: clean (fix any reported issues before the final commit; re-run the Step 1 suite after any fix).

- [ ] **Step 4: Commit any lint/type fixups (only if Step 3 required edits)**

```bash
git add -A
git commit -m "chore(cli): ruff/mypy fixups for metadata-csv deliverable copy"
```

---

## Phase 5 deliverable

A single-sourced `DELIVERABLES_METADATA_CSV` filename constant + `metadata_csv_deliverable_path(output_dir)` helper in `phenotypic.sdk_._io_constants` (re-exported through `phenotypic.sdk_`), and a **best-effort, never-raising** `shutil.copy` of the run's `--metadata` source CSV to `<output>/deliverables/metadata.csv` inside `finalize_post_master_outputs`. The copy reuses the already-threaded `metadata_csv` parameter — **no call-site changes** at `aggregate_measurements` or the `--recompile` worker, and **no** addition to the mid-run chunk writer (`_aggregate_chunks_locked`), matching the documented finalize-only convention. This gives Results-Timeline users (and anyone moving a run off the cluster) a portable, co-located copy of the full original metadata mapping — the piece the inner-join mirror drops. Pure CLI/IO: independent of the timeline GUI engine (Phases 1–4) and the §16 focus-navigate pivot.

## Relationship to the other phases

- **Phases 1–4** (shared engine, Browse, Results, Compare strip) build the GUI timeline surfaces; they read `deliverables/measurements.parquet` (the post-applied mirror with joined `Metadata_*` columns) for the core axes (spec §8.1) and do **not** depend on this copy. The `deliverables/metadata.csv` copy is an additive durability/portability artifact (spec §8.2), not a runtime dependency of the timeline.
- **Phase 6** (docs/CI) wraps the feature: a `code-simplifier` pass + regression run over touched areas. This phase adds no GUI affordance, so it touches **no** `gui/` files and therefore is **not** subject to the `FEATURES.md` / `WORKFLOWS.md` gates.
