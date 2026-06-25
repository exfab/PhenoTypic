# QC DuckDB Per-Module Tables — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat QC parquet artifact (`deliverables/qc/qc_summary.parquet` + `qc_members.parquet` + `qc_config.json`) with a single DuckDB database `deliverables/qc/qc.duckdb` holding one self-describing table per QC module plus a `qc_modules` catalog, and rewire the results-viewer Review + Error tabs onto a catalog-driven data API.

**Architecture:** `phenotypic.sdk_._qc_recipe._runner.run_qc` becomes the sole DuckDB writer (always an **atomic full rebuild** via temp-file + `os.replace`). Each `QualityCheck` describes itself via new `to_table()` + `table_spec()` methods. The GUI reads through short-lived `read_only` DuckDB connections in a new `_db.py` data layer, deleting the flat-schema reverse-engineering helpers. Both GUI live recompute triggers (curation + settings-edit) run the same full rebuild. Full cutover: the parquet artifact, the GUI "Export QC report" button, and the public `phenotypic.sdk_` QC-parquet symbols are removed.

**Tech Stack:** Python 3.12, `uv`, pydantic v2, polars + pandas, DuckDB (new dep), Dash (GUI), pytest.

**Design spec:** `docs/superpowers/specs/2026-06-25-qc-duckdb-per-module-tables-design.md`

## Global Constraints

- **Package manager/runner:** `uv` only. Never bare `python`/`pip`. Tests: `uv run pytest …`; type check: `uv run mypy src/phenotypic`; lint: `uv run ruff check --fix`.
- **Operations are pydantic v2 keyword-only models.** `QualityCheck` subclasses take no positional args; new fields are annotated class-level fields, no `__init__`.
- **Artifact paths resolve through `phenotypic.sdk_` helpers / `BundleLayout`** — never hand-join `deliverables/qc/...`.
- **DuckDB connections are short-lived and `read_only=True` on the read side** — opened per query, closed immediately, never held across Dash callbacks (Windows `os.replace` + single-writer requirement).
- **Atomic writes only:** temp file in the destination dir + `os.replace` (POSIX-atomic). The CLI rebuild wraps `os.replace` in a bounded retry on `PermissionError` (Windows open-handle race).
- **GUI changes require updating `src/phenotypic/gui/FEATURES.md`** (CI `features-md-gate` rejects `gui/` PRs that don't).
- **TDD:** failing test → run-red → minimal impl → run-green → commit. Frequent commits.
- **`run_qc` stays tolerant:** a check that fails to build or analyze is skipped with a WARNING, never aborting the rest of the rebuild.
- **Curation labels + review state are NOT touched as storage** — `curation_labels.parquet` and `review_state.json` remain separate files; only their QC-join code paths migrate.

---

## File Structure

**New files:**
- `src/phenotypic/analysis/abc_/_qc_table_spec.py` — `QcTableSpec` value type (catalog descriptor produced by `QualityCheck.table_spec()`).
- `src/phenotypic/gui/results_viewer/_qc_tab/review/_db.py` — catalog-driven DuckDB read API (`open_qc_db`, `QcModule`, `list_modules`, `module_summary`, `module_members`, `summary_stats`).
- `tests/unit/qc/test_qc_table_spec.py`, `tests/unit/qc/test_qc_duckdb_writer.py`, `tests/unit/gui/results_viewer/test_qc_db_api.py`.

**Modified files:**
- `src/phenotypic/sdk_/_io_constants.py` — add `QC_DUCKDB` + `qc_duckdb_path` + `BundleLayout.qc_duckdb`; later remove the 3 parquet constants, 3 path helpers, 3 layout accessors.
- `src/phenotypic/sdk_/__init__.py` — export `QC_DUCKDB` / `qc_duckdb_path`; later drop the removed symbols from `__all__`.
- `src/phenotypic/analysis/abc_/_quality_check.py` — `to_table()`, `table_spec()`, `supports_object_curation`, `member_key_cols` defaults.
- `src/phenotypic/analysis/qc/_grid_occupancy.py` — diagnostic-only override.
- `src/phenotypic/sdk_/_qc_recipe/_runner.py` — DuckDB writer.
- `src/phenotypic/gui/results_viewer/_qc_tab/review/_data.py` — delete flat-schema helpers; keep `build_recompute_frame`.
- `src/phenotypic/gui/results_viewer/_qc_tab/review/_callbacks.py` — read via `_db.py`.
- `src/phenotypic/gui/results_viewer/_qc_tab/review/_review_state.py` — add `reconcile_to_summary`.
- `src/phenotypic/gui/results_viewer/_qc_tab/_callbacks.py`, `_ids.py` — settings-edit durable recompute; remove Export button.
- `src/phenotypic/gui/results_viewer/_error_tab/_data.py` — rewire `verified_good_keys`.
- `src/phenotypic/gui/FEATURES.md`, root `CLAUDE.md`, `src/phenotypic/gui/CLAUDE.md` — docs.
- Tests listed per task.

**Dependency ordering:** add DB path + contract (T1–T4) → sanitizer + writer (T5–T6) → read API (T7) → rewire Review (T8) → recompute + reconcile (T9) → Error tab (T10) → remove Export (T11) → remove old public symbols + docs (T12) → regression (T13). Old parquet symbols stay alive until every consumer is migrated (T12), so the suite stays green throughout.

---

### Task 1: Add the DuckDB dependency

**Files:**
- Modify: `pyproject.toml` (core `dependencies`)
- Modify: `uv.lock` (regenerated)
- Test: `tests/unit/qc/test_qc_duckdb_writer.py` (new — import smoke only this task)

**Interfaces:**
- Produces: `import duckdb` available process-wide.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/qc/test_qc_duckdb_writer.py`:

```python
"""Tests for the DuckDB-backed QC artifact writer."""


def test_duckdb_importable():
    import duckdb

    con = duckdb.connect(":memory:")
    assert con.execute("SELECT 42").fetchone()[0] == 42
    con.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/qc/test_qc_duckdb_writer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'duckdb'`.

- [ ] **Step 3: Add the dependency**

Run: `uv add duckdb`
This adds `duckdb>=1.0` (or current) to `[project].dependencies` in `pyproject.toml` and updates `uv.lock`. DuckDB ships cross-platform wheels (macOS/Windows/Linux), so no Windows try/except is needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/qc/test_qc_duckdb_writer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/unit/qc/test_qc_duckdb_writer.py
git commit -m "build(deps): add duckdb for the QC artifact store"
```

---

### Task 2: Add the `qc.duckdb` path constant, helper, and layout accessor

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py` (near `QC_SUMMARY_PARQUET` ~line 287; `qc_summary_parquet_path` ~1325; `BundleLayout.qc_summary_parquet` ~1812)
- Modify: `src/phenotypic/sdk_/__init__.py` (exports + `__all__`)
- Test: `tests/unit/sdk_/test_bundle_layout.py`

**Interfaces:**
- Produces: `QC_DUCKDB: Final[str] = "qc.duckdb"`; `qc_duckdb_path(output_dir: Path) -> Path`; `BundleLayout.qc_duckdb -> Path`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/sdk_/test_bundle_layout.py`:

```python
def test_qc_duckdb_path_under_deliverables_qc(tmp_path):
    from phenotypic.sdk_ import qc_duckdb_path, qc_dir

    assert qc_duckdb_path(tmp_path) == qc_dir(tmp_path) / "qc.duckdb"


def test_bundle_layout_qc_duckdb_accessor(tmp_path):
    from phenotypic.sdk_ import BundleLayout

    layout = BundleLayout.detect(tmp_path)
    assert layout.qc_duckdb == layout.qc_dir / "qc.duckdb"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/sdk_/test_bundle_layout.py -k qc_duckdb -v`
Expected: FAIL — `ImportError: cannot import name 'qc_duckdb_path'`.

- [ ] **Step 3: Implement the constant, helper, and accessor**

In `_io_constants.py`, beside `QC_CONFIG_JSON` (~line 298):

```python
#: DuckDB QC analysis database filename: ``<output>/deliverables/qc/qc.duckdb``.
#: One self-describing table per QC module + a ``qc_modules`` catalog; written
#: by ``run_qc``. Supersedes :data:`QC_SUMMARY_PARQUET`/:data:`QC_MEMBERS_PARQUET`/
#: :data:`QC_CONFIG_JSON`.
QC_DUCKDB: Final[str] = "qc.duckdb"
```

Beside `qc_config_json_path` (~line 1335):

```python
def qc_duckdb_path(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/qc/qc.duckdb``."""
    return qc_dir(output_dir) / QC_DUCKDB
```

In `class BundleLayout`, beside the `qc_config_json` property (~line 1822):

```python
    @property
    def qc_duckdb(self) -> Path:
        """Return path to ``qc/qc.duckdb`` (the QC analysis database)."""
        return self.qc_dir / QC_DUCKDB
```

In `src/phenotypic/sdk_/__init__.py`, add `QC_DUCKDB` to the constants import block and `qc_duckdb_path` to the path-helpers import block, and add both strings to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/sdk_/test_bundle_layout.py -k qc_duckdb -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/__init__.py tests/unit/sdk_/test_bundle_layout.py
git commit -m "feat(sdk_): add qc.duckdb path constant, helper, and BundleLayout accessor"
```

---

### Task 3: `QcTableSpec` value type + `QualityCheck.table_spec()` / `to_table()`

**Files:**
- Create: `src/phenotypic/analysis/abc_/_qc_table_spec.py`
- Modify: `src/phenotypic/analysis/abc_/_quality_check.py`
- Modify: `src/phenotypic/analysis/abc_/__init__.py` (export `QcTableSpec`)
- Test: `tests/unit/qc/test_qc_table_spec.py` (new)

**Interfaces:**
- Produces:
  - `QcTableSpec` — frozen dataclass with fields: `instance_id: str`, `cls_name: str`, `name: str`, `groupby_cols: list[str]`, `metric_col: str`, `status_col: str`, `flag_col: str`, `on_col: str`, `member_key_cols: list[str]`, `supports_object_curation: bool`, `time_col: str | None`, `higher_is_bad: bool`, `extra_cols: list[str]`, `warn_threshold: float`, `fail_threshold: float`.
  - `QualityCheck.supports_object_curation: ClassVar[bool] = True`
  - `QualityCheck.member_key_cols: ClassVar[tuple[str, ...]] = ("Metadata_ImageFile", "Object_Label")`
  - `QualityCheck.to_table(self) -> pd.DataFrame` (member-level default; precondition: `analyze()` ran)
  - `QualityCheck.table_spec(self, instance_id: str) -> QcTableSpec` (precondition: `analyze()` ran)
- Consumes: `QualityCheck.results()`, `metric_col()/flag_col()/status_col()`, `self.groupby`, `self.on`, `self._HIGHER_IS_BAD`, `getattr(self, "time_label", None)`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/qc/test_qc_table_spec.py`:

```python
"""Tests for the self-describing QualityCheck table contract."""

import pandas as pd

from phenotypic.analysis.qc import MaxModifiedZScore


def _frame():
    return pd.DataFrame(
        {
            "Metadata_ImageFile": ["a.png", "a.png", "b.png", "b.png"],
            "Object_Label": [1, 2, 1, 2],
            "Plate": ["P1", "P1", "P1", "P1"],
            "Metadata_Time": [0, 0, 1, 1],
            "Size_Area": [10.0, 11.0, 100.0, 9.0],
        }
    )


def test_to_table_carries_check_specific_columns():
    chk = MaxModifiedZScore(on="Size_Area", groupby=["Plate"])
    chk.analyze(_frame())
    table = chk.to_table()
    # Member-level: one row per input object, with the check's QC columns.
    assert len(table) == 4
    assert "QC_ZMax_Metric" in table.columns
    assert "QC_ZMax_Status" in table.columns
    assert "QC_ZMax_Median" in table.columns  # check-specific extra (kept!)
    assert {"Metadata_ImageFile", "Object_Label", "Plate"} <= set(table.columns)


def test_table_spec_describes_roles():
    chk = MaxModifiedZScore(on="Size_Area", groupby=["Plate"])
    chk.analyze(_frame())
    spec = chk.table_spec("qc-ZMax-deadbeef")
    assert spec.instance_id == "qc-ZMax-deadbeef"
    assert spec.cls_name == "MaxModifiedZScore"
    assert spec.groupby_cols == ["Plate"]
    assert spec.metric_col == "QC_ZMax_Metric"
    assert spec.status_col == "QC_ZMax_Status"
    assert spec.supports_object_curation is True
    assert spec.member_key_cols == ["Metadata_ImageFile", "Object_Label"]
    assert spec.time_col == "Metadata_Time"   # ZMax declares a time_label field
    assert spec.higher_is_bad is True
    assert "QC_ZMax_Median" in spec.extra_cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/qc/test_qc_table_spec.py -v`
Expected: FAIL — `AttributeError: 'MaxModifiedZScore' object has no attribute 'to_table'`.

- [ ] **Step 3: Implement `QcTableSpec` and the ABC methods**

Create `src/phenotypic/analysis/abc_/_qc_table_spec.py`:

```python
"""Self-describing catalog descriptor for a QC module's persisted table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QcTableSpec:
    """Column-role descriptor for one QC module's DuckDB table.

    Produced by :meth:`QualityCheck.table_spec` and written as one row of the
    ``qc_modules`` catalog, so any consumer can render a module generically
    without hard-coding its schema.

    Attributes:
        instance_id: The recipe entry id (``qc-<name>-<hex>``).
        cls_name: The ``QualityCheck`` subclass name.
        name: The check's short ``name`` (e.g. ``"ICC"``).
        groupby_cols: Ordered group-key column names.
        metric_col / status_col / flag_col: The generic QC column names.
        on_col: The measurement column the check operates on.
        member_key_cols: Per-object curation-key columns (``[]`` when the
            module has no per-object key).
        supports_object_curation: Whether the table's rows map to curatable
            detected objects (``False`` for diagnostic-only modules).
        time_col: Time-course facet column, or ``None``.
        higher_is_bad: The check's ``_HIGHER_IS_BAD`` direction.
        extra_cols: Check-specific columns beyond the generic trio.
        warn_threshold / fail_threshold: For status legends.
    """

    instance_id: str
    cls_name: str
    name: str
    groupby_cols: list[str]
    metric_col: str
    status_col: str
    flag_col: str
    on_col: str
    member_key_cols: list[str]
    supports_object_curation: bool
    time_col: str | None
    higher_is_bad: bool
    extra_cols: list[str]
    warn_threshold: float
    fail_threshold: float
```

In `_quality_check.py`, add the two class attributes near `name`/`_HIGHER_IS_BAD` (~line 100):

```python
    #: Whether this check's rows map to curatable detected objects. False for
    #: diagnostic-only checks (e.g. GridOccupancy) — the Review tab hides the
    #: curation radial + tile gallery and verified-good skips them.
    supports_object_curation: ClassVar[bool] = True

    #: Per-object curation-key columns. Empty tuple when the check has no
    #: per-object key. Subclasses may narrow this.
    member_key_cols: ClassVar[tuple[str, ...]] = (
        "Metadata_ImageFile",
        str(OBJECT.LABEL),
    )
```

Add the two methods (after `results()`, ~line 368). Import `QcTableSpec` at top of the module:

```python
from ._qc_table_spec import QcTableSpec
```

```python
    def to_table(self) -> pd.DataFrame:
        """Return the module's self-describing frame to persist to DuckDB.

        Precondition: :meth:`analyze` has run (this reads
        :attr:`_latest_measurements`). The default is member-level: the
        augmented frame projected to group-key + member-key + ``on`` + every
        ``QC_<name>_*`` column (metric/flag/status AND check-specific extras)
        + context columns (``Metadata_Dataset`` and the column named by
        ``self.time_label``) when those columns are present.

        Diagnostic-only checks override to return a group-level frame.

        Returns:
            The projected DataFrame; columns vary per check (self-describing).
        """
        df = self._latest_measurements
        qc_cols = [c for c in df.columns if c.startswith(f"QC_{self.name}_")]
        context = [
            c
            for c in ("Metadata_Dataset", getattr(self, "time_label", None))
            if c and c in df.columns
        ]
        keep: list[str] = []
        for col in [
            *self.groupby,
            *self.member_key_cols,
            self.on,
            *context,
            *qc_cols,
        ]:
            if col in df.columns and col not in keep:
                keep.append(col)
        return df[keep].copy()

    def table_spec(self, instance_id: str) -> QcTableSpec:
        """Return the catalog descriptor for this analyzed check.

        Precondition: :meth:`analyze` has run. Reads column roles from the
        class + instance config and derives ``extra_cols`` from the augmented
        frame.

        Args:
            instance_id: The recipe entry id this check was built from.

        Returns:
            A populated :class:`QcTableSpec`.
        """
        df = self._latest_measurements
        generic = {self.metric_col(), self.flag_col(), self.status_col()}
        extra = [
            c
            for c in df.columns
            if c.startswith(f"QC_{self.name}_") and c not in generic
        ]
        time_col = getattr(self, "time_label", None)
        return QcTableSpec(
            instance_id=instance_id,
            cls_name=type(self).__name__,
            name=self.name,
            groupby_cols=list(self.groupby),
            metric_col=self.metric_col(),
            status_col=self.status_col(),
            flag_col=self.flag_col(),
            on_col=self.on,
            member_key_cols=list(self.member_key_cols),
            supports_object_curation=self.supports_object_curation,
            time_col=time_col if (time_col and time_col in df.columns) else None,
            higher_is_bad=self._HIGHER_IS_BAD,
            extra_cols=extra,
            warn_threshold=float(self.warn_threshold),
            fail_threshold=float(self.fail_threshold),
        )
```

Export `QcTableSpec` from `src/phenotypic/analysis/abc_/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/qc/test_qc_table_spec.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/analysis/abc_/_qc_table_spec.py src/phenotypic/analysis/abc_/_quality_check.py src/phenotypic/analysis/abc_/__init__.py tests/unit/qc/test_qc_table_spec.py
git commit -m "feat(analysis): self-describing QualityCheck table contract (to_table/table_spec)"
```

---

### Task 4: `GridOccupancy` diagnostic-only override

**Files:**
- Modify: `src/phenotypic/analysis/qc/_grid_occupancy.py`
- Test: `tests/unit/qc/test_qc_table_spec.py` (extend)

**Interfaces:**
- Produces: `GridOccupancy.supports_object_curation = False`; `GridOccupancy.to_table()` returns one group-level row per group with `groupby + QC_Occupancy_Filled/Expected/Vacant/Metric/Status/Flag`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/qc/test_qc_table_spec.py`:

```python
def test_grid_occupancy_is_group_level_and_diagnostic_only():
    import pandas as pd

    from phenotypic.analysis.qc import GridOccupancy

    metadata = pd.DataFrame(
        {"Metadata_ImageFile": ["a.png"] * 4, "cell_label": [1, 2, 3, 4]}
    )
    measured = pd.DataFrame(
        {
            "Metadata_ImageFile": ["a.png", "a.png"],
            "Object_Label": [1, 2],
            "cell_label": [1, 2],
        }
    )
    # cell_label defaults to "Grid_RowMajorIdx"; point it at this frame's column.
    chk = GridOccupancy(
        metadata=metadata, groupby=["Metadata_ImageFile"], cell_label="cell_label"
    )
    chk.analyze(measured)
    spec = chk.table_spec("qc-Occupancy-cafef00d")
    assert spec.supports_object_curation is False

    table = chk.to_table()
    # Group-level: one row per group (here one image), not per colony.
    assert len(table) == 1
    assert {"QC_Occupancy_Filled", "QC_Occupancy_Expected"} <= set(table.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/qc/test_qc_table_spec.py -k grid_occupancy -v`
Expected: FAIL — `assert spec.supports_object_curation is False` (defaults True) and/or row count (default `to_table` is per-colony → 2 rows).

- [ ] **Step 3: Implement the override**

In `_grid_occupancy.py`, set the class attribute beside `name`/`_HIGHER_IS_BAD` (~line 114):

```python
    supports_object_curation: ClassVar[bool] = False
```

Add a `to_table` override (after `_compute`, reusing the base `summary()` for group-level rows + the occupancy columns):

```python
    def to_table(self) -> pd.DataFrame:
        """Return one group-level row per group (occupancy is per-plate).

        Occupancy reports filled-vs-expected counts broadcast across a
        group's rows, so per-colony rows carry no extra signal. Collapse to
        one row per group: the base ``summary()`` (renamed to the generic
        QC columns) plus the occupancy-specific counts.

        Returns:
            A group-level frame: ``[*groupby, QC_Occupancy_Filled,
            QC_Occupancy_Expected, QC_Occupancy_Vacant, QC_Occupancy_Metric,
            QC_Occupancy_Status, QC_Occupancy_Flag]``.
        """
        df = self._latest_measurements
        occ_cols = [
            "QC_Occupancy_Filled",
            "QC_Occupancy_Expected",
            "QC_Occupancy_Vacant",
        ]
        first = (
            df.groupby(self.groupby, dropna=False)[
                [c for c in occ_cols if c in df.columns]
            ]
            .first()
            .reset_index()
        )
        summary = self.summary().rename(
            columns={
                "qc_worst_metric": self.metric_col(),
                "qc_status": self.status_col(),
            }
        )
        merged = first.merge(
            summary[[*self.groupby, self.metric_col(), self.status_col()]],
            on=list(self.groupby),
            how="left",
        )
        # Group-level flag: any member flagged → fail-status drives the flag.
        merged[self.flag_col()] = merged[self.status_col()] == "fail"
        return merged
```

> NOTE: confirm against `_grid_occupancy.py:148-179` that `_compute` emits `QC_Occupancy_Filled/Expected/Vacant`. If `cell_label` handling means a column name differs, adjust `occ_cols`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/qc/test_qc_table_spec.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/analysis/qc/_grid_occupancy.py tests/unit/qc/test_qc_table_spec.py
git commit -m "feat(analysis): GridOccupancy is diagnostic-only, group-level to_table"
```

---

### Task 5: SQL-safe table-name sanitizer

**Files:**
- Modify: `src/phenotypic/sdk_/_qc_recipe/_runner.py` (add helper)
- Test: `tests/unit/qc/test_qc_duckdb_writer.py` (extend)

**Interfaces:**
- Produces: `_safe_table_name(instance_id: str) -> str` — deterministic, valid DuckDB identifier (starts with a letter, only `[a-z0-9_]`), collision-resistant.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/qc/test_qc_duckdb_writer.py`:

```python
def test_safe_table_name_is_deterministic_and_identifier_safe():
    from phenotypic.sdk_._qc_recipe._runner import _safe_table_name

    name = _safe_table_name("qc-SE-1a2b3c4d")
    assert name == _safe_table_name("qc-SE-1a2b3c4d")  # deterministic
    assert name[0].isalpha()
    assert all(c.isalnum() or c == "_" for c in name)
    # Distinct ids → distinct names.
    assert _safe_table_name("qc-SE-1a2b3c4d") != _safe_table_name("qc-SE-99999999")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/qc/test_qc_duckdb_writer.py -k safe_table_name -v`
Expected: FAIL — `ImportError: cannot import name '_safe_table_name'`.

- [ ] **Step 3: Implement the sanitizer**

In `_runner.py`:

```python
import re

_IDENT_UNSAFE = re.compile(r"[^a-z0-9]+")


def _safe_table_name(instance_id: str) -> str:
    """Return a deterministic, DuckDB-identifier-safe table name.

    Lowercases, replaces any run of non-alphanumerics with ``_``, strips
    leading/trailing ``_``, and prefixes ``qc_`` so the result always starts
    with a letter. Instance ids are already unique (``qc-<name>-<8hex>``), so
    the lowercased, underscore-joined form stays unique.

    Args:
        instance_id: The recipe entry id.

    Returns:
        A valid table identifier (e.g. ``qc_se_1a2b3c4d``).
    """
    core = _IDENT_UNSAFE.sub("_", instance_id.lower()).strip("_")
    return core if core.startswith("qc_") else f"qc_{core}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/qc/test_qc_duckdb_writer.py -k safe_table_name -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_qc_recipe/_runner.py tests/unit/qc/test_qc_duckdb_writer.py
git commit -m "feat(sdk_): SQL-safe table-name sanitizer for the QC writer"
```

---

### Task 6: Rewrite `run_qc` as the DuckDB writer

**Files:**
- Modify: `src/phenotypic/sdk_/_qc_recipe/_runner.py` (rewrite the writer body; keep `_run_one_check`, `_rank_worst_first`, `_safe_table_name`)
- Test: `tests/unit/qc/test_qc_duckdb_writer.py` (extend), rewrite `tests/unit/qc/test_run_qc.py`

**Interfaces:**
- Produces: `run_qc(measurements_df, pipeline, output_dir, *, qc_output_dir=None) -> None` writes `qc.duckdb` containing `qc_modules` (catalog) + `<table_name>` (data) + `<table_name>__summary` (worklist) per enabled module. No-op when there are no enabled entries.
- Consumes: `check.to_table()`, `check.table_spec(instance_id)`, `check.summary()`, `_rank_worst_first`, `_safe_table_name`, `qc_duckdb_path` / `qc_output_dir`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/qc/test_qc_duckdb_writer.py` (build a tiny 2-check pipeline; reuse `load_synth_yeast_plate`-free synthetic frames):

```python
def _two_check_pipeline():
    from phenotypic import ImagePipeline
    from phenotypic.analysis.qc import MaxModifiedZScore, RelativeMAD

    pipe = ImagePipeline()
    pipe.set_qc(
        [
            _entry(MaxModifiedZScore, {"on": "Size_Area", "groupby": ["Plate"]}),
            _entry(RelativeMAD, {"on": "Size_Area", "groupby": ["Plate"]}),
        ]
    )
    return pipe


def _entry(cls, params):
    from phenotypic.sdk_._qc_recipe import QcRecipeEntry

    name = cls.name
    return QcRecipeEntry(
        cls=cls, params=params, instance_id=f"qc-{name}-00000001", enabled=True
    )


def _frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "Metadata_ImageFile": ["a.png"] * 6,
            "Object_Label": [1, 2, 3, 4, 5, 6],
            "Plate": ["P1"] * 6,
            "Size_Area": [10.0, 11.0, 12.0, 10.5, 11.5, 200.0],
        }
    )


def test_run_qc_writes_per_module_tables_and_catalog(tmp_path):
    import duckdb

    from phenotypic.sdk_ import qc_duckdb_path
    from phenotypic.sdk_._qc_recipe._runner import run_qc

    # NOTE: instance_ids must be unique per module; fix the second one.
    pipe = _two_check_pipeline()
    qc = pipe.get_qc()
    qc[1] = qc[1].__class__(
        cls=qc[1].cls, params=qc[1].params,
        instance_id="qc-MAD-00000002", enabled=True,
    )
    pipe.set_qc(qc)

    run_qc(_frame(), pipe, tmp_path)

    db = qc_duckdb_path(tmp_path)
    assert db.is_file()
    con = duckdb.connect(str(db), read_only=True)
    try:
        cat = con.execute("SELECT instance_id, table_name, summary_table, "
                          "supports_object_curation FROM qc_modules ORDER BY ordinal").fetchall()
        assert [r[0] for r in cat] == ["qc-ZMax-00000001", "qc-MAD-00000002"]
        # Each module's data + summary tables exist and the metric column is kept.
        for _iid, tname, stname, _curation in cat:
            cols = [c[0] for c in con.execute(f'DESCRIBE "{tname}"').fetchall()]
            assert any(c.startswith("QC_") and c.endswith("_Metric") for c in cols)
            scols = [c[0] for c in con.execute(f'DESCRIBE "{stname}"').fetchall()]
            assert {"metric", "status", "rank", "n_members", "n_flagged"} <= set(scols)
    finally:
        con.close()


def test_run_qc_no_enabled_checks_is_noop(tmp_path):
    from phenotypic import ImagePipeline
    from phenotypic.sdk_ import qc_duckdb_path
    from phenotypic.sdk_._qc_recipe._runner import run_qc

    run_qc(_frame(), ImagePipeline(), tmp_path)
    assert not qc_duckdb_path(tmp_path).exists()


def test_run_qc_all_disabled_is_noop(tmp_path):
    from phenotypic import ImagePipeline
    from phenotypic.sdk_ import qc_duckdb_path
    from phenotypic.sdk_._qc_recipe import QcRecipeEntry
    from phenotypic.sdk_._qc_recipe._runner import run_qc
    from phenotypic.analysis.qc import MaxModifiedZScore

    pipe = ImagePipeline()
    pipe.set_qc([QcRecipeEntry(cls=MaxModifiedZScore,
                               params={"on": "Size_Area", "groupby": ["Plate"]},
                               instance_id="qc-ZMax-00000001", enabled=False)])
    run_qc(_frame(), pipe, tmp_path)
    assert not qc_duckdb_path(tmp_path).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/qc/test_qc_duckdb_writer.py -k run_qc -v`
Expected: FAIL — `run_qc` still writes parquet; no `qc.duckdb`.

- [ ] **Step 3: Rewrite the writer**

Replace the `run_qc` body and the parquet helpers (`_build_member_rows`, `_write_parquet`, `_write_config`, `_concat_or_empty`, `_summary_empty`, `_members_empty`, and the `_SUMMARY_*`/`_MEMBERS_*` col constants) with the DuckDB writer. Keep `_run_one_check` (retitle to return `(spec, table_df, summary_df)`), `_rank_worst_first`, `_safe_table_name`. New core:

```python
import os
import duckdb

from phenotypic.sdk_ import qc_duckdb_path


def run_qc(measurements_df, pipeline, output_dir, *, qc_output_dir=None):
    """Run enabled QC checks and atomically (re)build ``qc.duckdb``.

    Always a FULL rebuild: a temp DB is built then ``os.replace``-d over the
    canonical path so readers never see a partial DB. No-op when the pipeline
    has no enabled QC entries. Never touches ``review_state.json`` or
    ``measurements.parquet``.
    """
    entries = [e for e in pipeline.get_qc() if e.enabled]
    if not entries:
        logger.debug("No enabled QC entries; skipping run_qc")
        return

    output_dir = Path(output_dir)
    if qc_output_dir is not None:
        target = Path(qc_output_dir) / "qc.duckdb"
    else:
        target = qc_duckdb_path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    # Thread `entry` alongside the built frames so the catalog row carries the
    # real params snapshot (QcTableSpec does not hold params).
    built: list[tuple[Any, Any, pd.DataFrame, pd.DataFrame]] = []
    for entry in entries:
        result = _run_one_check(entry, measurements_df)
        if result is not None:
            spec, table_df, summary_df = result
            built.append((entry, spec, table_df, summary_df))

    if not built:
        logger.info("No QC check produced a table; not writing qc.duckdb")
        return

    con = duckdb.connect(str(tmp))
    try:
        _create_catalog(con)
        for ordinal, (entry, spec, table_df, summary_df) in enumerate(built):
            tname = _safe_table_name(spec.instance_id)
            stname = f"{tname}__summary"
            con.register("_qc_data", table_df)
            con.execute(f'CREATE TABLE "{tname}" AS SELECT * FROM _qc_data')
            con.unregister("_qc_data")
            con.register("_qc_summary", summary_df)
            con.execute(f'CREATE TABLE "{stname}" AS SELECT * FROM _qc_summary')
            con.unregister("_qc_summary")
            _insert_catalog_row(
                con, spec, tname, stname, ordinal,
                params=entry.to_dict()["params"],
            )
    finally:
        con.close()

    _atomic_replace_with_retry(tmp, target)
    logger.info("Wrote QC DuckDB for %d module(s) -> %s", len(built), target)
```

Add the catalog DDL/insert, the `_run_one_check` rewrite (now also building the summary frame), and the Windows-safe replace:

```python
_CATALOG_DDL = """
CREATE TABLE qc_modules (
    instance_id TEXT, class TEXT, name TEXT,
    table_name TEXT, summary_table TEXT, ordinal INTEGER,
    groupby_cols TEXT, metric_col TEXT, status_col TEXT, flag_col TEXT,
    on_col TEXT, member_key_cols TEXT, supports_object_curation BOOLEAN,
    time_col TEXT, higher_is_bad BOOLEAN, extra_cols TEXT, params TEXT,
    warn_threshold DOUBLE, fail_threshold DOUBLE
)
"""


def _create_catalog(con) -> None:
    con.execute(_CATALOG_DDL)


def _insert_catalog_row(con, spec, tname, stname, ordinal, params) -> None:
    con.execute(
        "INSERT INTO qc_modules VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            spec.instance_id, spec.cls_name, spec.name, tname, stname, ordinal,
            json.dumps(spec.groupby_cols), spec.metric_col, spec.status_col,
            spec.flag_col, spec.on_col, json.dumps(spec.member_key_cols),
            spec.supports_object_curation, spec.time_col, spec.higher_is_bad,
            json.dumps(spec.extra_cols), json.dumps(params),  # real params snapshot
            spec.warn_threshold, spec.fail_threshold,
        ],
    )


def _atomic_replace_with_retry(tmp: Path, target: Path, attempts: int = 5) -> None:
    """``os.replace`` with a bounded retry on Windows PermissionError."""
    for i in range(attempts):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            # A reader handle overlaps the swap (Windows). Brief backoff.
            import time
            time.sleep(0.1 * (i + 1))
```

Rewrite `_run_one_check` to return `(spec, table_df, summary_df)`:

```python
def _run_one_check(entry, measurements_df):
    try:
        check = entry.instantiate()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping QC %s: instantiation failed: %s", entry.instance_id, exc)
        return None
    try:
        check.analyze(measurements_df)
        table_df = check.to_table()
        spec = check.table_spec(entry.instance_id)
        summary_df = _build_summary_frame(check)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping QC %s: analyze/build failed: %s", entry.instance_id, exc)
        return None
    return spec, table_df, summary_df
```

Add `_build_summary_frame` (group-level worklist, reusing `_rank_worst_first` — this is `_build_summary_rows` minus the instance/class columns, which now live in the catalog):

```python
def _build_summary_frame(check) -> pd.DataFrame:
    raw = check.summary()
    out = raw.rename(
        columns={
            "qc_worst_metric": "metric", "qc_status": "status",
            "qc_n_members": "n_members", "qc_n_flagged": "n_flagged",
        }
    )
    out["flag"] = out["n_flagged"].astype(int) > 0
    out["rank"] = _rank_worst_first(
        out["metric"], out["status"], higher_is_bad=check._HIGHER_IS_BAD
    )
    return out
```

The params snapshot is threaded via `built` (each tuple carries `entry`), so `_insert_catalog_row` receives `entry.to_dict()["params"]` — no separate plumbing step.

> **Deletions in `_runner.py`:** the `_build_summary_rows`/`_build_member_rows`/`_write_parquet`/`_write_config`/`_concat_or_empty`/`_summary_empty`/`_members_empty` functions and the `_SUMMARY_*`/`_MEMBERS_*` constants are deleted. The lazy `from phenotypic._cli._cli_output_manager import _atomic_write` imports are removed. **Also remove the now-unused top-of-file imports** `QC_SUMMARY_PARQUET`, `QC_MEMBERS_PARQUET`, `QC_CONFIG_JSON`, `qc_summary_parquet_path`, `qc_members_parquet_path`, `qc_config_json_path` (so T12's removal of those symbols doesn't find a survivor here).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/qc/test_qc_duckdb_writer.py -v`
Expected: PASS.

Then rewrite `tests/unit/qc/test_run_qc.py` to assert DuckDB behavior (catalog rows, per-module tables, tolerant-skip on a check that raises, empty-pipeline no-op) instead of parquet, and run it:
Run: `uv run pytest tests/unit/qc/test_run_qc.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_qc_recipe/_runner.py tests/unit/qc/test_qc_duckdb_writer.py tests/unit/qc/test_run_qc.py
git commit -m "feat(sdk_): run_qc writes per-module DuckDB tables + catalog (atomic rebuild)"
```

---

### Task 7: Catalog-driven GUI read API (`_db.py`)

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_qc_tab/review/_db.py`
- Test: `tests/unit/gui/results_viewer/test_qc_db_api.py` (new)

**Interfaces:**
- Produces:
  - `QcModule` — frozen dataclass mirroring a `qc_modules` row (`instance_id`, `cls_name`, `name`, `table_name`, `summary_table`, `groupby_cols: list[str]`, `metric_col`, `status_col`, `flag_col`, `member_key_cols: list[str]`, `supports_object_curation: bool`, `time_col: str | None`, `higher_is_bad: bool`, `extra_cols: list[str]`).
  - `open_qc_db(output_root) -> duckdb connection | None` (short-lived, `read_only=True`).
  - `list_modules(output_root) -> list[QcModule]` (ordered by `ordinal`).
  - `module_summary(output_root, instance_id) -> pl.DataFrame` (worst-first by `rank`).
  - `module_members(output_root, instance_id, group_key: tuple) -> pl.DataFrame`.
  - `summary_stats(module_summary_df) -> dict` (ported verbatim from `_data.summary_stats`).
- Consumes: `output_root.layout.qc_duckdb`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/gui/results_viewer/test_qc_db_api.py` (build a DB via `run_qc` against a tmp dir, then read it; use a tiny fake `output_root` exposing `.layout`):

```python
import pandas as pd

from phenotypic import ImagePipeline
from phenotypic.sdk_ import BundleLayout
from phenotypic.sdk_._qc_recipe import QcRecipeEntry
from phenotypic.sdk_._qc_recipe._runner import run_qc
from phenotypic.analysis.qc import MaxModifiedZScore


class _Root:
    def __init__(self, layout):
        self.layout = layout


def _seed_db(tmp_path):
    pipe = ImagePipeline()
    pipe.set_qc([QcRecipeEntry(cls=MaxModifiedZScore,
                               params={"on": "Size_Area", "groupby": ["Plate"]},
                               instance_id="qc-ZMax-00000001", enabled=True)])
    df = pd.DataFrame({
        "Metadata_ImageFile": ["a.png"] * 4, "Object_Label": [1, 2, 3, 4],
        "Plate": ["P1"] * 4, "Size_Area": [10.0, 11.0, 12.0, 99.0],
    })
    run_qc(df, pipe, tmp_path)
    return _Root(BundleLayout.detect(tmp_path))


def test_list_modules_reads_catalog(tmp_path):
    from phenotypic.gui.results_viewer._qc_tab.review import _db

    root = _seed_db(tmp_path)
    mods = _db.list_modules(root)
    assert [m.instance_id for m in mods] == ["qc-ZMax-00000001"]
    assert mods[0].groupby_cols == ["Plate"]
    assert mods[0].supports_object_curation is True


def test_module_summary_and_members(tmp_path):
    from phenotypic.gui.results_viewer._qc_tab.review import _db

    root = _seed_db(tmp_path)
    summ = _db.module_summary(root, "qc-ZMax-00000001")
    assert "rank" in summ.columns and summ.height == 1
    members = _db.module_members(root, "qc-ZMax-00000001", ("P1",))
    assert members.height == 4


def test_open_qc_db_missing_returns_none(tmp_path):
    from phenotypic.gui.results_viewer._qc_tab.review import _db

    assert _db.open_qc_db(_Root(BundleLayout.detect(tmp_path))) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/gui/results_viewer/test_qc_db_api.py -v`
Expected: FAIL — module `_db` does not exist.

- [ ] **Step 3: Implement `_db.py`**

```python
"""Catalog-driven DuckDB read API for the QC Review + Error tabs.

Connections are short-lived and ``read_only`` — opened per query and closed
immediately, never held across Dash callbacks (Windows os.replace + single
DuckDB writer). Returns polars frames / plain dataclasses; Dash-free.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import duckdb
import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QcModule:
    instance_id: str
    cls_name: str
    name: str
    table_name: str
    summary_table: str
    groupby_cols: list[str]
    metric_col: str
    status_col: str
    flag_col: str
    member_key_cols: list[str]
    supports_object_curation: bool
    time_col: str | None
    higher_is_bad: bool
    extra_cols: list[str]


def open_qc_db(output_root: "OutputRoot"):
    """Open a short-lived read_only connection, or None when absent/corrupt."""
    path = output_root.layout.qc_duckdb
    if not path.is_file():
        return None
    try:
        return duckdb.connect(str(path), read_only=True)
    except Exception:  # noqa: BLE001 - a corrupt/locked DB is non-fatal
        logger.warning("Failed to open QC DuckDB %s", path, exc_info=True)
        return None


def list_modules(output_root: "OutputRoot") -> list[QcModule]:
    con = open_qc_db(output_root)
    if con is None:
        return []
    try:
        rows = con.execute(
            "SELECT instance_id, class, name, table_name, summary_table, "
            "groupby_cols, metric_col, status_col, flag_col, member_key_cols, "
            "supports_object_curation, time_col, higher_is_bad, extra_cols "
            "FROM qc_modules ORDER BY ordinal"
        ).fetchall()
    finally:
        con.close()
    return [
        QcModule(
            instance_id=r[0], cls_name=r[1], name=r[2], table_name=r[3],
            summary_table=r[4], groupby_cols=json.loads(r[5]), metric_col=r[6],
            status_col=r[7], flag_col=r[8], member_key_cols=json.loads(r[9]),
            supports_object_curation=bool(r[10]), time_col=r[11],
            higher_is_bad=bool(r[12]), extra_cols=json.loads(r[13]),
        )
        for r in rows
    ]


def _module(output_root, instance_id) -> QcModule | None:
    return next(
        (m for m in list_modules(output_root) if m.instance_id == instance_id),
        None,
    )


def module_summary(output_root, instance_id) -> pl.DataFrame:
    mod = _module(output_root, instance_id)
    con = open_qc_db(output_root)
    if mod is None or con is None:
        return pl.DataFrame()
    try:
        return con.execute(
            f'SELECT * FROM "{mod.summary_table}" ORDER BY rank NULLS LAST'
        ).pl()
    finally:
        con.close()


def module_members(output_root, instance_id, group_key: tuple) -> pl.DataFrame:
    mod = _module(output_root, instance_id)
    con = open_qc_db(output_root)
    if mod is None or con is None:
        return pl.DataFrame()
    try:
        frame = con.execute(f'SELECT * FROM "{mod.table_name}"').pl()
    finally:
        con.close()
    # Filter in polars (avoids dynamic SQL on column names); null/NaN-safe.
    for col, val in zip(mod.groupby_cols, group_key):
        if col not in frame.columns:
            continue
        if val is None or (isinstance(val, float) and math.isnan(val)):
            frame = frame.filter(pl.col(col).is_null())
        else:
            frame = frame.filter(pl.col(col).cast(pl.String) == str(val))
    return frame


def summary_stats(module_summary_df: pl.DataFrame) -> dict[str, Any]:
    """Counts (fail/warn/pass/insufficient) + robust median. Ported verbatim
    from the old _data.summary_stats so the header tiles are unchanged."""
    # ... (copy the exact body of _data.summary_stats here) ...
```

Copy the `summary_stats` + `_robust_median` bodies verbatim from `_data.py:211-286`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/gui/results_viewer/test_qc_db_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_qc_tab/review/_db.py tests/unit/gui/results_viewer/test_qc_db_api.py
git commit -m "feat(gui): catalog-driven DuckDB read API for the QC tabs"
```

---

### Task 8: Rewire the Review tab onto `_db.py`; delete flat-schema helpers

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_qc_tab/review/_callbacks.py` (all `_data.load_qc_summary`/`load_qc_members`/`module_options`/`module_worklist`/`group_member_keys`/`group_record`/`groupby_cols_for`/`facet_keys_by_timepoint` call sites)
- Modify: `src/phenotypic/gui/results_viewer/_qc_tab/review/_data.py` (delete the flat-schema helpers + constants; KEEP `build_recompute_frame`, `_anti_join_removed`, and the `_KEY_*` constants)
- Test: rewrite `tests/unit/gui/results_viewer/test_qc_review_data.py`; update `tests/unit/gui/results_viewer/test_qc_review_layout.py`

**Interfaces:**
- Consumes: `_db.list_modules`, `_db.module_summary`, `_db.module_members`, `_db.summary_stats`, `_db.QcModule`.
- Produces: Review callbacks render module picker (from `list_modules`), worklist (from `module_summary`), detail/gallery (from `module_members`, faceted by `QcModule.time_col`). Diagnostic-only modules (`supports_object_curation == False`) hide the curation radial + tile gallery.

- [ ] **Step 1: Write the failing test**

Rewrite `tests/unit/gui/results_viewer/test_qc_review_data.py` to target the new seam. Example replacing the old `module_options`/`groupby_cols_for` tests:

```python
def test_module_picker_options_from_catalog(tmp_path):
    # seed a DB (reuse the _seed_db helper pattern from test_qc_db_api), then:
    from phenotypic.gui.results_viewer._qc_tab.review import _db

    mods = _db.list_modules(root)
    options = [{"label": f"{m.cls_name} ({m.instance_id.rsplit('-', 1)[-1]})",
                "value": m.instance_id} for m in mods]
    assert options[0]["value"] == "qc-ZMax-00000001"
```

(Keep `build_recompute_frame` tests as-is — that function is unchanged.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/gui/results_viewer/test_qc_review_data.py -v`
Expected: FAIL (new helpers not wired) — or import errors where deleted symbols are still referenced.

- [ ] **Step 3: Rewire callbacks and delete helpers**

In `_callbacks.py`, replace each call site:
- `_data.load_qc_summary(output_root)` + `module_options(summary)` → `_db.list_modules(output_root)` (build picker options inline as above).
- `module_worklist(summary, iid)` → `_db.module_summary(output_root, iid)`.
- `_data.load_qc_members(...)` + `group_member_keys(...)` → `_db.module_members(output_root, iid, key)`; resolve tiles directly from the member frame's `Metadata_ImageFile`/`Object_Label`/`Metadata_Dataset` columns (the table now carries dataset/time context, so drop the `dataset_by_image_map`/`time_by_key_map` calls).
- `_data.summary_stats(_data.module_worklist(summary, iid))` (~line 845) → `_db.summary_stats(_db.module_summary(output_root, iid))`.
- `_data.group_record(summary, iid, cols, key)` (~lines 662, 675, 928) → filter `_db.module_summary(output_root, iid)` to the group key and take the first matching row as a dict (the worklist summary carries the per-group columns the header reads).
- `_recompute_after_curation` (~line 729): replace `_data.load_qc_summary(output_root)` with `_db.module_summary(output_root, instance_id)` and read the new metric/status from it.
- For diagnostic-only modules: branch on `module.supports_object_curation` (from `_db.list_modules`) to hide the radial + gallery.

> Grep `_data\.` in `review/_callbacks.py` after the rewrite — only `_data.build_recompute_frame` should remain. Any other `_data.<helper>` call is a missed site (it will surface as `AttributeError` at test time).

In `_data.py`, DELETE the Review-only helpers now that the callbacks no longer use them: `_read_optional_parquet`, `module_options`, `_short_id`, `module_worklist`, `summary_stats`, `_robust_median`, `group_member_keys`, `group_record`, `facet_keys_by_timepoint`, `_time_sort_token`, `dataset_by_image_map`, `time_by_key_map`, and the `_SUMMARY_LEAD/_TAIL`, `_MEMBERS_LEAD/_TAIL` constants.

> **CRITICAL — do NOT delete these four yet:** `load_qc_summary`, `load_qc_members`, `groupby_cols_for`, `_eq_or_null`. `_error_tab/_data.py:31-35` still imports all four until it is rewired in **T10**; deleting them now turns the suite red (ImportError) for a whole task. They are removed at the end of T10, after the Error tab no longer needs them. Also KEEP `build_recompute_frame`, `_anti_join_removed`, `_KEY_*`. Update `__all__` to drop only the deleted names.

> `summary_stats`/`_robust_median` already live in `_db.py` (copied in T7). The T10 Error-tab rewrite does not need `_eq_or_null` (it filters via `_db.module_members`), so the final four are removed in T10.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/gui/results_viewer/test_qc_review_data.py tests/unit/gui/results_viewer/test_qc_review_layout.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_qc_tab/review/_callbacks.py src/phenotypic/gui/results_viewer/_qc_tab/review/_data.py tests/unit/gui/results_viewer/test_qc_review_data.py tests/unit/gui/results_viewer/test_qc_review_layout.py
git commit -m "refactor(gui): QC Review tab reads the DuckDB catalog; drop flat-schema helpers"
```

---

### Task 9: Durable settings-edit recompute + review_state reconciliation

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_qc_tab/_callbacks.py` (add a recompute on `STORE_QC_RECIPE_REVISION`; **add `CFG_QC_PIPELINE` to the `_config` import block** — this module currently imports only `CFG_QC_RECIPE`)
- Modify: `src/phenotypic/gui/results_viewer/_qc_tab/_ids.py` (define `STORE_QC_RECOMPUTE_DONE`)
- Modify: `src/phenotypic/gui/results_viewer/_qc_tab/_layout.py` (mount a `dcc.Store(id=STORE_QC_RECOMPUTE_DONE)` so the worklist can subscribe to it)
- Modify: `src/phenotypic/gui/results_viewer/_qc_tab/review/_review_state.py` (add `reconcile_to_summary`)
- Modify: `src/phenotypic/gui/results_viewer/_qc_tab/review/_callbacks.py` (call reconcile after recompute)
- Test: `tests/integration/gui/test_qc_review_recompute.py` (extend)

**Interfaces:**
- Produces: editing/saving a QC module persists via `QcRecipe` then runs `run_qc` (full rebuild) against the curated post-applied frame, so the worklist + DB reflect the new settings. `ReviewState.reconcile_to_summary(instance_id, present_keys: set[str]) -> None` drops reviewed encoded keys whose group no longer exists.

- [ ] **Step 1: Write the failing test**

Extend `tests/integration/gui/test_qc_review_recompute.py`:

```python
def test_settings_edit_durably_rewrites_db(tmp_path):
    # Build a run output with a QC module + a written qc.duckdb; capture the
    # module's fail_threshold. Simulate a settings edit (QcRecipe.update to a
    # threshold that flips a group's status), trigger the recompute path, then
    # assert _db.module_summary reflects the new status (not the stale one).
    ...


def test_reconcile_drops_vanished_reviewed_keys(tmp_path):
    from phenotypic.gui.results_viewer._qc_tab.review._review_state import (
        ReviewState, encode_group_key,
    )

    state = ReviewState(path=tmp_path / "review_state.json")
    state.mark_reviewed("qc-ZMax-1", ("P1",))
    state.mark_reviewed("qc-ZMax-1", ("P_GONE",))
    state.reconcile_to_summary("qc-ZMax-1", {encode_group_key(("P1",))})
    assert state.is_reviewed("qc-ZMax-1", ("P1",))
    assert not state.is_reviewed("qc-ZMax-1", ("P_GONE",))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/gui/test_qc_review_recompute.py -k "settings_edit or reconcile" -v`
Expected: FAIL — no settings-edit recompute; `reconcile_to_summary` missing.

- [ ] **Step 3: Implement**

In `_review_state.py`:

```python
    def reconcile_to_summary(self, instance_id: str, present_keys: set[str]) -> None:
        """Drop reviewed encoded keys whose group no longer exists.

        After a recompute the module's group keys may change (settings edit
        altered groupby/thresholds). Prune reviewed keys not in
        ``present_keys`` (encoded). Persists when anything changed.

        Args:
            instance_id: The recomputed module.
            present_keys: Encoded group keys present in the new summary.
        """
        progress = self.modules.get(instance_id)
        if progress is None:
            return
        stale = progress.reviewed - present_keys
        if stale:
            progress.reviewed -= stale
            if progress.last in stale:
                progress.last = None
            self.save()
```

In `_qc_tab/_callbacks.py`, add a callback that fires on `STORE_QC_RECIPE_REVISION` and runs the full-rebuild recompute (reuse the helper from `review/_callbacks.py`). Extend the existing `_recompute_after_curation` into a shared `_recompute_full_rebuild(output_root, pipeline, removed)` used by BOTH the curation path and the new settings-edit callback. After the rebuild, for each module call `state.reconcile_to_summary(iid, present_encoded_keys)` where the present keys come from `_db.module_summary(output_root, iid)` (encode each row's group-key tuple via `encode_group_key`).

**Sync the pipeline before `run_qc` (REQUIRED).** `CFG_QC_PIPELINE` is loaded once at app boot (`_app.py:243`) and is NOT auto-updated when `QcRecipe.update()` mutates `CFG_QC_RECIPE`. The settings-edit callback MUST refresh it before rebuilding, or `run_qc` rebuilds from stale boot-time entries and the edit is invisible:

```python
recipe = current_app.config.get(CFG_QC_RECIPE)
pipeline = current_app.config.get(CFG_QC_PIPELINE)
pipeline.set_qc(list(recipe.entries))   # reflect the just-saved recipe
```

**Dash duplicate-output gotcha.** `_qc_tab/_callbacks.py` already registers callbacks on `STORE_QC_RECIPE_REVISION` (≈ lines 445, 469). A new callback sharing this Input needs `allow_duplicate=True` + `prevent_initial_call=True` on its Output to avoid Dash's "duplicate callback outputs" error. Give the recompute its own Output (e.g. a dedicated `STORE_QC_RECOMPUTE_DONE` tick the worklist subscribes to) rather than colliding with an existing store.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/gui/test_qc_review_recompute.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_qc_tab/_callbacks.py src/phenotypic/gui/results_viewer/_qc_tab/review/_review_state.py src/phenotypic/gui/results_viewer/_qc_tab/review/_callbacks.py tests/integration/gui/test_qc_review_recompute.py
git commit -m "feat(gui): durable settings-edit QC recompute + review-state reconciliation"
```

---

### Task 10: Rewire the Error tab's `verified_good_keys`

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_error_tab/_data.py` (`verified_good_keys`, `_module_reviewed_member_keys`; drop the `groupby_cols_for`/`load_qc_*`/`_eq_or_null` imports)
- Modify: `src/phenotypic/gui/results_viewer/_qc_tab/review/_data.py` (delete the four leftover helpers kept back in T8, once the Error tab no longer imports them)
- Test: `tests/gui/results_viewer/error_tab/test_error_data.py`, `test_error_tab_integration.py`

**Interfaces:**
- Consumes: `_db.list_modules`, `_db.module_members`, `ReviewState`, `decode_group_key`.
- Produces: `verified_good_keys(output_root, labeled_keys)` unchanged signature/semantics (unlabeled members of reviewed groups, any module), now reading the DuckDB catalog; modules with `supports_object_curation == False` are skipped.

- [ ] **Step 1: Write the failing test**

Update `test_error_data.py`'s verified-good test to seed a `qc.duckdb` (via `run_qc`) + a `review_state.json` marking a group reviewed, then assert `verified_good_keys` returns that group's unlabeled members. Add an assertion that a `GridOccupancy` (diagnostic-only) reviewed group contributes nothing.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/gui/results_viewer/error_tab/test_error_data.py -v`
Expected: FAIL — old helpers import parquet artifact that no longer exists.

- [ ] **Step 3: Reimplement**

```python
from phenotypic.gui.results_viewer._qc_tab.review import _db
from phenotypic.gui.results_viewer._qc_tab.review._review_state import (
    ReviewState, decode_group_key,
)


def verified_good_keys(output_root, labeled_keys):
    state = ReviewState.load(output_root.layout)
    modules = {m.instance_id: m for m in _db.list_modules(output_root)}
    reviewed_members: set[LabelKey] = set()
    for instance_id, progress in state.modules.items():
        mod = modules.get(instance_id)
        if mod is None or not mod.supports_object_curation or not progress.reviewed:
            continue
        for encoded in progress.reviewed:
            key = decode_group_key(encoded)
            members = _db.module_members(output_root, instance_id, key)
            if members.is_empty():
                continue
            for img, lbl in zip(
                members.get_column(KEY_IMAGE_FILE).to_list(),
                members.get_column(KEY_OBJECT_LABEL).to_list(),
            ):
                reviewed_members.add((str(img), int(lbl)))
    return reviewed_members - labeled_keys
```

Delete `_module_reviewed_member_keys` and the now-unused imports (`groupby_cols_for`, `load_qc_members`, `load_qc_summary`, `_eq_or_null` from `review/_data.py`).

Now that **no consumer remains**, DELETE the four leftover helpers from `review/_data.py` that T8 kept back: `load_qc_summary`, `load_qc_members`, `groupby_cols_for`, `_eq_or_null` (and drop them from `__all__`). After this, `review/_data.py` contains only `build_recompute_frame`, `_anti_join_removed`, and the `_KEY_*` constants.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/gui/results_viewer/error_tab/ tests/unit/gui/results_viewer/test_qc_review_data.py -v`
Expected: PASS (and no `ImportError` from the now-deleted helpers).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_error_tab/_data.py src/phenotypic/gui/results_viewer/_qc_tab/review/_data.py tests/gui/results_viewer/error_tab/
git commit -m "refactor(gui): Error-tab verified-good reads the QC DuckDB catalog; drop flat helpers"
```

---

### Task 11: Remove the GUI "Export QC report" button

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_qc_tab/_callbacks.py` (delete `_on_export_click`, `_export_qc_report`)
- Modify: `src/phenotypic/gui/results_viewer/_qc_tab/_ids.py` (delete `QC_EXPORT_BTN_ID`, `QC_EXPORT_TOAST_ID`), and the layout that mounts the button
- Modify: `src/phenotypic/gui/FEATURES.md` (remove the Export row)
- Test: `tests/e2e/gui/test_qc_tab.py` (delete both export tests)

- [ ] **Step 1: Delete the export tests (red→green inverted: removal)**

Delete `test_export_emits_qc_parquet_and_summary` and `test_export_button_disabled_when_no_checks` from `tests/e2e/gui/test_qc_tab.py`.

- [ ] **Step 2: Remove the button, callback, helper, and ids**

Delete `_on_export_click` + `_export_qc_report` from `_callbacks.py`; delete the two export ids from `_ids.py`; remove the button + toast from the QC tab layout (`_qc_tab/_layout.py`). Remove the Export row from `FEATURES.md`.

- [ ] **Step 3: Run the QC e2e + layout tests**

Run: `uv run pytest tests/e2e/gui/test_qc_tab.py tests/unit/gui/results_viewer/test_qc_review_layout.py -v`
Expected: PASS (no references to the removed button remain).

- [ ] **Step 4: Verify no dangling references**

Run: `grep -rn "QC_EXPORT\|_export_qc_report\|qc_summary\.json\|qc\.parquet" src/phenotypic`
Expected: no matches (the Export button wrote both `qc.parquet` and `qc_summary.json`).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_qc_tab/ src/phenotypic/gui/FEATURES.md tests/e2e/gui/test_qc_tab.py
git commit -m "refactor(gui): remove the Export QC report button (superseded by qc.duckdb)"
```

---

### Task 12: Remove the legacy public QC-parquet symbols + docs

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py` (delete `QC_SUMMARY_PARQUET`, `QC_MEMBERS_PARQUET`, `QC_CONFIG_JSON`; `qc_summary_parquet_path`, `qc_members_parquet_path`, `qc_config_json_path`; `BundleLayout.qc_summary_parquet`, `qc_members_parquet`, `qc_config_json`)
- Modify: `src/phenotypic/sdk_/__init__.py` (drop the six names from imports + `__all__`)
- Modify: root `CLAUDE.md` (deliverables/qc gotchas), `src/phenotypic/gui/CLAUDE.md` (Error-analysis tab + QC artifact refs)
- Test: `tests/unit/sdk_/test_bundle_layout.py` (remove assertions on the deleted accessors)

- [ ] **Step 1: Find every remaining reference**

Run: `grep -rn "QC_SUMMARY_PARQUET\|QC_MEMBERS_PARQUET\|QC_CONFIG_JSON\|qc_summary_parquet\|qc_members_parquet\|qc_config_json" src/phenotypic tests`
Expected: matches only in `_io_constants.py` (definitions/accessors), `sdk_/__init__.py` (exports), and `tests/unit/sdk_/test_bundle_layout.py` (accessor assertions). `_runner.py` already dropped these imports in T6 and must NOT appear — if it does, finish T6's import cleanup before proceeding.

- [ ] **Step 2: Delete the nine symbols and their tests/docs**

Remove the constants, helpers, and accessors. Drop them from `sdk_/__init__.py` imports + `__all__`. Remove the `test_bundle_layout.py` assertions for the three deleted accessors. Update both `CLAUDE.md` files: replace the `deliverables/qc/` artifact descriptions (qc_summary/qc_members/qc_config) with the `qc.duckdb` model, and update the gui Error-analysis-tab section (verified-good now reads `qc.duckdb`).

- [ ] **Step 3: Verify the cutover is clean**

Run: `grep -rn "qc_summary.parquet\|qc_members.parquet\|qc_config.json" src/phenotypic`
Expected: no matches (legacy migration in `migrate_legacy_qc` may still relocate a legacy *directory*, but must not reference these filenames).
Run: `uv run python -c "import phenotypic.sdk_ as s; assert not hasattr(s, 'QC_SUMMARY_PARQUET')"`
Expected: exit 0.

- [ ] **Step 4: Run the affected suites**

Run: `uv run pytest tests/unit/sdk_/test_bundle_layout.py tests/integration/cli/test_finalize_qc.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/__init__.py tests/unit/sdk_/test_bundle_layout.py CLAUDE.md src/phenotypic/gui/CLAUDE.md
git commit -m "refactor(sdk_)!: remove legacy QC parquet constants/helpers/accessors (DuckDB cutover)"
```

---

### Task 13: Full regression — lint, types, and the QC + GUI suites

**Files:** none (verification only)

- [ ] **Step 1: Lint + format**

Run: `uv run ruff check --fix`
Expected: clean (no remaining errors).

- [ ] **Step 2: Type check**

Run: `uv run mypy src/phenotypic`
Expected: no new errors introduced by the change (pre-existing baseline unchanged).

- [ ] **Step 3: Run the QC + finalize + GUI QC/Error suites**

Run:
```bash
uv run pytest tests/unit/qc tests/integration/cli/test_finalize_qc.py \
  tests/unit/gui/results_viewer tests/integration/gui/test_qc_review_recompute.py \
  tests/gui/results_viewer/error_tab tests/e2e/gui/test_qc_tab.py \
  tests/unit/sdk_/test_bundle_layout.py -v
```
Expected: PASS.

- [ ] **Step 4: Regenerate GUI tutorial screenshots (chrome changed: Export button removed)**

Run: `uv run python scripts/capture_gui_tutorial_screenshots.py`
Then `git add` the refreshed PNGs (commit ALL of them — do not cherry-pick the collateral, per CLAUDE.md).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test(qc): regression pass + refreshed GUI screenshots for the DuckDB cutover"
```

---

## Self-Review

**Spec coverage:**
- §6.1 storage / table naming → T2 (path) + T5 (sanitizer) + T6 (writer).
- §6.2 catalog schema → T6 (`_create_catalog`/`_insert_catalog_row`) + T3 (`QcTableSpec`).
- §6.3 contract (`to_table`/`table_spec`) → T3; diagnostic-only override → T4.
- §6.4 `run_qc` atomic full rebuild + Windows retry → T6.
- §6.5 CLI wiring (finalize/recompile) → unchanged call site (T6 keeps the `run_qc(post_df, pipeline, output_dir)` signature; finalize already calls it; recompile reaches finalize). Verified in spec §3.1.
- §6.6 GUI data API → T7; deletions → T8.
- §6.7 Error tab → T10.
- §6.8 live recompute (both triggers full rebuild; settings-edit durable) → T9.
- §6.9 review_state reconciliation → T9.
- §7 dependency → T1.
- §8 cutover (Export removal, public-symbol removal, docs, tests) → T11, T12, T13.
- §9 side-stores stay separate → respected (only QC-join paths change, T9/T10).
- §11 no-op when no checks; corrupt DB → None → T6 (no-op test), T7 (`open_qc_db` None test).

**Placeholder scan:** the only deferred body is `summary_stats`/`_robust_median` in T7 ("copy verbatim from `_data.py:211-286`") and the T8/T9/T10 test bodies sketched with `...` where they depend on a shared seeding fixture — these reference concrete, existing code to copy, not invented behavior. No `TBD`/"add error handling"/invented signatures.

**Type consistency:** `QcTableSpec` fields (T3) ↔ catalog DDL columns (T6 `_CATALOG_DDL`) ↔ `QcModule` fields (T7) are aligned (instance_id, cls/class→cls_name, name, table_name, summary_table, groupby_cols, metric/status/flag_col, on_col, member_key_cols, supports_object_curation, time_col, higher_is_bad, extra_cols). `run_qc(measurements_df, pipeline, output_dir, *, qc_output_dir=None)` matches the existing finalize call `run_qc(post_df.to_pandas(), pipeline, output_dir)` and the GUI `run_qc(frame, pipeline, Path(root), qc_output_dir=layout.qc_dir)` — note T6/T7 must read the DB at `qc_output_dir/qc.duckdb` consistently with the GUI passing `layout.qc_dir`. `module_summary`/`module_members`/`list_modules` names are used identically in T7, T8, T9, T10.

**Open items deferred to implementation (from spec §13):** materialized summary table (chosen — T6 writes `<t>__summary`); table-name sanitization (T5); no-op vs schema-only catalog (chosen no-op — T6).
