# Standalone `deliverables/` Bundle GUI Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `deliverables/` a self-contained, portable bundle the results viewer can open on its own with full parity (view + curate + QC review + error triage), while opportunistically using `results/` (per-image HDF) for full-res crops + a per-layer toggle when present.

**Architecture:** A new `BundleLayout` value object in `phenotypic.sdk_._io_constants` becomes the single authority on on-disk topology — it separates the *deliverables base* (the folder holding `master_measurements.parquet`) from the *optional output root* (the parent holding `results/` + `.phenotypic/`) and carries the `has_results`/`hdf_path` capability flags. `qc/` relocates under `deliverables/qc/` (canonical), with a one-time MOVE migration from the legacy root `qc/`. The GUI's `OutputRoot` holds a `BundleLayout` and routes every deliverables/qc path through it; pixel viewers tier per-image between full-res HDF crops and the overlay-PNG fallback.

**Tech Stack:** Python 3, pydantic v2 (operations — not touched here), polars, Dash + Flask (GUI), h5py / PIL / numpy (pixel path), pytest + pytest-qt + Playwright (tests). `uv` is the sole runner.

## Global Constraints

- **`uv` is the sole runner.** Never bare `python`/`pip`. Tests: `uv run pytest ...`; type: `uv run mypy src/phenotypic`; lint: `uv run ruff check --fix`.
- **Operations/analyzers are pydantic v2 models** — not touched by this plan, but never add positional constructors.
- **Resolve output paths via `phenotypic.sdk_` helpers / `BundleLayout`**, never hand-join `deliverables`/`qc`/`results` names.
- **GUI shared constants** live in `gui/_config.py` (Python idents) / `gui/_design.py` (CSS-shaped); CLI-produced filenames live in `sdk_._io_constants` and are re-exported by `_config.py`. Never re-spell a literal.
- **Memory discipline:** images are large; read only the needed HDF layer, crop, release — do not hold full `Image` instances beyond the cached decoded layer.
- **CI gate — FEATURES.md:** any PR touching `src/phenotypic/gui/` MUST modify `src/phenotypic/gui/FEATURES.md`; every `✅ shipping` row needs a resolvable `Test ref` (`path::test`).
- **GUI callbacks must be verified in a live browser** (Playwright), not unit tests alone — curation/QC callback wiring bugs only fire on `/_dash-update-component`.
- **Commit frequently** — one commit per task minimum. End commit messages with the `Co-Authored-By` / `Claude-Session` trailers used in this repo.
- **Spec:** `docs/superpowers/specs/2026-06-24-deliverables-standalone-gui-design.md`.

---

## Review Revisions (independent plan review, 2026-06-25)

An independent reviewer verified the load-bearing assumptions against the live venv and found that **`OutputRoot.root` set to `deliverables_base` in standalone mode causes a `deliverables/` double-join** in three external consumers that internally call `deliverables_dir(...)`: `QcRecipe.load` (`_recipe.py:327`), `MeasurementSchema` (`schema/_schema_cache.py:95`), and `run_qc` (`_runner.py:149`). The governing rule for this plan, applied throughout Task 5:

> **NEVER pass `output_root.root` into a helper that internally joins `deliverables/`/`qc/`.** Route through `OutputRoot.layout` accessors, or give the consumer a `BundleLayout`-aware entry point. After Task 4, `output_root.root` is the *deliverables folder* in standalone mode, so any `deliverables_dir(output_root.root)` / `qc_dir(output_root.root)` call double-joins.

Incorporated fixes (folded into the tasks below): `from_layout` constructors on `QcRecipe`/`MeasurementSchema` (C1); a `qc_output_dir` param on `run_qc` (C2); `CurationLabels.load`/`ReviewState.load` take a `BundleLayout` and the CLI caller `_cli_error_outputs.py:59` is updated (C3); the hand-joined dead paths at `_qc_tab/_callbacks.py:1018-1020` are removed/routed (C4); viewer-cache read-only fallback + a `viewer_cache_dir` property (C5); the `results_dir is None` guard lands in Task 4 to avoid a crash window (W2); Task 6 reads only `/layers/<name>` via `h5py` instead of a full `Image` load (W5, satisfies the memory-discipline constraint); a full `output_root.root` audit step (Q1). Note: `migrate_legacy_qc`'s `shutil.move` of `qc/` is an atomic `os.rename` when source and destination share a filesystem (always true here — both under the output root), so the whole-directory move has no partial-resume hazard on the common path; only a cross-filesystem move degrades to copy+delete, accepted under the documented no-concurrent-GUI+CLI assumption (W1).

---

## File Structure

**Created:**
- (none — all changes extend existing modules)

**Modified — Phase 1 (sdk_ foundation):**
- `src/phenotypic/sdk_/_io_constants.py` — add `BundleLayout`; relocate `qc_dir`; add `resolve_qc_dir`, `migrate_legacy_qc`.
- `src/phenotypic/sdk_/__init__.py` — export `BundleLayout`, `resolve_qc_dir`, `migrate_legacy_qc`.
- `src/phenotypic/_cli/_cli_output_manager.py` — call `migrate_legacy_qc` in `finalize_post_master_outputs`.

**Modified — Phase 2 (GUI discovery + routing):**
- `src/phenotypic/gui/results_viewer/_output_root.py` — `BundleLayout`-backed discovery + accessors.
- `src/phenotypic/gui/results_viewer/_curation_labels.py`, `_filtered_state.py`,
  `_qc_tab/review/_data.py`, `_qc_tab/review/_review_state.py`, `_qc_tab/review/_callbacks.py`,
  `_qc_tab/_callbacks.py`, `_error_tab/_data.py`, `_error_tab/_callbacks.py`,
  `_app.py`, `_layout.py` — route path resolution through `OutputRoot.layout`.
- `src/phenotypic/sdk_/_qc_recipe/_recipe.py` — add `QcRecipe.from_layout(layout)` (C1).
- `src/phenotypic/sdk_/_qc_recipe/_runner.py` — add `run_qc(..., qc_output_dir=None)` (C2).
- `src/phenotypic/schema/_schema_cache.py` — add `MeasurementSchema.from_layout(layout)` (C1).
- `src/phenotypic/_cli/_cli_error_outputs.py` — update `CurationLabels.load` call to a `BundleLayout` (C3).

**Modified — Phase 3 (pixel tiering):**
- `src/phenotypic/gui/_shared/tiles.py` — `crop_hdf_rgb`, `crop_colony`, layer cache.
- `src/phenotypic/gui/results_viewer/_tile_routes.py`, `colony_view/*`, `timeline_view/_thumb_routes.py` — layer-aware tiling + toggle.
- `src/phenotypic/gui/FEATURES.md` — rows for toggle + fidelity indicator.

**Modified — Phase 4 (mode signaling):**
- `src/phenotypic/gui/results_viewer/_layout.py` (or header component), `shell/_classifier.py`, `FEATURES.md`.

**Modified — Phase 5 (docs + e2e):**
- `src/phenotypic/gui/CLAUDE.md`, root `CLAUDE.md`, `src/phenotypic/sdk_/_io_constants.py` docstrings, QC `CLAUDE.md`.
- `tests/.../test_deliverables_standalone_e2e.py` (new Playwright e2e).

**Test files (created/extended):**
- `tests/unit/tools_/test_bundle_layout.py` (new)
- `tests/unit/tools_/test_io_constants.py` (extend — qc relocation + migration)
- `tests/unit/gui/results_viewer/test_output_root.py` (extend — standalone discovery)
- `tests/unit/gui/_shared/test_tiles.py` (extend — `crop_hdf_rgb` / `crop_colony`)
- the GUI e2e suite (extend)

> Confirm exact test paths with `uv run pytest --collect-only` before writing; the directories above mirror the existing layout but verify per-file.

---

## Phase 1 — `sdk_` Foundation (BundleLayout + qc relocation + migration)

### Task 1: `BundleLayout` value object

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py` (add after the path-builder helpers section, ~line 600+)
- Modify: `src/phenotypic/sdk_/__init__.py` (import + `__all__`)
- Test: `tests/unit/tools_/test_bundle_layout.py` (create)

**Interfaces:**
- Consumes: existing constants `MASTER_MEASUREMENTS_PARQUET`, `MEASUREMENTS_PARQUET`, `MEASUREMENTS_CSV`, `MASTER_MEASUREMENTS_CSV`, `DIR_DELIVERABLES`, `DIR_RESULTS`, `DIR_HDF`, `DIR_OVERLAYS`, `DIR_QC`, `DIR_ERRORS`, `QC_SUMMARY_PARQUET`, `QC_MEMBERS_PARQUET`, `QC_CONFIG_JSON`, `QC_REVIEW_STATE_JSON`, `CURATION_LABELS_PARQUET`, `CUSTOM_CATEGORIES_JSON`, `PIPELINE_JSON`; helpers `dataset_hdf_dir`, `dataset_overlays_dir`, `resolve_pipeline_config_path`.
- Produces:
  ```python
  @dataclass(frozen=True)
  class BundleLayout:
      deliverables_base: Path
      output_root: Path | None
      @classmethod
      def detect(cls, path: Path) -> "BundleLayout"
      @property
      def has_results(self) -> bool
      def hdf_path(self, dataset: str, stem: str) -> Path | None
      @property
      def results_dir(self) -> Path | None
      @property
      def master_parquet(self) -> Path
      @property
      def master_csv(self) -> Path
      @property
      def mirror_parquet(self) -> Path
      @property
      def mirror_csv(self) -> Path
      @property
      def pipeline_config_path(self) -> Path
      @property
      def qc_dir(self) -> Path           # resolved: deliverables/qc, fallback legacy root/qc
      @property
      def qc_summary_parquet(self) -> Path
      @property
      def qc_members_parquet(self) -> Path
      @property
      def qc_config_json(self) -> Path
      @property
      def qc_review_state_path(self) -> Path
      @property
      def curation_labels_parquet(self) -> Path
      @property
      def custom_categories_json(self) -> Path
      @property
      def errors_dir(self) -> Path
      def error_category_parquet(self, category: str) -> Path
      def overlays_dir(self, dataset: str) -> Path
      def overlay_path(self, dataset: str, stem: str) -> Path
  ```
  > NOTE: `qc_dir` resolution is added in Task 2 (it depends on `resolve_qc_dir`). In Task 1 implement it as `self.deliverables_base / DIR_QC` and a test marked to update in Task 2; or sequence Task 2 first if implementing inline. The plan below assumes Task 1 ships the simple form and Task 2 upgrades it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/tools_/test_bundle_layout.py
from pathlib import Path

import polars as pl
import pytest

from phenotypic.sdk_ import BundleLayout


def _seed_deliverables(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"Metadata_Dataset": ["plate1"], "Object_Label": [1]}).write_parquet(
        base / "master_measurements.parquet"
    )


def test_detect_when_pointed_at_parent_containing_deliverables(tmp_path):
    out = tmp_path / "run"
    _seed_deliverables(out / "deliverables")
    layout = BundleLayout.detect(out)
    assert layout.deliverables_base == (out / "deliverables").resolve()
    assert layout.output_root == out.resolve()


def test_detect_when_pointed_at_deliverables_dir_standalone(tmp_path):
    base = tmp_path / "bundle" / "deliverables"
    _seed_deliverables(base)
    layout = BundleLayout.detect(base)
    assert layout.deliverables_base == base.resolve()
    # No sibling results/ -> standalone, no output_root.
    assert layout.output_root is None
    assert layout.has_results is False
    assert layout.results_dir is None
    assert layout.hdf_path("plate1", "img001") is None


def test_detect_deliverables_subdir_with_sibling_results_promotes_parent(tmp_path):
    out = tmp_path / "run"
    _seed_deliverables(out / "deliverables")
    (out / "results" / "plate1" / "hdf").mkdir(parents=True)
    layout = BundleLayout.detect(out / "deliverables")
    assert layout.output_root == out.resolve()
    assert layout.has_results is True


def test_promotion_guard_requires_deliverables_name(tmp_path):
    # A standalone bundle NOT named "deliverables" must not adopt a sibling results/.
    base = tmp_path / "shared_bundle"
    _seed_deliverables(base)
    (tmp_path / "results" / "plate1").mkdir(parents=True)
    layout = BundleLayout.detect(base)
    assert layout.output_root is None


def test_detect_rejects_non_bundle(tmp_path):
    with pytest.raises(FileNotFoundError):
        BundleLayout.detect(tmp_path)


def test_hdf_path_returns_path_when_h5_exists(tmp_path):
    out = tmp_path / "run"
    _seed_deliverables(out / "deliverables")
    hdf_dir = out / "results" / "plate1" / "hdf"
    hdf_dir.mkdir(parents=True)
    (hdf_dir / "img001.h5").write_bytes(b"")
    layout = BundleLayout.detect(out)
    assert layout.hdf_path("plate1", "img001") == hdf_dir / "img001.h5"
    assert layout.hdf_path("plate1", "missing") is None


def test_deliverables_accessors_anchor_on_base(tmp_path):
    out = tmp_path / "run"
    _seed_deliverables(out / "deliverables")
    layout = BundleLayout.detect(out)
    base = out / "deliverables"
    assert layout.master_parquet == base / "master_measurements.parquet"
    assert layout.mirror_parquet == base / "measurements.parquet"
    assert layout.qc_summary_parquet == base / "qc" / "qc_summary.parquet"
    assert layout.curation_labels_parquet == base / "qc" / "curation_labels.parquet"
    assert layout.overlay_path("plate1", "img001") == base / "overlays" / "plate1" / "img001.png"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/tools_/test_bundle_layout.py -v`
Expected: FAIL with `ImportError: cannot import name 'BundleLayout'`.

- [ ] **Step 3: Implement `BundleLayout`**

Add to `src/phenotypic/sdk_/_io_constants.py` (ensure `from dataclasses import dataclass` and `from typing import Optional` are imported at module top — `dataclass` likely is not yet; add it):

```python
@dataclass(frozen=True)
class BundleLayout:
    """Resolved on-disk topology of a run output or a standalone deliverables bundle.

    Separates the *deliverables base* (the folder directly holding
    ``master_measurements.parquet``) from the optional *output root* (the parent
    that also holds ``results/`` and ``.phenotypic/``). A standalone bundle has
    ``output_root is None``; deliverables-internal artefacts always resolve from
    ``deliverables_base`` so the bundle is portable.

    Attributes:
        deliverables_base: Folder containing ``master_measurements.parquet``.
        output_root: Parent run directory holding ``results/`` + machine state,
            or ``None`` for a standalone (deliverables-only) bundle.
    """

    deliverables_base: Path
    output_root: Optional[Path]

    @classmethod
    def detect(cls, path: Path) -> "BundleLayout":
        """Classify ``path`` as a run output dir or a standalone deliverables bundle.

        Case 1 — ``path`` directly holds ``master_measurements.parquet``: treat it
        as the deliverables base. Promote ``path.parent`` to ``output_root`` ONLY
        when ``path`` is literally named ``deliverables`` AND a sibling ``results/``
        exists (the "pointed at the deliverables subdir of a full run" case); this
        guard stops a renamed standalone bundle from adopting an unrelated sibling
        ``results/``.

        Case 2 — ``path`` contains ``deliverables/master_measurements.parquet``:
        ``deliverables_base = path/deliverables`` and ``output_root = path``.

        Raises:
            FileNotFoundError: ``path`` is neither.
        """
        path = Path(path).resolve()
        if (path / MASTER_MEASUREMENTS_PARQUET).is_file():
            output_root: Optional[Path] = None
            if path.name == DIR_DELIVERABLES and (path.parent / DIR_RESULTS).is_dir():
                output_root = path.parent
            return cls(deliverables_base=path, output_root=output_root)
        if (path / DIR_DELIVERABLES / MASTER_MEASUREMENTS_PARQUET).is_file():
            return cls(deliverables_base=path / DIR_DELIVERABLES, output_root=path)
        raise FileNotFoundError(
            f"{path} is neither a deliverables bundle nor a run output directory "
            f"containing {DIR_DELIVERABLES}/{MASTER_MEASUREMENTS_PARQUET}. Point the "
            "viewer at a `python -m phenotypic` output dir or a deliverables/ folder."
        )

    # -- capability ---------------------------------------------------------
    @property
    def has_results(self) -> bool:
        return self.output_root is not None and (self.output_root / DIR_RESULTS).is_dir()

    @property
    def results_dir(self) -> Optional[Path]:
        return (self.output_root / DIR_RESULTS) if self.has_results else None

    def hdf_path(self, dataset: str, stem: str) -> Optional[Path]:
        """Full-res per-image HDF for ``(dataset, stem)``, or ``None`` if unavailable."""
        if not self.has_results:
            return None
        candidate = dataset_hdf_dir(self.output_root, dataset) / f"{stem}.h5"
        return candidate if candidate.is_file() else None

    # -- deliverables-anchored artefacts ------------------------------------
    @property
    def master_parquet(self) -> Path:
        return self.deliverables_base / MASTER_MEASUREMENTS_PARQUET

    @property
    def master_csv(self) -> Path:
        return self.deliverables_base / MASTER_MEASUREMENTS_CSV

    @property
    def mirror_parquet(self) -> Path:
        return self.deliverables_base / MEASUREMENTS_PARQUET

    @property
    def mirror_csv(self) -> Path:
        return self.deliverables_base / MEASUREMENTS_CSV

    @property
    def pipeline_config_path(self) -> Path:
        # resolve_pipeline_config_path expects an output_dir whose deliverables_dir
        # is deliverables_base; pass deliverables_base.parent only when valid.
        return self.deliverables_base / PIPELINE_JSON

    @property
    def qc_dir(self) -> Path:
        # Upgraded in Task 2 to resolve the legacy-root fallback.
        return self.deliverables_base / DIR_QC

    @property
    def qc_summary_parquet(self) -> Path:
        return self.qc_dir / QC_SUMMARY_PARQUET

    @property
    def qc_members_parquet(self) -> Path:
        return self.qc_dir / QC_MEMBERS_PARQUET

    @property
    def qc_config_json(self) -> Path:
        return self.qc_dir / QC_CONFIG_JSON

    @property
    def qc_review_state_path(self) -> Path:
        return self.qc_dir / QC_REVIEW_STATE_JSON

    @property
    def curation_labels_parquet(self) -> Path:
        return self.qc_dir / CURATION_LABELS_PARQUET

    @property
    def custom_categories_json(self) -> Path:
        return self.qc_dir / CUSTOM_CATEGORIES_JSON

    @property
    def errors_dir(self) -> Path:
        return self.deliverables_base / DIR_ERRORS

    def error_category_parquet(self, category: str) -> Path:
        return self.errors_dir / f"{category}.parquet"

    def overlays_dir(self, dataset: str) -> Path:
        return self.deliverables_base / DIR_OVERLAYS / dataset

    def overlay_path(self, dataset: str, stem: str) -> Path:
        return self.overlays_dir(dataset) / f"{stem}.png"
```

> If `PIPELINE_JSON`, `MASTER_MEASUREMENTS_CSV`, `MEASUREMENTS_CSV`, `MEASUREMENTS_PARQUET` constant names differ, confirm via `grep -n "PIPELINE_JSON\|MEASUREMENTS_CSV\|MEASUREMENTS_PARQUET\b\|MASTER_MEASUREMENTS_CSV" src/phenotypic/sdk_/_io_constants.py` and use the exact names. `error_analysis_*` / `verified_parquet` / `analysis_*` accessors are added in Task 5 only if a call site needs them — keep Task 1 to the set the tests cover.

- [ ] **Step 4: Export from `sdk_/__init__.py`**

Add `BundleLayout` to the import block and `__all__` (alphabetical neighbours near `dataset_hdf_dir`):
```python
    BundleLayout,
```
and in `__all__`:
```python
    "BundleLayout",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/tools_/test_bundle_layout.py -v`
Expected: PASS (all 7 tests).
Then: `uv run mypy src/phenotypic/sdk_/_io_constants.py` — expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/__init__.py tests/unit/tools_/test_bundle_layout.py
git commit -m "feat(sdk_): add BundleLayout topology + capability value object"
```

---

### Task 2: Relocate `qc_dir` under `deliverables/` + legacy read-fallback

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py:1268` (`qc_dir`), add `resolve_qc_dir`; upgrade `BundleLayout.qc_dir`.
- Modify: `src/phenotypic/sdk_/__init__.py` (export `resolve_qc_dir`).
- Test: `tests/unit/tools_/test_io_constants.py` (extend)

**Interfaces:**
- Consumes: `deliverables_dir`, `DIR_QC` (Task 1 constants).
- Produces: `qc_dir(output_dir) -> deliverables/qc`; `resolve_qc_dir(output_dir) -> Path` (deliverables/qc if exists, else legacy `output_dir/qc` if exists, else deliverables/qc). `BundleLayout.qc_dir` returns the resolved path.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/tools_/test_io_constants.py  (append)
def test_qc_dir_is_now_under_deliverables(tmp_path):
    from phenotypic.sdk_ import qc_dir, deliverables_dir
    assert qc_dir(tmp_path) == deliverables_dir(tmp_path) / "qc"


def test_resolve_qc_dir_prefers_deliverables_then_legacy(tmp_path):
    from phenotypic.sdk_ import resolve_qc_dir, qc_dir
    # Neither exists -> canonical.
    assert resolve_qc_dir(tmp_path) == qc_dir(tmp_path)
    # Legacy only -> legacy.
    legacy = tmp_path / "qc"
    legacy.mkdir()
    assert resolve_qc_dir(tmp_path) == legacy
    # Canonical present -> canonical wins.
    qc_dir(tmp_path).mkdir(parents=True)
    assert resolve_qc_dir(tmp_path) == qc_dir(tmp_path)


def test_bundle_layout_qc_dir_resolves_legacy(tmp_path):
    import polars as pl
    from phenotypic.sdk_ import BundleLayout
    out = tmp_path / "run"
    (out / "deliverables").mkdir(parents=True)
    pl.DataFrame({"Metadata_Dataset": ["p1"]}).write_parquet(
        out / "deliverables" / "master_measurements.parquet"
    )
    (out / "qc").mkdir()  # legacy root qc, no deliverables/qc yet
    layout = BundleLayout.detect(out)
    assert layout.qc_dir == out / "qc"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/tools_/test_io_constants.py -k "qc_dir or resolve_qc" -v`
Expected: FAIL — `qc_dir` still returns `<output>/qc`; `resolve_qc_dir` undefined.

- [ ] **Step 3: Relocate `qc_dir` + add `resolve_qc_dir`**

Edit `qc_dir` (currently `_io_constants.py:1268`):
```python
def qc_dir(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/qc/`` — durable QC + curation state.

    Relocated under ``deliverables/`` so a deliverables bundle is self-contained
    and portable. Use :func:`resolve_qc_dir` for reads that must honour the legacy
    root ``<output>/qc/`` layout of pre-relocation runs.
    """
    return deliverables_dir(output_dir) / DIR_QC


def _legacy_qc_dir(output_dir: Path) -> Path:
    """Pre-relocation location: ``<output>/qc/``."""
    return output_dir / DIR_QC


def resolve_qc_dir(output_dir: Path) -> Path:
    """Return the qc dir that exists, preferring ``deliverables/qc/``.

    Read-only resolver: deliverables/qc if present, else legacy root qc if
    present, else the canonical deliverables/qc (for fresh writes).
    """
    new = qc_dir(output_dir)
    if new.exists():
        return new
    legacy = _legacy_qc_dir(output_dir)
    if legacy.exists():
        return legacy
    return new
```

Upgrade `BundleLayout.qc_dir` (Task 1) to resolve the fallback:
```python
    @property
    def qc_dir(self) -> Path:
        canonical = self.deliverables_base / DIR_QC
        if canonical.exists():
            return canonical
        if self.output_root is not None:
            legacy = self.output_root / DIR_QC
            if legacy.exists():
                return legacy
        return canonical
```

Export `resolve_qc_dir` from `sdk_/__init__.py` (import + `__all__`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/tools_/test_io_constants.py -k "qc" -v` and `uv run pytest tests/unit/tools_/test_bundle_layout.py -v`
Expected: PASS.

- [ ] **Step 5: Run the QC + CLI test suites to catch fixtures that hardcoded root `qc/`**

Run: `uv run pytest tests/unit/tools_/_qc_recipe tests/unit/_cli -q`
Expected: PASS, or a small set of fixture failures asserting `<output>/qc/...`. Fix each by switching the assertion to `qc_dir(out)` / `deliverables_dir(out)/"qc"`. List every fixture changed in the commit body.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/__init__.py tests/unit/tools_/test_io_constants.py
git commit -m "feat(sdk_)!: relocate qc/ under deliverables/qc with legacy read-fallback"
```

---

### Task 3: `migrate_legacy_qc` + wire into finalize

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py` (add `migrate_legacy_qc` next to `migrate_legacy_machine_state` ~line 707)
- Modify: `src/phenotypic/sdk_/__init__.py` (export)
- Modify: `src/phenotypic/_cli/_cli_output_manager.py` (`finalize_post_master_outputs` calls it)
- Test: `tests/unit/tools_/test_io_constants.py` (extend)

**Interfaces:**
- Consumes: `_legacy_qc_dir`, `qc_dir` (Task 2).
- Produces: `migrate_legacy_qc(output_dir: Path) -> bool` — MOVE legacy `<output>/qc/` → `<output>/deliverables/qc/` once; idempotent; returns `True` if it moved anything.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/tools_/test_io_constants.py  (append)
def test_migrate_legacy_qc_moves_once(tmp_path):
    from phenotypic.sdk_ import migrate_legacy_qc, qc_dir
    legacy = tmp_path / "qc"
    legacy.mkdir()
    (legacy / "curation_labels.parquet").write_bytes(b"x")
    (tmp_path / "deliverables").mkdir()

    assert migrate_legacy_qc(tmp_path) is True
    assert (qc_dir(tmp_path) / "curation_labels.parquet").is_file()
    assert not legacy.exists()
    # Idempotent: second call is a no-op.
    assert migrate_legacy_qc(tmp_path) is False


def test_migrate_legacy_qc_noop_when_no_legacy(tmp_path):
    from phenotypic.sdk_ import migrate_legacy_qc
    (tmp_path / "deliverables").mkdir()
    assert migrate_legacy_qc(tmp_path) is False


def test_migrate_legacy_qc_noop_when_canonical_exists(tmp_path):
    from phenotypic.sdk_ import migrate_legacy_qc, qc_dir
    (tmp_path / "qc").mkdir()
    (tmp_path / "qc" / "a.parquet").write_bytes(b"x")
    qc_dir(tmp_path).mkdir(parents=True)  # canonical already present
    assert migrate_legacy_qc(tmp_path) is False
    # Legacy is left untouched (no merge); resolver will still prefer canonical.
    assert (tmp_path / "qc" / "a.parquet").is_file()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/tools_/test_io_constants.py -k migrate_legacy_qc -v`
Expected: FAIL — `migrate_legacy_qc` undefined.

- [ ] **Step 3: Implement `migrate_legacy_qc`**

```python
def migrate_legacy_qc(output_dir: Path) -> bool:
    """Move a pre-relocation run's ``<output>/qc/`` into ``deliverables/qc/``.

    Hard cutover (MOVE, no duplication), mirroring
    :func:`migrate_legacy_machine_state`. A no-op when there is no legacy ``qc/``
    or when the canonical ``deliverables/qc/`` already exists (the move is
    whole-directory; we never merge a half-written canonical with legacy).

    Returns:
        ``True`` if this call moved the directory, else ``False``.
    """
    import shutil

    legacy = _legacy_qc_dir(output_dir)
    canonical = qc_dir(output_dir)
    if not legacy.is_dir() or canonical.exists():
        return False
    canonical.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(legacy), str(canonical))
    except (FileNotFoundError, shutil.Error):
        # Lost a race with a concurrent migrator; safe to skip.
        return False
    return True
```

Export from `sdk_/__init__.py`.

> **Atomicity (review W1):** `shutil.move` of a directory uses `os.rename` when source and destination share a filesystem — which they always do here (`<output>/qc/` → `<output>/deliverables/qc/` are both under the output root). `os.rename` is atomic, so there is no partial-state window on the common path; the whole-directory move (vs the per-artifact move in `migrate_legacy_machine_state`) is therefore safe. Only a cross-filesystem output dir degrades to copy+delete, accepted under the documented no-concurrent-GUI+CLI assumption.

- [ ] **Step 4: Wire into `finalize_post_master_outputs`**

In `src/phenotypic/_cli/_cli_output_manager.py`, near the top of `finalize_post_master_outputs(output_dir, master_df, pipeline, ...)`, before it writes any qc/deliverables artefact, add:
```python
    from phenotypic.sdk_ import migrate_legacy_qc

    migrate_legacy_qc(output_dir)
```
(Confirm the function name/signature with `grep -n "def finalize_post_master_outputs" src/phenotypic/_cli/_cli_output_manager.py`.)

- [ ] **Step 5: Run tests + a CLI smoke**

Run: `uv run pytest tests/unit/tools_/test_io_constants.py -k migrate_legacy_qc -v`
Expected: PASS.
Run: `uv run pytest tests/unit/_cli -k "finalize or recompile" -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/__init__.py src/phenotypic/_cli/_cli_output_manager.py tests/unit/tools_/test_io_constants.py
git commit -m "feat(sdk_): one-time MOVE migration of legacy qc/ into deliverables/qc"
```

---

## Phase 2 — GUI Discovery + Path Routing

### Task 4: `BundleLayout`-backed `OutputRoot.discover`

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_output_root.py` (discover + new `layout` field + passthrough accessors)
- Test: `tests/unit/gui/results_viewer/test_output_root.py` (extend)

**Interfaces:**
- Consumes: `BundleLayout` (Task 1), `migrate_legacy_qc` (Task 3).
- Produces: `OutputRoot.layout: BundleLayout`; `OutputRoot.has_results -> bool`; `OutputRoot.hdf_path(ds, stem) -> Path | None`; `OutputRoot.overlay_path` unchanged signature but routed through `layout`. `OutputRoot.discover(root)` no longer requires `results/`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/gui/results_viewer/test_output_root.py  (append)
import polars as pl
from phenotypic.gui.results_viewer._output_root import OutputRoot


def _seed_standalone_bundle(base):
    """Deliverables-only: master + mirror + one overlay, NO results/, NO root qc/."""
    base.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {"Metadata_Dataset": ["plate1", "plate1"],
         "Metadata_ImageFile": ["img001", "img001"],
         "Object_Label": [1, 2]}
    )
    df.write_parquet(base / "master_measurements.parquet")
    df.write_parquet(base / "measurements.parquet")
    ov = base / "overlays" / "plate1"
    ov.mkdir(parents=True)
    from PIL import Image as PILImage
    PILImage.new("RGB", (8, 8)).save(ov / "img001.png")


def test_discover_standalone_deliverables_only(tmp_path):
    base = tmp_path / "bundle" / "deliverables"
    _seed_standalone_bundle(base)
    root = OutputRoot.discover(base)
    assert root.has_results is False
    assert "plate1" in root.master_df["Metadata_Dataset"].unique().to_list()
    # Overlay-backed picker still works.
    assert root.has_overlay("plate1", "img001") is True
    assert root.hdf_path("plate1", "img001") is None


def test_discover_full_run_lights_up_results(tmp_path):
    out = tmp_path / "run"
    _seed_standalone_bundle(out / "deliverables")
    (out / "results" / "plate1" / "hdf").mkdir(parents=True)
    (out / "results" / "plate1" / "hdf" / "img001.h5").write_bytes(b"")
    root = OutputRoot.discover(out)
    assert root.has_results is True
    assert root.hdf_path("plate1", "img001") is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/gui/results_viewer/test_output_root.py -k "standalone or full_run" -v`
Expected: FAIL — discover raises `FileNotFoundError` (no `results/`), and `has_results`/`hdf_path` don't exist.

- [ ] **Step 3: Refactor `discover` + add fields/accessors**

In `_output_root.py`:

1. Add to imports: `from phenotypic.sdk_ import BundleLayout, migrate_legacy_qc` (extend the existing `from phenotypic.sdk_ import (...)`).
2. Add a `layout: BundleLayout` field to the dataclass (after `root`).
3. Replace the body of `discover` (lines ~111-208) so it builds from a `BundleLayout`:

```python
        layout = BundleLayout.detect(Path(root))
        if layout.output_root is not None:
            migrate_legacy_qc(layout.output_root)

        master_path = layout.master_parquet
        if not master_path.is_file():
            raise FileNotFoundError(
                f"Master measurements parquet not found at {master_path!s}. "
                "Point the viewer at a `python -m phenotypic` output dir or a "
                "deliverables/ bundle."
            )
        clean_master_df = pl.read_parquet(master_path)

        mirror_path = layout.mirror_parquet
        master_df = pl.read_parquet(mirror_path) if mirror_path.is_file() else clean_master_df

        # Datasets are data-driven: master frame is authoritative, unioned with
        # overlay subdirs (and results/ when present) to catch a dataset with
        # overlays but no surviving rows.
        datasets = _discover_datasets(master_df, layout)
        if not datasets:
            raise FileNotFoundError(
                f"No datasets found in {layout.deliverables_base!s}. Expected a "
                "Metadata_Dataset column or deliverables/overlays/<dataset>/ dirs."
            )

        master_df = _ensure_required_columns(master_df, layout, datasets)
        clean_master_df = _ensure_required_columns(clean_master_df, layout, datasets)

        datasets_with_overlays = [
            ds for ds in datasets if layout.overlays_dir(ds).is_dir()
        ]
        column_value_sets = _build_column_value_sets(master_df)

        # Cache root: the output root for full runs, else inside the bundle.
        # Read-only-mount fallback (spec Section 2 / review C5): if the chosen
        # location is not writable, use a per-session temp dir keyed by bundle path.
        cache_root = layout.output_root if layout.output_root is not None else layout.deliverables_base
        cache_dir = cache_root / _CACHE_RELATIVE
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            import hashlib
            import tempfile
            key = hashlib.sha1(str(layout.deliverables_base).encode()).hexdigest()[:12]
            cache_dir = Path(tempfile.gettempdir()) / f"phenotypic-viewer-{key}" / _CACHE_RELATIVE
            cache_dir.mkdir(parents=True, exist_ok=True)

        pipeline_summary = _read_pipeline_summary(layout.pipeline_config_path)
        overlay_index = _scan_overlay_index(layout, datasets_with_overlays)

        return cls(
            root=layout.deliverables_base if layout.output_root is None else layout.output_root,
            layout=layout,
            master_df=master_df,
            clean_master_df=clean_master_df,
            column_value_sets=column_value_sets,
            cache_dir=cache_dir,
            pipeline_summary=pipeline_summary,
            overlay_index=overlay_index,
        )
```

4. Add the helpers + accessors:
```python
    @property
    def has_results(self) -> bool:
        return self.layout.has_results

    def hdf_path(self, dataset: str, stem: str):
        return self.layout.hdf_path(dataset, stem)

    @property
    def results_dir(self):
        # None for a standalone bundle; callers must guard.
        return self.layout.results_dir

    @property
    def viewer_cache_dir(self) -> Path:
        # The cache *root* (parent of the dzi/ subdir held by cache_dir). Thumb
        # routes and any non-DZI cache write under here so the standalone path
        # stays inside the bundle. (review Q2 — distinct from cache_dir = .../dzi)
        return self.cache_dir.parent
```
Replace `overlay_path` body to `return self.layout.overlay_path(dataset, stem)`.

> **Guard `results_dir` now (review W2):** because `results_dir` becomes `Path | None`, immediately update its one existing consumer `_tile_routes.py:69` in THIS task (not Task 7) to avoid a `None`-division crash window between commits: change `if not (output_root.results_dir / dataset).is_dir():` to `if output_root.results_dir is None or not (output_root.results_dir / dataset).is_dir():`. Task 7 then replaces this gate entirely with the capability check.

5. Add module-level `_discover_datasets`:
```python
def _discover_datasets(master_df: pl.DataFrame, layout: "BundleLayout") -> list[str]:
    names: set[str] = set()
    if KEY_DATASET in master_df.columns:
        names.update(
            str(v) for v in master_df.get_column(KEY_DATASET).drop_nulls().unique().to_list()
        )
    overlays_root = layout.deliverables_base / "overlays"
    if overlays_root.is_dir():
        names.update(e.name for e in overlays_root.iterdir() if e.is_dir())
    if layout.results_dir is not None:
        names.update(e.name for e in layout.results_dir.iterdir() if e.is_dir())
    return sorted(names)
```
> Use `DIR_OVERLAYS` instead of the `"overlays"` literal — import it from `phenotypic.sdk_`.

6. Update `_ensure_required_columns` signature from `(df, results_dir, datasets)` to `(df, layout, datasets)`; replace the backfill block (lines 339-363) so it is capability-aware:
```python
    if KEY_DATASET in df.columns:
        return df
    if layout.results_dir is None:
        raise ValueError(
            "Master measurements parquet is missing column 'Metadata_Dataset' and "
            "this is a standalone deliverables bundle (no results/ to recover it "
            "from). Recompile the run with the current version: "
            "`python -m phenotypic --mode recompile --output <dir>`."
        )
    results_root = layout.results_dir
    # ... existing per-image backfill loop, using results_root in place of results_dir ...
```

7. Update `_scan_overlay_index(root, datasets)` → `_scan_overlay_index(layout, datasets)` and use `layout.overlays_dir(dataset)` instead of `dataset_overlays_dir(root, dataset)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/gui/results_viewer/test_output_root.py -v`
Expected: PASS (new + existing). If existing tests construct `results/` and assert `.results_dir`, they still pass (full-run path). Fix any that asserted discovery *fails* without `results/` — that is now intended behaviour.

- [ ] **Step 5: Type-check**

Run: `uv run mypy src/phenotypic/gui/results_viewer/_output_root.py`
Expected: no new errors (note `results_dir` is now `Path | None` — callers guarded in Task 5).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_output_root.py tests/unit/gui/results_viewer/test_output_root.py
git commit -m "feat(gui): BundleLayout-backed discovery; boot from deliverables/ alone"
```

---

### Task 5: Route GUI path resolution through `OutputRoot.layout`

**Why:** ~30 call sites pass `output_root.root` into `output_dir`-based helpers. In a standalone bundle `root` IS the deliverables folder, so `deliverables_dir(root)` would double-join. Route every deliverables/qc path through `layout`.

**Files (modify):**
- GUI: `_curation_labels.py`, `_filtered_state.py`, `_qc_tab/review/_data.py`,
  `_qc_tab/review/_review_state.py`, `_qc_tab/review/_callbacks.py`, `_qc_tab/_callbacks.py`,
  `_error_tab/_data.py`, `_error_tab/_callbacks.py`, `_app.py`, `_layout.py`,
  `_tile_routes.py`, `timeline_view/_thumb_routes.py`
- sdk_/CLI (review C1/C2/C3): `sdk_/_qc_recipe/_recipe.py` (`QcRecipe.from_layout`),
  `sdk_/_qc_recipe/_runner.py` (`run_qc(..., qc_output_dir=None)`),
  `schema/_schema_cache.py` (`MeasurementSchema.from_layout`),
  `_cli/_cli_error_outputs.py` (update `CurationLabels.load` caller)
- Test: `tests/unit/gui/results_viewer/test_curation_labels.py` (extend — standalone write target)

**Interfaces:**
- Consumes: `OutputRoot.layout` accessors (Task 1/2).
- Produces: `CurationLabels.load(layout: BundleLayout, master_df)` and `ReviewState.load(layout: BundleLayout)` (both take a `BundleLayout`, not a raw root). All curation/qc/error writes land under `layout.qc_dir` / `layout.errors_dir`, correct for both full-run and standalone roots.
- **Governing rule (review):** never pass `output_root.root` into a helper that internally joins `deliverables/`/`qc/` — in standalone mode `root` is the deliverables folder and the call double-joins. Route through `layout`, or give the consumer a `from_layout`/`qc_output_dir` entry point.

- [ ] **Step 1: Write the failing test (standalone curation persistence)**

```python
# tests/unit/gui/results_viewer/test_curation_labels.py  (append)
def test_curation_writes_into_deliverables_qc_for_standalone(tmp_path):
    import polars as pl
    from phenotypic.gui.results_viewer._output_root import OutputRoot
    from phenotypic.gui.results_viewer._curation_labels import CurationLabels

    base = tmp_path / "bundle" / "deliverables"
    base.mkdir(parents=True)
    df = pl.DataFrame(
        {"Metadata_Dataset": ["p1"], "Metadata_ImageFile": ["img001"], "Object_Label": [1]}
    )
    df.write_parquet(base / "master_measurements.parquet")
    df.write_parquet(base / "measurements.parquet")

    root = OutputRoot.discover(base)
    labels = CurationLabels.load(root.layout, root.master_df)  # NEW: takes BundleLayout
    labels.mark("img001", "1", "debris")

    # Durable store must live INSIDE the bundle, not at base.parent/qc.
    assert (base / "qc" / "curation_labels.parquet").is_file()
    assert not (base.parent / "qc").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/gui/results_viewer/test_curation_labels.py -k standalone -v`
Expected: FAIL — `CurationLabels.load` takes a path, writes to `base/deliverables/qc` (double-join) or `base.parent/qc`.

- [ ] **Step 3: Apply the routing edits**

Change `CurationLabels` to hold a `BundleLayout` instead of `root`:
- `load(cls, layout: BundleLayout, master_df)` — accept a `BundleLayout`; store `self._layout = layout`. (GUI passes `output_root.layout`; the CLI passes `BundleLayout.detect(output_dir)` — see the CLI-caller sub-step below for C3.)
- Replace internal path properties (`_curation_labels.py:198,202,206,210,267,277,292,316,671,692`):
  | Line | Before | After |
  |---|---|---|
  | 198 | `curation_labels_parquet_path(self.root)` | `self._layout.curation_labels_parquet` |
  | 202 | `custom_categories_json_path(self.root)` | `self._layout.custom_categories_json` |
  | 206 | `measurements_parquet_path(self.root)` | `self._layout.mirror_parquet` |
  | 210 | `measurements_csv_path(self.root)` | `self._layout.mirror_csv` |
  | 267 | `custom_categories_json_path(root)` | `layout.custom_categories_json` |
  | 277 | `curation_labels_parquet_path(root)` | `layout.curation_labels_parquet` |
  | 292 | `measurements_parquet_path(root)` | `layout.mirror_parquet` |
  | 316 | `master_measurements_parquet_path(root)` | `layout.master_parquet` |
  | 671 | `errors_dir(self.root)` | `self._layout.errors_dir` |
  | 692 | `error_category_parquet_path(self.root, token)` | `self._layout.error_category_parquet(token)` |
  (Thread `layout = output_root.layout` through the `load`/`_read_*` classmethods that currently take `root`.)

Change `ReviewState.load(output_root_path)` → `ReviewState.load(layout: BundleLayout)` storing `layout`, and `_review_state.py:143` `qc_review_state_path(Path(output_root_path))` → `layout.qc_review_state_path`. Update both call sites: `_error_tab/_data.py:135` `ReviewState.load(output_root.root)` → `ReviewState.load(output_root.layout)`; `_qc_tab/review/_callbacks.py:650` likewise.

QC review data (`_qc_tab/review/_data.py`):
- `96` `qc_summary_parquet_path(Path(output_root.root))` → `output_root.layout.qc_summary_parquet`
- `101` `qc_members_parquet_path(Path(output_root.root))` → `output_root.layout.qc_members_parquet`
- `522/535` `measurements_parquet_path(root)` / `master_measurements_parquet_path(root)` → `output_root.layout.mirror_parquet` / `output_root.layout.master_parquet`

Error callbacks (`_error_tab/_callbacks.py:293,294,306,613`): route through new `layout` accessors — add `error_analysis_parquet`, `error_analysis_csv`, `error_analysis_html`, `verified_parquet`, `analysis_parquet`, `analysis_csv` properties to `BundleLayout` (mirror the qc accessors, anchored on `deliverables_base`; reuse the existing `ERROR_ANALYSIS_*`, `VERIFIED_PARQUET`, `ANALYSIS_*` constants). Then:
- `293` `error_analysis_parquet_path(root)` → `output_root.layout.error_analysis_parquet`
- `294` `error_analysis_csv_path(root)` → `output_root.layout.error_analysis_csv`
- `306` `verified_parquet_path(Path(output_root.root))` → `output_root.layout.verified_parquet`
- `613` `error_analysis_html_path(Path(output_root.root))` → `output_root.layout.error_analysis_html`

`_filtered_state.py:240` `measurements_parquet_path(root)`: this module is now a utility/constants module (per gui/CLAUDE.md). Confirm the single caller with `grep -rn "_filtered_state\|measurements_parquet_path" src/phenotypic/gui/results_viewer` and route it through `output_root.layout.mirror_parquet`.

**C1 — `QcRecipe`/`MeasurementSchema` double-join (CRITICAL).** Both internally call `deliverables_dir(output_root)` (`_recipe.py:327` `pipeline_json_path`, `_schema_cache.py:95`). Passing `output_root.root` works for full runs but in standalone `root == deliverables_base`, so `deliverables_dir(deliverables_base)` → `deliverables_base/deliverables/...` (verified double-join → silent empty recipe / empty columns dropdown). Fix by adding `BundleLayout`-aware entry points that read from `deliverables_base` directly:
- In `_qc_recipe/_recipe.py`: add `@classmethod def from_layout(cls, layout: BundleLayout) -> "QcRecipe"` that reads `layout.pipeline_config_path` (already `deliverables_base/pipeline.json`) instead of `pipeline_json_path(output_root)`. Also add a `from_layout`-style path for `migrate_from_sidecar` (review W6, `_recipe.py:714`) reading `layout.deliverables_base / VIEWER_CACHE_DIRNAME / QC_RECIPE_FILENAME` — or skip the sidecar migration entirely when `layout.output_root is None` (standalone bundles never have a legacy sidecar).
- In `schema/_schema_cache.py`: add `@classmethod def from_layout(cls, layout: BundleLayout)` (or accept `deliverables_base` directly) that resolves measurements from `layout.mirror_parquet`/`layout.master_parquet` rather than `deliverables_dir(self.output_root)`.
- `_app.py:237,238,239` and `_layout.py:399,418`: replace `QcRecipe.migrate_from_sidecar(Path(output_root.root))` / `QcRecipe.load(Path(output_root.root))` / `_load_qc_pipeline(Path(output_root.root))` / `MeasurementSchema(output_root=Path(output_root.root))` / `QcRecipe.load(Path(output_root.root))` with the `*.from_layout(output_root.layout)` variants. `_load_qc_pipeline` itself should read `output_root.layout.pipeline_config_path`.
- `CurationLabels.load(output_root.root, ...)` (`_app.py:198`) → `CurationLabels.load(output_root.layout, ...)`.

**C2 — `run_qc` writes wrong dir (CRITICAL).** `_qc_tab/review/_callbacks.py:717` `run_qc(frame, pipeline, Path(output_root.root))`; `run_qc` (`_runner.py:149`) calls `qc_summary_parquet_path(output_dir)` etc. In standalone this double-joins and the broad `except` swallows the failure. Fix: add an optional `qc_output_dir: Path | None = None` param to `run_qc`; when provided, write `qc_output_dir / QC_SUMMARY_PARQUET` (and members/config) directly instead of via `qc_*_parquet_path(output_dir)`. The GUI call becomes `run_qc(frame, pipeline, Path(output_root.root), qc_output_dir=output_root.layout.qc_dir)`. The CLI caller passes nothing (unchanged behaviour). Also fix `_qc_tab/_callbacks.py:1018` `frame` write path if it reuses `qc_summary_parquet_path` (verify).

**C3 — CLI caller of `CurationLabels.load` (CRITICAL).** After the signature change, `_cli/_cli_error_outputs.py:59` `CurationLabels.load(output_dir, master_df)` breaks (it passes a `Path`). Update it to `CurationLabels.load(BundleLayout.detect(output_dir), master_df)` (the CLI always has a real output root → `detect` resolves case 2). Add `_cli_error_outputs.py` to this task's commit.

**C4 — hand-joined dead paths (CRITICAL).** `_qc_tab/_callbacks.py:1018-1020` hand-joins `root / "qc.parquet"` and `root / "qc_summary.json"` — not through any helper, and (per review) not read anywhere downstream. Investigate with `grep -rn "qc.parquet\|qc_summary.json" src/phenotypic/gui`; if unused, delete the block; if used, route through `output_root.layout.qc_dir`. Do not leave a raw `output_root.root` join here.

`_tile_routes.py:69` `output_root.results_dir / dataset` — the `None` guard is added in Task 4 (review W2); Task 7 then replaces this gate with the capability check ("overlay exists OR hdf exists").

`timeline_view/_thumb_routes.py:72,77` and `_tile_routes.py:140` use `output_root.root / VIEWER_CACHE_DIRNAME` — replace with `output_root.viewer_cache_dir` (added in Task 4) so the standalone cache path stays inside the bundle. `_layout.py:172` displays `str(output_root.root)` as the header subtitle — this is now the deliverables path in standalone mode; acceptable (informative), no change required, but note it in the commit body.

- [ ] **Step 4: Run the targeted + full viewer suites**

Run: `uv run pytest tests/unit/gui/results_viewer -q`
Expected: PASS. Then **audit EVERY `output_root.root` / `.root` use** (review Q1 — the path-helper grep alone misses sites like `QcRecipe.load`/`MeasurementSchema`/`run_qc`/`migrate_from_sidecar` that take a root and join internally):
- `grep -rn "output_root\.root\|\.root\b" src/phenotypic/gui/results_viewer` — for each hit, confirm it either (a) is a display/log string, or (b) routes through `output_root.layout` / a `from_layout`/`qc_output_dir` entry point. NO surviving call may pass `output_root.root` into a function that internally calls `deliverables_dir(...)` / `qc_dir(...)` / `qc_*_parquet_path(...)` / `pipeline_json_path(...)`.
- `grep -rn "deliverables_dir\|qc_dir\|qc_summary_parquet_path\|qc_members_parquet_path\|pipeline_json_path" src/phenotypic/gui src/phenotypic/sdk_/_qc_recipe src/phenotypic/schema` — confirm GUI-reached call sites receive a real output root (full run) or a `BundleLayout`-aware entry point (standalone), never `deliverables_base`.

- [ ] **Step 5: Type-check + CLI regression (C3 caller)**

Run: `uv run mypy src/phenotypic/gui/results_viewer src/phenotypic/sdk_/_qc_recipe src/phenotypic/schema`
Run: `uv run pytest tests/unit/_cli -k "error or curation or reemit" -q` (catches the `_cli_error_outputs.py` `CurationLabels.load` signature change)
Expected: no new errors / PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/results_viewer src/phenotypic/sdk_/_qc_recipe src/phenotypic/schema/_schema_cache.py src/phenotypic/_cli/_cli_error_outputs.py tests/unit/gui/results_viewer/test_curation_labels.py
git commit -m "refactor(gui): route deliverables/qc paths through BundleLayout; from_layout entry points"
```

---

## Phase 3 — Pixel-Fidelity Tiering

### Task 6: `crop_hdf_rgb` + layer cache

**Files:**
- Modify: `src/phenotypic/gui/_shared/tiles.py` (replace the `crop_overlay` TODO with a real sibling)
- Test: `tests/unit/gui/_shared/test_tiles.py` (extend)

**Interfaces:**
- Consumes: `h5py` (direct single-layer read of `/layers/<name>`); renderer helpers `_normalize_to_uint8`, `_label_map_to_rgb` (from `gui/builder/_image_renderer.py`). Does NOT use `load_image_from_hdf` (would load all layers — review W5).
- Produces:
  ```python
  LayerName = Literal["rgb", "detect_mat", "objmap"]
  def crop_hdf_rgb(h5_path: Path, layer: str, center_rr: float, center_cc: float,
                   size: int, mtime_ns: int, *, dim_alpha: float = 0.0,
                   bbox: tuple[float, float, float, float] | None = None) -> bytes
  ```

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/gui/_shared/test_tiles.py  (append)
def test_crop_hdf_rgb_returns_full_res_png(tmp_path):
    import numpy as np
    from PIL import Image as PILImage
    import io
    from phenotypic import Image
    from phenotypic.gui._shared.tiles import crop_hdf_rgb

    # Build a tiny image with a distinctive RGB layer and save to HDF.
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgb[10:30, 10:30] = (255, 0, 0)
    img = Image(arr=rgb)
    h5 = tmp_path / "img001.h5"
    img.save2hdf5(str(h5))

    out = crop_hdf_rgb(h5, "rgb", center_rr=20, center_cc=20, size=16,
                       mtime_ns=h5.stat().st_mtime_ns)
    crop = PILImage.open(io.BytesIO(out)).convert("RGB")
    assert crop.size == (16, 16)
    # Centre pixel falls inside the red square.
    assert crop.getpixel((8, 8)) == (255, 0, 0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/gui/_shared/test_tiles.py -k crop_hdf_rgb -v`
Expected: FAIL — `crop_hdf_rgb` undefined.

- [ ] **Step 3: Implement `crop_hdf_rgb` + a cached layer loader**

Add to `tiles.py` (factor the shared paste/dim logic out of `crop_overlay` into `_paste_centered_crop(source_rgb_ndarray, ...)` so both croppers reuse it — DRY):

```python
from typing import Literal

_HDF_LAYER_CACHE_SIZE = 4  # full-res layers are heavier than overlay PNGs


@functools.lru_cache(maxsize=_HDF_LAYER_CACHE_SIZE)
def _load_hdf_layer_rgb(path: str, mtime_ns: int, layer: str) -> PILImage.Image:
    """Decode one HDF layer to an RGB PIL image and cache it.

    rgb -> raw uint8; detect_mat -> contrast-normalised greyscale promoted to RGB;
    objmap -> label2rgb colourisation.

    Memory discipline (review W5 / spec Section 4): read ONLY the requested
    ``/layers/<name>`` dataset via h5py — do NOT call ``load_image_from_hdf``,
    which eagerly materialises every layer (rgb+gray+detect_mat+objmap, hundreds
    of MB) just to discard all but one.
    """
    del mtime_ns  # cache-key only
    import h5py

    from phenotypic.gui.builder._image_renderer import (
        _normalize_to_uint8, _label_map_to_rgb,
    )

    with h5py.File(path, "r") as fh:
        # Modern layout is /layers/<name>; legacy flat layout is /<name>.
        grp = fh["layers"] if "layers" in fh else fh
        if layer not in grp:
            raise KeyError(f"HDF {path} has no layer {layer!r}")
        arr = np.asarray(grp[layer][:])

    if layer == "rgb":
        rgb = arr.astype(np.uint8)
    elif layer == "objmap":
        rgb = _label_map_to_rgb(arr)
    else:  # detect_mat / gray-like float layer
        gray = _normalize_to_uint8(arr)
        rgb = np.stack([gray] * 3, axis=-1)
    return PILImage.fromarray(rgb, mode="RGB")


def crop_hdf_rgb(h5_path, layer, center_rr, center_cc, size, mtime_ns,
                 *, dim_alpha=0.0, bbox=None) -> bytes:
    """Full-resolution sibling of :func:`crop_overlay`, sourcing a chosen HDF layer.

    Same centering/padding/dimming contract as :func:`crop_overlay`; the only
    difference is the pixel source (raw HDF layer vs baked overlay PNG).
    """
    source = _load_hdf_layer_rgb(str(h5_path), mtime_ns, layer)
    return _crop_pil_source(source, center_rr, center_cc, size,
                            dim_alpha=dim_alpha, bbox=bbox)
```

Refactor `crop_overlay` to share the centering/paste/dim body: extract the code from `crop_overlay`'s `src_width,... return buf.getvalue()` (lines 183-229) into:
```python
def _crop_pil_source(source: PILImage.Image, center_rr, center_cc, size,
                     pad_value=(0, 0, 0), *, dim_alpha=0.0, bbox=None) -> bytes:
    # (move the existing crop/paste/dim/encode body here verbatim)
    ...
```
and have `crop_overlay` call `_crop_pil_source(_load_overlay_rgb(...), ...)`. This keeps both croppers byte-identical in geometry.

- [ ] **Step 4: Run to verify pass + overlay regression**

Run: `uv run pytest tests/unit/gui/_shared/test_tiles.py -v`
Expected: PASS, including the pre-existing `crop_overlay` tests (geometry unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_shared/tiles.py tests/unit/gui/_shared/test_tiles.py
git commit -m "feat(gui): crop_hdf_rgb full-res layer cropper sharing crop_overlay geometry"
```

---

### Task 7: `crop_colony` dispatcher + route wiring

**Files:**
- Modify: `src/phenotypic/gui/_shared/tiles.py` (`crop_colony`)
- Modify: `src/phenotypic/gui/results_viewer/_tile_routes.py` (capability gate + layer param)
- Modify: `src/phenotypic/gui/results_viewer/colony_view/_cropper.py` (re-export `crop_colony`)
- Modify: `src/phenotypic/gui/results_viewer/timeline_view/_thumb_routes.py`
- Test: `tests/unit/gui/_shared/test_tiles.py` (extend — dispatch)

**Interfaces:**
- Consumes: `OutputRoot` (`hdf_path`, `overlay_path`, `has_overlay`), `crop_hdf_rgb`/`crop_overlay`.
- Produces:
  ```python
  def crop_colony(output_root, dataset, stem, layer, center_rr, center_cc, size,
                  *, dim_alpha=0.0, bbox=None) -> bytes | None
  ```
  Returns `None` when neither an HDF nor an overlay exists (route → 404).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/gui/_shared/test_tiles.py  (append)
def test_crop_colony_prefers_hdf_falls_back_to_overlay(tmp_path, monkeypatch):
    from phenotypic.gui._shared import tiles

    calls = {}
    monkeypatch.setattr(tiles, "crop_hdf_rgb", lambda *a, **k: calls.setdefault("hdf", True) or b"H")
    monkeypatch.setattr(tiles, "crop_overlay", lambda *a, **k: calls.setdefault("ovl", True) or b"O")

    class FakeRoot:
        def __init__(self, hdf, overlay_ok):
            self._hdf = hdf
            self._overlay_ok = overlay_ok
        def hdf_path(self, ds, stem):
            return tmp_path / "x.h5" if self._hdf else None
        def has_overlay(self, ds, stem):
            return self._overlay_ok
        def overlay_path(self, ds, stem):
            return tmp_path / "x.png"

    if True:  # hdf present -> hdf path
        (tmp_path / "x.h5").write_bytes(b"")
        assert tiles.crop_colony(FakeRoot(True, True), "p", "s", "rgb", 1, 1, 8) == b"H"
    calls.clear()
    assert tiles.crop_colony(FakeRoot(False, True), "p", "s", "rgb", 1, 1, 8) == b"O"
    assert tiles.crop_colony(FakeRoot(False, False), "p", "s", "rgb", 1, 1, 8) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/gui/_shared/test_tiles.py -k crop_colony -v`
Expected: FAIL — `crop_colony` undefined.

- [ ] **Step 3: Implement `crop_colony` + wire routes**

```python
def crop_colony(output_root, dataset, stem, layer, center_rr, center_cc, size,
                *, dim_alpha=0.0, bbox=None):
    """Tier the crop source per-image: full-res HDF layer when available, else overlay.

    Returns PNG bytes, or None when neither source exists (caller serves 404).
    """
    h5 = output_root.hdf_path(dataset, stem)
    if h5 is not None:
        return crop_hdf_rgb(h5, layer, center_rr, center_cc, size,
                            os.stat(h5).st_mtime_ns, dim_alpha=dim_alpha, bbox=bbox)
    if output_root.has_overlay(dataset, stem):
        png = output_root.overlay_path(dataset, stem)
        return crop_overlay(png, center_rr, center_cc, size, dim_alpha=dim_alpha, bbox=bbox)
    return None
```

In `_tile_routes.py`: replace the `results_dir / dataset` existence gate (line 69) with a capability check that works standalone — accept the request when `output_root.hdf_path(ds, stem) is not None or output_root.has_overlay(ds, stem)`. Thread a `layer` query-param (default `"rgb"`) into the DZI source selection (full DZI layer support is Task 9; for the flat crop route here, pass `layer` to `crop_colony`). Replace the direct `crop_overlay(...)` call in the crop-serving route with `crop_colony(output_root, dataset, stem, layer, ...)` and 404 on `None`.

In `colony_view/_cropper.py`: add `crop_colony` to the re-export (`__all__`) and import.

In `timeline_view/_thumb_routes.py`: switch the thumb source to `crop_colony(output_root, ds, stem, layer, ...)` (thumbnails default to `layer="rgb"`); keep the existing downscale-cache wrapper. Use `output_root.cache_dir.parent`-based cache root (Task 5) rather than `output_root.root`.

- [ ] **Step 4: Run + manual route smoke**

Run: `uv run pytest tests/unit/gui/_shared/test_tiles.py tests/unit/gui/results_viewer -k "crop or tile or thumb" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_shared/tiles.py src/phenotypic/gui/results_viewer/_tile_routes.py src/phenotypic/gui/results_viewer/colony_view/_cropper.py src/phenotypic/gui/results_viewer/timeline_view/_thumb_routes.py tests/unit/gui/_shared/test_tiles.py
git commit -m "feat(gui): per-image crop_colony dispatcher (HDF full-res with overlay fallback)"
```

---

### Task 8: Layer toggle UI (gated on `has_results`)

**Files:**
- Modify: colony_view + timeline_view layout/callbacks (segmented control + store), `_tile_routes.py` (consume layer)
- Modify: `src/phenotypic/gui/FEATURES.md`
- Test: `tests/unit/gui/results_viewer/...` (toggle render gate) + extend the e2e in Phase 5

**Interfaces:**
- Consumes: `OutputRoot.has_results`.
- Produces: a `dcc.Store`/segmented `RadioItems` (`id` from the view's `_ids.py`) holding the active layer (`"rgb"|"detect_mat"|"objmap"`), default `"rgb"`; rendered only when `output_root.has_results`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/gui/results_viewer/test_layer_toggle.py  (create)
def test_layer_toggle_hidden_in_standalone(tmp_path):
    from phenotypic.gui.results_viewer.colony_view._layout import build_layer_toggle  # NEW
    class R: has_results = False
    assert build_layer_toggle(R()) is None  # or a hidden component


def test_layer_toggle_shown_in_full_run(tmp_path):
    from phenotypic.gui.results_viewer.colony_view._layout import build_layer_toggle
    class R: has_results = True
    comp = build_layer_toggle(R())
    assert comp is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/gui/results_viewer/test_layer_toggle.py -v`
Expected: FAIL — `build_layer_toggle` undefined.

- [ ] **Step 3: Implement the toggle**

Add `build_layer_toggle(output_root)` returning `None` when `not output_root.has_results`, else a `dbc`/`dcc` segmented control over `["rgb", "detect_mat", "objmap"]` (labels: "RGB", "Enhanced", "Labels") with id from `_ids.py` (`LAYER_TOGGLE`, `STORE_ACTIVE_LAYER`). Place it in the colony-view and timeline-view toolbars. Add a callback that writes the selection into the store, and thread the store value into the tile/crop URLs as the `layer` query-param (the crop routes already accept it from Task 7). Use `_design.py` tokens for styling — no inline hex.

- [ ] **Step 4: FEATURES.md rows**

Add rows under the results-viewer section for: "Pixel layer toggle (rgb/detect_mat/objmap)" and "the toggle is `✅ shipping` only when results/ present". Each `✅ shipping` row needs a `Test ref` resolving to `tests/unit/gui/results_viewer/test_layer_toggle.py::test_layer_toggle_shown_in_full_run` (and the e2e from Phase 5).

- [ ] **Step 5: Run unit + features gate**

Run: `uv run pytest tests/unit/gui/results_viewer/test_layer_toggle.py -v`
Run: `uv run python scripts/check_features_md.py` (or the pre-commit hook) — expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/results_viewer/colony_view src/phenotypic/gui/results_viewer/timeline_view src/phenotypic/gui/results_viewer/_tile_routes.py src/phenotypic/gui/FEATURES.md tests/unit/gui/results_viewer/test_layer_toggle.py
git commit -m "feat(gui): per-layer pixel toggle, gated on results/ availability"
```

---

### Task 9: DZI deep-zoom layer dimension

**Files:**
- Modify: `_tile_routes.py` (DZI source + cache key), `_dzi_tiler` usage
- Test: `tests/unit/gui/results_viewer/test_tile_routes.py` (extend — layer-keyed cache)

**Interfaces:**
- Consumes: `crop`/render-to-PNG of a selected HDF layer (Task 6 loader), `_dzi_tiler.tile(png, outdir)`.
- Produces: DZI cache under `.viewer_cache/dzi/<ds>/<stem>/<layer>/`; standalone falls back to tiling the overlay PNG (layer ignored).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/gui/results_viewer/test_tile_routes.py  (append)
def test_dzi_cache_path_includes_layer(tmp_path):
    from phenotypic.gui.results_viewer._tile_routes import _dzi_cache_dir_for  # NEW helper
    p_rgb = _dzi_cache_dir_for(tmp_path, "plate1", "img001", "rgb")
    p_obj = _dzi_cache_dir_for(tmp_path, "plate1", "img001", "objmap")
    assert p_rgb != p_obj
    assert p_rgb.parts[-1] == "rgb" and p_obj.parts[-1] == "objmap"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/gui/results_viewer/test_tile_routes.py -k dzi_cache -v`
Expected: FAIL — `_dzi_cache_dir_for` undefined.

- [ ] **Step 3: Implement layer-aware DZI**

Add `_dzi_cache_dir_for(cache_root, dataset, stem, layer)` returning `cache_root / dataset / stem / layer`. When `output_root.hdf_path(ds, stem)` is present and `layer != "overlay"`: render the full layer to a temp PNG via `_load_hdf_layer_rgb(...).save(...)`, tile that. Else tile the overlay PNG (existing path), keyed under a `overlay` layer dir. Read the `layer` query-param (default `"rgb"` when `has_results`, `"overlay"` otherwise).

- [ ] **Step 4: Run tests + a manual OSD smoke**

Run: `uv run pytest tests/unit/gui/results_viewer/test_tile_routes.py -q`
Expected: PASS. (OSD visual smoke happens in the Phase 5 e2e.)

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_tile_routes.py tests/unit/gui/results_viewer/test_tile_routes.py
git commit -m "feat(gui): layer-keyed DZI deep-zoom from HDF, overlay fallback"
```

---

## Phase 4 — Mode Signaling

### Task 10: Standalone/Full-run badge + sidebar classifier

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_layout.py` (header badge)
- Modify: `src/phenotypic/gui/shell/_classifier.py` (distinguish bundle vs full run)
- Modify: `src/phenotypic/gui/FEATURES.md`
- Test: `tests/unit/gui/results_viewer/test_mode_badge.py` (create); `tests/unit/gui/shell/test_classifier.py` (extend)

**Interfaces:**
- Consumes: `OutputRoot.has_results`; the classifier's existing directory-stat logic.
- Produces: `build_mode_badge(output_root) -> Component` ("Full run" vs "Standalone bundle"); classifier returns a `has_results`-aware label.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/gui/results_viewer/test_mode_badge.py  (create)
def test_mode_badge_text_by_capability():
    from phenotypic.gui.results_viewer._layout import build_mode_badge  # NEW
    class Full: has_results = True
    class Bundle: has_results = False
    assert "Full run" in str(build_mode_badge(Full()))
    assert "Standalone" in str(build_mode_badge(Bundle()))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/gui/results_viewer/test_mode_badge.py -v`
Expected: FAIL — `build_mode_badge` undefined.

- [ ] **Step 3: Implement badge + classifier**

`build_mode_badge(output_root)` → a small `dbc.Badge`/`html.Span` reading `Full run` (results present) or `Standalone bundle` (deliverables-only), styled via `_design.py` tokens. Mount it in the results-viewer header (`_layout.py`). In `shell/_classifier.py`, where a directory is classified for the sidebar, add a check: if `deliverables/master_measurements.parquet` exists but `results/` does not, label it as a deliverables bundle; else full run. Keep the existing badge contrast rules (DESIGN.md).

- [ ] **Step 4: FEATURES.md row + run**

Add a FEATURES.md row "Results-viewer mode badge (full-run vs standalone bundle)" with a `Test ref` to `test_mode_badge.py::test_mode_badge_text_by_capability`.
Run: `uv run pytest tests/unit/gui/results_viewer/test_mode_badge.py tests/unit/gui/shell/test_classifier.py -v` and `uv run python scripts/check_features_md.py`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_layout.py src/phenotypic/gui/shell/_classifier.py src/phenotypic/gui/FEATURES.md tests/unit/gui/results_viewer/test_mode_badge.py tests/unit/gui/shell/test_classifier.py
git commit -m "feat(gui): standalone-bundle vs full-run mode badge + sidebar classifier"
```

---

## Phase 5 — Docs + End-to-End

### Task 11: Docs/CLAUDE.md sweep + Playwright e2e

**Files:**
- Modify: `src/phenotypic/gui/CLAUDE.md`, root `CLAUDE.md`, QC `CLAUDE.md`, `_io_constants.py` qc-helper docstrings
- Test: `tests/e2e/gui/test_deliverables_standalone_e2e.py` (create — confirm the e2e dir with `uv run pytest --collect-only -k e2e`)

**Interfaces:**
- Consumes: the full stack from Tasks 1-10.
- Produces: docs reflect `qc/` under `deliverables/`; a live-browser e2e proving standalone parity.

- [ ] **Step 1: Write the e2e (Playwright, offscreen Qt not needed — this is web)**

```python
# tests/e2e/gui/test_deliverables_standalone_e2e.py  (create)
# Build a deliverables-only bundle on disk (master + mirror + overlays + qc),
# launch the results viewer pointed at the deliverables/ folder, then drive:
#   1. measurements table renders rows
#   2. open a colony, mark an error category, reload -> label persists
#   3. QC review tab renders tiles from deliverables/qc/qc_members.parquet
#   4. layer toggle is ABSENT (no results/)
# Assert curation_labels.parquet was written under <bundle>/deliverables/qc/.
# Follow the existing results-viewer e2e harness (Playwright MCP / pytest-playwright);
# mirror tests/e2e/gui/<existing results-viewer e2e>.py for app-launch + teardown.
```
> Use the existing results-viewer e2e as the template for launching the Werkzeug hub and tailing the viewer log (per the "verify Dash callbacks in a live browser" rule). Build the QC fixtures (`qc_summary.parquet`, `qc_members.parquet`) with the same schema `run_qc` writes.

- [ ] **Step 2: Run to verify it fails (or errors on missing app wiring)**

Run: `uv run pytest tests/e2e/gui/test_deliverables_standalone_e2e.py -v`
Expected: FAIL initially (assertions unmet) — iterate until green.

- [ ] **Step 3: Update docs**

- `gui/CLAUDE.md`: change the "Output layout — `deliverables/`" paragraph that says `QC_DIRNAME` (`qc/`) "stay at the output-dir root" — qc now lives under `deliverables/qc/`. Document the standalone-bundle mode + layer toggle in "Common gotchas".
- Root `CLAUDE.md`: update the "Output location — `deliverables/`" gotcha to note `qc/` relocated under `deliverables/` and that the GUI can open a deliverables bundle standalone.
- QC `CLAUDE.md` (`sdk_/_qc_recipe` or wherever QC paths are documented): qc artefacts now under `deliverables/qc/`.
- `_io_constants.py`: update `qc_*` helper docstrings (`<output>/qc/...` → `<output>/deliverables/qc/...`).

- [ ] **Step 4: Full regression + lint/type**

Run: `uv run pytest tests/unit/gui tests/unit/tools_ tests/unit/_cli -q`
Run: `uv run pytest tests/e2e/gui/test_deliverables_standalone_e2e.py -v`
Run: `uv run mypy src/phenotypic && uv run ruff check --fix`
Expected: all PASS / clean.

- [ ] **Step 5: Regenerate GUI tutorial screenshots (chrome changed)**

Run: `uv run python scripts/capture_gui_tutorial_screenshots.py`
Commit ALL refreshed PNGs (do not cherry-pick — per CLAUDE.md).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/CLAUDE.md CLAUDE.md src/phenotypic/sdk_/_io_constants.py tests/e2e/gui/test_deliverables_standalone_e2e.py docs
git add -A  # refreshed screenshots
git commit -m "docs+test(gui): standalone deliverables bundle e2e + qc-relocation doc sweep"
```

---

## Self-Review

**Spec coverage:**
- Section 1 (BundleLayout) → Task 1 ✅
- Section 1 (qc_dir relocation, CLI follows) → Task 2 ✅
- Section 2 (discovery refactor, drop results/ gate, data-driven datasets, capability-aware backfill, cache location) → Task 4 ✅; path-routing consequence → Task 5 ✅
- Section 3 (qc relocation + MOVE migration on discover + finalize, split-state hazard) → Task 2 + Task 3 + Task 4 (migrate on discover) ✅
- Section 4 (crop_hdf_rgb, crop_colony dispatcher, layer toggle, DZI layer dimension) → Tasks 6, 7, 8, 9 ✅
- Section 5 (mode badge, per-viewer fidelity via per-image dispatch, classifier) → Task 10 ✅ (fidelity hint falls out of Task 7 per-image dispatch; a dedicated per-tile "Overlay" pill is folded into the badge — if a per-tile indicator is wanted, add it as a Task 8 sub-step)
- Testing (unit + discovery + e2e + fixtures) → Tasks 1-11 ✅
- CI gates (FEATURES.md, screenshots) → Tasks 8, 10, 11 ✅
- Docs sweep → Task 11 ✅

**Placeholder scan:** No "TBD"/"handle edge cases" steps; every code step shows code. Two deliberate *verification* asks remain (confirm `QcRecipe`/`MeasurementSchema` path expectations in Task 5; confirm exact test dirs) — these are grep-confirm steps, not implementation placeholders, and are flagged inline.

**Type consistency:** `BundleLayout` accessor names are used identically across Tasks 1/2/4/5 (`qc_summary_parquet`, `curation_labels_parquet`, `mirror_parquet`, `master_parquet`, `error_category_parquet(token)`, `overlay_path`, `hdf_path`). `crop_colony`/`crop_hdf_rgb` signatures match across Tasks 6/7/9. `has_results`/`hdf_path` consistent across Tasks 1/4/7/8/10.

**Known follow-ups (not blockers):** the `error_analysis_*`/`verified`/`analysis_*` `BundleLayout` accessors are introduced in Task 5 (the error-tab task) rather than Task 1 to keep Task 1's surface to the tested set.

**Independent-review fixes incorporated (see "Review Revisions" near the top):** C1 (`QcRecipe`/`MeasurementSchema` double-join → `from_layout` entry points, Task 5); C2 (`run_qc` double-join → `qc_output_dir` param, Task 5); C3 (CLI `CurationLabels.load` caller updated, Task 5); C4 (hand-joined dead paths `_qc_tab/_callbacks.py:1018-1020`, Task 5); C5 (viewer-cache read-only fallback + `viewer_cache_dir`, Task 4); W2 (`results_dir` `None` guard moved into Task 4); W5 (single-layer h5py read, Task 6); W1 (atomic-rename note, Task 3); W6/Q1 (`migrate_from_sidecar` + full `.root` audit, Task 5). The verdict was GO-WITH-CHANGES; all CRITICAL findings are now addressed in the plan text.
