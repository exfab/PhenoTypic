# REMBI Metadata Reorganization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PhenoTypic's metadata model explicitly REMBI-shaped — classify every metadata field by REMBI module, add the missing Study module, always emit a REMBI manifest, and rename the metadata column namespace to per-enum self-describing prefixes while eliminating stringly-typed metadata access.

**Architecture:** Reuse the existing `schema/` classification machinery (`kind()`/`tier()` classmethods + per-member `Entry` overrides + a single resolver + a coverage gate) to add a parallel `rembi_module()` axis. A pure `build_rembi_manifest()` folds the per-colony measurements mirror up to each REMBI module's scope and a best-effort writer drops `deliverables/rembi.yaml` in finalize. The category-prefix rename (`Metadata_` → `Metadata<Topic>_`) is centralized behind three `sdk_` helpers so the ~57 hardcoded-string sites collapse to one source of truth, and the ad-hoc `Metadata_ImageFile` column is retired in favor of the canonical `Metadata_ImageName` (+ `Metadata_FileSuffix`).

**Tech Stack:** Python 3, pydantic v2, polars, pandas, PyYAML, pytest, `uv` (sole runner). Source under `src/phenotypic/`, tests under `tests/`.

**Source spec:** `docs/superpowers/docs/2026-06-29-rembi-metadata-reorg/design.md`

## Global Constraints

- **Runner:** never use bare `python`/`pip`. Use `uv run <cmd>` (e.g. `uv run pytest ...`, `uv run python -c "..."`). Sync env with `uv sync --group dev --group test-qt --group docs --extra gui --extra napari` for the GUI/Qt tests.
- **Branch:** all work stays on `fix/pipeline-output-cleanup` (do NOT create a new branch). Commit frequently.
- **Operations are keyword-only pydantic models;** `op.apply(image)` not `op(image)`. Not relevant to most tasks but applies if touching operations.
- **`Entry` is the only legal `MeasurementInfo` member value** (frozen dataclass, `_measurement_info.py`); raw tuples raise `TypeError`. Author only `label`/`desc` on new members; leave `bio_desc=""`/`image=None` (human-authored only).
- **Schema package is import-light:** modules in `schema/` import only stdlib + the sibling base (`_measurement_info`/`_tiers`). `_rembi.py` must follow this (no other `phenotypic` imports) to preserve the `phenotypic/__init__.py` load-order trick.
- **Closed value sets are `str, Enum`** (project convention). `REMBI_MODULE` is one.
- **Master-vs-mirror:** the manifest reads the post-applied **mirror** (`measurements.parquet`), never the master. Route any FINAL master write through `finalize_post_master_outputs`. Resolve paths via `phenotypic.sdk_` helpers, never hand-joined names.
- **Metadata is a recommended vocabulary, NOT a validator:** arbitrary metadata columns are always accepted; nothing fails a run for missing/unknown metadata. The manifest is best-effort and must never block finalize.
- **Category strings (Scheme B), verbatim:** `MetadataImage` (METADATA), `MetadataStudy` (STUDY_METADATA), `MetadataExperiment` (EXPERIMENT_METADATA), `MetadataGenetic` (GENETIC_METADATA), `MetadataSample` (SAMPLE_METADATA), `MetadataCondition` (CONDITION_METADATA), `MetadataCulture` (CULTURE_METADATA), `MetadataPlate` (PLATE_METADATA), `MetadataAcquisition` (ACQUISITION_METADATA).
- **REMBI module values, verbatim:** `Study, Biosample, SpecimenPreparation, ImageAcquisition, ImageData, AnalyzedData, Uncategorized`.

---

## File Structure

**Phase A — additive REMBI layer + manifest (back-compat, ships independently):**
- Create `src/phenotypic/schema/_rembi.py` — `REMBI_MODULE` enum + `header_to_module()` reverse index.
- Modify `src/phenotypic/schema/_measurement_info.py` — `Entry.rembi_module` field; `MeasurementInfo.rembi_module()` classmethod; `rembi_module_override` in `__new__`; `resolved_rembi_module` property.
- Modify `src/phenotypic/schema/__init__.py` — re-export `REMBI_MODULE`.
- Create `src/phenotypic/schema/_experimental_tags/_study.py` — `STUDY_METADATA` enum.
- Modify the 8 existing metadata enum files — add `rembi_module()`; CULTURE per-member overrides.
- Modify `src/phenotypic/schema/_experimental_tags/__init__.py` + `schema/__init__.py` — export `STUDY_METADATA`.
- Create `src/phenotypic/sdk_/_rembi_manifest.py` — `build_rembi_manifest()` (pure) + `write_rembi_manifest()`.
- Modify `src/phenotypic/sdk_/_io_constants.py` — `REMBI_MANIFEST_YAML`, `rembi_manifest_path()`.
- Modify `src/phenotypic/_cli/_cli_output_manager.py` — call the writer in `finalize_post_master_outputs`.
- Modify `src/phenotypic/phenotypicCLI.py` (or the arg parser) — `--study` flag.
- Modify `src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py` — `by_module()` view + REMBI column ordering.

**Phase B — category namespace migration (the rename):**
- Create `src/phenotypic/sdk_/_metadata_helpers.py` — `is_metadata_header()`, `metadata_category_prefixes()`, `metadata_category_for_label()`.
- Modify the 9 metadata enum files — change `category()` returns to Scheme B.
- Modify the 8 bare-prefix predicate sites (see Task B3).
- Modify `src/phenotypic/post/_utils.py` + the 4 post ops — schema-aware prefixing.
- Modify the 4 `Metadata_ImageFile` creation sites + ~6 `KEY_IMAGE_FILE` constant sites.
- Create `src/phenotypic/gui/results_viewer/_curation_migration.py` (or extend `_curation_labels.py`) — rename-on-load shim.
- Modify ~30 files replacing specific `Metadata_<X>` literals with enum refs.
- Modify ~128 test files (column-name updates).

---

# PHASE A — Additive REMBI layer + manifest

*Fully back-compat; touches none of the hardcoded-string sites. Categories stay `"Metadata"` until Phase B.*

## Task A1: `REMBI_MODULE` enum + classification machinery

**Files:**
- Create: `src/phenotypic/schema/_rembi.py`
- Modify: `src/phenotypic/schema/_measurement_info.py` (`Entry` ~60-101; `MeasurementInfo.__new__` ~374-398; classmethods ~351-359; properties ~412-420)
- Modify: `src/phenotypic/schema/__init__.py` (~18-24, `__all__`)
- Test: `tests/unit/schema/test_rembi_module.py`

**Interfaces:**
- Produces: `REMBI_MODULE(str, Enum)` with members `STUDY, BIOSAMPLE, SPECIMEN_PREP, IMAGE_ACQUISITION, IMAGE_DATA, ANALYZED_DATA, UNCATEGORIZED`; `Entry(..., rembi_module: REMBI_MODULE | None = None)`; `MeasurementInfo.rembi_module() -> REMBI_MODULE | None` (default `None`); `MeasurementInfo.resolved_rembi_module -> REMBI_MODULE` (total; fallback `ANALYZED_DATA`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/schema/test_rembi_module.py
from phenotypic.schema import Entry, REMBI_MODULE
from phenotypic.schema._tiers import IdentityInfo


class _ModEnum(IdentityInfo):
    @classmethod
    def category(cls) -> str:
        return "TestMod"

    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:
        return REMBI_MODULE.BIOSAMPLE

    PLAIN = Entry("Plain", "uses the enum-level module")
    OVERRIDDEN = Entry("Overridden", "per-member override",
                       rembi_module=REMBI_MODULE.SPECIMEN_PREP)


class _NoModEnum(IdentityInfo):
    @classmethod
    def category(cls) -> str:
        return "TestNoMod"

    LONELY = Entry("Lonely", "no module declared -> fallback")


def test_canonical_module_order():
    assert [m.value for m in REMBI_MODULE] == [
        "Study", "Biosample", "SpecimenPreparation", "ImageAcquisition",
        "ImageData", "AnalyzedData", "Uncategorized",
    ]


def test_enum_level_module():
    assert _ModEnum.PLAIN.resolved_rembi_module is REMBI_MODULE.BIOSAMPLE


def test_member_override_wins():
    assert _ModEnum.OVERRIDDEN.resolved_rembi_module is REMBI_MODULE.SPECIMEN_PREP


def test_fallback_is_analyzed_data():
    assert _NoModEnum.LONELY.resolved_rembi_module is REMBI_MODULE.ANALYZED_DATA


def test_entry_rejects_bad_module():
    import pytest
    with pytest.raises((ValueError, TypeError)):
        Entry("X", "bad", rembi_module="Biosample")  # not a REMBI_MODULE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_rembi_module.py -v`
Expected: FAIL with `ImportError: cannot import name 'REMBI_MODULE'`.

- [ ] **Step 3: Create the `REMBI_MODULE` enum**

```python
# src/phenotypic/schema/_rembi.py
"""REMBI module taxonomy for classifying metadata columns.

REMBI (Recommended Metadata for Biological Images; Sarkans et al. 2021) groups
bioimage provenance into modules. Each metadata enum declares its module via
``MeasurementInfo.rembi_module()``; measurement/locator enums fall back to
``ANALYZED_DATA``. Definition order is the canonical manifest/section order.

Import-light: stdlib only (see schema package load-order rule).
"""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._measurement_info import MeasurementInfo


class REMBI_MODULE(str, Enum):
    """REMBI metadata modules. Definition order is canonical."""

    STUDY = "Study"
    BIOSAMPLE = "Biosample"
    SPECIMEN_PREP = "SpecimenPreparation"
    IMAGE_ACQUISITION = "ImageAcquisition"
    IMAGE_DATA = "ImageData"
    ANALYZED_DATA = "AnalyzedData"
    UNCATEGORIZED = "Uncategorized"
```

- [ ] **Step 4: Add the `rembi_module` field to `Entry`**

In `src/phenotypic/schema/_measurement_info.py`, add to the `Entry` dataclass (after the `derives_from` field, ~line 83):

```python
    rembi_module: "REMBI_MODULE | None" = None
```

Add the import at the top of the file (under `TYPE_CHECKING` to keep it light):

```python
from typing import Final, TYPE_CHECKING
if TYPE_CHECKING:
    from ._rembi import REMBI_MODULE
```

In `Entry.__post_init__`, after the `derives_from` check (~line 99-100), add validation:

```python
        if self.rembi_module is not None:
            from ._rembi import REMBI_MODULE
            if not isinstance(self.rembi_module, REMBI_MODULE):
                raise TypeError("Entry.rembi_module must be a REMBI_MODULE or None")
```

- [ ] **Step 5: Add the classmethod default, `__new__` capture, and resolver**

In `MeasurementInfo` (same file), add next to `kind()`/`tier()` (~line 351-359):

```python
    @classmethod
    def rembi_module(cls) -> "REMBI_MODULE | None":
        """REMBI module for this enum, or None until a subclass declares it."""
        return None
```

Add the per-member attribute annotation next to `tier_override` (~line 329):

```python
    rembi_module_override: "REMBI_MODULE | None"
```

In `__new__`, after `obj.tier_override = entry.tier` (~line 395), add:

```python
        obj.rembi_module_override = entry.rembi_module
```

Add the resolver property next to `resolved_tier` (~line 417-420):

```python
    @property
    def resolved_rembi_module(self) -> "REMBI_MODULE":
        """Total REMBI-module resolver: override > enum declaration > fallback."""
        from ._rembi import REMBI_MODULE
        if self.rembi_module_override is not None:
            return self.rembi_module_override
        mod = type(self).rembi_module()
        if mod is not None:
            return mod
        return REMBI_MODULE.ANALYZED_DATA
```

- [ ] **Step 6: Re-export `REMBI_MODULE`**

In `src/phenotypic/schema/__init__.py`, add after the `_tiers` imports (~line 24):

```python
from ._rembi import REMBI_MODULE as REMBI_MODULE
```

and add `"REMBI_MODULE",` to `__all__`.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/unit/schema/test_rembi_module.py -v`
Expected: PASS (5 passed).

- [ ] **Step 8: Verify existing schema tests still pass**

Run: `uv run pytest tests/unit/schema/ -q`
Expected: PASS (no regressions from the `Entry`/`__new__` change).

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/schema/_rembi.py src/phenotypic/schema/_measurement_info.py src/phenotypic/schema/__init__.py tests/unit/schema/test_rembi_module.py
git commit -m "feat(schema): add REMBI_MODULE taxonomy + rembi_module classification"
```

---

## Task A1b: Rename `INCUBATION_METADATA` → `CULTURE_METADATA`

Standalone public-symbol rename (more accurate, less verbose). Independent of A1 —
can run in parallel with A1. The category string stays `"Metadata"` here; the
Scheme-B `MetadataCulture` value is set later in Task B2.

**Files:**
- Rename: `src/phenotypic/schema/_experimental_tags/_incubation.py` → `_culture.py`
- Modify: `src/phenotypic/schema/_experimental_tags/__init__.py` (import + `__all__`)
- Modify: `src/phenotypic/schema/__init__.py` (docstring mention line ~14, import line ~34, `__all__` line ~79)
- Modify: `tests/unit/schema/test_classification.py`, `tests/unit/docs/test_measurements_ref_extension.py`
- Rename/modify docs: `docs/source/measurements_ref/metadata/incubation_metadata.rst` → `culture_metadata.rst`; update `docs/source/measurements_ref/metadata/index.rst` and `docs/source/_extensions/measurements_ref.py` if either lists the enum by name.

**Interfaces:**
- Produces: `CULTURE_METADATA` (was `INCUBATION_METADATA`), importable from `phenotypic.schema`; same members/labels; `category()` still `"Metadata"` (renamed in B2).

- [ ] **Step 1: Rename the file and class**

```bash
git mv src/phenotypic/schema/_experimental_tags/_incubation.py src/phenotypic/schema/_experimental_tags/_culture.py
```

In `_culture.py`: rename `class INCUBATION_METADATA(IdentityInfo)` → `class CULTURE_METADATA(IdentityInfo)`; update the module docstring (`"""Incubation and time-course..."""` → `"""Culture and time-course metadata tags for the PhenoTypic module."""`) and the class docstring wording (Incubation → Culture); change the `TEMPERATURE` description `"Incubation temperature in degrees Celsius."` → `"Culture temperature in degrees Celsius."`.

- [ ] **Step 2: Update the schema exports**

`_experimental_tags/__init__.py`: `from ._culture import CULTURE_METADATA`; `"CULTURE_METADATA",` in `__all__`.
`schema/__init__.py`: update the docstring mention (line ~14), the import (line ~34: `CULTURE_METADATA,`), and `__all__` (line ~79: `"CULTURE_METADATA",`).

- [ ] **Step 3: Update existing references**

Edit `tests/unit/schema/test_classification.py` and `tests/unit/docs/test_measurements_ref_extension.py`: replace every `INCUBATION_METADATA` with `CULTURE_METADATA` (and any `MetadataIncubation`/`incubation_metadata` doc-id strings with `MetadataCulture`/`culture_metadata`).

- [ ] **Step 4: Verify the rename is complete**

Run:
```bash
uv run python -c "from phenotypic.schema import CULTURE_METADATA; print(CULTURE_METADATA.TIME.value)"
grep -rn "INCUBATION_METADATA\|_incubation\b" src tests docs/source || echo "CLEAN"
```
Expected: prints `Metadata_Time`; grep prints `CLEAN` (no stray references).

- [ ] **Step 5: Run schema + docs tests**

Run: `uv run pytest tests/unit/schema tests/unit/docs -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A src/phenotypic/schema tests/unit/schema tests/unit/docs docs/source/measurements_ref docs/source/_extensions
git commit -m "refactor(schema)!: rename INCUBATION_METADATA -> CULTURE_METADATA"
```

---

## Task A2: Declare REMBI modules on the 8 existing metadata enums + coverage gate

**Files:**
- Modify: `src/phenotypic/schema/_metadata.py`; `_experimental_tags/_genetic.py`, `_sample.py`, `_condition.py`, `_culture.py`, `_plate.py`, `_acquisition.py`, `_experiment.py`
- Test: `tests/unit/schema/test_rembi_coverage.py`

**Interfaces:**
- Consumes: `REMBI_MODULE`, `resolved_rembi_module` (Task A1).
- Produces: every `Metadata_`-namespace enum resolves to a real module; `CULTURE_METADATA.{TIME,TIME_UNIT,TIMEPOINT,FRAME_INDEX}` → `BIOSAMPLE`, all other CULTURE members → `SPECIMEN_PREP`.

- [ ] **Step 1: Write the failing coverage test**

```python
# tests/unit/schema/test_rembi_coverage.py
import phenotypic.schema as schema
from phenotypic.schema import MeasurementInfo, REMBI_MODULE
from phenotypic.schema._tiers import (
    DirectPhenotype, DescriptiveTrait, DiscriminativeFeature,
    IdentityInfo, QualityInfo, DerivedMeasure, PrimaryMeasure,
)

_BASES = (DirectPhenotype, DescriptiveTrait, DiscriminativeFeature,
          IdentityInfo, QualityInfo, DerivedMeasure, PrimaryMeasure)


def _column_enums():
    for name in schema.__all__:
        obj = getattr(schema, name)
        if (isinstance(obj, type) and issubclass(obj, MeasurementInfo)
                and obj not in _BASES and obj is not MeasurementInfo and list(obj)):
            yield obj


def test_metadata_enums_declare_a_real_module():
    bad = []
    for enum in _column_enums():
        if not enum.category().startswith("Metadata"):
            continue
        for m in enum:
            mod = m.resolved_rembi_module
            if mod in (REMBI_MODULE.ANALYZED_DATA, REMBI_MODULE.UNCATEGORIZED):
                bad.append(f"{enum.__name__}.{m.name} -> {mod}")
    assert not bad, "metadata members must declare a real REMBI module:\n" + "\n".join(bad)


def test_resolved_module_is_total():
    for enum in _column_enums():
        for m in enum:
            assert isinstance(m.resolved_rembi_module, REMBI_MODULE)


def test_culture_time_members_are_biosample():
    from phenotypic.schema import CULTURE_METADATA
    assert CULTURE_METADATA.TIME.resolved_rembi_module is REMBI_MODULE.BIOSAMPLE
    assert CULTURE_METADATA.TIMEPOINT.resolved_rembi_module is REMBI_MODULE.BIOSAMPLE
    assert CULTURE_METADATA.FRAME_INDEX.resolved_rembi_module is REMBI_MODULE.BIOSAMPLE
    assert CULTURE_METADATA.TEMPERATURE.resolved_rembi_module is REMBI_MODULE.SPECIMEN_PREP
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_rembi_coverage.py -v`
Expected: FAIL (`test_metadata_enums_declare_a_real_module` lists every metadata member as `AnalyzedData`).

- [ ] **Step 3: Add `rembi_module()` to each metadata enum**

In each file, add the import `from .._rembi import REMBI_MODULE` (for `_experimental_tags/*`) or `from ._rembi import REMBI_MODULE` (for `_metadata.py`), and a classmethod next to `category()`:

- `_metadata.py` (`METADATA`): `return REMBI_MODULE.IMAGE_DATA`
- `_experiment.py` (`EXPERIMENT_METADATA`): `return REMBI_MODULE.STUDY`
- `_genetic.py` (`GENETIC_METADATA`): `return REMBI_MODULE.BIOSAMPLE`
- `_sample.py` (`SAMPLE_METADATA`): `return REMBI_MODULE.BIOSAMPLE`
- `_condition.py` (`CONDITION_METADATA`): `return REMBI_MODULE.SPECIMEN_PREP`
- `_plate.py` (`PLATE_METADATA`): `return REMBI_MODULE.SPECIMEN_PREP`
- `_acquisition.py` (`ACQUISITION_METADATA`): `return REMBI_MODULE.IMAGE_ACQUISITION`
- `_culture.py` (`CULTURE_METADATA`): `return REMBI_MODULE.SPECIMEN_PREP`

Example (`_genetic.py`, after the `category()` classmethod):

```python
    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:
        return REMBI_MODULE.BIOSAMPLE
```

- [ ] **Step 4: Add the per-member overrides in `_culture.py`**

Change the four temporal members to carry `rembi_module=REMBI_MODULE.BIOSAMPLE`:

```python
    TIME = Entry("Time", "Elapsed growth time.",
                 rembi_module=REMBI_MODULE.BIOSAMPLE)
    TIME_UNIT = Entry("TimeUnit", "Unit for the Time value (e.g. hours, days).",
                      rembi_module=REMBI_MODULE.BIOSAMPLE)
    TIMEPOINT = Entry(
        "Timepoint",
        "Human-readable label for a discrete timepoint in a time series (e.g. "
        "'24h', 'stationary'); may be non-numeric. For the integer capture "
        "ordinal, use FrameIndex.",
        rembi_module=REMBI_MODULE.BIOSAMPLE,
    )
    FRAME_INDEX = Entry(
        "FrameIndex",
        "1-based ordinal index of the image within the time-course capture "
        "sequence; the monotonic-integer companion to the free-form Timepoint "
        "label.",
        rembi_module=REMBI_MODULE.BIOSAMPLE,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/schema/test_rembi_coverage.py tests/unit/schema/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/schema/_metadata.py src/phenotypic/schema/_experimental_tags/ tests/unit/schema/test_rembi_coverage.py
git commit -m "feat(schema): declare REMBI modules on metadata enums; time->Biosample"
```

---

## Task A3: `STUDY_METADATA` enum (Study module)

**Files:**
- Create: `src/phenotypic/schema/_experimental_tags/_study.py`
- Modify: `src/phenotypic/schema/_experimental_tags/__init__.py`; `src/phenotypic/schema/__init__.py`
- Test: `tests/unit/schema/test_study_metadata.py`

**Interfaces:**
- Produces: `STUDY_METADATA(IdentityInfo)` with `category()=="MetadataStudy"`... **NOTE: in Phase A the category is still `"Metadata"`** (Scheme-B rename happens in Phase B Task B2). Members: `TITLE, DESCRIPTION, PRIVATE_UNTIL_DATE, KEYWORDS, AUTHOR, LICENSE, FUNDING, PUBLICATIONS, LINKS, ACKNOWLEDGEMENTS`; `rembi_module()==STUDY`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/schema/test_study_metadata.py
from phenotypic.schema import STUDY_METADATA, REMBI_MODULE


def test_study_members_present():
    labels = STUDY_METADATA.get_labels()
    assert labels == [
        "Title", "Description", "PrivateUntilDate", "Keywords", "Author",
        "License", "Funding", "Publications", "Links", "Acknowledgements",
    ]


def test_study_module_and_namespace():
    assert STUDY_METADATA.TITLE.resolved_rembi_module is REMBI_MODULE.STUDY
    assert STUDY_METADATA.category().startswith("Metadata")
    assert STUDY_METADATA.TITLE.value.endswith("_Title")


def test_study_bio_desc_unset():
    # human-authored guardrail: agents leave bio_desc empty
    assert all(m.bio_desc == "" for m in STUDY_METADATA)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_study_metadata.py -v`
Expected: FAIL with `ImportError: cannot import name 'STUDY_METADATA'`.

- [ ] **Step 3: Create the enum**

```python
# src/phenotypic/schema/_experimental_tags/_study.py
"""Study-level (REMBI Study component) metadata tags."""

from .._measurement_info import Entry
from .._rembi import REMBI_MODULE
from .._tiers import IdentityInfo


class STUDY_METADATA(IdentityInfo):
    """Recommended ``Metadata_*`` tags for the REMBI Study component.

    One set per run (title, authors, license, …). Mirrors REMBI's Study field
    names. Structured REMBI lists (authors/publications/links) are flattened to
    scalar tags whose value may be a delimited string. Recommended vocabulary,
    not a validator.
    """

    @classmethod
    def category(cls) -> str:
        return "Metadata"

    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:
        return REMBI_MODULE.STUDY

    TITLE = Entry("Title", "Study title.")
    DESCRIPTION = Entry("Description", "Free-text study description.")
    PRIVATE_UNTIL_DATE = Entry(
        "PrivateUntilDate", "Embargo date until which the study stays private.")
    KEYWORDS = Entry("Keywords", "Keywords describing the study.")
    AUTHOR = Entry("Author", "Study author(s); delimited string when multiple.")
    LICENSE = Entry("License", "Data license (e.g. CC0, CC-BY-4.0).")
    FUNDING = Entry("Funding", "Funding statement or grant reference(s).")
    PUBLICATIONS = Entry("Publications", "Associated publication(s) or DOI(s).")
    LINKS = Entry("Links", "Related links or external resource URLs.")
    ACKNOWLEDGEMENTS = Entry("Acknowledgements", "Acknowledgements text.")
```

- [ ] **Step 4: Export it**

In `src/phenotypic/schema/_experimental_tags/__init__.py`: add `from ._study import STUDY_METADATA` and `"STUDY_METADATA",` to `__all__`.
In `src/phenotypic/schema/__init__.py`: add `STUDY_METADATA` to the experimental-tags import block and `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/schema/test_study_metadata.py tests/unit/schema/test_rembi_coverage.py -q`
Expected: PASS (coverage gate now also covers STUDY_METADATA).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/schema/_experimental_tags/_study.py src/phenotypic/schema/_experimental_tags/__init__.py src/phenotypic/schema/__init__.py tests/unit/schema/test_study_metadata.py
git commit -m "feat(schema): add STUDY_METADATA enum (REMBI Study module)"
```

---

## Task A4: `header_to_module()` reverse index

**Files:**
- Modify: `src/phenotypic/schema/_rembi.py`
- Modify: `src/phenotypic/schema/__init__.py` (export `header_to_module`)
- Test: `tests/unit/schema/test_header_to_module.py`

**Interfaces:**
- Consumes: `schema.__all__` enums, `resolved_rembi_module`.
- Produces: `header_to_module() -> dict[str, REMBI_MODULE]` mapping every known column header (`"Metadata_Strain"`, `"Shape_Area"`, …) to its module.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/schema/test_header_to_module.py
from phenotypic.schema import header_to_module, REMBI_MODULE


def test_metadata_headers_mapped():
    idx = header_to_module()
    assert idx["Metadata_Strain"] is REMBI_MODULE.BIOSAMPLE
    assert idx["Metadata_Dataset"] is REMBI_MODULE.STUDY
    assert idx["Metadata_Time"] is REMBI_MODULE.BIOSAMPLE
    assert idx["Metadata_ImageName"] is REMBI_MODULE.IMAGE_DATA


def test_measurement_headers_are_analyzed():
    idx = header_to_module()
    assert idx["Shape_Area"] is REMBI_MODULE.ANALYZED_DATA
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_header_to_module.py -v`
Expected: FAIL with `ImportError: cannot import name 'header_to_module'`.

- [ ] **Step 3: Implement `header_to_module()`**

Append to `src/phenotypic/schema/_rembi.py`:

```python
def header_to_module() -> "dict[str, REMBI_MODULE]":
    """Map every known column header to its REMBI module.

    Walks every ``MeasurementInfo`` subclass exported from ``phenotypic.schema``
    and reads each member's ``resolved_rembi_module``. Built fresh on each call
    (cheap; <1k members). Used by the manifest builder's column router.
    """
    from . import __all__ as _names
    from . import _measurement_info as _mi
    import phenotypic.schema as _schema

    out: "dict[str, REMBI_MODULE]" = {}
    for name in _names:
        obj = getattr(_schema, name)
        if (isinstance(obj, type) and issubclass(obj, _mi.MeasurementInfo)
                and obj is not _mi.MeasurementInfo and list(obj)):
            for member in obj:
                out[member.value] = member.resolved_rembi_module
    return out
```

In `schema/__init__.py`, export `header_to_module` (add to the `_rembi` import line and `__all__`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/schema/test_header_to_module.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/schema/_rembi.py src/phenotypic/schema/__init__.py tests/unit/schema/test_header_to_module.py
git commit -m "feat(schema): add header_to_module reverse index"
```

---

## Task A5: `build_rembi_manifest()` (pure builder)

**Files:**
- Create: `src/phenotypic/sdk_/_rembi_manifest.py`
- Test: `tests/unit/sdk_/test_rembi_manifest.py`

**Interfaces:**
- Consumes: `header_to_module()`, `REMBI_MODULE`, the per-enum module values.
- Produces: `build_rembi_manifest(measurements: pandas.DataFrame, image_metadata: list[dict], study_config: dict | None = None) -> dict`. Returns a nested dict keyed by lowercase module section names (`study, biosample, specimen_preparation, image_acquisition, image_data, analyzed_data, uncategorized`), omitting empty sections except `image_data`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/sdk_/test_rembi_manifest.py
import pandas as pd
from phenotypic.sdk_._rembi_manifest import build_rembi_manifest


def _df():
    return pd.DataFrame({
        "Metadata_Strain": ["BY4741", "by4742"],
        "Metadata_Media": ["YPD", "YPD"],
        "Metadata_Temperature": [30, 30],
        "Metadata_CustomTag": ["x", "x"],
        "Size_Area": [10, 12],
        "Shape_Circularity": [0.9, 0.8],
    })


def _imgmeta():
    return [{"ImageName": "p1", "UUID": "u1", "BitDepth": 8, "ImageType": "rgb"}]


def test_scalar_vs_list_collapse():
    m = build_rembi_manifest(_df(), _imgmeta())
    assert sorted(m["biosample"]["Strain"]) == ["BY4741", "by4742"]  # >1 -> list
    assert m["specimen_preparation"]["Media"] == "YPD"               # 1 -> scalar


def test_unknown_metadata_goes_uncategorized():
    m = build_rembi_manifest(_df(), _imgmeta())
    assert "CustomTag" in m["uncategorized"]


def test_analyzed_data_lists_features_grouped():
    m = build_rembi_manifest(_df(), _imgmeta())
    assert m["analyzed_data"]["features"]["Size"] == ["Area"]
    assert m["analyzed_data"]["features"]["Shape"] == ["Circularity"]


def test_image_data_always_present_even_empty():
    m = build_rembi_manifest(pd.DataFrame(), _imgmeta())
    assert m["image_data"]["n_images"] == 1
    assert m["image_data"]["files"][0]["uuid"] == "u1"
    assert "biosample" not in m  # empty sections omitted


def test_study_config_overrides_csv_constant():
    df = pd.DataFrame({"Metadata_Title": ["from_csv", "from_csv"]})
    m = build_rembi_manifest(df, _imgmeta(), study_config={"Title": "from_file"})
    assert m["study"]["Title"] == "from_file"


def test_study_ambiguity_collapses_to_list():
    df = pd.DataFrame({"Metadata_Title": ["a", "b"]})
    m = build_rembi_manifest(df, _imgmeta())
    assert sorted(m["study"]["Title"]) == ["a", "b"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/sdk_/test_rembi_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: ... _rembi_manifest`.

- [ ] **Step 3: Implement the builder**

```python
# src/phenotypic/sdk_/_rembi_manifest.py
"""Pure builder for the REMBI run manifest (deliverables/rembi.yaml).

Folds the per-colony measurements mirror up to each REMBI module's scope:
distinct-collapse (scalar-or-list) for biosample/specimen-prep/acquisition,
a per-image file list for image-data, a feature catalog for analyzed-data, and
a one-value-per-field study section (CSV constants overridable by a study file).
No I/O; see _write below for serialization.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from phenotypic.schema import REMBI_MODULE, header_to_module

# module -> manifest section key
_SECTION = {
    REMBI_MODULE.STUDY: "study",
    REMBI_MODULE.BIOSAMPLE: "biosample",
    REMBI_MODULE.SPECIMEN_PREP: "specimen_preparation",
    REMBI_MODULE.IMAGE_ACQUISITION: "image_acquisition",
    REMBI_MODULE.UNCATEGORIZED: "uncategorized",
}
_METADATA_FAMILY = "Metadata"


def _distinct(series: pd.Series) -> Any:
    vals = sorted({v for v in series.dropna().tolist()}, key=str)
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else vals


def _label_of(header: str) -> str:
    # strip the "<Category>_" prefix -> bare label
    return header.split("_", 1)[1] if "_" in header else header


def build_rembi_manifest(
    measurements: pd.DataFrame,
    image_metadata: list[dict],
    study_config: dict | None = None,
) -> dict:
    idx = header_to_module()
    manifest: dict[str, dict] = {}

    # --- distinct-collapse sections (study/biosample/specimen/acquisition/uncat)
    for col in measurements.columns:
        if not str(col).startswith(_METADATA_FAMILY):
            continue  # measurement/locator columns handled in analyzed_data
        module = idx.get(col, REMBI_MODULE.UNCATEGORIZED)
        section = _SECTION.get(module)
        if section is None:
            continue
        value = _distinct(measurements[col])
        if value is None:
            continue
        manifest.setdefault(section, {})[_label_of(col)] = value

    # --- study file overrides csv constants
    if study_config:
        study = manifest.setdefault("study", {})
        study.update({k: v for k, v in study_config.items() if v is not None})

    # --- image_data: per-image files + rollups (ALWAYS present)
    files = [
        {
            "name": im.get("ImageName"),
            "uuid": im.get("UUID"),
            "bit_depth": im.get("BitDepth"),
            "image_type": im.get("ImageType"),
        }
        for im in image_metadata
    ]
    manifest["image_data"] = {
        "n_images": len(files),
        "bit_depth": sorted({f["bit_depth"] for f in files if f["bit_depth"] is not None}),
        "files": files,
    }

    # --- analyzed_data: feature catalog grouped by category prefix
    features: dict[str, list[str]] = {}
    for col in measurements.columns:
        col = str(col)
        if col.startswith(_METADATA_FAMILY) or "_" not in col:
            continue
        cat, label = col.split("_", 1)
        features.setdefault(cat, []).append(label)
    if features:
        manifest["analyzed_data"] = {"features": {k: sorted(v) for k, v in features.items()}}

    return manifest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/sdk_/test_rembi_manifest.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_rembi_manifest.py tests/unit/sdk_/test_rembi_manifest.py
git commit -m "feat(sdk_): pure build_rembi_manifest folder"
```

---

## Task A6: manifest writer + paths + finalize hook + `--study` flag

**Files:**
- Modify: `src/phenotypic/sdk_/_rembi_manifest.py` (add `write_rembi_manifest`)
- Modify: `src/phenotypic/sdk_/_io_constants.py` (add `REMBI_MANIFEST_YAML`, `rembi_manifest_path`)
- Modify: `src/phenotypic/_cli/_cli_output_manager.py` (`finalize_post_master_outputs`)
- Modify: `src/phenotypic/phenotypicCLI.py` (add `--study` arg, thread to finalize)
- Test: `tests/unit/sdk_/test_rembi_manifest_write.py`; `tests/integration/cli/test_rembi_manifest_emitted.py`

**Interfaces:**
- Consumes: `build_rembi_manifest` (A5), `deliverables_dir` (existing in `_io_constants`).
- Produces: `write_rembi_manifest(output_dir, measurements, image_metadata, study_config=None) -> Path | None` (best-effort, returns path or None on failure); `rembi_manifest_path(output_dir) -> Path` == `<output>/deliverables/rembi.yaml`; CLI `--study PATH`.

- [ ] **Step 1: Write the failing writer test**

```python
# tests/unit/sdk_/test_rembi_manifest_write.py
import pandas as pd
import yaml
from phenotypic.sdk_._rembi_manifest import write_rembi_manifest
from phenotypic.sdk_._io_constants import rembi_manifest_path


def test_writes_parseable_yaml(tmp_path):
    (tmp_path / "deliverables").mkdir()
    df = pd.DataFrame({"Metadata_Strain": ["BY4741"]})
    p = write_rembi_manifest(tmp_path, df, [{"ImageName": "p1", "UUID": "u1",
                                             "BitDepth": 8, "ImageType": "rgb"}])
    assert p == rembi_manifest_path(tmp_path)
    data = yaml.safe_load(p.read_text())
    assert data["image_data"]["n_images"] == 1
    assert data["biosample"]["Strain"] == "BY4741"


def test_write_never_raises(tmp_path):
    # deliverables dir missing -> best-effort returns None, no exception
    result = write_rembi_manifest(tmp_path, pd.DataFrame(), [])
    assert result is None or result.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/sdk_/test_rembi_manifest_write.py -v`
Expected: FAIL (`write_rembi_manifest` / `rembi_manifest_path` not defined).

- [ ] **Step 3: Add the path helper**

In `src/phenotypic/sdk_/_io_constants.py`, near `DELIVERABLES_METADATA_CSV` add:

```python
#: REMBI run manifest filename, flat under deliverables/ beside metadata.csv.
REMBI_MANIFEST_YAML: Final[str] = "rembi.yaml"
```

and near `deliverables_dir` add:

```python
def rembi_manifest_path(output_dir: Path) -> Path:
    """``<output>/deliverables/rembi.yaml`` — the REMBI run manifest."""
    return deliverables_dir(output_dir) / REMBI_MANIFEST_YAML
```

- [ ] **Step 4: Add the writer**

Append to `src/phenotypic/sdk_/_rembi_manifest.py`:

```python
def write_rembi_manifest(
    output_dir,
    measurements: pd.DataFrame,
    image_metadata: list[dict],
    study_config: dict | None = None,
):
    """Best-effort: build + write the manifest to deliverables/rembi.yaml.

    Never raises — logs and returns None on any failure, so finalize is never
    blocked (same contract as the best-effort metadata.csv copy).
    """
    import logging
    from pathlib import Path

    import yaml

    from ._io_constants import rembi_manifest_path

    log = logging.getLogger(__name__)
    try:
        manifest = build_rembi_manifest(measurements, image_metadata, study_config)
        path = rembi_manifest_path(Path(output_dir))
        if not path.parent.exists():
            return None
        path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
        return path
    except Exception:  # noqa: BLE001 - best-effort, never block finalize
        log.warning("REMBI manifest write failed", exc_info=True)
        return None
```

- [ ] **Step 5: Run the writer test**

Run: `uv run pytest tests/unit/sdk_/test_rembi_manifest_write.py -v`
Expected: PASS.

- [ ] **Step 6: Wire into finalize + `--study`**

Read `finalize_post_master_outputs` in `src/phenotypic/_cli/_cli_output_manager.py` to find where the post-applied mirror DataFrame and per-image metadata are available (near where `metadata.csv` is copied / the mirror is written). Add, guarded so it never blocks:

```python
    from phenotypic.sdk_._rembi_manifest import write_rembi_manifest
    # mirror_df: the post-applied measurements mirror; image_meta: per-image dicts
    write_rembi_manifest(output_dir, mirror_df, image_meta, study_config=study_config)
```

Source `image_meta` from the per-image metadata already gathered in finalize (UUID/ImageName/BitDepth/ImageType). If a study config path was passed, load it: `study_config = yaml.safe_load(Path(study_path).read_text()) if study_path else None`.

In `src/phenotypic/phenotypicCLI.py`, add the CLI arg in the parser block for the default run mode:

```python
    parser.add_argument(
        "--study", dest="study", default=None, metavar="PATH",
        help="Optional study.yaml of REMBI Study-level fields (Title, License, "
             "Author, ...) folded into deliverables/rembi.yaml. CLI study file "
             "overrides constant Metadata_* columns.",
    )
```

Thread `args.study` through to `finalize_post_master_outputs` (follow the existing arg-threading pattern for output options).

- [ ] **Step 7: Write the integration test**

```python
# tests/integration/cli/test_rembi_manifest_emitted.py
import subprocess
import sys
import yaml
from pathlib import Path


def test_manifest_emitted_on_default_run(tmp_path):
    # Use the smallest synthetic-plate CLI run available in this repo's fixtures.
    # (Mirror an existing tests/integration/cli run; assert the manifest lands.)
    out = tmp_path / "out"
    # ... run `uv run python -m phenotypic <images> -o out` per existing CLI test harness ...
    manifest = out / "deliverables" / "rembi.yaml"
    assert manifest.exists()
    data = yaml.safe_load(manifest.read_text())
    assert "image_data" in data and data["image_data"]["n_images"] >= 1
```

> Implementer note: model this on the nearest existing `tests/integration/cli/` run (reuse its fixture images + invocation); the only new assertions are the four lines about `deliverables/rembi.yaml`. Add a sibling assertion in the existing `--mode process` integration test that `rembi.yaml` is **absent** there.

- [ ] **Step 8: Run the integration test**

Run: `uv run pytest tests/integration/cli/test_rembi_manifest_emitted.py -v`
Expected: PASS (manifest present on default run; absent under `--mode process`).

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/sdk_/_rembi_manifest.py src/phenotypic/sdk_/_io_constants.py src/phenotypic/_cli/_cli_output_manager.py src/phenotypic/phenotypicCLI.py tests/unit/sdk_/test_rembi_manifest_write.py tests/integration/cli/test_rembi_manifest_emitted.py
git commit -m "feat(cli): always emit deliverables/rembi.yaml in finalize; add --study"
```

---

## Task A7: read-side `by_module()` + REMBI column ordering

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py` (`insert_metadata` ~316-331; add `by_module`)
- Test: `tests/unit/core/test_metadata_by_module.py`

**Interfaces:**
- Consumes: `header_to_module`, `REMBI_MODULE`.
- Produces: `MetadataAccessor.by_module(module: REMBI_MODULE | str) -> dict[str, Any]`; `insert_metadata` orders the inserted metadata columns by canonical REMBI-module order then alpha.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_metadata_by_module.py
from phenotypic.schema import REMBI_MODULE
from phenotypic._core import Image  # adjust import to the public Image
import numpy as np


def _img():
    img = Image(np.zeros((8, 8, 3), dtype=np.uint8), name="sample")
    img.metadata["Strain"] = "BY4741"       # public tag
    return img


def test_by_module_groups_image_data():
    img = _img()
    image_data = img.metadata.by_module(REMBI_MODULE.IMAGE_DATA)
    # framework private/protected keys (e.g. ImageName) land in ImageData
    assert any("ImageName" in k for k in image_data)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/core/test_metadata_by_module.py -v`
Expected: FAIL (`by_module` not defined).

- [ ] **Step 3: Implement `by_module` and ordering**

Add to `MetadataAccessor`:

```python
    def by_module(self, module) -> dict:
        """Group metadata keys/values by REMBI module (read-only view).

        Framework private/protected keys map to IMAGE_DATA; public tags resolve
        via the schema reverse index; unrecognized keys fall to UNCATEGORIZED.
        """
        from phenotypic.schema import REMBI_MODULE, header_to_module
        target = REMBI_MODULE(module) if not isinstance(module, REMBI_MODULE) else module
        idx = header_to_module()
        out: dict = {}
        for key, value in self._combined_metadata.items():
            header = key if str(key).startswith("Metadata") else f"Metadata_{key}"
            mod = idx.get(header)
            if mod is None:
                mod = (REMBI_MODULE.IMAGE_DATA
                       if key in self._private_metadata or key in self._protected_metadata
                       else REMBI_MODULE.UNCATEGORIZED)
            if mod is target:
                out[key] = value
        return out
```

In `insert_metadata`, sort the iteration by REMBI-module order before inserting. Replace the `for key, value in self._public_protected_metadata.items():` loop driver with a sorted key list:

```python
        from phenotypic.schema import REMBI_MODULE, header_to_module
        idx = header_to_module()
        order = {m: i for i, m in enumerate(REMBI_MODULE)}

        def _rank(item):
            key = item[0]
            header = key if str(key).startswith("Metadata") else f"Metadata_{key}"
            mod = idx.get(header, REMBI_MODULE.UNCATEGORIZED)
            return (order[mod], str(key))

        items = sorted(self._public_protected_metadata.items(), key=_rank, reverse=True)
        for key, value in items:
            ...  # existing prefix + insert body unchanged
```

(The existing body inserts at `loc=0` right-to-left, so iterate in reverse rank to land columns in canonical order.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/core/test_metadata_by_module.py tests/unit/core/ -q`
Expected: PASS (and no regression in existing metadata-accessor tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py tests/unit/core/test_metadata_by_module.py
git commit -m "feat(core): metadata.by_module() view + REMBI column ordering"
```

---

**PHASE A CHECKPOINT.** Run the full additive suite + a smoke run:
```bash
uv run pytest tests/unit/schema tests/unit/sdk_ tests/unit/core tests/integration/cli -q
```
Phase A is now an independently shippable unit: REMBI classification + Study module + always-on manifest, fully back-compat (categories still `"Metadata"`). If splitting into PRs, this is the PR1 boundary.

---

# PHASE B — Category namespace migration (the rename)

*Renames `Metadata_` → `Metadata<Topic>_`, centralizes the prefix predicate, retires `Metadata_ImageFile`, and de-stringly-types metadata access. Touches ~49 source + ~128 test files.*

> **Execution order — decouple-then-flip (not strict numeric order).** The
> category flip (B2) changes real column strings, so do it *after* the codebase
> already routes through helpers/enum-refs:
> **B1 → {B3, B4, B5, B6, B7 — the "decouple" group} → B2 (flip) → B8 (gate+docs)**.
> Two rules keep every task green: (1) write decouple-task tests against the
> **live enum value** (`str(GENETIC_METADATA.STRAIN)`), never a hardcoded
> post-rename string — so they pass before *and* after B2; (2) **each task updates
> the tests it breaks** (e.g. B5 updates `Metadata_ImageFile` test assertions in
> the same commit). B2 then only has to update the remaining category-prefix test
> assertions not already handled by B5/B7. B8 is the final grep gate + docs + full
> sweep.

## Task B1: centralized metadata helpers

**Files:**
- Create: `src/phenotypic/sdk_/_metadata_helpers.py`
- Modify: `src/phenotypic/sdk_/__init__.py` (export the three helpers)
- Test: `tests/unit/sdk_/test_metadata_helpers.py`

**Interfaces:**
- Produces: `metadata_category_prefixes() -> tuple[str, ...]` (REMBI order, derived from enums); `is_metadata_header(col: str) -> bool`; `metadata_category_for_label(label: str) -> str | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/sdk_/test_metadata_helpers.py
from phenotypic.sdk_ import (
    is_metadata_header, metadata_category_prefixes, metadata_category_for_label,
)


def test_prefixes_in_rembi_order():
    pres = metadata_category_prefixes()
    assert pres[0] == "MetadataImage_"            # IMAGE_DATA-owning framework enum
    assert "MetadataGenetic_" in pres
    assert all(p.startswith("Metadata") and p.endswith("_") for p in pres)


def test_is_metadata_header():
    assert is_metadata_header("MetadataGenetic_Strain")
    assert is_metadata_header("MetadataImage_ImageName")
    assert not is_metadata_header("Shape_Area")
    assert not is_metadata_header("Object_Label")


def test_category_for_label():
    assert metadata_category_for_label("Strain") == "MetadataGenetic"
    assert metadata_category_for_label("Dataset") == "MetadataExperiment"
    assert metadata_category_for_label("NotARealTag") is None
```

> This test is written against the **post-rename** category strings, so it will stay red until Task B2 renames the categories. Run it after B2.

- [ ] **Step 2: Implement the helpers**

```python
# src/phenotypic/sdk_/_metadata_helpers.py
"""Single source of truth for the metadata column namespace.

Replaces every hardcoded ``"Metadata_"`` prefix literal across the codebase.
Prefixes/labels are derived from the schema enums, so they track the category
names automatically.
"""
from __future__ import annotations

from functools import lru_cache

import phenotypic.schema as _schema
from phenotypic.schema import MeasurementInfo, REMBI_MODULE


@lru_cache(maxsize=1)
def _metadata_enums() -> tuple[type, ...]:
    out = []
    for name in _schema.__all__:
        obj = getattr(_schema, name)
        if (isinstance(obj, type) and issubclass(obj, MeasurementInfo)
                and obj is not MeasurementInfo and list(obj)
                and obj.category().startswith("Metadata")):
            out.append(obj)
    return tuple(out)


@lru_cache(maxsize=1)
def metadata_category_prefixes() -> tuple[str, ...]:
    """All metadata category prefixes (e.g. 'MetadataGenetic_') in REMBI order."""
    order = {m: i for i, m in enumerate(REMBI_MODULE)}
    enums = sorted(
        _metadata_enums(),
        key=lambda e: (order.get(next(iter(e)).resolved_rembi_module, 99), e.category()),
    )
    seen, prefixes = set(), []
    for e in enums:
        p = f"{e.category()}_"
        if p not in seen:
            seen.add(p)
            prefixes.append(p)
    return tuple(prefixes)


def is_metadata_header(col: str) -> bool:
    """True if col is a metadata-family column (any MetadataXxx_ prefix)."""
    return any(str(col).startswith(p) for p in metadata_category_prefixes())


@lru_cache(maxsize=1)
def _label_to_category() -> dict[str, str]:
    out: dict[str, str] = {}
    for e in _metadata_enums():
        for m in e:
            out.setdefault(m.label, e.category())
    return out


def metadata_category_for_label(label: str) -> str | None:
    """Category that owns a bare label ('Strain' -> 'MetadataGenetic'), or None."""
    return _label_to_category().get(label)
```

In `src/phenotypic/sdk_/__init__.py`, export the three public helpers.

- [ ] **Step 3: Commit (test runs green after B2)**

```bash
git add src/phenotypic/sdk_/_metadata_helpers.py src/phenotypic/sdk_/__init__.py tests/unit/sdk_/test_metadata_helpers.py
git commit -m "feat(sdk_): centralized metadata namespace helpers"
```

---

## Task B2: rename the 9 `category()` returns to Scheme B

**Files:**
- Modify: the 9 metadata enum files (`_metadata.py` + 7 experimental tags + `_study.py`)
- Modify: `src/phenotypic/schema/CLAUDE.md` (docstring text stating `category()=="Metadata"`)
- Test: `tests/unit/schema/test_category_names.py`; re-run `test_metadata_helpers.py`

**Interfaces:**
- Produces: each enum's `category()` returns its Scheme-B string (Global Constraints).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/schema/test_category_names.py
from phenotypic import schema

EXPECTED = {
    "METADATA": "MetadataImage",
    "STUDY_METADATA": "MetadataStudy",
    "EXPERIMENT_METADATA": "MetadataExperiment",
    "GENETIC_METADATA": "MetadataGenetic",
    "SAMPLE_METADATA": "MetadataSample",
    "CONDITION_METADATA": "MetadataCondition",
    "CULTURE_METADATA": "MetadataCulture",
    "PLATE_METADATA": "MetadataPlate",
    "ACQUISITION_METADATA": "MetadataAcquisition",
}


def test_scheme_b_category_names():
    for enum_name, cat in EXPECTED.items():
        assert getattr(schema, enum_name).category() == cat


def test_headers_self_describing():
    assert schema.GENETIC_METADATA.STRAIN.value == "MetadataGenetic_Strain"
    assert schema.METADATA.IMAGE_NAME.value == "MetadataImage_ImageName"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_category_names.py -v`
Expected: FAIL (categories still `"Metadata"`).

- [ ] **Step 3: Change each `category()`**

In each of the 9 files, change the `return "Metadata"` in `category()` to the Scheme-B string from Global Constraints. Update each class docstring line that says `category() == "Metadata"` / "render as `Metadata_<Label>`" to the new prefix.

- [ ] **Step 4: Update `schema/CLAUDE.md`**

Update the lines describing `category() == "Metadata"` and the `Metadata_` namespace to reflect per-enum Scheme-B categories (the experimental tags + framework now return `Metadata<Topic>`).

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/schema/test_category_names.py tests/unit/sdk_/test_metadata_helpers.py tests/unit/schema/ -q`
Expected: PASS (B1 helpers now green too).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/schema/ tests/unit/schema/test_category_names.py
git commit -m "feat(schema)!: per-enum Scheme-B metadata category prefixes"
```

---

## Task B3: route bare-prefix predicates through `is_metadata_header`

**Files (8 sites + JS):**
- Modify: `src/phenotypic/post/_utils.py:3,6-8` (handled in Task B4 — schema-aware)
- Modify: `src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py:322-323` (handled in B4)
- Modify: `src/phenotypic/_cli/_dashboard/_analysis_helpers.py:22`
- Modify: `src/phenotypic/_cli/_dashboard/_analysis/_scatter_plot.py:89` (inject from Python)
- Modify: `src/phenotypic/gui/results_viewer/_output_root.py:547`
- Modify: `src/phenotypic/gui/results_viewer/_viewer_card.py:105` (delete unused const)
- Modify: `src/phenotypic/gui/results_viewer/colony_view/_grid.py:98`
- Modify: `src/phenotypic/gui/results_viewer/timeline_view/_grid.py:136`
- Test: `tests/unit/gui/results_viewer/test_metadata_prefix_predicates.py`

**Interfaces:** Consumes `is_metadata_header`, `metadata_category_prefixes` (B1).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/gui/results_viewer/test_metadata_prefix_predicates.py
from phenotypic.gui.results_viewer.colony_view._grid import selectable_axis_columns


def test_renamed_metadata_columns_bucket_first():
    cols = ["Shape_Area", "MetadataGenetic_Strain", "Grid_Row"]
    out = selectable_axis_columns(cols)
    # metadata column must sort into the metadata bucket (first), not be missed
    assert out.index("MetadataGenetic_Strain") < out.index("Shape_Area")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/gui/results_viewer/test_metadata_prefix_predicates.py -v`
Expected: FAIL (the `startswith("Metadata_")` bucket misses `MetadataGenetic_`).

- [ ] **Step 3: Replace each predicate**

- `_output_root.py:547`, `colony_view/_grid.py:98`, `timeline_view/_grid.py:136`: replace `column.startswith(_METADATA_PREFIX)` / `name.startswith("Metadata_")` with `is_metadata_header(column)` (import from `phenotypic.sdk_`). Delete the now-unused `_METADATA_PREFIX` constants. Keep `_GRID_PREFIX` as-is.
- `_viewer_card.py:105`: delete the unused `_METADATA_PREFIX` constant; inline any docstring references.
- `_analysis_helpers.py:22`: change `SCATTER_PREFIX_PRIORITY = ("Metadata_", "Grid_", ...)` to `SCATTER_PREFIX_PRIORITY = (*metadata_category_prefixes(), "Grid_", "Shape_", "Intensity_", "Color_")`.
- `_scatter_plot.py:89`: replace the hardcoded JS `var prefixes = ['Metadata_', ...]` by injecting the Python tuple — render `f"var prefixes = {json.dumps([*metadata_category_prefixes(), 'Grid_', 'Shape_', 'Intensity_', 'Color_'])};"` into the JS so Python is the single source of truth.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/gui/results_viewer/test_metadata_prefix_predicates.py tests/unit/gui/results_viewer -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_dashboard src/phenotypic/gui/results_viewer tests/unit/gui/results_viewer/test_metadata_prefix_predicates.py
git commit -m "refactor: route metadata-prefix predicates through is_metadata_header"
```

---

## Task B4: schema-aware prefixing in post ops + `insert_metadata`

**Files:**
- Modify: `src/phenotypic/post/_utils.py` (`_PREFIX`, `_ensure_prefix`)
- Modify: `src/phenotypic/post/_merge_metadata.py`, `_append_string.py`, `_prepend_string.py`, `_expand_metadata.py` (callers)
- Modify: `src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py:322-323`
- Test: `tests/unit/post/test_schema_aware_prefix.py`

**Interfaces:** Consumes `metadata_category_for_label`, `is_metadata_header` (B1).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/post/test_schema_aware_prefix.py
from phenotypic.post._utils import ensure_metadata_prefix


def test_known_label_gets_category_prefix():
    assert ensure_metadata_prefix("Strain") == "MetadataGenetic_Strain"
    assert ensure_metadata_prefix("Dataset") == "MetadataExperiment_Dataset"


def test_unknown_label_gets_generic_prefix():
    assert ensure_metadata_prefix("MyCustomTag") == "Metadata_MyCustomTag"


def test_already_prefixed_passthrough():
    assert ensure_metadata_prefix("MetadataGenetic_Strain") == "MetadataGenetic_Strain"
    assert ensure_metadata_prefix("Metadata_MyCustomTag") == "Metadata_MyCustomTag"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/post/test_schema_aware_prefix.py -v`
Expected: FAIL (`ensure_metadata_prefix` not defined).

- [ ] **Step 3: Implement schema-aware prefixing**

Replace the body of `src/phenotypic/post/_utils.py`:

```python
from phenotypic.sdk_ import is_metadata_header, metadata_category_for_label

_GENERIC_PREFIX = "Metadata_"


def ensure_metadata_prefix(name: str) -> str:
    """Prefix a bare metadata label with its schema category, else generic.

    'Strain' -> 'MetadataGenetic_Strain'; unknown 'Foo' -> 'Metadata_Foo'
    (kept, uncategorized). Already-prefixed names pass through unchanged.
    """
    if is_metadata_header(name) or name.startswith(_GENERIC_PREFIX):
        return name
    category = metadata_category_for_label(name)
    return f"{category}_{name}" if category else f"{_GENERIC_PREFIX}{name}"
```

Update the four post ops to call `ensure_metadata_prefix(...)` wherever they previously called `_ensure_prefix(...)` / used `_PREFIX`.

In `_metadata_accessor.py:322-323`, replace:

```python
            if not is_metadata_header(key):
                header = ensure_metadata_prefix(key)
            else:
                header = key
```

(import `is_metadata_header` from `phenotypic.sdk_` and `ensure_metadata_prefix` from `phenotypic.post._utils`, or relocate `ensure_metadata_prefix` to `sdk_/_metadata_helpers.py` to avoid the post→core dependency — preferred; then import both from `phenotypic.sdk_`.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/post tests/unit/core -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/post src/phenotypic/_core/_image_parts/accessors/_metadata_accessor.py src/phenotypic/sdk_ tests/unit/post/test_schema_aware_prefix.py
git commit -m "feat(post): schema-aware metadata prefixing with generic fallback"
```

---

## Task B5: retire `Metadata_ImageFile` → `Metadata_ImageName` (+ thread `FileSuffix`)

**Files:**
- Modify creation sites: `_cli/_cli_chunk_writer.py:237`, `_cli/_cli_output_manager.py:908-911`, `_cli/_dashboard/_analysis_data.py:147-151`, `_cli/_cli_recompile_worker.py:154-161`
- Modify key constants: `gui/results_viewer/_curation_labels.py:36`, `_filtered_state.py:53`, `_error_tab/_data.py:44`, `_heatmap_tab/_figure.py:69`, `_qc_tab/review/_data.py:31`
- Modify: `analysis/abc_/_quality_check.py:313-353`
- Test: `tests/unit/cli/test_imagename_consolidation.py`

**Interfaces:** the curation/QC key column becomes `str(METADATA.IMAGE_NAME)` (`"MetadataImage_ImageName"`); `str(METADATA.SUFFIX)` (`"MetadataImage_FileSuffix"`) is written alongside.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/cli/test_imagename_consolidation.py
import polars as pl
from phenotypic.schema import METADATA
from phenotypic._cli._cli_chunk_writer import _attach_image_identity  # see Step 3


def test_chunk_writer_emits_imagename_and_suffix():
    df = pl.DataFrame({"x": [1]})
    out = _attach_image_identity(df, stem="plate1", suffix=".tif")
    assert str(METADATA.IMAGE_NAME) in out.columns          # MetadataImage_ImageName
    assert str(METADATA.SUFFIX) in out.columns              # MetadataImage_FileSuffix
    assert "Metadata_ImageFile" not in out.columns
    assert out[str(METADATA.IMAGE_NAME)][0] == "plate1"
    assert out[str(METADATA.SUFFIX)][0] == ".tif"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/cli/test_imagename_consolidation.py -v`
Expected: FAIL.

- [ ] **Step 3: Update creation sites**

At each creation site, replace the `.alias("Metadata_ImageFile")` (stem) with the canonical column and add the suffix. Extract a small helper (`_attach_image_identity`) in `_cli_chunk_writer.py` so it's unit-testable:

```python
from phenotypic.schema import METADATA

def _attach_image_identity(df, stem: str, suffix: str):
    return df.with_columns(
        pl.lit(stem).alias(str(METADATA.IMAGE_NAME)),
        pl.lit(suffix).alias(str(METADATA.SUFFIX)),
    )
```

For the regex-extract sites (`_cli_output_manager.py:910`, `_dashboard/_analysis_data.py:150`, `_cli_recompile_worker.py:160`), capture both stem and extension and alias to `str(METADATA.IMAGE_NAME)` and `str(METADATA.SUFFIX)` respectively (regex `r"([^/\\]+)(\.[^.]+)$"` → group 1 = stem, group 2 = suffix).

- [ ] **Step 4: Update consumption key constants**

Set each key constant to the canonical column:

```python
from phenotypic.schema import METADATA
KEY_IMAGE_FILE: str = str(METADATA.IMAGE_NAME)   # was "Metadata_ImageFile"
```

in `_curation_labels.py`, `_filtered_state.py`, `_error_tab/_data.py`, `_heatmap_tab/_figure.py` (`_META_IMAGE_FILE`), `_qc_tab/review/_data.py` (`_KEY_IMAGE_FILE`). The `(key, Object_Label)` join logic is unchanged. `analysis/abc_/_quality_check.py` reads the column via the constant / dynamically — verify its references use the constant, not a literal; update any literal to `str(METADATA.IMAGE_NAME)`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/unit/cli/test_imagename_consolidation.py tests/unit/gui/results_viewer tests/unit/analysis -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli src/phenotypic/gui/results_viewer src/phenotypic/analysis/abc_/_quality_check.py tests/unit/cli/test_imagename_consolidation.py
git commit -m "feat!: retire Metadata_ImageFile in favor of Metadata_ImageName + FileSuffix"
```

---

## Task B6: curation rename-on-load shim

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_curation_labels.py` (the parquet read path)
- Test: `tests/migration/test_curation_imagefile_rename.py`

**Interfaces:** on read, if `curation_labels.parquet` has a legacy `Metadata_ImageFile` column and lacks `MetadataImage_ImageName`, rename it so old curation state survives.

- [ ] **Step 1: Write the failing test**

```python
# tests/migration/test_curation_imagefile_rename.py
import polars as pl
from phenotypic.schema import METADATA
from phenotypic.gui.results_viewer._curation_labels import _migrate_legacy_imagefile


def test_legacy_imagefile_column_renamed():
    legacy = pl.DataFrame({"Metadata_ImageFile": ["p1"], "Object_Label": [3]})
    out = _migrate_legacy_imagefile(legacy)
    assert str(METADATA.IMAGE_NAME) in out.columns
    assert "Metadata_ImageFile" not in out.columns
    assert out[str(METADATA.IMAGE_NAME)][0] == "p1"


def test_new_frame_unchanged():
    new = pl.DataFrame({str(METADATA.IMAGE_NAME): ["p1"], "Object_Label": [3]})
    out = _migrate_legacy_imagefile(new)
    assert out.columns == new.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/migration/test_curation_imagefile_rename.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the shim**

In `_curation_labels.py`:

```python
from phenotypic.schema import METADATA

_LEGACY_IMAGE_FILE = "Metadata_ImageFile"


def _migrate_legacy_imagefile(df):
    canonical = str(METADATA.IMAGE_NAME)
    if _LEGACY_IMAGE_FILE in df.columns and canonical not in df.columns:
        return df.rename({_LEGACY_IMAGE_FILE: canonical})
    return df
```

Call `_migrate_legacy_imagefile(df)` immediately after reading `curation_labels.parquet` (in the load/`_read_parquet` path), before any join on `KEY_COLUMNS`.

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/migration/test_curation_imagefile_rename.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_curation_labels.py tests/migration/test_curation_imagefile_rename.py
git commit -m "feat(gui): rename-on-load shim for legacy curation Metadata_ImageFile"
```

---

## Task B7: de-stringly-type specific literals + ICC default

**Files (~30):** `analysis/qc/*` (`_icc.py`, `_tukey_fraction.py`, `_relative_mad.py`, `_max_modz.py`, `_replicate_agreement.py`, `_expected_vs_detected.py`, `_grid_occupancy.py`), `analysis/abc_/_model_fitter.py`, `_edge_correction.py`, `_quality_check.py`, `tune/score/_supervised.py`, `_composite.py`, `_qc_scorer.py`, `post/_merge_metadata.py`, `_expand_metadata.py`, `gui/analysis/_callbacks.py`, `gui/tune/_space.py`, `_setup_authoring.py`, `_cli/_cli_parquet_agg.py`, `_cli_output_manager.py`, `_cli_recompile_worker.py`, `_cli_chunk_writer.py`, `_dashboard/_analysis_helpers.py`, `_dashboard/_analysis/_summary_stats.py`, `abc_/_post_measurement.py`, `_core/_image_parts/_image_io_handler.py`, `gui/results_viewer/_filtered_state.py`, `_qc_tab/review/_data.py`, `_heatmap_tab/_callbacks.py`, `_figure.py`.
- Test: `tests/unit/analysis/test_icc_replicate_default.py`

**Interfaces:** every schema-backed `Metadata_<X>` literal becomes `str(ENUM.MEMBER)`.

**Replacement map (apply throughout):**

| Literal | Replacement |
|---|---|
| `"Metadata_Time"` | `str(CULTURE_METADATA.TIME)` |
| `"Metadata_Dataset"` | `str(EXPERIMENT_METADATA.DATASET)` |
| `"Metadata_ImageName"` | `str(METADATA.IMAGE_NAME)` |
| `"Metadata_Strain"` | `str(GENETIC_METADATA.STRAIN)` |
| `"Metadata_SampleID"` / `"Metadata_ID"` | `str(SAMPLE_METADATA.SAMPLE_ID)` |
| `"Metadata_Temp"` | `str(CULTURE_METADATA.TEMPERATURE)` |
| `"Metadata_Well"` | `str(SAMPLE_METADATA.SOURCE_WELL)` |
| `"Metadata_Replicate"` (ICC default) | `str(SAMPLE_METADATA.BIO_REPLICATE)` |

Leave genuinely-arbitrary doctest example columns (`"Metadata_Condition"` in `_merge_metadata` doctest, `"Metadata_Flag"` in `_post_measurement` doctest) **as-is** — they demonstrate that arbitrary non-vocabulary columns are accepted.

- [ ] **Step 1: Write the failing ICC test**

```python
# tests/unit/analysis/test_icc_replicate_default.py
from phenotypic.schema import SAMPLE_METADATA
from phenotypic.analysis.qc._icc import ICC  # adjust to the real class name


def test_icc_default_replicate_is_bio_replicate():
    assert ICC().replicate_label == str(SAMPLE_METADATA.BIO_REPLICATE)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/analysis/test_icc_replicate_default.py -v`
Expected: FAIL (default still `"Metadata_Replicate"`).

- [ ] **Step 3: Apply the replacements**

For each file: add the needed `from phenotypic.schema import ...` imports and swap each literal per the map. Watch the **context-aware** spots:
- Polars: `.alias("Metadata_Dataset")` → `.alias(str(EXPERIMENT_METADATA.DATASET))`.
- Dash callback dict literals in `gui/analysis/_callbacks.py`: `{"to_column": "Metadata_Strain"}` → `{"to_column": str(GENETIC_METADATA.STRAIN)}`.
- Class field defaults: `time_label: str = "Metadata_Time"` → use `default_factory` or module-level `str(CULTURE_METADATA.TIME)` constant (pydantic models can't call a function in an annotated default; assign a module-level constant `_TIME = str(CULTURE_METADATA.TIME)` and use `= _TIME`).

- [ ] **Step 4: Run the broad suite**

Run: `uv run pytest tests/unit/analysis tests/unit/tune tests/unit/post tests/unit/cli -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic tests/unit/analysis/test_icc_replicate_default.py
git commit -m "refactor!: de-stringly-type metadata columns via schema enum refs"
```

---

## Task B8: test sweep + grep gate + docs

**Files:**
- Modify: ~128 test files referencing `Metadata_*` (update expected column names).
- Create: `tests/unit/schema/test_no_metadata_literals.py` (grep gate).
- Modify: `docs/source/...` schema/measurements reference + a short REMBI explanation page; release/migration note.

- [ ] **Step 1: Write the grep-gate test**

```python
# tests/unit/schema/test_no_metadata_literals.py
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "phenotypic"
# the ONLY legal bare-"Metadata_" literal homes: the generic fallback + helpers
_ALLOWED = {
    "sdk_/_metadata_helpers.py",
    "post/_utils.py",                       # _GENERIC_PREFIX fallback
    "gui/results_viewer/_curation_labels.py",  # _LEGACY_IMAGE_FILE shim
}
_PAT = re.compile(r"""["']Metadata_[A-Za-z]""")  # specific-column literal


def test_no_specific_metadata_literals_outside_allowed():
    offenders = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        if rel in _ALLOWED or "/schema/" in f"/{rel}":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _PAT.search(line) and "Metadata_ImageFile" not in line:
                # allow doctest example columns explicitly tagged
                if "# noqa: metadata-literal" in line:
                    continue
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "stringly-typed metadata literals remain:\n" + "\n".join(offenders)
```

- [ ] **Step 2: Run it — expect failures, then fix the test files**

Run: `uv run pytest tests/unit/schema/test_no_metadata_literals.py -v`
Expected: FAIL listing remaining literals. Update each test file's expected column names to the new prefixes; tag the two intentional doctest example columns with `# noqa: metadata-literal`.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q` (and `uv run pytest -m "smoke or slow" -q` for a nightly-equivalent pass on a final check). Fix any remaining column-name mismatches.

- [ ] **Step 4: Lint + type-check**

Run: `uv run ruff check --fix && uv run mypy src/phenotypic`
Expected: clean.

- [ ] **Step 5: Docs + migration note**

Add a short REMBI explanation page (reuse the §9 mapping table from the spec) under `docs/source/explanation/`; update the measurements/schema reference for the new category prefixes; add a release/migration note: "Metadata columns are now per-REMBI-module prefixed (`MetadataGenetic_Strain`); `Metadata_ImageFile` is consolidated into `MetadataImage_ImageName` (+ `MetadataImage_FileSuffix`). Old output folders: curation state auto-migrates on load; re-run `--mode recompile` to refresh measurement parquets."

- [ ] **Step 6: Commit**

```bash
git add tests docs src/phenotypic
git commit -m "test/docs: update metadata column names; grep gate; REMBI docs + migration note"
```

---

## Execution & Parallelization

**Dependency DAG** (→ = "must finish before"):

```
PHASE A
  A1  ─┬─► A2 ─┐
       │       ├─► A4 ─┬─► A5 ─► A6
  A1b ─┘  A3 ─┘       └─► A7
  (A1b is logic-independent of A1; both edit schema/__init__.py — sequence them)

PHASE B   (decouple-then-flip)
  B1 ─┬─► B3 ─┐
      ├─► B4 ─┤
      ├─► B5 ─┼─► B2 ─► B8
      ├─► B6 ─┤   (flip)  (gate+docs)
      └─► B7 ─┘
```

**Critical path:** A1 → A2 → A4 → A5 → A6  (Phase A), then B1 → {decouple} → B2 → B8.

**Genuinely independent (could go to parallel worktrees):**
- `A2 ∥ A3` — A2 edits the 8 enum files; A3 creates `_study.py`. *Overlap:* both
  add an export line to `schema/__init__.py` + `_experimental_tags/__init__.py` →
  trivial conflict; resolve by sequencing just the export edit.
- `A5 ∥ A7` — manifest builder vs. metadata-accessor; disjoint files.
- `A1b ∥ A1` — logic-independent, but both touch `schema/__init__.py`.
- `B3, B4, B5, B6, B7` — logically independent (different subsystems), but with
  real file overlaps: **B5 ∩ B7** share the CLI files (`_cli_output_manager.py`,
  `_cli_recompile_worker.py`, `_cli_chunk_writer.py`); **B3 ∩ B4** share
  `_metadata_accessor.py`. Parallel worktrees on these would conflict.

**Recommendation:** run **subagent-driven, sequential, with a review gate after
each task**, following the DAG order, and **pause at the Phase A checkpoint** (PR1)
before starting Phase B. The file overlaps above mean parallel worktrees buy
little here and add merge cost; the DAG's value is *ordering flexibility* (e.g.
slot A3 wherever convenient relative to A2; do A7 before or after A5/A6) and
knowing which one or two disjoint pairs (`A5 ∥ A7`) are safe to fan out if you
want wall-clock speed. The hard serialization points are: **A1 first**, **A4 after
A2+A3**, **B1 before the decouple group**, **B2 after the whole decouple group**,
**B8 last**.

---

## Self-Review Notes (for the executor)

- **Spec coverage:** Phase A covers spec §3–6 (classification, STUDY enum, manifest, read-side) + §9 mapping in docs. Phase B covers §10 (rename) + §11 (execution scope: helpers, predicates, post-op prefixing, ImageFile consolidation, curation shim, de-stringly-typing, tests).
- **Type/name consistency:** the curation key column is `str(METADATA.IMAGE_NAME)` everywhere (B5/B6/B7); the three helpers (`is_metadata_header`, `metadata_category_prefixes`, `metadata_category_for_label`) are defined in B1 and consumed in B3/B4/B7; `ensure_metadata_prefix` is defined in B4.
- **Known follow-ups the executor must resolve against the real code (not placeholders — verify exact line/var names at edit time):** the exact variable carrying the mirror DataFrame + per-image metadata inside `finalize_post_master_outputs` (A6 Step 6); the real ICC class/attr name (B7 Step 1); the exact `selectable_axis_columns` signature (B3 Step 1). These are interface lookups, not undefined behavior — confirm by reading the cited file before editing.
