# Metadata Schema Tags — Design & Implementation Checklist

**Date:** 2026-05-30
**Branch:** refactor/cli-output-structure
**Goal:** Move the framework `METADATA` enum out of `tools_/constants_.py` into the
public `phenotypic.schema` package, and add a standardized, documented vocabulary of
biological/experimental metadata tags so users converge on canonical `Metadata_*`
column names for inputs (`--metadata` CSV) and outputs.

---

## Design Summary

Two concepts, both rooted at `MeasurementInfo` and both rendering as `Metadata_<Label>`
(shared `category() == "Metadata"` namespace):

1. **Framework bookkeeping** — the existing `METADATA` enum (UUID, ImageName,
   BitDepth, …), auto-populated by the image pipeline. Moved verbatim (with the
   already-applied prefix fix: the prefix-stripping `__new__` override was removed, so
   members now render as `Metadata_UUID`, `Metadata_ImageName`, …).
2. **Experimental vocabulary** — 7 organizational classes grouping ~52 recommended
   tags for arrayed colony phenotyping. A *recommended vocabulary, not a validator*:
   the `--metadata` join still accepts arbitrary columns; these supply canonical
   names + descriptions + auto-generated RST tables.

### Prefix decision
All 8 classes return `category() == "Metadata"`. The 7 experimental classes are
**organizational groupings of one shared `Metadata_` namespace** (not 7 distinct
column prefixes). This matches the `post/` module, the `--metadata` CSV inner-join,
the CLI's `Metadata_Dataset`, and the QC `Metadata_Time`/`Metadata_Replicate` usages.
Labels are globally unique across all 8 classes so no two members claim the same column.

### File layout
```
schema/
  _metadata.py                  # METADATA (framework)
  _experimental_tags/
    __init__.py                 # re-exports the 7 classes
    _genetic.py                 # GENETIC_METADATA
    _sample.py                  # SAMPLE_METADATA
    _plate.py                   # PLATE_METADATA
    _condition.py               # CONDITION_METADATA
    _incubation.py              # INCUBATION_METADATA
    _acquisition.py             # ACQUISITION_METADATA
    _experiment.py              # EXPERIMENT_METADATA
```
Each module imports **only** `MeasurementInfo` from the schema base — no other
`phenotypic` imports — preserving the package's stdlib-only load-order rule.

### Tag roster (52 across 7 classes)
- **GENETIC_METADATA:** Organism, Strain, Genotype, Background, Allele, Plasmid,
  SelectionMarker, MatingType, Ploidy
- **SAMPLE_METADATA:** SampleID, Replicate, TechnicalReplicate, Clone, LibraryID,
  SourcePlate, SourceWell, Barcode, Control
- **PLATE_METADATA:** PlateID, Batch, ArrayDensity, IncubatorPosition
- **CONDITION_METADATA:** Media, CarbonSource, NitrogenSource, pH, Supplement,
  Antibiotic, Inducer, Treatment, Compound, Concentration, Dose, Stress
- **INCUBATION_METADATA:** Temperature, Time, TimeUnit, Timepoint, Day, Generation,
  Humidity, Atmosphere
- **ACQUISITION_METADATA:** ImagingDate, Instrument, Experimenter, Resolution,
  ExposureTime
- **EXPERIMENT_METADATA:** ExperimentID, Project, Dataset, Protocol, Notes

---

## Implementation Checklist

### Phase 1 — Move `METADATA` into `schema/`
- [x] Create `src/phenotypic/schema/_metadata.py` with `METADATA(MeasurementInfo)`
      (category `"Metadata"`, prefix-stripping `__new__` NOT reintroduced; clean the
      misplaced inner docstring into a proper class docstring).
- [x] Remove the `METADATA` class from `src/phenotypic/sdk_/constants_.py`.
- [x] Fix the `constants_.py` module docstring import example (drop `METADATA`).

### Phase 2 — Add the experimental tag vocabulary
- [x] `_experimental_tags/_genetic.py` → `GENETIC_METADATA`.
- [x] `_sample.py` → `SAMPLE_METADATA`.
- [x] `_plate.py` → `PLATE_METADATA`.
- [x] `_condition.py` → `CONDITION_METADATA`.
- [x] `_incubation.py` → `INCUBATION_METADATA`.
- [x] `_acquisition.py` → `ACQUISITION_METADATA`.
- [x] `_experiment.py` → `EXPERIMENT_METADATA`.
- [x] `_experimental_tags/__init__.py` re-exporting the 7 classes.

### Phase 3 — Wire up public exports
- [x] In `schema/__init__.py`, import + add to `__all__`: `METADATA` and the 7
      experimental classes.

### Phase 4 — Update all `METADATA` import sites (chosen: update, not re-export)
Source (9):
- [x] `_core/_image_parts/_grid_image_handler.py`
- [x] `_core/_image_parts/_image_io_handler.py`
- [x] `_core/_image_parts/_image_data_manager.py`
- [x] `_core/_image_parts/accessors/_metadata_accessor.py`
- [x] `_core/_image_parts/accessors/_objects_accessor.py`
- [x] `_core/_image_parts/_image_handler.py`
- [x] `_core/_image_parts/accessors/_grid_accessor.py`
- [x] `_core/_image_parts/accessor_abstracts/_multichannel_accessor.py`
- [x] `_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_mpl_handler.py`

Tests (2):
- [x] `tests/unit/tools_/test_metadata_io.py`
- [x] `tests/unit/grid/test_grid_image.py`

### Phase 5 — Docs / notes
- [x] Update `schema/CLAUDE.md` (remove "framework config stays in constants_" note;
      document `_metadata.py` + `_experimental_tags/`).
- [x] Update `tools_/CLAUDE.md` (`constants_.py` no longer hosts `METADATA`).
- [x] Update `tools_/_io_constants.py` doc comment listing `METADATA` among
      `constants_` enums.

### Phase 6 — Verify
- [x] `uv run ruff check --fix` on touched files.
- [x] `uv run mypy src/phenotypic` (no new errors).
- [x] `uv run pytest tests/unit/tools_/test_metadata_io.py tests/unit/grid/test_grid_image.py -q`.
- [x] Import smoke test → `SAMPLE_METADATA.REPLICATE.value == "Metadata_Replicate"`.

---

## Risks / Notes
- **Behavioral change (pre-existing, user-applied):** removing `METADATA.__new__`
  changes `image.metadata` dict keys + persisted HDF5/JSON metadata keys from bare
  (`"ImageName"`) to prefixed (`"Metadata_ImageName"`). Internally consistent; only
  cross-version persisted files would mismatch. Phase 6 tests gate this.
  - **Mitigated across all load paths:** `_image_io_handler._remap_legacy_metadata_key`
    (built from the `METADATA` enum) remaps legacy bare keys → prefixed on load.
    Idempotent and targeted: already-prefixed keys and arbitrary user keys pass
    through. Applied in:
    - **HDF5** — `_load_v2_grouped` and `_load_legacy_flat_group`.
    - **PNG / JPEG** — the shared `_phenotypic_data` restore block (`imread`):
      bare keys remapped (so the critical-field UUID/ImageName skip fires for
      old files), public keys remapped.
    - **Pickle** — `load_pickle` now uses `_BackCompatUnpickler`, whose
      `find_class` remaps the *moved* `METADATA` class
      (`phenotypic.sdk_.constants_` → `phenotypic.schema`) so pre-move pickles
      load at all; since enum members resolve by name, keys+values auto-upgrade
      to the prefixed members. (The plain bare-key remap is a no-op for pickles,
      whose metadata keys are enum members, not strings — the class remap is the
      real fix.) Trust model unchanged: callers load their own pickles.
    Tests: `tests/unit/tools_/test_metadata_io.py::TestLegacyMetadataKeyShim`
    (HDF5 v2 + flat, PNG/JPEG, pickle `find_class` + round-trip) + updated
    `test_back_compat_legacy_flat_hdf_loads`.
- **No circular imports:** `schema` is stdlib-only, so `_core` modules importing
  `METADATA` from `schema` introduces no cycle.
