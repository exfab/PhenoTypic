# Schema Module

Public, blessed API for PhenoTypic's measurement naming conventions.

- `MeasurementInfo` (`_measurement_info.py`) — `str, Enum` base. Subclasses
  declare `(label, description)` members and a `category()` classmethod; the
  enum value is the category-prefixed header (e.g. `Shape_Area`). Helpers:
  `get_labels()`, `get_headers()`, `rst_table()`, `append_rst_to_doc()`.
- 24 measurement-column enum modules (`_shape.py`, `_size.py`,
  `_color_lab.py`, …) — one `MeasurementInfo` subclass each, re-exported from
  `__init__.py`.
- `_metadata.py` — `METADATA`: framework-populated image bookkeeping
  (`UUID`, `ImageName`, `BitDepth`, …). `category() == "Metadata"`, so members
  render as `Metadata_<Label>`. These are `image.metadata` accessor keys set by
  the pipeline, not user input.
- `_experimental_tags/` — seven `MeasurementInfo` subclasses
  (`GENETIC_METADATA`, `SAMPLE_METADATA`, `PLATE_METADATA`,
  `CONDITION_METADATA`, `INCUBATION_METADATA`, `ACQUISITION_METADATA`,
  `EXPERIMENT_METADATA`), one per file, re-exported from `__init__.py`. All
  return `category() == "Metadata"`, so the grouping is organizational and every
  member shares the `Metadata_` namespace (`SAMPLE_METADATA.REPLICATE` →
  `Metadata_Replicate`). A **recommended vocabulary, not a validator**: it
  standardizes `--metadata` CSV columns + `post/` ops but arbitrary columns are
  still accepted.

Downstream users import headers directly:

    from phenotypic.schema import SHAPE, MeasurementInfo
    SHAPE.get_headers()  # ['Shape_Area', 'Shape_Perimeter', ...]

Conventions: one class per file (or per file under a grouping subpackage like
`_experimental_tags/`); bodies are pure data + `category()`; import **only**
stdlib and the sibling base (no other `phenotypic` imports) to keep the package
import-light and preserve the package load-order trick in
`phenotypic/__init__.py` (`abc_` imports the stdlib-only base from here before
`tools_.constants_` needs it). Metadata-naming enums (`METADATA` + the
experimental tags) live here because they name `Metadata_*` columns/keys.
Framework-config enums that are *not* about naming columns/keys (e.g.
`GAMMA_ENCODINGS`, `PIPE_STATUS`, `IMAGE_MODE`) stay in `tools_/constants_.py`.
