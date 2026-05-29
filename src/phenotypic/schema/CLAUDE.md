# Schema Module

Public, blessed API for PhenoTypic's measurement naming conventions.

- `MeasurementInfo` (`_measurement_info.py`) — `str, Enum` base. Subclasses
  declare `(label, description)` members and a `category()` classmethod; the
  enum value is the category-prefixed header (e.g. `Shape_Area`). Helpers:
  `get_labels()`, `get_headers()`, `rst_table()`, `append_rst_to_doc()`.
- 24 enum modules (`_shape.py`, `_size.py`, `_color_lab.py`, …) — one
  `MeasurementInfo` subclass each, re-exported from `__init__.py`.

Downstream users import headers directly:

    from phenotypic.schema import SHAPE, MeasurementInfo
    SHAPE.get_headers()  # ['Shape_Area', 'Shape_Perimeter', ...]

Conventions: one class per file; bodies are pure data + `category()`; import
**only** stdlib and the sibling base (no other `phenotypic` imports) to keep
the package import-light and preserve the package load-order trick in
`phenotypic/__init__.py` (`abc_` imports the stdlib-only base from here before
`tools_.constants_` needs it). New documented constant enums that are framework
config (not measurement columns) belong in `tools_/constants_.py`, not here.
