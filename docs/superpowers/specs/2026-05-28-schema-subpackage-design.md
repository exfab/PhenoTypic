# Design: `phenotypic.tools_.measurement_info` → `phenotypic.schema`

**Date:** 2026-05-28
**Status:** Approved (brainstorming complete; ready for implementation plan)
**Branch context:** `refactor/intuitive-names`

## Goal

Promote the internal `phenotypic.tools_.measurement_info` package to a public,
discoverable `phenotypic.schema` subpackage so downstream users can import the
measurement **headers / column-name enums** to align their own DataFrames and
code with PhenoTypic outputs. The trailing-underscore-free name `schema`
signals "blessed public API", in contrast to the framework-internal `tools_`,
`abc_`, and `settings_`.

## Decisions (locked during brainstorming)

1. **Scope:** `phenotypic.schema` holds **both** the `MeasurementInfo` base
   class (moved out of `abc_`) **and** the 24 measurement-info enums. The whole
   header/column-name machinery lives in one public home.
2. **Back-compat:** **hard break.** All internal call sites are rewritten to
   import from `phenotypic.schema`; the old `phenotypic.tools_.measurement_info`
   path is deleted. (Pre-1.0 library, version 0.15.2 — hard breaks acceptable.)
3. **Layout:** **Approach A — flat mirror.** Keep the current flat,
   one-class-per-file shape; the diff is almost purely *move files + rewrite
   import strings*. Grouped subpackages are a possible non-breaking follow-up.

## Non-goals (YAGNI)

- `ConstantLabels` / `GAMMA_ENCODINGS` and other documented enums stay in
  `phenotypic.tools_.constants_`. (They subclass `MeasurementInfo`, so they
  will now import the base from `phenotypic.schema`, but they do not move.)
- No header-registry / "dump all headers" helper.
- No operation `model_json_schema()` "contract hub".
- No grouped subpackages (`schema/color/`, `schema/grid/`, …).

Each is a clean non-breaking follow-up if wanted later.

## Architecture

### 1. Target structure

New public package `src/phenotypic/schema/` (no trailing underscore):

```
phenotypic/schema/
├── __init__.py            # re-exports base + all 24 enums; defines __all__
├── _measurement_info.py   # the MeasurementInfo base class (moved from abc_)
├── _bbox.py … _texture.py # the 24 enum modules, verbatim bodies
```

- The 24 enum modules move **verbatim** except for their base-class import:
  `from phenotypic.abc_._measurement_info import MeasurementInfo`
  → relative `from ._measurement_info import MeasurementInfo`.
- `schema/__init__.py` re-exports the base class **plus** all 24 enums (the
  existing `__init__` content, extended with `MeasurementInfo`).

### 2. Import-ordering (circular-import dance)

The existing trick in `phenotypic/__init__.py` survives intact because
`schema._measurement_info` is **stdlib-only** (`enum` + `textwrap`) and the
enum modules import nothing from `phenotypic` except their sibling base
(verified: a grep for non-stdlib, non-base imports across all 24 modules
returns empty).

New load order:

1. `phenotypic/__init__` → `from . import abc_`
2. `abc_/__init__` line 9 becomes `from phenotypic.schema import MeasurementInfo`
   — this runs `schema/__init__` (base + 24 enums, all stdlib-only), caches
   everything in `sys.modules`, returns the base.
3. `abc_` then imports `tools_` → `constants_` does
   `from phenotypic.schema import MeasurementInfo` — already cached. ✓

`abc_` keeps re-exporting `MeasurementInfo` (its `__all__` and the import on
line 9 stay valid), so `from phenotypic.abc_ import MeasurementInfo` continues
to work for internal callers.

### 3. Migration mechanics (the bulk — mechanical find/replace)

Three package-import shapes to rewrite across **31 source files + 12 test
files**:

- `from phenotypic.tools_.measurement_info import …`
  → `from phenotypic.schema import …` (~33 sites)
- `from ..tools_.measurement_info import …`
  → `from phenotypic.schema import …` (~7 sites; note relative→absolute)
- 3 deep imports, e.g. `…measurement_info._quality_se import QUALITY_SE`
  → `from phenotypic.schema import QUALITY_SE`

Base-class call sites:

- The 24 enum modules (moving) → relative `from ._measurement_info import …`.
- `tools_/constants_.py` → `from phenotypic.schema import MeasurementInfo`.
- `abc_/__init__.py` line 9 → re-export from `phenotypic.schema`.

Top-level:

- `phenotypic/__init__.py`: add `schema` to the public import block + `__all__`.

Deletions (hard break):

- `phenotypic/abc_/_measurement_info.py`
- `phenotypic/tools_/measurement_info/` (entire directory)

Doctest fix: in the moved base-class module, update
`>>> from phenotypic.abc_ import MeasurementInfo`
→ `>>> from phenotypic.schema import MeasurementInfo`.

### 4. Docs, guides, packaging

- **Sphinx extension** `docs/source/_extensions/measurements_ref.py`: rewrite
  the `_REGISTRY` qualified-path strings
  `phenotypic.tools_.measurement_info.SIZE` → `phenotypic.schema.SIZE`
  (all entries).
- `docs/source/measurements_ref/index.rst` is **auto-regenerated** at build
  time — rebuild docs and commit the regenerated file; do not hand-edit.
- **CLAUDE.md updates:**
  - `src/phenotypic/abc_/CLAUDE.md` — "Standalone: MeasurementInfo (enum base)"
    re-points to `phenotypic.schema`.
  - `src/phenotypic/tools_/CLAUDE.md` — `constants_.py` note clarifies the
    `MeasurementInfo` base now lives in `phenotypic.schema`.
  - Add a short `src/phenotypic/schema/CLAUDE.md` module guide.
  - Root `CLAUDE.md` "Module Guides" list gains a `schema` entry.
- **Packaging:** none — `[tool.setuptools.packages.find]` with `where=["src"]`
  auto-discovers the new package; no `pyproject` change.
- **Historical spec** `docs/superpowers/specs/2026-05-12-*.md` left as-is.

### 5. Testing & verification

- Rewrite imports in the 12 affected test files.
- Add `tests/unit/schema/` smoke test:
  - `from phenotypic.schema import MeasurementInfo, SHAPE, SIZE` succeeds.
  - `phenotypic.schema.SHAPE.get_headers()` returns the expected prefixed names.
  - `phenotypic.abc_.MeasurementInfo is phenotypic.schema.MeasurementInfo`
    (re-export identity).
  - `import phenotypic.tools_.measurement_info` raises `ModuleNotFoundError`
    (hard break confirmed).
- Gates:
  - `uv run ruff check --fix`
  - `uv run mypy src/phenotypic`
  - `uv run pytest`
  - docs build to confirm the extension regenerates `measurements_ref/index.rst`.

## Blast radius summary

| Bucket | Count | Risk |
|--------|-------|------|
| Package modules moved | 24 enums + `__init__` (~1,420 LOC) | Low (leaf code) |
| Base class moved | `_measurement_info.py` | Medium (import ordering) |
| Source import rewrites | 31 files | Low (mechanical) |
| Test import rewrites | 12 files | Low (mechanical) |
| Docs extension + guides | ~4 files | Low |
| Top-level `__init__` + `abc_/__init__` | 2 files | Medium |
| New smoke test | 1 file | Low |

**Total ≈ 45–50 files**, the overwhelming majority being find-and-replace on
import paths. The genuinely thoughtful surface is ~6 files: the new
`schema/__init__`, the moved base class, `abc_/__init__` (re-export),
`constants_.py`, the docs extension, and `phenotypic/__init__`.
