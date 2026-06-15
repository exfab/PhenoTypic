# MeasurementInfo Revamp — Design

**Date:** 2026-06-11
**Branch:** `worktree-measurement-info-revamp`
**Status:** Approved design → ready for implementation plan

## 1. Motivation

`MeasurementInfo` members are currently declared as positional 2-tuples
`(label, desc)`. We want to attach richer, optional documentation to each
metric:

- **`bio_desc`** — the metric's *biological* relevance / use-case (microbiology
  context), distinct from the algorithmic description.
- **`image`** — a path to an illustrative figure (under `_assets/`) that the
  Sphinx docs embed to help users understand the metric.

`desc` keeps its current meaning and content (the technical/algorithm
description). We are **not** renaming it to `tech_desc`, and we are **not**
splitting today's blended descriptions — that is deliberately out of scope (see
§10).

Adding two more positional tuple fields would be ambiguous and swap-prone
(`("Area", "tech…", "bio…", "img.png")`). The goal is a **stricter, uniform,
self-documenting** declaration that still reads like a string in normal use.

## 2. Goals & Non-Goals

### Goals
- One **universal** declaration format for every `MeasurementInfo` member.
- New fields are **optional / opt-in** per member.
- **Strict input**: impossible to swap `bio_desc`/`image`; raw tuples become
  import-time errors.
- Members still behave as strings (`str(SHAPE.AREA) == "Shape_Area"`); all
  existing `.value`/`.label`/`.desc` semantics are preserved **byte-for-byte**.
- `schema` stays **stdlib-only and import-light** (no pydantic, no new heavy
  imports) — preserves the package load-order trick in `phenotypic/__init__.py`.
- Sphinx renders `bio_desc` and `image` with **conditional columns** (a column
  appears only when ≥1 member in the enum populates it).
- Reference navigation groups enums into **5 captioned groups**.

### Non-Goals
- No rename of `desc`; no `tech_desc`.
- No rewrite/splitting of existing description text.
- No bulk authoring of `bio_desc`/`image` content — only **one worked example**
  end-to-end (§9).
- No change to measurement algorithms, DataFrame columns, or serialization.

## 3. The `Entry` wrapper

New, lives in `schema/_measurement_info.py` (stdlib only):

```python
from dataclasses import dataclass, KW_ONLY

@dataclass(frozen=True, slots=True)
class Entry:
    """Declarative value for a MeasurementInfo member.

    label/desc are positional (matching the old 2-tuple ergonomics); the new
    optional fields are keyword-only, so they can never be positionally swapped.
    """
    label: str
    desc: str = ""
    _: KW_ONLY                     # everything below is keyword-only
    bio_desc: str = ""             # biological relevance / use-case
    image: str | None = None       # relative path under _assets/measurements/

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("Entry.label must be a non-empty string")
        for fld in ("desc", "bio_desc"):
            if not isinstance(getattr(self, fld), str):
                raise TypeError(f"Entry.{fld} must be a string")
        if self.image is not None and not isinstance(self.image, str):
            raise TypeError("Entry.image must be a string path or None")
```

- **Name:** `Entry` (neutral — reads correctly for measurements, metadata, and
  config enums alike).
- **Public:** exported from `phenotypic.schema` (users subclass
  `MeasurementInfo` and must use `Entry`).
- `frozen=True, slots=True` — immutable, lightweight; it is a *non-tuple* object,
  which is what lets `__new__` reject anything that isn't an `Entry`.

### Why keyword-only optionals
`label`/`desc` stay positional so the lean case is barely more verbose than the
old tuple: `Entry("Perimeter", "Boundary length in px.")`. `bio_desc`/`image`
are keyword-only, so `Entry("Area", "tech", "bio")` is a `TypeError` — the swap
footgun is structurally impossible.

## 4. `MeasurementInfo.__new__` change

Replace the `(label, desc=None)` signature with a single `Entry` argument:

```python
def __new__(cls, entry: "Entry"):
    if not isinstance(entry, Entry):
        raise TypeError(
            f"{cls.__name__} members must be declared as Entry(...); "
            f"got {entry!r}. Raw tuples/strings are no longer accepted."
        )
    full = f"{cls.category()}_{entry.label}"
    obj = str.__new__(cls, full)
    obj._value_ = full
    obj.label = entry.label
    obj.desc = entry.desc
    obj.bio_desc = entry.bio_desc
    obj.image = entry.image
    obj.pair = (entry.label, entry.desc)   # unchanged: (label, desc)
    return obj
```

- **Enum mechanics:** an `Entry` is *not* a tuple, so Python passes it to
  `__new__` as a single argument (a tuple RHS would be unpacked — that's exactly
  the legacy path we are removing).
- **New instance attributes:** `.bio_desc: str`, `.image: str | None`.
- **Preserved:** `.label`, `.desc`, `.pair == (label, desc)`, `.CATEGORY`,
  `__str__`, value-by-string lookup (`SHAPE("Shape_Area")`).
- Declare `bio_desc`/`image` as bare class-level annotations (like the existing
  `label`/`desc`/`pair`) so type checkers see them without creating enum members.

## 5. Universal rewrite (value-preserving)

Every member of every `MeasurementInfo` subclass is rewritten from a 2-tuple to
an `Entry(...)`:

- `schema/_*.py` — all ~33 enums (~200 members; member-name case varies, e.g.
  `Colorxy.x_MEAN`, `DOUBLE_SOFTPLUS_MODEL.v` — the rewrite is name-agnostic).
- `tools_/constants_.py` — `GAMMA_ENCODINGS`, `PIPE_STATUS`, and the
  `ConstantLabels` base path (import `Entry` from `schema`).

```python
# before
AREA = ("Area", "Total number of pixels …")
# after
AREA = Entry("Area", "Total number of pixels …")
```

**Invariant — the rewrite changes only syntax, never identity.** For every
member, `(name, value, label, desc)` is identical before and after. This is what
makes a ~200-member change safe and keeps every existing `.desc` consumer
(`util/_measurement_outputs.py`, `_cli/_cli_readme_generator.py`,
`QUALITY_CHECK`) and `GAMMA_ENCODINGS._GAMMA_COERCE` working untouched.

**Safety net:** snapshot `[(m.name, m.value, m.label, m.desc) for m in cls]` for
all subclasses from the **pre-change** tree into a committed golden fixture, then
assert the post-change tree matches exactly (§8).

## 6. Assets & packaging

- **Location:** `src/phenotypic/_assets/measurements/<relpath>` (e.g.
  `shape/area.png`). `_assets` is already a packaged subpackage.
- **`image` field stores the explicit relative path** under that root
  (`image="shape/area.png"`) — no implicit category-munging ("explicit over
  magic").
- **Resolution stays out of `schema`:** the field is just a string. Joining it to
  the assets root / docs URL is done by the renderer and the docs generator, so
  `schema` gains no filesystem or path imports at module scope.
- **Packaging:** add to `[tool.setuptools.package-data]` under `phenotypic`:
  ```toml
  "_assets/measurements/*",
  "_assets/measurements/*/*",
  ```
  so the images ship in the wheel (and are available to any future runtime/GUI
  consumer).

## 7. Rendering — one conditional-column table everywhere

### 7.1 Shared renderer
Introduce a single private helper in `schema/_measurement_info.py`:

```python
_ASSET_URL_PREFIX: Final = "/_static/measurements"   # source-root-absolute

def _render_info_table(rows, *, title, name_header="Name") -> str:
    """rows: list of (name_cell, desc, bio_desc, image_relpath_or_None)."""
```

Behavior:
- Always emits `name_header | Description`.
- Emits a **Biology** column iff any row has non-empty `bio_desc`.
- Emits an **Image** column iff any row has non-empty `image`; the cell is a
  nested `.. image:: {_ASSET_URL_PREFIX}/{relpath}` (root-absolute so it resolves
  from *any* embedding location), `:width:` constrained for thumbnails.
- Escapes RST-hostile cell text. The small role/`|` escaper (`_RST_ROLE_RE` +
  `_rst_cell_text`) currently in the generator is **ported into this schema
  helper** (stdlib `re` only); the generator imports it for its own prose
  escaping so there is a single implementation.

### 7.2 Callers collapse onto the helper
- `MeasurementInfo.rst_table(...)` builds rows `(m.label, m.desc, m.bio_desc,
  m.image)` and calls the helper → docstrings now get the conditional columns
  (per the requirement *"the rst_table in docstrings should be the conditional
  column table"*).
- `rst_table` gains a flag to use the **prefixed header** (`m.value`) instead of
  `m.label` for the name cell. The generator's near-duplicate
  `_full_column_table` is **deleted** and replaced by
  `info_cls.rst_table(use_headers=True)`.
- `QUALITY_CHECK.append_rst_to_doc(..., check_name=…)` builds rows with
  `QC_<slug>_<label>` name cells and calls the same helper (its `bio_desc`/
  `image` are empty, so suppression keeps it text-only; the `check_name`
  substitution is preserved).

### 7.3 Why root-absolute image paths
The same table string is embedded in **two** doc locations: operation autodoc
pages (`api_reference/…`, via `append_rst_to_doc` baked into `__doc__` at import)
and the generated `measurements_ref/…` pages. A relative path can't resolve from
both; `/_static/measurements/<relpath>` resolves from either because Sphinx
treats a leading `/` as relative to the source root.

### 7.4 Layering note
`schema` already owns RST emission (`rst_table`/`append_rst_to_doc` predate this
work), so adding the `_ASSET_URL_PREFIX` string constant there is consistent with
its existing responsibility — it introduces no import, only a literal.

## 8. Sphinx generator (`docs/source/_extensions/measurements_ref.py`)

- **Asset copy step:** at `builder-inited` (before autodoc reads modules), copy
  `src/phenotypic/_assets/measurements/**` → `docs/source/_static/measurements/**`
  so every `/_static/measurements/...` reference resolves. Idempotent (mirrors the
  existing `rmtree`+regenerate pattern).
- **Cell-escaping** (`_RST_ROLE_RE`/`_rst_cell_text`) moves to the schema helper
  (§7.1); the generator imports it for prose escaping rather than keeping a copy.
- **5-group toctree IA** — replace the current Measurements/Metadata 2-way split
  with five captioned groups:

  | Group | Enums |
  |-------|-------|
  | Measurements | SIZE, SHAPE, BBOX, INTENSITY, TEXTURE, ColorLab, ColorHSV, Colorxy, ColorXYZ, ColorComposition, OBJECT, GRID, GRID_SPATIAL, GRID_LINREG_STATS, GRID_SPREAD, SYMMETRIC_ZONES, RADIAL_EXPANSION |
  | Models & Analysis | LOG_GROWTH_MODEL, LINEAR_SOFTPLUS_MODEL, DOUBLE_SOFTPLUS_MODEL, EDGE_CORRECTION, MODEL_METRICS |
  | Quality Control | QUALITY_CHECK, QUALITY_COUNT, QUALITY_ICC, QUALITY_MAD, QUALITY_SE, QUALITY_TUKEY, QUALITY_ZMAX |
  | Curation & Errors | CURATION, ErrorCategory |
  | Metadata | METADATA, ACQUISITION/CONDITION/EXPERIMENT/GENETIC/INCUBATION/PLATE/SAMPLE_METADATA |

  Membership is driven by a single explicit mapping in the extension (one source
  of truth; a test asserts every public schema enum lands in exactly one group so
  new enums can't silently fall out).
- Per-enum pages keep their lead-paragraph + table; the table now carries the
  conditional Biology/Image columns via the shared renderer.

## 9. Worked example (vertical slice)

Wire **`SHAPE.AREA`** end-to-end to prove the pipeline:

```python
AREA = Entry(
    "Area",
    "Total number of pixels occupied by the microbial colony. …",  # desc unchanged
    bio_desc="Primary proxy for colony biomass and growth extent; "
             "larger area tracks more robust growth or longer incubation.",
    image="shape/area.png",
)
```

- Commit a real `src/phenotypic/_assets/measurements/shape/area.png` (an
  illustrative figure — e.g. a synthetic colony from `load_synth_yeast_plate()`
  with its area highlighted; generated by a small committed helper or by hand).
- Result: the SHAPE per-enum page and `MeasureShape`'s autodoc page both render
  the Biology + Image columns; every other enum still renders Description-only.

All other members stay `desc`-only.

## 10. Out of scope / deferred

- Splitting existing blended `desc` text into pure-algorithm vs biology.
- Authoring `bio_desc`/`image` for any member other than `SHAPE.AREA`.
- These are follow-up content passes, decoupled from this plumbing change.

### 10.1 Authoring policy (guardrail in project-root `CLAUDE.md`)

`bio_desc` makes a **biological claim** that needs literature-grade validation;
it must not be machine-generated. Add this rule to the project root `CLAUDE.md`
(near the measurement-columns guidance), and apply it **first** so it is in place
before the member rewrite:

> **MeasurementInfo authoring.** When adding a new `MeasurementInfo` member or
> editing an existing one, only author/edit the **`label`** (name) and **`desc`**
> (the technical/algorithm description of what is computed). **Never author or
> auto-fill `bio_desc`** (and leave `image` unset) — biological-relevance claims
> must be written and verified by a human domain author, not generated. Agents
> may scaffold the `Entry(...)` and populate `label`/`desc`, but must leave
> `bio_desc=""`/`image=None` for human authoring.

This guardrail binds the implementation itself: the universal rewrite (§5) and the
worked example (§9) populate **only** `label`/`desc` for every member; the lone
exception is `SHAPE.AREA`, whose `bio_desc`/`image` the human author supplies.

## 11. Touch points

| File / area | Change |
|-------------|--------|
| `schema/_measurement_info.py` | Add `Entry`; rewrite `__new__`; add `.bio_desc`/`.image`; add `_render_info_table` + `_ASSET_URL_PREFIX`; refactor `rst_table` (conditional columns + `use_headers` flag); update docstring doctests to use `Entry` |
| `schema/__init__.py` | Export `Entry` |
| `schema/_*.py` (~33) | Rewrite all members to `Entry(...)` |
| `schema/_quality_check.py` | Rewrite members; route override through shared helper |
| `schema/_shape.py` | `AREA` worked example (`bio_desc` + `image`) |
| `tools_/constants_.py` | Rewrite `GAMMA_ENCODINGS`/`PIPE_STATUS` members; import `Entry` |
| `docs/source/_extensions/measurements_ref.py` | 5-group IA; delete `_full_column_table` → `rst_table(use_headers=True)`; asset copy step |
| `pyproject.toml` | Add `_assets/measurements/**` to package-data |
| `src/phenotypic/_assets/measurements/shape/area.png` | New example asset |
| **project-root `CLAUDE.md`** | Add the §10.1 authoring guardrail (applied first) |
| `schema/CLAUDE.md` | Document `Entry`, new fields, rendering |
| Doctests elsewhere | Grep for `MeasurementInfo` subclasses built with tuples; convert to `Entry` |

## 12. Testing

- **Strictness:** declaring a member with a raw tuple/string raises `TypeError`
  at class creation; `Entry("a","b","c")` (3rd positional) raises `TypeError`
  (keyword-only); empty `label` raises `ValueError`.
- **Format smoke test (every member uses `Entry`):** after `import phenotypic`
  (registers all enum modules), walk **every concrete `MeasurementInfo` subclass**
  recursively via `MeasurementInfo.__subclasses__()` — this catches the schema
  enums *and* `ConstantLabels`-derived `GAMMA_ENCODINGS`/`PIPE_STATUS`, not just
  `schema.__all__`. Assert each member exposes `.label: str`, `.desc: str`,
  `.bio_desc: str`, `.image: (str | None)` with correct types. Because `__new__`
  rejects any non-`Entry` value, a member's mere existence already proves it was
  declared via `Entry`; this test makes that guarantee explicit and guards against
  attribute/type drift (and asserts no subclass was missed).
- **Value-preserving golden:** committed snapshot of `(name, value, label, desc)`
  for all subclasses; post-rewrite must match exactly.
- **New attributes:** `.bio_desc`/`.image` present on all members; default `""` /
  `None`; `SHAPE.AREA` carries the example values.
- **Image resolution:** every declared `image` resolves to an existing file under
  `src/phenotypic/_assets/measurements/` (fails CI on a dangling reference).
- **Conditional rendering:** `rst_table` omits Biology/Image columns when no
  member populates them; includes them (and emits the `.. image::` directive)
  when one does; `SHAPE` shows both, a desc-only enum shows neither.
- **Group coverage:** every public schema enum maps to exactly one toctree group.
- **Docs smoke:** `sphinx-build` succeeds; the SHAPE page contains the image
  reference.
- **Regression:** existing schema / measurement-output / readme-generator tests
  pass unchanged (proves the value-preserving invariant downstream).

## 13. Open questions

None blocking. Wrapper name `Entry` is the working choice (was flagged "name
open" — `Spec`/`Column` are alternatives if preferred before implementation).
