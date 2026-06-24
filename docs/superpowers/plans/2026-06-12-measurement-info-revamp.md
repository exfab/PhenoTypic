# MeasurementInfo Revamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:
> subagent-driven-development (recommended) or superpowers:executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `MeasurementInfo`'s positional `(label, desc)` member tuples with a
strict, universal `Entry(...)` wrapper that keeps `desc` and adds optional keyword-only
`bio_desc` + `image`, rendered as conditional Sphinx columns.

**Architecture:** A frozen `Entry` dataclass becomes the sole legal member value (
`__new__` rejects everything else once migration completes). The bulk member rewrite is
value-preserving and guarded by a golden snapshot; a transient dual-accept `__new__`
keeps the suite green during the rewrite, then flips to strict. One conditional-column
table renderer is shared by docstrings, the Sphinx generator, and the `QUALITY_CHECK`
override. Images live under the packaged `_assets/measurements/` and are copied into the
docs `_static` tree at build time, referenced by a source-root-absolute path so they
resolve from any embedding.

**Tech Stack:** Python 3.12, stdlib `dataclasses`/`enum`, pydantic-free `schema`
package, pytest, Sphinx (custom `measurements_ref` extension), setuptools package-data,
`uv`.

**Spec:** `docs/superpowers/specs/2026-06-11-measurement-info-revamp-design.md`

**Working tree:** worktree `measurement-info-revamp`, branch
`worktree-measurement-info-revamp`. All paths below are relative to the worktree root.
Run everything with `uv run`.

---

## File Structure

**Created:**

- `tests/unit/schema/test_entry.py` — `Entry` construction/validation/strictness.
- `tests/unit/schema/test_measurement_info_format.py` — universal-format smoke test.
- `tests/unit/schema/test_measurement_info_golden.py` — value-preserving snapshot test.
- `tests/unit/schema/_golden/measurement_info_values.json` — committed golden snapshot.
- `tests/unit/schema/test_rst_rendering.py` — conditional-column renderer.
- `tests/unit/schema/test_measurement_assets.py` — declared-image resolution.
- `src/phenotypic/_assets/measurements/shape/area.png` — worked-example asset.
- `scripts/make_measurement_example_images.py` — regenerates example assets.

**Modified:**

- `CLAUDE.md` — authoring guardrail (Task 0).
- `src/phenotypic/schema/_measurement_info.py` — `Entry`, `__new__`, attrs,
  `_render_info_table`, `rst_table` refactor, escaping helpers, `_ASSET_URL_PREFIX`.
- `src/phenotypic/schema/__init__.py` — export `Entry`.
- `src/phenotypic/schema/_*.py` (~33 enum modules) — members → `Entry(...)`.
- `src/phenotypic/schema/_quality_check.py` — members + override via shared helper.
- `src/phenotypic/schema/_shape.py` — `AREA` worked example.
- `src/phenotypic/tools_/constants_.py` — `GAMMA_ENCODINGS`/`PIPE_STATUS` members.
- `docs/source/_extensions/measurements_ref.py` — shared renderer, 5-group IA, asset
  copy.
- `pyproject.toml` — package-data for `_assets/measurements/**`.
- `src/phenotypic/schema/CLAUDE.md` — document `Entry` + new fields + rendering.
- `tests/unit/docs/test_measurements_ref_extension.py` — group coverage + asset copy.

---

## Task 0: Authoring guardrail in project-root CLAUDE.md

Applied **first** so it is in force before any member is touched (spec §10.1).

**Files:**

- Modify: `CLAUDE.md` (Gotchas section, after the "Measurement columns are
  category-prefixed" bullet)

- [ ] **Step 1: Insert the guardrail bullet**

Find this existing bullet in `CLAUDE.md`:

```markdown
  `MeasurementInfo.get_labels()` returns unprefixed names; `get_headers()` returns the
  prefixed column names used in DataFrames.
```

Insert immediately after it:

```markdown
- **Authoring `MeasurementInfo` members:** members are declared with
  `Entry(label, desc, *, bio_desc="", image=None)` (the `Entry` value type in
  `phenotypic.schema`). When adding a new member or editing one, only author/edit
  the **`label`** (name) and **`desc`** (the technical/algorithm description of
  what is computed). **Never author or auto-fill `bio_desc`**, and leave `image`
  unset — biological-relevance claims must be written and verified by a human
  domain author, not generated. Agents may scaffold the `Entry(...)` and populate
  `label`/`desc`, but must leave `bio_desc=""`/`image=None` for human authoring.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): guardrail — agents author label/desc only, never bio_desc"
```

---

## Task 1: The `Entry` value type

**Files:**

- Modify: `src/phenotypic/schema/_measurement_info.py` (top of module, before the class)
- Test: `tests/unit/schema/test_entry.py`

- [ ] **Step 1: Write the failing test**

```python
"""Entry — the declarative value type for MeasurementInfo members."""

import pytest

from phenotypic.schema import Entry


def test_entry_minimal_defaults():
    e = Entry("Area")
    assert e.label == "Area"
    assert e.desc == ""
    assert e.bio_desc == ""
    assert e.image is None


def test_entry_positional_label_and_desc():
    e = Entry("Area", "Pixel count of the mask.")
    assert e.desc == "Pixel count of the mask."


def test_entry_optional_fields_are_keyword_only():
    # third positional must fail — bio_desc/image cannot be passed positionally
    with pytest.raises(TypeError):
        Entry("Area", "tech", "bio")  # type: ignore[misc]


def test_entry_rich():
    e = Entry("Area", "tech", bio_desc="biology", image="shape/area.png")
    assert (e.bio_desc, e.image) == ("biology", "shape/area.png")


def test_entry_is_frozen():
    e = Entry("Area")
    with pytest.raises(Exception):  # FrozenInstanceError
        e.label = "Other"  # type: ignore[misc]


def test_entry_rejects_empty_label():
    with pytest.raises(ValueError):
        Entry("")


def test_entry_rejects_non_string_desc():
    with pytest.raises(TypeError):
        Entry("Area", 123)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/schema/test_entry.py -q`
Expected: FAIL — `ImportError: cannot import name 'Entry'`.

- [ ] **Step 3: Add `Entry` to `_measurement_info.py`**

At the top of `src/phenotypic/schema/_measurement_info.py`, replace the existing
imports:

```python
from enum import Enum
from textwrap import dedent
```

with:

```python
from dataclasses import KW_ONLY, dataclass
from enum import Enum
from textwrap import dedent
from typing import Final


@dataclass(frozen=True, slots=True)
class Entry:
    """Declarative value for a :class:`MeasurementInfo` member.

    ``label`` and ``desc`` are positional (matching the old ``(label, desc)``
    tuple ergonomics). The new fields are keyword-only, so ``bio_desc``/``image``
    can never be positionally swapped with ``desc``.

    Args:
        label: Short, unprefixed measurement name (e.g. ``"Area"``).
        desc: Technical/algorithm description of what is computed.
        bio_desc: Biological relevance / use-case. Human-authored only.
        image: Path, relative to ``_assets/measurements/``, of an illustrative
            figure (e.g. ``"shape/area.png"``); ``None`` for no figure.
    """

    label: str
    desc: str = ""
    _: KW_ONLY
    bio_desc: str = ""
    image: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("Entry.label must be a non-empty string")
        for name in ("desc", "bio_desc"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"Entry.{name} must be a string")
        if self.image is not None and not isinstance(self.image, str):
            raise TypeError("Entry.image must be a string path or None")
```

- [ ] **Step 4: Export `Entry` from the package**

In `src/phenotypic/schema/__init__.py`, change:

```python
from ._measurement_info import MeasurementInfo
```

to:

```python
from ._measurement_info import Entry, MeasurementInfo
```

and add `"Entry",` to `__all__` immediately after `"MeasurementInfo",`.

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/unit/schema/test_entry.py -q`
Expected: PASS (7 passed).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/schema/_measurement_info.py src/phenotypic/schema/__init__.py tests/unit/schema/test_entry.py
git commit -m "feat(schema): add Entry value type for MeasurementInfo members"
```

---

## Task 2: `__new__` accepts `Entry` (transient dual-accept) + new attributes

The dual-accept branch is a **migration scaffold**: it keeps every existing
legacy-tuple member working while Task 4 rewrites them. Task 5 removes it.

**Files:**

- Modify: `src/phenotypic/schema/_measurement_info.py` (`__new__`, class-level
  annotations)
- Test: `tests/unit/schema/test_measurement_info_format.py`

- [ ] **Step 1: Write the failing smoke test**

```python
"""Every MeasurementInfo member exposes the universal Entry attribute surface."""

import phenotypic  # noqa: F401  (registers all enum modules)
import phenotypic.tools_.constants_  # noqa: F401  (GAMMA_ENCODINGS, PIPE_STATUS)
from phenotypic.schema import MeasurementInfo


def _all_concrete_info_classes():
    seen, out = set(), []
    stack = list(MeasurementInfo.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        # first-party enums only — excludes test/doctest-defined subclasses that
        # linger in __subclasses__() within a pytest session
        if cls.__module__.startswith("phenotypic") and len(list(cls)) > 0:
            out.append(cls)
    return out


def test_every_member_has_entry_attribute_surface():
    classes = _all_concrete_info_classes()
    assert classes, "no concrete MeasurementInfo subclasses discovered"
    for cls in classes:
        for member in cls:
            assert isinstance(member.label, str) and member.label
            assert isinstance(member.desc, str)
            assert isinstance(member.bio_desc, str)
            assert member.image is None or isinstance(member.image, str)
            assert member.pair == (member.label, member.desc)


def test_discovery_covers_known_enums():
    names = {c.__name__ for c in _all_concrete_info_classes()}
    assert {"SHAPE", "SIZE", "METADATA", "GAMMA_ENCODINGS", "PIPE_STATUS"} <= names
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/schema/test_measurement_info_format.py -q`
Expected: FAIL — members have no `bio_desc`/`image` attribute (`AttributeError`).

- [ ] **Step 3: Rewrite `__new__` (dual-accept) and declare new attributes**

In `src/phenotypic/schema/_measurement_info.py`, change the class-level annotation block
from:

```python
    label: str
    desc: str
    pair: tuple[str, str]
```

to:

```python
    label: str
    desc: str
    bio_desc: str
    image: str | None
    pair: tuple[str, str]
```

Replace the entire existing `__new__` method with:

```python
    def __new__(cls, *args):
        """Create a member from an ``Entry`` (preferred) or, transiently, a
        legacy ``(label, desc)`` tuple / bare ``label`` string.

        The legacy branch is a migration scaffold removed once all members use
        ``Entry`` (see plan Task 5). The enum value is the category-prefixed
        header (e.g. ``Shape_Area``); ``label``/``desc``/``bio_desc``/``image``
        are stored as instance attributes.
        """
        if len(args) == 1 and isinstance(args[0], Entry):
            entry = args[0]
        elif len(args) == 1 and isinstance(args[0], str):
            entry = Entry(args[0])
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
            entry = Entry(args[0], args[1])
        else:
            raise TypeError(
                f"{cls.__name__} members must be declared as Entry(...); "
                f"got {args!r}."
            )
        full = f"{cls.category()}_{entry.label}"
        obj = str.__new__(cls, full)
        obj._value_ = full
        obj.label = entry.label
        obj.desc = entry.desc
        obj.bio_desc = entry.bio_desc
        obj.image = entry.image
        obj.pair = (entry.label, entry.desc)
        return obj
```

- [ ] **Step 4: Run both schema test files**

Run:
`uv run pytest tests/unit/schema/test_measurement_info_format.py tests/unit/schema/test_entry.py -q`
Expected: PASS.

- [ ] **Step 5: Run the broader schema/regression sanity**

Run: `uv run pytest tests/unit/schema tests/unit/util/test_measurement_outputs.py -q`
Expected: PASS (existing `.desc`/`.value` consumers unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/schema/_measurement_info.py tests/unit/schema/test_measurement_info_format.py
git commit -m "feat(schema): MeasurementInfo.__new__ accepts Entry; add bio_desc/image attrs"
```

---

## Task 3: Capture the value-preserving golden snapshot

Captured **before** the rewrite (members are still legacy tuples here), so it
records the true current values that Task 4 must preserve.

**Files:**

- Create: `tests/unit/schema/_golden/measurement_info_values.json`
- Test: `tests/unit/schema/test_measurement_info_golden.py`

- [ ] **Step 1: Write the golden test (reads the snapshot)**

```python
"""The member rewrite must not change any (name, value, label, desc)."""

import json
from pathlib import Path

import phenotypic  # noqa: F401
import phenotypic.tools_.constants_  # noqa: F401
from phenotypic.schema import MeasurementInfo

_GOLDEN = Path(__file__).parent / "_golden" / "measurement_info_values.json"


def _snapshot() -> dict[str, list[list[str | None]]]:
    seen, snap = set(), {}
    stack = list(MeasurementInfo.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        if not cls.__module__.startswith("phenotypic"):
            continue  # exclude test/doctest-defined subclasses
        members = list(cls)
        if members:
            snap[f"{cls.__module__}.{cls.__name__}"] = [
                [m.name, m.value, m.label, m.desc] for m in members
            ]
    return dict(sorted(snap.items()))


def test_member_values_match_golden():
    assert _GOLDEN.exists(), "golden snapshot missing — generate it first"
    expected = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    assert _snapshot() == expected
```

- [ ] **Step 2: Generate the snapshot from the current (pre-rewrite) tree**

Run:

```bash
uv run python -c "
import json, sys
sys.path.insert(0, 'tests/unit/schema')
from test_measurement_info_golden import _snapshot, _GOLDEN
_GOLDEN.parent.mkdir(parents=True, exist_ok=True)
_GOLDEN.write_text(json.dumps(_snapshot(), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print('wrote', _GOLDEN, 'classes:', len(_snapshot()))
"
```

Expected: prints the path and a class count (~40).

- [ ] **Step 3: Run the golden test**

Run: `uv run pytest tests/unit/schema/test_measurement_info_golden.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/schema/test_measurement_info_golden.py tests/unit/schema/_golden/measurement_info_values.json
git commit -m "test(schema): golden snapshot of MeasurementInfo member values"
```

---

## Task 4: Universal member rewrite to `Entry(...)`

Value-preserving, mechanical. The golden test (Task 3) is the gate after **every**
file. Commit per group so a regression is bisectable.

**Transformation recipe** — wrap each member's right-hand side in `Entry(...)`:

| Before                                   | After                              |
|------------------------------------------|------------------------------------|
| `AREA = ("Area", "desc")`                | `AREA = Entry("Area", "desc")`     |
| `UUID = "UUID", "desc"` (bare tuple)     | `UUID = Entry("UUID", "desc")`     |
| multi-line `X = (\n  "L",\n  "desc",\n)` | `X = Entry(\n  "L",\n  "desc",\n)` |

Each rewritten module must import `Entry`. In `schema/_*.py` files, change
`from ._measurement_info import MeasurementInfo` to
`from ._measurement_info import Entry, MeasurementInfo`. In
`tools_/constants_.py`, change `from phenotypic.schema import MeasurementInfo`
to `from phenotypic.schema import Entry, MeasurementInfo`.

**Do not** add `bio_desc`/`image` to any member here (guardrail) — `SHAPE.AREA`'s
content is Task 10.

**Files (by commit group):**

- [ ] **Step 1: Group "Measurements"** — rewrite members in:
  `_size.py`, `_shape.py`, `_bbox.py`, `_intensity.py`, `_texture.py`,
  `_color_lab.py`, `_color_hsv.py`, `_color_xy.py`, `_color_xyz.py`,
  `_color_composition.py`, `_object.py`, `_grid.py`, `_grid_spatial.py`,
  `_grid_linreg_stats.py`, `_grid_spread.py`, `_symmetric_zones.py`,
  `_radial_expansion.py`.

  After editing, run the gate:
  `uv run pytest tests/unit/schema/test_measurement_info_golden.py tests/unit/schema/test_measurement_info_format.py -q`
  Expected: PASS. Then:
  ```bash
  git add src/phenotypic/schema/_size.py src/phenotypic/schema/_shape.py src/phenotypic/schema/_bbox.py src/phenotypic/schema/_intensity.py src/phenotypic/schema/_texture.py src/phenotypic/schema/_color_lab.py src/phenotypic/schema/_color_hsv.py src/phenotypic/schema/_color_xy.py src/phenotypic/schema/_color_xyz.py src/phenotypic/schema/_color_composition.py src/phenotypic/schema/_object.py src/phenotypic/schema/_grid.py src/phenotypic/schema/_neighbor_dist.py src/phenotypic/schema/_grid_linreg_stats.py src/phenotypic/schema/_grid_spread.py src/phenotypic/schema/_symmetric_zones.py src/phenotypic/schema/_radial_expansion.py
  git commit -m "refactor(schema): Entry(...) members — measurements group"
  ```

- [ ] **Step 2: Group "Models & Analysis"** — rewrite members in:
  `_log_growth_model.py`, `_linear_softplus_model.py`, `_double_softplus_model.py`,
  `_edge_correction.py`, `_model_metrics.py`.
  Run the gate; expected PASS. Commit:
  `git commit -m "refactor(schema): Entry(...) members — models & analysis group"`.

- [ ] **Step 3: Group "Quality Control"** — rewrite members in:
  `_quality_count.py`, `_quality_icc.py`, `_quality_mad.py`, `_quality_se.py`,
  `_quality_tukey.py`, `_quality_zmax.py`, and `_quality_check.py` (its three
  members `FLAG`/`METRIC`/`STATUS` → `Entry(...)`). **The members must convert
  now** so the strict flip in Task 5 doesn't break `_quality_check.py`'s import;
  its `append_rst_to_doc` override is reconciled separately in Task 6.
  Run the gate; expected PASS. Commit:
  `git commit -m "refactor(schema): Entry(...) members — quality-control group"`.

- [ ] **Step 4: Group "Curation & Errors"** — rewrite members in:
  `_curation.py`, `_error_category.py`.
  Run the gate; expected PASS. Commit:
  `git commit -m "refactor(schema): Entry(...) members — curation & errors group"`.

- [ ] **Step 5: Group "Metadata"** — rewrite members in:
  `_metadata.py` and every file under `_experimental_tags/`.
  Run the gate; expected PASS. Commit:
  `git commit -m "refactor(schema): Entry(...) members — metadata group"`.

- [ ] **Step 6: Group "Config constants"** — rewrite `GAMMA_ENCODINGS` and
  `PIPE_STATUS` members in `src/phenotypic/tools_/constants_.py` (add `Entry` to
  the `from phenotypic.schema import ...` line).
  Run the gate; expected PASS. Commit:
  `git commit -m "refactor(schema): Entry(...) members — GAMMA_ENCODINGS/PIPE_STATUS"`.

- [ ] **Step 7: Full schema regression**

Run:
`uv run pytest tests/unit/schema tests/unit/util/test_measurement_outputs.py tests/smoke/test_measurement.py -q`
Expected: PASS.

---

## Task 5: Flip `__new__` to strict (reject non-`Entry`) + update base doctests

**Files:**

- Modify: `src/phenotypic/schema/_measurement_info.py` (`__new__`, class docstring
  doctests)
- Test: `tests/unit/schema/test_entry.py` (add strictness cases)

- [ ] **Step 1: Add the strictness tests**

Append to `tests/unit/schema/test_entry.py`:

```python
def test_member_declared_with_raw_tuple_is_rejected():
    import pytest
    from phenotypic.schema import MeasurementInfo

    with pytest.raises(TypeError):
        class BAD(MeasurementInfo):  # noqa: N801
            @classmethod
            def category(cls):
                return "Bad"

            X = ("X", "raw tuple no longer allowed")


def test_member_declared_with_bare_string_is_rejected():
    import pytest
    from phenotypic.schema import MeasurementInfo

    with pytest.raises(TypeError):
        class BAD2(MeasurementInfo):  # noqa: N801
            @classmethod
            def category(cls):
                return "Bad2"

            X = "X"
```

- [ ] **Step 2: Run to verify the strictness tests fail (dual-accept still allows
  tuples)**

Run: `uv run pytest tests/unit/schema/test_entry.py -q -k rejected`
Expected: FAIL (no error raised yet).

- [ ] **Step 3: Replace dual-accept `__new__` with the strict version**

In `src/phenotypic/schema/_measurement_info.py`, replace the `__new__` body from Task 2
with:

```python
    def __new__(cls, entry: "Entry"):
        """Create a member from an :class:`Entry`.

        The enum value is the category-prefixed header (e.g. ``Shape_Area``);
        ``label``/``desc``/``bio_desc``/``image`` are stored as instance
        attributes. Anything other than an ``Entry`` raises ``TypeError`` at
        class-creation time.
        """
        if not isinstance(entry, Entry):
            raise TypeError(
                f"{cls.__name__} members must be declared as Entry(...); "
                f"got {entry!r}. Raw tuples/strings are not accepted."
            )
        full = f"{cls.category()}_{entry.label}"
        obj = str.__new__(cls, full)
        obj._value_ = full
        obj.label = entry.label
        obj.desc = entry.desc
        obj.bio_desc = entry.bio_desc
        obj.image = entry.image
        obj.pair = (entry.label, entry.desc)
        return obj
```

- [ ] **Step 4: Update the class docstring doctests to use `Entry`**

In the `MeasurementInfo` class docstring, change the `SHAPE` example from:

```python
        ...     AREA = ('Area', 'Total number of pixels in the detected object')
        ...     PERIMETER = ('Perimeter', 'Total length of object boundary in pixels')
```

to:

```python
        ...     AREA = Entry('Area', 'Total number of pixels in the detected object')
        ...     PERIMETER = Entry('Perimeter', 'Total length of object boundary in pixels')
```

and change the import line in that doctest from
`>>> from phenotypic.schema import MeasurementInfo` to
`>>> from phenotypic.schema import Entry, MeasurementInfo`.

- [ ] **Step 5: Run strictness tests + doctest**

Run: `uv run pytest tests/unit/schema/test_entry.py -q`
Run: `uv run pytest --doctest-modules src/phenotypic/schema/_measurement_info.py -q`
Expected: PASS.

- [ ] **Step 6: Full schema regression (strict mode)**

Run: `uv run pytest tests/unit/schema tests/smoke/test_measurement.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/schema/_measurement_info.py tests/unit/schema/test_entry.py
git commit -m "feat(schema): require Entry for members — raw tuples now error at import"
```

---

## Task 6: Shared conditional-column renderer + `rst_table` refactor + QC override

**Files:**

- Modify: `src/phenotypic/schema/_measurement_info.py` (add `_ASSET_URL_PREFIX`,
  `_RST_ROLE_RE`, `_rst_cell_text`, `_render_info_table`; refactor `rst_table`)
- Modify: `src/phenotypic/schema/_quality_check.py` (route override through helper)
- Test: `tests/unit/schema/test_rst_rendering.py`

(`_quality_check.py`'s members were already converted to `Entry` in Task 4 Step 3;
this task only changes its `append_rst_to_doc` override.)

- [ ] **Step 1: Write the failing renderer tests**

Use real (not functional-API) fixture enums — `category()` must be a classmethod,
which the functional `Enum(name, members)` API can't express:

```python
"""rst_table renders Biology/Image columns only when populated."""

from phenotypic.schema import Entry, MeasurementInfo


class _DescOnly(MeasurementInfo):
    @classmethod
    def category(cls):
        return "DescOnly"

    A = Entry("A", "alpha")
    B = Entry("B", "beta")


class _WithBio(MeasurementInfo):
    @classmethod
    def category(cls):
        return "WithBio"

    A = Entry("A", "alpha", bio_desc="grows")
    B = Entry("B", "beta")


class _WithImage(MeasurementInfo):
    @classmethod
    def category(cls):
        return "WithImage"

    A = Entry("A", "alpha", image="shape/area.png")


def test_desc_only_has_no_biology_or_image_columns():
    table = _DescOnly.rst_table()
    assert "Description" in table
    assert "Biology" not in table
    assert "Image" not in table
    assert "``A``" in table


def test_biology_column_appears_when_any_member_sets_bio_desc():
    table = _WithBio.rst_table()
    assert "Biology" in table
    assert "grows" in table
    assert "Image" not in table


def test_image_column_emits_directive_with_root_absolute_path():
    table = _WithImage.rst_table()
    assert "Image" in table
    assert ".. image:: /_static/measurements/shape/area.png" in table


def test_use_headers_renders_prefixed_value():
    assert "``DescOnly_A``" in _DescOnly.rst_table(use_headers=True)
    assert "``A``" in _DescOnly.rst_table(use_headers=False)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/schema/test_rst_rendering.py -q`
Expected: FAIL — `rst_table()` has no `use_headers` kwarg / no Biology/Image columns.

- [ ] **Step 3: Add module constants + escaping + the shared renderer**

In `src/phenotypic/schema/_measurement_info.py`, add `import re` to the imports, and
after the `Entry` dataclass add:

```python
#: Source-root-absolute URL prefix for measurement asset images. Root-absolute so
#: the same table resolves whether embedded in autodoc pages or measurements_ref
#: pages. The docs build copies _assets/measurements/** into <srcdir>/_static/.
_ASSET_URL_PREFIX: Final = "/_static/measurements"

_RST_ROLE_RE: Final = re.compile(r":[a-zA-Z0-9_.:]+:`([^`]+)`")


def _rst_cell_text(text: str) -> str:
    """Escape text that would otherwise be parsed as RST markup in a cell."""
    normalized = _RST_ROLE_RE.sub(lambda m: f"``{m.group(1)}``", text)
    return normalized.replace("|", r"\|")


def _render_info_table(
    rows: list[tuple[str, str, str, str | None]],
    *,
    title: str,
    name_header: str = "Name",
) -> str:
    """Render a list-table; Biology/Image columns appear only when populated.

    Args:
        rows: ``(name_cell, desc, bio_desc, image_relpath_or_None)`` per member.
        title: Bold table caption (rendered ``Category: **{title}**``).
        name_header: Header for the first (name) column.
    """
    has_bio = any(bio for (_n, _d, bio, _i) in rows)
    has_img = any(img for (_n, _d, _b, img) in rows)

    lines = [
        f".. list-table:: Category: **{title}**",
        "   :header-rows: 1",
        "",
        f"   * - {name_header}",
        "     - Description",
    ]
    if has_bio:
        lines.append("     - Biology")
    if has_img:
        lines.append("     - Image")

    for name, desc, bio, img in rows:
        lines.append(f"   * - ``{name}``")
        lines.append(f"     - {_rst_cell_text(desc)}")
        if has_bio:
            lines.append(f"     - {_rst_cell_text(bio)}")
        if has_img:
            if img:
                lines.append(f"     - .. image:: {_ASSET_URL_PREFIX}/{img}")
                lines.append("          :width: 110px")
            else:
                lines.append("     -")
    return dedent("\n".join(lines))
```

- [ ] **Step 4: Refactor `rst_table` to use the shared renderer**

Replace the existing `rst_table` classmethod body with:

```python
    @classmethod
    def rst_table(
        cls,
        *,
        title: str | None = None,
        header: tuple[str, str] = ("Name", "Description"),
        use_headers: bool = False,
    ) -> str:
        """Render an RST list-table of this enum's members.

        Adds a Biology column when any member sets ``bio_desc`` and an Image
        column when any sets ``image`` (each suppressed otherwise).

        Args:
            title: Table caption; defaults to the category name.
            header: ``(name_column_header, _description_header)``; only the name
                header is used (the description header is fixed to "Description").
            use_headers: Name cell shows the prefixed value (``Shape_Area``)
                instead of the bare label (``Area``).
        """
        title = title or cls.category()
        name_header = header[0]
        rows = [
            (
                m.value if use_headers else m.label,
                m.desc,
                m.bio_desc,
                m.image,
            )
            for m in cls
        ]
        return _render_info_table(rows, title=title, name_header=name_header)
```

(`append_rst_to_doc` is unchanged — it calls `rst_table()` with defaults.)

- [ ] **Step 5: Route the `QUALITY_CHECK` override through the shared renderer**

In `src/phenotypic/schema/_quality_check.py`, replace the import and the
`append_rst_to_doc` body. After Task 4 the import line reads
`from ._measurement_info import Entry, MeasurementInfo` and `from textwrap import
dedent` is still present. Change them to:

```python
from ._measurement_info import Entry, MeasurementInfo, _render_info_table
```

(drop the now-unused `from textwrap import dedent`), and replace the method body
(everything after the docstring) with:

```python
        slug = check_name if check_name is not None else "<name>"
        rows = [
            (f"QC_{slug}_{m.label}", m.desc, m.bio_desc, m.image) for m in cls
        ]
        table = _render_info_table(rows, title=f"QC_{slug}", name_header="Name")
        base = doc if isinstance(doc, str) else (doc.__doc__ or "")
        return base + "\n\n" + table
```

- [ ] **Step 6: Run renderer tests + QC doc + regression**

Run: `uv run pytest tests/unit/schema/test_rst_rendering.py tests/unit/schema -q`
Run: `uv run pytest tests/unit/docs/test_measurements_ref_extension.py -q`
Expected: PASS (QC table is text-only via suppression; generator still imports fine).

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/schema/_measurement_info.py src/phenotypic/schema/_quality_check.py tests/unit/schema/test_rst_rendering.py
git commit -m "feat(schema): conditional-column rst_table shared by docstrings + QC override"
```

---

## Task 7: Generator uses `rst_table(use_headers=True)`; delete the duplicate

**Files:**

- Modify: `docs/source/_extensions/measurements_ref.py`
- Test: `tests/unit/docs/test_measurements_ref_extension.py`

- [ ] **Step 1: Replace `_full_column_table` + local escaping with the shared API**

In `docs/source/_extensions/measurements_ref.py`:

1. Delete the `_full_column_table` function and the `_rst_cell_text` function and the
   `_RST_ROLE_RE` constant.
2. Add to the imports near the top:

```python
from phenotypic.schema._measurement_info import _rst_cell_text
```

3. In `_enum_page`, change the table line from:

```python
    out.append(_full_column_table(info_cls))
```

to:

```python
    out.append(info_cls.rst_table(header=("Column label", "Description"), use_headers=True))
```

(`_rst_cell_text` is still used by `_enum_page`/`_append_metadata_sections` for prose —
now imported from schema instead of locally defined.)

- [ ] **Step 2: Run the docs extension tests**

Run: `uv run pytest tests/unit/docs/test_measurements_ref_extension.py -q`
Expected: PASS (update any test asserting on the old `_full_column_table` to call
`rst_table(use_headers=True)` instead).

- [ ] **Step 3: Commit**

```bash
git add docs/source/_extensions/measurements_ref.py tests/unit/docs/test_measurements_ref_extension.py
git commit -m "refactor(docs): generator uses shared rst_table; drop _full_column_table"
```

---

## Task 8: 5-group toctree IA + asset-copy step

**Files:**

- Modify: `docs/source/_extensions/measurements_ref.py`
- Test: `tests/unit/docs/test_measurements_ref_extension.py`

- [ ] **Step 1: Write the group-coverage test**

Add to `tests/unit/docs/test_measurements_ref_extension.py`:

```python
def test_every_public_enum_lands_in_exactly_one_group(monkeypatch):
    ext = _load_extension(monkeypatch)
    public = set(ext._public_measurement_info_classes())  # {name: class} -> keys
    grouped = [name for names in ext._GROUPS.values() for name in names]
    assert len(grouped) == len(set(grouped)), "an enum appears in >1 group"
    assert set(grouped) == public, f"ungrouped or unknown: {public ^ set(grouped)}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/docs/test_measurements_ref_extension.py -q -k group`
Expected: FAIL — `_GROUPS` does not exist.

- [ ] **Step 3: Add the explicit group mapping**

In `docs/source/_extensions/measurements_ref.py`, add a single source-of-truth
mapping (ordered) near the top, after `_METADATA_INFO_NAMES`:

```python
#: Toctree groups (caption -> ordered enum names). Single source of truth; a test
#: asserts every public schema enum lands in exactly one group.
_GROUPS: dict[str, tuple[str, ...]] = {
    "Measurements"     : (
        "SIZE", "SHAPE", "BBOX", "INTENSITY", "TEXTURE", "ColorLab", "ColorHSV",
        "Colorxy", "ColorXYZ", "ColorComposition", "OBJECT", "GRID",
        "NEIGHBOR_DIST", "GRID_LINREG_STATS", "GRID_SPREAD", "SYMMETRIC_ZONES",
        "RADIAL_EXPANSION",
    ),
    "Models & Analysis": (
        "LOG_GROWTH_MODEL", "LINEAR_SOFTPLUS_MODEL", "DOUBLE_SOFTPLUS_MODEL",
        "EDGE_CORRECTION", "MODEL_METRICS",
    ),
    "Quality Control"  : (
        "QUALITY_CHECK", "QUALITY_COUNT", "QUALITY_ICC", "QUALITY_MAD",
        "QUALITY_SE", "QUALITY_TUKEY", "QUALITY_ZMAX",
    ),
    "Curation & Errors": ("CURATION", "ErrorCategory"),
    "Metadata"         : (
        "METADATA", "ACQUISITION_METADATA", "CONDITION_METADATA",
        "EXPERIMENT_METADATA", "GENETIC_METADATA", "INCUBATION_METADATA",
        "PLATE_METADATA", "SAMPLE_METADATA",
    ),
}
```

- [ ] **Step 4: Build per-group captioned toctrees + index cards**

Replace `_build_pages`'s two-way `metadata_names`/`measurement_names` split with
iteration over `_GROUPS`, and add the three builder functions below. The
per-enum `_enum_page` output is unchanged; the per-group index is a captioned
toctree of that group's enum pages (plus the metadata overview table for the
Metadata group). **Delete only** these now-superseded symbols:
`_build_measurements_index`,
`_build_metadata_index`, `_append_operator_sections`,
`_append_object_identifier_section`, `_append_metadata_sections`,
`_toctree_entries`, `_experimental_tag_list`, and the
`_ROOT_INTRO`/`_MEASUREMENTS_INTRO`/`_METADATA_INTRO`/`_REGISTRY` constants (the
per-enum pages + group toctrees replace the inline operator/metadata overviews;
`OBJECT` now renders as an ordinary per-enum page in the Measurements group).
**Leave intact** everything `_enum_page` still depends on — `_enum_page`,
`_lead_paragraphs`, `_strip_appended_table`, `_heading`, `_import`,
`_rst_cell_text` (imported from schema in Task 7) — plus `_metadata_overview_rows`,
`_METADATA_OVERVIEWS`, `_doc_stem`, `_write`, `_public_measurement_info_classes`.

```python
def _group_slug(caption: str) -> str:
    return caption.lower().replace(" & ", "-and-").replace(" ", "-")


def _build_root_index(groups, public_infos) -> str:
    toc, cards = [], []
    for caption, names in groups.items():
        if not any(n in public_infos for n in names):
            continue
        slug = _group_slug(caption)
        toc.append(f"   {slug}/index")
        cards += [
            f"   .. grid-item-card:: {caption}",
            "",
            "      +++",
            "",
            f"      .. button-ref:: {slug}/index",
            "         :ref-type: doc",
            "         :click-parent:",
            "         :color: secondary",
            "         :expand:",
            "",
            f"         Browse {caption}",
            "",
        ]
    out = [
        "Measurements",
        "============",
        "",
        "PhenoTypic uses ``MeasurementInfo`` enums to define stable column names",
        "for measurement outputs and metadata joins. Browse by group:",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "   :hidden:",
        "",
        *toc,
        "",
        ".. grid:: 1 2 2 2",
        "   :gutter: 3",
        "",
        *cards,
    ]
    return "\n".join(out)


def _metadata_overview_block(names) -> str:
    present = [n for n in names if n in _METADATA_OVERVIEWS]
    if not present:
        return ""
    return "\n".join([
        "Metadata Tag Overview",
        "---------------------",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Tag class",
        "     - Includes",
        "     - Use for",
        _metadata_overview_rows(present),
        "",
    ])


def _build_group_index(caption, names, public_infos) -> str:
    out = [
        caption,
        "=" * len(caption),
        "",
        f"Schema enums in the **{caption}** group. Each page documents an enum's",
        "DataFrame column labels and descriptions.",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        f"   :caption: {caption}",
        "",
    ]
    for name in names:
        out.append(f"   {public_infos[name].category()} <{_doc_stem(name)}>")
    out.append("")
    block = _metadata_overview_block(names) if caption == "Metadata" else ""
    if block:
        out += ["", block]
    return "\n".join(out)
```

and the rewritten `_build_pages`:

```python
def _build_pages(srcdir: str) -> None:
    output_dir = Path(srcdir) / "measurements_ref"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    public_infos = _public_measurement_info_classes()
    _write(output_dir / "index.rst", _build_root_index(_GROUPS, public_infos))
    for caption, names in _GROUPS.items():
        present = [n for n in names if n in public_infos]
        if not present:
            continue
        slug = _group_slug(caption)
        _write(
            output_dir / slug / "index.rst",
            _build_group_index(caption, present, public_infos),
        )
        for name in present:
            _write(
                output_dir / slug / f"{_doc_stem(name)}.rst",
                _enum_page(public_infos[name]),
            )
```

- [ ] **Step 5: Add the asset-copy step at `builder-inited`**

Add a copy helper and call it from `_generate`:

```python
def _copy_measurement_assets(srcdir: str) -> None:
    """Copy packaged measurement images into the docs static tree so that
    ``/_static/measurements/...`` references resolve in the built HTML."""
    import shutil
    import phenotypic

    src = Path(phenotypic.__file__).resolve().parent / "_assets" / "measurements"
    if not src.is_dir():
        return
    dest = Path(srcdir) / "_static" / "measurements"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
```

and in `_generate`:

```python
def _generate(app):
    _copy_measurement_assets(app.srcdir)
    _build_pages(app.srcdir)
    print(f"Generated {os.path.join(app.srcdir, 'measurements_ref')}")
```

- [ ] **Step 6: Run docs extension tests**

Run: `uv run pytest tests/unit/docs/test_measurements_ref_extension.py -q`
Expected: PASS. Update any test that assumed the old two-page IA to the new
group structure.

- [ ] **Step 7: Commit**

```bash
git add docs/source/_extensions/measurements_ref.py tests/unit/docs/test_measurements_ref_extension.py
git commit -m "feat(docs): 5-group measurements_ref IA + measurement asset copy step"
```

---

## Task 9: Assets directory + packaging + image-resolution test

**Files:**

- Create: `src/phenotypic/_assets/measurements/` (with a `.gitkeep` until Task 10 adds
  the PNG)
- Modify: `pyproject.toml`
- Test: `tests/unit/schema/test_measurement_assets.py`

- [ ] **Step 1: Write the image-resolution test**

```python
"""Every declared MeasurementInfo image resolves to a packaged file."""

from pathlib import Path

import phenotypic
import phenotypic.tools_.constants_  # noqa: F401
from phenotypic.schema import MeasurementInfo

_ASSETS = Path(phenotypic.__file__).resolve().parent / "_assets" / "measurements"


def _all_members():
    seen = set()
    stack = list(MeasurementInfo.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        if cls.__module__.startswith("phenotypic"):
            yield from list(cls)


def test_declared_images_exist():
    missing = [
        m.image for m in _all_members()
        if m.image and not (_ASSETS / m.image).is_file()
    ]
    assert not missing, f"declared images with no file: {missing}"
```

- [ ] **Step 2: Create the assets dir and run the test (no images yet → passes
  vacuously)**

```bash
mkdir -p src/phenotypic/_assets/measurements
touch src/phenotypic/_assets/measurements/.gitkeep
uv run pytest tests/unit/schema/test_measurement_assets.py -q
```

Expected: PASS (no member declares an image yet).

- [ ] **Step 3: Add package-data globs**

In `pyproject.toml`, under `[tool.setuptools.package-data]` `"phenotypic" = [ ... ]`,
add after `"_assets/vendor/*",`:

```toml
    "_assets/measurements/*",
    "_assets/measurements/*/*",
```

- [ ] **Step 4: Commit**

```bash
git add src/phenotypic/_assets/measurements/.gitkeep pyproject.toml tests/unit/schema/test_measurement_assets.py
git commit -m "feat(assets): measurements asset dir, package-data, resolution test"
```

---

## Task 10: `SHAPE.AREA` worked example (human-authored content)

**Guardrail:** the agent generates the *image* (a visualization, not a claim) and
wires the field, but the **`bio_desc` text is human-authored** — do not invent it.

**Files:**

- Create: `scripts/make_measurement_example_images.py`
- Create: `src/phenotypic/_assets/measurements/shape/area.png`
- Modify: `src/phenotypic/schema/_shape.py` (`AREA` member)

- [ ] **Step 1: Add the example-image generator script**

```python
"""Generate illustrative figures for MeasurementInfo `image` examples.

These are visualizations to aid understanding — not biological claims. Run:
    uv run python scripts/make_measurement_example_images.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector

_OUT = Path(__file__).resolve().parents[1] / "src" / "phenotypic" / "_assets" / "measurements"


def _shape_area() -> None:
    image = load_synth_yeast_plate()
    detected = OtsuDetector(ignore_zeros=True).apply(image)
    objmap = detected.objmap[:]
    labels = [v for v in set(objmap.ravel().tolist()) if v != 0]
    target = max(labels, key=lambda v: (objmap == v).sum())
    mask = objmap == target

    rr, cc = mask.nonzero()
    pad = 15
    r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad, mask.shape[0])
    c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad, mask.shape[1])

    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
    ax.imshow(detected.rgb[:][r0:r1, c0:c1])
    ax.imshow(
        mask[r0:r1, c0:c1], cmap="autumn", alpha=0.45 * mask[r0:r1, c0:c1],
    )
    ax.set_title(f"Shape_Area = {int(mask.sum())} px", fontsize=9)
    ax.axis("off")
    dest = _OUT / "shape" / "area.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    print("wrote", dest)


if __name__ == "__main__":
    _shape_area()
```

- [ ] **Step 2: Generate the image**

Run: `uv run python scripts/make_measurement_example_images.py`
Expected: prints `wrote .../_assets/measurements/shape/area.png`; the PNG exists.
(Import paths `phenotypic.data.load_synth_yeast_plate` and
`phenotypic.detect.OtsuDetector` are verified. If `.objmap[:]`/`.rgb[:]` accessor
names differ for the detected image, adjust per `_core/CLAUDE.md`.)

- [ ] **Step 3: Obtain the human-authored `bio_desc`**

**STOP — ask the user for the `SHAPE.AREA` bio_desc text** (one or two sentences
on the biological relevance of colony area). Do not generate it (guardrail §10.1).
Use the user's exact text in the next step. If the user is unavailable, leave the
member `desc`-only and skip Steps 4–6 (note the example is pending content).

- [ ] **Step 4: Wire the example into `SHAPE.AREA`**

In `src/phenotypic/schema/_shape.py`, change the `AREA` member to (substituting
the user's `bio_desc` text):

```python
    AREA = Entry(
        "Area",
        "Total number of pixels occupied by the microbial colony. Represents colony biomass and growth extent on agar plates. Larger areas typically indicate more robust growth or longer incubation times.",
        bio_desc="<USER-PROVIDED TEXT>",
        image="shape/area.png",
    )
```

- [ ] **Step 5: Run asset, golden, format, and rendering tests**

Run:
`uv run pytest tests/unit/schema/test_measurement_assets.py tests/unit/schema/test_measurement_info_golden.py tests/unit/schema/test_measurement_info_format.py tests/unit/schema/test_rst_rendering.py -q`
Expected: PASS. (The golden test still passes — it snapshots only
`name/value/label/desc`, which are unchanged; `bio_desc`/`image` are not part of
the golden contract.)

- [ ] **Step 6: Commit**

```bash
git add scripts/make_measurement_example_images.py src/phenotypic/_assets/measurements/shape/area.png src/phenotypic/schema/_shape.py
git rm --cached src/phenotypic/_assets/measurements/.gitkeep 2>/dev/null || true
git commit -m "feat(schema): SHAPE.AREA worked example (bio_desc + figure)"
```

---

## Task 11: Docs, doctests, and full regression

**Files:**

- Modify: `src/phenotypic/schema/CLAUDE.md`
- Modify: any doctest/test elsewhere that builds a `MeasurementInfo` subclass with a
  tuple

- [ ] **Step 1: Find any remaining tuple-style member declarations / doctests**

Run:

```bash
grep -rn "MeasurementInfo)" --include='*.py' src docs tests | grep -v "_measurement_info.py"
grep -rEn "= \(\"[A-Za-z]" --include='*.py' src/phenotypic/schema src/phenotypic/tools_/constants_.py
```

Convert any remaining `class X(MeasurementInfo)` member tuples or doctests to
`Entry(...)`. (Custom-base subclasses like `ConstantLabels` are already covered
by Task 4 Step 6.)

- [ ] **Step 2: Update `schema/CLAUDE.md`**

In `src/phenotypic/schema/CLAUDE.md`, update the `MeasurementInfo` bullet to
describe the `Entry` value type and the new fields. Replace:

```markdown
- `MeasurementInfo` (`_measurement_info.py`) — `str, Enum` base. Subclasses
  declare `(label, description)` members and a `category()` classmethod; the
  enum value is the category-prefixed header (e.g. `Shape_Area`). Helpers:
  `get_labels()`, `get_headers()`, `rst_table()`, `append_rst_to_doc()`.
```

with:

```markdown
- `MeasurementInfo` (`_measurement_info.py`) — `str, Enum` base. Subclasses
  declare members as `Entry(label, desc, *, bio_desc="", image=None)` plus a
  `category()` classmethod; the enum value is the category-prefixed header
  (e.g. `Shape_Area`). `Entry` is the only legal member value — raw tuples raise
  `TypeError` at import. `desc` is the technical description; **`bio_desc` is
  human-authored only** (biological claim — never machine-generated); `image` is
  a path under `_assets/measurements/`. Per-member attrs: `.label`, `.desc`,
  `.bio_desc`, `.image`, `.pair`, `.CATEGORY`. Helpers: `get_labels()`,
  `get_headers()`, `rst_table()` (conditional Biology/Image columns),
  `append_rst_to_doc()`.
```

- [ ] **Step 3: Type-check and lint**

Run: `uv run mypy src/phenotypic/schema src/phenotypic/tools_/constants_.py`
Run:
`uv run ruff check --fix src/phenotypic/schema src/phenotypic/tools_/constants_.py docs/source/_extensions/measurements_ref.py`
Expected: clean (fix any issues surfaced).

- [ ] **Step 4: Docs build smoke**

Run:
`uv run --group docs sphinx-build -b html -W --keep-going docs/source /tmp/phenodocs 2>&1 | tail -20`
Expected: build succeeds. Verify the SHAPE page embeds the image:

```bash
grep -r "measurements/shape/area.png" /tmp/phenodocs/measurements_ref | head
```

Expected: at least one hit (SHAPE per-enum page, and the MeasureShape autodoc page).

- [ ] **Step 5: Full targeted regression**

Run:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/schema tests/unit/docs tests/unit/util/test_measurement_outputs.py tests/smoke/test_measurement.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/schema/CLAUDE.md
git commit -m "docs(schema): document Entry value type and bio_desc/image fields"
```

---

## Self-Review notes (for the executor)

- **Migration safety:** Tasks 1–3 build `Entry` + dual-accept + golden snapshot
  *before* any member changes; Task 4 rewrites under the golden gate; Task 5
  flips to strict. Never skip the per-group gate in Task 4.
- **Guardrail:** only `SHAPE.AREA` gets a `bio_desc`, and that text comes from the
  user (Task 10 Step 3). No other member's `bio_desc`/`image` is populated.
- **Single renderer:** if a test references `_full_column_table` (deleted in
  Task 7) or the old two-page IA (changed in Task 8), update it to the new API.
- **Path resolution:** images use `/_static/measurements/...`; the copy step
  (Task 8 Step 5) must land before the docs build (Task 11 Step 4).
