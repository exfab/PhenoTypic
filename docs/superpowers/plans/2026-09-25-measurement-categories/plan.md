# Measurement Categories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repo-defined `CATEGORIES` axis to measurement `Entry` members, publish one
spreadsheet per category under `deliverables/measurements_by_category/` on the finalize/recompile
path, and document categories on a generated docs page. Before any of that, rename
`MeasurementInfo.category()` → `metric_family()`.

**Architecture:** Phase 0 is a hard rename of the per-enum header prefix. `schema/_categories.py`
declares a `CATEGORIES` `str` enum (a label and a desc per member, **not** a `MeasurementInfo`
subclass), and `Entry(categories=CATEGORIES.X)` tags members. `util/_measurement_outputs.py` gains
one shared grouping helper that serves both the existing by-measurer split and the new by-category
split. The CLI writes the category split beside the feature split inside
`finalize_post_master_outputs`, the single finalization path shared by `full`/`measure`/`recompile`.
The Sphinx `measurements_ref` extension generates a Categories subpage under the Measurements tab.

**Tech Stack:** Python 3.12, `enum`/`dataclasses`, pandas + polars, Sphinx + sphinx-design
(`:bdg-ref-<color>-line:`), pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-09-25-measurement-categories/design.md`. Read it before any
task: this plan argues from it.

## Global Constraints

- `uv run` for everything; never bare `python`/`pip`/`pytest`.
- `uv run ruff check --fix <explicit paths you changed>`: **never** a bare `ruff check --fix`.
- Operations and `Entry` are keyword-only; never author `bio_desc` or `image` on any `Entry`.
- `src/phenotypic/schema/_categories.py` imports **only stdlib** at module level (schema
  import-light rule); `phenotypic.schema` is imported only inside `CATEGORIES.members()`.
- `CATEGORIES` must **not** subclass `MeasurementInfo`.
- Hard rename: `category()` → `metric_family()`, `.CATEGORY` → `.METRIC_FAMILY`, **no alias**.
- Initial `STARTING_METRICS` membership is exactly the 18 headers pinned in Task 3.
- Deliverable paths are built only through `phenotypic.sdk_` helpers; never hand-join
  `"measurements_by_category"`.
- `split_master_by_category` is called from exactly one place: `finalize_post_master_outputs`.
- Test cadence: focused tests per task, affected surface per phase, the full sharded suite
  **once** at the end (Task 10). Never `-n auto`; never `-x` on a quoted run.
- `sphinx-build` runs only as a Slurm job (`build_docs_categories.sbatch` in this folder).
- Commit messages end with the two attribution lines:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01D1sMyGvSvvLnJSAM5skaEa`.

## Review Focus

These are the five inputs most likely to bite a user that the spec implies but does not spell
out as tests. Each has a pinning test in the task named.

1. **Bare member passed to `categories=`.** `CATEGORIES` is a `str` enum, so naive iteration of
   `CATEGORIES.STARTING_METRICS` yields its characters. Expected: exactly one category.
   Pinned in Task 2 (`test_bare_member_is_one_category_not_its_characters`).
2. **A categorized column that no measurer owns.** That column would be both a context column
   and a group column, and the split would select it twice (polars raises on duplicate names).
   Expected: every categorized header is producer-owned. Pinned in Task 4
   (`test_every_categorized_header_is_owned_by_a_producer`).
3. **A run with none of a category's columns** (e.g. only `MeasureShape`). Expected: no
   `StartingMetrics.csv` and no empty `measurements_by_category/` directory. Pinned in Task 5
   (`test_no_categorized_columns_writes_nothing`).
4. **Both integrated-intensity columns present** (`MeasureSize` + `MeasureIntensity`).
   Expected: both columns, once each, in master order. Pinned in Task 4
   (`test_both_integrated_intensities_appear_once_each`).
5. **Recompiling a run finalized before this feature.** Expected: the recompile finalizer
   creates `measurements_by_category/StartingMetrics.{csv,parquet}`. Pinned in Task 5
   (extension of `test_finalizer_writes_master_outputs_and_rebuilds_dashboard`).

## Dependency graph

```
T1 (rename) ─► T2 (CATEGORIES + Entry) ─► T3 (tag 18) ─► T4 (util split) ─► T5 (CLI) ─► T6 (README)
                         └──────────────► T7 (rst badges) ─► T8 (Categories page) ─► T9 (prose + docs build)
all ─► T10 (full regression)
```

T4→T6 and T7→T9 are independent chains after T3 and can run in parallel.

---

## Phase 0: rename

### Task 1: Hard-rename `category()` → `metric_family()`

**Files:**
- Modify (mechanical): every file listed by
  `git grep -lE 'def category\(|\.category\(\)|\.CATEGORY\b|def CATEGORY\(' -- src tests`
  (44 in `src/`, 17 in `tests/` on `a8b6e17c`)
- Modify: `src/phenotypic/schema/_measurement_info.py` (guard, docstrings, table caption)
- Modify: `src/phenotypic/measure/_measure_symzones.py:253-257` (remove dead filter)
- Modify (wording): `CLAUDE.md:513`, `src/phenotypic/schema/CLAUDE.md`,
  `.claude/skills/adding-an-operation/SKILL.md:37`,
  `docs/source/explanation/metadata_namespace.md:21`, and the docstrings listed in Step 7
- Create: `tests/unit/schema/test_metric_family.py`
- Create: `tests/unit/cli/test_readme_measurement_tables.py`

**Interfaces:**
- Produces: `MeasurementInfo.metric_family() -> str` (classmethod; base raises
  `NotImplementedError`), `member.METRIC_FAMILY -> str` (property). `category`/`CATEGORY` no
  longer exist. A subclass defining either raises `TypeError` mentioning `metric_family`.
  `rst_table()` caption reads `Metric family: **<family>**`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/schema/test_metric_family.py`:

```python
"""``MeasurementInfo.category()`` was hard-renamed to ``metric_family()``."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from phenotypic.schema import SIZE, Entry, MeasurementInfo

_SRC = Path(__file__).resolve().parents[3] / "src" / "phenotypic"
_LEGACY_API = re.compile(r"def category\(|\.category\(\)|\.CATEGORY\b|def CATEGORY\(")


def test_metric_family_is_the_header_prefix() -> None:
    assert SIZE.metric_family() == "Size"
    assert SIZE.AREA.METRIC_FAMILY == "Size"
    assert SIZE.AREA.value == "Size_Area"


def test_base_class_has_no_legacy_category_api() -> None:
    assert not hasattr(MeasurementInfo, "category")
    assert not hasattr(MeasurementInfo, "CATEGORY")


def test_legacy_category_override_is_refused_at_class_creation() -> None:
    with pytest.raises(TypeError, match="metric_family"):

        class LEGACY(MeasurementInfo):
            @classmethod
            def category(cls) -> str:
                return "Legacy"

            VALUE = Entry("Value", "A value.")


def test_legacy_category_on_memberless_subclass_is_refused() -> None:
    with pytest.raises(TypeError, match="metric_family"):

        class LEGACY_BASE(MeasurementInfo):
            @classmethod
            def category(cls) -> str:
                return "Legacy"


def test_no_legacy_category_api_left_in_src() -> None:
    offenders = [
        f"{path.relative_to(_SRC)}:{lineno}"
        for path in sorted(_SRC.rglob("*.py"))
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _LEGACY_API.search(line)
    ]
    assert offenders == []


def test_rst_table_caption_names_the_metric_family() -> None:
    assert ".. list-table:: Metric family: **Size**" in SIZE.rst_table()
```

`tests/unit/cli/test_readme_measurement_tables.py`:

```python
"""Every public schema renders a README table.

``READMEGenerator._generate_measurement_table`` wraps its body in a broad
``except Exception`` that returns ``""``. A missed rename inside it would
silently drop every measurement table from every run's README, so this guard
asserts each public, member-ful schema renders a non-empty table headed by its
metric family.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import phenotypic.schema as schema
from phenotypic import ImagePipeline
from phenotypic._cli._cli_readme_generator import READMEGenerator
from phenotypic.schema import MeasurementInfo


def _public_info_classes() -> list[type[MeasurementInfo]]:
    seen: list[type[MeasurementInfo]] = []
    for name in schema.__all__:
        value = getattr(schema, name, None)
        if (
            isinstance(value, type)
            and issubclass(value, MeasurementInfo)
            and value is not MeasurementInfo
            and list(value)
            and value not in seen
        ):
            seen.append(value)
    return seen


@pytest.mark.parametrize("info_cls", _public_info_classes(), ids=lambda c: c.__name__)
def test_every_public_schema_renders_a_readme_table(info_cls: type[MeasurementInfo]) -> None:
    generator = READMEGenerator(config=SimpleNamespace(), pipeline=ImagePipeline())
    table = generator._generate_measurement_table(info_cls)
    assert table.startswith(f"\n### {info_cls.metric_family()}\n")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/schema/test_metric_family.py tests/unit/cli/test_readme_measurement_tables.py -q -p no:cacheprovider`
Expected: FAIL with `AttributeError: type object 'SIZE' has no attribute 'metric_family'` (and the
parametrized README tests fail the same way).

- [ ] **Step 3: Mechanical rename**

```bash
git grep -lE 'def category\(|\.category\(\)|\.CATEGORY\b|def CATEGORY\(' -- src tests \
  | xargs perl -pi -e 's/\bdef category\(/def metric_family(/g; s/\bdef CATEGORY\(/def METRIC_FAMILY(/g; s/\.category\(\)/.metric_family()/g; s/\.CATEGORY\b/.METRIC_FAMILY/g'
git grep -nE 'def category\(|\.category\(\)|\.CATEGORY\b' -- src tests
```

Expected: the second command prints nothing. `category=` keyword arguments (GUI triage, operation
registry, migration scenarios) are unrelated and are deliberately left alone: the patterns above
never match them.

- [ ] **Step 4: Add the legacy-override guard in `_measurement_info.py`**

Python 3.12 builds enum members **before** `__init_subclass__` runs (verified: member `__new__`
fires first), so the guard must run at the top of member `__new__`. It also runs in
`__init_subclass__`, to cover member-less subclasses. Add this module-level helper above
`class MeasurementInfo`:

```python
#: Names removed by the ``category()`` → ``metric_family()`` hard rename.
_LEGACY_FAMILY_NAMES: Final = ("category", "CATEGORY")


def _refuse_legacy_category(cls: type) -> None:
    """Refuse a subclass still overriding the removed ``category`` API."""
    for klass in cls.__mro__:
        for name in _LEGACY_FAMILY_NAMES:
            if name in vars(klass):
                raise TypeError(
                    f"{cls.__name__} defines {name!r}, which was removed: "
                    "the category() classmethod is now metric_family() and "
                    "the CATEGORY property is now METRIC_FAMILY. Rename the override."
                )
```

The message deliberately avoids the literal dotted forms (`.category()`, `.CATEGORY`), because
`test_no_legacy_category_api_left_in_src` scans `src/` for exactly those patterns.

Inside `class MeasurementInfo`, add:

```python
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _refuse_legacy_category(cls)
```

At the very top of `MeasurementInfo.__new__` (before the `isinstance(entry, Entry)` check), add:

```python
        _refuse_legacy_category(cls)
```

- [ ] **Step 5: Rename the table caption**

In `_render_info_table`, change `f".. list-table:: Category: **{title}**",` to
`f".. list-table:: Metric family: **{title}**",`, and update its docstring line
`title: Bold table caption (rendered ``Category: **{title}**``).` to
`title: Bold table caption (rendered ``Metric family: **{title}**``).`

- [ ] **Step 6: Remove the dead symzones filter**

`src/phenotypic/measure/_measure_symzones.py`: the comprehension filter
`if feature != SYMMETRIC_ZONES.METRIC_FAMILY` (formerly `.CATEGORY`) compares a member with the
*property object* (class-level access to a plain `property`), so it never excludes anything.
Verified on `a8b6e17c`: `[f for f in SYMMETRIC_ZONES if f == SYMMETRIC_ZONES.CATEGORY] == []`.
Delete that line; the dict comprehension becomes:

```python
        measurements = {
            str(feature): np.full(image.num_objects, np.nan)
            for feature in SYMMETRIC_ZONES
        }
```

- [ ] **Step 7: Rewrite prefix wording**

Replace "category" with "metric family" (and "category-prefixed" with "family-prefixed") **only**
where it names the `MeasurementInfo` header prefix. These are the hits on `a8b6e17c`:

- `CLAUDE.md:513`: `**Measurement columns are category-prefixed:**` →
  `**Measurement columns are family-prefixed:**`
- `src/phenotypic/schema/_measurement_info.py`: the class docstring (lines ~242, 261, 272, 281,
  297), the `metric_family` docstring (was `category`, ~344-355: "Return the metric family for
  this measurement enumeration … the family name (e.g., 'Size', 'Color', 'Texture')"), the
  `METRIC_FAMILY` property docstring (~418-425), `__new__` (~432), `__str__` (~459),
  `get_labels`/`get_headers` (~521-547), and `rst_table` `title:` arg (~565).
- `src/phenotypic/schema/_error_category.py:25`, `_linear_cap_and_lag_model.py:12`,
  `_linear_lag_model.py:12`, `_log_growth_model.py:12`, `_tiers.py:32`
- `src/phenotypic/analysis/_linear_cap_and_lag_model.py:89`, `_linear_lag_model.py:81`,
  `_log_growth_model.py:61` ("measurement-category prefix" → "metric-family prefix")
- `src/phenotypic/post/_append_string.py:67`, `_prepend_string.py:67`,
  `_expand_metadata.py:31,89,95`, `_merge_metadata.py:25,28,86,100`
  ("schema category prefix" → "schema metric-family prefix")
- `src/phenotypic/refine/_remove_by_feature.py:53`, `src/phenotypic/sdk_/_rembi_manifest.py:88`,
  `src/phenotypic/tune/score/_reference_free_scorer.py:322`
- `src/phenotypic/util/_measurement_outputs.py`: rename `_known_categories` →
  `_known_families` (and its one caller in `metric_token`, plus the loop variable) and its
  docstring; `metric_token` docstring "schema **category** prefix" → "schema **metric family**
  prefix".
- `src/phenotypic/_gui/results_viewer/colony_view/_grid.py:105-135` and
  `_scatter_tab/_grouping.py:36`: docstrings that say ``category()`` (Step 3's perl already
  fixed the calls; fix the prose).
- `src/phenotypic/schema/CLAUDE.md:7,8,30,31,140,145` and
  `.claude/skills/adding-an-operation/SKILL.md:37`
- `docs/source/explanation/metadata_namespace.md:21`: `` `category()` `` → `` `metric_family()` ``

**Leave alone:** GUI "category name" hits in `_operation_registry.py`, `_triage_callbacks.py`,
`_curation_labels.py` and `builder/_layout.py` (error/curation/operation categories), all test
docstrings, and anything under `docs/superpowers/` (historical).

- [ ] **Step 8: Run the focused tests**

Run: `uv run pytest tests/unit/schema tests/unit/cli/test_readme_measurement_tables.py tests/unit/cli/test_readme_model_section.py tests/unit/util/test_metric_token.py tests/unit/docs/test_measurements_ref_extension.py tests/unit/sdk_/test_metadata_helpers.py tests/unit/gui/results_viewer/test_measurement_prefixes.py tests/unit/measure/test_measure_grid_spatial.py tests/unit/core/test_metadata_cluster_order.py -q -p no:cacheprovider`
Expected: all PASS.

Then the symzones measurer, since Step 6 touched it:
Run: `uv run pytest tests/unit/measure -q -p no:cacheprovider -k "symzone or SymZone or symmetric"`
Expected: PASS (unchanged column set).

- [ ] **Step 9: Lint the changed files and commit**

```bash
uv run ruff check --fix $(git diff --name-only -- '*.py')
git add -A src tests CLAUDE.md .claude/skills/adding-an-operation/SKILL.md docs/source/explanation/metadata_namespace.md
git commit -m "refactor(schema)!: rename MeasurementInfo.category() to metric_family()

BREAKING: category()/.CATEGORY are removed with no alias. A subclass that
still defines category raises TypeError at class creation naming
metric_family(). Frees the word 'category' for curated measurement
categories. Also drops a no-op symzones filter that compared members with
the CATEGORY property object.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01D1sMyGvSvvLnJSAM5skaEa"
```

**Phase 0 gate:** run the affected surface once. That is every test file importing
`phenotypic.schema`, `phenotypic.util`, or `_cli_readme_generator`, derived mechanically:
`git grep -lE 'phenotypic\.schema|phenotypic\.util|_cli_readme_generator' -- tests | grep '\.py$'`.
Run it as a Slurm job (see the `run-phenotypic-test` skill); quote the pass/fail counts.

---

## Phase 1: schema

### Task 2: `CATEGORIES` enum and `Entry.categories`

**Files:**
- Create: `src/phenotypic/schema/_categories.py`
- Modify: `src/phenotypic/schema/_measurement_info.py` (`Entry` field + normalizer; member attr)
- Modify: `src/phenotypic/schema/__init__.py` (export `CATEGORIES`, `CategoryEntry`)
- Create: `tests/unit/schema/test_categories.py`

**Interfaces:**
- Consumes: `MeasurementInfo` (Task 1 names).
- Produces:
  - `CategoryEntry(label: str, desc: str)`: frozen dataclass; `label` must match
    `^[A-Z][A-Za-z0-9]*$`; `desc` non-empty.
  - `CATEGORIES(str, Enum)`: `member.value == member.label`; `.label: str`, `.desc: str`,
    `.display_name: str` (CamelCase split), `.anchor: str`
    (`"measurement-category-" + label.lower()`), `.members() -> tuple[MeasurementInfo, ...]`,
    classmethod `CATEGORIES.in_order(cats: Iterable[CATEGORIES]) -> tuple[CATEGORIES, ...]`
    (enum declaration order).
  - `Entry(..., categories=CATEGORIES.X | Iterable[CATEGORIES])`; `entry.categories:
    frozenset[CATEGORIES]`; every `MeasurementInfo` member has `.categories:
    frozenset[CATEGORIES]`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/schema/test_categories.py`:

```python
"""The CATEGORIES vocabulary and Entry.categories normalization."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import phenotypic.schema as schema
from phenotypic.schema import (
    CATEGORIES,
    CategoryEntry,
    Entry,
    MeasurementInfo,
    MetadataInfo,
)

_CATEGORIES_MODULE = (
    Path(__file__).resolve().parents[3] / "src" / "phenotypic" / "schema" / "_categories.py"
)
_STDLIB_ONLY = {"__future__", "re", "dataclasses", "enum", "typing", "collections.abc"}


def test_categories_is_not_a_measurement_info() -> None:
    assert not issubclass(CATEGORIES, MeasurementInfo)


def test_value_is_the_label_with_no_family_prefix() -> None:
    assert CATEGORIES.STARTING_METRICS.value == "StartingMetrics"
    assert CATEGORIES.STARTING_METRICS.label == "StartingMetrics"
    assert str(CATEGORIES.STARTING_METRICS) == "StartingMetrics"
    assert CATEGORIES.STARTING_METRICS.desc


def test_display_name_splits_camel_case() -> None:
    assert CATEGORIES.STARTING_METRICS.display_name == "Starting Metrics"


@pytest.mark.parametrize("category", list(CATEGORIES), ids=lambda c: c.name)
def test_every_display_name_is_readable(category: CATEGORIES) -> None:
    words = category.display_name.split(" ")
    assert "".join(words) == category.label
    assert all(word[:1].isupper() or word[:1].isdigit() for word in words)


def test_anchor_is_stable() -> None:
    assert CATEGORIES.STARTING_METRICS.anchor == "measurement-category-startingmetrics"


def test_members_must_be_category_entries() -> None:
    # __new__ delegates to _validate_entry; an Enum with members cannot be
    # subclassed, so the validator is exercised directly.
    with pytest.raises(TypeError, match="CategoryEntry"):
        CATEGORIES._validate_entry("not-an-entry")


@pytest.mark.parametrize("label", ["startingMetrics", "Starting Metrics", "", "Starting_Metrics"])
def test_category_entry_rejects_non_camel_case_labels(label: str) -> None:
    with pytest.raises(ValueError, match="CamelCase"):
        CategoryEntry(label, "desc")


def test_category_entry_rejects_empty_desc() -> None:
    with pytest.raises(ValueError, match="desc"):
        CategoryEntry("Valid", "   ")


def test_bare_member_is_one_category_not_its_characters() -> None:
    entry = Entry("Value", "A value.", categories=CATEGORIES.STARTING_METRICS)
    assert entry.categories == frozenset({CATEGORIES.STARTING_METRICS})


def test_iterable_of_members_normalizes_to_frozenset() -> None:
    entry = Entry("Value", "A value.", categories=[CATEGORIES.STARTING_METRICS] * 2)
    assert entry.categories == frozenset({CATEGORIES.STARTING_METRICS})


def test_default_is_empty() -> None:
    assert Entry("Value", "A value.").categories == frozenset()


def test_raw_string_equal_to_a_member_value_is_rejected() -> None:
    assert "StartingMetrics" == CATEGORIES.STARTING_METRICS  # str enum equality
    with pytest.raises(TypeError, match="CATEGORIES"):
        Entry("Value", "A value.", categories="StartingMetrics")


def test_non_member_element_is_rejected() -> None:
    with pytest.raises(TypeError, match="CATEGORIES"):
        Entry("Value", "A value.", categories=[CATEGORIES.STARTING_METRICS, "Other"])


def test_non_iterable_is_rejected() -> None:
    with pytest.raises(TypeError, match="CATEGORIES"):
        Entry("Value", "A value.", categories=3)  # type: ignore[arg-type]


def test_member_exposes_its_categories() -> None:
    class TAGGED(MeasurementInfo):
        @classmethod
        def metric_family(cls) -> str:
            return "Tagged"

        VALUE = Entry("Value", "A value.", categories=CATEGORIES.STARTING_METRICS)
        OTHER = Entry("Other", "Another value.")

    assert TAGGED.VALUE.categories == frozenset({CATEGORIES.STARTING_METRICS})
    assert TAGGED.OTHER.categories == frozenset()


def test_members_finds_tagged_public_members(monkeypatch: pytest.MonkeyPatch) -> None:
    class FUTURE_TAGGED(MeasurementInfo):
        @classmethod
        def metric_family(cls) -> str:
            return "FutureTagged"

        VALUE = Entry("Value", "A value.", categories=CATEGORIES.STARTING_METRICS)

    monkeypatch.setattr(schema, "FUTURE_TAGGED", FUTURE_TAGGED, raising=False)
    monkeypatch.setattr(schema, "__all__", [*schema.__all__, "FUTURE_TAGGED"])
    assert FUTURE_TAGGED.VALUE in CATEGORIES.STARTING_METRICS.members()


def test_in_order_follows_declaration_order() -> None:
    assert CATEGORIES.in_order({CATEGORIES.STARTING_METRICS}) == (CATEGORIES.STARTING_METRICS,)


def test_no_metadata_member_carries_a_category() -> None:
    for name in schema.__all__:
        value = getattr(schema, name, None)
        if isinstance(value, type) and issubclass(value, MetadataInfo):
            tagged = [m for m in value if m.categories]
            assert tagged == [], f"{name} metadata members may not be categorized: {tagged}"


def test_categories_module_imports_only_stdlib_at_module_level() -> None:
    tree = ast.parse(_CATEGORIES_MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert {alias.name for alias in node.names} <= _STDLIB_ONLY
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module in _STDLIB_ONLY, ast.dump(node)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/schema/test_categories.py -q -p no:cacheprovider`
Expected: FAIL at collection with `ImportError: cannot import name 'CATEGORIES' from 'phenotypic.schema'`.

- [ ] **Step 3: Create `src/phenotypic/schema/_categories.py`**

```python
"""Curated, repo-defined groupings of measurement columns across metric families.

A category is a many-to-many label: one column may carry several categories and
one category spans several metric families (``SIZE``, ``ColorLab``, …). Unlike
the kind/tier classification in :mod:`._tiers`, a category makes no trust
claim. The CLI writes one spreadsheet per category under
``deliverables/measurements_by_category/``.

Declare membership on the measurement itself::

    AREA = Entry("Area", "...", categories=CATEGORIES.STARTING_METRICS)

This module imports only the standard library at module level, keeping
``phenotypic.schema`` import-light; :meth:`CATEGORIES.members` imports the
schema package lazily.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._measurement_info import MeasurementInfo

_LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
#: Word boundaries in a CamelCase label: lower/digit→Upper, and the last capital
#: of an acronym before a capitalised word ("CIELabColor" → "CIE Lab Color").
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_ANCHOR_PREFIX = "measurement-category-"


@dataclass(frozen=True, slots=True)
class CategoryEntry:
    """Declarative value for a :class:`CATEGORIES` member.

    Args:
        label: CamelCase token used as the output file stem and anchor slug
            (e.g. ``"StartingMetrics"``).
        desc: Technical description of what the category groups and why.
    """

    label: str
    desc: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not _LABEL_RE.match(self.label):
            raise ValueError(
                f"CategoryEntry.label must be a CamelCase token, got {self.label!r}"
            )
        if not isinstance(self.desc, str) or not self.desc.strip():
            raise ValueError("CategoryEntry.desc must be a non-empty string")


class CATEGORIES(str, Enum):
    """Repo-defined measurement categories.

    Each member's value is its label (no metric-family prefix). Members expose
    ``label``, ``desc``, ``display_name`` and ``anchor``.
    """

    label: str
    desc: str

    @staticmethod
    def _validate_entry(entry: object) -> CategoryEntry:
        if not isinstance(entry, CategoryEntry):
            raise TypeError(
                "CATEGORIES members must be declared as CategoryEntry(...); "
                f"got {entry!r}"
            )
        return entry

    def __new__(cls, entry: CategoryEntry) -> "CATEGORIES":
        entry = cls._validate_entry(entry)
        obj = str.__new__(cls, entry.label)
        obj._value_ = entry.label
        obj.label = entry.label
        obj.desc = entry.desc
        return obj

    def __str__(self) -> str:
        return self._value_

    @property
    def display_name(self) -> str:
        """Human-readable name: the label split on CamelCase boundaries."""
        return _CAMEL_BOUNDARY_RE.sub(" ", self.label)

    @property
    def anchor(self) -> str:
        """Stable Sphinx label of this category's section on the Categories page."""
        return f"{_ANCHOR_PREFIX}{self.label.lower()}"

    @classmethod
    def in_order(cls, categories: Iterable["CATEGORIES"]) -> tuple["CATEGORIES", ...]:
        """Return *categories* sorted by declaration order."""
        order = list(cls)
        return tuple(sorted(set(categories), key=order.index))

    def members(self) -> tuple["MeasurementInfo", ...]:
        """Every public schema member carrying this category.

        Ordered by ``phenotypic.schema.__all__`` export order, then member
        order; compatibility aliases are deduplicated to their class. Not
        cached, so schema classes registered later are seen.
        """
        import phenotypic.schema as schema
        from ._measurement_info import MeasurementInfo

        seen: set[type] = set()
        found: list[MeasurementInfo] = []
        for name in schema.__all__:
            info = getattr(schema, name, None)
            if (
                not isinstance(info, type)
                or not issubclass(info, MeasurementInfo)
                or info is MeasurementInfo
                or info in seen
            ):
                continue
            seen.add(info)
            found.extend(member for member in info if self in member.categories)
        return tuple(found)

    STARTING_METRICS = CategoryEntry(
        "StartingMetrics",
        "Core per-colony magnitudes to examine first: the size measurements, "
        "integrated grayscale intensity, and the CIELAB medoid colour.",
    )
```

- [ ] **Step 4: Add `Entry.categories`**

In `src/phenotypic/schema/_measurement_info.py`:

1. Add imports: `from collections.abc import Iterable` and `from ._categories import CATEGORIES`
   (sibling, stdlib-only, so there's no cycle).
2. Above `class Entry`, add:

```python
def _normalize_categories(value: object) -> frozenset[CATEGORIES]:
    """Coerce ``Entry(categories=...)`` to a frozenset of CATEGORIES members.

    A bare member is checked **before** iteration: CATEGORIES is a ``str``
    enum, so iterating a member would yield its characters. Membership is by
    ``isinstance``, never equality, so a raw string equal to a member's value
    (``"StartingMetrics"``) is rejected.
    """
    if isinstance(value, CATEGORIES):
        return frozenset({value})
    if isinstance(value, str):
        raise TypeError(
            f"Entry.categories takes CATEGORIES members, not strings; got {value!r}. "
            "Use CATEGORIES.<NAME>."
        )
    if not isinstance(value, Iterable):
        raise TypeError(
            f"Entry.categories must be a CATEGORIES member or an iterable of them; got {value!r}"
        )
    items = tuple(value)
    bad = [item for item in items if not isinstance(item, CATEGORIES)]
    if bad:
        raise TypeError(f"Entry.categories accepts only CATEGORIES members; got {bad!r}")
    return frozenset(items)
```

3. In `class Entry`, after `rembi_module: ...`, add the field
   `categories: "CATEGORIES | Iterable[CATEGORIES]" = frozenset()`. Document it in the `Args:`
   block: `categories: Curated categories (CATEGORIES members) this measurement belongs to. A
   bare member or an iterable; normalized to a frozenset.`
4. At the end of `Entry.__post_init__`, add
   `object.__setattr__(self, "categories", _normalize_categories(self.categories))`.
5. In `MeasurementInfo.__new__`, after `obj.rembi_module_override = entry.rembi_module`, add
   `obj.categories = entry.categories`. Add `categories: frozenset[CATEGORIES]` next to
   `rembi_module_override: "REMBI_MODULE | None"` in the class-level annotations.

- [ ] **Step 5: Export from `phenotypic.schema`**

In `src/phenotypic/schema/__init__.py`, add `from ._categories import CATEGORIES, CategoryEntry`
next to the `_tiers` import, and add `"CATEGORIES", "CategoryEntry",` to `__all__` right after
`"Entry",`.

- [ ] **Step 6: Run to verify they pass**

Run: `uv run pytest tests/unit/schema/test_categories.py tests/unit/schema/test_entry.py tests/unit/ci/test_startup_imports.py tests/unit/ci/test_deferred_imports.py -q -p no:cacheprovider`
Expected: PASS. `test_members_finds_tagged_public_members` passes here even though no real member
is tagged yet, because it uses its own test-local class.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/schema/_categories.py src/phenotypic/schema/_measurement_info.py src/phenotypic/schema/__init__.py tests/unit/schema/test_categories.py
git add src/phenotypic/schema/_categories.py src/phenotypic/schema/_measurement_info.py src/phenotypic/schema/__init__.py tests/unit/schema/test_categories.py
git commit -m "feat(schema): CATEGORIES vocabulary and Entry.categories

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01D1sMyGvSvvLnJSAM5skaEa"
```

### Task 3: Tag the 18 `STARTING_METRICS` members

**Files:**
- Modify: `src/phenotypic/schema/_size.py` (all 14 members)
- Modify: `src/phenotypic/schema/_color_lab.py:26-28` (`L_STAR_MEDOID`, `A_STAR_MEDOID`,
  `B_STAR_MEDOID`)
- Modify: `src/phenotypic/schema/_intensity.py:21` (`INTEGRATED_INTENSITY`)
- Modify: `tests/unit/schema/test_categories.py` (append)

**Interfaces:**
- Consumes: `CATEGORIES`, `Entry(categories=...)` (Task 2).
- Produces: `CATEGORIES.STARTING_METRICS.members()` returns exactly the 18 members below.

- [ ] **Step 1: Write the failing pin test** (append to `tests/unit/schema/test_categories.py`)

```python
_STARTING_METRICS_HEADERS = {
    "Size_Area",
    "Size_IntegratedIntensity",
    "Size_Perimeter",
    "Size_ConvexArea",
    "Size_BboxArea",
    "Size_MajorAxisLength",
    "Size_MinorAxisLength",
    "Size_MinFeretDiameter",
    "Size_MaxFeretDiameter",
    "Size_InscribedRadius",
    "Size_MedianRadius",
    "Size_MeanRadius",
    "Size_RobustMeanRadius",
    "Size_MaxRadius",
    "ColorLab_L*Medoid",
    "ColorLab_a*Medoid",
    "ColorLab_b*Medoid",
    "Intensity_IntegratedIntensity",
}


def test_starting_metrics_membership_is_pinned() -> None:
    members = CATEGORIES.STARTING_METRICS.members()
    assert {m.value for m in members} == _STARTING_METRICS_HEADERS
    assert len(members) == 18


def test_every_category_has_a_member() -> None:
    for category in CATEGORIES:
        assert category.members(), f"{category.name} has no members"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/schema/test_categories.py -q -p no:cacheprovider -k "starting_metrics_membership or every_category_has"`
Expected: FAIL (`set() != {...}`).

- [ ] **Step 3: Tag the members**

In each of the three files, add `from ._categories import CATEGORIES` beside the existing
`from ._measurement_info import Entry`, and add `categories=CATEGORIES.STARTING_METRICS` as the
**last** keyword argument of each listed `Entry(...)`. For single-line entries, split the call so
the keyword is on its own line, for example:

```python
    INTEGRATED_INTENSITY = Entry(
        "IntegratedIntensity",
        "The sum of the object's pixels",
        categories=CATEGORIES.STARTING_METRICS,
    )
```

`_size.py`: AREA, INTEGRATED_INTENSITY, PERIMETER, CONVEX_AREA, BBOX_AREA, MAJOR_AXIS_LENGTH,
MINOR_AXIS_LENGTH, MIN_FERET_DIAMETER, MAX_FERET_DIAMETER, INSCRIBED_RADIUS, MEDIAN_RADIUS,
MEAN_RADIUS, ROBUST_MEAN_RADIUS, MAX_RADIUS. `_color_lab.py`: L_STAR_MEDOID, A_STAR_MEDOID,
B_STAR_MEDOID. `_intensity.py`: INTEGRATED_INTENSITY. Do not touch any `desc`, `bio_desc` or
`image`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/schema -q -p no:cacheprovider`
Expected: PASS (includes the classification coverage gate, which must stay green).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/schema/_size.py src/phenotypic/schema/_color_lab.py src/phenotypic/schema/_intensity.py tests/unit/schema/test_categories.py
git add src/phenotypic/schema/_size.py src/phenotypic/schema/_color_lab.py src/phenotypic/schema/_intensity.py tests/unit/schema/test_categories.py
git commit -m "feat(schema): tag the 18 StartingMetrics members

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01D1sMyGvSvvLnJSAM5skaEa"
```

---

## Phase 2: split and CLI

### Task 4: Shared grouping helper and `split_measurements_by_category`

**Files:**
- Modify: `src/phenotypic/util/_measurement_outputs.py`
- Modify: `src/phenotypic/util/__init__.py:6,32-33`
- Create: `tests/unit/util/test_split_by_category.py`

**Interfaces:**
- Consumes: `CATEGORIES`, `member.categories` (Tasks 2–3).
- Produces: `phenotypic.util.split_measurements_by_category(df: pd.DataFrame | pl.DataFrame)
  -> dict[str, same-type frame]`, keyed by category **label** (`"StartingMetrics"`). Each frame
  is the context columns followed by that category's present columns, in master order.
  `split_measurements` output is unchanged.

- [ ] **Step 1: Write the failing tests**

`tests/unit/util/test_split_by_category.py`:

```python
"""split_measurements_by_category: one frame per category, feature-split context."""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from phenotypic.schema import CATEGORIES, Entry, MeasurementInfo
from phenotypic.util import split_measurements, split_measurements_by_category
from phenotypic.util import _measurement_outputs as mo

_CONTEXT = ["Metadata_Dataset", "Object_Label"]


def _frame() -> dict[str, list[object]]:
    return {
        "Metadata_Dataset": ["ds1", "ds1"],
        "Object_Label": [1, 2],
        "Size_Area": [10.0, 12.0],
        "Shape_Circularity": [0.9, 0.8],
        "ColorLab_L*Medoid": [50.0, 51.0],
        "Intensity_IntegratedIntensity": [100.0, 110.0],
        "Size_IntegratedIntensity": [100.0, 110.0],
    }


@pytest.mark.parametrize("ctor", [pd.DataFrame, pl.DataFrame], ids=["pandas", "polars"])
def test_starting_metrics_split_holds_context_then_categorized_columns(ctor) -> None:
    df = ctor(_frame())
    splits = split_measurements_by_category(df)
    assert set(splits) == {"StartingMetrics"}
    out = splits["StartingMetrics"]
    assert type(out) is type(df)
    assert list(out.columns) == [
        *_CONTEXT,
        "Size_Area",
        "ColorLab_L*Medoid",
        "Intensity_IntegratedIntensity",
        "Size_IntegratedIntensity",
    ]


def test_uncategorized_measurements_are_not_context() -> None:
    out = split_measurements_by_category(pd.DataFrame(_frame()))["StartingMetrics"]
    assert "Shape_Circularity" not in out.columns


def test_both_integrated_intensities_appear_once_each() -> None:
    out = split_measurements_by_category(pd.DataFrame(_frame()))["StartingMetrics"]
    cols = list(out.columns)
    assert cols.count("Size_IntegratedIntensity") == 1
    assert cols.count("Intensity_IntegratedIntensity") == 1


def test_context_matches_the_feature_split() -> None:
    df = pd.DataFrame(_frame())
    feature_ctx = list(split_measurements(df)["MeasureShape"].columns)[: len(_CONTEXT)]
    category_ctx = list(split_measurements_by_category(df)["StartingMetrics"].columns)[: len(_CONTEXT)]
    assert feature_ctx == category_ctx == _CONTEXT


def test_category_with_no_present_columns_has_no_key() -> None:
    df = pd.DataFrame({"Metadata_Dataset": ["ds1"], "Object_Label": [1], "Shape_Circularity": [0.9]})
    assert split_measurements_by_category(df) == {}


def test_frame_with_no_measurements_splits_to_nothing() -> None:
    assert split_measurements_by_category(pd.DataFrame({"Metadata_Dataset": ["ds1"]})) == {}


def test_every_categorized_header_is_owned_by_a_producer() -> None:
    headers = [m.value for c in CATEGORIES for m in c.members()]
    groups = mo._producer_column_groups(headers)
    owned = {col for cols in groups.values() for col in cols}
    assert set(headers) <= owned, sorted(set(headers) - owned)


def test_column_in_two_categories_appears_in_both(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only one real category exists today, so substitute a two-member stand-in
    # for the module's CATEGORIES and a resolver returning a member tagged with
    # both. Declaration order (FIRST, then SECOND) must drive key order.
    class FakeCategories(str, Enum):
        FIRST = "First"
        SECOND = "Second"

        @property
        def label(self) -> str:
            return self.value

        @classmethod
        def in_order(cls, cats):
            order = list(cls)
            return tuple(sorted(set(cats), key=order.index))

    member = SimpleNamespace(categories=frozenset({FakeCategories.SECOND, FakeCategories.FIRST}))
    monkeypatch.setattr(mo, "CATEGORIES", FakeCategories)
    monkeypatch.setattr(mo, "_member_for_column", lambda column: member if column == "Size_Area" else None)
    groups = mo._category_column_groups(["Metadata_Dataset", "Size_Area"])
    assert list(groups) == ["First", "Second"]
    assert groups == {"First": ["Size_Area"], "Second": ["Size_Area"]}


def test_dynamic_header_resolves_through_member_for_header(monkeypatch: pytest.MonkeyPatch) -> None:
    class DYNAMIC(MeasurementInfo):
        @classmethod
        def metric_family(cls) -> str:
            return "Dynamic"

        @classmethod
        def member_for_header(cls, column: str):
            return cls.VALUE if column.startswith("Dynamic_Value-scale") else None

        VALUE = Entry("Value", "A value.", categories=CATEGORIES.STARTING_METRICS)

    monkeypatch.setattr(mo, "_public_info_classes", lambda: (DYNAMIC,))
    assert mo._category_column_groups(["Dynamic_Value-scale05"]) == {
        "StartingMetrics": ["Dynamic_Value-scale05"]
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/util/test_split_by_category.py -q -p no:cacheprovider`
Expected: FAIL at collection with `ImportError: cannot import name 'split_measurements_by_category'`.

- [ ] **Step 3: Implement in `_measurement_outputs.py`**

Add `CATEGORIES` to the schema import: `from phenotypic.schema import CATEGORIES, MeasurementInfo`.

Replace the body of `split_measurements` and add the new public function directly after it:

```python
def split_measurements(df: MeasurementFrame) -> dict[str, MeasurementFrame]:
    """(docstring unchanged)"""
    columns = _columns(df)
    groups = _producer_column_groups(columns)
    if not groups:
        return {}
    return _split_by_groups(df, _context_columns(columns, groups), groups)


def split_measurements_by_category(df: MeasurementFrame) -> dict[str, MeasurementFrame]:
    """Split a measurements table into one data frame per measurement category.

    Context columns are the same as :func:`split_measurements`: every column
    not owned by a producer's ``MeasurementInfo`` (metadata, object label,
    grid, joined external metadata). Each category frame holds those context
    columns followed by the present columns whose member carries the category
    (see :class:`phenotypic.schema.CATEGORIES`), in input order. Measurement
    columns outside a category are dropped from its frame. A column in several
    categories appears in each.

    Args:
        df: A pandas or polars measurements DataFrame.

    Returns:
        Mapping of category label (e.g. ``"StartingMetrics"``) to a same-type
        DataFrame. Categories with no present columns are omitted.

    Raises:
        TypeError: If *df* is not a pandas or polars DataFrame.
    """
    columns = _columns(df)
    producer_groups = _producer_column_groups(columns)
    if not producer_groups:
        return {}
    context = _context_columns(columns, producer_groups)
    context_set = set(context)
    measured = [column for column in columns if column not in context_set]
    return _split_by_groups(df, context, _category_column_groups(measured))


def _context_columns(columns: list[str], groups: dict[str, list[str]]) -> list[str]:
    """Columns not claimed by any producer group, in input order."""
    owned = {column for group_columns in groups.values() for column in group_columns}
    return [column for column in columns if column not in owned]


def _split_by_groups(
    df: MeasurementFrame,
    context: list[str],
    groups: dict[str, list[str]],
) -> dict[str, MeasurementFrame]:
    """Select ``context + group`` columns for every group, preserving frame type."""
    return {key: _select_columns(df, context + cols) for key, cols in groups.items()}


def _category_column_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    """Map category labels to the *columns* whose member carries that category.

    Keys follow ``CATEGORIES`` declaration order; values follow *columns* order.
    """
    by_category: dict[CATEGORIES, list[str]] = {category: [] for category in CATEGORIES}
    for column in columns:
        member = _member_for_column(column)
        if member is None:
            continue
        for category in CATEGORIES.in_order(member.categories):
            by_category[category].append(column)
    return {category.label: cols for category, cols in by_category.items() if cols}


def _member_for_column(column: str) -> MeasurementInfo | None:
    """Resolve *column* to its public schema member across all header schemes."""
    for info in _public_info_classes():
        member = info.member_for_header(column)
        if member is not None:
            return member
    return None
```

`_category_column_groups` reads the module-level name `CATEGORIES` at call time. That is the
seam the two-category test patches; don't bind it to a local alias. Rewrite `_describe_column`
on the new resolver:

```python
def _describe_column(column: str) -> str | None:
    """Resolve *column* to its member's ``desc`` across all schemes, or None."""
    member = _member_for_column(column)
    return member.desc if member is not None else None
```

Update `__all__` to `["generate_output_key", "split_measurements", "split_measurements_by_category"]`.

- [ ] **Step 4: Export from `phenotypic.util`**

`src/phenotypic/util/__init__.py`: change line 6 to
`from ._measurement_outputs import generate_output_key, split_measurements, split_measurements_by_category`
and add `"split_measurements_by_category",` to `__all__` after `"split_measurements",`.

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/unit/util -q -p no:cacheprovider`
Expected: PASS, including the pre-existing `split_measurements`/`generate_output_key` tests
(the refactor must not change their output).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/util/_measurement_outputs.py src/phenotypic/util/__init__.py tests/unit/util/test_split_by_category.py
git add src/phenotypic/util/_measurement_outputs.py src/phenotypic/util/__init__.py tests/unit/util/test_split_by_category.py
git commit -m "feat(util): split_measurements_by_category on a shared grouping helper

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01D1sMyGvSvvLnJSAM5skaEa"
```

### Task 5: CLI writes `measurements_by_category/` on the finalization path

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py:30,785-787,2182-2184` (constant + helper + module
  doc list)
- Modify: `src/phenotypic/sdk_/__init__.py:169,295,613,703` (imports + `__all__`)
- Modify: `src/phenotypic/_cli/_cli_output_manager.py` (imports ~line 58; `_write_split`;
  `split_master_by_feature` at 1348; new `split_master_by_category`; call in
  `finalize_post_master_outputs` after line 1298)
- Modify: `tests/unit/sdk_/test_io_constants.py:435-438,525`
- Modify: `tests/unit/cli/test_cli_output_manager.py` (new class + aggregate test)
- Modify: `tests/unit/cli/test_cli_recompile_slurm.py:821-835` (extend finalizer test)
- Create: `tests/unit/cli/test_category_split_call_site.py`

**Interfaces:**
- Consumes: `split_measurements_by_category` (Task 4).
- Produces: `phenotypic.sdk_.DIR_MEASUREMENTS_BY_CATEGORY == "measurements_by_category"`,
  `phenotypic.sdk_.measurements_by_category_dir(output_dir: Path) -> Path`, and
  `_cli_output_manager.split_master_by_category(master_df: pl.DataFrame, output_dir: Path) ->
  dict[str, Path]` (label → CSV path; `{}` when nothing to split, and then no directory is
  created).

- [ ] **Step 1: Write the failing tests**

In `tests/unit/sdk_/test_io_constants.py`, add `measurements_by_category_dir` to the import list
next to `measurements_by_feature_dir`, add this test after `test_measurements_by_feature_dir`:

```python
    def test_measurements_by_category_dir(self, output: Path) -> None:
        assert measurements_by_category_dir(output) == (
            output / "deliverables" / "measurements_by_category"
        )
```

and add `"measurements_by_category_dir": measurements_by_category_dir,` beside the
`"measurements_by_feature_dir"` entry in the helper dict near line 525.

In `tests/unit/cli/test_cli_output_manager.py`, import `measurements_by_category_dir` (from
`phenotypic.sdk_`, beside `measurements_by_feature_dir`) and `split_master_by_category` (beside
`split_master_by_feature`), then add:

```python
class TestSplitMasterByCategory:
    """``split_master_by_category`` writes one CSV + Parquet per category."""

    @staticmethod
    def _master() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "Metadata_Dataset": ["ds1"],
                "Object_Label": [1],
                "Size_Area": [10.0],
                "Shape_Circularity": [0.9],
            }
        )

    def test_writes_starting_metrics_csv_and_parquet(self, tmp_path: Path) -> None:
        written = split_master_by_category(self._master(), tmp_path)
        split_dir = measurements_by_category_dir(tmp_path)
        assert written == {"StartingMetrics": split_dir / "StartingMetrics.csv"}
        csv_df = pl.read_csv(split_dir / "StartingMetrics.csv")
        pq_df = pl.read_parquet(split_dir / "StartingMetrics.parquet")
        assert csv_df.columns == ["Metadata_Dataset", "Object_Label", "Size_Area"]
        assert pq_df.columns == csv_df.columns
        assert pq_df["Size_Area"].to_list() == [10.0]

    def test_no_categorized_columns_writes_nothing(self, tmp_path: Path) -> None:
        master = self._master().drop("Size_Area")
        assert split_master_by_category(master, tmp_path) == {}
        assert not measurements_by_category_dir(tmp_path).exists()
```

Also add a sibling of `test_no_state_file_keeps_master_and_splits_known_columns` in the same
class as that test (the full/measure path through `aggregate_measurements`):

```python
    def test_aggregate_writes_category_split_beside_feature_split(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        ds_dir = output_dir / "results" / "ds1" / "measurements"
        ds_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "Metadata_Dataset": ["ds1"],
                str(IMAGE.IMAGE_NAME): ["img1"],
                "Object_Label": [1],
                "Size_Area": [10.0],
                "Shape_Circularity": [0.9],
            }
        ).write_parquet(ds_dir / "img1.parquet")

        aggregate_measurements(
            output_dir=output_dir, dataset_names=["ds1"], include_dataset_column=True
        )

        category_csv = measurements_by_category_dir(output_dir) / "StartingMetrics.csv"
        assert category_csv.exists()
        df = pl.read_csv(category_csv)
        assert "Size_Area" in df.columns
        assert "Shape_Circularity" not in df.columns
        assert str(IMAGE.IMAGE_NAME) in df.columns

    def test_failing_category_split_does_not_block_publication(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        ds_dir = output_dir / "results" / "ds1" / "measurements"
        ds_dir.mkdir(parents=True)
        pl.DataFrame(
            {
                "Metadata_Dataset": ["ds1"],
                str(IMAGE.IMAGE_NAME): ["img1"],
                "Object_Label": [1],
                "Size_Area": [10.0],
            }
        ).write_parquet(ds_dir / "img1.parquet")

        with patch(
            "phenotypic._cli._cli_output_manager.split_measurements_by_category",
            side_effect=RuntimeError("boom"),
        ):
            master_path = aggregate_measurements(
                output_dir=output_dir, dataset_names=["ds1"], include_dataset_column=True
            )

        assert master_path is not None and master_path.exists()
        assert measurements_csv_path(output_dir).exists()
        assert (measurements_by_feature_dir(output_dir) / "MeasureSize.csv").exists()
        assert not measurements_by_category_dir(output_dir).exists()
```

In `tests/unit/cli/test_cli_recompile_slurm.py`, import `measurements_by_category_dir` from
`phenotypic.sdk_`, and in `test_finalizer_writes_master_outputs_and_rebuilds_dashboard` add after
the `measurements_parquet_path(output_dir).exists()` assertion:

```python
    # The recompile finalizer reaches the category split through the same
    # finalize_post_master_outputs path as full/measure (spec §5.3).
    category_csv = measurements_by_category_dir(output_dir) / "StartingMetrics.csv"
    assert category_csv.exists()
    assert pl.read_csv(category_csv)["Size_Area"].to_list() == [1, 2]
```

`tests/unit/cli/test_category_split_call_site.py`:

```python
"""split_master_by_category has exactly one caller: finalize_post_master_outputs.

That function is the single finalization path shared by full, measure and
recompile (spec §5.3). A second call site would let one mode drift.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src" / "phenotypic"


def _calls_by_enclosing_function(name: str) -> list[str]:
    found: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == name
                ):
                    found.append(f"{path.name}:{func.name}")
    return found


def test_category_split_is_called_only_from_finalize_post_master_outputs() -> None:
    assert _calls_by_enclosing_function("split_master_by_category") == [
        "_cli_output_manager.py:finalize_post_master_outputs"
    ]
```

The lambda inside `finalize_post_master_outputs` is walked as part of that function's subtree,
so the call is attributed to it.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/sdk_/test_io_constants.py tests/unit/cli/test_cli_output_manager.py tests/unit/cli/test_category_split_call_site.py tests/unit/cli/test_cli_recompile_slurm.py::test_finalizer_writes_master_outputs_and_rebuilds_dashboard -q -p no:cacheprovider`
Expected: FAIL at collection with `ImportError: cannot import name 'measurements_by_category_dir'`.

- [ ] **Step 3: Constant and path helper**

`src/phenotypic/sdk_/_io_constants.py`, after `DIR_MEASUREMENTS_BY_FEATURE`:

```python
#: Per-category spreadsheet split written by
#: :func:`phenotypic._cli._cli_output_manager.split_master_by_category`.
DIR_MEASUREMENTS_BY_CATEGORY: Final[str] = "measurements_by_category"
```

After `measurements_by_feature_dir`:

```python
def measurements_by_category_dir(output_dir: Path) -> Path:
    """Return ``<output>/deliverables/measurements_by_category/``."""
    return deliverables_dir(output_dir) / DIR_MEASUREMENTS_BY_CATEGORY
```

In the module docstring list (line ~30), add ``` ``measurements_by_category_dir``, ``` after
``` ``measurements_by_feature_dir``, ```. In `src/phenotypic/sdk_/__init__.py`, add
`DIR_MEASUREMENTS_BY_CATEGORY` beside `DIR_MEASUREMENTS_BY_FEATURE` (import ~169, `__all__`
~613) and `measurements_by_category_dir` beside `measurements_by_feature_dir` (import ~295,
`__all__` ~703).

- [ ] **Step 4: Extract `_write_split` and add `split_master_by_category`**

In `_cli_output_manager.py`, add `measurements_by_category_dir` to the `phenotypic.sdk_` import
block (~line 58), and change line 47 to
`from phenotypic.util import split_measurements, split_measurements_by_category`. It must stay a
module-level name: the failure-isolation test patches
`phenotypic._cli._cli_output_manager.split_measurements_by_category`. Add `Mapping` to the
`from typing import (...)` block at line 19 if it isn't already there. Replace `split_master_by_feature` (1348-1414) with:

```python
def _write_split(
    split_dir: Path,
    split_frames: "Mapping[str, pd.DataFrame | pl.DataFrame]",
) -> Dict[str, Path]:
    """Atomically write one CSV + Parquet per split frame into *split_dir*.

    A failed CSV skips that key; a failed Parquet keeps the CSV. Returns
    key → CSV path for every key whose CSV was written.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}
    for key, subset_frame in split_frames.items():
        if isinstance(subset_frame, pd.DataFrame):
            subset = pl.from_pandas(subset_frame)
        else:
            subset = subset_frame
        csv_path = split_dir / f"{key}.csv"
        pq_path = split_dir / f"{key}.parquet"

        def _write_parquet(
            path: str, _subset: "pl.DataFrame" = subset
        ) -> None:
            _subset.write_parquet(path, **PARQUET_WRITE_OPTIONS)

        try:
            atomic_write_with_writer(csv_path, subset.write_csv)
        except Exception:
            logger.warning(
                "Failed to write split CSV for %r", key, exc_info=True
            )
            continue

        try:
            atomic_write_with_writer(pq_path, _write_parquet)
        except Exception:
            logger.warning(
                "Failed to write split Parquet for %r (CSV was saved)",
                key,
                exc_info=True,
            )

        written[key] = csv_path
        logger.info(
            "Split %r: %d rows x %d cols -> %s",
            key,
            subset.height,
            subset.width,
            csv_path.name,
        )

    return written


def split_master_by_feature(
    master_df: "pl.DataFrame",
    output_dir: Path,
    pipeline: Optional["ImagePipeline"] = None,
) -> Dict[str, Path]:
    """(docstring unchanged)"""
    del pipeline

    master_df = normalize_measurement_metadata_columns(master_df)
    split_frames = split_measurements(master_df)
    if not split_frames:
        logger.info("No recognized MeasurementInfo columns -- skipping split")
        return {}
    return _write_split(measurements_by_feature_dir(output_dir), split_frames)


def split_master_by_category(
    master_df: "pl.DataFrame",
    output_dir: Path,
) -> Dict[str, Path]:
    """Write one CSV + Parquet per measurement category into *output_dir*.

    Creates ``deliverables/measurements_by_category/`` and emits
    ``<label>.{csv,parquet}`` for every key returned by
    :func:`phenotypic.util.split_measurements_by_category`: the feature
    split's context columns plus that category's present columns. Called only
    from :func:`finalize_post_master_outputs`, the finalization path shared by
    ``full``, ``measure`` and ``recompile``.

    Args:
        master_df: The post-applied, metadata-joined measurements frame.
        output_dir: Run output root.

    Returns:
        Mapping of category label → path to the emitted CSV. Empty, with no
        directory created, when no categorized column is present.
    """
    master_df = normalize_measurement_metadata_columns(master_df)
    split_frames = split_measurements_by_category(master_df)
    if not split_frames:
        logger.info("No categorized measurement columns -- skipping category split")
        return {}
    return _write_split(measurements_by_category_dir(output_dir), split_frames)
```

Add `Mapping` to the module's `typing` import if it isn't there.

- [ ] **Step 5: Call it in `finalize_post_master_outputs`**

Directly after the existing `split_master_by_feature` `_guarded_terminal_best_effort(...)` block
(ends ~line 1298), insert:

```python
    # Same frame, same guard: one spreadsheet per measurement category
    # (deliverables/measurements_by_category/). This is the only call site, so
    # full, measure and recompile all publish it (spec §5.3).
    _guarded_terminal_best_effort(
        commit_guard,
        lambda: split_master_by_category(post_df, output_dir),
        warning=(
            "Per-category measurement split failed (master files still written)"
        ),
        default={},
    )
```

Update `finalize_post_master_outputs`'s docstring step 4 to mention both splits:
"Split ``post_df`` into per-feature and per-category spreadsheets
(``measurements_by_feature/`` and ``measurements_by_category/``)".

- [ ] **Step 6: Run to verify they pass**

Run: `uv run pytest tests/unit/sdk_/test_io_constants.py tests/unit/cli/test_cli_output_manager.py tests/unit/cli/test_category_split_call_site.py tests/unit/cli/test_cli_recompile_slurm.py tests/unit/cli/test_cli_recompile.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/__init__.py src/phenotypic/_cli/_cli_output_manager.py tests/unit/sdk_/test_io_constants.py tests/unit/cli/test_cli_output_manager.py tests/unit/cli/test_cli_recompile_slurm.py tests/unit/cli/test_category_split_call_site.py
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/__init__.py src/phenotypic/_cli/_cli_output_manager.py tests/unit/sdk_/test_io_constants.py tests/unit/cli/test_cli_output_manager.py tests/unit/cli/test_cli_recompile_slurm.py tests/unit/cli/test_category_split_call_site.py
git commit -m "feat(cli): publish measurements_by_category/ on the finalize path

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01D1sMyGvSvvLnJSAM5skaEa"
```

### Task 6: README documents both splits and categories

**Files:**
- Modify: `src/phenotypic/_cli/_cli_readme_generator.py` (`generate` sections list,
  `_generate_output_structure`, `_generate_measurement_table`, new `_generate_categories_section`)
- Create: `tests/unit/cli/test_readme_categories.py`

**Interfaces:**
- Consumes: `CATEGORIES`, `member.categories`, `metric_family()`.
- Produces: `READMEGenerator._generate_categories_section() -> str` (`""` when no configured
  measurer emits a categorized column).

- [ ] **Step 1: Write the failing tests**

```python
"""README: split folders in the tree, a Categories column, a Categories section."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from phenotypic import ImagePipeline
from phenotypic._cli._cli_readme_generator import READMEGenerator
from phenotypic.measure import MeasureShape, MeasureSize
from phenotypic.schema import CATEGORIES, SHAPE, SIZE


def _generator(*measurers) -> READMEGenerator:
    # generate() reads config.image_type and config.pipeline_json.name.
    return READMEGenerator(
        config=SimpleNamespace(image_type="Image", pipeline_json=Path("pipeline.json")),
        pipeline=ImagePipeline(meas=list(measurers)),
    )


def test_output_tree_lists_both_split_folders() -> None:
    tree = _generator(MeasureSize())._generate_output_structure([])
    assert "measurements_by_feature/" in tree
    assert "measurements_by_category/" in tree


def test_categorized_table_gains_a_categories_column() -> None:
    table = _generator()._generate_measurement_table(SIZE)
    assert "| Column | Description | Categories |" in table
    assert f"| `{SIZE.AREA}` |" in table
    assert "Starting Metrics |" in table


def test_uncategorized_table_has_no_categories_column() -> None:
    table = _generator()._generate_measurement_table(SHAPE)
    assert "Categories" not in table


def test_categories_section_lists_present_columns_only() -> None:
    section = _generator(MeasureSize())._generate_categories_section()
    assert "## Measurement Categories" in section
    assert "### Starting Metrics" in section
    assert CATEGORIES.STARTING_METRICS.desc in section
    assert "measurements_by_category/StartingMetrics.csv" in section
    assert f"`{SIZE.AREA}`" in section
    assert "ColorLab_L*Medoid" not in section  # MeasureColor not configured


def test_categories_section_is_empty_without_categorized_measurers() -> None:
    assert _generator(MeasureShape())._generate_categories_section() == ""


def test_generate_includes_the_categories_section(tmp_path) -> None:
    path = _generator(MeasureSize()).generate(tmp_path, [])
    assert "## Measurement Categories" in path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/cli/test_readme_categories.py -q -p no:cacheprovider`
Expected: FAIL (`AttributeError: ... '_generate_categories_section'`, tree asserts fail).

- [ ] **Step 3: Implement**

In `_generate_output_structure`, insert after the `measurements.parquet` line:

```
|   +-- measurements_by_feature/      # One CSV + Parquet per measurer: shared context columns + that measurer's columns
|   +-- measurements_by_category/     # One CSV + Parquet per measurement category (see Measurement Categories)
```

Replace `_generate_measurement_table` with:

```python
    def _generate_measurement_table(self, info_cls) -> str:
        """Generate markdown table for a MeasurementInfo class."""
        from phenotypic.schema import CATEGORIES

        try:
            family = info_cls.metric_family()
            members = list(info_cls)

            if not members:
                return ""

            has_categories = any(m.categories for m in members)
            table = f"\n### {family}\n\n"
            if has_categories:
                table += "| Column | Description | Categories |\n"
                table += "|--------|-------------|------------|\n"
            else:
                table += "| Column | Description |\n"
                table += "|--------|-------------|\n"

            for member in members:
                col_name = str(member)
                desc = member.desc if hasattr(member, "desc") else ""
                desc = desc.replace("|", "\\|").replace("\n", " ")
                if len(desc) > 200:
                    desc = desc[:197] + "..."
                row = f"| `{col_name}` | {desc} |"
                if has_categories:
                    names = ", ".join(
                        c.display_name for c in CATEGORIES.in_order(member.categories)
                    )
                    row += f" {names} |"
                table += row + "\n"

            return table
        except Exception as e:
            logger.warning(f"Could not generate table for {info_cls}: {e}")
            return ""
```

Add, after `_generate_measurements_section`:

```python
    def _generate_categories_section(self) -> str:
        """Document each category the configured measurers emit columns for.

        Only columns this pipeline's measurers can produce are listed, so a
        category whose members all come from an unconfigured measurer is
        omitted. Returns ``""`` when no category applies.
        """
        from phenotypic.abc_ import MeasureFeatures
        from phenotypic.schema import CATEGORIES

        infos = [
            info
            for measurer in (self.pipeline._meas or {}).values()
            if isinstance(measurer, MeasureFeatures)
            for info in self._get_measurement_infoclasses(measurer)
        ]
        blocks: list[str] = []
        for category in CATEGORIES:
            columns = [
                str(member)
                for info in infos
                for member in info
                if category in member.categories
            ]
            if not columns:
                continue
            listed = "\n".join(f"- `{column}`" for column in dict.fromkeys(columns))
            blocks.append(
                f"### {category.display_name}\n\n{category.desc}\n\n"
                f"Written to `deliverables/measurements_by_category/{category.label}.csv` "
                f"(and `.parquet`), alongside every context column.\n\n{listed}"
            )
        if not blocks:
            return ""
        return "## Measurement Categories\n\n" + "\n\n".join(blocks)
```

In `generate`, insert `self._generate_categories_section(),` after
`self._generate_measurements_section(),`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/cli/test_readme_categories.py tests/unit/cli/test_readme_measurement_tables.py tests/unit/cli/test_readme_model_section.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_readme_generator.py tests/unit/cli/test_readme_categories.py
git add src/phenotypic/_cli/_cli_readme_generator.py tests/unit/cli/test_readme_categories.py
git commit -m "feat(cli): README documents measurement splits and categories

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01D1sMyGvSvvLnJSAM5skaEa"
```

**Phase 2 gate:** run the affected surface once as a Slurm job: `tests/unit/cli`,
`tests/unit/util`, `tests/unit/sdk_`, plus
`git grep -lE 'split_measurements|measurements_by_feature|_cli_readme_generator' -- tests`. Quote
the pass/fail counts.

---

## Phase 3: docs

### Task 7: Category badges in `rst_table()`

**Files:**
- Modify: `src/phenotypic/schema/_measurement_info.py` (`_CATEGORY_BADGE_COLOR`,
  `category_badges` property, `_render_info_table`, `rst_table`)
- Modify: `tests/unit/schema/test_rst_rendering.py` (append)
- Modify: `tests/unit/schema/test_classification.py` (extend color test)

**Interfaces:**
- Consumes: `CATEGORIES.display_name`, `.anchor`, `.in_order` (Task 2).
- Produces: `member.category_badges -> str` (space-joined
  `:bdg-ref-dark-line:`<display_name> <anchor>`` roles; `""` when uncategorized). `rst_table()`
  gains a `Categories` column after `Type`, only when some member has a category.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/schema/test_rst_rendering.py`:

```python
def test_categorized_table_has_a_categories_badge_column() -> None:
    from phenotypic.schema import SIZE

    table = SIZE.rst_table()
    assert "     - Categories" in table
    assert (
        ":bdg-ref-dark-line:`Starting Metrics <measurement-category-startingmetrics>`"
        in table
    )


def test_uncategorized_table_has_no_categories_column() -> None:
    from phenotypic.schema import SHAPE

    assert "Categories" not in SHAPE.rst_table()


def test_category_badges_is_empty_for_an_uncategorized_member() -> None:
    from phenotypic.schema import SHAPE, SIZE

    assert next(iter(SHAPE)).category_badges == ""
    assert SIZE.AREA.category_badges.count(":bdg-ref-") == 1
```

In `tests/unit/schema/test_classification.py`, extend
`test_badge_spec_colors_are_valid_sphinx_design_semantic_colors`: import `_CATEGORY_BADGE_COLOR`
beside `_BADGE_SPECS` and change the `used` line to
`used = {color for _text, color, _anchor in _BADGE_SPECS.values()} | {_CATEGORY_BADGE_COLOR}`.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/schema/test_rst_rendering.py tests/unit/schema/test_classification.py -q -p no:cacheprovider`
Expected: FAIL (`"     - Categories" not in table`; `ImportError: _CATEGORY_BADGE_COLOR`).

- [ ] **Step 3: Implement**

After `_BADGE_SPECS`:

```python
#: Category badges render as *outline* pills (``:bdg-ref-{color}-line:``) so
#: they read as a separate axis from the solid Type pills. A sphinx-design
#: semantic color (asserted in test_classification). Each links to the
#: category's section on the generated Categories page.
_CATEGORY_BADGE_COLOR: Final = "dark"
```

On `MeasurementInfo`, after `use_badge`:

```python
    @property
    def category_badges(self) -> str:
        """RST outline badges, one per category, linking to the Categories page.

        Empty when the member has no category.
        """
        return " ".join(
            f":bdg-ref-{_CATEGORY_BADGE_COLOR}-line:`{c.display_name} <{c.anchor}>`"
            for c in CATEGORIES.in_order(self.categories)
        )
```

`_render_info_table`: the row tuple gains a 6th element. Change the `rows` annotation to
`list[tuple[str, str, str, str | None, str, str]]` and the docstring to
`rows: (name_cell, desc, bio_desc, image_relpath_or_None, type_badge, category_badges) per
member. Both badge cells are raw RST inserted unescaped.` Add
`has_cat = any(row[5] for row in rows)`; after the `if has_use: lines.append("     - Type")`
header line, add `if has_cat: lines.append("     - Categories")`; change the loop to
`for name, desc, bio, img, use, cats in rows:` and after the `if has_use:` cell, add
`if has_cat: lines.append(f"     - {cats}")`. Update the docstring summary to "Type/Categories/
Biology/Image columns appear only when populated."

`rst_table`: append `m.category_badges,` as the sixth tuple element.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/schema -q -p no:cacheprovider`
Expected: PASS. The color test runs only where sphinx-design is installed. Run it once
explicitly with the docs group:
`uv run --group docs pytest tests/unit/schema/test_classification.py -q -p no:cacheprovider -k semantic`
Expected: PASS (not skipped).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/schema/_measurement_info.py tests/unit/schema/test_rst_rendering.py tests/unit/schema/test_classification.py
git add src/phenotypic/schema/_measurement_info.py tests/unit/schema/test_rst_rendering.py tests/unit/schema/test_classification.py
git commit -m "feat(schema): outline category badges in measurement tables

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01D1sMyGvSvvLnJSAM5skaEa"
```

### Task 8: Generated Categories subpage under the Measurements tab

**Files:**
- Modify: `docs/source/_extensions/measurements_ref.py`
- Modify: `docs/source/_templates/navbar-nav.html:86-91` (third dropdown entry)
- Modify: `tests/unit/docs/test_measurements_ref_extension.py:67-78,93-113,115-125` (+ new tests)

**Interfaces:**
- Consumes: `CATEGORIES` (`.label`, `.desc`, `.display_name`, `.anchor`, `.members()`),
  `member.use_badge`, `member.category_badges`, `_section_label(info_cls)`.
- Produces: `measurements_ref/categories/index.rst`; the Measurements page's hidden toctree is
  `../metadata/index` then `../categories/index`.

- [ ] **Step 1: Update and add the failing tests**

In `tests/unit/docs/test_measurements_ref_extension.py`:

1. Rename `test_build_pages_creates_exactly_two_reference_pages` →
   `test_build_pages_creates_exactly_three_reference_pages`, expecting
   `["categories/index.rst", "measurements/index.rst", "metadata/index.rst"]`.
2. In `test_setup_generates_pages_before_sphinx_source_discovery`, add
   `assert (docs_root / "categories" / "index.rst").is_file()`.
3. Replace `test_measurements_page_has_metadata_as_its_only_toctree_child` with:

```python
def test_measurements_page_has_metadata_and_categories_as_toctree_children(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)
    measurements_page = (docs_root / "measurements" / "index.rst").read_text()

    assert (
        ".. toctree::\n   :hidden:\n\n   ../metadata/index\n   ../categories/index"
        in measurements_page
    )
    for child in ("metadata", "categories"):
        assert ".. toctree::" not in (docs_root / child / "index.rst").read_text()
```

4. Append:

```python
def test_categories_page_has_one_section_per_category_with_verbatim_desc(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    from phenotypic.schema import CATEGORIES

    page = (_build_reference_tree(tmp_path, monkeypatch) / "categories" / "index.rst").read_text()
    for category in CATEGORIES:
        assert f".. _{category.anchor}:" in page
        assert category.display_name in page
        assert category.desc in page
        assert f"measurements_by_category/{category.label}.csv" in page
    assert "``Size_Area``" in page
    assert ":ref:`Size <measurement-info-size>`" in page


def test_categories_page_is_generated_from_member_tags(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    from phenotypic.schema import CATEGORIES

    class FUTURE_TAGGED(MeasurementInfo):
        @classmethod
        def metric_family(cls) -> str:
            return "FutureTagged"

        VALUE = Entry("Value", "A value.", categories=CATEGORIES.STARTING_METRICS)

    monkeypatch.setattr(schema, "FUTURE_TAGGED", FUTURE_TAGGED, raising=False)
    monkeypatch.setattr(schema, "__all__", [*schema.__all__, "FUTURE_TAGGED"])
    page = (_build_reference_tree(tmp_path, monkeypatch) / "categories" / "index.rst").read_text()
    assert "``FutureTagged_Value``" in page


def test_every_badge_anchor_resolves_on_the_categories_page(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    import re

    page = (_build_reference_tree(tmp_path, monkeypatch) / "categories" / "index.rst").read_text()
    anchors = {
        match
        for info_cls in _canonical_public_classes()
        for match in re.findall(r"<(measurement-category-[a-z0-9]+)>", info_cls.rst_table())
    }
    assert anchors, "no category badge rendered anywhere"
    for anchor in anchors:
        assert f".. _{anchor}:" in page


def test_navbar_lists_categories_under_measurements() -> None:
    navbar = (_REPO_ROOT / "docs" / "source" / "_templates" / "navbar-nav.html").read_text()
    assert "pathto('measurements_ref/categories/index')" in navbar
    assert navbar.index("measurements_ref/metadata/index") < navbar.index(
        "measurements_ref/categories/index"
    )
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/docs/test_measurements_ref_extension.py -q -p no:cacheprovider`
Expected: FAIL (no `categories/index.rst`; navbar lacks the entry).

- [ ] **Step 3: Implement the page in `measurements_ref.py`**

Module docstring: "two deterministic pages: Measurements and Metadata" → "three deterministic
pages: Measurements, Metadata, and Categories (a subpage of Measurements, listing each
``CATEGORIES`` member's ``desc`` and columns)". Bump `setup`'s version to `"0.5"`.

Replace `_build_reference_page`'s `metadata_child: bool = False` parameter with
`child_pages: tuple[str, ...] = ()` and its toctree block with:

```python
    if child_pages:
        out.extend([".. toctree::", "   :hidden:", ""])
        out.extend(f"   ../{child}/index" for child in child_pages)
        out.append("")
```

Add:

```python
_CATEGORIES_INTRO = (
    "Categories are curated groupings of measurement columns drawn from "
    "several metric families; one column can belong to several categories. "
    "Unlike the Type badge, a category makes no claim about how far a single "
    "value can be trusted (see :ref:`measurement-categories`). Every run that "
    "measures objects writes one spreadsheet per category under "
    "``deliverables/measurements_by_category/``, holding the shared context "
    "columns (metadata, object label, grid) plus that category's columns."
)


def _category_section(category: Any) -> str:
    """Render one category: anchor, heading, verbatim desc, output file, column table."""
    out = [
        f".. _{category.anchor}:",
        "",
        *_heading(category.display_name, "-"),
        category.desc,
        "",
        f"Written to ``deliverables/measurements_by_category/{category.label}.csv`` "
        "(and ``.parquet``).",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Column",
        "     - Metric family",
        "     - Type",
    ]
    for member in category.members():
        info_cls = type(member)
        out.extend(
            [
                f"   * - ``{member.value}``",
                f"     - :ref:`{info_cls.metric_family()} <{_section_label(info_cls)}>`",
                f"     - {member.use_badge}",
            ]
        )
    out.append("")
    return "\n".join(out)


def _build_categories_page() -> str:
    """Build the generated Categories page, one section per CATEGORIES member."""
    from phenotypic.schema import CATEGORIES

    out = [*_heading("Categories", "="), _CATEGORIES_INTRO, ""]
    out.extend(_category_section(category) for category in CATEGORIES)
    return "\n".join(out)
```

In `_build_pages`, pass `child_pages=("metadata", "categories")` to the Measurements page build
and add:

```python
    _write(output_dir / "categories" / "index.rst", _build_categories_page())
```

- [ ] **Step 4: Navbar entry**

In `docs/source/_templates/navbar-nav.html`, after the Metadata `<li>…</li>` (ends ~line 91),
add:

```html
        <li>
          <a class="nav-link dropdown-item nav-internal{% if _pn.startswith('measurements_ref/categories/') %} current active{% endif %}"
             href="{{ pathto('measurements_ref/categories/index') }}">
            Categories
          </a>
        </li>
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/unit/docs/test_measurements_ref_extension.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix docs/source/_extensions/measurements_ref.py tests/unit/docs/test_measurements_ref_extension.py
git add docs/source/_extensions/measurements_ref.py docs/source/_templates/navbar-nav.html tests/unit/docs/test_measurements_ref_extension.py
git commit -m "docs: generated Categories subpage under the Measurements tab

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01D1sMyGvSvvLnJSAM5skaEa"
```

### Task 9: Prose, module guides, and a verified docs build

**Files:**
- Modify: `docs/source/explanation/measurement_classification_system.md` (new section)
- Modify: `docs/source/tutorials/pages/cli_modes.md:85` (output tree)
- Modify: `src/phenotypic/schema/CLAUDE.md` (Categories section)
- Modify: `CLAUDE.md` (Gotchas `deliverables/` bullet)
- Use: `docs/superpowers/plans/2026-09-25-measurement-categories/build_docs_categories.sbatch`

**Interfaces:**
- Consumes: the `measurement-categories` anchor referenced by Task 8's intro.
- Produces: `(measurement-categories)=` MyST anchor.

- [ ] **Step 1: Explanation section**

Append to `measurement_classification_system.md`:

```markdown
(measurement-categories)=
## Categories are groupings, not trust claims

Every measurement column is named `<Family>_<Label>`: `Size_Area` belongs to the
**Size** metric family. The family says only which schema the column comes from.

A **category** is a separate, curated grouping that cuts across families. *Starting
Metrics*, for example, gathers the size magnitudes, integrated intensity, and the CIELAB
medoid colour, which come from three different families. One column can belong to
several categories. A category tells you where to look first, not how far to trust a
single value; that is the job of the kind and tier above.

Each run that measures objects writes one spreadsheet per category under
`deliverables/measurements_by_category/`. The full list of categories, with each one's
columns, is on the {doc}`Categories </measurements_ref/categories/index>` page.
```

- [ ] **Step 2: CLI tree**

In `cli_modes.md`, after `│   ├── measurements_by_feature/            # one file per measurer`,
add
`│   ├── measurements_by_category/           # one file per measurement category`.

- [ ] **Step 3: Module guides**

`src/phenotypic/schema/CLAUDE.md`: add a `## Measurement categories` section after
"Classification badges in the docs", containing:

- `CATEGORIES` (`_categories.py`) is a closed, repo-defined `str` enum of `CategoryEntry(label,
  desc)`; value == label; it deliberately does **not** subclass `MeasurementInfo` (at least six
  discovery sites use `issubclass(x, MeasurementInfo)`).
- Tag a member with `Entry(..., categories=CATEGORIES.X)` (bare member or iterable; a raw string
  is refused). Metadata owners may not be categorized.
- To add a category: add a `CategoryEntry` member (CamelCase label, technical `desc`; agents may
  write it), tag members, and update the pin test in `tests/unit/schema/test_categories.py`. The
  Categories docs page, the README section, and `measurements_by_category/<label>.*` follow
  automatically.
- Badges: `_CATEGORY_BADGE_COLOR` outline pills link to `CATEGORIES.X.anchor` on the generated
  Categories page.

Root `CLAUDE.md`, Gotchas **Output layout** bullet: after the sentence that introduces
`measurements.{csv,parquet}`, add: "`measurements_by_feature/` (one file per measurer) and
`measurements_by_category/` (one file per `CATEGORIES` member) are both written from that mirror
by `finalize_post_master_outputs`, the finalization path shared by full, measure and recompile."

- [ ] **Step 4: Build the docs as a Slurm job**

```bash
jid=$(sbatch --parsable --export=ALL,WORKTREE="$PWD" docs/superpowers/plans/2026-09-25-measurement-categories/build_docs_categories.sbatch)
[[ "$jid" =~ ^[0-9]+$ ]] || { echo "submit failed: $jid"; exit 1; }
scontrol show job "$jid" | grep -E 'StartTime|Reason'
```

Wait for completion (`sacct -j "$jid" --format=State,ExitCode`). Then read the log for warnings
touching the changed pages:
`grep -nE 'measurements_ref|measurement-categor|categories' /bigdata/exfab/anguy344/slurm_logs/cat-docs_${jid}.log`
Expected: no `WARNING`/`ERROR` lines about them.

- [ ] **Step 5: Read the rendered HTML** (exit 0 is not the check)

```bash
B=docs/_build/categories
grep -c 'measurement-category-startingmetrics' $B/measurements_ref/categories/index.html
grep -o 'Starting Metrics' $B/measurements_ref/categories/index.html | head -1
grep -c 'sd-outline-dark' $B/measurements_ref/measurements/index.html
grep -o 'measurements_ref/categories/index.html' $B/index.html | head -1
grep -c 'Metric family: <strong>Size</strong>\|Metric family: ' $B/measurements_ref/measurements/index.html
grep -c 'href="#measurement-categories"\|id="measurement-categories"' $B/explanation/measurement_classification_system.html
```

Expected: each count ≥ 1, and the navbar link is present on the index page. Also open the
Categories page section in a text dump (`sed -n '/Starting Metrics/,/<\/table>/p'`) and confirm
the `desc` sentence and 18 rows are there.

- [ ] **Step 6: Commit**

```bash
git add docs/source/explanation/measurement_classification_system.md docs/source/tutorials/pages/cli_modes.md src/phenotypic/schema/CLAUDE.md CLAUDE.md
git commit -m "docs: explain measurement categories; document measurements_by_category/

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01D1sMyGvSvvLnJSAM5skaEa"
```

---

## Task 10: Full regression (once)

- [ ] **Step 1:** Invoke the `run-phenotypic-test` skill, then run the full sharded `tests/unit`
  gate as a Slurm array against a **detached worktree at the branch HEAD SHA** (never the live
  checkout), using the committed recipe
  (`docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch` with
  `WORKTREE` pointed at the detached tree). Clean the worktree up from an `afterany` finalizer.
- [ ] **Step 2:** Compare against the latest `main` baseline (13,462 tests / 0 failed at
  `ebb6d7fc`, 2026-09-23). Run each failure in isolation before attributing it; report the
  counts measured, not remembered.
- [ ] **Step 3:** `uv run mypy src/phenotypic/schema src/phenotypic/util src/phenotypic/_cli/_cli_output_manager.py src/phenotypic/_cli/_cli_readme_generator.py`
  and report new errors relative to `main`.
