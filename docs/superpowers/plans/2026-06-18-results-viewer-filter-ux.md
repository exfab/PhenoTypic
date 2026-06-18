# Results Viewer Filter UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate the results-viewer Filters button onto a sticky tab row, give each filter row a Method dropdown (is any of / is none of / range / compare / contains), and sort numeric value options numerically.

**Architecture:** Three independent features over `src/phenotypic/gui/results_viewer/`. The pure data layer (`_filter_state.py`) grows a per-row `method` discriminator and a `to_expr()` that branches into polars predicates; `_output_root.py` gains numeric-aware sorting + an `is_numeric_column` gate; `_filter_panel.py` renders the method-specific controls and syncs them to `STORE_FILTER_SPEC` (via extracted pure helpers, unit-tested per the project's "make callback bodies unit-testable" rule); `_layout.py` + `results_viewer.css` move the Filters button to a sticky tab-bar strip. `dbc.Tabs`/`TABS_ID` and the offcanvas mechanism are untouched.

**Tech Stack:** Python 3.12, polars 1.41, Dash + dash-bootstrap-components, pytest, Playwright (e2e).

## Global Constraints

- **Package manager/runner:** `uv` only — never bare `python`/`pip`. Run tests with `uv run pytest …`.
- **Design tokens:** no hardcoded colors / fonts / z-index-as-magic in Python inline styles — import from `phenotypic.gui._design` (`COLOR_*`, `FONT_*`). CSS uses `var(--…)` tokens injected by `_design.py`; never declare a `:root{--color-*}` block in a tool CSS file.
- **Shared constants:** new shared identifiers go in `_config.py`; output-artifact filenames in `phenotypic.sdk_`. Component ids go in the tool's `_ids.py`.
- **`FEATURES.md` gate:** the `gui-checks` `features-md-gate` job rejects any PR touching `src/phenotypic/gui/` that does not modify `src/phenotypic/gui/FEATURES.md`. Task 10 satisfies this; do not skip it.
- **Unset = skip:** a filter row that is not fully configured is a no-op (never "match nothing"). Rows AND together.
- **Backward compatibility:** legacy `STORE_FILTER_SPEC` rows shaped `{id, column, values}` (no `method`) must keep working — default `method` to `is_any_of`.
- **Commit trailers:** every commit message ends with these two lines:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01WZfxi1tTb2dBuYJtwV3MMt
  ```
  Per-task commit commands below show only the subject; append the trailers.
- **Spec:** `docs/superpowers/specs/2026-06-18-results-viewer-filter-ux-design.md`.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `src/phenotypic/gui/results_viewer/_output_root.py` | numeric-aware value sorting + `is_numeric_column` | 1, 2 |
| `src/phenotypic/gui/results_viewer/_filter_state.py` | extended `FilterRow`/`FilterSpec`, method constants, `to_expr()`, method-aware `apply_to` | 3, 4 |
| `src/phenotypic/gui/results_viewer/_ids.py` | new pattern-matching id-builders for method/range/compare/contains controls | 6 |
| `src/phenotypic/gui/results_viewer/_filter_panel.py` | method dropdown + per-method controls, pure spec-merge helpers, sync callbacks | 5, 6, 7 |
| `src/phenotypic/gui/results_viewer/_filter_offcanvas.py` | `row_is_active` + method-aware `active_filter_count` | 8 |
| `src/phenotypic/gui/results_viewer/_layout.py` | relocate Filters button to sticky tab-bar strip | 9 |
| `src/phenotypic/gui/results_viewer/_assets/results_viewer.css` | sticky nav + actions strip rules | 9 |
| `src/phenotypic/gui/FEATURES.md` | ledger rows for the new affordances | 10 |
| `tests/gui/results_viewer/test_output_root.py` | Feature C + `is_numeric_column` tests | 1, 2 |
| `tests/gui/results_viewer/test_filter_state.py` | method/predicate tests | 3, 4 |
| `tests/unit/gui/results_viewer/test_filter_panel.py` (new) | render + pure spec-merge helper tests | 5, 6, 7 |
| `tests/unit/gui/results_viewer/test_filter_offcanvas.py` | `row_is_active`/`active_filter_count` | 8 |
| `tests/unit/gui/results_viewer/test_navigation_layout.py` | header/tab-bar placement assertions | 9 |
| `tests/e2e/gui/test_filter_offcanvas.py` | live method-swap + range filter + sticky scroll | 11 |

---

## Task 1: Numeric-aware value sorting (Feature C)

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_output_root.py` (add `_all_parse_as_float`; rewrite `_LazyColumnValueSets._compute`)
- Test: `tests/gui/results_viewer/test_output_root.py`

**Interfaces:**
- Produces: `_all_parse_as_float(values: list[str]) -> bool` (module-level, reused by Task 2); numeric-ordered output from `_LazyColumnValueSets._compute`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/gui/results_viewer/test_output_root.py`:

```python
from phenotypic.gui.results_viewer._output_root import _all_parse_as_float


def test_all_parse_as_float_true_for_numeric_strings() -> None:
    assert _all_parse_as_float(["2", "10", "1.5"]) is True


def test_all_parse_as_float_false_for_mixed_or_empty() -> None:
    assert _all_parse_as_float(["2", "x", "10"]) is False
    assert _all_parse_as_float([]) is False


def test_column_value_sets_sorts_numeric_columns_numerically(tmp_path) -> None:
    """An all-numeric metadata column sorts 2 < 10, not lexically '10' < '2'."""
    (tmp_path / "results" / "d1" / "overlays").mkdir(parents=True)
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    df = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"] * 3,
            "Metadata_ImageFile": ["a", "b", "c"],
            "Metadata_Time": ["10", "2", "1"],
        }
    )
    _write_master_parquet(tmp_path, df)
    for stem in ("a", "b", "c"):
        (tmp_path / "results" / "d1" / "overlays" / f"{stem}.png").touch()

    out = OutputRoot.discover(tmp_path)
    assert out.column_value_sets["Metadata_Time"] == ["1", "2", "10"]


def test_column_value_sets_keeps_lexical_for_text_columns(tmp_path) -> None:
    df = _make_minimal_output(tmp_path)  # has Metadata_Strain = s1, s2
    out = OutputRoot.discover(tmp_path)
    assert out.column_value_sets["Metadata_Strain"] == sorted(
        df.get_column("Metadata_Strain").to_list()
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/gui/results_viewer/test_output_root.py -q -k "all_parse or sorts_numeric or keeps_lexical"`
Expected: FAIL — `ImportError: cannot import name '_all_parse_as_float'`.

- [ ] **Step 3: Implement the helper + numeric sort**

In `_output_root.py`, add the helper just above `class _LazyColumnValueSets` (after the `_METADATA_PREFIX = "Metadata_"` line):

```python
def _all_parse_as_float(values: list[str]) -> bool:
    """Return True iff ``values`` is non-empty and every entry parses as float.

    Used to decide whether a column's filter options should be sorted
    numerically (``"2"`` before ``"10"``) and whether a column is
    range/compare-eligible. Empty input returns ``False`` (nothing to
    sort numerically).
    """
    if not values:
        return False
    for value in values:
        try:
            float(value)
        except (TypeError, ValueError):
            return False
    return True
```

Replace the body of `_LazyColumnValueSets._compute`:

```python
    def _compute(self, column: str) -> list[str]:
        values = (
            self._df.get_column(column)
            .cast(pl.String)
            .drop_nulls()
            .unique()
            .to_list()
        )
        if _all_parse_as_float(values):
            return sorted(values, key=float)
        return sorted(values)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/gui/results_viewer/test_output_root.py -q`
Expected: PASS (all, including the pre-existing ones).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_output_root.py tests/gui/results_viewer/test_output_root.py
git commit -m "feat(results-viewer): sort numeric filter options numerically"
```

---

## Task 2: `OutputRoot.is_numeric_column` (Feature B gate)

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_output_root.py` (add method to `OutputRoot`)
- Test: `tests/gui/results_viewer/test_output_root.py`

**Interfaces:**
- Consumes: `_all_parse_as_float` (Task 1).
- Produces: `OutputRoot.is_numeric_column(column: str) -> bool` (used by Task 6 render gating).

- [ ] **Step 1: Write the failing tests**

Append to `tests/gui/results_viewer/test_output_root.py`:

```python
def test_is_numeric_column_true_for_float_measurement(tmp_path) -> None:
    _make_minimal_output(tmp_path)  # Size_Area is Float64
    out = OutputRoot.discover(tmp_path)
    assert out.is_numeric_column("Size_Area") is True


def test_is_numeric_column_true_for_numeric_string_metadata(tmp_path) -> None:
    (tmp_path / "results" / "d1" / "overlays").mkdir(parents=True)
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    df = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1", "d1"],
            "Metadata_ImageFile": ["a", "b"],
            "Metadata_Time": ["6", "24"],
        }
    )
    _write_master_parquet(tmp_path, df)
    for stem in ("a", "b"):
        (tmp_path / "results" / "d1" / "overlays" / f"{stem}.png").touch()
    out = OutputRoot.discover(tmp_path)
    assert out.is_numeric_column("Metadata_Time") is True


def test_is_numeric_column_false_for_text_and_missing(tmp_path) -> None:
    _make_minimal_output(tmp_path)  # Metadata_Strain = s1, s2
    out = OutputRoot.discover(tmp_path)
    assert out.is_numeric_column("Metadata_Strain") is False
    assert out.is_numeric_column("NoSuchColumn") is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/gui/results_viewer/test_output_root.py -q -k is_numeric_column`
Expected: FAIL — `AttributeError: 'OutputRoot' object has no attribute 'is_numeric_column'`.

- [ ] **Step 3: Implement the method**

Add to the `OutputRoot` class in `_output_root.py` (e.g. right after `image_pairs`):

```python
    def is_numeric_column(self, column: str) -> bool:
        """Return ``True`` if ``column`` can be filtered numerically.

        True when the column's polars dtype is numeric (covers every
        ``Size_*`` / ``Shape_*`` / ``Intensity_*`` measurement column for
        free), or when its filter value-set parses entirely as floats
        (covers numeric-valued string metadata like ``Metadata_Time``).
        Unknown columns return ``False``. Drives the Range/Compare gate in
        the filter sidebar.
        """
        if column not in self.master_df.columns:
            return False
        if self.master_df.schema[column].is_numeric():
            return True
        return _all_parse_as_float(self.column_value_sets.get(column, []))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/gui/results_viewer/test_output_root.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_output_root.py tests/gui/results_viewer/test_output_root.py
git commit -m "feat(results-viewer): add OutputRoot.is_numeric_column gate"
```

---

## Task 3: Extend the filter row model (Feature B data model)

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_filter_state.py`
- Test: `tests/gui/results_viewer/test_filter_state.py`

**Interfaces:**
- Produces: method constants `METHOD_IS_ANY_OF`, `METHOD_IS_NONE_OF`, `METHOD_RANGE`, `METHOD_COMPARE`, `METHOD_CONTAINS`; `VALID_METHODS: frozenset[str]`; `COMPARE_OPS: frozenset[str]`; `_coerce_float(v) -> float | None`; extended `FilterRow` dataclass with `from_dict`/`to_dict`; `FilterSpec.from_store`/`to_store` round-trip. Consumed by Tasks 4, 6, 7, 8.

- [ ] **Step 1: Write the failing tests**

Append to `tests/gui/results_viewer/test_filter_state.py`:

```python
from phenotypic.gui.results_viewer._filter_state import (
    COMPARE_OPS,
    METHOD_COMPARE,
    METHOD_CONTAINS,
    METHOD_IS_ANY_OF,
    METHOD_IS_NONE_OF,
    METHOD_RANGE,
    _coerce_float,
)


def test_legacy_row_defaults_to_is_any_of() -> None:
    """A pre-method store row keeps working as an is_any_of list filter."""
    spec = FilterSpec.from_store([{"column": "a", "values": ["x"]}])
    assert spec.rows[0].method == METHOD_IS_ANY_OF
    assert spec.rows[0].values == ["x"]


def test_coerce_float_handles_blanks_and_numbers() -> None:
    assert _coerce_float("") is None
    assert _coerce_float(None) is None
    assert _coerce_float("3.5") == 3.5
    assert _coerce_float(7) == 7.0
    assert _coerce_float("not-a-number") is None


def test_from_store_reads_range_and_compare_and_contains() -> None:
    payload = [
        {"column": "Size_Area", "method": METHOD_RANGE,
         "range_min": "100", "range_max": "5000"},
        {"column": "Shape_Circularity", "method": METHOD_COMPARE,
         "compare_op": ">=", "compare_value": "0.85"},
        {"column": "Metadata_ImageFile", "method": METHOD_CONTAINS,
         "text_pattern": "plate_02", "text_regex": False,
         "text_case_sensitive": True},
    ]
    spec = FilterSpec.from_store(payload)
    assert spec.rows[0].method == METHOD_RANGE
    assert spec.rows[0].range_min == 100.0 and spec.rows[0].range_max == 5000.0
    assert spec.rows[1].method == METHOD_COMPARE
    assert spec.rows[1].compare_op == ">=" and spec.rows[1].compare_value == 0.85
    assert spec.rows[2].method == METHOD_CONTAINS
    assert spec.rows[2].text_pattern == "plate_02"
    assert spec.rows[2].text_case_sensitive is True


def test_invalid_compare_op_coerced_to_none() -> None:
    spec = FilterSpec.from_store(
        [{"column": "a", "method": METHOD_COMPARE, "compare_op": "~=",
          "compare_value": "1"}]
    )
    assert spec.rows[0].compare_op is None


def test_to_store_round_trips_all_methods() -> None:
    original = FilterSpec.from_store(
        [
            {"column": "Size_Area", "method": METHOD_RANGE, "range_min": 1.0,
             "range_max": None},
            {"column": "n", "method": METHOD_IS_NONE_OF, "values": ["1"]},
        ]
    )
    rebuilt = FilterSpec.from_store(original.to_store())
    assert rebuilt.rows[0].method == METHOD_RANGE
    assert rebuilt.rows[0].range_min == 1.0 and rebuilt.rows[0].range_max is None
    assert rebuilt.rows[1].method == METHOD_IS_NONE_OF
    assert rebuilt.rows[1].values == ["1"]


def test_compare_ops_set_is_ordering_only() -> None:
    assert COMPARE_OPS == frozenset({">", ">=", "<", "<="})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/gui/results_viewer/test_filter_state.py -q -k "legacy_row or coerce_float or reads_range or invalid_compare or round_trips_all or ordering_only"`
Expected: FAIL — `ImportError: cannot import name 'METHOD_RANGE'`.

- [ ] **Step 3: Implement the extended model**

Replace the top of `_filter_state.py` from the imports down through the end of the `FilterRow`/`FilterSpec.from_store`/`to_store` definitions. First, after the existing imports, add:

```python
METHOD_IS_ANY_OF = "is_any_of"
METHOD_IS_NONE_OF = "is_none_of"
METHOD_RANGE = "range"
METHOD_COMPARE = "compare"
METHOD_CONTAINS = "contains"

VALID_METHODS: frozenset[str] = frozenset(
    {METHOD_IS_ANY_OF, METHOD_IS_NONE_OF, METHOD_RANGE, METHOD_COMPARE, METHOD_CONTAINS}
)

#: Ordering-only comparison operators. Equality is intentionally excluded —
#: exact float equality is fragile; use list mode for exact match.
COMPARE_OPS: frozenset[str] = frozenset({">", ">=", "<", "<="})


def _coerce_float(value: Any) -> float | None:
    """Best-effort float coercion; blanks / unparseable values become None."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
```

Replace the `FilterRow` dataclass with:

```python
@dataclass
class FilterRow:
    """A single filter clause: one column matched by one ``method``.

    Only the fields relevant to ``method`` are read by :meth:`to_expr`:

    - ``is_any_of`` / ``is_none_of`` → ``values``
    - ``range`` → ``range_min`` / ``range_max`` (either bound optional)
    - ``compare`` → ``compare_op`` (in :data:`COMPARE_OPS`) / ``compare_value``
    - ``contains`` → ``text_pattern`` / ``text_regex`` / ``text_case_sensitive``

    An unset clause (no column, or no usable payload for ``method``) is a
    no-op at apply time — never a "match nothing" sentinel.
    """

    column: str
    method: str = METHOD_IS_ANY_OF
    values: list[str] = field(default_factory=list)
    range_min: float | None = None
    range_max: float | None = None
    compare_op: str | None = None
    compare_value: float | None = None
    text_pattern: str = ""
    text_regex: bool = False
    text_case_sensitive: bool = False

    @classmethod
    def from_dict(cls, entry: dict[str, Any]) -> "FilterRow":
        """Build a row from a (possibly legacy / partial) store dict."""
        method = entry.get("method") or METHOD_IS_ANY_OF
        if method not in VALID_METHODS:
            method = METHOD_IS_ANY_OF
        raw_values = entry.get("values") or []
        if not isinstance(raw_values, list):
            raw_values = []
        compare_op = entry.get("compare_op")
        if compare_op not in COMPARE_OPS:
            compare_op = None
        return cls(
            column=str(entry.get("column", "") or ""),
            method=method,
            values=[str(v) for v in raw_values],
            range_min=_coerce_float(entry.get("range_min")),
            range_max=_coerce_float(entry.get("range_max")),
            compare_op=compare_op,
            compare_value=_coerce_float(entry.get("compare_value")),
            text_pattern=str(entry.get("text_pattern", "") or ""),
            text_regex=bool(entry.get("text_regex", False)),
            text_case_sensitive=bool(entry.get("text_case_sensitive", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat, JSON-store-friendly dict."""
        return {
            "column": self.column,
            "method": self.method,
            "values": list(self.values),
            "range_min": self.range_min,
            "range_max": self.range_max,
            "compare_op": self.compare_op,
            "compare_value": self.compare_value,
            "text_pattern": self.text_pattern,
            "text_regex": self.text_regex,
            "text_case_sensitive": self.text_case_sensitive,
        }
```

Replace `FilterSpec.from_store` and `to_store` bodies:

```python
    @classmethod
    def from_store(cls, payload: list[dict] | None) -> "FilterSpec":
        if not payload:
            return cls(rows=[])
        rows: list[FilterRow] = []
        for entry in payload:
            if not isinstance(entry, dict) or "column" not in entry:
                continue
            rows.append(FilterRow.from_dict(entry))
        return cls(rows=rows)

    def to_store(self) -> list[dict]:
        return [row.to_dict() for row in self.rows]
```

Leave `apply_to` unchanged for now (Task 4 rewrites it). The existing `apply_to` still references `row.column`/`row.values`, which remain valid.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/gui/results_viewer/test_filter_state.py -q`
Expected: PASS — including the pre-existing `is_any_of` tests (the legacy `apply_to` still works on the new dataclass).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_filter_state.py tests/gui/results_viewer/test_filter_state.py
git commit -m "feat(results-viewer): extend FilterRow with method + payloads"
```

---

## Task 4: `to_expr()` + method-aware `apply_to` (Feature B predicates)

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_filter_state.py`
- Test: `tests/gui/results_viewer/test_filter_state.py`

**Interfaces:**
- Consumes: `FilterRow` fields + method constants (Task 3).
- Produces: `FilterRow.to_expr() -> pl.Expr | None`; method-aware `FilterSpec.apply_to`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/gui/results_viewer/test_filter_state.py`:

```python
def _numeric_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Size_Area": [50.0, 150.0, 1000.0, 6000.0],
            "name": ["plate_01", "Plate_02", "ctrl_02", "x"],
            "rep": ["1", "2", "3", "4"],
        }
    )


def test_is_none_of_excludes_listed_values() -> None:
    df = _make_frame()
    spec = FilterSpec.from_store(
        [{"column": "a", "method": METHOD_IS_NONE_OF, "values": ["x"]}]
    )
    out = spec.apply_to(df)
    assert "x" not in out.get_column("a").to_list()
    assert out.height == 2  # y, z


def test_range_between_inclusive_optional_bounds() -> None:
    df = _numeric_frame()
    both = FilterSpec.from_store(
        [{"column": "Size_Area", "method": METHOD_RANGE, "range_min": 100,
          "range_max": 1000}]
    ).apply_to(df)
    assert sorted(both.get_column("Size_Area").to_list()) == [150.0, 1000.0]

    only_min = FilterSpec.from_store(
        [{"column": "Size_Area", "method": METHOD_RANGE, "range_min": 1000,
          "range_max": None}]
    ).apply_to(df)
    assert sorted(only_min.get_column("Size_Area").to_list()) == [1000.0, 6000.0]

    only_max = FilterSpec.from_store(
        [{"column": "Size_Area", "method": METHOD_RANGE, "range_min": None,
          "range_max": 150}]
    ).apply_to(df)
    assert sorted(only_max.get_column("Size_Area").to_list()) == [50.0, 150.0]


def test_range_both_bounds_blank_is_no_op() -> None:
    df = _numeric_frame()
    out = FilterSpec.from_store(
        [{"column": "Size_Area", "method": METHOD_RANGE}]
    ).apply_to(df)
    assert out.height == df.height


def test_compare_operators() -> None:
    df = _numeric_frame()
    gt = FilterSpec.from_store(
        [{"column": "Size_Area", "method": METHOD_COMPARE, "compare_op": ">",
          "compare_value": 150}]
    ).apply_to(df)
    assert sorted(gt.get_column("Size_Area").to_list()) == [1000.0, 6000.0]

    le = FilterSpec.from_store(
        [{"column": "Size_Area", "method": METHOD_COMPARE, "compare_op": "<=",
          "compare_value": 150}]
    ).apply_to(df)
    assert sorted(le.get_column("Size_Area").to_list()) == [50.0, 150.0]


def test_contains_literal_case_sensitive_and_insensitive() -> None:
    df = _numeric_frame()
    cs = FilterSpec.from_store(
        [{"column": "name", "method": METHOD_CONTAINS, "text_pattern": "plate",
          "text_regex": False, "text_case_sensitive": True}]
    ).apply_to(df)
    assert cs.get_column("name").to_list() == ["plate_01"]

    ci = FilterSpec.from_store(
        [{"column": "name", "method": METHOD_CONTAINS, "text_pattern": "plate",
          "text_regex": False, "text_case_sensitive": False}]
    ).apply_to(df)
    assert sorted(ci.get_column("name").to_list()) == ["Plate_02", "plate_01"]


def test_contains_regex() -> None:
    df = _numeric_frame()
    out = FilterSpec.from_store(
        [{"column": "name", "method": METHOD_CONTAINS, "text_pattern": r"_0\d$",
          "text_regex": True, "text_case_sensitive": True}]
    ).apply_to(df)
    assert sorted(out.get_column("name").to_list()) == ["Plate_02", "ctrl_02", "plate_01"]


def test_contains_blank_pattern_is_no_op() -> None:
    df = _numeric_frame()
    out = FilterSpec.from_store(
        [{"column": "name", "method": METHOD_CONTAINS, "text_pattern": "  "}]
    ).apply_to(df)
    assert out.height == df.height


def test_invalid_regex_skips_row_without_raising(caplog) -> None:
    df = _numeric_frame()
    spec = FilterSpec.from_store(
        [{"column": "name", "method": METHOD_CONTAINS, "text_pattern": "(",
          "text_regex": True}]
    )
    out = spec.apply_to(df)  # must not raise
    assert out.height == df.height


def test_range_on_mixed_column_drops_non_numeric() -> None:
    df = pl.DataFrame({"mix": ["10", "2", "x"]})
    out = FilterSpec.from_store(
        [{"column": "mix", "method": METHOD_RANGE, "range_min": 1, "range_max": 100}]
    ).apply_to(df)
    assert sorted(out.get_column("mix").to_list()) == ["10", "2"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/gui/results_viewer/test_filter_state.py -q -k "is_none_of or range_between or both_bounds_blank or compare_operators or contains_ or invalid_regex or mixed_column"`
Expected: FAIL — `AttributeError: 'FilterRow' object has no attribute 'to_expr'`.

- [ ] **Step 3: Implement `to_expr` + rewrite `apply_to`**

At the top of `_filter_state.py`, add `import functools` and `import operator` to the imports (keep `import polars as pl`). Add a `to_expr` method to `FilterRow` (place after `to_dict`):

```python
    def to_expr(self) -> "pl.Expr | None":
        """Return the polars predicate for this row, or None if unset.

        None means "skip this row" (no column, or no usable payload for the
        active method). Numeric methods cast the column with
        ``strict=False`` so non-numeric cells become null and fall out of
        the match without raising.
        """
        if not self.column:
            return None
        method = self.method or METHOD_IS_ANY_OF

        if method in (METHOD_IS_ANY_OF, METHOD_IS_NONE_OF):
            if not self.values:
                return None
            expr = pl.col(self.column).cast(pl.String).is_in(self.values)
            return ~expr if method == METHOD_IS_NONE_OF else expr

        if method == METHOD_RANGE:
            if self.range_min is None and self.range_max is None:
                return None
            numeric = pl.col(self.column).cast(pl.Float64, strict=False)
            bounds: list[pl.Expr] = []
            if self.range_min is not None:
                bounds.append(numeric >= self.range_min)
            if self.range_max is not None:
                bounds.append(numeric <= self.range_max)
            return functools.reduce(operator.and_, bounds)

        if method == METHOD_COMPARE:
            if self.compare_op not in COMPARE_OPS or self.compare_value is None:
                return None
            numeric = pl.col(self.column).cast(pl.Float64, strict=False)
            ops = {
                ">": operator.gt,
                ">=": operator.ge,
                "<": operator.lt,
                "<=": operator.le,
            }
            return ops[self.compare_op](numeric, self.compare_value)

        if method == METHOD_CONTAINS:
            pattern = self.text_pattern or ""
            if not pattern.strip():
                return None
            as_str = pl.col(self.column).cast(pl.String)
            if self.text_regex:
                rx = pattern if self.text_case_sensitive else f"(?i){pattern}"
                return as_str.str.contains(rx, literal=False)
            if self.text_case_sensitive:
                return as_str.str.contains(pattern, literal=True)
            return as_str.str.to_lowercase().str.contains(
                pattern.lower(), literal=True
            )

        return None
```

Replace `FilterSpec.apply_to` with:

```python
    def apply_to(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply every active row as ``AND`` across rows.

        Each row contributes ``FilterRow.to_expr()``; ``None`` (unset) rows
        are skipped. Rows naming a column absent from the frame log a
        warning and skip. A row whose expression fails to evaluate (e.g. an
        invalid user-supplied regex raising ``ComputeError``) is logged and
        skipped rather than crashing the viewer.
        """
        result = df
        for row in self.rows:
            if not row.column:
                continue
            if row.column not in result.columns:
                logger.warning(
                    "Filter column %r is not in the DataFrame; skipping row.",
                    row.column,
                )
                continue
            expr = row.to_expr()
            if expr is None:
                continue
            try:
                result = result.filter(expr)
            except Exception:
                logger.exception(
                    "Filter row failed to evaluate; skipping. column=%r method=%r",
                    row.column,
                    row.method,
                )
                continue
        return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/gui/results_viewer/test_filter_state.py -q`
Expected: PASS (all old + new). The pre-existing `test_string_coercion_uses_polars_default_cast` and missing-column tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_filter_state.py tests/gui/results_viewer/test_filter_state.py
git commit -m "feat(results-viewer): method-aware FilterSpec.apply_to via to_expr"
```

---

## Task 5: Pure spec-merge helpers (Feature B callback logic)

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_filter_panel.py`
- Test: `tests/unit/gui/results_viewer/test_filter_panel.py` (new)

**Interfaces:**
- Consumes: method constants from `_filter_state` (Task 3).
- Produces: pure helpers used by Task 7 callbacks: `_blank_row(column="") -> dict`, `set_row_method(rows, idx, method) -> list[dict]`, `set_row_range(rows, idx, lo, hi) -> list[dict]`, `set_row_compare(rows, idx, op, value) -> list[dict]`, `set_row_text(rows, idx, pattern, regex, case) -> list[dict]`. Also an extended `_normalise_spec` that carries every method field with defaults.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/gui/results_viewer/test_filter_panel.py`:

```python
"""Unit tests for the filter-panel pure helpers (no Dash runtime)."""

from __future__ import annotations

from phenotypic.gui.results_viewer._filter_panel import (
    _blank_row,
    _normalise_spec,
    set_row_compare,
    set_row_method,
    set_row_range,
    set_row_text,
)
from phenotypic.gui.results_viewer._filter_state import (
    METHOD_COMPARE,
    METHOD_IS_ANY_OF,
    METHOD_RANGE,
)


def test_blank_row_has_all_keys_and_defaults() -> None:
    row = _blank_row()
    for key in (
        "id", "column", "method", "values", "range_min", "range_max",
        "compare_op", "compare_value", "text_pattern", "text_regex",
        "text_case_sensitive",
    ):
        assert key in row
    assert row["method"] == METHOD_IS_ANY_OF
    assert row["values"] == []


def test_normalise_spec_backfills_legacy_rows() -> None:
    rows = _normalise_spec([{"column": "a", "values": ["x"]}])
    assert rows[0]["method"] == METHOD_IS_ANY_OF
    assert rows[0]["range_min"] is None
    assert isinstance(rows[0]["id"], str) and rows[0]["id"]


def test_set_row_method_resets_payload() -> None:
    rows = [_blank_row()]
    idx = rows[0]["id"]
    rows[0]["values"] = ["keep-me?"]
    rows[0]["range_min"] = 5.0
    out = set_row_method(rows, idx, METHOD_RANGE)
    assert out[0]["method"] == METHOD_RANGE
    assert out[0]["values"] == []          # payload reset
    assert out[0]["range_min"] is None     # payload reset


def test_set_row_range_writes_bounds() -> None:
    rows = [_blank_row()]
    idx = rows[0]["id"]
    out = set_row_range(rows, idx, 1.0, 9.0)
    assert out[0]["range_min"] == 1.0 and out[0]["range_max"] == 9.0


def test_set_row_compare_writes_op_and_value() -> None:
    rows = [_blank_row()]
    idx = rows[0]["id"]
    out = set_row_compare(rows, idx, ">=", 0.5)
    assert out[0]["method"] == METHOD_COMPARE or out[0]["compare_op"] == ">="
    assert out[0]["compare_op"] == ">=" and out[0]["compare_value"] == 0.5


def test_set_row_text_writes_pattern_and_flags() -> None:
    rows = [_blank_row()]
    idx = rows[0]["id"]
    out = set_row_text(rows, idx, "plate", regex=True, case=False)
    assert out[0]["text_pattern"] == "plate"
    assert out[0]["text_regex"] is True
    assert out[0]["text_case_sensitive"] is False


def test_setters_ignore_unknown_idx() -> None:
    rows = [_blank_row()]
    out = set_row_range(rows, "nope", 1.0, 2.0)
    assert out[0]["range_min"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/gui/results_viewer/test_filter_panel.py -q`
Expected: FAIL — `ImportError: cannot import name '_blank_row'`.

- [ ] **Step 3: Implement the helpers**

In `_filter_panel.py`, add imports near the existing `_filter_state` import:

```python
from phenotypic.gui.results_viewer._filter_state import (
    COMPARE_OPS,
    FilterSpec,
    METHOD_COMPARE,
    METHOD_CONTAINS,
    METHOD_IS_ANY_OF,
    METHOD_IS_NONE_OF,
    METHOD_RANGE,
    VALID_METHODS,
    _coerce_float,
)
```

Add these module-level helpers (e.g. just below `_MAX_PASTE_PREVIEW_CHIPS`):

```python
# Canonical empty payload for every method field, used by _blank_row and
# the per-method setters to reset a row when its method or column changes.
_EMPTY_PAYLOAD: dict[str, Any] = {
    "values": [],
    "range_min": None,
    "range_max": None,
    "compare_op": None,
    "compare_value": None,
    "text_pattern": "",
    "text_regex": False,
    "text_case_sensitive": False,
}


def _blank_row(column: str = "") -> dict[str, Any]:
    """Return a fresh, fully-defaulted row dict with a new uuid id."""
    return {
        "id": uuid.uuid4().hex,
        "column": column,
        "method": METHOD_IS_ANY_OF,
        **{k: (list(v) if isinstance(v, list) else v) for k, v in _EMPTY_PAYLOAD.items()},
    }


def _reset_payload(row: dict[str, Any]) -> None:
    """Clear every method-specific field on ``row`` in place."""
    for key, empty in _EMPTY_PAYLOAD.items():
        row[key] = list(empty) if isinstance(empty, list) else empty


def _find(rows: list[dict[str, Any]], idx: str) -> dict[str, Any] | None:
    return next((r for r in rows if r.get("id") == idx), None)


def set_row_method(
    rows: list[dict[str, Any]], idx: str, method: str
) -> list[dict[str, Any]]:
    """Set a row's method and reset its payload (cached values are stale)."""
    if method not in VALID_METHODS:
        method = METHOD_IS_ANY_OF
    row = _find(rows, idx)
    if row is not None:
        row["method"] = method
        _reset_payload(row)
    return rows


def set_row_range(
    rows: list[dict[str, Any]], idx: str, lo: Any, hi: Any
) -> list[dict[str, Any]]:
    row = _find(rows, idx)
    if row is not None:
        row["range_min"] = _coerce_float(lo)
        row["range_max"] = _coerce_float(hi)
    return rows


def set_row_compare(
    rows: list[dict[str, Any]], idx: str, op: Any, value: Any
) -> list[dict[str, Any]]:
    row = _find(rows, idx)
    if row is not None:
        row["compare_op"] = op if op in COMPARE_OPS else None
        row["compare_value"] = _coerce_float(value)
    return rows


def set_row_text(
    rows: list[dict[str, Any]], idx: str, pattern: Any, *, regex: Any, case: Any
) -> list[dict[str, Any]]:
    row = _find(rows, idx)
    if row is not None:
        row["text_pattern"] = str(pattern or "")
        row["text_regex"] = bool(regex)
        row["text_case_sensitive"] = bool(case)
    return rows
```

Replace `_normalise_spec` with a version that backfills every field:

```python
def _normalise_spec(stored: Any) -> list[dict[str, Any]]:
    """Coerce the store payload into a list of full, defaulted row dicts.

    Each row gains ``id``, ``column``, ``method`` (default is_any_of), and
    every method payload field. Malformed entries are dropped; legacy rows
    (``{column, values}`` with no ``method``) are backfilled.
    """
    if not isinstance(stored, list):
        return []
    rows: list[dict[str, Any]] = []
    for entry in stored:
        if not isinstance(entry, dict):
            continue
        row = _blank_row(str(entry.get("column", "") or ""))
        row["id"] = entry.get("id") or row["id"]
        method = entry.get("method") or METHOD_IS_ANY_OF
        row["method"] = method if method in VALID_METHODS else METHOD_IS_ANY_OF
        raw_values = entry.get("values") or []
        row["values"] = [str(v) for v in raw_values] if isinstance(raw_values, list) else []
        row["range_min"] = _coerce_float(entry.get("range_min"))
        row["range_max"] = _coerce_float(entry.get("range_max"))
        op = entry.get("compare_op")
        row["compare_op"] = op if op in COMPARE_OPS else None
        row["compare_value"] = _coerce_float(entry.get("compare_value"))
        row["text_pattern"] = str(entry.get("text_pattern", "") or "")
        row["text_regex"] = bool(entry.get("text_regex", False))
        row["text_case_sensitive"] = bool(entry.get("text_case_sensitive", False))
        rows.append(row)
    return rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/gui/results_viewer/test_filter_panel.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_filter_panel.py tests/unit/gui/results_viewer/test_filter_panel.py
git commit -m "feat(results-viewer): pure spec-merge helpers for filter methods"
```

---

## Task 6: Method dropdown + per-method controls (Feature B render)

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_ids.py` (new id-builders)
- Modify: `src/phenotypic/gui/results_viewer/_filter_panel.py` (`_render_filter_row`, `_render_filter_rows`)
- Modify: `tests/unit/gui/results_viewer/test_filter_offcanvas.py` (update `_render_filter_row` call to new signature)
- Test: `tests/unit/gui/results_viewer/test_filter_panel.py`

**Interfaces:**
- Consumes: `_blank_row`/`_normalise_spec` (Task 5), `OutputRoot.is_numeric_column` (Task 2), method constants (Task 3).
- Produces: id-builders `filter_row_method_id`, `filter_row_range_min_id`, `filter_row_range_max_id`, `filter_row_compare_op_id`, `filter_row_compare_value_id`, `filter_row_text_pattern_id`, `filter_row_text_regex_id`, `filter_row_text_case_id`; new `_render_filter_row(idx, row, column_options, *, is_numeric)` signature.

- [ ] **Step 1: Add the id-builders**

In `_ids.py`, after `filter_row_remove_id` and before the viewer-cards section, add:

```python
def filter_row_method_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's method dropdown.

    Returns ``{"type": "filter-row-method", "index": idx}``.
    """
    return {"type": "filter-row-method", "index": idx}


def filter_row_range_min_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's range-min numeric input."""
    return {"type": "filter-row-range-min", "index": idx}


def filter_row_range_max_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's range-max numeric input."""
    return {"type": "filter-row-range-max", "index": idx}


def filter_row_compare_op_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's compare-operator dropdown."""
    return {"type": "filter-row-compare-op", "index": idx}


def filter_row_compare_value_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's compare-threshold numeric input."""
    return {"type": "filter-row-compare-value", "index": idx}


def filter_row_text_pattern_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's contains text input."""
    return {"type": "filter-row-text-pattern", "index": idx}


def filter_row_text_regex_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's contains regex checkbox."""
    return {"type": "filter-row-text-regex", "index": idx}


def filter_row_text_case_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's contains case-sensitive checkbox."""
    return {"type": "filter-row-text-case", "index": idx}
```

Add all eight names to the `__all__` list in `_ids.py` (next to the existing `filter_row_*` entries).

- [ ] **Step 2: Write the failing render tests**

Append to `tests/unit/gui/results_viewer/test_filter_panel.py`:

```python
from phenotypic.gui.results_viewer._filter_panel import _render_filter_row
from phenotypic.gui.results_viewer._filter_state import METHOD_CONTAINS, METHOD_RANGE


def _iter(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _iter(child)


def _type_ids(node):
    return {
        c.id["type"]
        for c in _iter(node)
        if isinstance(getattr(c, "id", None), dict) and "type" in c.id
    }


def test_render_row_has_method_dropdown() -> None:
    row = _normalise_spec([{"id": "r1", "column": "Metadata_Strain"}])[0]
    node = _render_filter_row("r1", row, [], is_numeric=False)
    assert "filter-row-method" in _type_ids(node)


def test_range_method_renders_min_max_inputs() -> None:
    row = _normalise_spec(
        [{"id": "r1", "column": "Size_Area", "method": METHOD_RANGE}]
    )[0]
    node = _render_filter_row("r1", row, [], is_numeric=True)
    ids_present = _type_ids(node)
    assert "filter-row-range-min" in ids_present
    assert "filter-row-range-max" in ids_present
    assert "filter-row-values" not in ids_present


def test_contains_method_renders_text_controls() -> None:
    row = _normalise_spec(
        [{"id": "r1", "column": "Metadata_ImageFile", "method": METHOD_CONTAINS}]
    )[0]
    node = _render_filter_row("r1", row, [], is_numeric=False)
    ids_present = _type_ids(node)
    assert "filter-row-text-pattern" in ids_present
    assert "filter-row-text-regex" in ids_present
    assert "filter-row-text-case" in ids_present


def test_method_dropdown_disables_range_compare_for_text_column() -> None:
    row = _normalise_spec([{"id": "r1", "column": "Metadata_Strain"}])[0]
    node = _render_filter_row("r1", row, [], is_numeric=False)
    dropdown = next(
        c for c in _iter(node)
        if isinstance(getattr(c, "id", None), dict)
        and c.id.get("type") == "filter-row-method"
    )
    disabled = {o["value"] for o in dropdown.options if o.get("disabled")}
    assert {"range", "compare"} <= disabled


def test_method_dropdown_enables_range_compare_for_numeric_column() -> None:
    row = _normalise_spec([{"id": "r1", "column": "Size_Area"}])[0]
    node = _render_filter_row("r1", row, [], is_numeric=True)
    dropdown = next(
        c for c in _iter(node)
        if isinstance(getattr(c, "id", None), dict)
        and c.id.get("type") == "filter-row-method"
    )
    disabled = {o["value"] for o in dropdown.options if o.get("disabled")}
    assert "range" not in disabled and "compare" not in disabled
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/gui/results_viewer/test_filter_panel.py -q -k "method_dropdown or range_method or contains_method"`
Expected: FAIL — `_render_filter_row()` got an unexpected keyword / signature mismatch.

- [ ] **Step 4: Rewrite `_render_filter_row` and `_render_filter_rows`**

In `_filter_panel.py`, replace `_render_filter_rows` and `_render_filter_row`. First add a method-label constant near the top:

```python
_METHOD_LABELS: list[tuple[str, str]] = [
    (METHOD_IS_ANY_OF, "Is any of"),
    (METHOD_IS_NONE_OF, "Is none of"),
    (METHOD_RANGE, "Range (between)"),
    (METHOD_COMPARE, "Compare"),
    (METHOD_CONTAINS, "Contains"),
]
_NUMERIC_ONLY_METHODS = {METHOD_RANGE, METHOD_COMPARE}
```

Replace `_render_filter_rows`:

```python
def _render_filter_rows(
    rows: list[dict[str, Any]], df: pl.DataFrame, output_root: OutputRoot
) -> list[Component]:
    """Render the dynamic filter rows for the rows-container."""
    column_options = _column_options(df)
    children: list[Component] = []
    for row in rows:
        column = row["column"]
        is_numeric = bool(column) and output_root.is_numeric_column(column)
        children.append(
            _render_filter_row(row["id"], row, column_options, is_numeric=is_numeric)
        )
    return children
```

Replace `_render_filter_row` (full new body). The column dropdown, remove button, and the list-mode value dropdown + paste popover are preserved; a method dropdown and method-specific controls are added:

```python
def _build_method_dropdown(
    idx: str, method: str, *, is_numeric: bool
) -> Component:
    """Method selector; range/compare options disabled for non-numeric cols."""
    options = [
        {
            "label": label,
            "value": value,
            "disabled": (value in _NUMERIC_ONLY_METHODS and not is_numeric),
        }
        for value, label in _METHOD_LABELS
    ]
    return dcc.Dropdown(
        id=ids.filter_row_method_id(idx),
        options=options,
        value=method or METHOD_IS_ANY_OF,
        clearable=False,
        searchable=False,
        className="mb-2",
    )


def _build_list_controls(idx: str, values: list[str]) -> list[Component]:
    """The shared multi-select + bulk-paste controls (is_any_of / is_none_of)."""
    values_dropdown = dcc.Dropdown(
        id=ids.filter_row_values_id(idx),
        options=[{"label": v, "value": v} for v in values],
        value=values,
        multi=True,
        placeholder="values",
    )
    paste_button = dbc.Button(
        "Paste",
        id=ids.filter_row_paste_btn_id(idx),
        color="secondary",
        outline=True,
        size="sm",
        n_clicks=0,
    )
    paste_popover = dbc.Popover(
        dbc.PopoverBody(
            [
                html.Div(
                    "Paste values separated by newline, comma, or tab.",
                    className="text-muted small mb-2",
                ),
                dbc.Textarea(
                    id=ids.filter_row_paste_textarea_id(idx),
                    placeholder="value1\nvalue2\nvalue3",
                    style={"width": "100%", "minHeight": "8rem"},
                ),
                dbc.Button(
                    "Apply",
                    id=ids.filter_row_paste_apply_id(idx),
                    color="primary",
                    size="sm",
                    className="mt-2",
                    n_clicks=0,
                ),
                html.Div(
                    id=ids.filter_row_paste_chips_id(idx),
                    className="mt-2",
                    children=_render_paste_chips([], []),
                ),
            ]
        ),
        id=ids.filter_row_paste_popover_id(idx),
        target=ids.filter_row_paste_btn_id(idx),
        is_open=False,
        placement="left",
        trigger=None,
        style={"minWidth": "20rem", "maxWidth": "28rem"},
    )
    return [
        html.Div(values_dropdown, className="mb-2"),
        html.Div(paste_button, className="d-flex gap-1"),
        paste_popover,
    ]


def _build_range_controls(idx: str, row: dict[str, Any]) -> list[Component]:
    return [
        html.Div(
            [
                dcc.Input(
                    id=ids.filter_row_range_min_id(idx),
                    type="number",
                    value=row["range_min"],
                    placeholder="min",
                    className="form-control form-control-sm",
                    style={"width": "45%"},
                ),
                html.Span("–", className="mx-1"),
                dcc.Input(
                    id=ids.filter_row_range_max_id(idx),
                    type="number",
                    value=row["range_max"],
                    placeholder="max",
                    className="form-control form-control-sm",
                    style={"width": "45%"},
                ),
            ],
            className="d-flex align-items-center mb-2",
        )
    ]


def _build_compare_controls(idx: str, row: dict[str, Any]) -> list[Component]:
    return [
        html.Div(
            [
                dcc.Dropdown(
                    id=ids.filter_row_compare_op_id(idx),
                    options=[{"label": op, "value": op} for op in (">", ">=", "<", "<=")],
                    value=row["compare_op"],
                    clearable=False,
                    searchable=False,
                    placeholder="op",
                    style={"width": "40%"},
                ),
                dcc.Input(
                    id=ids.filter_row_compare_value_id(idx),
                    type="number",
                    value=row["compare_value"],
                    placeholder="value",
                    className="form-control form-control-sm ms-1",
                    style={"width": "55%"},
                ),
            ],
            className="d-flex align-items-center mb-2",
        )
    ]


def _build_contains_controls(idx: str, row: dict[str, Any]) -> list[Component]:
    return [
        dbc.Input(
            id=ids.filter_row_text_pattern_id(idx),
            type="text",
            value=row["text_pattern"],
            placeholder="contains…",
            size="sm",
            className="mb-2",
        ),
        html.Div(
            [
                dbc.Checkbox(
                    id=ids.filter_row_text_regex_id(idx),
                    label="regex",
                    value=row["text_regex"],
                    className="me-3",
                ),
                dbc.Checkbox(
                    id=ids.filter_row_text_case_id(idx),
                    label="case-sensitive",
                    value=row["text_case_sensitive"],
                ),
            ],
            className="d-flex small mb-2",
        ),
    ]


def _render_filter_row(
    idx: str,
    row: dict[str, Any],
    column_options: list[dict[str, str]],
    *,
    is_numeric: bool,
) -> Component:
    """Build a single filter-row component tree for the row's active method."""
    column = row["column"]
    method = row["method"]

    column_dropdown = dcc.Dropdown(
        id=ids.filter_row_column_id(idx),
        options=column_options,
        value=column or None,
        searchable=True,
        clearable=False,
        placeholder="column",
        className="mb-2",
    )
    method_dropdown = _build_method_dropdown(idx, method, is_numeric=is_numeric)

    if method == METHOD_RANGE:
        method_controls = _build_range_controls(idx, row)
    elif method == METHOD_COMPARE:
        method_controls = _build_compare_controls(idx, row)
    elif method == METHOD_CONTAINS:
        method_controls = _build_contains_controls(idx, row)
    else:  # is_any_of / is_none_of
        method_controls = _build_list_controls(idx, row["values"])

    remove_button = dbc.Button(
        "✕",
        id=ids.filter_row_remove_id(idx),
        color="danger",
        outline=True,
        size="sm",
        n_clicks=0,
        title="Remove this filter",
    )

    return html.Div(
        [
            html.Div(column_dropdown),
            method_dropdown,
            *method_controls,
            html.Div(remove_button, className="d-flex justify-content-end"),
        ],
        id=ids.filter_row_id(idx),
        className="filter-row mb-2",
        style={
            "borderLeft": f"2px solid {COLOR_BLUE}",
            "paddingLeft": "0.5rem",
            "paddingTop": "0.25rem",
            "paddingBottom": "0.25rem",
        },
    )
```

- [ ] **Step 5: Update the `_render_rows` callback call site**

In `register_callbacks`, the `_render_rows` callback currently calls `_render_filter_rows(rows, df)`. Change it to pass `output_root`:

```python
    @app.callback(
        Output(ids.FILTER_ROWS_CONTAINER_ID, "children"),
        Input(ids.STORE_FILTER_SPEC, "data"),
    )
    def _render_rows(stored: Any) -> list[Component]:
        """Render one component tree per row in the spec store."""
        rows = _normalise_spec(stored)
        return _render_filter_rows(rows, df, output_root)
```

- [ ] **Step 6: Update the existing offcanvas test for the new signature**

In `tests/unit/gui/results_viewer/test_filter_offcanvas.py`, replace the body of `test_bulk_paste_popover_opens_left`:

```python
def test_bulk_paste_popover_opens_left() -> None:
    """The per-row bulk-paste popover opens leftward so it stays on-screen
    inside the right-docked offcanvas."""
    from phenotypic.gui.results_viewer._filter_panel import (
        _normalise_spec,
        _render_filter_row,
    )

    row = _normalise_spec([{"id": "idx1", "column": "Metadata_Dataset",
                            "values": ["WT"]}])[0]
    node = _render_filter_row("idx1", row, [], is_numeric=False)
    popovers = [
        n for n in _iter_components(node) if getattr(n, "_type", None) == "Popover"
    ]
    assert popovers, "expected a bulk-paste popover in the rendered row"
    assert all(getattr(p, "placement", None) == "left" for p in popovers)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/gui/results_viewer/test_filter_panel.py tests/unit/gui/results_viewer/test_filter_offcanvas.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_ids.py src/phenotypic/gui/results_viewer/_filter_panel.py tests/unit/gui/results_viewer/test_filter_panel.py tests/unit/gui/results_viewer/test_filter_offcanvas.py
git commit -m "feat(results-viewer): method dropdown + per-method filter controls"
```

---

## Task 7: Per-method sync callbacks (Feature B wiring)

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_filter_panel.py` (`register_callbacks`)
- Test: `tests/unit/gui/results_viewer/test_filter_panel.py`

**Interfaces:**
- Consumes: pure setters (Task 5), new ids (Task 6).
- Produces: callbacks writing method/range/compare/text changes into `STORE_FILTER_SPEC`; an extended column-change reset.

- [ ] **Step 1: Write the failing registration test**

Append to `tests/unit/gui/results_viewer/test_filter_panel.py`:

```python
def test_register_callbacks_wires_method_controls(tmp_path) -> None:
    """register_callbacks adds the per-method sync callbacks."""
    import dash
    import polars as pl

    from phenotypic.gui.results_viewer._curation_labels import CurationLabels
    from phenotypic.gui.results_viewer._output_root import OutputRoot
    from phenotypic.gui.results_viewer import _filter_panel

    (tmp_path / "results" / "d1" / "overlays").mkdir(parents=True)
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    df = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1", "d1"],
            "Metadata_ImageFile": ["a", "b"],
            "Size_Area": [1.0, 2.0],
        }
    )
    from phenotypic.sdk_ import master_measurements_parquet_path

    target = master_measurements_parquet_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(target)
    for stem in ("a", "b"):
        (tmp_path / "results" / "d1" / "overlays" / f"{stem}.png").touch()

    output_root = OutputRoot.discover(tmp_path)
    state = CurationLabels.load(output_root.root, output_root.clean_master_df)
    app = dash.Dash(__name__)
    _filter_panel.register_callbacks(app, output_root, state)

    keys = " ".join(app.callback_map.keys())
    assert "filter-row-method" in keys
    assert "filter-row-range-min" in keys
    assert "filter-row-compare-op" in keys
    assert "filter-row-text-pattern" in keys
```

> Note: confirm the `CurationLabels` constructor/loader signature in
> `_curation_labels.py` while implementing; if `CurationLabels.load` differs,
> adapt this fixture to the real API (the assertion on `callback_map` is the
> point). Reuse the pattern already used in `test_curation_labels.py`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/gui/results_viewer/test_filter_panel.py -q -k wires_method_controls`
Expected: FAIL — no callback contains `filter-row-method`.

- [ ] **Step 3: Add the sync callbacks**

In `register_callbacks` in `_filter_panel.py`, add these callbacks (after the existing `_update_values` callback, before `_populate_value_options`). They mirror the existing `_update_columns` structure:

```python
    # --- Method dropdown → spec (resets payload) -------------------------

    @app.callback(
        Output(ids.STORE_FILTER_SPEC, "data", allow_duplicate=True),
        Input({"type": "filter-row-method", "index": ALL}, "value"),
        State({"type": "filter-row-method", "index": ALL}, "id"),
        State(ids.STORE_FILTER_SPEC, "data"),
        prevent_initial_call=True,
    )
    def _update_methods(
        methods: list[Any],
        component_ids: list[dict[str, str]],
        stored: Any,
    ) -> Any:
        rows = _normalise_spec(stored)
        changed = False
        for comp_id, method in zip(component_ids, methods, strict=False):
            idx = comp_id["index"]
            current = _find(rows, idx)
            new_method = method or METHOD_IS_ANY_OF
            if current is not None and current["method"] != new_method:
                set_row_method(rows, idx, new_method)
                changed = True
        return rows if changed else no_update

    # --- Range inputs → spec ---------------------------------------------

    @app.callback(
        Output(ids.STORE_FILTER_SPEC, "data", allow_duplicate=True),
        Input({"type": "filter-row-range-min", "index": ALL}, "value"),
        Input({"type": "filter-row-range-max", "index": ALL}, "value"),
        State({"type": "filter-row-range-min", "index": ALL}, "id"),
        State(ids.STORE_FILTER_SPEC, "data"),
        prevent_initial_call=True,
    )
    def _update_ranges(
        mins: list[Any],
        maxes: list[Any],
        component_ids: list[dict[str, str]],
        stored: Any,
    ) -> Any:
        rows = _normalise_spec(stored)
        changed = False
        for comp_id, lo, hi in zip(component_ids, mins, maxes, strict=False):
            idx = comp_id["index"]
            row = _find(rows, idx)
            if row is None:
                continue
            new_lo, new_hi = _coerce_float(lo), _coerce_float(hi)
            if row["range_min"] != new_lo or row["range_max"] != new_hi:
                set_row_range(rows, idx, lo, hi)
                changed = True
        return rows if changed else no_update

    # --- Compare op/value → spec -----------------------------------------

    @app.callback(
        Output(ids.STORE_FILTER_SPEC, "data", allow_duplicate=True),
        Input({"type": "filter-row-compare-op", "index": ALL}, "value"),
        Input({"type": "filter-row-compare-value", "index": ALL}, "value"),
        State({"type": "filter-row-compare-op", "index": ALL}, "id"),
        State(ids.STORE_FILTER_SPEC, "data"),
        prevent_initial_call=True,
    )
    def _update_compares(
        ops: list[Any],
        vals: list[Any],
        component_ids: list[dict[str, str]],
        stored: Any,
    ) -> Any:
        rows = _normalise_spec(stored)
        changed = False
        for comp_id, op, value in zip(component_ids, ops, vals, strict=False):
            idx = comp_id["index"]
            row = _find(rows, idx)
            if row is None:
                continue
            new_op = op if op in COMPARE_OPS else None
            new_val = _coerce_float(value)
            if row["compare_op"] != new_op or row["compare_value"] != new_val:
                set_row_compare(rows, idx, op, value)
                changed = True
        return rows if changed else no_update

    # --- Contains text/flags → spec --------------------------------------

    @app.callback(
        Output(ids.STORE_FILTER_SPEC, "data", allow_duplicate=True),
        Input({"type": "filter-row-text-pattern", "index": ALL}, "value"),
        Input({"type": "filter-row-text-regex", "index": ALL}, "value"),
        Input({"type": "filter-row-text-case", "index": ALL}, "value"),
        State({"type": "filter-row-text-pattern", "index": ALL}, "id"),
        State(ids.STORE_FILTER_SPEC, "data"),
        prevent_initial_call=True,
    )
    def _update_texts(
        patterns: list[Any],
        regexes: list[Any],
        cases: list[Any],
        component_ids: list[dict[str, str]],
        stored: Any,
    ) -> Any:
        rows = _normalise_spec(stored)
        changed = False
        for comp_id, pattern, regex, case in zip(
            component_ids, patterns, regexes, cases, strict=False
        ):
            idx = comp_id["index"]
            row = _find(rows, idx)
            if row is None:
                continue
            new_pat = str(pattern or "")
            if (
                row["text_pattern"] != new_pat
                or row["text_regex"] != bool(regex)
                or row["text_case_sensitive"] != bool(case)
            ):
                set_row_text(rows, idx, pattern, regex=regex, case=case)
                changed = True
        return rows if changed else no_update
```

Then update the existing `_update_columns` callback so a column change also resets every method payload, and downgrades range/compare to `is_any_of` if the new column is non-numeric. Replace its loop body:

```python
        changed = False
        for row in rows:
            new_column = new_by_id.get(row["id"], row["column"])
            if new_column != row["column"]:
                row["column"] = new_column
                _reset_payload(row)
                if row["method"] in _NUMERIC_ONLY_METHODS and not (
                    new_column and output_root.is_numeric_column(new_column)
                ):
                    row["method"] = METHOD_IS_ANY_OF
                changed = True
        if not changed:
            return no_update
        return rows
```

(`output_root` is already in scope in `register_callbacks`.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/gui/results_viewer/test_filter_panel.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full filter-related unit suite (regression)**

Run: `uv run pytest tests/gui/results_viewer/test_filter_state.py tests/gui/results_viewer/test_output_root.py tests/unit/gui/results_viewer/test_filter_panel.py tests/unit/gui/results_viewer/test_filter_offcanvas.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_filter_panel.py tests/unit/gui/results_viewer/test_filter_panel.py
git commit -m "feat(results-viewer): sync method/range/compare/contains to spec store"
```

---

## Task 8: Method-aware active-filter count

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_filter_offcanvas.py`
- Test: `tests/unit/gui/results_viewer/test_filter_offcanvas.py`

**Interfaces:**
- Consumes: method constants + `COMPARE_OPS` (Task 3).
- Produces: `row_is_active(row: dict) -> bool`; `active_filter_count` counts only rows with a usable constraint.

- [ ] **Step 1: Write the failing tests**

In `tests/unit/gui/results_viewer/test_filter_offcanvas.py`, add to `class TestActiveFilterCount` and a new class:

```python
    def test_counts_configured_range_and_contains_rows(self) -> None:
        spec = [
            {"id": "a", "column": "Size_Area", "method": "range",
             "range_min": 100, "range_max": None},
            {"id": "b", "column": "Metadata_ImageFile", "method": "contains",
             "text_pattern": "plate"},
            {"id": "c", "column": "Size_Area", "method": "range"},  # unset → 0
            {"id": "d", "column": "", "method": "is_any_of", "values": ["x"]},
        ]
        assert active_filter_count(spec) == 2


class TestRowIsActive:
    def test_list_methods_need_values(self) -> None:
        from phenotypic.gui.results_viewer._filter_offcanvas import row_is_active

        assert row_is_active({"column": "a", "method": "is_any_of", "values": ["x"]})
        assert not row_is_active({"column": "a", "method": "is_any_of", "values": []})
        assert row_is_active({"column": "a", "method": "is_none_of", "values": ["x"]})

    def test_range_needs_a_bound(self) -> None:
        from phenotypic.gui.results_viewer._filter_offcanvas import row_is_active

        assert row_is_active({"column": "a", "method": "range", "range_min": 1})
        assert row_is_active({"column": "a", "method": "range", "range_max": 9})
        assert not row_is_active({"column": "a", "method": "range"})

    def test_compare_needs_op_and_value(self) -> None:
        from phenotypic.gui.results_viewer._filter_offcanvas import row_is_active

        assert row_is_active(
            {"column": "a", "method": "compare", "compare_op": ">", "compare_value": 1}
        )
        assert not row_is_active(
            {"column": "a", "method": "compare", "compare_op": "~", "compare_value": 1}
        )
        assert not row_is_active(
            {"column": "a", "method": "compare", "compare_op": ">"}
        )

    def test_contains_needs_nonblank_pattern(self) -> None:
        from phenotypic.gui.results_viewer._filter_offcanvas import row_is_active

        assert row_is_active({"column": "a", "method": "contains", "text_pattern": "x"})
        assert not row_is_active(
            {"column": "a", "method": "contains", "text_pattern": "  "}
        )

    def test_no_column_is_inactive(self) -> None:
        from phenotypic.gui.results_viewer._filter_offcanvas import row_is_active

        assert not row_is_active({"column": "", "method": "is_any_of", "values": ["x"]})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/gui/results_viewer/test_filter_offcanvas.py -q -k "configured_range or RowIsActive"`
Expected: FAIL — `ImportError: cannot import name 'row_is_active'`.

- [ ] **Step 3: Implement `row_is_active` + rewrite `active_filter_count`**

In `_filter_offcanvas.py`, add the import and helper. Add near the top:

```python
from phenotypic.gui.results_viewer._filter_state import (
    COMPARE_OPS,
    METHOD_COMPARE,
    METHOD_CONTAINS,
    METHOD_IS_ANY_OF,
    METHOD_IS_NONE_OF,
    METHOD_RANGE,
    _coerce_float,
)
```

Add `row_is_active` and rewrite `active_filter_count`:

```python
def row_is_active(row: Any) -> bool:
    """Return True if a spec row contributes a real constraint.

    Mirrors the "unset = skip" rule in
    :meth:`phenotypic.gui.results_viewer._filter_state.FilterRow.to_expr`:
    a row needs a column AND a usable payload for its method.
    """
    if not isinstance(row, dict):
        return False
    if not str(row.get("column", "") or "").strip():
        return False
    method = row.get("method") or METHOD_IS_ANY_OF
    if method in (METHOD_IS_ANY_OF, METHOD_IS_NONE_OF):
        values = row.get("values") or []
        return isinstance(values, list) and len(values) > 0
    if method == METHOD_RANGE:
        return (
            _coerce_float(row.get("range_min")) is not None
            or _coerce_float(row.get("range_max")) is not None
        )
    if method == METHOD_COMPARE:
        return (
            row.get("compare_op") in COMPARE_OPS
            and _coerce_float(row.get("compare_value")) is not None
        )
    if method == METHOD_CONTAINS:
        return bool(str(row.get("text_pattern", "") or "").strip())
    return False


def active_filter_count(spec: Any) -> int:
    """Count rows that contribute a real constraint (see :func:`row_is_active`)."""
    if not isinstance(spec, list):
        return 0
    return sum(1 for row in spec if row_is_active(row))
```

Add `"row_is_active"` to `__all__`.

> Note: `test_counts_only_rows_with_a_column` (existing) expects count `2`
> for `[{col+values}, {empty}, {col, values=[]}]`. Under the new rule the
> third row (column set, empty values) is **inactive**, so the existing test
> would now expect `1`. Update that existing test's assertion to `== 1` and its
> docstring to "counts only rows with a usable constraint".

- [ ] **Step 4: Update the existing test assertion**

In `test_counts_only_rows_with_a_column`, change `assert active_filter_count(spec) == 2` to `assert active_filter_count(spec) == 1` and rename the method to `test_counts_only_rows_with_a_usable_constraint`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/gui/results_viewer/test_filter_offcanvas.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_filter_offcanvas.py tests/unit/gui/results_viewer/test_filter_offcanvas.py
git commit -m "feat(results-viewer): count active filters across all methods"
```

---

## Task 9: Relocate Filters button to a sticky tab row (Feature A)

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_layout.py` (`_build_header`, `build_app_layout`)
- Modify: `src/phenotypic/gui/results_viewer/_assets/results_viewer.css`
- Test: `tests/unit/gui/results_viewer/test_navigation_layout.py`

**Interfaces:**
- Consumes: `ids.BTN_FILTERS_TOGGLE`, `ids.FILTER_TOGGLE_BADGE_ID`, `ids.TABS_ID` (unchanged).
- Produces: a `_build_filters_toggle()` helper returning the button; a `results-viewer-tabbar` wrapper around the tabs with the button in a sticky actions strip.

- [ ] **Step 1: Write the failing tests**

Create or append to `tests/unit/gui/results_viewer/test_navigation_layout.py`:

```python
"""Layout placement tests for the sticky tab-row Filters button."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._layout import _build_header, build_app_layout
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.sdk_ import master_measurements_parquet_path


def _iter(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _iter(child)


def _ids_in(component) -> set:
    return {getattr(c, "id", None) for c in _iter(component)}


def _make_output(tmp_path: Path) -> OutputRoot:
    (tmp_path / "results" / "d1" / "overlays").mkdir(parents=True)
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    df = pl.DataFrame(
        {"Metadata_Dataset": ["d1"], "Metadata_ImageFile": ["a"], "Size_Area": [1.0]}
    )
    target = master_measurements_parquet_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(target)
    (tmp_path / "results" / "d1" / "overlays" / "a.png").touch()
    return OutputRoot.discover(tmp_path)


def test_header_no_longer_contains_filters_toggle(tmp_path) -> None:
    out = _make_output(tmp_path)
    header = _build_header(out)
    assert ids.BTN_FILTERS_TOGGLE not in _ids_in(header)


def test_app_layout_keeps_filters_toggle_near_tabs(tmp_path) -> None:
    out = _make_output(tmp_path)
    state = CurationLabels.load(out.root, out.clean_master_df)
    layout = build_app_layout(out, state)
    all_ids = _ids_in(layout)
    assert ids.BTN_FILTERS_TOGGLE in all_ids
    assert ids.FILTER_TOGGLE_BADGE_ID in all_ids
    assert ids.TABS_ID in all_ids
```

> Note: confirm `CurationLabels.load(root, clean_master_df)` matches the real
> signature in `_curation_labels.py` (see `test_curation_labels.py`); adapt the
> fixture if needed.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/gui/results_viewer/test_navigation_layout.py -q -k "filters_toggle or near_tabs"`
Expected: FAIL — header still contains `BTN_FILTERS_TOGGLE`.

- [ ] **Step 3: Move the button out of the header**

In `_layout.py`, extract the button into a helper and remove it from `_build_header`. Add this helper above `_build_header`:

```python
def _build_filters_toggle() -> Component:
    """The Filters offcanvas toggle (with active-filter count badge).

    Rendered into the sticky tab-bar actions strip by
    :func:`build_app_layout` (not the header), so it rides on the tab row
    and stays pinned while tab content scrolls.
    """
    return dbc.Button(
        [
            "Filters",
            dbc.Badge(
                "",
                id=ids.FILTER_TOGGLE_BADGE_ID,
                color="primary",
                className="ms-2",
                style={"display": "none"},
            ),
        ],
        id=ids.BTN_FILTERS_TOGGLE,
        color="secondary",
        outline=True,
        size="sm",
        n_clicks=0,
    )
```

In `_build_header`, delete the local `filters_toggle = dbc.Button(...)` block and remove `filters_toggle` from the `top_row` children list (leave the spacer and `lock_switch`).

- [ ] **Step 4: Wrap the tabs in a sticky tab-bar strip**

In `build_app_layout`, replace the `body = html.Div(tabs, …)` block with a wrapper that overlays the Filters button on the sticky nav row:

```python
    tabbar = html.Div(
        [
            html.Div(
                _build_filters_toggle(),
                className="results-viewer-tabbar__actions",
            ),
            tabs,
        ],
        className="results-viewer-tabbar",
    )

    body = html.Div(
        tabbar,
        className="results-viewer-body",
        style={
            "background": _BG,
            "minHeight": "calc(100vh - 7rem)",
        },
    )
```

- [ ] **Step 5: Add the sticky CSS**

Append to `src/phenotypic/gui/results_viewer/_assets/results_viewer.css`:

```css
/* --- Sticky tab bar with right-aligned Filters button ------------------ */
/* The dbc.Tabs nav row sticks to the top of the scroll container while the
   tab panes scroll beneath it. The Filters button rides on that row via a
   zero-height sticky actions strip layered above the nav (higher z-index),
   so it never consumes vertical layout space and stays pinned on scroll. */
.results-viewer-tabbar {
    position: relative;
}

.results-viewer-tabbar .nav-tabs {
    position: sticky;
    top: 0;
    z-index: 1020;
    background: var(--color-bg);
    /* reserve room so long tab lists never slide under the button */
    padding-right: 7rem;
}

.results-viewer-tabbar__actions {
    position: sticky;
    top: 0;
    z-index: 1030;
    height: 0;
    display: flex;
    justify-content: flex-end;
    pointer-events: none;
}

.results-viewer-tabbar__actions > * {
    pointer-events: auto;
    margin: 0.25rem 0.5rem 0 0;
}
```

> `--color-bg` is injected by `_design.py`; do not redefine it. The
> offcanvas (`position: fixed`, bootstrap z-index ≈1045) stays above both
> sticky layers.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/gui/results_viewer/test_navigation_layout.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_layout.py src/phenotypic/gui/results_viewer/_assets/results_viewer.css tests/unit/gui/results_viewer/test_navigation_layout.py
git commit -m "feat(results-viewer): sticky tab-row Filters button"
```

---

## Task 10: Update FEATURES.md ledger + screenshots

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md`
- Modify: any refreshed PNGs under `docs/source/tutorials/gui/` (from screenshot capture)

**Interfaces:** none (documentation gate).

- [ ] **Step 1: Inspect the current filter rows in FEATURES.md**

Run: `grep -n -iE "filter|Filters toggle|filter-row" src/phenotypic/gui/FEATURES.md`
Read the surrounding table to match the exact column format (id, description, status, Test ref).

- [ ] **Step 2: Edit FEATURES.md**

Update the existing "Filters toggle" row to note it now lives on the sticky
tab row (right-aligned), and add one row per new affordance, following the
file's existing table columns. Use these Test refs:

- Filters toggle (sticky tab row) → `tests/unit/gui/results_viewer/test_navigation_layout.py`
- Filter row Method dropdown → `tests/unit/gui/results_viewer/test_filter_panel.py`
- Filter method: Is none of / Range / Compare / Contains → `tests/gui/results_viewer/test_filter_state.py`
- Numeric-aware filter option sort → `tests/gui/results_viewer/test_output_root.py`

Match the exact `✅ shipping` / status token the surrounding rows use (the
pre-commit `Test ref` validator only checks shipping rows).

- [ ] **Step 3: Regenerate tutorial screenshots**

Run: `uv run python scripts/capture_gui_tutorial_screenshots.py`
Expected: regenerates the full PNG set (unrelated tutorials shift a few bytes — this is expected; commit them all, do not cherry-pick).

- [ ] **Step 4: Verify the FEATURES.md gate locally**

Run: `uv run python scripts/check_workflows_md.py` (if present) and re-read the
diff to confirm the table is well-formed.
Expected: no errors. (The `features-md-gate` itself runs in CI; this is a
sanity check.)

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/FEATURES.md docs/source/tutorials/gui
git commit -m "docs(results-viewer): FEATURES.md rows + screenshots for filter UX"
```

---

## Task 11: Live Playwright verification (Feature A + B end-to-end)

**Files:**
- Modify: `tests/e2e/gui/test_filter_offcanvas.py`
- Test: same file.

**Interfaces:** none — drives the live viewer.

- [ ] **Step 1: Read the existing e2e harness**

Run: `sed -n '1,80p' tests/e2e/gui/test_filter_offcanvas.py` and review
`tests/e2e/gui/conftest.py` for the `live_server` / fixture that boots the
results viewer against a fake output root. Match those fixtures exactly.

- [ ] **Step 2: Add the e2e test (open offcanvas, switch to Range, filter)**

Append a test that mirrors the file's existing fixtures. Skeleton (adapt
selectors/fixtures to the real harness — do not invent fixture names):

```python
def test_range_method_filters_picker(page, results_viewer_server):
    page.goto(results_viewer_server.url)
    # open the filter offcanvas from the sticky tab-row button
    page.get_by_role("button", name="Filters").click()
    page.get_by_role("button", name="+ Add filter").click()
    # choose a numeric column, switch method to Range, type a bound
    # (use the dcc.Dropdown + dcc.Input ids from _ids.py via CSS/text selectors)
    # then assert the "N images match" chip text changes.
    ...
```

- [ ] **Step 3: Add a sticky-scroll assertion**

Add a check that the Filters button remains in the viewport after scrolling
the tab content (e.g. `page.mouse.wheel(0, 2000)` then assert the button is
still visible via `expect(page.get_by_role("button", name="Filters")).to_be_visible()`).

- [ ] **Step 4: Run the e2e test locally**

Run: `PLAYWRIGHT=1 uv run pytest tests/e2e/gui/test_filter_offcanvas.py -q`
Expected: PASS locally. If it passes locally but is timing-sensitive on CI,
apply the `ci_flaky` marker per `tests/CLAUDE.md` only after the documented
re-validation workflow.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/gui/test_filter_offcanvas.py
git commit -m "test(results-viewer): e2e range filter + sticky tab-bar"
```

---

## Final verification

- [ ] **Run the full results-viewer unit + integration suite**

Run:
```bash
uv run pytest tests/gui/results_viewer tests/unit/gui/results_viewer tests/integration/gui/test_filter_offcanvas_layout.py -q
```
Expected: PASS.

- [ ] **Type-check + lint the touched modules**

Run:
```bash
uv run mypy src/phenotypic/gui/results_viewer/_filter_state.py src/phenotypic/gui/results_viewer/_filter_panel.py src/phenotypic/gui/results_viewer/_output_root.py src/phenotypic/gui/results_viewer/_filter_offcanvas.py src/phenotypic/gui/results_viewer/_layout.py src/phenotypic/gui/results_viewer/_ids.py
uv run ruff check --fix src/phenotypic/gui/results_viewer
```
Expected: no errors.

- [ ] **Run a code-review + simplify pass** (per project convention: simplifier after implementation, then regression test). Apply fixes, then re-run the suite above.

---

## Self-Review (plan author)

**Spec coverage:**
- Feature A (sticky tab-row button) → Task 9 (+ live check Task 11).
- Feature B methods (is any of / none of / range / compare / contains) → Tasks 3, 4 (logic), 6 (controls), 7 (wiring); numeric gating → Task 2 + Task 6/7; backward compat → Task 3/5; active-count → Task 8.
- Feature C (numeric sort) → Task 1.
- Testing strategy (unit + live) → Tasks 1–8 unit, Task 11 e2e.
- `FEATURES.md` gate → Task 10.
- All covered.

**Placeholder scan:** Task 11's e2e selectors are intentionally adaptive (the real `conftest` fixtures must be matched, not invented) and flagged as such; every pure-logic task carries complete code. No `TBD`/`add error handling`/`similar to Task N`.

**Type consistency:** method constants (`METHOD_*`), `COMPARE_OPS`, `_coerce_float`, `_blank_row`, `set_row_*`, `is_numeric_column`, `_render_filter_row(idx, row, column_options, *, is_numeric)`, `row_is_active` are defined once and referenced with the same signatures across tasks. Store row keys (`range_min`, `range_max`, `compare_op`, `compare_value`, `text_pattern`, `text_regex`, `text_case_sensitive`) are spelled identically in `_filter_state`, `_filter_panel`, and `_filter_offcanvas`.
