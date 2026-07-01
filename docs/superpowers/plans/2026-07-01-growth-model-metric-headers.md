# Growth-Model Metric-Qualified Headers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Embed the fitted metric into growth-model output column headers as `<Category>_<metric>_<label>` (e.g. `LinearLagModel_Area_v`, `ModelMetrics_Area_RMSE`) and surface it in the CLI README and the docs.

**Architecture:** (1) The three growth fitters keep all internal machinery keyed by static `MeasurementInfo` members; `ModelFitter.analyze()` renames member-columns to qualified strings only at the boundary, keeping a private member-keyed frame for plotting. (2) Recognition becomes scheme-agnostic — each `MeasurementInfo` enum knows how to match its own headers (`owns_header`/`member_for_header`), which also fixes the pre-existing gap where `MeasureTexture`'s dynamic headers were unrecognized.

**Tech Stack:** Python 3.10+, pydantic v2, pandas, polars, scipy, pytest, Sphinx. Design spec: `docs/superpowers/specs/2026-07-01-growth-model-metric-headers/design.md`.

## Global Constraints

- **`uv` is the sole runner.** Every command is `uv run <cmd>`; never bare `python`/`pip`/`pytest`.
- **Operations/analyzers are keyword-only pydantic models.** Construct with kwargs (`LinearLagModel(on="Shape_Area", groupby=[...])`), never positional.
- **Google-style docstrings**; doctest examples must be runnable via `load_synth_yeast_plate()`. New illustrative header examples in docstrings are plain RST prose (NOT doctests), following the `TEXTURE` precedent.
- **Never author `bio_desc`/`image` on `Entry` members** (human-only). This plan adds **no** new enum members, so the classification tier/kind gate is untouched.
- **Schema modules import only stdlib + the sibling base** (`_measurement_info`) — do not add cross-`phenotypic` imports at module top in `schema/`.
- **`_measurement_outputs.py` import discipline:** its `phenotypic.analysis`/`phenotypic.measure` imports are lazy (inside functions). Keep them lazy — `analysis → util → schema` has no back-edge and must stay that way.
- **Hard cutover:** emission is always qualified; legacy un-qualified columns read from old artifacts degrade gracefully (unrecognized/undescribed, no crash). No read alias.

---

## Task graph (dependencies)

- **Task 1** (schema helpers + recognition API) — no deps.
- **Task 2** (`metric_token`) — no deps (only needs schema importable).
- **Task 3** (generic recognition in `_measurement_outputs`) — needs Task 1.
- **Task 4** (`ModelFitter` boundary rename) — needs Tasks 1 + 2.
- **Task 5** (update the 3 model-fitter tests) — needs Task 4.
- **Task 6** (CLI README model section) — needs Tasks 1 + 2.
- **Task 7** (docs: enum + fitter docstrings + notebook prose) — needs Task 1.

Tasks 1 and 2 can proceed in parallel; 3/4/6/7 fan out after them; 5 follows 4.

---

## Task 1: Schema — emission helpers + scheme-aware recognition

**Files:**
- Modify: `src/phenotypic/schema/_measurement_info.py` (add module funcs + 3 base classmethods)
- Modify: `src/phenotypic/schema/_texture.py` (texture-scheme override + regex)
- Modify: `src/phenotypic/schema/_linear_lag_model.py`, `_linear_cap_and_lag_model.py`, `_log_growth_model.py`, `_model_metrics.py` (one `header_scheme` classmethod each)
- Modify: `src/phenotypic/schema/__init__.py` (export `qualified_header`, `parse_qualified_header`)
- Test: `tests/unit/schema/test_dynamic_headers.py` (new)

**Interfaces:**
- Produces:
  - `qualified_header(member: MeasurementInfo, token: str) -> str` → `f"{member.CATEGORY}_{token}_{member.label}"`
  - `parse_qualified_header(info_cls: type[MeasurementInfo], column: str) -> tuple[str, MeasurementInfo] | None`
  - `MeasurementInfo.header_scheme(cls) -> str` (`"static"` default; `"metric_qualified"`; `"texture"`)
  - `MeasurementInfo.member_for_header(cls, column: str) -> MeasurementInfo | None`
  - `MeasurementInfo.owns_header(cls, column: str) -> bool`
  - Both helpers re-exported from `phenotypic.schema`.

- [ ] **Step 1: Write the failing test** — create `tests/unit/schema/test_dynamic_headers.py`:

```python
"""Dynamic output-header emission and scheme-aware recognition."""

import phenotypic.schema as schema
from phenotypic.schema import (
    LINEAR_CAP_AND_LAG_MODEL,
    LINEAR_LAG_MODEL,
    LOG_GROWTH_MODEL,
    MODEL_METRICS,
    MeasurementInfo,
    SHAPE,
    TEXTURE,
    parse_qualified_header,
    qualified_header,
)


def test_qualified_header_format():
    assert qualified_header(LINEAR_LAG_MODEL.v, "Area") == "LinearLagModel_Area_v"
    assert qualified_header(MODEL_METRICS.RMSE, "Radius") == "ModelMetrics_Radius_RMSE"


def test_qualified_roundtrip_including_underscored_metric():
    for token in ("Area", "Radius", "x", "my_custom"):
        for member in list(LINEAR_LAG_MODEL) + list(MODEL_METRICS):
            header = qualified_header(member, token)
            assert parse_qualified_header(type(member), header) == (token, member)


def test_metric_qualified_scheme_recognition():
    header = qualified_header(LINEAR_LAG_MODEL.s0, "Area")  # LinearLagModel_Area_s0
    assert LINEAR_LAG_MODEL.header_scheme() == "metric_qualified"
    assert LINEAR_LAG_MODEL.owns_header(header)
    assert LINEAR_LAG_MODEL.member_for_header(header) is LINEAR_LAG_MODEL.s0
    # legacy unqualified is NOT recognized (graceful degrade, hard cutover)
    assert not LINEAR_LAG_MODEL.owns_header("LinearLagModel_s0")
    assert LINEAR_LAG_MODEL.member_for_header("LinearLagModel_s0") is None


def test_static_scheme_is_default():
    assert SHAPE.header_scheme() == "static"
    assert SHAPE.owns_header("Shape_Area")
    assert SHAPE.member_for_header("Shape_Area") is SHAPE.AREA
    assert not SHAPE.owns_header("Shape_Area_extra")


def test_texture_scheme_recognition():
    headers = TEXTURE.get_headers(scale=5, matrix_name="Gray")
    directional = headers[0]  # e.g. Texture_AngularSecondMoment-deg000-scale05
    assert TEXTURE.header_scheme() == "texture"
    assert TEXTURE.owns_header(directional)
    member = TEXTURE.member_for_header(directional)
    assert member is not None and member.label in directional
    avg = next(h for h in headers if "-avg-scale" in h)
    assert TEXTURE.owns_header(avg)
    # a bare base label is not an emitted texture header
    assert not TEXTURE.owns_header("Texture_AngularSecondMoment")


def test_no_label_is_underscore_suffix_of_another_label():
    """Guardrail: protects parse_qualified_header's suffix anchoring."""
    for name in schema.__all__:
        obj = getattr(schema, name, None)
        if not (
            isinstance(obj, type)
            and issubclass(obj, MeasurementInfo)
            and obj is not MeasurementInfo
            and list(obj)
        ):
            continue
        labels = [m.label for m in obj]
        for a in labels:
            for b in labels:
                if a is not b:
                    assert not a.endswith("_" + b), (obj.__name__, a, b)


def test_double_softplus_and_log_growth_own_qualified_headers():
    for member in list(LINEAR_CAP_AND_LAG_MODEL):
        header = qualified_header(member, "Area")
        assert LINEAR_CAP_AND_LAG_MODEL.owns_header(header)
    for member in list(LOG_GROWTH_MODEL):
        header = qualified_header(member, "Area")
        assert LOG_GROWTH_MODEL.owns_header(header)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/unit/schema/test_dynamic_headers.py -q`
Expected: FAIL — `ImportError: cannot import name 'qualified_header'`.

- [ ] **Step 3: Add the helpers + base classmethods** in `src/phenotypic/schema/_measurement_info.py`.

Add these three classmethods inside `class MeasurementInfo`, next to the existing `kind`/`tier` classmethods (after `rembi_module`):

```python
    @classmethod
    def header_scheme(cls) -> str:
        """Naming scheme for this enum's DataFrame output headers.

        ``"static"`` (default) → exact ``{category}_{label}``;
        ``"metric_qualified"`` → ``{category}_{metric}_{label}`` (growth
        models + model metrics, where ``{metric}`` is a runtime value);
        ``"texture"`` → TEXTURE's ``-deg/-scale`` suffix scheme.
        """
        return "static"

    @classmethod
    def member_for_header(cls, column: str) -> "MeasurementInfo | None":
        """Return the member *column* is an output header for, or ``None``."""
        if cls.header_scheme() == "metric_qualified":
            parsed = parse_qualified_header(cls, column)
            return parsed[1] if parsed is not None else None
        for member in cls:
            if member.value == column:
                return member
        return None

    @classmethod
    def owns_header(cls, column: str) -> bool:
        """Whether *column* is one of this enum's output headers."""
        return cls.member_for_header(column) is not None
```

Add these two module-level functions at the end of `src/phenotypic/schema/_measurement_info.py` (after the class; class methods resolve names at call time, so ordering is fine):

```python
def qualified_header(member: "MeasurementInfo", token: str) -> str:
    """Runtime output header ``{Category}_{token}_{label}``.

    Embeds the fitted-metric *token* so a reader can tell which measurement a
    growth-model parameter was trained on. Inverse of
    :func:`parse_qualified_header`.
    """
    return f"{member.CATEGORY}_{token}_{member.label}"


def parse_qualified_header(
    info_cls: "type[MeasurementInfo]", column: str
) -> "tuple[str, MeasurementInfo] | None":
    """Inverse of :func:`qualified_header` for one enum.

    Returns ``(metric_token, member)`` when *column* is a metric-qualified
    header of *info_cls* (``{cat}_{metric}_{label}`` with a non-empty metric),
    else ``None``. Anchors on the longest matching member-label suffix, so an
    underscore inside the metric token stays unambiguous.
    """
    prefix = info_cls.category() + "_"
    if not column.startswith(prefix):
        return None
    best: "tuple[str, MeasurementInfo] | None" = None
    for member in info_cls:
        suffix = "_" + member.label
        if column.endswith(suffix):
            metric = column[len(prefix): len(column) - len(suffix)]
            if metric and (best is None or len(member.label) > len(best[1].label)):
                best = (metric, member)
    return best
```

- [ ] **Step 4: Export from the schema package** — in `src/phenotypic/schema/__init__.py`, change line 18 and the `__all__` list:

```python
from ._measurement_info import (
    Entry,
    MeasurementInfo,
    parse_qualified_header,
    qualified_header,
)
```

and add to `__all__` (after `"MeasurementInfo",`):

```python
    "parse_qualified_header",
    "qualified_header",
```

- [ ] **Step 5: Add the metric-qualified scheme to the 4 enums.** In each of `src/phenotypic/schema/_linear_lag_model.py`, `_linear_cap_and_lag_model.py`, `_log_growth_model.py`, and `_model_metrics.py`, add this classmethod immediately after the existing `category` classmethod in the class body:

```python
    @classmethod
    def header_scheme(cls) -> str:
        return "metric_qualified"
```

- [ ] **Step 6: Add the texture scheme to `TEXTURE`** in `src/phenotypic/schema/_texture.py`.

At the top of the file (after the existing imports), add:

```python
import re

_TEXTURE_HEADER_RE = re.compile(
    r"^(?P<cat>[A-Za-z0-9]+)_(?P<label>[^-]+)-(?:deg\d{3}|avg)-scale\d{2}$"
)
```

Inside `class TEXTURE`, add these two classmethods next to `get_headers`:

```python
    @classmethod
    def header_scheme(cls) -> str:
        return "texture"

    @classmethod
    def member_for_header(cls, column: str):
        """Recognize TEXTURE's ``{cat}_{label}-deg###-scale##`` / ``-avg-scale##``."""
        match = _TEXTURE_HEADER_RE.match(column)
        if match is None or match.group("cat") != cls.category():
            return None
        label = match.group("label")
        for member in cls:
            if member.label == label:
                return member
        return None
```

(`owns_header` is inherited from the base and delegates to `member_for_header`.)

- [ ] **Step 7: Run the tests to confirm they pass**

Run: `uv run pytest tests/unit/schema/test_dynamic_headers.py -q`
Expected: PASS (all 7 tests).

- [ ] **Step 8: Regression — schema + classification gates unaffected**

Run: `uv run pytest tests/unit/schema -q`
Expected: PASS (no member added → classification/coverage/public-api gates green).

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/schema/_measurement_info.py src/phenotypic/schema/_texture.py \
        src/phenotypic/schema/_linear_lag_model.py src/phenotypic/schema/_linear_cap_and_lag_model.py \
        src/phenotypic/schema/_log_growth_model.py src/phenotypic/schema/_model_metrics.py \
        src/phenotypic/schema/__init__.py tests/unit/schema/test_dynamic_headers.py
git commit -m "feat(schema): scheme-aware output-header recognition + qualified-header helpers"
```

---

## Task 2: `metric_token` — smart-strip helper

**Files:**
- Modify: `src/phenotypic/util/_measurement_outputs.py` (add `metric_token`, `_known_categories`, `_sanitize_token`)
- Test: `tests/unit/util/test_metric_token.py` (new)

**Interfaces:**
- Produces: `metric_token(on: str) -> str` — strips the longest known schema category prefix from `on`, else returns `on` verbatim, then sanitizes whitespace.

- [ ] **Step 1: Write the failing test** — create `tests/unit/util/test_metric_token.py`:

```python
"""Metric-token derivation from a fitter's `on` column."""

import phenotypic.schema as schema
from phenotypic.schema import MeasurementInfo
from phenotypic.util._measurement_outputs import metric_token


def test_strips_known_category():
    assert metric_token("Shape_Area") == "Area"
    assert metric_token("Size_IntegratedIntensity") == "IntegratedIntensity"


def test_endorsed_examples():
    # category-strip works even though "Radius" is not a SIZE member
    assert metric_token("Size_Radius") == "Radius"
    assert metric_token("x") == "x"            # unknown token → verbatim
    assert metric_token("Area") == "Area"      # bare label → verbatim


def test_longest_prefix_wins_for_qc_family():
    # "QC" and "QC_Tukey" are both real categories; the longest must win
    assert metric_token("QC_Tukey_NumOutliers") == "NumOutliers"


def test_every_category_strips_to_the_remainder():
    for name in schema.__all__:
        obj = getattr(schema, name, None)
        if not (
            isinstance(obj, type)
            and issubclass(obj, MeasurementInfo)
            and obj is not MeasurementInfo
            and list(obj)
        ):
            continue
        cat = obj.category()
        assert metric_token(f"{cat}_Foo") == "Foo", cat


def test_sanitizes_whitespace():
    assert metric_token("  Shape_Area  ") == "Area"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/unit/util/test_metric_token.py -q`
Expected: FAIL — `ImportError: cannot import name 'metric_token'`.

- [ ] **Step 3: Implement** in `src/phenotypic/util/_measurement_outputs.py`. Add `import re` to the top imports if not present, then add near the other private helpers:

```python
import re

_SANITIZE_TOKEN_RE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _known_categories() -> tuple[str, ...]:
    """All public schema categories, sorted longest-first for prefix matching."""
    import phenotypic.schema as schema

    cats: set[str] = set()
    for name in getattr(schema, "__all__", ()):
        obj = getattr(schema, name, None)
        if not _is_info_class(obj):
            continue
        try:
            cats.add(obj.category())
        except NotImplementedError:  # member-less classification bases
            continue
    return tuple(sorted(cats, key=len, reverse=True))


def _sanitize_token(token: str) -> str:
    return _SANITIZE_TOKEN_RE.sub("", token.strip())


def metric_token(on: str) -> str:
    """Derive the ``<metric>`` header segment from a fitter's ``on`` column.

    Strips the longest known schema **category** prefix if present
    (``Shape_Area`` → ``Area``), else returns the value verbatim
    (``x`` → ``x``); then removes whitespace.
    """
    value = str(on).strip()
    for category in _known_categories():
        if value.startswith(category + "_"):
            return _sanitize_token(value[len(category) + 1:])
    return _sanitize_token(value)
```

(`_is_info_class` and `lru_cache` already exist in this module.)

- [ ] **Step 4: Run to confirm pass**

Run: `uv run pytest tests/unit/util/test_metric_token.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/util/_measurement_outputs.py tests/unit/util/test_metric_token.py
git commit -m "feat(util): metric_token smart-strip helper (longest-prefix category match)"
```

---

## Task 3: Generic recognition in `_measurement_outputs`

**Files:**
- Modify: `src/phenotypic/util/_measurement_outputs.py` (rewrite `_producer_column_groups`, replace `_measurement_descriptions`, update `generate_output_key`; delete `_headers_for_infos`)
- Test: `tests/unit/util/test_measurement_outputs.py` (update 2 tests, add 1)

**Interfaces:**
- Consumes: `MeasurementInfo.owns_header` / `member_for_header` (Task 1), `qualified_header` (Task 1).
- Produces: `split_measurements` / `generate_output_key` now recognize static, metric-qualified, AND texture dynamic headers; legacy un-qualified model columns are unrecognized (graceful).

- [ ] **Step 1: Update the two now-broken tests + add a texture test** in `tests/unit/util/test_measurement_outputs.py`.

Add `qualified_header` to the schema import (line 9-17 block):

```python
from phenotypic.schema import (
    ColorHSV,
    ColorLab,
    LINEAR_LAG_MODEL,
    MODEL_METRICS,
    OBJECT,
    SHAPE,
    SIZE,
    TEXTURE,
    qualified_header,
)
```

Replace `test_split_measurements_groups_model_metrics_with_linear_softplus` (lines 108-123) with:

```python
def test_split_measurements_groups_model_metrics_with_linear_softplus() -> None:
    """Shared model metrics are included with a present model-specific split."""
    v = qualified_header(LINEAR_LAG_MODEL.v, "Area")
    s0 = qualified_header(LINEAR_LAG_MODEL.s0, "Area")
    rmse = qualified_header(MODEL_METRICS.RMSE, "Area")
    r2 = qualified_header(MODEL_METRICS.R2, "Area")
    frame = pd.DataFrame(
        {
            "MetadataGenetic_Strain": ["WT", "KO"],
            v: [1.1, 1.2],
            s0: [0.1, 0.2],
            rmse: [0.01, 0.02],
            r2: [0.99, 0.98],
        }
    )

    splits = split_measurements(frame)

    assert set(splits) == {"LinearLagModel"}
    assert list(splits["LinearLagModel"].columns) == list(frame.columns)


def test_split_measurements_recognizes_texture_dynamic_headers() -> None:
    """MeasureTexture's runtime -deg/-scale headers are now recognized."""
    headers = TEXTURE.get_headers(scale=5, matrix_name="Gray")[:3]
    frame = pd.DataFrame(
        {str(OBJECT.LABEL): [1], **{h: [0.0] for h in headers}}
    )

    splits = split_measurements(frame)

    assert "MeasureTexture" in splits
    assert all(h in splits["MeasureTexture"].columns for h in headers)
```

Replace `test_generate_output_key_returns_known_measurement_descriptions` (lines 126-153) with:

```python
def test_generate_output_key_returns_known_measurement_descriptions() -> None:
    """Static, metric-qualified, and texture columns all resolve to a description."""
    v = qualified_header(LINEAR_LAG_MODEL.v, "Area")
    rmse = qualified_header(MODEL_METRICS.RMSE, "Area")
    texture = TEXTURE.get_headers(scale=5, matrix_name="Gray")[0]
    frame = pd.DataFrame(
        {
            "Custom_Note": ["a"],
            str(OBJECT.LABEL): [1],
            str(SIZE.AREA): [10.0],
            v: [1.1],
            rmse: [0.01],
            texture: [0.5],
        }
    )

    key = generate_output_key(frame)

    assert list(key.columns) == ["column_header", "description"]
    assert key["column_header"].tolist() == [
        str(OBJECT.LABEL),
        str(SIZE.AREA),
        v,
        rmse,
        texture,
    ]
    descriptions = dict(zip(key["column_header"], key["description"]))
    assert descriptions[str(OBJECT.LABEL)] == OBJECT.LABEL.desc
    assert descriptions[str(SIZE.AREA)] == SIZE.AREA.desc
    assert descriptions[v] == LINEAR_LAG_MODEL.v.desc
    assert descriptions[rmse] == MODEL_METRICS.RMSE.desc
    assert descriptions[texture] == TEXTURE.ANGULAR_SECOND_MOMENT.desc
```

- [ ] **Step 2: Run to confirm the updated tests fail against current code**

Run: `uv run pytest tests/unit/util/test_measurement_outputs.py -q`
Expected: FAIL — old exact-match recognition returns `{}` for qualified/texture columns; `LinearLagModel` split and the texture assertions fail.

- [ ] **Step 3: Rewrite recognition** in `src/phenotypic/util/_measurement_outputs.py`.

Replace `_producer_column_groups` (lines 110-133) with a version that uses `owns_header`:

```python
def _producer_column_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    """Map producer class names to their present measurement columns."""
    ordered_columns = list(columns)
    groups: dict[str, list[str]] = {}

    for producer in _discover_measurement_producers():
        present_primary = [
            column
            for column in ordered_columns
            if any(info.owns_header(column) for info in producer.primary_infos)
        ]
        if not present_primary:
            continue

        producer_headers = set(present_primary)
        producer_headers.update(
            column
            for column in ordered_columns
            if any(info.owns_header(column) for info in producer.shared_infos)
        )
        groups[producer.output_key] = [
            column for column in ordered_columns if column in producer_headers
        ]

    return groups
```

Delete `_headers_for_infos` (lines 136-138) — it is no longer referenced.

Replace `generate_output_key` (lines 63-83) body's description source and `_measurement_descriptions` (lines 234-248) with a per-column resolver. New `generate_output_key`:

```python
def generate_output_key(df: MeasurementFrame) -> pd.DataFrame:
    """Generate a column-description key for recognized output columns.

    Args:
        df: A pandas or polars measurements DataFrame.

    Returns:
        A pandas DataFrame with ``column_header`` and ``description`` columns,
        preserving input column order and omitting columns not backed by a
        public ``MeasurementInfo`` member (in any header scheme).

    Raises:
        TypeError: If *df* is not a pandas or polars DataFrame.
    """
    records = [
        {"column_header": column, "description": desc}
        for column in _columns(df)
        if (desc := _describe_column(column)) is not None
    ]
    return pd.DataFrame(records, columns=["column_header", "description"])
```

Replace `_measurement_descriptions` with:

```python
@lru_cache(maxsize=1)
def _public_info_classes() -> tuple[type[MeasurementInfo], ...]:
    """Public, member-ful ``MeasurementInfo`` subclasses exported by schema."""
    import phenotypic.schema as schema

    classes: list[type[MeasurementInfo]] = []
    for name in getattr(schema, "__all__", ()):
        obj = getattr(schema, name, None)
        if _is_info_class(obj) and list(obj):
            classes.append(obj)
    return tuple(classes)


def _describe_column(column: str) -> str | None:
    """Resolve *column* to its member's ``desc`` across all schemes, or None."""
    for info in _public_info_classes():
        member = info.member_for_header(column)
        if member is not None:
            return member.desc
    return None
```

- [ ] **Step 4: Run the updated tests + the full file**

Run: `uv run pytest tests/unit/util/test_measurement_outputs.py -q`
Expected: PASS (all tests, including the new texture recognition).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/util/_measurement_outputs.py tests/unit/util/test_measurement_outputs.py
git commit -m "feat(util): scheme-agnostic column recognition (fixes texture + qualified model headers)"
```

---

## Task 4: `ModelFitter` boundary rename

**Files:**
- Modify: `src/phenotypic/analysis/abc_/_model_fitter.py`
- Test: `tests/unit/analysis/test_model_fitter_headers.py` (new)

**Interfaces:**
- Consumes: `qualified_header`, `MeasurementInfo` (Task 1); `metric_token` (Task 2).
- Produces: `analyze()` and `results()` return a frame whose model/metric columns are `qualified_header(member, metric_token(self.on))`. Internal plotting frame `_latest_fit_internal` stays member-keyed. Subclasses unchanged.

- [ ] **Step 1: Write the failing test** — create `tests/unit/analysis/test_model_fitter_headers.py`:

```python
"""ModelFitter emits metric-qualified headers and still plots."""

import matplotlib
import pandas as pd

from phenotypic.analysis import LinearLagModel
from phenotypic.schema import LINEAR_LAG_MODEL, MODEL_METRICS, qualified_header

matplotlib.use("Agg")


def _toy_df() -> pd.DataFrame:
    rows = []
    for strain in ("A", "B"):
        for t in range(8):
            rows.append(
                {
                    "MetadataGenetic_Strain": strain,
                    "MetadataCulture_Time": float(t),
                    "Shape_Area": 1.0 + 2.0 * t,
                }
            )
    return pd.DataFrame(rows)


def test_analyze_returns_metric_qualified_columns():
    model = LinearLagModel(on="Shape_Area", groupby=["MetadataGenetic_Strain"])
    res = model.analyze(_toy_df())
    assert qualified_header(LINEAR_LAG_MODEL.v, "Area") in res.columns
    assert qualified_header(MODEL_METRICS.RMSE, "Area") in res.columns
    assert "LinearLagModel_v" not in res.columns  # hard cutover, no legacy header


def test_results_returns_the_qualified_frame():
    model = LinearLagModel(on="Shape_Area", groupby=["MetadataGenetic_Strain"])
    model.analyze(_toy_df())
    assert qualified_header(LINEAR_LAG_MODEL.s0, "Area") in model.results().columns


def test_show_works_after_qualified_analyze():
    model = LinearLagModel(on="Shape_Area", groupby=["MetadataGenetic_Strain"])
    model.analyze(_toy_df())
    fig, ax = model.show()
    assert ax is not None
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run pytest tests/unit/analysis/test_model_fitter_headers.py -q`
Expected: FAIL — `res.columns` currently holds `LinearLagModel_v` (unqualified), so the qualified-header asserts fail.

- [ ] **Step 3: Edit `src/phenotypic/analysis/abc_/_model_fitter.py`.**

3a. Extend the schema import (line 16) to include `MeasurementInfo` and `qualified_header`:

```python
from phenotypic.schema import CULTURE_METADATA, MeasurementInfo, MODEL_METRICS, qualified_header
```

3b. Add the internal plotting frame next to `_latest_model_scores` (after line 84):

```python
    _latest_fit_internal: pd.DataFrame = PrivateAttr(
        default_factory=pd.DataFrame
    )
```

3c. Add a metric-token property and a rename-map helper (place after `_nan_fit_columns`, before `_apply2group_func`):

```python
    @property
    def _metric_token(self) -> str:
        """The ``<metric>`` header segment derived from ``self.on``."""
        from phenotypic.util._measurement_outputs import metric_token

        return metric_token(str(self.on))

    def _qualified_rename_map(self, results: pd.DataFrame) -> Dict[Any, str]:
        """Map member-object columns to their metric-qualified header strings."""
        token = self._metric_token
        return {
            column: qualified_header(column, token)
            for column in results.columns
            if isinstance(column, MeasurementInfo)
        }
```

3d. In `analyze` (lines 331-337), replace the tail so the internal frame is kept and the public frame is renamed:

```python
        results = pd.concat(model_res, axis=0).reset_index(drop=False)

        for col_key, val in self._post_fit_columns().items():
            results.insert(loc=len(results.columns), column=col_key, value=val)

        self._latest_fit_internal = results
        self._latest_model_scores = results.rename(
            columns=self._qualified_rename_map(results)
        )
        return self._latest_model_scores
```

3e. In `_filter_for_plot` (lines 346-360), read `_latest_fit_internal` (member-keyed) instead of `_latest_model_scores`:

```python
    def _filter_for_plot(
            self, criteria: Dict[str, Union[Any, List[Any]]] | None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply `criteria` (if any) to both the model-scores and measurements frames."""
        if criteria is not None:
            model_scores = self._filter_by(
                    df=self._latest_fit_internal, criteria=criteria, copy=True
            )
            measurements = self._filter_by(
                    df=self._latest_measurements, criteria=criteria, copy=True
            )
        else:
            model_scores = self._latest_fit_internal.copy()
            measurements = self._latest_measurements.copy()
        return model_scores, measurements
```

- [ ] **Step 4: Run the new test + the ModelFitter's own suite**

Run: `uv run pytest tests/unit/analysis/test_model_fitter_headers.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/analysis/abc_/_model_fitter.py tests/unit/analysis/test_model_fitter_headers.py
git commit -m "feat(analysis): emit metric-qualified model headers; keep member-keyed frame for plots"
```

---

## Task 5: Update the growth-model fitter tests

**Files:**
- Modify: `tests/unit/analysis/test_linear_softplus.py`
- Modify: `tests/unit/analysis/test_double_softplus.py`
- Modify: `tests/unit/analysis/test_log_growth_model.py`
- Audit: `tests/unit/core/test_pipeline_analyze.py`

**Interfaces:**
- Consumes: `analyze()` public frames now have qualified string columns (Task 4).

**The transform (identical for all three files):** every subscript of an `analyze()`/`results()` frame by a **model member** (`results[MEMBER]`, `res[MEMBER]`, `row[MEMBER]`, `df.loc[key, MEMBER]`) must go through a metric-qualifying helper, because the column is now a qualified string, not a member. All three files fit on `on="Shape_Area"`, whose token is `"Area"`.

- [ ] **Step 1: Add the helper to each of the three files** (just below the imports). Import `qualified_header` from `phenotypic.schema` if not already imported, then:

```python
from phenotypic.schema import qualified_header


def _q(member):
    """Qualified header for tests that fit on ``Shape_Area`` (token ``Area``)."""
    return qualified_header(member, "Area")
```

- [ ] **Step 2: Apply the transform in `test_linear_softplus.py`.**

- Membership loop (lines ~155-169): keep the `expected` list of members, and change the assertion loop to qualify each:

```python
        for col in expected:
            assert _q(col) in results.columns, f"missing column: {col}"
```

- Every value access — wrap the member. Representative edits (apply to ALL such lines: 185, 206-208, 212-214, 252-253, 271-272, 289-291, and any others in the file):

```python
        r2 = results[_q(MODEL_METRICS.R2)]
        ...
        assert abs(row1[_q(LINEAR_LAG_MODEL.v)] - 5.0) < 0.3
        assert abs(row1[_q(LINEAR_LAG_MODEL.s0)] - 1.0) < 0.5
        assert abs(row1[_q(LINEAR_LAG_MODEL.lam)] - 2.0) < 0.5
```

Rule of thumb: `grep -n "LINEAR_LAG_MODEL\.\|MODEL_METRICS\." tests/unit/analysis/test_linear_softplus.py`; for each hit that indexes a results/row frame, wrap the member in `_q(...)`. Do NOT wrap members used as fit inputs or attribute checks (there are none here beyond subscripts).

- [ ] **Step 3: Apply the transform in `test_double_softplus.py`.**

- Membership loop (lines 123-140): change to `assert _q(col) in results.columns`.
- Wrap every results/row/`.loc` member subscript. Representative (apply to lines 167-174, 187-188, 197, 216-219, 235-237, 252-253, 273-275, 403-405):

```python
        assert abs(row1[_q(LINEAR_CAP_AND_LAG_MODEL.v)] - 5.0) < 0.3
        ...
        assert abs(results.loc["Strain1", _q(LINEAR_CAP_AND_LAG_MODEL.smax)] - 50.0) < 1.0
        ...
        assert res[_q(LINEAR_CAP_AND_LAG_MODEL.mode)].iloc[0] == "fitted_beta"
        ...
        assert np.isnan(float(res[_q(LINEAR_CAP_AND_LAG_MODEL.v)].iloc[0]))
        assert pd.isna(res[_q(LINEAR_CAP_AND_LAG_MODEL.beta)].iloc[0])
        assert pd.isna(res[_q(LINEAR_CAP_AND_LAG_MODEL.mode)].iloc[0])
```

- [ ] **Step 4: Apply the transform in `test_log_growth_model.py`.**

- Expected-columns membership (lines 127-143): keep the members list, change the loop to `assert _q(col) in results.columns`.
- Wrap value accesses (lines 146-148, 157-158, 180): `results[_q(LOG_GROWTH_MODEL.R_FIT)]`, etc.
- **Gotcha — the `GROWTH_RATE` series-equality check (lines 157-161):** the derived series' `.name` is now the qualified header, so `assert_series_equal` must ignore names. Rewrite as:

```python
        r_values = results[_q(LOG_GROWTH_MODEL.R_FIT)]
        k_values = results[_q(LOG_GROWTH_MODEL.K_FIT)]
        growth_rate_calc = (r_values * k_values) / 4
        pd.testing.assert_series_equal(
                results[_q(LOG_GROWTH_MODEL.GROWTH_RATE)],
                growth_rate_calc,
                check_names=False,
        )
```

- `K_MAX` check (line 180): `assert (results[_q(LOG_GROWTH_MODEL.K_MAX)] == 1200).all()`.
- The config-defaults tests that only assert `model.on == "Area"` (lines 80-110) need **no** change (they check attributes, not columns).

- [ ] **Step 5: Audit `tests/unit/core/test_pipeline_analyze.py`.**

Run: `uv run grep -n "LOG_GROWTH_MODEL\|LINEAR_\|MODEL_METRICS\|\.columns\|results\[" tests/unit/core/test_pipeline_analyze.py` (via `uv run python -c` or plain `grep`). The model there uses `on="x"` (token `"x"`). If any assertion indexes the analyze result by a model member, wrap it with `qualified_header(member, "x")` (add a local `_qx` helper analogous to `_q`). If the test only checks that `analyze()` returns a non-empty frame / row count, no change is needed.

- [ ] **Step 6: Run all affected analysis + core tests**

Run: `uv run pytest tests/unit/analysis/test_linear_softplus.py tests/unit/analysis/test_double_softplus.py tests/unit/analysis/test_log_growth_model.py tests/unit/core/test_pipeline_analyze.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/unit/analysis/test_linear_softplus.py tests/unit/analysis/test_double_softplus.py \
        tests/unit/analysis/test_log_growth_model.py tests/unit/core/test_pipeline_analyze.py
git commit -m "test(analysis): assert metric-qualified growth-model output headers"
```

---

## Task 6: CLI README — Models & Analysis section

**Files:**
- Modify: `src/phenotypic/_cli/_cli_readme_generator.py`
- Test: `tests/unit/cli/test_readme_model_section.py` (new; create the `tests/unit/cli/` dir if absent)

**Interfaces:**
- Consumes: `pipeline.get_model()`, `model.on`, `model._measurement_infoclass`; `qualified_header` (Task 1), `metric_token` (Task 2), `MODEL_METRICS`.
- Produces: a `## Models & Analysis` markdown section rendered only when a model is configured.

- [ ] **Step 1: Write the failing test** — create `tests/unit/cli/test_readme_model_section.py`:

```python
"""README Models & Analysis section renders metric-qualified headers."""

from types import SimpleNamespace

from phenotypic import ImagePipeline
from phenotypic.analysis import LinearLagModel
from phenotypic._cli._cli_readme_generator import READMEGenerator
from phenotypic.schema import LINEAR_LAG_MODEL, MODEL_METRICS, qualified_header


def _generator_with_model() -> READMEGenerator:
    pipe = ImagePipeline()
    pipe.set_model(
        LinearLagModel(on="Shape_Area", groupby=["MetadataGenetic_Strain"])
    )
    return READMEGenerator(config=SimpleNamespace(), pipeline=pipe)


def test_model_section_documents_qualified_headers():
    section = _generator_with_model()._generate_model_section()
    assert "## Models & Analysis" in section
    assert "LinearLagModel" in section
    assert "Shape_Area" in section
    assert f"`{qualified_header(LINEAR_LAG_MODEL.v, 'Area')}`" in section
    assert f"`{qualified_header(MODEL_METRICS.RMSE, 'Area')}`" in section


def test_model_section_is_empty_without_a_model():
    pipe = ImagePipeline()
    gen = READMEGenerator(config=SimpleNamespace(), pipeline=pipe)
    assert gen._generate_model_section() == ""
```

(If `ImagePipeline()` requires arguments, mirror the construction used in `tests/unit/core/test_pipeline_analyze.py`.)

- [ ] **Step 2: Run to confirm it fails**

Run: `uv run pytest tests/unit/cli/test_readme_model_section.py -q`
Expected: FAIL — `AttributeError: 'READMEGenerator' object has no attribute '_generate_model_section'`.

- [ ] **Step 3: Implement.** In `src/phenotypic/_cli/_cli_readme_generator.py`, add `self._generate_model_section()` into the `sections` list in `generate` (between measurements and footer, lines 53-59):

```python
        sections = [
            self._generate_header(),
            self._generate_output_structure(datasets),
            self._generate_layers_section(),
            self._generate_measurements_section(),
            self._generate_model_section(),
            self._generate_footer(),
        ]
```

Add the method (place it after `_generate_measurements_section`'s helpers, before `_generate_footer`):

```python
    def _generate_model_section(self) -> str:
        """Document the configured analysis model's metric-qualified columns.

        Renders only when the pipeline has a ``model`` configured. Column
        headers embed the fitted metric (``model.on``) so the README matches
        the actual ``analysis.csv`` produced by this run.
        """
        from phenotypic.schema import MODEL_METRICS, qualified_header
        from phenotypic.util._measurement_outputs import metric_token

        model = self.pipeline.get_model()
        if model is None:
            return ""
        info_cls = getattr(model, "_measurement_infoclass", None)
        if info_cls is None:
            return ""

        token = metric_token(str(model.on))
        model_name = type(model).__name__

        lines = [
            "## Models & Analysis",
            "",
            f"Model `{model_name}` fit on metric `{model.on}` "
            "(output written to `deliverables/analysis.csv`).",
            "",
            "Output columns follow `<Model>_<metric>_<parameter>`; for this "
            f"run `<metric>` = `{token}`.",
            "",
            "| Column | Description |",
            "|--------|-------------|",
        ]
        for member in list(info_cls) + list(MODEL_METRICS):
            header = qualified_header(member, token)
            desc = (member.desc or "").replace("|", "\\|").replace("\n", " ")
            if len(desc) > 200:
                desc = desc[:197] + "..."
            lines.append(f"| `{header}` | {desc} |")
        return "\n".join(lines)
```

- [ ] **Step 4: Run to confirm pass**

Run: `uv run pytest tests/unit/cli/test_readme_model_section.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_readme_generator.py tests/unit/cli/test_readme_model_section.py
git commit -m "feat(cli): document metric-qualified model columns in run README"
```

---

## Task 7: Docs — enum + fitter docstrings + notebook prose

**Files:**
- Modify: `src/phenotypic/schema/_linear_lag_model.py`, `_linear_cap_and_lag_model.py`, `_log_growth_model.py` (add class docstrings)
- Modify: `src/phenotypic/analysis/_linear_lag_model.py`, `_linear_cap_and_lag_model.py`, `_log_growth_model.py` (add an "Output column naming" note)
- Modify: `docs/source/explanation/notebooks/linear_softplus_model.ipynb` (prose fix)
- Modify: `src/phenotypic/schema/CLAUDE.md` (add a "Dynamic output headers" contributor section)

**Interfaces:**
- Consumes: the `metric_qualified` scheme (Task 1). Docs are illustrative RST prose (no doctests).

- [ ] **Step 1: Add a class docstring to each growth-model ENUM.** These enums currently have none; the Sphinx measurements-ref page renders it above the static label table. For `src/phenotypic/schema/_linear_lag_model.py`, insert immediately after `class LINEAR_LAG_MODEL(DerivedMeasure):`:

```python
    """Fitted parameters of the linear-softplus lag model (no saturation).

    Output columns are **metric-qualified**: each header is
    ``LinearLagModel_<metric>_<parameter>``, where ``<metric>`` records the
    measurement the model was fit on (``self.on`` with its category prefix
    stripped, e.g. ``Shape_Area`` → ``Area``). For example, fitting on
    ``Shape_Area`` emits ``LinearLagModel_Area_v`` (post-lag growth rate) and
    ``LinearLagModel_Area_s0`` (initial size). The labels below are the
    ``<parameter>`` segment; the ``<metric>`` infix is filled in at fit time.
    """
```

Do the same in `_linear_cap_and_lag_model.py` (category `LinearCapAndLagModel`, example `LinearCapAndLagModel_Area_v`) and `_log_growth_model.py` (category `LogGrowthModel`, example `LogGrowthModel_Area_r`, `LogGrowthModel_Area_µmax`), adjusting the class name and example in the prose accordingly.

- [ ] **Step 2: Add an "Output column naming" note to each FITTER class docstring.** In `src/phenotypic/analysis/_linear_lag_model.py`, add a paragraph to the `LinearLagModel` class docstring (e.g. just before the `.. note::` block):

```
    Output column naming:
        ``analyze`` emits metric-qualified columns
        ``LinearLagModel_<metric>_<parameter>`` (e.g. fitting ``on="Shape_Area"``
        yields ``LinearLagModel_Area_v``), plus qualified fit-quality columns
        ``ModelMetrics_<metric>_<label>``. The ``<metric>`` segment is
        ``self.on`` with a recognized measurement-category prefix stripped.
```

Add the analogous paragraph to `LinearCapAndLagModel` and `LogGrowthModel` (adjust class name / example).

- [ ] **Step 3: Fix stale prose in the notebook.** In `docs/source/explanation/notebooks/linear_softplus_model.ipynb`, find the markdown cell (~line 154) referencing `LinearCapAndLagModel_mode` and change it to reflect qualification, e.g.:

> "...recorded in the metric-qualified ``LinearCapAndLagModel_<metric>_mode`` column of the results (e.g. ``LinearCapAndLagModel_Area_mode`` when fitting on ``Shape_Area``)."

Use `uv run python -c "import json,glob"` or a JSON-aware edit; keep it a markdown cell (no code cell reads the column).

- [ ] **Step 4: Add the "Dynamic output headers" contributor section to `src/phenotypic/schema/CLAUDE.md`.** Append this section (place it after the "Measurement classification" sections, before "Downstream users import headers directly:"):

````markdown
## Dynamic output headers (recognition schemes)

Output column names are an *encoding* of `(member, runtime-params)`. Each enum owns
**both** directions — emit and decode — so downstream code
(`util/_measurement_outputs.py` `split_measurements`/`generate_output_key`, the CLI
README, Sphinx) is scheme-agnostic and only ever calls the recognition interface on the
base:

- `header_scheme() -> "static" | "metric_qualified" | "texture"` — dispatch hint
  (default `"static"`).
- `member_for_header(column) -> member | None` — decode a column to its member.
- `owns_header(column) -> bool` — `member_for_header(...) is not None`; **never** override.

Emission (write side) lives with the enum or as shared functions in `_measurement_info.py`:

- **static** — the header *is* `member.value` (`Shape_Area`); base default, no override.
- **metric_qualified** — `{cat}_{metric}_{label}` (e.g. `LinearLagModel_Area_v`):
  `qualified_header(member, token)` / `parse_qualified_header(info_cls, column)`; the enum
  sets `header_scheme() -> "metric_qualified"` (the 3 growth models + `MODEL_METRICS`).
  The token comes from `metric_token(on)` in `util/_measurement_outputs.py`
  (strips the longest known category prefix from `self.on`).
- **texture** — `{cat}_{label}-deg###-scale##` / `-avg-scale##`:
  `TEXTURE.get_headers(scale, matrix_name)` plus a `member_for_header` regex override.

**Invariant:** the format must be invertible — `parse(emit(member, token)) == (token,
member)`. `metric_qualified` anchors on the category prefix + the known member-label
suffix, so a guardrail in `tests/unit/schema/test_dynamic_headers.py` asserts no label is
a `_`-suffix of another. Emission (in the producer) and recognition (on the enum) live in
two files that must agree; the round-trip test keeps them honest. Docs/`rst_table` render
the **base** labels; only run-specific surfaces (the CLI README) fill in the real token.

### Adding a new dynamic scheme

1. Pick an invertible format — the member label must be recoverable without the token.
2. Add an emission helper — co-locate on the enum (like `get_headers`) or a shared func;
   reuse `qualified_header`/`parse_qualified_header` if the shape is the
   `{cat}_{token}_{label}` infix (then you only set `header_scheme()`, no parser).
3. Override `member_for_header` on the enum (and set `header_scheme()`). `owns_header`
   is inherited.
4. In the producer, name columns via the helper and declare
   `_measurement_infoclass = <enum>` — that one attribute wires
   split/output-key/recognition. (For the CLI README's measurement tables only, also add
   the measurer to `_get_measurement_infoclasses` in `_cli/_cli_readme_generator.py`.)

A `MeasureFeatures` emits via the enum (never hand-built strings):

    class MeasureTexture(MeasureFeatures):
        _measurement_infoclass: ClassVar[type] = TEXTURE
        scale: List[int] = [5]
        def _operate(self, image):
            cols = TEXTURE.get_headers(self.scale[0], "Gray")   # runtime params -> headers
            meas = pd.DataFrame(data, columns=cols)
            meas.insert(loc=0, column=OBJECT.LABEL, value=image.objects.labels2series())
            return meas

`ModelFitter` subclasses are the exception: they stay member-keyed internally and the ABC
(`analyze()`) renames to qualified headers at the boundary, so a fitter never touches
header strings.
````

- [ ] **Step 5: Verify the docstrings import and the enum tables still build.**

Run: `uv run python -c "import phenotypic.schema as s; print(s.LINEAR_LAG_MODEL.__doc__.splitlines()[0]); print(s.LINEAR_LAG_MODEL.rst_table()[:60])"`
Expected: prints the new first docstring line and an RST list-table header (base labels), no exception.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/schema/_linear_lag_model.py src/phenotypic/schema/_linear_cap_and_lag_model.py \
        src/phenotypic/schema/_log_growth_model.py src/phenotypic/analysis/_linear_lag_model.py \
        src/phenotypic/analysis/_linear_cap_and_lag_model.py src/phenotypic/analysis/_log_growth_model.py \
        docs/source/explanation/notebooks/linear_softplus_model.ipynb src/phenotypic/schema/CLAUDE.md
git commit -m "docs: explain metric-qualified growth-model header convention + dynamic-header contributor guide"
```

---

## Final verification

- [ ] **Step 1: Full affected test sweep**

Run:
```bash
uv run pytest tests/unit/schema tests/unit/util tests/unit/analysis tests/unit/cli tests/unit/core/test_pipeline_analyze.py -q
```
Expected: PASS.

- [ ] **Step 2: Lint + type-check the touched modules**

Run:
```bash
uv run ruff check --fix src/phenotypic/schema src/phenotypic/util/_measurement_outputs.py \
    src/phenotypic/analysis/abc_/_model_fitter.py src/phenotypic/_cli/_cli_readme_generator.py
uv run mypy src/phenotypic/schema/_measurement_info.py src/phenotypic/util/_measurement_outputs.py \
    src/phenotypic/analysis/abc_/_model_fitter.py src/phenotypic/_cli/_cli_readme_generator.py
```
Expected: no new lint errors; mypy clean (or no worse than baseline).

- [ ] **Step 3: End-to-end smoke of an actual analysis frame** (proves emission + recognition + description round-trip agree):

```bash
uv run python - <<'PY'
import pandas as pd
from phenotypic.analysis import LinearLagModel
from phenotypic.util import generate_output_key, split_measurements

rows = [{"MetadataGenetic_Strain": s, "MetadataCulture_Time": float(t), "Shape_Area": 1.0 + 2.0*t}
        for s in ("A", "B") for t in range(8)]
res = LinearLagModel(on="Shape_Area", groupby=["MetadataGenetic_Strain"]).analyze(pd.DataFrame(rows))
cols = [c for c in res.columns if c != "MetadataGenetic_Strain"]
assert any(c == "LinearLagModel_Area_v" for c in cols), cols
key = generate_output_key(res)
assert "LinearLagModel_Area_v" in key["column_header"].tolist()
splits = split_measurements(res)
assert "LinearLagModel" in splits, list(splits)
print("OK:", [c for c in cols[:6]])
PY
```
Expected: prints `OK:` with qualified headers; no assertion error.

- [ ] **Step 4: Confirm the CLAUDE.md contributor section landed** — the "Dynamic output headers" section is added in Task 7 Step 4. Verify it is present in `src/phenotypic/schema/CLAUDE.md` and mentions `header_scheme`/`owns_header`/`member_for_header`, the three schemes, and the round-trip invariant.

---

## Self-review notes (author)

- **Spec coverage:** §4 recognition API → Task 1; §5 `metric_token` → Task 2; §6 boundary rename → Task 4; §7 generic recognition (incl. texture, per locked decision 4) → Task 3; §8 docs → Task 7; §9 README → Task 6; §10 tests → Tasks 1–7 (guardrail in Task 1, category enumeration in Task 2, plotting round-trip in Task 4, texture+qualified recognition in Task 3, model-test updates in Task 5, README in Task 6). §11 blast radius all covered. §12 non-goals respected (no `filters` docs, no read alias, no member changes).
- **Type consistency:** `qualified_header`/`parse_qualified_header`/`metric_token`/`owns_header`/`member_for_header`/`header_scheme` names used identically across Tasks 1–7. `_latest_fit_internal` introduced (Task 4) and read only in `_filter_for_plot`.
- **Hard-cutover contract** verified by explicit "legacy unqualified NOT recognized" assertions (Tasks 1, 4).
