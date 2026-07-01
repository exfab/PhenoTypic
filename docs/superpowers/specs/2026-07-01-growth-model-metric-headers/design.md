# Design: metric-qualified growth-model output headers

- **Date:** 2026-07-01
- **Status:** Approved design → ready for implementation plan
- **Topic:** Embed the fitted measurement into growth-model output column headers so
  users can see which metric a growth rate was trained on, and surface that in the CLI
  README and the docs.

---

## 1. Problem & goal

The three growth-rate fitters — `LinearLagModel`, `LinearCapAndLagModel`,
`LogGrowthModel` (`src/phenotypic/analysis/`) — emit output columns keyed by static
`MeasurementInfo` members that stringify to `<Category>_<label>`:

    LinearLagModel_v, LinearLagModel_s0, LinearLagModel_lambda, ...
    ModelMetrics_RMSE, ModelMetrics_R2, ...

Nothing in the header records **which measurement was fitted** (`self.on`, e.g.
`Shape_Area` or a radius column). A user reading `analysis.csv` cannot tell whether
`LinearLagModel_v` is a growth rate in area-units, radius-units, or intensity-units.

**Goal:** make growth-model output headers carry the fitted metric, in the form
`<Category>_<metric>_<label>` (e.g. `LinearLagModel_Radius_v`,
`ModelMetrics_Radius_RMSE`). This mirrors the existing dynamic-header pattern used by
`MeasureTexture` / `TEXTURE.get_headers(scale, matrix_name)`
(`src/phenotypic/schema/_texture.py`), which encodes runtime parameters
(`scale`, angle) into the header. The metric must also appear in:

- the **CLI README** (`deliverables/README.md`) for a run that configures a model, and
- the **docs** (Sphinx measurements-reference pages + the fitter/enum docstrings).

## 2. Locked decisions

Confirmed with the user before drafting; not to be relitigated:

1. **Metric token = smart-strip to label.** From `self.on`, strip a recognized schema
   **category** prefix if present, else use the value verbatim; then sanitize.
   Endorsed examples: `Shape_Area → Area`, `Size_Radius → Radius` (works via
   category-strip even though `Radius` is not a `SIZE` member), `x → x`, `Area → Area`.
2. **Scope = ALL fitter-emitted columns get the qualifier**, including the shared
   `ModelMetrics_*` fit-quality/diagnostic columns → `ModelMetrics_<metric>_<label>`.
   (Rationale: uniform transformation; avoids collisions if the same model is fit on two
   metrics and the results are merged.)
3. **Hard cutover.** No backward-compat alias on **emission**. Legacy un-qualified
   columns read from old artifacts **degrade gracefully** (unrecognized/undescribed, no
   crash — the same behaviour dynamic texture columns already have today). No read-time
   alias.
4. **Fix the pre-existing texture gap too.** The recognition machinery being rebuilt in
   `util/_measurement_outputs.py` also currently fails to recognize `MeasureTexture`'s
   dynamic headers. Recognition is generalized so **both** dynamic schemes are handled.

## 3. Design overview

Two ideas carry the whole design:

**(A) Emission renames only at the boundary.** All internal fitter machinery stays keyed
by static enum members (zero changes to the three model subclasses' fit/predict/hover
logic). `ModelFitter.analyze()` keeps a private member-keyed frame for plotting and
produces the public frame by renaming member-columns to qualified strings.

**(B) Recognition is scheme-agnostic.** Each `MeasurementInfo` enum knows how to match
its own headers via two new classmethods (`owns_header` / `member_for_header`). The
default matches the exact `member.value`; `TEXTURE` overrides for its `-deg/-scale`
suffix scheme; the four metric-qualified enums override for the `<cat>_<metric>_<label>`
infix scheme. `util/_measurement_outputs.py` calls these methods instead of comparing
against a static header set, which fixes texture and growth in one stroke.

## 4. Header schemes & the recognition interface

Three schemes now exist. The enum is the single source of truth for matching its own
columns.

| Scheme | Producer | Emission format | Example |
|---|---|---|---|
| `static` (default) | most `MeasureFeatures`, QC, metadata | `{cat}_{label}` | `Shape_Area` |
| `metric_qualified` | 3 growth enums + `MODEL_METRICS` | `{cat}_{metric}_{label}` | `LinearLagModel_Radius_v` |
| `texture` | `TEXTURE` | `{cat}_{label}-deg{a:03d}-scale{s:02d}` and `{cat}_{label}-avg-scale{s:02d}` | `Texture_Contrast-deg000-scale05` |

### New API on `MeasurementInfo` base (`schema/_measurement_info.py`)

```python
@classmethod
def owns_header(cls, column: str) -> bool:
    """True if `column` is an output header produced by this enum."""
    # default: exact match against member values
    return column in {m.value for m in cls}

@classmethod
def member_for_header(cls, column: str) -> "MeasurementInfo | None":
    """Return the member that `column` corresponds to, or None."""
    # default: exact-value reverse lookup
```

- **`TEXTURE`** overrides both, co-located with its existing `get_headers` (which already
  breaks the "pure data" rule for a good reason): match the `-deg###-scale##` /
  `-avg-scale##` suffix and resolve the leading `{cat}_{label}` back to a member.
- **Metric-qualified enums** (`LINEAR_LAG_MODEL`, `LINEAR_CAP_AND_LAG_MODEL`,
  `LOG_GROWTH_MODEL`, `MODEL_METRICS`) override both to the infix scheme via a shared
  helper (below). A non-empty metric middle is **required**, so a legacy un-qualified
  `ModelMetrics_RMSE` does **not** match (graceful degrade, per decision 3).

Preferred factoring for the metric-qualified override: a small shared mixin
(`_MetricQualifiedInfo`) supplying the two classmethod overrides, mixed into each of the
four enums. If the Enum MRO makes a mixin fragile, fall back to a one-line delegation in
each enum to module-level helpers. This is an implementation detail to settle in the plan;
the **interface** above is fixed.

### Shared emission + parse helpers (`schema/_measurement_info.py`)

```python
def qualified_header(member, token: str) -> str:
    return f"{member.CATEGORY}_{token}_{member.label}"

def parse_qualified_header(info_cls, column: str) -> tuple[str, member] | None:
    cat = info_cls.category()
    if not column.startswith(cat + "_"):
        return None
    # anchor on a known member-label suffix; the middle (non-empty) is the metric
    for member in info_cls:
        suffix = "_" + member.label
        if column.endswith(suffix):
            metric = column[len(cat) + 1 : -len(suffix)]
            if metric:
                return metric, member
    return None
```

Suffix-anchoring safely disambiguates the current label sets (verified: `MSE` vs `RMSE`,
`K` vs `Kmax`, `s0` vs `smax`, `alpha` vs `lambda` — no label is a `_`-anchored suffix of
another within one enum). This safety is currently incidental, so a **guardrail test**
(Section 9) enforces it going forward.

## 5. Metric token (`metric_token`)

Home: `util/_measurement_outputs.py` (the existing hub for schema-header introspection;
`analysis → util → schema` import DAG has no back-edge, and util's `phenotypic.analysis`
import is already lazy — preserve that).

```python
@lru_cache(maxsize=1)
def _known_categories() -> list[str]:
    # every public schema enum's category(), sorted LONGEST-FIRST
    ...

def metric_token(on: str) -> str:
    s = str(on).strip()
    for cat in _known_categories():          # longest-prefix wins
        if s.startswith(cat + "_"):
            return _sanitize(s[len(cat) + 1:])
    return _sanitize(s)
```

**Longest-prefix match is mandatory** (blocking bug found in review): `QC` is a real
category *and* `QC_Tukey`, `QC_Count`, `QC_ICC`, `QC_MAD`, `QC_Occupancy`, `QC_SE`,
`QC_ZMax` are real categories. A naive "split on first `_`" turns
`on="QC_Tukey_NumOutliers"` into `Tukey_NumOutliers` instead of `NumOutliers`. Sorting
categories longest-first and testing `startswith(cat + "_")` picks `QC_Tukey` correctly.

`_sanitize` collapses/strip characters that are awkward in a column header (whitespace);
it leaves underscores intact — `parse_qualified_header` anchors on the known label suffix,
so an internally-underscored metric token stays parseable.

A regression test enumerates **all** public categories and asserts `metric_token`
produces the expected token for each `cat + "_X"` input.

## 6. `ModelFitter` ABC changes (`analysis/abc_/_model_fitter.py`)

Rename at the boundary; subclasses untouched.

- New `PrivateAttr` `_latest_fit_internal: pd.DataFrame` (member-keyed; plotting only).
- New cached property `_metric_token` → `metric_token(str(self.on))`.
- New `_rename_map()` → `{col: qualified_header(col, self._metric_token)
  for col in results.columns if isinstance(col, MeasurementInfo)}`. Non-member columns
  (the `groupby` keys) are left untouched.
- `analyze()`:
  1. build member-keyed `results` exactly as today (concat of `_apply2group_func`
     rows + `_post_fit_columns`);
  2. `self._latest_fit_internal = results` (member-keyed snapshot for plotting);
  3. `public = results.rename(columns=self._rename_map())`;
  4. `self._latest_model_scores = public`; `return public`.
- `_filter_for_plot()` reads **`_latest_fit_internal`** (member-keyed) instead of
  `_latest_model_scores`, so `show`/`dash` → `_predict_kwargs(row)` / `_hover_fields`
  (`row[LINEAR_LAG_MODEL.v]`) keep working with zero subclass edits.
- `results()` continues to return `_latest_model_scores` (the **qualified/public**
  frame) — an explicit, tested contract.

Why the rename is safe: `MeasurementInfo(str, Enum)` uses `str.__hash__`/`__eq__`
(no override), so `pd.concat`/`reset_index` preserve member-object columns and
`df.rename(columns={member: str})` retargets them correctly. `pl.from_pandas` in
`_cli_output_manager.py` runs *after* the rename, so on-disk `analysis.{csv,parquet}`
headers are plain qualified strings.

The three subclasses (`_linear_lag_model.py`, `_linear_cap_and_lag_model.py`,
`_log_growth_model.py`) need **no logic changes** — only docstring additions (Section 8).

## 7. `util/_measurement_outputs.py` — generic recognition

Replace static header-set membership with the enum's own `owns_header` /
`member_for_header`, handling all three schemes uniformly (fixes texture + growth):

- `_producer_column_groups` / `_headers_for_infos`: a column belongs to a producer's info
  class when `info.owns_header(column)` is true (was: `column in {m.value ...}`).
- `generate_output_key` / `_measurement_descriptions`: resolve each column by trying
  `member_for_header` across the public enums and using the resolved `member.desc`
  (was: a static `column → desc` dict, which cannot represent runtime metric/scale
  tokens). Cache the public-enum list; resolution is per-column and small.
- Non-model `MeasureFeatures`/QC/metadata producers keep working unchanged (their default
  `owns_header` is the old exact match). The `shared_infos=(MODEL_METRICS,)` handling for
  `ModelFitter` producers is retained; MODEL_METRICS simply now matches the infix scheme.

## 8. Docs

Follow the `TEXTURE` precedent — the Sphinx enum page shows **base labels** and the
docstring explains the runtime infix; the run-specific README shows the **actual** metric.

- **Enum docstrings** (currently absent on the three growth enums): add a class docstring
  to `LINEAR_LAG_MODEL`, `LINEAR_CAP_AND_LAG_MODEL`, `LOG_GROWTH_MODEL` explaining
  `<Category>_<metric>_<label>` with a concrete example. `measurements_ref.py`'s
  `_enum_page` renders `_lead_paragraphs(_strip_appended_table(__doc__))` above the static
  `rst_table(use_headers=True)`; `rst_table` keeps showing base labels (`LinearLagModel_v`)
  — matching how `TEXTURE` documents `Texture_<feature>-deg<axis>-scale<scale>`.
- **Fitter docstrings**: add a short "Output column naming" note to each of the three
  fitter classes. Kept as illustrative RST (no new doctests, per the runnable-doctest
  rule).
- **Stale prose fix:** `docs/source/explanation/notebooks/linear_softplus_model.ipynb`
  (~cell at line 154) says the mode is "recorded in the `LinearCapAndLagModel_mode`
  column"; update to the qualified form. (Markdown only; no code cell reads the column.)
- The generated `docs/source/measurements_ref/**` tree is git-ignored and rebuilt every
  build; no stale fixtures. CI docs build is not `-W`-strict.

## 9. CLI README (`_cli/_cli_readme_generator.py`)

The generator today documents **only** `MeasureFeatures` (iterates `pipeline._meas`) and
has zero model references. Add a new section, rendered only when
`self.pipeline.get_model()` is not `None`:

- `_generate_model_section()`: read `model = pipeline.get_model()`,
  `token = metric_token(str(model.on))`, and for every member of
  `model._measurement_infoclass` **and** `MODEL_METRICS`, emit a row
  `| \`{qualified_header(member, token)}\` | {member.desc} |`. Include a line stating the
  fitted metric (`model.on`) and the model class name.
- A metric-aware variant of `_generate_measurement_table` (or a small dedicated
  table builder) that formats `f"{category}_{token}_{label}"` instead of `str(member)`.

`pipeline.get_model()` returns `Optional[ModelFitter]`; `model.on` and
`model._measurement_infoclass` are both available at README-generation time.

Out of scope: documenting pipeline `filters` (edge correction, QC analyzers) in the
README.

## 10. Testing plan

- **Unit — token & parsing:** `metric_token` over all 41 categories (incl. the `QC_*`
  longest-prefix cases) and bare/arbitrary `on`; `qualified_header` round-trips through
  `parse_qualified_header`; `owns_header`/`member_for_header` for static, texture, and
  metric-qualified schemes; empty-metric legacy strings return `None`.
- **Unit — guardrail:** for every `MeasurementInfo` subclass, assert no member label is a
  `_`-anchored suffix of another member's label in the same enum (protects
  `parse_qualified_header`).
- **Update model tests:** `tests/unit/analysis/test_log_growth_model.py`,
  `test_double_softplus.py`, `test_linear_softplus.py` — assertions like
  `LOG_GROWTH_MODEL.R_FIT in results.columns` become the qualified header for the test's
  `on=` (e.g. `on="Shape_Area"` → `LogGrowthModel_Area_r`). Add a test that plotting
  (`show`/`dash`) still works after `analyze()` (exercises the `_latest_fit_internal`
  round-trip), and that `results()` returns qualified columns.
- **Update `tests/unit/util/test_measurement_outputs.py`:** add texture + growth-model
  cases proving `split_measurements` and `generate_output_key` now recognize both dynamic
  schemes (texture currently has zero coverage there).
- **README:** a test that a pipeline with a configured model renders the Models & Analysis
  section with the metric-qualified headers and the fitted-metric line; and that a
  model-less pipeline omits the section.
- **Unaffected (spot-checked in review):** `test_classification.py`,
  `test_classification_coverage.py`, `test_schema_public_api.py`, the tune
  annotation-coverage gate, and `to_json`/`from_json` — none reference output headers.

## 11. Blast radius (files touched)

- `src/phenotypic/schema/_measurement_info.py` — base `owns_header`/`member_for_header`,
  `qualified_header`, `parse_qualified_header` (+ mixin or module helpers).
- `src/phenotypic/schema/_texture.py` — texture-scheme `owns_header`/`member_for_header`.
- `src/phenotypic/schema/_linear_lag_model.py`, `_linear_cap_and_lag_model.py`,
  `_log_growth_model.py`, `_model_metrics.py` — metric-qualified scheme + enum docstrings.
- `src/phenotypic/analysis/abc_/_model_fitter.py` — boundary rename, `_latest_fit_internal`,
  `_metric_token`, `_filter_for_plot` repoint, `results()` contract.
- `src/phenotypic/analysis/_linear_lag_model.py`, `_linear_cap_and_lag_model.py`,
  `_log_growth_model.py` — docstring "Output column naming" note only.
- `src/phenotypic/util/_measurement_outputs.py` — `metric_token`, generic recognition.
- `src/phenotypic/_cli/_cli_readme_generator.py` — `_generate_model_section`.
- `docs/source/explanation/notebooks/linear_softplus_model.ipynb` — prose fix.
- Tests as in Section 10.

No consumer in `src` hardcodes `LinearLagModel_*`/`ModelMetrics_*` header strings outside
the schema and the three model files (verified), so the runtime blast radius is contained.

## 12. Non-goals

- Documenting pipeline `filters`/QC analyzers in the README.
- Any backward-compat alias for reading old un-qualified artifacts (graceful degrade only).
- Changing the fitters' math, parameters, or the `MeasurementInfo` member set (so the
  per-member classification tiers/gate are untouched).
