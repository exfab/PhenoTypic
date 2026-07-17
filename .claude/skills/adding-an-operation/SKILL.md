---
name: adding-an-operation
description: Conventions for adding or editing a PhenoTypic operation, parameter, closed-value-set field, or tunable numeric field. Use when authoring/modifying any operation under detect/, enhance/, refine/, grid/, correction/, measure/, or an analyzer — covers pydantic field declaration, Enum/Literal normalization, MeasurementInfo/ConstantLabels choice, and the tune annotation-coverage gate.
---

# Adding or editing a PhenoTypic operation

Operations and analyzers are **pydantic v2 models** rooted at `BaseOperation`.
There is no hand-written `__init__`; construction is **keyword-only**; invalid
input raises `pydantic.ValidationError`. Parameters are **annotated class-level
fields**. Put input normalization and guards in a `field_validator`, never an
`__init__`. Raw-array params use `NdArrayField`; operation-valued params use
`OperationField` (both in `phenotypic.sdk_.typing_`). Algorithm bodies
(`_operate`/`apply`/`measure`) are unchanged by the model migration.

For *which ABC to subclass* and ABC-authoring rules (instance-method `_operate`,
no-required-args for `from_json`, the tuple round-trip `field_validator`,
docstring layout), see [`abc_/CLAUDE.md`](../../../src/phenotypic/abc_/CLAUDE.md).

## Closed value sets

Public parameters with a closed set of values:

- Type as `EnumType | Literal["a", "b", ...]`, normalize in a
  `field_validator(mode="before")` with `value = EnumType(value)`, and use only
  enum members internally.
- Define the `Literal` alias once as a `TypeAlias` and reuse it.
- If both an `Enum` and a `Literal` exist, add a test asserting their values
  match (`set(get_args(MyLiteral)) == {m.value for m in MyEnum}`). When the
  `Literal` intentionally covers only a subset of the enum, assert with
  `issubset` and document the partial coverage in the test docstring.
- **Never** accept bare `str` for a closed set, never propagate raw strings past
  the boundary, never derive a `Literal` from a runtime expression.

**When the closed set needs user-visible documentation, prefer `MeasurementInfo`
/ `ConstantLabels`.** Each member is a `(label, description)` pair whose
description is accessible to callers; override the `category()` classmethod (and
optionally `__new__` for bare-label values) per the existing convention.

- The `MeasurementInfo` base and the per-feature **measurement-column** enums
  (`SHAPE`, `SIZE`, …) live in the public `phenotypic.schema` package — see
  [`schema/CLAUDE.md`](../../../src/phenotypic/schema/CLAUDE.md).
- Framework-config constant enums (`GAMMA_ENCODINGS`, `PIPE_STATUS`, `METADATA`)
  stay in `phenotypic.sdk_.constants_`. Don't alter these enums' bespoke
  coercion (e.g. `_GAMMA_COERCE`) just to satisfy generic `MyEnum(value)`
  normalization — it's intentional.

## Measurement docstrings: `MeasureFeatures` vs `MeasurementInfo`

When authoring a `measure/` operation, the documentation is **split** across the
op and its schema enum — don't overlap them:

- The **`MeasureFeatures` op docstring** explains **what its parameters mean** and
  gives a **high-level overview** of the measurements it emits (what the op does,
  when to use it). Stay at overview altitude; don't restate each output column.
- The **`MeasurementInfo` enum members** carry the **detailed per-column
  explanation of the measurements themselves** — what each value is, how it is
  computed, and **how to read the output** (prefixed column header, units, range).
  Each member's `desc` is the canonical per-column documentation.

This split is load-bearing: the deliverables `README.md` generator
(`_cli/_cli_readme_generator.py`) maps each configured `MeasureFeatures` op to its
`MeasurementInfo` enum(s) and emits every member's `desc` verbatim as the public
column reference — so the enum `desc` is what end users read. Author only `label`
and `desc` on a member; **never** author `bio_desc` (human-only). Also kept in the
root `CLAUDE.md` (Code Style + Gotchas) and the contributing guide.

### Fixed measurement schemas are explicit

For a fixed set of output columns, declare every member directly as
`Entry(label, desc)`, following the neighboring files in `phenotypic.schema`:

```python
class MY_MEASUREMENTS(DescriptiveTrait):
    @classmethod
    def category(cls) -> str:
        return "MyMeasurements"

    VALUE = Entry(
        "Value",
        "Complete per-column calculation and interpretation. Reported in pixels.",
    )
```

Keep the complete public description beside its member, including the exact
calculation, selector or region, units, range or sign convention, and missing-value
conditions. Repeat zone- or variant-specific wording where necessary. Do **not**
generate fixed member descriptions from shared dictionaries, templates, formatter
functions, comprehensions, or metaprogramming. Avoiding repetition is less important
than making each public column independently readable and reviewable. Reserve dynamic
header generation for schemas whose columns are genuinely runtime-parameterized.

**For type-only enforcement** of a closed set with no documentation surface (CLI
dispatch keys, internal mode flags), a `Literal[...]` `TypeAlias` in
`phenotypic.sdk_.typing_` is sufficient — no Enum needed. Examples:
`FootprintShape`, `DetectMode`, `ExecutionMode`, `ImageTypeName`,
`ProcessingStatus`, `NormOut`, `InputLayer`.

Pair an Enum with a `Literal` alias **only** when both forms are used at boundary
code (string-typed external input + enum-typed internal storage). Canonical
alignment test:
`tests/unit/sdk_/test_io_constants.py::TestEnumLiteralAlignment`.

**Parameterized strings are not enumerations:** keep the template as a private
`Final[str]` and expose a typed render function whose parameters are the public
API.

## Output-range guards use `norm`, never `clip: bool`

Any operation that clamps or normalizes its output declares `norm: NormOut`
(from `phenotypic.sdk_.typing_`) by inheriting `NormalizedOutputMixin`, and
applies it via `self._apply_norm(arr)`:

- `"clip"` (default) saturates — the identity in-range, so absolute intensity
  and cross-batch comparability survive.
- `"rescale"` remaps the full observed range onto [0, 1]. Ordering survives,
  absolute scale does not: a single specular highlight sets the max.
- `None` passes through — the escape hatch GAT regions and `CompositeEnhance`
  depend on, and what `NormControlMixin._disable_normalization` sets.

**`"rescale"` divides out any purely multiplicative `gain`.** `rescale_intensity`
maps `[k·min, k·max]` onto [0, 1], so a post-curve factor `k` cancels exactly.
With `norm="rescale"`, `ContrastGamma` and `ContrastLog` give the same output for
`gain=1` and `gain=3` (max abs difference of order 1e-7, i.e. float32 rounding); the
knob is a no-op. `ContrastSigmoid` escapes this because its `gain` sits *inside* the
exponent rather than after the curve. Before adding a multiplicative parameter to
an operation that can rescale, check the parameter still does something.

A bare `clip: bool` cannot express `"rescale"`, and the attribute name is claimed
by `NormControlMixin`, which duck-types on it. Removed in 0.18.0 — the mixin's
`_reject_legacy_clip` validator turns a stale `clip=` key into a migration error
instead of pydantic's opaque "Extra inputs are not permitted".

Skimage passthrough parameters that merely *sound* like range guards
(`rescale_sigma`, forwarded to `denoise_wavelet`) are **not** `norm` and stay as
they are.

Inside a GAT region, declare the inert value in `_GAT_DEFER_VALUES` (a
`ClassVar[dict[str, Any]]` mapping attribute name to inert value):
`{"norm": None, "rescale_sigma": False}`. This is not a rounding concern — the
stabilized signal runs to ~30, so leaving *either* policy active drives the
inverse transform to all zeros.

## Cross-cutting fields append; they do not frontload

Pydantic collects fields in reverse-MRO order, so a mixin's field lands *before*
the operation's own parameters — wrong for both `model_json_schema()` and
`to_json()`. A field-append mixin fixes this in `__pydantic_init_subclass__`:

```python
@classmethod
def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
    super().__pydantic_init_subclass__(**kwargs)  # cooperative: BaseOperation's runs
    fields = cls.__pydantic_fields__
    if "norm" in fields and list(fields)[-1] != "norm":
        fields["norm"] = fields.pop("norm")
        cls.model_rebuild(force=True)
```

`TuneSpec` metadata survives the forced rebuild. With two such mixins the order is
deterministic — each hook calls `super()` *before* popping, so the mixin **earliest
in the MRO ends up last**. `class ContrastGamma(InputLayerMixin,
NormalizedOutputMixin, ContrastAdjustment)` yields field order
`['gamma', 'gain', 'norm', 'input_layer']`.

Canonical: `sdk_/mixin/_normalized_output_mixin.py`, `sdk_/mixin/_input_layer_mixin.py`.

## Tunable numeric fields — the annotation-coverage gate

A new numeric (`int`/`float`) field on any `detect/`, `enhance/`, `refine/`,
`grid/`, or `correction/` operation is pulled into the annotation-coverage gate
(`tests/unit/tune/test_annotation_coverage.py`) and **must be covered** — by a
`TuneSpec` or a pydantic `Field` bound — or CI fails. Pick the annotation by
intent, not just to pass the gate:

- **Has a fixed, sensible search window** →
  `Annotated[float, TuneSpec(low, high, log=...)]`.
- **Should never be tuned** (scene-derived, structural) →
  `TuneSpec(tunable=False)`.
- **Worth tuning but the range depends on runtime context** (e.g. a filter
  cutoff on a measured value whose scale varies by feature) → a **bare
  `TuneSpec()`** (tunable, no `low`/`high`). It satisfies the gate and declares
  intent-to-tune; auto-search deliberately surfaces it as range-less
  (`_resolve_tune_spec` → `Excluded("non_numeric")`) instead of fabricating a
  window, and the concrete range is supplied per-run in the tune spec.

Don't reach for `tunable=False` just to silence the gate when the field is
genuinely a knob. Canonical: `refine/_remove_by_feature.py` (`RemoveByFeature`,
`min_value`/`max_value`).
