# Schema Module

Public, blessed API for PhenoTypic's measurement naming conventions.

## Layout

Files are grouped by the stage that emits their columns. Every subfolder is
private; `__init__.py` re-exports every class, and nothing else is public (see
"Importing schema classes" below).

```
schema/
  __init__.py      the only public surface: flat re-exports + legacy-name __getattr__
  _base/           MeasurementInfo, Entry, tiers, CATEGORIES, UNITS, REMBI, change notes
  _shared/         columns no single operation owns: OBJECT, GRID, METADATA_MATCH,
                   CURATION, ErrorCategory, RADIAL_EXPANSION
  _measure/        one file per MeasureFeatures schema: SIZE, SHAPE, BBOX, Color*, ...
  _analysis/       SetAnalyzer schemas: EDGE_CORRECTION
    _qc/           QUALITY_CHECK (shared QC columns) + the eight QUALITY_* enums
    _models/       the three growth models + MODEL_METRICS
  _metadata/       _image.py (IMAGE) + _experimental_tags/ (the eight owners)
  _experimental_tags.py  compatibility module at the pre-2026-09 path; it only
                   re-exports the owners and their one-release transition aliases,
                   and goes away with those aliases
```

The folders mirror the groups of the Measurements reference sidebar, but they
do not drive it: the docs generator groups pages by *operation*, from each
operation's declared schemas, so a file's folder never changes what the docs
show. Put a new schema in the folder of the stage that emits it.

- `MeasurementInfo` (`_base/_measurement_info.py`) — `str, Enum` base. Subclasses
  declare members as `Entry(label, desc, *, bio_desc="", image=None, tier=None,
  derivation_type=None, derives_from=None, rembi_module=None, categories=frozenset(),
  unit=None)` plus a
  `metric_family()` classmethod; the
  enum value is the family-prefixed header (e.g. `Size_Area`). `Entry` (a
  frozen dataclass, also in `_base/_measurement_info.py` and exported from the package)
  is the **only** legal member value — raw tuples raise `TypeError` at import.
  `desc` is the technical/algorithm description; **`bio_desc` is human-authored
  only** (a biological claim — never machine-generated, see the root `CLAUDE.md`
  guardrail); `image` is a path under `_assets/measurements/` rendered into the
  Sphinx docs; `tier`/`derivation_type`/`derives_from` drive the classification
  system (below). Per-member attrs: `.label`, `.desc`, `.bio_desc`, `.image`,
  `.pair`, `.METRIC_FAMILY`, plus classification: `.resolved_kind`, `.resolved_tier`,
  `.use_label`, `.use_badge`, categories: `.categories`, `.category_badges`
  (see "Measurement categories" below), and `.unit` (see "Units" below). Helpers: `get_labels()`, `get_headers()`,
  `rst_table()` (conditional Type/Categories/Biology/Image columns, suppressed
  when empty),
  `append_rst_to_doc()`. `metric_family()`/`.METRIC_FAMILY` were hard-renamed
  from the `category()` classmethod and `CATEGORY` property with no alias; a subclass that still defines
  either legacy name raises `TypeError` at class creation.
- Measurement-column enum modules under `_shared/`, `_measure/` and `_analysis/`
  (`_measure/_shape.py`, `_measure/_size.py`, …) — one `MeasurementInfo`
  subclass each, re-exported from `__init__.py`.
- `_metadata/_image.py` — `IMAGE`: framework-populated image bookkeeping
  (`UUID`, `ImageName`, `BitDepth`, …). It subclasses `MetadataInfo`; members
  render in the shared flat namespace (for example `Metadata_ImageName`). These
  are `image.metadata` accessor keys set by the pipeline, not user input.
- `_metadata/_experimental_tags/` — eight `MetadataInfo` subclasses (`GENETIC`, `SAMPLE`,
  `PLATE`, `CONDITION`, `CULTURE`, `ACQUISITION`, `EXPERIMENT`, and `STUDY`),
  one per file and re-exported from `__init__.py`. Every owner has
  `metric_family() == "Metadata"`, so `SAMPLE.BIO_REPLICATE` renders as
  `Metadata_BioReplicate`. The owner class, not a metric-family prefix, carries the
  semantic type. Use `phenotypic.sdk_.metadata_owner_for_header` or
  `metadata_member_for_header` for routing, membership, and set checks. A
  **recommended vocabulary, not a validator**: it standardizes `--metadata` CSV
  columns and `post/` ops, while arbitrary labels remain valid and become
  `Metadata_<Label>` with REMBI module `Uncategorized`.
- Exact previous-release stored headers such as `MetadataGenetic_Strain` remain
  readable permanently and normalize in memory to `Metadata_Strain`. Previous
  Python class names resolve through package `__getattr__` with a
  `DeprecationWarning` for one transition release, but are excluded from
  discovery and `__all__`. Do not use them in new code or examples.
- **Front-block column order** is the bio-semantic cluster order
  (Identity → Strain → Condition → Design; see `sdk_/_metadata_helpers.py`
  `_METADATA_OWNER_ORDER` / `order_measurement_columns`). REMBI
  (`rembi_module()` / `header_to_module`) is a *separate* provenance axis and does
  not drive column order.

## Importing schema classes: public interface only

Code outside this package reaches schema classes only through `phenotypic.schema`:

```python
from phenotypic.schema import SIZE, QUALITY_ICC, IdentityInfo  # correct
from phenotypic.schema._measure._size import SIZE              # wrong
from phenotypic.schema._base._tiers import IdentityInfo         # wrong
```

Every module file and folder here is private (underscore-prefixed), and the
layout has already moved once (2026-09, flat files into the stage folders above).
A private import couples the caller to a file path; the package namespace survives
a move. This applies to
`src/`, `tests/` and `docs/` (including Sphinx extensions).

"Public interface" means **what the package namespace exposes, which is wider than
`__all__`**. The tier bases (`IdentityInfo`, `QualityInfo`, `PrimaryMeasure`,
`DirectPhenotype`, …) are importable from `phenotypic.schema` but kept out of
`__all__`, so that `__all__`-driven discovery (the docs and README generators)
does not treat member-less bases as measurement families. Import them from the
package anyway. If something you need is not importable from the package, export
it from `__init__.py`; do not reach into the module.

Exceptions, and only these:

1. **Imports inside this package** (`from .._base._tiers import …`) are normal.
2. **A test that pins a private helper the package does not export** (`_classify`,
   `_BADGE_SPECS`, `_CATEGORY_BADGE_COLOR`, `_rst_cell_text`, `_VALID_KINDS`,
   `_CAMEL_BOUNDARY_RE`) may import that helper from its module. Only the helper:
   any public class the same test uses still comes from `phenotypic.schema`.
3. **A test whose subject is a private path**, such as
   `test_direct_experimental_tags_package_imports_have_the_same_transition_aliases`,
   which checks the deprecation alias on both `phenotypic.schema._experimental_tags`
   (the pre-2026-09 path, kept as a compatibility module) and
   `phenotypic.schema._metadata._experimental_tags`.
4. **Historical module-path strings in
   `_BackCompatUnpickler._MOVED_CLASSES`** (`_core/_image_parts/_image_io_handler.py`,
   mirrored in `tests/unit/sdk_/test_metadata_io.py`). These record where a class
   lived when an old pickle was written. They are persisted data, not imports:
   never update one to a current path and never delete one.

**Moving a schema file is safe for pickles, and it is the unpickler that makes it
so.** Enum members pickle by their class's `__module__`, which is the private file
the class lived in at write time (a pickle written before 2026-09 refers to
`phenotypic.schema._metadata.IMAGE`; one written after refers to
`phenotypic.schema._metadata._image.IMAGE`). `_BackCompatUnpickler` resolves any
class pickled under a `phenotypic.schema._*` path by name through the public
namespace (`_public_schema_class`), so no per-move row is needed. That guarantee
holds only while every schema class keeps a unique public name: **never rename a
public schema class, and never give two schema classes the same name**, without
adding a `_MOVED_CLASSES` row for the old `(module, name)`. Guarded by
`test_backcompat_unpickler_resolves_any_private_schema_path`.

## Measurement classification (kind + tier)

Every member resolves to a coarse **kind** and, for primary/derived measurements,
a trust **tier**. `_classify(member) -> (kind, tier)` in `_base/_measurement_info.py` is
the single resolver; members expose it via `.resolved_kind` (`"identity"` |
`"quality"` | `"primary"` | `"derived"`) and `.resolved_tier` (`1` | `2` | `3` |
`None`). The four kinds and the three primary tiers are explained for users on
the Tier System page, `docs/source/measurements_ref/tier_system.md` (hand-written,
tracked, and the one page in `measurements_ref/` the generator does not
overwrite) — keep that page and this section consistent.

An enum declares its classification **structurally**, by subclassing a member-less
base in `_base/_tiers.py` instead of `MeasurementInfo` directly (all exported from the
package):

- `IdentityInfo` → `kind="identity"` (design factors / locators; not outcomes).
- `QualityInfo` → `kind="quality"` (trust gates; never a biological claim).
- `DerivedMeasure` → `kind="derived"` (model/derived outputs; per-member tier).
- `PrimaryMeasure` → `kind="primary"`, **no fixed tier** (straddler base).
- `DirectPhenotype` / `DescriptiveTrait` / `DiscriminativeFeature` →
  `PrimaryMeasure` subclasses fixing `tier` 1 / 2 / 3.

**Straddlers** (enums whose members span tiers; no primary enum straddles since 0.20.0,
when SHAPE's Feret diameters moved to SIZE) subclass the neutral
`PrimaryMeasure`/`DerivedMeasure`, set a **class default tier** (via an overridden
`tier()` classmethod), and tag the minority members with an `Entry(tier=...)`
override. Resolution precedence in `_classify` (highest first): `derivation_type`
(`"diagnostic"`→quality; `"normalization"`→derived/tier-deferred;
`"parameterization"`→derived + **requires** `Entry(tier=...)`) > `Entry(tier=...)`
override + class `kind()` > class `kind()`/`tier()`. A primary member with no tier
(from any source) **raises** `ValueError` — every member must classify, enforced by
`tests/unit/schema/test_classification.py` + a coverage gate.

A "straddler" is a primary enum whose class `tier()` sets the default while some
members override it with `Entry(..., tier=N)`. No primary enum straddles since
0.20.0: `SHAPE(PrimaryMeasure)` overrides `tier()` to return `2` and every member
(circularity, eccentricity, the boundary distances, …) takes that default, while the
size magnitudes — area, perimeter, radii, axis lengths and the Feret diameters,
which were SHAPE's `tier=1` members before 0.20.0 — live in `SIZE(DirectPhenotype)`,
which resolves every member to tier 1 without tags. Derived enums (the growth
models) tag their parameterization members with `Entry(tier=1, derivation_type=...)`;
their diagnostic members carry no tier.

## Classification badges in the docs

`rst_table()` renders a **"Type"** column whose cells are sphinx-design
`:bdg-ref-{color}:` pills (one per member) linking to the Tier System page —
this is how the classification surfaces in the Measurements reference (the
Sphinx extension `docs/source/_extensions/measurements_ref.py` calls `rst_table`).

- `.use_label` → plain text (e.g. `"Direct phenotype (Tier 1)"`); a **frozen
  contract** (asserted in tests). Empty for non-tiered kinds. **Do not change
  its strings.**
- `.use_badge` → the RST badge string for the Type column; covers **every** kind
  (Identity/Quality/Derived included, unlike `use_label`), returns `""` only when
  a member fails to classify. Consumed by `_analysis/_qc/_quality_check.py` for QC column headers.
- `_BADGE_SPECS` (keyed `(tier, kind)`) maps to `(text, color, anchor)`; colors
  are sphinx-design semantic names (Tier1=`success`, Tier2=`primary`,
  Tier3=`warning`, Quality=`secondary`, Identity=`muted`, Derived=`info`). A
  `test_classification.py` unit test asserts every color is a real sphinx-design
  `SEMANTIC_COLORS`, so a typo'd color fails fast. The two anchor targets are MyST
  `(label)=` anchors in `tier_system.md` (`measurement-tiers` for tier badges,
  `measurement-classification` for the rest); badge xrefs use `reftype="any"`, so
  a typo'd **anchor** only warns at build time and the CI docs build
  (`uv run make html`, no `-W`) won't fail on it — a broken anchor would ship as a
  dead link, so verify anchor names against the md by hand when editing either.

Gotchas: keep the tier rows of `_USE_LABELS` and `_BADGE_SPECS` in sync (both
encode the same `(tier, kind)` taxonomy). Badge cells are inserted into the
list-table **unescaped** (they bypass `_rst_cell_text`) so the role renders — the
badge text must stay free of literal `|`. When adding a new tier/kind, update
`_base/_tiers.py`, both badge maps, and the Tier System page together.

## Measurement categories

A **category** is a curated, many-to-many grouping of columns that cuts across
metric families (`Size_Area` is in the **Size** family and in the *Starting
Metrics* category). Unlike kind/tier it makes **no trust claim**. Users read about
it in the intro of the generated Categories page (`measurement-categories` anchor,
`_CATEGORIES_INTRO` in `docs/source/_extensions/measurements_ref.py`). Keep that
intro and this section consistent.

- `CATEGORIES` (`_base/_categories.py`) is a closed, repo-defined `str` enum of
  `CategoryEntry(label, desc)`, and value == label. Members expose `.label`,
  `.desc`, `.display_name` (the label split on CamelCase), `.anchor`
  (`measurement-category-<label.lower()>`) and `.members()` (every public schema
  member carrying it, in `__all__` then member order). It deliberately does
  **not** subclass `MeasurementInfo`, because at least six discovery sites use
  `issubclass(x, MeasurementInfo)` and would pick it up as a measurement family.
  `_base/_categories.py` imports only stdlib at module level; `.members()` imports the
  schema package lazily.
- Tag a member with `Entry(..., categories=CATEGORIES.X)`. It takes a bare member
  or an iterable of members. A raw string is refused, even one equal to a
  member's value. Metadata owners (`MetadataInfo`) may not be categorized, which
  `test_no_metadata_member_carries_a_category` enforces.
- **`member.categories` is an unordered `frozenset`.** `CATEGORIES` members hash
  as their `str` value, and `str` hashing is randomized per process, so
  iterating the set directly gives a different order in each process once a
  column has two or more categories. Always order through
  `CATEGORIES.in_order(...)`, and iterate `CATEGORIES` itself for declaration
  order.
- To add a category, add a `CategoryEntry` member with a CamelCase label and a
  technical `desc`. Agents may write the `desc`, unlike an `Entry`'s `bio_desc`.
  Then tag the members and update the pin test in
  `tests/unit/schema/test_categories.py`. Three things follow automatically:
  the generated Categories docs page (`measurements_ref/categories/index.rst`,
  written by `docs/source/_extensions/measurements_ref.py`), the README
  section, and `deliverables/measurements_by_category/<label>.{csv,parquet}`.
- Badges: `member.category_badges` renders one
  `:bdg-ref-{_CATEGORY_BADGE_COLOR}-line:` outline pill per category, in
  `in_order` order, and each links to `CATEGORIES.X.anchor` on the Categories
  page. `rst_table()` gives them a "Categories" column after "Type", shown only
  when some member of the enum is categorized. `_CATEGORY_BADGE_COLOR = "info"`
  (the outline variant, so it reads as distinct from the solid Derived `info`
  Type pill). `dark` was rejected because pydata renders it `#222832` in both
  themes, which is invisible in dark mode.

## Units

Every non-metadata measurement column carries **exactly one unit** from a closed
list, so a reader with a pixels-per-mm value can convert any length or area
column with one recipe:
`df[SIZE.AREA] / px_per_mm ** SIZE.AREA.unit.length_power`. Output tables are
unchanged and stay in pixels; a unit is metadata only.

- `UNITS` (`_base/_units.py`) is a closed, repo-defined `str` enum of
  `UnitEntry(key, symbol, desc, *, length_power=0, input_power=0,
  per_time=False)`, and value == key (a unique lower-case key such as `"px2"`;
  equal values would alias members, which `@unique` refuses). Members expose
  `.symbol`, `.desc`, `.length_power`, `.input_power`, `.per_time`,
  `.is_input_dependent` and `.resolve(input_unit)`. Like `CATEGORIES`, it does
  **not** subclass `MeasurementInfo`, and the module imports only the stdlib.
- **`length_power`** is the power of length in the unit: 1 for `PX`, 2 for
  `PX2` and `INTENSITY_PX2`, 0 for anything scale-free, negative for a per-px
  rate. `NOT_A_QUANTITY` (labels, strings, flags, statuses, and numbers with
  no physical unit that must never be converted, such as `MODEL_METRICS.LOSS`)
  has `None`.
- **A unit on each member works the way a tier does.** A class sets its default
  with a `default_unit()` classmethod (base default `None`), and the exceptions
  carry `Entry(..., unit=UNITS.X)`, stored as `member.unit_override`.
  `member.unit` returns the override, else the class default. The classmethod is
  `default_unit`, not `unit`, because a classmethod would shadow the per-member
  `unit` property; a subclass that defines anything named `unit` raises
  `TypeError` at class creation. `Entry(unit=)` refuses a raw string, even one
  equal to a member's value.
- **Input-dependent units** (`OF_INPUT`, `OF_INPUT_SQUARED`,
  `OF_INPUT_PER_TIME`, and `PER_INPUT` with `input_power=-1`) tag columns whose
  unit depends on the column the producer ran on: fitted rates, model metrics,
  QC statistics, a sharpness that multiplies a difference of input values. The producer's
  `output_unit(member, on)`, next to `output_header`, resolves them through
  `UNITS.resolve(...)` into a concrete `ResolvedUnit(symbol, length_power)`:
  `v` fitted on a radius is `px/time`, length power 1. Reading `.length_power`
  on an unresolved input-dependent member raises `TypeError` pointing at
  `output_unit`, so `px_per_mm ** None` can never happen silently. A unitless
  input gives `"1/time"`, not `"/time"`, and a compound input is wrapped first
  (`(°/px)/time`). Unresolved, their
  `.symbol` is a readable placeholder (`input unit`, `(input unit)²`,
  `input unit/time`), never `""`, so nothing renders a fitted rate as unitless.
- **`output_unit` has one implementation.** `MeasureFeatures.output_unit` and
  `SetAnalyzer.output_unit` both call `resolve_output_unit` in
  `util/_measurement_outputs.py`, and **no producer overrides it**
  (`test_no_producer_overrides_output_unit`); `MeasurementProducer.output_unit`
  delegates to it. A static unit resolves to itself and ignores `on`. An
  input-dependent one resolves against `unit_of_column(on)`, the static unit of
  the member that column decodes to. `unit_of_column` returns `None` for an
  unknown header, metadata, and any column whose own unit is input-dependent,
  including the edge-corrected `EdgeCorrection_NewVal-<column>`, which is never
  resolved recursively (K0-17). `output_unit` then raises `ValueError` naming
  `on`, as it does for an `on` that is not a quantity (a label) or is `None`; a
  member with no unit raises naming the member. It never makes a unit up.
  Pinned by `tests/unit/util/test_output_units.py`.
- `None` means the member's unit has not been reviewed yet, and **the gate
  fails on it**: `test_every_non_metadata_member_has_a_unit` in
  `tests/unit/schema/test_units.py` covers every public non-metadata member,
  so a new schema without units fails CI rather than rendering "None" in the
  docs. **Metadata members (`MetadataInfo`) are user vocabulary and stay
  unit-less**; never tag one (`test_no_metadata_member_has_a_unit`).
- **A wrong tag that looks plausible** is caught by
  `tests/unit/measure/test_unit_scaling.py`: it measures synthetic scenes at
  two image scales (k = 2) and checks every size, shape, intensity,
  bounding-box, neighbour-distance, grid-spread, grid-regression and
  symmetric-zone column against `k ** length_power`. A failure means the tag
  is wrong; fix the tag, never the tolerance. Members it skips as
  `NOT_A_QUANTITY` must match an explicit allowlist there.
- **Every tag is pinned** by `tests/unit/schema/test_unit_tags_snapshot.py`, a
  literal `{"<class>.<member>": "<unit key>"}` mapping. It catches what scaling
  cannot: a swap between units of the same power (`RADIAN` for `DEGREE`), and
  any tag in a family no synthetic scene measures (orientation zones, the
  `GRID` intervals). **A tag change edits that line on purpose**; never
  regenerate the mapping wholesale.
- A unit describes the **value**, not the header: the texture `scale` in a
  `Texture_*-scale##` header is a parameter in pixels, and the value is still
  `UNITLESS`.
- **To add a unit, add one `UnitEntry` line** to `UNITS` with a technical
  `desc` (agents may write it), then update the member-order pin in
  `tests/unit/schema/test_units.py`.

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

Emission (write side) lives with the enum or as shared functions in `_base/_measurement_info.py`:

- **static** — the header *is* `member.value` (`Size_Area`); base default, no override.
- **metric_qualified** — `{family}_{metric}_{label}` (e.g. `LinearLagModel_Area_v`):
  `qualified_header(member, token)` / `parse_qualified_header(info_cls, column)`; the enum
  sets `header_scheme() -> "metric_qualified"` (the 3 growth models + `MODEL_METRICS`).
  The token comes from `metric_token(on)` in `util/_measurement_outputs.py`
  (strips the longest known metric-family prefix from `self.on`).
- **texture** — `{family}_{label}-deg###-scale##` / `-avg-scale##`:
  `TEXTURE.get_headers(scale, matrix_name)` plus a `member_for_header` regex override.

**Invariant:** the format must be invertible — `parse(emit(member, token)) == (token,
member)`. `metric_qualified` anchors on the metric-family prefix + the known member-label
suffix, so a guardrail in `tests/unit/schema/test_dynamic_headers.py` asserts no label is
a `_`-suffix of another. Emission (in the producer) and recognition (on the enum) live in
two files that must agree; the round-trip test keeps them honest.

**The Measurements reference shows the header written, not the enum value.** Every table
producer (`MeasureFeatures`, `SetAnalyzer`) has a classmethod
`output_header(member, on=None)` and `output_header_placeholders()`. The default is
`member.value` and no placeholders; a producer whose headers differ overrides both, and
**its emitter must call `output_header`** (or the enum helper it wraps) so docs and table
share one formatter. Overrides today: `QualityCheck` (`QUALITY_CHECK.header`,
`QC_<name>_<label>`; only the abstract base renders `<name>`, and a concrete check
without `name` raises), `ModelFitter`
(`qualified_header`, `<metric>`), `EdgeCorrector` (`<value>-<column>`), `MeasureTexture`
(`TEXTURE.header`, `<direction>` and `<x>` for the scale). Placeholders are `<lowercase>`
tokens; each maps to an RST sentence, with an example, printed under **Placeholders** on
the operation's page. The registry is `phenotypic.util.measurement_producers()`, and
`tests/unit/util/test_output_headers.py` runs the producers to prove the documented
names are the written ones. The generator
(`docs/source/_extensions/measurements_ref.py`) fails the build for a public schema that
no producer declares and that is not listed in its `_SHARED_ONLY` or
`_NOT_IN_OUTPUT_TABLES`; members never written go in `_OMITTED_MEMBERS`.

### Adding a new dynamic scheme

1. Pick an invertible format — the member label must be recoverable without the token.
2. Add an emission helper — co-locate on the enum (like `get_headers`) or a shared func;
   reuse `qualified_header`/`parse_qualified_header` if the shape is the
   `{family}_{token}_{label}` infix (then you only set `header_scheme()`, no parser).
3. Override `member_for_header` on the enum (and set `header_scheme()`). `owns_header`
   is inherited.
4. In the producer, name columns via the helper and declare
   `_measurement_infoclass = <enum>` — that one attribute wires
   split/output-key/recognition, CLI README documentation and a page in the
   Measurements reference. When parameters
   enable or disable individual schemas, override
   `MeasureFeatures.get_measurement_infoclasses()` to return the active subset
   (the reference finds the enabling boolean parameter by construction).
5. Override the producer's `output_header` and `output_header_placeholders` so the
   reference shows the pattern and defines each placeholder.

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

Downstream users import headers directly:

    from phenotypic.schema import SHAPE, MeasurementInfo
    SHAPE.get_headers()  # ['Shape_Circularity', 'Shape_Eccentricity', ...]

Conventions: one class per file (or per file under a grouping subpackage like
`_experimental_tags/`); bodies are pure data and inherit their metric family; import **only**
stdlib and the sibling base (no other `phenotypic` imports) to keep the package
import-light and preserve the package load-order trick in
`phenotypic/__init__.py` (`abc_` imports the stdlib-only base from here before
`sdk_.constants_` needs it). Metadata-naming enums (`IMAGE` + the semantic
owners) live here because they name `Metadata_*` columns/keys.
Framework-config enums that are *not* about naming columns/keys (e.g.
`GAMMA_ENCODINGS`, `PIPE_STATUS`, `IMAGE_MODE`) stay in `sdk_/constants_.py`.
