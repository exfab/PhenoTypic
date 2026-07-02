# Schema Module

Public, blessed API for PhenoTypic's measurement naming conventions.

- `MeasurementInfo` (`_measurement_info.py`) — `str, Enum` base. Subclasses
  declare members as `Entry(label, desc, *, bio_desc="", image=None, tier=None,
  derivation_type=None, derives_from=None)` plus a `category()` classmethod; the
  enum value is the category-prefixed header (e.g. `Shape_Area`). `Entry` (a
  frozen dataclass, also in `_measurement_info.py` and exported from the package)
  is the **only** legal member value — raw tuples raise `TypeError` at import.
  `desc` is the technical/algorithm description; **`bio_desc` is human-authored
  only** (a biological claim — never machine-generated, see the root `CLAUDE.md`
  guardrail); `image` is a path under `_assets/measurements/` rendered into the
  Sphinx docs; `tier`/`derivation_type`/`derives_from` drive the classification
  system (below). Per-member attrs: `.label`, `.desc`, `.bio_desc`, `.image`,
  `.pair`, `.CATEGORY`, plus classification: `.resolved_kind`, `.resolved_tier`,
  `.use_label`, `.use_badge`. Helpers: `get_labels()`, `get_headers()`,
  `rst_table()` (conditional Type/Biology/Image columns, suppressed when empty),
  `append_rst_to_doc()`.
- 32 measurement-column enum modules (`_shape.py`, `_size.py`,
  `_color_lab.py`, …) — one `MeasurementInfo` subclass each, re-exported from
  `__init__.py`.
- `_metadata.py` — `METADATA`: framework-populated image bookkeeping
  (`UUID`, `ImageName`, `BitDepth`, …). `category() == "MetadataImage"`, so members
  render as `MetadataImage_<Label>` (e.g. `MetadataImage_ImageName`). These are
  `image.metadata` accessor keys set by the pipeline, not user input.
- `_experimental_tags/` — eight `MeasurementInfo` subclasses
  (`GENETIC_METADATA`, `SAMPLE_METADATA`, `PLATE_METADATA`,
  `CONDITION_METADATA`, `CULTURE_METADATA`, `ACQUISITION_METADATA`,
  `EXPERIMENT_METADATA`, `STUDY_METADATA`), one per file, re-exported from
  `__init__.py`. Each returns its own per-topic Scheme-B category
  (`MetadataGenetic`, `MetadataSample`, `MetadataPlate`, `MetadataCondition`,
  `MetadataCulture`, `MetadataAcquisition`, `MetadataExperiment`, `MetadataStudy`),
  so members render as `Metadata<Topic>_<Label>`
  (`SAMPLE_METADATA.BIO_REPLICATE` → `MetadataSample_BioReplicate`). All prefixes
  share the `Metadata` column family — recognize any of them via
  `phenotypic.sdk_.is_metadata_header` rather than a bare `"Metadata_"` literal.
  A **recommended vocabulary, not a validator**: it standardizes `--metadata` CSV
  columns + `post/` ops but arbitrary columns are still accepted (unknown labels
  fall back to a generic `Metadata_<Label>` → REMBI `Uncategorized`).

## Measurement classification (kind + tier)

Every member resolves to a coarse **kind** and, for primary/derived measurements,
a trust **tier**. `_classify(member) -> (kind, tier)` in `_measurement_info.py` is
the single resolver; members expose it via `.resolved_kind` (`"identity"` |
`"quality"` | `"primary"` | `"derived"`) and `.resolved_tier` (`1` | `2` | `3` |
`None`). The four kinds and the three primary tiers are explained for users in
`docs/source/explanation/measurement_classification_system.md` — keep that page
and this section consistent.

An enum declares its classification **structurally**, by subclassing a member-less
base in `_tiers.py` instead of `MeasurementInfo` directly (all exported from the
package):

- `IdentityInfo` → `kind="identity"` (design factors / locators; not outcomes).
- `QualityInfo` → `kind="quality"` (trust gates; never a biological claim).
- `DerivedMeasure` → `kind="derived"` (model/derived outputs; per-member tier).
- `PrimaryMeasure` → `kind="primary"`, **no fixed tier** (straddler base).
- `DirectPhenotype` / `DescriptiveTrait` / `DiscriminativeFeature` →
  `PrimaryMeasure` subclasses fixing `tier` 1 / 2 / 3.

**Straddlers** (enums whose members span tiers, e.g. `SHAPE`) subclass the neutral
`PrimaryMeasure`/`DerivedMeasure`, set a **class default tier** (via an overridden
`tier()` classmethod), and tag the minority members with an `Entry(tier=...)`
override. Resolution precedence in `_classify` (highest first): `derivation_type`
(`"diagnostic"`→quality; `"normalization"`→derived/tier-deferred;
`"parameterization"`→derived + **requires** `Entry(tier=...)`) > `Entry(tier=...)`
override + class `kind()` > class `kind()`/`tier()`. A primary member with no tier
(from any source) **raises** `ValueError` — every member must classify, enforced by
`tests/unit/schema/test_classification.py` + a coverage gate.

Example straddler: `class SHAPE(PrimaryMeasure)` overrides `tier()` to return `2`
(form descriptors default to Descriptive trait); its size-magnitude members carry
`Entry(..., tier=1)` (e.g. `AREA`, `PERIMETER`, radii, Feret diameters) so they
resolve to Direct phenotype while `CIRCULARITY`/`ECCENTRICITY` take the class
default of 2.

## Classification badges in the docs

`rst_table()` renders a **"Type"** column whose cells are sphinx-design
`:bdg-ref-{color}:` pills (one per member) linking to the explanation page —
this is how the classification surfaces in the Measurements reference (the
Sphinx extension `docs/source/_extensions/measurements_ref.py` calls `rst_table`).

- `.use_label` → plain text (e.g. `"Direct phenotype (Tier 1)"`); a **frozen
  contract** (asserted in tests). Empty for non-tiered kinds. **Do not change
  its strings.**
- `.use_badge` → the RST badge string for the Type column; covers **every** kind
  (Identity/Quality/Derived included, unlike `use_label`), returns `""` only when
  a member fails to classify. Consumed by `_quality_check.py` for QC column headers.
- `_BADGE_SPECS` (keyed `(tier, kind)`) maps to `(text, color, anchor)`; colors
  are sphinx-design semantic names (Tier1=`success`, Tier2=`primary`,
  Tier3=`warning`, Quality=`secondary`, Identity=`muted`, Derived=`info`). A
  `test_classification.py` unit test asserts every color is a real sphinx-design
  `SEMANTIC_COLORS`, so a typo'd color fails fast. The two anchor targets are MyST
  `(label)=` anchors in the explanation md (`measurement-tiers` for tier badges,
  `measurement-classification` for the rest); badge xrefs use `reftype="any"`, so
  a typo'd **anchor** only warns at build time and the CI docs build
  (`uv run make html`, no `-W`) won't fail on it — a broken anchor would ship as a
  dead link, so verify anchor names against the md by hand when editing either.

Gotchas: keep the tier rows of `_USE_LABELS` and `_BADGE_SPECS` in sync (both
encode the same `(tier, kind)` taxonomy). Badge cells are inserted into the
list-table **unescaped** (they bypass `_rst_cell_text`) so the role renders — the
badge text must stay free of literal `|`. When adding a new tier/kind, update
`_tiers.py`, both badge maps, and the explanation page together.

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

Downstream users import headers directly:

    from phenotypic.schema import SHAPE, MeasurementInfo
    SHAPE.get_headers()  # ['Shape_Area', 'Shape_Perimeter', ...]

Conventions: one class per file (or per file under a grouping subpackage like
`_experimental_tags/`); bodies are pure data + `category()`; import **only**
stdlib and the sibling base (no other `phenotypic` imports) to keep the package
import-light and preserve the package load-order trick in
`phenotypic/__init__.py` (`abc_` imports the stdlib-only base from here before
`sdk_.constants_` needs it). Metadata-naming enums (`METADATA` + the
experimental tags) live here because they name `Metadata<Topic>_*` columns/keys.
Framework-config enums that are *not* about naming columns/keys (e.g.
`GAMMA_ENCODINGS`, `PIPE_STATUS`, `IMAGE_MODE`) stay in `sdk_/constants_.py`.
