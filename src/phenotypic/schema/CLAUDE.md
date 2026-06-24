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
  (`UUID`, `ImageName`, `BitDepth`, …). `category() == "Metadata"`, so members
  render as `Metadata_<Label>`. These are `image.metadata` accessor keys set by
  the pipeline, not user input.
- `_experimental_tags/` — seven `MeasurementInfo` subclasses
  (`GENETIC_METADATA`, `SAMPLE_METADATA`, `PLATE_METADATA`,
  `CONDITION_METADATA`, `INCUBATION_METADATA`, `ACQUISITION_METADATA`,
  `EXPERIMENT_METADATA`), one per file, re-exported from `__init__.py`. All
  return `category() == "Metadata"`, so the grouping is organizational and every
  member shares the `Metadata_` namespace (`SAMPLE_METADATA.REPLICATE` →
  `Metadata_Replicate`). A **recommended vocabulary, not a validator**: it
  standardizes `--metadata` CSV columns + `post/` ops but arbitrary columns are
  still accepted.

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
  contract** (asserted in tests, consumed by `_quality_check.py`). Empty for
  non-tiered kinds. **Do not change its strings.**
- `.use_badge` → the RST badge string for the Type column; covers **every** kind
  (Identity/Quality/Derived included, unlike `use_label`), returns `""` only when
  a member fails to classify.
- `_BADGE_SPECS` (keyed `(tier, kind)`) maps to `(text, color, anchor)`; colors
  are sphinx-design semantic names (Tier1=`success`, Tier2=`primary`,
  Tier3=`warning`, Quality=`secondary`, Identity=`muted`, Derived=`info`). A
  `test_classification.py` unit test asserts every color is a real sphinx-design
  `SEMANTIC_COLOR`, so a typo'd color fails fast. The two anchor targets are MyST
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

Downstream users import headers directly:

    from phenotypic.schema import SHAPE, MeasurementInfo
    SHAPE.get_headers()  # ['Shape_Area', 'Shape_Perimeter', ...]

Conventions: one class per file (or per file under a grouping subpackage like
`_experimental_tags/`); bodies are pure data + `category()`; import **only**
stdlib and the sibling base (no other `phenotypic` imports) to keep the package
import-light and preserve the package load-order trick in
`phenotypic/__init__.py` (`abc_` imports the stdlib-only base from here before
`sdk_.constants_` needs it). Metadata-naming enums (`METADATA` + the
experimental tags) live here because they name `Metadata_*` columns/keys.
Framework-config enums that are *not* about naming columns/keys (e.g.
`GAMMA_ENCODINGS`, `PIPE_STATUS`, `IMAGE_MODE`) stay in `sdk_/constants_.py`.
