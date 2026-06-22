# Measurement Classification System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a queryable tier/kind classification (Direct phenotype / Descriptive trait / Discriminative feature, plus Identity / Quality / Derived) to every `phenotypic.schema` measurement column, via an intermediate-class spine + per-`Entry` overrides, with a CI coverage gate and a user-facing docs page — **without renaming any column**.

**Architecture:** Member-less `MeasurementInfo` subclasses (`DirectPhenotype`, `DescriptiveTrait`, `DiscriminativeFeature`, `IdentityInfo`, `QualityInfo`, `DerivedMeasure`) carry `kind()`/`tier()` classmethods. Tier-uniform enums re-parent to the matching base; the straddlers (`SHAPE`, the three growth-model enums, `EDGE_CORRECTION`) keep their members and tag the minority via new optional `Entry` fields (`tier`, `derivation_type`, `derives_from`). A single `resolved_kind`/`resolved_tier` accessor is the one read path. Enum **values** (header strings) never change, so the operation layer, goldens, and serialization are untouched.

**Tech Stack:** Python 3, stdlib `enum`/`dataclasses`, pytest, Sphinx (MyST markdown), `uv` runner.

## Global Constraints

- **`uv` is the sole runner.** Tests: `uv run pytest ...`; types: `uv run mypy src/phenotypic`; lint: `uv run ruff check --fix`. Never bare `python`/`pip`.
- **`schema/` is import-light:** modules import **only** stdlib and the sibling `_measurement_info` / `_tiers` base. No other `phenotypic` imports (preserves the package load-order trick in `phenotypic/__init__.py`).
- **No column renames.** Enum `.value` (e.g. `Shape_Area`) is invariant. This is the Path 3 invariant — any change that alters a header string is out of scope.
- **`bio_desc` is human-authored only.** Do not author or auto-fill `bio_desc`. The mechanical tier/kind is fine to auto-assign; biological caveat prose is not.
- **`derives_from` is a string token** (e.g. `"SIZE"`), never a typed class reference (import-light rule).
- **Google-style docstrings** on new public classes/methods.
- **One class per file** convention in `schema/`; the intermediate bases are the sole exception (grouped in `_tiers.py`).
- Spec: `docs/superpowers/specs/measurement-classification-system/2026-06-18-measurement-classification-system-design.md`.

---

### Task 1: Foundation — `Entry` fields, base classmethods, resolution, intermediate classes

**Files:**
- Modify: `src/phenotypic/schema/_measurement_info.py`
- Create: `src/phenotypic/schema/_tiers.py`
- Modify: `src/phenotypic/schema/__init__.py`
- Modify: `src/phenotypic/util/_measurement_outputs.py` (defensive guard)
- Test: `tests/unit/schema/test_classification.py`

**Interfaces:**
- Produces:
  - `Entry(label, desc, *, bio_desc="", image=None, tier=None, derivation_type=None, derives_from=None)` — `tier ∈ {1,2,3,None}`, `derivation_type ∈ {"parameterization","normalization","diagnostic",None}`, `derives_from: str | None`.
  - On `MeasurementInfo`: classmethods `kind() -> str | None` (default `None`), `tier() -> int | None` (default `None`); instance attrs `tier_override`, `derivation_type`, `derives_from`; properties `resolved_kind -> str` and `resolved_tier -> int | None` (raise `ValueError` for an unclassified member).
  - In `_tiers.py`: `IdentityInfo`, `QualityInfo`, `PrimaryMeasure`, `DirectPhenotype` (tier 1), `DescriptiveTrait` (tier 2), `DiscriminativeFeature` (tier 3), `DerivedMeasure` — all member-less.
  - Valid kinds: `"identity" | "quality" | "primary" | "derived"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/schema/test_classification.py
import pytest
from phenotypic.schema import Entry
from phenotypic.schema._tiers import (
    DirectPhenotype, DescriptiveTrait, DiscriminativeFeature,
    IdentityInfo, QualityInfo, DerivedMeasure,
)


def _make(cls, **entries):
    # Build a throwaway enum subclass for testing resolution.
    from phenotypic.schema import MeasurementInfo  # noqa: F401
    members = {name: Entry(name, "d", **kw) for name, kw in entries.items()}
    return type("T", (cls,), {"category": classmethod(lambda c: "T"), **members})


def test_tier_bases_are_memberless():
    for base in (DirectPhenotype, DescriptiveTrait, DiscriminativeFeature,
                 IdentityInfo, QualityInfo, DerivedMeasure):
        assert list(base) == []


def test_class_level_tier_resolution():
    E = _make(DiscriminativeFeature, A={})
    assert E.A.resolved_kind == "primary"
    assert E.A.resolved_tier == 3


def test_entry_override_beats_class():
    E = _make(DescriptiveTrait, A={}, B={"tier": 1})
    assert E.A.resolved_tier == 2          # class default
    assert E.B.resolved_tier == 1          # Entry override


def test_diagnostic_resolves_to_quality():
    E = _make(DerivedMeasure, A={"derivation_type": "diagnostic"})
    assert E.A.resolved_kind == "quality"
    assert E.A.resolved_tier is None


def test_normalization_is_covered_with_deferred_tier():
    E = _make(DerivedMeasure, A={"derivation_type": "normalization", "derives_from": "SIZE"})
    assert E.A.resolved_kind == "derived"
    assert E.A.resolved_tier is None


def test_unclassified_primary_member_raises():
    E = _make(DerivedMeasure, A={})        # derived, no tier, no derivation_type
    with pytest.raises(ValueError):
        _ = E.A.resolved_tier


def test_entry_validates_tier_and_derivation_type():
    with pytest.raises(ValueError):
        Entry("x", "d", tier=4)
    with pytest.raises(ValueError):
        Entry("x", "d", derivation_type="bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_classification.py -q`
Expected: FAIL — `ImportError`/`AttributeError` (`_tiers` missing, `resolved_tier` undefined).

- [ ] **Step 3: Extend `Entry` and `MeasurementInfo` in `_measurement_info.py`**

In the `Entry` dataclass, add three keyword-only fields after `image`:

```python
    label: str
    desc: str = ""
    _: KW_ONLY
    bio_desc: str = ""
    image: str | None = None
    tier: int | None = None
    derivation_type: str | None = None
    derives_from: str | None = None
```

Add the allowed set near the top of the module (after imports):

```python
_DERIVATION_TYPES: Final = frozenset({"parameterization", "normalization", "diagnostic"})
_VALID_KINDS: Final = frozenset({"identity", "quality", "primary", "derived"})
```

Extend `Entry.__post_init__` with validation (append to the existing body):

```python
        if self.tier is not None and self.tier not in (1, 2, 3):
            raise ValueError("Entry.tier must be 1, 2, 3, or None")
        if self.derivation_type is not None and self.derivation_type not in _DERIVATION_TYPES:
            raise ValueError(
                f"Entry.derivation_type must be one of {sorted(_DERIVATION_TYPES)} or None"
            )
        if self.derives_from is not None and not isinstance(self.derives_from, str):
            raise TypeError("Entry.derives_from must be a string token or None")
```

In `MeasurementInfo`, add the bare annotations (next to the existing `label`/`desc`/...):

```python
    tier_override: int | None
    derivation_type: str | None
    derives_from: str | None
```

Add default classmethods (next to `category`):

```python
    @classmethod
    def kind(cls) -> str | None:
        """Coarse classification kind, or None until assigned by a base class."""
        return None

    @classmethod
    def tier(cls) -> int | None:
        """Trust tier (1/2/3) for primary measurements, or None."""
        return None
```

In `__new__`, after `obj.image = entry.image`, add:

```python
        obj.tier_override = entry.tier
        obj.derivation_type = entry.derivation_type
        obj.derives_from = entry.derives_from
```

Add the resolution function (module level) and properties:

```python
def _classify(member: "MeasurementInfo") -> tuple[str, int | None]:
    """Resolve (kind, tier) for a member; raise if unclassified.

    Precedence: diagnostic/normalization derivation_type, then an explicit
    Entry tier override, then the member's base-class kind()/tier().
    """
    if member.derivation_type == "diagnostic":
        return ("quality", None)
    if member.derivation_type == "normalization":
        return ("derived", None)  # tier inherited from the runtime target
    cls = type(member)
    if member.tier_override is not None:
        return (cls.kind() or "primary", member.tier_override)
    kind, tier = cls.kind(), cls.tier()
    if kind is None:
        raise ValueError(f"{member!r}: no kind assigned (re-parent to a classification base)")
    if kind == "primary" and tier is None:
        raise ValueError(
            f"{member!r}: primary member needs a tier (a tier base class or Entry(tier=...))"
        )
    return (kind, tier)
```

Add these properties to `MeasurementInfo`:

```python
    @property
    def resolved_kind(self) -> str:
        """The coarse kind for this member (identity/quality/primary/derived)."""
        return _classify(self)[0]

    @property
    def resolved_tier(self) -> int | None:
        """The trust tier (1/2/3) for this member, or None for non-primary."""
        return _classify(self)[1]
```

- [ ] **Step 4: Create `_tiers.py`**

```python
# src/phenotypic/schema/_tiers.py
"""Intermediate classification base classes for MeasurementInfo enums.

These member-less bases carry the coarse ``kind()`` and (for primary
measurements) ``tier()`` for the measurement-classification framework. A
measurement enum declares its classification by subclassing the matching
base instead of ``MeasurementInfo`` directly. Straddling enums subclass the
neutral parent (``PrimaryMeasure``/``DerivedMeasure``) and tag the minority
members with ``Entry(tier=...)`` / ``Entry(derivation_type=...)``.
"""

from ._measurement_info import MeasurementInfo


class IdentityInfo(MeasurementInfo):
    """Identity / design-factor columns (metadata, locators)."""

    @classmethod
    def kind(cls) -> str:
        return "identity"


class QualityInfo(MeasurementInfo):
    """Quality / trust columns that gate analysis."""

    @classmethod
    def kind(cls) -> str:
        return "quality"


class DerivedMeasure(MeasurementInfo):
    """Model/derived outputs; per-member tier via Entry overrides."""

    @classmethod
    def kind(cls) -> str:
        return "derived"


class PrimaryMeasure(MeasurementInfo):
    """Primary measured signal with no fixed tier (used by straddlers)."""

    @classmethod
    def kind(cls) -> str:
        return "primary"


class DirectPhenotype(PrimaryMeasure):
    """Tier 1 — semantic readout, safe to interpret a single value."""

    @classmethod
    def tier(cls) -> int:
        return 1


class DescriptiveTrait(PrimaryMeasure):
    """Tier 2 — interpretable named trait; interpret directionally."""

    @classmethod
    def tier(cls) -> int:
        return 2


class DiscriminativeFeature(PrimaryMeasure):
    """Tier 3 — agnostic fingerprint; use in aggregate for discrimination."""

    @classmethod
    def tier(cls) -> int:
        return 3
```

- [ ] **Step 5: Export the bases from `schema/__init__.py`**

Add after the `from ._measurement_info import Entry, MeasurementInfo` line:

```python
from ._tiers import (
    DerivedMeasure,
    DescriptiveTrait,
    DirectPhenotype,
    DiscriminativeFeature,
    IdentityInfo,
    PrimaryMeasure,
    QualityInfo,
)
```

And add those seven names to `__all__` (after `"MeasurementInfo",`).

- [ ] **Step 6: Guard `_measurement_descriptions()` against member-less classes**

In `src/phenotypic/util/_measurement_outputs.py`, inside the `for name in getattr(schema, "__all__", ()):` loop of `_measurement_descriptions`, after the `if not _is_info_class(obj): continue` line add:

```python
        if not list(obj):  # member-less classification bases contribute no columns
            continue
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/unit/schema/test_classification.py -q`
Expected: PASS (all 7 tests).

- [ ] **Step 8: Type-check and lint**

Run: `uv run mypy src/phenotypic/schema && uv run ruff check --fix src/phenotypic/schema`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/schema/_measurement_info.py src/phenotypic/schema/_tiers.py \
        src/phenotypic/schema/__init__.py src/phenotypic/util/_measurement_outputs.py \
        tests/unit/schema/test_classification.py
git commit -m "feat(schema): add measurement classification foundation (tiers + resolution)"
```

---

### Task 2: Re-parent Identity enums

**Files:**
- Modify: `src/phenotypic/schema/_metadata.py`, `_bbox.py`, `_object.py`, `_grid.py`, and the seven `_experimental_tags/*.py` (`_genetic.py`, `_sample.py`, `_plate.py`, `_condition.py`, `_incubation.py`, `_acquisition.py`, `_experiment.py`)
- Test: `tests/unit/schema/test_classification.py`

**Interfaces:**
- Consumes: `IdentityInfo` (Task 1).

- [ ] **Step 1: Write the failing test** (append to `test_classification.py`)

```python
def test_identity_enums_resolve_identity():
    from phenotypic.schema import (
        METADATA, BBOX, OBJECT, GRID,
        GENETIC_METADATA, SAMPLE_METADATA, PLATE_METADATA, CONDITION_METADATA,
        INCUBATION_METADATA, ACQUISITION_METADATA, EXPERIMENT_METADATA,
    )
    for enum in (METADATA, BBOX, OBJECT, GRID, GENETIC_METADATA, SAMPLE_METADATA,
                 PLATE_METADATA, CONDITION_METADATA, INCUBATION_METADATA,
                 ACQUISITION_METADATA, EXPERIMENT_METADATA):
        assert all(m.resolved_kind == "identity" for m in enum), enum.__name__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_identity_enums_resolve_identity -q`
Expected: FAIL — members resolve via base `MeasurementInfo` (kind None → `ValueError`).

- [ ] **Step 3: Re-parent each enum**

In each file, change the import and class declaration. Pattern (shown for `_metadata.py`):

```python
# from: from ._measurement_info import Entry, MeasurementInfo
from ._measurement_info import Entry
from ._tiers import IdentityInfo
# from: class METADATA(MeasurementInfo):
class METADATA(IdentityInfo):
```

Apply the identical swap (`MeasurementInfo` → `IdentityInfo`) to: `BBOX` (`_bbox.py`), `OBJECT` (`_object.py`), `GRID` (`_grid.py`), `GENETIC_METADATA`, `SAMPLE_METADATA`, `PLATE_METADATA`, `CONDITION_METADATA`, `INCUBATION_METADATA`, `ACQUISITION_METADATA`, `EXPERIMENT_METADATA`. If a file still references `MeasurementInfo` elsewhere, keep that import; otherwise drop it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_identity_enums_resolve_identity -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/schema/_metadata.py src/phenotypic/schema/_bbox.py \
        src/phenotypic/schema/_object.py src/phenotypic/schema/_grid.py \
        src/phenotypic/schema/_experimental_tags/ tests/unit/schema/test_classification.py
git commit -m "feat(schema): classify identity/metadata columns"
```

---

### Task 3: Re-parent Quality enums

**Files:**
- Modify: `src/phenotypic/schema/_quality_check.py`, `_quality_count.py`, `_quality_icc.py`, `_quality_mad.py`, `_quality_se.py`, `_quality_tukey.py`, `_quality_zmax.py`, `_curation.py`, `_error_category.py`, `_model_metrics.py`, `_grid_linreg_stats.py`, `_grid_spatial.py`, `_grid_spread.py`
- Test: `tests/unit/schema/test_classification.py`

**Interfaces:**
- Consumes: `QualityInfo` (Task 1).

- [ ] **Step 1: Write the failing test** (append)

```python
def test_quality_enums_resolve_quality():
    from phenotypic.schema import (
        QUALITY_CHECK, QUALITY_COUNT, QUALITY_ICC, QUALITY_MAD, QUALITY_SE,
        QUALITY_TUKEY, QUALITY_ZMAX, CURATION, ErrorCategory, MODEL_METRICS,
        GRID_LINREG_STATS, GRID_SPATIAL, GRID_SPREAD,
    )
    for enum in (QUALITY_CHECK, QUALITY_COUNT, QUALITY_ICC, QUALITY_MAD, QUALITY_SE,
                 QUALITY_TUKEY, QUALITY_ZMAX, CURATION, ErrorCategory, MODEL_METRICS,
                 GRID_LINREG_STATS, GRID_SPATIAL, GRID_SPREAD):
        assert all(m.resolved_kind == "quality" for m in enum), enum.__name__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_quality_enums_resolve_quality -q`
Expected: FAIL.

- [ ] **Step 3: Re-parent each enum** — swap `MeasurementInfo` → `QualityInfo` (import + class line) in all 13 files, same pattern as Task 2 Step 3.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_quality_enums_resolve_quality -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/schema/_quality_*.py src/phenotypic/schema/_curation.py \
        src/phenotypic/schema/_error_category.py src/phenotypic/schema/_model_metrics.py \
        src/phenotypic/schema/_grid_linreg_stats.py src/phenotypic/schema/_grid_spatial.py \
        src/phenotypic/schema/_grid_spread.py tests/unit/schema/test_classification.py
git commit -m "feat(schema): classify quality/QC and grid-diagnostic columns"
```

---

### Task 4: Re-parent Tier-1 primary enums (`SIZE`, `INTENSITY`)

**Files:**
- Modify: `src/phenotypic/schema/_size.py`, `_intensity.py`
- Test: `tests/unit/schema/test_classification.py`

**Interfaces:**
- Consumes: `DirectPhenotype` (Task 1).

- [ ] **Step 1: Write the failing test** (append)

```python
def test_tier1_primary_enums():
    from phenotypic.schema import SIZE, INTENSITY
    for enum in (SIZE, INTENSITY):
        assert all(m.resolved_kind == "primary" for m in enum), enum.__name__
        assert all(m.resolved_tier == 1 for m in enum), enum.__name__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_tier1_primary_enums -q`
Expected: FAIL.

- [ ] **Step 3: Re-parent** — swap `MeasurementInfo` → `DirectPhenotype` in `_size.py` and `_intensity.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_tier1_primary_enums -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/schema/_size.py src/phenotypic/schema/_intensity.py \
        tests/unit/schema/test_classification.py
git commit -m "feat(schema): classify Tier-1 direct phenotypes (size, intensity)"
```

---

### Task 5: Re-parent Tier-2 primary enums (`ColorLab`, `ColorHSV`, `RADIAL_EXPANSION`, `SYMMETRIC_ZONES`)

**Files:**
- Modify: `src/phenotypic/schema/_color_lab.py`, `_color_hsv.py`, `_radial_expansion.py`, `_symmetric_zones.py`
- Test: `tests/unit/schema/test_classification.py`

**Interfaces:**
- Consumes: `DescriptiveTrait` (Task 1).

- [ ] **Step 1: Write the failing test** (append)

```python
def test_tier2_primary_enums():
    from phenotypic.schema import ColorLab, ColorHSV, RADIAL_EXPANSION, SYMMETRIC_ZONES
    for enum in (ColorLab, ColorHSV, RADIAL_EXPANSION, SYMMETRIC_ZONES):
        assert all(m.resolved_tier == 2 for m in enum), enum.__name__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_tier2_primary_enums -q`
Expected: FAIL.

- [ ] **Step 3: Re-parent** — swap `MeasurementInfo` → `DescriptiveTrait` in all four files.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_tier2_primary_enums -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/schema/_color_lab.py src/phenotypic/schema/_color_hsv.py \
        src/phenotypic/schema/_radial_expansion.py src/phenotypic/schema/_symmetric_zones.py \
        tests/unit/schema/test_classification.py
git commit -m "feat(schema): classify Tier-2 descriptive traits (Lab/HSV color, radial, zones)"
```

---

### Task 6: Re-parent Tier-3 primary enums (`TEXTURE`, `ColorXYZ`, `Colorxy`, `ColorComposition`)

**Files:**
- Modify: `src/phenotypic/schema/_texture.py`, `_color_xyz.py`, `_color_xy.py`, `_color_composition.py`
- Test: `tests/unit/schema/test_classification.py`

**Interfaces:**
- Consumes: `DiscriminativeFeature` (Task 1).

- [ ] **Step 1: Write the failing test** (append)

```python
def test_tier3_primary_enums():
    from phenotypic.schema import TEXTURE, ColorXYZ, Colorxy, ColorComposition
    for enum in (TEXTURE, ColorXYZ, Colorxy, ColorComposition):
        assert all(m.resolved_tier == 3 for m in enum), enum.__name__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_tier3_primary_enums -q`
Expected: FAIL.

- [ ] **Step 3: Re-parent** — swap `MeasurementInfo` → `DiscriminativeFeature` in all four files.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_tier3_primary_enums -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/schema/_texture.py src/phenotypic/schema/_color_xyz.py \
        src/phenotypic/schema/_color_xy.py src/phenotypic/schema/_color_composition.py \
        tests/unit/schema/test_classification.py
git commit -m "feat(schema): classify Tier-3 discriminative features (texture, XYZ/xy/composition color)"
```

---

### Task 7: Straddler — `SHAPE` (form descriptors Tier 2; size-magnitude members Tier 1)

**Files:**
- Modify: `src/phenotypic/schema/_shape.py`
- Test: `tests/unit/schema/test_classification.py`

**Interfaces:**
- Consumes: `PrimaryMeasure` (Task 1), `Entry(tier=1)`.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_shape_straddles_tier1_and_tier2():
    from phenotypic.schema import SHAPE
    tier1 = {SHAPE.AREA, SHAPE.CONVEX_AREA, SHAPE.MEDIAN_RADIUS, SHAPE.MEAN_RADIUS,
             SHAPE.MAX_RADIUS, SHAPE.MIN_FERET_DIAMETER, SHAPE.MAX_FERET_DIAMETER,
             SHAPE.MAJOR_AXIS_LENGTH, SHAPE.MINOR_AXIS_LENGTH, SHAPE.BBOX_AREA,
             SHAPE.PERIMETER}
    tier2 = {SHAPE.CIRCULARITY, SHAPE.ECCENTRICITY, SHAPE.SOLIDITY, SHAPE.EXTENT,
             SHAPE.COMPACTNESS, SHAPE.ORIENTATION}
    for m in tier1:
        assert m.resolved_tier == 1, m
    for m in tier2:
        assert m.resolved_tier == 2, m
    assert tier1 | tier2 == set(SHAPE)   # full coverage, no member missed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_shape_straddles_tier1_and_tier2 -q`
Expected: FAIL.

- [ ] **Step 3: Re-parent `SHAPE` to `PrimaryMeasure` and add `tier=1` to the size-magnitude members**

Change the base: `from ._tiers import PrimaryMeasure` and `class SHAPE(PrimaryMeasure):`. Add `tier=1` (keyword) to the `Entry(...)` of exactly these members, preserving their existing `desc`/`bio_desc`/`image`: `AREA`, `PERIMETER`, `CONVEX_AREA`, `MEDIAN_RADIUS`, `MEAN_RADIUS`, `MAX_RADIUS`, `MIN_FERET_DIAMETER`, `MAX_FERET_DIAMETER`, `MAJOR_AXIS_LENGTH`, `MINOR_AXIS_LENGTH`, `BBOX_AREA`. Example for the first two:

```python
    AREA = Entry(
        "Area",
        "Total number of pixels occupied by the microbial colony. Represents colony biomass and growth extent on agar plates. Larger areas typically indicate more robust growth or longer incubation times.",
        bio_desc=(
            "Projected 2D footprint of the colony in pixels — a common proxy "
            "for colony size and overall growth in arrayed plate assays. With "
            "matched imaging and incubation, larger area generally reflects "
            "greater proliferation or spreading; it captures only the 2D "
            "footprint, not colony height or cell density."
        ),
        image="shape/area.png",
        tier=1,
    )
    PERIMETER = Entry(
        "Perimeter",
        "Total length of the colony's outer boundary in pixels. Measures colony edge complexity and surface irregularity. Smooth, circular colonies have shorter perimeters relative to their area compared to irregular or filamentous colonies.",
        tier=1,
    )
```

PrimaryMeasure has no class-level `tier()`, so the remaining members (`CIRCULARITY`, `CONVEX_AREA`'s neighbors that are ratios, `ECCENTRICITY`, `SOLIDITY`, `EXTENT`, `COMPACTNESS`, `ORIENTATION`) would resolve as primary-with-no-tier and **raise**. To give them Tier 2, add a class-level default by overriding `tier` on `SHAPE` itself:

```python
    @classmethod
    def category(cls):
        return "Shape"

    @classmethod
    def tier(cls) -> int:
        return 2  # default for form descriptors; size-magnitude members override via Entry(tier=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_shape_straddles_tier1_and_tier2 -q`
Expected: PASS.

- [ ] **Step 5: Verify no value changed**

Run: `uv run python -c "from phenotypic.schema import SHAPE; assert SHAPE.AREA.value=='Shape_Area'; print('ok')"`
Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/schema/_shape.py tests/unit/schema/test_classification.py
git commit -m "feat(schema): classify SHAPE (size-magnitude Tier 1, form descriptors Tier 2)"
```

---

### Task 8: Derived enums (`LOG_GROWTH_MODEL`, `LINEAR_SOFTPLUS_MODEL`, `DOUBLE_SOFTPLUS_MODEL`, `EDGE_CORRECTION`)

**Files:**
- Modify: `src/phenotypic/schema/_log_growth_model.py`, `_linear_softplus_model.py`, `_double_softplus_model.py`, `_edge_correction.py`
- Test: `tests/unit/schema/test_classification.py`

**Interfaces:**
- Consumes: `DerivedMeasure` (Task 1), `Entry(tier=..., derivation_type=..., derives_from=...)`.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_derived_growth_models_and_edge_correction():
    from phenotypic.schema import (
        LOG_GROWTH_MODEL, LINEAR_SOFTPLUS_MODEL, DOUBLE_SOFTPLUS_MODEL, EDGE_CORRECTION,
    )
    # LOG_GROWTH: kinetics -> Tier 1; regularization knobs -> diagnostic/quality
    assert LOG_GROWTH_MODEL.R_FIT.resolved_tier == 1
    assert LOG_GROWTH_MODEL.K_FIT.resolved_tier == 1
    assert LOG_GROWTH_MODEL.N0_FIT.resolved_tier == 1
    assert LOG_GROWTH_MODEL.GROWTH_RATE.resolved_tier == 1
    for knob in (LOG_GROWTH_MODEL.LAM, LOG_GROWTH_MODEL.BETA, LOG_GROWTH_MODEL.K_MAX):
        assert knob.resolved_kind == "quality"
    # LINEAR_SOFTPLUS: v/s0/lam Tier 1; alpha Tier 2
    assert LINEAR_SOFTPLUS_MODEL.v.resolved_tier == 1
    assert LINEAR_SOFTPLUS_MODEL.s0.resolved_tier == 1
    assert LINEAR_SOFTPLUS_MODEL.lam.resolved_tier == 1
    assert LINEAR_SOFTPLUS_MODEL.alpha.resolved_tier == 2
    # DOUBLE_SOFTPLUS: v/s0/lam/smax Tier 1; alpha/beta Tier 2; mode diagnostic
    for m in (DOUBLE_SOFTPLUS_MODEL.v, DOUBLE_SOFTPLUS_MODEL.s0,
              DOUBLE_SOFTPLUS_MODEL.lam, DOUBLE_SOFTPLUS_MODEL.smax):
        assert m.resolved_tier == 1, m
    assert DOUBLE_SOFTPLUS_MODEL.alpha.resolved_tier == 2
    assert DOUBLE_SOFTPLUS_MODEL.beta.resolved_tier == 2
    assert DOUBLE_SOFTPLUS_MODEL.mode.resolved_kind == "quality"
    # EDGE_CORRECTION: normalization, tier deferred to target
    for m in EDGE_CORRECTION:
        assert m.resolved_kind == "derived"
        assert m.resolved_tier is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_derived_growth_models_and_edge_correction -q`
Expected: FAIL.

- [ ] **Step 3: Edit `_log_growth_model.py`**

Re-parent to `DerivedMeasure` and tag members:

```python
from ._measurement_info import Entry
from ._tiers import DerivedMeasure


class LOG_GROWTH_MODEL(DerivedMeasure):
    @classmethod
    def category(cls) -> str:
        return "LogGrowthModel"

    R_FIT = Entry("r", "The intrinsic growth rate",
                  tier=1, derivation_type="parameterization", derives_from="SIZE")
    K_FIT = Entry("K", "The carrying capacity",
                  tier=1, derivation_type="parameterization", derives_from="SIZE")
    N0_FIT = Entry("N0", "The initial number of the colony size metric being fitted",
                   tier=1, derivation_type="parameterization", derives_from="SIZE")
    LAM = Entry(
        "lambda",
        "The regularization factor applied to the max specific growth rate "
        "and initial population size",
        derivation_type="diagnostic",
    )
    BETA = Entry(
        "beta",
        (
            "The penalty factor applied to relative difference of "
            "the carrying capacity from the largest measurement"
        ),
        derivation_type="diagnostic",
    )
    GROWTH_RATE = Entry("µmax", "The growth rate of the colony calculated as (K*r)/4",
                        tier=1, derivation_type="parameterization", derives_from="SIZE")
    K_MAX = Entry("Kmax", "The upper bound of the carrying capacity for model fitting",
                  derivation_type="diagnostic")
```

- [ ] **Step 4: Edit `_linear_softplus_model.py`**

```python
from ._measurement_info import Entry
from ._tiers import DerivedMeasure


class LINEAR_SOFTPLUS_MODEL(DerivedMeasure):
    @classmethod
    def category(cls) -> str:
        return "LinearSoftplus"

    v = Entry("v", "The post-lag phase growth rate.",
              bio_desc="The post-lag phase growth rate "
                       "using the target metric (usually radius)",
              tier=1, derivation_type="parameterization", derives_from="SIZE")
    s0 = Entry("s0", "The initial value of the target metric",
               bio_desc="The initial size",
               tier=1, derivation_type="parameterization", derives_from="SIZE")
    lam = Entry("lambda", "The duration of the lag phase",
                tier=1, derivation_type="parameterization", derives_from="SIZE")
    alpha = Entry("alpha", "lag phase transition sharpness",
                  tier=2, derivation_type="parameterization", derives_from="SIZE")
```

- [ ] **Step 5: Edit `_double_softplus_model.py`**

Re-parent to `DerivedMeasure`; add `tier=1` to `v`/`s0`/`lam`/`smax`, `tier=2` to `alpha`/`beta`, and `derivation_type="diagnostic"` to `mode` (leave each existing `desc` text intact; add `derivation_type="parameterization", derives_from="SIZE"` to the Tier-1/Tier-2 members). For example:

```python
    v = Entry("v", "The post-lag phase growth rate.",
              tier=1, derivation_type="parameterization", derives_from="SIZE")
    ...
    alpha = Entry("alpha", "lag phase transition sharpness",
                  tier=2, derivation_type="parameterization", derives_from="SIZE")
    smax = Entry(
        "smax",
        "Carrying capacity used by the model. Either the user-provided "
        "scalar or the per-group observed maximum.",
        tier=1, derivation_type="parameterization", derives_from="SIZE",
    )
    beta = Entry(
        "beta",
        "Saturation transition sharpness. Fitted per-group when a "
        "saturation shoulder is detected and ``beta`` is ``None`` at "
        "construction; held at the user-provided scalar (or the "
        "module default) when no shoulder is present.",
        tier=2, derivation_type="parameterization", derives_from="SIZE",
    )
    mode = Entry(
        "mode",
        "Fit variant selected per-group: 'fixed_beta' (beta held at "
        "the user-provided or module-default value) or 'fitted_beta' "
        "(beta fitted as a 5th free parameter when a saturation "
        "shoulder is detected).",
        derivation_type="diagnostic",
    )
```

- [ ] **Step 6: Edit `_edge_correction.py`**

Re-parent to `DerivedMeasure`; tag both members as normalization:

```python
from ._measurement_info import Entry
from ._tiers import DerivedMeasure


class EDGE_CORRECTION(DerivedMeasure):
    ...
    @classmethod
    def category(cls) -> str:
        return "EdgeCorrection"

    CORRECTED_CAP = Entry("Cap", "The carrying capacity for the target measurement",
                          derivation_type="normalization", derives_from="SIZE")
    NEW_VAL = Entry("NewVal", "The new value of the target measurement",
                    derivation_type="normalization", derives_from="SIZE")
```

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_derived_growth_models_and_edge_correction -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/schema/_log_growth_model.py src/phenotypic/schema/_linear_softplus_model.py \
        src/phenotypic/schema/_double_softplus_model.py src/phenotypic/schema/_edge_correction.py \
        tests/unit/schema/test_classification.py
git commit -m "feat(schema): classify derived growth-model + edge-correction columns"
```

---

### Task 9: Render Tier/Use column in `rst_table()`

**Files:**
- Modify: `src/phenotypic/schema/_measurement_info.py`
- Test: `tests/unit/schema/test_classification.py`

**Interfaces:**
- Consumes: `resolved_kind`/`resolved_tier` (Task 1).
- Produces: a human-readable `use_label` per member used in docs rendering.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_rst_table_includes_use_column():
    from phenotypic.schema import TEXTURE, METADATA
    txt = TEXTURE.rst_table()
    assert "Use" in txt
    assert "Discriminative feature" in txt
    # Identity enums have no tier/use semantics -> column suppressed
    assert "Use" not in METADATA.rst_table()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_rst_table_includes_use_column -q`
Expected: FAIL.

- [ ] **Step 3: Add a `use_label` helper and thread it through rendering**

In `_measurement_info.py`, add a module-level mapping and a property:

```python
_USE_LABELS: Final = {
    (1, "primary"): "Direct phenotype (Tier 1)",
    (2, "primary"): "Descriptive trait (Tier 2)",
    (3, "primary"): "Discriminative feature (Tier 3)",
}
```

Add to `MeasurementInfo`:

```python
    @property
    def use_label(self) -> str:
        """Short human-readable 'how to apply' label, empty for non-primary."""
        kind, tier = _classify(self)
        return _USE_LABELS.get((tier, kind), "")
```

Extend `_render_info_table` to accept a 5-tuple `(name, desc, bio, img, use)` and emit a `Use` column when any `use` is non-empty (mirror the `has_bio` pattern):

```python
def _render_info_table(
    rows: list[tuple[str, str, str, str | None, str]],
    *,
    title: str,
    name_header: str = "Name",
    desc_header: str = "Description",
) -> str:
    has_bio = any(row[2] for row in rows)
    has_img = any(row[3] for row in rows)
    has_use = any(row[4] for row in rows)

    lines = [
        f".. list-table:: Category: **{title}**",
        "   :header-rows: 1",
        "",
        f"   * - {name_header}",
        f"     - {desc_header}",
    ]
    if has_use:
        lines.append("     - Use")
    if has_bio:
        lines.append("     - Biology")
    if has_img:
        lines.append("     - Image")

    for name, desc, bio, img, use in rows:
        lines.append(f"   * - ``{name}``")
        lines.append(f"     - {_rst_cell_text(desc)}")
        if has_use:
            lines.append(f"     - {use}")
        if has_bio:
            lines.append(f"     - {_rst_cell_text(bio)}")
        if has_img:
            if img:
                lines.append(f"     - .. image:: {_ASSET_URL_PREFIX}/{img}")
                lines.append("          :width: 110px")
            else:
                lines.append("     -")
    return "\n".join(lines)
```

Update `rst_table()` to build 5-tuples:

```python
        rows = [
            (
                m.value if use_headers else m.label,
                m.desc,
                m.bio_desc,
                m.image,
                m.use_label,
            )
            for m in cls
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/schema/test_classification.py::test_rst_table_includes_use_column -q`
Expected: PASS.

- [ ] **Step 5: Run the full schema suite (catch any rst_table callers)**

Run: `uv run pytest tests/unit/schema -q`
Expected: PASS (existing `rst_table`/`append_rst_to_doc` tests still green).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/schema/_measurement_info.py tests/unit/schema/test_classification.py
git commit -m "feat(schema): render Tier/Use column in measurement rst tables"
```

---

### Task 10: CI coverage gate

**Files:**
- Test: `tests/unit/schema/test_classification_coverage.py`

**Interfaces:**
- Consumes: every exported enum's `resolved_kind`/`resolved_tier` (Tasks 2–8).

- [ ] **Step 1: Write the gate test**

```python
# tests/unit/schema/test_classification_coverage.py
"""Gate: every measurement column resolves to a valid (kind, tier)."""
import phenotypic.schema as schema
from phenotypic.schema import MeasurementInfo
from phenotypic.schema._measurement_info import _VALID_KINDS
from phenotypic.schema._tiers import (
    DirectPhenotype, DescriptiveTrait, DiscriminativeFeature,
    IdentityInfo, QualityInfo, DerivedMeasure, PrimaryMeasure,
)

_BASES = (DirectPhenotype, DescriptiveTrait, DiscriminativeFeature,
          IdentityInfo, QualityInfo, DerivedMeasure, PrimaryMeasure)


def _column_enums():
    for name in schema.__all__:
        obj = getattr(schema, name)
        if (isinstance(obj, type) and issubclass(obj, MeasurementInfo)
                and obj not in _BASES and obj is not MeasurementInfo and list(obj)):
            yield obj


def test_every_member_is_classified():
    failures = []
    for enum in _column_enums():
        for m in enum:
            try:
                kind, tier = m.resolved_kind, m.resolved_tier
            except ValueError as exc:
                failures.append(f"{enum.__name__}.{m.name}: {exc}")
                continue
            if kind not in _VALID_KINDS:
                failures.append(f"{enum.__name__}.{m.name}: bad kind {kind!r}")
            if tier not in (None, 1, 2, 3):
                failures.append(f"{enum.__name__}.{m.name}: bad tier {tier!r}")
            if kind == "primary" and tier is None:
                failures.append(f"{enum.__name__}.{m.name}: primary w/o tier")
    assert not failures, "Unclassified measurement columns:\n" + "\n".join(failures)


def test_intermediate_bases_have_no_members():
    for base in _BASES:
        assert list(base) == [], f"{base.__name__} must stay member-less"
```

- [ ] **Step 2: Run the gate**

Run: `uv run pytest tests/unit/schema/test_classification_coverage.py -q`
Expected: PASS (if a column is unclassified, it lists exactly which — that is the gate working).

- [ ] **Step 3: Run the full schema suite + type check**

Run: `uv run pytest tests/unit/schema -q && uv run mypy src/phenotypic/schema`
Expected: PASS, no type errors.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/schema/test_classification_coverage.py
git commit -m "test(schema): CI gate requiring every measurement column be classified"
```

---

### Task 11: User-facing documentation page

**Files:**
- Create: `docs/source/explanation/measurement_classification_system.md`
- Modify: `docs/source/explanation/index.rst`
- Modify: `docs/source/explanation/measurement_metrics_biological_meaning.md` (add a cross-link)

**Interfaces:**
- Consumes: the conceptual framework (spec Sections 2–7) and `use_label` rendering (Task 9).

- [ ] **Step 1: Create the explanation page**

```markdown
# Measurement Classification: Phenotypes vs. Features

PhenoTypic measures many columns per colony. This page explains *how to apply*
each one — which numbers you can report directly as a biological result, and
which are best used together as inputs to classification or clustering — without
needing the underlying math.

## Two questions place every measurement

- **Interpretability** — does the number name a *biological thing* (a diameter, a
  pigment, biomass), or is it a *mathematical descriptor* (a texture value, a
  colour coordinate)?
- **Analytical role** — do you use it *as a result* (quantify an effect), or *as a
  feature* (feed many of them into a classifier/clustering)?

## Four kinds, then three tiers

Every column is first one of four **kinds**:

- **Identity / design factors** — the variables you analyse *against* (metadata,
  locators). Not outcomes.
- **Quality** — gates whether to trust a row/plate. Never a biological claim.
- **Primary measurement** — the measured signal. These get a **tier** (below).
- **Derived / model output** — computed from primary measurements; classified by
  *how* they were derived.

Primary measurements fall on a three-tier spectrum:

| Tier | Name | What it is | How to apply it |
|---|---|---|---|
| **1** | Direct phenotype | A real biological quantity with units/meaning (size, intensity/opacity) | Report a single value as a result; compare across conditions; dose–response. |
| **2** | Descriptive trait | A named, interpretable form/colour property, usually unitless (shape descriptors, Lab/HSV colour, radial/zone structure) | Interpret the *direction* of change against a control; also good clustering input. |
| **3** | Discriminative feature | A mathematical fingerprint with no single biological meaning (texture, XYZ/xy/composition colour) | Don't read one value; use the whole block together for classification/clustering. |

## The trust contract

The tier is a promise about what a single number licenses:

- **Tier 1** — pre-validated for direct biological claims; safe to report alone.
- **Tier 2** — interpret directionally, anchored to a control.
- **Tier 3** — make no single-value biological claim; its job is discrimination,
  judged by how well groups separate.

## Derived outputs inherit by *how* they were made

A model fit on a primary phenotype is classified by its transformation:

- **Parameterization** (e.g. logistic/softplus growth: growth rate, lag, carrying
  capacity) → same tier as the input phenotype. Colony size and growth rate are
  interchangeable fitness proxies, so growth parameters are Tier 1.
- **Normalization** (e.g. edge correction) → the input's tier, cleaned.
- **Fit diagnostics** (R², RMSE, optimizer state, regularization knobs) → Quality.

See also: [Measurement metrics and their biological meaning](measurement_metrics_biological_meaning.md)
for per-metric detail, and the
[measurement reference](../measurements_ref/index.rst) for the Use/Tier badge on
every column.
```

- [ ] **Step 2: Wire it into the toctree**

In `docs/source/explanation/index.rst`, under the `:caption: Measurement & Analysis` toctree, add `measurement_classification_system` as the **first** entry (before `measurement_metrics_biological_meaning`):

```rst
.. toctree::
   :maxdepth: 1
   :caption: Measurement & Analysis

   measurement_classification_system
   measurement_metrics_biological_meaning
   edge_effects_in_plate_assays
   image_quality_noise_contrast_structure
   notebooks/linear_softplus_model
```

- [ ] **Step 3: Add the reverse cross-link**

At the top of `docs/source/explanation/measurement_metrics_biological_meaning.md`, after the intro paragraph, add:

```markdown
> For *how to apply* each metric — which to report directly vs. use as a
> classifier feature — see
> [Measurement Classification: Phenotypes vs. Features](measurement_classification_system.md).
```

- [ ] **Step 4: Build the docs to verify the page renders and links resolve**

Run: `uv run --group docs sphinx-build -b html -q docs/source docs/_build/html`
Expected: build succeeds; no warning about `measurement_classification_system` missing from a toctree or broken cross-references.

- [ ] **Step 5: Commit**

```bash
git add docs/source/explanation/measurement_classification_system.md \
        docs/source/explanation/index.rst \
        docs/source/explanation/measurement_metrics_biological_meaning.md
git commit -m "docs(explanation): add measurement classification system page"
```

---

## Final verification

- [ ] Run the full unit suite for touched areas:
  `uv run pytest tests/unit/schema tests/unit/util -q` → PASS
- [ ] Type check: `uv run mypy src/phenotypic` → no new errors
- [ ] Lint: `uv run ruff check src/phenotypic` → clean
- [ ] Sanity: no header changed —
  `uv run python -c "from phenotypic.schema import SHAPE,SIZE; print(SHAPE.AREA.value, SIZE.AREA.value)"`
  → `Shape_Area Size_Area`

---

## Self-Review notes

- **Spec coverage:** Foundation (Task 1) ↔ §10.1/10.3; re-parenting (Tasks 2–6) ↔ §10.2; straddlers (Tasks 7–8) ↔ §10.2; rst rendering (Task 9) ↔ §10.6 reference annotation; coverage gate (Task 10) ↔ §10.5 + resolved-design-decision #5; docs page (Task 11) ↔ §10.6 conceptual page; non-breaking guards (Task 1 Step 6, Task 7 Step 5) ↔ §10.4.
- **Straddler note:** all three growth-model enums straddle (kinetics Tier 1, curve-shape Tier 2, `mode`/knobs diagnostic), not just `LOG_GROWTH_MODEL`; Task 8 handles all three.
- **`EDGE_CORRECTION`** uses `derivation_type="normalization"` → resolves as kind `derived`, tier `None` (deferred to the runtime target); the coverage gate accepts `None` tier for non-primary kinds.
```
