# Contrast Enhancers + `input_layer` Mixin — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ContrastGamma` / `ContrastLog` / `ContrastSigmoid` enhancers that can read either `detect_mat` or the pristine `rgb` layer, and retire `clip: bool` in favour of a single `norm: NormOut` field repo-wide.

**Architecture:** Two pydantic **field-append mixins** (`NormalizedOutputMixin`, `InputLayerMixin`) supply cross-cutting fields that land *after* each operation's own parameters. When `input_layer="rgb"`, an op adjusts all three channels and then collapses to 2-D by projecting through the image's own `detect_mode` — which requires a new `DetectionMode.compute_from_rgb()` and an `rgb_to_xyz()` extraction, because `LabL/A/B` depend on per-image gamma/illuminant/observer. Output is always written to `detect_mat` only.

**Tech Stack:** Python 3.11+, pydantic v2.12.5, scikit-image 0.25.2, colour-science, numpy, pytest, `uv` as the sole runner.

**Spec:** `docs/superpowers/specs/2026-07-09-contrast-enhancers-input-layer/design.md` (§ references below point there).

---

## Global Constraints

- **`uv` is the sole package manager and runner.** Never use bare `python` or `pip`. Every command below is `uv run ...`.
- **Operations are pydantic v2 models.** No hand-written `__init__`. Parameters are annotated class-level fields. Construction is keyword-only. Normalization goes in a `field_validator`, never `__init__`.
- **Operations must be constructible with no required args**, or `from_json()` breaks.
- **Enhancers subclass a purpose-group marker ABC** (`ContrastAdjustment`, `FocusEdge`, …), never `ImageEnhancer` directly.
- **Enhancers write `detect_mat` only.** `@validate_operation_integrity("image.rgb", "image.gray")` enforces this. Always use the accessor: `image.detect_mat[:] = arr`.
- **Google-style docstrings** everywhere; every doctest must run against `load_synth_yeast_plate()`. Field descriptions are auto-derived from the `Args:` block, so every field needs an `Args:` entry.
- **`MeasurementInfo` members:** author `label`/`desc` only. Never author `bio_desc`; leave `image` unset.
- **Target version: `0.18.0`.** The migration error message must name `0.18.0` verbatim.
- **A test that cannot run must fail, not skip.** Derive tolerances from a mechanism, never a guess. Prove each new test can fail before trusting it.
- Lint/typecheck gates: `uv run ruff check --fix` and `uv run mypy src/phenotypic`.

### Type aliases introduced (used across many tasks)

```python
# src/phenotypic/sdk_/typing_.py
InputLayer: TypeAlias = Literal["detect_mat", "rgb"]
NormOut: TypeAlias = Optional[Literal["clip", "rescale"]]
```

---

## Phase Order — corrected from the brief

The requested order had Phases 1–3 as mutually independent. **They are not.** The
`clip → norm` migration and the `FocusEdgeLaplace` fix both consume
`NormalizedOutputMixin`, so the mixins must land first. Verified dependency graph:

```
Phase 1 (mixins + type aliases)  ──┬──> Phase 2 (clip -> norm migration)
                                   └──> Phase 4 (FocusEdgeLaplace + invariant gate)

Phase 3 (rgb_to_xyz + compute_from_rgb)   [genuinely independent of 1, 2, 4]

Phase 1 + Phase 3 ──> Phase 5 (Contrast ops + ContrastStretching retrofit)
Phase 2 + Phase 5 ──> Phase 6 (skill convention)
```

Phase 3 may be executed in parallel with Phases 1–2 (disjoint files). Phases 1→2 and
1→4 are strictly sequential.

**Version bump lands in Phase 2**, with the migration message that names it.

---

## File Structure

**New files**

| File | Responsibility |
|---|---|
| `src/phenotypic/sdk_/mixin/_normalized_output_mixin.py` | `norm` field, `_apply_norm`, legacy-`clip` rejection |
| `src/phenotypic/sdk_/mixin/_input_layer_mixin.py` | `input_layer` field, `_read_input_layer`, `_project_to_detect_mat` |
| `src/phenotypic/sdk_/mixin/_norm_control_mixin.py` | `NormControlMixin` (replaces `_clip_control_mixin.py`) |
| `src/phenotypic/_core/_image_parts/color_space_accessors/_xyz_conversion.py` | `rgb_to_xyz()` free function |
| `src/phenotypic/enhance/_contrast_gamma.py` | `ContrastGamma` |
| `src/phenotypic/enhance/_contrast_log.py` | `ContrastLog` |
| `src/phenotypic/enhance/_contrast_sigmoid.py` | `ContrastSigmoid` |

**Deleted**

- `src/phenotypic/sdk_/mixin/_clip_control_mixin.py` (replaced)

---

# Phase 1 — Type aliases + the two field-append mixins

No consumers yet. Nothing else can proceed without this.

---

### Task 1.1: Add `InputLayer` and `NormOut` type aliases

**Files:**
- Modify: `src/phenotypic/sdk_/typing_.py` (near `DetectMode`, line ~28)
- Test: `tests/unit/sdk_/test_typing_aliases.py`

**Interfaces:**
- Produces: `InputLayer`, `NormOut` — importable from `phenotypic.sdk_.typing_`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_typing_aliases.py`:

```python
"""Pin the closed value sets introduced for the contrast enhancers."""
from typing import get_args

from phenotypic.sdk_.typing_ import InputLayer, NormOut


def test_input_layer_values():
    assert set(get_args(InputLayer)) == {"detect_mat", "rgb"}


def test_norm_out_values():
    # Optional[Literal[...]] -> (Literal[...], NoneType)
    literal, none_type = get_args(NormOut)
    assert set(get_args(literal)) == {"clip", "rescale"}
    assert none_type is type(None)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/sdk_/test_typing_aliases.py -v
```
Expected: FAIL — `ImportError: cannot import name 'InputLayer'`

- [ ] **Step 3: Add the aliases**

In `src/phenotypic/sdk_/typing_.py`, immediately after the `DetectMode` alias:

```python
#: Source layer an ``InputLayerMixin`` operation reads from. ``"detect_mat"``
#: is the 2-D detection matrix; ``"rgb"`` is the pristine 3-D colour layer,
#: collapsed back to 2-D via the image's own ``detect_mode``. Literal-only —
#: no Enum partner, mirroring ``DetectMode`` / ``ExecutionMode``.
InputLayer: TypeAlias = Literal["detect_mat", "rgb"]

#: Output-range policy for operations that guard ``detect_mat``'s [0, 1]
#: contract. ``"clip"`` saturates, ``"rescale"`` normalizes the full
#: histogram, ``None`` passes values through untouched (the escape hatch
#: GAT regions and ``CompositeEnhance`` depend on). Replaces the former
#: ``clip: bool`` field as of 0.18.0.
NormOut: TypeAlias = Optional[Literal["clip", "rescale"]]
```

Ensure `Optional` and `TypeAlias` are imported at the top of the module (add to the existing `typing` import if absent).

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/test_typing_aliases.py -v
```
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/typing_.py tests/unit/sdk_/test_typing_aliases.py
git commit -m "feat(sdk_): add InputLayer and NormOut type aliases"
```

---

### Task 1.2: `NormalizedOutputMixin`

**Files:**
- Create: `src/phenotypic/sdk_/mixin/_normalized_output_mixin.py`
- Modify: `src/phenotypic/sdk_/mixin/__init__.py`, `src/phenotypic/sdk_/__init__.py`
- Test: `tests/unit/sdk_/mixin/test_normalized_output_mixin.py`

**Interfaces:**
- Consumes: `NormOut` (Task 1.1).
- Produces:
  - `class NormalizedOutputMixin(BaseModel)` with field `norm: NormOut = "clip"`
  - `def _apply_norm(self, arr: np.ndarray) -> np.ndarray`
  - Appends `norm` to the **end** of `cls.__pydantic_fields__`.
  - Raises `ValidationError` (wrapping `ValueError`) when constructed with a legacy `clip` key.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/mixin/test_normalized_output_mixin.py`:

```python
"""Contract for NormalizedOutputMixin: field append, norm semantics, clip rejection."""
import numpy as np
import pytest
from pydantic import ValidationError

from phenotypic.abc_ import ImageEnhancer
from phenotypic.sdk_ import NormalizedOutputMixin


class _Probe(NormalizedOutputMixin, ImageEnhancer):
    """Probe enhancer.

    Args:
        sigma: Width.
        norm: Output normalization policy.
    """

    sigma: float = 1.0

    def _operate(self, image):
        return image


def test_norm_is_appended_last():
    assert list(_Probe.model_fields) == ["sigma", "norm"]


def test_norm_appended_last_in_json_schema():
    assert list(_Probe.model_json_schema()["properties"]) == ["sigma", "norm"]


def test_default_is_clip():
    assert _Probe().norm == "clip"


@pytest.mark.parametrize(
    ("norm", "expected"),
    [
        ("clip", [0.0, 0.5, 1.0]),
        ("rescale", [0.0, 0.5, 1.0]),
        (None, [-0.5, 0.5, 1.5]),
    ],
)
def test_apply_norm(norm, expected):
    arr = np.array([-0.5, 0.5, 1.5], dtype=np.float32)
    np.testing.assert_allclose(_Probe(norm=norm)._apply_norm(arr), expected, atol=1e-6)


def test_rescale_differs_from_clip_when_input_is_inside_unit_range():
    """`clip` is the identity in-range; `rescale` stretches. Distinguishes the two."""
    arr = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    np.testing.assert_allclose(_Probe(norm="clip")._apply_norm(arr), arr, atol=1e-6)
    np.testing.assert_allclose(
        _Probe(norm="rescale")._apply_norm(arr), [0.0, 0.5, 1.0], atol=1e-6
    )


def test_legacy_clip_key_raises_migration_message():
    with pytest.raises(ValidationError, match=r"`clip` was replaced by `norm` in 0\.18\.0"):
        _Probe(clip=True)


def test_invalid_norm_rejected():
    with pytest.raises(ValidationError):
        _Probe(norm="passthrough")


def test_setattr_to_none_under_validate_assignment():
    """The GAT defer path uses setattr; validate_assignment must accept None."""
    op = _Probe(norm="clip")
    op.norm = None
    assert op.norm is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/sdk_/mixin/test_normalized_output_mixin.py -v
```
Expected: FAIL — `ImportError: cannot import name 'NormalizedOutputMixin'`

- [ ] **Step 3: Write the mixin**

Create `src/phenotypic/sdk_/mixin/_normalized_output_mixin.py`:

```python
"""Mixin supplying the ``norm`` output-range policy as an appended pydantic field."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, model_validator
from skimage.exposure import rescale_intensity

from phenotypic.sdk_.typing_ import NormOut


class NormalizedOutputMixin(BaseModel):
    """Adds a ``norm`` field controlling how an operation's output is range-guarded.

    ``detect_mat`` is contractually [0, 1]. ``norm`` selects how an operation
    upholds that contract:

    - ``"clip"`` (default) saturates out-of-range values. It is the identity for
      in-range pixels, so absolute intensity is preserved and ``detect_mat``
      stays comparable across a batch of plates.
    - ``"rescale"`` linearly remaps the full observed range onto [0, 1]. Ordering
      survives, absolute scale does not: a single specular highlight sets the max.
    - ``None`` passes values through untouched. Required inside a Generalized
      Anscombe Transform region (where the signal is deliberately not in [0, 1])
      and by ``CompositeEnhance`` on non-normalized maps.

    The field is **appended** to the end of the subclass's field order rather than
    frontloaded, so an operation's own parameters keep their natural position in
    ``model_json_schema()`` and ``to_json()``.

    Note:
        Replaces the ``clip: bool`` field removed in 0.18.0. A bool cannot express
        ``"rescale"``, and the attribute name ``clip`` is claimed by
        :class:`NormControlMixin`, which duck-types on it.
    """

    norm: NormOut = "clip"

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Move ``norm`` to the end of the subclass's field order."""
        super().__pydantic_init_subclass__(**kwargs)
        fields = cls.__pydantic_fields__
        if "norm" in fields and list(fields)[-1] != "norm":
            fields["norm"] = fields.pop("norm")
            cls.model_rebuild(force=True)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_clip(cls, data: Any) -> Any:
        """Turn the 0.17.x ``clip`` key into an actionable migration error.

        ``BaseOperation`` sets ``extra="forbid"``, so without this the user sees
        pydantic's opaque "Extra inputs are not permitted".
        """
        if isinstance(data, dict) and "clip" in data:
            raise ValueError(
                f"{cls.__name__}: `clip` was replaced by `norm` in 0.18.0. "
                f"Use norm='clip' (was clip=True) or norm=None (was clip=False)."
            )
        return data

    def _apply_norm(self, arr: np.ndarray) -> np.ndarray:
        """Apply the configured output-range policy to *arr*."""
        match self.norm:
            case "clip":
                return np.clip(arr, 0.0, 1.0)
            case "rescale":
                return rescale_intensity(arr, out_range=(0.0, 1.0))
            case None:
                return arr
        raise ValueError(f"Unknown norm policy: {self.norm!r}")
```

In `src/phenotypic/sdk_/mixin/__init__.py` add the import and `__all__` entry:

```python
from ._normalized_output_mixin import NormalizedOutputMixin
```

Do the same in `src/phenotypic/sdk_/__init__.py` (import + `__all__`).

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/mixin/test_normalized_output_mixin.py -v
```
Expected: PASS (9 passed)

- [ ] **Step 5: Prove `test_norm_is_appended_last` can fail**

Temporarily comment out the `fields["norm"] = fields.pop("norm")` line, re-run, and confirm the order test reports `['norm', 'sigma']`. Restore the line.

```bash
uv run pytest tests/unit/sdk_/mixin/test_normalized_output_mixin.py::test_norm_is_appended_last -v
```
Expected while broken: FAIL. Expected after restoring: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/mixin/_normalized_output_mixin.py \
        src/phenotypic/sdk_/mixin/__init__.py src/phenotypic/sdk_/__init__.py \
        tests/unit/sdk_/mixin/test_normalized_output_mixin.py
git commit -m "feat(sdk_): add NormalizedOutputMixin with appended norm field"
```

---

### Task 1.3: `InputLayerMixin`

**Files:**
- Create: `src/phenotypic/sdk_/mixin/_input_layer_mixin.py`
- Modify: `src/phenotypic/sdk_/mixin/__init__.py`, `src/phenotypic/sdk_/__init__.py`
- Test: `tests/unit/sdk_/mixin/test_input_layer_mixin.py`

**Interfaces:**
- Consumes: `InputLayer` (Task 1.1); `NormalizedOutputMixin` (Task 1.2, for the stacking test).
- Produces:
  - `class InputLayerMixin(BaseModel)` with field `input_layer: InputLayer = "detect_mat"`
  - `def _read_input_layer(self, image) -> np.ndarray` — 2-D float32 `detect_mat`, or 3-D float32 RGB in [0,1]
  - `def _project_to_detect_mat(self, image, arr) -> np.ndarray` — 3-D collapses via `image.detect_mode`; 2-D passes through
  - `def _guard_input_range(self, arr) -> np.ndarray` — rescales to [0,1] iff out of range **and** `self.norm is not None`

> **Note on `_guard_input_range`:** it reads `self.norm`, which lives on
> `NormalizedOutputMixin`. Access it defensively via `getattr(self, "norm", "clip")`
> so `InputLayerMixin` stays usable standalone.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/mixin/test_input_layer_mixin.py`:

```python
"""Contract for InputLayerMixin: field append, layer read, projection, range guard."""
import numpy as np
import pytest
from pydantic import ValidationError

from phenotypic.abc_ import ContrastAdjustment
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import InputLayerMixin, NormalizedOutputMixin
from phenotypic.sdk_.typing_ import TuneSpec
from typing import Annotated


class _Probe(InputLayerMixin, NormalizedOutputMixin, ContrastAdjustment):
    """Probe enhancer.

    Args:
        gamma: Exponent.
        norm: Output normalization policy.
        input_layer: Source layer.
    """

    gamma: Annotated[float, TuneSpec(0.1, 5.0, log=True)] = 1.0

    def _operate(self, image):
        return image


def test_two_mixins_append_in_deterministic_order():
    """norm then input_layer, both after the op's own params."""
    assert list(_Probe.model_fields) == ["gamma", "norm", "input_layer"]


def test_order_holds_in_json_schema_and_serialization():
    assert list(_Probe.model_json_schema()["properties"]) == ["gamma", "norm", "input_layer"]
    import json
    params = json.loads(_Probe().to_json())["params"]
    assert list(params) == ["gamma", "norm", "input_layer"]


def test_tunespec_survives_both_forced_rebuilds():
    """Each mixin calls model_rebuild(force=True); Annotated metadata must persist."""
    meta = _Probe.model_fields["gamma"].metadata
    assert any(isinstance(m, TuneSpec) for m in meta)


def test_docstring_descriptions_reach_appended_fields():
    assert _Probe.model_fields["input_layer"].description == "Source layer."


def test_invalid_input_layer_rejected():
    with pytest.raises(ValidationError):
        _Probe(input_layer="gray")


def test_read_detect_mat_returns_2d():
    image = load_synth_yeast_plate()
    arr = _Probe(input_layer="detect_mat")._read_input_layer(image)
    assert arr.ndim == 2
    assert arr.dtype == np.float32


def test_read_rgb_returns_3d_float32_unit_range():
    image = load_synth_yeast_plate()
    arr = _Probe(input_layer="rgb")._read_input_layer(image)
    assert arr.ndim == 3 and arr.shape[2] == 3
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_project_collapses_3d_via_detect_mode():
    image = load_synth_yeast_plate()
    image.set_detect_mode("MinRGB")
    op = _Probe(input_layer="rgb")
    rgb = op._read_input_layer(image)
    out = op._project_to_detect_mat(image, rgb)
    assert out.shape == image.detect_mat[:].shape
    np.testing.assert_allclose(out, np.min(rgb, axis=2), atol=1e-6)


def test_project_passes_2d_through_unchanged():
    image = load_synth_yeast_plate()
    op = _Probe()
    arr = image.detect_mat[:]
    assert op._project_to_detect_mat(image, arr) is arr


def test_guard_rescales_negative_input():
    op = _Probe(norm="clip")
    arr = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    out = op._guard_input_range(arr)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-6)


def test_guard_is_identity_for_in_range_input():
    op = _Probe(norm="clip")
    arr = np.array([0.25, 0.75], dtype=np.float32)
    assert op._guard_input_range(arr) is arr


def test_guard_skipped_when_norm_is_none():
    """A GAT-stabilized signal (~[1.9, 2.3]) must not be normalized on the way in."""
    op = _Probe(norm=None)
    arr = np.array([1.9185, 2.3065], dtype=np.float32)
    assert op._guard_input_range(arr) is arr
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/sdk_/mixin/test_input_layer_mixin.py -v
```
Expected: FAIL — `ImportError: cannot import name 'InputLayerMixin'`

- [ ] **Step 3: Write the mixin**

Create `src/phenotypic/sdk_/mixin/_input_layer_mixin.py`:

```python
"""Mixin letting an operation read either ``detect_mat`` or the pristine ``rgb`` layer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel
from skimage.exposure import rescale_intensity

from phenotypic.sdk_.typing_ import InputLayer

if TYPE_CHECKING:
    from phenotypic._core._image import Image


class InputLayerMixin(BaseModel):
    """Adds an ``input_layer`` field selecting the operation's source array.

    Pointwise intensity curves are non-linear, so applying one to the three RGB
    channels and *then* collapsing to a detection matrix gives a different — often
    better — colony/background separation than collapsing first. This mixin exposes
    that choice without changing the output contract: the only layer an enhancer
    ever writes is still ``detect_mat``.

    When ``input_layer="rgb"`` the 3-D result is collapsed back to 2-D by projecting
    it through the image's own ``detect_mode``, so an upstream
    ``SetDetectMode(mode="MinRGB")`` is honoured.

    The field is **appended** to the end of the subclass's field order. When stacked
    with :class:`NormalizedOutputMixin`, list this mixin first; the resulting order
    is ``[…op params…, norm, input_layer]``.

    Note:
        Reading ``rgb`` discards any enhancement a prior operation wrote to
        ``detect_mat`` — the same behaviour as ``SetDetectMode``. This is documented,
        not enforced.
    """

    input_layer: InputLayer = "detect_mat"

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Move ``input_layer`` to the end of the subclass's field order."""
        super().__pydantic_init_subclass__(**kwargs)
        fields = cls.__pydantic_fields__
        if "input_layer" in fields and list(fields)[-1] != "input_layer":
            fields["input_layer"] = fields.pop("input_layer")
            cls.model_rebuild(force=True)

    def _read_input_layer(self, image: "Image") -> np.ndarray:
        """Return the source array for this operation.

        Returns:
            The 2-D ``detect_mat``, or a 3-D float32 RGB copy normalized to [0, 1].

        Raises:
            EmptyImageError: If ``input_layer="rgb"`` on a grayscale-only image.
        """
        if self.input_layer == "rgb":
            # ``normed()`` returns float64; halve the intermediate on large plates.
            return image.rgb.normed().astype(np.float32)
        return image.detect_mat[:]

    def _project_to_detect_mat(self, image: "Image", arr: np.ndarray) -> np.ndarray:
        """Collapse a 3-D array to 2-D via the image's ``detect_mode``.

        A 2-D array is returned unchanged (identity, not a copy).
        """
        if arr.ndim == 2:
            return arr
        from phenotypic._core._image_parts.detection_modes import get_detection_mode

        mode = get_detection_mode(image.detect_mode)
        return mode.compute_from_rgb(arr, image=image)

    def _guard_input_range(self, arr: np.ndarray) -> np.ndarray:
        """Rescale *arr* into [0, 1] when it strays outside, else return it unchanged.

        skimage's ``adjust_gamma`` / ``adjust_log`` / ``adjust_sigmoid`` raise
        ``ValueError`` on negative input, which a signed filter such as
        ``FocusEdgeLaplace`` produces. Skipped entirely when ``norm is None`` so a
        deliberately non-normalized (e.g. GAT-stabilized) signal is left alone.
        """
        if getattr(self, "norm", "clip") is None:
            return arr
        if arr.min() < 0.0 or arr.max() > 1.0:
            return rescale_intensity(arr, out_range=(0.0, 1.0))
        return arr
```

Export from `src/phenotypic/sdk_/mixin/__init__.py` and `src/phenotypic/sdk_/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/mixin/test_input_layer_mixin.py -v
```
Expected: PASS — **except** `test_project_collapses_3d_via_detect_mode`, which requires `compute_from_rgb` from Phase 3.

Mark that one test with `@pytest.mark.xfail(reason="compute_from_rgb lands in Phase 3", strict=True)` and **remove the marker in Task 3.3**. A strict xfail fails loudly once the feature arrives, so this cannot silently rot into a permanent skip.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/mixin/_input_layer_mixin.py \
        src/phenotypic/sdk_/mixin/__init__.py src/phenotypic/sdk_/__init__.py \
        tests/unit/sdk_/mixin/test_input_layer_mixin.py
git commit -m "feat(sdk_): add InputLayerMixin with appended input_layer field"
```

---

# Phase 2 — Retire `clip: bool` repo-wide

Depends on Phase 1. Eight classes, one public rename, one ClassVar contract change, plus the version bump.

---

### Task 2.1: `ClipControlMixin` → `NormControlMixin`

**Files:**
- Create: `src/phenotypic/sdk_/mixin/_norm_control_mixin.py`
- Delete: `src/phenotypic/sdk_/mixin/_clip_control_mixin.py`
- Modify: `src/phenotypic/sdk_/mixin/__init__.py`, `src/phenotypic/sdk_/__init__.py`, `src/phenotypic/sdk_/CLAUDE.md`
- Test: `tests/unit/sdk_/mixin/test_norm_control_mixin.py`

**Interfaces:**
- Consumes: `NormalizedOutputMixin` (Task 1.2).
- Produces: `NormControlMixin._disable_normalization(operation)` — returns a copy with `norm=None`; recurses into `ImagePipeline._ops`.

The old implementation duck-typed on `.clip` and set it to `False`. The new one duck-types on `.norm` and sets it to `None`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/mixin/test_norm_control_mixin.py`:

```python
"""NormControlMixin replaces ClipControlMixin: duck-types on `.norm`, sets None."""
import pytest

from phenotypic import ImagePipeline
from phenotypic.enhance import GaussianBlur, LocalEdgeDenoise
from phenotypic.sdk_ import NormControlMixin


def test_disable_normalization_sets_norm_none():
    enh = LocalEdgeDenoise(sigma_spatial=5, norm="clip")
    copied = NormControlMixin._disable_normalization(enh)
    assert enh.norm == "clip", "original must be untouched"
    assert copied.norm is None


def test_disable_normalization_is_noop_without_norm_field():
    blur = GaussianBlur(sigma=1.0)
    assert not hasattr(blur, "norm")
    assert NormControlMixin._disable_normalization(blur) is not None


def test_disable_normalization_recurses_into_pipeline():
    pipe = ImagePipeline(pipe_cfgs=[GaussianBlur(sigma=1.0), LocalEdgeDenoise(norm="clip")])
    copied = NormControlMixin._disable_normalization(pipe)
    assert list(copied._ops.values())[1].norm is None


def test_old_symbol_is_gone():
    with pytest.raises(ImportError):
        from phenotypic.sdk_ import ClipControlMixin  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/sdk_/mixin/test_norm_control_mixin.py -v
```
Expected: FAIL — `ImportError: cannot import name 'NormControlMixin'`

- [ ] **Step 3: Port the mixin**

`git mv src/phenotypic/sdk_/mixin/_clip_control_mixin.py src/phenotypic/sdk_/mixin/_norm_control_mixin.py`, then rewrite its body:

```python
"""Mixin for disabling output normalization on nested operations."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from phenotypic.abc_ import ImageEnhancer
    from phenotypic._core._image_pipeline import ImagePipeline


class NormControlMixin:
    """Disable output normalization on an inner operation or pipeline.

    Composite operations sometimes run an inner enhancer on data that is
    deliberately **not** in [0, 1] — e.g. variance-stabilized values from the
    Generalized Anscombe Transform, typically in the range ~1–32. Normalizing
    such a signal destroys the inverse transform.

    Duck-types on a ``norm`` attribute: any operation exposing one gets a copy
    with ``norm=None``. The original is left unchanged.

    Note:
        Renamed from ``ClipControlMixin`` in 0.18.0, when ``clip: bool`` became
        :data:`~phenotypic.sdk_.typing_.NormOut`. The old name is gone.

    Example:
        >>> from phenotypic.sdk_ import NormControlMixin
        >>> from phenotypic.enhance import LocalEdgeDenoise
        >>> enh = LocalEdgeDenoise(sigma_spatial=5, norm="clip")
        >>> copied = NormControlMixin._disable_normalization(enh)
        >>> enh.norm, copied.norm
        ('clip', None)
    """

    @staticmethod
    def _disable_normalization(
        operation: Union["ImageEnhancer", "ImagePipeline"],
    ) -> Union["ImageEnhancer", "ImagePipeline"]:
        """Return a copy of *operation* with ``norm=None`` wherever the field exists."""
        copied = copy.copy(operation)
        if hasattr(copied, "_ops"):
            copied._ops = {
                key: NormControlMixin._disable_normalization(op)
                for key, op in copied._ops.items()
            }
            return copied
        if hasattr(copied, "norm"):
            copied.norm = None
        return copied
```

Preserve whatever pipeline-recursion shape the original used — read it before rewriting and keep its `_ops` handling identical.

Update `src/phenotypic/sdk_/mixin/__init__.py`, `src/phenotypic/sdk_/__init__.py` (both the import and `__all__`), and the `### ClipControlMixin` heading in `src/phenotypic/sdk_/CLAUDE.md`.

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/sdk_/mixin/test_norm_control_mixin.py -v
```
Expected: PASS (4 passed). `test_disable_normalization_sets_norm_none` requires Task 2.3's `LocalEdgeDenoise` migration — run this task's tests again after 2.3.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/sdk_/ tests/unit/sdk_/mixin/test_norm_control_mixin.py
git commit -m "refactor(sdk_)!: rename ClipControlMixin to NormControlMixin"
```

---

### Task 2.2: `_GAT_DEFER_ATTRS` → `_GAT_DEFER_VALUES`

**Files:**
- Modify: `src/phenotypic/sdk_/mixin/_gat_support_mixin.py:60-61,113-126`
- Test: `tests/unit/sdk_/mixin/test_gat_support_mixin.py`

**Interfaces:**
- Produces: `_GAT_DEFER_VALUES: ClassVar[dict[str, Any]]` — maps attribute name → inert value.

A tuple cannot express this: `norm`'s inert value is `None` while `rescale_sigma`'s is `False`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/sdk_/mixin/test_gat_support_mixin.py`:

```python
def test_defer_values_is_a_mapping_with_correct_inert_values():
    from phenotypic.enhance import VisuShrinkEnhancer

    assert VisuShrinkEnhancer._GAT_DEFER_VALUES == {"norm": None, "rescale_sigma": False}


def test_defer_restores_original_values_after_gat_region(monkeypatch):
    """norm must be None *inside* the GAT region and restored afterwards."""
    from phenotypic.enhance import VisuShrinkEnhancer

    op = VisuShrinkEnhancer(use_gat=True, norm="clip", gat_scale_factor=255.0)
    seen = {}

    original = op._operate

    def spy(image):
        seen["norm_inside"] = op.norm
        return original(image)

    monkeypatch.setattr(op, "_operate", spy)
    from phenotypic.data import load_synth_yeast_plate

    op.apply(load_synth_yeast_plate())
    assert seen["norm_inside"] is None
    assert op.norm == "clip"
```

> The spy must wrap whatever callable `_gat_apply` receives. Read `_gat_apply`'s
> signature at `_gat_support_mixin.py:70` and target the actual inner `fn`, not
> `_operate`, if they differ.

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/sdk_/mixin/test_gat_support_mixin.py::test_defer_values_is_a_mapping_with_correct_inert_values -v
```
Expected: FAIL — `AttributeError: _GAT_DEFER_VALUES`

- [ ] **Step 3: Change the ClassVar contract**

In `_gat_support_mixin.py`, replace line 61:

```python
    _GAT_DEFER_VALUES: ClassVar[dict[str, Any]] = {}
```

Replace the snapshot and defer loop (lines ~113–126):

```python
        snapshot = {
            k: getattr(self, k)
            for k in (*self._GAT_NOISE_PARAMS, *self._GAT_DEFER_VALUES)
        }
        try:
            for k, v in self._GAT_NOISE_PARAMS.items():
                setattr(self, k, v)
            for k, v in self._GAT_DEFER_VALUES.items():
                setattr(self, k, v)
            fn(image)
        finally:
            for k, v in snapshot.items():
                setattr(self, k, v)
```

Update the class docstring: `_GAT_DEFER_ATTRS (ClassVar[tuple[str, ...]]): Boolean attributes that must be False…` becomes:

```
        ``_GAT_DEFER_VALUES`` (ClassVar[dict[str, Any]]):
            Attributes that must hold an inert value inside the GAT region,
            mapped to that value. Use for output-normalization policies
            (``{"norm": None}``) and skimage's ``rescale_sigma``
            (``{"rescale_sigma": False}``) -- any knob that would corrupt the
            stabilized round-trip. Restored after the inner call returns.
```

Ensure `Any` is imported.

- [ ] **Step 4: Update the six declaring classes**

| File | New value |
|---|---|
| `enhance/_visushrink_enhancer.py:125` | `{"norm": None, "rescale_sigma": False}` |
| `enhance/_bayesshrink_enhancer.py:137` | `{"norm": None, "rescale_sigma": False}` |
| `enhance/_local_edge_denoise.py:111` | `{"norm": None}` |
| `enhance/_enhance_block_match.py:128` | `{"norm": None}` |
| `correction/_visushrink_corrector.py:131` | `{"norm": None, "rescale_sigma": False}` |
| `correction/_bayesshrink_corrector.py:137` | `{"norm": None, "rescale_sigma": False}` |
| `enhance/_non_local_means.py:116` | `{}` |

Each becomes `_GAT_DEFER_VALUES: ClassVar[dict[str, Any]] = {...}`.

- [ ] **Step 5: Run the full GAT suite**

```bash
uv run pytest tests/unit/sdk_/mixin/test_gat_support_mixin.py -v
```
Expected: PASS. Existing `clip=True` call sites in this file (lines 75, 123, 185, 206) must be changed to `norm="clip"`.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/mixin/_gat_support_mixin.py src/phenotypic/enhance/ \
        src/phenotypic/correction/ tests/unit/sdk_/mixin/test_gat_support_mixin.py
git commit -m "refactor(sdk_)!: _GAT_DEFER_ATTRS becomes _GAT_DEFER_VALUES mapping"
```

---

### Task 2.3: Migrate the eight `clip: bool` classes

**Files (all Modify):**
- `src/phenotypic/enhance/_local_edge_denoise.py:124,150`
- `src/phenotypic/enhance/_bayesshrink_enhancer.py:143,162`
- `src/phenotypic/enhance/_enhance_block_match.py:133,153`
- `src/phenotypic/enhance/_visushrink_enhancer.py:131,150`
- `src/phenotypic/enhance/_composite_enhance.py:115,154`
- `src/phenotypic/correction/_color_denoise.py:127,218`
- `src/phenotypic/correction/_visushrink_corrector.py:139`
- `src/phenotypic/correction/_bayesshrink_corrector.py:145`
- `src/phenotypic/prefab/_heavy_round_peaks_pipeline.py:241`
- Test: `tests/unit/sdk_/test_norm_migration.py`

**Interfaces:**
- Consumes: `NormalizedOutputMixin` (Task 1.2).
- Produces: each class inherits `NormalizedOutputMixin`; `clip: bool` field deleted; `if self.clip:` → `self._apply_norm(...)`.

Mapping: `clip=True → norm="clip"`, `clip=False → norm=None`.
**`rescale_sigma` is untouched** — it is forwarded to skimage's `denoise_wavelet` and governs noise-sigma scaling, not output range.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_norm_migration.py`:

```python
"""The clip -> norm migration, across every class that carried `clip: bool`."""
import pytest
from pydantic import ValidationError

from phenotypic.correction import BayesShrinkCorrector, ColorDenoise, VisuShrinkCorrector
from phenotypic.enhance import (
    BayesShrinkEnhancer,
    CompositeEnhance,
    EnhanceBlockMatch,
    LocalEdgeDenoise,
    VisuShrinkEnhancer,
)

MIGRATED = [
    LocalEdgeDenoise, BayesShrinkEnhancer, EnhanceBlockMatch,
    VisuShrinkEnhancer, CompositeEnhance,
    ColorDenoise, VisuShrinkCorrector, BayesShrinkCorrector,
]


@pytest.mark.parametrize("cls", MIGRATED, ids=lambda c: c.__name__)
def test_clip_field_is_gone(cls):
    assert "clip" not in cls.model_fields


@pytest.mark.parametrize("cls", MIGRATED, ids=lambda c: c.__name__)
def test_norm_field_exists_and_is_last(cls):
    assert list(cls.model_fields)[-1] == "norm"


@pytest.mark.parametrize("cls", MIGRATED, ids=lambda c: c.__name__)
def test_legacy_clip_kwarg_raises_migration_message(cls):
    with pytest.raises(ValidationError, match=r"`clip` was replaced by `norm` in 0\.18\.0"):
        cls(clip=True)


def test_composite_enhance_defaults_to_none():
    """Preserves the old `clip: bool = False` default."""
    assert CompositeEnhance().norm is None


@pytest.mark.parametrize(
    "cls", [c for c in MIGRATED if c is not CompositeEnhance], ids=lambda c: c.__name__
)
def test_others_default_to_clip(cls):
    assert cls().norm == "clip"


@pytest.mark.parametrize("cls", MIGRATED, ids=lambda c: c.__name__)
def test_norm_none_round_trips_through_json(cls):
    loaded = type(cls()).from_json(cls(norm=None).to_json())
    assert loaded.norm is None


def test_rescale_sigma_untouched():
    assert VisuShrinkEnhancer.model_fields["rescale_sigma"].annotation is bool
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/sdk_/test_norm_migration.py -v
```
Expected: FAIL — `assert "clip" not in cls.model_fields`

- [ ] **Step 3: Migrate each class**

For each of the five enhancers and three correctors:

1. Add `NormalizedOutputMixin` as the **first** base:
   `class LocalEdgeDenoise(NormalizedOutputMixin, _GATSupportMixin, ImageDenoiser):`
   (preserve the existing base order after it).
2. Delete the `clip: bool = True` field line.
3. Delete the `clip:` entry from the docstring `Args:` block; add a `norm:` entry:

```
        norm: Output range policy. ``"clip"`` (default) saturates values outside
            [0, 1]; ``"rescale"`` remaps the full observed range onto [0, 1];
            ``None`` passes values through untouched. Automatically deferred to
            ``None`` when ``use_gat=True``.
```

   (Drop the final sentence for classes without `_GATSupportMixin`, i.e. `CompositeEnhance` and `ColorDenoise`.)

4. Replace the guard. `enhance/_visushrink_enhancer.py:150` currently reads:

```python
        if self.clip:
            denoised = denoised.clip(0.0, 1.0)
        image.detect_mat[:] = denoised
```

becomes:

```python
        image.detect_mat[:] = self._apply_norm(denoised)
```

Apply the same shape at each of the eight sites.

5. `CompositeEnhance` overrides the default:

```python
    norm: NormOut = None
```

   and its `if self.clip: combined = np.clip(combined, 0.0, 1.0)` at line 154 becomes `combined = self._apply_norm(combined)`.

6. `prefab/_heavy_round_peaks_pipeline.py:241` — the local `bm3d_clip` variable feeding `clip=bm3d_clip` becomes `norm=bm3d_norm`, typed `NormOut`. Trace the variable to its definition and update its type and default (`True → "clip"`).

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/sdk_/test_norm_migration.py \
              tests/unit/sdk_/mixin/test_norm_control_mixin.py \
              tests/unit/sdk_/mixin/test_gat_support_mixin.py -v
```
Expected: PASS

- [ ] **Step 5: Update remaining `clip=` call sites**

```bash
grep -rn "clip=" tests/ src/phenotypic/ --exclude-dir=__pycache__
```
Expected remaining hits: `tests/unit/correction/test_color_denoise.py:55`, `tests/unit/enhance/test_composite_enhance.py:74,139`. Change `clip=False → norm=None`, `clip=True → norm="clip"`.

Then run the full affected suites:

```bash
uv run pytest tests/unit/enhance tests/unit/correction -q
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add -A src/phenotypic tests/
git commit -m "refactor!: replace clip: bool with norm: NormOut across 8 classes"
```

---

### Task 2.4: Regenerate back-compat fixtures + bump to `0.18.0`

**Files:**
- Modify: `src/phenotypic/__init__.py:17`
- Modify: `tests/fixtures/tune/back_compat_pipelines/local_edge_denoise_small_sigma.json`
- Modify: `tests/fixtures/tune/back_compat_pipelines/bm3d_zero_sigma.json`
- Test: `tests/unit/sdk_/test_norm_migration.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/sdk_/test_norm_migration.py`:

```python
def test_version_is_0_18_0():
    import phenotypic

    assert phenotypic.__version__ == "0.18.0"


def test_back_compat_fixtures_carry_norm_not_clip():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "fixtures" / "tune" / "back_compat_pipelines"
    fixtures = sorted(root.glob("*.json"))
    assert fixtures, "fixture directory must not be empty"
    for fp in fixtures:
        blob = fp.read_text()
        assert '"clip"' not in blob, f"{fp.name} still pins the removed `clip` key"


def test_back_compat_fixtures_still_deserialize():
    from pathlib import Path

    from phenotypic import ImagePipeline

    root = Path(__file__).resolve().parents[2] / "fixtures" / "tune" / "back_compat_pipelines"
    for fp in sorted(root.glob("*.json")):
        assert ImagePipeline.from_json(fp) is not None
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/unit/sdk_/test_norm_migration.py::test_version_is_0_18_0 -v
```
Expected: FAIL — `assert '0.17.3' == '0.18.0'`

- [ ] **Step 3: Bump the version**

`src/phenotypic/__init__.py:17`:

```python
__version__ = "0.18.0"
```

- [ ] **Step 4: Regenerate the fixtures**

In each fixture's `params` block, replace `"clip": true` with `"norm": "clip"`, and update `"version"` to `"0.18.0"`. Do **not** hand-edit blindly — verify by round-tripping:

```bash
uv run python -c "
from pathlib import Path
from phenotypic import ImagePipeline
for fp in sorted(Path('tests/fixtures/tune/back_compat_pipelines').glob('*.json')):
    print(fp.name, '->', ImagePipeline.from_json(fp) is not None)
"
```
Expected: each prints `-> True` with no `ValidationError`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/unit/sdk_/test_norm_migration.py -v && uv run pytest tests/unit/tune -q
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/__init__.py tests/fixtures/tune/back_compat_pipelines/ \
        tests/unit/sdk_/test_norm_migration.py
git commit -m "chore!: bump to 0.18.0 and regenerate back-compat fixtures for norm"
```

---

# Phase 3 — `rgb_to_xyz` extraction + `DetectionMode.compute_from_rgb`

**Independent of Phases 1, 2, 4.** Safe to run in parallel (disjoint files).

---

### Task 3.1: Extract `rgb_to_xyz()` from `XyzAccessor`

**Files:**
- Create: `src/phenotypic/_core/_image_parts/color_space_accessors/_xyz_conversion.py`
- Modify: `src/phenotypic/_core/_image_parts/color_space_accessors/_xyz_accessor.py:108-152`
- Test: `tests/unit/core/test_xyz_conversion.py`

**Interfaces:**
- Produces: `rgb_to_xyz(rgb_normed: np.ndarray, *, gamma, illuminant: str, observer: str) -> np.ndarray`

This is a **pure lift**. No logic changes. The class is `XyzAccessor` (lowercase `yz`).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/core/test_xyz_conversion.py`:

```python
"""rgb_to_xyz is a pure lift out of XyzAccessor: byte-identical output."""
import numpy as np

from phenotypic._core._image_parts.color_space_accessors._xyz_conversion import rgb_to_xyz
from phenotypic.data import load_synth_yeast_plate


def test_matches_accessor_exactly():
    """The extraction must not perturb a single value. Same code, same inputs."""
    image = load_synth_yeast_plate()
    via_accessor = image.color.XYZ[:]
    via_function = rgb_to_xyz(
        image.rgb.normed(),
        gamma=image.gamma,
        illuminant=image.illuminant,
        observer=image._observer,
    )
    np.testing.assert_array_equal(via_function, via_accessor)


def test_unknown_illuminant_raises():
    import pytest

    image = load_synth_yeast_plate()
    with pytest.raises(ValueError, match="Unknown color_profile|illuminant"):
        rgb_to_xyz(image.rgb.normed(), gamma=image.gamma, illuminant="D99",
                   observer=image._observer)
```

`assert_array_equal` (not `allclose`) is correct here: a pure code move that produces
*any* different bit is a bug, not a rounding artifact.

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/core/test_xyz_conversion.py -v
```
Expected: FAIL — `ModuleNotFoundError: ..._xyz_conversion`

- [ ] **Step 3: Create the free function**

Create `_xyz_conversion.py`, moving the `match` block from `_xyz_accessor.py:110-152` **verbatim**, substituting parameters for `self._root_image.*`:

```python
"""Pure RGB -> CIE XYZ conversion, independent of any Image instance.

Lifted out of :class:`XyzAccessor` so a *substitute* RGB array (e.g. one an
enhancer has just gamma-corrected) can be projected through the same colour
pipeline the accessor uses, carrying the source image's colour configuration.
"""

from __future__ import annotations

import colour
import numpy as np

from phenotypic.sdk_.colourspace import sRGB_D50
from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS


def rgb_to_xyz(
    rgb_normed: np.ndarray,
    *,
    gamma: GAMMA_ENCODINGS,
    illuminant: str,
    observer: str,
) -> np.ndarray:
    """Convert a normalized RGB array to CIE XYZ tristimulus values.

    Args:
        rgb_normed: RGB array normalized to [0, 1], shape ``(rows, cols, 3)``.
        gamma: The image's gamma encoding (``GAMMA_ENCODINGS.SRGB`` or ``LINEAR``).
        illuminant: ``"D50"`` or ``"D65"``.
        observer: CIE standard observer name, e.g.
            ``"CIE 1931 2 Degree Standard Observer"``.

    Returns:
        np.ndarray: XYZ array, shape ``(rows, cols, 3)``, dtype float64.

    Raises:
        ValueError: If the gamma/illuminant combination is unrecognized.
    """
    match (gamma, illuminant):
        case (GAMMA_ENCODINGS.SRGB, "D50"):
            sRGB_D50.whitepoint = colour.CCS_ILLUMINANTS[observer]["D50"]
            return colour.RGB_to_XYZ(
                RGB=rgb_normed,
                colourspace=sRGB_D50,
                illuminant=sRGB_D50.whitepoint,
                apply_cctf_decoding=True,
            )
        case (GAMMA_ENCODINGS.SRGB, "D65"):
            return colour.RGB_to_XYZ(
                RGB=rgb_normed,
                colourspace=colour.RGB_COLOURSPACES["sRGB"],
                illuminant=colour.CCS_ILLUMINANTS[observer]["D65"],
                apply_cctf_decoding=True,
            )
        case (GAMMA_ENCODINGS.LINEAR, "D50"):
            sRGB_D50.whitepoint = colour.CCS_ILLUMINANTS[observer]["D50"]
            return colour.RGB_to_XYZ(
                RGB=rgb_normed,
                colourspace=colour.RGB_COLOURSPACES["sRGB"],
                illuminant=sRGB_D50.whitepoint,
                apply_cctf_decoding=False,
            )
        case (GAMMA_ENCODINGS.LINEAR, "D65"):
            return colour.RGB_to_XYZ(
                RGB=rgb_normed,
                colourspace=colour.RGB_COLOURSPACES["sRGB"],
                illuminant=colour.CCS_ILLUMINANTS[observer]["D65"],
                apply_cctf_decoding=False,
            )
        case _:
            raise ValueError(
                f"Unknown color_profile: {gamma} or illuminant: {illuminant}"
            )
```

Then reduce `XyzAccessor._subject_arr` to:

```python
    @property
    def _subject_arr(self) -> np.ndarray:
        if self._root_image.rgb.isempty():
            raise AttributeError("XYZ conversion is not available for grayscale images")
        return rgb_to_xyz(
            self._root_image.rgb.normed(),
            gamma=self._root_image.gamma,
            illuminant=self._root_image.illuminant,
            observer=self._root_image._observer,
        )
```

Keep the original docstring on `_subject_arr`. Remove now-unused imports (`colour`, `sRGB_D50`, `GAMMA_ENCODINGS`) from `_xyz_accessor.py` if nothing else uses them.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/core/test_xyz_conversion.py -v && uv run pytest tests/unit/core -q -k "color or xyz or lab"
```
Expected: PASS

- [ ] **Step 5: Prove the equality test can fail**

Temporarily flip `apply_cctf_decoding=True → False` in the `(SRGB, "D65")` arm. Re-run `test_matches_accessor_exactly`; it must FAIL. Restore.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_core/_image_parts/color_space_accessors/ tests/unit/core/test_xyz_conversion.py
git commit -m "refactor(core): extract rgb_to_xyz free function from XyzAccessor"
```

---

### Task 3.2: Add `compute_from_rgb` to `DetectionMode` and all 11 modes

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/detection_modes/_detection_mode.py`
- Modify: `detection_modes/_gray_mode.py`, `_color_channel_modes.py`, `_min_rgb_mode.py`, `_lab_channel_modes.py`, `_hsv_channel_modes.py`, `_inv_saturation_mode.py`
- Test: `tests/unit/core/test_detection_modes_from_rgb.py` (Task 3.3)

**Interfaces:**
- Consumes: `rgb_to_xyz` (Task 3.1).
- Produces: `DetectionMode.compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray`
  where `rgb` is float, shape `(rows, cols, 3)`, values in [0, 1]. `image` supplies colour
  configuration only — `image._data.rgb` is **never** read.

- [ ] **Step 1: Add the abstract method**

In `_detection_mode.py`, inside `class DetectionMode(ABC)`:

```python
    @abstractmethod
    def compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray:
        """Project a substitute RGB array to a 2-D detection matrix.

        Unlike :meth:`compute`, the pixel data comes from *rgb* rather than from
        *image*. ``image`` supplies colour configuration only (``gamma``,
        ``illuminant``, ``_observer``), which the CIE L*a*b* modes require and the
        others ignore.

        Args:
            rgb: Float RGB array normalized to [0, 1], shape ``(rows, cols, 3)``.
            image: Source image, consulted for colour configuration only.

        Returns:
            A 2-D float32 array normalized to [0, 1].
        """
```

- [ ] **Step 2: Implement per mode**

`_gray_mode.py` — **keep `compute()` as-is**; it must stay usable on grayscale-only images (`requires_rgb` is `False`). Add:

```python
    def compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray:
        from skimage.color import rgb2gray

        return rgb2gray(rgb).astype(np.float32)
```

`_color_channel_modes.py` — on `_ColorChannelMode`:

```python
    def compute(self, image: Image) -> np.ndarray:
        assert image._data.rgb is not None
        return self.compute_from_rgb(normalize_rgb_bitdepth(image._data.rgb), image=image)

    def compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray:
        return rgb[:, :, self._channel_index].astype(np.float32)
```

`_min_rgb_mode.py`:

```python
    def compute(self, image: Image) -> np.ndarray:
        assert image._data.rgb is not None
        return self.compute_from_rgb(normalize_rgb_bitdepth(image._data.rgb), image=image)

    def compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray:
        return np.min(rgb, axis=2).astype(np.float32)
```

`_hsv_channel_modes.py` — on `_HsvChannelMode`:

```python
    def compute(self, image: Image) -> np.ndarray:
        return self.compute_from_rgb(normalize_rgb_bitdepth(image._data.rgb), image=image)

    def compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray:
        from skimage.color import rgb2hsv

        return rgb2hsv(rgb)[:, :, self._channel_index].astype(np.float32)
```

`_inv_saturation_mode.py`:

```python
    def compute(self, image: Image) -> np.ndarray:
        return self.compute_from_rgb(normalize_rgb_bitdepth(image._data.rgb), image=image)

    def compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray:
        from skimage.color import rgb2hsv

        return 1.0 - rgb2hsv(rgb)[:, :, 1].astype(np.float32)
```

`_lab_channel_modes.py` — on `_LabChannelMode`:

```python
    def compute(self, image: Image) -> np.ndarray:
        return self.compute_from_rgb(image.rgb.normed(), image=image)

    def compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray:
        import colour

        from phenotypic._core._image_parts.color_space_accessors._xyz_conversion import (
            rgb_to_xyz,
        )

        xyz = rgb_to_xyz(
            rgb,
            gamma=image.gamma,
            illuminant=image.illuminant,
            observer=image._observer,
        )
        lab = colour.XYZ_to_Lab(
            XYZ=xyz,
            illuminant=colour.CCS_ILLUMINANTS[image._observer][image.illuminant],
        )
        return self._normalize_channel(lab[:, :, self._channel_index])
```

Add `import numpy as np` and the `normalize_rgb_bitdepth` import where each file needs them.

- [ ] **Step 3: Run the existing detect-mode suite (no regressions)**

```bash
uv run pytest tests/unit -q -k "detect_mode or detection_mode"
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/phenotypic/_core/_image_parts/detection_modes/
git commit -m "feat(core): add DetectionMode.compute_from_rgb across all 11 modes"
```

---

### Task 3.3: The golden equivalence gate

**Files:**
- Create: `tests/unit/core/test_detection_modes_from_rgb.py`
- Modify: `tests/unit/sdk_/mixin/test_input_layer_mixin.py` (remove the Task 1.3 xfail)

- [ ] **Step 1: Write the test**

```python
"""Golden gate: compute() and compute_from_rgb() must agree, for every mode.

Tolerance derivation (mechanism, not guess): the two paths differ only by float
accumulation order in a 3-term dot product (rgb2gray) or a 3x3 matmul
(RGB_to_XYZ), computed in float64 then cast to float32. float32 eps is 1.19e-7;
three fused operations bound the discrepancy at ~3 x 1.19e-7 = 3.6e-7. atol=1e-6
sits ~3x above that -- loose enough to survive reassociation, tight enough that a
channel swap or a dropped CCTF decode (which move values by >1e-2) fails.
"""
import numpy as np
import pytest

from phenotypic._core._image_parts.detection_modes import available_modes, get_detection_mode
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_.funcs_ import normalize_rgb_bitdepth

ATOL = 1e-6


@pytest.fixture(scope="module")
def image():
    return load_synth_yeast_plate()


def test_all_eleven_modes_are_registered():
    assert len(available_modes()) == 11


@pytest.mark.parametrize("name", available_modes())
def test_compute_from_rgb_matches_compute(name, image):
    mode = get_detection_mode(name)
    expected = mode.compute(image)
    actual = mode.compute_from_rgb(normalize_rgb_bitdepth(image.rgb[:]), image=image)
    assert actual.shape == expected.shape
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=ATOL)


@pytest.mark.parametrize("name", available_modes())
def test_compute_from_rgb_output_is_unit_range(name, image):
    out = get_detection_mode(name).compute_from_rgb(
        normalize_rgb_bitdepth(image.rgb[:]), image=image
    )
    assert out.min() >= -ATOL and out.max() <= 1.0 + ATOL


def test_compute_from_rgb_ignores_the_images_own_rgb(image):
    """Feeding a different array must produce a different result -- proving the
    method reads `rgb`, not `image._data.rgb`."""
    mode = get_detection_mode("MinRGB")
    baseline = mode.compute_from_rgb(normalize_rgb_bitdepth(image.rgb[:]), image=image)
    darkened = mode.compute_from_rgb(normalize_rgb_bitdepth(image.rgb[:]) * 0.5, image=image)
    assert not np.allclose(baseline, darkened, atol=ATOL)
    np.testing.assert_allclose(darkened, baseline * 0.5, atol=ATOL)
```

- [ ] **Step 2: Run it**

```bash
uv run pytest tests/unit/core/test_detection_modes_from_rgb.py -v
```
Expected: PASS (24 passed)

- [ ] **Step 3: Prove the golden test can fail (mutation check)**

In `_lab_channel_modes.py::_LabChannelMode.compute_from_rgb`, temporarily pass
`observer="CIE 1964 10 Degree Standard Observer"` (a different whitepoint) to
`rgb_to_xyz`. Re-run:

```bash
uv run pytest tests/unit/core/test_detection_modes_from_rgb.py -k "LabA" -v
```
Expected: **FAIL**, confirming `atol=1e-6` is tight enough to catch a wrong colour
configuration. Restore the correct argument and re-run to green.

- [ ] **Step 4: Remove the Phase-1 xfail**

Delete the `@pytest.mark.xfail(...)` marker from
`test_project_collapses_3d_via_detect_mode` in `tests/unit/sdk_/mixin/test_input_layer_mixin.py`.
(If Phase 1 has not landed yet, defer this step to whichever phase finishes second.)

```bash
uv run pytest tests/unit/sdk_/mixin/test_input_layer_mixin.py -v
```
Expected: PASS, no xfail/xpass

- [ ] **Step 5: Commit**

```bash
git add tests/unit/core/test_detection_modes_from_rgb.py tests/unit/sdk_/mixin/test_input_layer_mixin.py
git commit -m "test(core): golden gate pinning compute_from_rgb against compute"
```

---

# Phase 4 — `FocusEdgeLaplace` + the `[0,1]` invariant gate

Depends on Phase 1 (needs `norm`).

---

### Task 4.1: Normalize `FocusEdgeLaplace` and pin the invariant

**Files:**
- Modify: `src/phenotypic/enhance/_focus_edge_laplace.py:64-70`
- Create: `tests/unit/enhance/test_detect_mat_invariant.py`

**Interfaces:**
- Consumes: `NormalizedOutputMixin` (Task 1.2).

**Why `norm="rescale"` is the default here, not `"clip"`:** a Laplacian is signed by
construction; it currently emits `[-1.5157, +1.4787]` on the synth plate (114,268 negative
pixels). Clipping would map the entire negative lobe to zero, destroying half the edge
response. Rescaling preserves the full bipolar structure inside [0, 1]. This is the one
place in the codebase where `"rescale"` is the right default.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/enhance/test_detect_mat_invariant.py`:

```python
"""Repo-wide gate: every enhancer leaves detect_mat within [0, 1].

Before this gate, FocusEdgeLaplace violated the contract at [-1.5157, +1.4787].
"""
import numpy as np
import pytest

import phenotypic.enhance as enhance
from phenotypic.data import load_synth_yeast_plate

TOL = 1e-6


def _zero_arg_enhancers():
    """Every exported enhancer constructible with no arguments.

    A class that cannot be constructed is a hard failure, not a skip -- a silent
    skip would report green while covering nothing.
    """
    names = []
    for name in enhance.__all__:
        cls = getattr(enhance, name)
        cls()  # must not raise; no try/except -- see module docstring
        names.append(name)
    return names


@pytest.mark.parametrize("name", _zero_arg_enhancers())
def test_enhancer_output_stays_in_unit_range(name):
    op = getattr(enhance, name)()
    out = op.apply(load_synth_yeast_plate())
    dm = out.detect_mat[:]
    assert dm.min() >= -TOL, f"{name} emits {dm.min():+.4f} < 0"
    assert dm.max() <= 1.0 + TOL, f"{name} emits {dm.max():+.4f} > 1"


def test_laplace_preserves_bipolar_structure():
    """Rescale, not clip: the negative lobe must survive as sub-midpoint values."""
    from phenotypic.enhance import FocusEdgeLaplace

    assert FocusEdgeLaplace().norm == "rescale"
    dm = FocusEdgeLaplace().apply(load_synth_yeast_plate()).detect_mat[:]
    assert (dm < 0.5).any() and (dm > 0.5).any(), "response collapsed to one sign"
    assert np.isclose(dm.min(), 0.0, atol=1e-3) and np.isclose(dm.max(), 1.0, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/enhance/test_detect_mat_invariant.py -v
```
Expected: FAIL — `FocusEdgeLaplace emits -1.5157 < 0`, plus `AttributeError: norm`

- [ ] **Step 3: Migrate `FocusEdgeLaplace`**

```python
class FocusEdgeLaplace(NormalizedOutputMixin, FocusEdge):
    ...
    norm: NormOut = "rescale"

    def _operate(self, image: Image) -> Image:
        response = laplace(
            image=image.detect_mat[:],
            ksize=self.kernel_size,
            mask=self.mask,
        )
        image.detect_mat[:] = self._apply_norm(response)
        return image
```

Add to the docstring `Args:` block:

```
        norm: Output range policy. Defaults to ``"rescale"`` because a Laplacian
            is signed: clipping would map the entire negative lobe to zero and
            discard half the edge response. ``None`` passes the raw bipolar
            response through, leaving ``detect_mat`` outside its [0, 1] contract.
```

Add to the class docstring a `Note:` recording the behaviour change:

```
    Note:
        Changed in 0.18.0: the output is now normalized to [0, 1] (previously it
        could span roughly [-1.5, +1.5]). Downstream thresholds tuned against the
        old raw response must be re-tuned.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/enhance/test_detect_mat_invariant.py -v
```
Expected: PASS (31 passed — 30 enhancers + the Laplace structure test)

- [ ] **Step 5: Prove the gate can fail**

Temporarily set `norm: NormOut = None` on `FocusEdgeLaplace`. Re-run; the gate must
report `FocusEdgeLaplace emits -1.5157 < 0`. Restore `"rescale"`.

- [ ] **Step 6: Run the full enhance suite for regressions**

```bash
uv run pytest tests/unit/enhance -q
```
Expected: PASS. `FocusEdgeLaplace`'s own existing tests may assert the old range — update them to the new contract, and note the change in their docstrings.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/enhance/_focus_edge_laplace.py tests/unit/enhance/
git commit -m "fix(enhance)!: normalize FocusEdgeLaplace output; gate detect_mat in [0,1]"
```

---

# Phase 5 — The four contrast operations

Depends on Phase 1 (mixins) and Phase 3 (`compute_from_rgb`).

---

### Task 5.1: `ContrastGamma`

**Files:**
- Create: `src/phenotypic/enhance/_contrast_gamma.py`
- Modify: `src/phenotypic/enhance/__init__.py`
- Modify: `tests/unit/abc_/test_enhancer_taxonomy.py:66` (roster)
- Test: `tests/unit/enhance/test_contrast_ops.py`

**Interfaces:**
- Consumes: `InputLayerMixin`, `NormalizedOutputMixin` (Phase 1); `compute_from_rgb` (Phase 3).
- Produces: `ContrastGamma(gamma: float = 1.0, gain: float = 1.0, norm: NormOut = "clip", input_layer: InputLayer = "detect_mat")`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/enhance/test_contrast_ops.py`:

```python
"""Behavioural contract for the Contrast* enhancers."""
import numpy as np
import pytest
from skimage.exposure import adjust_gamma

from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import ContrastGamma


def test_field_order_is_params_then_norm_then_input_layer():
    assert list(ContrastGamma.model_fields) == ["gamma", "gain", "norm", "input_layer"]


def test_identity_at_defaults():
    """gamma=1, gain=1 is the identity curve; clip is a no-op on in-range input."""
    image = load_synth_yeast_plate()
    before = image.detect_mat[:].copy()
    after = ContrastGamma().apply(image).detect_mat[:]
    np.testing.assert_allclose(after, before, atol=1e-6)


def test_gamma_gt_one_darkens_midtones():
    image = load_synth_yeast_plate()
    before = image.detect_mat[:].copy()
    after = ContrastGamma(gamma=2.0).apply(image).detect_mat[:]
    assert after.mean() < before.mean()


def test_matches_skimage_on_detect_mat():
    image = load_synth_yeast_plate()
    src = image.detect_mat[:].copy()
    expected = np.clip(adjust_gamma(src, gamma=2.0, gain=1.5), 0.0, 1.0)
    actual = ContrastGamma(gamma=2.0, gain=1.5).apply(image).detect_mat[:]
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_gain_is_meaningful_under_clip():
    """The whole reason `norm` defaults to clip rather than rescale."""
    image = load_synth_yeast_plate()
    a = ContrastGamma(gamma=2.0, gain=1.0, norm="clip").apply(image).detect_mat[:]
    b = ContrastGamma(gamma=2.0, gain=1.9, norm="clip").apply(image).detect_mat[:]
    assert np.abs(a - b).max() > 1e-2


def test_gain_is_absorbed_under_rescale():
    """Documented consequence of norm='rescale': gain is a uniform post-curve scale."""
    image = load_synth_yeast_plate()
    a = ContrastGamma(gamma=2.0, gain=1.0, norm="rescale").apply(image).detect_mat[:]
    b = ContrastGamma(gamma=2.0, gain=1.9, norm="rescale").apply(image).detect_mat[:]
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_rgb_path_differs_from_detect_mat_path():
    """Non-linear curve then projection != projection then curve. The whole point."""
    image = load_synth_yeast_plate()
    image.set_detect_mode("MinRGB")
    via_dm = ContrastGamma(gamma=2.5, input_layer="detect_mat").apply(image).detect_mat[:]
    via_rgb = ContrastGamma(gamma=2.5, input_layer="rgb").apply(image).detect_mat[:]
    assert np.abs(via_dm - via_rgb).max() > 1e-3


def test_rgb_path_writes_only_detect_mat():
    image = load_synth_yeast_plate()
    rgb_before = image.rgb[:].copy()
    gray_before = image.gray[:].copy()
    out = ContrastGamma(gamma=2.0, input_layer="rgb").apply(image)
    np.testing.assert_array_equal(out.rgb[:], rgb_before)
    np.testing.assert_array_equal(out.gray[:], gray_before)


def test_negative_input_is_rescaled_not_raised():
    """FocusEdgeLaplace can emit negatives; skimage would raise ValueError."""
    from phenotypic.enhance import FocusEdgeLaplace

    image = FocusEdgeLaplace(norm=None).apply(load_synth_yeast_plate())
    assert image.detect_mat[:].min() < 0
    out = ContrastGamma(gamma=2.0).apply(image)
    assert 0.0 <= out.detect_mat[:].min() and out.detect_mat[:].max() <= 1.0


def test_json_round_trip():
    from phenotypic.abc_ import ImageOperation

    op = ContrastGamma(gamma=2.0, gain=1.5, norm=None, input_layer="rgb")
    loaded = ImageOperation.from_json(op.to_json())
    assert loaded.gamma == 2.0 and loaded.norm is None and loaded.input_layer == "rgb"


def test_rgb_on_grayscale_image_raises():
    from phenotypic import Image
    from phenotypic.sdk_.exceptions_ import EmptyImageError

    gray_only = Image(np.full((32, 32), 0.5, dtype=np.float32))
    with pytest.raises((EmptyImageError, AttributeError)):
        ContrastGamma(input_layer="rgb").apply(gray_only)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/enhance/test_contrast_ops.py -v
```
Expected: FAIL — `ImportError: cannot import name 'ContrastGamma'`

- [ ] **Step 3: Write `ContrastGamma`**

Create `src/phenotypic/enhance/_contrast_gamma.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.exposure import adjust_gamma

from ..abc_ import ContrastAdjustment
from ..sdk_.mixin import InputLayerMixin, NormalizedOutputMixin
from ..sdk_.typing_ import TuneSpec


class ContrastGamma(InputLayerMixin, NormalizedOutputMixin, ContrastAdjustment):
    """Apply a power-law (gamma) intensity curve to boost faint or washed-out colonies.

    Raises each pixel to the power ``gamma`` after normalizing to [0, 1], then scales
    by ``gain``. Values above 1 darken midtones and deepen the agar background, making
    bright colonies stand out. Values below 1 brighten midtones, rescuing faint or
    translucent colonies that a global threshold would otherwise miss.

    Unlike :class:`ContrastStretching`, the mapping is non-linear, so it redistributes
    tonal weight rather than merely rescaling the range.

    Best For:
        - Faint or translucent colonies lost against a bright agar background
          (``gamma`` below 1).
        - Over-exposed plates where colony interiors saturate (``gamma`` above 1).
        - Pigmented colonies whose colour separation is stronger in a single channel:
          set ``input_layer="rgb"`` so the curve applies per-channel before the
          detection matrix is derived.

    Consider Also:
        - :class:`ContrastStretching` when the histogram is merely narrow and a
          linear remap suffices.
        - :class:`ContrastSigmoid` when you want to steepen contrast around a
          specific intensity rather than across the whole range.

    Args:
        gamma: Power-law exponent. Below 1 brightens midtones; above 1 darkens them.
            ``1.0`` is the identity. Typical range: 0.5--2.5. Default: 1.0.
        gain: Constant multiplier applied after the curve. Default: 1.0.
            Has no effect when ``norm="rescale"``, which divides it back out.
        norm: Output range policy. ``"clip"`` (default) saturates values outside
            [0, 1]; ``"rescale"`` remaps the full observed range onto [0, 1];
            ``None`` passes values through untouched.
        input_layer: Source layer. ``"detect_mat"`` (default) applies the curve to
            the 2-D detection matrix. ``"rgb"`` applies it to all three colour
            channels, then collapses the result to 2-D through the image's own
            ``detect_mode``. Because the curve is non-linear, the two routes give
            different results.

    Returns:
        Image: Input image with ``detect_mat`` gamma-corrected. ``rgb`` and ``gray``
        are unchanged. With ``input_layer="rgb"``, any enhancement a prior operation
        wrote to ``detect_mat`` is discarded, as with :class:`SetDetectMode`.

    Examples:
        Darken the background to sharpen bright yeast colonies:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import ContrastGamma
        >>> plate = load_synth_yeast_plate()
        >>> enhanced = ContrastGamma(gamma=2.0).apply(plate)
        >>> float(enhanced.detect_mat[:].max()) <= 1.0
        True

        Apply the curve in colour space before deriving the detection matrix:

        >>> plate = load_synth_yeast_plate()
        >>> plate.set_detect_mode('MinRGB')
        >>> enhanced = ContrastGamma(gamma=2.0, input_layer='rgb').apply(plate)
        >>> enhanced.detect_mat[:].ndim
        2
    """

    gamma: Annotated[float, TuneSpec(0.1, 5.0, log=True)] = 1.0
    gain: Annotated[float, TuneSpec(0.5, 2.0)] = 1.0

    def _operate(self, image: Image) -> Image:
        src = self._guard_input_range(self._read_input_layer(image))
        adjusted = adjust_gamma(src, gamma=self.gamma, gain=self.gain)
        collapsed = self._project_to_detect_mat(image, adjusted)
        image.detect_mat[:] = self._apply_norm(collapsed).astype(np.float32)
        return image
```

Add to `src/phenotypic/enhance/__init__.py`:

```python
from ._contrast_gamma import ContrastGamma
```

plus `"ContrastGamma"` in `__all__`.

Add `"ContrastGamma"` to the `ContrastAdjustment` roster in `tests/unit/abc_/test_enhancer_taxonomy.py`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/enhance/test_contrast_ops.py tests/unit/abc_/test_enhancer_taxonomy.py -v
uv run pytest --doctest-modules src/phenotypic/enhance/_contrast_gamma.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/enhance/_contrast_gamma.py src/phenotypic/enhance/__init__.py \
        tests/unit/enhance/test_contrast_ops.py tests/unit/abc_/test_enhancer_taxonomy.py
git commit -m "feat(enhance): add ContrastGamma with input_layer support"
```

---

### Task 5.2: `ContrastLog`

**Files:**
- Create: `src/phenotypic/enhance/_contrast_log.py`
- Modify: `src/phenotypic/enhance/__init__.py`, `tests/unit/abc_/test_enhancer_taxonomy.py`
- Test: `tests/unit/enhance/test_contrast_ops.py` (extend)

**Interfaces:**
- Produces: `ContrastLog(gain: float = 1.0, inv: bool = False, norm: NormOut = "clip", input_layer: InputLayer = "detect_mat")`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/enhance/test_contrast_ops.py`:

```python
def test_contrast_log_field_order():
    from phenotypic.enhance import ContrastLog

    assert list(ContrastLog.model_fields) == ["gain", "inv", "norm", "input_layer"]


def test_contrast_log_brightens_dark_regions():
    """log compresses highlights and expands shadows -> mean rises."""
    from phenotypic.enhance import ContrastLog

    image = load_synth_yeast_plate()
    before = image.detect_mat[:].copy()
    after = ContrastLog().apply(image).detect_mat[:]
    assert after.mean() > before.mean()


def test_contrast_log_inv_is_the_inverse_curve():
    from phenotypic.enhance import ContrastLog

    image = load_synth_yeast_plate()
    fwd = ContrastLog(inv=False).apply(load_synth_yeast_plate()).detect_mat[:]
    inv = ContrastLog(inv=True).apply(image).detect_mat[:]
    assert np.abs(fwd - inv).max() > 1e-2


def test_contrast_log_matches_skimage():
    from skimage.exposure import adjust_log

    from phenotypic.enhance import ContrastLog

    image = load_synth_yeast_plate()
    src = image.detect_mat[:].copy()
    expected = np.clip(adjust_log(src, gain=1.0, inv=False), 0.0, 1.0)
    actual = ContrastLog().apply(image).detect_mat[:]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/unit/enhance/test_contrast_ops.py -k contrast_log -v
```
Expected: FAIL — `ImportError: cannot import name 'ContrastLog'`

- [ ] **Step 3: Write `ContrastLog`**

Create `src/phenotypic/enhance/_contrast_log.py`, mirroring `ContrastGamma`'s structure:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.exposure import adjust_log

from ..abc_ import ContrastAdjustment
from ..sdk_.mixin import InputLayerMixin, NormalizedOutputMixin
from ..sdk_.typing_ import TuneSpec


class ContrastLog(InputLayerMixin, NormalizedOutputMixin, ContrastAdjustment):
    """Apply a logarithmic intensity curve to lift faint colonies out of dark agar.

    Computes ``gain * log2(1 + I)``, which expands the dark end of the histogram
    while compressing highlights. Faint colonies sitting just above the agar
    background gain contrast; already-bright colonies compress toward saturation.
    Setting ``inv=True`` applies the inverse exponential curve, which expands the
    bright end instead.

    Best For:
        - Dark-field or transmitted-light plates where colonies are dim and the
          background is near black.
        - Recovering small colonies whose intensity sits within a few percent of
          the agar background.

    Consider Also:
        - :class:`ContrastGamma` for a tunable power-law curve rather than a fixed
          logarithmic shape.
        - :class:`ContrastSigmoid` when contrast should steepen around one
          intensity rather than across the shadows.

    Args:
        gain: Constant multiplier applied after the curve. Default: 1.0.
            Has no effect when ``norm="rescale"``, which divides it back out.
        inv: When ``True``, apply the inverse (exponential) curve, expanding the
            bright end rather than the dark end. Default: ``False``.
        norm: Output range policy. ``"clip"`` (default) saturates values outside
            [0, 1]; ``"rescale"`` remaps the full observed range onto [0, 1];
            ``None`` passes values through untouched.
        input_layer: Source layer. ``"detect_mat"`` (default) applies the curve to
            the 2-D detection matrix. ``"rgb"`` applies it to all three colour
            channels, then collapses the result to 2-D through the image's own
            ``detect_mode``.

    Returns:
        Image: Input image with ``detect_mat`` log-corrected. ``rgb`` and ``gray``
        are unchanged. With ``input_layer="rgb"``, any enhancement a prior operation
        wrote to ``detect_mat`` is discarded.

    Examples:
        Lift dim colonies on a dark plate:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import ContrastLog
        >>> plate = load_synth_yeast_plate()
        >>> enhanced = ContrastLog().apply(plate)
        >>> float(enhanced.detect_mat[:].mean()) > float(plate.detect_mat[:].mean())
        True
    """

    gain: Annotated[float, TuneSpec(0.5, 2.0)] = 1.0
    inv: bool = False

    def _operate(self, image: Image) -> Image:
        src = self._guard_input_range(self._read_input_layer(image))
        adjusted = adjust_log(src, gain=self.gain, inv=self.inv)
        collapsed = self._project_to_detect_mat(image, adjusted)
        image.detect_mat[:] = self._apply_norm(collapsed).astype(np.float32)
        return image
```

> **Doctest caution:** `plate.detect_mat[:]` is read *after* `apply()` returns a copy,
> so `plate` is unmodified. Verify the doctest's `True` empirically before committing —
> if `apply()` mutated in place the comparison would be `False`.

Register in `__init__.py` and the taxonomy roster.

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/unit/enhance/test_contrast_ops.py -k contrast_log -v
uv run pytest --doctest-modules src/phenotypic/enhance/_contrast_log.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/enhance/_contrast_log.py src/phenotypic/enhance/__init__.py \
        tests/unit/enhance/test_contrast_ops.py tests/unit/abc_/test_enhancer_taxonomy.py
git commit -m "feat(enhance): add ContrastLog with input_layer support"
```

---

### Task 5.3: `ContrastSigmoid`

**Files:**
- Create: `src/phenotypic/enhance/_contrast_sigmoid.py`
- Modify: `src/phenotypic/enhance/__init__.py`, `tests/unit/abc_/test_enhancer_taxonomy.py`
- Test: `tests/unit/enhance/test_contrast_ops.py` (extend)

**Interfaces:**
- Produces: `ContrastSigmoid(cutoff: float = 0.5, gain: float = 10.0, inv: bool = False, norm: NormOut = "clip", input_layer: InputLayer = "detect_mat")`

`cutoff=0.5` and `gain=10.0` are skimage's own defaults. **Unlike the other two ops, this
`gain` survives `norm="rescale"`** — it sits inside the exponent
(`scale/(1+exp(gain*(cutoff - I/scale)))`), so it reshapes the curve rather than scaling it.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/enhance/test_contrast_ops.py`:

```python
def test_contrast_sigmoid_field_order():
    from phenotypic.enhance import ContrastSigmoid

    assert list(ContrastSigmoid.model_fields) == [
        "cutoff", "gain", "inv", "norm", "input_layer",
    ]


def test_contrast_sigmoid_defaults_match_skimage():
    from phenotypic.enhance import ContrastSigmoid

    assert ContrastSigmoid().cutoff == 0.5
    assert ContrastSigmoid().gain == 10.0


def test_contrast_sigmoid_gain_survives_rescale():
    """Contrast with ContrastGamma: sigmoid's gain is inside exp(), so it reshapes."""
    from phenotypic.enhance import ContrastSigmoid

    a = ContrastSigmoid(gain=5.0, norm="rescale").apply(load_synth_yeast_plate())
    b = ContrastSigmoid(gain=15.0, norm="rescale").apply(load_synth_yeast_plate())
    assert np.abs(a.detect_mat[:] - b.detect_mat[:]).max() > 1e-2


def test_contrast_sigmoid_increases_contrast_about_cutoff():
    from phenotypic.enhance import ContrastSigmoid

    image = load_synth_yeast_plate()
    before = image.detect_mat[:].copy()
    after = ContrastSigmoid(cutoff=float(before.mean()), gain=10.0).apply(image).detect_mat[:]
    assert after.std() > before.std()


def test_contrast_sigmoid_matches_skimage():
    from skimage.exposure import adjust_sigmoid

    from phenotypic.enhance import ContrastSigmoid

    image = load_synth_yeast_plate()
    src = image.detect_mat[:].copy()
    expected = np.clip(adjust_sigmoid(src, cutoff=0.4, gain=8.0, inv=False), 0.0, 1.0)
    actual = ContrastSigmoid(cutoff=0.4, gain=8.0).apply(image).detect_mat[:]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/unit/enhance/test_contrast_ops.py -k contrast_sigmoid -v
```
Expected: FAIL — `ImportError: cannot import name 'ContrastSigmoid'`

- [ ] **Step 3: Write `ContrastSigmoid`**

Same structure as Task 5.2. Fields:

```python
    cutoff: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5
    gain: Annotated[float, TuneSpec(1.0, 20.0)] = 10.0
    inv: bool = False

    def _operate(self, image: Image) -> Image:
        src = self._guard_input_range(self._read_input_layer(image))
        adjusted = adjust_sigmoid(src, cutoff=self.cutoff, gain=self.gain, inv=self.inv)
        collapsed = self._project_to_detect_mat(image, adjusted)
        image.detect_mat[:] = self._apply_norm(collapsed).astype(np.float32)
        return image
```

Docstring `Args:` entries:

```
        cutoff: Intensity about which the sigmoid is centred, in [0, 1]. Pixels
            below it are pushed toward 0, above it toward 1. Set near the
            agar/colony boundary intensity. Default: 0.5.
        gain: Steepness of the sigmoid. Larger values approach a hard threshold;
            smaller values blend gradually. Typical range: 5--15. Default: 10.0.
            Unlike :class:`ContrastGamma`, this ``gain`` survives ``norm="rescale"``
            because it reshapes the curve rather than scaling its output.
        inv: When ``True``, invert the sigmoid so bright regions are suppressed.
            Default: ``False``.
        norm: Output range policy. ``"clip"`` (default) saturates values outside
            [0, 1]; ``"rescale"`` remaps the full observed range onto [0, 1];
            ``None`` passes values through untouched.
        input_layer: Source layer. ``"detect_mat"`` (default) or ``"rgb"``.
```

Class docstring `Best For:` should mention pushing a soft colony/agar boundary toward a
binary decision before an Otsu or Triangle threshold.

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/unit/enhance/test_contrast_ops.py -v
uv run pytest --doctest-modules src/phenotypic/enhance/_contrast_sigmoid.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/enhance/_contrast_sigmoid.py src/phenotypic/enhance/__init__.py \
        tests/unit/enhance/test_contrast_ops.py tests/unit/abc_/test_enhancer_taxonomy.py
git commit -m "feat(enhance): add ContrastSigmoid with input_layer support"
```

---

### Task 5.4: Retrofit `ContrastStretching` with `input_layer` + `keep_colors`

**Files:**
- Modify: `src/phenotypic/enhance/_contrast_streching.py`
- Test: `tests/unit/enhance/test_contrast_ops.py` (extend)

**Interfaces:**
- Produces: `ContrastStretching(lower_percentile: int = 2, upper_percentile: int = 98, keep_colors: bool = True, input_layer: InputLayer = "detect_mat")`

**No `norm` field.** Percentile rescaling to [0, 1] *is* this operation's algorithm; `norm`
could only no-op (`"clip"`) or undo it (`None`).

`keep_colors` is GIMP's name for the joint/per-channel choice and is **ignored for 2-D
input**. `keep_colors=True` (default) takes one `(p_lo, p_hi)` from the flattened `H×W×3`
array — matching skimage's `rescale_intensity(in_range='image')`, which reduces globally
with `np.min`/`np.max`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/enhance/test_contrast_ops.py`:

```python
def test_stretching_field_order():
    from phenotypic.enhance import ContrastStretching

    assert list(ContrastStretching.model_fields) == [
        "lower_percentile", "upper_percentile", "keep_colors", "input_layer",
    ]


def test_stretching_has_no_norm_field():
    """Percentile rescaling IS the algorithm; a norm field could only no-op or undo it."""
    from phenotypic.enhance import ContrastStretching

    assert "norm" not in ContrastStretching.model_fields


def test_stretching_keep_colors_preserves_channel_balance():
    """Joint percentiles: a red-dominant plate stays red-dominant."""
    import numpy as np

    from phenotypic import Image
    from phenotypic.enhance import ContrastStretching

    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[..., 0] = np.linspace(120, 220, 16, dtype=np.uint8)[None, :]
    rgb[..., 1] = 40
    rgb[..., 2] = 20
    image = Image(rgb)
    image.set_detect_mode("MinRGB")

    joint = ContrastStretching(input_layer="rgb", keep_colors=True).apply(image)
    split = ContrastStretching(input_layer="rgb", keep_colors=False).apply(image)
    assert not np.allclose(joint.detect_mat[:], split.detect_mat[:], atol=1e-4)


def test_stretching_keep_colors_ignored_for_2d_input():
    import numpy as np

    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.enhance import ContrastStretching

    a = ContrastStretching(keep_colors=True).apply(load_synth_yeast_plate()).detect_mat[:]
    b = ContrastStretching(keep_colors=False).apply(load_synth_yeast_plate()).detect_mat[:]
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_stretching_output_always_unit_range():
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.enhance import ContrastStretching

    dm = ContrastStretching().apply(load_synth_yeast_plate()).detect_mat[:]
    assert abs(float(dm.min())) < 1e-6 and abs(float(dm.max()) - 1.0) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/unit/enhance/test_contrast_ops.py -k stretching -v
```
Expected: FAIL — `assert [...] == [..., 'keep_colors', 'input_layer']`

- [ ] **Step 3: Retrofit the class**

```python
class ContrastStretching(InputLayerMixin, ContrastAdjustment):
    ...
    lower_percentile: Annotated[int, TuneSpec(1, 5)] = 2
    upper_percentile: Annotated[int, TuneSpec(95, 99)] = 98
    keep_colors: bool = True

    def _operate(self, image: Image) -> Image:
        src = self._read_input_layer(image)
        if src.ndim == 3 and not self.keep_colors:
            adjusted = np.empty_like(src)
            for channel in range(src.shape[2]):
                p_lower, p_upper = np.percentile(
                    src[..., channel], (self.lower_percentile, self.upper_percentile)
                )
                adjusted[..., channel] = rescale_intensity(
                    image=src[..., channel], in_range=(p_lower, p_upper), out_range=(0, 1)
                )
        else:
            p_lower, p_upper = np.percentile(
                src, (self.lower_percentile, self.upper_percentile)
            )
            adjusted = rescale_intensity(
                image=src, in_range=(p_lower, p_upper), out_range=(0, 1)
            )
        collapsed = self._project_to_detect_mat(image, adjusted)
        image.detect_mat[:] = collapsed.astype(np.float32)
        return image
```

Add the two new `Args:` entries:

```
        keep_colors: When ``input_layer="rgb"``, take a single pair of percentiles
            jointly across all three colour channels and rescale them together,
            preserving channel balance and hue. When ``False``, compute percentiles
            per channel and rescale each independently -- effectively a white
            balance, which removes colour casts but shifts hue. Ignored for 2-D
            input. Default: ``True``.
        input_layer: Source layer. ``"detect_mat"`` (default) or ``"rgb"``.
```

Add to `Returns:`: *"The output always fills [0, 1] by construction, so no ``norm`` field is offered."*

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/unit/enhance/test_contrast_ops.py -v
uv run pytest tests/unit/core/test_image_pipeline.py tests/unit/tune/test_enhance_annotations.py -q
```
Expected: PASS. `ContrastStretching` is used by `detect/_filamentous_fungi_detector.py:415` and `detect/_inoculum_detector.py:189` and by `prefab/_grid_section_pipeline.py:176` — all construct it with defaults, so behaviour is unchanged. Confirm:

```bash
uv run pytest tests/unit/detect -q -k "fungi or inoculum"
```

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/enhance/_contrast_streching.py tests/unit/enhance/test_contrast_ops.py
git commit -m "feat(enhance): add input_layer and keep_colors to ContrastStretching"
```

---

### Task 5.5: Tune annotation coverage

**Files:**
- Modify: `tests/unit/tune/test_enhance_annotations.py`
- Verify: `tests/unit/tune/test_annotation_coverage.py` (no edit expected)

`inv`, `norm`, `input_layer`, and `keep_colors` are `bool` / `Literal` / `Optional[Literal]`,
all excluded by `is_numeric_tunable`. Only `gamma`, `gain`, and `cutoff` enter the gate,
and each carries a `TuneSpec`.

- [ ] **Step 1: Extend the annotation test**

Add to the parametrized list in `tests/unit/tune/test_enhance_annotations.py`. The file
already imports `FloatRange, IntRange` from `phenotypic.tune` (line 37); `FloatRange`
entries carry a **3-tuple** `(low, high, log)` — see `GaussianBlur` at line 59:

```python
            (ContrastGamma(), "gamma", FloatRange, (0.1, 5.0, True)),
            (ContrastGamma(), "gain", FloatRange, (0.5, 2.0, False)),
            (ContrastLog(), "gain", FloatRange, (0.5, 2.0, False)),
            (ContrastSigmoid(), "cutoff", FloatRange, (0.0, 1.0, False)),
            (ContrastSigmoid(), "gain", FloatRange, (1.0, 20.0, False)),
```

Import `ContrastGamma`, `ContrastLog`, `ContrastSigmoid` in the existing
`from phenotypic.enhance import (...)` block at line 20.

- [ ] **Step 2: Assert the closed-set fields stay out of the gate**

```python
def test_closed_set_fields_are_not_numeric_tunable():
    from tests.unit.tune._annotation_introspect import is_numeric_tunable

    from phenotypic.enhance import ContrastGamma, ContrastStretching

    assert not is_numeric_tunable(ContrastGamma.model_fields["norm"])
    assert not is_numeric_tunable(ContrastGamma.model_fields["input_layer"])
    assert not is_numeric_tunable(ContrastStretching.model_fields["keep_colors"])
```

- [ ] **Step 3: Run the gate**

```bash
uv run pytest tests/unit/tune/test_enhance_annotations.py tests/unit/tune/test_annotation_coverage.py -v
```
Expected: PASS, with **no new entries** required in `tests/fixtures/tune/annotation_allowlist.json`. If the coverage gate demands an allowlist entry, a `TuneSpec` is missing — add it rather than allowlisting.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/tune/
git commit -m "test(tune): cover Contrast* annotations in the coverage gate"
```

---

# Phase 6 — Conventions, docs, and full verification

Depends on Phases 2 and 5.

---

### Task 6.1: Record the conventions

**Files:**
- Modify: `.claude/skills/adding-an-operation/SKILL.md`
- Modify: `src/phenotypic/enhance/CLAUDE.md`
- Modify: `src/phenotypic/sdk_/CLAUDE.md`

Now — and only now — `NormOut`, `NormalizedOutputMixin`, and `InputLayerMixin` exist and
are importable, so documenting them cannot send a reader after a symbol that isn't there.

- [ ] **Step 1: Add the convention to the skill**

Append to `.claude/skills/adding-an-operation/SKILL.md`, after the "Closed value sets" section:

```markdown
## Output-range guards use `norm`, never `clip: bool`

Any operation that clamps or normalizes its output declares `norm: NormOut`
(from `sdk_.typing_`) by inheriting `NormalizedOutputMixin`, and applies it via
`self._apply_norm(arr)`:

- `"clip"` (default) saturates — the identity in-range, so absolute intensity
  and cross-batch comparability survive.
- `"rescale"` remaps the full observed range onto [0, 1]. Note this **divides out**
  any purely multiplicative `gain` parameter, making such a knob a no-op.
- `None` passes through — the escape hatch GAT regions and `CompositeEnhance`
  depend on, and what `NormControlMixin._disable_normalization` sets.

A bare `clip: bool` cannot express `"rescale"`, and the attribute name is claimed
by `NormControlMixin`, which duck-types on it. Removed in 0.18.0.

Skimage passthrough parameters that merely *sound* like range guards
(`rescale_sigma`, forwarded to `denoise_wavelet`) are **not** `norm` and stay as
they are.

Inside a GAT region, declare the inert value in `_GAT_DEFER_VALUES`:
`{"norm": None, "rescale_sigma": False}`.

## Cross-cutting fields append; they do not frontload

Pydantic collects fields in reverse-MRO order, so a mixin's field lands *before*
the operation's own parameters — wrong for both `model_json_schema()` and
`to_json()`. A field-append mixin fixes this in `__pydantic_init_subclass__`:

```python
@classmethod
def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
    super().__pydantic_init_subclass__(**kwargs)   # cooperative: BaseOperation's runs
    fields = cls.__pydantic_fields__
    if "norm" in fields and list(fields)[-1] != "norm":
        fields["norm"] = fields.pop("norm")
        cls.model_rebuild(force=True)
```

`TuneSpec` metadata survives the forced rebuild. With two such mixins the order is
deterministic — each calls `super()` first, so the mixin **earliest in the MRO ends
up last**. `class Op(InputLayerMixin, NormalizedOutputMixin, ContrastAdjustment)`
yields `[…op params…, norm, input_layer]`.

Canonical: `sdk_/mixin/_normalized_output_mixin.py`, `sdk_/mixin/_input_layer_mixin.py`.
```

- [ ] **Step 2: Update `enhance/CLAUDE.md`**

Add to the Implementation Conventions list:

```markdown
- `ContrastGamma`, `ContrastLog`, `ContrastSigmoid`, and `ContrastStretching` inherit
  `InputLayerMixin`, so they can read the pristine `rgb` layer instead of `detect_mat`.
  The 3-D result is collapsed back to 2-D through the image's own `detect_mode` — they
  still write `detect_mat` and nothing else. `ContrastStretching` alone has no `norm`
  field: percentile rescaling to [0, 1] is its algorithm.
- `FocusEdgeLaplace` defaults to `norm="rescale"`, not `"clip"`: a Laplacian is signed,
  and clipping would discard the entire negative lobe.
```

- [ ] **Step 3: Update `sdk_/CLAUDE.md`**

Replace the `### ClipControlMixin` section with `### NormControlMixin`, and add
`### NormalizedOutputMixin` and `### InputLayerMixin` entries with one-line summaries.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/adding-an-operation/SKILL.md src/phenotypic/enhance/CLAUDE.md \
        src/phenotypic/sdk_/CLAUDE.md
git commit -m "docs: record norm and field-append mixin conventions"
```

---

### Task 6.2: Full verification sweep

- [ ] **Step 1: Confirm no `clip` survives**

```bash
grep -rnE "clip: bool|self\.clip\b|[^_]clip=|_GAT_DEFER_ATTRS|ClipControlMixin" \
     src/phenotypic/ tests/ --exclude-dir=__pycache__ --exclude-dir=_assets
```
Expected: **zero hits.**

Word boundaries matter here. `EnhanceLocalContrast` has a legitimate `clip_limit` field,
so a bare `self\.clip` matches `self.clip_limit` and a bare `clip=` matches `clip_limit=`.
`\b` and `[^_]` exclude both. Plain `np.clip(...)` calls are also fine and unmatched.

- [ ] **Step 2: Lint and typecheck**

```bash
uv run ruff check --fix
uv run mypy src/phenotypic
```
Expected: clean

- [ ] **Step 3: Full test suite**

```bash
uv run pytest tests/unit -q
```
Expected: PASS. Record the pass count; compare against the pre-branch baseline
(`git stash && uv run pytest tests/unit -q`) so no test silently vanished.

- [ ] **Step 4: Doctests for every new/changed module**

```bash
uv run pytest --doctest-modules \
  src/phenotypic/enhance/_contrast_gamma.py \
  src/phenotypic/enhance/_contrast_log.py \
  src/phenotypic/enhance/_contrast_sigmoid.py \
  src/phenotypic/enhance/_contrast_streching.py \
  src/phenotypic/enhance/_focus_edge_laplace.py \
  src/phenotypic/sdk_/mixin/_norm_control_mixin.py -v
```
Expected: PASS

- [ ] **Step 5: Confirm the GUI picks up the new ops automatically**

```bash
uv run python -c "
import phenotypic.enhance as e
for n in ('ContrastGamma','ContrastLog','ContrastSigmoid'):
    assert n in e.__all__, n
print('all three exported; the GUI builder registry walks phenotypic.enhance')
"
```

- [ ] **Step 6: Commit any final fixes**

```bash
git add -A && git commit -m "chore: final lint and verification pass for 0.18.0"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task:

| Spec § | Task |
|---|---|
| §3 `InputLayerMixin` | 1.3 |
| §4.1–4.2 `rgb_to_xyz` + `compute_from_rgb` | 3.1, 3.2 |
| §4.3 golden test | 3.3 |
| §5 `norm` field | 1.1, 1.2 |
| §5.1 `NormalizedOutputMixin` + two-mixin order | 1.2, 1.3 |
| §5.2 migration, 8 classes, hard break | 2.3, 2.4 |
| §5.3 `_GAT_DEFER_VALUES`, `NormControlMixin` | 2.1, 2.2 |
| §5.4 defensive input normalization | 1.3 (`_guard_input_range`) |
| §6 the three ops | 5.1, 5.2, 5.3 |
| §6.1 `ContrastStretching` retrofit | 5.4 |
| §7 `FocusEdgeLaplace` + invariant gate | 4.1 |
| §8 discard semantics documented | 5.1–5.4 docstrings |
| §9 file list | all |
| §11 release `0.18.0` | 2.4 |
| §12 conventions → skill | 6.1 |
| Tune coverage gate | 5.5 |

**Type consistency.** `_apply_norm`, `_read_input_layer`, `_project_to_detect_mat`,
`_guard_input_range`, `_disable_normalization`, `compute_from_rgb`, `rgb_to_xyz`,
`_GAT_DEFER_VALUES`, `NormOut`, `InputLayer` are each defined once and used with
identical signatures downstream.

**Known ordering hazards, handled explicitly:**

1. Task 1.3's `test_project_collapses_3d_via_detect_mode` needs Phase 3. It is marked
   `xfail(strict=True)` and the marker is removed in Task 3.3 — a strict xfail turns into a
   failure the moment the feature lands, so it cannot rot into a silent skip.
2. Task 2.1's `test_disable_normalization_sets_norm_none` needs Task 2.3's migration.
   Re-run Task 2.1's suite after 2.3.

**Facts verified against the tree while writing this plan** (do not re-derive):

- `available_modes()` returns exactly 11 names: `HsvS, HsvV, InvS, LabA, LabB, LabL,
  MinRGB, blue, gray, green, red`. Both it and `get_detection_mode` are exported from
  `detection_modes/__init__.py`.
- `tests/unit/tune/test_enhance_annotations.py:37` already imports `FloatRange, IntRange`.
  `FloatRange` expectations are 3-tuples `(low, high, log)`.
- The correction classes are exported as `ColorDenoise`, `VisuShrinkCorrector`,
  `BayesShrinkCorrector` (used in Task 2.3's test imports).
- The XYZ accessor class is `XyzAccessor` — lowercase `yz`, not `XYZAccessor`.
- Two stacked field-append mixins yield `['gamma', 'gain', 'norm', 'input_layer']`;
  `TuneSpec` metadata survives both `model_rebuild(force=True)` calls; `norm=None`
  round-trips through `to_json`; `setattr(op, "norm", None)` works under
  `validate_assignment=True` (the GAT-defer path depends on this).

---

## Execution Handoff

Plan complete and saved to
`docs/superpowers/plans/2026-07-09-contrast-enhancers-input-layer/plan.md`.
