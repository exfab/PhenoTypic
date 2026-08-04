# GpuDetector Batched Interface Refactor — Implementation Plan (Spec 1, Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:
> subagent-driven-development (recommended) or superpowers:executing-plans to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `GpuDetector` a full batched/streaming interface (`input_layer`,
`supports_batching`, `output_kind`, `preprocess`/`collate`/`infer_batch`) with a correct
single-image default, and refactor the existing `Sam2`/`MicroSamDetector` onto it — with
**no behavior change** for notebook users.

**Architecture:** `GpuDetector` becomes a concrete-`_operate` ABC: it reads the declared
`input_layer`, runs `preprocess → collate → infer_batch`, and writes the result via
`output_kind` (`objmap` for instance, `objmask` for semantic). The default `infer_batch`
loops a per-detector `_infer_one`, so non-batchable models (SAM2/micro-sam) need only
`_infer_one` + `_ensure_model_loaded`. Spec 2's batchable models will later override
`infer_batch` with a true `(N,C,H,W)` forward — no engine changes.

**Tech Stack:** Python, pydantic v2 (operations are `BaseModel`s with class-annotated
fields), numpy, scikit-image (`skimage.measure.label`), pytest. `uv` is the sole runner.

**Source of truth:**
`docs/superpowers/specs/gpu-detection-optimization/2026-06-16-staged-batched-gpu-detection-design.md`
§3–§4, §8 (decisions D4, D8, D9).

**Plan set:** Plan 1 (this) = interface refactor. Plan 2 = CLI splitter + local staged
engine + sidecar. Plan 3 = SLURM per-stage chaining + licensing scaffolding.

---

## File Structure

| File                                             | Responsibility                                                                                                                                         | Action |
|--------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| `src/phenotypic/tools_/typing_.py`               | `GpuInputLayer`, `GpuOutputKind` Literal aliases                                                                                                       | Modify |
| `src/phenotypic/abc_/_gpu_detector.py`           | Interface fields + `preprocess`/`collate`/`infer_batch`/`_write_object_output`/concrete `_operate`; abstract `_ensure_model_loaded`; `_infer_one` hook | Modify |
| `src/phenotypic/detect/nn/_sam2_detector.py`     | Set capability fields; move core into `_infer_one`; drop bespoke `_operate`                                                                            | Modify |
| `src/phenotypic/detect/nn/_microsam_detector.py` | Same as SAM2                                                                                                                                           | Modify |
| `tests/unit/abc_/test_gpu_detector_interface.py` | Interface, channel-stacking, default-loop, both routes (via a CPU `_FakeGpuDetector`)                                                                  | Create |
| `tests/unit/detect/nn/test_sam2_detector.py`     | Add capability-field assertions                                                                                                                        | Modify |
| `tests/unit/detect/nn/test_microsam_detector.py` | Add capability-field assertions                                                                                                                        | Modify |

**Convention note for the engineer:** Operations are pydantic v2 models — parameters are
**class-level annotated fields** (e.g. `input_layer: GpuInputLayer = "rgb"`), there is
no `__init__`, construction is keyword-only. Private runtime state uses `PrivateAttr`
and never serializes. Tests construct detectors **without** torch installed; only
`.apply()` functional tests require the `phenotypic[torch]` extra and are skipped
otherwise.

---

### Task 1: Typing aliases for the interface

**Files:**

- Modify: `src/phenotypic/tools_/typing_.py`
- Test: `tests/unit/abc_/test_gpu_detector_interface.py` (Create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/abc_/test_gpu_detector_interface.py`:

```python
"""Tests for the GpuDetector batched/streaming interface (Spec 1, Plan 1).

All tests construct detectors WITHOUT torch — the interface and the CPU
``_FakeGpuDetector`` exercise the engine contract with no GPU dependency.
"""

from typing import get_args

from phenotypic.tools_.typing_ import GpuInputLayer, GpuOutputKind


class TestTypingAliases:
    def test_input_layer_values(self):
        assert set(get_args(GpuInputLayer)) == {"rgb", "gray", "detect_mat"}

    def test_output_kind_values(self):
        assert set(get_args(GpuOutputKind)) == {"instance", "semantic"}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/unit/abc_/test_gpu_detector_interface.py::TestTypingAliases -v`
Expected: FAIL — `ImportError: cannot import name 'GpuInputLayer'`.

- [ ] **Step 3: Add the aliases**

In `src/phenotypic/tools_/typing_.py`, next to the existing `ProcessOnlyLayer` /
`DetectMode` Literal aliases, add:

```python
#: Image layer a GpuDetector consumes as model input. Single-channel layers
#: (gray/detect_mat) are stacked to (H, W, 3) by GpuDetector.preprocess.
GpuInputLayer = Literal["rgb", "gray", "detect_mat"]

#: Object output a GpuDetector produces. "instance" -> labeled objmap;
#: "semantic" -> binary objmask (auto-labels into objmap, like a threshold detector).
GpuOutputKind = Literal["instance", "semantic"]
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest tests/unit/abc_/test_gpu_detector_interface.py::TestTypingAliases -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/tools_/typing_.py tests/unit/abc_/test_gpu_detector_interface.py
git commit -m "feat(gpu): add GpuInputLayer/GpuOutputKind typing aliases"
```

---

### Task 2: GpuDetector capability fields + abstract `_ensure_model_loaded`

**Files:**

- Modify: `src/phenotypic/abc_/_gpu_detector.py`
- Test: `tests/unit/abc_/test_gpu_detector_interface.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/abc_/test_gpu_detector_interface.py`:

```python
from phenotypic.abc_ import GpuDetector
from phenotypic.detect.nn import Sam2


class TestCapabilityFields:
    def test_defaults_on_existing_detector(self):
        det = Sam2()
        assert det.input_layer == "rgb"
        assert det.supports_batching is False
        assert det.output_kind == "instance"

    def test_fields_are_serializable_pydantic_fields(self):
        # capability markers are real fields (not ClassVar) -> in model_fields
        assert "input_layer" in GpuDetector.model_fields
        assert "supports_batching" in GpuDetector.model_fields
        assert "output_kind" in GpuDetector.model_fields
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/unit/abc_/test_gpu_detector_interface.py::TestCapabilityFields -v`
Expected: FAIL — `AttributeError: 'Sam2' object has no attribute 'input_layer'`.

- [ ] **Step 3: Add fields + abstract `_ensure_model_loaded`**

In `src/phenotypic/abc_/_gpu_detector.py`, add the imports and replace the class body's
field/abstract section. The new top of the class:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List

import numpy as np

from phenotypic.tools_.typing_ import GpuInputLayer, GpuOutputKind

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from ._object_detector import ObjectDetector


class GpuDetector(ObjectDetector, ABC):
    """Marker + interface ABC for GPU-accelerated object detectors.

    (keep the existing long docstring body here unchanged)
    """

    # Capability / routing markers — pydantic FIELDS (not ClassVar) so they
    # serialize and round-trip (Spec 1 §4, review S4). Subclasses override the
    # defaults; "instance" keeps existing SAM behavior unchanged.
    input_layer: GpuInputLayer = "rgb"
    supports_batching: bool = False
    output_kind: GpuOutputKind = "instance"

    @abstractmethod
    def _ensure_model_loaded(self) -> None:
        """Build/load the GPU model on first use (idempotent)."""
```

(Leave the existing class docstring text intact; only add the fields +
`_ensure_model_loaded` abstract method. The existing abstract `_operate` stays for now —
it is replaced in Task 5.)

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest tests/unit/abc_/test_gpu_detector_interface.py::TestCapabilityFields tests/unit/detect/nn -v`
Expected: PASS (existing SAM2/micro-sam construction + serialization tests still pass —
they already implement `_ensure_model_loaded`).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/abc_/_gpu_detector.py tests/unit/abc_/test_gpu_detector_interface.py
git commit -m "feat(gpu): add GpuDetector capability fields + abstract _ensure_model_loaded"
```

---

### Task 3: `GpuDetector.preprocess` channel-stacking

**Files:**

- Modify: `src/phenotypic/abc_/_gpu_detector.py`
- Test: `tests/unit/abc_/test_gpu_detector_interface.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
import numpy as np


class TestPreprocess:
    def test_2d_layer_stacked_to_3_channels(self):
        det = Sam2()
        gray = np.zeros((4, 5), dtype=np.float32)
        out = det.preprocess(gray)
        assert out.shape == (4, 5, 3)

    def test_rgb_passthrough(self):
        det = Sam2()
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        out = det.preprocess(rgb)
        assert out.shape == (4, 5, 3)
        assert out is rgb  # no copy for already-3-channel input
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/abc_/test_gpu_detector_interface.py::TestPreprocess -v`
Expected: FAIL — `AttributeError: ... has no attribute 'preprocess'`.

- [ ] **Step 3: Implement `preprocess`**

Add to `GpuDetector`:

```python
    def preprocess(self, array: np.ndarray) -> Any:
        """Turn a raw ``input_layer`` array into a model-ready sample (CPU).

        Default: a single-channel 2D layer (``gray``/``detect_mat``) is stacked
        into an ``(H, W, 3)`` block so 3-channel models (SAM/DINO ViT) consume
        it unchanged; ``rgb`` passes through untouched. Subclasses may override
        for model-specific normalization (e.g. uint8 coercion).
        """
        if array.ndim == 2:
            return np.stack([array, array, array], axis=-1)
        return array
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/abc_/test_gpu_detector_interface.py::TestPreprocess -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/abc_/_gpu_detector.py tests/unit/abc_/test_gpu_detector_interface.py
git commit -m "feat(gpu): add GpuDetector.preprocess channel-stacking default"
```

---

### Task 4: `collate` + `infer_batch` default loop + `_infer_one` hook

**Files:**

- Modify: `src/phenotypic/abc_/_gpu_detector.py`
- Test: `tests/unit/abc_/test_gpu_detector_interface.py`

- [ ] **Step 1: Write the failing test**

Append the CPU fake detector (reused by later tasks) and its tests:

```python
from pydantic import PrivateAttr
from skimage.measure import label as _sk_label

from phenotypic.abc_ import GpuDetector


class _FakeGpuDetector(GpuDetector):
    """CPU-only GpuDetector for interface tests (no torch).

    Thresholds the (stacked) input and either labels it (instance) or returns
    the binary mask (semantic). ``supports_batching``/``output_kind``/
    ``input_layer`` are overrideable per test.
    """

    threshold: float = 0.5
    _loaded: bool = PrivateAttr(default=False)

    def _ensure_model_loaded(self) -> None:
        self._loaded = True

    def _infer_one(self, sample):
        gray = sample.mean(axis=-1) if sample.ndim == 3 else sample
        peak = gray.max()
        mask = gray > (self.threshold * peak) if peak > 0 else gray > 0
        if self.output_kind == "instance":
            return _sk_label(mask).astype(np.uint16)
        return mask


class TestInferBatchDefault:
    def test_collate_passthrough(self):
        det = _FakeGpuDetector()
        samples = [np.zeros((2, 2, 3)), np.ones((2, 2, 3))]
        assert det.collate(samples) == samples

    def test_infer_batch_loops_infer_one(self):
        det = _FakeGpuDetector(output_kind="instance")
        a = np.zeros((3, 3, 3), dtype=np.float32)
        a[1, 1, :] = 1.0
        results = det.infer_batch([a, a])
        assert len(results) == 2
        assert results[0].dtype == np.uint16
        assert results[0].max() == 1  # one labeled blob

    def test_infer_batch_loads_model(self):
        det = _FakeGpuDetector()
        det.infer_batch([np.zeros((2, 2, 3))])
        assert det._loaded is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/unit/abc_/test_gpu_detector_interface.py::TestInferBatchDefault -v`
Expected: FAIL — `_FakeGpuDetector` can't instantiate (abstract `_operate` not
implemented) **or** `AttributeError: ... 'collate'`. (Either failure is fine; Task 5
makes `_operate` concrete so the fake can construct.)

- [ ] **Step 3: Implement `collate`, `infer_batch`, and the `_infer_one` hook**

Add to `GpuDetector`:

```python
    def collate(self, samples: List[Any]) -> Any:
        """Merge per-sample ``preprocess`` outputs into a batch.

        Default returns the list unchanged (consumed by the looped
        ``infer_batch``). Batchable subclasses override to stack into a tensor.
        """
        return samples

    def infer_batch(self, batch: Any) -> List[np.ndarray]:
        """Run inference over a collated batch; return one result per sample.

        Each result is a uint16 labeled map (``output_kind="instance"``) or a
        boolean mask (``output_kind="semantic"``). The default loops
        ``_infer_one`` (correct for ``supports_batching=False``); batchable
        subclasses override with a true ``(N, C, H, W)`` forward.
        """
        self._ensure_model_loaded()
        return [self._infer_one(sample) for sample in batch]

    def _infer_one(self, sample: Any) -> np.ndarray:
        """Run the model on ONE preprocessed sample. Subclasses must implement.

        Returns a uint16 labeled objmap (instance) or a boolean mask (semantic).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _infer_one()"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
`uv run pytest tests/unit/abc_/test_gpu_detector_interface.py::TestInferBatchDefault -v`
Expected: PASS (after Task 5 makes `_operate` concrete the fake constructs; if this task
is run before Task 5, temporarily the fake still can't construct — run Step 4 only after
Task 5 if executing strictly in order). To keep this task self-contained, **proceed to
Task 5 and run both test classes together** in Task 5 Step 4.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/abc_/_gpu_detector.py tests/unit/abc_/test_gpu_detector_interface.py
git commit -m "feat(gpu): add GpuDetector.collate + infer_batch default loop + _infer_one hook"
```

---

### Task 5: Concrete `_operate` + `_write_object_output` (both routes)

**Files:**

- Modify: `src/phenotypic/abc_/_gpu_detector.py`
- Test: `tests/unit/abc_/test_gpu_detector_interface.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from phenotypic.data import load_synth_yeast_plate


class TestOperateRoutes:
    def test_instance_route_writes_objmap(self):
        image = load_synth_yeast_plate()
        det = _FakeGpuDetector(output_kind="instance", threshold=0.3)
        out = det.apply(image, inplace=False)
        assert out.objmap[:].max() >= 1
        # objmask is the derived view of objmap
        np.testing.assert_array_equal(out.objmap[:] > 0, out.objmask[:])

    def test_semantic_route_writes_objmask(self):
        image = load_synth_yeast_plate()
        det = _FakeGpuDetector(output_kind="semantic", threshold=0.3)
        out = det.apply(image, inplace=False)
        assert out.objmask[:].any()

    def test_input_layer_detect_mat_is_read_and_stacked(self):
        image = load_synth_yeast_plate()
        det = _FakeGpuDetector(input_layer="detect_mat", output_kind="instance",
                               threshold=0.3)
        # detect_mat is 2D -> preprocess stacks to (H,W,3); must not raise
        out = det.apply(image, inplace=False)
        assert out.objmap[:].shape == image.shape[:2]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/unit/abc_/test_gpu_detector_interface.py::TestOperateRoutes -v`
Expected: FAIL — the base `_operate` is still the abstract stub, so `_FakeGpuDetector`
can't instantiate.

- [ ] **Step 3: Replace abstract `_operate` with concrete + add `_write_object_output`**

In `src/phenotypic/abc_/_gpu_detector.py`, **remove** the old
`@abstractmethod def _operate(...)` and add:

```python
    def _write_object_output(self, image: "Image", result: np.ndarray) -> None:
        """Write one ``infer_batch`` result onto the image per ``output_kind``.

        - ``instance`` -> ``image.objmap[:]`` (detector-controlled labels).
        - ``semantic`` -> ``image.objmask[:]`` (auto-labels into the shared
          ``objmap`` backend, exactly like a threshold detector; see Spec 1 §8).
        """
        if self.output_kind == "instance":
            image.objmap[:] = result.astype(np.uint16)
        else:  # semantic
            image.objmask[:] = result.astype(bool)

    def _operate(self, image: "Image") -> "Image":
        """Run GPU detection on one image (notebook / single-image path).

        Reads the declared ``input_layer``, preprocesses, runs a one-element
        batch through ``collate`` + ``infer_batch``, and writes the result via
        ``output_kind``. The batched CLI engine drives the same
        ``preprocess``/``collate``/``infer_batch`` methods over many images.
        """
        self._ensure_model_loaded()
        array = getattr(image, self.input_layer)[:]
        sample = self.preprocess(array)
        batch = self.collate([sample])
        results = self.infer_batch(batch)
        self._write_object_output(image, results[0])
        return image
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/abc_/test_gpu_detector_interface.py -v`
Expected: PASS (all interface test classes, including Task 4's `TestInferBatchDefault`).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/abc_/_gpu_detector.py tests/unit/abc_/test_gpu_detector_interface.py
git commit -m "feat(gpu): concrete GpuDetector._operate + per-output_kind write-back"
```

---

### Task 6: Refactor `Sam2` onto the interface

**Files:**

- Modify: `src/phenotypic/detect/nn/_sam2_detector.py`
- Test: `tests/unit/detect/nn/test_sam2_detector.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/detect/nn/test_sam2_detector.py` inside
`TestSam2DetectorConstruction`:

```python
    def test_capability_fields(self):
    det = Sam2()
    assert det.input_layer == "rgb"
    assert det.output_kind == "instance"
    assert det.supports_batching is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest "tests/unit/detect/nn/test_sam2_detector.py::TestSam2DetectorConstruction::test_capability_fields" -v`
Expected: PASS already (fields inherited from Task 2) — this guards the refactor. If it
errors on import, fix imports first. Proceed to refactor the body.

- [ ] **Step 3: Move the inference core into `_infer_one`; delete the bespoke `_operate`
  **

In `src/phenotypic/detect/nn/_sam2_detector.py`: **delete** the existing
`def _operate(self, image)` method and the trailing
`Sam2.apply.__doc__ = Sam2._operate.__doc__` line. Add `_infer_one` (the same algorithm,
now operating on the preprocessed sample):

```python
    def _infer_one(self, sample):
        """Segment colonies in one preprocessed RGB sample via SAM2 AMG.

        Returns a uint16 labeled objmap (largest-first painting preserves
        small-colony identity at overlaps).
        """
        import numpy as np

        rgb = sample
        if rgb.dtype != np.uint8:
            max_val = rgb.max()
            if max_val > 0:
                rgb = (rgb / max_val * 255).astype(np.uint8)
            else:
                rgb = np.zeros(rgb.shape, dtype=np.uint8)

        masks = self._generator.generate(rgb)  # type: ignore[attr-defined]

        h, w = rgb.shape[:2]
        objmap = np.zeros((h, w), dtype=np.uint16)
        if masks:
            max_labels = int(np.iinfo(np.uint16).max)
            if len(masks) > max_labels:
                import warnings

                warnings.warn(
                    f"SAM2 generated {len(masks)} masks, exceeding uint16 "
                    f"range. Only the first {max_labels} will be labeled.",
                    UserWarning,
                    stacklevel=2,
                )
                masks = masks[:max_labels]
            masks = sorted(masks, key=lambda m: m["area"], reverse=True)
            for idx, m in enumerate(masks, start=1):
                objmap[m["segmentation"]] = idx
        return objmap
```

Then re-point the autodoc line to the class docstring (replace the deleted `_operate`
-based line at the bottom of the file):

```python
# Expose the class docstring on .apply() for Sphinx autodoc
Sam2.apply.__doc__ = Sam2.__doc__
```

Capability fields are inherited (`input_layer="rgb"`, `supports_batching=False`,
`output_kind="instance"`) — no need to redeclare them.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/detect/nn/test_sam2_detector.py -v`
Expected: PASS for construction/hierarchy/serialization. Functional tests (
`TestSam2DetectorFunctional`) **skip** unless `phenotypic[torch]` + a cached SAM2 tiny
checkpoint are present — that skip is correct.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/detect/nn/_sam2.py tests/unit/detect/nn/test_sam2_detector.py
git commit -m "refactor(gpu): move Sam2 onto the batched interface (_infer_one)"
```

---

### Task 7: Refactor `MicroSamDetector` onto the interface

**Files:**

- Modify: `src/phenotypic/detect/nn/_microsam_detector.py`
- Test: `tests/unit/detect/nn/test_microsam_detector.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/detect/nn/test_microsam_detector.py` (inside the construction test
class — match the file's existing class name):

```python
    def test_capability_fields(self):
        det = MicroSamDetector()
        assert det.input_layer == "rgb"
        assert det.output_kind == "instance"
        assert det.supports_batching is False
```

- [ ] **Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/unit/detect/nn/test_microsam_detector.py -k capability_fields -v`
Expected: PASS already if the construction class imports cleanly (fields inherited).
Proceed to refactor the body.

- [ ] **Step 3: Move the inference core into `_infer_one`; delete the bespoke `_operate`
  **

In `src/phenotypic/detect/nn/_microsam_detector.py`: **delete** the existing
`def _operate(self, image)` and the trailing
`MicroSamDetector.apply.__doc__ = MicroSamDetector._operate.__doc__` line. Add:

```python
    def _infer_one(self, sample):
        """Segment colonies in one preprocessed RGB sample via micro-sam AIS.

        Returns a uint16 labeled objmap.
        """
        import numpy as np
        from micro_sam.automatic_segmentation import (
            automatic_instance_segmentation,
        )

        rgb = sample
        if rgb.dtype != np.uint8:
            max_val = rgb.max()
            if max_val > 0:
                rgb = (rgb / max_val * 255).astype(np.uint8)
            else:
                rgb = np.zeros(rgb.shape, dtype=np.uint8)

        labeled = automatic_instance_segmentation(
            predictor=self._predictor,
            segmenter=self._segmenter,
            input_path=rgb,
            ndim=2,
            verbose=False,
        )
        return labeled.astype(np.uint16)
```

Re-point autodoc:

```python
# Propagate the class docstring to the public apply method
MicroSamDetector.apply.__doc__ = MicroSamDetector.__doc__
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/detect/nn/test_microsam_detector.py -v`
Expected: PASS for construction/serialization (functional tests skip without
`micro_sam`).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/detect/nn/_microsam_detector.py tests/unit/detect/nn/test_microsam_detector.py
git commit -m "refactor(gpu): move MicroSamDetector onto the batched interface (_infer_one)"
```

---

### Task 8: Full regression (interface + detectors + migration + lint/types)

**Files:** none (verification + any golden updates surfaced)

- [ ] **Step 1: Run the interface + detector + migration suites**

Run:

```bash
uv run pytest tests/unit/abc_/test_gpu_detector_interface.py tests/unit/detect/nn tests/migration -v
```

Expected: PASS / SKIP only (no failures). The SAM2/micro-sam migration goldens are
`structural_only`, so the new capability fields do not break them. **If** a migration
test fails on a changed serialized shape, open the reported golden under
`tests/migration/_goldens/` and reconcile it with the new params, then re-run; commit
the updated golden with message
`test(gpu): update migration goldens for GpuDetector capability fields`.

- [ ] **Step 2: Type-check the changed modules**

Run:
`uv run mypy src/phenotypic/abc_/_gpu_detector.py src/phenotypic/detect/nn/_sam2_detector.py src/phenotypic/detect/nn/_microsam_detector.py src/phenotypic/tools_/typing_.py`
Expected: no new errors. (`sam2.*` / `micro_sam.*` are already in the mypy
`ignore_missing_imports` list.)

- [ ] **Step 3: Lint/format**

Run:
`uv run ruff check --fix src/phenotypic/abc_/_gpu_detector.py src/phenotypic/detect/nn tests/unit/abc_/test_gpu_detector_interface.py`
Expected: clean (auto-fixes applied).

- [ ] **Step 4: Run the broader detector + pipeline smoke to catch regressions**

Run: `uv run pytest tests/unit/detect tests/smoke -q -m "not slow"`
Expected: PASS / SKIP. No behavior change for non-GPU detectors or notebook usage.

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "test(gpu): green regression for GpuDetector interface refactor" --allow-empty
```

---

## Self-Review (run before handing off)

**Spec coverage (Spec 1 §4, D4/D8/D9):**

- `input_layer` / `supports_batching` / `output_kind` as pydantic fields → Task 2. ✓
- `preprocess` channel-stacking for 2D layers (D9) → Task 3. ✓
- `collate` + default-loop `infer_batch` + `_infer_one` hook → Task 4. ✓
- Concrete `_operate` + instance/semantic write-back (D8, §8) → Task 5. ✓
- SAM2/micro-sam refactored, no behavior change → Tasks 6–7. ✓
- `_operate`↔`infer_batch` shared core (notebook + batched paths) → Tasks 4–5. ✓

**Type consistency:** `_infer_one(sample) -> np.ndarray`,
`infer_batch(batch) -> List[np.ndarray]`, `preprocess(array) -> Any`,
`collate(List) -> Any`, `_write_object_output(image, result)`,
`_operate(image) -> Image` — names/signatures consistent across Tasks 4–7. ✓

**Out of scope (later plans):** the CLI splitter, the staged engine, the sidecar,
resume, SLURM, and licensing scaffolding are **Plan 2 / Plan 3** — not implemented here.
This plan ships a self-contained, notebook-testable interface refactor.

---

## Execution Handoff

Plan complete. Next: choose how to execute, or have me draft **Plan 2** (CLI splitter +
local staged engine + sidecar) first.
