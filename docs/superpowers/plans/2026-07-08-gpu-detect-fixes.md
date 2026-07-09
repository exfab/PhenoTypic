# GPU Detector Resolution & Tiling Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the DINO-backed GPU detectors from silently downsampling every tile to 224×224, and replace SAM3's fragment-producing tile merge with a centroid-in-core assignment that cannot duplicate or delete colonies.

**Architecture:** Two shared modules absorb the policy so detectors get thinner, not fatter. `_dino_support.py` becomes the single resize policy (`NATIVE_PROCESSOR_KWARGS` + a grid↔image geometry pair that always uses the *covered* extent `hp*patch × wp*patch`). `_tiling.py` becomes the single tiling policy (owns the instance merge, which currently lives in `_sam3_detector.py`). Detectors then only choose `tile_px` and call into these.

**Tech Stack:** Python 3.12, pydantic v2 operations, `transformers` (DINOv2/DINOv3), `sam2`, numpy, scikit-image, pytest. Runner is `uv`.

## Global Constraints

- **`uv` is the sole runner.** Never bare `python`/`pip`. Test env: `uv sync --group dev --extra foundation`.
- **Operations are pydantic v2 models.** Keyword-only construction, class-level annotated fields, no `__init__`. Normalize in a `field_validator`.
- **Every new numeric field needs `Annotated[T, TuneSpec(lo, hi)]` or `TuneSpec(tunable=False)`**, or the coverage gate against `tests/fixtures/tune/annotation_allowlist.json` fails.
- **Resolution invariant:** under 1:1, native px per patch `== patch_size`, independent of `tile_px`. `tile_px` is a compute/context knob only, and **smaller is cheaper at equal fidelity**.
- **DINOv2 is patch 14 / 0 registers; DINOv3 is patch 16 / 4 registers.** Never hardcode 14 or 16 — read `model.config.patch_size`.
- **`config.image_size` is not a native-resolution signal** (DINOv2 → 518, DINOv3 → 224). Do not use it to pick defaults.
- **Semantic tiling (`stitch_semantic_tiles`) is already correct.** Do not apply instance-merge or edge logic to it.
- Google-style docstrings; doctests must run against `load_synth_yeast_plate()` (600×800, 96 ground-truth colonies).
- Commit after every task.

---

## File Structure

**Modified:**
- `src/phenotypic/detect/nn/_dino_support.py` — resize policy + grid geometry. Gains `NATIVE_PROCESSOR_KWARGS`, `patch_grid_hw`, `covered_hw`, `upsample_grid_to_image`, `pool_prototype_tiled`. Existing `resize_mask_to_grid`, `align_mask_to_grid`, `cosine_match_to_mask`, `pool_prototype` gain a `patch` parameter.
- `src/phenotypic/detect/nn/_tiling.py` — tiling policy. Gains `_iou`, `_merge_tiles_iou_nms` (moved in), `tile_overlap_px`, `owning_tile_index`, `assign_by_centroid_core`.
- `src/phenotypic/detect/nn/_sam3_detector.py` — merge moves out; switches to centroid-in-core.
- `src/phenotypic/detect/nn/_fssdino_detector.py` — `tile_px` default, `_segment_crop` upsample.
- `src/phenotypic/detect/nn/_insid3_detector.py` — `tile_px` default, patch threading.
- `src/phenotypic/detect/nn/_dinosam2_detector.py` — tiled DINO, `crop_*` pass-through.
- `src/phenotypic/detect/nn/_sam2_detector.py` — `crop_n_layers` default, `box_nms_thresh`, docstring corrections.
- `docs/source/how_to/pages/gpu_detection_setup.md` — lines 167, 178, 261, 299.

**Test files:**
- `tests/unit/detect/nn/test_dino_support.py`
- `tests/unit/detect/nn/test_tiling.py`
- `tests/unit/detect/nn/test_sam3_detector.py`
- `tests/unit/detect/nn/test_fssdino_detector.py`
- `tests/unit/detect/nn/test_insid3_detector.py`
- `tests/unit/detect/nn/test_dinosam2_detector.py`
- `tests/unit/detect/nn/test_sam2_detector.py`

**Created:**
- `scripts/accuracy_gate_gpu_detectors.py` — measures IoU against `synth_plate` ground truth.

---

## Task 1: Native processor kwargs + patch geometry

**Files:**
- Modify: `src/phenotypic/detect/nn/_dino_support.py:154-280`
- Test: `tests/unit/detect/nn/test_dino_support.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `NATIVE_PROCESSOR_KWARGS: dict[str, bool]`
  - `patch_grid_hw(pixel_hw: Tuple[int, int], patch: int) -> Tuple[int, int]`
  - `covered_hw(grid_hw: Tuple[int, int], patch: int) -> Tuple[int, int]`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/detect/nn/test_dino_support.py`:

```python
class TestPatchGeometry:
    def test_patch_grid_hw_floors(self):
        from phenotypic.detect.nn._dino_support import patch_grid_hw

        assert patch_grid_hw((518, 518), 14) == (37, 37)
        assert patch_grid_hw((512, 512), 14) == (36, 36)
        assert patch_grid_hw((512, 512), 16) == (32, 32)
        assert patch_grid_hw((600, 800), 14) == (42, 57)

    def test_covered_hw_is_grid_times_patch(self):
        from phenotypic.detect.nn._dino_support import covered_hw

        assert covered_hw((37, 37), 14) == (518, 518)
        assert covered_hw((42, 57), 14) == (588, 798)

    def test_native_kwargs_disable_resize_and_crop(self):
        from phenotypic.detect.nn._dino_support import NATIVE_PROCESSOR_KWARGS

        assert NATIVE_PROCESSOR_KWARGS == {
            "do_resize": False,
            "do_center_crop": False,
        }


class TestProcessorPolicy:
    def test_extract_patch_features_requests_native_geometry(self):
        """The three extract_* fns must never let the checkpoint's
        classification preset (224 center-crop) decide the input size."""
        import numpy as np
        import torch

        from phenotypic.detect.nn._dino_support import extract_patch_features

        seen = {}

        class FakeInputs(dict):
            def to(self, _device):
                return self

        class FakeProcessor:
            def __call__(self, images, return_tensors=None, **kwargs):
                seen.update(kwargs)
                h, w = images.shape[:2]
                return FakeInputs({"pixel_values": torch.zeros((1, 3, h, w))})

        class FakeConfig:
            patch_size = 14
            num_register_tokens = 0

        class FakeModel:
            config = FakeConfig()

            def __call__(self, pixel_values=None, **_):
                hp, wp = 518 // 14, 518 // 14
                tokens = torch.zeros((1, 1 + hp * wp, 8))

                class Out:
                    last_hidden_state = tokens

                return Out()

        rgb = np.zeros((518, 518, 3), dtype=np.uint8)
        dense = extract_patch_features(
            FakeModel(), FakeProcessor(), rgb, device="cpu"
        )
        assert seen == {"do_resize": False, "do_center_crop": False}
        assert dense.shape == (37, 37, 8)
```

`FakeInputs` subclasses `dict` because `extract_patch_features` calls `.to(device)`
on the processor's return value.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/detect/nn/test_dino_support.py::TestPatchGeometry -v`
Expected: FAIL with `ImportError: cannot import name 'patch_grid_hw'`

- [ ] **Step 3: Write minimal implementation**

In `_dino_support.py`, after the `_DINOV3_SIZE_TO_REPO` block (line ~40):

```python
#: Processor kwargs that pin the model input to the tile's native geometry.
#: Without these, ``AutoImageProcessor`` applies the checkpoint's *classification*
#: preset — DINOv2 resizes shortest-edge to 256 then center-crops to 224; DINOv3
#: resizes squarely to 224 — so every tile reaches the ViT at 224x224 regardless
#: of ``tile_px``. Under these kwargs the patch grid is ``(h // patch, w // patch)``
#: and native px per patch is exactly ``patch_size``.
NATIVE_PROCESSOR_KWARGS: dict[str, bool] = {
    "do_resize": False,
    "do_center_crop": False,
}


def patch_grid_hw(pixel_hw: Tuple[int, int], patch: int) -> Tuple[int, int]:
    """Return the ``(Hp, Wp)`` patch grid a ViT produces for ``pixel_hw``.

    Patch embedding is a stride-``patch`` convolution, so it floors: a
    non-multiple input silently drops up to ``patch - 1`` pixels off the bottom
    and right. Use :func:`covered_hw` for the extent the grid actually spans.

    Args:
        pixel_hw: ``(H, W)`` of the tensor handed to the model.
        patch: ``model.config.patch_size`` (14 for DINOv2, 16 for DINOv3).

    Returns:
        ``(Hp, Wp)`` patch-grid shape.
    """
    return (int(pixel_hw[0]) // int(patch), int(pixel_hw[1]) // int(patch))


def covered_hw(grid_hw: Tuple[int, int], patch: int) -> Tuple[int, int]:
    """Return the pixel extent a ``grid_hw`` patch grid actually covers.

    ``(hp * patch, wp * patch)`` — never the original ``(H, W)``. Mapping a grid
    onto ``(H, W)`` instead introduces a scale error of up to
    ``(patch - 1) / H`` (2.0% vertically for a 600-px tile at patch 14).

    Args:
        grid_hw: ``(Hp, Wp)`` patch grid.
        patch: ``model.config.patch_size``.

    Returns:
        ``(Hp * patch, Wp * patch)``.
    """
    return (int(grid_hw[0]) * int(patch), int(grid_hw[1]) * int(patch))
```

Then in **all three** extract functions, replace the processor call at lines 175, 216, 263:

```python
    inputs = processor(
        images=rgb_uint8, return_tensors="pt", **NATIVE_PROCESSOR_KWARGS
    ).to(device)
```

And in `extract_patch_features`, replace the inline grid computation (line ~183):

```python
    patch = int(getattr(model.config, "patch_size", 16))
    grid_hw = patch_grid_hw((in_h, in_w), patch)
```

Apply the same substitution in `extract_reference_features` and
`extract_hidden_layer_features`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/detect/nn/test_dino_support.py -v`
Expected: PASS

- [ ] **Step 5: Verify against the real backbones**

Run:

```bash
uv run python -c "
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from phenotypic.detect.nn._dino_support import NATIVE_PROCESSOR_KWARGS, patch_grid_hw
for rid, tile in [('facebook/dinov2-base', 518), ('facebook/dinov3-vitb16-pretrain-lvd1689m', 512)]:
    p = AutoImageProcessor.from_pretrained(rid); m = AutoModel.from_pretrained(rid)
    px = p(images=np.zeros((tile,tile,3),dtype=np.uint8), return_tensors='pt', **NATIVE_PROCESSOR_KWARGS)['pixel_values']
    hw = tuple(px.shape[-2:]); patch = m.config.patch_size
    print(rid.split('/')[-1], hw, patch_grid_hw(hw, patch), f'{tile/patch_grid_hw(hw,patch)[0]:.1f} px/patch')
"
```

Expected:
```
dinov2-base (518, 518) (37, 37) 14.0 px/patch
dinov3-vitb16-pretrain-lvd1689m (512, 512) (32, 32) 16.0 px/patch
```

If DINOv3 is not downloaded, this step needs `PHENOTYPIC_ACCEPT_MODEL_LICENSE=dinov3` and an approved HF gate. Skip the DINOv3 half and note it, rather than guessing.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/detect/nn/_dino_support.py tests/unit/detect/nn/test_dino_support.py
git commit -m "fix(nn): feed DINO tiles at native geometry, not the 224 classification preset"
```

---

## Task 2: Covered-extent grid↔image mapping

**Files:**
- Modify: `src/phenotypic/detect/nn/_dino_support.py:282-360` (`resize_mask_to_grid`, `align_mask_to_grid`), `:428-464` (`cosine_match_to_mask`), `:361-398` (`pool_prototype`)
- Test: `tests/unit/detect/nn/test_dino_support.py`

**Interfaces:**
- Consumes: `patch_grid_hw`, `covered_hw` (Task 1).
- Produces:
  - `upsample_grid_to_image(grid, image_hw: Tuple[int, int], patch: int, *, order: int = 0) -> np.ndarray`
  - `resize_mask_to_grid(mask, grid_hw, patch: int | None = None) -> np.ndarray`
  - `align_mask_to_grid(mask, proc_hw, grid_hw, patch: int | None = None) -> np.ndarray`
  - `cosine_match_to_mask(features, prototype, thresh, out_shape, patch: int | None = None) -> np.ndarray`
  - `pool_prototype(features, mask, proc_hw=None, patch: int | None = None) -> np.ndarray`

`patch` is keyword-with-default `None` on the four existing functions so present callers and tests keep working; `None` reproduces today's whole-extent behaviour. Every in-repo caller passes it.

- [ ] **Step 1: Write the failing test**

```python
class TestCoveredExtentMapping:
    def test_upsample_grid_preserves_centroid(self):
        """A disc centred in the grid must stay centred after upsampling.

        Mapping a (42, 57) grid onto a 600x800 tile instead of its covered
        588x798 extent displaces objects by ~2% vertically.
        """
        import numpy as np

        from phenotypic.detect.nn._dino_support import upsample_grid_to_image

        grid = np.zeros((42, 57), dtype=bool)
        grid[20:23, 27:30] = True  # centred block
        gy, gx = np.nonzero(grid)
        grid_cy = (gy.mean() + 0.5) / 42
        grid_cx = (gx.mean() + 0.5) / 57

        full = upsample_grid_to_image(grid, (600, 800), 14)
        assert full.shape == (600, 800)
        fy, fx = np.nonzero(full)
        assert abs(fy.mean() / 600 - grid_cy) < 0.01
        assert abs(fx.mean() / 800 - grid_cx) < 0.01

    def test_upsample_pads_the_truncated_remainder(self):
        import numpy as np

        from phenotypic.detect.nn._dino_support import upsample_grid_to_image

        grid = np.ones((42, 57), dtype=bool)
        full = upsample_grid_to_image(grid, (600, 800), 14)
        assert full.shape == (600, 800)
        assert full.all()  # edge-padded, no False stripe at 588..600

    def test_resize_mask_to_grid_crops_to_covered_extent(self):
        import numpy as np

        from phenotypic.detect.nn._dino_support import resize_mask_to_grid

        mask = np.zeros((600, 800), dtype=bool)
        mask[588:600, :] = True  # lives ONLY in the truncated remainder
        small = resize_mask_to_grid(mask, (42, 57), patch=14)
        assert not small.any()  # the ViT never saw those rows

    def test_round_trip_grid_image_grid_is_identity(self):
        import numpy as np

        from phenotypic.detect.nn._dino_support import (
            resize_mask_to_grid,
            upsample_grid_to_image,
        )

        rng = np.random.default_rng(0)
        grid = rng.random((42, 57)) > 0.5
        full = upsample_grid_to_image(grid, (600, 800), 14)
        back = resize_mask_to_grid(full, (42, 57), patch=14)
        assert (back == grid).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/detect/nn/test_dino_support.py::TestCoveredExtentMapping -v`
Expected: FAIL with `ImportError: cannot import name 'upsample_grid_to_image'`

- [ ] **Step 3: Write minimal implementation**

Add after `covered_hw`:

```python
def upsample_grid_to_image(
    grid: "np.ndarray",
    image_hw: Tuple[int, int],
    patch: int,
    *,
    order: int = 0,
) -> "np.ndarray":
    """Upsample a patch grid to full resolution through its covered extent.

    The grid spans ``covered_hw(grid.shape, patch)``, **not** ``image_hw``: a
    stride-``patch`` conv floors, so up to ``patch - 1`` rows/columns at the
    bottom/right were never seen by the model. Resizing straight to ``image_hw``
    stretches the grid over pixels it does not describe (a 2.0% vertical scale
    error for a 600-px tile at patch 14). Resize to the covered extent, then
    edge-pad the remainder.

    Args:
        grid: ``(Hp, Wp)`` patch-grid array (boolean mask or float score map).
        image_hw: ``(H, W)`` of the tile the grid came from.
        patch: ``model.config.patch_size``.
        order: Interpolation order — ``0`` (nearest) for masks, ``1`` for
            score maps.

    Returns:
        ``(H, W)`` array with ``grid``'s dtype semantics preserved.
    """
    import numpy as np
    from skimage.transform import resize

    h, w = int(image_hw[0]), int(image_hw[1])
    ch, cw = covered_hw(grid.shape, patch)
    arr = np.asarray(grid)
    is_bool = arr.dtype == bool

    covered = resize(
        arr.astype(np.float32),
        (ch, cw),
        order=order,
        preserve_range=True,
        anti_aliasing=False,
    )
    if (ch, cw) != (h, w):
        covered = np.pad(
            covered, ((0, max(0, h - ch)), (0, max(0, w - cw))), mode="edge"
        )[:h, :w]
    return (covered > 0.5) if is_bool else covered
```

Amend `resize_mask_to_grid` to accept `patch` and crop first:

```python
def resize_mask_to_grid(
    mask: "np.ndarray", grid_hw: Tuple[int, int], patch: int | None = None
) -> "np.ndarray":
    """Downsample a full-resolution boolean mask onto the patch grid.

    Nearest-neighbour (``order=0``) so labels are not interpolated. When
    ``patch`` is given the mask is first cropped to
    ``covered_hw(grid_hw, patch)`` — the extent the grid actually describes —
    so the truncated bottom/right remainder cannot leak into a patch.

    Args:
        mask: ``(H, W)`` boolean (or 0/1) mask.
        grid_hw: ``(Hp, Wp)`` target patch grid.
        patch: ``model.config.patch_size``. When *None*, the whole mask is
            resized (legacy behaviour; introduces a sub-patch scale error).

    Returns:
        ``(Hp, Wp)`` boolean mask.
    """
    import numpy as np
    from skimage.transform import resize

    hp, wp = int(grid_hw[0]), int(grid_hw[1])
    arr = np.asarray(mask)
    if patch is not None:
        ch, cw = covered_hw((hp, wp), patch)
        arr = arr[:ch, :cw]
    small = (
        resize(
            arr.astype(np.float32),
            (hp, wp),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )
        > 0.5
    )
    return small.astype(bool)
```

Thread `patch` through `align_mask_to_grid` (final line becomes
`return resize_mask_to_grid(processed, grid_hw, patch)`), through `pool_prototype`
(both branches forward `patch`), and rewrite `cosine_match_to_mask`'s upsample:

```python
    sim = cosine_similarity_map(features, proto)
    if patch is None:
        sim_full = resize(
            sim.astype(np.float32), out_shape, order=1,
            preserve_range=True, anti_aliasing=False,
        )
    else:
        sim_full = upsample_grid_to_image(
            sim.astype(np.float32), out_shape, patch, order=1
        )
    return (sim_full > thresh).astype(bool)
```

adding `patch: int | None = None` to its signature and documenting it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/detect/nn/test_dino_support.py -v`
Expected: PASS (all classes, including Task 1's)

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/detect/nn/_dino_support.py tests/unit/detect/nn/test_dino_support.py
git commit -m "fix(nn): map patch grids through their covered extent, not the full tile"
```

---

## Task 3: FssDinoDetector — 1:1 tiles

**Files:**
- Modify: `src/phenotypic/detect/nn/_fssdino_detector.py:386` (`tile_px`), `:545-572` (`_segment_crop`)
- Test: `tests/unit/detect/nn/test_fssdino_detector.py`

**Interfaces:**
- Consumes: `upsample_grid_to_image` (Task 2).
- Produces: nothing downstream.

- [ ] **Step 1: Write the failing test**

```python
class TestFssDinoResolution:
    def test_tile_px_default_is_a_dinov2_patch_multiple(self):
        from phenotypic.detect.nn import FssDinoDetector

        det = FssDinoDetector()
        assert det.dino_version == 2
        assert det.tile_px == 518
        assert det.tile_px % 14 == 0   # 14 * 37

    def test_segment_crop_upsamples_through_covered_extent(self, monkeypatch):
        """A 600x800 crop at patch 14 has a (42, 57) grid covering 588x798.
        The returned mask must be 600x800 with no scale error."""
        import numpy as np

        from phenotypic.detect.nn import FssDinoDetector

        det = FssDinoDetector(dino_version=2)
        det._device = "cpu"
        det._model = type("M", (), {"config": type("C", (), {"patch_size": 14})()})()
        det._processor = object()
        det._fg_prototypes = np.ones((1, 4), dtype=np.float64)
        det._bg_prototypes = np.zeros((1, 4), dtype=np.float64)
        det._fg_gram = np.eye(4)
        det._bg_gram = np.eye(4)

        monkeypatch.setattr(
            "phenotypic.detect.nn._dino_support.extract_hidden_layer_features",
            lambda *a, **k: np.ones((42, 57, 4), dtype=np.float32),
        )
        out = det._segment_crop(np.zeros((600, 800, 3), dtype=np.uint8))
        assert out.shape == (600, 800)
        assert out.dtype == bool
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/detect/nn/test_fssdino_detector.py::TestFssDinoResolution -v`
Expected: FAIL — `assert 512 == 518`

- [ ] **Step 3: Write minimal implementation**

At `_fssdino_detector.py:386`:

```python
    # 518 = 14 * 37 — an exact DINOv2 patch multiple. Under the native
    # processor kwargs the resolution is pinned at patch_size (14.0 native
    # px/patch) regardless of tile_px, so this is a COMPUTE choice, not a
    # fidelity one: 518 and 1022 both give 14.0 px/patch, but 1022 costs 3.3x
    # more (attention is quadratic in tokens per tile). Smaller wins.
    tile_px: Annotated[int, TuneSpec(256, 1024)] = 518
```

Rewrite `_segment_crop`'s tail (replacing the `resize(...)` at line 565):

```python
        grid_mask = assign_foreground(fg_score, bg_score, self.similarity_thresh)
        patch = int(getattr(self._model.config, "patch_size", 14))
        return upsample_grid_to_image(
            grid_mask.astype(bool), (rgb.shape[0], rgb.shape[1]), patch
        )
```

and import `upsample_grid_to_image` alongside `extract_hidden_layer_features`.

Add a load-time guard at the end of `_ensure_model_loaded`:

```python
        patch = int(getattr(self._model.config, "patch_size", 14))
        if self.tile_px % patch:
            warnings.warn(
                f"tile_px={self.tile_px} is not a multiple of the backbone's "
                f"patch_size={patch}; the ViT will silently drop "
                f"{self.tile_px % patch} px off each tile's bottom and right. "
                f"Use {(self.tile_px // patch) * patch}.",
                UserWarning,
                stacklevel=2,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/detect/nn/test_fssdino_detector.py -v`
Expected: PASS

- [ ] **Step 5: Run the functional regression (needs DINOv2, ungated)**

Add to the existing `TestFssDinoFunctionalDinoV2` class (`test_fssdino_detector.py:201`):

```python
    def test_dense_grid_is_native_not_224(self, synth_plate):
        """Direct F1 regression: synth_plate is 600x800, so at patch 14 the
        grid must be (42, 57). Before the fix every tile was squashed to
        224x224 -> a (16, 16) grid."""
        import numpy as np

        from phenotypic.detect.nn._dino_support import extract_patch_features
        from transformers import AutoImageProcessor, AutoModel

        m = AutoModel.from_pretrained("facebook/dinov2-small").eval()
        p = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        rgb = np.asarray(synth_plate.rgb[:], dtype=np.uint8)
        dense = extract_patch_features(m, p, rgb, device="cpu")
        assert dense.shape[:2] == (600 // 14, 800 // 14) == (42, 57)
```

Run: `uv run pytest tests/unit/detect/nn/test_fssdino_detector.py -v -k Functional`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/detect/nn/_fssdino_detector.py tests/unit/detect/nn/test_fssdino_detector.py
git commit -m "fix(fssdino): 1:1 native tiles (32.0 -> 14.0 native px/patch)"
```

---

## Task 4: Insid3Detector — 1:1 tiles + gate error

**Files:**
- Modify: `src/phenotypic/detect/nn/_insid3_detector.py:266` (`tile_px`), `:409-425` (`_match_crop`)
- Test: `tests/unit/detect/nn/test_insid3_detector.py`

**Interfaces:**
- Consumes: `cosine_match_to_mask(..., patch=...)` (Task 2).
- Produces: nothing downstream.

- [ ] **Step 1: Write the failing test**

```python
class TestInsid3Resolution:
    def test_tile_px_default_is_a_dinov3_patch_multiple(self):
        from phenotypic.detect.nn import Insid3Detector

        det = Insid3Detector()
        assert det.dino_version == 3
        assert det.tile_px == 512
        assert det.tile_px % 16 == 0   # 16 * 32

    def test_match_crop_threads_patch_into_cosine_match(self, monkeypatch):
        import numpy as np

        from phenotypic.detect.nn import Insid3Detector

        det = Insid3Detector()
        det._device = "cpu"
        det._model = type("M", (), {"config": type("C", (), {"patch_size": 16})()})()
        det._processor = object()
        det._basis = np.zeros((4, 0))
        det._prototype = np.ones(4)

        seen = {}
        monkeypatch.setattr(
            "phenotypic.detect.nn._dino_support.extract_patch_features",
            lambda *a, **k: np.ones((32, 32, 4), dtype=np.float32),
        )

        def fake_match(features, prototype, thresh, out_shape, patch=None):
            seen["patch"] = patch
            return np.zeros(out_shape, dtype=bool)

        monkeypatch.setattr(
            "phenotypic.detect.nn._dino_support.cosine_match_to_mask", fake_match
        )
        det._match_crop(np.zeros((512, 512, 3), dtype=np.uint8))
        assert seen["patch"] == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/detect/nn/test_insid3_detector.py::TestInsid3Resolution -v`
Expected: FAIL — `assert 1024 == 512`

- [ ] **Step 3: Write minimal implementation**

At `_insid3_detector.py:266`:

```python
    # 512 = 16 * 32 — an exact DINOv3 patch multiple. Resolution is pinned at
    # patch_size (16.0 native px/patch) regardless of tile_px, so this is a
    # COMPUTE choice: 512 and 1024 both give 16.0 px/patch, but 1024 costs
    # 2.6x more. NOTE: DINOv3's config.image_size reports 224 (a classification
    # preset) and must NOT be used to pick this default.
    tile_px: Annotated[int, TuneSpec(256, 1024)] = 512
```

In `_match_crop`, thread the patch size:

```python
        patch = int(getattr(self._model.config, "patch_size", 16))
        return cosine_match_to_mask(
            feats_deb,
            self._prototype,
            thresh=self.similarity_thresh,
            out_shape=(rgb.shape[0], rgb.shape[1]),
            patch=patch,
        )
```

Add the same non-multiple `warnings.warn` guard as Task 3 to `_ensure_model_loaded`
(default `patch` fallback `16`).

Raise the DINOv3 gate error at construction rather than first `apply()` — add to
`Insid3Detector`:

```python
    @field_validator("dino_version")
    @classmethod
    def _warn_gated_default(cls, v: int) -> int:
        """DINOv3 is gated; surface that at construction, not at first apply()."""
        if v == 3:
            warnings.warn(
                "Insid3Detector defaults to dino_version=3 (DINOv3), whose "
                "weights are gated: request access at "
                "https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m, "
                "run `uv run hf auth login`, and set "
                "PHENOTYPIC_ACCEPT_MODEL_LICENSE=dinov3. "
                "Pass dino_version=2 for the ungated DINOv2 backbone.",
                UserWarning,
                stacklevel=2,
            )
        return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/detect/nn/test_insid3_detector.py -v`
Expected: PASS. Existing construction tests may now emit the gate warning — if any
assert on `pytest.warns(None)` or run under `-W error`, wrap those constructions in
`pytest.warns(UserWarning)`.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/detect/nn/_insid3_detector.py tests/unit/detect/nn/test_insid3_detector.py
git commit -m "fix(insid3): 1:1 native tiles (73.1 -> 16.0 native px/patch); surface the DINOv3 gate at construction"
```

---

## Task 5: `_tiling.py` — centroid-in-core instance merge

**Files:**
- Modify: `src/phenotypic/detect/nn/_tiling.py`
- Modify: `src/phenotypic/detect/nn/_sam3_detector.py:27-93` (remove `_iou`, `_merge_tiles_iou_nms`)
- Test: `tests/unit/detect/nn/test_tiling.py`

**Interfaces:**
- Consumes: `_Tile`, `_plan_tiles` (existing).
- Produces:
  - `_iou(mask_a, mask_b) -> float` (moved verbatim)
  - `_merge_tiles_iou_nms(objmaps: List[np.ndarray], iou_thresh: float) -> np.ndarray` (moved verbatim; retained for the single-tile relabel path)
  - `tile_overlap_px(tiles: List[_Tile]) -> int`
  - `owning_tile_index(tiles: List[_Tile], centroid_yx: Tuple[float, float]) -> int`
  - `assign_by_centroid_core(tiles: List[_Tile], tile_objmaps: List[np.ndarray], out_shape: Tuple[int, int]) -> np.ndarray`

`tile_objmaps[i]` is **tile-local**, shape `(tiles[i].h, tiles[i].w)`. This differs
from `_merge_tiles_iou_nms`, which takes full-image-offset objmaps.

**Why not port SAM2's `is_box_near_crop_edge`:** edge rejection requires
`overlap_px >= d + 2*atol` or the colony is rejected from *every* tile and vanishes.
SAM2 survives this only because its pyramid always runs a full-image layer 0 and its
`1/box_area` NMS outvotes that coarse copy. Uniform tiles have neither. At
`tile_px=512, overlap=0.15, atol=20` the safe diameter is 37 px while `synth_plate`'s
median colony is 39 px.

- [ ] **Step 1: Write the failing test**

```python
class TestAssignByCentroidCore:
    def _two_tiles(self):
        from phenotypic.detect.nn._tiling import _Tile
        # 100x180 image, two 100x100 tiles overlapping by 20 px.
        return [_Tile(0, 0, 100, 100), _Tile(0, 80, 100, 180)]

    def test_fragment_regression_one_colony_stays_one(self):
        """A colony fully inside tile A also appears as a fragment in tile B.
        Under _merge_tiles_iou_nms(iou_thresh=0.5) the fragment survives
        (IoU == area fraction) and paints OVER the colony. Centroid-in-core
        must yield exactly one instance with its area intact."""
        import numpy as np

        from phenotypic.detect.nn._tiling import assign_by_centroid_core

        tiles = self._two_tiles()
        # Colony spans image cols 70..90 -> inside A (0..100); B (80..180)
        # sees only cols 80..90 as a fragment.
        a = np.zeros((100, 100), dtype=np.uint16)
        a[40:60, 70:90] = 1                      # whole, tile-local
        b = np.zeros((100, 100), dtype=np.uint16)
        b[40:60, 0:10] = 1                       # fragment, tile-local

        merged = assign_by_centroid_core(tiles, [a, b], (100, 180))
        labels = [l for l in np.unique(merged) if l]
        assert len(labels) == 1
        assert int((merged == labels[0]).sum()) == 20 * 20

    def test_instance_claimed_by_exactly_one_tile(self):
        import numpy as np

        from phenotypic.detect.nn._tiling import assign_by_centroid_core

        tiles = self._two_tiles()
        # Colony wholly inside the overlap band, cols 82..88: both tiles see it whole.
        a = np.zeros((100, 100), dtype=np.uint16); a[10:20, 82:88] = 1
        b = np.zeros((100, 100), dtype=np.uint16); b[10:20, 2:8] = 1
        merged = assign_by_centroid_core(tiles, [a, b], (100, 180))
        assert len([l for l in np.unique(merged) if l]) == 1

    def test_single_tile_relabels_contiguously(self):
        import numpy as np

        from phenotypic.detect.nn._tiling import _Tile, assign_by_centroid_core

        t = [_Tile(0, 0, 50, 50)]
        om = np.zeros((50, 50), dtype=np.uint16)
        om[5:10, 5:10] = 7
        om[20:25, 20:25] = 9
        merged = assign_by_centroid_core(t, [om], (50, 50))
        assert sorted(int(l) for l in np.unique(merged) if l) == [1, 2]

    def test_overlap_guard_warns_when_colony_exceeds_overlap(self):
        import numpy as np
        import pytest

        from phenotypic.detect.nn._tiling import assign_by_centroid_core

        tiles = self._two_tiles()          # overlap_px == 20
        a = np.zeros((100, 100), dtype=np.uint16)
        a[10:70, 30:90] = 1                # d == 60 > 20
        b = np.zeros((100, 100), dtype=np.uint16)
        with pytest.warns(UserWarning, match="overlap"):
            merged = assign_by_centroid_core(tiles, [a, b], (100, 180))
        assert len([l for l in np.unique(merged) if l]) == 1   # not deleted


class TestTileOverlapPx:
    def test_overlap_of_two_tiles(self):
        from phenotypic.detect.nn._tiling import _Tile, tile_overlap_px

        assert tile_overlap_px([_Tile(0, 0, 100, 100), _Tile(0, 80, 100, 180)]) == 20

    def test_single_tile_has_no_overlap(self):
        from phenotypic.detect.nn._tiling import _Tile, tile_overlap_px

        assert tile_overlap_px([_Tile(0, 0, 10, 10)]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/detect/nn/test_tiling.py -v`
Expected: FAIL with `ImportError: cannot import name 'assign_by_centroid_core'`

- [ ] **Step 3: Write minimal implementation**

Move `_iou` and `_merge_tiles_iou_nms` verbatim from `_sam3_detector.py:27-93` into
`_tiling.py` (change the warning text `"SAM3 merged"` → `"Tiled merge kept"`), and
add:

```python
def tile_overlap_px(tiles: List[_Tile]) -> int:
    """Smallest overlap in pixels between any two overlapping tiles.

    Zero when a single tile covers the image. This is the bound that decides
    whether a colony can be lost: an instance wider than the overlap is cleaved
    in every tile that contains it.

    Args:
        tiles: Crop rectangles from :func:`_plan_tiles`.

    Returns:
        Minimum positive pairwise overlap along either axis, or ``0``.
    """
    if len(tiles) < 2:
        return 0
    best: int | None = None
    for i, a in enumerate(tiles):
        for b in tiles[i + 1:]:
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            if oy > 0 and ox > 0:  # genuinely overlapping, not merely abutting
                cand = min(oy, ox)
                best = cand if best is None else min(best, cand)
    return int(best) if best is not None else 0


def owning_tile_index(
    tiles: List[_Tile], centroid_yx: tuple[float, float]
) -> int:
    """Index of the tile whose *core* contains ``centroid_yx``.

    A tile's core is the region closer to its centre than to any other tile's
    centre — a Voronoi partition of the tile centres, intersected with the tile.
    Since :func:`_plan_tiles` guarantees the tiles cover the image, every point
    lies in at least one tile, and the nearest-centre rule picks exactly one.
    Border tiles' cores therefore reach the image edge with no gap.

    This is what makes cross-tile duplicates impossible: a colony fully inside
    one tile is claimed by whichever core holds its true centroid; the same
    colony's *fragment* in a neighbouring tile has a centroid pushed within
    ``d/2`` of that tile's edge, while the core begins ``overlap_px / 2`` inside
    it — so when ``overlap_px >= d`` the fragment is never claimed.

    Args:
        tiles: Crop rectangles from :func:`_plan_tiles`.
        centroid_yx: ``(y, x)`` in full-image coordinates.

    Returns:
        Index into *tiles*.
    """
    cy, cx = float(centroid_yx[0]), float(centroid_yx[1])
    best_i, best_d = 0, None
    for i, t in enumerate(tiles):
        if not (t.y0 <= cy < t.y1 and t.x0 <= cx < t.x1):
            continue
        ty = (t.y0 + t.y1) / 2.0
        tx = (t.x0 + t.x1) / 2.0
        d = (cy - ty) ** 2 + (cx - tx) ** 2
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return best_i


def assign_by_centroid_core(
    tiles: List[_Tile],
    tile_objmaps: List["np.ndarray"],
    out_shape: tuple[int, int],
) -> "np.ndarray":
    """Merge tile-local instance maps by centroid-in-core assignment.

    Each instance is kept by exactly the one tile whose core contains its
    centroid (:func:`owning_tile_index`); every other copy is discarded. No NMS,
    no edge tolerance, no duplicates by construction. Fragments are dropped
    because nobody claims them.

    Contrast :func:`_merge_tiles_iou_nms`, whose IoU between a whole colony and
    its cross-tile fragment equals the fragment's area fraction ``f`` — so every
    fragment with ``f <= iou_thresh`` survives and, being painted later in the
    largest-first order, overwrites the colony it came from.

    Survivors are relabelled ``1..N`` largest-first, matching
    :func:`_merge_tiles_iou_nms`.

    Args:
        tiles: Crop rectangles in full-image coordinates.
        tile_objmaps: Per-tile uint16 objmaps, each ``(tile.h, tile.w)``,
            **tile-local** (not offset).
        out_shape: ``(H, W)`` of the full image.

    Returns:
        A full-image uint16 objmap with contiguous labels ``1..N``.

    Raises:
        ValueError: If *tiles* and *tile_objmaps* differ in length.

    Warns:
        UserWarning: When the largest retained instance is wider than
            :func:`tile_overlap_px` — the condition under which a colony can be
            cleaved in every tile and lost.
    """
    import warnings

    import numpy as np

    if len(tiles) != len(tile_objmaps):
        raise ValueError(
            f"assign_by_centroid_core: {len(tiles)} tiles vs "
            f"{len(tile_objmaps)} objmaps"
        )

    kept: list["np.ndarray"] = []
    for i, (tile, om) in enumerate(zip(tiles, tile_objmaps)):
        om = np.asarray(om)
        for label in np.unique(om):
            if label == 0:
                continue
            local = om == label
            ys, xs = np.nonzero(local)
            cy = ys.mean() + tile.y0
            cx = xs.mean() + tile.x0
            if owning_tile_index(tiles, (cy, cx)) != i:
                continue
            full = np.zeros(out_shape, dtype=bool)
            full[tile.y0:tile.y1, tile.x0:tile.x1] = local
            kept.append(full)

    if not kept:
        return np.zeros(out_shape, dtype=np.uint16)

    kept.sort(key=lambda m: int(m.sum()), reverse=True)

    overlap = tile_overlap_px(tiles)
    if overlap:
        ys, xs = np.nonzero(kept[0])
        d = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
        if d > overlap:
            warnings.warn(
                f"Largest instance is {d} px across but tiles overlap by only "
                f"{overlap} px; an instance wider than the overlap can be "
                f"cleaved in every tile and lost. Raise tile_overlap.",
                UserWarning,
                stacklevel=2,
            )

    max_labels = int(np.iinfo(np.uint16).max)
    if len(kept) > max_labels:
        warnings.warn(
            f"Tiled merge kept {len(kept)} instances, exceeding uint16 range. "
            f"Only the first {max_labels} (largest) will be labeled.",
            UserWarning,
            stacklevel=2,
        )
        kept = kept[:max_labels]

    objmap = np.zeros(out_shape, dtype=np.uint16)
    for idx, mask in enumerate(kept, start=1):
        objmap[mask] = idx
    return objmap
```

Update `_tiling.py`'s module docstring: the instance merge now lives here, not in
`_sam3_detector.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/detect/nn/test_tiling.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/detect/nn/_tiling.py src/phenotypic/detect/nn/_sam3_detector.py tests/unit/detect/nn/test_tiling.py
git commit -m "feat(tiling): centroid-in-core instance merge; move IoU-NMS out of _sam3_detector"
```

---

## Task 6: Sam3Detector — adopt the new merge

**Files:**
- Modify: `src/phenotypic/detect/nn/_sam3_detector.py:337-396` (`_infer_batch`), imports at `:14-24`
- Test: `tests/unit/detect/nn/test_sam3_detector.py:117-137` (existing merge tests import from `_sam3_detector`)

**Interfaces:**
- Consumes: `assign_by_centroid_core`, `_merge_tiles_iou_nms`, `_plan_tiles`, `_Tile` (Task 5).
- Produces: nothing downstream.

- [ ] **Step 1: Write the failing test**

```python
class TestSam3UsesCentroidCore:
    def test_infer_batch_merges_by_centroid_core(self, monkeypatch):
        """A colony straddling a tile seam must yield one instance, not a
        colony plus its fragment."""
        import numpy as np

        from phenotypic.detect.nn import Sam3Detector

        det = Sam3Detector(tile_px=100, tile_overlap=0.2)
        det._model = object()
        det._processor = object()
        monkeypatch.setattr(det, "_ensure_model_loaded", lambda: None)

        # _plan_tiles((100, 180), 100, 0.2) -> [(0,0,100,100), (0,80,100,180)].
        # Tile 0 sees the whole colony at global cols 70..90; tile 1 sees only
        # its fragment at global cols 80..90.
        def fake_forward(crops):
            out = []
            for i, c in enumerate(crops):
                om = np.zeros(c.shape[:2], dtype=np.uint16)
                if i == 0:
                    om[40:60, 70:90] = 1     # whole colony, tile-local
                else:
                    om[40:60, 0:10] = 1      # fragment, tile-local
                out.append(om)
            return out

        monkeypatch.setattr(det, "_forward_tiles", fake_forward)
        sample = np.zeros((100, 180, 3), dtype=np.uint8)
        (result,) = det._infer_batch([sample])
        assert result.shape == (100, 180)
        labels = [l for l in np.unique(result) if l]
        assert len(labels) == 1                          # not colony + fragment
        assert int((result == labels[0]).sum()) == 20 * 20   # area uncorrupted
```

Also retarget the two existing merge tests (`test_sam3_detector.py:117`, `:129`)
to import from the new home:

```python
        from phenotypic.detect.nn._tiling import _merge_tiles_iou_nms
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/detect/nn/test_sam3_detector.py -v`
Expected: FAIL — `ImportError: cannot import name '_merge_tiles_iou_nms' from '_sam3_detector'`

- [ ] **Step 3: Write minimal implementation**

Replace `_sam3_detector.py`'s import block:

```python
from phenotypic.detect.nn._tiling import (
    _merge_tiles_iou_nms,
    _plan_tiles,
    _Tile,
    _tile_starts,
    assign_by_centroid_core,
)

_ = (_Tile, _tile_starts, _merge_tiles_iou_nms)
```

Rewrite the tail of `_infer_batch` so tile objmaps stay **tile-local** (delete the
`full = np.zeros(full_shape); full[...] = crop_obj` offsetting loop):

```python
        # Group tile-local objmaps by sample (no offsetting — the merge does it).
        per_sample_local: list[list[np.ndarray]] = [[] for _ in batch]
        cursor = 0
        for s_idx in range(len(batch)):
            for _t in plans[s_idx]:
                per_sample_local[s_idx].append(crop_objmaps[cursor])
                cursor += 1

        results: list[np.ndarray] = []
        for s_idx in range(len(batch)):
            tile_objmaps = per_sample_local[s_idx]
            if not tile_objmaps:
                results.append(np.zeros(full_shapes[s_idx], dtype=np.uint16))
            else:
                results.append(
                    assign_by_centroid_core(
                        plans[s_idx], tile_objmaps, full_shapes[s_idx]
                    )
                )
        return results
```

`tile_merge_iou` is now unused by this path. Keep the field (removing it would break
`from_json` on existing payloads) and mark it deprecated in its docstring:

```python
    # Deprecated: the tiled instance merge is centroid-in-core
    # (_tiling.assign_by_centroid_core), which needs no IoU threshold. Retained
    # so existing serialized pipelines keep deserializing.
    tile_merge_iou: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/detect/nn/test_sam3_detector.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/detect/nn/_sam3_detector.py tests/unit/detect/nn/test_sam3_detector.py
git commit -m "fix(sam3): merge tiles by centroid-in-core; a seam-straddling colony is no longer split"
```

---

## Task 7: DinoSam2Detector — tiled DINO + crop pass-through

**Files:**
- Modify: `src/phenotypic/detect/nn/_dinosam2_detector.py:250-266` (fields), `:304-318` (`_ensure_model_loaded`), `:328-362` (`_infer_one`)
- Modify: `src/phenotypic/detect/nn/_dino_support.py` (add `pool_prototype_tiled`)
- Test: `tests/unit/detect/nn/test_dinosam2_detector.py`

**Interfaces:**
- Consumes: `extract_patch_features`, `pool_prototype` (Tasks 1–2); `_plan_tiles`, `owning_tile_index` (Task 5).
- Produces: `pool_prototype_tiled(dense_by_tile: List[np.ndarray], tiles: List[_Tile], mask: np.ndarray, patch: int) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
class TestDinoSam2Tiling:
    def test_has_tiling_fields(self):
        from phenotypic.detect.nn import DinoSam2Detector

        det = DinoSam2Detector()
        assert det.tile_px == 518
        assert det.tile_overlap == 0.15
        assert det.crop_n_layers == 1

    def test_pool_prototype_tiled_is_nonzero_for_a_small_colony(self):
        """F3 regression: on a full plate a 30px colony is 0.16 patches wide,
        so pool_prototype rounds it to empty and returns a zero vector."""
        import numpy as np

        from phenotypic.detect.nn._dino_support import pool_prototype_tiled
        from phenotypic.detect.nn._tiling import _Tile

        tiles = [_Tile(0, 0, 518, 518)]
        dense = [np.ones((37, 37, 8), dtype=np.float32)]
        mask = np.zeros((518, 518), dtype=bool)
        mask[250:280, 250:280] = True          # 30 px colony

        proto = pool_prototype_tiled(dense, tiles, mask, 14)
        assert proto.shape == (8,)
        assert np.any(proto)                   # NOT the zero vector
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/detect/nn/test_dinosam2_detector.py::TestDinoSam2Tiling -v`
Expected: FAIL — `assert 0 == 518` (no `tile_px` field)

- [ ] **Step 3: Write minimal implementation**

Add to `_dino_support.py`:

```python
def pool_prototype_tiled(
    dense_by_tile: List["np.ndarray"],
    tiles: List["_Tile"],
    mask: "np.ndarray",
    patch: int,
) -> "np.ndarray":
    """Pool a full-image mask's prototype from per-tile dense features.

    The mask is assigned to the tile whose core contains its centroid
    (:func:`~phenotypic.detect.nn._tiling.owning_tile_index`), cropped to that
    tile, and pooled against that tile's feature grid. Pooling against a
    full-plate grid instead makes every colony sub-patch — a 30 px colony on a
    3000x4000 plate spans 0.16 patches — so the mask rounds to empty and
    :func:`pool_prototype` returns its zero-vector fail-safe for every proposal.

    Args:
        dense_by_tile: One ``(Hp, Wp, D)`` feature grid per tile.
        tiles: Crop rectangles, aligned with *dense_by_tile*.
        mask: ``(H, W)`` boolean mask in full-image coordinates.
        patch: ``model.config.patch_size``.

    Returns:
        ``(D,)`` mean-pooled prototype.
    """
    import numpy as np

    from phenotypic.detect.nn._tiling import owning_tile_index

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros(dense_by_tile[0].shape[-1], dtype=np.float32)
    i = owning_tile_index(tiles, (ys.mean(), xs.mean()))
    t = tiles[i]
    local = mask[t.y0:t.y1, t.x0:t.x1]
    return pool_prototype(dense_by_tile[i], local, patch=patch)
```

Add fields to `DinoSam2Detector` (mirroring `Sam2Detector`):

```python
    tile_px: Annotated[int, TuneSpec(256, 1024)] = 518
    tile_overlap: Annotated[float, TuneSpec(0.0, 0.4)] = 0.15
    crop_n_layers: Annotated[int, TuneSpec(0, 2)] = 1
    crop_nms_thresh: Annotated[float, TuneSpec(0.0, 1.0)] = 0.7
    crop_overlap_ratio: Annotated[float, TuneSpec(0.0, 0.5)] = 512 / 1500
    crop_n_points_downscale_factor: Annotated[int, TuneSpec(1, 2)] = 1
```

Pass them through at `_dinosam2_detector.py:311`:

```python
        self._generator = build_sam2_generator(
            self.sam2_model_size,
            device=self._device,
            min_mask_region_area=self.min_proposal_area,
            crop_n_layers=self.crop_n_layers,
            crop_nms_thresh=self.crop_nms_thresh,
            crop_overlap_ratio=self.crop_overlap_ratio,
            crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
        )
```

Replace the feature extraction in `_infer_one`:

```python
        from phenotypic.detect.nn._dino_support import (
            extract_patch_features,
            pool_prototype_tiled,
        )
        from phenotypic.detect.nn._tiling import _plan_tiles

        patch = int(getattr(self._dino_model.config, "patch_size", 14))
        tiles = _plan_tiles(rgb.shape[:2], self.tile_px, self.tile_overlap)
        dense_by_tile = [
            extract_patch_features(
                self._dino_model,
                self._dino_processor,
                rgb[t.y0:t.y1, t.x0:t.x1],
                device=self._device,
            )
            for t in tiles
        ]
        features = np.stack(
            [
                pool_prototype_tiled(dense_by_tile, tiles, p, patch).astype(np.float64)
                for p in proposals
            ]
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/detect/nn/test_dinosam2_detector.py -v`
Expected: PASS

- [ ] **Step 5: Run the functional F3 regression (DINOv2, ungated)**

Add to the DINOv2-guarded functional class:

```python
    def test_prototypes_are_not_all_zero(self, synth_plate):
        """Direct F3 regression: before the fix every proposal pooled an empty
        mask and got the zero vector, so all scores were identical."""
        import numpy as np

        from phenotypic.detect.nn._dino_support import (
            extract_patch_features,
            pool_prototype_tiled,
        )
        from phenotypic.detect.nn._tiling import _plan_tiles
        from transformers import AutoImageProcessor, AutoModel

        m = AutoModel.from_pretrained("facebook/dinov2-small").eval()
        p = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        rgb = np.asarray(synth_plate.rgb[:], dtype=np.uint8)
        om = np.asarray(synth_plate.objmap[:])

        tiles = _plan_tiles(rgb.shape[:2], 518, 0.15)
        dense = [
            extract_patch_features(m, p, rgb[t.y0:t.y1, t.x0:t.x1], device="cpu")
            for t in tiles
        ]
        protos = [
            pool_prototype_tiled(dense, tiles, om == lab, 14)
            for lab in np.unique(om)[1:6]
        ]
        assert all(np.any(pr) for pr in protos)
        assert len({tuple(np.round(pr, 4)) for pr in protos}) > 1
```

Run: `uv run pytest tests/unit/detect/nn/test_dinosam2_detector.py -v -k Functional`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/detect/nn/_dinosam2_detector.py src/phenotypic/detect/nn/_dino_support.py tests/unit/detect/nn/test_dinosam2_detector.py
git commit -m "fix(dinosam2): tile DINO features; stop pooling every colony into a zero vector"
```

---

## Task 8: Sam2Detector — engage the crop pyramid

**Files:**
- Modify: `src/phenotypic/detect/nn/_sam2_detector.py:18-32` (`build_sam2_generator` signature), `:108-280` (docstring + fields), `:284-309` (`_ensure_model_loaded`)
- Test: `tests/unit/detect/nn/test_sam2_detector.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `build_sam2_generator(..., box_nms_thresh: float = 0.7)`.

- [ ] **Step 1: Write the failing test**

```python
class TestSam2CropPyramid:
    def test_crop_pyramid_is_engaged_by_default(self):
        from phenotypic.detect.nn import Sam2Detector

        det = Sam2Detector()
        assert det.crop_n_layers == 1
        assert det.box_nms_thresh == 0.7

    def test_build_sam2_generator_accepts_box_nms_thresh(self):
        """`box_nms_thresh` dedups the dense point grid's redundant proposals
        within one crop. SAM2 exposes it; Sam2Detector did not."""
        import inspect

        from phenotypic.detect.nn._sam2_detector import build_sam2_generator

        sig = inspect.signature(build_sam2_generator)
        assert "box_nms_thresh" in sig.parameters
        assert sig.parameters["box_nms_thresh"].default == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/detect/nn/test_sam2_detector.py::TestSam2CropPyramid -v`
Expected: FAIL — `assert 0 == 1`

- [ ] **Step 3: Write minimal implementation**

Add `box_nms_thresh: float = 0.7` to `build_sam2_generator`'s signature (after
`stability_score_thresh`), document it, and forward it to
`SAM2AutomaticMaskGenerator(..., box_nms_thresh=box_nms_thresh, ...)`.

On `Sam2Detector`:

```python
    box_nms_thresh: Annotated[float, TuneSpec(0.0, 1.0)] = 0.7
    # 1 crop layer = 5 encoder passes (1 full image + 2**(1+1)**2 = 4 crops).
    # Engages SAM2's edge rejection, crop overlap, full-image fallback, and
    # resolution-preferring NMS. ~3.91 -> ~1.9 native px per encoder px on a
    # 4000x3000 plate, at ~5x the inference cost.
    crop_n_layers: Annotated[int, TuneSpec(0, 2)] = 1
```

and forward `box_nms_thresh=self.box_nms_thresh` in `_ensure_model_loaded`.

Fix the three docstring defects. Replace the `crop_n_layers` paragraph
(`_sam2_detector.py:155-165`):

```
        crop_n_layers: Number of additional **crop-pyramid layers** SAM2 runs
            for higher accuracy on large or dense plates.  SAM2's encoder
            resizes the whole image to a fixed **1024x1024 square** -- a
            non-aspect-preserving squash, so a 4:3 plate enters the model as
            ellipses -- and small colonies on a multi-megapixel plate can be
            lost to downsampling.  ``0`` keeps a single full-image pass; each
            added layer ``i`` re-tiles the *entire* image into
            ``(2 ** (i + 1)) ** 2`` overlapping crops (4 at layer 1, 16 at
            layer 2), each encoded nearer native resolution, and merges them by
            NMS that prefers masks from smaller crops.  The full-image pass is
            always included.  Default 1 (5 encoder passes).
```

Add to the `Sam2Detector` docstring's parameter list:

```
        box_nms_thresh: Box-IoU cutoff for non-maximum suppression between the
            dense point grid's redundant proposals *within* one crop (distinct
            from ``crop_nms_thresh``, which deduplicates *across* crops).
            Typical range 0.5--0.9.  Default 0.7 (the SAM2 default).
```

Update `build_sam2_generator`'s docstring: replace "SAM2's native sliding-window
crop mechanism" with "SAM2's native crop pyramid", and correct the `crop_n_layers`
line to `(2 ** (i + 1)) ** 2` crops per layer.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/detect/nn/test_sam2_detector.py -v`
Expected: PASS

- [ ] **Step 5: Verify the doctest still runs**

Run: `uv run pytest --doctest-modules src/phenotypic/detect/nn/_sam2_detector.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/detect/nn/_sam2_detector.py tests/unit/detect/nn/test_sam2_detector.py
git commit -m "fix(sam2): engage the crop pyramid by default; expose box_nms_thresh; correct crop-count docs"
```

---

## Task 9: Tune-gate, docs, changelog

**Files:**
- Modify: `docs/source/how_to/pages/gpu_detection_setup.md:167,178,261,299`
- Verify: `tests/fixtures/tune/annotation_allowlist.json`

**Interfaces:** none.

- [ ] **Step 1: Run the annotation-coverage gate**

Run: `uv run pytest tests/unit/tune -v -k "annotation or coverage or ln"`
Expected: PASS. If it fails, the new numeric fields (`DinoSam2.tile_px`,
`tile_overlap`, `crop_n_layers`, `crop_nms_thresh`, `crop_overlap_ratio`,
`crop_n_points_downscale_factor`; `Sam2.box_nms_thresh`) are missing a `TuneSpec`.
Add the annotation — do **not** add them to the allowlist.

- [ ] **Step 2: Run the full nn suite and mypy**

Run:
```bash
uv run pytest tests/unit/detect/nn -v
uv run mypy src/phenotypic/detect/nn
uv run ruff check --fix src/phenotypic/detect/nn
```
Expected: all pass.

- [ ] **Step 3: Update the how-to page**

At `docs/source/how_to/pages/gpu_detection_setup.md`, replace the four lines:

- line 178: ``- `tile_px` / `tile_overlap` — dense-plate tiling (defaults 1008 / 0.15). Under the native processor geometry the resolution is pinned at the backbone's `patch_size` regardless of `tile_px`, so **larger tiles are slower at identical fidelity** — 1008 and 2016 both give the same px/patch, but 2016 costs ~3x more.``
- line 261 (`Insid3Detector`): ``- `tile_px` / `tile_overlap` — large-plate tiling (defaults 512 / 0.15). `512 = 16 * 32` is an exact DINOv3 patch multiple. Do not raise it for accuracy; it buys none.``
- line 299 (`FssDinoDetector`): ``- `tile_px` / `tile_overlap` — large-plate tiling (defaults 518 / 0.15). `518 = 14 * 37` is an exact DINOv2 patch multiple.``
- line 167: replace the "SAM3 resizes to 1008 internally" claim with "SAM3's internal resize is unverified (`facebook/sam3` is gated); `tile_px=1008` is carried as an assumption."

Add a **Behaviour changes** admonition near the top:

```markdown
```{warning}
**Behaviour change (this release).** DINO-backed detectors previously fed every
tile to the ViT at 224x224 regardless of `tile_px`. They now feed tiles at native
geometry. Existing pipelines deserialized from JSON keep their pinned `tile_px`
but **will produce different (higher-resolution) masks**, and cost 6.6x (FssDino)
to 26x (Insid3) more GPU time. Re-serialize to pick up the new `tile_px` defaults.
`Sam2Detector`'s `crop_n_layers` default moves 0 -> 1 (~5x cost) for
newly-constructed detectors only.
```
```

- [ ] **Step 4: Commit**

```bash
git add docs/source/how_to/pages/gpu_detection_setup.md
git commit -m "docs(gpu): tile_px is a compute knob, not a fidelity knob; record behaviour changes"
```

---

## Task 10: Accuracy gate

**Files:**
- Create: `scripts/accuracy_gate_gpu_detectors.py`
- Modify: `docs/superpowers/specs/2026-07-08-gpu-detect-fixes/2026-07-08-gpu-detect-fixes-design.md` (Accuracy budget section)

**Interfaces:** none.

This task decides whether the two **costly default changes** ship. The code from
Tasks 3, 4, and 8 lands regardless; only the defaults are gated.

- [ ] **Step 1: Write the gate script**

```python
"""Measure objmask IoU against synth_plate's 96 ground-truth colonies.

Run before/after the resolution fixes to decide whether the new tile_px and
crop_n_layers defaults ship. Spec: docs/superpowers/specs/2026-07-08-gpu-detect-fixes/
"""

from __future__ import annotations

import numpy as np

from phenotypic.data import load_synth_yeast_plate


def mask_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    """Foreground IoU of two boolean masks."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    union = (pred | truth).sum()
    return float((pred & truth).sum() / union) if union else 0.0


def evaluate(detector, label: str) -> None:
    image = load_synth_yeast_plate()
    truth = np.asarray(image.objmap[:]) > 0
    n_truth = int(image.num_objects)

    detector.apply(image)
    pred = np.asarray(image.objmask[:])
    print(
        f"{label:<34} IoU {mask_iou(pred, truth):.4f}  "
        f"objects {image.num_objects:>4} / {n_truth}"
    )


if __name__ == "__main__":
    from phenotypic.detect.nn import FssDinoDetector, Insid3Detector

    # Baseline reproduces the pre-fix geometry by forcing the old tile_px.
    evaluate(FssDinoDetector(dino_version=2, dino_size="small", tile_px=512,
                             device="cpu"), "FssDino  tile_px=512 (old default)")
    evaluate(FssDinoDetector(dino_version=2, dino_size="small", tile_px=518,
                             device="cpu"), "FssDino  tile_px=518 (new default)")
```

- [ ] **Step 2: Run the gate**

Run: `uv run python scripts/accuracy_gate_gpu_detectors.py`

Record the actual printed numbers. Do **not** invent them.

- [ ] **Step 3: Decide, and record the decision**

- If the new default's IoU **≥** the old default's: keep the new defaults from
  Tasks 3, 4, 8.
- If it is **lower**: revert only the *default values* (`tile_px` back to 512 /
  1024, `crop_n_layers` back to 0), keep every code change, and record the
  measurement. The `tile_px` semantics fix stands either way — it makes the
  parameter mean what its docstring says.

Replace the spec's Accuracy budget placeholder text with a table of the measured
IoU and object counts, and state which branch was taken.

- [ ] **Step 4: Commit**

```bash
git add scripts/accuracy_gate_gpu_detectors.py docs/superpowers/specs/2026-07-08-gpu-detect-fixes/
git commit -m "test(nn): accuracy gate for the GPU detector resolution fixes"
```

---

## Self-Review

**Spec coverage:**

| Spec item | Task |
|---|---|
| F1 — `tile_px` inert | 1, 3, 4 |
| F2 — query-side misregistration + truncation | 2, 3, 4 |
| F3 — DinoSam2 zero prototypes + `crop_*` pass-through | 7 |
| F4 — Sam2 crop pyramid off; `box_nms_thresh`; docstring errors | 8 |
| F5 — Sam3 fragment bug; merge moves to `_tiling` | 5, 6 |
| `assign_by_centroid_core` + overlap guard | 5 |
| `tile_px` default rationale (not `config.image_size`) | 3, 4 (comments) |
| Compatibility: no field removed; `tile_merge_iou` retained | 6 |
| Tune annotation gate | 9 |
| Docs lines 167/178/261/299 + behaviour-change notice | 9 |
| Accuracy budget | 10 |
| Non-goals: micro_sam, SAM3 `tile_px` verification | untouched by design |

**Known deviations from the spec, discovered while reading the code:**

1. The spec places INSID3's mask upsample in `_insid3_detector.py`. It is actually in
   the **shared** `cosine_match_to_mask` (`_dino_support.py:428`), so Task 2 fixes it
   once for both callers rather than per-detector.
2. The spec's `assign_by_centroid_core` assumed a stride-derived core. `_plan_tiles`
   does not expose stride, and its last tile is clamped (overlapping more), so cores
   are defined by **nearest tile centre** (a Voronoi partition of tile centres,
   intersected with the tiles). This partitions exactly, needs no stride, and handles
   the clamped border tiles with no gap.
3. `reject_edge_instances` is **not implemented** — see Task 5's rationale.

**Type consistency:** `patch` is `int` everywhere; `patch_grid_hw`/`covered_hw` both
take/return `Tuple[int, int]`; `assign_by_centroid_core` takes **tile-local** objmaps
while `_merge_tiles_iou_nms` takes full-image-offset ones (Task 6 deletes the
offsetting loop accordingly).
