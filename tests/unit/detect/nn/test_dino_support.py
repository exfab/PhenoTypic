"""Shared frozen-DINO prototype-match core (Spec 2b, Task 2).

Pure math on synthetic features/masks — no model, no weights. Covers the C1
register-token grid-reshape fix and the prototype-pool + cosine-match core that
``Insid3Detector`` and ``FssDinoDetector`` build on.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Backbone id mapping
# ---------------------------------------------------------------------------


def test_hf_dino_id_v2_and_v3():
    from phenotypic.detect.nn._dino_support import hf_dino_id

    assert hf_dino_id(2, "base") == "facebook/dinov2-base"
    assert hf_dino_id(2, "small") == "facebook/dinov2-small"
    assert (
        hf_dino_id(3, "large")
        == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    assert (
        hf_dino_id(3, "base")
        == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    )


# ---------------------------------------------------------------------------
# C1 — register-token-aware patch-grid reshape
# ---------------------------------------------------------------------------


class TestPatchGridReshape:
    def test_drops_cls_and_register_tokens_exact_grid(self):
        """``1 + num_register_tokens + Hp*Wp`` tokens reshape to (Hp, Wp, D)."""
        from phenotypic.detect.nn._dino_support import reshape_patch_tokens

        Hp, Wp, D, n_reg = 3, 4, 8, 4
        n_tokens = 1 + n_reg + Hp * Wp
        # Token 0 = CLS, tokens 1..n_reg = registers, rest = patches.
        tokens = np.zeros((n_tokens, D), np.float32)
        # Tag each patch token with its flat index so we can verify ordering.
        for i in range(Hp * Wp):
            tokens[1 + n_reg + i, 0] = i + 1.0
        grid = reshape_patch_tokens(
            tokens, grid_hw=(Hp, Wp), num_register_tokens=n_reg
        )
        assert grid.shape == (Hp, Wp, D)
        # Row-major patch order preserved (no register contamination).
        assert grid[0, 0, 0] == 1.0
        assert grid[0, 1, 0] == 2.0
        assert grid[1, 0, 0] == Wp + 1.0

    def test_v2_zero_registers(self):
        from phenotypic.detect.nn._dino_support import reshape_patch_tokens

        Hp, Wp, D = 2, 2, 5
        tokens = np.zeros((1 + Hp * Wp, D), np.float32)
        for i in range(Hp * Wp):
            tokens[1 + i, 0] = i + 1.0
        grid = reshape_patch_tokens(
            tokens, grid_hw=(Hp, Wp), num_register_tokens=0
        )
        assert grid.shape == (Hp, Wp, D)
        assert grid[0, 0, 0] == 1.0
        assert grid[1, 1, 0] == 4.0


# ---------------------------------------------------------------------------
# Prototype pooling + cosine match
# ---------------------------------------------------------------------------


def test_pool_prototype_is_masked_mean():
    from phenotypic.detect.nn._dino_support import pool_prototype

    feats = np.zeros((4, 4, 8), np.float32)
    feats[1:3, 1:3] = 1.0
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    proto = pool_prototype(feats, mask, patch=1)  # 4x4 grid covers the 4x4 mask
    assert np.allclose(proto, np.ones(8))  # mean over masked patches


def test_pool_prototype_resizes_full_res_mask_to_grid():
    """A full-resolution mask is downsampled to the patch grid before pooling."""
    from phenotypic.detect.nn._dino_support import pool_prototype

    feats = np.zeros((4, 4, 8), np.float32)
    feats[1:3, 1:3] = 1.0
    # Full-res 8x8 mask covering the same central region (2x upsampled).
    mask = np.zeros((8, 8), bool)
    mask[2:6, 2:6] = True
    proto = pool_prototype(feats, mask, patch=2)  # 4x4 grid covers the 8x8 mask
    assert np.allclose(proto, np.ones(8))


def test_pool_prototype_empty_mask_returns_zero_vector():
    from phenotypic.detect.nn._dino_support import pool_prototype

    feats = np.ones((4, 4, 8), np.float32)
    mask = np.zeros((4, 4), bool)
    proto = pool_prototype(feats, mask, patch=1)
    assert proto.shape == (8,)
    assert np.allclose(proto, 0.0)


def test_cosine_match_recovers_prototype_region():
    from phenotypic.detect.nn._dino_support import cosine_match_to_mask

    feats = np.zeros((4, 4, 8), np.float32)
    feats[1:3, 1:3] = 1.0
    proto = np.ones(8, np.float32)
    out = cosine_match_to_mask(feats, proto, thresh=0.9, out_shape=(8, 8), patch=2)
    assert out.dtype == bool and out.shape == (8, 8)
    assert out.any() and not out.all()  # a region, not everything


def test_cosine_match_zero_prototype_is_all_false():
    from phenotypic.detect.nn._dino_support import cosine_match_to_mask

    feats = np.ones((4, 4, 8), np.float32)
    proto = np.zeros(8, np.float32)
    out = cosine_match_to_mask(feats, proto, thresh=0.5, out_shape=(8, 8), patch=2)
    assert out.dtype == bool and not out.any()  # fail-safe


def test_cosine_similarity_map_shapes_and_range():
    from phenotypic.detect.nn._dino_support import cosine_similarity_map

    feats = np.zeros((3, 5, 6), np.float32)
    # One patch parallel to the prototype (cosine 1), one orthogonal (cosine 0).
    feats[1, 2, 0] = 1.0  # parallel to proto = e0
    feats[0, 0, 1] = 1.0  # orthogonal to proto
    proto = np.zeros(6, np.float32)
    proto[0] = 1.0
    sim = cosine_similarity_map(feats, proto)
    assert sim.shape == (3, 5)
    assert -1.0001 <= float(sim.min()) and float(sim.max()) <= 1.0001
    assert sim[1, 2] == pytest.approx(1.0, abs=1e-6)  # parallel → cosine 1
    assert sim[0, 0] == pytest.approx(0.0, abs=1e-6)  # orthogonal → cosine 0
    assert sim[2, 4] == pytest.approx(0.0, abs=1e-6)  # zero patch → fail-safe 0


# ---------------------------------------------------------------------------
# resize_mask_to_grid (W4 — non-square exemplar alignment)
# ---------------------------------------------------------------------------


def test_resize_mask_to_grid_non_square():
    from phenotypic.detect.nn._dino_support import resize_mask_to_grid

    # Non-square full-res mask → square-ish patch grid (order=0 nearest).
    mask = np.zeros((40, 90), bool)
    mask[10:30, 30:60] = True
    small = resize_mask_to_grid(mask, grid_hw=(4, 9), patch=10)
    assert small.shape == (4, 9)
    assert small.dtype == bool
    assert small.any() and not small.all()


def test_align_mask_to_grid_non_square_through_processed_geometry():
    """W4: a non-square exemplar mask aligns via the (square) processed geom."""
    from phenotypic.detect.nn._dino_support import align_mask_to_grid

    # Non-square 220x300 exemplar; the processor squashes to a 224x224 square
    # → a 16x16 patch grid (patch=14). The mask must follow the same path.
    mask = np.zeros((220, 300), bool)
    mask[40:180, 60:240] = True  # central block
    grid = align_mask_to_grid(mask, proc_hw=(224, 224), grid_hw=(16, 16), patch=14)
    assert grid.shape == (16, 16)
    assert grid.dtype == bool
    assert grid.any() and not grid.all()  # central region survives, edges off
    # Corners (always background) stay off after the two-step alignment.
    assert not grid[0, 0] and not grid[15, 15]


def test_pool_prototype_with_proc_hw_aligns_non_square():
    """pool_prototype honours proc_hw for the W4 two-step alignment."""
    from phenotypic.detect.nn._dino_support import pool_prototype

    feats = np.zeros((16, 16, 4), np.float32)
    feats[4:12, 4:12] = 1.0  # central patch-grid block is "foreground"
    # A non-square full-res mask covering the central region of a 220x300 image.
    mask = np.zeros((220, 300), bool)
    mask[55:165, 75:225] = True
    proto = pool_prototype(feats, mask, proc_hw=(224, 224), patch=14)
    assert proto.shape == (4,)
    # Foreground patches are the all-ones block → prototype ≈ ones.
    assert np.all(proto > 0.5)


# ---------------------------------------------------------------------------
# F1 — native processor geometry + patch grid arithmetic
# ---------------------------------------------------------------------------


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
    """The extract_* fns must never let the checkpoint's classification preset
    (224 center-crop) decide the input size."""

    @staticmethod
    def _fakes(patch=14, n_reg=0, grid=(37, 37)):
        import torch

        seen: dict = {}

        class FakeInputs(dict):
            def to(self, _device):
                return self

        class FakeProcessor:
            def __call__(self, images, return_tensors=None, **kwargs):
                seen.update(kwargs)
                h, w = images.shape[:2]
                return FakeInputs({"pixel_values": torch.zeros((1, 3, h, w))})

        class FakeConfig:
            patch_size = patch
            num_register_tokens = n_reg

        class FakeModel:
            config = FakeConfig()

            def __call__(self, pixel_values=None, **_):
                hp, wp = grid
                tokens = torch.zeros((1, 1 + n_reg + hp * wp, 8))

                class Out:
                    last_hidden_state = tokens
                    hidden_states = [tokens, tokens]

                return Out()

        return seen, FakeModel(), FakeProcessor()

    def test_extract_patch_features_requests_native_geometry(self):
        from phenotypic.detect.nn._dino_support import extract_patch_features

        seen, model, processor = self._fakes()
        rgb = np.zeros((518, 518, 3), dtype=np.uint8)
        dense = extract_patch_features(model, processor, rgb, device="cpu")
        assert seen == {"do_resize": False, "do_center_crop": False}
        assert dense.shape == (37, 37, 8)

    def test_extract_reference_features_requests_native_geometry(self):
        from phenotypic.detect.nn._dino_support import extract_reference_features

        seen, model, processor = self._fakes(patch=16, n_reg=4, grid=(32, 32))
        rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        dense, proc_hw = extract_reference_features(
            model, processor, rgb, device="cpu"
        )
        assert seen == {"do_resize": False, "do_center_crop": False}
        assert dense.shape == (32, 32, 8)
        assert proc_hw == (512, 512)

    def test_extract_hidden_layer_features_requests_native_geometry(self):
        from phenotypic.detect.nn._dino_support import (
            extract_hidden_layer_features,
        )

        seen, model, processor = self._fakes()
        rgb = np.zeros((518, 518, 3), dtype=np.uint8)
        dense = extract_hidden_layer_features(
            model, processor, rgb, device="cpu", layer=-1
        )
        assert seen == {"do_resize": False, "do_center_crop": False}
        assert dense.shape == (37, 37, 8)


# ---------------------------------------------------------------------------
# F1 — covered-extent grid ↔ image mapping
# ---------------------------------------------------------------------------


class TestCoveredExtentMapping:
    def test_upsample_grid_preserves_centroid(self):
        """A block centred in the grid keeps its position in the covered extent.

        A (42, 57) grid at patch 14 covers 588x798 of a 600x800 tile. The
        block's fractional centroid must be preserved relative to that covered
        extent, NOT to the full tile: stretching the grid onto 600x800 instead
        displaces it by ~2% (6.1 px vertically here).
        """
        from phenotypic.detect.nn._dino_support import upsample_grid_to_image

        grid = np.zeros((42, 57), dtype=bool)
        grid[20:23, 27:30] = True  # centred block
        gy, gx = np.nonzero(grid)
        grid_cy = (gy.mean() + 0.5) / 42
        grid_cx = (gx.mean() + 0.5) / 57

        full = upsample_grid_to_image(grid, (600, 800), 14)
        assert full.shape == (600, 800)
        fy, fx = np.nonzero(full)
        # Covered extent = covered_hw((42, 57), 14) = (588, 798).
        assert abs((fy.mean() + 0.5) / 588 - grid_cy) < 1e-6
        assert abs((fx.mean() + 0.5) / 798 - grid_cx) < 1e-6
        # And it is NOT the stretched mapping the fix removes.
        assert abs((fy.mean() + 0.5) - grid_cy * 600) > 5.0

    def test_upsample_pads_the_truncated_remainder(self):
        from phenotypic.detect.nn._dino_support import upsample_grid_to_image

        grid = np.ones((42, 57), dtype=bool)
        full = upsample_grid_to_image(grid, (600, 800), 14)
        assert full.shape == (600, 800)
        assert full.all()  # edge-padded, no False stripe at 588..600

    def test_resize_mask_to_grid_crops_to_covered_extent(self):
        from phenotypic.detect.nn._dino_support import resize_mask_to_grid

        mask = np.zeros((600, 800), dtype=bool)
        mask[588:600, :] = True  # lives ONLY in the truncated remainder
        small = resize_mask_to_grid(mask, (42, 57), patch=14)
        assert not small.any()  # the ViT never saw those rows

    def test_round_trip_grid_image_grid_is_identity(self):
        from phenotypic.detect.nn._dino_support import (
            resize_mask_to_grid,
            upsample_grid_to_image,
        )

        rng = np.random.default_rng(0)
        grid = rng.random((42, 57)) > 0.5
        full = upsample_grid_to_image(grid, (600, 800), 14)
        back = resize_mask_to_grid(full, (42, 57), patch=14)
        assert (back == grid).all()

    def test_upsample_float_score_map_keeps_dtype_semantics(self):
        from phenotypic.detect.nn._dino_support import upsample_grid_to_image

        grid = np.linspace(0.0, 1.0, 37 * 37, dtype=np.float32).reshape(37, 37)
        full = upsample_grid_to_image(grid, (520, 520), 14, order=1)
        assert full.shape == (520, 520)
        assert full.dtype != bool
        assert float(full.max()) <= 1.0 + 1e-6

    def test_align_mask_to_grid_accepts_patch(self):
        from phenotypic.detect.nn._dino_support import align_mask_to_grid

        mask = np.zeros((600, 800), dtype=bool)
        mask[588:600, :] = True  # only in the truncated remainder
        grid = align_mask_to_grid(mask, (600, 800), (42, 57), patch=14)
        assert grid.shape == (42, 57)
        assert not grid.any()

    def test_pool_prototype_accepts_patch(self):
        from phenotypic.detect.nn._dino_support import pool_prototype

        feats = np.zeros((4, 4, 8), np.float32)
        feats[1:3, 1:3] = 1.0
        mask = np.zeros((60, 60), bool)
        mask[16:40, 16:40] = True
        proto = pool_prototype(feats, mask, patch=14)
        assert proto.shape == (8,)
        assert np.all(proto > 0.5)

    def test_cosine_match_to_mask_uses_covered_extent_when_patch_given(self):
        from phenotypic.detect.nn._dino_support import cosine_match_to_mask

        feats = np.zeros((42, 57, 8), np.float32)
        feats[20:23, 27:30] = 1.0
        proto = np.ones(8, np.float32)
        out = cosine_match_to_mask(
            feats, proto, thresh=0.9, out_shape=(600, 800), patch=14
        )
        assert out.shape == (600, 800)
        assert out.dtype == bool
        ys, xs = np.nonzero(out)
        # Grid block centre (21.5, 28.5) patches → ~(301, 399) px, not stretched.
        assert abs(ys.mean() - 21.5 * 14) < 14
        assert abs(xs.mean() - 28.5 * 14) < 14


class TestCoveredExtentIsRequiredOnBothDirections:
    """F6: omitting `patch` is a silent scale error, not a crash.

    The bundled exemplar is 220x300 -- a multiple of neither patch size -- so a
    grid built from it covers only 208x288 (patch 16) or 210x294 (patch 14).
    Mapping the exemplar mask over the full 220x300 selects foreground patches
    the ViT never saw, corrupting the prototype that defines "colony".
    """

    def test_patch_cannot_be_omitted(self):
        """`patch` is required, so the silent-scale-error path is unreachable.

        It was optional once. Omitting it mapped the grid over the whole mask
        instead of its covered extent, which is a rescale rather than a crash —
        and it reached production twice, on the query path and then on the
        exemplar path. A TypeError is the point of this test.
        """
        import numpy as np
        import pytest

        from phenotypic.detect.nn._dino_support import (
            align_mask_to_grid,
            cosine_match_to_mask,
            pool_prototype,
            resize_mask_to_grid,
        )

        mask = np.zeros((220, 300), dtype=bool)
        feats = np.ones((13, 18, 4), dtype=np.float32)

        with pytest.raises(TypeError):
            resize_mask_to_grid(mask, (13, 18))
        with pytest.raises(TypeError):
            align_mask_to_grid(mask, (220, 300), (13, 18))
        with pytest.raises(TypeError):
            pool_prototype(feats, mask)
        with pytest.raises(TypeError):
            cosine_match_to_mask(feats, np.ones(4), 0.5, (220, 300))

    def test_covered_extent_excludes_the_truncated_remainder(self):
        import numpy as np

        from phenotypic.detect.nn._dino_support import align_mask_to_grid

        # Rows 208..220 are outside the (13, 18) grid's covered extent at patch 16.
        mask = np.zeros((220, 300), dtype=bool)
        mask[208:220, :] = True
        assert not align_mask_to_grid(mask, (220, 300), (13, 18), 16).any()


class TestBackbonePatchSize:
    """One source of truth for the patch size, and it never guesses.

    Ten call sites once wrote `int(getattr(cfg, "patch_size", N))` with N=14 in
    some files and N=16 in others. A backbone missing `patch_size` would have
    silently disagreed with itself within one run — DINOv2 is patch-14, DINOv3
    patch-16, and the mismatch rescales every mask rather than raising.
    """

    def test_reads_the_value_from_the_model(self):
        import types

        from phenotypic.detect.nn._dino_support import backbone_patch_size

        model = types.SimpleNamespace(config=types.SimpleNamespace(patch_size=16))
        assert backbone_patch_size(model) == 16

    def test_raises_rather_than_guessing(self):
        import types

        import pytest

        from phenotypic.detect.nn._dino_support import backbone_patch_size

        with pytest.raises(ValueError, match="patch_size"):
            backbone_patch_size(types.SimpleNamespace(config=types.SimpleNamespace()))
        with pytest.raises(ValueError, match="patch_size"):
            backbone_patch_size(types.SimpleNamespace())
