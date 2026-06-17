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
    proto = pool_prototype(feats, mask)
    assert np.allclose(proto, np.ones(8))  # mean over masked patches


def test_pool_prototype_resizes_full_res_mask_to_grid():
    """A full-resolution mask is downsampled to the patch grid before pooling."""
    from phenotypic.detect.nn._dino_support import pool_prototype

    feats = np.zeros((4, 4, 8), np.float32)
    feats[1:3, 1:3] = 1.0
    # Full-res 8x8 mask covering the same central region (2x upsampled).
    mask = np.zeros((8, 8), bool)
    mask[2:6, 2:6] = True
    proto = pool_prototype(feats, mask)
    assert np.allclose(proto, np.ones(8))


def test_pool_prototype_empty_mask_returns_zero_vector():
    from phenotypic.detect.nn._dino_support import pool_prototype

    feats = np.ones((4, 4, 8), np.float32)
    mask = np.zeros((4, 4), bool)
    proto = pool_prototype(feats, mask)
    assert proto.shape == (8,)
    assert np.allclose(proto, 0.0)


def test_cosine_match_recovers_prototype_region():
    from phenotypic.detect.nn._dino_support import cosine_match_to_mask

    feats = np.zeros((4, 4, 8), np.float32)
    feats[1:3, 1:3] = 1.0
    proto = np.ones(8, np.float32)
    out = cosine_match_to_mask(feats, proto, thresh=0.9, out_shape=(8, 8))
    assert out.dtype == bool and out.shape == (8, 8)
    assert out.any() and not out.all()  # a region, not everything


def test_cosine_match_zero_prototype_is_all_false():
    from phenotypic.detect.nn._dino_support import cosine_match_to_mask

    feats = np.ones((4, 4, 8), np.float32)
    proto = np.zeros(8, np.float32)
    out = cosine_match_to_mask(feats, proto, thresh=0.5, out_shape=(8, 8))
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
    small = resize_mask_to_grid(mask, grid_hw=(4, 9))
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
    grid = align_mask_to_grid(mask, proc_hw=(224, 224), grid_hw=(16, 16))
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
    proto = pool_prototype(feats, mask, proc_hw=(224, 224))
    assert proto.shape == (4,)
    # Foreground patches are the all-ones block → prototype ≈ ones.
    assert np.all(proto > 0.5)
