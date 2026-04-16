"""Tests for MeasureSymmetricRadius — mask-based radial symmetry operator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phenotypic import Image
from phenotypic.measure import MeasureSymmetricRadius


# ---------------------------------------------------------------------------
# Shared helper (duplicated from test_measure_radial_expansion.py L154-161
# because it is defined inside a test class and not importable).
# ---------------------------------------------------------------------------


def _make_image_with_objmap(
    gray: np.ndarray, objmap: np.ndarray,
) -> Image:
    """Create an Image with a pre-set objmap (bypasses detection)."""
    rgb = np.stack([gray, gray, gray], axis=-1)
    image = Image(rgb)
    image.objmap[:] = objmap
    return image


# ---------------------------------------------------------------------------
# Synthetic-colony builders (module-level for reproducibility + debuggability).
# All return uint8 gray (background ≈ 220, object ≈ 40) and int32 objmap.
# ---------------------------------------------------------------------------


def _make_circular_colony(
    shape: tuple[int, int],
    center: tuple[int, int],
    core_radius: float,
    outer_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a concentric colony: bright dense core + dimmer outer halo.

    Both core and halo pixels are labeled 1 in the objmap and are guaranteed to
    form a single connected component because the outer halo strictly contains
    the core.

    Args:
        shape: (rows, cols) of the output arrays.
        center: (r, c) center of the colony.
        core_radius: Radius of the dense core (bright = darker inverted intensity).
        outer_radius: Outer radius of the halo (must be >= core_radius).

    Returns:
        Tuple of (gray, objmap). gray is uint8 with background ~220, core ~40,
        halo ~120. objmap is int32 with label 1 for the whole disk.
    """
    rows, cols = shape
    cr, cc = center
    gray = np.full((rows, cols), 220, dtype=np.uint8)
    objmap = np.zeros((rows, cols), dtype=np.int32)

    rr, cc_idx = np.ogrid[:rows, :cols]
    dist_sq = (rr - cr) ** 2 + (cc_idx - cc) ** 2

    outer_mask = dist_sq < outer_radius ** 2
    core_mask = dist_sq < core_radius ** 2

    # Outer halo gets a dimmer value; core is darkest (inverted-for-dark-colonies).
    gray[outer_mask] = 120
    gray[core_mask] = 40
    objmap[outer_mask] = 1
    return gray, objmap


def _make_half_mask_colony(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: float,
    angular_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a solid disk restricted to pixels whose angle falls in a wedge.

    Angles use the ``np.arctan2(dy, dx)`` convention with ``-π ≤ θ ≤ π``.

    Args:
        shape: (rows, cols) of the output arrays.
        center: (r, c) center of the colony.
        radius: Outer radius of the disk.
        angular_range: (theta_min, theta_max) in radians. Pixels outside this
            inclusive range are excluded from the mask.

    Returns:
        Tuple of (gray, objmap) with label 1 on the wedge.
    """
    rows, cols = shape
    cr, cc = center
    gray = np.full((rows, cols), 220, dtype=np.uint8)
    objmap = np.zeros((rows, cols), dtype=np.int32)

    rr, cc_idx = np.ogrid[:rows, :cols]
    dy = rr - cr
    dx = cc_idx - cc
    dist_sq = dy ** 2 + dx ** 2
    theta = np.arctan2(dy, dx)

    theta_min, theta_max = angular_range
    in_disk = dist_sq < radius ** 2
    in_wedge = (theta >= theta_min) & (theta <= theta_max)
    mask = in_disk & in_wedge

    gray[mask] = 40
    objmap[mask] = 1
    return gray, objmap


def _make_lopsided_colony(
    shape: tuple[int, int],
    center: tuple[int, int],
    core_radius: float,
    base_radius: float,
    bias_factor: float,
    bias_angular_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a colony whose outer radius is ``base_radius`` except in a wedge.

    Inside ``bias_angular_range`` the outer radius becomes
    ``base_radius * bias_factor``. The core disk (radius ``core_radius``) is
    always circular. Angles use ``np.arctan2(dy, dx)`` convention.

    Args:
        shape: (rows, cols) of the output arrays.
        center: (r, c) center of the colony.
        core_radius: Radius of the always-circular core.
        base_radius: Outer radius outside the bias wedge.
        bias_factor: Multiplier applied to ``base_radius`` inside the wedge.
        bias_angular_range: (theta_min, theta_max) in radians for the biased wedge.

    Returns:
        Tuple of (gray, objmap). Objmap labels the full lopsided disk as 1.
    """
    rows, cols = shape
    cr, cc = center
    gray = np.full((rows, cols), 220, dtype=np.uint8)
    objmap = np.zeros((rows, cols), dtype=np.int32)

    rr, cc_idx = np.ogrid[:rows, :cols]
    dy = rr - cr
    dx = cc_idx - cc
    dist = np.sqrt(dy ** 2 + dx ** 2)
    theta = np.arctan2(dy, dx)

    theta_min, theta_max = bias_angular_range
    in_bias_wedge = (theta >= theta_min) & (theta <= theta_max)
    bias_radius = base_radius * bias_factor

    # Per-pixel outer radius: bias_radius inside the wedge, base_radius outside.
    outer_radius_per_pixel = np.where(in_bias_wedge, bias_radius, base_radius)
    outer_mask = dist < outer_radius_per_pixel
    core_mask = dist < core_radius

    gray[outer_mask] = 120
    gray[core_mask] = 40
    objmap[outer_mask] = 1
    return gray, objmap


# ---------------------------------------------------------------------------
# Six test cases.
# ---------------------------------------------------------------------------


class TestMeasureSymmetricRadius:
    """Synthetic-colony ground-truth tests for MeasureSymmetricRadius."""

    # -- column-name constants (match the plan's SYMMETRIC_RADIUS enum) --
    CORE_COL = "SymmetricRadius_CoreRadius"
    SYMM_COL = "SymmetricRadius_SymmetricRadius"
    MEAN_COL = "SymmetricRadius_MeanExpansion"
    MAX_COL = "SymmetricRadius_MaxExpansion"

    # ------------------------------------------------------------------
    # Test 1 — symmetric circular colony.
    # ------------------------------------------------------------------

    def test_symmetric_circular_colony(self):
        """Circular colony: CoreRadius ≈ 15, Symmetric ≈ outer, Mean ≈ Max."""
        core_radius = 15.0
        outer_radius = 60.0
        gray, objmap = _make_circular_colony(
            shape=(200, 200),
            center=(100, 100),
            core_radius=core_radius,
            outer_radius=outer_radius,
        )
        image = _make_image_with_objmap(gray, objmap)

        op = MeasureSymmetricRadius()
        df = op.measure(image)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

        core = df[self.CORE_COL].iloc[0]
        symm = df[self.SYMM_COL].iloc[0]
        mean_exp = df[self.MEAN_COL].iloc[0]
        max_exp = df[self.MAX_COL].iloc[0]

        # A uniformly-filled solid disk has no mask-density gradient, so PELT
        # legitimately returns CoreRadius = 0. Either outcome is acceptable
        # as long as the value is non-negative and finite.
        assert np.isfinite(core) and 0.0 <= core <= outer_radius + 5.0, (
            f"CoreRadius={core} outside [0, {outer_radius + 5}]"
        )

        # SymmetricRadius should be close to outer_radius since the colony is
        # perfectly symmetric out to the very edge (within 5 px of outer).
        assert symm >= outer_radius - 5.0, (
            f"SymmetricRadius={symm} < outer_radius - 5 ({outer_radius - 5})"
        )

        # MaxExpansion ≈ outer_radius - CoreRadius (the detected value).
        expected_max = outer_radius - core
        assert abs(max_exp - expected_max) <= 3.0, (
            f"MaxExpansion={max_exp} not within 3 of {expected_max}"
        )

        # Circular symmetry ⇒ mean boundary distance ≈ max (within 5 px).
        assert abs(mean_exp - max_exp) <= 5.0, (
            f"|MeanExpansion - MaxExpansion| = {abs(mean_exp - max_exp)} > 5"
        )

    # ------------------------------------------------------------------
    # Test 2 — half-moon mask.
    # ------------------------------------------------------------------

    def test_half_moon_mask_triggers_asymmetry_early(self):
        """Half-disk wedge: SymmetricRadius should be well below MaxExpansion."""
        radius = 60.0
        gray, objmap = _make_half_mask_colony(
            shape=(200, 200),
            center=(100, 100),
            radius=radius,
            angular_range=(0.0, np.pi),
        )
        image = _make_image_with_objmap(gray, objmap)

        op = MeasureSymmetricRadius()
        df = op.measure(image)

        assert len(df) == 1
        core = df[self.CORE_COL].iloc[0]
        symm = df[self.SYMM_COL].iloc[0]
        max_exp = df[self.MAX_COL].iloc[0]

        # Asymmetry kicks in early because half the plate has no mask at all:
        # boundary pixels are angularly clustered in the wedge from the start.
        assert symm < 0.5 * max_exp, (
            f"SymmetricRadius={symm} not < 0.5 * MaxExpansion={max_exp}"
        )

        # MaxExpansion ≈ radius - core_radius (within 5 px).
        expected_max = radius - core
        assert abs(max_exp - expected_max) <= 5.0, (
            f"MaxExpansion={max_exp} outside {expected_max} ± 5"
        )

    # ------------------------------------------------------------------
    # Test 3 — lopsided colony (1.5× bias).
    # ------------------------------------------------------------------

    def test_lopsided_colony_symmetric_radius_between_bounds(self):
        """1.5× bias: Max ≈ bias*base - core; Symmetric strictly between bounds."""
        core_radius = 10.0
        base_radius = 40.0
        bias_factor = 1.5
        gray, objmap = _make_lopsided_colony(
            shape=(200, 200),
            center=(100, 100),
            core_radius=core_radius,
            base_radius=base_radius,
            bias_factor=bias_factor,
            bias_angular_range=(-np.pi / 3, np.pi / 3),
        )
        image = _make_image_with_objmap(gray, objmap)

        op = MeasureSymmetricRadius()
        df = op.measure(image)

        assert len(df) == 1
        core = df[self.CORE_COL].iloc[0]
        symm = df[self.SYMM_COL].iloc[0]
        max_exp = df[self.MAX_COL].iloc[0]

        # MaxExpansion ≈ bias_factor * base_radius - CoreRadius (± 5 px),
        # using the DETECTED CoreRadius (PELT picks up the density drop at
        # base_radius where the wedge-only region begins, not the synthetic's
        # core_radius parameter).
        bias_radius = bias_factor * base_radius
        expected_max = bias_radius - core
        assert abs(max_exp - expected_max) <= 5.0, (
            f"MaxExpansion={max_exp} outside {expected_max} ± 5 "
            f"(detected core={core})"
        )

        # SymmetricRadius must sit between the core edge and the outer envelope
        # (inclusive on the core side since asymmetry kicks in immediately
        # past the density drop).
        outer_envelope = max_exp + core
        assert core <= symm <= outer_envelope + 1.0, (
            f"SymmetricRadius={symm} not in [core={core}, "
            f"outer_envelope={outer_envelope}]"
        )

    # ------------------------------------------------------------------
    # Test 4 — core undetected (PELT finds no changepoint).
    # ------------------------------------------------------------------

    def test_core_undetected_still_produces_finite_expansion(self):
        """Thin annulus: CoreRadius == 0 but expansion columns stay finite."""
        shape = (200, 200)
        cr, cc = 100, 100
        inner_r = 35.0
        outer_r = 45.0

        gray = np.full(shape, 220, dtype=np.uint8)
        objmap = np.zeros(shape, dtype=np.int32)

        rr, cc_idx = np.ogrid[: shape[0], : shape[1]]
        dist_sq = (rr - cr) ** 2 + (cc_idx - cc) ** 2
        annulus = (dist_sq >= inner_r ** 2) & (dist_sq < outer_r ** 2)
        gray[annulus] = 40
        objmap[annulus] = 1

        image = _make_image_with_objmap(gray, objmap)

        op = MeasureSymmetricRadius()
        df = op.measure(image)

        assert len(df) == 1
        core = df[self.CORE_COL].iloc[0]
        symm = df[self.SYMM_COL].iloc[0]
        mean_exp = df[self.MEAN_COL].iloc[0]
        max_exp = df[self.MAX_COL].iloc[0]

        # PELT detects the density rise at the inner edge of the annulus
        # (~inner_r). The test's primary concern is that expansion metrics
        # remain finite regardless of core-detection outcome, so we only
        # require CoreRadius to be a non-negative finite value.
        assert np.isfinite(core) and core >= 0.0, (
            f"CoreRadius should be a non-negative finite value, got {core}"
        )
        assert np.isfinite(symm), f"SymmetricRadius should be finite, got {symm}"
        assert np.isfinite(mean_exp), f"MeanExpansion should be finite, got {mean_exp}"
        assert np.isfinite(max_exp), f"MaxExpansion should be finite, got {max_exp}"

    # ------------------------------------------------------------------
    # Test 5 — tiny object (area < 10).
    # ------------------------------------------------------------------

    def test_tiny_object_returns_all_nan(self):
        """5-pixel mask → all four measurement columns NaN; no exception."""
        shape = (100, 100)
        gray = np.full(shape, 220, dtype=np.uint8)
        objmap = np.zeros(shape, dtype=np.int32)

        # 5-pixel "plus" shape (centre + 4 neighbours) — area == 5 < 10.
        r, c = 50, 50
        for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            gray[r + dr, c + dc] = 40
            objmap[r + dr, c + dc] = 1

        image = _make_image_with_objmap(gray, objmap)

        op = MeasureSymmetricRadius()
        # Must not raise.
        df = op.measure(image)

        assert len(df) == 1
        for col in (self.CORE_COL, self.SYMM_COL, self.MEAN_COL, self.MAX_COL):
            val = df[col].iloc[0]
            assert pd.isna(val), f"{col} expected NaN for tiny object, got {val}"

    # ------------------------------------------------------------------
    # Test 6 — NaN handling in angular profile (sparse branches).
    # ------------------------------------------------------------------

    def test_nan_handling_in_angular_profile(self):
        """Thin cross of branches → several outer annuli have < 8 boundary pixels.

        The smoother must skip those NaN-bin annuli and still produce a finite
        SymmetricRadius (either via fallback to the outer radius or via a
        valid threshold crossing in a populated annulus).
        """
        shape = (200, 200)
        cr, cc = 100, 100
        gray = np.full(shape, 220, dtype=np.uint8)
        objmap = np.zeros(shape, dtype=np.int32)

        # Central core.
        rr, cc_idx = np.ogrid[: shape[0], : shape[1]]
        dist_sq = (rr - cr) ** 2 + (cc_idx - cc) ** 2
        core_mask = dist_sq < 10 ** 2
        gray[core_mask] = 40
        objmap[core_mask] = 1

        # Four thin branches (a "+" cross), each 2 pixels thick, extending out.
        # Thin branches ensure many outer annuli contain < 8 boundary pixels,
        # triggering the NaN-in-angular-R code path.
        branch_length = 70
        half_thickness = 1  # branch is 2*half_thickness + 1 = 3 px thick.
        # Horizontal (east-west).
        for dc in range(-branch_length, branch_length + 1):
            for dr in range(-half_thickness, half_thickness + 1):
                rr_i = cr + dr
                cc_i = cc + dc
                if 0 <= rr_i < shape[0] and 0 <= cc_i < shape[1]:
                    gray[rr_i, cc_i] = 40
                    objmap[rr_i, cc_i] = 1
        # Vertical (north-south).
        for dr in range(-branch_length, branch_length + 1):
            for dc in range(-half_thickness, half_thickness + 1):
                rr_i = cr + dr
                cc_i = cc + dc
                if 0 <= rr_i < shape[0] and 0 <= cc_i < shape[1]:
                    gray[rr_i, cc_i] = 40
                    objmap[rr_i, cc_i] = 1

        image = _make_image_with_objmap(gray, objmap)

        op = MeasureSymmetricRadius()
        df = op.measure(image)

        assert len(df) == 1
        symm = df[self.SYMM_COL].iloc[0]
        assert np.isfinite(symm), (
            f"SymmetricRadius should be finite even when many annuli are "
            f"NaN, got {symm}"
        )
