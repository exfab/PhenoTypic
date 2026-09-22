from __future__ import annotations

import numpy as np
import pytest

from phenotypic.correction._color_correction._checker_measure import (
    candidate_medoid,
    clipped_fraction,
    extract_patch,
    impurity,
    measure_tile,
    robust_shift,
)


# ---------------------------------------------------------------------------
# Independent reference: the exhaustive medoid, implemented here on purpose.
# Importing the module's own search would make this test a tautology.
# ---------------------------------------------------------------------------
def _exhaustive_medoid_index(lab: np.ndarray, chunk: int = 128) -> int:
    import colour

    n = lab.shape[0]
    totals = np.empty(n)
    for start in range(0, n, chunk):
        block = lab[start : start + chunk]
        totals[start : start + chunk] = np.asarray(
                colour.difference.delta_E_CIE2000(block[:, None, :], lab[None, :, :])
        ).sum(axis=1)
    return int(totals.argmin())


def _tile_cloud(seed: int, n: int = 1200) -> np.ndarray:
    """A checker-tile-like Lab cloud: unimodal, sensor-noise dominated."""
    rng = np.random.default_rng(seed)
    centre = np.array(
            [rng.uniform(20, 85), rng.uniform(-45, 60), rng.uniform(-55, 65)]
    )
    sd = np.array([rng.uniform(1.5, 4), rng.uniform(1, 3), rng.uniform(1, 3)])
    return centre + rng.normal(0, 1, (n, 3)) * sd


# ---------------------------------------------------------------------------
# The medoid contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_candidate_medoid_equals_the_exhaustive_medoid(seed: int) -> None:
    """The restriction is a cost optimisation, not an approximation."""
    lab = _tile_cloud(seed)
    result = candidate_medoid(lab, k=256)

    assert result.index == _exhaustive_medoid_index(lab)
    assert result.rank < 0.8 * 256
    assert result.widened is False


def test_candidate_medoid_is_deterministic() -> None:
    """No RNG anywhere on this path, unlike the seeded-subsample estimator."""
    lab = _tile_cloud(0)
    first, second = candidate_medoid(lab), candidate_medoid(lab)

    assert first.index == second.index
    assert np.array_equal(first.lab, second.lab)


def test_candidate_medoid_returns_a_real_pixel() -> None:
    lab = _tile_cloud(3)
    result = candidate_medoid(lab)

    assert np.array_equal(result.lab, lab[result.index])


def _bimodal_cloud() -> np.ndarray:
    """Two colour lobes: the medoid is not near the geometric median."""
    rng = np.random.default_rng(11)
    major = np.array([50.0, 0.0, 0.0]) + rng.normal(0, 0.6, (900, 3))
    minor = np.array([50.0, 30.0, 0.0]) + rng.normal(0, 0.6, (300, 3))
    return np.vstack([major, minor])


def test_candidate_medoid_widens_and_recovers_on_a_bimodal_cloud() -> None:
    """At the default candidate count the edge guard fires and fixes the answer."""
    lab = _bimodal_cloud()

    result = candidate_medoid(lab, k=256)

    assert result.widened is True
    assert result.index == _exhaustive_medoid_index(lab)


def test_candidate_medoid_guard_does_not_rescue_a_tiny_candidate_set() -> None:
    """Documents the guard's reach, so nobody lowers ``candidates`` expecting it.

    With only 64 candidates on the same bimodal cloud the winner sits at rank
    3 -- well inside the set, so nothing looks wrong -- yet it is the wrong
    pixel. The guard is a safety net at the default count, not at any count.
    """
    lab = _bimodal_cloud()

    small = candidate_medoid(lab, k=64)

    assert small.widened is False
    assert small.rank < 0.8 * 64
    assert small.index != _exhaustive_medoid_index(lab)


def test_candidate_medoid_returns_a_result_rather_than_crashing_when_still_at_edge()\
        -> None:
    """The final attempt always returns; falling through would be a crash."""
    lab = _bimodal_cloud()

    result = candidate_medoid(lab, k=2)

    assert 0 <= result.index < lab.shape[0]


@pytest.mark.parametrize("bad", [np.zeros((4,)), np.zeros((4, 2)), np.zeros((2, 2, 3))])
def test_candidate_medoid_rejects_non_lab_input(bad) -> None:
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        candidate_medoid(bad)


def test_candidate_medoid_handles_empty_and_single_pixel_tiles() -> None:
    empty = candidate_medoid(np.zeros((0, 3)))
    assert empty.index == -1 and np.isnan(empty.lab).all()

    lone = candidate_medoid(np.array([[50.0, 1.0, -2.0]]))
    assert lone.index == 0 and lone.total_delta_e == 0.0


# ---------------------------------------------------------------------------
# Box extraction
# ---------------------------------------------------------------------------
def test_extract_patch_clips_to_the_array_bounds() -> None:
    arr = np.zeros((100, 50, 3))
    assert extract_patch(arr, (-10.0, 20.0, 5.0, 500.0)).shape == (20, 45, 3)


def test_extract_patch_returns_empty_when_the_box_misses() -> None:
    arr = np.zeros((10, 10, 3))
    assert extract_patch(arr, (50.0, 60.0, 0.0, 5.0)).size == 0


# ---------------------------------------------------------------------------
# Contamination statistics
# ---------------------------------------------------------------------------
def _clean_tile(h: int = 40, w: int = 40) -> np.ndarray:
    rng = np.random.default_rng(5)
    return np.array([55.0, 10.0, -20.0]) + rng.normal(0, 0.4, (h, w, 3))


def test_impurity_is_near_zero_on_a_clean_tile() -> None:
    assert impurity(_clean_tile()) < 0.01


def test_impurity_rises_with_an_occluder_crossing_the_tile() -> None:
    tile = _clean_tile()
    tile[:, :10] = np.array([95.0, 0.0, 0.0])  # a bright object over a quarter

    assert impurity(tile) > 0.2


def _contaminated(sd: float, fraction: float, width: int = 40) -> np.ndarray:
    """A tile with *fraction* of its columns covered by a bright object."""
    rng = np.random.default_rng(5)
    tile = np.array([55.0, 10.0, -20.0]) + rng.normal(0, sd, (width, width, 3))
    tile[:, : int(fraction * width)] = np.array([95.0, 0.0, 0.0])
    return tile


def test_impurity_and_robust_shift_answer_different_questions() -> None:
    """Impurity says something is there; robust shift says it moved the answer.

    This is the pair the QC gate needs, and why only the second rejects a
    band: a tenth of the tile can be covered while the per-channel median
    barely moves. Matches the observed field case of 8.7 % contaminated
    pixels shifting the median by 0.57 delta-E.
    """
    tile = _contaminated(sd=3.0, fraction=0.10)

    assert impurity(tile) >= 0.10
    assert robust_shift(tile) < 0.5


def test_robust_shift_grows_with_the_contaminated_fraction() -> None:
    """Below the point where the median flips lobes, more cover moves it more."""
    shifts = [robust_shift(_contaminated(3.0, f)) for f in (0.05, 0.10, 0.20, 0.35)]

    assert shifts == sorted(shifts)
    assert shifts[0] < shifts[-1]


def test_robust_shift_collapses_once_contamination_takes_the_median() -> None:
    """A known blind spot, asserted so it cannot regress unnoticed.

    Past ~45 % cover the median sits inside the contaminant and the trimmed
    median follows it, so the statistic reads ~0. Two-lobe contamination
    therefore never drives it to the 1.5 delta-E gate; that limit was
    calibrated on real graded contamination (a barcode sticker, 2.5-18
    delta-E), which these synthetic tiles do not reproduce. Occlusion at this
    level is caught by impurity instead.
    """
    heavy = _contaminated(sd=3.0, fraction=0.55)

    assert robust_shift(heavy) < 0.1
    assert impurity(heavy) > 0.4


def test_clipped_fraction_catches_a_channel_at_the_sensor_floor() -> None:
    rgb = np.full((20, 20, 3), 0.5)
    rgb[:10, :, 0] = 0.0

    assert clipped_fraction(rgb) == pytest.approx(0.5)


def test_clipped_fraction_is_blind_to_contamination_and_vice_versa() -> None:
    """The three statistics are independent faults, which is why all are kept."""
    clean_but_clipped = np.full((20, 20, 3), 0.5)
    clean_but_clipped[..., 2] = 1.0

    assert clipped_fraction(clean_but_clipped) == pytest.approx(1.0)
    lab_of_it = np.zeros((20, 20, 3))  # uniform -> no spread whatever the colour
    assert impurity(lab_of_it) == pytest.approx(0.0)


@pytest.mark.parametrize("fn", [impurity, robust_shift, clipped_fraction])
def test_statistics_return_nan_for_an_empty_patch(fn) -> None:
    assert np.isnan(fn(np.zeros((0, 0, 3))))


def test_robust_shift_returns_nan_for_a_tiny_patch() -> None:
    assert np.isnan(robust_shift(np.zeros((3, 3, 3))))


# ---------------------------------------------------------------------------
# The assembled measurement
# ---------------------------------------------------------------------------
def test_measure_tile_reports_the_medoid_pixel_s_own_srgb() -> None:
    """The fit consumes sRGB, and the medoid is a real pixel, so its own
    sRGB value goes through untouched -- no Lab round-trip."""
    rng = np.random.default_rng(2)
    srgb = np.clip(0.4 + rng.normal(0, 0.01, (30, 30, 3)), 0, 1)
    lab = srgb * np.array([100.0, 1.0, 1.0])  # monotone stand-in, shape-preserving

    measured = measure_tile(lab, srgb, row=2, col=1, roi_index=0)

    flat_srgb = srgb.reshape(-1, 3)
    flat_lab = lab.reshape(-1, 3)
    winner = candidate_medoid(flat_lab).index
    assert measured.srgb == pytest.approx(tuple(flat_srgb[winner]))
    assert measured.n_pixels == 900
    assert (measured.row, measured.col, measured.roi_index) == (2, 1, 0)


def test_measure_tile_spread_is_within_tile_consistency_not_accuracy() -> None:
    tight = _clean_tile()
    loose = _clean_tile() * 1.0
    rng = np.random.default_rng(7)
    loose += rng.normal(0, 3.0, loose.shape)
    srgb = np.full_like(tight, 0.5)

    a = measure_tile(tight, srgb, row=0, col=0, roi_index=0)
    b = measure_tile(loose, srgb, row=0, col=0, roi_index=0)

    assert b.spread_delta_e > a.spread_delta_e


def test_measure_tile_rejects_mismatched_patches() -> None:
    with pytest.raises(ValueError, match="same pixels"):
        measure_tile(
                np.zeros((4, 4, 3)), np.zeros((5, 5, 3)),
                row=0, col=0, roi_index=0,
        )
