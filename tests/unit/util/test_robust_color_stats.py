import numpy as np
import pytest
from phenotypic.util._robust_color_stats import (
    cone_to_hsv,
    delta_e2000_spread,
    hsv_to_cone,
    lab_to_srgb_hex,
    medoid_ciede2000,
    robust_color_center,
)


def test_robust_center_symmetric_cloud():
    pts = np.array([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    assert np.allclose(robust_color_center(pts), [0.0, 0.0, 0.0], atol=1e-3)


def test_robust_center_resists_single_outlier():
    cluster = np.tile([50.0, 10.0, 20.0], (99, 1))
    pts = np.vstack([cluster, [50_000.0, 50_000.0, 50_000.0]])
    assert np.allclose(robust_color_center(pts), [50.0, 10.0, 20.0], atol=1.0)


def test_robust_center_single_point_returns_it():
    assert np.allclose(robust_color_center(np.array([[3.0, 4.0, 5.0]])), [3.0, 4.0, 5.0])


def test_robust_center_identical_points():
    assert np.allclose(robust_color_center(np.tile([7.0, 7.0, 7.0], (5, 1))), [7.0, 7.0, 7.0])


def test_robust_center_empty_returns_nan():
    out = robust_color_center(np.empty((0, 3)))
    assert out.shape == (3,) and np.isnan(out).all()


def test_medoid_is_an_actual_input_pixel():
    rng = np.random.default_rng(0)
    lab = rng.uniform([20, -10, -10], [80, 40, 40], size=(50, 3))
    center, deltas = medoid_ciede2000(lab, max_pixels=1000, seed=0)
    assert any(np.allclose(center, p) for p in lab)  # center IS a real pixel
    assert deltas.shape == (50,)
    assert np.all(deltas >= 0)


def test_medoid_central_for_one_outlier():
    cluster = np.tile([50.0, 10.0, 20.0], (40, 1))
    lab = np.vstack([cluster, [10.0, -50.0, 60.0]])
    center, _ = medoid_ciede2000(lab, max_pixels=1000, seed=0)
    assert np.allclose(center, [50.0, 10.0, 20.0])


def test_medoid_subsample_is_reproducible():
    rng = np.random.default_rng(1)
    lab = rng.uniform([20, -10, -10], [80, 40, 40], size=(5000, 3))
    c1, d1 = medoid_ciede2000(lab, max_pixels=500, seed=7)
    c2, d2 = medoid_ciede2000(lab, max_pixels=500, seed=7)
    assert np.allclose(c1, c2)
    assert d1.shape == (5000,)  # spread uses ALL pixels, not the subsample


def test_medoid_chunk_size_is_invariant():
    # Chunked row-sum selection must give the identical medoid + deltas
    # regardless of chunk_size (it only bounds peak memory, not the result).
    rng = np.random.default_rng(2)
    lab = rng.uniform([20, -10, -10], [80, 40, 40], size=(300, 3))
    c_full, d_full = medoid_ciede2000(lab, max_pixels=1000, seed=0, chunk_size=10_000)
    for cs in (1, 7, 64, 299):
        c, d = medoid_ciede2000(lab, max_pixels=1000, seed=0, chunk_size=cs)
        assert np.array_equal(c, c_full)  # bit-identical center
        assert np.array_equal(d, d_full)  # bit-identical spread


def test_medoid_single_and_empty():
    c, d = medoid_ciede2000(np.array([[40.0, 5.0, -5.0]]))
    assert np.allclose(c, [40.0, 5.0, -5.0]) and np.allclose(d, [0.0])
    c0, d0 = medoid_ciede2000(np.empty((0, 3)))
    assert np.isnan(c0).all() and d0.size == 0


def test_delta_e2000_spread_values():
    deltas = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    med, mean, p95 = delta_e2000_spread(deltas)
    assert med == 2.0
    assert mean == pytest.approx(21.2)
    assert p95 == pytest.approx(np.percentile(deltas, 95))


def test_delta_e2000_spread_empty_is_nan():
    med, mean, p95 = delta_e2000_spread(np.array([]))
    assert np.isnan(med) and np.isnan(mean) and np.isnan(p95)


def test_cone_roundtrip_recovers_hsv():
    hsv = np.array([[0.0, 1.0, 1.0], [0.25, 0.5, 0.8], [0.99, 0.3, 0.6]])
    back = cone_to_hsv(hsv_to_cone(hsv))
    assert np.allclose(back, hsv, atol=1e-6)


def test_cone_collapses_unreliable_hue_at_zero_saturation():
    # Two grays with different (meaningless) hues map to the same cone point.
    a = hsv_to_cone(np.array([0.1, 0.0, 0.5]))
    b = hsv_to_cone(np.array([0.7, 0.0, 0.5]))
    assert np.allclose(a, b)


def test_cone_handles_hue_wraparound():
    # Hues near 0 and near 1 are adjacent, not opposite.
    near_zero = hsv_to_cone(np.array([0.001, 1.0, 1.0]))
    near_one = hsv_to_cone(np.array([0.999, 1.0, 1.0]))
    assert np.linalg.norm(near_zero - near_one) < 0.05


def test_lab_to_srgb_hex_format():
    h = lab_to_srgb_hex(np.array([60.0, 20.0, 30.0]))
    assert h == "#c1825d"


def test_lab_to_srgb_hex_nan_returns_empty():
    assert lab_to_srgb_hex(np.array([np.nan, 0.0, 0.0])) == ""


def test_candidate_medoid_is_exported_from_util():
    """One implementation, shared: util exports it and the checker path re-imports it."""
    from phenotypic.correction._color_correction import _checker_measure
    from phenotypic.util import MedoidResult, candidate_medoid

    assert candidate_medoid is _checker_measure.candidate_medoid
    assert MedoidResult is _checker_measure.MedoidResult
    result = candidate_medoid(np.array([[50.0, 0.0, 0.0], [52.0, 1.0, -1.0], [51.0, 0.5, 0.0]]))
    assert isinstance(result, MedoidResult)


def _bimodal_lab_cloud() -> np.ndarray:
    """900 px at one colour and 300 px 30 a* away: exercises the widened pass too."""
    rng = np.random.default_rng(11)
    major = np.array([50.0, 0.0, 0.0]) + rng.normal(0, 0.6, (900, 3))
    minor = np.array([50.0, 30.0, 0.0]) + rng.normal(0, 0.6, (300, 3))
    return np.vstack([major, minor])


@pytest.mark.parametrize(
    "lab",
    [
        _bimodal_lab_cloud(),
        np.array([55.0, 5.0, 20.0]) + np.random.default_rng(3).normal(0, 2.0, (2000, 3)),
    ],
    ids=["bimodal-widened", "unimodal"],
)
def test_candidate_medoid_is_chunk_size_invariant(lab):
    """Chunking bounds memory and nothing else: every field is bit-identical."""
    from phenotypic.util import candidate_medoid

    results = {cs: candidate_medoid(lab, chunk_size=cs) for cs in (1, 7, 64, None)}
    reference = results[64]
    for cs, result in results.items():
        assert result.index == reference.index, cs
        assert result.rank == reference.rank, cs
        assert result.widened == reference.widened, cs
        assert result.total_delta_e == reference.total_delta_e, cs  # exact, not approx
        assert np.array_equal(result.lab, reference.lab), cs


def test_candidate_medoid_peak_memory_is_bounded():
    """Peak memory stays near the byte budget however large the object is.

    Before the budget, ``chunk_size=64`` scored 64 candidates against all N
    pixels at once: ~265 B per candidate-pixel pair in colour's ΔE2000
    temporaries, i.e. ~1.7 GB at N = 100 000.

    Allowance above the budget, from the non-chunk allocations: the (N, 3)
    ``points - seed`` difference (24 B/px), its norms and their argsort (16
    B/px) -- measured together with the Weiszfeld seed at ~40 B/px (peak minus
    the pair term at ``chunk_size=1``), i.e. ~4 MB here. Measured peak after
    the budget: ~256 MiB. The scoring block is sized with a bytes-per-pair
    constant measured on macOS; 2x the budget leaves room for that constant to
    run higher on another platform or numpy build while still failing the old
    ~1.6 GiB peak by a factor of 3.
    """
    import tracemalloc

    from phenotypic.util import candidate_medoid
    from phenotypic.util._robust_color_stats import MEDOID_MEMORY_BUDGET_BYTES

    n = 100_000
    lab = np.array([55.0, 5.0, 20.0]) + np.random.default_rng(0).normal(0, 2.0, (n, 3))
    candidate_medoid(lab[:10])  # import colour outside the traced window

    tracemalloc.start()
    try:
        candidate_medoid(lab)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    bound = 2 * MEDOID_MEMORY_BUDGET_BYTES
    assert peak < bound, f"peak {peak / 2**20:.0f} MiB exceeds {bound / 2**20:.0f} MiB"
