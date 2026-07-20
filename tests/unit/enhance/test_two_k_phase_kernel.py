# tests/unit/enhance/test_two_k_phase_kernel.py
import numpy as np

from phenotypic.enhance._two_k_phase_kernel import two_k_phase


def _synthetic_branches():
    # a bright ring (branches) around a dark core, on a mid-gray field with a faint outlier speck
    img = np.full((80, 80), 0.4, dtype=np.float32)
    yy, xx = np.ogrid[:80, :80]
    ring = ((yy - 40) ** 2 + (xx - 40) ** 2)
    img[(ring > 18 ** 2) & (ring < 24 ** 2)] = 0.95   # edge ring -> strong PCT
    img[70:72, 6:8] = 0.7                              # isolated faint speck (loose-only, no seed)
    return img


def test_two_k_phase_returns_gated_response_and_loose_result():
    img = _synthetic_branches()
    gated, loose = two_k_phase(
        img, k_strict=6.0, k_loose=4.5, seed_thresh="otsu",
        cand_thresh="triangle", n_orient=8, min_wavelength=5.0,
    )
    assert gated.shape == img.shape
    # loose result exposes the cost-surface arrays, all finite & same-shaped
    for arr in (loose.pc_sum, loose.M, loose.m, loose.orientation):
        assert arr.shape == img.shape
        assert np.all(np.isfinite(arr))
    # gated response is continuous (not just 0/1) where branches are confirmed
    nz = gated[gated > 0]
    assert nz.size > 0
    assert np.unique(nz).size > 2  # magnitudes, not a binary mask


def test_two_k_phase_rejects_isolated_loose_only_agar():
    img = _synthetic_branches()
    gated, _ = two_k_phase(
        img, k_strict=6.0, k_loose=4.5, seed_thresh="otsu",
        cand_thresh="triangle", n_orient=8, min_wavelength=5.0,
    )
    # the isolated faint speck has no strict seed -> rejected (stays 0)
    assert gated[70, 6] == 0.0


def test_two_k_phase_otsu_otsu_admits_no_more_than_hysteresis():
    # mutation guard: using otsu on candidates (instead of triangle) must not
    # recover MORE branch pixels than the loose-triangle hysteresis (it recovers fewer).
    img = _synthetic_branches()
    tri, _ = two_k_phase(img, k_strict=6.0, k_loose=4.5, seed_thresh="otsu",
                         cand_thresh="triangle", n_orient=8, min_wavelength=5.0)
    ott, _ = two_k_phase(img, k_strict=6.0, k_loose=4.5, seed_thresh="otsu",
                         cand_thresh="otsu", n_orient=8, min_wavelength=5.0)
    assert (tri > 0).sum() >= (ott > 0).sum()
