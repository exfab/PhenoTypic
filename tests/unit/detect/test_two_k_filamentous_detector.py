import numpy as np
import pytest

from phenotypic.data import load_synth_filamentous_plate
from phenotypic.detect import TwoKFilamentousDetector, OtsuDetector


def test_defaults_and_construction():
    d = TwoKFilamentousDetector()
    assert d.k_strict == 6.0 and d.k_loose == 4.5
    # scene-derived scalars populate after validation
    assert d.tile_size is not None and d.mad_window is not None and d.mad_window % 2 == 1


def test_k_loose_must_be_below_k_strict():
    from pydantic import ValidationError
    # inverting the hysteresis scales (loose >= strict) is rejected by the validator
    with pytest.raises(ValidationError):
        TwoKFilamentousDetector(k_loose=6.0, k_strict=6.0)
    with pytest.raises(ValidationError):
        TwoKFilamentousDetector(k_loose=7.0, k_strict=6.0)


def test_end_to_end_labels_colonies():
    image = load_synth_filamentous_plate().copy()
    d = TwoKFilamentousDetector(center_detector=OtsuDetector(ignore_zeros=True))
    result = d.apply(image, inplace=False)
    assert result.objmap[:].max() > 0
    assert result.objmask[:].sum() > 0


def test_reconnection_reduces_fragments():
    # mutation-style: disabling reconnection (tile smaller than any gap) leaves >= as many
    # connected components as the full run. Full run should not have MORE fragments.
    image = load_synth_filamentous_plate().copy()
    from skimage.measure import label
    full = TwoKFilamentousDetector(center_detector=OtsuDetector(ignore_zeros=True))
    r_full = full.apply(image.copy(), inplace=False)
    n_full = label(r_full.objmap[:] > 0).max()
    no_recon = TwoKFilamentousDetector(
        center_detector=OtsuDetector(ignore_zeros=True), max_gap_length=1, frag_reach_px=1,
    )
    r_no = no_recon.apply(image.copy(), inplace=False)
    n_no = label(r_no.objmap[:] > 0).max()
    assert n_full <= n_no


def test_serialization_round_trip():
    d = TwoKFilamentousDetector(k_loose=4.0, max_colony_radius_px=200.0)
    payload = d.model_dump_json()
    restored = TwoKFilamentousDetector.model_validate_json(payload)
    assert restored.k_loose == 4.0
    assert restored.max_colony_radius_px == 200.0
    assert restored.tile_size == d.tile_size          # derived scalars survive


def test_cost_surface_uses_loose_result_no_extra_pct(monkeypatch):
    # two_k_phase must be called exactly once per _operate (the only phase-congruency
    # work); the loose result it returns feeds the cost surface — no third PCT pass.
    import phenotypic.detect._two_k_filamentous_detector as mod
    calls = {"n": 0}
    real = mod.two_k_phase

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mod, "two_k_phase", counting)
    from phenotypic.data import load_synth_filamentous_plate
    from phenotypic.detect import OtsuDetector
    image = load_synth_filamentous_plate().copy()
    TwoKFilamentousDetector(center_detector=OtsuDetector(ignore_zeros=True)).apply(image, inplace=False)
    assert calls["n"] == 1


from phenotypic import GridImage
from phenotypic.detect import ManualGridPointDetector


def _plate_two_colonies_and_a_stray():
    """1x2 grid: two colonies (solid core + branch tendril) on known wells, plus one bright
    blob far from both wells. Returns (GridImage, well0_rc, well1_rc, stray_rc, stray_bbox).

    ``stray_bbox`` is a (row_slice, col_slice) tight around the stray blob but clear of both
    colonies, for a region assertion that no stray pixel is labeled. (A single-point check at the
    blob centre is not enough: a solid blob's centre is a phase-congruency hole and reads 0
    regardless, so the region covers the rim where PCT actually responds.)
    """
    H, W = 200, 400
    g = np.full((H, W), 60, dtype=np.uint8)
    yy, xx = np.ogrid[:H, :W]

    def disk(cy, cx, r, val):
        g[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = val

    well0, well1 = (100, 100), (100, 300)
    disk(*well0, 22, 235); g[98:103, 100:150] = 215        # colony 0: core + tendril
    disk(*well1, 22, 235); g[98:103, 250:300] = 215        # colony 1: core + tendril
    stray_rc = (32, 200)
    disk(*stray_rc, 18, 240)                                # stray blob, far from both wells
    stray_bbox = (slice(10, 55), slice(178, 222))          # tight around the stray, clear of colonies
    rgb = np.repeat(g[..., None], 3, axis=2)
    return GridImage(rgb, nrows=1, ncols=2), well0, well1, stray_rc, stray_bbox


def test_final_objmap_excludes_objects_not_overlapping_centers():
    """End-to-end: the final objmap keeps only center-overlapping objects (never "all objects").

    This is the requirement the detector must satisfy. It is enforced jointly by the
    ``filter_mask_by_overlap`` pre-filter AND the grid-Voronoi marker-drop (a marker-less
    connected component is zeroed by ``partition_by_grid_voronoi``), so this integration test
    guards the *behavior* rather than isolating the filter — the filter itself is unit-tested in
    ``tests/unit/sdk_/reconnect/test_colony_labeling.py``.
    """
    img, well0, well1, stray_rc, stray_bbox = _plate_two_colonies_and_a_stray()
    detector = TwoKFilamentousDetector(
        center_detector=ManualGridPointDetector(coord1=well0, coord2=well1,
                                                shape="disk", width=40),
    )
    objmap = np.asarray(detector.apply(img, inplace=False).objmap[:])
    assert objmap[well0] > 0 and objmap[well1] > 0     # both colonies are labeled
    assert objmap[stray_rc] == 0                       # stray object is not labeled (overlap-keep)
    # No pixel anywhere in the stray blob may be labeled — it overlaps no center-fill location.
    assert objmap[stray_bbox].max() == 0
