"""Behaviour tests for MeasureOrientationZones."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic.data import load_synth_yeast_plate, load_synth_filamentous_plate
from phenotypic.schema import ORIENTATION_ZONES
from phenotypic.measure import MeasureOrientationZones
from phenotypic.measure._measure_orientation_zones import (
    aggregate_orientation,
    zone_selector,
)


def test_aggregate_parallel_field_high_R_zero_turning():
    n = 40
    phi = np.full((n, n), 0.3)          # constant orientation
    coh = np.ones((n, n))
    grad = np.zeros((n, n))
    sel = np.ones((n, n), dtype=bool)
    R, turning, coh_mean = aggregate_orientation(phi, coh, grad, sel, eps=1e-9)
    assert R == pytest.approx(1.0, abs=1e-6)
    assert turning == pytest.approx(0.0, abs=1e-9)
    assert coh_mean == pytest.approx(1.0, abs=1e-9)


def test_aggregate_zero_coherence_is_nan():
    n = 20
    out = aggregate_orientation(
        np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n)),
        np.ones((n, n), dtype=bool), eps=1e-9,
    )
    assert all(np.isnan(v) for v in out)


def test_aggregate_empty_selector_is_nan():
    n = 20
    out = aggregate_orientation(
        np.zeros((n, n)), np.ones((n, n)), np.zeros((n, n)),
        np.zeros((n, n), dtype=bool), eps=1e-9,
    )
    assert all(np.isnan(v) for v in out)


def test_zone_selector_radial_vs_mask():
    n = 21
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.hypot(yy - c, xx - c)
    obj = dist < 6                      # imperfect mask: only inner disk
    radial = zone_selector(dist, 0.0, 8.0, obj, "Radial")
    masked = zone_selector(dist, 0.0, 8.0, obj, "Mask")
    assert radial.sum() > masked.sum()          # mask carves out the ring 6..8
    assert np.array_equal(masked, radial & obj)


def test_zone_restriction_inner_vs_outer_orientation():
    # Inner disk oriented one way, outer ring another -> per-zone R directions differ.
    n = 61
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.hypot(yy - c, xx - c)
    phi = np.where(dist < 15, 0.0, np.pi / 2 - 1e-6)
    coh = np.ones((n, n))
    grad = np.zeros((n, n))
    obj = np.ones((n, n), dtype=bool)
    inner = aggregate_orientation(phi, coh, grad, zone_selector(dist, 0.0, 15.0, obj, "Radial"))
    outer = aggregate_orientation(phi, coh, grad, zone_selector(dist, 20.0, 28.0, obj, "Radial"))
    assert inner[0] == pytest.approx(1.0, abs=1e-6)   # each zone internally aligned
    assert outer[0] == pytest.approx(1.0, abs=1e-6)


def test_measure_returns_18_columns_one_row_per_object():
    image = load_synth_filamentous_plate()
    df = MeasureOrientationZones().measure(image)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == image.num_objects
    for h in ORIENTATION_ZONES.get_headers():
        assert h in df.columns
    # R and coherence within [0,1] where finite
    for col in df.columns:
        if col.startswith(("OrientZones_Concentration", "OrientZones_Coherence")):
            vals = df[col].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            assert np.all((finite >= -1e-9) & (finite <= 1 + 1e-9))


def test_rotation_invariance_of_R_magnitude_and_turning():
    # A single synthetic tile rotated 90 deg: R magnitude and turning invariant.
    from phenotypic.measure._measure_orientation_zones import (
        aggregate_orientation, zone_selector,
    )
    n = 61
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.hypot(yy - c, xx - c)
    base = np.sin(2 * np.pi * xx / 7.0)
    from phenotypic.util._orientation_field import orientation_field
    obj = np.ones((n, n), dtype=bool)

    def metrics(field):
        phi, coh, grad = orientation_field(field, 1.5, 4.0)
        sel = zone_selector(dist, 0.0, 20.0, obj, "Radial")
        return aggregate_orientation(phi, coh, grad, sel)

    R0, t0, _ = metrics(base)
    R90, t90, _ = metrics(np.rot90(base))
    assert R0 == pytest.approx(R90, abs=0.05)
    assert t0 == pytest.approx(t90, abs=0.05)


def test_tiny_objects_are_all_nan():
    image = load_synth_yeast_plate()
    df = MeasureOrientationZones().measure(image)
    # every row has the full column set; NaN allowed, no exceptions
    assert set(ORIENTATION_ZONES.get_headers()).issubset(df.columns)


def test_measure_cache_is_compact():
    # Guard against memory bloat: after measure(), the per-object cache must hold
    # NO full-res arrays and NO seg dataclass — only scalars + the block quiver.
    image = load_synth_filamentous_plate()
    op = MeasureOrientationZones()
    op.measure(image)
    assert op._cache, "cache should be populated"
    forbidden = {"tile", "phi", "coherence", "grad_phi", "dist_map", "seg"}
    for rec in op._cache.values():
        assert forbidden.isdisjoint(rec), f"full-res leaked: {forbidden & set(rec)}"
        assert "quiver" in rec
        for v in rec.values():
            if isinstance(v, np.ndarray):
                assert v.size <= 4096, "only the block-resolution quiver may be cached"
        # the block quiver must be far smaller than a full tile
        rows, cols, pb, cb = rec["quiver"]
        assert pb.shape == cb.shape and pb.size <= 4096


def test_inspect_builds_figure():
    import plotly.graph_objects as go
    image = load_synth_filamentous_plate()
    op = MeasureOrientationZones()
    op.measure(image)
    fig = op.inspect(image)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    fig_save = op.inspect(image, for_save=True)
    assert isinstance(fig_save, go.Figure)


def test_dashboard_builds_composed_figure():
    import plotly.graph_objects as go
    image = load_synth_filamentous_plate()
    op = MeasureOrientationZones()
    op.measure(image)
    fig = op.dashboard(image, show=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    # the go.Table summary panel must survive composition (base composer can't
    # host it — proves the custom dash() override with per-row specs works).
    assert any(getattr(tr, "type", None) == "table" for tr in fig.data)
    # coherence heatmap present too
    assert any(getattr(tr, "type", None) == "heatmap" for tr in fig.data)


def test_non_grid_image_uses_expanded_crop_fallback():
    # A grid section extracted via image.grid[idx] is a plain Image with no
    # .grid accessor — the ONLY way to exercise _resolve_tile's expanded-crop
    # fallback (all repo fixtures are GridImages). Spec §5: non-grid → no error.
    image = load_synth_filamentous_plate()
    section = image.grid[18]
    assert not hasattr(section, "grid"), "grid section should be a plain Image"
    df = MeasureOrientationZones().measure(section)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == section.num_objects
    assert set(ORIENTATION_ZONES.get_headers()).issubset(df.columns)
    # the object is a real colony → the fallback path still yields a finite R
    r = df["OrientZones_Concentration-Radial-Overall"].to_numpy(float)
    assert np.isfinite(r).any()


def test_collapsed_zones_yield_all_nan():
    # zones_computed==False (collapsed symmetric envelope) → all 18 metrics NaN,
    # including Overall once symmetric_radius==0 (empty selector). Spec §5. This
    # branch is never hit by the fixtures, so drive _fill_metrics directly using
    # real per-object arrays with a mutated (collapsed) segmentation.
    image = load_synth_filamentous_plate()
    op = MeasureOrientationZones()
    props, label2section = op._prep(image)
    _prop, seg, obj_mask, phi, coh, grad, dist_map, _centre = next(
        op._iter_object_fields(image, props, label2section)
    )
    seg.zones_computed = False
    seg.symmetric_radius = 0.0
    row: dict = {}
    op._fill_metrics(row, seg, obj_mask, phi, coh, grad, dist_map)
    assert len(row) == 18
    assert all(np.isnan(v) for v in row.values())


def test_radial_and_mask_variants_diverge_on_real_plate():
    # The Mask variant exists so the imperfect mask's distortion can be *seen*
    # (spec §1/§2). In the sparse ring the mask carves holes the Radial variant
    # keeps, so the two concentration reads must differ for at least some objects.
    image = load_synth_filamentous_plate()
    df = MeasureOrientationZones().measure(image)
    rad = df["OrientZones_Concentration-Radial-Sparse"].to_numpy(float)
    msk = df["OrientZones_Concentration-Mask-Sparse"].to_numpy(float)
    both = np.isfinite(rad) & np.isfinite(msk)
    assert both.any(), "need objects with a finite sparse ring in both variants"
    assert np.nanmax(np.abs(rad[both] - msk[both])) > 1e-6
