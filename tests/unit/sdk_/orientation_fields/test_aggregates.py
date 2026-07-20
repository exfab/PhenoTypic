"""Tests for robust literal-crossing zone aggregates."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.sdk_.orientation_fields import (
    LiteralCrossingRingProfile,
    aggregate_literal_crossing_zone,
)


def _profile(
    changes_degrees: list[float],
    *,
    run_ids: list[int] | None = None,
    crossing_count: int = 3,
) -> LiteralCrossingRingProfile:
    """Build a controlled cumulative profile at eight-pixel spacing."""
    changes = np.radians(np.asarray(changes_degrees, dtype=float))
    supported = np.isfinite(changes)
    if run_ids is None:
        run_id = np.where(supported, 0, -1)
    else:
        run_id = np.asarray(run_ids, dtype=np.int64)
    return LiteralCrossingRingProfile(
        radii=8.0 + np.arange(changes.size, dtype=float) * 8.0,
        consensus_tilt=np.where(supported, 0.0, np.nan),
        resultant=np.where(supported, 0.8, np.nan),
        crossing_count=np.where(supported, crossing_count, 0),
        contiguous_change=changes,
        run_id=run_id,
    )


def test_linear_profile_recovers_primary_metrics_and_units() -> None:
    """A one-degree-per-pixel ramp should recover its exact robust rate."""
    profile = _profile([0, 8, 16, 24, 32, 40, 48, 56, 64, 72])

    metrics = aggregate_literal_crossing_zone(profile, 8.0, 88.0)

    assert np.degrees(metrics.sustained_peak) == pytest.approx(64.0)
    assert np.degrees(metrics.net_rotation) == pytest.approx(64.0)
    assert np.degrees(metrics.rotation_rate) == pytest.approx(1.0)
    assert metrics.consistency == pytest.approx(1.0)
    assert np.degrees(metrics.raw_peak) == pytest.approx(72.0)
    assert np.degrees(metrics.median_magnitude) == pytest.approx(36.0)
    assert np.degrees(metrics.absolute_area) == pytest.approx(36.0)
    assert np.degrees(metrics.total_variation) == pytest.approx(72.0)
    assert np.degrees(metrics.rate_gradient) == pytest.approx(0.0)
    assert metrics.ring_support == pytest.approx(1.0)
    assert metrics.run_span_support == pytest.approx(1.0)
    assert metrics.median_resultant == pytest.approx(0.8)


def test_sustained_peak_rejects_one_ring_spike() -> None:
    """A one-ring maximum must not set the rolling-median peak."""
    baseline = _profile([0, 4, 8, 12, 16, 20, 24])
    spiked = _profile([0, 4, 8, 500, 16, 20, 24])

    baseline_metrics = aggregate_literal_crossing_zone(baseline, 8.0, 64.0)
    spiked_metrics = aggregate_literal_crossing_zone(spiked, 8.0, 64.0)

    assert np.degrees(spiked_metrics.raw_peak) == pytest.approx(500.0)
    assert spiked_metrics.sustained_peak == pytest.approx(
        baseline_metrics.sustained_peak
    )
    assert spiked_metrics.rotation_rate == pytest.approx(
        baseline_metrics.rotation_rate
    )


def test_sustained_peak_searches_runs_without_bridging_gap() -> None:
    """A shorter high-turn run may set the peak but gaps cannot be crossed."""
    profile = _profile(
        [0, 100, 101, np.nan, 0, 1, 2, 3, 4, 5],
        run_ids=[0, 0, 0, -1, 1, 1, 1, 1, 1, 1],
    )

    metrics = aggregate_literal_crossing_zone(profile, 8.0, 88.0)

    assert np.degrees(metrics.sustained_peak) == pytest.approx(100.0)
    assert np.degrees(metrics.net_rotation) == pytest.approx(4.0)
    assert metrics.ring_support == pytest.approx(0.9)
    assert metrics.run_span_support == pytest.approx(5.0 / 9.0)


def test_aggregates_ignore_crossing_count_after_ring_is_supported() -> None:
    """Branch-count replication must not weight an accepted ring aggregate."""
    sparse = _profile([0, 8, 16, 24, 32, 40], crossing_count=3)
    dense = _profile([0, 8, 16, 24, 32, 40], crossing_count=30)

    sparse_metrics = aggregate_literal_crossing_zone(sparse, 8.0, 56.0)
    dense_metrics = aggregate_literal_crossing_zone(dense, 8.0, 56.0)

    assert sparse_metrics == dense_metrics


def test_short_run_keeps_peak_and_diagnostics_but_not_trend_metrics() -> None:
    """Trend estimates should be missing below the minimum run length."""
    profile = _profile([0, 8, 16, 24, 32])

    metrics = aggregate_literal_crossing_zone(profile, 8.0, 48.0)

    assert np.degrees(metrics.sustained_peak) == pytest.approx(24.0)
    assert np.degrees(metrics.raw_peak) == pytest.approx(32.0)
    assert np.isnan(metrics.net_rotation)
    assert np.isnan(metrics.rotation_rate)
    assert np.isnan(metrics.consistency)
    assert np.isnan(metrics.rate_gradient)


@pytest.mark.parametrize("window", [2, 4, True])
def test_peak_window_must_be_odd_integer(window: object) -> None:
    """Invalid persistence windows should fail explicitly."""
    profile = _profile([0, 8, 16, 24, 32, 40])

    with pytest.raises(ValueError, match="odd integer"):
        aggregate_literal_crossing_zone(
            profile,
            8.0,
            56.0,
            peak_window_rings=window,  # type: ignore[arg-type]
        )
