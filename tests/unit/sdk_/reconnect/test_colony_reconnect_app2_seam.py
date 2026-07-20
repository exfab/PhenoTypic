"""Unit tests for the APP2 GWDT reconnection seam in ``_colony_reconnect``.

These exercise the pure array machinery extracted from FilamentousFungiDetector:
the APP2 endpoint-averaged Dijkstra kernel, the full-image GI adapter, tile
generation/dispatch, and the tiled reconnection driver. Detector-owned helpers
are imported into (or defined in) ``_colony_reconnect``, so they are monkeypatched
on that module rather than on the detector.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from phenotypic.sdk_.branch_pathfinding import DijkstraResult
from phenotypic.sdk_.reconnect import (
    ReconnectConfig,
    app2_gwdt_cost,
    compute_full_image_app2_gi_cost,
    grey_weighted_distance,
    reconnect_fragments_tiled,
)
from phenotypic.sdk_.reconnect import _colony_reconnect
from phenotypic.sdk_.reconnect._colony_reconnect import (
    _generate_tiles,
    _process_tile,
    _run_app2_gwdt_dijkstra,
)


def _cfg(**overrides: object) -> ReconnectConfig:
    """Build a valid ReconnectConfig with reasonable defaults for these seams."""
    base = dict(
        beta=2.0,
        gamma=1.2,
        delta=1.0,
        coherence_window_radius=15,
        mad_window=7,
        gap_crossing_penalty=4.0,
        border_margin_px=50,
        frag_reach_px=10,
        tile_size=1200,
        tile_overlap=500,
        max_gap_length=30,
        path_dilation_radius=2,
        snr_margin=5,
        reconnection_tolerance=2.5,
    )
    base.update(overrides)
    return ReconnectConfig(**base)  # type: ignore[arg-type]


def test_app2_axis_edges_use_endpoint_average_not_destination_cost() -> None:
    """Axial propagation charges each endpoint exactly once per edge."""
    gi_cost = np.array([[1.0, 3.0, 5.0]], dtype=np.float64)
    colony_labels = np.array([[1, 0, 0]], dtype=np.int32)

    result = _run_app2_gwdt_dijkstra(gi_cost, colony_labels)

    np.testing.assert_array_equal(result.cost_distance, [[0.0, 2.0, 6.0]])
    np.testing.assert_array_equal(result.predecessor, [[-1, 0, 1]])


def test_app2_diagonal_edges_use_pinned_source_factor() -> None:
    """The APP2 constant is 1.414214, not an exact square root."""
    gi_cost = np.array([[2.0, 100.0], [100.0, 4.0]], dtype=np.float64)
    colony_labels = np.array([[1, 0], [0, 0]], dtype=np.int32)

    result = _run_app2_gwdt_dijkstra(gi_cost, colony_labels)

    assert result.cost_distance[1, 1] == 3.0 * 1.414214
    assert result.cost_distance[1, 1] != 3.0 * np.sqrt(2.0)


def test_app2_tree_allows_finite_gap_pixels_without_a_threshold_gate() -> None:
    """The detector tree traverses every finite GI pixel, including a costly gap."""
    gi_cost = np.array([[1.0, 1.0, 1000.0, 1.0, 1.0]], dtype=np.float64)
    colony_labels = np.array([[1, 0, 0, 0, 0]], dtype=np.int32)

    result = _run_app2_gwdt_dijkstra(gi_cost, colony_labels)

    assert np.all(np.isfinite(result.cost_distance))
    np.testing.assert_array_equal(result.colony_id, [[1, 1, 1, 1, 1]])
    assert result.predecessor[0, 3] == 2


@pytest.mark.parametrize(
    ("colony_labels", "expected_owner"),
    [
        (np.array([[2, 0, 1]], dtype=np.int32), 2),
        (np.array([[9, 0, 3]], dtype=np.int32), 9),
    ],
)
def test_equal_cost_ownership_uses_first_row_major_boundary_seed(
    colony_labels: np.ndarray,
    expected_owner: int,
) -> None:
    """Strict ties retain the first row-major seed, independent of label value."""
    result = _run_app2_gwdt_dijkstra(
        np.ones((1, 3), dtype=np.float64),
        colony_labels,
    )

    assert result.cost_distance[0, 1] == 1.0
    assert result.colony_id[0, 1] == expected_owner
    assert result.predecessor[0, 1] == 0


def test_equal_cost_predecessor_uses_fixed_detector_neighbor_order() -> None:
    """Detector order differs observably from the exact Vaa3D source order."""
    gi_cost = np.array(
        [
            [3.0, 1.0, 4.0, 5.0],
            [2.0, 3.0, 1.0, 1.0],
            [2.0, 5.0, 5.0, 4.0],
            [3.0, 3.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    colony_labels = np.zeros((4, 4), dtype=np.int32)
    colony_labels[1, 1] = 1

    result = _run_app2_gwdt_dijkstra(gi_cost, colony_labels)

    assert result.colony_id[0, 2] == 1
    assert result.cost_distance[0, 1] == result.cost_distance[1, 2] == 2.0
    assert result.cost_distance[0, 2] == 4.5
    assert result.predecessor[0, 2] == 6


@pytest.mark.parametrize(
    "background",
    [np.zeros((2, 2), dtype=np.bool_), np.ones((2, 2), dtype=np.bool_)],
)
def test_app2_full_image_cost_rejects_single_class_masks(
    background: np.ndarray,
) -> None:
    """The adapter guards the two undefined threshold-class seams."""
    with pytest.raises(ValueError, match="both background and foreground"):
        compute_full_image_app2_gi_cost(
            np.ones((2, 2), dtype=np.float32),
            background=background,
        )


def test_full_image_adapter_applies_gi_lookup_not_raw_distance() -> None:
    """The cumulative GWDT map is transformed before it becomes an edge term."""
    image = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
    background = np.array([[True, False, False]], dtype=np.bool_)
    expected_distance = grey_weighted_distance(
        image, background, connectivity=8
    )
    expected_gi = app2_gwdt_cost(expected_distance)

    actual = compute_full_image_app2_gi_cost(
        image,
        background=background,
    )

    np.testing.assert_array_equal(actual, expected_gi)
    assert not np.array_equal(actual, expected_distance)


@pytest.mark.parametrize("use_app2", [False, True])
def test_tile_dispatch_keeps_app2_separate_from_legacy_dijkstra(
    monkeypatch: pytest.MonkeyPatch,
    use_app2: bool,
) -> None:
    """APP2 never feeds its GI map into the destination-only kernel."""
    calls: list[str] = []
    empty_result = DijkstraResult(
        cost_distance=np.zeros((2, 2), dtype=np.float64),
        colony_id=np.ones((2, 2), dtype=np.int32),
        predecessor=np.full((2, 2), -1, dtype=np.int32),
        colony_centroids={1: (0.0, 0.0)},
    )

    def fake_legacy(*args: object, **kwargs: object) -> DijkstraResult:
        calls.append("dijkstra")
        return empty_result

    def fake_app2(*args: object, **kwargs: object) -> DijkstraResult:
        calls.append("app2_gwdt")
        return empty_result

    monkeypatch.setattr(
        _colony_reconnect, "run_multisource_dijkstra", fake_legacy
    )
    monkeypatch.setattr(
        _colony_reconnect, "_run_app2_gwdt_dijkstra", fake_app2
    )
    monkeypatch.setattr(
        _colony_reconnect, "assign_fragments_to_colonies", lambda *a: {}
    )
    monkeypatch.setattr(
        _colony_reconnect,
        "extract_fragment_paths",
        lambda *a: ({}, []),
    )

    tile_colony = np.array([[1, 0], [0, 0]], dtype=np.int32)
    tile_frags = np.array([[0, 0], [0, 1]], dtype=np.int32)
    zeros = np.zeros((2, 2), dtype=np.float32)
    _process_tile(
        zeros,
        zeros,
        tile_colony,
        tile_frags,
        zeros,
        zeros,
        0.0,
        _cfg(),
        np.ones((2, 2), dtype=np.float64) if use_app2 else None,
    )

    assert calls == (["app2_gwdt"] if use_app2 else ["dijkstra"])


def test_tiled_app2_uses_overlap_as_halo_and_exact_global_map_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeze row-major overlap, edge clipping, GI slicing, and first-write merge."""
    shape = (5, 7)
    app2_gi = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    fragments = np.ones(shape, dtype=np.int32)
    zeros_float = np.zeros(shape, dtype=np.float32)
    expected_tiles = [
        (0, 4, 0, 4),
        (0, 4, 2, 6),
        (0, 4, 4, 7),
        (2, 5, 0, 4),
        (2, 5, 2, 6),
        (2, 5, 4, 7),
    ]
    captured_gi: list[np.ndarray] = []

    monkeypatch.setattr(
        _colony_reconnect,
        "_compute_screening_envelope",
        lambda *args, **kwargs: (np.zeros(shape, dtype=np.float64), None),
    )
    monkeypatch.setattr(
        _colony_reconnect,
        "calibrate_screening_threshold",
        lambda *args, **kwargs: (0.0, None),
    )
    monkeypatch.setattr(
        _colony_reconnect,
        "prescreen_fragments",
        lambda *args, **kwargs: SimpleNamespace(
            screened_fragment_labels=fragments
        ),
    )
    monkeypatch.setattr(_colony_reconnect, "threshold_otsu", lambda _values: 0.0)

    def fake_process_tile(
        tile_cost: np.ndarray,
        tile_raw: np.ndarray,
        tile_colony: np.ndarray,
        tile_frags: np.ndarray,
        tile_pct: np.ndarray,
        tile_gray: np.ndarray,
        pct_noise_ceil: float,
        cfg: ReconnectConfig,
        tile_app2_gi: np.ndarray | None = None,
    ) -> np.ndarray:
        del tile_cost, tile_raw, tile_frags, tile_pct, tile_gray
        del pct_noise_ceil, cfg
        assert tile_app2_gi is not None
        captured_gi.append(tile_app2_gi.copy())
        return np.full(tile_colony.shape, 10 + len(captured_gi) - 1, dtype=np.int32)

    monkeypatch.setattr(
        _colony_reconnect,
        "_process_tile",
        fake_process_tile,
    )

    result = reconnect_fragments_tiled(
        colony_labels=np.zeros(shape, dtype=np.int32),
        fragment_labels=fragments,
        cost_surface=zeros_float,
        unmasked_cost=zeros_float,
        pct_energy=zeros_float,
        grayscale=zeros_float,
        cfg=_cfg(tile_size=4, tile_overlap=2),
        app2_gi_cost=app2_gi,
    )

    assert _generate_tiles(shape, 4, 2) == expected_tiles
    assert len(captured_gi) == len(expected_tiles)
    for actual, (row_start, row_end, col_start, col_end) in zip(
        captured_gi, expected_tiles, strict=True
    ):
        np.testing.assert_array_equal(
            actual,
            app2_gi[row_start:row_end, col_start:col_end],
        )

    expected = np.array(
        [
            [10, 10, 10, 10, 11, 11, 12],
            [10, 10, 10, 10, 11, 11, 12],
            [10, 10, 10, 10, 11, 11, 12],
            [10, 10, 10, 10, 11, 11, 12],
            [13, 13, 13, 13, 14, 14, 15],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(result, expected)
