"""Integration tests for the opt-in APP2 GWDT reconnection seam."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import FilamentousFungiDetector, OtsuDetector
from phenotypic.detect import _filamentous_fungi_detector as fungi_module
from phenotypic.detect._filamentous_fungi_detector import (
    _run_app2_gwdt_dijkstra,
)
from phenotypic.sdk_.branch_pathfinding import DijkstraResult
from phenotypic.sdk_.reconnect import app2_gwdt_cost, grey_weighted_distance


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


@pytest.mark.parametrize(
    "background",
    [np.zeros((2, 2), dtype=np.bool_), np.ones((2, 2), dtype=np.bool_)],
)
def test_app2_full_image_cost_rejects_single_class_masks(
    background: np.ndarray,
) -> None:
    """The detector guards the two undefined threshold-class seams."""
    with pytest.raises(ValueError, match="both background and foreground"):
        FilamentousFungiDetector._compute_full_image_app2_gi_cost(
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

    actual = FilamentousFungiDetector._compute_full_image_app2_gi_cost(
        image,
        background=background,
    )

    np.testing.assert_array_equal(actual, expected_gi)
    assert not np.array_equal(actual, expected_distance)


def test_reconnect_strategy_round_trips_and_defaults_to_legacy() -> None:
    """Serialization preserves the opt-in strategy and old JSON defaults."""
    default = FilamentousFungiDetector()
    assert default.reconnect_strategy == "dijkstra"

    pipeline = ImagePipeline(
        ops=[FilamentousFungiDetector(reconnect_strategy="app2_gwdt")]
    )
    restored = ImagePipeline.from_json(pipeline.to_json())
    restored_detector = next(iter(restored._ops.values()))

    assert isinstance(restored_detector, FilamentousFungiDetector)
    assert restored_detector.reconnect_strategy == "app2_gwdt"


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

    monkeypatch.setattr(fungi_module, "run_multisource_dijkstra", fake_legacy)
    monkeypatch.setattr(fungi_module, "_run_app2_gwdt_dijkstra", fake_app2)
    monkeypatch.setattr(
        fungi_module, "assign_fragments_to_colonies", lambda *a: {}
    )
    monkeypatch.setattr(
        fungi_module,
        "extract_fragment_paths",
        lambda *a: ({}, []),
    )

    detector = FilamentousFungiDetector()
    tile_colony = np.array([[1, 0], [0, 0]], dtype=np.int32)
    tile_frags = np.array([[0, 0], [0, 1]], dtype=np.int32)
    zeros = np.zeros((2, 2), dtype=np.float32)
    detector._process_tile(
        zeros,
        zeros,
        tile_colony,
        tile_frags,
        zeros,
        zeros,
        0.0,
        np.ones((2, 2), dtype=np.float64) if use_app2 else None,
    )

    assert calls == (["app2_gwdt"] if use_app2 else ["dijkstra"])


def test_app2_cost_is_computed_once_on_full_image_before_tiling(
    synth_plate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tiles receive a global GI map instead of recomputing a local transform."""
    computed_shapes: list[tuple[int, int]] = []
    forwarded_maps: list[np.ndarray] = []

    def fake_compute(
        image: np.ndarray, *, background: np.ndarray
    ) -> np.ndarray:
        assert image.shape == background.shape
        computed_shapes.append(image.shape)
        return np.arange(image.size, dtype=np.float64).reshape(image.shape)

    def fake_reconnect(
        self: FilamentousFungiDetector,
        colony_labels: np.ndarray,
        fragment_labels: np.ndarray,
        cost_surface: np.ndarray,
        unmasked_cost: np.ndarray,
        pct_energy: np.ndarray,
        grayscale: np.ndarray,
        app2_gi_cost: np.ndarray | None = None,
    ) -> np.ndarray:
        assert app2_gi_cost is not None
        forwarded_maps.append(app2_gi_cost)
        return colony_labels

    monkeypatch.setattr(
        FilamentousFungiDetector,
        "_compute_full_image_app2_gi_cost",
        staticmethod(fake_compute),
    )
    monkeypatch.setattr(
        FilamentousFungiDetector,
        "_reconnect_fragments_tiled",
        fake_reconnect,
    )

    image = synth_plate.copy()
    FilamentousFungiDetector(
        inoculum_detector=OtsuDetector(ignore_zeros=True),
        reconnect_strategy="app2_gwdt",
    ).apply(image)

    assert computed_shapes == [image.detect_mat[:].shape]
    assert len(forwarded_maps) == 1
    assert forwarded_maps[0].shape == image.detect_mat[:].shape


def test_explicit_legacy_strategy_is_byte_identical_to_default(
    synth_plate,
) -> None:
    """Opt-in integration does not change the existing detector output."""
    common = {"inoculum_detector": OtsuDetector(ignore_zeros=True)}
    implicit = FilamentousFungiDetector(**common).apply(synth_plate.copy())
    explicit = FilamentousFungiDetector(
        **common,
        reconnect_strategy="dijkstra",
    ).apply(synth_plate.copy())

    np.testing.assert_array_equal(implicit.objmap[:], explicit.objmap[:])
