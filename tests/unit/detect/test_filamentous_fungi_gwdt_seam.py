"""Integration tests for the opt-in APP2 GWDT reconnection seam.

The pure ``_colony_reconnect`` array machinery is covered in
``tests/unit/sdk_/reconnect/test_colony_reconnect_app2_seam.py``. These tests
exercise the FilamentousFungiDetector integration: strategy serialization, the
full-image-cost-before-tiling wiring, and legacy byte-identity. The detector now
imports ``compute_full_image_app2_gi_cost`` / ``reconnect_fragments_tiled`` from
``phenotypic.sdk_.reconnect``, so they are attributes of ``fungi_module`` and are
monkeypatched there as plain free functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import FilamentousFungiDetector, OtsuDetector
from phenotypic.detect import _filamentous_fungi_detector as fungi_module


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
        colony_labels: np.ndarray,
        fragment_labels: np.ndarray,
        cost_surface: np.ndarray,
        unmasked_cost: np.ndarray,
        pct_energy: np.ndarray,
        grayscale: np.ndarray,
        cfg: object,
        app2_gi_cost: np.ndarray | None = None,
    ) -> np.ndarray:
        assert app2_gi_cost is not None
        forwarded_maps.append(app2_gi_cost)
        return colony_labels

    monkeypatch.setattr(
        fungi_module,
        "compute_full_image_app2_gi_cost",
        fake_compute,
    )
    monkeypatch.setattr(
        fungi_module,
        "reconnect_fragments_tiled",
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
