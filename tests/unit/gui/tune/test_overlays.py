"""Unit tests for the candidate-overlay render backend (tune Curate B-i, task B1).

``render_candidate_overlay`` is the pure helper the Curate Dash surface (B-ii)
calls to turn a sampled parameter combo into a Plotly-ready RGB overlay array:
``build_pipeline(base, params) -> pipeline.apply(plate) -> label2rgb(objmap over
detect_mat)``. It returns a NumPy array (for ``go.Image``), not PNG bytes.
"""
from __future__ import annotations

import numpy as np

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.gui.tune._overlays import render_candidate_overlay


def test_render_candidate_overlay_returns_rgb() -> None:
    base = ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()])
    img = render_candidate_overlay(base, {"0.sigma": 2.0}, load_synth_yeast_plate())
    assert img.ndim == 3 and img.shape[2] in (3, 4)


def test_render_candidate_overlay_is_uint8_downscaled() -> None:
    base = ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()])
    img = render_candidate_overlay(
        base, {"0.sigma": 2.0}, load_synth_yeast_plate(), max_dim=128
    )
    # The longer spatial side is clamped to max_dim and the array is display-ready.
    assert max(img.shape[0], img.shape[1]) <= 128
    assert img.dtype == np.uint8


def test_render_candidate_overlay_param_changes_segmentation() -> None:
    base = ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()])
    plate = load_synth_yeast_plate()
    weak = render_candidate_overlay(base, {"0.sigma": 0.5}, plate)
    strong = render_candidate_overlay(base, {"0.sigma": 6.0}, plate)
    # A heavier Gaussian blur shifts the detected colonies, so the overlays differ.
    assert weak.shape == strong.shape
    assert not np.array_equal(weak, strong)
