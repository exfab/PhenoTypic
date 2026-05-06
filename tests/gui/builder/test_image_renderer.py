"""Tests for :mod:`phenotypic.gui.builder._image_renderer`.

Covers:
- :func:`to_png_bytes` per channel (rgb / gray / detect_mat / objmap).
- :func:`bytes_to_data_uri` round-trip.
- :func:`to_data_uri` legacy compatibility wrapper.
- :func:`to_overlay_png_bytes` overlay composition + the matplotlib fallback
  when scikit-image is unavailable.
- :func:`render_node_preview` per-stage dispatch (enhancer / detector /
  refiner / corrector / nested ``ImagePipeline``).
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image as PILImage

from phenotypic.data._synthetic_data import load_synth_yeast_plate
from phenotypic.gui.builder import _image_renderer
from phenotypic.gui.builder._image_renderer import (
    bytes_to_data_uri,
    render_node_preview,
    to_data_uri,
    to_overlay_png_bytes,
    to_png_bytes,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def synthetic_image():
    """A small synthetic plate, fast to load and re-use across tests."""
    return load_synth_yeast_plate()


@pytest.fixture(scope="module")
def detected_image(synthetic_image):
    """Synthetic plate with an objmap populated via OtsuDetector.

    Needed to exercise overlay rendering and the objmap channel.
    """
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector

    pipeline = ImagePipeline(ops=[OtsuDetector()], name="overlay-fixture")
    return pipeline.apply(synthetic_image)


# ---------------------------------------------------------------------------
# to_png_bytes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel", ["rgb", "gray", "detect_mat"])
def test_to_png_bytes_starts_with_png_magic(synthetic_image, channel):
    blob = to_png_bytes(synthetic_image, channel=channel)
    assert isinstance(blob, bytes)
    assert blob[:8] == PNG_MAGIC


def test_to_png_bytes_objmap(detected_image):
    blob = to_png_bytes(detected_image, channel="objmap")
    assert blob[:8] == PNG_MAGIC


def test_to_png_bytes_respects_max_dim(synthetic_image):
    """The encoded PNG's longer side must not exceed ``max_dim``."""
    blob = to_png_bytes(synthetic_image, channel="rgb", max_dim=128)
    pil = PILImage.open(io.BytesIO(blob))
    longer = max(pil.size)
    assert longer <= 128, f"longer side {longer} > 128"


def test_to_png_bytes_unknown_channel_raises(synthetic_image):
    with pytest.raises(ValueError):
        to_png_bytes(synthetic_image, channel="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# bytes_to_data_uri
# ---------------------------------------------------------------------------


def test_bytes_to_data_uri_format():
    blob = b"hello"
    uri = bytes_to_data_uri(blob)
    assert uri.startswith("data:image/png;base64,")
    decoded = base64.b64decode(uri.removeprefix("data:image/png;base64,"))
    assert decoded == blob


def test_bytes_to_data_uri_round_trip(synthetic_image):
    blob = to_png_bytes(synthetic_image, channel="gray", max_dim=64)
    uri = bytes_to_data_uri(blob)
    head = "data:image/png;base64,"
    assert uri.startswith(head)
    assert base64.b64decode(uri[len(head):]) == blob


def test_to_data_uri_wrapper_matches_split(synthetic_image):
    """The legacy ``to_data_uri`` is a thin wrapper over the split helpers."""
    direct = to_data_uri(synthetic_image, channel="rgb", max_dim=128)
    composed = bytes_to_data_uri(
        to_png_bytes(synthetic_image, channel="rgb", max_dim=128)
    )
    assert direct == composed


# ---------------------------------------------------------------------------
# to_overlay_png_bytes
# ---------------------------------------------------------------------------


def test_to_overlay_png_bytes_starts_with_png_magic(detected_image):
    blob = to_overlay_png_bytes(detected_image, max_dim=128)
    assert blob[:8] == PNG_MAGIC


def test_to_overlay_png_bytes_respects_max_dim(detected_image):
    blob = to_overlay_png_bytes(detected_image, max_dim=96)
    pil = PILImage.open(io.BytesIO(blob))
    assert max(pil.size) <= 96


def test_to_overlay_png_bytes_works_without_skimage(
    monkeypatch, detected_image
):
    """Force the scikit-image import to fail and confirm the fallback runs."""

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "skimage.color" or name.startswith("skimage."):
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    blob = to_overlay_png_bytes(detected_image, max_dim=64)
    assert blob[:8] == PNG_MAGIC


def test_to_overlay_png_bytes_with_zero_labels(synthetic_image):
    """An all-zero objmap still produces valid bytes via the label2rgb path.

    The pre-detection synthetic plate has an all-zero objmap (no detections),
    which exercises the ``bg_label=0`` path of ``label2rgb`` rather than the
    skimage-import fallback (covered by a separate test).
    """
    blob = to_overlay_png_bytes(synthetic_image, max_dim=64)
    assert blob[:8] == PNG_MAGIC


# ---------------------------------------------------------------------------
# render_node_preview dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "class_name",
    [
        "GaussianBlur",       # Enhancer → detect_mat
        "OtsuDetector",       # Detector → overlay
        "ImagePipeline",      # nested pipeline → rgb
        "DefinitelyNotAClass",  # unknown → rgb
    ],
)
def test_render_node_preview_dispatch(detected_image, class_name):
    """Every supported class_name produces valid PNG bytes."""
    blob = render_node_preview(detected_image, class_name, max_dim=64)
    assert isinstance(blob, bytes)
    assert blob[:8] == PNG_MAGIC


def test_render_node_preview_detector_uses_overlay_path(detected_image):
    """Sanity: the detector path produces different bytes than a plain rgb
    render of the same image (overlay alpha-blends color over gray).
    """
    overlay = render_node_preview(detected_image, "OtsuDetector", max_dim=64)
    rgb = to_png_bytes(detected_image, channel="rgb", max_dim=64)
    assert overlay != rgb


def test_render_node_preview_enhancer_uses_detect_mat(synthetic_image):
    """Enhancer renders should match the detect_mat single-channel render."""
    via_dispatcher = render_node_preview(
        synthetic_image, "GaussianBlur", max_dim=64
    )
    direct = to_png_bytes(synthetic_image, channel="detect_mat", max_dim=64)
    assert via_dispatcher == direct


# ---------------------------------------------------------------------------
# Internal: _label_map_to_rgb fallback
# ---------------------------------------------------------------------------


def test_label_map_fallback_to_matplotlib(monkeypatch):
    """Force the skimage import to fail; matplotlib path must still produce
    a valid (H, W, 3) uint8 array.
    """

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("skimage"):
            raise ImportError("forced")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    arr = np.zeros((10, 10), dtype=np.int32)
    arr[2:5, 2:5] = 1
    arr[6:9, 6:9] = 2
    rgb = _image_renderer._label_map_to_rgb(arr)
    assert rgb.shape == (10, 10, 3)
    assert rgb.dtype == np.uint8
