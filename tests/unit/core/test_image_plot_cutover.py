"""Regression tests for the hard removal of the ``Image.plot`` accessor."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from phenotypic import GridImage, Image
from phenotypic._core._image_parts import ImageVisualizationHandler
from phenotypic._core._image_parts import accessors


def _image_with_objects() -> Image:
    image = Image(np.zeros((8, 8, 3), dtype=np.uint8))
    image.objmap[1:3, 1:3] = 1
    return image


def test_image_plot_and_accessor_cache_are_removed() -> None:
    image = _image_with_objects()

    assert not hasattr(image, "plot")
    assert not hasattr(image._accessors, "plot")
    assert "PlotAccessor" not in accessors.__all__
    assert not hasattr(accessors, "PlotAccessor")


def test_legacy_plot_modules_and_registry_api_are_removed() -> None:
    sdk = importlib.import_module("phenotypic.sdk_")
    assert not hasattr(sdk, "register")

    for module in (
        "phenotypic._core._image_parts._image_plot_handler",
        "phenotypic._core._image_parts.accessors._plot_accessor",
        "phenotypic._core._image_parts.accessors._dash_plot_accessor",
        "phenotypic.sdk_.register",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)


def test_visualization_handler_preserves_image_and_grid_image_mro() -> None:
    assert ImageVisualizationHandler in Image.__mro__
    assert ImageVisualizationHandler in GridImage.__mro__
    assert Image.napari is ImageVisualizationHandler.napari


def test_image_and_channel_dash_methods_survive_plot_accessor_removal() -> None:
    go = pytest.importorskip("plotly.graph_objects")
    image = _image_with_objects()

    assert isinstance(image.dash(), go.Figure)
    for accessor_name in ("rgb", "gray", "detect_mat", "objmap", "objmask"):
        assert isinstance(getattr(image, accessor_name).dash(), go.Figure)
