"""Grid, preset and detector checks of the run preflight.

Spec: ``docs/superpowers/specs/2026-09-24-cli-preflight/design.md`` §4
(findings F4, F5, F6; review R4, R5). Plan: Task 5. Every positive case has a
counterpart proving that an out-of-scope slot or mode produces nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import GridImage, Image, ImagePipeline
from phenotypic._cli._cli_preflight import (
    check_detector_present,
    check_grid_image,
    check_grid_preset,
)
from phenotypic.detect import CompositeDetector, OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.grid import AutoGridFinder
from phenotypic.measure import MeasureGridSpread, MeasureSize
from phenotypic.refine import RemoveGridOutliers
from tests.unit.cli._preflight_support import make_context, make_datasets


def _codes(findings) -> list[str]:
    return [f.code for f in findings]


# --- PF-GRID-IMAGE ------------------------------------------------------------


@pytest.fixture
def nested_grid_refiner() -> ImagePipeline:
    """A grid refiner hidden inside a composite branch: proves the tree walk."""
    return ImagePipeline(
        ops={
            "detect": CompositeDetector(
                ops=[OtsuDetector(), ImagePipeline(ops={"r": RemoveGridOutliers()})]
            )
        }
    )


def test_a_nested_grid_op_under_plain_image_is_an_error(nested_grid_refiner) -> None:
    findings = check_grid_image(make_context(nested_grid_refiner, image_type="Image"))

    assert _codes(findings) == ["PF-GRID-IMAGE"]
    assert findings[0].severity == "error"
    assert "detect/ops[1]/r" in findings[0].message


def test_a_grid_measurer_under_plain_image_is_an_error() -> None:
    pipeline = ImagePipeline(ops={"det": OtsuDetector()}, meas={"g": MeasureGridSpread()})

    findings = check_grid_image(make_context(pipeline, image_type="Image"))

    assert _codes(findings) == ["PF-GRID-IMAGE"]
    assert "meas:g" in findings[0].message


def test_grid_ops_under_grid_image_are_fine(nested_grid_refiner) -> None:
    assert check_grid_image(make_context(nested_grid_refiner, image_type="GridImage")) == []


def test_a_grid_measurer_is_out_of_scope_in_process_mode() -> None:
    """``process`` never measures (review R4)."""
    pipeline = ImagePipeline(ops={"det": OtsuDetector()}, meas={"g": MeasureGridSpread()})

    assert check_grid_image(make_context(pipeline, "process", image_type="Image")) == []


def _store(path: Path, cls: type) -> Path:
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[8:16, 8:16] = 200
    image = cls(arr, name=path.name.split(".")[0])
    image.save2zarr(path)
    return path


def test_measure_mode_reads_each_stores_recorded_image_class(tmp_path: Path) -> None:
    """``measure`` loads each store as its recorded class, so one plain store warns."""
    plain = _store(tmp_path / "plain.ome.zarr", Image)
    grid = _store(tmp_path / "grid.ome.zarr", GridImage)
    pipeline = ImagePipeline(meas={"g": MeasureGridSpread()})

    findings = check_grid_image(
        make_context(pipeline, "measure", make_datasets(plain, grid), image_type="GridImage")
    )

    assert _codes(findings) == ["PF-GRID-IMAGE"]
    assert findings[0].severity == "warning"
    assert findings[0].subjects == (str(plain),)


# --- PF-GRID-PRESET -----------------------------------------------------------


def _preset(**kwargs) -> ImagePipeline:
    return ImagePipeline(ops={"det": OtsuDetector()}, meas={"s": MeasureSize()}, **kwargs)


def test_a_preset_with_both_dimensions_under_plain_image_is_an_error() -> None:
    findings = check_grid_preset(make_context(_preset(nrows=8, ncols=12), image_type="Image"))

    assert _codes(findings) == ["PF-GRID-PRESET"]
    assert findings[0].severity == "error"


def test_a_preset_with_one_dimension_injects_nothing() -> None:
    """``measure()`` injects the finder only when both are set (review R5)."""
    assert check_grid_preset(make_context(_preset(nrows=8), image_type="Image")) == []


def test_a_preset_with_a_grid_finder_already_in_meas_injects_nothing() -> None:
    pipeline = ImagePipeline(
        ops={"det": OtsuDetector()},
        meas={"finder": AutoGridFinder(nrows=8, ncols=12)},
        nrows=8,
        ncols=12,
    )

    assert check_grid_preset(make_context(pipeline, image_type="Image")) == []


def test_a_preset_is_irrelevant_in_process_mode_and_under_grid_image() -> None:
    pipeline = _preset(nrows=8, ncols=12)

    assert check_grid_preset(make_context(pipeline, "process", image_type="Image")) == []
    assert check_grid_preset(make_context(pipeline, image_type="GridImage")) == []


# --- PF-NO-DETECTOR -----------------------------------------------------------


def _enhancer_only() -> ImagePipeline:
    return ImagePipeline(ops={"blur": BlurGauss()}, meas={"s": MeasureSize()})


@pytest.mark.parametrize("image_type", ["Image", "GridImage"])
def test_a_forward_run_without_a_detector_is_an_error(image_type: str) -> None:
    findings = check_detector_present(make_context(_enhancer_only(), image_type=image_type))

    assert _codes(findings) == ["PF-NO-DETECTOR"]
    assert findings[0].severity == "error"


@pytest.mark.parametrize("mode", ["measure", "process"])
def test_no_detector_is_fine_outside_full_mode(mode: str) -> None:
    assert check_detector_present(make_context(_enhancer_only(), mode)) == []


def test_a_detector_nested_in_a_composite_satisfies_the_check() -> None:
    pipeline = ImagePipeline(
        ops={"blur": BlurGauss(), "c": CompositeDetector(ops=[OtsuDetector()])},
        meas={"s": MeasureSize()},
    )

    assert check_detector_present(make_context(pipeline)) == []
