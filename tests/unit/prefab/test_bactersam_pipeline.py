"""``BacterSamPipeline`` reproduces the bacterial run it was built from.

The fixture is that run's ``pipeline.json.pht-pipe`` trimmed to its operation
configuration: ``pipe_cfgs``, ``meas`` and the grid shape. The run's plot, its
name and description, and every experiment-specific metadata column are left
out, because the prefab drops the plot (it is bound to that experiment's
metadata) and a fixture is no place for experiment data.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from phenotypic import ImagePipeline
from phenotypic.prefab import BacterSamPipeline
from phenotypic.sdk_._operation_tree import find_operations

FIXTURE = Path(__file__).parent / "fixtures" / "bactersam_reference_pipeline.json"

#: Colour correction and denoising: the prefab assumes both happened upstream.
_COLOUR_AND_DENOISE = {
    "ColorCorrector",
    "CalibrateColorRpcc",
    "ColorDenoise",
    "DenoiseBlockMatch",
    "EnhanceBlockMatch",
    "BayesShrinkCorrector",
    "VisuShrinkCorrector",
}


def _sections(pipeline: ImagePipeline) -> dict:
    """The parts of a pipeline's JSON that say what it computes."""
    data = json.loads(pipeline.to_json())
    return {key: data[key] for key in ("pipe_cfgs", "meas", "nrows", "ncols")}


def test_default_config_equals_the_reference_run():
    # Normalise the fixture through the current schema, so a field added to an
    # operation since the run's 0.20.0 is not reported as a difference.
    with pytest.warns(UserWarning, match="saved with phenotypic version"):
        reference = ImagePipeline.from_json(FIXTURE)
    assert _sections(BacterSamPipeline()) == _sections(reference)


def test_no_colour_correction_or_denoising_anywhere_in_the_tree():
    found = {
        type(op).__name__
        for _, op in find_operations(BacterSamPipeline(), lambda op: True)
    }
    assert not found & _COLOUR_AND_DENOISE


def test_construction_loads_no_torch():
    code = (
        "import sys\n"
        "from phenotypic.prefab import BacterSamPipeline\n"
        "BacterSamPipeline()\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_sam2_reads_detect_mat():
    assert BacterSamPipeline().get_ops()["Detector"].input_layer == "detect_mat"


def test_texture_measurers_one_per_scale():
    meas = BacterSamPipeline(texture_scales=(7, 9)).meas
    texture_keys = [key for key in meas if key.startswith("MeasureTexture")]
    assert texture_keys == ["MeasureTexture", "MeasureTexture_1"]
    assert [meas[key].scale for key in texture_keys] == [[7], [9]]


@pytest.mark.parametrize(
    ("kwarg", "value", "op_key", "field"),
    [
        ("crop_left", 11, "Crop", "left"),
        ("crop_right", 12, "Crop", "right"),
        ("crop_top", 13, "Crop", "top"),
        ("crop_bottom", 14, "Crop", "bottom"),
        ("blur_sigma", 3.0, "Blur", "sigma"),
        ("background_sigma", 400.0, "RemoveBG", "sigma"),
        ("sam_model_size", "small", "Detector", "model_size"),
        ("sam_points_per_side", 16, "Detector", "points_per_side"),
        ("sam_pred_iou_thresh", 0.8, "Detector", "pred_iou_thresh"),
        ("sam_stability_score_thresh", 0.9, "Detector", "stability_score_thresh"),
        ("sam_min_mask_region_area", 200, "Detector", "min_mask_region_area"),
        ("sam_crop_n_layers", 0, "Detector", "crop_n_layers"),
        ("sam_device", "cpu", "Detector", "device"),
        ("circularity_cutoff", 0.6, "CircularityFilter", "cutoff"),
        ("border_size", 5, "BorderFilter", "border_size"),
        ("min_object_size", 700, "SmallObjectFilter", "min_size"),
    ],
)
def test_kwargs_reach_their_ops(kwarg, value, op_key, field):
    op = BacterSamPipeline(**{kwarg: value}).get_ops()[op_key]
    assert getattr(op, field) == value


def test_grid_shape_kwargs():
    pipeline = BacterSamPipeline(nrows=16, ncols=24)
    assert (pipeline.nrows, pipeline.ncols) == (16, 24)


def test_construction_is_keyword_only():
    with pytest.raises(TypeError):
        BacterSamPipeline(450)  # type: ignore[misc]


def test_stages_at_the_detector():
    from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu

    plan = split_pipeline_at_gpu(BacterSamPipeline())
    assert plan.gpu_path == ("Detector",)


def _paths_requiring(pipeline, attribute: str) -> set[tuple[str, ...]]:
    def requires(op) -> bool:
        declared = getattr(op, "preflight_requirements", None)
        return declared is not None and getattr(declared(), attribute)

    return {path for path, _ in find_operations(pipeline, requires)}


def test_requires_grid_image():
    assert ("GridLargest",) in _paths_requiring(BacterSamPipeline(), "grid_image")


def test_requires_rgb():
    assert ("meas:MeasureColor",) in _paths_requiring(BacterSamPipeline(), "rgb_input")


def test_json_round_trip_keeps_class_and_config():
    pipeline = BacterSamPipeline(min_object_size=700)
    loaded = BacterSamPipeline.from_json(pipeline.to_json())
    assert type(loaded) is BacterSamPipeline
    assert _sections(loaded) == _sections(pipeline)
