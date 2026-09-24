import pytest

from phenotypic import ImagePipeline
from phenotypic.enhance import BlurGauss
from phenotypic.detect import OtsuDetector
from phenotypic.refine import SmallObjectRemover
from phenotypic.measure import MeasureSize
from phenotypic.measure import MeasureSymZones
from phenotypic.abc_.plotting import PlotImage
from phenotypic._cli._cli_pipeline_split import (
    split_pipeline_at_gpu,
    StagePlan,
)
from tests._fakes.fake_gpu_detector import FakeGpuDetector


def test_splits_at_first_gpu_detector():
    pipe = ImagePipeline(
            ops=[BlurGauss(), FakeGpuDetector(), SmallObjectRemover()],
            meas=[MeasureSize()],
    )
    plan = split_pipeline_at_gpu(pipe)
    assert isinstance(plan, StagePlan)
    assert list(plan.pre_pipeline.get_ops().keys()) == ["BlurGauss"]
    assert isinstance(plan.gpu_detector, FakeGpuDetector)
    # The detector's own slot is now INCLUDED in post_pipeline: the cut is
    # taken at the detector's top-level ancestor, and for a top-level detector
    # that ancestor is the detector itself. Stage 3 substitutes a
    # ReplayDetector into this slot instead of calling _write_object_output
    # before the post pipeline. Omitting it would make Stage 3 re-run the REAL
    # detector on a CPU node.
    assert list(plan.post_pipeline.get_ops().keys()) == [
        "FakeGpuDetector",
        "SmallObjectRemover",
    ]
    # post pipeline carries the measurements
    assert "MeasureSize" in plan.post_pipeline.get_meas()


def test_rejects_more_than_one_gpu_detector():
    pipe = ImagePipeline(ops=[FakeGpuDetector(), FakeGpuDetector()])
    with pytest.raises(ValueError, match="more than one GpuDetector"):
        split_pipeline_at_gpu(pipe)


def test_rejects_no_gpu_detector():
    pipe = ImagePipeline(ops=[BlurGauss(), OtsuDetector()])
    with pytest.raises(ValueError, match="no GpuDetector"):
        split_pipeline_at_gpu(pipe)


def test_measurer_plot_binding_survives_into_stage_three():
    zones = MeasureSymZones()
    pipe = ImagePipeline(
            ops=[FakeGpuDetector()], meas={"zones": zones}, plots=[zones]
    )
    plan = split_pipeline_at_gpu(pipe)
    assert plan.post_pipeline.get_plots()[0].plot is zones
    assert plan.post_pipeline.get_plots()[0].ref.key == "zones"


class _PreGpuPlot(BlurGauss, PlotImage):
    pass


def test_a_pre_gpu_plot_goes_to_stage_one_only():
    """Stage 1 applies the operation, so Stage 1 draws its figure and Stage 3
    carries it (figures spec §3a); ``post_pipeline`` no longer holds the op,
    so the binding could not even resolve there."""
    pre = _PreGpuPlot()
    zones = MeasureSymZones()
    pipe = ImagePipeline(
            ops={"pre": pre, "gpu": FakeGpuDetector()},
            meas={"zones": zones},
            plots=[pre, zones],
    )
    plan = split_pipeline_at_gpu(pipe)
    [stage1] = plan.pre_pipeline.get_plots()
    assert (stage1.id, stage1.ref.key) == ("pre", "pre")
    assert stage1.plot is pre is plan.pre_pipeline.get_ops()["pre"]
    assert [b.id for b in plan.post_pipeline.get_plots()] == ["zones"]
