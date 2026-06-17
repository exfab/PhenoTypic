import pytest

from phenotypic import ImagePipeline
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector
from phenotypic.refine import SmallObjectRemover
from phenotypic.measure import MeasureSize
from phenotypic._cli._cli_pipeline_split import (
    split_pipeline_at_gpu,
    StagePlan,
)
from tests._fakes.fake_gpu_detector import FakeGpuDetector


def test_splits_at_first_gpu_detector():
    pipe = ImagePipeline(
        ops=[GaussianBlur(), FakeGpuDetector(), SmallObjectRemover()],
        meas=[MeasureSize()],
    )
    plan = split_pipeline_at_gpu(pipe)
    assert isinstance(plan, StagePlan)
    assert list(plan.pre_pipeline.get_ops().keys()) == ["GaussianBlur"]
    assert isinstance(plan.gpu_detector, FakeGpuDetector)
    assert list(plan.post_pipeline.get_ops().keys()) == ["SmallObjectRemover"]
    # post pipeline carries the measurements
    assert "MeasureSize" in plan.post_pipeline.get_meas()


def test_rejects_more_than_one_gpu_detector():
    pipe = ImagePipeline(ops=[FakeGpuDetector(), FakeGpuDetector()])
    with pytest.raises(ValueError, match="more than one GpuDetector"):
        split_pipeline_at_gpu(pipe)


def test_rejects_no_gpu_detector():
    pipe = ImagePipeline(ops=[GaussianBlur(), OtsuDetector()])
    with pytest.raises(ValueError, match="no GpuDetector"):
        split_pipeline_at_gpu(pipe)
