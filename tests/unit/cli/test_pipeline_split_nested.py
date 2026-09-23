"""Path-shaped ``StagePlan``: the cut, the branch prefix, and the plot guard.

These tests build pipelines IN MEMORY (no ``from_json``), so a plain import of
``FakeGpuDetector`` is enough -- no namespace registration needed. Same as
``test_cli_pipeline_split.py:14``.
"""

import pytest

from phenotypic import ImagePipeline
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
from phenotypic._cli._cli_validation import UnstageableGpuDetectorError
from phenotypic.abc_.plotting import PlotImage
from phenotypic.detect import CompositeDetector, ManualPointDetector
from phenotypic.enhance import BlurGauss, ContrastStretching, SubtractGaussian
from tests._fakes.fake_gpu_detector import FakeGpuDetector

CENTERS = [(10.0, 10.0), (10.0, 40.0)]


def _manual():
    return ManualPointDetector(centers=CENTERS, shape="disk", width=11)


class _PlottingComposite(CompositeDetector, PlotImage):
    """Plot capability is a MIXIN ON THE OP CLASS, not retrofitted onto an
    instance -- see ``test_cli_pipeline_split.py:53-54``.
    ``normalize_plot_bindings`` raises on a non-plot-capable entry, so a plain
    ``CompositeDetector`` instance will not do."""


class _PlottingGpu(FakeGpuDetector, PlotImage):
    """A plot-capable GpuDetector, for the top-level plot refusal."""


class _PreGpuPlot(BlurGauss, PlotImage):
    """A plot-capable pre-GPU enhancer (mirrors ``test_cli_pipeline_split.py``)."""


def _with_plot_on(pipeline: ImagePipeline, key: str) -> ImagePipeline:
    """Rebuild *pipeline* with a plot bound by identity to ``ops[key]``."""
    ops = pipeline.get_ops()
    return ImagePipeline(ops=dict(ops), plots=[ops[key]])


# --------------------------------------------------------------------------
# The cut
# --------------------------------------------------------------------------


def test_split_cuts_at_the_top_level_ancestor():
    pipe = ImagePipeline(
        ops={
            "BlurGauss": BlurGauss(sigma=2.0),
            "SubtractGaussian": SubtractGaussian(sigma=50.0),
            "CompositeDetector": CompositeDetector(
                ops=[FakeGpuDetector(), _manual()], mode="overlap"
            ),
            "ContrastStretching": ContrastStretching(input_layer="detect_mat"),
        }
    )
    plan = split_pipeline_at_gpu(pipe)

    assert plan.gpu_path == ("CompositeDetector", "ops[0]")
    assert list(plan.pre_pipeline.get_ops()) == ["BlurGauss", "SubtractGaussian"]
    # the ANCESTOR heads the post pipeline -- it has not run yet
    assert list(plan.post_pipeline.get_ops()) == [
        "CompositeDetector",
        "ContrastStretching",
    ]


def test_a_top_level_detector_stays_in_the_post_pipeline():
    """Its slot must survive so Stage 3's stub can land in it.

    Dropping it (the pre-change behaviour) would make Stage 3 re-run the REAL
    detector on a CPU node.
    """
    pipe = ImagePipeline(
        ops={
            "BlurGauss": BlurGauss(sigma=2.0),
            "FakeGpuDetector": FakeGpuDetector(),
            "ContrastStretching": ContrastStretching(input_layer="detect_mat"),
        }
    )
    plan = split_pipeline_at_gpu(pipe)

    assert list(plan.pre_pipeline.get_ops()) == ["BlurGauss"]
    assert list(plan.post_pipeline.get_ops()) == [
        "FakeGpuDetector",
        "ContrastStretching",
    ]


def test_the_detector_returned_is_the_one_at_the_path():
    """Identity, not merely type -- substitution addresses this exact node."""
    detector = FakeGpuDetector(threshold=0.31)
    composite = CompositeDetector(ops=[_manual(), detector], mode="overlap")
    plan = split_pipeline_at_gpu(ImagePipeline(ops={"C": composite}))

    assert plan.gpu_path == ("C", "ops[1]")
    assert plan.gpu_detector is composite.ops[1]
    assert plan.gpu_detector.threshold == 0.31


# --------------------------------------------------------------------------
# The Stage-2 branch prefix
# --------------------------------------------------------------------------


def test_a_bare_leaf_needs_no_stage2_prefix():
    pipe = ImagePipeline(
        ops={
            "CompositeDetector": CompositeDetector(
                ops=[FakeGpuDetector(), _manual()], mode="overlap"
            )
        }
    )
    assert split_pipeline_at_gpu(pipe).stage2_prefix == []


def test_a_branch_pipeline_contributes_its_preceding_ops():
    branch = ImagePipeline(
        ops={
            "ContrastStretching": ContrastStretching(input_layer="detect_mat"),
            "FakeGpu": FakeGpuDetector(),
        }
    )
    pipe = ImagePipeline(
        ops={
            "CompositeDetector": CompositeDetector(
                ops=[branch, _manual()], mode="overlap"
            )
        }
    )
    plan = split_pipeline_at_gpu(pipe)

    assert plan.gpu_path == ("CompositeDetector", "ops[0]", "FakeGpu")
    assert [type(op).__name__ for op in plan.stage2_prefix] == ["ContrastStretching"]
    # the identical objects, not equal copies -- Stage 2 applies these
    assert plan.stage2_prefix[0] is branch.get_ops()["ContrastStretching"]


def test_a_branch_pipeline_contributes_ONLY_the_ops_before_the_detector():
    branch = ImagePipeline(
        ops={
            "ContrastStretching": ContrastStretching(input_layer="detect_mat"),
            "FakeGpu": FakeGpuDetector(),
            "BlurGauss": BlurGauss(sigma=1.0),
        }
    )
    pipe = ImagePipeline(
        ops={"C": CompositeDetector(ops=[branch, _manual()], mode="overlap")}
    )
    plan = split_pipeline_at_gpu(pipe)

    # BlurGauss follows the detector INSIDE the branch; Stage 3 runs it as part
    # of the normal branch run, so Stage 2 must not.
    assert [type(op).__name__ for op in plan.stage2_prefix] == ["ContrastStretching"]


def test_composite_siblings_contribute_nothing_to_the_prefix():
    """Composite ops are PARALLEL branches applied to the same input."""
    pipe = ImagePipeline(
        ops={
            "CompositeDetector": CompositeDetector(
                ops=[_manual(), FakeGpuDetector()], mode="overlap"
            )
        }
    )
    plan = split_pipeline_at_gpu(pipe)
    assert plan.gpu_path == ("CompositeDetector", "ops[1]")
    assert plan.stage2_prefix == []


def test_a_composite_inside_a_composite_contributes_nothing():
    inner = CompositeDetector(ops=[FakeGpuDetector(), _manual()], mode="union")
    pipe = ImagePipeline(
        ops={"CompositeDetector": CompositeDetector(ops=[inner, _manual()],
                                                    mode="overlap")}
    )
    plan = split_pipeline_at_gpu(pipe)
    assert plan.gpu_path == ("CompositeDetector", "ops[0]", "ops[0]")
    assert plan.stage2_prefix == []


def test_a_top_level_detector_needs_no_prefix():
    """The case the spike got wrong (spec §4.2).

    The spike's ``branch_prefix`` lacks a root guard on its final block, so for
    a top-level detector it returns every preceding top-level op -- ops Stage 1
    has ALREADY applied and written to the store, which Stage 2 would then
    re-run on top of themselves. Nothing pinned this because all three spike
    shapes nest.
    """
    pipe = ImagePipeline(
        ops={
            "BlurGauss": BlurGauss(sigma=2.0),
            "SubtractGaussian": SubtractGaussian(sigma=50.0),
            "FakeGpuDetector": FakeGpuDetector(),
        }
    )
    plan = split_pipeline_at_gpu(pipe)

    assert plan.gpu_path == ("FakeGpuDetector",)
    assert plan.stage2_prefix == []


def test_a_top_level_nested_pipeline_contributes_its_own_preceding_ops():
    """The ancestor is itself a sequence container, so its head ops HAVE NOT run.

    Stage 1 runs only the ops before ``"branch"``; ``"branch"`` is in
    ``post_pipeline``, so everything inside it ahead of the detector is Stage
    2's to apply. This is the one shape where the root guard must NOT suppress
    a sequence container's contribution.
    """
    branch = ImagePipeline(
        ops={
            "ContrastStretching": ContrastStretching(input_layer="detect_mat"),
            "FakeGpu": FakeGpuDetector(),
        }
    )
    pipe = ImagePipeline(ops={"BlurGauss": BlurGauss(sigma=2.0), "branch": branch})
    plan = split_pipeline_at_gpu(pipe)

    assert plan.gpu_path == ("branch", "FakeGpu")
    assert list(plan.pre_pipeline.get_ops()) == ["BlurGauss"]
    assert list(plan.post_pipeline.get_ops()) == ["branch"]
    assert [type(op).__name__ for op in plan.stage2_prefix] == ["ContrastStretching"]


def test_a_nested_branch_does_not_pick_up_root_level_ops():
    """The root guard must survive the presence of a deeper sequence container."""
    branch = ImagePipeline(
        ops={
            "ContrastStretching": ContrastStretching(input_layer="detect_mat"),
            "FakeGpu": FakeGpuDetector(),
        }
    )
    pipe = ImagePipeline(
        ops={
            "BlurGauss": BlurGauss(sigma=2.0),
            "SubtractGaussian": SubtractGaussian(sigma=50.0),
            "C": CompositeDetector(ops=[branch, _manual()], mode="overlap"),
        }
    )
    plan = split_pipeline_at_gpu(pipe)

    # BlurGauss/SubtractGaussian are Stage 1's and are already in the store.
    assert [type(op).__name__ for op in plan.stage2_prefix] == ["ContrastStretching"]


# --------------------------------------------------------------------------
# The plot guard
# --------------------------------------------------------------------------


def test_a_plot_referencing_the_ancestor_is_now_allowed():
    """The ancestor runs in Stage 3, so a plot may reference it.

    The old guard refused ``ref.key == gpu_key``; under the new cut that key is
    the ANCESTOR and lives in ``post_pipeline``.
    """
    pipe = ImagePipeline(
        ops={
            "CompositeDetector": _PlottingComposite(
                ops=[FakeGpuDetector(), _manual()], mode="overlap"
            )
        }
    )
    plan = split_pipeline_at_gpu(_with_plot_on(pipe, "CompositeDetector"))
    assert "CompositeDetector" in plan.post_pipeline.get_ops()


def test_a_plot_referencing_a_pre_gpu_op_goes_to_stage_one():
    """Stage 1 applies it, so Stage 1 draws it (figures spec §3a)."""
    pipe = ImagePipeline(
        ops={
            "BlurGauss": _PreGpuPlot(sigma=2.0),
            "CompositeDetector": CompositeDetector(
                ops=[FakeGpuDetector(), _manual()], mode="overlap"
            ),
        }
    )
    plan = split_pipeline_at_gpu(_with_plot_on(pipe, "BlurGauss"))
    assert [b.id for b in plan.pre_pipeline.get_plots()] == ["BlurGauss"]
    assert plan.post_pipeline.get_plots() == []


def test_a_plot_on_a_TOP_LEVEL_gpu_detector_is_still_refused():
    """The disjunct that ``ref.key in pre_ops`` alone would silently drop.

    When the detector is top-level, ``gpu_path[0]`` IS the detector and now
    lives in ``post_ops`` -- but Stage 3 never runs it, and the substituted
    ``ReplayDetector`` is not plot-capable. Reducing the guard to
    ``ref.key in pre_ops`` would let this through.
    """
    pipe = ImagePipeline(ops={"gpu": _PlottingGpu()})
    with pytest.raises(ValueError, match="references the GPU detector 'gpu'"):
        split_pipeline_at_gpu(_with_plot_on(pipe, "gpu"))


# --------------------------------------------------------------------------
# Refusals, now tree-wide
# --------------------------------------------------------------------------


def test_no_gpu_detector_still_raises():
    pipe = ImagePipeline(
        ops={"CompositeDetector": CompositeDetector(ops=[_manual()], mode="union")}
    )
    with pytest.raises(ValueError, match="no GpuDetector"):
        split_pipeline_at_gpu(pipe)


def test_two_detectors_NESTED_in_one_composite_are_refused():
    """The old top-level-only scan here missed this entirely."""
    pipe = ImagePipeline(
        ops={
            "C": CompositeDetector(
                ops=[FakeGpuDetector(), FakeGpuDetector()], mode="union"
            )
        }
    )
    with pytest.raises(UnstageableGpuDetectorError,
                       match="more than one GpuDetector"):
        split_pipeline_at_gpu(pipe)


def test_two_detectors_in_DIFFERENT_branches_are_refused():
    pipe = ImagePipeline(
        ops={
            "A": CompositeDetector(ops=[FakeGpuDetector(), _manual()],
                                   mode="union"),
            "B": CompositeDetector(ops=[_manual(), FakeGpuDetector()],
                                   mode="union"),
        }
    )
    with pytest.raises(UnstageableGpuDetectorError,
                       match="more than one GpuDetector"):
        split_pipeline_at_gpu(pipe)
