"""`pipeline_requires_gpu` sees the whole operation tree, and refuses loudly.

Before Task 3 it looked only at the ROOT pipeline's top-level ops, so a
``GpuDetector`` nested inside a ``CompositeDetector`` routed to the non-staged
CPU path and silently ran per-image inference -- a wrong-answer bug, not a slow
one.

Every test here drives the **production** entry point where it can. A test that
calls ``find_gpu_detectors`` directly still passes when the refusal is
unreachable from production, which is exactly the defect this file exists to
prevent: the "composition primitives only" narrowing was implemented once
before and lost, because its only caller was a prefix builder that runs after
routing has already happened.
"""

import pytest

import phenotypic
from phenotypic import ImagePipeline
from phenotypic._cli._cli_validation import (
    _CHILD_CONTRACT,
    UnstageableGpuDetectorError,
    _populate_child_contract,
    find_gpu_detectors,
    pipeline_requires_gpu,
)
from phenotypic.detect import CompositeDetector, ManualPointDetector
from tests._fakes.fake_gpu_detector import FakeGpuDetector


@pytest.fixture(autouse=True)
def _register_fake_gpu_detector(monkeypatch):
    """from_json resolves classes by bare name in the phenotypic namespace."""
    monkeypatch.setattr(
        phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False
    )


CENTERS = [[10.0, 10.0], [10.0, 40.0]]


def _cpu_detector():
    return ManualPointDetector(centers=CENTERS, shape="disk", width=11)


def _write(tmp_path, pipeline):
    path = tmp_path / "pipeline.json"
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return path


def _nested_gpu_pipeline():
    return ImagePipeline(
        ops={
            "CompositeDetector": CompositeDetector(
                ops=[FakeGpuDetector(), _cpu_detector()], mode="overlap"
            )
        }
    )


def test_a_nested_gpu_detector_is_detected(tmp_path):
    assert pipeline_requires_gpu(_write(tmp_path, _nested_gpu_pipeline())) is True


def test_the_detected_path_addresses_the_branch(tmp_path):
    hits = find_gpu_detectors(
        ImagePipeline.from_json(_write(tmp_path, _nested_gpu_pipeline()))
    )
    assert [p for p, _ in hits] == [("CompositeDetector", "ops[0]")]


def test_a_top_level_gpu_detector_is_still_detected(tmp_path):
    """The pre-Task-3 shape keeps working, and its path has one segment."""
    pipe = ImagePipeline(ops={"FakeGpuDetector": FakeGpuDetector()})
    path = _write(tmp_path, pipe)
    assert pipeline_requires_gpu(path) is True
    hits = find_gpu_detectors(ImagePipeline.from_json(path))
    assert [p for p, _ in hits] == [("FakeGpuDetector",)]


def test_a_cpu_only_pipeline_is_still_false(tmp_path):
    pipe = ImagePipeline(
        ops={
            "CompositeDetector": CompositeDetector(
                ops=[_cpu_detector()], mode="union"
            )
        }
    )
    assert pipeline_requires_gpu(_write(tmp_path, pipe)) is False


def test_two_gpu_detectors_anywhere_are_refused(tmp_path):
    pipe = ImagePipeline(
        ops={
            "CompositeDetector": CompositeDetector(
                ops=[FakeGpuDetector(), FakeGpuDetector()], mode="union"
            )
        }
    )
    with pytest.raises(UnstageableGpuDetectorError, match="more than one"):
        find_gpu_detectors(
            ImagePipeline.from_json(_write(tmp_path, pipe)), strict=True
        )


def test_two_gpu_detectors_are_not_refused_without_strict(tmp_path):
    """The GUI asks "is this a GPU pipeline?", not "can this be split?".

    ``gui/run_console/_callbacks.py:253`` calls the non-strict path, where a
    multi-detector pipeline must report True rather than raise -- the
    multi-detector refusal belongs to ``split_pipeline_at_gpu``.
    """
    pipe = ImagePipeline(
        ops={
            "CompositeDetector": CompositeDetector(
                ops=[FakeGpuDetector(), FakeGpuDetector()], mode="union"
            )
        }
    )
    assert pipeline_requires_gpu(_write(tmp_path, pipe)) is True


def _meas_slot_pipeline():
    from phenotypic.measure import MeasureSymZones

    return ImagePipeline(
        ops={"ManualPointDetector": _cpu_detector()},
        meas={"MeasureSymZones": MeasureSymZones(center_detector=FakeGpuDetector())},
    )


def test_a_gpu_detector_in_the_ROOT_meas_slot_is_refused(tmp_path):
    """Stage 3 runs measurers on a CPU node, so a GPU op there cannot stage.

    Scope is the **root** pipeline's ``meas`` only -- that is what
    ``_CPU_ONLY_SLOTS`` queries, since it calls ``get_meas()`` on the root
    object. A nested pipeline's own ``meas`` is a different case and is NOT
    covered; see
    ``test_a_gpu_detector_in_a_NESTED_pipelines_meas_slot_is_refused`` below.

    Drives ``pipeline_requires_gpu`` -- the PRODUCTION entry point -- not
    ``find_gpu_detectors(strict=True)``. A test that calls the helper directly
    passes even when the refusal is unreachable from production, which is
    exactly the bug this test exists to prevent.
    """
    with pytest.raises(UnstageableGpuDetectorError, match="meas"):
        pipeline_requires_gpu(_write(tmp_path, _meas_slot_pipeline()))


@pytest.mark.xfail(
    strict=False,
    reason=(
        "KNOWN GAP. `iter_child_operations` short-circuits on "
        "ImagePipelineCore (_operation_tree.py:41-43) and yields only "
        "get_ops(), and `_CPU_ONLY_SLOTS` queries the ROOT pipeline's "
        "accessors alone -- so a GpuDetector in a NESTED pipeline's own "
        "meas/post/filters/model is invisible to both halves of the scan. "
        "Measured: find_gpu_detectors returns [], pipeline_requires_gpu "
        "returns False, and uses_staged_gpu_strategy returns False, so the "
        "run routes to LocalParallelStrategy and the detector performs "
        "per-image inference on a CPU node with nothing reported. That is "
        "the same silent-CPU failure Task 3 exists to remove, surviving in a "
        "shape the walker cannot see. Not strict: when the walker learns to "
        "descend a nested pipeline's non-ops slots this XPASSes and should "
        "be unmarked rather than fail the suite."
    ),
)
def test_a_gpu_detector_in_a_NESTED_pipelines_meas_slot_is_refused(tmp_path):
    """The desired behaviour, pinned as a test rather than as prose.

    A gap recorded only in a docstring is one that gets closed by accident and
    reopened by accident. This asserts what *should* happen, so closing the
    gap turns the marker green instead of leaving nothing to notice.
    """
    from phenotypic.measure import MeasureSymZones

    pipe = ImagePipeline(
        ops={
            "inner": ImagePipeline(
                ops={"ManualPointDetector": _cpu_detector()},
                meas={
                    "MeasureSymZones": MeasureSymZones(
                        center_detector=FakeGpuDetector()
                    )
                },
            )
        }
    )
    with pytest.raises(UnstageableGpuDetectorError, match="meas"):
        pipeline_requires_gpu(_write(tmp_path, pipe))


def test_a_ROOT_meas_slot_gpu_detector_does_not_route_to_the_cpu_strategy(tmp_path):
    """The refusal must fire BEFORE strategy selection.

    Without it ``pipeline_requires_gpu`` returns False, the run routes to
    ``LocalParallelStrategy`` (``_cli_execution_strategies.py:1341``) and the
    GPU op runs on CPU -- the wrong-answer bug this change exists to kill.
    """
    from phenotypic._cli._cli_execution_strategies import uses_staged_gpu_strategy
    from phenotypic._cli._cli_types import ExecutionConfig

    path = _write(tmp_path, _meas_slot_pipeline())
    config = ExecutionConfig(
        pipeline_json=path, input_path=tmp_path, output_dir=tmp_path,
        image_type="Image", nrows=None, ncols=None, bit_depth=None,
        n_jobs=1, slurm_args={}, force_local=True, wait=False, ext=".tiff",
        overlay_alpha=0.5, include_dataset_column=False, dry_run=False,
        sample=None, resume=False, retry_failures=False, skip_validation=True,
        save_overlays=False, measure_only=False, process_only_layer=None,
    )
    with pytest.raises(UnstageableGpuDetectorError):
        uses_staged_gpu_strategy(config)


# ---------------------------------------------------------------------------
# Only composition primitives may carry a staged GpuDetector (spec 4.3).
# ---------------------------------------------------------------------------


def test_the_child_contract_table_holds_exactly_the_two_composites():
    """Coverage is asserted on the TABLE, not by enumerating the tree.

    The set is closed by rule, so a new container needs no entry and no
    decision -- it is refused by default, which is the correct answer for it.
    ``ImagePipeline`` is absent deliberately: it is matched by ``isinstance``,
    so a class key would refuse its subclasses.
    """
    from phenotypic.enhance import CompositeEnhance

    _populate_child_contract()
    assert set(_CHILD_CONTRACT) == {CompositeDetector, CompositeEnhance}


def test_a_gpu_detector_inside_a_composite_enhance_is_allowed(tmp_path):
    from phenotypic.enhance import CompositeEnhance, ContrastStretching

    pipe = ImagePipeline(
        ops={
            "CompositeEnhance": CompositeEnhance(
                ops=[FakeGpuDetector(), ContrastStretching()], mode="max"
            )
        }
    )
    path = _write(tmp_path, pipe)
    assert pipeline_requires_gpu(path) is True
    hits = find_gpu_detectors(ImagePipeline.from_json(path))
    assert [p for p, _ in hits] == [("CompositeEnhance", "ops[0]")]


def test_a_gpu_detector_inside_a_domain_detector_is_refused(tmp_path):
    """``FilamentousFungiDetector`` would classify as "parallel" today.

    It is refused anyway. That classification is incidental to an algorithm
    that also runs an inline ContrastStretching and a destructive background
    subtraction; nothing about being a fungus detector constrains it to keep
    handing its child the container's own image.
    """
    from phenotypic.detect import FilamentousFungiDetector

    pipe = ImagePipeline(
        ops={
            "FilamentousFungiDetector": FilamentousFungiDetector(
                inoculum_detector=FakeGpuDetector()
            )
        }
    )
    with pytest.raises(
        UnstageableGpuDetectorError, match="only composition primitives"
    ):
        pipeline_requires_gpu(_write(tmp_path, pipe))


def test_a_gpu_detector_inside_the_two_k_detector_is_refused(tmp_path):
    """Three fields, three different child inputs, inside one algorithm."""
    from phenotypic.detect import TwoKFilamentousDetector

    pipe = ImagePipeline(
        ops={
            "TwoKFilamentousDetector": TwoKFilamentousDetector(
                branch_base=FakeGpuDetector()
            )
        }
    )
    with pytest.raises(
        UnstageableGpuDetectorError, match="only composition primitives"
    ):
        pipeline_requires_gpu(_write(tmp_path, pipe))


def test_the_domain_detector_refusal_names_the_offending_class(tmp_path):
    """The message has to tell the user *which* container to lift out of."""
    from phenotypic.detect import FilamentousFungiDetector

    pipe = ImagePipeline(
        ops={
            "FilamentousFungiDetector": FilamentousFungiDetector(
                inoculum_detector=FakeGpuDetector()
            )
        }
    )
    with pytest.raises(
        UnstageableGpuDetectorError, match="FilamentousFungiDetector"
    ):
        pipeline_requires_gpu(_write(tmp_path, pipe))


def test_a_domain_detector_with_no_gpu_detector_is_untouched(tmp_path):
    """The refusal keys on a GpuDetector's placement, not on the class.

    A CPU-only pipeline containing a domain detector must stay a perfectly
    ordinary CPU run -- the narrowing forbids *staging* inside one, not the
    operation itself.
    """
    from phenotypic.detect import FilamentousFungiDetector

    pipe = ImagePipeline(
        ops={"FilamentousFungiDetector": FilamentousFungiDetector()}
    )
    assert pipeline_requires_gpu(_write(tmp_path, pipe)) is False
