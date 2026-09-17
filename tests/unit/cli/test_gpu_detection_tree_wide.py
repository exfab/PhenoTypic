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

from typing import Union

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
from phenotypic.analysis import LogGrowthModel, TukeyOutlierRemover
from phenotypic.detect import CompositeDetector, ManualPointDetector
from phenotypic.post import AppendString
from phenotypic.sdk_.typing_ import OperationField
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

    ``_gui/run_console/_callbacks.py:_staged_gpu_capability`` calls the non-strict path, where a
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

    Drives ``pipeline_requires_gpu`` -- the PRODUCTION entry point -- not
    ``find_gpu_detectors(strict=True)``. A test that calls the helper directly
    passes even when the refusal is unreachable from production, which is
    exactly the bug this test exists to prevent.

    The match pins the slot-namespaced path spelling AND the slot refusal's
    own wording, not just ``"meas"``: the ancestor-contract refusal also
    fires for this shape (MeasureSymZones is not a composition primitive), and
    a bare ``"meas"`` could not tell which of the two refused.
    """
    with pytest.raises(
        UnstageableGpuDetectorError,
        match=(
            r"GpuDetector at meas:MeasureSymZones/center_detector cannot be "
            r"staged: it sits in the 'meas' slot of the root pipeline"
        ),
    ):
        pipeline_requires_gpu(_write(tmp_path, _meas_slot_pipeline()))


def test_a_gpu_detector_in_a_NESTED_pipelines_meas_slot_is_refused(tmp_path):
    """Was an xfail (non-strict) until the walker descended nested slots.

    Before, ``walk_operations`` yielded only a nested pipeline's ``ops``, so
    this shape was neither staged nor refused and routed to the CPU strategy.
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
    with pytest.raises(
        UnstageableGpuDetectorError,
        match=(
            r"GpuDetector at inner/meas:MeasureSymZones/center_detector cannot "
            r"be staged: it sits in the 'meas' slot of the pipeline at inner"
        ),
    ):
        pipeline_requires_gpu(_write(tmp_path, pipe))


# ---------------------------------------------------------------------------
# Every CPU-only slot, at the root and nested. `meas` carries its detector in a
# real measurer; `post`/`filters`/`model` have no shipped class with an
# operation-valued field, so each uses a carrier subclass of a real one.
# ---------------------------------------------------------------------------


class PostCarryingDetector(AppendString):
    """A real ``PostMeasurement`` with an operation-valued field."""

    detector: Union[OperationField, None] = None  # type: ignore[valid-type]


class FilterCarryingDetector(TukeyOutlierRemover):
    """A real ``SetAnalyzer`` with an operation-valued field."""

    detector: Union[OperationField, None] = None  # type: ignore[valid-type]


class ModelCarryingDetector(LogGrowthModel):
    """A real ``ModelFitter`` with an operation-valued field."""

    detector: Union[OperationField, None] = None  # type: ignore[valid-type]


@pytest.fixture
def _register_slot_carriers(monkeypatch):
    for cls in (PostCarryingDetector, FilterCarryingDetector, ModelCarryingDetector):
        monkeypatch.setattr(phenotypic, cls.__name__, cls, raising=False)


#: slot -> (segment the slot entry is addressed by, field holding the detector)
_SLOT_ADDRESS = {
    "meas": ("meas:MeasureSymZones", "center_detector"),
    "post": ("post:PostCarryingDetector", "detector"),
    "filters": ("filters:FilterCarryingDetector", "detector"),
    "model": ("model:ModelCarryingDetector", "detector"),
}


def _slot_kwargs(slot, detector):
    from phenotypic.measure import MeasureSymZones

    if slot == "meas":
        return {"meas": {"MeasureSymZones": MeasureSymZones(center_detector=detector)}}
    if slot == "post":
        return {"post": {"PostCarryingDetector": PostCarryingDetector(
            column="Temperature", value="C", detector=detector)}}
    if slot == "filters":
        return {"filters": {"FilterCarryingDetector": FilterCarryingDetector(
            on="Size_Area", groupby=["Metadata_Plate"], detector=detector)}}
    return {"model": ModelCarryingDetector(
        on="Size_Area", groupby=["Metadata_Plate"], detector=detector)}


def _slot_pipeline(slot, *, nested):
    inner = ImagePipeline(
        ops={"ManualPointDetector": _cpu_detector()},
        **_slot_kwargs(slot, FakeGpuDetector()),
    )
    return ImagePipeline(ops={"inner": inner}) if nested else inner


def _detector_in_slot(pipeline, slot, *, nested):
    """The detector as it sits after a JSON round trip -- the test premise."""
    owner = pipeline.get_ops()["inner"] if nested else pipeline
    entry = owner.get_model() if slot == "model" else next(
        iter(getattr(owner, f"get_{slot}")().values())
    )
    return getattr(entry, _SLOT_ADDRESS[slot][1])


@pytest.mark.usefixtures("_register_slot_carriers")
@pytest.mark.parametrize("nested", [False, True], ids=["ROOT", "NESTED"])
@pytest.mark.parametrize("slot", ["meas", "post", "filters", "model"])
def test_a_gpu_detector_in_every_cpu_only_slot_is_refused(tmp_path, slot, nested):
    """One case per slot, per depth, through the production entry point.

    Premise first: the detector must survive ``to_json``/``from_json`` in the
    slot, or a refusal test on a pipeline with no GpuDetector passes vacuously.
    """
    path = _write(tmp_path, _slot_pipeline(slot, nested=nested))
    assert isinstance(
        _detector_in_slot(ImagePipeline.from_json(path), slot, nested=nested),
        FakeGpuDetector,
    )

    segment, field = _SLOT_ADDRESS[slot]
    prefix = "inner/" if nested else ""
    where = "the pipeline at inner" if nested else "the root pipeline"
    with pytest.raises(UnstageableGpuDetectorError) as caught:
        pipeline_requires_gpu(path)
    message = str(caught.value)
    assert message.startswith(
        f"GpuDetector at {prefix}{segment}/{field} cannot be staged: "
        f"it sits in the '{slot}' slot of {where}"
    ), message
    # Refusal ORDER: the ancestor-contract check would also refuse this
    # shape, blaming the slot entry's class instead of the slot.
    assert "cannot be nested inside" not in message


@pytest.mark.parametrize("nested", [False, True], ids=["ROOT", "NESTED"])
def test_a_slot_entry_that_is_also_a_composite_is_refused(nested):
    """The one shape ONLY the slot refusal catches.

    Every shipped slot type (MeasureFeatures, PostMeasurement, SetAnalyzer,
    ModelFitter) is refused by the ancestor-contract check anyway, with a
    misleading "cannot be nested inside" message. A class that is BOTH a slot
    type and a composition primitive passes that check -- it inherits
    ``CompositeDetector._operate`` -- so without the slot refusal its detector
    is an ordinary hit: staged, run by Stage 2 on the stored image, and
    replayed. This test fails by NO RAISE AT ALL if the slot refusal goes.

    Calls ``find_gpu_detectors`` on the live pipeline rather than through
    JSON: the local class is not in the ``phenotypic`` namespace, and the
    refusal under test does not depend on serialization.
    """
    from phenotypic._cli._cli_validation import _child_contract
    from phenotypic.abc_ import MeasureFeatures

    class MeasuringComposite(CompositeDetector, MeasureFeatures):
        pass

    entry = MeasuringComposite(ops=[FakeGpuDetector()], mode="union")
    # Premise: the ancestor check ADMITS this entry, so it cannot be what
    # makes the test pass.
    assert _child_contract(entry) == "parallel"

    inner = ImagePipeline(meas={"MeasuringComposite": entry})
    pipe = ImagePipeline(ops={"inner": inner}) if nested else inner
    prefix = "inner/" if nested else ""
    with pytest.raises(
        UnstageableGpuDetectorError,
        match=(
            rf"GpuDetector at {prefix}meas:MeasuringComposite/ops\[0\] cannot "
            r"be staged: it sits in the 'meas' slot"
        ),
    ):
        find_gpu_detectors(pipe)


@pytest.mark.usefixtures("_register_slot_carriers")
def test_the_same_nested_pipeline_is_staged_when_the_detector_is_in_its_ops(
    tmp_path,
):
    """Control: the refusal is about the SLOT, not about nesting.

    Same nested-pipeline shape as the slot cases, with the detector moved into
    the inner pipeline's ``ops``: an ordinary staged hit.
    """
    pipe = ImagePipeline(
        ops={
            "inner": ImagePipeline(
                ops={
                    "ManualPointDetector": _cpu_detector(),
                    "FakeGpuDetector": FakeGpuDetector(),
                },
                **_slot_kwargs("post", _cpu_detector()),
            )
        }
    )
    path = _write(tmp_path, pipe)
    assert pipeline_requires_gpu(path) is True
    hits = find_gpu_detectors(ImagePipeline.from_json(path))
    assert [p for p, _ in hits] == [("inner", "FakeGpuDetector")]


def test_an_ops_key_spelled_like_a_slot_segment_is_not_a_slot(tmp_path):
    """Slot membership is decided on the live pipeline, not by the string.

    A user may key an op ``"meas:Fake"``. That detector is in ``ops`` and must
    stage normally rather than be refused as a ``meas`` entry.
    """
    pipe = ImagePipeline(ops={"meas:Fake": FakeGpuDetector()})
    path = _write(tmp_path, pipe)
    assert pipeline_requires_gpu(path) is True
    hits = find_gpu_detectors(ImagePipeline.from_json(path))
    assert [p for p, _ in hits] == [("meas:Fake",)]


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
