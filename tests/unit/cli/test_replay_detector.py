"""``ReplayDetector``: recorded mask in, wrapped detector's identity out.

Built in memory, never round-tripped through JSON, so no registration fixture
is needed here. ``threshold`` is a real field on the shared fake and is what the
parameter-delegation assertion below keys on.
"""

import numpy as np
import pytest

from phenotypic import ImagePipeline
from phenotypic._cli._cli_replay_detector import ReplayDetector
from phenotypic.abc_ import ObjectDetector
from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import BlurGauss
from tests._fakes.fake_gpu_detector import FakeGpuDetector


def test_replay_writes_the_recorded_array():
    image = load_synth_yeast_plate()
    recorded = np.zeros(image.gray[:].shape, dtype=np.uint16)
    recorded[20:60, 20:60] = 1
    recorded[120:160, 120:160] = 2

    detector = FakeGpuDetector(
        drop_frame_background=False, split_disconnected_labels=False
    )
    ReplayDetector(detector=detector, result=recorded).apply(image, inplace=True)

    assert image.num_objects == 2


def test_replay_applies_the_detectors_post_inference_cleanup():
    """``_write_object_output`` owns ``drop_frame_background`` /
    ``split_disconnected_labels``; the stub must delegate to it rather than
    assigning ``objmap`` itself."""
    image = load_synth_yeast_plate()
    recorded = np.zeros(image.gray[:].shape, dtype=np.uint16)
    recorded[:] = 9  # a background-spanning label
    recorded[20:60, 20:60] = 1

    detector = FakeGpuDetector(
        drop_frame_background=True, split_disconnected_labels=True
    )
    ReplayDetector(detector=detector, result=recorded).apply(image, inplace=True)

    assert image.num_objects == 1, "frame background was not dropped"


def test_a_semantic_detectors_result_goes_to_objmask():
    """``output_kind`` is read off the WRAPPED detector, not assumed."""
    image = load_synth_yeast_plate()
    recorded = np.zeros(image.gray[:].shape, dtype=bool)
    recorded[20:60, 20:60] = True
    recorded[120:160, 120:160] = True

    detector = FakeGpuDetector(output_kind="semantic")
    ReplayDetector(detector=detector, result=recorded).apply(image, inplace=True)

    assert image.num_objects == 2
    assert image.objmask[:].sum() == recorded.sum()


def test_the_stub_is_an_object_detector():
    """So it can be substituted into any slot the real detector occupied."""
    stub = ReplayDetector(
        detector=FakeGpuDetector(), result=np.zeros((4, 4), np.uint16)
    )
    assert isinstance(stub, ObjectDetector)


def test_the_stub_is_keyword_only_like_every_other_operation():
    with pytest.raises(TypeError):
        ReplayDetector(FakeGpuDetector(), np.zeros((4, 4), np.uint16))


# --------------------------------------------------------------------------
# Provenance identity delegation
# --------------------------------------------------------------------------


def test_provenance_identity_is_the_wrapped_detector():
    """The journal must name the real detector, not ``ReplayDetector``, or a
    staged run's provenance stops matching a single-pass run's."""
    detector = FakeGpuDetector(threshold=0.37)
    stub = ReplayDetector(detector=detector, result=np.zeros((4, 4), np.uint16))

    assert stub.provenance_operation_class().endswith("FakeGpuDetector")
    assert stub.provenance_operation_name() == "FakeGpuDetector"
    assert stub.provenance_parameters()["threshold"] == 0.37


def test_provenance_parameters_do_not_carry_the_recorded_objmap():
    """The stub holds an ``NdArrayField``; the DEFAULT ``model_dump`` would
    serialise the whole recorded map into the journal."""
    stub = ReplayDetector(
        detector=FakeGpuDetector(), result=np.arange(64, dtype=np.uint16).reshape(8, 8)
    )
    parameters = stub.provenance_parameters()

    assert "result" not in parameters
    assert "detector" not in parameters
    assert "detector_duration_seconds" not in parameters


def test_the_stub_carries_the_stage2_duration():
    """The stub's own wall time is the MERGE only; GPU cost lives in the token.

    ``tests/integration/cli/test_staged_store_stages.py:128`` asserts the
    recorded duration is >= the token's ``detector_duration_seconds``.
    """
    stub = ReplayDetector(
        detector=FakeGpuDetector(),
        result=np.zeros((4, 4), np.uint16),
        detector_duration_seconds=12.5,
    )
    assert stub.provenance_duration_offset() == 12.5


def test_the_duration_offset_defaults_to_zero():
    stub = ReplayDetector(
        detector=FakeGpuDetector(), result=np.zeros((4, 4), np.uint16)
    )
    assert stub.provenance_duration_offset() == 0.0


# --------------------------------------------------------------------------
# The hooks are only useful if `append_operation_provenance` reads them
# --------------------------------------------------------------------------


def _journal_entries(image):
    return [
        entry
        for application in image._metadata.provenance_journal["applications"]
        for entry in application["operations"]
    ]


def test_the_journal_records_the_wrapped_detector_not_the_stub():
    """End-to-end through ``append_operation_provenance``.

    The four methods on the stub buy nothing until the journal writer prefers
    them, so assert the recorded entry rather than the stub's own return
    values.
    """
    image = load_synth_yeast_plate()
    recorded = np.zeros(image.gray[:].shape, dtype=np.uint16)
    recorded[20:60, 20:60] = 1

    stub = ReplayDetector(
        detector=FakeGpuDetector(threshold=0.37, drop_frame_background=False),
        result=recorded,
        detector_duration_seconds=12.5,
    )
    result = ImagePipeline(ops={"Sam": stub}).apply(image)

    entry = _journal_entries(result)[-1]
    assert entry["operation_name"] == "FakeGpuDetector"
    assert entry["operation_class"].endswith("FakeGpuDetector")
    assert entry["parameters"]["threshold"] == 0.37
    assert "result" not in entry["parameters"], "the recorded objmap leaked"
    # own merge time PLUS the Stage-2 inference time carried on the stub
    assert entry["duration_seconds"] >= 12.5


def test_an_ordinary_operation_still_records_its_own_identity():
    """Control: the hooks are opt-in and the fallback path is unchanged.

    Without this, the journal-writer edit could break every operation in the
    codebase and only the stub's own tests would notice.
    """
    result = ImagePipeline(ops={"Blur": BlurGauss(sigma=2.0)}).apply(
        load_synth_yeast_plate()
    )

    entry = _journal_entries(result)[-1]
    assert entry["operation_name"] == "BlurGauss"
    assert entry["operation_class"].endswith("BlurGauss")
    assert entry["parameters"]["sigma"] == 2.0
    assert entry["duration_seconds"] >= 0.0
