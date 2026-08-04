"""Foundation-detector exports + annotation coverage (Spec 2a, Task 7 / S1).

The annotation-coverage gate (``tests/unit/tune/test_annotation_coverage.py``)
iterates ``phenotypic.detect.__all__`` (top level), so the ``nn`` subpackage's
GPU detectors are out of its denominator by the existing design — ``Sam2``
is not in the gate either. This module pins the meaningful S1 guarantee directly:
both new detectors are exported from ``detect/nn/__init__.__all__`` and EVERY
numeric (int/float) field on them carries a ``TuneSpec`` (search window or
``tunable=False``), so they stay tune-ready regardless of the gate's scope.
"""

import typing

from phenotypic.detect.nn import (
    DinoSam2Detector,
    FssDinoDetector,
    Insid3Detector,
    Sam3,
)
from phenotypic.sdk_.typing_ import TuneSpec


def _walk_metadata(annotation) -> list:
    found = list(getattr(annotation, "__metadata__", ()))
    for arg in typing.get_args(annotation):
        found.extend(_walk_metadata(arg))
    return found


def _core_type(annotation):
    if typing.get_origin(annotation) is typing.Annotated:
        return typing.get_args(annotation)[0]
    return annotation


def _numeric_fields(cls):
    for name, field_info in cls.model_fields.items():
        core = _core_type(field_info.annotation)
        if core in (int, float):
            yield name, field_info


def test_detectors_exported_from_nn_all():
    import phenotypic.detect.nn as nn

    assert "Sam3" in nn.__all__
    assert "DinoSam2Detector" in nn.__all__
    assert "Insid3Detector" in nn.__all__
    assert "FssDinoDetector" in nn.__all__
    assert nn.Sam3 is Sam3
    assert nn.DinoSam2Detector is DinoSam2Detector
    assert nn.Insid3Detector is Insid3Detector
    assert nn.FssDinoDetector is FssDinoDetector


def test_every_numeric_field_carries_a_tune_spec():
    # W1: the annotation-coverage gate is scoped to detect.__all__ (not
    # detect.nn), so pin the Spec 2a + 2b GPU detectors' tune-readiness here.
    for cls in (Sam3, DinoSam2Detector, Insid3Detector, FssDinoDetector):
        for name, field_info in _numeric_fields(cls):
            metadata = list(field_info.metadata) + _walk_metadata(
                    field_info.annotation
            )
            assert any(isinstance(m, TuneSpec) for m in metadata), (
                f"{cls.__name__}.{name} (numeric) lacks a TuneSpec annotation"
            )


def test_semantic_detectors_set_semantic_output_kind():
    # Spec 2b: both new detectors are semantic → write objmask (Spec 1 §8).
    for cls in (Insid3Detector, FssDinoDetector):
        det = cls()
        assert det.output_kind == "semantic"
        assert det.input_layer == "rgb"
        assert det.supports_batching is False
