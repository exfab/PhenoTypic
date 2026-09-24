"""Environment checks of the run preflight: packages, licenses, cached weights.

Spec ``2026-09-24-cli-preflight`` §5 (F9, F11, F12). ``find_spec`` and the cache
probes are patched so the results do not depend on what this environment has
installed.
"""

from __future__ import annotations

import importlib.util

import pytest

from phenotypic import ImagePipeline
from phenotypic._cli import _cli_preflight
from phenotypic._cli._cli_preflight import (
    check_model_licenses,
    check_model_weights_cached,
    check_optional_modules,
)
from phenotypic.abc_ import OperationRequirements, WeightRequirement
from phenotypic.detect import OtsuDetector
from phenotypic.detect.nn import Insid3Detector, MicroSamDetector, Sam2
from phenotypic.measure import MeasureSize
from tests.unit.cli._preflight_support import make_context


def _absent(*names: str):
    real = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        return None if name in names else real(name, *args, **kwargs)

    return find_spec


def test_a_missing_package_is_an_error_naming_its_extra(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _absent("sam2", "torch"))
    pipeline = ImagePipeline(ops={"sam": Sam2()}, meas={"s": MeasureSize()})

    findings = check_optional_modules(make_context(pipeline))

    assert {f.code for f in findings} == {"PF-MISSING-MODULE"}
    assert all(f.severity == "error" for f in findings)
    assert any("'sam2'" in f.message and "'torch' extra" in f.message for f in findings)


def test_micro_sam_names_no_extra(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", _absent("micro_sam"))
    pipeline = ImagePipeline(ops={"ms": MicroSamDetector()})

    (finding,) = check_optional_modules(make_context(pipeline))

    assert "extra" not in finding.message
    assert "conda" in finding.hint


def test_installed_packages_produce_nothing(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda *a, **k: object())
    pipeline = ImagePipeline(ops={"sam": Sam2()})

    assert check_optional_modules(make_context(pipeline)) == []


def test_a_detector_out_of_scope_needs_nothing(monkeypatch) -> None:
    """``measure`` mode never applies ops, so an ops detector's packages are moot."""
    monkeypatch.setattr(importlib.util, "find_spec", _absent("sam2", "torch"))
    pipeline = ImagePipeline(ops={"sam": Sam2()}, meas={"s": MeasureSize()})

    assert check_optional_modules(make_context(pipeline, "measure")) == []


def test_an_unaccepted_gated_license_is_an_error(monkeypatch) -> None:
    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    pipeline = ImagePipeline(ops={"insid3": Insid3Detector()})

    (finding,) = check_model_licenses(make_context(pipeline))

    assert finding.code == "PF-LICENSE" and finding.severity == "error"
    assert "'dinov3'" in finding.message


def test_an_accepted_license_passes(monkeypatch) -> None:
    monkeypatch.setenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", "sam3, DINOv3")
    pipeline = ImagePipeline(ops={"insid3": Insid3Detector()})

    assert check_model_licenses(make_context(pipeline)) == []


class _Weighted(OtsuDetector):
    """An ordinary detector that declares one weight with a controllable probe."""

    def preflight_requirements(self):
        return OperationRequirements(
            weights=(
                WeightRequirement(
                    model="demo:weights", license_key=None, is_cached=_Weighted.state
                ),
            )
        )

    state = staticmethod(lambda: None)


@pytest.mark.parametrize(("cached", "codes"), [(False, ["PF-WEIGHTS-UNCACHED"]), (True, []), (None, [])])
def test_uncached_weights_warn_and_unknown_is_silent(monkeypatch, cached, codes) -> None:
    monkeypatch.setattr(_Weighted, "state", staticmethod(lambda: cached))
    pipeline = ImagePipeline(ops={"w": _Weighted()})

    findings = check_model_weights_cached(make_context(pipeline))

    assert [f.code for f in findings] == codes
    assert all(f.severity == "warning" for f in findings)


def test_every_environment_check_is_registered() -> None:
    for check in (check_optional_modules, check_model_licenses, check_model_weights_cached):
        assert check in _cli_preflight.CHECKS
