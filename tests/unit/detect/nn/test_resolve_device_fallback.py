"""``device="auto"`` falls back to CPU when no accelerator exists.

Every GPU detector defaults to ``device="auto"`` and resolves it through
``resolve_device(self.device)``. These tests pin that a machine without a GPU
gets a CPU run with a warning rather than a ``RuntimeError``, while an
explicitly requested accelerator still refuses. A fake ``torch``
(``tests/_fakes/fake_torch.py``) decides which accelerators exist, so the
tests run without PyTorch and do not depend on the host's hardware.
"""

from __future__ import annotations

import logging
import sys
import types
import warnings

import pytest

from phenotypic.detect.nn._helper._checkpoint_manager import resolve_device
from tests._fakes.fake_torch import install_fake_torch


class TestAutoResolution:
    def test_auto_without_an_accelerator_falls_back_to_cpu(
        self, monkeypatch, caplog
    ):
        install_fake_torch(monkeypatch)

        with caplog.at_level(logging.WARNING), pytest.warns(
            UserWarning, match="falling back to CPU"
        ):
            assert resolve_device("auto") == "cpu"
        # The log line is what survives inside a SLURM or loky worker.
        assert any("falling back to CPU" in r.message for r in caplog.records)

    def test_auto_prefers_an_available_accelerator_silently(self, monkeypatch):
        install_fake_torch(monkeypatch, cuda=True)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_device("auto") == "cuda"

    def test_auto_probes_mps_when_cuda_is_absent(self, monkeypatch):
        install_fake_torch(monkeypatch, mps=True)

        assert resolve_device("auto") == "mps"

    def test_allow_cpu_false_still_demands_an_accelerator(self, monkeypatch):
        install_fake_torch(monkeypatch)

        with pytest.raises(RuntimeError, match="No accelerator available"):
            resolve_device("auto", allow_cpu=False)


class TestExplicitDevice:
    @pytest.mark.parametrize("device", ["cuda", "mps", "xpu"])
    def test_an_unavailable_explicit_accelerator_is_never_replaced(
        self, monkeypatch, device
    ):
        install_fake_torch(monkeypatch)

        with pytest.raises(RuntimeError, match=f"device='{device}' requested"):
            resolve_device(device)

    def test_explicit_cpu_resolves_without_warning(self, monkeypatch):
        install_fake_torch(monkeypatch)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_device("cpu") == "cpu"


class TestDetectorsReachTheFallback:
    """The detectors pass ``self.device`` through, so they inherit the default."""

    def test_sam2_builds_its_generator_on_cpu(self, monkeypatch):
        from phenotypic.detect.nn import Sam2
        from phenotypic.detect.nn import _sam2 as sam2_mod

        install_fake_torch(monkeypatch)
        seen: dict = {}
        monkeypatch.setattr(
            sam2_mod,
            "build_sam2_generator",
            lambda *args, **kwargs: seen.update(kwargs) or object(),
        )

        with pytest.warns(UserWarning, match="falling back to CPU"):
            Sam2()._ensure_model_loaded()

        assert seen["device"] == "cpu"

    def test_microsam_builds_its_predictor_on_cpu(self, monkeypatch):
        from phenotypic.detect.nn import MicroSamDetector

        install_fake_torch(monkeypatch)
        seen: dict = {}
        auto_seg = types.ModuleType("micro_sam.automatic_segmentation")
        auto_seg.get_predictor_and_segmenter = (
            lambda **kwargs: seen.update(kwargs) or (object(), object())
        )
        monkeypatch.setitem(sys.modules, "micro_sam", types.ModuleType("micro_sam"))
        monkeypatch.setitem(
            sys.modules, "micro_sam.automatic_segmentation", auto_seg
        )

        with pytest.warns(UserWarning, match="falling back to CPU"):
            MicroSamDetector()._ensure_model_loaded()

        assert seen["device"] == "cpu"

    def test_an_explicit_cuda_detector_still_refuses(self, monkeypatch):
        from phenotypic.detect.nn import Sam2
        from phenotypic.detect.nn import _sam2 as sam2_mod

        install_fake_torch(monkeypatch)
        monkeypatch.setattr(
            sam2_mod, "build_sam2_generator", lambda *a, **k: object()
        )

        with pytest.raises(RuntimeError, match="device='cuda' requested"):
            Sam2(device="cuda")._ensure_model_loaded()
