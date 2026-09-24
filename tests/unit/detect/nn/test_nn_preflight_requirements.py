"""GPU detectors declare packages and weights; none prompts inside a pipeline.

Spec ``2026-09-24-cli-preflight`` §3, §5, §10.3 (F9-F12; review R8, R14, R17).
The optional packages (``transformers``, ``torch``, ...) may be absent here, so
the runtime-fix tests inject fake modules instead of skipping: a test that
must fail on the base commit may not skip for want of a dependency.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from phenotypic.detect.nn import (
    DinoSam2Detector,
    FssDinoDetector,
    Insid3Detector,
    MicroSamDetector,
    Sam2,
    Sam3,
)
from phenotypic.detect.nn._helper import _checkpoint_manager as ckpt


# --- runtime fixes (§10.3) ------------------------------------------------------


def test_sam3_refuses_an_unaccepted_license_before_loading(monkeypatch) -> None:
    """F11: the load called from_pretrained with no PhenoTypic gate."""
    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("from_pretrained reached before the license gate")

    fake = types.ModuleType("transformers")
    fake.Sam3Model = types.SimpleNamespace(from_pretrained=forbidden)
    fake.Sam3Processor = types.SimpleNamespace(from_pretrained=forbidden)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("prompted"))

    with pytest.raises(RuntimeError, match="PHENOTYPIC_ACCEPT_MODEL_LICENSE=sam3"):
        Sam3()._ensure_model_loaded()


@pytest.mark.parametrize(
    "load",
    [
        lambda: DinoSam2Detector(dino_version=3)._ensure_model_loaded(),
        lambda: __import__(
            "phenotypic.detect.nn._helper._dino_support", fromlist=["x"]
        ).load_dino_backbone(3, "base", "cpu"),
    ],
    ids=["DinoSam2Detector", "load_dino_backbone"],
)
def test_dinov3_paths_never_prompt(monkeypatch, load) -> None:
    """F10: both runtime call sites passed the default interactive=True."""
    seen: list[dict] = []

    class Stop(Exception):
        pass

    def record(self, **kwargs):
        seen.append(kwargs)
        raise Stop

    monkeypatch.setattr(ckpt.Dinov3CheckpointManager, "download", record)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("prompted"))
    fake = types.ModuleType("transformers")
    fake.AutoModel = types.SimpleNamespace(from_pretrained=lambda *a, **k: None)
    fake.AutoImageProcessor = fake.AutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    with pytest.raises(Stop):
        load()
    assert seen == [{"interactive": False}]


# --- declarations (§3) ----------------------------------------------------------


def test_sam2_declares_packages_and_size_specific_weights() -> None:
    requirements = Sam2(model_size="large").preflight_requirements()

    assert requirements.modules == ("sam2", "torch")
    assert requirements.extra == "torch"
    assert [w.model for w in requirements.weights] == ["sam2:large"]
    assert requirements.rgb_input


def test_sam2_with_an_explicit_checkpoint_needs_no_download(tmp_path: Path) -> None:
    requirements = Sam2(checkpoint=tmp_path / "weights.pt").preflight_requirements()

    assert requirements.weights == ()


def test_sam3_weights_are_gated() -> None:
    (weight,) = Sam3().preflight_requirements().weights

    assert weight.model == "facebook/sam3"
    assert weight.license_key == "sam3"


@pytest.mark.parametrize(
    ("detector", "gated"),
    [
        (FssDinoDetector(), False),  # DINOv2 default
        (FssDinoDetector(dino_version=3), True),
        (Insid3Detector(), True),  # DINOv3 default
    ],
)
def test_dino_weights_are_gated_only_for_v3(detector, gated: bool) -> None:
    requirements = detector.preflight_requirements()
    (weight,) = requirements.weights

    assert (weight.license_key == "dinov3") is gated
    assert ("huggingface_hub" in requirements.modules) is gated
    assert requirements.extra == "foundation"


def test_dinosam2_needs_both_backbones() -> None:
    requirements = DinoSam2Detector(sam2_model_size="small").preflight_requirements()

    assert "sam2" in requirements.modules
    assert [w.model for w in requirements.weights][1] == "sam2:small"


def test_microsam_follows_its_input_layer_and_has_no_extra() -> None:
    """Review R14: RGB comes from the inherited GpuDetector rule."""
    assert not MicroSamDetector().preflight_requirements().rgb_input
    assert MicroSamDetector(input_layer="rgb").preflight_requirements().rgb_input
    requirements = MicroSamDetector().preflight_requirements()
    assert requirements.modules == ("micro_sam",) and requirements.extra is None


def test_filfinder_declares_the_topology_extra() -> None:
    from phenotypic.detect import FilFinderDetector

    requirements = FilFinderDetector().preflight_requirements()

    assert requirements.modules == ("fil_finder", "astropy")
    assert requirements.extra == "topology"


# --- cache probes (§5, review R8) ----------------------------------------------


def test_torch_hub_dir_follows_torch_home_then_xdg(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "th"))
    assert ckpt.torch_hub_checkpoint_dir() == tmp_path / "th" / "hub" / "checkpoints"

    monkeypatch.delenv("TORCH_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert ckpt.torch_hub_checkpoint_dir() == tmp_path / "xdg" / "torch" / "hub" / "checkpoints"


def test_an_empty_torch_home_is_honored_as_torch_honors_it(monkeypatch) -> None:
    """Review C16: torch reads ``os.getenv("TORCH_HOME", default)``, so an empty
    value is the empty path, not "unset" (torch/hub.py ``_get_torch_home``)."""
    monkeypatch.setenv("TORCH_HOME", "")
    monkeypatch.setenv("XDG_CACHE_HOME", "/elsewhere")

    assert ckpt.torch_hub_checkpoint_dir() == Path("hub") / "checkpoints"


def test_torch_hub_dir_matches_torch_when_torch_is_installed(monkeypatch, tmp_path) -> None:
    """A pin, not this task's failing test: it only runs where torch exists."""
    torch_hub = pytest.importorskip("torch.hub")
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "th"))

    assert ckpt.torch_hub_checkpoint_dir() == Path(torch_hub.get_dir()) / "checkpoints"


def test_sam2_cache_probe_sees_a_downloaded_checkpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    (weight,) = Sam2(model_size="tiny").preflight_requirements().weights
    assert weight.is_cached() is False

    target = tmp_path / "hub" / "checkpoints" / ckpt.Sam2CheckpointManager.MODELS["tiny"]["filename"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    assert weight.is_cached() is True


#: Packages the run preflight must never import in the submitting process.
#: Each is stubbed as an importable package, so an import is observable
#: whether or not the real package is installed (review C2: without stubs a
#: ``try: import micro_sam except ImportError`` regression passes wherever
#: micro_sam is absent, which is everywhere CI runs).
_HEAVY_PACKAGES = ("torch", "micro_sam", "transformers", "sam2", "fil_finder", "astropy")


def test_the_run_preflight_imports_no_heavy_package(tmp_path: Path) -> None:
    stubs = tmp_path / "stubs"
    for name in _HEAVY_PACKAGES:
        (stubs / name).mkdir(parents=True)
        (stubs / name / "__init__.py").write_text("", encoding="utf-8")
    repo = Path(__file__).resolve().parents[4]
    code = (
        "import sys\n"
        "from phenotypic import ImagePipeline\n"
        "from phenotypic._cli._cli_preflight import run_preflight\n"
        "from phenotypic.detect import FilFinderDetector\n"
        "from phenotypic.detect.nn import (Sam2, Sam3, DinoSam2Detector, FssDinoDetector,\n"
        "    Insid3Detector, MicroSamDetector)\n"
        "from phenotypic.measure import MeasureSize\n"
        "from tests.unit.cli._preflight_support import make_context\n"
        "ops = {type(d).__name__ + str(i): d for i, d in enumerate((Sam2(), Sam3(),\n"
        "    DinoSam2Detector(dino_version=3), FssDinoDetector(), Insid3Detector(),\n"
        "    MicroSamDetector(), FilFinderDetector()))}\n"
        "report = run_preflight(make_context(ImagePipeline(ops=ops, meas={'s': MeasureSize()})))\n"
        "assert not [f for f in report.findings if f.code == 'PF-CHECK-CRASHED'], report.findings\n"
        f"print(sorted(m for m in {_HEAVY_PACKAGES!r} if m in sys.modules))\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(stubs), str(repo)])}

    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env, cwd=repo
    )

    assert done.stdout.strip().splitlines()[-1] == "[]", done.stdout


def test_the_heavy_package_guard_sees_an_import(tmp_path: Path) -> None:
    """Self-test: the stubs make an import visible even when caught."""
    stubs = tmp_path / "stubs"
    (stubs / "micro_sam").mkdir(parents=True)
    (stubs / "micro_sam" / "__init__.py").write_text("", encoding="utf-8")
    code = (
        "import sys\n"
        "try:\n    import micro_sam\nexcept ImportError:\n    pass\n"
        "print('micro_sam' in sys.modules)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(stubs)}

    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)

    assert done.stdout.strip() == "True"
