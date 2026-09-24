"""``PHENOTYPIC_PRELOAD_MODULES`` reaches every process that loads a pipeline.

Spec ``2026-09-24-cli-preflight`` §10.2 and F13; review R2. At 81d19ec the main
CLI never read the variable, so a custom operation failed validation, and local
joblib/loky workers -- fresh processes that never run ``main`` -- could not
resolve it either. These tests run real subprocesses on purpose: the earlier
live test patched the submitting process in-process and hid both gaps.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from phenotypic import ImagePipeline
from phenotypic.measure import MeasureSize
from phenotypic.sdk_._preload import (
    preload_custom_operation_modules,
    preload_module_names,
)
from tests._fakes.custom_threshold_detector import CustomThresholdDetector

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTER = "tests._fakes.register_custom_threshold_detector"


@pytest.fixture
def custom_pipeline(tmp_path: Path) -> Path:
    path = tmp_path / "custom.json"
    pipeline = ImagePipeline(
        ops={"det": CustomThresholdDetector(thresh=0.3)},
        meas={"size": MeasureSize()},
    )
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return path


@pytest.fixture
def two_images(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    (root / "plate1").mkdir(parents=True)
    for index in range(2):
        image = np.full((32, 32, 3), 20, dtype=np.uint8)
        image[6 + index : 14 + index, 6:14] = 220
        tifffile.imwrite(root / "plate1" / f"img{index:03d}.tiff", image)
    return root


def _env(preload: str | None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), *filter(None, [env.get("PYTHONPATH")])]
    )
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.pop("PHENOTYPIC_PRELOAD_MODULES", None)
    if preload is not None:
        env["PHENOTYPIC_PRELOAD_MODULES"] = preload
    return env


def _cli(*args: str, preload: str | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "phenotypic", *args],
        capture_output=True,
        text=True,
        env=_env(preload),
        cwd=REPO_ROOT,
        timeout=600,
    )


def test_a_custom_op_validates_in_the_main_cli_process(
    custom_pipeline: Path, two_images: Path, tmp_path: Path
) -> None:
    result = _cli(
        "--pipeline", str(custom_pipeline), "--input", str(two_images),
        "--output", str(tmp_path / "out"), "--image-type", "Image", "--dry-run",
        preload=REGISTER,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Pipeline loaded successfully" in result.stdout


def test_a_custom_op_runs_in_local_parallel_workers(
    custom_pipeline: Path, two_images: Path, tmp_path: Path
) -> None:
    """Review R2: loky workers never pass through ``main``."""
    result = _cli(
        "--pipeline", str(custom_pipeline), "--input", str(two_images),
        "--output", str(tmp_path / "out"), "--image-type", "Image",
        "--njobs", "2", "--no-qc",
        preload=REGISTER,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Completed: 2/2" in output, output
    assert "not found in phenotypic namespace" not in output


def test_an_unregistered_custom_op_is_refused_with_the_remedy(
    custom_pipeline: Path, two_images: Path, tmp_path: Path
) -> None:
    result = _cli(
        "--pipeline", str(custom_pipeline), "--input", str(two_images),
        "--output", str(tmp_path / "out"), "--image-type", "Image", "--dry-run",
        preload=None,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "CustomThresholdDetector" in output
    assert "PHENOTYPIC_PRELOAD_MODULES" in output
    assert "[PF-CUSTOM-OP]" in output  # the code and its hint print (review C3)
    assert not (tmp_path / "out").exists()


def test_class_resolution_preloads_in_a_process_that_never_ran_the_cli(
    custom_pipeline: Path,
) -> None:
    """The mechanism workers rely on: a bare ``from_json`` in a fresh process."""
    code = (
        "import sys; from phenotypic import ImagePipeline; "
        f"p = ImagePipeline.from_json({str(custom_pipeline)!r}); "
        "print(type(p.get_ops()['det']).__name__)"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=_env(REGISTER), cwd=REPO_ROOT, timeout=300,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "CustomThresholdDetector"


def test_a_module_that_only_defines_the_class_is_not_enough(custom_pipeline: Path) -> None:
    """Report §1b: registration, not import, is what resolution needs."""
    code = (
        "from phenotypic import ImagePipeline; "
        f"ImagePipeline.from_json({str(custom_pipeline)!r})"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=_env("tests._fakes.custom_threshold_detector"), cwd=REPO_ROOT, timeout=300,
    )

    assert done.returncode != 0
    assert "UnknownOperationClassError" in done.stderr
    assert "attaches the class to the phenotypic namespace" in done.stderr


def _counting_module(tmp_path: Path, name: str) -> Path:
    """A module that appends one line to a log each time its body executes."""
    log = tmp_path / f"{name}.log"
    (tmp_path / f"{name}.py").write_text(
        f"with open({str(log)!r}, 'a', encoding='utf-8') as handle:\n"
        "    handle.write('imported\\n')\n",
        encoding="utf-8",
    )
    return log


def test_preload_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Blank entries are ignored and a repeated name executes the module once."""
    log = _counting_module(tmp_path, "pht_counting_preload")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "pht_counting_preload", raising=False)
    name = "pht_counting_preload"
    monkeypatch.setenv("PHENOTYPIC_PRELOAD_MODULES", f" {name} , ,{name}")

    assert preload_module_names() == (name, name)
    preload_custom_operation_modules()
    preload_custom_operation_modules()

    assert log.read_text(encoding="utf-8").splitlines() == ["imported"]


def test_resolution_preloads_even_when_every_name_resolves(tmp_path: Path) -> None:
    """Review C4: a process whose classes are all built-ins still preloads."""
    log = _counting_module(tmp_path, "pht_side_effect_preload")
    pipeline = tmp_path / "builtin.json"
    pipeline.write_text(
        ImagePipeline(meas={"size": MeasureSize()}).to_json(), encoding="utf-8"
    )
    env = _env("pht_side_effect_preload")
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), env["PYTHONPATH"]])
    code = (
        "from phenotypic import ImagePipeline; "
        f"ImagePipeline.from_json({str(pipeline)!r}); "
        f"ImagePipeline.from_json({str(pipeline)!r})"
    )

    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=env, cwd=REPO_ROOT, timeout=300,
    )

    assert done.returncode == 0, done.stderr
    assert log.read_text(encoding="utf-8").splitlines() == ["imported"]


def test_a_broken_preload_name_is_one_error_line_at_startup(
    two_images: Path, tmp_path: Path
) -> None:
    """Review C6: the startup preload names the variable; no traceback."""
    pipeline = tmp_path / "builtin.json"
    pipeline.write_text(
        ImagePipeline(meas={"size": MeasureSize()}).to_json(), encoding="utf-8"
    )

    result = _cli(
        "--pipeline", str(pipeline), "--input", str(two_images),
        "--output", str(tmp_path / "out"), "--dry-run",
        preload="no_such_module_xyz",
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "PHENOTYPIC_PRELOAD_MODULES" in output and "no_such_module_xyz" in output
    assert "Traceback" not in output
    assert not (tmp_path / "out").exists()
