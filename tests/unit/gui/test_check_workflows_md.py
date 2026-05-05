"""Regression tests for scripts/check_workflows_md.py.

The validator is a CI gate that wires WORKFLOWS.md to the capture
script and the on-disk screenshots. These tests lock in the four
failure modes the gate must catch (orphan, undispatched, missing
folder, missing tutorial) plus the happy path against the real
repo.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_workflows_md.py"


@pytest.fixture(scope="module")
def validator():
    """Import scripts/check_workflows_md.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "check_workflows_md", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_workflows_md"] = module
    spec.loader.exec_module(module)
    return module


def _write_fake_pair(
    tmp_path: Path,
    *,
    rows_md: str,
    capture_py: str,
) -> tuple[Path, Path]:
    """Materialise a fake WORKFLOWS.md + capture script under tmp_path."""
    md = tmp_path / "WORKFLOWS.md"
    md.write_text(
        "| ID | Title | Description | Capture function | Tutorial page | Status |\n"
        "| -- | ----- | ----------- | ---------------- | ------------- | ------ |\n"
        + rows_md,
        encoding="utf-8",
    )
    script = tmp_path / "capture.py"
    script.write_text(capture_py, encoding="utf-8")
    return md, script


def _patch_paths(monkeypatch, validator, *, md, script, screenshots, tutorials):
    monkeypatch.setattr(validator, "WORKFLOWS_MD", md)
    monkeypatch.setattr(validator, "CAPTURE_SCRIPT", script)
    monkeypatch.setattr(validator, "SCREENSHOTS_ROOT", screenshots)
    monkeypatch.setattr(validator, "TUTORIAL_ROOT", tutorials)


# ---------------------------------------------------------------------------
# Happy path — the real, committed repo
# ---------------------------------------------------------------------------


def test_real_repo_passes(validator):
    """Committed WORKFLOWS.md + capture script must be self-consistent."""
    rc = validator.main([])
    assert rc == 0


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_orphan_capture_function_is_flagged(
    tmp_path, monkeypatch, capsys, validator
):
    """A `_capture_*` defined+dispatched but not referenced fails."""
    md, script = _write_fake_pair(
        tmp_path,
        rows_md=(
            "| setup | Setup | desc | `_capture_setup` "
            "| `gui_walkthrough/01_setup.md` | ✅ shipping |\n"
        ),
        capture_py=(
            "def _capture_setup(c, b): pass\n"
            "def _capture_orphan(c, b): pass\n"
            "def capture_workflow_screenshots(b, headed=False):\n"
            "    _capture_setup(None, b)\n"
            "    _capture_orphan(None, b)\n"
            "def capture_standalone_viewer_screenshots(headed=False): pass\n"
        ),
    )
    screenshots = tmp_path / "shots"
    (screenshots / "setup").mkdir(parents=True)
    (screenshots / "setup" / "01.png").write_bytes(b"PNG")
    tutorials = tmp_path / "tutorials"
    (tutorials / "gui_walkthrough").mkdir(parents=True)
    (tutorials / "gui_walkthrough" / "01_setup.md").write_text("# Setup")
    _patch_paths(
        monkeypatch, validator,
        md=md, script=script,
        screenshots=screenshots, tutorials=tutorials,
    )
    rc = validator.main([])
    err = capsys.readouterr().err
    assert rc == 1
    assert "_capture_orphan" in err
    assert "no WORKFLOWS.md row references it" in err


def test_referenced_function_not_dispatched_is_flagged(
    tmp_path, monkeypatch, capsys, validator
):
    """A row references a function that is defined but never called."""
    md, script = _write_fake_pair(
        tmp_path,
        rows_md=(
            "| setup | Setup | desc | `_capture_setup` "
            "| `gui_walkthrough/01_setup.md` | ✅ shipping |\n"
        ),
        capture_py=(
            "def _capture_setup(c, b): pass\n"
            "def capture_workflow_screenshots(b, headed=False): pass\n"
            "def capture_standalone_viewer_screenshots(headed=False): pass\n"
        ),
    )
    screenshots = tmp_path / "shots"
    (screenshots / "setup").mkdir(parents=True)
    (screenshots / "setup" / "01.png").write_bytes(b"PNG")
    tutorials = tmp_path / "tutorials"
    (tutorials / "gui_walkthrough").mkdir(parents=True)
    (tutorials / "gui_walkthrough" / "01_setup.md").write_text("# Setup")
    _patch_paths(
        monkeypatch, validator,
        md=md, script=script,
        screenshots=screenshots, tutorials=tutorials,
    )
    rc = validator.main([])
    err = capsys.readouterr().err
    assert rc == 1
    assert "_capture_setup" in err
    assert "not dispatched" in err


def test_shipping_row_with_empty_screenshot_folder_is_flagged(
    tmp_path, monkeypatch, capsys, validator
):
    """✅ shipping rows must have at least one PNG in the folder."""
    md, script = _write_fake_pair(
        tmp_path,
        rows_md=(
            "| setup | Setup | desc | `_capture_setup` "
            "| `gui_walkthrough/01_setup.md` | ✅ shipping |\n"
        ),
        capture_py=(
            "def _capture_setup(c, b): pass\n"
            "def capture_workflow_screenshots(b, headed=False):\n"
            "    _capture_setup(None, b)\n"
            "def capture_standalone_viewer_screenshots(headed=False): pass\n"
        ),
    )
    screenshots = tmp_path / "shots"  # empty -- folder doesn't even exist
    tutorials = tmp_path / "tutorials"
    (tutorials / "gui_walkthrough").mkdir(parents=True)
    (tutorials / "gui_walkthrough" / "01_setup.md").write_text("# Setup")
    _patch_paths(
        monkeypatch, validator,
        md=md, script=script,
        screenshots=screenshots, tutorials=tutorials,
    )
    rc = validator.main([])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no PNGs under" in err


def test_shipping_row_with_missing_tutorial_is_flagged(
    tmp_path, monkeypatch, capsys, validator
):
    """✅ shipping rows must point at an existing tutorial page."""
    md, script = _write_fake_pair(
        tmp_path,
        rows_md=(
            "| setup | Setup | desc | `_capture_setup` "
            "| `gui_walkthrough/missing.md` | ✅ shipping |\n"
        ),
        capture_py=(
            "def _capture_setup(c, b): pass\n"
            "def capture_workflow_screenshots(b, headed=False):\n"
            "    _capture_setup(None, b)\n"
            "def capture_standalone_viewer_screenshots(headed=False): pass\n"
        ),
    )
    screenshots = tmp_path / "shots"
    (screenshots / "setup").mkdir(parents=True)
    (screenshots / "setup" / "01.png").write_bytes(b"PNG")
    tutorials = tmp_path / "tutorials"
    (tutorials / "gui_walkthrough").mkdir(parents=True)
    # NB: no missing.md created
    _patch_paths(
        monkeypatch, validator,
        md=md, script=script,
        screenshots=screenshots, tutorials=tutorials,
    )
    rc = validator.main([])
    err = capsys.readouterr().err
    assert rc == 1
    assert "missing.md" in err
    assert "does not exist" in err


def test_planned_row_skips_screenshot_and_tutorial_checks(
    tmp_path, monkeypatch, capsys, validator
):
    """🔭 planned rows are exempt from PNG/tutorial existence checks."""
    md, script = _write_fake_pair(
        tmp_path,
        rows_md=(
            "| future | Future | desc | `_capture_future` "
            "| `gui_walkthrough/99_future.md` | 🔭 planned |\n"
        ),
        capture_py=(
            "def _capture_future(c, b): pass\n"
            "def capture_workflow_screenshots(b, headed=False):\n"
            "    _capture_future(None, b)\n"
            "def capture_standalone_viewer_screenshots(headed=False): pass\n"
        ),
    )
    screenshots = tmp_path / "shots"
    tutorials = tmp_path / "tutorials"
    _patch_paths(
        monkeypatch, validator,
        md=md, script=script,
        screenshots=screenshots, tutorials=tutorials,
    )
    rc = validator.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "1 workflows" in out
