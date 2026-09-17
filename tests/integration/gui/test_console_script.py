"""Phase 7 console-script integration tests.

Confirms the ``phenotypic-gui`` console script (defined in
``[project.scripts]``) is reachable on PATH after install and that
``--help`` succeeds without spinning up the server. The actual
boot test launches the launcher with a tiny timeout via subprocess
to verify ``--root`` validation works end-to-end.

The companion ``phenotypic`` CLI must continue to work unchanged
(regression check for the click-refactor that was investigated and
dropped — see ``GUI_SPEC_V1.md`` § 7).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest


def _phenotypic_gui_argv() -> list[str]:
    """Return argv for ``phenotypic-gui``, the hub's only entry point.

    Looks only in this interpreter's scripts directory -- never ``PATH`` -- so
    the test exercises this checkout's install. A missing script fails the
    test: there is no module fallback to hide behind.
    """
    scripts_dir = sysconfig.get_path("scripts")
    binary = shutil.which("phenotypic-gui", path=scripts_dir)
    if binary is None:
        pytest.fail(f"phenotypic-gui console script is not installed in {scripts_dir}")
    return [binary]


def test_phenotypic_gui_help_succeeds() -> None:
    """``phenotypic-gui --help`` returns 0 and prints usage."""
    argv = _phenotypic_gui_argv() + ["--help"]
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "phenotypic-gui" in result.stdout
    assert "--root" in result.stdout
    assert "--port" in result.stdout
    assert "--url-prefix" in result.stdout


def test_phenotypic_gui_rejects_missing_root(tmp_path: Path) -> None:
    """Pointing ``--root`` at a non-existent dir exits with a clean error."""
    bad_root = tmp_path / "does-not-exist"
    argv = _phenotypic_gui_argv() + ["--root", str(bad_root), "--port", "0"]
    result = subprocess.run(
        argv, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    # Error message should mention the bad path, not a stack trace.
    assert "Traceback" not in result.stderr
    assert "does-not-exist" in result.stderr or "not exist" in result.stderr.lower()


@pytest.mark.timeout(30)
def test_phenotypic_cli_still_works(tmp_path: Path) -> None:
    """Regression: ``phenotypic`` CLI (the click-group target) still parses.

    The Phase 7 design dropped the click refactor; the old CLI MUST keep
    accepting its argv shape. We assert ``phenotypic --help`` succeeds.
    """
    binary = shutil.which("phenotypic")
    argv = [binary] if binary else [sys.executable, "-m", "phenotypic"]
    result = subprocess.run(
        argv + ["--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
