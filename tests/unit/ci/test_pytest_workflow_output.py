"""Tests for pytest GitHub Actions output policy."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_full_pytest_workflow_uses_quiet_failure_focused_output() -> None:
    """The full pytest lane should show failures/warnings, not every pass."""
    workflow = REPO_ROOT / ".github" / "workflows" / "run-pytest-full.yml"
    text = workflow.read_text(encoding="utf-8")

    run_test_blocks = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("pytest ")
    ]

    assert run_test_blocks
    assert all("-q" in line for line in run_test_blocks)
    assert all("--tb=long" in line for line in run_test_blocks)
    assert all('--no-header' in line for line in run_test_blocks)
    assert all("-r " in line and "Efw" in line for line in run_test_blocks)
    assert all("--verbose" not in line for line in run_test_blocks)
    assert all("--capture=no" not in line for line in run_test_blocks)
