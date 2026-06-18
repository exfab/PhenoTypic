"""Unit tests for the pure run-picker bind helper (Chunk C).

:func:`~phenotypic.gui.tune._run_picker.discover_run_payload` is the headless
seam the bind callback wraps: it resolves a candidate directory inside the
sandbox, runs ``TuneRunRoot.discover``, and returns either the
``tune-run-root-store`` payload (``{"path": <abs>}``) or ``(None, <note>)`` with
a clear reason. These tests pin the three branches — success, non-tune
directory, out-of-sandbox — without Dash.
"""
from __future__ import annotations

import json
from pathlib import Path

from phenotypic.gui.shell import SandboxRoot


def _write_tune_marker(run_dir: Path) -> None:
    """Write a minimal discoverable ``.pht-tune-cache/run.json`` marker."""
    from phenotypic.sdk_ import tune_cache_run_marker_path

    marker = tune_cache_run_marker_path(run_dir)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "storage_url": None,
                "study_name": "tune",
                "is_multi_objective": False,
                "images_dir": None,
            }
        )
    )


def test_discover_run_payload_success_returns_path(tmp_path: Path) -> None:
    from phenotypic.gui.tune._run_picker import discover_run_payload

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_tune_marker(run_dir)
    sandbox = SandboxRoot.from_path(tmp_path)

    payload, note = discover_run_payload(sandbox, str(sandbox.resolve("run")))
    assert payload == {"path": str(sandbox.resolve("run"))}
    assert note == ""


def test_discover_run_payload_non_tune_dir_returns_note(tmp_path: Path) -> None:
    from phenotypic.gui.tune._run_picker import discover_run_payload

    plain = tmp_path / "plain"
    plain.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)

    payload, note = discover_run_payload(sandbox, str(sandbox.resolve("plain")))
    assert payload is None
    assert "Not a tune output" in note


def test_discover_run_payload_out_of_sandbox_is_refused(tmp_path: Path) -> None:
    from phenotypic.gui.tune._run_picker import discover_run_payload

    sandbox = SandboxRoot.from_path(tmp_path)

    payload, note = discover_run_payload(sandbox, "/etc")
    assert payload is None
    assert "escapes the sandbox" in note


def test_discover_run_payload_blank_candidate_returns_note(tmp_path: Path) -> None:
    from phenotypic.gui.tune._run_picker import discover_run_payload

    sandbox = SandboxRoot.from_path(tmp_path)

    payload, note = discover_run_payload(sandbox, "")
    assert payload is None
    assert note  # a non-empty guidance note
