"""Tune runs are discoverable via their ``.pht-tune-cache/run.json`` marker.

A tune run writes a ``run.json`` marker into the hidden ``.pht-tune-cache/``
at run START (before any deliverable lands), so the classifier must surface it
via a dedicated ``is_tune_output`` capability — mirroring the process-only
``.phenotypic/progress/manifest.json`` detection. A plain directory and a
forward CLI run (``.phenotypic`` machine-state) are NOT tune outputs.
"""

from pathlib import Path

import pytest

from phenotypic.gui.shell._classifier import classify, invalidate_cache
from phenotypic.tools_ import (
    manifest_json_path,
    tune_cache_run_marker_path,
)


@pytest.fixture(autouse=True)
def _flush_classifier_cache() -> None:
    invalidate_cache()


def test_tune_run_is_discoverable(tmp_path: Path) -> None:
    marker = tune_cache_run_marker_path(tmp_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('{"version": 1, "study_name": "tune"}', encoding="utf-8")
    caps = classify(tmp_path)
    assert caps.is_tune_output is True
    # A tune run is not a forward CLI output (no results/ + master parquet).
    assert caps.is_cli_output is False


def test_plain_dir_not_tune_output(tmp_path: Path) -> None:
    (tmp_path / "plateA.png").write_bytes(b"\x89PNG")
    caps = classify(tmp_path)
    assert caps.is_tune_output is False


def test_forward_run_not_tune_output(tmp_path: Path) -> None:
    # A forward CLI run carries .phenotypic machine-state but NO tune marker.
    (tmp_path / "results").mkdir()
    deliv = tmp_path / "deliverables"
    deliv.mkdir()
    (deliv / "master_measurements.parquet").write_bytes(b"x")
    mp = manifest_json_path(tmp_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text('{"is_complete": true}', encoding="utf-8")
    caps = classify(tmp_path)
    assert caps.is_cli_output is True
    assert caps.is_tune_output is False
