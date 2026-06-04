"""Tests for processing-state IO routing through the ``.phenotypic`` cache."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from phenotypic._cli._cli_state_management import load_processing_state, save_processing_state
from phenotypic._cli._cli_types import ProcessingState
from phenotypic.tools_ import processing_state_path


def _make_state(out: Path) -> ProcessingState:
    now = datetime(2026, 6, 3, 12, 0, 0)
    return ProcessingState(
        version="2",
        pipeline_path=Path("p.json"),
        input_path=Path("in"),
        output_dir=out,
        timestamp=now,
        execution_mode="local",
        last_updated=now,
        datasets={},
        config={},
    )


def test_save_writes_under_phenotypic(tmp_path: Path) -> None:
    save_processing_state(_make_state(tmp_path), tmp_path)
    assert processing_state_path(tmp_path).is_file()
    assert not (tmp_path / "processing_state.json").exists()


def test_load_migrates_and_reads_legacy_run(tmp_path: Path) -> None:
    # Simulate a pre-migration run: state at the output root.
    save_processing_state(_make_state(tmp_path), tmp_path)
    new = processing_state_path(tmp_path)
    legacy = tmp_path / "processing_state.json"
    (tmp_path / ".phenotypic").rename(tmp_path / "_tmp")  # extract
    (tmp_path / "_tmp" / "processing_state.json").rename(legacy)
    (tmp_path / "_tmp").rmdir()
    assert legacy.is_file() and not new.exists()
    loaded = load_processing_state(tmp_path)
    assert loaded is not None
    # migrate-on-load moved it into .phenotypic
    assert new.is_file()
