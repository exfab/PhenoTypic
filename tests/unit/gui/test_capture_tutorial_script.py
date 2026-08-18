"""Focused regressions for the GUI tutorial capture harness."""

from __future__ import annotations

from pathlib import Path

import pytest
import polars as pl

import scripts.capture_gui_tutorial_screenshots as capture
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.schema import EXPERIMENT, IMAGE
from phenotypic.sdk_ import (
    dataset_overlays_dir,
    deliverables_dir,
    master_measurements_parquet_path,
)


def test_results_timeline_seed_loads_as_current_output(tmp_path, monkeypatch):
    """The hermetic timeline seed must satisfy the current viewer schema."""
    output_dir = tmp_path / "results_timeline_run"
    monkeypatch.setattr(capture, "RESULTS_TIMELINE_OUTPUT_DIR", output_dir)

    capture._seed_results_timeline_output()
    root = OutputRoot.discover(
        output_dir,
        cache_root=tmp_path / ".test-phenotypic-viewer-cache",
    )

    assert str(IMAGE.IMAGE_NAME) in root.master_df.columns
    assert str(EXPERIMENT.DATASET) in root.master_df.columns
    assert root.master_df.height == (
            capture.RESULTS_TIMELINE_N_PLATES
            * capture.RESULTS_TIMELINE_N_TIMES
    )
    overlays = dataset_overlays_dir(output_dir, capture.RESULTS_TIMELINE_DATASET)
    assert len(list(overlays.glob("*.png"))) == root.master_df.height


def test_results_timeline_seed_replaces_obsolete_schema(tmp_path, monkeypatch):
    """A cached pre-migration seed must not poison later capture runs."""
    output_dir = tmp_path / "results_timeline_run"
    monkeypatch.setattr(capture, "RESULTS_TIMELINE_OUTPUT_DIR", output_dir)
    deliverables_dir(output_dir).mkdir(parents=True)
    pl.DataFrame(
        {
            "Metadata_Dataset": [capture.RESULTS_TIMELINE_DATASET],
            "Metadata_ImageFile": ["obsolete"],
        }
    ).write_parquet(master_measurements_parquet_path(output_dir))

    capture._seed_results_timeline_output()
    root = OutputRoot.discover(
        output_dir,
        cache_root=tmp_path / ".test-phenotypic-viewer-cache",
    )

    assert str(IMAGE.IMAGE_NAME) in root.master_df.columns
    assert "Metadata_ImageFile" not in root.master_df.columns


def test_error_triage_seed_repairs_cached_legacy_master(tmp_path: Path) -> None:
    """The capture-only cache repair rewrites the exact historical key."""
    master_path = master_measurements_parquet_path(tmp_path)
    master_path.parent.mkdir(parents=True)
    legacy_image_name = "MetadataImage_ImageName"
    pl.DataFrame(
        {
            legacy_image_name: ["plate-a"],
            "Object_Label": [1],
            "Size_Area": [10.0],
        }
    ).write_parquet(master_path)

    repaired, canonical_image_name = capture._repair_cached_legacy_master(
        pl.read_parquet(master_path), master_path
    )

    assert canonical_image_name == str(IMAGE.IMAGE_NAME)
    assert repaired.columns == [
        canonical_image_name,
        "Object_Label",
        "Size_Area",
    ]
    persisted = pl.read_parquet(master_path)
    assert persisted.columns == repaired.columns
    assert legacy_image_name not in persisted.columns


class _FakeProcess:
    """Minimal ``Popen`` double for readiness-failure diagnostics."""

    def __init__(self) -> None:
        self.terminated = False

    def poll(self):
        return -15 if self.terminated else None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None):
        return -15

    def kill(self) -> None:
        self.terminated = True


def test_readiness_failure_terminates_child_and_includes_log(
        tmp_path: Path, monkeypatch,
):
    log_path = tmp_path / "viewer.log"
    log_path.write_text("startup\nfatal schema mismatch\n", encoding="utf-8")
    proc = _FakeProcess()

    def fail_readiness(*_args, **_kwargs):
        raise RuntimeError("GUI did not respond")

    monkeypatch.setattr(capture, "_wait_for_http_200", fail_readiness)

    with pytest.raises(RuntimeError) as exc_info:
        capture._wait_for_process_http_200(
                "http://127.0.0.1:9999/", proc, log_path, timeout=0.01,
        )

    message = str(exc_info.value)
    assert proc.terminated
    assert "Child exit code: -15" in message
    assert "fatal schema mismatch" in message
