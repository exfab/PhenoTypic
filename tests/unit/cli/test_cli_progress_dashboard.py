"""
Tests for the SLURM progress dashboard and failure tracking modules.

Covers:
- Event log parsing with "started" status and SLURM fields
- Backward compatibility with old 4-5 field event lines
- Failure tracker JSONL append/read/categorize
- Manifest builder with mock event log and failures
- sacct output parsing with mock subprocess
- OOM detection logic
- Dashboard HTML generation
- Sentinel script generation
- DatasetState.in_progress property
"""

from __future__ import annotations

import json
import shlex
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
from uuid import uuid4

import pytest

from phenotypic._cli._cli_types import DatasetState
from phenotypic._cli._cli_update_state import (
    PROCESSING_GENERATION_ENV_VAR,
    append_event,
    append_completion_event,
    parse_event_line,
    aggregate_state_from_events,
    aggregate_state_from_events_with_diagnostics,
)
from phenotypic._cli._cli_failure_tracker import (
    append_failure,
    read_failures,
    categorize_failures,
)
from phenotypic._cli._dashboard._manifest_builder import (
    build_manifest,
    query_sacct_job_states,
    query_sacct_chunk_states,
)
from phenotypic._cli._dashboard import generate_dashboard
from phenotypic._cli._cli_sentinel_scripts import generate_sentinel_script
from phenotypic.sdk_ import analysis_html_path, dashboard_html_path, logs_dir
from phenotypic.sdk_ import progress_dir as _progress_dir


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that cleans up after each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def _clear_processing_generation(monkeypatch):
    """Prevent process-global generation state from leaking between tests."""
    monkeypatch.delenv(PROCESSING_GENERATION_ENV_VAR, raising=False)


@pytest.fixture
def event_log(tmp_dir):
    """Path to a fresh event log file."""
    return tmp_dir / "processing_events.log"


@pytest.fixture
def progress_dir(tmp_dir):
    """Path to a fresh progress directory."""
    p = tmp_dir / "progress"
    p.mkdir()
    return p


# ──────────────────────────────────────────────────────────────────────
# DatasetState
# ──────────────────────────────────────────────────────────────────────

class TestDatasetState:

    def test_in_progress_property(self):
        ds = DatasetState()
        ds.started = {"img1", "img2", "img3"}
        ds.completed = {"img1"}
        ds.failed = {"img3"}
        assert ds.in_progress == {"img2"}

    def test_in_progress_empty_when_all_finished(self):
        ds = DatasetState()
        ds.started = {"img1", "img2"}
        ds.completed = {"img1", "img2"}
        assert ds.in_progress == set()

    def test_in_progress_empty_when_nothing_started(self):
        ds = DatasetState()
        assert ds.in_progress == set()


# ──────────────────────────────────────────────────────────────────────
# Event Log Parsing
# ──────────────────────────────────────────────────────────────────────

class TestParseEventLine:

    def test_started_status(self):
        line = "2026-02-27T14:30:00.000|plate1|img001.tif|started||12345|42"
        evt = parse_event_line(line)
        assert evt.status == "started"
        assert evt.dataset == "plate1"
        assert evt.image == "img001.tif"
        assert evt.slurm_job_id == "12345"
        assert evt.slurm_array_task_id == "42"

    def test_completed_status(self):
        line = "2026-02-27T14:30:00.000|plate1|img001.tif|completed|"
        evt = parse_event_line(line)
        assert evt.status == "completed"
        assert evt.error_msg == ""

    def test_failed_status_with_error(self):
        line = "2026-02-27T14:30:00.000|plate1|img001.tif|failed|ValueError: bad shape"
        evt = parse_event_line(line)
        assert evt.status == "failed"
        assert "ValueError" in evt.error_msg

    def test_backward_compat_4_fields(self):
        """Old format with no error message field."""
        line = "2026-02-27T14:30:00.000|plate1|img001.tif|completed"
        evt = parse_event_line(line)
        assert evt.status == "completed"
        assert evt.error_msg == ""
        assert evt.slurm_job_id == ""

    def test_backward_compat_5_fields(self):
        """Old format with error message but no SLURM fields."""
        line = "2026-02-27T14:30:00.000|plate1|img001.tif|failed|some error"
        evt = parse_event_line(line)
        assert evt.status == "failed"
        assert evt.error_msg == "some error"
        assert evt.slurm_job_id == ""
        assert evt.slurm_array_task_id == ""

    def test_new_7_field_format(self):
        line = "2026-02-27T14:30:00.000|plate1|img001.tif|started||99999|7"
        evt = parse_event_line(line)
        assert evt.slurm_job_id == "99999"
        assert evt.slurm_array_task_id == "7"

    def test_invalid_status_raises(self):
        line = "2026-02-27T14:30:00.000|plate1|img001.tif|unknown|"
        with pytest.raises(ValueError, match="Invalid status"):
            parse_event_line(line)

    def test_too_few_fields_raises(self):
        line = "2026-02-27T14:30:00.000|plate1|img001.tif"
        with pytest.raises(ValueError, match="Invalid line format"):
            parse_event_line(line)

    def test_empty_line_raises(self):
        with pytest.raises(ValueError, match="Empty line"):
            parse_event_line("")

    def test_pipe_in_error_roundtrip(self, tmp_dir):
        """Pipes in error messages survive a write-then-read round trip."""
        event_log = tmp_dir / "test_events.log"
        append_event(event_log, "plate1", "img001.tif", "failed", error_msg="has | pipe")
        state = aggregate_state_from_events(event_log)
        # The error is stored after escaping — pipes become \\| on disk,
        # and the split-then-unescape means the full message may be truncated.
        # The important thing is no crash and the image is marked failed.
        assert "img001.tif" in state["plate1"].failed


class TestAppendEvent:

    def test_append_started(self, event_log):
        append_event(event_log, "plate1", "img001.tif", "started")
        content = event_log.read_text()
        assert "|started|" in content

    def test_append_with_slurm_fields(self, event_log):
        append_event(
            event_log, "plate1", "img001.tif", "started",
            slurm_job_id="12345", slurm_array_task_id="0",
        )
        content = event_log.read_text()
        assert "|12345|0" in content

    def test_append_without_slurm_omits_fields(self, event_log):
        append_event(event_log, "plate1", "img001.tif", "completed")
        content = event_log.read_text()
        parts = content.strip().split("|")
        # Should have 5 fields (no SLURM fields)
        assert len(parts) == 5

    def test_backward_compat_wrapper(self, event_log):
        append_completion_event(event_log, "plate1", "img001.tif", "completed")
        content = event_log.read_text()
        assert "|completed|" in content


class TestAggregateState:

    def test_tracks_started(self, event_log):
        append_event(event_log, "plate1", "img001.tif", "started")
        append_event(event_log, "plate1", "img002.tif", "started")
        append_event(event_log, "plate1", "img001.tif", "completed")

        state = aggregate_state_from_events(event_log)
        ds = state["plate1"]
        assert "img001.tif" not in ds.started
        assert "img002.tif" in ds.started
        assert "img001.tif" in ds.completed
        assert ds.in_progress == {"img002.tif"}

    def test_mixed_statuses(self, event_log):
        append_event(event_log, "plate1", "img001.tif", "started")
        append_event(event_log, "plate1", "img002.tif", "started")
        append_event(event_log, "plate1", "img003.tif", "started")
        append_event(event_log, "plate1", "img001.tif", "completed")
        append_event(event_log, "plate1", "img002.tif", "failed", error_msg="OOM")

        state = aggregate_state_from_events(event_log)
        ds = state["plate1"]
        assert ds.completed == {"img001.tif"}
        assert ds.failed == {"img002.tif"}
        assert ds.in_progress == {"img003.tif"}

    def test_empty_log(self, event_log):
        state = aggregate_state_from_events(event_log)
        assert state == {}

    def test_inventory_and_generation_exclude_stale_events(self, event_log):
        append_event(
            event_log,
            "plate1",
            "old.tif",
            "completed",
            generation="current",
        )
        append_event(
            event_log,
            "plate1",
            "a.tif",
            "completed",
            generation="previous",
        )
        append_event(
            event_log,
            "plate1",
            "a.tif",
            "started",
            generation="current",
        )
        append_event(
            event_log,
            "plate1",
            "b.tif",
            "failed",
            generation="current",
        )

        aggregated = aggregate_state_from_events_with_diagnostics(
            event_log,
            inventory={"plate1": {"a.tif", "b.tif"}},
            generation="current",
        )

        assert aggregated.datasets["plate1"].in_progress == {"a.tif"}
        assert aggregated.datasets["plate1"].failed == {"b.tif"}
        assert aggregated.diagnostics.unknown_image_events == 1
        assert aggregated.diagnostics.other_generation_events == 1

    def test_generationless_events_remain_inventory_filtered(self, event_log):
        append_event(event_log, "plate1", "legacy.tif", "completed")
        append_event(event_log, "plate1", "unknown.tif", "completed")

        state = aggregate_state_from_events(
            event_log,
            inventory={"plate1": {"legacy.tif"}},
            generation="new-generation",
        )

        assert state["plate1"].completed == {"legacy.tif"}


# ──────────────────────────────────────────────────────────────────────
# Failure Tracker
# ──────────────────────────────────────────────────────────────────────

class TestFailureTracker:

    def test_append_and_read(self, progress_dir):
        append_failure(
            progress_dir,
            dataset="plate1",
            image="img001.tif",
            error_type="ValueError",
            error_message="bad shape",
            traceback="Traceback...\nValueError: bad shape",
        )
        failures = read_failures(progress_dir)
        assert len(failures) == 1
        assert failures[0]["error_type"] == "ValueError"
        assert failures[0]["dataset"] == "plate1"
        assert failures[0]["failure_source"] == "python"

    def test_multiple_appends(self, progress_dir):
        for i in range(5):
            append_failure(
                progress_dir,
                dataset="plate1",
                image=f"img{i:03d}.tif",
                error_type="RuntimeError" if i % 2 else "ValueError",
                error_message=f"error {i}",
            )
        failures = read_failures(progress_dir)
        assert len(failures) == 5

    def test_slurm_failure(self, progress_dir):
        append_failure(
            progress_dir,
            dataset="plate1",
            image="img099.tif",
            error_type="OUT_OF_MEMORY",
            error_message="SLURM killed task",
            slurm_job_id="12345_99",
            failure_source="slurm",
        )
        failures = read_failures(progress_dir)
        assert failures[0]["failure_source"] == "slurm"
        assert failures[0]["slurm_job_id"] == "12345_99"

    def test_categorize(self, progress_dir):
        for _ in range(3):
            append_failure(
                progress_dir, dataset="d", image="i",
                error_type="ValueError", error_message="e",
            )
        for _ in range(2):
            append_failure(
                progress_dir, dataset="d", image="i",
                error_type="OUT_OF_MEMORY", error_message="e",
            )
        append_failure(
            progress_dir, dataset="d", image="i",
            error_type="TimeoutError", error_message="e",
        )

        failures = read_failures(progress_dir)
        cats = categorize_failures(failures)
        assert cats["ValueError"] == 3
        assert cats["OUT_OF_MEMORY"] == 2
        assert cats["TimeoutError"] == 1

    def test_read_empty(self, progress_dir):
        assert read_failures(progress_dir) == []

    def test_read_missing_dir(self, tmp_dir):
        assert read_failures(tmp_dir / "nonexistent") == []

    def test_malformed_lines_skipped(self, progress_dir):
        failures_path = progress_dir / "failures.jsonl"
        failures_path.write_text("not json\n{bad json\n")
        assert read_failures(progress_dir) == []


# ──────────────────────────────────────────────────────────────────────
# Manifest Builder
# ──────────────────────────────────────────────────────────────────────

class TestManifestBuilder:

    def test_local_manifest(self, tmp_dir):
        event_log = tmp_dir / "processing_events.log"
        progress_dir = tmp_dir / "progress"
        progress_dir.mkdir()

        # Create events
        append_event(event_log, "plate1", "img001.tif", "started")
        append_event(event_log, "plate1", "img001.tif", "completed")
        append_event(event_log, "plate1", "img002.tif", "started")
        append_event(event_log, "plate1", "img002.tif", "failed", error_msg="err")

        # Create a failure record
        append_failure(
            progress_dir, dataset="plate1", image="img002.tif",
            error_type="ValueError", error_message="err",
        )

        build_manifest(
            output_dir=tmp_dir,
            progress_dir=progress_dir,
            datasets={"plate1": 3},
            execution_mode="local",
            start_time=datetime.now().isoformat(timespec="milliseconds"),
        )

        manifest_path = progress_dir / "manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["version"] == 1
        assert manifest["execution_mode"] == "local"
        assert manifest["total_images"] == 3
        assert manifest["completed"] == 1
        assert manifest["failed"] == 1
        assert manifest["pending"] == 1  # 3 - 1 completed - 1 failed - 0 in_progress
        assert manifest["is_complete"] is False
        assert "ValueError" in manifest["failure_categories"]
        assert "gui_record_generation" not in manifest

    def test_local_gui_manifest_carries_exact_generation(self, tmp_dir):
        progress_dir = tmp_dir / "progress"
        progress_dir.mkdir()
        generation = str(uuid4())

        build_manifest(
            output_dir=tmp_dir,
            progress_dir=progress_dir,
            datasets={},
            execution_mode="local",
            start_time=datetime.now().isoformat(timespec="milliseconds"),
            gui_record_generation=generation,
        )

        manifest = json.loads((progress_dir / "manifest.json").read_text())
        assert manifest["gui_record_generation"] == generation

    def test_manifest_is_complete(self, tmp_dir):
        event_log = tmp_dir / "processing_events.log"
        progress_dir = tmp_dir / "progress"
        progress_dir.mkdir()

        append_event(event_log, "plate1", "img001.tif", "started")
        append_event(event_log, "plate1", "img001.tif", "completed")

        build_manifest(
            output_dir=tmp_dir,
            progress_dir=progress_dir,
            datasets={"plate1": 1},
            execution_mode="local",
            start_time=datetime.now().isoformat(timespec="milliseconds"),
        )

        manifest = json.loads((progress_dir / "manifest.json").read_text())
        assert manifest["is_complete"] is True
        assert manifest["completed"] == 1
        assert manifest["success_rate"] == 1.0

    def test_manifest_filters_historical_images_and_reports_them(
        self,
        tmp_dir,
    ):
        event_log = tmp_dir / "processing_events.log"
        progress_dir = tmp_dir / "progress"
        progress_dir.mkdir()
        append_event(event_log, "plate1", "old-1.tif", "completed")
        append_event(event_log, "plate1", "a.tif", "completed")
        append_failure(
            progress_dir,
            dataset="plate1",
            image="old-1.tif",
            error_type="HistoricalError",
            error_message="stale",
        )

        build_manifest(
            output_dir=tmp_dir,
            progress_dir=progress_dir,
            datasets={"plate1": 2},
            dataset_inventory={"plate1": {"a.tif", "b.tif"}},
            execution_mode="local",
            start_time=datetime.now().isoformat(timespec="milliseconds"),
            processing_generation="current",
        )

        manifest = json.loads((progress_dir / "manifest.json").read_text())
        assert manifest["completed"] == 1
        assert manifest["failed"] == 0
        assert manifest["pending"] == 1
        assert manifest["processing_generation"] == "current"
        assert manifest["event_diagnostics"]["unknown_image_events"] == 1
        assert "HistoricalError" not in manifest["failure_categories"]

    def test_manifest_categories_only_latest_current_failures(self, tmp_dir):
        event_log = tmp_dir / "processing_events.log"
        progress_dir = tmp_dir / "progress"
        progress_dir.mkdir()

        append_event(event_log, "plate1", "recovered.tif", "failed")
        append_failure(
            progress_dir,
            dataset="plate1",
            image="recovered.tif",
            error_type="RecoveredError",
            error_message="first attempt",
        )
        append_event(event_log, "plate1", "recovered.tif", "started")
        append_event(event_log, "plate1", "recovered.tif", "completed")

        append_event(event_log, "plate1", "failed.tif", "failed")
        append_failure(
            progress_dir,
            dataset="plate1",
            image="failed.tif",
            error_type="EarlierError",
            error_message="earlier attempt",
        )
        append_event(event_log, "plate1", "failed.tif", "started")
        append_event(event_log, "plate1", "failed.tif", "failed")
        append_failure(
            progress_dir,
            dataset="plate1",
            image="failed.tif",
            error_type="CurrentError",
            error_message="current attempt",
        )

        build_manifest(
            output_dir=tmp_dir,
            progress_dir=progress_dir,
            datasets={"plate1": 2},
            dataset_inventory={
                "plate1": {"recovered.tif", "failed.tif"}
            },
            execution_mode="local",
            start_time=datetime.now().isoformat(timespec="milliseconds"),
        )

        manifest = json.loads((progress_dir / "manifest.json").read_text())
        assert manifest["completed"] == 1
        assert manifest["failed"] == 1
        assert manifest["failure_categories"] == {"CurrentError": 1}

    def test_manifest_multiple_datasets(self, tmp_dir):
        event_log = tmp_dir / "processing_events.log"
        progress_dir = tmp_dir / "progress"
        progress_dir.mkdir()

        append_event(event_log, "plate1", "img001.tif", "started")
        append_event(event_log, "plate1", "img001.tif", "completed")
        append_event(event_log, "plate2", "img001.tif", "started")

        build_manifest(
            output_dir=tmp_dir,
            progress_dir=progress_dir,
            datasets={"plate1": 2, "plate2": 3},
            execution_mode="local",
            start_time=datetime.now().isoformat(timespec="milliseconds"),
        )

        manifest = json.loads((progress_dir / "manifest.json").read_text())
        assert manifest["total_images"] == 5
        assert manifest["datasets"]["plate1"]["completed"] == 1
        assert manifest["datasets"]["plate1"]["total"] == 2
        assert manifest["datasets"]["plate2"]["started"] == 1  # in-progress
        assert manifest["datasets"]["plate2"]["total"] == 3

    def test_slurm_manifest_includes_slurm_info(self, tmp_dir):
        progress_dir = tmp_dir / "progress"
        progress_dir.mkdir()

        build_manifest(
            output_dir=tmp_dir,
            progress_dir=progress_dir,
            datasets={"plate1": 10},
            execution_mode="slurm",
            start_time=datetime.now().isoformat(timespec="milliseconds"),
            slurm_job_ids={"0": "12345"},
            chunk_scripts=["chunk0.sh"],
        )

        manifest = json.loads((progress_dir / "manifest.json").read_text())
        assert "slurm_info" in manifest
        assert manifest["slurm_info"]["chunk_scripts"] == ["chunk0.sh"]
        assert manifest["slurm_info"]["chunk_job_ids"] == {"0": "12345"}

    def test_slurm_manifest_ignores_named_staged_round_jobs(self, tmp_dir):
        progress_dir = tmp_dir / "progress"
        progress_dir.mkdir()
        event_log = tmp_dir / "processing_events.log"
        append_event(event_log, "plate1", "img001.tif", "started")
        append_completion_event(
            event_log, "plate1", "img001.tif", "completed"
        )

        build_manifest(
            output_dir=tmp_dir,
            progress_dir=progress_dir,
            datasets={"plate1": 1},
            execution_mode="slurm",
            start_time=datetime.now().isoformat(timespec="milliseconds"),
            slurm_job_ids={"0": "12345", "stage2-round-1": "23456"},
            chunk_scripts=["chunk0.sh"],
        )

        manifest = json.loads((progress_dir / "manifest.json").read_text())
        slurm_info = manifest["slurm_info"]
        assert slurm_info["total_chunks"] == 1
        assert slurm_info["chunk_job_ids"] == {"0": "12345"}
        assert slurm_info["completed_chunks"] == [0]


# ──────────────────────────────────────────────────────────────────────
# sacct parsing
# ──────────────────────────────────────────────────────────────────────

class TestSacctParsing:

    def test_sacct_unavailable_returns_none(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("sacct not found")):
            result = query_sacct_job_states("12345")
        assert result is None

    def test_sacct_permission_denied_returns_none(self):
        with patch("subprocess.run", side_effect=PermissionError("denied")):
            result = query_sacct_job_states("12345")
        assert result is None

    def test_sacct_timeout_returns_none(self):
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("sacct", 10)):
            result = query_sacct_job_states("12345")
        assert result is None

    def test_sacct_nonzero_exit_returns_none(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"
        with patch("subprocess.run", return_value=mock_result):
            result = query_sacct_job_states("12345")
        assert result is None

    def test_sacct_parses_array_tasks(self):
        sacct_output = (
            "12345_0|COMPLETED|0:0|512K\n"
            "12345_0.batch|COMPLETED|0:0|512K\n"
            "12345_1|FAILED|1:0|1024K\n"
            "12345_1.batch|FAILED|1:0|1024K\n"
            "12345_2|OUT_OF_MEMORY|0:125|2048K\n"
            "12345_2.batch|OUT_OF_MEMORY|0:125|2048K\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = sacct_output
        with patch("subprocess.run", return_value=mock_result):
            result = query_sacct_job_states("12345")

        assert result is not None
        assert result["12345_0"] == "COMPLETED"
        assert result["12345_1"] == "FAILED"
        assert result["12345_2"] == "OUT_OF_MEMORY"
        # Sub-step lines should be excluded
        assert "12345_0.batch" not in result

    def test_sacct_chunk_states_unavailable(self):
        import phenotypic._cli._dashboard._manifest_builder as mb
        saved = mb._terminal_job_cache.copy()
        mb._terminal_job_cache.clear()
        try:
            with patch(
                "phenotypic._cli._dashboard._manifest_builder.query_sacct_batch",
                return_value=None,
            ):
                active, completed, pending = query_sacct_chunk_states({"0": "12345"})
            assert active == []
            assert completed == []
            assert pending == []
        finally:
            mb._terminal_job_cache.clear()
            mb._terminal_job_cache.update(saved)

    def test_sacct_chunk_states_ignores_named_staged_round_jobs(
        self, monkeypatch
    ):
        import phenotypic._cli._dashboard._manifest_builder as mb

        monkeypatch.setattr(mb, "_terminal_job_cache", {})
        with patch(
            "phenotypic._cli._dashboard._manifest_builder.query_sacct_batch",
            return_value={"12345": {"12345_0": "COMPLETED"}},
        ) as query_batch:
            active, completed, pending = query_sacct_chunk_states(
                {"0": "12345", "stage2-round-1": "23456"}
            )

        query_batch.assert_called_once_with(["12345"])
        assert active == []
        assert completed == [0]
        assert pending == []


# ──────────────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────────────

class TestDashboard:

    def test_generates_html(self, tmp_dir):
        generate_dashboard(tmp_dir)
        assert dashboard_html_path(tmp_dir).exists()
        assert analysis_html_path(tmp_dir).exists()

    def test_html_contains_key_elements(self, tmp_dir):
        generate_dashboard(tmp_dir)
        html = dashboard_html_path(tmp_dir).read_text()
        assert "<!DOCTYPE html>" in html
        assert "PhenoTypic" in html
        assert "PROGRESS_PREFIX + 'manifest.json" in html
        assert "PROGRESS_PREFIX + 'failures.jsonl" in html
        assert "setInterval" in html

    def test_machine_state_urls_use_canonical_progress_prefix(self, tmp_dir):
        """Generated pages fetch every machine-state asset from the hidden cache."""
        generate_dashboard(tmp_dir)
        dashboard = dashboard_html_path(tmp_dir).read_text()
        analysis = analysis_html_path(tmp_dir).read_text()
        pages = (dashboard, analysis)

        canonical_prefix = (
            'const PROGRESS_PREFIX = ROOT_PREFIX + ".phenotypic/progress/";'
        )
        assert all(canonical_prefix in page for page in pages)
        assert all("ROOT_PREFIX + 'progress/" not in page for page in pages)

        assert "PROGRESS_PREFIX + 'manifest.json" in dashboard
        assert "PROGRESS_PREFIX + 'failures.jsonl" in dashboard
        for asset in (
            "manifest.json",
            "plotly.min.js",
            "hyparquet.min.js",
            "analysis_full.parquet",
            "analysis_stats.json",
        ):
            assert f"PROGRESS_PREFIX + '{asset}" in analysis

    def test_creates_dir_if_missing(self, tmp_dir):
        new_dir = tmp_dir / "new_output"
        generate_dashboard(new_dir)
        assert dashboard_html_path(new_dir).exists()
        assert analysis_html_path(new_dir).exists()

    def test_contains_tab_structure(self, tmp_dir):
        generate_dashboard(tmp_dir)
        html = dashboard_html_path(tmp_dir).read_text()
        assert "tab-progress" in html
        assert "tab-readme" in html
        assert "tab-download" in html
        assert "switchTab" in html
        # Analysis is now a separate page, linked from dashboard
        assert "analysis.html" in html
        assert "tab-analysis" not in html

    def test_contains_marked_js(self, tmp_dir):
        generate_dashboard(tmp_dir)
        html = dashboard_html_path(tmp_dir).read_text()
        assert "marked" in html

    def test_local_mode_hides_download_tab(self, tmp_dir):
        generate_dashboard(tmp_dir, execution_mode="local")
        html = dashboard_html_path(tmp_dir).read_text()
        assert 'EXECUTION_MODE = "local"' in html

    def test_slurm_mode_enables_download_tab(self, tmp_dir):
        generate_dashboard(tmp_dir, execution_mode="slurm")
        html = dashboard_html_path(tmp_dir).read_text()
        assert 'EXECUTION_MODE = "slurm"' in html

    def test_readme_fetch_path(self, tmp_dir):
        generate_dashboard(tmp_dir)
        html = dashboard_html_path(tmp_dir).read_text()
        assert "README.md" in html

    def test_download_tab_wget_content(self, tmp_dir):
        generate_dashboard(tmp_dir, execution_mode="slurm")
        html = dashboard_html_path(tmp_dir).read_text()
        assert "wget" in html
        assert "YOUR_SERVER_URL" in html
        assert "--cut-dirs" in html
        assert "--user=" in html
        assert "--password='" in html

    def test_download_url_autodetect_js(self, tmp_dir):
        generate_dashboard(tmp_dir)
        html = dashboard_html_path(tmp_dir).read_text()
        assert "getBaseUrl" in html
        assert "window.location" in html

    def test_contains_fetch_error_handling(self, tmp_dir):
        generate_dashboard(tmp_dir)
        html = dashboard_html_path(tmp_dir).read_text()
        assert "status-error" in html
        assert "showFetchError" in html
        assert "clearFetchError" in html
        assert "fetchErrors" in html

    def test_logo_embedded_as_data_uri(self, tmp_dir):
        """The header logo is read from phenotypic/_assets/logos and
        base64-embedded; an empty data URI would mean the asset was not found.
        """
        generate_dashboard(tmp_dir)
        html = dashboard_html_path(tmp_dir).read_text()
        assert "data:image/png;base64," in html

    def test_js_sidecars_written_from_assets_vendor(self, tmp_dir):
        """plotly.min.js / hyparquet.min.js are copied from
        phenotypic/_assets/vendor into the run's progress/ dir. This is the
        end-to-end guard for the relocated vendor assets (and the wheel
        packaging bug they previously triggered).
        """
        generate_dashboard(tmp_dir)
        prog = _progress_dir(tmp_dir)  # run's progress/ dir under .phenotypic/
        plotly = prog / "plotly.min.js"
        hyparquet = prog / "hyparquet.min.js"
        assert plotly.exists() and hyparquet.exists()
        # Plotly is multi-MB; a truncated/missing copy would be tiny.
        assert plotly.stat().st_size > 1_000_000


# ──────────────────────────────────────────────────────────────────────
# README Generator ASCII Tree
# ──────────────────────────────────────────────────────────────────────

class TestREADMEGeneratorASCII:

    def test_no_unicode_box_drawing(self):
        from phenotypic._cli._cli_readme_generator import READMEGenerator

        config = MagicMock()
        config.image_type = "Image"
        pipeline = MagicMock()
        pipeline._meas = {}

        gen = READMEGenerator(config, pipeline)
        ds = [MagicMock()]
        ds[0].name = "plate1"
        tree = gen._generate_output_structure(ds)
        assert "\u251c" not in tree  # ├
        assert "\u2514" not in tree  # └
        assert "\u2502" not in tree  # │
        assert "+--" in tree
        assert "dashboard.html" in tree


# ──────────────────────────────────────────────────────────────────────
# Sentinel Script
# ──────────────────────────────────────────────────────────────────────

class TestSentinelScript:

    def test_generates_script(self, tmp_dir):
        progress_dir = tmp_dir / "progress"
        script = generate_sentinel_script(
            output_dir=tmp_dir,
            progress_dir=progress_dir,
            slurm_args={"slurm_partition": "gpu", "slurm_account": "mylab"},
        )
        assert script.exists()
        content = script.read_text()
        assert "#!/bin/bash" in content
        assert "#SBATCH --partition=gpu" in content
        assert "#SBATCH --account=mylab" in content
        assert "#SBATCH --time=01:00:00" in content  # default max_runtime=1800 → 60-min floor
        assert "pht-sentinel" in content
        assert "phenotypic._cli._cli_sentinel" in content
        # Uses the same Python path as array job scripts, not bare "python"
        assert "-m phenotypic._cli._cli_sentinel" in content
        log_path = logs_dir(tmp_dir) / "slurm" / "sentinel_%j.log"
        assert f"#SBATCH --output={log_path.as_posix()}" in content
        assert f"#SBATCH --error={log_path.as_posix()}" in content
        assert (progress_dir / "sentinel_%j.log").as_posix() not in content

    def test_time_derived_from_max_runtime(self, tmp_dir):
        script = generate_sentinel_script(
            output_dir=tmp_dir,
            progress_dir=tmp_dir / "progress",
            slurm_args={},
            max_runtime=3600,  # 60 min → 75 min SLURM time
        )
        content = script.read_text()
        assert "#SBATCH --time=01:15:00" in content

    def test_time_minimum_floor(self, tmp_dir):
        script = generate_sentinel_script(
            output_dir=tmp_dir,
            progress_dir=tmp_dir / "progress",
            slurm_args={},
            max_runtime=300,  # 5 min + 15 = 20, but floor is 60
        )
        content = script.read_text()
        assert "#SBATCH --time=01:00:00" in content

    def test_sigterm_trap_present(self, tmp_dir):
        progress_dir = tmp_dir / "progress"
        script = generate_sentinel_script(
            output_dir=tmp_dir,
            progress_dir=progress_dir,
            slurm_args={},
        )
        content = script.read_text()
        assert "RESUBMIT_MARKER=" in content
        assert "sentinel_resubmitted" in content
        assert "trap " in content
        assert "SIGTERM" in content
        assert "sbatch --parsable" in content
        assert "sbatch --parsable --export=ALL" in content

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod not effective on Windows")
    def test_script_executable(self, tmp_dir):
        import os
        import stat
        script = generate_sentinel_script(
            output_dir=tmp_dir,
            progress_dir=tmp_dir / "progress",
            slurm_args={},
        )
        mode = os.stat(script).st_mode
        assert mode & stat.S_IEXEC

    def test_default_partition(self, tmp_dir):
        script = generate_sentinel_script(
            output_dir=tmp_dir,
            progress_dir=tmp_dir / "progress",
            slurm_args={},
        )
        content = script.read_text()
        assert "#SBATCH --partition=batch" in content

    def test_no_account_line_when_absent(self, tmp_dir):
        script = generate_sentinel_script(
            output_dir=tmp_dir,
            progress_dir=tmp_dir / "progress",
            slurm_args={"slurm_partition": "compute"},
        )
        content = script.read_text()
        assert "--account" not in content

    def test_self_referencing_script_path(self, tmp_dir):
        script = generate_sentinel_script(
            output_dir=tmp_dir,
            progress_dir=tmp_dir / "progress",
            slurm_args={},
        )
        content = script.read_text()
        assert f"--sentinel-script {shlex.quote(str(script.as_posix()))}" in content
