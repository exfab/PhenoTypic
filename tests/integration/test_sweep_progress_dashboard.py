"""Tests for sweep progress dashboard generation."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path


from phenotypic.sweep._sweep_cli._sweep_progress_dashboard import (
    _DASHBOARD_FILENAME,
    _META_FILENAME,
    _RATE_LIMIT_SECONDS,
    generate_sweep_progress_dashboard,
    load_sweep_progress_metadata,
    maybe_regenerate_dashboard,
    write_sweep_progress_metadata,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _write_mock_event_log(path: Path, events: list[tuple[str, str]]) -> None:
    """Write a mock event log with (image_id, status) pairs."""
    lines = []
    base = datetime(2026, 1, 15, 10, 0, 0)
    for i, (image_id, status) in enumerate(events):
        ts = (base + timedelta(seconds=i * 5)).isoformat(
            timespec="milliseconds"
        )
        error = "some error" if status == "failed" else ""
        lines.append(f"{ts}|sweep|{image_id}|{status}|{error}")
    path.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------
# Metadata round-trip
# ------------------------------------------------------------------


class TestMetadata:
    def test_write_and_load(self, tmp_path: Path) -> None:
        start = datetime(2026, 1, 15, 10, 0, 0)
        write_sweep_progress_metadata(
            tmp_path, total_tasks=24, num_images=4,
            num_pipelines=6, start_time=start,
        )
        meta = load_sweep_progress_metadata(tmp_path)
        assert meta is not None
        assert meta["total_tasks"] == 24
        assert meta["num_images"] == 4
        assert meta["num_pipelines"] == 6
        assert meta["start_time"] == start.isoformat(
            timespec="milliseconds"
        )

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        assert load_sweep_progress_metadata(tmp_path) is None

    def test_json_file_valid(self, tmp_path: Path) -> None:
        write_sweep_progress_metadata(
            tmp_path, total_tasks=10, num_images=2,
            num_pipelines=5, start_time=datetime.now(),
        )
        raw = json.loads((tmp_path / _META_FILENAME).read_text())
        assert isinstance(raw, dict)
        assert set(raw.keys()) == {
            "total_tasks", "num_images", "num_pipelines", "start_time",
        }


# ------------------------------------------------------------------
# HTML generation
# ------------------------------------------------------------------


class TestGenerateDashboard:
    def test_basic_html_generation(self, tmp_path: Path) -> None:
        event_log = tmp_path / "processing_events.log"
        _write_mock_event_log(event_log, [
            ("img1.tif::pipe_A", "completed"),
            ("img1.tif::pipe_B", "completed"),
            ("img2.tif::pipe_A", "failed"),
        ])
        out = tmp_path / _DASHBOARD_FILENAME

        generate_sweep_progress_dashboard(
            event_log=event_log,
            output_path=out,
            total_tasks=6,
            start_time=datetime.now() - timedelta(minutes=5),
        )

        html = out.read_text()
        assert "<!DOCTYPE html>" in html
        assert "PhenoTypic Sweep Progress" in html
        # Stat cards
        assert ">6<" in html  # total
        assert ">2<" in html  # completed
        assert ">1<" in html  # failed
        assert ">3<" in html  # remaining
        # Auto-refresh present
        assert 'http-equiv="refresh"' in html

    def test_is_complete_omits_refresh(self, tmp_path: Path) -> None:
        event_log = tmp_path / "processing_events.log"
        _write_mock_event_log(event_log, [
            ("img1.tif::pipe_A", "completed"),
        ])
        out = tmp_path / _DASHBOARD_FILENAME

        generate_sweep_progress_dashboard(
            event_log=event_log,
            output_path=out,
            total_tasks=1,
            start_time=datetime.now() - timedelta(seconds=30),
            is_complete=True,
        )

        html = out.read_text()
        assert 'http-equiv="refresh"' not in html
        assert "Complete" in html

    def test_empty_event_log(self, tmp_path: Path) -> None:
        event_log = tmp_path / "processing_events.log"
        # Event log doesn't exist yet
        out = tmp_path / _DASHBOARD_FILENAME

        generate_sweep_progress_dashboard(
            event_log=event_log,
            output_path=out,
            total_tasks=10,
            start_time=datetime.now(),
        )

        html = out.read_text()
        assert ">0<" in html  # completed = 0
        assert ">10<" in html  # total and remaining
        assert "No events recorded yet." in html

    def test_recent_events_table(self, tmp_path: Path) -> None:
        event_log = tmp_path / "processing_events.log"
        _write_mock_event_log(event_log, [
            ("img1.tif::pipe_A", "completed"),
            ("img2.tif::pipe_A", "failed"),
        ])
        out = tmp_path / _DASHBOARD_FILENAME

        generate_sweep_progress_dashboard(
            event_log=event_log,
            output_path=out,
            total_tasks=4,
            start_time=datetime.now() - timedelta(minutes=1),
        )

        html = out.read_text()
        assert "img1.tif::pipe_A" in html
        assert "img2.tif::pipe_A" in html
        assert "Recent Events" in html

    def test_failure_details_section(self, tmp_path: Path) -> None:
        event_log = tmp_path / "processing_events.log"
        _write_mock_event_log(event_log, [
            ("img1.tif::pipe_A", "failed"),
        ])
        out = tmp_path / _DASHBOARD_FILENAME

        generate_sweep_progress_dashboard(
            event_log=event_log,
            output_path=out,
            total_tasks=2,
            start_time=datetime.now(),
        )

        html = out.read_text()
        assert "Failures (1)" in html
        assert "some error" in html

    def test_atomic_write_no_corruption(self, tmp_path: Path) -> None:
        """Verify that the .tmp file is cleaned up after write."""
        event_log = tmp_path / "processing_events.log"
        _write_mock_event_log(event_log, [
            ("img1.tif::pipe_A", "completed"),
        ])
        out = tmp_path / _DASHBOARD_FILENAME

        generate_sweep_progress_dashboard(
            event_log=event_log,
            output_path=out,
            total_tasks=1,
            start_time=datetime.now(),
        )

        assert out.exists()
        # Temp file should be cleaned up by os.replace
        assert not out.with_suffix(".html.tmp").exists()
        # HTML should be valid (not truncated)
        html = out.read_text()
        assert html.endswith("</html>")


# ------------------------------------------------------------------
# Rate-limiting
# ------------------------------------------------------------------


class TestMaybeRegenerateDashboard:
    def test_generates_when_no_dashboard_exists(
        self, tmp_path: Path
    ) -> None:
        event_log = tmp_path / "processing_events.log"
        _write_mock_event_log(event_log, [
            ("img1.tif::pipe_A", "completed"),
        ])
        write_sweep_progress_metadata(
            tmp_path, total_tasks=2, num_images=1,
            num_pipelines=2, start_time=datetime.now(),
        )

        maybe_regenerate_dashboard(tmp_path, event_log)

        assert (tmp_path / _DASHBOARD_FILENAME).exists()

    def test_skips_when_recently_updated(self, tmp_path: Path) -> None:
        event_log = tmp_path / "processing_events.log"
        _write_mock_event_log(event_log, [
            ("img1.tif::pipe_A", "completed"),
        ])
        write_sweep_progress_metadata(
            tmp_path, total_tasks=2, num_images=1,
            num_pipelines=2, start_time=datetime.now(),
        )

        # Generate once
        maybe_regenerate_dashboard(tmp_path, event_log)
        first_mtime = (tmp_path / _DASHBOARD_FILENAME).stat().st_mtime

        # Immediately try again — should be rate-limited
        maybe_regenerate_dashboard(tmp_path, event_log)
        second_mtime = (tmp_path / _DASHBOARD_FILENAME).stat().st_mtime

        assert first_mtime == second_mtime

    def test_regenerates_after_rate_limit_expires(
        self, tmp_path: Path
    ) -> None:
        event_log = tmp_path / "processing_events.log"
        _write_mock_event_log(event_log, [
            ("img1.tif::pipe_A", "completed"),
        ])
        write_sweep_progress_metadata(
            tmp_path, total_tasks=2, num_images=1,
            num_pipelines=2, start_time=datetime.now(),
        )

        maybe_regenerate_dashboard(tmp_path, event_log)
        dashboard = tmp_path / _DASHBOARD_FILENAME

        # Backdate file mtime beyond rate limit
        old_time = time.time() - _RATE_LIMIT_SECONDS - 1
        os.utime(dashboard, (old_time, old_time))
        old_mtime = dashboard.stat().st_mtime

        maybe_regenerate_dashboard(tmp_path, event_log)
        new_mtime = dashboard.stat().st_mtime

        assert new_mtime > old_mtime

    def test_skips_when_no_metadata(self, tmp_path: Path) -> None:
        event_log = tmp_path / "processing_events.log"
        _write_mock_event_log(event_log, [
            ("img1.tif::pipe_A", "completed"),
        ])

        # No metadata written — should silently skip
        maybe_regenerate_dashboard(tmp_path, event_log)

        assert not (tmp_path / _DASHBOARD_FILENAME).exists()
