"""Sweep progress dashboard generator.

Generates a self-contained HTML dashboard that shows live progress
for parameter sweeps. Uses ``<meta http-equiv="refresh">`` for
auto-refresh since browsers cannot read local files via JavaScript.

SLURM workers call :func:`maybe_regenerate_dashboard` after logging
each completion event. Rate-limiting (skip if HTML was updated <10 s
ago) prevents I/O storms from concurrent workers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

_META_FILENAME = "sweep_progress_meta.json"
_DASHBOARD_FILENAME = "sweep_progress.html"
_RATE_LIMIT_SECONDS = 10


def write_sweep_progress_metadata(
    output_dir: Path,
    total_tasks: int,
    num_images: int,
    num_pipelines: int,
    start_time: datetime,
) -> Path:
    """Write sweep metadata JSON for dashboard generation.

    Args:
        output_dir: Sweep output directory.
        total_tasks: Total number of (image, pipeline) pairs.
        num_images: Number of images in the sweep.
        num_pipelines: Number of pipeline configurations.
        start_time: Sweep start time.

    Returns:
        Path to the written metadata file.
    """
    meta = {
        "total_tasks": total_tasks,
        "num_images": num_images,
        "num_pipelines": num_pipelines,
        "start_time": start_time.isoformat(timespec="milliseconds"),
    }
    meta_path = output_dir / _META_FILENAME
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def load_sweep_progress_metadata(
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Read sweep metadata JSON.

    Args:
        output_dir: Sweep output directory.

    Returns:
        Metadata dict, or ``None`` if file not found.
    """
    meta_path = output_dir / _META_FILENAME
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


# ---------------------------------------------------------------------------
# Rate-limited regeneration entry point
# ---------------------------------------------------------------------------


def maybe_regenerate_dashboard(
    output_dir: Path,
    event_log: Path,
) -> None:
    """Regenerate dashboard if not recently updated.

    Skips regeneration if the HTML file was modified less than
    :data:`_RATE_LIMIT_SECONDS` ago to avoid I/O storms when many
    SLURM workers finish near-simultaneously.

    Args:
        output_dir: Sweep output directory.
        event_log: Path to ``processing_events.log``.
    """
    dashboard_path = output_dir / _DASHBOARD_FILENAME

    # Rate-limit: skip if HTML was updated recently
    if dashboard_path.exists():
        age = time.time() - dashboard_path.stat().st_mtime
        if age < _RATE_LIMIT_SECONDS:
            return

    meta = load_sweep_progress_metadata(output_dir)
    if meta is None:
        return

    total_tasks = meta["total_tasks"]
    start_time = datetime.fromisoformat(meta["start_time"])

    generate_sweep_progress_dashboard(
        event_log=event_log,
        output_path=dashboard_path,
        total_tasks=total_tasks,
        start_time=start_time,
    )


# ---------------------------------------------------------------------------
# Dashboard generation
# ---------------------------------------------------------------------------


def generate_sweep_progress_dashboard(
    event_log: Path,
    output_path: Path,
    total_tasks: int,
    start_time: datetime,
    is_complete: bool = False,
) -> None:
    """Generate a self-contained HTML progress dashboard.

    Args:
        event_log: Path to ``processing_events.log``.
        output_path: Where to write the HTML file.
        total_tasks: Total (image, pipeline) pairs expected.
        start_time: Sweep start time.
        is_complete: If ``True``, omit auto-refresh meta tag.
    """
    completed, failed, failure_details, recent_lines = (
        _read_event_state(event_log)
    )
    remaining = max(0, total_tasks - completed - failed)
    elapsed = (datetime.now() - start_time).total_seconds()
    done = completed + failed

    if done > 0 and remaining > 0:
        eta_seconds = (elapsed / done) * remaining
        eta_str = _format_duration(eta_seconds)
    elif remaining == 0:
        eta_str = "Complete"
    else:
        eta_str = "Calculating..."

    pct = (done / total_tasks * 100) if total_tasks > 0 else 0

    html = _build_html(
        total=total_tasks,
        completed=completed,
        failed=failed,
        remaining=remaining,
        elapsed_str=_format_duration(elapsed),
        eta_str=eta_str,
        pct=pct,
        recent_lines=recent_lines,
        failure_details=failure_details,
        is_complete=is_complete,
    )

    # Atomic write: write to .tmp then os.replace
    tmp_path = output_path.with_suffix(".html.tmp")
    tmp_path.write_text(html)
    os.replace(tmp_path, output_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_event_state(
    event_log: Path,
) -> tuple[int, int, List[Dict[str, str]], List[str]]:
    """Read event log and extract dashboard state.

    Returns:
        Tuple of (completed_count, failed_count, failure_details,
        recent_raw_lines).
    """
    from phenotypic._cli._cli_update_state import (
        aggregate_state_from_events,
        parse_event_line,
    )

    completed = 0
    failed = 0
    failure_details: List[Dict[str, str]] = []

    if event_log.exists():
        states = aggregate_state_from_events(event_log)
        for ds in states.values():
            completed += len(ds.completed)
            failed += len(ds.failed)
            for img_name in ds.failed:
                failure_details.append({
                    "image": img_name,
                    "error": ds.errors.get(img_name, ""),
                })

    # Read last 20 raw lines for recent events table
    recent_lines: List[str] = []
    if event_log.exists():
        try:
            all_lines = event_log.read_text().strip().splitlines()
            recent_lines = all_lines[-20:]
        except Exception:
            pass

    return completed, failed, failure_details, recent_lines


def _build_html(
    *,
    total: int,
    completed: int,
    failed: int,
    remaining: int,
    elapsed_str: str,
    eta_str: str,
    pct: float,
    recent_lines: List[str],
    failure_details: List[Dict[str, str]],
    is_complete: bool,
) -> str:
    """Build the full HTML string for the dashboard."""
    refresh_tag = (
        ""
        if is_complete
        else '    <meta http-equiv="refresh" content="5">\n'
    )

    status_text = "Complete" if is_complete else "In Progress"
    status_class = "success" if is_complete else "info"

    recent_rows = _build_recent_events_table(recent_lines)
    failure_section = _build_failure_section(failure_details)

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>PhenoTypic Sweep Progress</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
{refresh_tag}{_get_styles()}
</head>
<body>
    <div class="container">
        <h1>PhenoTypic Sweep Progress</h1>
        <div class="metadata">
            <p><strong>Status:</strong> \
<span class="{status_class}-text">{status_text}</span></p>
            <p><strong>Elapsed:</strong> {elapsed_str}</p>
            <p><strong>ETA:</strong> {eta_str}</p>
        </div>

        <div class="summary">
            <div class="stat-card info">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Tasks</div>
            </div>
            <div class="stat-card success">
                <div class="stat-value">{completed}</div>
                <div class="stat-label">Completed</div>
            </div>
            <div class="stat-card failure">
                <div class="stat-value">{failed}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{remaining}</div>
                <div class="stat-label">Remaining</div>
            </div>
        </div>

        <div class="progress-container">
            <div class="progress-label">\
{completed + failed}/{total} processed ({pct:.1f}%)</div>
            <progress value="{completed + failed}" max="{total}"></progress>
        </div>

        <h2>Recent Events</h2>
{recent_rows}

{failure_section}
    </div>
</body>
</html>"""


def _build_recent_events_table(recent_lines: List[str]) -> str:
    """Build HTML table of recent events from raw log lines."""
    from phenotypic._cli._cli_update_state import parse_event_line

    if not recent_lines:
        return "        <p>No events recorded yet.</p>"

    rows: List[str] = []
    for line in reversed(recent_lines):
        try:
            ev = parse_event_line(line)
        except ValueError:
            continue
        status_cls = (
            "success-text" if ev.status == "completed" else "failure-text"
        )
        ts = ev.timestamp.strftime("%H:%M:%S")
        rows.append(
            f"                <tr>"
            f"<td class=\"timestamp\">{ts}</td>"
            f"<td>{_escape_html(ev.image)}</td>"
            f"<td class=\"{status_cls}\">{ev.status}</td>"
            f"</tr>"
        )

    return f"""        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Task</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
{chr(10).join(rows)}
            </tbody>
        </table>"""


def _build_failure_section(
    failure_details: List[Dict[str, str]],
) -> str:
    """Build collapsible failure details section."""
    if not failure_details:
        return ""

    rows: List[str] = []
    for f in failure_details:
        error_text = _escape_html(f["error"]) if f["error"] else "—"
        rows.append(
            f"                        <tr>"
            f"<td><strong>{_escape_html(f['image'])}</strong></td>"
            f"<td>{error_text}</td>"
            f"</tr>"
        )

    return f"""        <details>
            <summary>Failures ({len(failure_details)})</summary>
            <table>
                <thead>
                    <tr><th>Task</th><th>Error</th></tr>
                </thead>
                <tbody>
{chr(10).join(rows)}
                </tbody>
            </table>
        </details>"""


def _get_styles() -> str:
    """Inline CSS matching _cli_report_generator.py style."""
    return """    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                         'Roboto', 'Helvetica', 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f7fa;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
            font-size: 2.5em;
        }

        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }

        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }

        .stat-card:hover {
            transform: translateY(-5px);
        }

        .stat-card.success {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }

        .stat-card.failure {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }

        .stat-card.info {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }

        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .success-text { color: #27ae60; font-weight: bold; }
        .failure-text { color: #e74c3c; font-weight: bold; }
        .info-text { color: #3498db; font-weight: bold; }

        progress {
            width: 100%;
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            border: none;
            background: #ecf0f1;
        }

        progress::-webkit-progress-bar {
            background: #ecf0f1;
            border-radius: 15px;
        }

        progress::-webkit-progress-value {
            background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
            border-radius: 15px;
        }

        progress::-moz-progress-bar {
            background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
            border-radius: 15px;
        }

        .progress-container {
            margin: 20px 0;
        }

        .progress-label {
            margin-bottom: 10px;
            font-size: 1.1em;
            color: #555;
        }

        .metadata {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 6px;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }

        .metadata p {
            margin: 5px 0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }

        th {
            background: #34495e;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }

        tr:hover {
            background: #f8f9fa;
        }

        tr:last-child td {
            border-bottom: none;
        }

        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
        }

        details {
            margin: 15px 0;
            border: 1px solid #ddd;
            border-radius: 6px;
            overflow: hidden;
        }

        summary {
            cursor: pointer;
            padding: 15px;
            background: #ecf0f1;
            font-weight: 600;
            user-select: none;
            transition: background 0.2s;
        }

        summary:hover {
            background: #d5dbdb;
        }

        summary::-webkit-details-marker {
            display: none;
        }

        summary::before {
            content: '\25b6';
            display: inline-block;
            margin-right: 10px;
            transition: transform 0.2s;
        }

        details[open] summary::before {
            transform: rotate(90deg);
        }

        @media print {
            body { background: white; }
            .container { box-shadow: none; }
        }
    </style>"""


def _format_duration(seconds: float) -> str:
    """Format duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / 3600:.1f} hr"


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
