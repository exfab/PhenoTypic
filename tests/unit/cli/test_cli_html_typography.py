"""The CLI's standalone HTML outputs follow the GUI typography (DESIGN.md).

Nunito Sans carries general text and JetBrains Mono every data and table value,
loaded from the same stylesheet URL as the GUI; the retired DM font trio is gone.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from phenotypic._cli._cli_report_generator import HTMLReportGenerator
from phenotypic._cli._cli_types import ExecutionResults
from phenotypic._cli._dashboard._generator import generate_dashboard
from phenotypic._gui._design import FONT_FAMILY_BODY, FONT_FAMILY_MONO, GOOGLE_FONTS_URL


def _dashboard_html(tmp_path: Path) -> str:
    generate_dashboard(tmp_path, execution_mode="local")
    [path] = list(tmp_path.rglob("*.html"))
    return path.read_text(encoding="utf-8")


def _report_html(tmp_path: Path) -> str:
    results = ExecutionResults(
        datasets={},
        total_images=0,
        total_completed=0,
        total_failed=0,
        execution_mode="local",
        start_time=datetime(2026, 9, 29, 12, 0),
        end_time=datetime(2026, 9, 29, 12, 5),
    )
    path = tmp_path / "processing_report.html"
    HTMLReportGenerator().generate_report(results, path)
    return path.read_text(encoding="utf-8")


@pytest.fixture(params=["dashboard", "report"])
def html(request, tmp_path: Path) -> str:
    builder = {"dashboard": _dashboard_html, "report": _report_html}[request.param]
    return builder(tmp_path)


def test_loads_the_gui_font_stylesheet(html: str) -> None:
    assert GOOGLE_FONTS_URL in html


def test_declares_the_gui_role_fonts(html: str) -> None:
    assert f"--font-body:    {FONT_FAMILY_BODY};" in html
    assert f"--font-mono:    {FONT_FAMILY_MONO};" in html
    assert "--font-size-data: 0.9375rem;" in html


def test_retired_dm_fonts_are_gone(html: str) -> None:
    for retired in ("DM Serif Display", "DM Sans", "DM Mono", "DM+Sans"):
        assert retired not in html


def test_no_placeholder_leaks(html: str) -> None:
    assert "__PHENOTYPIC_TYPE_TOKENS__" not in html


def test_display_headings_use_weight_600(html: str) -> None:
    """Display styles moved from 400 to 600 with the Nunito Sans chrome."""
    blocks = [block for block in html.split("}") if "font-family: var(--font-display)" in block]
    assert blocks
    for block in blocks:
        assert "font-weight: 400" not in block
