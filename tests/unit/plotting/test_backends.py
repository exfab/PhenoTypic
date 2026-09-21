"""Rendering capability and the hoisted plotly.min.js bundle."""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.plotting._pipeline._backends import (
    chrome_available,
    ensure_plotlyjs_bundle,
    plotlyjs_src_for,
    reset_chrome_probe,
)


@pytest.fixture(autouse=True)
def _clear_probe():
    reset_chrome_probe()
    yield
    reset_chrome_probe()


def test_the_bundle_is_written_once(tmp_path: Path) -> None:
    first = ensure_plotlyjs_bundle(tmp_path)
    assert first == tmp_path / "plotly.min.js"
    assert first.stat().st_size > 1_000_000
    stamp = first.stat().st_mtime_ns

    second = ensure_plotlyjs_bundle(tmp_path)
    assert second == first
    assert second.stat().st_mtime_ns == stamp, "bundle was rewritten"


def test_the_bundle_is_rewritten_if_truncated(tmp_path: Path) -> None:
    bundle = ensure_plotlyjs_bundle(tmp_path)
    bundle.write_text("corrupted")
    assert ensure_plotlyjs_bundle(tmp_path).stat().st_size > 1_000_000


@pytest.mark.parametrize(
    "page_dir, expected",
    [
        ("plots/sym", "../plotly.min.js"),
        ("plots/sym/ds-1", "../../plotly.min.js"),
        ("plots/sym/ds-1/A01-abc123", "../../../plotly.min.js"),
    ],
)
def test_the_src_resolves_from_every_layout(
    tmp_path: Path, page_dir: str, expected: str
) -> None:
    """Aggregate, single-page image, and multi-page image layouts."""
    bundle = tmp_path / "plots" / "plotly.min.js"
    assert plotlyjs_src_for(tmp_path / page_dir, bundle) == expected


def test_the_probe_is_memoised(monkeypatch) -> None:
    calls = {"n": 0}

    def _counting_to_image(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("Kaleido requires Google Chrome to be installed.")

    import plotly.io as pio

    monkeypatch.setattr(pio, "to_image", _counting_to_image)
    assert chrome_available() is False
    assert chrome_available() is False
    assert calls["n"] == 1, "the probe ran more than once"


def test_the_probe_reports_success(monkeypatch) -> None:
    import plotly.io as pio

    monkeypatch.setattr(pio, "to_image", lambda *a, **k: b"\x89PNG")
    assert chrome_available() is True
