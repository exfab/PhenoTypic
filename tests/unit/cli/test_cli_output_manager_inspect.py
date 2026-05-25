"""Unit tests for :meth:`OutputManager.save_inspect` figure dispatch.

The CLI's ``--save-inspect`` flag iterates the pipeline's measurers and
dispatches each one through :meth:`OutputManager.save_inspect`, which
type-dispatches the returned figure (matplotlib vs plotly) and writes a
PNG. This module pins the per-figure-type behavior, the unsupported-type
warning, and the inspect-raises path — all the cases the CLI integration
test cannot cleanly exercise.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from tests.unit.cli._kaleido_utils import requires_kaleido_chrome

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="OutputManager uses POSIX atomic writes",
)

from phenotypic._cli._cli_output_manager import OutputManager


class _MplStubMeasurer:
    """Returns a tiny matplotlib Figure when ``inspect()`` is called."""

    def inspect(self, image, *, for_save: bool = False):
        import matplotlib

        matplotlib.use("Agg")  # headless backend; safe under pytest
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        return fig


class _PlotlyStubMeasurer:
    """Returns a tiny plotly Figure when ``inspect()`` is called."""

    def inspect(self, image, *, for_save: bool = False):
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]))
        fig.update_layout(title={"text": "stub-title"})
        return fig


class _UnsupportedStubMeasurer:
    """Returns a plain string — not a supported figure type."""

    def inspect(self, image, *, for_save: bool = False):
        return "this is not a figure"


class _RaisingStubMeasurer:
    """Raises whenever ``inspect()`` is called."""

    def inspect(self, image, *, for_save: bool = False):
        raise RuntimeError("synthetic inspect failure")


def _make_output_manager(base_dir: Path) -> OutputManager:
    """Construct an OutputManager with save_inspects=True and create the
    base dataset directory structure save_inspect() expects.
    """
    om = OutputManager.from_config(
        base_dir=base_dir,
        ext=".png",
        include_dataset_column=False,
        overlay_alpha=0.3,
        save_overlays=False,
        save_inspects=True,
    )
    # save_inspect() requires the results/<ds>/inspect/<key>/ tree;
    # _atomic_write creates intermediate dirs via Path.parent.mkdir,
    # but the dataset root itself is created by create_structure().
    om.results_dir.mkdir(parents=True, exist_ok=True)
    return om


class TestSaveInspect:
    """End-to-end behavior of :meth:`OutputManager.save_inspect`."""

    def test_mpl_figure_writes_non_empty_png(self, tmp_path: Path) -> None:
        om = _make_output_manager(tmp_path)
        result = om.save_inspect(
            _MplStubMeasurer(), image=None,
            dataset_name="ds1", image_stem="img1",
            measurer_key="MplStub",
        )
        assert result is not None
        expected = tmp_path / "results" / "ds1" / "inspect" / "MplStub" / "img1.png"
        assert result == expected
        assert expected.exists()
        assert expected.stat().st_size > 0
        # PNG magic bytes
        assert expected.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    @requires_kaleido_chrome
    def test_plotly_figure_writes_non_empty_png(self, tmp_path: Path) -> None:
        om = _make_output_manager(tmp_path)
        result = om.save_inspect(
            _PlotlyStubMeasurer(), image=None,
            dataset_name="ds1", image_stem="img2",
            measurer_key="PlotlyStub",
        )
        assert result is not None
        expected = (
            tmp_path / "results" / "ds1" / "inspect" / "PlotlyStub" / "img2.png"
        )
        assert result == expected
        assert expected.exists()
        assert expected.stat().st_size > 0
        assert expected.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_unsupported_figure_type_logs_warning_no_file(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        om = _make_output_manager(tmp_path)
        with caplog.at_level(logging.WARNING):
            result = om.save_inspect(
                _UnsupportedStubMeasurer(), image=None,
                dataset_name="ds1", image_stem="img3",
                measurer_key="UnsupportedStub",
            )
        assert result is None
        expected = (
            tmp_path / "results" / "ds1" / "inspect" / "UnsupportedStub" / "img3.png"
        )
        assert not expected.exists()
        assert any("unsupported figure type" in r.message for r in caplog.records)

    def test_inspect_raising_logs_warning_no_file_no_propagate(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        om = _make_output_manager(tmp_path)
        with caplog.at_level(logging.WARNING):
            result = om.save_inspect(
                _RaisingStubMeasurer(), image=None,
                dataset_name="ds1", image_stem="img4",
                measurer_key="RaisingStub",
            )
        assert result is None
        expected = (
            tmp_path / "results" / "ds1" / "inspect" / "RaisingStub" / "img4.png"
        )
        assert not expected.exists()
        # Worker must keep going — no exception propagates, and the
        # warning message names the failure type.
        assert any(
            "inspect() raised" in r.message and "RuntimeError" in r.message
            for r in caplog.records
        )

    @requires_kaleido_chrome
    def test_title_at_write_time_prepends_stem_to_existing_plotly_title(
        self, tmp_path: Path,
    ) -> None:
        """At the moment the PNG is written, the figure's title must
        lead with the image stem so a folder of PNGs is self-describing.

        The writer restores the original title after the write so that
        a cached figure object is left exactly as the measurer
        returned it (see ``test_title_does_not_compound``); the
        intermediate prepended state is only observable by hooking
        ``write_image`` itself.
        """
        om = _make_output_manager(tmp_path)
        m = _PlotlyStubMeasurer()
        fig = m.inspect(None, for_save=True)
        assert str(fig.layout.title.text) == "stub-title"

        title_at_write: list[str] = []
        original_write_image = fig.write_image

        def _spy_write_image(path, **kwargs):
            title_at_write.append(str(fig.layout.title.text))
            return original_write_image(path, **kwargs)

        fig.write_image = _spy_write_image  # type: ignore[method-assign]
        writer = om._build_inspect_writer(fig, image_stem="2024-03-14_plate_A2")
        writer(str(tmp_path / "out.png"))

        assert title_at_write, "write_image was never invoked"
        captured = title_at_write[0]
        assert captured.startswith("2024-03-14_plate_A2")
        assert "stub-title" in captured

    @requires_kaleido_chrome
    def test_title_does_not_compound_on_repeat_calls(
        self, tmp_path: Path,
    ) -> None:
        """Repeated saves on the same cached figure must NOT compound
        the prepended stem.

        Regression guard against a future measurer that returns a
        cached figure across calls; without the restore-after-write
        step in :meth:`_build_inspect_writer`, this would produce
        ``"img2 — img1 — original"`` titles.
        """
        om = _make_output_manager(tmp_path)
        m = _PlotlyStubMeasurer()
        fig = m.inspect(None, for_save=True)

        title_at_write: list[str] = []
        original_write_image = fig.write_image

        def _spy_write_image(path, **kwargs):
            title_at_write.append(str(fig.layout.title.text))
            return original_write_image(path, **kwargs)

        fig.write_image = _spy_write_image  # type: ignore[method-assign]

        om._build_inspect_writer(fig, image_stem="img1")(
            str(tmp_path / "a.png"),
        )
        om._build_inspect_writer(fig, image_stem="img2")(
            str(tmp_path / "b.png"),
        )

        assert len(title_at_write) == 2
        first, second = title_at_write
        assert first.startswith("img1")
        assert second.startswith("img2")
        assert "img1" not in second, f"compounding detected: {second!r}"
        # And the figure object is restored to its pristine state.
        assert str(fig.layout.title.text) == "stub-title"

    def test_create_structure_provisions_inspect_dir_when_enabled(
        self, tmp_path: Path,
    ) -> None:
        from phenotypic._cli._cli_types import Dataset

        om = OutputManager.from_config(
            base_dir=tmp_path,
            ext=".png",
            include_dataset_column=False,
            overlay_alpha=0.3,
            save_overlays=False,
            save_inspects=True,
        )
        om.create_structure([Dataset(
            name="ds1", images=[],
            input_dir=tmp_path, output_dir=tmp_path,
        )])
        assert (tmp_path / "results" / "ds1" / "inspect").is_dir()

    def test_create_structure_skips_inspect_dir_when_disabled(
        self, tmp_path: Path,
    ) -> None:
        from phenotypic._cli._cli_types import Dataset

        om = OutputManager.from_config(
            base_dir=tmp_path,
            ext=".png",
            include_dataset_column=False,
            overlay_alpha=0.3,
            save_overlays=False,
            save_inspects=False,
        )
        om.create_structure([Dataset(
            name="ds1", images=[],
            input_dir=tmp_path, output_dir=tmp_path,
        )])
        assert not (tmp_path / "results" / "ds1" / "inspect").exists()
