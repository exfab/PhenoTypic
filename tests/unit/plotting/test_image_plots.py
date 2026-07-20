"""Standalone image plot replacements for the removed accessor."""

import gc
import weakref

import numpy as np

from phenotypic import Image
from phenotypic._core._image_parts.plot_accessor._diagnostics_plotter import (
    DiagnosticsPlotter,
)
from phenotypic.abc_.plotting import PlotImage
from phenotypic.data import load_synth_yeast_plate
from phenotypic.plotting import PlotDetectModes, PlotDiagnostics


def test_detect_modes_is_standalone_plot_image() -> None:
    plot = PlotDetectModes()
    figure = plot.inspect(load_synth_yeast_plate())
    assert isinstance(plot, PlotImage)
    assert len(figure.data) > 0


def test_diagnostics_is_standalone_plot_image() -> None:
    plot = PlotDiagnostics()
    figure = plot.inspect(load_synth_yeast_plate())
    assert isinstance(plot, PlotImage)
    assert len(figure.data) > 0


def test_diagnostics_report_does_not_retain_image_or_accessor(
    monkeypatch,
) -> None:
    plot = PlotDiagnostics()
    accessor_refs: list[weakref.ReferenceType[DiagnosticsPlotter]] = []
    original_init = DiagnosticsPlotter.__init__

    def record_accessor(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        accessor_refs.append(weakref.ref(self))

    monkeypatch.setattr(DiagnosticsPlotter, "__init__", record_accessor)
    # Construct directly because pytest's tracing can retain frames from the
    # file-backed synthetic-data loader independently of the plotting report.
    image = Image(
        arr=np.arange(64 * 64, dtype=np.uint8).reshape(64, 64),
        bit_depth=8,
    )
    image_ref = weakref.ref(image)

    report = plot.report(image)
    del image
    gc.collect()

    assert report is not None
    assert accessor_refs
    assert all(accessor_ref() is None for accessor_ref in accessor_refs)
    assert image_ref() is None
