"""CalibrateColorRpcc as a PlotImage: the provider contract (figures spec §3a).

The overlay shows the as-shot pixels that ``apply()`` overwrites, so it can
be drawn only for the image the last ``apply()`` ran on, in that process.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import warnings
import weakref

import pytest

from phenotypic import Image, ImagePipeline
from phenotypic.abc_.plotting import FigureInputUnavailable, PlotImage
from phenotypic.correction import CalibrateColorRpcc
from phenotypic.plotting._pipeline._store_figures import build_image_figures
from phenotypic.plotting._pipeline._store_formats import serialize_store_format
from phenotypic.sdk_._image_figures import FigureRun

from phenotypic.correction._color_correction._calibration_overlay import (
    render_calibration_overlay,
)

from ._checker_frames import band_prior, band_rois, frozen_op, render_frame


def _png(figure) -> bytes:
    return serialize_store_format("png", figure, binding_id="cal", page_key="default")


def _applied() -> tuple[CalibrateColorRpcc, Image]:
    frame = Image(arr=render_frame())
    operation = frozen_op()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        operation.apply(frame, inplace=True)
    return operation, frame


def test_the_signal_is_public_and_imports_no_plotting_library():
    # The plotting contract package is stdlib-only at runtime; `phenotypic.abc_`
    # itself is not light, so only the plotting libraries are asked about.
    probe = (
        "import sys\n"
        "from phenotypic.abc_.plotting import FigureInputUnavailable\n"
        "assert issubclass(FigureInputUnavailable, RuntimeError)\n"
        "print(sorted(m for m in ('matplotlib', 'plotly', 'kaleido')"
        " if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_it_is_an_image_plot():
    assert isinstance(frozen_op(), PlotImage)


def _ids(output) -> list[tuple[str, str]]:
    return [(page.plot_name, page.key) for page in output.pages]


def test_inspect_draws_one_overlay_per_roi_and_the_delta_e_chart():
    operation, frame = _applied()
    record = operation.calibration_record
    assert len(record.rois) == 2
    output = operation.inspect(frame, for_save=True)
    assert _ids(output) == [("tiles", "roi_0"), ("tiles", "roi_1"), ("delta_e", "delta_e")]
    for roi, page in zip(record.rois, output.pages):
        alone = render_calibration_overlay(record.model_copy(update={"rois": [roi]}))
        assert _png(page.figure) == _png(alone)  # exactly that ROI, nothing else
    assert _png(output.pages[2].figure) == _png(operation.show_delta_bar_plot())
    assert _png(output.pages[0].figure) != _png(output.pages[1].figure)
    # No subject: the held image, which is still alive.
    held = operation.inspect()
    assert [_png(p.figure) for p in held.pages] == [_png(p.figure) for p in output.pages]


def test_a_skipped_frame_still_draws_every_roi():
    operation = frozen_op(on_qc_fail="skip")
    frame = Image(arr=render_frame(gain=1.6))  # saturated card
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        operation.apply(frame, inplace=True)
    assert operation.calibration_record.verdict == "skipped"
    output = operation.inspect(frame)
    assert _ids(output) == [("tiles", "roi_0"), ("tiles", "roi_1"), ("delta_e", "delta_e")]
    (ax,) = output.pages[2].figure.axes
    assert ax.containers == []
    assert all(_png(page.figure).startswith(b"\x89PNG") for page in output.pages)


def test_one_roi_stores_one_overlay():
    """Review Focus 4."""
    # One band is 12 patches; a degree-3 fit needs 13 (`require_rank`), and that
    # fails before QC, so `on_qc_fail` cannot help. Degree 2 fits 12 (probed
    # 2026-09-30: verdict "corrected", 12/24 fitted).
    operation = frozen_op(rois=band_rois()[:1], lattice_prior=[band_prior()], degree=2)
    frame = Image(arr=render_frame())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        operation.apply(frame, inplace=True)
    assert operation.calibration_record.verdict == "corrected"
    assert _ids(operation.inspect(frame)) == [("tiles", "roi_0"), ("delta_e", "delta_e")]


def test_another_image_is_unavailable():
    operation, frame = _applied()
    for other in (frame.copy(), Image(arr=render_frame())):
        with pytest.raises(FigureInputUnavailable, match="another image"):
            operation.inspect(other)


def test_a_fresh_instance_is_unavailable():
    operation = frozen_op()
    with pytest.raises(FigureInputUnavailable, match="no calibration record"):
        operation.inspect()
    with pytest.raises(FigureInputUnavailable, match="no calibration record"):
        operation.inspect(Image(arr=render_frame()))


def test_the_record_does_not_pin_the_image():
    operation, frame = _applied()
    released = weakref.ref(frame)
    del frame
    gc.collect()
    assert released() is None
    assert operation.calibration_record is not None
    with pytest.raises(FigureInputUnavailable, match="no calibration record"):
        operation.inspect()


def test_overrides_are_refused():
    operation, frame = _applied()
    with pytest.raises(ValueError, match="takes none"):
        operation.inspect(frame, figsize=(1, 1))


_FIELDS = {
    "rois", "checker_type", "target_illuminant", "degree", "grid",
    "lattice_prior", "refine_method", "core_trim", "medoid_candidates",
    "outlier_sigma", "min_patches", "qc_limits", "on_qc_fail",
    "fitted_profile", "qc",
}


def test_serialization_is_unchanged():
    assert set(CalibrateColorRpcc.model_json_schema()["properties"]) == _FIELDS
    operation, _frame = _applied()
    assert set(json.loads(operation.to_json())["params"]) == _FIELDS
    fresh = frozen_op()
    restored = CalibrateColorRpcc.from_json(fresh.to_json())
    assert restored.model_dump() == fresh.model_dump()


def test_a_pipeline_binds_it_by_reference_and_round_trips():
    operation = frozen_op()
    pipeline = ImagePipeline(ops={"cal": operation}, plots=[operation])
    [binding] = pipeline.get_plots()
    assert (binding.id, binding.ref.slot, binding.ref.key) == ("cal", "ops", "cal")
    restored = ImagePipeline.from_json(pipeline.to_json())
    [restored_binding] = restored.get_plots()
    assert restored_binding.plot is restored.get_ops()["cal"]


def test_an_image_pipeline_listing_it_under_plots_stores_every_figure_in_its_plot_folder():
    operation = frozen_op()
    pipeline = ImagePipeline(ops={"cal": operation}, plots=[operation])
    frame = Image(arr=render_frame())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        corrected = pipeline.apply(frame)
    run = FigureRun(date="2026-09-23", pipeline_sha256="0" * 64)
    stored = build_image_figures(pipeline, corrected, run=run)

    assert (stored.failed, stored.unavailable) == ((), ())
    [binding] = stored.bindings
    assert (binding.binding_id, binding.plot_class) == ("cal", "CalibrateColorRpcc")
    assert [(p.directory, p.key, p.files[0].filename) for p in binding.pages] == [
        ("tiles", "roi_0", "roi_0.png"), ("tiles", "roi_1", "roi_1.png"),
        ("delta_e", "delta_e", "delta_e.png"),
    ]
    assert {p.backend for p in binding.pages} == {"mpl"}
    for page in binding.pages:
        [file] = page.files
        assert file.format == "png"
        assert file.data.startswith(b"\x89PNG")
    assert binding.pages[0].files[0].data != binding.pages[1].files[0].data
