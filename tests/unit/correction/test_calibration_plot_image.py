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
from phenotypic.plotting._pipeline._store_formats import serialize_store_format

from ._checker_frames import frozen_op, render_frame


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


def test_inspect_draws_what_show_tiles_draws():
    operation, frame = _applied()
    expected = _png(operation.show_tiles())
    assert _png(operation.inspect(frame, for_save=True)) == expected
    # No subject: the held image, which is still alive.
    assert _png(operation.inspect()) == expected


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
