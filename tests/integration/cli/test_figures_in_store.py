"""Every mode writes figures into the store; deliverables are a copy (spec §3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic.sdk_ import zarr_store_path
from phenotypic.sdk_._image_figures import read_image_figures_descriptor


def _write_inputs(root: Path, *, with_plot: bool):
    from skimage.io import imsave

    from phenotypic import ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize, MeasureSymZones

    root.mkdir(parents=True, exist_ok=True)
    image = root / "in" / "plate.tiff"
    image.parent.mkdir(exist_ok=True)
    imsave(str(image), load_synth_yeast_plate().rgb[:], check_contrast=False)
    sym = MeasureSymZones()
    pipeline = ImagePipeline(
        ops={"detect": OtsuDetector()},
        meas={"size": MeasureSize(), "sym": sym},
        plots=[sym] if with_plot else [],
    )
    path = root / "pipeline.json"
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return image, path


def _full(tmp_path, pipeline, image):
    from phenotypic._cli._cli_process_single import process_single_image_core

    out = tmp_path / "out"
    process_single_image_core(
        pipeline, image, out, "ds", "Image", {},
        OutputManager.from_config(out, ".tiff", save_overlays=False),
    )
    return out, zarr_store_path(out, "ds", "plate")


def test_full_mode_stores_figures_and_copies_them_out(tmp_path):
    image, pipeline = _write_inputs(tmp_path, with_plot=True)
    out, store = _full(tmp_path, pipeline, image)
    descriptor = read_image_figures_descriptor(store)
    assert descriptor["failed"] == []
    [page] = descriptor["bindings"]["sym"]["pages"]
    assert [f["format"] for f in page["files"]] == ["plotly-json"]
    deliverable = list((out / "deliverables/plots/sym/ds").glob("plate-*.plotly.json"))
    assert len(deliverable) == 1
    assert deliverable[0].read_bytes() == (store / page["files"][0]["path"]).read_bytes()
    assert len(list((out / "deliverables/plots/sym/ds").glob("plate-*.html"))) == 1


def test_measure_mode_rebuilds_figures_from_the_current_pipeline(tmp_path):
    from phenotypic._cli._cli_process_single import process_single_store_measure_core

    image, with_plot = _write_inputs(tmp_path, with_plot=True)
    out, store = _full(tmp_path, with_plot, image)
    _, without_plot = _write_inputs(tmp_path / "second", with_plot=False)
    manager = OutputManager.from_config(out, ".tiff", save_overlays=False)
    process_single_store_measure_core(without_plot, store, out, "ds", "Image", manager)
    assert read_image_figures_descriptor(store) is None
    process_single_store_measure_core(with_plot, store, out, "ds", "Image", manager)
    assert list(read_image_figures_descriptor(store)["bindings"]) == ["sym"]


@pytest.mark.parametrize("fmt", ["zarr", "tiff"])
def test_process_mode_carries_figures_only_in_a_store(tmp_path, fmt):
    from phenotypic._cli._cli_process_only import process_single_apply_only_core

    image, pipeline = _write_inputs(tmp_path, with_plot=True)
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline, image_path=image, input_root=image.parent,
        output_dir=out, image_type="Image", layer="rgb", read_kwargs={},
        process_format=fmt,
    )
    if fmt == "zarr":
        store = out / "plate.ome.zarr"
        descriptor = read_image_figures_descriptor(store)
        assert list(descriptor["bindings"]) == ["sym"] and descriptor["failed"] == []
        assert not (store / "tables").exists(), "process mode writes no table"
    else:
        assert not list(out.rglob("*.plotly.json"))
    assert not (out / "deliverables").exists()


def test_process_revision_is_3():
    from phenotypic._cli import _cli_failure_tracker as tracker

    assert tracker.PROCESS_LAYER_SEMANTICS_REVISION == 3
