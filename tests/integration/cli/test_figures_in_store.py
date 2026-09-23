"""Every mode writes a run folder into the store; deliverables are a copy
(spec §1a, §3)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic.sdk_ import zarr_store_path
from phenotypic.sdk_._image_figures import read_image_figures_descriptor


def _write_inputs(root: Path, *, with_plot: bool, variant: int = 0):
    """The image and a pipeline file; *variant* changes the pipeline's bytes."""
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
    meas = {"size": MeasureSize(), "sym": sym}
    if variant:
        meas[f"size{variant}"] = MeasureSize()
    pipeline = ImagePipeline(
        ops={"detect": OtsuDetector()},
        meas=meas,
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


def _runs(store: Path) -> dict:
    return read_image_figures_descriptor(store)["runs"]


def _run_of(pipeline: Path) -> str:
    """The run id this pipeline file writes under today (spec §1a)."""
    from phenotypic.sdk_._image_figures import FigureRun, utc_run_date

    sha = hashlib.sha256(pipeline.read_bytes()).hexdigest()
    return FigureRun(date=utc_run_date(), pipeline_sha256=sha).run_id


def _folder_bytes(store: Path, run_id: str) -> dict[str, bytes]:
    folder = store / "figures" / run_id
    return {
        p.relative_to(store).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*")) if p.is_file()
    }


def test_full_mode_stores_figures_and_copies_them_out(tmp_path):
    image, pipeline = _write_inputs(tmp_path, with_plot=True)
    out, store = _full(tmp_path, pipeline, image)
    [(run_id, run)] = _runs(store).items()
    assert run_id == _run_of(pipeline)
    assert run["failed"] == [] and run["unavailable"] == []
    [page] = run["bindings"]["sym"]["pages"]
    assert [f["format"] for f in page["files"]] == ["plotly-json"]
    assert page["files"][0]["path"] == f"figures/{run_id}/sym/default.plotly.json"
    deliverable = list((out / "deliverables/plots/sym/ds").glob("plate-*.plotly.json"))
    assert len(deliverable) == 1
    assert deliverable[0].read_bytes() == (store / page["files"][0]["path"]).read_bytes()
    assert len(list((out / "deliverables/plots/sym/ds").glob("plate-*.html"))) == 1


def test_a_measure_run_with_another_pipeline_adds_a_run_and_keeps_the_first(tmp_path):
    """Two runs, one store (spec §1a): the first folder is byte-identical."""
    from phenotypic._cli._cli_process_single import process_single_store_measure_core

    image, first = _write_inputs(tmp_path, with_plot=True)
    out, store = _full(tmp_path, first, image)
    first_run = _run_of(first)
    first_entry, first_bytes = _runs(store)[first_run], _folder_bytes(store, first_run)

    _, second = _write_inputs(tmp_path / "second", with_plot=True, variant=1)
    manager = OutputManager.from_config(out, ".tiff", save_overlays=False)
    process_single_store_measure_core(second, store, out, "ds", "Image", manager)

    second_run = _run_of(second)
    assert second_run != first_run
    assert set(_runs(store)) == {first_run, second_run}
    assert _runs(store)[first_run] == first_entry
    assert _folder_bytes(store, first_run) == first_bytes
    assert list(_runs(store)[second_run]["bindings"]) == ["sym"]


def test_a_measure_run_without_image_plots_touches_no_figure(tmp_path):
    from phenotypic._cli._cli_process_single import process_single_store_measure_core

    image, with_plot = _write_inputs(tmp_path, with_plot=True)
    out, store = _full(tmp_path, with_plot, image)
    before = read_image_figures_descriptor(store)
    _, without_plot = _write_inputs(tmp_path / "second", with_plot=False)
    manager = OutputManager.from_config(out, ".tiff", save_overlays=False)
    process_single_store_measure_core(without_plot, store, out, "ds", "Image", manager)
    assert read_image_figures_descriptor(store) == before


def test_a_same_day_same_pipeline_rerun_replaces_only_its_folder(tmp_path):
    from phenotypic._cli._cli_process_single import process_single_store_measure_core

    image, first = _write_inputs(tmp_path, with_plot=True)
    out, store = _full(tmp_path, first, image)
    _, second = _write_inputs(tmp_path / "second", with_plot=True, variant=1)
    manager = OutputManager.from_config(out, ".tiff", save_overlays=False)
    process_single_store_measure_core(second, store, out, "ds", "Image", manager)
    second_bytes = _folder_bytes(store, _run_of(second))

    process_single_store_measure_core(first, store, out, "ds", "Image", manager)
    assert set(_runs(store)) == {_run_of(first), _run_of(second)}
    assert _folder_bytes(store, _run_of(second)) == second_bytes


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
        run = _runs(store)[_run_of(pipeline)]
        assert list(run["bindings"]) == ["sym"] and run["failed"] == []
        assert not (store / "tables").exists(), "process mode writes no table"
    else:
        assert not list(out.rglob("*.plotly.json"))
    assert not (out / "deliverables").exists()


def test_process_revision_is_3():
    from phenotypic._cli import _cli_failure_tracker as tracker

    assert tracker.PROCESS_LAYER_SEMANTICS_REVISION == 3
