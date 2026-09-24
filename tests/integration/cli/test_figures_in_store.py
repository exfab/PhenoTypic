"""Every mode writes a run folder into the store; deliverables are a copy
(spec §1a, §3, §3a).

Every run date here is pinned, so no assertion depends on the wall clock.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import BaseModel

from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic.abc_.plotting import FigureInputUnavailable, PlotImage
from phenotypic.sdk_ import zarr_store_path
from phenotypic.sdk_._image_figures import (
    FigureRun,
    RunInitiation,
    read_image_figures_descriptor,
)

DAY = "2026-09-22"
LATER = "2026-09-30"
LATEST = "2026-10-07"

#: Whether ApplyStateFake can draw -- True stands for "apply() ran here".
_APPLY_STATE = {"drawable": True}


class ApplyStateFake(BaseModel, PlotImage):
    """A §3a provider: draws only in the process where its apply-state exists."""

    def inspect(self, subject=None, *, for_save=False, **overrides):
        from matplotlib.figure import Figure

        if not _APPLY_STATE["drawable"]:
            raise FigureInputUnavailable("the as-shot pixels are gone")
        fig = Figure()
        fig.subplots().plot([0, 1])
        return fig


@pytest.fixture(autouse=True)
def _drawable():
    _APPLY_STATE["drawable"] = True
    yield
    _APPLY_STATE["drawable"] = True


def _write_inputs(root: Path, *, with_plot: bool, variant: int = 0, apply_state=False):
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
    plots: list = [sym] if with_plot else []
    if apply_state:
        plots.append(ApplyStateFake())
    pipeline = ImagePipeline(ops={"detect": OtsuDetector()}, meas=meas, plots=plots)
    path = root / "pipeline.json"
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return image, path


def _call(date: str) -> RunInitiation:
    """A pinned initial CLI call on *date*; the pid tells the calls apart."""
    return RunInitiation(date=date, at_utc=f"{date}T12:00:00.000Z", pid=int(date[-2:]))


def _manager(out: Path, date: str) -> OutputManager:
    return OutputManager.from_config(
        out, ".tiff", save_overlays=False, run_initiation=_call(date)
    )


def _full(tmp_path, pipeline, image, date=DAY):
    from phenotypic._cli._cli_process_single import process_single_image_core

    out = tmp_path / "out"
    process_single_image_core(pipeline, image, out, "ds", "Image", {}, _manager(out, date))
    return out, zarr_store_path(out, "ds", "plate")


def _measure(out: Path, store: Path, pipeline: Path, date: str) -> None:
    from phenotypic._cli._cli_process_single import process_single_store_measure_core

    process_single_store_measure_core(pipeline, store, out, "ds", "Image", _manager(out, date))


def _runs(store: Path) -> dict:
    return read_image_figures_descriptor(store)["runs"]


def _run_of(pipeline: Path, date: str = DAY) -> str:
    """The run id this pipeline file writes under on *date* (spec §1a)."""
    sha = hashlib.sha256(pipeline.read_bytes()).hexdigest()
    return FigureRun(date=date, pipeline_sha256=sha).run_id


def _folder(store: Path, run_id: str) -> dict[str, tuple[bytes, int]]:
    """Bytes and inode of every file in one run folder."""
    folder = store / "figures" / run_id
    return {
        p.relative_to(store).as_posix(): (p.read_bytes(), p.stat().st_ino)
        for p in sorted(folder.rglob("*")) if p.is_file()
    }


def _bytes(folder: dict) -> dict[str, bytes]:
    return {name: data for name, (data, _inode) in folder.items()}


def test_full_mode_stores_figures_and_copies_them_out(tmp_path):
    image, pipeline = _write_inputs(tmp_path, with_plot=True)
    out, store = _full(tmp_path, pipeline, image)
    [(run_id, run)] = _runs(store).items()
    assert run_id == _run_of(pipeline) and run_id.startswith(f"{DAY}-")
    assert (run["initiated_at_utc"], run["initiated_pid"]) == (
        _call(DAY).at_utc, _call(DAY).pid
    )
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
    image, first = _write_inputs(tmp_path, with_plot=True)
    out, store = _full(tmp_path, first, image)
    first_run = _run_of(first)
    first_entry, first_files = _runs(store)[first_run], _folder(store, first_run)

    _, second = _write_inputs(tmp_path / "second", with_plot=True, variant=1)
    _measure(out, store, second, LATER)

    second_run = _run_of(second, LATER)
    assert set(_runs(store)) == {first_run, second_run}
    assert _runs(store)[first_run] == first_entry
    assert _folder(store, first_run) == first_files
    assert list(_runs(store)[second_run]["bindings"]) == ["sym"]


def test_a_measure_run_without_image_plots_touches_no_figure(tmp_path):
    image, with_plot = _write_inputs(tmp_path, with_plot=True)
    out, store = _full(tmp_path, with_plot, image)
    before = read_image_figures_descriptor(store)
    _, without_plot = _write_inputs(tmp_path / "second", with_plot=False)
    _measure(out, store, without_plot, LATER)
    assert read_image_figures_descriptor(store) == before


def test_a_measure_run_with_the_same_pipeline_overwrites_that_runs_folder(tmp_path):
    """(i) Revision 14: a later day, the same pipeline, one folder, new files."""
    image, pipeline = _write_inputs(tmp_path, with_plot=True)
    out, store = _full(tmp_path, pipeline, image)
    run_id = _run_of(pipeline)
    before = _folder(store, run_id)

    _measure(out, store, pipeline, LATER)

    assert list(_runs(store)) == [run_id]
    run = _runs(store)[run_id]
    assert (run["date"], run["initiated_at_utc"], run["initiated_pid"]) == (
        DAY, _call(LATER).at_utc, _call(LATER).pid
    )
    after = _folder(store, run_id)
    assert after.keys() == before.keys()
    # Rewritten, not carried: every file is a new inode.
    assert all(after[name][1] != before[name][1] for name in after)


def test_measure_reuses_the_most_recent_folder_of_its_pipeline(tmp_path):
    """(ii) Two folders of one pipeline: the latest is overwritten, the other kept."""
    image, pipeline = _write_inputs(tmp_path, with_plot=True)
    out, store = _full(tmp_path, pipeline, image, DAY)
    _full(tmp_path, pipeline, image, LATER)
    older, newer = _run_of(pipeline, DAY), _run_of(pipeline, LATER)
    assert set(_runs(store)) == {older, newer}
    older_files, newer_files = _folder(store, older), _folder(store, newer)

    _measure(out, store, pipeline, LATEST)

    assert set(_runs(store)) == {older, newer}
    assert _folder(store, older) == older_files
    after = _folder(store, newer)
    assert all(after[name][1] != newer_files[name][1] for name in after)


def test_a_reused_folder_keeps_its_apply_state_figure(tmp_path):
    """(iv) §3a: the calibration-style figure in the reused folder is kept."""
    image, pipeline = _write_inputs(tmp_path, with_plot=True, apply_state=True)
    out, store = _full(tmp_path, pipeline, image)
    run_id = _run_of(pipeline)
    entry = _runs(store)[run_id]["bindings"]["ApplyStateFake"]
    kept = {n: d for n, d in _bytes(_folder(store, run_id)).items() if "/ApplyStateFake/" in n}
    assert kept

    _APPLY_STATE["drawable"] = False
    _measure(out, store, pipeline, LATER)

    run = _runs(store)[run_id]
    assert run["bindings"]["ApplyStateFake"] == entry
    assert run["unavailable"] == [] and run["failed"] == []
    now = {n: d for n, d in _bytes(_folder(store, run_id)).items() if "/ApplyStateFake/" in n}
    assert now == kept


def test_a_new_run_lists_an_apply_state_figure_unavailable_and_leaves_the_old(tmp_path):
    """§3a: no copy between run folders; the earlier folder is untouched."""
    image, first = _write_inputs(tmp_path, with_plot=True, apply_state=True)
    out, store = _full(tmp_path, first, image)
    first_run = _run_of(first)
    first_files = _folder(store, first_run)

    _APPLY_STATE["drawable"] = False
    _, second = _write_inputs(tmp_path / "second", with_plot=True, variant=1, apply_state=True)
    _measure(out, store, second, LATER)

    new = _runs(store)[_run_of(second, LATER)]
    assert new["unavailable"] == ["ApplyStateFake"]
    assert "ApplyStateFake" not in new["bindings"] and new["failed"] == []
    assert _folder(store, first_run) == first_files


@pytest.mark.parametrize("fmt", ["zarr", "tiff"])
def test_process_mode_carries_figures_only_in_a_store(tmp_path, fmt):
    from phenotypic._cli._cli_process_only import process_single_apply_only_core

    image, pipeline = _write_inputs(tmp_path, with_plot=True)
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline, image_path=image, input_root=image.parent,
        output_dir=out, image_type="Image", layer="rgb", read_kwargs={},
        process_format=fmt, run_initiation=_call(DAY),
    )
    if fmt == "zarr":
        store = out / "plate.ome.zarr"
        run = _runs(store)[_run_of(pipeline)]
        assert list(run["bindings"]) == ["sym"] and run["failed"] == []
        # Omitted for same-day byte identity (spec §1a); a full store has them.
        assert "initiated_at_utc" not in run and "initiated_pid" not in run
        assert not (store / "tables").exists(), "process mode writes no table"
    else:
        assert not list(out.rglob("*.plotly.json"))
    assert not (out / "deliverables").exists()


def test_process_revision_is_3():
    from phenotypic._cli import _cli_failure_tracker as tracker

    assert tracker.PROCESS_LAYER_SEMANTICS_REVISION == 3
