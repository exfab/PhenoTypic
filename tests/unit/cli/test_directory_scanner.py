"""Store directories reaching the peripheral CLI consumers (Phase 3 Task 3.6).

Two failure shapes are pinned here, and every test in this file exists to catch
one of them.

**Shape 1 — the scan.** ``results/<ds>/zarr/`` holds *directories*, not files.
An ``is_file()`` filter finds nothing at all, and a recursive glob descends into
every store (roughly forty stat calls each, 400k at 10k images) and re-finds
them nested inside themselves.

**Shape 2 — ``Path.stem`` on a store.** ``.ome.zarr`` is a *double* suffix, so
``Path("img.ome.zarr").stem`` is ``"img.ome"``. Nothing raises. The run writes
``img.ome.parquet``, publishes a marker keyed ``"img.ome"``, then looks for
``img.ome.ome.zarr``, finds nothing, and reprocesses every image on every run
forever. :func:`phenotypic.sdk_.store_stem` is the only correct spelling, and it
*raises* on a non-store path rather than falling back — the silent fallback
being precisely the failure it prevents.

Each ``store_stem`` site below is asserted through an observable name — a
parquet filename, an overlay filename, a task-manifest entry, a marker key — so
that reverting the call to ``.stem`` fails a test rather than passing quietly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from phenotypic.sdk_ import (
    STORE_SUFFIX,
    dataset_zarr_dir,
    store_stem,
    zarr_store_path,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="CLI store scanning uses POSIX atomic writes",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fake_store(output_dir: Path, dataset: str, stem: str) -> Path:
    """A path-shaped store: enough for scanning, not for loading."""
    store = zarr_store_path(output_dir, dataset, stem)
    store.mkdir(parents=True, exist_ok=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    return store


def _detected(image):
    """Give the image an objmap: MeasureSize needs objects to measure."""
    from phenotypic.detect import OtsuDetector

    return OtsuDetector().apply(image)


def _synth_image():
    from phenotypic import Image
    from phenotypic.data import load_synth_yeast_plate

    return _detected(Image(load_synth_yeast_plate().rgb[:]))


def _real_store(output_dir: Path, dataset: str, stem: str) -> Path:
    """A genuinely readable store, written by the production writer."""
    from phenotypic._cli._cli_output_manager import OutputManager

    manager = OutputManager.from_config(output_dir, ".tiff", save_overlays=False)
    saved = manager.save_image_store(_synth_image(), dataset, stem)
    assert saved is not None
    return saved


# ---------------------------------------------------------------------------
# scan_store_outputs
# ---------------------------------------------------------------------------


def test_scan_finds_store_directories_not_files(tmp_path: Path) -> None:
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs

    for stem in ("a", "b"):
        _fake_store(tmp_path, "ds", stem)

    datasets = scan_store_outputs(tmp_path)
    assert [p.name for p in datasets[0].images] == [
        f"a{STORE_SUFFIX}",
        f"b{STORE_SUFFIX}",
    ]
    assert datasets[0].name == "ds"
    assert datasets[0].input_dir == dataset_zarr_dir(tmp_path, "ds")


def test_scan_is_non_recursive(tmp_path: Path) -> None:
    """A recursive scan walks INTO every store: 400k stat calls at 10k images."""
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs

    store = _fake_store(tmp_path, "ds", "a")
    (store / "gray" / "0").mkdir(parents=True)
    (store / "gray" / "0" / f"nested{STORE_SUFFIX}").mkdir()

    assert len(scan_store_outputs(tmp_path)[0].images) == 1


def test_scan_skips_part_and_trash_directories(tmp_path: Path) -> None:
    """The dotfile guard now also covers promote_store's in-flight siblings."""
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs

    _fake_store(tmp_path, "ds", "a")
    zarr_dir = dataset_zarr_dir(tmp_path, "ds")
    (zarr_dir / f".a{STORE_SUFFIX}.deadbeef.part").mkdir()
    (zarr_dir / f".a{STORE_SUFFIX}.deadbeef.trash").mkdir()
    # And the AppleDouble sidecar the guard was originally written for.
    (zarr_dir / f"._a{STORE_SUFFIX}").mkdir()

    assert len(scan_store_outputs(tmp_path)[0].images) == 1


def test_scan_ignores_a_leftover_h5_file(tmp_path: Path) -> None:
    """A converted run keeps its .h5 by default; it is not an image store."""
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs

    hdf_dir = tmp_path / "results" / "ds" / "hdf"
    hdf_dir.mkdir(parents=True)
    (hdf_dir / "a.h5").write_bytes(b"not a store")

    with pytest.raises(ValueError, match="No OME-Zarr outputs"):
        scan_store_outputs(tmp_path)


def test_scan_skips_datasets_with_an_empty_zarr_dir(tmp_path: Path) -> None:
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs

    dataset_zarr_dir(tmp_path, "empty").mkdir(parents=True)
    _fake_store(tmp_path, "full", "a")

    datasets = scan_store_outputs(tmp_path)
    assert [d.name for d in datasets] == ["full"]


def test_scan_raises_when_nothing_is_found(tmp_path: Path) -> None:
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs

    with pytest.raises(ValueError, match="No OME-Zarr outputs"):
        scan_store_outputs(tmp_path)


# ---------------------------------------------------------------------------
# store_stem site: process_single_store_measure_core
# ---------------------------------------------------------------------------


def _measure_pipeline(tmp_path: Path) -> Path:
    from phenotypic import ImagePipeline
    from phenotypic.measure import MeasureSize

    pipeline = ImagePipeline(meas=[MeasureSize()])
    path = tmp_path / "pipeline.json"
    # to_json(filepath) rewrites the suffix; write the string ourselves so
    # the path the CLI is handed is the path that exists.
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return path


def test_measure_core_names_the_parquet_by_the_bare_stem(tmp_path: Path) -> None:
    """`.stem` here writes `img.ome.parquet` -- a wrong name nothing raises on."""
    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic._cli._cli_process_single import (
        process_single_store_measure_core,
    )
    from phenotypic.sdk_ import dataset_measurements_dir

    store = _real_store(tmp_path, "ds", "img")
    manager = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    manager.create_structure(
        [
            type(
                "_D",
                (),
                {"name": "ds", "images": [], "input_dir": tmp_path, "output_dir": tmp_path},
            )()
        ]
    )

    process_single_store_measure_core(
        pipeline_path=_measure_pipeline(tmp_path),
        store_path=store,
        output_dir=tmp_path,
        dataset_name="ds",
        image_type="Image",
        output_manager=manager,
    )

    written = sorted(p.name for p in dataset_measurements_dir(tmp_path, "ds").glob("*.parquet"))
    assert written == ["img.parquet"]


def test_measure_core_dispatches_on_the_stores_image_class(tmp_path: Path) -> None:
    """A GridImage store must rehydrate as a GridImage, not the configured fallback."""
    from phenotypic import GridImage
    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic._cli._cli_process_single import (
        process_single_store_measure_core,
    )
    from phenotypic.data import load_synth_yeast_plate

    manager = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    saved = manager.save_image_store(
        _detected(GridImage(load_synth_yeast_plate().rgb[:])), "ds", "img"
    )
    assert saved is not None

    seen: list[type] = []
    from phenotypic._core import _image_pipeline as _ip

    original = _ip.ImagePipeline.measure

    def _spy(self, image, *args, **kwargs):
        seen.append(type(image))
        return original(self, image, *args, **kwargs)

    _ip.ImagePipeline.measure = _spy  # type: ignore[method-assign]
    try:
        process_single_store_measure_core(
            pipeline_path=_measure_pipeline(tmp_path),
            store_path=saved,
            output_dir=tmp_path,
            dataset_name="ds",
            # The WRONG fallback on purpose: the store's own image_class wins.
            image_type="Image",
            output_manager=manager,
        )
    finally:
        _ip.ImagePipeline.measure = original  # type: ignore[method-assign]

    assert seen == [GridImage]


# ---------------------------------------------------------------------------
# store_stem site: _regenerate_missing_overlays
# ---------------------------------------------------------------------------


def test_regenerate_overlays_writes_the_bare_stem_png(tmp_path: Path) -> None:
    """`.stem` writes `img.ome.png`, so the real overlay stays missing forever."""
    from phenotypic.phenotypicCLI import _regenerate_missing_overlays
    from phenotypic.sdk_ import dataset_overlays_dir

    _real_store(tmp_path, "ds", "img")
    _regenerate_missing_overlays(tmp_path, overlay_alpha=0.3, n_jobs=1)

    assert sorted(p.name for p in dataset_overlays_dir(tmp_path, "ds").glob("*.png")) == [
        "img.png"
    ]


def test_regenerate_overlays_skips_an_overlay_that_already_exists(
    tmp_path: Path,
) -> None:
    """`.stem` probes for `img.ome.png`, never finds it, and re-renders every run."""
    from phenotypic.phenotypicCLI import _regenerate_missing_overlays
    from phenotypic.sdk_ import dataset_overlays_dir

    _real_store(tmp_path, "ds", "img")
    overlays = dataset_overlays_dir(tmp_path, "ds")
    overlays.mkdir(parents=True, exist_ok=True)
    sentinel = overlays / "img.png"
    sentinel.write_bytes(b"already here")

    _regenerate_missing_overlays(tmp_path, overlay_alpha=0.3, n_jobs=1)

    assert sentinel.read_bytes() == b"already here"


# ---------------------------------------------------------------------------
# store_stem site: the SLURM recompile overlay task list
# ---------------------------------------------------------------------------


def test_recompile_overlay_tasks_skip_present_overlays(tmp_path: Path) -> None:
    """`.stem` tests for `<stem>.ome.png`, so every overlay is queued forever."""
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        _overlay_tasks_for_dataset,
    )
    from phenotypic.sdk_ import dataset_overlays_dir

    _fake_store(tmp_path, "ds", "a")
    _fake_store(tmp_path, "ds", "b")
    overlays = dataset_overlays_dir(tmp_path, "ds")
    overlays.mkdir(parents=True, exist_ok=True)
    (overlays / "a.png").write_bytes(b"present")

    tasks = _overlay_tasks_for_dataset(tmp_path, "ds", 0.3)
    assert [Path(t["store_path"]).name for t in tasks] == [f"b{STORE_SUFFIX}"]


def test_recompile_overlay_tasks_ignore_part_directories(tmp_path: Path) -> None:
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        _overlay_tasks_for_dataset,
    )

    _fake_store(tmp_path, "ds", "a")
    (dataset_zarr_dir(tmp_path, "ds") / f".a{STORE_SUFFIX}.cafe.part").mkdir()

    assert len(_overlay_tasks_for_dataset(tmp_path, "ds", 0.3)) == 1


def test_recompile_overlay_worker_writes_the_bare_stem_png(tmp_path: Path) -> None:
    from phenotypic._cli._cli_recompile_worker import _run_overlay_task
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic.sdk_ import dataset_overlays_dir

    store = _real_store(tmp_path, "ds", "img")
    initialize_slurm_lifecycle(tmp_path, generation="gen-1", mode="recompile")
    dataset_overlays_dir(tmp_path, "ds").mkdir(parents=True, exist_ok=True)

    result = _run_overlay_task(
        tmp_path,
        {
            "task_type": "overlay",
            "dataset_name": "ds",
            "store_path": str(store),
            "overlay_alpha": 0.3,
        },
        slurm_generation="gen-1",
    )

    assert result["overlay_failed"] is False, result.get("error")
    assert sorted(
        p.name for p in dataset_overlays_dir(tmp_path, "ds").glob("*.png")
    ) == ["img.png"]


# ---------------------------------------------------------------------------
# store_stem site: recompile dataset/image discovery
# ---------------------------------------------------------------------------


def test_recompile_image_names_come_back_bare(tmp_path: Path) -> None:
    """These names key the SLURM recompile job metadata; `img.ome` matches nothing."""
    from phenotypic.phenotypicCLI import _recompile_dataset_image_names

    _fake_store(tmp_path, "ds", "b")
    _fake_store(tmp_path, "ds", "a")

    assert _recompile_dataset_image_names(tmp_path, "ds") == ["a", "b"]


def test_recompile_discovers_a_dataset_that_only_has_stores(tmp_path: Path) -> None:
    from phenotypic.phenotypicCLI import _discover_recompile_dataset_names

    _fake_store(tmp_path, "ds", "a")

    assert _discover_recompile_dataset_names(tmp_path, None) == ["ds"]


# ---------------------------------------------------------------------------
# The single-pass forward path now writes a store, not an .h5
# ---------------------------------------------------------------------------


def test_single_pass_writes_a_store_and_no_h5(tmp_path: Path) -> None:
    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic._cli._cli_process_single import process_single_image_core
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize

    from skimage.io import imsave

    image_path = tmp_path / "img.tiff"
    imsave(
        str(image_path), load_synth_yeast_plate().rgb[:], check_contrast=False
    )

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        ImagePipeline(ops=[OtsuDetector()], meas=[MeasureSize()]).to_json(),
        encoding="utf-8",
    )

    out = tmp_path / "out"
    manager = OutputManager.from_config(out, ".tiff", save_overlays=False)
    process_single_image_core(
        pipeline_path=pipeline_path,
        image_path=image_path,
        output_dir=out,
        dataset_name="ds",
        image_type="Image",
        read_kwargs={},
        output_manager=manager,
    )

    assert zarr_store_path(out, "ds", "img").is_dir()
    assert not list(out.rglob("*.h5"))


def test_process_single_cli_marker_certifies_the_store(tmp_path: Path) -> None:
    """The standalone worker publishes its own marker, bypassing the strategy.

    It hard-coded ``"hdf": results/<ds>/hdf/<stem>.h5``. Once the forward path
    writes a store instead, ``publish_image_success`` resolves every artifact
    ``strict=True``, so that key names a file nothing writes any more and every
    SLURM array task dies with ``FileNotFoundError`` after doing all its work.
    """
    from click.testing import CliRunner

    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_process_single import main
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize

    from skimage.io import imsave

    input_root = tmp_path / "in"
    input_root.mkdir()
    image_path = input_root / "img.tiff"
    imsave(
        str(image_path), load_synth_yeast_plate().rgb[:], check_contrast=False
    )

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        ImagePipeline(ops=[OtsuDetector()], meas=[MeasureSize()]).to_json(),
        encoding="utf-8",
    )

    out = tmp_path / "out"
    published: dict = {}

    import phenotypic._cli._cli_process_single as mod

    real_publish = mod.publish_image_success

    def _spy(output_dir, **kwargs):
        published.update(kwargs)
        return real_publish(output_dir, **kwargs)

    mod.publish_image_success = _spy  # type: ignore[assignment]
    try:
        result = CliRunner().invoke(
            main,
            [
                "--pipeline", str(pipeline_path),
                "--image", str(image_path),
                "--output-dir", str(out),
                "--dataset-name", "in",
                "--input-root", str(input_root),
                "--no-save-overlays",
            ],
        )
    finally:
        mod.publish_image_success = real_publish  # type: ignore[assignment]

    assert result.exit_code == 0, result.output
    artifacts = published["artifacts"]
    assert "hdf" not in artifacts
    assert artifacts["store"] == zarr_store_path(out, "in", "img") / "zarr.json"


def test_local_success_marker_certifies_the_store(tmp_path: Path) -> None:
    """`image_data_artifact` must resolve to the store once single-pass writes one."""
    from phenotypic._cli._cli_completion import image_data_artifact
    from phenotypic._cli._cli_output_manager import OutputManager

    manager = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    _real_store(tmp_path, "ds", "img")

    key, path = image_data_artifact(tmp_path, manager, "ds", "img")
    assert key == "store"
    assert path == zarr_store_path(tmp_path, "ds", "img") / "zarr.json"
    assert path.is_file()


# ---------------------------------------------------------------------------
# tune: a run can be pointed at a previous run's zarr/ directory
# ---------------------------------------------------------------------------


def test_tune_loads_store_directories(tmp_path: Path) -> None:
    from phenotypic import GridImage
    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.tune._tune_cli._run import _load_images

    manager = OutputManager.from_config(tmp_path, ".tiff", save_overlays=False)
    assert (
        manager.save_image_store(
            _detected(GridImage(load_synth_yeast_plate().rgb[:])), "ds", "img"
        )
        is not None
    )

    images = _load_images(dataset_zarr_dir(tmp_path, "ds"))
    assert len(images) == 1
    assert isinstance(images[0], GridImage)


def test_tune_ignores_part_directories(tmp_path: Path) -> None:
    from phenotypic.tune._tune_cli._run import _load_images

    zarr_dir = dataset_zarr_dir(tmp_path, "ds")
    zarr_dir.mkdir(parents=True)
    (zarr_dir / f".img{STORE_SUFFIX}.beef.part").mkdir()

    assert _load_images(zarr_dir) == []


# ---------------------------------------------------------------------------
# store_stem itself refuses to guess
# ---------------------------------------------------------------------------


def test_store_stem_raises_on_a_non_store_path() -> None:
    """The guard that makes every site above a hard failure, not a silent one."""
    with pytest.raises(ValueError, match="not an OME-Zarr store"):
        store_stem(Path("img.h5"))
    assert store_stem(Path(f"img{STORE_SUFFIX}")) == "img"


# ---------------------------------------------------------------------------
# The README generator documents the new layout
# ---------------------------------------------------------------------------


def test_readme_documents_the_store_and_its_interoperability(tmp_path: Path) -> None:
    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_readme_generator import READMEGenerator
    from types import SimpleNamespace

    from phenotypic._cli._cli_types import Dataset
    from phenotypic.measure import MeasureSize

    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        ImagePipeline(meas=[MeasureSize()]).to_json(), encoding="utf-8"
    )
    # READMEGenerator reads only these three config attributes.
    config = SimpleNamespace(
        pipeline_json=pipeline_path, image_type="Image", nrows=None, ncols=None
    )
    dataset = Dataset(
        name="ds", images=[], input_dir=tmp_path, output_dir=tmp_path
    )
    written = READMEGenerator(
        config, ImagePipeline(meas=[MeasureSize()])
    ).generate(tmp_path, [dataset])
    text = written.read_text(encoding="utf-8")

    assert "zarr/" in text
    assert STORE_SUFFIX in text
    assert "load_zarr" in text
    for viewer in ("napari", "QuPath", "Vizarr"):
        assert viewer in text
    assert ".h5" not in text
    assert "load_hdf5" not in text


# ---------------------------------------------------------------------------
# The scan is what measure mode consumes
# ---------------------------------------------------------------------------


def test_measure_mode_scan_feeds_paths_store_stem_accepts(tmp_path: Path) -> None:
    """End-to-end on the contract: every scanned path must survive store_stem."""
    from phenotypic._cli._cli_directory_scanner import scan_store_outputs

    for stem in ("plate_01", "plate.02", "plate 03"):
        _fake_store(tmp_path, "ds", stem)

    scanned = scan_store_outputs(tmp_path)[0].images
    assert sorted(store_stem(p) for p in scanned) == [
        "plate 03",
        "plate.02",
        "plate_01",
    ]
    # And the round trip closes: store_stem -> zarr_store_path -> the same dir.
    for path in scanned:
        assert zarr_store_path(tmp_path, "ds", store_stem(path)) == path


def test_scan_result_json_round_trips_through_a_task_manifest(tmp_path: Path) -> None:
    """Store paths cross a JSON boundary on the SLURM recompile path."""
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        _overlay_tasks_for_dataset,
    )

    _fake_store(tmp_path, "ds", "a")
    tasks = _overlay_tasks_for_dataset(tmp_path, "ds", 0.3)
    revived = json.loads(json.dumps(tasks))
    assert store_stem(Path(revived[0]["store_path"])) == "a"
