"""Real OME-Zarr input loading through Tune submitter and worker paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def _write_group(path: Path, attributes: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {"zarr_format": 3, "node_type": "group", "attributes": attributes}
        ),
        encoding="utf-8",
    )


def _write_array(path: Path, array: np.ndarray, axes: list[str]) -> None:
    """Write one uncompressed, single-chunk Zarr-v3 array."""
    array = np.ascontiguousarray(array)
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": list(array.shape),
                "data_type": array.dtype.name,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": list(array.shape)},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "fill_value": 0,
                "codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}}
                ],
                "attributes": {},
                "dimension_names": axes,
            }
        ),
        encoding="utf-8",
    )
    chunk = path / "c"
    for _index in array.shape:
        chunk /= "0"
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.write_bytes(array.tobytes(order="C"))


def _write_minimal_ome_zarr(store: Path) -> Path:
    """Create a real tiny run-bundle store without invoking the slow writer."""
    from phenotypic.sdk_ import ngff_

    pixels = np.arange(12 * 12 * 3, dtype=np.uint16).reshape(3, 12, 12)
    gray = np.arange(12 * 12, dtype=np.float64).reshape(12, 12) / 144
    detect_mat = gray.copy()
    objmap = np.zeros((12, 12), dtype=np.uint32)
    arrays = {
        "rgb": (pixels, ["c", "y", "x"]),
        "gray": (gray, ["y", "x"]),
        "detect_mat": (detect_mat, ["y", "x"]),
        "rgb/labels/objmap": (objmap, ["y", "x"]),
    }
    for relative, (array, axes) in arrays.items():
        _write_array(store / relative / "0", array, axes)

    for series, (array, _axes) in list(arrays.items())[:3]:
        _write_group(
            store / series,
            {
                "ome": {
                    "version": ngff_.NGFF_VERSION,
                    **ngff_.build_multiscales(
                        series=series, level_shapes=[array.shape]
                    ),
                }
            },
        )
    _write_group(
        store / "rgb" / "labels",
        {"ome": {"version": ngff_.NGFF_VERSION, "labels": ["objmap"]}},
    )
    _write_group(
        store / "rgb" / "labels" / "objmap",
        {
            "ome": {
                "version": ngff_.NGFF_VERSION,
                **ngff_.build_multiscales(
                    series="objmap", level_shapes=[objmap.shape]
                ),
            }
        },
    )
    _write_group(
        store / "OME",
        {
            "ome": {
                "version": ngff_.NGFF_VERSION,
                "series": ["rgb", "gray", "detect_mat"],
            }
        },
    )
    block = ngff_.build_phenotypic_attributes(
        image_class="GridImage",
        series_names=["rgb", "gray", "detect_mat"],
        pyramid_levels=1,
        metadata_sections={
            "protected": {
                "Metadata_ImageName": "minimal_plate",
                "Metadata_BitDepth": 16,
            },
            "public": {},
            "imported": {},
        },
        detect_mode="gray",
        illuminant=None,
        gamma=None,
        grid={"nrows": 2, "ncols": 3},
    )
    _write_group(store, {"phenotypic": block})
    return store


def _store_bytes(store: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(store)): path.read_bytes()
        for path in sorted(store.rglob("*"))
        if path.is_file()
    }


def _require_working_zarr_sync_bridge(store: Path) -> None:
    """Probe the installed zarr sync bridge, failing when the lane requires it."""
    probe = (
        "import sys, zarr; "
        "a = zarr.open_array(store=sys.argv[1], mode='r'); "
        "assert a[...].shape == (12, 12)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe, str(store / "gray" / "0")],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        if os.environ.get("PHENOTYPIC_REQUIRE_OME_ZARR_PROBE") == "1":
            pytest.fail("OME-Zarr probe timed out while the Slurm lane requires it")
        pytest.skip("installed zarr sync bridge deadlocks on local store reads")
    assert result.returncode == 0, result.stderr


def test_required_ome_zarr_probe_fails_instead_of_skipping(
    tmp_path: Path, monkeypatch
) -> None:
    """The Slurm verification lane must expose a broken sync bridge."""
    store = tmp_path / "minimal.ome.zarr"
    monkeypatch.setenv("PHENOTYPIC_REQUIRE_OME_ZARR_PROBE", "1")

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="zarr probe", timeout=5)

    monkeypatch.setattr(subprocess, "run", _timeout)

    with pytest.raises(pytest.fail.Exception, match="OME-Zarr probe"):
        _require_working_zarr_sync_bridge(store)


def _spec(*, optuna: bool):
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.tune import Budget, Evaluator, SearchSpace
    from phenotypic.tune.score import ReferenceFreeScorer
    from phenotypic.tune.strategy import GridConfig, OptunaConfig
    from phenotypic.tune._spec import TuningSpec

    return TuningSpec(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=SearchSpace(knobs=()),
        scorer=ReferenceFreeScorer(),
        evaluator=Evaluator(),
        strategy=OptunaConfig(n_trials=1) if optuna else GridConfig(),
        budget=Budget(n_trials=1),
    )


def test_real_ome_zarr_is_loaded_by_submitter_and_worker_without_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    """Replacing the canonical store loader breaks processed-image tuning."""
    from phenotypic import GridImage
    from phenotypic.tune import __main__ as cli
    from phenotypic.tune._tune_cli import _worker

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    store = _write_minimal_ome_zarr(images_dir / "minimal_plate.ome.zarr")
    _require_working_zarr_sync_bridge(store)
    before = _store_bytes(store)

    submit_spec = tmp_path / "submit.json"
    submit_spec.write_text(_spec(optuna=False).model_dump_json(), encoding="utf-8")
    submitter_seen: list[GridImage] = []

    def _capture_submitter(_spec, images, _output, **_kwargs):
        submitter_seen.extend(images)

    monkeypatch.setattr(cli, "run_tuning", _capture_submitter)
    cli.main(
        [
            "run",
            str(submit_spec),
            "--input",
            str(images_dir),
            "--output",
            str(tmp_path / "submit-out"),
        ]
    )

    worker_spec = tmp_path / "worker.json"
    worker_spec.write_text(_spec(optuna=True).model_dump_json(), encoding="utf-8")
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "calibration": ["minimal_plate"],
                "held_out": [],
                "kind": "none",
                "group_key": None,
                "dataset_identity": "fixture",
                "seed_entropy": [1],
            }
        ),
        encoding="utf-8",
    )
    worker_seen: list[GridImage] = []

    class _Engine:
        def __init__(self, _spec, *, store):
            assert store is not None

        def optimize(self, images):
            worker_seen.extend(images)

    monkeypatch.setattr(_worker, "build_worker_store", lambda **_kwargs: object())
    monkeypatch.setattr("phenotypic.tune._engine.TuningEngine", _Engine)
    _worker.run_worker(
        spec_path=worker_spec,
        images_dir=images_dir,
        split_path=split_path,
        storage_url=f"journal://{tmp_path}/journal.log",
        study_name="tune_cost_v1",
    )

    assert [type(item) for item in submitter_seen] == [GridImage]
    assert [type(item) for item in worker_seen] == [GridImage]
    assert submitter_seen[0].name == worker_seen[0].name == "minimal_plate"
    assert (submitter_seen[0].nrows, submitter_seen[0].ncols) == (2, 3)
    assert (worker_seen[0].nrows, worker_seen[0].ncols) == (2, 3)
    assert _store_bytes(store) == before
