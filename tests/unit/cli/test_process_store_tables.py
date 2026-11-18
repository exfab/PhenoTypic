"""Process stores carry the pipeline's run folder (#281 P5 T2).

Spec E4: ``process_single_apply_only_core`` embeds after ``apply()`` and
writes the run folder into the store's ``.part`` before consolidation, so
the store stays write-once and consolidated, and stays byte-reproducible on
one UTC day on one device. ``PROCESS_LAYER_SEMANTICS_REVISION`` goes 4 -> 5.

Torch: the ``dinov2-vitb14`` tests run the stub backbone and skip without
torch. The toy-embedder tests pin the same store shape in every environment.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from phenotypic._cli._cli_failure_tracker import PROCESS_LAYER_SEMANTICS_REVISION
from tests._tiny_run import (
    TOY_DIM,
    TOY_GRID_KEY,
    grid_process_pipeline,
    run_cli,
    toy_grid_process_pipeline,
    tree_digest,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="Requires torch (stub backbone)"
)

ZARR_RGB = ["--layer", "rgb", "--process-format", "zarr"]
DINO_MODEL = "dinov2-vitb14"


def _stores(out: Path) -> list[Path]:
    return sorted(out.rglob("*.ome.zarr"))


def _consolidated_nodes(store: Path) -> dict:
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    return root["consolidated_metadata"]["metadata"]


def _assert_grid_run_folder(store: Path, model: str) -> None:
    """The store's one run folder holds the 2 x 2 grid table and nothing else."""
    import pyarrow.parquet as pq
    import zarr

    from phenotypic.sdk_ import list_table_runs

    (run,) = list_table_runs(store)
    assert run.run_id.startswith("1-") and not run.run_id.endswith("-unknown")
    assert not run.has_measurements
    assert set(run.descriptor["embeddings"]) == {f"{model}/grid"}
    entry = run.descriptor["embeddings"][f"{model}/grid"]
    assert entry["stale"] is False
    path = run.embedding_path(model, "grid")
    assert path.is_file()
    # The store stays consolidated: the consolidated view names the new group.
    assert f"tables/{run.run_id}/embeddings/{model}/grid" in _consolidated_nodes(store)
    # Independent readers: zarr-python over the consolidated root, pyarrow.
    group = zarr.open_consolidated(store, mode="r")
    assert f"tables/{run.run_id}/embeddings/{model}/grid" in dict(
        group.members(max_depth=None)
    )
    assert pq.read_table(path).num_rows == 4
    # No object map: the store is label-less, as process stores always are.
    assert not (store / "rgb" / "labels").exists()


def test_revision_bump_invalidates_old_process_continuation():
    assert PROCESS_LAYER_SEMANTICS_REVISION == 5


def test_a_toy_process_store_has_its_grid_table(tmp_path, tiny_inputs, registered_toy_embedder):
    out = tmp_path / "out"
    result = run_cli(
        tiny_inputs, out, mode="process", pipeline=toy_grid_process_pipeline(tmp_path),
        extra=ZARR_RGB,
    )
    assert result.exit_code == 0, result.output
    stores = _stores(out)
    assert len(stores) == 2
    for store in stores:
        _assert_grid_run_folder(store, TOY_GRID_KEY.partition("/")[0])


def test_a_toy_process_store_records_its_pipeline_digest(
    tmp_path, tiny_inputs, registered_toy_embedder
):
    """The folder's ``<sha12>`` is the pipeline file's, which the journal records."""
    import hashlib

    from phenotypic.sdk_ import list_table_runs

    pipeline = toy_grid_process_pipeline(tmp_path)
    out = tmp_path / "out"
    assert run_cli(tiny_inputs, out, mode="process", pipeline=pipeline, extra=ZARR_RGB).exit_code == 0
    sha = hashlib.sha256(pipeline.read_bytes()).hexdigest()
    (run,) = list_table_runs(_stores(out)[0])
    assert run.run_id == f"1-{sha[:12]}" and run.pipeline_sha256 == sha


def test_a_toy_process_store_stays_byte_reproducible(
    tmp_path, tiny_inputs, registered_toy_embedder
):
    pipeline = toy_grid_process_pipeline(tmp_path)
    for name in ("a", "b"):
        result = run_cli(tiny_inputs, tmp_path / name, mode="process", pipeline=pipeline, extra=ZARR_RGB)
        assert result.exit_code == 0, result.output
    a, b = (_stores(tmp_path / name) for name in ("a", "b"))
    assert [s.relative_to(tmp_path / "a") for s in a] == [s.relative_to(tmp_path / "b") for s in b]
    assert [tree_digest(s) for s in a] == [tree_digest(s) for s in b]


def test_a_toy_grid_table_holds_the_cells_vectors(tmp_path, tiny_inputs, registered_toy_embedder):
    """Row i is cell i in row-major order, with the toy's box-sum vector."""
    import numpy as np
    import pyarrow.parquet as pq

    from phenotypic.sdk_ import list_table_runs
    from tests._tiny_run import TOY_GRID_COL_EDGES, TOY_GRID_ROW_EDGES

    out = tmp_path / "out"
    run_cli(tiny_inputs, out, mode="process", pipeline=toy_grid_process_pipeline(tmp_path), extra=ZARR_RGB)
    (run,) = list_table_runs(_stores(out)[0])
    table = pq.read_table(run.embedding_path("toy-v1", "grid")).to_pydict()
    assert table["Grid_RowMajorIdx"] == [0, 1, 2, 3]
    vectors = np.asarray(table["embedding"], dtype=np.float32)
    assert vectors.shape == (4, TOY_DIM)
    # The toy vector of a cell is the sum of its box's four edges, repeated.
    r, c = TOY_GRID_ROW_EDGES, TOY_GRID_COL_EDGES
    sums = [r[i] + r[i + 1] + c[j] + c[j + 1] for i in range(2) for j in range(2)]
    assert (vectors == vectors[:, :1]).all()
    assert vectors[:, 0].tolist() == sums


def test_a_process_store_without_meas_has_no_tables(tmp_path, tiny_inputs):
    """A pipeline with no ``meas`` writes the store it always did."""
    from phenotypic import ImagePipeline
    from phenotypic.enhance import BlurGauss

    pipeline = tmp_path / "no_meas.pht-pipe.json"
    pipeline.write_text(ImagePipeline(ops=[BlurGauss(sigma=1.0)]).to_json() or "", encoding="utf-8")
    out = tmp_path / "out"
    assert run_cli(tiny_inputs, out, mode="process", pipeline=pipeline, extra=ZARR_RGB).exit_code == 0
    for store in _stores(out):
        assert not (store / "tables").exists()
        block = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
        assert "tables" not in block["attributes"]["phenotypic"]


def test_a_flat_export_refuses_embeddings(tmp_path):
    """``write_process_only_layer`` never drops a table on the floor."""
    from phenotypic import Image
    from phenotypic._cli._cli_process_only import write_process_only_layer
    from phenotypic.data import load_synth_yeast_plate

    image = Image(load_synth_yeast_plate())
    with pytest.raises(ValueError, match="nowhere to put embedding tables"):
        write_process_only_layer(
            image, "rgb", tmp_path / "plate.tiff", fmt="tiff",
            embeddings={"toy-v1/grid": object()},  # type: ignore[dict-item]
        )
    assert not (tmp_path / "plate.tiff").exists()


@requires_torch
def test_process_store_has_grid_table_and_stays_consolidated(tmp_path, tiny_inputs, stub_backbone):
    out = tmp_path / "out"
    result = run_cli(
        tiny_inputs, out, mode="process", pipeline=grid_process_pipeline(tmp_path), extra=ZARR_RGB
    )
    assert result.exit_code == 0, result.output
    for store in _stores(out):
        _assert_grid_run_folder(store, DINO_MODEL)


@requires_torch
def test_process_store_stays_byte_reproducible(tmp_path, tiny_inputs, stub_backbone):
    pipeline = grid_process_pipeline(tmp_path)
    for name in ("a", "b"):
        result = run_cli(tiny_inputs, tmp_path / name, mode="process", pipeline=pipeline, extra=ZARR_RGB)
        assert result.exit_code == 0, result.output
    stores = [_stores(tmp_path / name)[0] for name in ("a", "b")]
    assert tree_digest(stores[0]) == tree_digest(stores[1])


@requires_torch
def test_the_grid_entry_records_the_cpu_device(tmp_path, tiny_inputs, stub_backbone):
    from phenotypic.sdk_ import list_table_runs

    out = tmp_path / "out"
    run_cli(tiny_inputs, out, mode="process", pipeline=grid_process_pipeline(tmp_path), extra=ZARR_RGB)
    (run,) = list_table_runs(_stores(out)[0])
    assert run.descriptor["embeddings"][f"{DINO_MODEL}/grid"]["model"]["device"] == "cpu"
