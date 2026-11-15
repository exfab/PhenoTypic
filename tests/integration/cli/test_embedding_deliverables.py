"""CLI wiring of the embedding producer and its deliverables (#281 P4 T4).

Spec E2 (every per-image writer embeds beside measuring and writes one run
folder), E5 (``deliverables/embeddings/``), D6a (aggregation reads the folder
each image's record certifies), D7 (stale entries are skipped) and D9 (a
corrupt embedding never revokes measurement authority).

Every test here runs ``MeasureDinoEmbedding`` on the ``stub_backbone``
fixture, which needs torch, so the module skips in an environment without
it. The torch-free half of the aggregation is pinned by
``tests/unit/cli/test_embedding_aggregation.py``.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="Requires torch (stub backbone)"
)


def _is_stale(run) -> bool:
    """Whether *run*'s crop entry, as its store's root declares it now, is stale."""
    from phenotypic.sdk_ import list_table_runs

    from tests._tiny_run import DINO_CROPS_KEY

    (current,) = [r for r in list_table_runs(run.store) if r.run_id == run.run_id]
    return current.descriptor["embeddings"][DINO_CROPS_KEY]["stale"] is True


def test_full_run_publishes_embedding_deliverable(finished_tiny_run_with_embeddings):
    import polars as pl
    import pyarrow.parquet as pq

    out = finished_tiny_run_with_embeddings
    t = pq.read_table(out / "deliverables/embeddings/dinov2-vitb14/crops.parquet")
    assert {"Metadata_Dataset", "Object_Label", "embedding"} <= set(t.column_names)
    master = pl.read_parquet(out / "deliverables/master_measurements.parquet")
    assert "embedding" not in master.columns
    metrics = json.loads((out / "deliverables/metrics.json").read_text(encoding="utf-8"))
    assert metrics["embeddings"]["dinov2-vitb14/crops"]["rows"] == t.num_rows
    # One row per measured colony, keyed as the master is.
    assert t.num_rows == master.height
    keys = {"Metadata_Dataset", "Metadata_ImageName", "Object_Label"}
    joined = master.select(sorted(keys)).join(
        pl.from_arrow(t.select(sorted(keys))), on=sorted(keys), how="inner"
    )
    assert joined.height == master.height


def test_the_manifest_describes_each_deliverable(finished_tiny_run_with_embeddings):
    import hashlib

    deliverables = finished_tiny_run_with_embeddings / "deliverables"
    manifest = json.loads(
        (deliverables / "embeddings" / "manifest.json").read_text(encoding="utf-8")
    )
    entry = manifest["entries"]["dinov2-vitb14/crops"]
    path = deliverables / entry["path"]
    assert entry["path"] == "embeddings/dinov2-vitb14/crops.parquet"
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert entry["stores"] == 2 and entry["stale_skipped"] == []
    (descriptor,) = entry["descriptors"]
    assert descriptor["producer"]["class"] == "MeasureDinoEmbedding"
    assert {"files", "binds", "stale"}.isdisjoint(descriptor)


def test_embeddings_get_no_split_and_no_mirror_section(finished_tiny_run_with_embeddings):
    from phenotypic.schema import DINO_EMBEDDING

    d = finished_tiny_run_with_embeddings / "deliverables"
    assert not any("Embedding" in p.name for p in (d / "measurements_by_feature").iterdir())
    schema = json.loads((d / "schema.json").read_text(encoding="utf-8"))
    vector = next(c for c in schema["tables"]["embeddings/dinov2-vitb14/crops"] if c["name"] == "embedding")
    # schema.json names a member as ``<enum>.<member>`` (spec C12, ``SIZE.AREA``).
    member = DINO_EMBEDDING.EMBEDDING
    assert vector["member"] == f"{type(member).__name__}.{member.name}"
    assert vector["desc"] == member.desc
    assert vector["dtype"] == "FixedSizeList<float16>[8]"  # the stub's hidden size
    assert vector["model"] == "dinov2-vitb14"
    assert vector["producer"] == "MeasureDinoEmbedding"
    readme = (d / "README.md").read_text(encoding="utf-8")
    assert "DinoEmbedding_Embedding" not in readme
    assert "deliverables/embeddings/dinov2-vitb14/crops.parquet" in readme


def test_stale_entries_are_skipped_and_counted(tree_with_one_stale_store):
    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables

    from tests._tiny_run import rows_in

    out, runs = tree_with_one_stale_store            # 3 images, one stale
    counts = aggregate_embedding_tables(out, runs)
    assert counts["dinov2-vitb14/crops"]["stale_skipped"] == 1
    assert counts["dinov2-vitb14/crops"]["rows"] == sum(rows_in(r) for r in runs if not _is_stale(r))
    manifest = json.loads(
        (out / "deliverables/embeddings/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["entries"]["dinov2-vitb14/crops"]["stale_skipped"] == [
        "results/ds/zarr/plate_3.ome.zarr"
    ]


def test_corrupt_embedding_does_not_revoke_measurements(finished_tiny_run_with_embeddings):
    from phenotypic.sdk_ import (
        EmbeddingTableUnreadableError,
        latest_measurement_run,
        read_embedding_table,
        resolve_run_state,
    )

    from tests._tiny_run import one_store

    # spec D9: embeddings are additive to measurement authority
    out = finished_tiny_run_with_embeddings
    store = one_store(out, stem="plate_1")
    latest_measurement_run(store).embedding_path("dinov2-vitb14", "crops").write_bytes(b"corrupt")
    state = resolve_run_state(out, depth="deep")
    assert all(image.verdict == "verified" for image in state.images.values())
    with pytest.raises(EmbeddingTableUnreadableError):
        read_embedding_table(store, "dinov2-vitb14/crops")


def test_a_corrupt_embedding_fails_finalization_naming_the_store(
    finished_tiny_run_with_embeddings,
):
    """Spec E5: an unreadable embedding fails finalization, like the master.

    The images stay verified (D9), so the remedy the message names is a
    same-pipeline ``--mode measure``, which rewrites the embeddings without
    re-running detection.
    """
    from phenotypic.sdk_ import latest_measurement_run

    from tests._tiny_run import one_store, run_cli

    out = finished_tiny_run_with_embeddings
    store = one_store(out, stem="plate_1")
    latest_measurement_run(store).embedding_path("dinov2-vitb14", "crops").write_bytes(b"corrupt")
    result = run_cli(None, out, mode="recompile")
    assert result.exit_code != 0, result.output
    text = result.output + repr(result.exception)
    assert "plate_1.ome.zarr" in text
    assert "--mode measure" in text


def test_same_pipeline_measure_keeps_the_runs_crop_embeddings(
    finished_tiny_run_with_embeddings,
):
    """OQ5: ``replace_image_tables(embeddings=None)`` drops the run's embeddings,
    so measure mode must pass ``pipeline.embed(image)``."""
    import pyarrow.parquet as pq

    from phenotypic.sdk_ import list_table_runs, read_embedding_table

    from tests._tiny_run import DINO_CROPS_KEY, crop_entry, one_store, run_cli

    out = finished_tiny_run_with_embeddings
    store = one_store(out, stem="plate_1")
    before = read_embedding_table(store, DINO_CROPS_KEY).num_rows
    pipeline = out.parent / "dino_crops.pht-pipe.json"

    result = run_cli(None, out, mode="measure", pipeline=pipeline)

    assert result.exit_code == 0, result.output
    assert len(list_table_runs(store)) == 1
    table = read_embedding_table(store, DINO_CROPS_KEY)
    assert table.num_rows == before
    assert crop_entry(store)["stale"] is False
    published = pq.read_metadata(
        out / "deliverables/embeddings/dinov2-vitb14/crops.parquet"
    ).num_rows
    assert published == 2 * before


def test_a_rerun_without_embeddings_drops_the_deliverable(
    finished_tiny_run_with_embeddings,
):
    """A finalization publishes only what its certified folders hold (D6a)."""
    from tests._tiny_run import run_cli

    out = finished_tiny_run_with_embeddings
    assert (out / "deliverables" / "embeddings").is_dir()
    result = run_cli(None, out, mode="measure")  # tiny_pipeline: no embedder
    assert result.exit_code == 0, result.output
    assert not (out / "deliverables" / "embeddings").exists()
    metrics = json.loads((out / "deliverables/metrics.json").read_text(encoding="utf-8"))
    assert "embeddings" not in metrics


def test_byte_stable_embedding_deliverables(tmp_path, tiny_inputs, stub_backbone):
    """Spec C11: identical runs give identical ``deliverables/embeddings/`` bytes."""
    from tests._tiny_run import run_cli, tree_digest, with_dino_crops

    pipeline = with_dino_crops(tmp_path)
    digests = []
    for name in ("a", "b"):
        out = tmp_path / name
        assert run_cli(tiny_inputs, out, pipeline=pipeline).exit_code == 0
        assert (out / "deliverables" / "embeddings" / "manifest.json").is_file()
        digests.append(tree_digest(out / "deliverables" / "embeddings"))
    assert digests[0] == digests[1]


def test_stage3_writes_the_crop_embeddings(tmp_path, monkeypatch, stub_backbone):
    """Stage 3 embeds beside measuring and writes one run folder (spec E2)."""
    import phenotypic
    from click.testing import CliRunner
    from phenotypic import ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.measure import MeasureDinoEmbedding, MeasureSize
    from phenotypic.phenotypicCLI import phenotypic_cli
    from phenotypic.sdk_ import read_embedding_table

    from tests._fakes.fake_gpu_detector import FakeGpuDetector
    from tests._tiny_run import DINO_CROPS_KEY, crop_entry, one_store

    monkeypatch.setattr(phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False)
    images = tmp_path / "images"
    images.mkdir()
    load_synth_yeast_plate().rgb.imsave(filepath=images / "img.tiff")
    pipe = ImagePipeline(
        ops=[FakeGpuDetector(threshold=0.3)],
        meas={"size": MeasureSize(), "emb": MeasureDinoEmbedding(device="cpu")},
    )
    pipe_path = tmp_path / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    out = tmp_path / "out"
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline", str(pipe_path), "--input", str(images), "--output", str(out),
            "--image-type", "Image", "--force-local", "--skip-validation",
            "--njobs", "1",
        ],
    )
    assert result.exit_code == 0, result.output
    store = one_store(out)
    assert read_embedding_table(store, DINO_CROPS_KEY).num_rows > 0
    assert crop_entry(store)["stale"] is False
    assert (out / "deliverables/embeddings/dinov2-vitb14/crops.parquet").is_file()
