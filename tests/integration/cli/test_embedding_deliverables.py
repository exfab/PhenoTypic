"""CLI wiring of the embedding producer and its deliverables (#281 P4 T4, P4 review).

Spec E2 (every per-image writer embeds beside measuring and writes one run
folder), E5 (``deliverables/embeddings/``), D6a (aggregation reads the folder
each image's record certifies), D7 (stale entries are skipped), D9 (a
corrupt embedding never revokes measurement authority), C11 (byte-stable
deliverables), and the review's I1 (a failure changes no deliverable) and
M1a (the aggregate proof declares the files).

Every test here runs ``MeasureDinoEmbedding`` on the ``stub_backbone``
fixture, which needs torch, so the module skips in an environment without
it. The torch-free half of the aggregation is pinned by
``tests/unit/cli/test_embedding_aggregation.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="Requires torch (stub backbone)"
)

OBJECTS = "dinov2-vitb14/rgb/objects"
GRID = "dinov2-vitb14/rgb/grid"


def _is_stale(run) -> bool:
    """Whether *run*'s crop entry, as its store's root declares it now, is stale."""
    from phenotypic.sdk_ import list_table_runs

    (current,) = [r for r in list_table_runs(run.store) if r.run_id == run.run_id]
    return current.descriptor["embeddings"][OBJECTS]["stale"] is True


def _deliverable_snapshot(out: Path) -> dict[str, tuple[bytes, int]]:
    """Every file under ``deliverables/``: its bytes and ``st_mtime_ns``."""
    root = out / "deliverables"
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _corrupt(store: Path, key: str) -> None:
    from phenotypic.sdk_ import latest_measurement_run

    latest_measurement_run(store).embedding_path(key).write_bytes(b"corrupt")


def _rows_of(table, image: str):
    import pyarrow.compute as pc

    return table.filter(pc.equal(table["Metadata_ImageName"], image))


def test_full_run_publishes_embedding_deliverable(finished_tiny_run_with_embeddings):
    import polars as pl
    import pyarrow.parquet as pq

    out = finished_tiny_run_with_embeddings
    t = pq.read_table(out / "deliverables/embeddings/dinov2-vitb14/rgb/objects.parquet")
    assert {"Metadata_Dataset", "Object_Label", "embedding"} <= set(t.column_names)
    master = pl.read_parquet(out / "deliverables/master_measurements.parquet")
    assert "embedding" not in master.columns
    metrics = json.loads((out / "deliverables/metrics.json").read_text(encoding="utf-8"))
    assert metrics["embeddings"]["dinov2-vitb14/rgb/objects"]["rows"] == t.num_rows
    # One row per measured colony, keyed as the master is.
    assert t.num_rows == master.height
    keys = {"Metadata_Dataset", "Metadata_ImageName", "Object_Label"}
    joined = master.select(sorted(keys)).join(
        pl.from_arrow(t.select(sorted(keys))), on=sorted(keys), how="inner"
    )
    assert joined.height == master.height


def test_each_stores_rows_are_its_own_table(finished_tiny_run_with_embeddings):
    """Review T4: plate_2 is mirrored, so its vectors differ from plate_1's,
    and each store's deliverable rows are exactly its own embedding table."""
    import pyarrow.parquet as pq

    from phenotypic.sdk_ import read_embedding_table

    from tests._tiny_run import one_store

    out = finished_tiny_run_with_embeddings
    published = pq.read_table(out / "deliverables/embeddings/dinov2-vitb14/rgb/objects.parquet")
    own = {}
    for stem in ("plate_1", "plate_2"):
        own[stem] = read_embedding_table(one_store(out, stem=stem), OBJECTS)
        rows = _rows_of(published, stem).select(own[stem].column_names)
        assert rows.equals(own[stem]), stem
    assert not own["plate_1"].column("embedding").equals(own["plate_2"].column("embedding"))


def test_the_manifest_describes_each_deliverable(finished_tiny_run_with_embeddings):
    import hashlib

    deliverables = finished_tiny_run_with_embeddings / "deliverables"
    manifest = json.loads(
        (deliverables / "embeddings" / "manifest.json").read_text(encoding="utf-8")
    )
    entry = manifest["entries"]["dinov2-vitb14/rgb/objects"]
    path = deliverables / entry["path"]
    assert entry["path"] == "embeddings/dinov2-vitb14/rgb/objects.parquet"
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
    vector = next(c for c in schema["tables"]["embeddings/dinov2-vitb14/rgb/objects"] if c["name"] == "embedding")
    # schema.json names a member as ``<enum>.<member>`` (spec C12, ``SIZE.AREA``).
    member = DINO_EMBEDDING.EMBEDDING
    assert vector["member"] == f"{type(member).__name__}.{member.name}"
    assert vector["desc"] == member.desc
    assert vector["dtype"] == "FixedSizeList<float16>[8]"  # the stub's hidden size
    assert vector["model"] == "dinov2-vitb14"
    assert vector["producer"] == "MeasureDinoEmbedding"
    # P4 review Q1: the per-row descriptor index, 0 for a one-descriptor run.
    index = next(c for c in schema["tables"]["embeddings/dinov2-vitb14/rgb/objects"] if c["name"] == "descriptor_index")
    assert index["dtype"] == "Int16" and "descriptors" in index["desc"]
    readme = (d / "README.md").read_text(encoding="utf-8")
    assert "DinoEmbedding_Embedding" not in readme
    assert "deliverables/embeddings/dinov2-vitb14/rgb/objects.parquet" in readme
    assert "`descriptor_index`" in readme


def test_the_aggregate_proof_declares_and_enforces_the_embedding_files(
    finished_tiny_run_with_embeddings,
):
    """Review M1a: manifest.json and each published table are required
    outputs, so an edit to either stales the proof, as for metrics.json."""
    import hashlib

    from phenotypic.sdk_ import aggregate_publication_marker_path
    from phenotypic.sdk_._run_state import aggregate_proof_refusal

    out = finished_tiny_run_with_embeddings
    proof = json.loads(aggregate_publication_marker_path(out).read_text(encoding="utf-8"))
    outputs = proof["required_outputs"]
    table = out / "deliverables/embeddings/dinov2-vitb14/rgb/objects.parquet"
    manifest = out / "deliverables/embeddings/manifest.json"
    assert outputs["embeddings/dinov2-vitb14/rgb/objects"]["path"] == "deliverables/embeddings/dinov2-vitb14/rgb/objects.parquet"
    assert outputs["embeddings/dinov2-vitb14/rgb/objects"]["sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
    assert outputs["embeddings_manifest"]["path"] == "deliverables/embeddings/manifest.json"
    for path in (table, manifest):
        original = path.read_bytes()
        assert aggregate_proof_refusal(out) is None
        path.write_bytes(original + b"\0")
        assert "embeddings" in (aggregate_proof_refusal(out) or "")
        path.write_bytes(original)
    assert aggregate_proof_refusal(out) is None


def test_stale_entries_are_skipped_and_counted(tree_with_one_stale_store):
    """A unit test of ``aggregate_embedding_tables`` over a CLI-built tree (review T5).

    It drives the aggregator directly on runs whose stale store's record no
    longer verifies, because no CLI path brings a stale certified folder to
    finalization (review M3): a write that changes the objmap rewrites the
    root and voids the image's record. It is not end-to-end D7 coverage; it
    pins the aggregator's own skip-and-count rule on real stores.
    """
    from phenotypic._cli._cli_embedding_deliverables import aggregate_embedding_tables

    from tests._tiny_run import rows_in

    out, runs = tree_with_one_stale_store            # 3 images, one stale
    counts = aggregate_embedding_tables(out, runs)
    assert counts["dinov2-vitb14/rgb/objects"]["stale_skipped"] == 1
    assert counts["dinov2-vitb14/rgb/objects"]["rows"] == sum(rows_in(r) for r in runs if not _is_stale(r))
    manifest = json.loads(
        (out / "deliverables/embeddings/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["entries"]["dinov2-vitb14/rgb/objects"]["stale_skipped"] == [
        "results/ds/zarr/plate_3.ome.zarr"
    ]


def test_corrupt_embedding_does_not_revoke_measurements(finished_tiny_run_with_embeddings):
    from phenotypic.sdk_ import (
        EmbeddingTableUnreadableError,
        read_embedding_table,
        resolve_run_state,
    )

    from tests._tiny_run import one_store

    # spec D9: embeddings are additive to measurement authority
    out = finished_tiny_run_with_embeddings
    store = one_store(out, stem="plate_1")
    _corrupt(store, OBJECTS)
    state = resolve_run_state(out, depth="deep")
    assert all(image.verdict == "verified" for image in state.images.values())
    with pytest.raises(EmbeddingTableUnreadableError):
        read_embedding_table(store, "dinov2-vitb14/rgb/objects")


@pytest.fixture
def two_key_run(tmp_path, tiny_inputs, stub_backbone) -> Path:
    """A finished full run of :func:`~tests._tiny_run.with_dino_objects_and_grid`: two keys."""
    from tests._tiny_run import differentiate_plate_2, run_cli, with_dino_objects_and_grid

    differentiate_plate_2(tiny_inputs)
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, pipeline=with_dino_objects_and_grid(tmp_path))
    assert result.exit_code == 0, result.output
    return out


def test_a_corrupt_embedding_fails_finalization_and_changes_no_deliverable(two_key_run):
    """Review I1 and T6. The second key (``grid``) of plate_2 is corrupt, so
    the first (``objects``) is the one a non-atomic writer would already have
    replaced. Finalization fails naming the store and the remedy, and no file
    under ``deliverables/`` changes: not the master, the mirror, the
    summaries, the manifest or either table, in bytes or in mtime."""
    from tests._tiny_run import one_store, run_cli

    out = two_key_run
    _corrupt(one_store(out, stem="plate_2"), GRID)
    before = _deliverable_snapshot(out)

    result = run_cli(None, out, mode="recompile")

    assert result.exit_code != 0, result.output
    text = result.output + repr(result.exception)
    assert "plate_2.ome.zarr" in text
    assert "--mode measure" in text and "every image" in text  # review Q2
    after = _deliverable_snapshot(out)
    assert after.keys() == before.keys()
    changed = [name for name in before if after[name] != before[name]]
    assert changed == []
    assert not list((out / "deliverables").rglob("*.tmp"))


def test_a_rerun_after_a_failed_finalization_does_not_hide_it(finished_tiny_run_with_embeddings):
    """Review I1's re-run. A failed finalization changed nothing, so the last
    good publication still stands and its proof still verifies; a plain
    forward re-run finds nothing to do and leaves every deliverable as that
    publication wrote it. Once the image is re-measured, the corruption is
    repaired and a recompile publishes again."""
    from phenotypic.sdk_ import read_embedding_table, resolve_run_state

    from tests._tiny_run import DINO_OBJECTS_KEY, one_store, run_cli

    out = finished_tiny_run_with_embeddings
    store = one_store(out, stem="plate_2")
    _corrupt(store, OBJECTS)
    before = _deliverable_snapshot(out)
    assert run_cli(None, out, mode="recompile").exit_code != 0

    rerun = run_cli(out.parent / "in", out, pipeline=out.parent / "dino_objects.pht-pipe.json")

    assert rerun.exit_code == 0, rerun.output
    assert "already processed" in rerun.output
    assert _deliverable_snapshot(out) == before
    assert resolve_run_state(out, depth="deep").completion == "complete"
    # The remedy the error names repairs the store and publishes again.
    measured = run_cli(None, out, mode="measure", pipeline=out.parent / "dino_objects.pht-pipe.json")
    assert measured.exit_code == 0, measured.output
    assert read_embedding_table(store, DINO_OBJECTS_KEY).num_rows > 0


def test_same_pipeline_measure_keeps_the_runs_crop_embeddings(
    finished_tiny_run_with_embeddings,
):
    """OQ5: ``replace_image_tables(embeddings=None)`` drops the run's embeddings,
    so measure mode must pass ``pipeline.embed(image)``."""
    import pyarrow.parquet as pq

    from phenotypic.sdk_ import list_table_runs, read_embedding_table

    from tests._tiny_run import DINO_OBJECTS_KEY, objects_entry, one_store, run_cli

    out = finished_tiny_run_with_embeddings
    stores = [one_store(out, stem=stem) for stem in ("plate_1", "plate_2")]
    before = [read_embedding_table(store, DINO_OBJECTS_KEY) for store in stores]
    pipeline = out.parent / "dino_objects.pht-pipe.json"

    result = run_cli(None, out, mode="measure", pipeline=pipeline)

    assert result.exit_code == 0, result.output
    for store, table in zip(stores, before):
        assert len(list_table_runs(store)) == 1
        assert read_embedding_table(store, DINO_OBJECTS_KEY).equals(table)
        assert objects_entry(store)["stale"] is False
    published = pq.read_metadata(
        out / "deliverables/embeddings/dinov2-vitb14/rgb/objects.parquet"
    ).num_rows
    assert published == sum(table.num_rows for table in before)


def test_a_rerun_without_embeddings_drops_the_deliverable(
    finished_tiny_run_with_embeddings,
):
    """A finalization publishes only what its certified folders hold (D6a)."""
    from phenotypic.sdk_ import aggregate_publication_marker_path

    from tests._tiny_run import run_cli

    out = finished_tiny_run_with_embeddings
    assert (out / "deliverables" / "embeddings").is_dir()
    result = run_cli(None, out, mode="measure")  # tiny_pipeline: no embedder
    assert result.exit_code == 0, result.output
    assert not (out / "deliverables" / "embeddings").exists()
    metrics = json.loads((out / "deliverables/metrics.json").read_text(encoding="utf-8"))
    assert "embeddings" not in metrics
    proof = json.loads(aggregate_publication_marker_path(out).read_text(encoding="utf-8"))
    assert not any(name.startswith("embeddings") for name in proof["required_outputs"])


def test_byte_stable_embedding_deliverables(tmp_path, tiny_inputs, stub_backbone):
    """Spec C11: identical runs give identical ``deliverables/embeddings/`` bytes."""
    from tests._tiny_run import run_cli, tree_digest, with_dino_objects

    pipeline = with_dino_objects(tmp_path)
    digests = []
    for name in ("a", "b"):
        out = tmp_path / name
        assert run_cli(tiny_inputs, out, pipeline=pipeline).exit_code == 0
        assert (out / "deliverables" / "embeddings" / "manifest.json").is_file()
        digests.append(tree_digest(out / "deliverables" / "embeddings"))
    assert digests[0] == digests[1]


#: A whole CLI run in a fresh interpreter on the stub backbone: argv[1] is
#: the input root, argv[2] the output, argv[3] the pipeline file.
_SUBPROCESS_RUN = """
import sys
import pytest
from tests._tiny_run import install_stub_backbone, run_cli
patch = pytest.MonkeyPatch()
install_stub_backbone(patch)
result = run_cli(__import__("pathlib").Path(sys.argv[1]), __import__("pathlib").Path(sys.argv[2]),
                 pipeline=__import__("pathlib").Path(sys.argv[3]))
sys.exit(result.exit_code)
"""


def test_byte_stable_across_two_processes(tmp_path, tiny_inputs):
    """Review T7: two fresh interpreters, two backbone loads, the same bytes."""
    from tests._tiny_run import tree_digest, with_dino_objects

    root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root), *filter(None, [env.get("PYTHONPATH")])]
    )
    env["QT_QPA_PLATFORM"] = "offscreen"
    pipeline = with_dino_objects(tmp_path)
    digests = []
    for name in ("a", "b"):
        out = tmp_path / name
        done = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_RUN, str(tiny_inputs), str(out), str(pipeline)],
            cwd=root, env=env, capture_output=True, text=True, timeout=600,
        )
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
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
    from tests._tiny_run import DINO_OBJECTS_KEY, objects_entry, one_store

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
    assert read_embedding_table(store, DINO_OBJECTS_KEY).num_rows > 0
    assert objects_entry(store)["stale"] is False
    assert (out / "deliverables/embeddings/dinov2-vitb14/rgb/objects.parquet").is_file()


# ---------------------------------------------------------------------------
# Review T8: more CLI paths
# ---------------------------------------------------------------------------


def test_a_grid_embedder_publishes_its_grid_table(two_key_run):
    """A grid embedder through the CLI: one row per cell per image, keyed by
    the ``Grid_*`` cell columns, beside the objects table."""
    import pyarrow.parquet as pq

    from phenotypic.schema import GRID as GRID_SCHEMA

    from tests._tiny_run import TOY_GRID_COL_EDGES, TOY_GRID_ROW_EDGES

    out = two_key_run
    table = pq.read_table(out / "deliverables/embeddings/dinov2-vitb14/rgb/grid.parquet")
    cells = (len(TOY_GRID_ROW_EDGES) - 1) * (len(TOY_GRID_COL_EDGES) - 1)
    assert table.num_rows == 2 * cells
    assert str(GRID_SCHEMA.ROW_MAJOR_IDX) in table.column_names
    assert table.column_names[:2] == ["Metadata_Dataset", "Metadata_ImageName"]
    assert table.column_names[-1] == "embedding"
    metrics = json.loads((out / "deliverables/metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["embeddings"]) == {OBJECTS, GRID}
    assert metrics["embeddings"][GRID] == {"rows": 2 * cells, "stale_skipped": 0}


def test_restart_republishes_the_embeddings(finished_tiny_run_with_embeddings):
    import pyarrow.parquet as pq

    from tests._tiny_run import run_cli

    out = finished_tiny_run_with_embeddings
    path = out / "deliverables/embeddings/dinov2-vitb14/rgb/objects.parquet"
    before = path.read_bytes()
    result = run_cli(
        out.parent / "in", out, pipeline=out.parent / "dino_objects.pht-pipe.json",
        extra=["--restart"],
    )
    assert result.exit_code == 0, result.output
    assert path.read_bytes() == before                      # byte-stable (C11)
    assert pq.read_metadata(path).num_row_groups == 2


def test_continuation_refinalizes_with_the_new_images_embeddings(
    finished_tiny_run_with_embeddings,
):
    """A third plate arrives; the plain re-run processes it and re-finalizes,
    and the deliverable then holds all three stores."""
    import pyarrow.parquet as pq

    from tests._tiny_run import run_cli

    out = finished_tiny_run_with_embeddings
    inputs = out.parent / "in"
    (inputs / "ds" / "plate_3.tif").write_bytes((inputs / "ds" / "plate_1.tif").read_bytes())
    result = run_cli(inputs, out, pipeline=out.parent / "dino_objects.pht-pipe.json")
    assert result.exit_code == 0, result.output
    manifest = json.loads(
        (out / "deliverables/embeddings/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["entries"][OBJECTS]["stores"] == 3
    table = pq.read_table(out / "deliverables/embeddings/dinov2-vitb14/rgb/objects.parquet")
    assert set(table.column("Metadata_ImageName").to_pylist()) == {
        "plate_1", "plate_2", "plate_3",
    }


def test_two_named_dino_grids_publish_two_deliverables(tmp_path, tiny_inputs, stub_backbone):
    """#281 named grid regions, the DINO twin of the toy run in
    ``tests/unit/cli/test_embedding_aggregation.py``: rig1's ``wells`` and
    ``agar`` grids of one model, one backbone load."""
    from tests._tiny_run import (
        assert_named_grids_published,
        differentiate_plate_2,
        grid_embedder,
        run_cli,
        with_named_grids,
    )

    differentiate_plate_2(tiny_inputs)
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, pipeline=with_named_grids(tmp_path, grid_embedder))
    assert result.exit_code == 0, result.output
    assert_named_grids_published(out, "dinov2-vitb14")
