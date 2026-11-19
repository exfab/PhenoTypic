"""In-place process runs: ``--output`` equal to ``--input`` (#281 P5 T3, spec E7).

An in-place run adds the pipeline's run folder to each input store where it
is, through the measure rewrite's transaction, and never rewrites a pixel.

Every behaviour runs twice, through the ``kit`` fixture: ``toy`` (the
torch-free toy grid embedder) and ``dino`` (``MeasureDinoEmbedding`` on the
stub backbone, which needs torch and skips without it).
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from phenotypic.sdk_ import RunExitCode, list_table_runs
from tests._tiny_run import (
    ZARR_RGB,
    grid_embedder,
    grid_process_pipeline,
    run_cli,
    toy_grid_embedder_of,
    toy_grid_process_pipeline,
    touch_root,
    tree_digest,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="Requires torch (stub backbone)"
)


@dataclass
class Kit:
    """One embedder flavour: its store tree and the pipelines run over it."""

    tree: Path
    #: ``pipeline(directory, n)``: ``n == 0`` built the tree; 1 and 2 are others.
    pipeline: Callable[[Path, int], Path]
    #: The model token of pipeline ``n``.
    model: Callable[[int], str]
    #: A meas entry for :func:`with_meas_and_ops`.
    embedder: Callable[[], object]


def _toy_pipeline(directory: Path, n: int) -> Path:
    return toy_grid_process_pipeline(directory, **({"token": f"toy-v{n + 1}"} if n else {}))


_DINO_POOLING = ("crop_cls", "crop_mean", "masked_dense")


@pytest.fixture(params=["toy", pytest.param("dino", marks=requires_torch)])
def kit(request, tmp_path: Path) -> Kit:
    if request.param == "toy":
        tree = request.getfixturevalue("toy_process_store_tree")
        return Kit(tree, _toy_pipeline, lambda n: f"toy-v{n + 1}", toy_grid_embedder_of)
    tree = request.getfixturevalue("process_store_tree")
    return Kit(
        tree,
        lambda directory, n: grid_process_pipeline(directory, pooling=_DINO_POOLING[n]),
        lambda n: "dinov2-vitb14",
        grid_embedder,
    )


def in_place(tree: Path, pipeline: Path, extra=ZARR_RGB):
    return run_cli(tree, tree, mode="process", pipeline=pipeline, extra=list(extra))


def _stores(tree: Path) -> list[Path]:
    return sorted(tree.rglob("*.ome.zarr"))


def with_meas_and_ops(directory: Path, ops: list, embedder) -> Path:
    """A pipeline file: *ops*, and *embedder* as ``meas["grid_embedding"]``."""
    import hashlib

    from phenotypic import ImagePipeline

    text = ImagePipeline(ops=ops, meas={"grid_embedding": embedder}).to_json() or ""
    path = directory / f"in_place_ops-{hashlib.sha256(text.encode()).hexdigest()[:8]}.pht-pipe.json"
    path.write_text(text, encoding="utf-8")
    return path


_PIXEL_NUDGER: type | None = None


def pixel_nudging_class() -> type:
    """A bare ``ImageOperation`` (no corrector, no enhancer) that adds 1 to ``rgb[0, 0]``."""
    global _PIXEL_NUDGER
    if _PIXEL_NUDGER is None:
        from phenotypic.abc_ import ImageOperation

        class PixelNudgingOp(ImageOperation):
            """Changes one pixel of ``rgb``, which no class check catches."""

            def _operate(self, image):
                pixels = image.rgb[:].copy()
                pixels[0, 0] = (pixels[0, 0].astype("int64") + 1) % 256
                image.set_image(pixels)
                return image

        _PIXEL_NUDGER = PixelNudgingOp
    return _PIXEL_NUDGER


@pytest.fixture
def registered_pixel_nudger(monkeypatch):
    import phenotypic

    cls = pixel_nudging_class()
    monkeypatch.setattr(phenotypic, cls.__name__, cls, raising=False)
    return cls


# ---------------------------------------------------------------------------
# The write
# ---------------------------------------------------------------------------


def test_in_place_adds_folder_2_and_leaves_folder_1_byte_identical(kit, tmp_path):
    store = _stores(kit.tree)[0]
    first = list_table_runs(store)[-1]
    before = tree_digest(store / "tables" / first.run_id)
    result = in_place(kit.tree, kit.pipeline(tmp_path, 1))
    assert result.exit_code == 0, result.output
    for each in _stores(kit.tree):
        assert [r.order for r in list_table_runs(each)] == [1, 2]
    assert tree_digest(store / "tables" / first.run_id) == before
    second = list_table_runs(store)[-1]
    assert set(second.descriptor["embeddings"]) == {f"{kit.model(1)}/grid"}
    assert second.descriptor["embeddings"][f"{kit.model(1)}/grid"]["stale"] is False


def test_pixel_arrays_are_never_rewritten(kit, tmp_path):
    from phenotypic._cli._cli_failure_tracker import file_sha256

    store = _stores(kit.tree)[0]
    chunks = {
        p: (p.stat().st_ino, file_sha256(p))
        for p in (store / "rgb").rglob("*")
        if p.is_file()
    }
    assert chunks
    assert in_place(kit.tree, kit.pipeline(tmp_path, 1)).exit_code == 0
    assert {p: (p.stat().st_ino, file_sha256(p)) for p in chunks} == chunks


def test_an_independent_reader_sees_the_new_folder(kit, tmp_path):
    import pyarrow.parquet as pq
    import zarr

    assert in_place(kit.tree, kit.pipeline(tmp_path, 1)).exit_code == 0
    store = _stores(kit.tree)[0]
    run = list_table_runs(store)[-1]
    group = zarr.open_consolidated(store, mode="r")
    members = dict(group.members(max_depth=None))
    assert f"tables/{run.run_id}/embeddings/{kit.model(1)}/grid" in members
    # The first folder is still in the consolidated view too.
    first = list_table_runs(store)[0]
    assert f"tables/{first.run_id}/embeddings/{kit.model(0)}/grid" in members
    assert pq.read_table(run.embedding_path(kit.model(1), "grid")).num_rows == 4


def test_the_journal_gains_one_process_application(kit, tmp_path):
    import hashlib

    store = _stores(kit.tree)[0]

    def applications():
        root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
        return root["attributes"]["phenotypic"]["provenance"]["applications"]

    before = applications()
    pipeline = kit.pipeline(tmp_path, 1)
    assert in_place(kit.tree, pipeline).exit_code == 0
    after = applications()
    assert after[: len(before)] == before
    (added,) = after[len(before):]
    assert added["kind"] == "process" and added["status"] == "complete"
    assert added["pipeline"]["sha256"] == hashlib.sha256(pipeline.read_bytes()).hexdigest()


def test_rerunning_does_no_work(kit, tmp_path):
    pipeline = kit.pipeline(tmp_path, 1)
    assert in_place(kit.tree, pipeline).exit_code == 0
    before = {store: tree_digest(store) for store in _stores(kit.tree)}
    result = in_place(kit.tree, pipeline)
    assert result.exit_code == 0, result.output
    assert {store: tree_digest(store) for store in _stores(kit.tree)} == before


def test_the_core_does_no_work_for_a_store_holding_its_folder(kit, tmp_path):
    """Without the CLI's records (a direct caller), the store itself says it is done."""
    from phenotypic._cli._cli_process_in_place import process_in_place_core
    from phenotypic._cli._cli_utils import input_read_settings

    store = _stores(kit.tree)[0]
    before = tree_digest(store)
    # Pipeline 0 built the tree, so its folder is there, every entry fresh.
    assert process_in_place_core(
        kit.pipeline(tmp_path, 0),
        store,
        kit.tree,
        layer="rgb",
        read_kwargs=input_read_settings(None, "gray"),
    )
    assert tree_digest(store) == before


def test_the_work_id_ignores_the_runs_own_write(kit, tmp_path):
    from phenotypic._cli._cli_failure_tracker import file_sha256
    from phenotypic._cli._cli_process_in_place import in_place_input_digest

    store = _stores(kit.tree)[0]
    before = (in_place_input_digest(store), file_sha256(store))
    assert in_place(kit.tree, kit.pipeline(tmp_path, 1)).exit_code == 0
    assert in_place_input_digest(store) == before[0]
    assert file_sha256(store) != before[1]
    # A change to the pixels the run reads does change it.
    chunk = next(p for p in sorted((store / "rgb" / "0").rglob("*")) if p.is_file() and p.name != "zarr.json")
    chunk.write_bytes(chunk.read_bytes() + b"\0")
    assert in_place_input_digest(store) != before[0]


def test_the_processing_digest_marks_in_place():
    from phenotypic._cli._cli_failure_tracker import (
        processing_configuration_digest_from_values,
    )

    values = dict(
        image_type="Image", nrows=None, ncols=None, bit_depth=None, detect_mode="gray",
        process_only_layer="rgb", ext=".tiff", process_format="zarr",
        include_dataset_column=True, overlay_alpha=0.2, save_overlays=True,
    )
    copy = processing_configuration_digest_from_values(**values)
    assert processing_configuration_digest_from_values(**values, in_place=False) == copy
    assert processing_configuration_digest_from_values(**values, in_place=True) != copy


def test_dry_run_writes_nothing(kit, tmp_path):
    before = tree_digest(kit.tree)
    result = in_place(kit.tree, kit.pipeline(tmp_path, 1), [*ZARR_RGB, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert tree_digest(kit.tree) == before


def test_machine_state_lives_under_the_input_root_and_is_not_an_input(kit, tmp_path):
    from phenotypic._cli._cli_directory_scanner import scan_directory_structure

    assert in_place(kit.tree, kit.pipeline(tmp_path, 1)).exit_code == 0
    assert (kit.tree / ".phenotypic").is_dir()
    # A leftover part of an interrupted transaction is not an input either.
    store = _stores(kit.tree)[0]
    leftover = store.parent / f".{store.name}.0123abcd.part"
    leftover.mkdir()
    (leftover / "zarr.json").write_text("{}", encoding="utf-8")
    scanned = scan_directory_structure(kit.tree)
    assert {name: sorted(p.name for p in paths) for name, paths in scanned.items()} == {
        "ds": ["plate_1.ome.zarr", "plate_2.ome.zarr"]
    }


def _files(stores: list[Path]) -> dict[Path, tuple[int, bytes]]:
    """Every file under *stores*, with its inode and its bytes."""
    return {
        path: (path.stat().st_ino, path.read_bytes())
        for store in stores
        for path in store.rglob("*")
        if path.is_file()
    }


def test_restart_in_place_touches_no_store(kit, tmp_path):
    """--restart clears only .phenotypic/: every store file keeps its inode and bytes."""
    pipeline = kit.pipeline(tmp_path, 1)
    assert in_place(kit.tree, pipeline).exit_code == 0
    stores = _stores(kit.tree)
    before = _files(stores)
    result = in_place(kit.tree, pipeline, [*ZARR_RGB, "--restart"])
    assert result.exit_code == 0, result.output
    assert "Cleared previous machine-state" in result.output
    assert _files(stores) == before


def test_another_pipeline_in_place_needs_restart(kit, tmp_path):
    assert in_place(kit.tree, kit.pipeline(tmp_path, 1)).exit_code == 0
    store = _stores(kit.tree)[0]
    folders = {run.run_id: tree_digest(run.folder) for run in list_table_runs(store)}
    assert len(folders) == 2
    before = tree_digest(kit.tree)

    refused = in_place(kit.tree, kit.pipeline(tmp_path, 2))
    assert refused.exit_code == RunExitCode.ERROR, refused.output
    assert "Pipeline contents changed" in refused.output
    assert "--restart" in refused.output
    assert tree_digest(kit.tree) == before

    result = in_place(kit.tree, kit.pipeline(tmp_path, 2), [*ZARR_RGB, "--restart"])
    assert result.exit_code == 0, result.output
    for each in _stores(kit.tree):
        assert [r.order for r in list_table_runs(each)] == [1, 2, 3]
    runs = list_table_runs(store)
    assert {run.run_id: tree_digest(run.folder) for run in runs[:2]} == folders
    assert set(runs[2].descriptor["embeddings"]) == {f"{kit.model(2)}/grid"}


# ---------------------------------------------------------------------------
# Refusals: one usage-error line each, nothing written
# ---------------------------------------------------------------------------


def _assert_refused(result, needle: str) -> None:
    errors = [line for line in result.output.splitlines() if line.startswith("Error:")]
    assert result.exit_code == 2, result.output
    assert len(errors) == 1 and needle in errors[0], result.output


@pytest.mark.parametrize(
    ("pipeline", "extra", "needle"),
    [
        ("base", [*ZARR_RGB, "--overwrite"], "--overwrite"),
        ("base", ["--layer", "rgb", "--process-format", "tiff"], "store"),
        ("base", ["--layer", "gray", "--process-format", "zarr"], "primary series"),
        ("detector", ZARR_RGB, "ObjectDetector"),
        ("corrector", ZARR_RGB, "ImageCorrector"),
    ],
)
def test_in_place_refusals_write_nothing(kit, tmp_path, pipeline, extra, needle):
    from phenotypic import ImagePipeline
    from phenotypic.correction import ColorDenoise
    from phenotypic.detect import OtsuDetector

    path = {
        "base": lambda: kit.pipeline(tmp_path, 1),
        "detector": lambda: with_meas_and_ops(
            tmp_path, [ImagePipeline(ops=[OtsuDetector()])], kit.embedder()
        ),
        "corrector": lambda: with_meas_and_ops(
            tmp_path, [ImagePipeline(ops=[ColorDenoise()])], kit.embedder()
        ),
    }[pipeline]()
    before = tree_digest(kit.tree)
    _assert_refused(in_place(kit.tree, path, extra), needle)
    assert tree_digest(kit.tree) == before


def test_a_detector_inside_a_skipped_measurer_is_refused(kit, tmp_path):
    """Any slot, any depth: a measurer's own detector, which process mode would skip."""
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSymZones

    import hashlib

    from phenotypic import ImagePipeline

    pipeline = ImagePipeline(
        meas={
            "grid_embedding": kit.embedder(),
            "zones": MeasureSymZones(center_detector=OtsuDetector()),
        }
    )
    text = pipeline.to_json() or ""
    path = tmp_path / f"zones-{hashlib.sha256(text.encode()).hexdigest()[:8]}.pht-pipe.json"
    path.write_text(text, encoding="utf-8")
    before = tree_digest(kit.tree)
    _assert_refused(in_place(kit.tree, path), "ObjectDetector")
    assert tree_digest(kit.tree) == before


def test_a_raw_input_is_refused(tiny_inputs, tmp_path, registered_toy_embedder):
    before = tree_digest(tiny_inputs)
    _assert_refused(in_place(tiny_inputs, toy_grid_process_pipeline(tmp_path)), "PhenoTypic store")
    assert tree_digest(tiny_inputs) == before


def test_a_third_party_store_is_refused(kit, tmp_path):
    store = _stores(kit.tree)[-1]
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    del root["attributes"]["phenotypic"]
    (store / "zarr.json").write_text(json.dumps(root), encoding="utf-8")
    before = tree_digest(kit.tree)
    _assert_refused(in_place(kit.tree, kit.pipeline(tmp_path, 1)), "PhenoTypic store")
    assert tree_digest(kit.tree) == before


def test_a_v1_store_is_refused_with_the_migrate_remedy(v1_store, tmp_path, registered_toy_embedder):
    tree = v1_store.parent
    before = tree_digest(tree)
    result = in_place(tree, toy_grid_process_pipeline(tmp_path))
    _assert_refused(result, "v1 flat layout")
    assert "--mode migrate" in result.output
    assert tree_digest(tree) == before


def test_a_single_store_input_is_refused(kit, tmp_path):
    store = _stores(kit.tree)[0]
    before = tree_digest(kit.tree)
    _assert_refused(in_place(store, kit.pipeline(tmp_path, 1)), "names one image or store")
    assert tree_digest(kit.tree) == before


@pytest.mark.parametrize("direction", ["output inside input", "input inside output"])
def test_strict_containment_stays_refused(kit, tmp_path, direction):
    inner = kit.tree / "ds"
    inputs, output = (kit.tree, inner) if direction == "output inside input" else (inner, kit.tree)
    before = tree_digest(kit.tree)
    result = run_cli(inputs, output, mode="process", pipeline=kit.pipeline(tmp_path, 1), extra=list(ZARR_RGB))
    _assert_refused(result, "must not overlap")
    assert tree_digest(kit.tree) == before


# ---------------------------------------------------------------------------
# Per image: the level-0 check and the root check
# ---------------------------------------------------------------------------


def test_a_pixel_changing_op_the_class_check_misses_fails_the_image(
    kit, tmp_path, registered_pixel_nudger
):
    from phenotypic._cli._cli_failure_tracker import read_terminal_failures

    before = {store: tree_digest(store) for store in _stores(kit.tree)}
    sneaky = with_meas_and_ops(tmp_path, [registered_pixel_nudger()], kit.embedder())
    result = in_place(kit.tree, sneaky)
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, result.output
    for store in _stores(kit.tree):
        assert len(list_table_runs(store)) == 1
    # Nothing written into any store, and the level-0 check is what failed it.
    assert {store: tree_digest(store) for store in _stores(kit.tree)} == before
    failures = read_terminal_failures(kit.tree)
    assert len(failures) == 2
    assert {f.exception_type for f in failures} == {"InPlacePixelsChangedError"}


def test_root_changed_before_swap_rebuilds_once_then_fails(kit, tmp_path, monkeypatch):
    # Review Focus 5
    from phenotypic._cli import _cli_process_in_place

    store = _stores(kit.tree)[0]
    touches = iter([True, False])
    real = _cli_process_in_place.write_run_entry

    def writer(part, *args, **kw):
        # Only while *store*'s own part is built: touching it during the other
        # store's write would change a root whose success record is already
        # published, which is a different failure from the one pinned here.
        if Path(part).name.startswith(f".{store.name}.") and next(touches, True):
            touch_root(store)  # a reprocess or a second writer, mid-transaction
        return real(part, *args, **kw)

    monkeypatch.setattr(_cli_process_in_place, "write_run_entry", writer)
    result = in_place(kit.tree, kit.pipeline(tmp_path, 1))
    assert result.exit_code == 0, result.output
    assert [r.order for r in list_table_runs(store)] == [1, 2]

    touches = iter([True, True])
    before = tree_digest(store / "rgb")
    # Another pipeline over the same stores starts a new run: --restart.
    result = in_place(kit.tree, kit.pipeline(tmp_path, 2), [*ZARR_RGB, "--restart"])
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES, result.output
    assert [r.order for r in list_table_runs(store)] == [1, 2]
    assert tree_digest(store / "rgb") == before
    assert not list(store.parent.glob(f".{store.name}.*.part"))


# ---------------------------------------------------------------------------
# The transaction's two new options (measure mode passes neither)
# ---------------------------------------------------------------------------


def _noop_plan(phenotypic):
    return lambda part: None


def test_the_rewrite_refuses_a_root_that_changed_and_leaves_no_part(kit):
    from phenotypic.sdk_._measurement_tables import (
        StoreRootChangedError,
        _rewrite_store_tables,
    )

    store = _stores(kit.tree)[0]
    before = tree_digest(store)
    with pytest.raises(StoreRootChangedError):
        _rewrite_store_tables(
            store,
            plan=_noop_plan,
            durable=False,
            commit_guard=None,
            clear_tables=False,
            expected_root_sha256="0" * 64,
        )
    assert tree_digest(store) == before
    assert not list(store.parent.glob(f".{store.name}.*.part"))


def test_the_rewrite_consolidates_on_request(kit):
    import hashlib

    from phenotypic.sdk_._measurement_tables import _rewrite_store_tables

    store = _stores(kit.tree)[0]
    root = store / "zarr.json"
    document = json.loads(root.read_text(encoding="utf-8"))
    document.pop("consolidated_metadata")
    root.write_text(json.dumps(document), encoding="utf-8")
    _rewrite_store_tables(
        store,
        plan=_noop_plan,
        durable=False,
        commit_guard=None,
        clear_tables=False,
        expected_root_sha256=hashlib.sha256(root.read_bytes()).hexdigest(),
        consolidate=True,
    )
    consolidated = json.loads(root.read_text(encoding="utf-8"))["consolidated_metadata"]
    run = list_table_runs(store)[0]
    assert f"tables/{run.run_id}/embeddings/{kit.model(0)}/grid" in consolidated["metadata"]
