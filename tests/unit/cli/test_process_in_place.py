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
    inodes = {
        path: path.stat().st_ino
        for path in (store / "tables" / first.run_id).rglob("*")
        if path.is_file()
    }
    assert inodes
    result = in_place(kit.tree, kit.pipeline(tmp_path, 1))
    assert result.exit_code == 0, result.output
    for each in _stores(kit.tree):
        assert [r.order for r in list_table_runs(each)] == [1, 2]
    assert tree_digest(store / "tables" / first.run_id) == before
    # Carried as hard links: the same files, not copies of them.
    assert {path: path.stat().st_ino for path in inodes} == inodes
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


def test_a_v1_store_is_refused_with_the_migrate_remedy(toy_process_store_tree, tmp_path):
    """A process store whose root still declares the v1 flat tables layout."""
    tree = toy_process_store_tree
    store = _stores(tree)[-1]
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    root["attributes"]["phenotypic"]["tables"] = {
        "measurements": {"path": "tables/measurements/table.parquet"}
    }
    (store / "zarr.json").write_text(json.dumps(root), encoding="utf-8")
    before = tree_digest(tree)
    result = in_place(tree, _toy_pipeline(tmp_path, 1))
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


# ---------------------------------------------------------------------------
# P5 review (C21r): refusals the user decided, and the fixes beside them
# ---------------------------------------------------------------------------


def _tree_from(tmp_path: Path, inputs: Path, pipeline: Path, extra, name: str) -> Path:
    """A copy process run's output, its machine state removed (an AutoConvertRaw-GC tree)."""
    import shutil

    out = tmp_path / name
    result = run_cli(inputs, out, mode="process", pipeline=pipeline, extra=list(extra))
    assert result.exit_code == 0, result.output
    shutil.rmtree(out / ".phenotypic")
    return out


def _write_plates(root: Path, arrays: list) -> Path:
    """``<root>/ds/plate_<i>.tif`` for each array."""
    import tifffile

    (root / "ds").mkdir(parents=True)
    for index, array in enumerate(arrays, start=1):
        tifffile.imwrite(root / "ds" / f"plate_{index}.tif", array)
    return root


@pytest.mark.parametrize(
    "legacy", ["progress", "processing_state.json", "processing_events.log"]
)
@pytest.mark.parametrize("extra", [(), ("--restart",)])
def test_a_legacy_machine_state_name_at_the_input_root_is_refused(
    toy_process_store_tree, tmp_path, legacy, extra
):
    """Q2 (review C1): the legacy migration would move user data into .phenotypic/."""
    tree = toy_process_store_tree
    if legacy == "progress":
        (tree / "ds").rename(tree / "progress")  # a dataset named progress
    else:
        (tree / legacy).write_text("{}", encoding="utf-8")
    before = tree_digest(tree)
    result = in_place(tree, _toy_pipeline(tmp_path, 1), [*ZARR_RGB, *extra])
    _assert_refused(result, legacy)
    assert "rename" in result.output
    assert tree_digest(tree) == before
    assert not (tree / ".phenotypic").exists()


def test_restart_refuses_a_legacy_target_that_holds_an_input(tmp_path, tiny_inputs):
    """No mode deletes an input through a legacy restart target (review C1).

    ``--input`` is not inside ``progress/``, so the old check passed it; a
    scanned image is.
    """
    tree = tiny_inputs
    (tree / "ds").rename(tree / "progress")
    before = tree_digest(tree)
    result = run_cli(tree, tree, mode="full", extra=["--restart"])
    _assert_refused(result, "progress")
    assert tree_digest(tree) == before


def test_a_bundle_store_is_refused(tmp_path, tiny_inputs, registered_toy_embedder):
    """Q1 (review I1): a full run's store carries image_class; in place is for process stores."""
    from phenotypic.sdk_ import resolve_run_state

    out = tmp_path / "full"
    assert run_cli(tiny_inputs, out, mode="full").exit_code == 0
    completion = resolve_run_state(out, depth="deep").completion
    zarr_dir = out / "results" / "ds" / "zarr"
    before = tree_digest(out)
    _assert_refused(in_place(zarr_dir, toy_grid_process_pipeline(tmp_path)), "process stores")
    assert tree_digest(out) == before
    assert resolve_run_state(out, depth="deep").completion == completion


def test_in_place_appends_and_never_rewrites_earlier_applications(kit, tmp_path):
    """Review I1: only the new application is stripped of wall-clock fields."""
    store = _stores(kit.tree)[0]
    root_path = store / "zarr.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    applications = root["attributes"]["phenotypic"]["provenance"]["applications"]
    for operation in applications[0]["operations"]:
        operation["applied_at_utc"] = "2026-01-01T00:00:00Z"
        operation["duration_seconds"] = 1.5
    root_path.write_text(json.dumps(root), encoding="utf-8")
    assert in_place(kit.tree, kit.pipeline(tmp_path, 1)).exit_code == 0
    after = json.loads(root_path.read_text(encoding="utf-8"))
    kept = after["attributes"]["phenotypic"]["provenance"]["applications"]
    assert kept[: len(applications)] == applications
    for operation in kept[-1]["operations"]:
        assert "applied_at_utc" not in operation and "duration_seconds" not in operation


@pytest.mark.parametrize("shape", ["store link", "dataset link"])
def test_a_symlinked_store_is_refused(toy_process_store_tree, tmp_path, shape):
    """Q4 (review I2): a store that is a link, or resolves outside --input."""
    import os
    import shutil

    tree = toy_process_store_tree
    elsewhere = tmp_path / "elsewhere"
    if shape == "store link":
        store = _stores(tree)[0]
        elsewhere.mkdir()
        shutil.move(str(store), str(elsewhere / store.name))
        os.symlink(elsewhere / store.name, store)
    else:
        shutil.move(str(tree / "ds"), str(elsewhere))
        os.symlink(elsewhere, tree / "ds")
    before = tree_digest(elsewhere)
    result = in_place(tree, _toy_pipeline(tmp_path, 1))
    _assert_refused(result, "does not follow symlinked stores")
    assert tree_digest(elsewhere) == before
    if shape == "store link":
        assert _stores(tree)[0].is_symlink()


def _legacy_integer_gray_tree(tmp_path: Path, dtype) -> Path:
    """Two process gray stores whose stored level-0 ``gray`` is *dtype* (pre-0.20)."""
    import numpy as np

    from phenotypic import Image
    from phenotypic._cli._cli_process_only import write_process_only_layer
    from phenotypic.data import load_synth_yeast_plate

    peak = np.iinfo(dtype).max
    gray = (np.asarray(load_synth_yeast_plate().gray[:]) * peak).astype(dtype)
    tree = tmp_path / f"legacy_{np.dtype(dtype).name}"
    for index in (1, 2):
        image = Image(arr=gray.copy(), name=f"plate_{index}")
        image._data.gray = gray.copy()
        image._data.detect_mat = gray.copy()
        write_process_only_layer(
            image, "gray", tree / "ds" / f"plate_{index}.ome.zarr", fmt="zarr"
        )
    return tree


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_a_legacy_integer_gray_store_runs_in_place(tmp_path, registered_toy_embedder, dtype):
    """Q3 (review I3): the check compares the layer before and after apply()."""
    import numpy as np
    import zarr

    from phenotypic._cli._cli_process_in_place import level0_digest

    tree = _legacy_integer_gray_tree(tmp_path, getattr(np, dtype))
    store = _stores(tree)[0]
    stored = np.asarray(zarr.open_array(store=str(store / "gray" / "0"), mode="r"))
    assert stored.dtype == np.dtype(dtype), "fixture did not write an integer gray"
    extra = ["--layer", "gray", "--process-format", "zarr"]
    result = in_place(tree, toy_grid_process_pipeline(tmp_path), extra)
    assert result.exit_code == 0, result.output
    (run,) = list_table_runs(store)
    entry = run.descriptor["embeddings"]["toy-v1/grid"]
    # The stored array's digest binds the entry, so it reads fresh.
    assert entry["binds"]["array_sha256"] == level0_digest(stored)
    assert entry["stale"] is False
    before = {each: tree_digest(each) for each in _stores(tree)}
    assert in_place(tree, toy_grid_process_pipeline(tmp_path), extra).exit_code == 0
    assert {each: tree_digest(each) for each in _stores(tree)} == before


@pytest.mark.parametrize("variant", ["uint16 rgb", "gray"])
def test_in_place_over_other_store_shapes(tmp_path, registered_toy_embedder, variant):
    import numpy as np

    from phenotypic.data import load_synth_yeast_plate

    plate = load_synth_yeast_plate()
    if variant == "uint16 rgb":
        arrays = [np.asarray(plate.rgb[:]).astype(np.uint16) * 257] * 2
        layer = "rgb"
    else:
        arrays = [(np.asarray(plate.gray[:]) * 255).astype(np.uint8)] * 2
        layer = "gray"
    inputs = _write_plates(tmp_path / "raw", arrays)
    extra = ["--layer", layer, "--process-format", "zarr"]
    tree = _tree_from(tmp_path, inputs, toy_grid_process_pipeline(tmp_path), extra, "stores")
    store = _stores(tree)[0]
    chunks = {p: p.stat().st_ino for p in (store / layer).rglob("*") if p.is_file()}
    result = in_place(tree, _toy_pipeline(tmp_path, 1), extra)
    assert result.exit_code == 0, result.output
    assert [r.order for r in list_table_runs(store)] == [1, 2]
    assert list_table_runs(store)[-1].descriptor["embeddings"]["toy-v2/grid"]["stale"] is False
    assert {p: p.stat().st_ino for p in chunks} == chunks


def test_a_copy_runs_machine_state_needs_restart(tmp_path, tiny_inputs, registered_toy_embedder):
    """A tree still holding a copy run's .phenotypic/ is another run's state."""
    tree = tmp_path / "stores"
    pipeline = toy_grid_process_pipeline(tmp_path)
    assert run_cli(tiny_inputs, tree, mode="process", pipeline=pipeline, extra=list(ZARR_RGB)).exit_code == 0
    before = tree_digest(tree)
    refused = in_place(tree, _toy_pipeline(tmp_path, 1))
    assert refused.exit_code == RunExitCode.ERROR, refused.output
    assert "Cannot continue" in refused.output and "--restart" in refused.output
    assert tree_digest(tree) == before
    result = in_place(tree, _toy_pipeline(tmp_path, 1), [*ZARR_RGB, "--restart"])
    assert result.exit_code == 0, result.output
    assert [r.order for r in list_table_runs(_stores(tree)[0])] == [1, 2]


def test_an_orphaned_trash_is_refused_with_its_restore(toy_process_store_tree, tmp_path):
    """Review I5: a crash between promote_store's renames leaves only the .trash."""
    tree = toy_process_store_tree
    store = _stores(tree)[0]
    trash = store.parent / f".{store.name}.0123456789abcdef.trash"
    store.rename(trash)
    before = tree_digest(tree)
    result = in_place(tree, _toy_pipeline(tmp_path, 1))
    _assert_refused(result, trash.name)
    assert f"mv {trash} {store}" in " ".join(result.output.split())
    assert tree_digest(tree) == before


def test_part_debris_is_listed_and_left_alone(toy_process_store_tree, tmp_path):
    """Review M6: a crashed in-place run's .part is named at startup, never deleted."""
    tree = toy_process_store_tree
    store = _stores(tree)[0]
    part = store.parent / f".{store.name}.fedcba9876543210.part"
    part.mkdir()
    (part / "zarr.json").write_text("{}", encoding="utf-8")
    result = in_place(tree, _toy_pipeline(tmp_path, 1))
    assert result.exit_code == 0, result.output
    assert part.name in result.output and "Warning" in result.output
    assert part.is_dir()


def test_an_unselected_input_is_not_blamed(toy_process_store_tree, tmp_path):
    """Review M7: the refusal checks the images the run would process, after --image-manifest."""
    import numpy as np
    import tifffile

    tree = toy_process_store_tree
    tifffile.imwrite(tree / "ds" / "notes.tif", np.zeros((8, 8), dtype=np.uint8))
    manifest = tmp_path / "approved.images"
    manifest.write_text("ds/plate_1.ome.zarr\nds/plate_2.ome.zarr\n", encoding="utf-8")
    result = in_place(tree, _toy_pipeline(tmp_path, 1), [*ZARR_RGB, "--image-manifest", str(manifest)])
    assert result.exit_code == 0, result.output
    assert [r.order for r in list_table_runs(_stores(tree)[0])] == [1, 2]


# ---------------------------------------------------------------------------
# P5 review (C21r): the core's continuation and errors
# ---------------------------------------------------------------------------


def _block_with_run(entry: dict) -> dict:
    sha = "a" * 64
    return {
        "tables": {
            "schema_version": 2,
            "runs": {f"1-{sha[:12]}": {"order": 1, "pipeline_sha256": sha, **entry}},
            "upstream": {},
        }
    }, sha


@pytest.mark.parametrize(
    ("entry", "keys", "current"),
    [
        ({}, {"toy-v1/grid"}, False),
        ({"embeddings": {}}, {"toy-v1/grid"}, False),
        ({"embeddings": {"toy-v1/grid": {"stale": False}}}, {"toy-v1/grid"}, True),
        ({"embeddings": {"toy-v1/grid": {"stale": True}}}, {"toy-v1/grid"}, False),
        ({"embeddings": {"toy-v2/grid": {"stale": False}}}, {"toy-v1/grid"}, False),
        ({"embeddings": {"toy-v1/grid": {"stale": False}}}, set(), False),
    ],
)
def test_the_owned_run_is_current_only_with_every_own_key_fresh(entry, keys, current):
    """Review M1."""
    from phenotypic._cli._cli_process_in_place import _owned_run_is_current

    block, sha = _block_with_run(entry)
    assert _owned_run_is_current(block, sha, keys) is current


def test_a_corrupt_root_in_the_skip_check_is_a_per_image_failure(toy_process_store_tree, tmp_path):
    """Review M3: never an error outside the scientific boundary, retried for ever."""
    from phenotypic._cli._cli_failure_tracker import PerImageScientificError
    from phenotypic._cli._cli_process_in_place import process_in_place_core
    from phenotypic._cli._cli_utils import input_read_settings

    store = _stores(toy_process_store_tree)[0]
    (store / "zarr.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(PerImageScientificError):
        process_in_place_core(
            _toy_pipeline(tmp_path, 1),
            store,
            toy_process_store_tree,
            layer="rgb",
            read_kwargs=input_read_settings(None, "gray"),
        )


def test_the_skip_warning_is_logged_once_per_process(toy_process_store_tree, tmp_path, caplog):
    """Review M5."""
    import logging

    from phenotypic._cli import _cli_process_in_place
    from phenotypic._cli._cli_process_in_place import process_in_place_core
    from phenotypic._cli._cli_utils import input_read_settings
    from phenotypic.measure import MeasureSize

    from phenotypic import ImagePipeline

    text = ImagePipeline(
        meas={"grid_embedding": toy_grid_embedder_of(token="toy-v2"), "size": MeasureSize()}
    ).to_json() or ""
    pipeline = tmp_path / "with_size.pht-pipe.json"
    pipeline.write_text(text, encoding="utf-8")
    _cli_process_in_place._WARNED_SKIPS.clear()
    with caplog.at_level(logging.WARNING, logger=_cli_process_in_place.__name__):
        for store in _stores(toy_process_store_tree):
            process_in_place_core(
                pipeline,
                store,
                toy_process_store_tree,
                layer="rgb",
                read_kwargs=input_read_settings(None, "gray"),
            )
    skips = [r for r in caplog.records if "meas:size skipped" in r.getMessage()]
    assert len(skips) == 1


# ---------------------------------------------------------------------------
# P5 review (C21r): the transaction re-consolidates what it found consolidated
# ---------------------------------------------------------------------------


def _group_adding_plan(phenotypic):
    def populate(part):
        from phenotypic.sdk_._measurement_tables import _ensure_group

        _ensure_group(part / "tables" / "added-by-test")

    return populate


def test_a_consolidated_store_is_reconsolidated_without_asking(kit):
    """Measure mode passes no consolidate=; a consolidated root must not go stale."""
    from phenotypic.sdk_._measurement_tables import _rewrite_store_tables

    store = _stores(kit.tree)[0]
    old = json.loads((store / "zarr.json").read_text(encoding="utf-8"))["consolidated_metadata"]
    assert "tables/added-by-test" not in old["metadata"]
    _rewrite_store_tables(
        store, plan=_group_adding_plan, durable=False, commit_guard=None, clear_tables=False
    )
    new = json.loads((store / "zarr.json").read_text(encoding="utf-8"))["consolidated_metadata"]
    assert "tables/added-by-test" in new["metadata"]


def test_an_unconsolidated_store_stays_unconsolidated(kit):
    from phenotypic.sdk_._measurement_tables import _rewrite_store_tables

    store = _stores(kit.tree)[0]
    root = store / "zarr.json"
    document = json.loads(root.read_text(encoding="utf-8"))
    document.pop("consolidated_metadata")
    root.write_text(json.dumps(document), encoding="utf-8")
    _rewrite_store_tables(
        store, plan=_group_adding_plan, durable=False, commit_guard=None, clear_tables=False
    )
    after = json.loads(root.read_text(encoding="utf-8"))
    assert "consolidated_metadata" not in after
    assert after["attributes"] == document["attributes"]
