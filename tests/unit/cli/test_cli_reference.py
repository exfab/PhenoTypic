"""Reference planning, the run manifest, and the worker context."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from phenotypic import ImagePipeline, ReferenceContext
from phenotypic._cli import _cli_reference as ref
from phenotypic._cli._cli_types import Dataset
from phenotypic._core._reference_context import ReferenceContextError
from phenotypic.enhance import SubtractBlank
from phenotypic.sdk_._io_constants import (
    metadata_csv_deliverable_path,
    reference_manifest_path,
    reference_metadata_snapshot_path,
)
from tests.unit.cli._preflight_support import make_config


def _write(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.full((8, 8), value, dtype=np.uint8))
    return path


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "in" / "plate1"
    for stem, value in (("blank", 20), ("t01", 60), ("t02", 90), ("t03", 90), ("orphan", 5)):
        _write(root / f"{stem}.tif", value)
    table = tmp_path / "layout.csv"
    pd.DataFrame({
        "Metadata_ImageName": ["t01", "t02", "t02", "t03"],
        "Metadata_BlankImage": ["blank", "blank", "other", "t03"],
    }).to_csv(table, index=False)
    dataset = Dataset(
        name="plate1",
        images=[root / f"{s}.tif" for s in ("t01", "t02", "t03", "orphan")],
        input_dir=root,
        output_dir=tmp_path / "out" / "plate1",
    )
    return table, dataset


def _pipe():
    return ImagePipeline(ops={"sb": SubtractBlank()})


def test_plan_classifies_every_image(tree):
    """t03 names itself: the reference plate is planned like every plate."""
    table, dataset = tree
    plan = ref.plan_references(ReferenceContext(table), _pipe(), [dataset], hash_images=True)
    assert plan.total_images == 4
    assert plan.unmatched == ("plate1/orphan",)
    assert plan.ambiguous == ("plate1/t02",)
    assert plan.unresolved == ()
    assert plan.images_by_dataset["plate1"]["blank"].endswith("plate1/blank.tif")
    assert set(plan.digests["plate1"]) == {"t01", "t03"}


def test_the_reference_plate_is_planned_as_its_own_blank(tree):
    """Its blank is its own file: resolved, hashed and digested like any blank."""
    table, dataset = tree
    plan = ref.plan_references(ReferenceContext(table), _pipe(), [dataset], hash_images=True)
    own = str((dataset.input_dir / "t03.tif").resolve())
    assert plan.images_by_dataset["plate1"]["t03"] == own
    assert own in plan.file_digests
    assert "t03" not in plan.unplanned["plate1"]
    assert not hasattr(plan, "self_referenced")


def test_an_empty_blank_cell_is_still_unplanned(tree, tmp_path):
    _, dataset = tree
    empty = tmp_path / "empty.csv"
    pd.DataFrame({"Metadata_ImageName": ["t01"], "Metadata_BlankImage": [None]}).to_csv(empty, index=False)
    single = Dataset(name="plate1", images=[dataset.input_dir / "t01.tif"],
                     input_dir=dataset.input_dir, output_dir=dataset.output_dir)
    plan = ref.plan_references(ReferenceContext(empty), _pipe(), [single], hash_images=True)
    assert plan.ambiguous == ("plate1/t01",)
    assert plan.unplanned["plate1"]["t01"] == "unplanned:ambiguous"


def test_plan_without_hashing_still_resolves(tree):
    table, dataset = tree
    plan = ref.plan_references(ReferenceContext(table), _pipe(), [dataset], hash_images=False)
    assert "blank" in plan.images_by_dataset["plate1"]
    assert plan.digests == {"plate1": {}}


def test_plan_without_hashing_never_reads_a_reference_image(tree, monkeypatch):
    """The preflight's contract is headers only (S7): resolve names, hash nothing."""
    from phenotypic._core import _reference_context

    def refuse(path):
        raise AssertionError(f"planning without hashing hashed {path}")

    monkeypatch.setattr(_reference_context, "reference_file_digest", refuse)
    table, dataset = tree
    plan = ref.plan_references(ReferenceContext(table), _pipe(), [dataset], hash_images=False)
    assert "blank" in plan.images_by_dataset["plate1"]


def test_a_blank_written_with_its_own_extension_is_planned_like_any_blank(tree, tmp_path):
    table, dataset = tree
    named = tmp_path / "named.csv"
    pd.DataFrame({"Metadata_ImageName": ["t01"], "Metadata_BlankImage": ["t01.tif"]}).to_csv(named, index=False)
    single = Dataset(name="plate1", images=[dataset.input_dir / "t01.tif"],
                     input_dir=dataset.input_dir, output_dir=dataset.output_dir)
    plan = ref.plan_references(ReferenceContext(named), _pipe(), [single], hash_images=True)
    assert set(plan.digests["plate1"]) == {"t01"}
    assert set(plan.file_digests) == {str((dataset.input_dir / "t01.tif").resolve())}


def test_a_blank_matching_no_single_file_is_unresolved(tree, tmp_path):
    table, dataset = tree
    missing = tmp_path / "missing.csv"
    pd.DataFrame({"Metadata_ImageName": ["t01"], "Metadata_BlankImage": ["nope"]}).to_csv(missing, index=False)
    single = Dataset(name="plate1", images=[dataset.input_dir / "t01.tif"],
                     input_dir=dataset.input_dir, output_dir=dataset.output_dir)
    plan = ref.plan_references(ReferenceContext(missing), _pipe(), [single], hash_images=True)
    assert plan.unresolved == ("plate1/t01",)
    assert plan.digests == {"plate1": {}}


def test_digest_changes_only_for_the_image_whose_blank_changed(tree, tmp_path):
    table, dataset = tree
    root = dataset.input_dir
    _write(root / "blank2.tif", 25)
    good = tmp_path / "good.csv"
    pd.DataFrame({"Metadata_ImageName": ["t01", "t02"], "Metadata_BlankImage": ["blank", "blank"]}).to_csv(good, index=False)
    edited = tmp_path / "edited.csv"
    pd.DataFrame({"Metadata_ImageName": ["t01", "t02"], "Metadata_BlankImage": ["blank2", "blank"]}).to_csv(edited, index=False)
    a = ref.plan_references(ReferenceContext(good), _pipe(), [dataset], hash_images=True).digests["plate1"]
    b = ref.plan_references(ReferenceContext(edited), _pipe(), [dataset], hash_images=True).digests["plate1"]
    assert a["t01"] != b["t01"]
    assert a["t02"] == b["t02"]


# ---- a blank value that is a file path --------------------------------------


def _one_image(dataset: Dataset, stem: str = "t01") -> Dataset:
    return Dataset(name="plate1", images=[dataset.input_dir / f"{stem}.tif"],
                   input_dir=dataset.input_dir, output_dir=dataset.output_dir)


def _layout(tmp_path: Path, value: str, name: str = "paths.csv") -> Path:
    path = tmp_path / name
    pd.DataFrame({"Metadata_ImageName": ["t01"], "Metadata_BlankImage": [value]}).to_csv(
        path, index=False
    )
    return path


def test_a_relative_file_path_is_planned_once_to_an_absolute_path(tree, tmp_path, monkeypatch):
    """Startup resolves it in the submitting process; a worker, whose cwd is
    not the submitter's, uses only the planned absolute path."""
    _, dataset = tree
    run = tmp_path / "run"
    blank = _write(run / "blanks" / "b0.tif", 33)
    layout = _layout(tmp_path, "blanks/b0.tif")
    monkeypatch.chdir(run)
    ctx = ReferenceContext(layout)
    plan = ref.plan_references(ctx, _pipe(), [_one_image(dataset)], hash_images=True)
    expected = str(blank.resolve())
    assert plan.images_by_dataset["plate1"]["blanks/b0.tif"] == expected
    assert set(plan.file_digests) == {expected}
    out = tmp_path / "out"
    ref.write_reference_manifest(
        out, plan=plan, table_path=layout, table_sha256=ctx.table_sha256, read_kwargs={}
    )

    elsewhere = tmp_path / "elsewhere"
    _write(elsewhere / "blanks" / "b0.tif", 99)  # a different file at the same relative path
    monkeypatch.chdir(elsewhere)
    with ref.worker_reference_context(out, "plate1") as active:
        assert active.resolve_image("blanks/b0.tif") == blank.resolve()
        assert active.load_image("blanks/b0.tif").gray.shape == (8, 8)


@pytest.mark.parametrize("kind", ["missing", "unsupported suffix"])
def test_a_file_path_that_is_not_an_image_is_unresolved(tree, tmp_path, kind):
    _, dataset = tree
    target = tmp_path / ("missing.tif" if kind == "missing" else "notes.json")
    if kind != "missing":
        target.write_text("{}", encoding="utf-8")
    plan = ref.plan_references(
        ReferenceContext(_layout(tmp_path, str(target))), _pipe(), [_one_image(dataset)],
        hash_images=True,
    )
    assert plan.unresolved == ("plate1/t01",)


def test_a_file_path_to_the_images_own_file_is_planned_like_any_blank(tree, tmp_path):
    _, dataset = tree
    own = dataset.input_dir / "t01.tif"
    plan = ref.plan_references(
        ReferenceContext(_layout(tmp_path, str(own))), _pipe(), [_one_image(dataset)],
        hash_images=True,
    )
    assert set(plan.digests["plate1"]) == {"t01"}
    assert set(plan.file_digests) == {str(own.resolve())}


def test_a_same_stem_blank_in_another_folder_is_planned(tree, tmp_path):
    """``blanks/t01.tif`` for ``in/plate1/t01.tif``: an ordinary blank."""
    _, dataset = tree
    blank = _write(tmp_path / "blanks" / "t01.tif", 33)
    plan = ref.plan_references(
        ReferenceContext(_layout(tmp_path, str(blank))), _pipe(), [_one_image(dataset)],
        hash_images=True,
    )
    assert set(plan.digests["plate1"]) == {"t01"}
    assert set(plan.file_digests) == {str(blank.resolve())}


def test_digest_changes_when_the_blank_file_is_rewritten(tree):
    """Same table, new blank pixels: the image's work must be redone."""
    table, dataset = tree
    before = ref.plan_references(ReferenceContext(table), _pipe(), [dataset], hash_images=True)
    _write(dataset.input_dir / "blank.tif", 21)
    after = ref.plan_references(ReferenceContext(table), _pipe(), [dataset], hash_images=True)
    assert before.digests["plate1"]["t01"] != after.digests["plate1"]["t01"]


def test_manifest_round_trip_and_worker_context(tree, tmp_path):
    table, dataset = tree
    out = tmp_path / "out"
    ctx = ReferenceContext(table)
    plan = ref.plan_references(ctx, _pipe(), [dataset], hash_images=True)
    ref.write_reference_manifest(out, plan=plan, table_path=table, table_sha256=ctx.table_sha256, read_kwargs={})
    assert reference_manifest_path(out).is_file()
    with ref.worker_reference_context(out, "plate1") as active:
        assert ReferenceContext.current() is active
        assert active.dataset == "plate1"
        assert active.load_image("blank").gray.shape == (8, 8)
    assert ReferenceContext.current() is None
    assert ref.reference_digest_for(out, "plate1", "t01") == plan.digests["plate1"]["t01"]
    # An image planning failed is pinned to WHY it failed (review F4).
    assert ref.reference_digest_for(out, "plate1", "orphan") == "unplanned:unmatched"
    # An image the manifest does not know at all (arrived after startup).
    assert ref.reference_digest_for(out, "plate1", "late") == ref.UNPLANNED_DIGEST


def test_no_manifest_means_no_context_and_no_digest(tmp_path):
    with ref.worker_reference_context(tmp_path, "plate1") as active:
        assert active is None
    assert ReferenceContext.current() is None
    assert ref.reference_digest_for(tmp_path, "plate1", "t01") is None
    assert ref.reference_digest_for(None, "plate1", "t01") is None


def test_a_removed_manifest_stops_answering(tree, tmp_path):
    table, dataset = tree
    out = tmp_path / "out"
    ctx = ReferenceContext(table)
    plan = ref.plan_references(ctx, _pipe(), [dataset], hash_images=True)
    ref.write_reference_manifest(out, plan=plan, table_path=table, table_sha256=ctx.table_sha256, read_kwargs={})
    assert ref.reference_digest_for(out, "plate1", "t01") is not None
    ref.remove_reference_manifest(out)
    ref.remove_reference_manifest(out)  # idempotent
    assert ref.reference_digest_for(out, "plate1", "t01") is None


def test_a_worker_with_a_manifest_needs_the_dataset_name(tree, tmp_path):
    table, dataset = tree
    out = tmp_path / "out"
    ctx = ReferenceContext(table)
    plan = ref.plan_references(ctx, _pipe(), [dataset], hash_images=False)
    ref.write_reference_manifest(out, plan=plan, table_path=table, table_sha256=ctx.table_sha256, read_kwargs={})
    with pytest.raises(ValueError, match="dataset name"):
        with ref.worker_reference_context(out, None):
            pass


def test_worker_refuses_a_table_changed_after_planning(tree, tmp_path):
    table, dataset = tree
    out = tmp_path / "out"
    ctx = ReferenceContext(table)
    plan = ref.plan_references(ctx, _pipe(), [dataset], hash_images=False)
    ref.write_reference_manifest(out, plan=plan, table_path=table, table_sha256=ctx.table_sha256, read_kwargs={})
    table.write_text(table.read_text() + "t09,blank\n", encoding="utf-8")
    ref._manifest_base_context.cache_clear()
    with pytest.raises(ref.ReferencePlanStaleError, match="changed") as caught:
        with ref.worker_reference_context(out, "plate1"):
            pass
    # Not a property of the image, so not a ReferenceContextError that an apply
    # site would wrap as a per-image scientific failure (review F2).
    assert not isinstance(caught.value, ReferenceContextError)


# ---- each planned blank's bytes (code review #1) ---------------------------


def _planned_run(tree, tmp_path) -> tuple[Path, Dataset]:
    table, dataset = tree
    out = tmp_path / "out"
    ctx = ReferenceContext(table)
    plan = ref.plan_references(ctx, _pipe(), [dataset], hash_images=True)
    ref.write_reference_manifest(
        out, plan=plan, table_path=table, table_sha256=ctx.table_sha256, read_kwargs={}
    )
    return out, dataset


def _load_as_an_apply_would(active: ReferenceContext, name: str) -> None:
    """Every enclosing operation and pipeline wraps a failure in RuntimeError."""
    try:
        active.load_image(name)
    except Exception as exc:
        raise RuntimeError("[SubtractBlank] (step 1/1, key='sb')") from exc


def test_the_manifest_records_each_planned_files_digest(tree, tmp_path):
    from phenotypic._core._reference_context import reference_file_digest

    out, dataset = _planned_run(tree, tmp_path)
    # t01's blank, and t03's: the reference plate is its own blank.
    files = [(dataset.input_dir / f"{s}.tif").resolve() for s in ("blank", "t03")]
    manifest = ref.read_reference_manifest(out)
    assert manifest["schema_version"] == ref.MANIFEST_SCHEMA_VERSION
    assert manifest["reference_files"] == {str(f): reference_file_digest(f) for f in files}


def test_a_blank_rewritten_after_planning_leaves_the_worker_as_stale(tree, tmp_path):
    out, dataset = _planned_run(tree, tmp_path)
    _write(dataset.input_dir / "blank.tif", 33)
    with pytest.raises(ref.ReferencePlanStaleError, match="blank") as caught:
        with ref.worker_reference_context(out, "plate1") as active:
            _load_as_an_apply_would(active, "blank")
    assert not isinstance(caught.value, ReferenceContextError)


def test_a_version_1_manifest_refuses_every_reference_file_load(tree, tmp_path):
    """No planned file digests: refused (non-terminal) rather than unchecked."""
    import json

    out, _ = _planned_run(tree, tmp_path)
    path = reference_manifest_path(out)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["reference_files"]
    manifest["schema_version"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ref.ReferencePlanStaleError, match="blank"):
        with ref.worker_reference_context(out, "plate1") as active:
            _load_as_an_apply_would(active, "blank")


@pytest.mark.parametrize("kind", ["plain", "wrapped reference error"])
def test_every_other_failure_leaves_the_worker_context_unchanged(tree, tmp_path, kind):
    """Only a changed reference file is converted; terminal classification of
    everything else is untouched (same object, same type)."""
    from phenotypic._core._reference_context import ReferenceImageError

    out, _ = _planned_run(tree, tmp_path)
    if kind == "plain":
        error: Exception = ValueError("a scientific failure")
    else:
        error = RuntimeError("[SubtractBlank] failed")
        error.__cause__ = ReferenceImageError("Reference image 'gone' matches 0 files")
    with pytest.raises(type(error)) as caught:
        with ref.worker_reference_context(out, "plate1"):
            raise error
    assert caught.value is error


def test_an_unplanned_digest_follows_the_failure_reason(tree, tmp_path):
    """Review F4: fixing one cause and hitting another re-derives the image."""
    table, dataset = tree
    single = Dataset(name="plate1", images=[dataset.input_dir / "t01.tif"],
                     input_dir=dataset.input_dir, output_dir=dataset.output_dir)

    def unplanned(rows):
        path = tmp_path / "rows.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        plan = ref.plan_references(ReferenceContext(path), _pipe(), [single], hash_images=True)
        assert plan.digests == {"plate1": {}}
        return plan.unplanned["plate1"]["t01"]

    unmatched = unplanned({"Metadata_ImageName": ["t99"], "Metadata_BlankImage": ["blank"]})
    missing = unplanned({"Metadata_ImageName": ["t01"], "Metadata_BlankImage": ["nope"]})
    other_missing = unplanned({"Metadata_ImageName": ["t01"], "Metadata_BlankImage": ["gone"]})
    empty = unplanned({"Metadata_ImageName": ["t01"], "Metadata_BlankImage": [None]})
    assert unmatched == "unplanned:unmatched"
    assert missing.startswith("unplanned:unresolved:")
    assert len({unmatched, missing, other_missing, empty}) == 4
    # Never equal to a real (hex) digest.
    assert all(value.startswith("unplanned") for value in (missing, empty))


def test_snapshot_copies_bytes_and_reuses_existing(tmp_path, tree):
    table, _ = tree
    out = tmp_path / "out"
    snap = ref.snapshot_reference_metadata(out, table)
    assert snap == reference_metadata_snapshot_path(out)
    assert snap.read_bytes() == table.read_bytes()
    assert ref.snapshot_reference_metadata(out, None) == snap
    assert ref.snapshot_reference_metadata(tmp_path / "fresh", None) is None


def test_invalid_bytes_never_replace_a_valid_snapshot(tmp_path, tree):
    table, _ = tree
    out = tmp_path / "out"
    snap = ref.snapshot_reference_metadata(out, table)
    before = snap.read_bytes()
    invalid = tmp_path / "invalid.csv"
    invalid.write_bytes(b'"unterminated')
    with pytest.raises(Exception):
        ref.snapshot_reference_metadata(out, invalid)
    assert snap.read_bytes() == before


def test_manifest_is_parsed_once_for_many_lookups(tree, tmp_path, monkeypatch):
    """Work-ids are computed per image in several startup passes (review M2)."""
    table, dataset = tree
    out = tmp_path / "out"
    ctx = ReferenceContext(table)
    plan = ref.plan_references(ctx, _pipe(), [dataset], hash_images=True)
    ref.write_reference_manifest(out, plan=plan, table_path=table, table_sha256=ctx.table_sha256, read_kwargs={})
    parses = []
    real = ref._parse_manifest
    monkeypatch.setattr(ref, "_parse_manifest", lambda text: (parses.append(1), real(text))[1])
    ref._MANIFEST_CACHE.clear()
    for _ in range(50):
        ref.reference_digest_for(out, "plate1", "t01")
    assert len(parses) == 1


def test_a_manifest_replaced_with_the_same_size_and_mtime_is_reread(tree, tmp_path):
    """Another process's atomic rewrite is a new file; size and mtime can collide.

    A changed blank name of the same length leaves the manifest's size unchanged,
    and two writes inside one timestamp tick share an mtime, so the memo must
    also notice that the path now names a different file.
    """
    _, dataset = tree
    _write(dataset.input_dir / "blanc.tif", 30)
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    pd.DataFrame({"Metadata_ImageName": ["t01"], "Metadata_BlankImage": ["blank"]}).to_csv(first, index=False)
    pd.DataFrame({"Metadata_ImageName": ["t01"], "Metadata_BlankImage": ["blanc"]}).to_csv(second, index=False)
    out, elsewhere = tmp_path / "out", tmp_path / "elsewhere"
    for root, csv in ((out, first), (elsewhere, second)):
        ctx = ReferenceContext(csv)
        plan = ref.plan_references(ctx, _pipe(), [dataset], hash_images=True)
        ref.write_reference_manifest(root, plan=plan, table_path=csv, table_sha256=ctx.table_sha256, read_kwargs={})
    before = ref.reference_digest_for(out, "plate1", "t01")
    live, replacement = reference_manifest_path(out), reference_manifest_path(elsewhere)
    assert live.stat().st_size == replacement.stat().st_size  # the premise
    stamp = live.stat()
    os.replace(replacement, live)
    os.utime(live, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    after = ref.reference_digest_for(out, "plate1", "t01")
    assert after is not None and after != before


def test_store_inputs_are_keyed_by_source_stem(tmp_path):
    """x.ome.zarr is named "x" by imread, not "x.ome" (review M1)."""
    root = tmp_path / "in" / "plate1"
    (root / "t01.ome.zarr").mkdir(parents=True)
    _write(root / "blank.tif", 20)
    table = tmp_path / "layout.csv"
    pd.DataFrame({"Metadata_ImageName": ["t01"], "Metadata_BlankImage": ["blank"]}).to_csv(table, index=False)
    dataset = Dataset(name="plate1", images=[root / "t01.ome.zarr"], input_dir=root, output_dir=tmp_path / "out")
    plan = ref.plan_references(ReferenceContext(table), _pipe(), [dataset], hash_images=True)
    assert plan.unmatched == ()
    assert set(plan.digests["plate1"]) == {"t01"}


def test_snapshot_is_preserved_on_restart():
    from phenotypic.sdk_._io_constants import REFERENCE_METADATA_CSV, preserved_on_restart_names

    assert REFERENCE_METADATA_CSV in preserved_on_restart_names()


def test_the_manifest_is_cleared_by_restart(tree, tmp_path):
    """Re-derived at every startup, so a restart may drop it (it is not an input)."""
    from phenotypic.sdk_._io_constants import machine_state_restart_targets

    table, dataset = tree
    out = tmp_path / "out"
    ctx = ReferenceContext(table)
    plan = ref.plan_references(ctx, _pipe(), [dataset], hash_images=False)
    ref.write_reference_manifest(out, plan=plan, table_path=table, table_sha256=ctx.table_sha256, read_kwargs={})
    ref.snapshot_reference_metadata(out, table)
    targets = machine_state_restart_targets(out)
    assert reference_manifest_path(out) in targets
    assert reference_metadata_snapshot_path(out) not in targets


def test_reference_table_path_follows_the_mode(tmp_path):
    out = tmp_path / "out"
    given = tmp_path / "given.csv"
    given.write_text("ImageName,BlankImage\nt01,blank\n", encoding="utf-8")
    full_snapshot = metadata_csv_deliverable_path(out)
    process_snapshot = reference_metadata_snapshot_path(out)

    # Nothing given and nothing to fall back to.
    assert ref.resolve_reference_table_path(make_config(output_dir=out), out) is None
    for snapshot in (full_snapshot, process_snapshot):
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text("ImageName,BlankImage\nt01,blank\n", encoding="utf-8")

    full = make_config(output_dir=out)
    process = make_config(output_dir=out, process_only_layer="gray")
    measure = make_config(output_dir=out, measure_only=True)
    assert ref.resolve_reference_table_path(full, out) == full_snapshot
    assert ref.resolve_reference_table_path(process, out) == process_snapshot
    assert ref.resolve_reference_table_path(full, None) is None
    # Measure mode applies no operation, so it reads no reference table.
    assert ref.resolve_reference_table_path(measure, out) is None
    assert ref.resolve_reference_table_path(
        make_config(output_dir=out, measure_only=True, metadata_csv=given), out
    ) is None
    # --metadata wins over either snapshot.
    for config in (
        make_config(output_dir=out, metadata_csv=given),
        make_config(output_dir=out, metadata_csv=given, process_only_layer="gray"),
    ):
        assert ref.resolve_reference_table_path(config, out) == given


def test_input_read_kwargs_carry_the_inputs_bit_depth():
    assert ref.input_read_kwargs(make_config()) == {}
    assert ref.input_read_kwargs(make_config(bit_depth=12)) == {"bit_depth": 12}
