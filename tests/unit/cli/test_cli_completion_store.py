"""Per-image completion markers over a store directory (Phase 3 Task 3.8).

``_sha256`` opens its argument as a file and ``valid_image_success`` required
``is_file()``, so an OME-Zarr store — a *directory* — killed the publishing
worker and, once published, reclassified as ``"stage3"`` forever. These tests
pin the ``kind``-tagged descriptor that fixes both halves, and the fingerprint
rules that keep a store descriptor honest (keyed on the root ``zarr.json``'s
contents, never on the directory, never on the absolute path).
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic._cli._cli_completion import (
    SUCCESS_MARKER_VERSION,
    image_data_artifact,
    publish_image_success,
    refresh_success_markers_after_metadata_migration,
    valid_image_success,
)
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic.sdk_ import (
    RECORD_VERSION,
    atomic_write_json,
    file_fingerprint,
    image_record_path,
    zarr_store_path,
)

DATASET = "ds"
STEM = "img"
WORK_ID = "w-1"


@dataclass
class PublishedStore:
    """A published run whose per-image RECORD certifies an OME-Zarr store.

    The field is ``record``, not ``marker``: ``publish_image_success`` returns
    the path it wrote, and after P3's clean break that is
    ``images/<ds>/<stem>.json``. Leaving the old name would have made every
    reader of this fixture believe they were looking at ``image_complete/``,
    which is the class of wrongness this change exists to remove -- a name
    that survives the thing it named.
    """

    output_dir: Path
    store: Path
    record: Path


def _output_manager(output_dir: Path) -> OutputManager:
    return OutputManager.from_config(
        base_dir=output_dir, ext=".tiff", save_overlays=False
    )


def _write_store(output_dir: Path, *, work_id: str | None = WORK_ID) -> Path:
    store = zarr_store_path(output_dir, DATASET, STEM)
    store.parent.mkdir(parents=True, exist_ok=True)
    return Image(np.zeros((8, 8, 3), dtype=np.uint8)).save2zarr(
        store, work_id=work_id
    )


@pytest.fixture
def published_store(tmp_path: Path) -> PublishedStore:
    """Publish a marker through the real helper + the real publisher."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    store = _write_store(output_dir)
    key, path = image_data_artifact(
        output_dir, _output_manager(output_dir), DATASET, STEM
    )
    record = publish_image_success(
        output_dir,
        work_id=WORK_ID,
        dataset=DATASET,
        relative_image_path=f"{STEM}.tif",
        image_stem=STEM,
        mode="full",
        attempt_id="a-1",
        lifecycle_epoch="e-1",
        artifacts={key: path},
    )
    return PublishedStore(output_dir=output_dir, store=store, record=record)


@pytest.fixture
def legacy_file_marker(tmp_path: Path) -> dict[str, object]:
    """A hand-written record whose file descriptor carries no ``kind``.

    **The "legacy" here is the DESCRIPTOR's shape, not the file's location.**
    A v1 descriptor omitted ``kind``, and an absent ``kind`` must still read
    as ``"file"`` -- that is the compatibility this fixture exists to
    exercise, and it is a property of `fenced_artifact_path`, which the
    current reader reaches through the record.

    So it plants at ``image_record_path`` with ``RECORD_VERSION``: pointing
    it at ``image_complete/`` would test nothing, because `valid_image_success`
    no longer looks there and would return ``False`` for the trivial reason
    rather than the interesting one. The fixture name is kept -- the subject
    really is the legacy descriptor shape.
    """
    output_dir = tmp_path / "out"
    artifact = output_dir / "results" / DATASET / "measurements" / "img.parquet"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"measurements")
    atomic_write_json(
        image_record_path(output_dir, DATASET, STEM),
        {
            "version": RECORD_VERSION,
            "work_id": WORK_ID,
            "dataset": DATASET,
            "relative_image_path": f"{STEM}.tif",
            "image_stem": STEM,
            "mode": "full",
            "attempt_id": "a-1",
            "lifecycle_epoch": "e-1",
            "artifacts": {
                "measurements": {
                    "path": artifact.relative_to(output_dir).as_posix(),
                    "size": artifact.stat().st_size,
                    "sha256": file_fingerprint(artifact).removeprefix(
                        "sha256:"
                    ),
                }
            },
            "completed_at": "2026-08-19T00:00:00.000+00:00",
        },
    )
    return {
        "output_dir": output_dir,
        "dataset": DATASET,
        "image_stem": STEM,
        "work_id": WORK_ID,
    }


# --- the version bump -----------------------------------------------------


def test_marker_version_is_bumped() -> None:
    """A v1 marker describes the RETAINED .h5, which still validates.

    ``keep_source=True`` is the default, so without the bump a v1 marker
    passes against a stale artifact while the store goes unverified -- a false
    ``complete``, not a spurious reprocess. See FLOW-23.
    """
    assert SUCCESS_MARKER_VERSION >= 2


def test_a_record_with_an_uninterpretable_version_is_rejected(
    published_store: PublishedStore,
) -> None:
    """The version check has to bite: a record this build cannot read fails.

    **Renamed and re-numbered at P3, and the number is the whole story.** It
    was `test_a_v1_marker_is_rejected`, planting `version: 1` against
    `SUCCESS_MARKER_VERSION == 2`. The publisher now writes a record, and
    `RECORD_VERSION == 1` -- so the old body plants the **current** version
    and the assertion inverts.

    That inversion fails loudly rather than passing green, which is the
    saving grace: `assert True is False`. The hazard is the *repair* -- the
    two-character `is False` -> `is True` that makes the red go away and
    leaves a test named "rejected" asserting acceptance. Nothing in a diff
    objects to it, least of all inside a sweep updating dozens of files.

    So the fix is not a new number but a stated property: **the invariant is
    "a version this build cannot interpret is refused", and `1` was only ever
    the stale value by accident of the marker constant having been bumped.**
    `RECORD_VERSION + 1` says that directly and cannot rot when the constant
    next moves.
    """
    marker = json.loads(published_store.record.read_text(encoding="utf-8"))
    marker["version"] = RECORD_VERSION + 1
    atomic_write_json(published_store.record, marker)
    assert (
        valid_image_success(
            published_store.output_dir,
            dataset=DATASET,
            image_stem=STEM,
            work_id=WORK_ID,
        )
        is False
    )


# --- the resolver ---------------------------------------------------------


def test_image_data_artifact_names_the_store_directory(tmp_path: Path) -> None:
    """The certified artifact is the store itself, not its root zarr.json."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    store = _write_store(output_dir)
    key, path = image_data_artifact(
        output_dir, _output_manager(output_dir), DATASET, STEM
    )
    assert key == "store"
    assert path == store
    assert path.is_dir()


def test_image_data_artifact_falls_back_to_the_hdf_for_a_legacy_tree(
    tmp_path: Path,
) -> None:
    """Legacy trees have no store; ``_migrate_legacy_success_evidence`` runs
    on exactly those, so the fallback stays reachable."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manager = _output_manager(output_dir)
    legacy = manager.get_output_path(DATASET, "hdf", STEM)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"legacy hdf")
    key, path = image_data_artifact(output_dir, manager, DATASET, STEM)
    assert (key, path) == ("hdf", legacy)


# --- publish / validate over a store --------------------------------------


def test_publishing_a_store_artifact_does_not_raise(
    published_store: PublishedStore,
) -> None:
    """_sha256 opens its argument as a file; on a directory that is fatal."""
    assert published_store.record.is_file()


def test_the_store_descriptor_keys_on_the_root_zarr_json(
    published_store: PublishedStore,
) -> None:
    """Kind-tagged, content-only, and relative -- all three at once."""
    marker = json.loads(published_store.record.read_text(encoding="utf-8"))
    descriptor = marker["artifacts"]["store"]
    assert descriptor["kind"] == "store"
    assert descriptor["path"] == published_store.store.relative_to(
        published_store.output_dir
    ).as_posix()
    assert descriptor["sha256"] == file_fingerprint(
        published_store.store / "zarr.json"
    )


def test_a_published_store_validates(published_store: PublishedStore) -> None:
    assert (
        valid_image_success(
            published_store.output_dir,
            dataset=DATASET,
            image_stem=STEM,
            work_id=WORK_ID,
        )
        is True
    )


def test_a_rewritten_store_invalidates_the_marker(
    published_store: PublishedStore,
) -> None:
    """Keying on the directory instead of zarr.json would miss this."""
    root = published_store.store / "zarr.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["attributes"]["phenotypic"]["work_id"] = "different"
    root.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        valid_image_success(
            published_store.output_dir,
            dataset=DATASET,
            image_stem=STEM,
            work_id=WORK_ID,
        )
        is False
    )


def test_a_relocated_output_tree_still_validates(
    published_store: PublishedStore, tmp_path: Path
) -> None:
    """Store descriptors must be relocatable, like every file descriptor.

    ``paths_fingerprint`` would fold the absolute path into the digest, so
    moving the tree -- or reaching it through a different symlink/automount,
    which on this cluster is routine -- would silently invalidate every marker
    and trigger a full re-finalization with no message saying why (FLOW-3).
    """
    moved = tmp_path / "relocated"
    shutil.copytree(published_store.output_dir, moved)
    assert (
        valid_image_success(
            moved, dataset=DATASET, image_stem=STEM, work_id=WORK_ID
        )
        is True
    )


def test_a_deleted_store_invalidates_the_marker(
    published_store: PublishedStore,
) -> None:
    shutil.rmtree(published_store.store)
    assert (
        valid_image_success(
            published_store.output_dir,
            dataset=DATASET,
            image_stem=STEM,
            work_id=WORK_ID,
        )
        is False
    )


def test_a_store_without_its_root_json_invalidates_the_marker(
    published_store: PublishedStore,
) -> None:
    """An interrupted re-promote leaves the directory but not a valid root."""
    (published_store.store / "zarr.json").unlink()
    assert (
        valid_image_success(
            published_store.output_dir,
            dataset=DATASET,
            image_stem=STEM,
            work_id=WORK_ID,
        )
        is False
    )


def test_a_store_replaced_by_a_regular_file_invalidates_the_marker(
    published_store: PublishedStore,
) -> None:
    """The store branch must not silently accept a same-named file."""
    shutil.rmtree(published_store.store)
    published_store.store.write_bytes(b"not a store")
    assert (
        valid_image_success(
            published_store.output_dir,
            dataset=DATASET,
            image_stem=STEM,
            work_id=WORK_ID,
        )
        is False
    )


# --- descriptor kind dispatch ---------------------------------------------


def test_a_file_descriptor_without_kind_still_validates(
    legacy_file_marker: dict[str, object],
) -> None:
    """Defaulting kind to 'file' keeps older markers parseable."""
    assert valid_image_success(**legacy_file_marker) is True


def test_a_file_descriptor_is_tagged_kind_file(tmp_path: Path) -> None:
    parquet = tmp_path / "results" / DATASET / "measurements" / "img.parquet"
    parquet.parent.mkdir(parents=True)
    parquet.write_bytes(b"measurements")
    marker = publish_image_success(
        tmp_path,
        work_id=WORK_ID,
        dataset=DATASET,
        relative_image_path=f"{STEM}.tif",
        image_stem=STEM,
        mode="full",
        attempt_id="a-1",
        lifecycle_epoch="e-1",
        artifacts={"measurements": parquet},
    )
    descriptor = json.loads(marker.read_text(encoding="utf-8"))["artifacts"][
        "measurements"
    ]
    assert descriptor["kind"] == "file"
    assert descriptor["size"] == parquet.stat().st_size


def test_an_unknown_descriptor_kind_is_rejected(
    legacy_file_marker: dict[str, object],
) -> None:
    """An unrecognized kind must fail closed, never fall through to 'file'.

    Tested over a descriptor that is otherwise a *valid file* descriptor:
    retagging a store descriptor would be caught by the ``is_file()`` guard in
    the file branch for the wrong reason, and the test would pass against a
    build that silently treats every unknown kind as a file.
    """
    marker_path = image_record_path(
        legacy_file_marker["output_dir"],  # type: ignore[arg-type]
        DATASET,
        STEM,
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert valid_image_success(**legacy_file_marker) is True
    marker["artifacts"]["measurements"]["kind"] = "something-else"
    atomic_write_json(marker_path, marker)
    assert valid_image_success(**legacy_file_marker) is False


# --- the shared descriptor loop in the migration bridge -------------------


def _state_requiring_markers(output_dir: Path) -> None:
    from datetime import datetime

    from phenotypic._cli._cli_state_management import save_processing_state
    from phenotypic._cli._cli_types import DatasetState, ProcessingState

    now = datetime.now()
    save_processing_state(
        ProcessingState(
            version="3.0.0",
            pipeline_path=output_dir / "pipeline.json",
            input_path=output_dir / "input",
            output_dir=output_dir,
            timestamp=now,
            execution_mode="local",
            last_updated=now,
            datasets={DATASET: DatasetState(completed={f"{STEM}.tif"})},
            config={
                "success_markers_required": True,
                "work_ids": {DATASET: {f"{STEM}.tif": WORK_ID}},
            },
        ),
        output_dir,
    )


#: The third deferred consumer, alongside `--mode recompile` (P4) and the two
#: GUI readers (P6). `refresh_success_markers_after_metadata_migration` reads
#: `image_completion_marker_path` at `_cli_completion.py:431`, which D1's clean
#: break stopped writing -- so `read_text` raises `FileNotFoundError`, `:436`
#: catches it as `OSError`, and EVERY image is skipped. The function returns 0
#: having examined nothing.
#:
#: **Deferred, not fixed, and the fix is not one line.** `:441` compares
#: `marker.get("version") != SUCCESS_MARKER_VERSION`, which is **2**, while a
#: record carries `RECORD_VERSION`, which is **1**. Repointing only the path
#: leaves every record failing the version check and `continue`-ing -- the same
#: silent skip through a different door. Path and constant move together, or
#: the repoint is cosmetic. That is REUSE-F10's work, deferred by ruling at the
#: P2 gate and owned by P7.
#:
#: Nothing user-facing is broken meanwhile: the function has **zero production
#: callers** (`grep -rn` finds the definition, one docstring reference in
#: `sdk_/_run_state.py:532`, and these tests). But the failure direction is the
#: dangerous one -- it returns 0 and raises nothing for a tree whose artifact
#: changed uncertified, which is a verdict improving without verification.
#:
#: **All THREE tests below carry this mark, and the third had to be made to
#: fail before it could.** Two failed loudly (`DID NOT RAISE`); the third
#: asserted `== 0` and stayed GREEN, because a bridge that examines nothing
#: returns 0 just as surely as one that examines a store and finds it
#: unchanged. It was given a spy as a positive control so it fails for the
#: real reason. A vacuous green between two reds reporting the same cause is
#: worse than a third red: it tells a reviewer the store handling is fine.
#:
#: So P7 gets three XPASSes at once, which is the point -- the repoint is not
#: done until all three go green together.
_BRIDGE_READS_THE_LEGACY_MARKER_UNTIL_P7 = pytest.mark.xfail(
    strict=True,
    reason=(
        "refresh_success_markers_after_metadata_migration reads "
        "image_completion_marker_path (_cli_completion.py:431), which D1 "
        "stopped writing, so every image is skipped at :436 and the guard "
        "never runs. REUSE-F10, deferred to P7; the path and "
        "SUCCESS_MARKER_VERSION must move together."
    ),
)


@_BRIDGE_READS_THE_LEGACY_MARKER_UNTIL_P7
def test_the_refresh_bridge_tolerates_a_store_descriptor(
    published_store: PublishedStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FLOW-31: the whole comparison block dispatches on kind, not just the
    ``is_file()`` guard -- ``_sha256`` on a store raises IsADirectoryError.

    **This assertion was green and vacuous, which made it the dangerous third
    of the trio.** Since D1 the bridge skips every image at
    ``_cli_completion.py:436`` and returns 0 without reaching the store branch
    at all, so ``== 0`` held for the trivial reason instead of the interesting
    one -- identically true of a bridge that dispatched on nothing. A green
    "the store branch works" sitting between two reds reporting the same cause
    tells a reviewer the store handling is fine.

    **The positive control is the spy below**, and it is what converts this
    from a comment into a test: a comment saying "this proves nothing" does
    not run. The spy fails today, XPASSes when P7 repoints the bridge, and is
    self-cleaning by the same mechanism as the two GUI tripwires.

    It returns ``True`` so the original claim is preserved exactly: the bridge
    must still reach the store branch, find it unchanged, and refresh nothing.
    """
    from phenotypic._cli import _cli_completion

    reached: list[Path] = []

    def _spy(artifact: Path, descriptor: object) -> bool:
        reached.append(artifact)
        return True

    monkeypatch.setattr(_cli_completion, "_store_artifact_matches", _spy)

    _state_requiring_markers(published_store.output_dir)
    refreshed = refresh_success_markers_after_metadata_migration(
        published_store.output_dir
    )

    assert reached, (
        "the bridge never reached the store branch, so `refreshed == 0` "
        "below would hold for the trivial reason -- it is skipping every "
        "image at _cli_completion.py:436 (REUSE-F10)"
    )
    assert refreshed == 0, (
        "a store descriptor is verified, never refreshed: metadata migration "
        "does not rewrite a store"
    )


@_BRIDGE_READS_THE_LEGACY_MARKER_UNTIL_P7
def test_the_refresh_bridge_rejects_an_uncertified_store_change(
    published_store: PublishedStore,
) -> None:
    """A store that changed with no receipt is an uncertified change, exactly
    as a file that changed with no receipt is."""
    _state_requiring_markers(published_store.output_dir)
    root = published_store.store / "zarr.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["attributes"]["phenotypic"]["work_id"] = "different"
    root.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Uncertified artifact change"):
        refresh_success_markers_after_metadata_migration(
            published_store.output_dir
        )


@_BRIDGE_READS_THE_LEGACY_MARKER_UNTIL_P7
def test_the_refresh_bridge_rejects_a_missing_store(
    published_store: PublishedStore,
) -> None:
    _state_requiring_markers(published_store.output_dir)
    shutil.rmtree(published_store.store)
    with pytest.raises(RuntimeError, match="artifact is missing"):
        refresh_success_markers_after_metadata_migration(
            published_store.output_dir
        )


# --- the source gate ------------------------------------------------------

#: Modules where a ``"hdf"`` key or comparison is correct and must survive
#: this phase. Everything outside this set is a per-image artifact
#: declaration this task ports to ``"store"``.
_KEEPS_AN_HDF_KEY = {
    # `TargetKind == "hdf"` -- legacy-tree metadata migration. Retained by
    # decision D10 (reachable for legacy trees, not dead); Task 6.4 records the
    # reasoning in this module's docstring. 7 lines.
    "sdk_/_metadata_migration.py",
    # OutputManager's legacy `save_layers` / `extensions` dict keys, their
    # docstring, and the `layer == "hdf"` extension dispatch -- the HDF writer
    # itself is kept until Phase 6 (Task 3.1). 4 lines; Phase 6 Task 6.3
    # removes them and this allowlist entry with them.
    "_cli/_cli_output_manager.py",
}


def _src_root() -> Path:
    return Path(__file__).resolve().parents[3] / "src" / "phenotypic"


def test_every_hdf_artifact_declaration_is_ported() -> None:
    """The per-image image-state artifact is never declared as ``"hdf"``.

    Scoped, not a bare zero-hit sweep: before this task ``"hdf":`` matched
    **12** lines under ``src/phenotypic`` and 11 of them were correct (7
    ``TargetKind``
    comparisons in ``sdk_/_metadata_migration.py`` and 4 ``save_layers`` /
    extension-dispatch lines in ``_cli/_cli_output_manager.py``). The one this
    task ports is ``phenotypicCLI.py``'s
    ``_migrate_legacy_success_evidence``; the other four publishers were
    routed through ``image_data_artifact`` by earlier clusters.
    ``gui/builder/_preview_cache.py`` is deliberately **not** allowlisted --
    Phase 2 Task 2.4 renamed it to ``"store"``, so a hit there is a
    regression.
    """
    src = _src_root()
    hits = [
        f"{p.relative_to(src)}:{n}"
        for p in src.rglob("*.py")
        if str(p.relative_to(src)) not in _KEEPS_AN_HDF_KEY
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r'"hdf"\s*:', line)
    ]
    assert hits == [], hits


def test_the_allowlist_itself_is_not_stale() -> None:
    """An allowlist that stops matching anything is a silent no-op."""
    src = _src_root()
    for rel in _KEEPS_AN_HDF_KEY:
        assert (src / rel).is_file(), rel
        assert re.search(
            r'"hdf"\s*:', (src / rel).read_text(encoding="utf-8")
        ), rel


# --- the live break: legacy success-evidence promotion --------------------
#
# `_migrate_legacy_success_evidence` mints markers for runs that have
# completion evidence but no marker, and its staged evidence test includes
# `stage3_completion_exists` -- a STORE-era signal. So it fires on a staged
# run interrupted between its Stage-3 marker and its success marker, and a
# hard-coded `"hdf"` there names a file the run never wrote. The publish then
# raises FileNotFoundError inside `resolve(strict=True)`, which the caller's
# `except OSError: continue` swallows -- so the symptom is not a crash but a
# silent refusal to promote, and the image is reprocessed from scratch.


def _legacy_migration_inputs(tmp_path: Path):
    """Build a real input tree, pipeline and config for the legacy migrator."""
    from PIL import Image as PILImage

    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_types import DatasetState, ExecutionConfig
    from phenotypic.detect import OtsuDetector

    input_root = tmp_path / "in"
    (input_root / DATASET).mkdir(parents=True)
    image_path = input_root / DATASET / f"{STEM}.tif"
    PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)

    pipeline_path = tmp_path / "pipe.json"
    pipeline_path.write_text(
        ImagePipeline(ops=[OtsuDetector()]).to_json(), encoding="utf-8"
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    config = ExecutionConfig(
        pipeline_json=pipeline_path,
        input_path=input_root,
        output_dir=output_dir,
        image_type="Image",
        nrows=None,
        ncols=None,
        bit_depth=None,
        n_jobs=1,
        slurm_args={},
        force_local=True,
        wait=False,
        ext=".tiff",
        overlay_alpha=0.5,
        include_dataset_column=False,
        dry_run=False,
        sample=None,
        resume=False,
        retry_failures=False,
        skip_validation=True,
        save_overlays=False,
        measure_only=False,
        process_only_layer=None,
    )
    return image_path, output_dir, config, DatasetState


def _run_legacy_migration(tmp_path: Path, *, with_store: bool):
    """Drive the production migrator over one image with legacy evidence."""
    from datetime import datetime

    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic._cli._cli_types import Dataset, ProcessingState
    from phenotypic.phenotypicCLI import _migrate_legacy_success_evidence

    image_path, output_dir, config, DatasetStateCls = _legacy_migration_inputs(
        tmp_path
    )
    manager = OutputManager.from_config(
        base_dir=output_dir, ext=".tiff", save_overlays=False
    )
    parquet = manager.get_output_path(DATASET, "measurements", STEM)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"measurements")
    if with_store:
        _write_store(output_dir, work_id=None)
    else:
        legacy = manager.get_output_path(DATASET, "hdf", STEM)
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(b"legacy hdf")

    now = datetime.now()
    state = ProcessingState(
        version="3.0.0",
        pipeline_path=config.pipeline_json,
        input_path=config.input_path,
        output_dir=output_dir,
        timestamp=now,
        execution_mode="local",
        last_updated=now,
        datasets={
            DATASET: DatasetStateCls(
                completed={f"{STEM}.tif"}, initial_images={f"{STEM}.tif"}
            )
        },
        config={},
    )
    datasets = [
        Dataset(
            name=DATASET,
            images=[image_path],
            input_dir=image_path.parent,
            output_dir=output_dir,
        )
    ]
    promoted = _migrate_legacy_success_evidence(
        state, config, datasets, output_dir
    )
    return promoted, output_dir


def test_legacy_migration_certifies_the_store_for_a_store_era_run(
    tmp_path: Path,
) -> None:
    """The one remaining raw ``"hdf"`` declaration: it names a file a
    store-era run never wrote, so the promotion is silently refused."""
    promoted, output_dir = _run_legacy_migration(tmp_path, with_store=True)
    assert promoted == 1
    marker = json.loads(
        image_record_path(output_dir, DATASET, STEM).read_text(
            encoding="utf-8"
        )
    )
    assert "hdf" not in marker["artifacts"]
    descriptor = marker["artifacts"]["store"]
    assert descriptor["kind"] == "store"
    assert descriptor["sha256"] == file_fingerprint(
        zarr_store_path(output_dir, DATASET, STEM) / "zarr.json"
    )


def test_legacy_migration_still_certifies_the_hdf_for_a_legacy_run(
    tmp_path: Path,
) -> None:
    """The ``"hdf"`` fallback stays reachable: a genuine legacy tree has an
    ``.h5`` and no store, and this function exists to serve exactly it."""
    promoted, output_dir = _run_legacy_migration(tmp_path, with_store=False)
    assert promoted == 1
    marker = json.loads(
        image_record_path(output_dir, DATASET, STEM).read_text(
            encoding="utf-8"
        )
    )
    assert marker["artifacts"]["hdf"]["kind"] == "file"
