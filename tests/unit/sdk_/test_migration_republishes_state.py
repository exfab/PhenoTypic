"""Migration must leave the run's published state valid, not just its pixels.

``--mode migrate`` converted images but never re-published the per-image
completion markers that gate them, and three mechanisms make that fatal:

1. ``valid_image_success`` rejects on **strict equality** against
   ``SUCCESS_MARKER_VERSION``, which Phase 3 bumped to 2. After migration
   every finished image in every legacy tree is unknown-to-complete.
2. Without the bump it is worse, not better: a v1 marker keeps validating
   against the retained ``.h5`` (``keep_source=True`` is the default),
   asserting completeness for an artifact the forward path no longer reads.
3. The one bridge that exists, ``refresh_success_markers_after_metadata_migration``,
   skips markers whose version differs before any descriptor is read.
"""

from __future__ import annotations

import json
from pathlib import Path

from phenotypic._cli._cli_completion import (
    SUCCESS_MARKER_VERSION,
    current_aggregate_is_current,
    valid_image_success,
)
from phenotypic.sdk_ import image_completion_marker_path
from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

from tests.unit.sdk_._migration_fixtures import LegacyRun


def _marker(run: LegacyRun, stem: str) -> dict:
    return json.loads(
        image_completion_marker_path(run.path, "ds", stem).read_text(
            encoding="utf-8"
        )
    )


def test_the_fixture_starts_INVALID(finished_legacy_run: LegacyRun) -> None:
    """Guard every test below from passing on a tree that was already fine."""
    for stem in finished_legacy_run.stems:
        assert (
            valid_image_success(
                finished_legacy_run.path,
                dataset="ds",
                image_stem=stem,
                work_id=finished_legacy_run.work_id_for(stem),
            )
            is False
        )


def test_every_image_still_validates_after_migration(
    finished_legacy_run: LegacyRun,
) -> None:
    migrate_run_hdf_to_zarr(finished_legacy_run.path)
    for stem in finished_legacy_run.stems:
        assert (
            valid_image_success(
                finished_legacy_run.path,
                dataset="ds",
                image_stem=stem,
                work_id=finished_legacy_run.work_id_for(stem),
            )
            is True
        ), stem


def test_the_aggregate_publication_survives_migration(
    finished_legacy_run: LegacyRun,
) -> None:
    """``aggregate_publication_is_valid`` is not a symbol in this codebase.

    ``current_aggregate_is_current`` is the predicate that compares the
    published ``source_set_digest`` against the current marker-authorized
    success set, which is what this is about.
    """
    migrate_run_hdf_to_zarr(finished_legacy_run.path)
    assert current_aggregate_is_current(finished_legacy_run.path) is True


def test_work_id_and_epoch_are_preserved(finished_legacy_run: LegacyRun) -> None:
    """Rewriting them would falsely re-attribute the result to the migration."""
    stem = finished_legacy_run.stems[0]
    before = _marker(finished_legacy_run, stem)
    migrate_run_hdf_to_zarr(finished_legacy_run.path)
    after = _marker(finished_legacy_run, stem)

    for key in ("work_id", "attempt_id", "lifecycle_epoch"):
        assert after[key] == before[key], key
    assert after["version"] == SUCCESS_MARKER_VERSION
    # The artifact key for a store is ``"store"`` -- that is what
    # ``image_data_artifact`` returns and what ``ARTIFACT_KIND_STORE`` spells.
    assert after["artifacts"]["store"]["kind"] == "store"


def test_republication_REPLACES_the_artifact_set(
    finished_legacy_run: LegacyRun,
) -> None:
    """MIG-22: adding a store descriptor beside the stale ``hdf`` one hides
    the defect entirely, because ``keep_source=True`` leaves the ``.h5`` on
    disk and the stale descriptor still validates.

    The whole mapping is asserted, not just one entry. ``measurements`` and
    ``overlay`` survive **under their literal keys** --
    ``_current_success_work_ids`` indexes ``artifacts["measurements"]`` by
    name -- but their descriptors are re-fingerprinted, because the bytes
    they describe are what the migration's metadata pass just rewrote.
    """
    stem = finished_legacy_run.stems[0]
    before = _marker(finished_legacy_run, stem)
    assert "hdf" in before["artifacts"]

    migrate_run_hdf_to_zarr(finished_legacy_run.path)

    after = _marker(finished_legacy_run, stem)
    assert set(after["artifacts"]) == {"measurements", "overlay", "store"}
    assert "hdf" not in after["artifacts"]
    assert after["artifacts"]["store"]["path"].endswith(".ome.zarr")


def test_a_markerless_tree_is_a_documented_no_op(
    markerless_legacy_run: Path,
) -> None:
    """MIG-23: a pre-markers archive is a likely migration subject.

    ``publish_aggregate_snapshot`` RAISES when state is missing or no marker
    is authorized, and resolves four deliverables paths with ``strict=True``.
    Aborting there would leave the stores written and the run reported failed.
    """
    report = migrate_run_hdf_to_zarr(markerless_legacy_run)
    assert report.converted > 0
    assert report.failed == ()


def test_republication_never_CREATES_a_marker(
    markerless_legacy_run: Path,
) -> None:
    """FLOW-37: a *missing* marker also "does not describe the store".

    Under the looser wording republication fired on every image of a
    pre-markers archive, where ``publish_image_success`` has no ``work_id``,
    ``attempt_id`` or ``lifecycle_epoch`` to be given -- and, unlike its three
    siblings, it does not short-circuit on ``success_markers_required``.
    """
    from phenotypic.sdk_ import progress_dir

    migrate_run_hdf_to_zarr(markerless_legacy_run)
    marker_root = progress_dir(markerless_legacy_run) / "image_complete"
    assert not marker_root.exists() or not list(marker_root.rglob("*.json"))


def test_republication_is_keyed_on_MARKER_state_not_conversion_state(
    finished_legacy_run: LegacyRun,
) -> None:
    """FLOW-22: trace an interruption.

    Migration promotes image X's store, then dies before rewriting X's
    marker. On resume X is SKIPPED, because its store already passes
    ``valid_staged_store``. If republication rode on "was converted this run"
    X's marker would stay v1 forever, and the next local run would reprocess
    it from source inputs a migrated archive may no longer have.
    """
    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_._hdf_to_zarr import _dataset_hdf_dir
    from phenotypic.sdk_._hdf_to_zarr import migrate_hdf_to_zarr

    tree = finished_legacy_run.path
    stem = finished_legacy_run.stems[0]

    # Convert one image WITHOUT touching its marker -- the interrupted state.
    migrate_hdf_to_zarr(
        _dataset_hdf_dir(tree, "ds") / f"{stem}.h5",
        zarr_store_path(tree, "ds", stem),
    )
    assert (
        valid_image_success(
            tree,
            dataset="ds",
            image_stem=stem,
            work_id=finished_legacy_run.work_id_for(stem),
        )
        is False
    )

    report = migrate_run_hdf_to_zarr(tree)

    assert report.skipped >= 1, "the interrupted image must be skipped"
    assert (
        valid_image_success(
            tree,
            dataset="ds",
            image_stem=stem,
            work_id=finished_legacy_run.work_id_for(stem),
        )
        is True
    ), "a skipped image's marker was never republished"


def test_a_migrated_run_does_no_work_on_the_next_full_run(
    finished_legacy_run: LegacyRun,
) -> None:
    """The end-to-end consequence: migration must not cause reprocessing."""
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli
    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_.ngff_ import STORE_ROOT_JSON

    tree = finished_legacy_run.path
    migrate_run_hdf_to_zarr(tree)
    roots = {
        stem: (zarr_store_path(tree, "ds", stem) / STORE_ROOT_JSON).read_bytes()
        for stem in finished_legacy_run.stems
    }

    parquets = {
        stem: (tree / "results" / "ds" / "measurements" / f"{stem}.parquet").read_bytes()
        for stem in finished_legacy_run.stems
    }

    result = CliRunner().invoke(phenotypic_cli, finished_legacy_run.full_run_args())

    assert result.exit_code == 0, result.output
    # Asserted on BYTES, not on a message. The run's wording has changed
    # before, and a string check that happens to match is satisfied by a run
    # that reprocessed every image -- which is the thing being ruled out.
    for stem, root_bytes in roots.items():
        assert (
            zarr_store_path(tree, "ds", stem) / STORE_ROOT_JSON
        ).read_bytes() == root_bytes, f"{stem} store was re-promoted"
    for stem, parquet_bytes in parquets.items():
        assert (
            tree / "results" / "ds" / "measurements" / f"{stem}.parquet"
        ).read_bytes() == parquet_bytes, f"{stem} was re-measured"


def test_stage3_markers_need_no_work(finished_legacy_run: LegacyRun) -> None:
    """``migrate_legacy_stage3_markers`` regenerates them from PARQUET
    presence, not from the image artifact, so they self-heal on the next run.
    """
    from phenotypic._cli._cli_staged_resume import (
        StagedResumeItem,
        StagedResumePlan,
        migrate_legacy_stage3_markers,
        stage3_completion_exists,
    )

    tree = finished_legacy_run.path
    migrate_run_hdf_to_zarr(tree)
    for stem in finished_legacy_run.stems:
        assert not stage3_completion_exists(tree, "ds", stem)

    plan = StagedResumePlan(
        datasets=[],
        items=tuple(
            StagedResumeItem("ds", Path(f"{stem}.png"), "complete")
            for stem in finished_legacy_run.stems
        ),
        initial_stage="complete",
    )
    assert migrate_legacy_stage3_markers(tree, plan) == len(
        finished_legacy_run.stems
    )
    for stem in finished_legacy_run.stems:
        assert stage3_completion_exists(tree, "ds", stem)
