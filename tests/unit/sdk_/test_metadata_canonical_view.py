"""Migration derives a canonical view; it never rewrites the snapshot.

The in-place rewrite this task originally specified is **withdrawn** (user
ruling, ledger FLOW-4 / PRE-D9). It could not have worked: ``metadata_sha256``
is recomputed from the file on every run rather than read back from state, so
the moment migration rewrote ``deliverables/metadata.csv`` the next run's
``expected_finalization`` diverged from the published
``finalization_input_digest`` and the whole tree re-finalized. The three
negative assertions below are regression guards against it creeping back.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tests.unit.sdk_._migration_fixtures import LegacyRun


def test_the_snapshot_is_never_rewritten(legacy_run: Path) -> None:
    """flat-metadata decision #7, unnarrowed."""
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    source = legacy_run / "deliverables" / "metadata.csv"
    before = source.read_bytes()
    migrate_run_hdf_to_zarr(legacy_run)
    assert source.read_bytes() == before


def test_no_original_copy_is_created(legacy_run: Path) -> None:
    """metadata.original.csv was an artifact of the withdrawn rewrite."""
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    migrate_run_hdf_to_zarr(legacy_run)
    assert not (legacy_run / "deliverables" / "metadata.original.csv").exists()


def test_a_canonical_view_is_emitted_beside_it(legacy_run: Path) -> None:
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    migrate_run_hdf_to_zarr(legacy_run)
    view = legacy_run / "deliverables" / "metadata.canonical.csv"
    assert view.is_file()
    with view.open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert all(column.startswith("Metadata_") for column in header if column), header


def test_the_view_actually_canonicalizes_a_legacy_header(legacy_run: Path) -> None:
    """Not merely "every column is prefixed" -- the LEGACY spelling is gone.

    A view that copied the snapshot byte for byte would satisfy the prefix
    check on a snapshot whose columns already happen to be prefixed. The
    fixture's ``MetadataGenetic_Strain`` is prefixed AND legacy, so only a real
    canonicalization turns it into ``Metadata_Strain``.
    """
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    migrate_run_hdf_to_zarr(legacy_run)
    with (legacy_run / "deliverables" / "metadata.canonical.csv").open(
        encoding="utf-8"
    ) as handle:
        header = next(csv.reader(handle))
    assert "MetadataGenetic_Strain" not in header, header
    assert "Metadata_Strain" in header, header


def test_the_view_preserves_every_row(legacy_run: Path) -> None:
    """A canonical *view* renames columns; it must not drop or reorder rows."""
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    migrate_run_hdf_to_zarr(legacy_run)
    deliverables = legacy_run / "deliverables"
    with (deliverables / "metadata.csv").open(encoding="utf-8") as handle:
        source_rows = list(csv.reader(handle))[1:]
    with (deliverables / "metadata.canonical.csv").open(encoding="utf-8") as handle:
        view_rows = list(csv.reader(handle))[1:]
    assert view_rows == source_rows


def test_no_view_is_emitted_without_a_snapshot(legacy_run: Path) -> None:
    """Skip silently when there is no metadata.csv to derive from."""
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    (legacy_run / "deliverables" / "metadata.csv").unlink()
    migrate_run_hdf_to_zarr(legacy_run)
    assert not (legacy_run / "deliverables" / "metadata.canonical.csv").exists()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "`_hdf_to_zarr._republish_image_marker` rewrites the LEGACY marker "
        "(`_hdf_to_zarr.py:614,647`) while `valid_image_success` now reads "
        "the record, so after migration no image validates and the aggregate "
        "digest describes an empty success set. The republisher must write a "
        "record with provenance='migrated' -- P7's U-10 item, which "
        "`publish_image_record` already supports."
    ),
)
def test_the_aggregate_publication_survives_migration(
    finished_legacy_run: LegacyRun,
) -> None:
    """The test that could not pass under the rewrite -- now it can.

    It needs Task 5.6 for the marker half: ``source_set_digest`` is computed
    from ``valid_image_success``, so without the republication it describes a
    marker set that no longer validates.

    ``aggregate_publication_is_valid`` does not exist in this codebase;
    ``current_aggregate_is_current`` is the predicate that compares the
    published ``source_set_digest`` against the current marker-authorized
    success set, which is what this is about.

    **The fourth deferred consumer, and the one my sweep classified wrong.**
    I read ``test_migration_republishes_state.py`` as "no change -- the
    migrator rewrites the legacy marker in place", which describes what the
    code does and was never a reason to leave it. After D1 a migrator that
    republishes legacy markers republishes something nothing reads: the
    republication test stays green because it asserts the legacy marker was
    written, which is still true and no longer means anything.

    **This test is the one that noticed**, because it asks the question one
    layer up -- not "was a marker written?" but "does the aggregate still
    validate against the success set?" A description of behaviour passed for
    a justification; only a test asking about the *consequence* caught it.
    """
    from phenotypic._cli._cli_completion import current_aggregate_is_current
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    migrate_run_hdf_to_zarr(finished_legacy_run.path)
    assert current_aggregate_is_current(finished_legacy_run.path) is True


def test_the_view_sits_beside_the_snapshot(legacy_run: Path) -> None:
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    migrate_run_hdf_to_zarr(legacy_run)
    deliverables = legacy_run / "deliverables"
    assert (deliverables / "metadata.canonical.csv").is_file()
    assert (deliverables / "metadata.csv").is_file()  # snapshot untouched
