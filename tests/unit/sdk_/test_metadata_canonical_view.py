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
        "Needs Task 5.6: migration must re-publish the per-image markers and "
        "the aggregate. Until it does, source_set_digest describes a marker "
        "set that no longer validates."
    ),
)
def test_the_aggregate_publication_survives_migration(
    finished_legacy_run: LegacyRun,
) -> None:
    """The test that could not pass under the rewrite -- now it can.

    ``aggregate_publication_is_valid`` does not exist in this codebase;
    ``current_aggregate_is_current`` is the predicate that compares the
    published ``source_set_digest`` against the current marker-authorized
    success set, which is what this is about.
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
