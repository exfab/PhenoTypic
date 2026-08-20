"""Legacy-run fixtures for the ``--mode migrate`` suites (Phase 5).

Deliberately **not** in the repo-root ``conftest.py``: these six fixtures are
migration-specific and would otherwise be global to the whole suite.

The expensive part -- one real CLI run -- happens once per session and is
copied per test. A copied output tree still continues correctly (the run's
``--input`` and ``--pipeline`` live outside it and are unchanged), so
``full_run_args()`` works against the copy.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.unit.sdk_._migration_fixtures import (
    DATASET,
    LegacyRun,
    build_completed_run,
    demote_run_to_hdf,
    make_markerless,
    run_stems,
    run_work_id,
)


@pytest.fixture(scope="session")
def _completed_run_one(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real completed run over a single image, built once per session."""
    workspace = tmp_path_factory.mktemp("legacy_one")
    return build_completed_run(workspace, ("img",))


@pytest.fixture(scope="session")
def _completed_run_two(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One real completed run over two images in the SAME dataset.

    Both in one dataset on purpose: the half-migrated state Task 5.7 exists to
    catch has converted and unconverted images side by side, which a
    dataset-level predicate would miss.
    """
    workspace = tmp_path_factory.mktemp("legacy_two")
    return build_completed_run(workspace, ("img", "img2"))


def _copy_run(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


@pytest.fixture
def legacy_run(_completed_run_one: Path, tmp_path: Path) -> Path:
    """A one-image ``.h5`` archive with measurements and a metadata snapshot.

    No completion markers -- this is the plain conversion subject.
    """
    output_dir = _copy_run(_completed_run_one, tmp_path / "legacy")
    demote_run_to_hdf(output_dir, keep_markers=False)
    return output_dir


@pytest.fixture
def finished_legacy_run(_completed_run_two: Path, tmp_path: Path) -> LegacyRun:
    """A COMPLETED legacy run: real markers, real work ids, real aggregate."""
    workspace = _completed_run_two.parent
    output_dir = _copy_run(_completed_run_two, tmp_path / "finished")
    demote_run_to_hdf(output_dir, keep_markers=True)
    stems = run_stems(output_dir)
    return LegacyRun(
        path=output_dir,
        work_ids={stem: run_work_id(output_dir, stem) for stem in stems},
        stems=stems,
        pipeline_json=workspace / "pipeline.json",
        input_dir=workspace / DATASET,
    )


@pytest.fixture
def markerless_legacy_run(_completed_run_one: Path, tmp_path: Path) -> Path:
    """A pre-markers archive: ``success_markers_required`` falsey, no aggregate."""
    output_dir = _copy_run(_completed_run_one, tmp_path / "markerless")
    demote_run_to_hdf(output_dir, keep_markers=False)
    make_markerless(output_dir)
    return output_dir


@pytest.fixture
def half_migrated_run(_completed_run_two: Path, tmp_path: Path) -> Path:
    """Two images in one dataset, exactly ONE of them converted.

    The expected state after any interruption, since migration is resumable.
    """
    from phenotypic.sdk_._hdf_to_zarr import migrate_hdf_to_zarr
    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_._hdf_to_zarr import _dataset_hdf_dir

    output_dir = _copy_run(_completed_run_two, tmp_path / "half")
    demote_run_to_hdf(output_dir, keep_markers=False)
    first = sorted(run_stems(output_dir))[0]
    migrate_hdf_to_zarr(
        _dataset_hdf_dir(output_dir, DATASET) / f"{first}.h5",
        zarr_store_path(output_dir, DATASET, first),
    )
    return output_dir


@pytest.fixture
def migrated_run(_completed_run_two: Path, tmp_path: Path) -> Path:
    """A fully converted tree with its ``.h5`` sources retained."""
    from phenotypic.sdk_._hdf_to_zarr import migrate_run_hdf_to_zarr

    output_dir = _copy_run(_completed_run_two, tmp_path / "migrated")
    demote_run_to_hdf(output_dir, keep_markers=False)
    migrate_run_hdf_to_zarr(output_dir)
    return output_dir
