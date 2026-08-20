"""Legacy-run builders shared by the ``--mode migrate`` suites (Phase 5).

Every fixture here starts from a **real** completed run: the CLI is invoked
once per session on two tiny synthetic plates, producing genuine per-image
completion markers, work ids, a lifecycle epoch and an aggregate publication.
That tree is then *demoted* to the legacy shape -- each OME-Zarr store is
rewritten as a per-image ``.h5`` and the markers re-minted over it -- so the
fixtures carry evidence that was actually published rather than hand-modelled.

Hand-writing the markers instead would defeat the tests that consume them:
``valid_image_success`` and ``current_aggregate_is_current`` are exactly what
Task 5.6 exists to keep true, and a fixture whose markers were invented cannot
show that migration preserved anything.

:class:`LegacyRun` is deliberately **not** a ``Path``. Annotate it as
``LegacyRun``; ``from __future__ import annotations`` hides a wrong annotation
from the runtime but not from mypy.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image as PILImage

from phenotypic._cli._cli_completion import _artifact_descriptor
from phenotypic._cli._cli_state_management import (
    load_processing_state,
    save_processing_state,
)
from phenotypic.sdk_ import (
    atomic_write_json,
    dataset_hdf_dir,
    dataset_zarr_dir,
    deliverables_dir,
    image_completion_marker_path,
    load_image_from_store,
    metadata_csv_deliverable_path,
    zarr_store_path,
)
from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes

DATASET = "ds"

#: A legacy per-topic metadata CSV. Both columns are metadata-family, so the
#: canonical view Task 5.2 derives is all-``Metadata_`` by construction, and
#: ``MetadataGenetic_Strain`` is a spelling ``ensure_metadata_prefix``
#: genuinely resolves (to ``Metadata_Strain``).
LEGACY_METADATA_CSV = (
    "MetadataImage_ImageFile,MetadataGenetic_Strain\n"
    "img.png,BY4741\n"
    "img2.png,BY4742\n"
)


@dataclass(frozen=True)
class LegacyRun:
    """A completed legacy run and the handles its tests need."""

    path: Path
    work_id: str
    stems: tuple[str, ...]
    pipeline_json: Path
    input_dir: Path

    def full_run_args(self) -> list[str]:
        """Return the CLI argv that re-runs this run against its own tree."""
        return [
            "--pipeline",
            str(self.pipeline_json),
            "--input",
            str(self.input_dir),
            "-o",
            str(self.path),
            "--njobs",
            "1",
            "--skip-validation",
            "--force-local",
        ]


# ---------------------------------------------------------------------------
# The one real run
# ---------------------------------------------------------------------------


def write_synthetic_plate(target: Path) -> None:
    """Write a 128x128 plate with four discs -- enough to detect and measure."""
    array = np.zeros((128, 128, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:128, :128]
    for centre_y, centre_x in ((30, 30), (30, 90), (90, 30), (90, 90)):
        array[(yy - centre_y) ** 2 + (xx - centre_x) ** 2 <= 12**2] = 220
    PILImage.fromarray(array).save(target)


def build_completed_run(workspace: Path, stems: tuple[str, ...]) -> Path:
    """Run the real CLI once and return its output root.

    Args:
        workspace: Directory to build the input tree, pipeline and output in.
        stems: Image stems to render into the single dataset ``ds``.

    Returns:
        The run's output root.

    Raises:
        AssertionError: If the CLI run did not succeed.
    """
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli
    from phenotypic.prefab import RoundPeaksPipeline

    input_dir = workspace / DATASET
    input_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        write_synthetic_plate(input_dir / f"{stem}.png")

    pipeline_json = workspace / "pipeline.json"
    pipeline_json.write_text(
        RoundPeaksPipeline(
            blur_sigma=1,
            detector_thresh_method="otsu",
            detector_subtract_background=False,
            detector_remove_noise=False,
        ).to_json(),
        encoding="utf-8",
    )

    output_dir = workspace / "out"
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(pipeline_json),
            "--input",
            str(input_dir),
            "-o",
            str(output_dir),
            "--njobs",
            "1",
            "--skip-validation",
            "--force-local",
        ],
    )
    assert result.exit_code == 0, (
        f"fixture run failed (exit {result.exit_code}):\n{result.output}"
    )
    return output_dir


# ---------------------------------------------------------------------------
# Demotion: store -> .h5
# ---------------------------------------------------------------------------


def demote_store_to_hdf(output_dir: Path, dataset: str, stem: str) -> Path:
    """Rewrite one image's store as a legacy ``.h5`` and delete the store.

    The store's ``work_id`` is carried onto the ``.h5``'s root
    ``phenotypic_work_id`` attribute -- the post-write patch the CLI used to
    apply, and the field FLOW-1 is about.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        stem: Image stem.

    Returns:
        The written ``.h5`` path.
    """
    store = zarr_store_path(output_dir, dataset, stem)
    work_id = read_phenotypic_attributes(store).get(PhenotypicAttr.WORK_ID)
    image = load_image_from_store(store)

    hdf_dir = dataset_hdf_dir(output_dir, dataset)
    hdf_dir.mkdir(parents=True, exist_ok=True)
    hdf_path = hdf_dir / f"{stem}.h5"
    image.save2hdf5(hdf_path)
    if work_id is not None:
        with h5py.File(hdf_path, mode="a") as handle:
            handle.attrs["phenotypic_work_id"] = work_id

    shutil.rmtree(store)
    return hdf_path


#: A legacy per-topic column injected into every per-dataset measurement
#: parquet, so a demoted tree is legacy in its **tables** as well as in its
#: image format.
#:
#: Without it the fixture's non-image targets are already canonical, pass 1
#: short-circuits to ``_compatible_result`` before writing a receipt, and
#: nothing about pass 1 is exercised at all -- verified: four separate
#: mutations of the pass-1 scope, the receipt's ``kinds``, its validation, and
#: its dry-run guard all survived the suite until this existed.
LEGACY_MEASUREMENT_COLUMN = "MetadataGenetic_Strain"
LEGACY_MEASUREMENT_VALUE = "BY4741"


def make_measurement_headers_legacy(output_dir: Path, dataset: str) -> int:
    """Add a legacy per-topic column to every per-image measurement parquet.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.

    Returns:
        How many parquets were rewritten.
    """
    import polars as pl

    from phenotypic.sdk_ import dataset_measurements_dir

    rewritten = 0
    measurements = dataset_measurements_dir(output_dir, dataset)
    if not measurements.is_dir():
        return 0
    for parquet in sorted(measurements.glob("*.parquet")):
        if parquet.name.startswith(("_", ".")):
            continue
        frame = pl.read_parquet(parquet)
        frame.with_columns(
            pl.lit(LEGACY_MEASUREMENT_VALUE).alias(LEGACY_MEASUREMENT_COLUMN)
        ).write_parquet(parquet)
        rewritten += 1
    return rewritten


#: The marker version a genuine legacy tree carries. Phase 3 bumped
#: ``SUCCESS_MARKER_VERSION`` to 2 **and** added the ``kind`` tag; a legacy
#: marker predates both.
LEGACY_SUCCESS_MARKER_VERSION = 1


def repoint_marker_at_hdf(output_dir: Path, dataset: str, stem: str) -> Path:
    """Rewrite a marker into the shape a genuine legacy tree carries.

    Three changes, and each one matters:

    * ``version`` drops to 1. ``valid_image_success`` rejects on **strict
      equality** against ``SUCCESS_MARKER_VERSION``, which Phase 3 bumped to
      2, so every finished image in every legacy tree reads as
      unknown-to-complete until Task 5.6 re-publishes it.
    * The ``store`` descriptor becomes an ``hdf`` one.
    * That descriptor carries **no** ``kind`` key -- the tag arrived with the
      bump, so a v1 descriptor cannot have one.

    A fixture that instead minted a *version 2* marker over an ``.h5`` would
    describe a state that never existed, and -- because ``keep_source=True``
    leaves the ``.h5`` in place -- would keep validating straight through a
    migration, quietly certifying that Task 5.6 was unnecessary.

    Everything else -- ``work_id``, ``attempt_id``, ``lifecycle_epoch``,
    ``completed_at``, and the ``measurements`` and ``overlay`` descriptors --
    is preserved verbatim, so the marker stays the one the real run published.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        stem: Image stem.

    Returns:
        The marker path.
    """
    marker_path = image_completion_marker_path(output_dir, dataset, stem)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["version"] = LEGACY_SUCCESS_MARKER_VERSION
    artifacts = marker["artifacts"]
    artifacts.pop("store", None)
    output_root = Path(output_dir).resolve()

    # Re-fingerprint every surviving descriptor over the bytes as they are
    # NOW. The legacy headers were injected into the measurement parquets
    # before this ran, and a marker in a real legacy tree bound the bytes that
    # existed when it was published -- so pass 1's rewrite of those parquets
    # genuinely invalidates it. Preserving the pre-injection digest instead
    # would make the marker stale for the wrong reason and hide MIG-15.
    for name in list(artifacts):
        relative = artifacts[name]["path"]
        resolved = (output_root / relative).resolve(strict=True)
        descriptor = _artifact_descriptor(
            resolved, resolved.relative_to(output_root)
        )
        descriptor.pop("kind", None)
        artifacts[name] = descriptor

    hdf_path = dataset_hdf_dir(output_dir, dataset) / f"{stem}.h5"
    resolved = hdf_path.resolve(strict=True)
    descriptor = _artifact_descriptor(resolved, resolved.relative_to(output_root))
    descriptor.pop("kind", None)
    artifacts["hdf"] = descriptor
    atomic_write_json(marker_path, marker)
    return marker_path


def demote_run_to_hdf(output_dir: Path, *, keep_markers: bool) -> None:
    """Turn a store-backed completed run into a legacy ``.h5`` archive.

    Args:
        output_dir: Run output root, modified in place.
        keep_markers: Keep (and repoint) the per-image markers and republish
            the aggregate. ``False`` strips every completion marker, which is
            the ``legacy_run`` shape.
    """
    for stem in sorted(run_stems(output_dir)):
        demote_store_to_hdf(output_dir, DATASET, stem)
    zarr_dir = dataset_zarr_dir(output_dir, DATASET)
    if zarr_dir.is_dir() and not any(zarr_dir.iterdir()):
        zarr_dir.rmdir()

    make_measurement_headers_legacy(output_dir, DATASET)

    if keep_markers:
        # The aggregate the real run published stands as-is. Re-publishing it
        # here would fail anyway (``publish_aggregate_snapshot`` refuses when
        # no marker is currently valid, and the v1 markers below are not) --
        # and a real legacy tree's aggregate is likewise one an older build
        # published while its own markers were current.
        for stem in sorted(run_stems(output_dir)):
            repoint_marker_at_hdf(output_dir, DATASET, stem)
    else:
        strip_completion_evidence(output_dir)

    metadata_csv_deliverable_path(output_dir).write_text(
        LEGACY_METADATA_CSV, encoding="utf-8"
    )


def run_stems(output_dir: Path) -> tuple[str, ...]:
    """Return the run's image stems, read from its own state."""
    state = load_processing_state(output_dir)
    assert state is not None, f"no processing state under {output_dir}"
    images = state.config.get("work_ids", {}).get(DATASET, {})
    return tuple(sorted(Path(name).stem for name in images))


def run_work_id(output_dir: Path, stem: str) -> str:
    """Return the work id the run assigned to *stem*."""
    state = load_processing_state(output_dir)
    assert state is not None
    for name, work_id in state.config["work_ids"][DATASET].items():
        if Path(name).stem == stem:
            return str(work_id)
    raise KeyError(stem)


def strip_completion_evidence(output_dir: Path) -> None:
    """Delete every per-image marker and the aggregate/run publications."""
    from phenotypic.sdk_ import (
        aggregate_publication_marker_path,
        progress_dir,
        run_completion_marker_path,
    )

    marker_root = progress_dir(output_dir) / "image_complete"
    if marker_root.is_dir():
        shutil.rmtree(marker_root)
    for path in (
        aggregate_publication_marker_path(output_dir),
        run_completion_marker_path(output_dir),
    ):
        path.unlink(missing_ok=True)


def make_markerless(output_dir: Path) -> None:
    """Turn a run into a pre-markers archive.

    ``success_markers_required`` goes falsey and the aggregate publication is
    removed, which is the state MIG-23 says must be a documented no-op rather
    than a ``RuntimeError`` out of ``publish_aggregate_snapshot``.
    """
    strip_completion_evidence(output_dir)
    state = load_processing_state(output_dir)
    assert state is not None
    state.config["success_markers_required"] = False
    save_processing_state(state, output_dir)


def deliverables_metadata_csv(output_dir: Path) -> Path:
    """Return the run's immutable metadata snapshot path."""
    return deliverables_dir(output_dir) / "metadata.csv"
