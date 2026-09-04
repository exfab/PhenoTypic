"""Shared helpers for seeding fake CLI output directories in tests.

A PhenoTypic CLI run writes its user-facing deliverables under
``<output>/deliverables/`` (``master_measurements.*``, the
``measurements.*`` mirror, ``analysis.*``, per-feature splits,
``pipeline.json``, ``README.md``, ``dashboard.html``,
``processing_report.html``). Per-image artifacts
(``results/<ds>/...``), QC outputs (``qc/``), progress sidecars
(``progress/``), and run state (``processing_state.json``) stay at the
output root.

Tests that synthesize one of these layouts should route through these
helpers (which compose from the production path-builders in
``phenotypic.sdk_``) so the on-disk layout auto-tracks any future
relocation of the deliverables folder. Never hard-code
``tmp_path / "master_measurements.parquet"`` — it will silently drift.

The helpers accept either a polars or a pandas frame for the master.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from phenotypic.sdk_ import (
    deliverables_dir,
    master_measurements_csv_path,
    master_measurements_parquet_path,
    measurements_csv_path,
    measurements_parquet_path,
    pipeline_json_path,
    resolve_manifest_json_path,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl


def _ensure_deliverables(root: Path) -> Path:
    """Create ``<root>/deliverables/`` (the folder won't exist yet) and return it."""
    deliv = deliverables_dir(root)
    deliv.mkdir(parents=True, exist_ok=True)
    return deliv


def _to_polars(df: Any) -> "pl.DataFrame":
    """Coerce a pandas or polars frame to polars (for ``.write_parquet``/``.write_csv``)."""
    import polars as pl

    if isinstance(df, pl.DataFrame):
        return df
    # Assume a pandas frame (or anything polars can ingest from pandas).
    return pl.from_pandas(df)


def write_master(root: Path, master: Any, *, csv: bool = True, parquet: bool = True) -> Path:
    """Write ``master_measurements.{csv,parquet}`` under ``<root>/deliverables/``.

    Args:
        root: Run output root (``tmp_path``).
        master: A polars or pandas DataFrame.
        csv: Write the CSV master archive.
        parquet: Write the parquet master archive.

    Returns:
        The deliverables directory.
    """
    deliv = _ensure_deliverables(root)
    frame = _to_polars(master)
    if csv:
        frame.write_csv(master_measurements_csv_path(root))
    if parquet:
        frame.write_parquet(master_measurements_parquet_path(root))
    return deliv


def write_measurements_mirror(
    root: Path, df: Any, *, csv: bool = True, parquet: bool = True
) -> Path:
    """Write the post-applied ``measurements.{csv,parquet}`` mirror.

    This is the frame the GUI viewer reads/curates (see CLAUDE.md
    "Master vs. mirror outputs"). Lives under ``<root>/deliverables/``.
    """
    deliv = _ensure_deliverables(root)
    frame = _to_polars(df)
    if csv:
        frame.write_csv(measurements_csv_path(root))
    if parquet:
        frame.write_parquet(measurements_parquet_path(root))
    return deliv


def write_pipeline_json(root: Path, pipeline: Any) -> Path:
    """Serialize ``pipeline`` to ``<root>/deliverables/pipeline.json``.

    ``pipeline`` may be an ``ImagePipeline`` (uses ``.to_json()``) or a raw
    JSON string already produced by the caller.
    """
    _ensure_deliverables(root)
    path = pipeline_json_path(root)
    text = pipeline if isinstance(pipeline, str) else (pipeline.to_json() or "")
    path.write_text(text, encoding="utf-8")
    return path


def write_dashboard(root: Path, *, execution_mode: str = "local") -> Path:
    """Generate a real ``<root>/deliverables/dashboard.html`` via the producer."""
    from phenotypic._cli._dashboard._generator import generate_dashboard

    _ensure_deliverables(root)
    generate_dashboard(root, execution_mode=execution_mode)
    from phenotypic.sdk_ import dashboard_html_path

    return dashboard_html_path(root)


def write_complete_manifest(root: Path, *, total_images: int) -> Path:
    """Publish coherent terminal manifest evidence for a synthetic output.

    Args:
        root: Synthetic CLI output root.
        total_images: Non-negative completed image count.

    Returns:
        The production-resolved manifest path.

    Raises:
        ValueError: If ``total_images`` is negative.
    """
    if total_images < 0:
        raise ValueError("total_images must be non-negative")
    manifest = resolve_manifest_json_path(root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "is_complete": True,
                "completed": total_images,
                "failed": 0,
                "total_images": total_images,
            }
        ),
        encoding="utf-8",
    )
    return manifest


#: The one dataset and the two image stems ``build_complete_run`` writes.
#: Named rather than repeated so a test can address the second image without
#: knowing how the builder spelled it.
FIXTURE_DATASET = "plate"
FIXTURE_STEMS = ("a", "b")


def _fixture_measurement_table() -> Any:
    """One embedded per-object measurement payload for a fixture store.

    Its presence is what writes ``attributes.phenotypic.tables``, which is the
    discriminator between "Stage 1 done, Stage 3 pending" and "finished, and
    the detector found nothing". A fixture store without it is a store no
    forward path produces.
    """
    import pandas as pd

    from phenotypic.schema import OBJECT
    from phenotypic.sdk_._measurement_tables import (
        PreparedEmbeddedMeasurementTable,
    )

    return PreparedEmbeddedMeasurementTable(
        frame=pd.DataFrame({str(OBJECT.LABEL): [1], "Size_Area": [16.0]}),
        measurement_columns=("Size_Area",),
        join_status="not_requested",
        join_keys=(),
        metadata_snapshot_sha256="0" * 64,
    )


def _promote_minimal_store(
    root: Path, *, dataset: str, stem: str, work_id: str
) -> Path:
    """Promote one real 8x8 OME-Zarr store with an embedded table.

    Through :meth:`Image.save2zarr`, so the store is built the way the writer
    builds one -- arrays, then ``OME/zarr.json``, then the root ``zarr.json``
    last. Hand-assembling the directory would produce a tree that keeps
    validating after the store format changes.

    The table is passed to the promote rather than written afterwards, which
    is decision **D-A**: per-store tables are written in the store's original
    ``.part``, before the root, so no artifact carrying a content proof is
    ever mutated.
    """
    import numpy as np

    from phenotypic import Image
    from phenotypic.sdk_ import zarr_store_path

    store = zarr_store_path(root, dataset, stem)
    store.parent.mkdir(parents=True, exist_ok=True)
    return Image(np.zeros((8, 8, 3), dtype=np.uint8)).save2zarr(
        store,
        work_id=work_id,
        measurement_table=_fixture_measurement_table(),
    )


def _write_overlay(root: Path, *, dataset: str, stem: str) -> Path:
    """Write one overlay PNG at the production-resolved path.

    A real PNG, not a stub: the overlay is marker-bound, so its bytes are what
    ``valid_image_success`` digests, and a test that tampers with it needs a
    file whose content actually changed.
    """
    import numpy as np
    from PIL import Image as PILImage

    from phenotypic.sdk_ import dataset_overlays_dir

    overlays = dataset_overlays_dir(root, dataset)
    overlays.mkdir(parents=True, exist_ok=True)
    path = overlays / f"{stem}.png"
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:4, :4, 0] = 255
    PILImage.fromarray(pixels).save(path)
    return path


def _fixture_measurements_frame(work_ids: dict[str, dict[str, str]]) -> Any:
    """Return the master/mirror frame matching ``work_ids``' inventory."""
    import polars as pl

    from phenotypic.schema import OBJECT

    rows = [
        {
            "Metadata_Dataset": dataset,
            "Metadata_ImageFile": image_name,
            str(OBJECT.LABEL): 1,
            "Size_Area": 16.0,
        }
        for dataset, images in work_ids.items()
        for image_name in images
    ]
    return pl.DataFrame(rows)


def write_processing_state(
    root: Path,
    *,
    work_ids: dict[str, dict[str, str]],
    pipeline_sha256: str = "a" * 64,
    metadata_sha256: str = "b" * 64,
) -> Path:
    """Write ``processing_state.json`` through the production writer.

    Carries the schema-3 config a marker-authorized run has: ``work_ids``,
    ``success_markers_required``, and the four fields the run identity's
    digests are composed from (``pipeline_sha256`` is
    ``scientific_config_digest``; ``metadata_sha256``,
    ``include_dataset_column`` and ``no_qc`` are the finalization inputs).

    Args:
        root: Run output root.
        work_ids: ``{dataset: {image filename: work id}}`` -- the accepted
            inventory, which is one of spec §4.1's three written authorities.
        pipeline_sha256: The scientific-config digest to record.
        metadata_sha256: The metadata snapshot digest to record.

    Returns:
        The written state file's path.
    """
    from datetime import datetime

    from phenotypic._cli._cli_state_management import save_processing_state
    from phenotypic._cli._cli_types import DatasetState, ProcessingState

    now = datetime.now()
    state = ProcessingState(
        version="3.0.0",
        pipeline_path=pipeline_json_path(root),
        input_path=root / "input",
        output_dir=root,
        timestamp=now,
        execution_mode="local",
        last_updated=now,
        datasets={
            dataset: DatasetState(initial_images=set(images))
            for dataset, images in work_ids.items()
        },
        config={
            "success_markers_required": True,
            "work_ids": work_ids,
            "processing_generation": "fixture-generation",
            "pipeline_sha256": pipeline_sha256,
            "metadata_sha256": metadata_sha256,
            "include_dataset_column": True,
            "no_qc": False,
            "process_only_layer": None,
        },
    )
    return save_processing_state(state, root)


def bump_scientific_config_digest(
    root: Path, *, digest: str = "c" * 64
) -> str:
    """Change the run's identity by rewriting ``config.pipeline_sha256``.

    The run identity is composed from ``processing_state.json``'s ``config``
    block, **not** from ``deliverables/pipeline.json`` -- editing the latter
    changes no identity token and no digest. Tests that need "the same tree
    under a new identity" must come through here.

    Args:
        root: Run output root.
        digest: The replacement ``scientific_config_digest``.

    Returns:
        The digest written.
    """
    from phenotypic.sdk_ import resolve_processing_state_path

    path = resolve_processing_state_path(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["pipeline_sha256"] = digest
    path.write_text(json.dumps(payload), encoding="utf-8")
    return digest


def build_complete_run(tmp_path: Path) -> Path:
    """Return an output tree whose deep verdict is ``complete``.

    Deliberately minimal: two images in one dataset, each with a promoted
    store, an embedded measurement table and an overlay; a success marker for
    each; an aggregate proof; a run proof. Anything more makes a failing test
    hard to read.

    Built by calling the **real** publishers, never by hand-writing JSON: a
    fixture that hand-writes the format under test keeps passing after the
    format changes, which is the failure mode this whole plan is about. P3
    swaps ``publish_image_success`` for the record writer and this function
    does not change.

    The publication order is the contract's own -- artifacts, then per-image
    markers, then state, then the aggregated outputs, then the aggregate
    proof, then the run proof. Reordering it would build a tree no run
    produces.

    Args:
        tmp_path: A directory to build under. ``<tmp_path>/run`` is created.

    Returns:
        The run output root.
    """
    from phenotypic._cli._cli_completion import (
        publish_aggregate_snapshot,
        publish_image_success,
        publish_run_completion_evidence,
    )

    output = tmp_path / "run"
    output.mkdir(parents=True, exist_ok=True)
    work_ids = {
        FIXTURE_DATASET: {
            f"{stem}.tif": f"work-{stem}" for stem in FIXTURE_STEMS
        }
    }

    for stem in FIXTURE_STEMS:
        work_id = f"work-{stem}"
        store = _promote_minimal_store(
            output, dataset=FIXTURE_DATASET, stem=stem, work_id=work_id
        )
        overlay = _write_overlay(output, dataset=FIXTURE_DATASET, stem=stem)
        publish_image_success(
            output,
            work_id=work_id,
            dataset=FIXTURE_DATASET,
            relative_image_path=f"{stem}.tif",
            image_stem=stem,
            mode="full",
            attempt_id=f"attempt-{stem}",
            # `scheduler_epoch` from P2 Task 4 onward.
            lifecycle_epoch="local",
            artifacts={"store": store, "overlay": overlay},
        )

    write_processing_state(output, work_ids=work_ids)
    frame = _fixture_measurements_frame(work_ids)
    write_master(output, frame)
    write_measurements_mirror(output, frame)
    publish_aggregate_snapshot(output)
    publish_run_completion_evidence(output, execution_epoch="local")
    return output


def build_incomplete_run(tmp_path: Path) -> Path:
    """The same tree with the second image's success marker removed.

    Removing the **marker** rather than the artifacts is deliberate: it is the
    state a run killed between promoting a store and publishing its proof
    actually leaves, and the one the verdict ladder has to call ``incomplete``
    rather than ``complete``.

    Args:
        tmp_path: A directory to build under.

    Returns:
        The run output root.
    """
    from phenotypic.sdk_ import image_completion_marker_path

    output = build_complete_run(tmp_path)
    image_completion_marker_path(
        output, FIXTURE_DATASET, FIXTURE_STEMS[1]
    ).unlink()
    return output


def seed_output_dir(
    root: Path,
    master: Any,
    *,
    pipeline: Any | None = None,
    mirror: Any | None = None,
    dashboard: bool = False,
    results_dataset: str | None = None,
) -> Path:
    """Seed a fake CLI output directory under ``root``.

    Writes the master archive (always), and optionally the measurements
    mirror, ``pipeline.json``, a real ``dashboard.html``, and an empty
    per-image ``results/<dataset>/`` tree (so the shell classifier
    recognizes ``root`` as a CLI output).

    Args:
        root: Run output root (``tmp_path`` or a subdir of it).
        master: Master DataFrame (polars or pandas).
        pipeline: Optional ``ImagePipeline`` or JSON string for
            ``pipeline.json``.
        mirror: Optional frame for ``measurements.{csv,parquet}``. If
            ``None`` and ``master`` is given, no mirror is written (callers
            that need the mirror to match the master should pass it
            explicitly).
        dashboard: When True, generate a real ``dashboard.html``.
        results_dataset: When set, create ``results/<dataset>/`` at the
            root so the classifier's ``is_cli_output`` check passes.

    Returns:
        ``root`` (the output dir), for chaining.
    """
    write_master(root, master)
    if mirror is not None:
        write_measurements_mirror(root, mirror)
    if pipeline is not None:
        write_pipeline_json(root, pipeline)
    if dashboard:
        write_dashboard(root)
    if results_dataset is not None:
        (root / "results" / results_dataset).mkdir(parents=True, exist_ok=True)
    return root
