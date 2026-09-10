"""An image that detected nothing is processed, not missing.

``--mode recompile`` used to leave any run containing a zero-detection
image flagged read-only by the results viewer. The manifest it wrote
counted ``total_images`` from :func:`authorized_measurement_sources` --
images carrying an embedded measurements table -- while the viewer's
completion guard compares that against the processing inventory. An image
whose detector found no colonies is legitimately in the second set and
not the first, so the two disagreed and the guard, correctly, refused to
trust the run.

Observed on a real 36-image run where 4 images detected nothing:
``total_images: 32`` against an inventory of 36, and ``completed: 0``
beside ``is_complete: true`` -- contradictory on the manifest's own terms
before anything else read it.

The two sets these tests keep apart:

* **processing inventory** -- every image with a valid success marker.
  The basis for completion accounting.
* **authorized measurement sources** -- markers that also carry a
  ``measurements`` artifact. The basis for aggregation, and still correct
  for that.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from phenotypic._cli._cli_completion import (
    authorized_measurement_sources,
    current_run_is_complete,
    current_success_inventory,
    manifest_completion_inventory,
    publish_aggregate_snapshot,
    publish_image_success,
)
from phenotypic._cli._cli_state_management import save_processing_state
from phenotypic._cli._cli_types import DatasetState, ProcessingState
from phenotypic._cli._dashboard._generator import regenerate_dashboard_artifacts
from phenotypic.sdk_ import (
    manifest_json_path,
    master_measurements_csv_path,
    master_measurements_parquet_path,
    measurements_csv_path,
    measurements_parquet_path,
)

DATASET = "plate"

#: The image whose detector found colonies, and the one whose did not.
#: Both complete; only the first has measurements to publish. Mirrors the
#: real markers exactly -- the zero-detection image's carried
#: ``['overlay', 'store']`` and no ``measurements`` key.
MEASURED = "with_colonies.tif"
EMPTY = "no_colonies.tif"


def _build_run(tmp_path: Path) -> Path:
    """A two-image run, one of which detected nothing.

    Args:
        tmp_path: Root for the synthetic output directory.

    Returns:
        The output directory.
    """
    results = tmp_path / "results" / DATASET
    results.mkdir(parents=True)

    measurements = results / "with_colonies.parquet"
    measurements.write_bytes(b"one measured colony")
    store = results / "no_colonies.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_bytes(b"{}")

    publish_image_success(
        tmp_path,
        work_id="work-measured",
        dataset=DATASET,
        relative_image_path=MEASURED,
        image_stem=Path(MEASURED).stem,
        mode="full",
        attempt_id="attempt",
        lifecycle_epoch="epoch",
        artifacts={"measurements": measurements},
    )
    # No `measurements` key: there were no colonies to measure. This is
    # the whole fixture -- everything else here is scaffolding for it.
    publish_image_success(
        tmp_path,
        work_id="work-empty",
        dataset=DATASET,
        relative_image_path=EMPTY,
        image_stem=Path(EMPTY).stem,
        mode="full",
        attempt_id="attempt",
        lifecycle_epoch="epoch",
        artifacts={"store": store / "zarr.json"},
    )

    now = datetime.now()
    save_processing_state(
        ProcessingState(
            version="3.0.0",
            pipeline_path=tmp_path / "pipeline.json",
            input_path=tmp_path / "input",
            output_dir=tmp_path,
            timestamp=now,
            execution_mode="local",
            last_updated=now,
            datasets={
                DATASET: DatasetState(initial_images={MEASURED, EMPTY})
            },
            config={
                "success_markers_required": True,
                "work_ids": {
                    DATASET: {
                        MEASURED: "work-measured",
                        EMPTY: "work-empty",
                    }
                },
            },
        ),
        tmp_path,
    )

    # Publish the aggregate snapshot too, without which
    # `current_run_is_complete` is False and `is_complete` never reaches
    # the manifest as True -- which would leave the invariant test below
    # vacuously passing on a run that never claimed completion. The real
    # run had published one; a fixture that had not was not reproducing
    # it.
    for core in (
        master_measurements_csv_path(tmp_path),
        master_measurements_parquet_path(tmp_path),
        measurements_csv_path(tmp_path),
        measurements_parquet_path(tmp_path),
    ):
        core.parent.mkdir(parents=True, exist_ok=True)
        core.write_bytes(b"aggregated")
    publish_aggregate_snapshot(tmp_path)
    assert current_run_is_complete(tmp_path) is True
    return tmp_path


def test_the_two_bases_disagree_exactly_on_the_zero_detection_image(
    tmp_path: Path,
) -> None:
    """Pin the distinction the bug conflated, before relying on it.

    Neither number is wrong. ``authorized_measurement_sources`` returning
    one is the correct answer to "what can be aggregated"; the inventory
    returning two is the correct answer to "what was processed". The
    defect was asking the first function the second question.
    """
    output_dir = _build_run(tmp_path)

    inventory = current_success_inventory(output_dir)
    assert inventory is not None
    assert inventory[DATASET] == frozenset({MEASURED, EMPTY})

    authorized = authorized_measurement_sources(output_dir)
    assert authorized is not None
    assert len(authorized) == 1


def test_the_recompiled_manifest_counts_a_zero_detection_image(
    tmp_path: Path,
) -> None:
    """The manifest a recompile writes must agree with the inventory.

    Asserted on the numbers the viewer's guard actually compares, because
    those are what went wrong: the guard reported ``32!=36`` and
    ``0!=36``. A test that only checked ``is_complete`` would have passed
    throughout the bug -- it was ``true`` the entire time.
    """
    output_dir = _build_run(tmp_path)
    inventory = current_success_inventory(output_dir)
    assert inventory is not None
    totals = {name: len(images) for name, images in inventory.items()}

    regenerate_dashboard_artifacts(
        output_dir, None, totals, dataset_inventory=inventory
    )

    manifest = json.loads(
        manifest_json_path(output_dir).read_text(encoding="utf-8")
    )
    assert manifest["total_images"] == 2
    assert manifest["completed"] == 2
    assert manifest["successful"] == 2
    assert manifest["pending"] == 0
    assert manifest["remaining"] == 0


def test_the_manifest_never_claims_completion_it_did_not_count(
    tmp_path: Path,
) -> None:
    """``is_complete`` and the counts must tell the same story.

    The shipped manifest asserted ``is_complete: true`` beside
    ``completed: 0``, which needs no second file to be recognised as
    wrong. This is the invariant that makes that self-evident rather than
    something only the viewer notices.
    """
    output_dir = _build_run(tmp_path)
    inventory = current_success_inventory(output_dir)
    assert inventory is not None
    totals = {name: len(images) for name, images in inventory.items()}

    regenerate_dashboard_artifacts(
        output_dir, None, totals, dataset_inventory=inventory
    )

    manifest = json.loads(
        manifest_json_path(output_dir).read_text(encoding="utf-8")
    )
    # Asserted unconditionally: the fixture publishes an aggregate
    # snapshot precisely so `is_complete` is True here. Guarding this
    # behind `if manifest["is_complete"]` would let the whole test pass
    # by never entering.
    assert manifest["is_complete"] is True
    assert manifest["completed"] == manifest["total_images"], (
        "a complete run must have counted every image it declares"
    )


def test_a_counting_shortfall_is_reported_when_markers_say_complete(
    tmp_path: Path, caplog
) -> None:
    """The contradiction is named at the write boundary, not just downstream.

    Reproduces the shipped shape by withholding the inventory, which is
    what a run with no ``job_metadata.json`` did: markers still certify
    the run complete, so ``is_complete`` is ``true``, while the counting
    path has nothing to count from and reports zero.

    Deliberately a warning rather than a raise. The manifest is progress
    reporting, and failing the finalisation of a run whose per-image
    evidence says it completed would be a worse outcome than a wrong
    progress number that something else already refuses to act on.
    """
    output_dir = _build_run(tmp_path)

    with caplog.at_level("WARNING"):
        regenerate_dashboard_artifacts(
            output_dir, None, {DATASET: 2}, dataset_inventory=None
        )

    manifest = json.loads(
        manifest_json_path(output_dir).read_text(encoding="utf-8")
    )
    assert manifest["is_complete"] is True
    assert manifest["completed"] == 0  # the shipped shape, reproduced
    assert "Manifest completion is inconsistent" in caplog.text
    assert "0 of 2" in caplog.text


def test_a_consistent_manifest_reports_nothing(
    tmp_path: Path, caplog
) -> None:
    """The guard must stay quiet on the run the fix produces.

    Without this, the warning above could pass while firing on every
    healthy run, which would train a reader to ignore it.
    """
    output_dir = _build_run(tmp_path)
    inventory = current_success_inventory(output_dir)
    assert inventory is not None
    totals = {name: len(images) for name, images in inventory.items()}

    with caplog.at_level("WARNING"):
        regenerate_dashboard_artifacts(
            output_dir, None, totals, dataset_inventory=inventory
        )

    assert "Manifest completion is inconsistent" not in caplog.text


def test_the_recompile_basis_counts_processed_not_measured_images(
    tmp_path: Path,
) -> None:
    """Pin the basis the recompile path itself chooses.

    The tests above call ``regenerate_dashboard_artifacts`` with an
    inventory they computed, so they prove the manifest builder handles
    one correctly -- and prove nothing about whether the recompile path
    supplies one. It did not, and a mutation restoring the old
    `authorized_measurement_sources` basis at that call site passed all
    of them. This is the test that fails on it.
    """
    output_dir = _build_run(tmp_path)

    totals, inventory = manifest_completion_inventory(output_dir, [DATASET])

    assert totals == {DATASET: 2}, (
        "the zero-detection image was processed and must be counted"
    )
    assert inventory is not None
    assert inventory[DATASET] == frozenset({MEASURED, EMPTY})


def test_a_dataset_with_no_surviving_images_is_reported_as_zero(
    tmp_path: Path,
) -> None:
    """A named dataset must not vanish from the totals it belongs in.

    ``build_manifest`` reconciles the totals against the inventory key by
    key, so a dataset silently dropped from one and not the other raises
    there instead of reporting an empty dataset.
    """
    output_dir = _build_run(tmp_path)

    totals, inventory = manifest_completion_inventory(
        output_dir, [DATASET, "never_ran"]
    )

    assert totals["never_ran"] == 0
    assert inventory is not None
    assert inventory["never_ran"] == frozenset()


def test_a_legacy_run_without_markers_reports_no_inventory(
    tmp_path: Path,
) -> None:
    """The legacy fallback must say it has no inventory, not invent one.

    ``build_manifest`` would otherwise reconcile the totals against an
    empty mapping and raise on every legacy tree.
    """
    (tmp_path / "results").mkdir()

    totals, inventory = manifest_completion_inventory(tmp_path, ["plate"])

    assert inventory is None
    assert totals == {"plate": 0}
