"""Rolling input: arrivals invalidate scope, never per-image proofs.

Spec §14's third named test, and the scenario the whole design is shaped
around -- the audit's running example is a 6,000-image run that grows.

**The property under test:** per-image proofs survive an arrival; only
aggregate-level proofs invalidate. §9.2 is why it matters -- adding 10 images
to a 6,000-image run today re-derives the worklist by validating 6,000
markers, each re-hashing its measurements Parquet and overlay PNG.

D7 is the identity half of the same property: a new image changes
``inventory_digest`` but **not** ``processing_generation``, so live progress is
not reset and in-flight workers are not fenced.

⛔ STANDING RULE, inherited from ``test_finalize_run.py`` and applied
throughout: every assertion is preceded by an assertion that the fixture
produced the thing whose presence, absence or equality is claimed. And per
register entry 62, every guard here states **what input would make it fire** --
a four-scenario matrix is exactly where a row quietly asserts nothing.

**Why this file lives in ``tests/integration/cli/`` and not
``tests/integration/``** as the plan's file table says: `synth_plate_dir` and
`simple_pipeline_json` are defined in this package's ``conftest.py``, and a
sibling directory would not see them. Duplicating them is what the P5 conftest
promotion existed to stop. Rolling input is CLI behaviour, so this is also
where it belongs on subject.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import (
    measurements_parquet_path,
    resolve_run_state,
    run_identity,
    zarr_store_path,
)

from .conftest import _write_synth_image


def _run(
    input_dir: Path,
    out: Path,
    pipeline: Path,
    *,
    metadata: Path | None = None,
    skip_validation: bool = True,
) -> None:
    """Invoke the real CLI, local, one worker.

    ``skip_validation`` is a parameter rather than a constant because the
    unready-file test **must not** pass it: input validation is the mechanism
    that keeps a half-written file out of the inventory, so skipping it would
    disable the very guard that test exists to exercise.
    """
    args = [
        "--pipeline", str(pipeline),
        "--input", str(input_dir),
        "--output", str(out),
        "--force-local", "--njobs", "1",
    ]
    if skip_validation:
        args.append("--skip-validation")
    if metadata is not None:
        args.extend(["--metadata", str(metadata)])
    result = CliRunner().invoke(phenotypic_cli, args)
    assert result.exit_code == 0, result.output


def _stages(out: Path) -> dict[str, dict]:
    """Per-image stage maps, keyed by work_id."""
    return {
        work_id: dict(image.stages)
        for work_id, image in resolve_run_state(out, depth="deep").images.items()
    }


def _store_mtimes(out: Path) -> dict[str, float]:
    """Each store root's mtime -- the witness that no store was rewritten."""
    return {
        str(path): path.stat().st_mtime
        for path in sorted(out.glob("results/*/zarr/*.ome.zarr/zarr.json"))
    }


def test_a_new_image_changes_the_inventory_but_not_the_generation(
    tmp_path: Path, synth_plate_dir: Path, simple_pipeline_json: Path
) -> None:
    """D7, and it is the mechanism behind ``batch_added_mid_run``.

    An image arriving mid-run must not fence in-flight workers. What makes
    that true is that ``processing_generation`` is a function of the *scientific
    configuration* while ``inventory_digest`` is a function of the accepted
    scope, so an arrival moves the second and not the first.

    Asserted on the identity directly rather than by racing a real arrival
    against a live run: ``CliRunner`` is single-threaded, so a "mid-run" test
    built on it would be a between-runs test wearing a different name -- which
    is the vacuity entry 62 is about.

    **What would make this fire:** folding ``inventory_digest`` into the
    generation digest, which is precisely the mistake D7 exists to prevent.
    """
    out = tmp_path / "out"
    _run(synth_plate_dir, out, simple_pipeline_json)
    before = run_identity(out)
    assert before is not None and before.processing_generation, (
        "no identity was minted, so both comparisons below would be vacuous"
    )

    _write_synth_image(synth_plate_dir / "plate_002.png")
    _run(synth_plate_dir, out, simple_pipeline_json)
    after = run_identity(out)
    assert after is not None

    assert after.inventory_digest != before.inventory_digest, (
        "the arrival did not change inventory_digest, so this test cannot "
        "distinguish the two digests' behaviour"
    )
    assert after.processing_generation == before.processing_generation, (
        "an arrival moved processing_generation -- live progress would be "
        "reset and in-flight workers fenced (D7)"
    )


def test_only_aggregate_proofs_invalidate_when_the_input_grows(
    tmp_path: Path, synth_plate_dir: Path, simple_pipeline_json: Path
) -> None:
    """``batch_added_between_runs``: existing per-image proofs survive.

    **What would make this fire:** anything that re-verifies or re-mints an
    existing image's record because the inventory changed -- the §9.2 cost the
    whole design exists to remove.
    """
    out = tmp_path / "out"
    _run(synth_plate_dir, out, simple_pipeline_json)
    before = _stages(out)
    assert before, "no per-image state; the comparison below would be vacuous"

    _write_synth_image(synth_plate_dir / "plate_002.png")
    _run(synth_plate_dir, out, simple_pipeline_json)
    after = resolve_run_state(out, depth="deep")

    assert len(after.images) > len(before), (
        f"the arrival did not enter the inventory ({len(after.images)} images "
        f"after vs {len(before)} before); nothing rolled"
    )
    for work_id, stages in before.items():
        assert work_id in after.images, (
            f"{work_id} left the inventory when a new image arrived"
        )
        assert dict(after.images[work_id].stages) == stages, (
            "an arrival invalidated an existing image's proof; only "
            "aggregate-level proofs may invalidate when scope changes"
        )
    assert after.completion == "complete"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "UNIMPLEMENTED, and by the composition of two deliberate decisions "
        "rather than by omission -- see register entry 65. Out of P5's scope: "
        "admission checking is new product behaviour belonging to whoever "
        "owns input handling, not to a fan-out phase. strict=True so this "
        "fails loudly the day the behaviour appears without the marker going."
    ),
)
def test_an_unready_file_is_not_accepted_into_the_inventory(
    tmp_path: Path, synth_plate_dir: Path, simple_pipeline_json: Path
) -> None:
    """A file still being written must not enter ``work_ids``.

    **This currently fails, and the product is what is wrong.** A truncated
    file is admitted, processed, and raises
    ``PerImageScientificError: broken PNG file``.

    The hazard is real and arises from **two individually-correct decisions**:

    * ``_cli_directory_scanner.py:28-32`` tests candidates **by name, never by
      opening them** -- deliberate, and justified: "reading a root
      ``zarr.json`` per entry would cost an open per file at 10k-image scale."
      Unreadable input is left to fail "later, loudly, in ``imread``".
    * ``current_success_counts`` (``_cli_completion.py:710-730``) counts
      **every image the state claims** in ``total``, so "a run with a failed
      image reports ``successful < total``" -- and ``current_run_is_complete``
      requires equality.

    Composed: a file admitted while still being copied can never succeed, and
    ``total`` never shrinks, so the run is **permanently incomplete**. Neither
    decision is wrong alone; the conjunction has no owner.

    **``--skip-validation`` is NOT the mechanism**, and an earlier version of
    this test opted out of it believing otherwise. That flag gates *config and
    pipeline* validation (``phenotypicCLI.py:2260``; help text: "Skip pipeline
    validation") and never opens an input image, so opting out changed
    nothing. The opt-out is kept only because running the fuller path costs
    nothing here -- it is not load-bearing, and saying so is the point.
    """
    out = tmp_path / "out"
    _run(synth_plate_dir, out, simple_pipeline_json, skip_validation=False)
    assert resolve_run_state(out, depth="deep").completion == "complete", (
        "the baseline run did not complete, so a later 'still complete' "
        "assertion would prove nothing"
    )

    partial = synth_plate_dir / "still-copying.png"
    partial.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    assert partial.is_file()

    _run(synth_plate_dir, out, simple_pipeline_json, skip_validation=False)
    assert resolve_run_state(out, depth="deep").completion == "complete", (
        "a half-written file entered the inventory and parked the run at "
        "incomplete"
    )


def test_metadata_arriving_later_re_runs_finalize_and_nothing_else(
    tmp_path: Path, synth_plate_dir: Path, simple_pipeline_json: Path
) -> None:
    """§7.4, as narrowed by D-A.

    A metadata edit changes ``finalization_input_digest``, so the next
    invocation re-joins the mirror. Stores keep the snapshot they were built
    against and are **not** rewritten.

    **What would make this fire:** a metadata arrival that re-promoted stores
    (the mtime assertion), or one that reached the mirror only as
    metadata-only phantoms (the null-count assertion).
    """
    out = tmp_path / "out"
    _run(synth_plate_dir, out, simple_pipeline_json)
    mtimes = _store_mtimes(out)
    assert mtimes, "no stores were written; the mtime comparison is vacuous"

    metadata = tmp_path / "meta.csv"
    metadata.write_text(
        f"{IMAGE.IMAGE_NAME},Metadata_Strain\nplate_001,WT\n", encoding="utf-8"
    )
    _run(synth_plate_dir, out, simple_pipeline_json, metadata=metadata)

    mirror = pl.read_parquet(measurements_parquet_path(out))
    assert "Metadata_Strain" in mirror.columns
    measured = mirror.filter(pl.col("QC_MetadataOnly").fill_null(False).not_())
    assert measured.height > 0, "fixture produced no measured rows to check"
    # CAN-2: the first draft asserted only that the column exists, which the
    # PHANTOM rows satisfy while every measured row is null -- the test passed
    # on broken data.
    assert measured["Metadata_Strain"].null_count() == 0, (
        "user metadata reached the mirror only as metadata-only phantoms"
    )
    assert _store_mtimes(out) == mtimes, "a metadata edit rewrote a store"
