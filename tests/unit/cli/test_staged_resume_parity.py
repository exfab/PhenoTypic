"""Differential resume parity: the zarr classifier must agree with the HDF one.

This is the test that would have caught all three resume defects the spec's
independent review found. It enumerates every combination
``classify_staged_image`` currently distinguishes and asserts the two artifact
worlds produce the same stage, rather than asserting a hand-written table that
could itself encode the bug.

``tests/unit/cli/test_staged_resume.py`` already parameterizes
``markers_required``; this mirrors that shape across all four axes.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pytest

from phenotypic._cli._cli_staged_resume import classify_staged_image
from phenotypic.sdk_ import image_record_path

PROCESS_ONLY_LAYERS = [None, "objmap", "gray"]
MARKERS_REQUIRED = [True, False]
WORK_IDS = [None, "w-1"]
#: Which durable artifacts exist, as
#: (image_state, stage2_signal, parquet, stage3_marker, image_success).
#:
#: The FIFTH axis is load-bearing. classify_staged_image's first branch
#: consults valid_image_success, which reads the per-image RECORD -- P3's clean
#: break moved it from image_complete/ to images/, and this comment named the
#: old file until then. Without this axis that branch is never exercised --
#: valid_image_success returns False in both worlds -- and the parity test
#: passes while production breaks. See Task 3.8 / OPEN-QUESTIONS D2.
#:
#: The warning came true on its own axis. The break left the axis inert
#: wherever parquet=False (16 of these 32 combinations, 192 of the 384 cases)
#: because both values then produced "no record" rather than a stale one, and
#: nothing failed. test_the_stale_record_is_rejected_only_for_its_missing
#: _artifact is what now holds that half up.
ARTIFACTS = list(itertools.product([False, True], repeat=5))

CASES = [
    pytest.param(
        layer,
        markers,
        work_id,
        artifacts,
        id=f"{layer}-{markers}-{work_id}-{''.join('1' if a else '0' for a in artifacts)}",
    )
    for layer, markers, work_id, artifacts in itertools.product(
        PROCESS_ONLY_LAYERS, MARKERS_REQUIRED, WORK_IDS, ARTIFACTS
    )
]


@pytest.mark.parametrize(("layer", "markers", "work_id", "artifacts"), CASES)
def test_zarr_classifier_matches_the_hdf_classifier(
    layer, markers, work_id, artifacts, hdf_world, zarr_world
):
    """hdf_world / zarr_world build the same artifact set in the two formats."""
    hdf_root = hdf_world(artifacts, work_id=work_id)
    zarr_root = zarr_world(artifacts, work_id=work_id)
    common = dict(
        dataset="ds",
        image=Path("img.tif"),
        input_root=Path("/in"),
        process_only_layer=layer,
        markers_required=markers,
        expected_work_id=work_id,
    )
    assert classify_staged_image(output_dir=zarr_root, **common) == (
        hdf_world.classify(output_dir=hdf_root, **common)
    )


def test_the_stale_record_is_rejected_only_for_its_missing_artifact(zarr_world):
    """The fifth axis's *stale* half, pinned by cause rather than by verdict.

    ``ArtifactWorld`` has two success branches. With a parquet it publishes
    through the real writer, so it follows the schema automatically. Without
    one it hand-writes a record describing a file that is not there -- the
    stale case -- and **that half went dead at P3's clean break without
    failing**: ``valid_image_success`` moved from ``image_complete/`` to
    ``images/``, the hand-written legacy marker stopped being read, and the
    verdict stayed ``False`` for a different reason. A parity test comparing
    ``False`` to ``False`` passes.

    Half the suite went with it. ``success`` and its negation both produced
    "no record" wherever ``parquet=False`` -- 16 of the 32 ``ARTIFACTS``
    combinations, so 192 of these 384 cases -- at full green.

    So this asserts the **cause**, in two halves that only mean something
    together, because no single call names staleness: ``record_rejection``
    checks identity and shape and returns ``None`` here, while
    ``fenced_artifact_path`` collapses malformed, escapes-the-root and
    missing-on-disk into one ``None``.

    * The fixture's own ``record_rejection(...) is None`` assertion proves no
      shape or identity clause fires, so the record's *form* is not what
      rejects it.
    * This test supplies the missing file and shows the verdict flip, which
      establishes the artifact's absence as the cause rather than a fact that
      merely co-occurs with it.

    Restoring only one of those would reproduce the defect: the first alone is
    consistent with rejection for some unrelated reason, and the second alone
    is the verdict check that failed to notice the break.
    """
    from phenotypic._cli._cli_completion import valid_image_success

    # success=True, parquet=False -- the stale-record cell.
    root = zarr_world(
        (True, False, False, False, True), work_id="w-1"
    )
    common = dict(dataset="ds", image_stem="img", work_id="w-1")

    assert not valid_image_success(root, **common)

    described = (
        root
        / json.loads(
            image_record_path(root, "ds", "img").read_text(encoding="utf-8")
        )["artifacts"]["measurements"]["path"]
    )
    assert not described.exists(), "the stale case requires an absent artifact"

    described.parent.mkdir(parents=True, exist_ok=True)
    described.write_bytes(b"measurements")
    # `b"measurements"` is 12 bytes, which is exactly the size the fixture's
    # descriptor claims -- so the SIZE now matches and only the digest does
    # not. That is the sharper intermediate: presence alone is not what the
    # verdict turns on.
    assert not valid_image_success(root, **common), (
        "the descriptor's sha256 still disagrees, so this proves nothing yet"
    )

    # Now make the bytes match what the record actually claims.
    record_path = image_record_path(root, "ds", "img")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    descriptor = record["artifacts"]["measurements"]
    descriptor["size"] = described.stat().st_size
    descriptor["sha256"] = hashlib.sha256(described.read_bytes()).hexdigest()
    record_path.write_text(json.dumps(record), encoding="utf-8")

    assert valid_image_success(root, **common), (
        "with the described artifact present and matching, the record must "
        "verify -- if it does not, the rejection was never about the artifact"
    )

    described.unlink()
    assert not valid_image_success(root, **common)


def test_the_axes_actually_reach_every_return_value(hdf_world):
    """A parity test that only ever compares "stage1" to "stage1" is vacuous.

    Pins that the enumeration exercises all four outcomes of the frozen
    reference, so a future narrowing of the axes cannot quietly hollow the
    comparison out.
    """
    seen = set()
    for layer, markers, work_id, artifacts in itertools.product(
        PROCESS_ONLY_LAYERS, MARKERS_REQUIRED, WORK_IDS, ARTIFACTS
    ):
        root = hdf_world(artifacts, work_id=work_id)
        seen.add(
            hdf_world.classify(
                output_dir=root,
                dataset="ds",
                image=Path("img.tif"),
                input_root=Path("/in"),
                process_only_layer=layer,
                markers_required=markers,
                expected_work_id=work_id,
            )
        )
    assert seen == {"stage1", "stage2", "stage3", "complete"}
