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

import itertools
from pathlib import Path

import pytest

from phenotypic._cli._cli_staged_resume import classify_staged_image

PROCESS_ONLY_LAYERS = [None, "objmap", "gray"]
MARKERS_REQUIRED = [True, False]
WORK_IDS = [None, "w-1"]
#: Which durable artifacts exist, as
#: (image_state, stage2_signal, parquet, stage3_marker, image_success_marker).
#:
#: The FIFTH axis is load-bearing. classify_staged_image's first branch
#: consults valid_image_success, which reads the per-image completion marker.
#: Without this axis that branch is never exercised -- valid_image_success
#: returns False in both worlds -- and the parity test passes while production
#: breaks. See Task 3.8 / OPEN-QUESTIONS D2.
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
