"""Curating a run must not make it undiscoverable (O-3).

`publish_aggregate_snapshot` fences three artifacts by size + sha256 --
`master_measurements.parquet` and **both mirror files**. `CurationLabels`
rewrites the mirror on every save and republishes nothing, so marking a
colony breaks the proof and `OutputRoot.discover` refuses the run for good.

The disposition is not "re-issue it". The fence asserts *these are
the bytes finalization published*, and after a curation click that is
**false by design** -- the curated mirror is not the CLI's output. So the
statement stops being made about the mirror, and goes on being made about the
master, which no GUI writer touches.

`test_a_rewritten_master_still_breaks_the_fence` is the control that keeps this
a narrowing rather than a disabling.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._filtered_state import (
    KEY_IMAGE_FILE,
    KEY_OBJECT_LABEL,
)
from phenotypic.gui.results_viewer._output_root import (
    OutputRoot,
    core_readable,
)
from phenotypic.sdk_ import BundleLayout, aggregate_proof_is_current
from tests._output_layout import build_complete_viewer_run

# Scalar columns only: a nested column makes the mirror's CSV write raise
# ``ComputeError: CSV format does not support nested data``.
FRAME = pl.DataFrame(
    {
        "Metadata_Dataset": ["plate", "plate"],
        KEY_IMAGE_FILE: ["a", "b"],
        KEY_OBJECT_LABEL: [1, 2],
        "Size_Area": [10.0, 20.0],
    }
)


@pytest.fixture()
def published_run(tmp_path: Path) -> Path:
    """A marker-authorized run, built by the production publishers."""
    return build_complete_viewer_run(
        tmp_path / "run", frame=FRAME, stems=("a", "b")
    )


def _curate_one_colony(root: Path, cache_root: Path) -> None:
    """Mark one object exactly as the Colony radial menu does."""
    bound = OutputRoot.discover(root, cache_root=cache_root)
    frame = bound.clean_master_df
    row = frame.row(0, named=True)
    labels = CurationLabels.load(bound.layout, frame)
    labels.mark(
        str(row[KEY_IMAGE_FILE]),
        int(row[KEY_OBJECT_LABEL]),
        labels.categories()[0],
    )


def test_curating_a_run_leaves_it_discoverable(
    published_run: Path, tmp_path: Path
) -> None:
    """O-3. One colony marked, and the run refuses to open ever again.

    Measured end to end before this fix: ``core_readable`` goes ``True`` ->
    ``False`` and ``discover`` raises ``ValueError: Core aggregate files are
    not authorized by a valid aggregate publication marker``. The sidebar
    classifier consults neither, so the directory still presents as openable
    and the open fails; the standalone launcher does not catch it at all.
    """
    layout = BundleLayout.detect(published_run)
    assert core_readable(layout) is True

    _curate_one_colony(published_run, tmp_path / "cache-bind")

    assert core_readable(layout) is True
    assert aggregate_proof_is_current(published_run) is True
    OutputRoot.discover(published_run, cache_root=tmp_path / "cache-after")


def test_a_rewritten_master_still_breaks_the_fence(
    published_run: Path, tmp_path: Path
) -> None:
    """The control: this is a narrowing, not a disabling.

    ``master_measurements.parquet`` has no GUI writer, so the claim the proof
    makes about it survives curation and must keep being enforced. If this
    goes green-by-vacuum the fix has removed the fence instead of narrowing
    it, and O-3's whole argument -- *stop asserting the statement that became
    false, keep asserting the one that did not* -- would be unsupported.
    """
    from tests._output_layout import write_master

    layout = BundleLayout.detect(published_run)
    assert core_readable(layout) is True

    write_master(
        published_run,
        FRAME.with_columns(pl.col("Size_Area") + 1.0),
    )

    assert aggregate_proof_is_current(published_run) is False
    assert core_readable(layout) is False
    with pytest.raises(ValueError, match="aggregate publication marker"):
        OutputRoot.discover(published_run, cache_root=tmp_path / "cache")


def test_the_proof_still_records_every_published_artifact(
    published_run: Path,
) -> None:
    """Record what happened; enforce what must not change.

    The writer is deliberately untouched: the proof stays a complete record of
    what finalization published, including the two mirror descriptors it no
    longer enforces. A fix that stopped *writing* them would lose provenance
    and could not repair the trees already on disk, which is the reason this
    is reader-side.
    """
    import json

    proof = json.loads(
        (published_run / ".phenotypic" / "aggregate_publication.json")
        .read_text(encoding="utf-8")
    )
    assert set(proof["required_outputs"]) == {
        "master_parquet",
        "measurements_csv",
        "measurements_parquet",
    }


# ----------------------------------------------------------------------
# The refusal diagnostic (O-3). Five distinguishable causes, one bare None.
# ----------------------------------------------------------------------


def _proof_path(root: Path) -> Path:
    return root / ".phenotypic" / "aggregate_publication.json"


def _rewrite_proof(root: Path, mutate) -> None:
    """Edit the marker in place. Hand-written on purpose.

    There is no production writer for a *corrupt* marker, and these tests are
    about the reader's response to one. Every other fixture in this file goes
    through the real publishers.
    """
    import json

    path = _proof_path(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_a_current_proof_refuses_nothing(published_run: Path) -> None:
    from phenotypic.sdk_._run_state import aggregate_proof_refusal

    assert aggregate_proof_refusal(published_run) is None


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        pytest.param(
            lambda payload: payload.__setitem__("version", 99),
            "version",
            id="wrong-version",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("required_outputs", {}),
            "lists no required outputs",
            id="empty-outputs",
        ),
        pytest.param(
            lambda payload: payload.__setitem__("required_outputs", []),
            "lists no required outputs",
            id="non-mapping-outputs",
        ),
        pytest.param(
            lambda payload: payload["required_outputs"]["master_parquet"]
            .__setitem__("sha256", "0" * 64),
            "'master_parquet' no longer matches",
            id="master-tampered",
        ),
    ],
)
def test_each_refusal_names_its_own_cause(
    published_run: Path, mutate, fragment: str
) -> None:
    """A bare ``None`` for five causes is unanswerable in a support request.

    *"Your master was tampered with"* and *"your curation broke it"* have
    opposite dispositions, and before this the reader could not tell them
    apart. The master case names **which** output moved, because with three
    descriptors "one of them" is not actionable either.
    """
    from phenotypic.sdk_._run_state import aggregate_proof_refusal

    _rewrite_proof(published_run, mutate)
    reason = aggregate_proof_refusal(published_run)
    assert reason is not None
    assert fragment in reason


def test_an_absent_marker_is_its_own_cause(published_run: Path) -> None:
    from phenotypic.sdk_._run_state import aggregate_proof_refusal

    _proof_path(published_run).unlink()
    reason = aggregate_proof_refusal(published_run)
    assert reason is not None
    assert "no readable aggregate publication marker" in reason


def test_a_reason_never_downgrades_the_verdict(published_run: Path) -> None:
    """A reason is for reporting. Every cause is still a refusal.

    The hazard in adding a diagnostic is that a caller starts branching on
    *which* cause and treats some as recoverable. ``core_readable`` stays a
    boolean and still refuses on every one.
    """
    layout = BundleLayout.detect(published_run)
    _rewrite_proof(
        published_run,
        lambda payload: payload["required_outputs"]["master_parquet"]
        .__setitem__("size", 1),
    )
    assert aggregate_proof_is_current(published_run) is False
    assert core_readable(layout) is False


def test_an_unrecognized_descriptor_is_enforced_not_skipped(
    published_run: Path,
) -> None:
    """The narrowing is a **named** exemption, closed against the unknown.

    A fence whose default is "do not check" is not a fence. If the skip list
    were ever inverted -- enforce only what is named -- a descriptor added by
    a future writer would go unchecked and nothing would say so. This is the
    test that fails if someone flips that polarity.
    """
    from phenotypic.sdk_._run_state import aggregate_proof_refusal

    _rewrite_proof(
        published_run,
        lambda payload: payload["required_outputs"].__setitem__(
            "future_artifact",
            {
                "path": "deliverables/nope.parquet",
                "size": 1,
                "sha256": "0" * 64,
            },
        ),
    )
    reason = aggregate_proof_refusal(published_run)
    assert reason is not None
    assert "future_artifact" in reason
