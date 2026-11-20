"""``PF-EMBED-REVISION-UNPINNED``: a DINO embedder whose weights are not pinned (#281 P4 review Q1).

With ``revision=None`` each process resolves the hub's latest commit, so two
runs, or two workers of one run, can embed with different weights while the
pipeline file (and its sha) stays the same. Finalization then publishes the
key with several descriptors. The preflight says so before the run starts.
A warning, never an error, scoped to what the mode runs. Torch-free:
constructing the measurer loads no weights.
"""

from __future__ import annotations

import pytest

from phenotypic import ImagePipeline
from phenotypic._cli import _cli_preflight
from phenotypic._cli._cli_preflight import check_embedding_revision_pinned
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureDinoEmbedding
from tests.unit.cli._preflight_support import make_context

PINNED = "0123456789abcdef0123456789abcdef01234567"


def _pipeline(revision: str | None) -> ImagePipeline:
    return ImagePipeline(
        ops={"det": OtsuDetector()},
        meas={"dino": MeasureDinoEmbedding(size="small", revision=revision)},
    )


@pytest.mark.parametrize("mode", ["full", "measure"])
def test_an_unpinned_revision_in_scope_warns(mode) -> None:
    findings = check_embedding_revision_pinned(make_context(_pipeline(None), mode))

    assert [f.code for f in findings] == ["PF-EMBED-REVISION-UNPINNED"]
    (finding,) = findings
    assert finding.severity == "warning"
    assert "meas:dino" in finding.message
    assert "not pinned" in finding.message
    assert "revision" in finding.hint and "commit" in finding.hint


@pytest.mark.parametrize("mode", ["full", "measure"])
def test_a_pinned_revision_is_silent(mode) -> None:
    assert check_embedding_revision_pinned(make_context(_pipeline(PINNED), mode)) == []


def test_an_embedder_process_mode_never_runs_is_silent() -> None:
    """``process`` mode skips a crop embedder, which needs objects, so it never loads."""
    assert check_embedding_revision_pinned(make_context(_pipeline(None), "process")) == []


def test_an_unpinned_grid_embedder_warns_in_process_mode() -> None:
    """``process`` mode runs a grid embedder (#281 P5, spec E4), so it is in scope."""
    from phenotypic.grid import ManualGridFinder

    pipeline = ImagePipeline(
        meas={
            "grid": MeasureDinoEmbedding(
                size="small",
                regions="grid",
                grid=ManualGridFinder(row_edges=[0, 100, 200], col_edges=[0, 150, 300]),
            )
        },
    )
    findings = check_embedding_revision_pinned(make_context(pipeline, "process"))
    assert [f.code for f in findings] == ["PF-EMBED-REVISION-UNPINNED"]
    assert "meas:grid" in findings[0].message


def test_the_check_is_registered() -> None:
    assert check_embedding_revision_pinned in _cli_preflight.CHECKS
