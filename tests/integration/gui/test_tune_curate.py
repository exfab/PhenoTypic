"""Integration + unit tests for the Curate layout, shortlist, and A/B pin (B4).

The Curate view exposes shortlist card ids (one per shortlisted trial) and the
two side-by-side graph ids; the pure :func:`pinned_pair` helper assigns slot A,
then slot B, then re-pins (cycling back to A) without any Dash machinery.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.gui.shell import SandboxRoot


def _curate_app(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Build a loaded Curate app over a 3-trial journal + an Image Source."""
    from phenotypic.gui.tune import create_app
    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.tools_ import trials_parquet_path
    from phenotypic.tune._study_store import JournalStudyStore, Trial

    images = tmp_path / "calibration"
    images.mkdir()
    parquet = trials_parquet_path(tmp_path)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    JournalStudyStore(
        trials=[
            Trial(number=0, params={"0.sigma": 1.0}, score=0.30, terms={}, n_images=2),
            Trial(number=1, params={"0.sigma": 2.0}, score=0.60, terms={}, n_images=2),
            Trial(number=2, params={"0.sigma": 3.0}, score=0.45, terms={}, n_images=2),
        ]
    ).to_parquet(parquet)
    root = TuneRunRoot.discover(tmp_path)
    sandbox = SandboxRoot.from_path(tmp_path)
    return create_app(root=root, url_prefix="/tune/", sandbox=sandbox)


def test_curate_exposes_graph_ids(tmp_path: Path) -> None:
    app = _curate_app(tmp_path)
    layout = str(app.layout)
    for component_id in (
        "tune-graph-a",
        "tune-graph-b",
        "tune-graph-diff",
        "tune-shortlist",
        "tune-plate-picker",
        "tune-overlay-poll",
    ):
        assert component_id in layout


def test_curate_exposes_one_card_per_shortlisted_trial(tmp_path: Path) -> None:
    app = _curate_app(tmp_path)
    layout_repr = str(app.layout)
    # The three trials are all shortlisted (top-k seeds with k=5); each card
    # carries a pattern-matching id ``{"type": ..., "trial": <n>}``.
    for trial in (0, 1, 2):
        assert f"'trial': {trial}" in layout_repr


# ---------------------------------------------------------------------------
# Pure pin helper — assign A, then B, then re-pin (no Dash)
# ---------------------------------------------------------------------------

def test_pinned_pair_assigns_a_first() -> None:
    from phenotypic.gui.tune._callbacks import pinned_pair

    assert pinned_pair(1, {"a": None, "b": None}) == {"a": 1, "b": None}


def test_pinned_pair_assigns_b_second() -> None:
    from phenotypic.gui.tune._callbacks import pinned_pair

    assert pinned_pair(2, {"a": 1, "b": None}) == {"a": 1, "b": 2}


def test_pinned_pair_repins_a_when_both_full() -> None:
    from phenotypic.gui.tune._callbacks import pinned_pair

    # Both slots full → re-pin into A (the oldest slot cycles out).
    assert pinned_pair(3, {"a": 1, "b": 2}) == {"a": 3, "b": 2}


def test_pinned_pair_same_trial_into_empty_slot_is_idempotent() -> None:
    from phenotypic.gui.tune._callbacks import pinned_pair

    # Clicking the already-A trial while B is empty does not duplicate it into B.
    assert pinned_pair(1, {"a": 1, "b": None}) == {"a": 1, "b": None}


@pytest.mark.parametrize("bad_store", [None, {}, {"a": None}])
def test_pinned_pair_tolerates_missing_store(bad_store: object) -> None:
    from phenotypic.gui.tune._callbacks import pinned_pair

    result = pinned_pair(5, bad_store)  # type: ignore[arg-type]
    assert result["a"] == 5
