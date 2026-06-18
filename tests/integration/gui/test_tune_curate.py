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
    from phenotypic.sdk_ import trials_parquet_path
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


# ---------------------------------------------------------------------------
# Difference-mode CSS specificity (the side-by-side panels must hide)
# ---------------------------------------------------------------------------

def _tune_css() -> str:
    import phenotypic.gui.tune as tune_pkg

    css_path = Path(tune_pkg.__file__).parent / "_assets" / "tune.css"
    return css_path.read_text(encoding="utf-8")


def test_difference_mode_hide_rule_outspecifies_grid() -> None:
    """A compound (two-class) rule hides the side-by-side panels in Difference.

    ``.tune-view-hidden{display:none}`` (one class) loses the cascade to the
    equally-specific ``.tune-curate-sidebyside{display:grid}`` declared later in
    the file, so the panels leaked through in Difference mode. The fix is a
    higher-specificity ``.tune-curate-sidebyside.tune-view-hidden{display:none}``
    rule; this guards it isn't dropped, and that it sits AFTER the base grid rule
    so the cascade resolves correctly.
    """
    css = _tune_css()
    compound = ".tune-curate-sidebyside.tune-view-hidden"
    assert compound in css, "missing the higher-specificity Difference-hide rule"

    # The compound rule must appear after the base `.tune-curate-sidebyside {`
    # declaration (cascade order matters for equal specificity, and this is
    # higher specificity regardless — but ordering keeps intent obvious).
    base_idx = css.index(".tune-curate-sidebyside {")
    compound_idx = css.index(compound)
    assert compound_idx > base_idx

    # The compound rule resolves to display:none (the hide).
    tail = css[compound_idx : compound_idx + 160]
    assert "display: none" in tail


def test_switch_curate_mode_classes_toggle_hidden(tmp_path: Path) -> None:
    """In Difference mode the side-by-side container carries ``tune-view-hidden``;
    in Side-by-side mode the difference container does. Pairs with the CSS rule
    above (class wiring + specificity together make the panels hide)."""
    app = _curate_app(tmp_path)
    client = app.server.test_client()

    out_key = next(
        k
        for k in app.callback_map
        if "tune-curate-mode-store.data" in k and "tune-side-by-side" in k
    )

    def _classes(mode_value: str) -> dict[str, str]:
        resp = client.post(
            "/_dash-update-component",
            json={
                "output": out_key,
                "outputs": [
                    {"id": "tune-curate-mode-store", "property": "data"},
                    {"id": "tune-side-by-side", "property": "className"},
                    {"id": "tune-difference", "property": "className"},
                ],
                "inputs": [
                    {
                        "id": "tune-curate-mode-toggle",
                        "property": "value",
                        "value": mode_value,
                    }
                ],
                "state": [],
                "changedPropIds": ["tune-curate-mode-toggle.value"],
            },
        )
        assert resp.status_code == 200
        r = resp.get_json()["response"]
        return {
            "side": r["tune-side-by-side"]["className"],
            "diff": r["tune-difference"]["className"],
        }

    difference = _classes("difference")
    assert "tune-view-hidden" in difference["side"]  # side-by-side hidden
    assert "tune-view-hidden" not in difference["diff"]

    side = _classes("side")
    assert "tune-view-hidden" not in side["side"]
    assert "tune-view-hidden" in side["diff"]  # difference hidden
