from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic.correction._color_correction._checker_identity import (
    MIN_PLACEMENT_MARGIN,
    assign_placement,
    chart_grid,
    identity_features,
    placements,
)

RESOURCES = Path(__file__).parent / "resources"


@pytest.fixture(scope="module")
def cards():
    """Eight real half-cards: 4 Rhodotorula frames x left/right bands.

    Committed as a small ``.npz`` rather than read from the external
    ``patch_measurements.npz`` so the suite has no dependency outside the repo.
    """
    data = np.load(RESOURCES / "checker_identity_cards.npz", allow_pickle=False)
    names = [str(n) for n in data["patch_names"]]
    return {
        "blocks"    : data["cards"],
        "truths"    : data["truths"],
        "labels"    : [str(x) for x in data["labels"]],
        "names"     : names,
        "ref_linear": {n: data["ref_linear"][i] for i, n in enumerate(names)},
    }


def _correct(placement, truth) -> int:
    return sum(
            a == b
            for arow, brow in zip(placement.names, truth)
            for a, b in zip(arow, brow)
    )


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
def test_chart_grid_lays_patches_out_row_major(cards) -> None:
    grid = chart_grid(cards["names"])

    assert len(grid) == 4 and len(grid[0]) == 6
    assert grid[0][0] == "dark skin"
    assert grid[3][0].startswith("white")


def test_chart_grid_rejects_a_shape_that_does_not_fit() -> None:
    with pytest.raises(ValueError, match="does not hold"):
        chart_grid(["a", "b", "c", "d"], shape=(3, 3))


def test_chart_grid_needs_a_shape_for_an_unknown_chart() -> None:
    with pytest.raises(ValueError, match="No known grid shape"):
        chart_grid(["a", "b", "c", "d", "e"])


def test_placements_enumerates_only_contiguous_blocks(cards) -> None:
    """A card cut in half gives two ADJACENT chart rows, never rows 1 and 3.

    12 = 3 adjacent row pairs x 2 row directions x 2 column directions.
    Enumerating impossible placements only lowers the margin.
    """
    found = placements(chart_grid(cards["names"]), (6, 2))

    assert len(found) == 12
    for placement in found:
        rows = {
            r for row in placement.names for name in row
            for r in [cards["names"].index(name) // 6]
        }
        assert max(rows) - min(rows) == 1


def test_placements_is_empty_when_the_block_cannot_fit(cards) -> None:
    assert placements(chart_grid(cards["names"]), (7, 7)) == []


def test_identity_features_are_invariant_to_exposure() -> None:
    """An exposure difference must not be able to choose the placement."""
    rng = np.random.default_rng(0)
    linear = rng.uniform(0.05, 0.9, (12, 3))

    np.testing.assert_allclose(
            identity_features(linear), identity_features(linear * 1.6), atol=1e-9
    )


# ---------------------------------------------------------------------------
# The real cards
# ---------------------------------------------------------------------------
def test_placement_is_recovered_without_being_told(cards) -> None:
    """Every tile on every real card, identified from pixels alone."""
    grid = chart_grid(cards["names"])
    candidates = placements(grid, (6, 2))

    for block, truth, label in zip(cards["blocks"], cards["truths"], cards["labels"]):
        result = assign_placement(block, candidates, cards["ref_linear"])

        assert _correct(result.placement, truth) == 12, label
        assert result.margin >= 0.47, label
        assert result.determined, label


def test_a_rotated_card_is_assigned_correctly_not_merely_flagged(cards) -> None:
    """A card mounted 180 degrees out is read, not rejected."""
    grid = chart_grid(cards["names"])
    candidates = placements(grid, (6, 2))
    block = cards["blocks"][0]
    truth = cards["truths"][0]

    result = assign_placement(block[::-1, ::-1], candidates, cards["ref_linear"])

    assert _correct(result.placement, truth[::-1, ::-1]) == 12
    assert result.determined


def test_occluded_tiles_are_refused_on_margin_not_mislabelled(cards) -> None:
    """The gate is the evidence that identity is trustworthy.

    Bright occlusions are injected into 1-6 of the 12 tiles. Across the
    trials no card is ever mislabelled while still clearing the gate: every
    wrong answer comes with a margin below it.
    """
    grid = chart_grid(cards["names"])
    candidates = placements(grid, (6, 2))
    rng = np.random.default_rng(0)
    silent_failures = 0
    wrong = 0

    for block, truth in zip(cards["blocks"], cards["truths"]):
        for n_hit in range(1, 7):
            for _ in range(40):
                observed = block.copy()
                for flat in rng.choice(12, n_hit, replace=False):
                    row, col = divmod(int(flat), 2)
                    observed[row, col] = np.clip(
                            block[row, col] * rng.uniform(0.1, 0.4)
                            + rng.uniform(0.3, 0.9), 0, 1,
                    )
                result = assign_placement(observed, candidates, cards["ref_linear"])
                if _correct(result.placement, truth) < 12:
                    wrong += 1
                    if result.margin >= MIN_PLACEMENT_MARGIN:
                        silent_failures += 1

    assert wrong > 0, "the injection was too gentle to test anything"
    assert silent_failures == 0


def test_a_blank_card_is_refused_rather_than_guessed_at(cards) -> None:
    """The deterministic catastrophic case: no colour structure left at all."""
    grid = chart_grid(cards["names"])
    candidates = placements(grid, (6, 2))
    blank = np.full_like(cards["blocks"][0], 0.42)

    result = assign_placement(blank, candidates, cards["ref_linear"])

    assert not result.determined
    assert result.margin < MIN_PLACEMENT_MARGIN


def test_a_missing_tile_is_left_out_rather_than_poisoning_the_score(cards) -> None:
    """A NaN tile (a box outside the ROI) must not vote or normalise.

    Luminance is normalised by the block's brightest tile, so one NaN used to
    turn every feature NaN and crash the Hungarian corroboration.
    """
    block = cards["blocks"][0].astype(float).copy()
    candidates = placements(chart_grid(cards["names"]), block.shape[:2])
    clean = assign_placement(block, candidates, cards["ref_linear"])

    block[-1, :] = np.nan
    result = assign_placement(block, candidates, cards["ref_linear"])

    assert np.isfinite(result.margin)
    assert result.placement == clean.placement
    assert result.n_tiles == block.shape[0] * block.shape[1] - block.shape[1]


def test_hungarian_corroborates_but_never_decides(cards) -> None:
    """A free assignment agrees on clean cards; it is reported, not obeyed."""
    grid = chart_grid(cards["names"])
    candidates = placements(grid, (6, 2))

    result = assign_placement(cards["blocks"][0], candidates, cards["ref_linear"])

    assert result.hungarian_agreement == 12
    assert result.n_tiles == 12


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_scoring_refuses_a_single_candidate(cards) -> None:
    """Without a runner-up there is no margin, so no evidence of determination."""
    grid = chart_grid(cards["names"])
    one = placements(grid, (6, 2))[:1]

    with pytest.raises(ValueError, match="at least two candidates"):
        assign_placement(cards["blocks"][0], one, cards["ref_linear"])


def test_scoring_rejects_a_shape_mismatch(cards) -> None:
    grid = chart_grid(cards["names"])
    candidates = placements(grid, (6, 2))

    with pytest.raises(ValueError, match="tiles were observed"):
        assign_placement(
                cards["blocks"][0][:4], candidates, cards["ref_linear"]
        )


def test_scoring_rejects_a_flat_observation_array(cards) -> None:
    grid = chart_grid(cards["names"])
    candidates = placements(grid, (6, 2))

    with pytest.raises(ValueError, match=r"\(nrows, ncols, 3\)"):
        assign_placement(
                cards["blocks"][0].reshape(-1, 3), candidates, cards["ref_linear"]
        )
