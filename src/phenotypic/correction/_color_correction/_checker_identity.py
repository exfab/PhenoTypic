"""Recover which chart patch sits in each detected tile.

Nothing upstream knows what the tiles are: the ROI is a rectangle and the
lattice is geometry.  This module decides identity from colour, and the fit
consumes its answer directly -- so a wrong answer here does not cost a quality
counter, it puts the wrong reference colour on two rows of the least-squares
input.  Two choices follow from that.

**Score whole placements, never a free per-tile assignment.**  A free
permutation lets each label wander onto whichever reference fits it best, so an
occluded or contaminated tile is silently relabelled.  A placement -- which
contiguous block of the chart the tiles occupy, in which orientation -- is
decided by every tile at once, so one bad tile cannot move it.  The two
genuinely similar patch pairs on a ColorChecker 24 (purplish blue against blue
flower; the two mid-greys) swap freely under per-tile matching and not at all
under placement scoring.  A free assignment still runs, but only as a
corroborating signal.

**Score in gain-invariant features.**  Chromaticity plus relative luminance, so
an exposure difference between the card and the reference cannot choose the
answer.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple, Sequence

import numpy as np

#: Grid shape of a chart, keyed by patch count.  Used when the caller does not
#: name one explicitly.
CHART_SHAPES: dict[int, tuple[int, int]] = {24: (4, 6)}

#: Placement margin below which the placement is not determined by the tiles
#: that voted, and the card must be refused rather than guessed at.
#:
#: There is deliberately **no fallback to a declared layout** -- the operation
#: is never given one.  A refusal is visible; a mislabelled fit is not.
MIN_PLACEMENT_MARGIN = 0.20

#: Below this the placement is weaker than any clean card observed and is
#: worth a warning, though still used.
WARN_PLACEMENT_MARGIN = 0.25

#: Free-assignment disagreements beyond which the tiles are worth a warning:
#: something about them does not look like the patch the geometry implies.
MAX_HUNGARIAN_DISAGREEMENT = 2


class Placement(NamedTuple):
    """One discrete way a tile block can sit on the chart.

    Attributes:
        name: Human-readable description, e.g. ``"rows 1-2, flipped"``.
        names: ``names[tile_row][tile_col]`` -> chart patch name.
    """

    name: str
    names: tuple[tuple[str, ...], ...]


class PlacementResult(NamedTuple):
    """Outcome of placement scoring.

    Attributes:
        placement: The winning :class:`Placement`.
        score: Its mean feature distance; lower is better.
        runner_up: The second-best score.
        margin: ``runner_up - score``.  The gate is on this, not on *score*.
        hungarian_agreement: How many tiles a free assignment labels the same
            way as the winner.
        n_tiles: Tiles that voted.
    """

    placement: Placement
    score: float
    runner_up: float
    margin: float
    hungarian_agreement: int
    n_tiles: int

    @property
    def determined(self) -> bool:
        """Whether the margin clears :data:`MIN_PLACEMENT_MARGIN`."""
        return self.margin >= MIN_PLACEMENT_MARGIN


def chart_grid(
        patch_names: Sequence[str],
        shape: tuple[int, int] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Lay chart patch names out on their grid, row-major.

    Args:
        patch_names: Patch names in the chart's canonical order, as
            ``colour.CCS_COLOURCHECKERS`` gives them.
        shape: ``(rows, cols)``.  Inferred from the patch count when omitted.

    Returns:
        ``grid[row][col]`` -> patch name.

    Raises:
        ValueError: If the shape is unknown or does not match the count.
    """
    if shape is None:
        shape = CHART_SHAPES.get(len(patch_names))
        if shape is None:
            raise ValueError(
                    f"No known grid shape for a {len(patch_names)}-patch chart; "
                    "pass shape=(rows, cols) explicitly."
            )
    rows, cols = shape
    if rows * cols != len(patch_names):
        raise ValueError(
                f"Chart shape {shape} does not hold {len(patch_names)} patches."
        )
    return tuple(
            tuple(patch_names[r * cols : (r + 1) * cols]) for r in range(rows)
    )


def placements(
        grid: Sequence[Sequence[str]],
        tile_shape: tuple[int, int],
) -> list[Placement]:
    """Every way a tile block of *tile_shape* can sit on *grid*.

    A detected block occupies a **contiguous** sub-block of the chart -- a card
    cut in half gives two adjacent rows, never rows 1 and 3 -- so only
    contiguous placements are enumerated, in both transpositions and all four
    reflections.  For a 6x2 block on a 4x6 chart that is 12 hypotheses.

    Args:
        grid: ``grid[row][col]`` -> patch name, from :func:`chart_grid`.
        tile_shape: ``(nrows, ncols)`` of the detected tile block.

    Returns:
        Every admissible :class:`Placement`, de-duplicated.  Empty when the
        block cannot fit the chart in any orientation.
    """
    chart = np.array([list(row) for row in grid], dtype=object)
    chart_rows, chart_cols = chart.shape
    n_rows, n_cols = tile_shape

    found: dict[tuple[tuple[str, ...], ...], Placement] = {}
    for transposed in (False, True):
        block_rows, block_cols = (n_cols, n_rows) if transposed else (n_rows, n_cols)
        if block_rows > chart_rows or block_cols > chart_cols:
            continue
        for row0 in range(chart_rows - block_rows + 1):
            for col0 in range(chart_cols - block_cols + 1):
                window = chart[row0 : row0 + block_rows, col0 : col0 + block_cols]
                for flip_rows in (False, True):
                    for flip_cols in (False, True):
                        placed = window.T if transposed else window
                        if flip_rows:
                            placed = placed[::-1]
                        if flip_cols:
                            placed = placed[:, ::-1]
                        names = tuple(tuple(r) for r in placed.tolist())
                        if names in found:
                            continue
                        label = (
                            f"rows {row0 + 1}-{row0 + block_rows}, "
                            f"cols {col0 + 1}-{col0 + block_cols}"
                            f"{' transposed' if transposed else ''}"
                            f"{' flip-rows' if flip_rows else ''}"
                            f"{' flip-cols' if flip_cols else ''}"
                        )
                        found[names] = Placement(label, names)
    return list(found.values())


def identity_features(linear_rgb: np.ndarray) -> np.ndarray:
    """Gain-invariant colour features: chromaticity plus relative luminance.

    An exposure or illumination difference scales all three channels together,
    which chromaticity divides out and the luminance normalisation absorbs.
    The placement is therefore chosen by colour *relationships*, not by how
    bright the card happened to be.

    Args:
        linear_rgb: ``(N, 3)`` **linear** RGB.  Gamma-encoded input silently
            gives different features and a smaller margin.

    Returns:
        ``(N, 3)`` feature vectors.
    """
    values = np.asarray(linear_rgb, dtype=np.float64)
    total = values.sum(axis=1, keepdims=True) + 1e-9
    return np.column_stack(
            [values[:, :2] / total * 3.0, values[:, 1] / (values[:, 1].max() + 1e-9)]
    )


def assign_placement(
        observed_linear: np.ndarray,
        candidates: Sequence[Placement],
        reference_linear: Mapping[str, np.ndarray],
        voting: np.ndarray | None = None,
) -> PlacementResult:
    """Choose the placement that best explains the observed tile colours.

    Args:
        observed_linear: ``(nrows, ncols, 3)`` linear RGB, one entry per tile,
            indexed the way the lattice yields them.
        candidates: Placements to score, from :func:`placements`.
        reference_linear: Patch name -> reference linear RGB.
        voting: Optional ``(nrows, ncols)`` boolean mask of tiles allowed to
            vote.  Every tile is still labelled.  Restricting the vote is
            rarely worth it: six same-row tiles do not constrain the
            orientation, and a restricted vote has been observed to pick the
            wrong placement at a margin low enough that only the gate caught
            it.

    Returns:
        A :class:`PlacementResult`.

    Raises:
        ValueError: If there are fewer than two candidates to compare, if the
            observed array is the wrong shape, or if a placement names a patch
            the reference does not have.
    """
    observed = np.asarray(observed_linear, dtype=np.float64)
    if observed.ndim != 3 or observed.shape[-1] != 3:
        raise ValueError(
                f"observed_linear must be (nrows, ncols, 3); got {observed.shape}."
        )
    if len(candidates) < 2:
        raise ValueError(
                "Placement scoring needs at least two candidates: the margin to "
                "the runner-up is the only evidence that the answer is determined."
        )
    n_rows, n_cols = observed.shape[:2]
    flat_observed = observed.reshape(-1, 3)
    # A tile whose box missed the ROI measured nothing. Excluding it only
    # from the vote is not enough: luminance is normalised by the block's
    # brightest tile, so a NaN anywhere would make every feature NaN.
    finite = np.isfinite(flat_observed).all(axis=1)
    obs_features = np.full_like(flat_observed, np.nan)
    obs_features[finite] = identity_features(flat_observed[finite])

    mask = (
        np.ones(n_rows * n_cols, dtype=bool)
        if voting is None
        else np.asarray(voting, dtype=bool).reshape(-1)
    ) & finite
    if not mask.any():
        raise ValueError("No tiles are allowed to vote.")

    scored: list[tuple[float, Placement]] = []
    for placement in candidates:
        if (len(placement.names), len(placement.names[0])) != (n_rows, n_cols):
            raise ValueError(
                    f"Placement {placement.name!r} is "
                    f"{len(placement.names)}x{len(placement.names[0])}, but "
                    f"{n_rows}x{n_cols} tiles were observed."
            )
        try:
            reference = np.vstack(
                    [reference_linear[name] for row in placement.names for name in row]
            )
        except KeyError as exc:  # pragma: no cover - guards a caller mistake
            raise ValueError(
                    f"Placement {placement.name!r} names patch {exc} which the "
                    "reference chart does not have."
            ) from None
        ref_features = np.full_like(reference, np.nan)
        ref_features[finite] = identity_features(reference[finite])
        distance = np.linalg.norm(
                obs_features[mask] - ref_features[mask], axis=1
        ).mean()
        scored.append((float(distance), placement))

    scored.sort(key=lambda item: item[0])
    best_score, best = scored[0]
    runner_up = scored[1][0]

    return PlacementResult(
            placement=best,
            score=best_score,
            runner_up=runner_up,
            margin=runner_up - best_score,
            hungarian_agreement=_hungarian_agreement(
                    obs_features, best, reference_linear, finite
            ),
            n_tiles=int(mask.sum()),
    )


def _hungarian_agreement(
        obs_features: np.ndarray,
        placement: Placement,
        reference_linear: Mapping[str, np.ndarray],
        rows: np.ndarray,
) -> int:
    """Tiles a free assignment labels the same way as *placement*.

    Corroboration only.  This must never decide identity: a free permutation
    relabels an occluded tile onto whatever reference happens to fit it.
    Only tiles flagged in *rows* take part.
    """
    from scipy.optimize import linear_sum_assignment

    names = [name for row in placement.names for name in row]
    names = [name for name, keep in zip(names, rows) if keep]
    reference = np.vstack([reference_linear[name] for name in names])
    ref_features = identity_features(reference)
    cost = np.linalg.norm(
            obs_features[rows][:, None, :] - ref_features[None, :, :], axis=2
    )
    assigned_rows, assigned_cols = linear_sum_assignment(cost)
    return int(sum(1 for r, c in zip(assigned_rows, assigned_cols) if r == c))
