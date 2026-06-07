"""Object matching — per-grid-cell + IoU-greedy (supervised-scorers §C).

Pairing predicted segmentation objects with ground-truth objects, the prerequisite
for object-level supervised metrics. Two strategies:

* :func:`match_per_grid_cell` — the **default on a** ``GridImage``: assign each
  predicted object and each GT object to its grid cell (the dominant grid section the
  object overlaps), then pair within each cell. The arrayed plate's grid *is* the
  spatial prior, so no IoU tolerance is needed — a cell holds one biological colony.
* :func:`match_iou_greedy` — the tolerance-based fallback for non-gridded layouts:
  rank every candidate pair by IoU and greedily assign the highest first. A pair
  is accepted only when its IoU is **strictly greater** than ``τ``; at the default
  ``τ = 0.5`` every accepted pair therefore has IoU > 0.5, which makes the
  assignment provably **one-to-one** (no predicted or GT object can exceed 0.5 IoU
  with two disjoint counterparts), so a merge leaves a GT object unmatched and a
  split leaves a predicted object unmatched.

Both return a ``list[tuple]`` of ``(pred_label, gt_label)`` pairs: a matched pair has
both non-``None``; an unmatched predicted object is ``(pred_label, None)`` and an
unmatched GT object is ``(None, gt_label)``. ``None`` labels are never paired.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from ._metrics import iou

#: Background label in a label/objmap array — never an object, never matched.
_BACKGROUND = 0

#: Grid sections number from 1; section 0 is the gridline/inter-cell gutter and is
#: not a cell an object can be assigned to.
_NO_CELL = 0

#: One match pair: ``(pred_label, gt_label)`` with ``None`` on the unmatched side.
MatchPair = tuple[Optional[int], Optional[int]]


def _object_labels(labels: npt.NDArray[Any]) -> list[int]:
    """The non-background object labels present in a label array, sorted.

    Args:
        labels: An integer label/objmap array (``0`` is background).

    Returns:
        The distinct positive labels in ascending order.
    """
    present = np.unique(labels)
    return [int(v) for v in present if int(v) != _BACKGROUND]


def match_iou_greedy(
    pred_labels: npt.ArrayLike,
    gt_labels: npt.ArrayLike,
    tau: float = 0.5,
) -> list[MatchPair]:
    """Greedily match predicted vs. GT objects by descending IoU (one-to-one).

    Computes the IoU of every overlapping predicted–GT label pair, sorts the pairs
    by descending IoU, and assigns the highest first; a pair is accepted only when
    its IoU is **strictly greater** than ``tau`` and neither object is already
    matched. With this strict-greater rule the default ``tau = 0.5`` accepts only
    pairs whose IoU exceeds 0.5, and the result is then provably one-to-one
    (supervised-scorers §C): no object can exceed 0.5 IoU with two disjoint
    counterparts, so a merge (one predicted blob over two GT objects) can claim at
    most one GT, leaving the other GT unmatched, and a split (two predicted objects
    over one GT) leaves the losing predicted object unmatched. (A ``tau`` below 0.5
    relaxes acceptance and can break uniqueness — e.g. the ``tau = 0.4`` merge/split
    cases in the tests.)

    Args:
        pred_labels: The predicted label/objmap array (``0`` is background).
        gt_labels: The ground-truth label array, the same shape as
            ``pred_labels`` (``0`` is background).
        tau: The IoU acceptance threshold. A pair is matched only when its IoU is
            **strictly greater** than ``tau``; the default ``0.5`` guarantees a
            unique one-to-one assignment.

    Returns:
        A list of ``(pred_label, gt_label)`` pairs — matched pairs first (in
        assignment order), then ``(pred_label, None)`` for every unmatched
        predicted object and ``(None, gt_label)`` for every unmatched GT object
        (both ascending by label).
    """
    pred = np.asarray(pred_labels)
    gt = np.asarray(gt_labels)
    pred_objects = _object_labels(pred)
    gt_objects = _object_labels(gt)

    candidates: list[tuple[float, int, int]] = []
    for p in pred_objects:
        p_mask = pred == p
        for g in gt_objects:
            score = iou(p_mask, gt == g)
            if score > tau:
                candidates.append((score, p, g))
    # Highest IoU first; ties broken deterministically by (pred, gt) label.
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    matches: list[MatchPair] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for _score, p, g in candidates:
        if p in used_pred or g in used_gt:
            continue
        matches.append((p, g))
        used_pred.add(p)
        used_gt.add(g)

    matches.extend(
        (p, None) for p in pred_objects if p not in used_pred
    )
    matches.extend(
        (None, g) for g in gt_objects if g not in used_gt
    )
    return matches


def _cell_assignment(
    labels: npt.NDArray[Any], section_map: npt.NDArray[Any]
) -> dict[int, int]:
    """Assign each object label to the grid cell (section) it most overlaps.

    Args:
        labels: An integer label/objmap array (``0`` is background).
        section_map: The grid section id per pixel, the same shape as ``labels``
            (``0`` is the inter-cell gutter, sections number from ``1``).

    Returns:
        A ``{object_label: section_id}`` mapping. An object lying entirely on the
        gutter (no positive section) maps to :data:`_NO_CELL`.
    """
    assignment: dict[int, int] = {}
    for label in _object_labels(labels):
        sections = section_map[labels == label]
        values, counts = np.unique(sections, return_counts=True)
        # Pick the most-overlapped *cell* (a positive section), ignoring the
        # gutter; an object entirely on the gutter falls back to _NO_CELL.
        best_cell = _NO_CELL
        best_count = -1
        for value, count in zip(values, counts):
            cell = int(value)
            if cell == _NO_CELL:
                continue
            if count > best_count:
                best_cell, best_count = cell, int(count)
        assignment[label] = best_cell
    return assignment


def match_per_grid_cell(image: Any, gt: npt.ArrayLike) -> list[MatchPair]:
    """Match predicted vs. GT objects per grid cell on a ``GridImage``.

    Each predicted object (from ``image.objmap``) and each GT object (from ``gt``)
    is assigned to the grid cell it most overlaps (``image.grid.get_section_map``),
    then objects are paired **within each cell** — the arrayed plate's grid is the
    spatial prior, so no IoU tolerance is applied (supervised-scorers §C). A cell
    normally holds one colony, so the within-cell pairing is a direct pairing; when
    a cell holds several objects on each side they are paired by descending IoU
    (the same one-to-one rule as :func:`match_iou_greedy`, with ``τ = 0`` so every
    co-located pair is eligible). Objects with no counterpart in their cell are
    returned unmatched. An object lying **entirely on the inter-cell gutter**
    (grid section ``0``) belongs to no colony pocket; it is excluded from the
    per-cell pairing and returned unmatched (``(label, None)`` /
    ``(None, label)``) rather than scored as a spurious cross-gutter pair.

    Args:
        image: A ``GridImage`` exposing ``objmap`` (the predicted labels) and
            ``grid.get_section_map()`` (the per-pixel grid cell id).
        gt: The ground-truth label array, the same shape as ``image.objmap``
            (``0`` is background).

    Returns:
        A list of ``(pred_label, gt_label)`` pairs (same convention as
        :func:`match_iou_greedy`): matched pairs, then unmatched predicted
        objects ``(pred_label, None)``, then unmatched GT objects
        ``(None, gt_label)``.
    """
    pred = np.asarray(image.objmap[:])
    gt_arr = np.asarray(gt)
    section_map = np.asarray(image.grid.get_section_map())

    pred_cells = _cell_assignment(pred, section_map)
    gt_cells = _cell_assignment(gt_arr, section_map)

    # Exclude the gutter (``_NO_CELL`` / 0): objects lying entirely on the
    # inter-cell gutter are not in any colony pocket, so they are not scored as
    # spurious pairs. They surface as unmatched (``(label, None)`` /
    # ``(None, label)``) — never paired across the gutter — consistent with the
    # module's "``None`` labels are never paired" contract.
    cells = sorted(
        (set(pred_cells.values()) | set(gt_cells.values())) - {_NO_CELL}
    )
    matches: list[MatchPair] = []
    for cell in cells:
        pred_here = [lab for lab, c in pred_cells.items() if c == cell]
        gt_here = [lab for lab, c in gt_cells.items() if c == cell]
        matches.extend(
            _pair_within_cell(pred, gt_arr, sorted(pred_here), sorted(gt_here))
        )
    # Carry gutter-only objects through as unmatched (never paired across the
    # gutter), preserving the (matched, unmatched-pred, unmatched-gt) ordering.
    matches.extend(
        (lab, None) for lab, c in sorted(pred_cells.items()) if c == _NO_CELL
    )
    matches.extend(
        (None, lab) for lab, c in sorted(gt_cells.items()) if c == _NO_CELL
    )
    return matches


def _pair_within_cell(
    pred: npt.NDArray[Any],
    gt: npt.NDArray[Any],
    pred_here: list[int],
    gt_here: list[int],
) -> list[MatchPair]:
    """Pair the objects sharing one grid cell by descending IoU (τ = 0).

    Args:
        pred: The full predicted label array.
        gt: The full GT label array.
        pred_here: Predicted object labels assigned to this cell.
        gt_here: GT object labels assigned to this cell.

    Returns:
        The within-cell ``(pred_label, gt_label)`` pairs, with unmatched objects
        on either side carried as ``(label, None)`` / ``(None, label)``.
    """
    candidates: list[tuple[float, int, int]] = []
    for p in pred_here:
        p_mask = pred == p
        for g in gt_here:
            # τ = 0: any positive co-located overlap is an eligible pair; a colony
            # is alone in its cell in the common case, so this just pairs them.
            score = iou(p_mask, gt == g)
            if score > 0.0:
                candidates.append((score, p, g))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))

    matches: list[MatchPair] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for _score, p, g in candidates:
        if p in used_pred or g in used_gt:
            continue
        matches.append((p, g))
        used_pred.add(p)
        used_gt.add(g)

    matches.extend((p, None) for p in pred_here if p not in used_pred)
    matches.extend((None, g) for g in gt_here if g not in used_gt)
    return matches
