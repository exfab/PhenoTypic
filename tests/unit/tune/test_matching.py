"""4.2 — object matching: per-grid-cell + IoU-greedy (supervised-scorers §C).

``match_per_grid_cell`` assigns each predicted/GT object to its grid cell on a
``GridImage`` and pairs them per cell (no tolerance — the grid *is* the assignment).
``match_iou_greedy`` sorts candidate pairs by descending IoU and assigns the highest
first, which is provably one-to-one at ``τ > 0.5``. A merge leaves a GT object
unmatched; a split leaves a predicted object unmatched.
"""
from __future__ import annotations

import numpy as np

from phenotypic.data import load_synth_yeast_plate
from phenotypic.tune._scoring._matching import (
    match_iou_greedy,
    match_per_grid_cell,
)


# --------------------------------------------------------------------------- #
# match_per_grid_cell — on a real GridImage
# --------------------------------------------------------------------------- #
def test_grid_cell_self_match_pairs_every_object():
    # Predicted == GT == the plate's own objmap → every object matches itself.
    image = load_synth_yeast_plate()
    gt = np.asarray(image.objmap[:])
    matches = match_per_grid_cell(image, gt)
    # One pair per object, all matched (no None on either side).
    paired = [(p, g) for (p, g) in matches if p is not None and g is not None]
    assert len(paired) == int(image.num_objects)
    assert all(p is not None and g is not None for p, g in matches)


def test_grid_cell_missing_gt_object_is_unmatched_pred():
    # GT drops one object's pixels → that cell's predicted object is unmatched.
    image = load_synth_yeast_plate()
    pred = np.asarray(image.objmap[:])
    dropped_label = int(np.unique(pred[pred > 0])[0])
    gt = pred.copy()
    gt[gt == dropped_label] = 0  # remove this object from the GT
    matches = match_per_grid_cell(image, gt)
    unmatched_pred = [p for (p, g) in matches if g is None]
    assert dropped_label in unmatched_pred


def test_grid_cell_extra_gt_object_is_unmatched_gt():
    # An extra GT object whose cell has no predicted counterpart → unmatched GT.
    image = load_synth_yeast_plate()
    pred = np.asarray(image.objmap[:])
    # Remove one predicted object entirely, but keep it (relabeled) in GT — the
    # GT object now lives in a cell with no predicted object.
    dropped_label = int(np.unique(pred[pred > 0])[0])
    extra_label = int(pred.max()) + 1
    pred_cleared = pred.copy()
    pred_cleared[pred_cleared == dropped_label] = 0
    gt = pred.copy()
    gt[gt == dropped_label] = extra_label  # GT-only object in that cell
    matches = match_per_grid_cell(pred_image_wrap(image, pred_cleared), gt)
    unmatched_gt = [g for (p, g) in matches if p is None]
    assert extra_label in unmatched_gt


# --------------------------------------------------------------------------- #
# match_iou_greedy — label arrays
# --------------------------------------------------------------------------- #
def test_iou_greedy_unique_assignment_above_tau():
    # Two predicted objects, two GT objects, near-perfect overlap → 1:1.
    pred = np.array([[1, 1, 0, 2, 2]])
    gt = np.array([[1, 1, 0, 2, 2]])
    matches = match_iou_greedy(pred, gt, tau=0.5)
    paired = sorted((p, g) for (p, g) in matches if p is not None and g is not None)
    assert paired == [(1, 1), (2, 2)]


def test_iou_greedy_is_one_to_one():
    # No predicted label and no GT label appears in more than one match.
    pred = np.array([[1, 1, 1, 0, 2, 2]])
    gt = np.array([[1, 1, 0, 0, 2, 2]])
    matches = match_iou_greedy(pred, gt, tau=0.5)
    preds = [p for (p, g) in matches if p is not None and g is not None]
    gts = [g for (p, g) in matches if p is not None and g is not None]
    assert len(preds) == len(set(preds))
    assert len(gts) == len(set(gts))


def test_iou_greedy_below_tau_leaves_both_unmatched():
    # Overlap is 1/5 < 0.5 → no accepted pair; both objects unmatched.
    pred = np.array([[1, 1, 1, 1, 0, 0]])
    gt = np.array([[0, 0, 0, 1, 1, 1]])
    matches = match_iou_greedy(pred, gt, tau=0.5)
    assert (1, None) in matches
    assert (None, 1) in matches
    assert not any(p is not None and g is not None for p, g in matches)


def test_iou_greedy_merge_leaves_gt_unmatched():
    # One predicted blob covers two GT objects (a merge). At τ>0.5 the predicted
    # object can win at most one GT; the other GT is left unmatched.
    pred = np.array([[1, 1, 1, 1, 1, 1]])
    gt = np.array([[1, 1, 1, 2, 2, 2]])
    matches = match_iou_greedy(pred, gt, tau=0.4)
    matched_gts = {g for (p, g) in matches if p is not None and g is not None}
    unmatched_gts = {g for (p, g) in matches if p is None}
    assert len(matched_gts) == 1
    assert unmatched_gts  # the merged-away GT object is unmatched


def test_iou_greedy_split_leaves_pred_unmatched():
    # Two predicted objects split one GT object. Only one predicted object can
    # claim the GT at τ>0.5; the other predicted object is unmatched.
    pred = np.array([[1, 1, 1, 2, 2, 2]])
    gt = np.array([[1, 1, 1, 1, 1, 1]])
    matches = match_iou_greedy(pred, gt, tau=0.4)
    matched_preds = {p for (p, g) in matches if p is not None and g is not None}
    unmatched_preds = {p for (p, g) in matches if g is None}
    assert len(matched_preds) == 1
    assert unmatched_preds


# --------------------------------------------------------------------------- #
# match_per_grid_cell — the gutter (section 0) is excluded from pairing
# --------------------------------------------------------------------------- #
class _FakeGridImage:
    """A minimal duck-typed GridImage with a fully controlled section map."""

    def __init__(self, labels: np.ndarray, sections: np.ndarray) -> None:
        self.objmap = _ArrayAccessor(labels)
        self.grid = _FakeGrid(sections)


class _FakeGrid:
    def __init__(self, sections: np.ndarray) -> None:
        self._sections = sections

    def get_section_map(self) -> np.ndarray:
        return self._sections


def test_grid_cell_gutter_only_objects_are_not_paired():
    # Two objects share the inter-cell gutter (section 0): a predicted blob and a
    # GT blob over the SAME gutter pixels. Before the fix the gutter id leaked
    # into the cell loop and these were scored as a (spurious) matched pair; now
    # the gutter is excluded, so each surfaces as unmatched and they are never
    # paired across the gutter. A real colony in cell 1 still matches normally.
    section_map = np.array([[0, 0, 1, 1, 0, 0]])  # cols 0,1,4,5 = gutter
    pred = np.array([[7, 7, 5, 5, 0, 0]])  # obj 7 in gutter, obj 5 in cell 1
    gt = np.array([[8, 8, 5, 5, 0, 0]])  # obj 8 in gutter (overlaps 7), 5 in cell 1
    matches = match_per_grid_cell(_FakeGridImage(pred, section_map), gt)

    # The cross-gutter pair must NOT appear despite the gutter pixels overlapping.
    assert (7, 8) not in matches
    # Gutter-only objects come through as unmatched.
    assert (7, None) in matches
    assert (None, 8) in matches
    # The genuine in-cell colony still matches one-to-one.
    assert (5, 5) in matches


def test_grid_cell_gutter_only_objects_excluded_even_without_real_cells():
    # If EVERY object lies on the gutter, no pairs are produced — they are all
    # returned unmatched rather than paired with each other.
    section_map = np.array([[0, 0, 0, 0]])  # all gutter
    pred = np.array([[1, 1, 0, 0]])
    gt = np.array([[2, 2, 0, 0]])  # overlaps pred 1 exactly, but on the gutter
    matches = match_per_grid_cell(_FakeGridImage(pred, section_map), gt)
    assert not any(p is not None and g is not None for p, g in matches)
    assert (1, None) in matches
    assert (None, 2) in matches


# --------------------------------------------------------------------------- #
# helper: wrap a GridImage so it reports a custom predicted objmap
# --------------------------------------------------------------------------- #
class _PredObjmapWrapper:
    """Forwards grid access to a real GridImage but serves a custom objmap."""

    def __init__(self, image, pred_objmap):
        self._image = image
        self._pred = np.asarray(pred_objmap)

    @property
    def grid(self):
        return self._image.grid

    @property
    def objmap(self):
        return _ArrayAccessor(self._pred)

    @property
    def num_objects(self):
        labels = np.unique(self._pred)
        return int(np.sum(labels != 0))


class _ArrayAccessor:
    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, key):
        return self._arr[key]


def pred_image_wrap(image, pred_objmap):
    return _PredObjmapWrapper(image, pred_objmap)
