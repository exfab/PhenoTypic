"""4.1 — Dice/IoU region-overlap metrics on boolean masks (supervised-scorers §A).

Pure NumPy, no GT-loader coupling. The §A.5 empty-mask convention: two empty masks
agree perfectly (1.0); disjoint masks score 0.0. One region metric *pair* only — no
redundant panel.
"""
from __future__ import annotations

import numpy as np
import pytest

from phenotypic.data import load_synth_yeast_plate
from phenotypic.tune._scoring._metrics import dice, iou


def test_identical_masks_score_one():
    mask = np.array([[True, True], [False, True]])
    assert dice(mask, mask) == pytest.approx(1.0)
    assert iou(mask, mask) == pytest.approx(1.0)


def test_disjoint_masks_score_zero():
    a = np.array([[True, True], [False, False]])
    b = np.array([[False, False], [True, True]])
    assert dice(a, b) == 0.0
    assert iou(a, b) == 0.0


def test_both_empty_masks_score_one():
    # §A.5: two empty predictions perfectly agree (no false positives/negatives).
    empty = np.zeros((3, 3), dtype=bool)
    assert dice(empty, empty) == pytest.approx(1.0)
    assert iou(empty, empty) == pytest.approx(1.0)


def test_one_empty_one_nonempty_scores_zero():
    empty = np.zeros((3, 3), dtype=bool)
    full = np.ones((3, 3), dtype=bool)
    assert dice(empty, full) == 0.0
    assert iou(empty, full) == 0.0
    assert dice(full, empty) == 0.0
    assert iou(full, empty) == 0.0


def test_known_half_overlap_values():
    # a: 4 px; b: 4 px; intersection 2 px; union 6 px.
    a = np.array([[True, True, False, False]])
    b = np.array([[False, True, True, False]])
    # wait: |a|=2, |b|=2, intersection=1, union=3
    assert int(a.sum()) == 2 and int(b.sum()) == 2
    assert dice(a, b) == pytest.approx(2 * 1 / (2 + 2))  # 0.5
    assert iou(a, b) == pytest.approx(1 / 3)


def test_dice_iou_relationship():
    # Dice = 2·IoU / (1 + IoU) for any overlap.
    rng = np.random.default_rng(0)
    a = rng.random((20, 20)) > 0.5
    b = rng.random((20, 20)) > 0.5
    d, j = dice(a, b), iou(a, b)
    assert d == pytest.approx(2 * j / (1 + j))


def test_non_boolean_input_is_coerced():
    # Integer / float arrays are treated as truthy masks.
    a = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    b = np.array([[1.0, 1.0], [0.0, 1.0]])
    assert dice(a, b) == pytest.approx(1.0)
    assert iou(a, b) == pytest.approx(1.0)


def test_self_overlap_on_synth_plate_is_perfect():
    # A real plate's foreground compared to itself scores 1.0 (sanity, no GT).
    image = load_synth_yeast_plate()
    mask = np.asarray(image.objmap[:]) > 0
    assert dice(mask, mask) == pytest.approx(1.0)
    assert iou(mask, mask) == pytest.approx(1.0)
