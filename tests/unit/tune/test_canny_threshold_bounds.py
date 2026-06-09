"""Phase 3: ``CannyDetector`` low/high threshold search windows can't cross.

The two ``TuneSpec`` windows are deliberately non-overlapping
(``low ∈ [0.05, 0.2]`` and ``high ∈ [0.2, 0.4]``) so the optimizer can never
sample ``low > high`` — a degenerate Canny config. The constructor params are
unchanged (no derived/delta field), so serialized pipelines keep round-tripping.
A belt-and-suspenders ``model_validator`` rejects a crossed pair on a
hand-constructed detector without rejecting the existing defaults.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from phenotypic import ImagePipeline
from phenotypic.detect import CannyDetector
from phenotypic.tune import infer_search_space


def _threshold_windows():
    """Return the inferred ``(low_domain, high_domain)`` for a Canny detector."""
    space = infer_search_space(ImagePipeline(ops=[CannyDetector()]))
    low = next(k for k in space.knobs if k.key == "0.low_threshold").domain
    high = next(k for k in space.knobs if k.key == "0.high_threshold").domain
    return low, high


def test_inferred_windows_are_non_overlapping():
    # The whole point: low's *upper* edge does not exceed high's *lower* edge,
    # so no sampled (low, high) pair can satisfy low > high.
    low, high = _threshold_windows()
    assert (low.low, low.high) == (0.05, 0.2)
    assert (high.low, high.high) == (0.2, 0.4)
    assert low.high <= high.low


def test_windows_cannot_produce_low_above_high():
    # Exhaustive over the window edges: max(low) == min(high) == 0.2, so the
    # worst case is low == high, never low > high.
    low, high = _threshold_windows()
    assert max(low.low, low.high) <= min(high.low, high.high)


def test_defaults_remain_valid():
    det = CannyDetector()
    assert det.low_threshold == 0.1
    assert det.high_threshold == 0.2
    # The default high == low edge case (0.2 vs 0.1) is strictly ordered.
    assert det.high_threshold > det.low_threshold


def test_validator_rejects_crossed_thresholds():
    with pytest.raises(ValidationError):
        CannyDetector(low_threshold=0.3, high_threshold=0.2)


def test_validator_rejects_equal_thresholds():
    # high == low is still degenerate for hysteresis (no separating band).
    with pytest.raises(ValidationError):
        CannyDetector(low_threshold=0.2, high_threshold=0.2)


def test_validator_accepts_a_manual_valid_pair():
    det = CannyDetector(low_threshold=0.15, high_threshold=0.35)
    assert det.low_threshold == 0.15
    assert det.high_threshold == 0.35


def test_threshold_change_preserves_serialization_roundtrip():
    # Non-overlapping windows are the fix (not a derived field), so the public
    # constructor params survive a JSON round-trip unchanged.
    det = CannyDetector(low_threshold=0.12, high_threshold=0.3)
    rebuilt = CannyDetector.model_validate(det.model_dump(mode="json"))
    assert rebuilt.low_threshold == 0.12
    assert rebuilt.high_threshold == 0.3
