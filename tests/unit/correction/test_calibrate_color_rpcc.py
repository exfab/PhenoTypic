from __future__ import annotations

import numpy as np
import pytest

from phenotypic.correction import CalibrateColorRpcc, CheckerRoi, ColorCorrector


def test_rois_are_required() -> None:
    with pytest.raises(ValueError):
        CalibrateColorRpcc()


def test_bbox_shorthand_is_coerced_and_stored_canonically() -> None:
    operation = CalibrateColorRpcc(
            rois=[[1170, 0, 2840, 340],
                  CheckerRoi(row=(1170, 2840), col=(5856, 6016))],
    )

    assert all(isinstance(roi, CheckerRoi) for roi in operation.rois)
    assert operation.rois[0] == CheckerRoi(row=(1170, 2840), col=(0, 340))


def test_serialisation_round_trips_the_rectangles() -> None:
    """A saved configuration must reproduce the same rectangles.

    ``ColorCheckerProfile.rois`` is ``exclude=True`` and does not round-trip;
    here the ROIs are the defining input, so they must.
    """
    operation = CalibrateColorRpcc(rois=[[1170, 0, 2840, 340]])

    restored = CalibrateColorRpcc.model_validate(operation.model_dump())

    assert restored.rois == operation.rois


def test_medoid_candidates_must_be_positive_at_construction() -> None:
    """Rejected when the operation is built, not on the first tile at apply time.

    ``MeasureColor.medoid_candidates`` is the same knob on the same estimator,
    so both share one validated type.
    """
    from phenotypic.measure import MeasureColor

    with pytest.raises(ValueError, match="medoid_candidates"):
        CalibrateColorRpcc(rois=[[0, 0, 10, 10]], medoid_candidates=0)
    with pytest.raises(ValueError, match="medoid_candidates"):
        MeasureColor(medoid_candidates=0)
    assert (
        CalibrateColorRpcc.model_fields["medoid_candidates"].metadata
        == MeasureColor.model_fields["medoid_candidates"].metadata
    )


def test_degree_is_a_plain_integer_with_no_auto_mode() -> None:
    """Fixed for the whole run; never adapted to what a frame detected."""
    assert CalibrateColorRpcc(rois=[[0, 0, 10, 10]]).degree == 3
    with pytest.raises(ValueError):
        CalibrateColorRpcc(rois=[[0, 0, 10, 10]], degree="auto")


@pytest.mark.parametrize("degree", [0, 5, -1])
def test_an_undefined_degree_is_rejected(degree: int) -> None:
    with pytest.raises(ValueError, match="degree must be 1-4"):
        CalibrateColorRpcc(rois=[[0, 0, 10, 10]], degree=degree)


def test_unknown_fields_are_rejected() -> None:
    """Strict construction: a layout kwarg is a typo, not a feature."""
    with pytest.raises(ValueError):
        CalibrateColorRpcc(rois=[[0, 0, 10, 10]], layout=(6, 2))


def test_it_is_a_different_operation_from_colorcorrector() -> None:
    """One derives the correction and applies it; the other only applies."""
    assert not issubclass(CalibrateColorRpcc, ColorCorrector)
    assert "profile" not in CalibrateColorRpcc.model_fields


def test_an_unfitted_operation_has_no_profile() -> None:
    operation = CalibrateColorRpcc(rois=[[0, 0, 10, 10]])

    assert operation.fitted_profile is None
    assert operation.qc == []


def test_it_refuses_an_roi_that_holds_no_card() -> None:
    from phenotypic import Image

    rng = np.random.default_rng(0)
    flat = (rng.normal(120, 2, (200, 200, 3))).clip(0, 255).astype(np.uint8)
    operation = CalibrateColorRpcc(rois=[[0, 0, 200, 200]])

    with pytest.raises(Exception, match="No patch columns|too short|cannot sit"):
        operation.apply(Image(arr=flat))
