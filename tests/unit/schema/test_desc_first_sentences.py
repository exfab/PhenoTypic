"""The first sentence of every ``desc`` reads as a plain technical sentence.

The Measurements table shows the first sentence of ``desc`` as "What it is"
(design §8), so a first sentence must not be an RST math role, must not be cut
short by an abbreviation, and must say what the code computes. The splitter
below is the one the DataTables spike used
(``spike/docs-datatables:docs/source/_spike/make_rows.py``); the table generator
is expected to cut the same way.
"""

from __future__ import annotations

import inspect
import re

import pytest

import phenotypic.schema as schema
from phenotypic.schema import (
    EDGE_CORRECTION,
    GRID,
    INTENSITY,
    LOG_GROWTH_MODEL,
    MODEL_METRICS,
    QUALITY_OCCUPANCY,
    SHAPE,
    TEXTURE,
    MeasurementInfo,
)

_ROLE = re.compile(r":[a-zA-Z0-9_.:]+:`([^`]+)`")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _first_sentence(desc: str) -> str:
    text = " ".join(_ROLE.sub(r"\1", desc).split())
    return _SENTENCE_END.split(text, maxsplit=1)[0]


def _all_members() -> list:
    classes = {
        obj
        for name in schema.__all__
        if inspect.isclass(obj := getattr(schema, name))
        and issubclass(obj, MeasurementInfo)
    }
    ordered = sorted(classes, key=lambda c: c.__name__)
    return [member for cls in ordered for member in cls]


@pytest.mark.parametrize("member", _all_members(), ids=str)
def test_first_sentence_is_not_cut_short_or_a_math_role(member) -> None:
    collapsed = " ".join(member.desc.split())
    raw_first = _SENTENCE_END.split(collapsed, maxsplit=1)[0]
    assert ":math:" not in raw_first
    assert not _first_sentence(member.desc).endswith(("e.g.", "i.e."))


@pytest.mark.parametrize(
    "member",
    [
        INTENSITY.INTEGRATED_INTENSITY,
        INTENSITY.MINIMUM_INTENSITY,
        INTENSITY.MAXIMUM_INTENSITY,
        INTENSITY.MEAN_INTENSITY,
        INTENSITY.MEDIAN_INTENSITY,
        INTENSITY.STANDARD_DEVIATION_INTENSITY,
        INTENSITY.COEFFICIENT_VARIANCE_INTENSITY,
        INTENSITY.Q1_INTENSITY,
        INTENSITY.Q3_INTENSITY,
        INTENSITY.IQR_INTENSITY,
    ],
)
def test_intensity_first_sentence_names_what_is_summarized(member) -> None:
    # "The mean intensity of the object" told the reader nothing the column
    # name did not; MeasureIntensity summarizes the gray layer's values.
    assert "grayscale" in _first_sentence(member.desc).lower()


@pytest.mark.parametrize("member", list(TEXTURE), ids=str)
def test_texture_first_sentence_is_words_not_a_formula(member) -> None:
    first = _first_sentence(member.desc)
    assert not any(symbol in first for symbol in ("Σ", "√", "p(i,j)"))


def test_orientation_is_measured_from_the_vertical_axis() -> None:
    # skimage regionprops measures orientation from axis 0 (rows): a colony
    # elongated left to right reads about pi/2, not 0.
    first = _first_sentence(SHAPE.ORIENTATION.desc)
    assert "vertical" in first
    assert "horizontal axis" not in first


def test_diff_variance_says_it_is_the_variance_of_the_probabilities() -> None:
    # mahotas' default feature 10 is var(p_{x-y}), not the variance of |i - j|.
    assert "probabilities" in _first_sentence(TEXTURE.DIFFERENCE_VARIANCE.desc)


def test_log_growth_beta_compares_k_with_the_last_time_point() -> None:
    # LogGrowthModel._loss_func uses y at the largest t, not max(y).
    assert "last time point" in LOG_GROWTH_MODEL.BETA.desc
    assert "largest measurement" not in LOG_GROWTH_MODEL.BETA.desc


def test_log_growth_lambda_names_the_rate_it_penalizes() -> None:
    # The penalty is on dN/dt = r*K/4, the maximum absolute rate.
    assert "specific growth rate" not in LOG_GROWTH_MODEL.LAM.desc
    assert "K·r/4" in LOG_GROWTH_MODEL.LAM.desc


def test_vacant_is_floored_at_zero() -> None:
    # GridOccupancy writes max(expected - filled, 0).
    assert "floored at 0" in QUALITY_OCCUPANCY.VACANT.desc


@pytest.mark.parametrize(
    "member",
    [
        EDGE_CORRECTION.CORRECTED_CAP,
        EDGE_CORRECTION.NEW_VAL,
        GRID.ROW_NUM,
        GRID.COL_NUM,
        MODEL_METRICS.LOSS,
        MODEL_METRICS.STATUS,
    ],
)
def test_first_sentence_does_not_restate_the_label(member) -> None:
    first = _first_sentence(member.desc).lower()
    assert not first.startswith(("the new value", "the carrying capacity"))
    assert "of the object" not in first
    assert "loss of model fitting" not in first
