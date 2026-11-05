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
    ACQUISITION,
    BBOX,
    EDGE_CORRECTION,
    GRID,
    GRID_SPREAD,
    IMAGE,
    INTENSITY,
    LOG_GROWTH_MODEL,
    METADATA_MATCH,
    MODEL_METRICS,
    QUALITY_OCCUPANCY,
    QUALITY_SE,
    SHAPE,
    TEXTURE,
    ColorLab,
    MeasurementInfo,
)

_ROLE = re.compile(r":[a-zA-Z0-9_.:]+:`([^`]+)`")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_ABBREVIATIONS = ("e.g.", "i.e.", "vs.", "etc.", "approx.", "cf.")


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
    assert not _first_sentence(member.desc).endswith(_ABBREVIATIONS)


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
    # mahotas' default feature 10 is var(p_{x-y}), not the variance of |i - j|,
    # over a vector as long as the colony's highest quantized gray level + 1.
    desc = TEXTURE.DIFFERENCE_VARIANCE.desc
    assert "probabilities" in _first_sentence(desc)
    assert "highest quantized gray" in desc


def test_diff_variance_reads_high_as_uniform() -> None:
    # var(p_{x-y}) is highest when differences concentrate on a few values:
    # measured with mahotas 1.4.18, uniform 0.0040 > half smooth 0.0006 > rough 0.00002.
    desc = " ".join(TEXTURE.DIFFERENCE_VARIANCE.desc.split())
    assert "High values mean the differences concentrate on a few values" in desc
    assert "that reading runs backwards" in desc
    assert "smooth margins with wrinkled centers" not in desc


@pytest.mark.parametrize(
    "member",
    [TEXTURE.SUM_ENTROPY, TEXTURE.ENTROPY, TEXTURE.DIFFERENCE_ENTROPY],
    ids=str,
)
def test_texture_entropies_are_base_2(member) -> None:
    # mahotas' _entropy uses log2.
    assert "base-2" in _first_sentence(member.desc)


def test_correlation_compares_pixels_one_scale_apart() -> None:
    # mahotas builds the co-occurrence matrix at distance=scale.
    first = _first_sentence(TEXTURE.CORRELATION.desc)
    assert "distance apart" in first
    assert "neighboring" not in first


def test_info_correlation_1_states_its_sign() -> None:
    # (HXY - HXY1) / max(HX, HY) is minus a mutual information: never above 0.
    # The maintainer kept the earlier "Positive values" wording, because projects
    # already read the column against it; a note maps it onto the written sign.
    desc = " ".join(TEXTURE.IMC1.desc.split())
    assert "0 or below" in _first_sentence(desc)
    assert "Positive values → structured" in desc
    assert 'read "positive" there as "more negative"' in desc
    assert "Haralick (1973)" in desc


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


def test_replicate_se_uses_the_sample_standard_deviation() -> None:
    # ReplicateAgreement: values.std(ddof=1).
    assert "n - 1" in QUALITY_SE.VALUE.desc


def test_intensity_cv_carries_the_small_sample_factor() -> None:
    # MeasureFeatures._calculate_coeff_variation: (1 + 1/n) * std / mean.
    assert "1 + 1/n" in INTENSITY.COEFFICIENT_VARIANCE_INTENSITY.desc


@pytest.mark.parametrize(
    "member",
    [
        GRID.ROW_NUM,
        GRID.COL_NUM,
        BBOX.CENTER_RR,
        BBOX.CENTER_CC,
        GRID_SPREAD.OBJECT_SPREAD,
    ],
    ids=str,
)
def test_grid_cells_and_bbox_centers_are_centroids(member) -> None:
    # MeasureBounds writes skimage's centroid as Bbox_CenterRR/CC; GridFinder
    # cuts rows and columns on those columns, and MeasureGridSpread takes its
    # pairwise distances between them.
    first = _first_sentence(member.desc)
    assert "centroid" in first
    assert "center of the bounding box" not in first
    assert "bounding-box center" not in first


@pytest.mark.parametrize(
    "member", [BBOX.DIST_WEIGHTED_CENTER_RR, BBOX.DIST_WEIGHTED_CENTER_CC]
)
def test_distance_weighted_center_is_a_weighted_centroid(member) -> None:
    # MeasureBounds: ndi.center_of_mass(dt), not the distance-transform argmax.
    first = _first_sentence(member.desc)
    assert "distance-weighted centroid" in first
    assert "maximum" not in first
    # The plain-centroid fallback fires only when every pixel touches another
    # colony (its zeroed contact ring leaves no distance weight).
    assert "another colony among its eight neighbors" in member.desc


@pytest.mark.parametrize("member", [BBOX.MAX_RR, BBOX.MAX_CC], ids=str)
def test_bbox_max_bounds_are_exclusive(member) -> None:
    # skimage regionprops bbox is half-open: [min_row, max_row) and
    # [min_col, max_col), so MaxRR/MaxCC are one past the last pixel.
    rest = member.desc[len(_first_sentence(member.desc)) :]
    assert "one past" in rest
    assert "exclusive" in rest


@pytest.mark.parametrize("member", [BBOX.MIN_RR, BBOX.MIN_CC], ids=str)
def test_bbox_min_bounds_stay_inclusive(member) -> None:
    # MinRR/MinCC are the first pixel row/column itself; nothing to qualify.
    assert "one past" not in member.desc
    assert "exclusive" not in member.desc


def test_optimizer_status_lists_only_codes_trf_returns() -> None:
    # The fitter always uses method="trf"; -1 is MINPACK-only, and a failed
    # fit writes NaN.
    desc = MODEL_METRICS.STATUS.desc
    assert "least_squares" in _first_sentence(desc)
    assert "-1" not in desc
    assert "NaN" in desc


def test_optimizer_loss_applies_rho_to_each_squared_residual() -> None:
    # scipy's cost is 0.5 * sum(C**2 * rho(f**2 / C**2)): the loss acts on each
    # squared residual, not on their sum.
    assert "applied to each squared residual" in MODEL_METRICS.LOSS.desc


def test_logistic_n0_and_kmax_bounds_are_stated() -> None:
    # LogGrowthModel._bounds: N0 in [0, min(y)], upper 1 if min(y) <= 0;
    # _kmax_for: max of Kmax_label, else max(y), and max(y) + 1 when the
    # Kmax_label column is empty for the group.
    assert "1 if that value is not positive" in LOG_GROWTH_MODEL.N0_FIT.desc
    assert "largest measured value plus 1" in LOG_GROWTH_MODEL.K_MAX.desc


@pytest.mark.parametrize(
    "member", [INTENSITY.DENSITY, INTENSITY.CONVEX_DENSITY], ids=str
)
def test_density_first_sentence_names_the_colony(member) -> None:
    assert _first_sentence(member.desc).startswith("The colony's")


def test_bit_depth_says_a_float_array_infers_16() -> None:
    assert "float" in IMAGE.BIT_DEPTH.desc


@pytest.mark.parametrize(
    "member",
    [
        ColorLab.DELTA_E2000_MEDIAN,
        ColorLab.DELTA_E2000_MEAN,
        ColorLab.DELTA_E2000_P95,
    ],
    ids=str,
)
def test_delta_e2000_spread_names_the_colony(member) -> None:
    first = _first_sentence(member.desc)
    assert "colony's pixels" in first
    assert "object pixels" not in first


#: The openings these members had when their first sentence only restated
#: the column name.
_RESTATED_OPENINGS = (
    "the new value",
    "the carrying capacity",
    "the intrinsic growth rate",
    "the initial number",
    "the upper bound",
    "the start of",
    "the end of",
    "the name of",
    "the uuid of",
    "the format of",
    "the output of",
)


@pytest.mark.parametrize(
    "member",
    [
        EDGE_CORRECTION.CORRECTED_CAP,
        EDGE_CORRECTION.NEW_VAL,
        GRID.ROW_NUM,
        GRID.COL_NUM,
        GRID.ROW_INTERVAL_START,
        GRID.ROW_INTERVAL_END,
        GRID.COL_INTERVAL_START,
        GRID.COL_INTERVAL_END,
        IMAGE.PARENT_IMAGE_NAME,
        IMAGE.PARENT_UUID,
        IMAGE.IMFORMAT,
        LOG_GROWTH_MODEL.R_FIT,
        LOG_GROWTH_MODEL.K_FIT,
        LOG_GROWTH_MODEL.N0_FIT,
        LOG_GROWTH_MODEL.K_MAX,
        MODEL_METRICS.LOSS,
        MODEL_METRICS.STATUS,
    ],
    ids=str,
)
def test_first_sentence_does_not_restate_the_label(member) -> None:
    first = _first_sentence(member.desc).lower()
    assert not first.startswith(_RESTATED_OPENINGS)
    assert "of the object" not in first
    assert "loss of model fitting" not in first


@pytest.mark.parametrize(
    "member", [SHAPE.CIRCULARITY, SHAPE.COMPACTNESS], ids=str
)
def test_shape_ratio_states_the_ideal_circle_value(member) -> None:
    # 4*pi*A / P**2 is exactly 1 for a continuous circle; pixel colonies read
    # off it, which the next sentence of each desc quantifies.
    assert "An ideal circle gives exactly 1." in member.desc


def test_mean_intensity_states_the_gray_scale() -> None:
    # The gray layer is float32 in [0, 1]: rgb2gray luminance for RGB, the
    # integer value over its 8- or 16-bit maximum for a single-channel scan.
    first = _first_sentence(INTENSITY.MEAN_INTENSITY.desc)
    assert "from 0 for black to 1" in first


def test_resolution_is_recorded_in_pixels_per_mm() -> None:
    # The unit recipe divides by px_per_mm; no code parses this column.
    desc = ACQUISITION.RESOLUTION.desc
    assert "pixels per mm" in _first_sentence(desc)
    assert "DPI / 25.4" in desc
    assert "DPI or pixels per mm" not in desc


def test_metadata_only_reads_in_plain_terms() -> None:
    # join_metadata(how="left") keeps an unmatched metadata row with null
    # measurements and QC_MetadataOnly = True.
    desc = METADATA_MATCH.METADATA_ONLY.desc
    assert "no colony was detected" in _first_sentence(desc)
    assert "empty, not zero" in desc
    assert "CSV key" not in desc
