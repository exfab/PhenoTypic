"""Tests for the error-cutoff finder (good-vs-category measurement screen)."""

import numpy as np
import pandas as pd
import pytest

from phenotypic.analysis import ErrorCutoffFinder
from phenotypic.analysis._error_cutoffs import RESULT_COLUMNS
from phenotypic.schema import METADATA


def _frame(values: dict[str, list[float]], n: int) -> pd.DataFrame:
    """Build a frame with the given measurement columns + filler metadata."""
    base = {
        str(METADATA.IMAGE_NAME): ["p.tif"] * n,
        "Object_Label": list(range(1, n + 1)),
    }
    base.update(values)
    return pd.DataFrame(base)


def test_measurement_columns_detects_only_numeric_measurements():
    finder = ErrorCutoffFinder()
    df = _frame(
        {
            "Size_Area": [1.0, 2.0, 3.0],
            "Shape_Circularity": [0.1, 0.2, 0.3],
            "Intensity_MeanIntensity": [10.0, 11.0, 12.0],
            "Grid_RowNum": [1, 1, 2],  # grid context, not a measurement
        },
        n=3,
    )
    cols = finder.measurement_columns(df)
    assert "Size_Area" in cols
    assert "Shape_Circularity" in cols
    assert "Intensity_MeanIntensity" in cols
    # Metadata / object-id / grid-context columns are excluded.
    assert str(METADATA.IMAGE_NAME) not in cols
    assert "Object_Label" not in cols
    assert "Grid_RowNum" not in cols


def test_default_min_n_fields():
    finder = ErrorCutoffFinder()
    assert finder.min_error_n == 8
    assert finder.min_good_n == 8
    # keyword-only pydantic construction; bad kwargs raise.
    with pytest.raises(Exception):
        ErrorCutoffFinder(min_error_n=5, bogus=1)


def test_enough_data_predicate():
    finder = ErrorCutoffFinder(min_error_n=3, min_good_n=3)
    good = _frame({"Size_Area": [1.0] * 5}, n=5)
    error = _frame({"Size_Area": [9.0] * 2}, n=2)  # below min_error_n
    assert finder.enough_data(good, error) is False
    error2 = _frame({"Size_Area": [9.0] * 4}, n=4)
    assert finder.enough_data(good, error2) is True


def _separating(n_good=40, n_err=20, seed=0):
    """Good ~ N(0,1); error ~ N(4,1) on Size_Area (clearly separable),
    plus a non-separating Shape_Circularity ~ N(0.5,0.05) in both."""
    rng = np.random.default_rng(seed)
    good = pd.DataFrame(
        {
            str(METADATA.IMAGE_NAME): ["p.tif"] * n_good,
            "Object_Label": list(range(1, n_good + 1)),
            "Size_Area": rng.normal(0.0, 1.0, n_good),
            "Shape_Circularity": rng.normal(0.5, 0.05, n_good),
        }
    )
    error = pd.DataFrame(
        {
            str(METADATA.IMAGE_NAME): ["p.tif"] * n_err,
            "Object_Label": list(range(1, n_err + 1)),
            "Size_Area": rng.normal(4.0, 1.0, n_err),
            "Shape_Circularity": rng.normal(0.5, 0.05, n_err),
        }
    )
    return good, error


def test_separating_measurement_ranks_first_with_high_auc():
    good, error = _separating()
    res = ErrorCutoffFinder().analyze(good, error)
    assert list(res.columns) == [
        "measurement", "auc", "direction", "cutoff", "recall", "specificity",
        "good_flagged", "f_stat", "p_value", "p_bh", "good_n", "error_n",
    ]
    # Size_Area separates; it ranks first with high AUC.
    assert res.iloc[0]["measurement"] == "Size_Area"
    assert res.iloc[0]["auc"] > 0.9
    # Error is the HIGH side -> flag when measurement is ABOVE the cutoff.
    assert res.iloc[0]["direction"] == ">"
    # The cutoff sits between the two means.
    assert 0.5 < res.iloc[0]["cutoff"] < 4.0
    # Recall / specificity are sane fractions; good_flagged is a small count.
    assert 0.5 <= res.iloc[0]["recall"] <= 1.0
    assert 0.5 <= res.iloc[0]["specificity"] <= 1.0
    assert 0 <= res.iloc[0]["good_flagged"] <= 40
    # The non-separating measurement has AUC near 0.5.
    circ = res[res["measurement"] == "Shape_Circularity"].iloc[0]
    assert abs(circ["auc"] - 0.5) < 0.15
    # n columns reflect inputs.
    assert res.iloc[0]["good_n"] == 40
    assert res.iloc[0]["error_n"] == 20


def test_direction_below_when_error_is_low_side():
    # Error LOWER than good -> flag when measurement is BELOW the cutoff.
    rng = np.random.default_rng(1)
    good = pd.DataFrame({"Object_Label": range(40), "Intensity_MeanIntensity": rng.normal(5, 0.5, 40)})
    error = pd.DataFrame({"Object_Label": range(20), "Intensity_MeanIntensity": rng.normal(1, 0.5, 20)})
    res = ErrorCutoffFinder().analyze(good, error)
    row = res.iloc[0]
    assert row["measurement"] == "Intensity_MeanIntensity"
    assert row["direction"] == "<"
    assert 1.0 < row["cutoff"] < 5.0


def test_bh_adjusted_p_is_monotone_and_ge_raw():
    good, error = _separating()
    res = ErrorCutoffFinder().analyze(good, error)
    # BH-adjusted p >= raw p for every measurement.
    assert (res["p_bh"] >= res["p_value"] - 1e-9).all()


def test_specificity_and_good_flagged_match_the_reported_cutoff_on_overlap():
    """Pin the midpoint-nudge invariant on OVERLAPPING data: the reported
    specificity / good_flagged equal the actual good-class classification at the
    returned cutoff (would fail if the nudge moved a point across the boundary)."""
    good, error = _separating()
    res = ErrorCutoffFinder().analyze(good, error)
    row = res[res["measurement"] == "Size_Area"].iloc[0]
    vals = good["Size_Area"]
    flagged = vals > row["cutoff"] if row["direction"] == ">" else vals < row["cutoff"]
    assert int(flagged.sum()) == row["good_flagged"]
    assert (1.0 - flagged.mean()) == pytest.approx(row["specificity"], abs=1e-9)


def test_min_sample_sizes_below_two_are_rejected():
    with pytest.raises(Exception):
        ErrorCutoffFinder(min_good_n=1)
    with pytest.raises(Exception):
        ErrorCutoffFinder(min_error_n=0)


def test_empty_result_has_numeric_dtypes():
    # Insufficient error -> empty frame whose numeric columns stay numeric
    # (so Phase-4/5 concat/parquet doesn't infer object).
    finder = ErrorCutoffFinder(min_good_n=3, min_error_n=3)
    res = finder.analyze(_frame({"Size_Area": [1.0] * 5}, n=5),
                         _frame({"Size_Area": [9.0] * 2}, n=2))
    assert res.empty
    assert list(res.columns) == list(RESULT_COLUMNS)
    assert str(res["auc"].dtype) == "float64"
    assert str(res["good_flagged"].dtype) == "int64"
    assert str(res["good_n"].dtype) == "int64"


def test_prefix_set_detects_phenotype_headers_and_excludes_position():
    # Drift guard: every listed phenotype prefix matches a representative
    # column; absolute position (Bbox_) and ids are excluded.
    from phenotypic.schema import BBOX

    pheno = [
        "Size_Area", "Shape_Circularity", "Intensity_MeanIntensity",
        "SymZones_Foo", "GridSpatial_Foo", "RadialExpansion_Foo",
        "TextureGray_Contrast",
    ]
    df = pd.DataFrame({c: [1.0, 2.0, 3.0] for c in pheno})
    df[str(BBOX.CENTER_RR)] = [1.0, 2.0, 3.0]  # absolute plate position
    df["Object_Label"] = [1, 2, 3]
    cols = set(ErrorCutoffFinder().measurement_columns(df))
    assert set(pheno).issubset(cols)
    assert str(BBOX.CENTER_RR) not in cols
    assert "Object_Label" not in cols


def test_insufficient_error_returns_empty_frame():
    good, error = _separating(n_good=40, n_err=3)
    res = ErrorCutoffFinder(min_error_n=8).analyze(good, error)
    assert res.empty
    assert list(res.columns) == list(__import__("phenotypic.analysis._error_cutoffs",
                                                fromlist=["RESULT_COLUMNS"]).RESULT_COLUMNS)


def test_all_nan_measurement_is_skipped_not_crashed():
    good = pd.DataFrame({"Object_Label": range(20), "Size_Area": np.r_[np.full(20, np.nan)],
                         "Shape_Circularity": np.random.default_rng(0).normal(0.5, 0.1, 20)})
    error = pd.DataFrame({"Object_Label": range(10), "Size_Area": np.full(10, np.nan),
                          "Shape_Circularity": np.random.default_rng(1).normal(0.7, 0.1, 10)})
    res = ErrorCutoffFinder(min_good_n=10, min_error_n=8).analyze(good, error)
    assert "Size_Area" not in set(res["measurement"])  # all-NaN -> skipped
    assert "Shape_Circularity" in set(res["measurement"])


def test_constant_measurement_is_skipped():
    good = pd.DataFrame({"Object_Label": range(20), "Size_Area": np.full(20, 3.0)})
    error = pd.DataFrame({"Object_Label": range(10), "Size_Area": np.full(10, 3.0)})
    res = ErrorCutoffFinder(min_good_n=10, min_error_n=8).analyze(good, error)
    assert res.empty  # no separable measurement


def test_perfect_separation_clean_cutoff_and_metrics():
    good = pd.DataFrame({"Object_Label": range(20), "Size_Area": np.linspace(0, 1, 20)})
    error = pd.DataFrame({"Object_Label": range(10), "Size_Area": np.linspace(10, 11, 10)})
    res = ErrorCutoffFinder(min_good_n=10, min_error_n=8).analyze(good, error)
    row = res.iloc[0]
    assert row["auc"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)
    assert row["specificity"] == pytest.approx(1.0)
    assert row["good_flagged"] == 0
    # Midpoint-nudge puts the cutoff IN the gap (≈ (1 + 10)/2), not on the edge.
    assert 1.0 < row["cutoff"] < 10.0
    assert row["cutoff"] == pytest.approx((1.0 + 10.0) / 2, abs=0.2)


def test_measurement_only_in_good_is_ignored():
    good = pd.DataFrame({"Object_Label": range(20), "Size_Area": np.random.default_rng(0).normal(0, 1, 20),
                         "Shape_OnlyHere": np.random.default_rng(0).normal(0, 1, 20)})
    error = pd.DataFrame({"Object_Label": range(10), "Size_Area": np.random.default_rng(1).normal(3, 1, 10)})
    res = ErrorCutoffFinder(min_good_n=10, min_error_n=8).analyze(good, error)
    assert "Shape_OnlyHere" not in set(res["measurement"])
