"""NaN handling for :class:`TukeyOutlierRemover` and ``tukey_fences``.

Regression suite for a bug where a single NaN measurement in a group produced
``(nan, nan)`` fences, made every inlier comparison ``False``, and silently
deleted the *entire* group -- including all of its valid colony measurements.
"""

import warnings

import numpy as np
import pandas as pd

from phenotypic.analysis import TukeyOutlierRemover
from phenotypic.analysis._helper._qc_math import tukey_fences


def _legacy_filter_group(group: pd.DataFrame, on: str, k: float) -> pd.DataFrame:
    """Reproduce the pre-fix group filter (plain percentile + inlier mask).

    Used as the reference for the "NaN-free groups are unchanged" test: the new
    implementation must agree with this one exactly whenever no value is NaN.

    Args:
        group: One group of rows.
        on: Measurement column to filter on.
        k: IQR multiplier.

    Returns:
        The rows the old implementation would have kept.
    """
    values = group[on]
    arr = values.to_numpy()
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - iqr * k
    upper = q3 + iqr * k
    return group[(values >= lower) & (values <= upper)]


class TestTukeyFencesNaN:
    """``tukey_fences`` ignores NaN rather than propagating it."""

    def test_single_nan_does_not_destroy_fences(self):
        # Non-NaN values are [10, 11, 12, 13, 100]: Q1 = 11, Q3 = 13, IQR = 2,
        # so the fences are 11 - 3 = 8 and 13 + 3 = 16.
        lower, upper = tukey_fences(
            np.array([10.0, 11.0, 12.0, np.nan, 13.0, 100.0]), 1.5
        )
        assert (lower, upper) == (8.0, 16.0)

    def test_nan_is_ignored_not_treated_as_a_value(self):
        """Fences with a NaN appended equal the fences of the values alone."""
        clean = np.array([10.0, 11.0, 12.0, 13.0, 100.0])
        assert tukey_fences(np.append(clean, np.nan), 1.5) == tukey_fences(clean, 1.5)

    def test_nan_free_fences_match_plain_percentile(self):
        values = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 100.0])
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        expected = (float(q1 - iqr * 1.5), float(q3 + iqr * 1.5))
        assert tukey_fences(values, 1.5) == expected

    def test_all_nan_group_yields_nan_fences_without_warning(self):
        # A bare np.nanpercentile emits "RuntimeWarning: All-NaN slice
        # encountered"; the early return must avoid it.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            lower, upper = tukey_fences(np.array([np.nan, np.nan, np.nan]), 1.5)
        assert np.isnan(lower) and np.isnan(upper)

    def test_empty_input_yields_nan_fences(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            lower, upper = tukey_fences(np.array([]), 1.5)
        assert np.isnan(lower) and np.isnan(upper)


class TestTukeyOutlierRemoverNaN:
    """A NaN colony measurement must not delete its whole group."""

    def test_nan_group_keeps_inliers_and_nan_but_removes_outlier(self):
        # Fences from the five non-NaN colonies are (8.0, 16.0), so only the
        # 100 is an outlier. Before the fix the fences were (nan, nan) and all
        # six rows -- valid measurements included -- were deleted.
        data = pd.DataFrame(
            {
                "ImageName": ["img1"] * 6,
                "Area": [10.0, 11.0, 12.0, np.nan, 13.0, 100.0],
            }
        )
        result = TukeyOutlierRemover(on="Area", groupby=["ImageName"]).analyze(data)

        assert len(result) == 5
        kept = result["Area"].to_numpy(dtype=float)
        assert sorted(kept[~np.isnan(kept)].tolist()) == [10.0, 11.0, 12.0, 13.0]
        assert int(np.isnan(kept).sum()) == 1
        assert 100.0 not in kept.tolist()

    def test_all_nan_group_passes_through_untouched(self):
        data = pd.DataFrame(
            {
                "ImageName": ["img1"] * 3,
                "Area": [np.nan, np.nan, np.nan],
            }
        )
        result = TukeyOutlierRemover(on="Area", groupby=["ImageName"]).analyze(data)

        assert len(result) == 3
        assert bool(np.isnan(result["Area"].to_numpy(dtype=float)).all())

    def test_nan_group_does_not_starve_a_clean_neighbour_group(self):
        data = pd.DataFrame(
            {
                "ImageName": ["img1"] * 6 + ["img2"] * 4,
                "Area": [
                    10.0, 11.0, 12.0, np.nan, 13.0, 100.0,  # NaN + one outlier
                    10.0, 11.0, 12.0, 13.0,  # clean, no outliers
                ],
            }
        )
        result = TukeyOutlierRemover(on="Area", groupby=["ImageName"]).analyze(data)

        from phenotypic.schema import IMAGE

        image_name = str(IMAGE.IMAGE_NAME)
        assert len(result[result[image_name] == "img1"]) == 5
        assert len(result[result[image_name] == "img2"]) == 4

    def test_nan_free_group_is_identical_to_legacy_implementation(self):
        rng = np.random.default_rng(42)
        areas = np.concatenate([rng.normal(200, 30, 95), [500, 550, 600, 50, 40]])
        data = pd.DataFrame({"ImageName": ["img1"] * 100, "Area": areas})

        result = TukeyOutlierRemover(on="Area", groupby=["ImageName"], k=1.5).analyze(
            data
        )
        expected = _legacy_filter_group(data, "Area", 1.5)

        assert len(result) == len(expected)
        np.testing.assert_array_equal(
            np.sort(result["Area"].to_numpy()), np.sort(expected["Area"].to_numpy())
        )
