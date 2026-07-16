"""Analysis ops must treat ``--metadata`` phantom rows as missing data.

The CLI's ``--metadata`` join is a **left** join, so the measurements mirror
carries a row for every metadata key — including strains that matched no
measured object. Every measurement/info column on such a "phantom" row is null.

These ops are public API: a notebook calls them on frames that never went
through the CLI and carry no ``QC_MetadataOnly`` flag. So none of them branch on
that flag — they detect a phantom the way they detect any missing value, by
ignoring NaN. Each test below therefore comes in a pair: the phantom behaves
correctly, **and** a frame with no nulls is untouched (the no-op invariant that
keeps existing runs' numbers from moving).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from phenotypic.analysis.qc._expected_vs_detected import ExpectedVsDetectedCount
from phenotypic.schema import OBJECT

LABEL = str(OBJECT.LABEL)


class TestDetectedCountIgnoresPhantomRows:
    """``ExpectedVsDetectedCount`` must not count a phantom as a detection."""

    def test_phantom_row_is_not_counted_as_detected(self):
        # One real colony + one strain that was never detected.
        group = pd.DataFrame({LABEL: [1.0, np.nan], "Size_Area": [10.0, np.nan]})
        assert ExpectedVsDetectedCount._detected_count(group) == 1

    def test_all_phantom_group_reports_zero_detected(self):
        """The load-bearing case: a strain detected nowhere.

        With ``len(group)`` this returned 2 == expected -> metric 0.0 -> the
        check PASSES for a strain that grew nowhere, inverting its meaning.
        """
        group = pd.DataFrame({LABEL: [np.nan, np.nan]})
        assert ExpectedVsDetectedCount._detected_count(group) == 0

    def test_no_op_on_a_frame_without_phantoms(self):
        """Every frame today, and every notebook call: identical to len()."""
        group = pd.DataFrame({LABEL: [1, 2, 3], "Size_Area": [1.0, 2.0, 3.0]})
        assert ExpectedVsDetectedCount._detected_count(group) == len(group) == 3

    def test_frame_without_a_label_column_falls_back_to_len(self):
        """No flag and no label column -> behave exactly as before."""
        group = pd.DataFrame({"Size_Area": [1.0, 2.0]})
        assert ExpectedVsDetectedCount._detected_count(group) == 2

    def test_reads_no_flag_column(self):
        """The count must not depend on ``QC_MetadataOnly`` existing.

        The flag is a user-facing output column, not internal machinery; a
        notebook frame never has it. A frame whose flag DISAGREES with the
        labels must follow the labels.
        """
        group = pd.DataFrame({LABEL: [1.0, np.nan], "QC_MetadataOnly": [True, False]})
        assert ExpectedVsDetectedCount._detected_count(group) == 1
