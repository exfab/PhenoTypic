"""Tests for the :class:`QualityCheck` ABC contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phenotypic.analysis.abc_._quality_check import QualityCheck


class DummyQC(QualityCheck):
    """Concrete QC that copies a caller-supplied severity column.

    Test fixtures pre-populate an ``input_severity`` column on the input
    frame; ``_compute`` mirrors it into ``QC_Dummy_Severity`` so the base
    class can derive ``Flag`` and ``Status``. This isolates the base-class
    severity-to-status machinery from any subclass computation logic.
    """

    name = "Dummy"
    _measurement_infoclass = None

    def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
        out = group.copy()
        out[self.severity_col()] = group["input_severity"].astype(float)
        return out


def _frame_with_severities(severities: list[float]) -> pd.DataFrame:
    n = len(severities)
    return pd.DataFrame({
        "Metadata_ImageFile": [f"img_{i // 2}.png" for i in range(n)],
        "Metadata_Strain": ["WT" if i % 2 == 0 else "KO" for i in range(n)],
        "ObjectLabel": list(range(1, n + 1)),
        "Size_Area": np.linspace(100.0, 200.0, n),
        "input_severity": severities,
    })


class TestClassAttributes:
    """Class-level defaults declared by :class:`QualityCheck`."""

    def test_class_attrs_have_defaults(self):
        assert QualityCheck.severity_warn == 0.05
        assert QualityCheck.severity_fail == 0.10
        assert QualityCheck._exposes_agg_func is False


class TestSeverityToStatus:
    """Severity -> Flag/Status tri-state derivation."""

    @pytest.mark.parametrize(
        ("severity", "expected_status", "expected_flag"),
        [
            (0.0, "pass", False),
            (0.04, "pass", False),
            (0.05, "warn", False),
            (0.07, "warn", False),
            (0.10, "fail", True),
            (0.50, "fail", True),
            (np.nan, "pass", False),
        ],
    )
    def test_severity_to_status_pass_warn_fail(
        self, severity: float, expected_status: str, expected_flag: bool
    ):
        data = _frame_with_severities([severity])
        check = DummyQC(on="Size_Area", groupby=["Metadata_ImageFile"])
        out = check.analyze(data)

        assert out[check.status_col()].iloc[0] == expected_status
        assert bool(out[check.flag_col()].iloc[0]) is expected_flag


class TestColumnNameClassmethods:
    """Severity/flag/status column-name classmethods."""

    def test_severity_col_flag_col_status_col_classmethods(self):
        assert DummyQC.severity_col() == "QC_Dummy_Severity"
        assert DummyQC.flag_col() == "QC_Dummy_Flag"
        assert DummyQC.status_col() == "QC_Dummy_Status"


class TestAnalyzeValidation:
    """Input-column validation performed by :meth:`analyze`."""

    def test_analyze_raises_on_missing_groupby_column(self):
        data = _frame_with_severities([0.0, 0.2])
        check = DummyQC(on="Size_Area", groupby=["Missing_Col"])

        with pytest.raises(KeyError):
            check.analyze(data)

    def test_analyze_raises_on_missing_on_column(self):
        data = _frame_with_severities([0.0, 0.2])
        check = DummyQC(on="Missing_On", groupby=["Metadata_ImageFile"])

        with pytest.raises(KeyError):
            check.analyze(data)


class TestAnalyzeShape:
    """Row preservation through :meth:`analyze`."""

    def test_analyze_never_drops_rows(self):
        severities = [0.0, 0.05, 0.07, 0.15, 0.30, np.nan]
        data = _frame_with_severities(severities)
        check = DummyQC(on="Size_Area", groupby=["Metadata_ImageFile"])

        out = check.analyze(data)

        assert len(out) == len(data)


class TestSummary:
    """Per-group summary computed from the analyzed frame."""

    def test_summary_shape_and_columns(self):
        data = pd.DataFrame({
            "Metadata_ImageFile": [
                "img_0.png", "img_0.png", "img_0.png", "img_0.png",
                "img_1.png", "img_1.png", "img_1.png", "img_1.png",
            ],
            "Metadata_Strain": [
                "WT", "WT", "KO", "KO",
                "WT", "WT", "KO", "KO",
            ],
            "ObjectLabel": [1, 2, 3, 4, 5, 6, 7, 8],
            "Size_Area": [100.0] * 8,
            "input_severity": [0.02, 0.04, 0.15, 0.20, 0.06, 0.08, 0.30, 0.50],
        })
        check = DummyQC(
            on="Size_Area",
            groupby=["Metadata_ImageFile", "Metadata_Strain"],
        )
        check.analyze(data)

        summary = check.summary()

        assert list(summary.columns) == [
            "Metadata_ImageFile",
            "Metadata_Strain",
            "num_rows",
            "num_flagged",
            "max_severity",
            "status",
        ]
        assert len(summary) == 4

    def test_summary_status_is_worst_in_group(self):
        data = pd.DataFrame({
            "Metadata_ImageFile": [
                "fail_mix.png", "fail_mix.png",
                "warn_mix.png", "warn_mix.png",
                "all_pass.png", "all_pass.png",
            ],
            "ObjectLabel": [1, 2, 3, 4, 5, 6],
            "Size_Area": [100.0] * 6,
            "input_severity": [0.00, 0.30, 0.02, 0.07, 0.00, 0.01],
        })
        check = DummyQC(on="Size_Area", groupby=["Metadata_ImageFile"])
        check.analyze(data)

        summary = check.summary().set_index("Metadata_ImageFile")

        assert summary.loc["fail_mix.png", "status"] == "fail"
        assert summary.loc["warn_mix.png", "status"] == "warn"
        assert summary.loc["all_pass.png", "status"] == "pass"


class TestFlaggedKeys:
    """``flagged_keys()`` hand-off to GUI curation."""

    def test_flagged_keys_returns_image_file_object_label_pairs(self):
        data = _frame_with_severities([0.00, 0.20, 0.06, 0.50])
        check = DummyQC(on="Size_Area", groupby=["Metadata_ImageFile"])
        check.analyze(data)

        keys = check.flagged_keys()

        flagged_rows = data[data["input_severity"] >= 0.10]
        expected = list(zip(
            flagged_rows["Metadata_ImageFile"].astype(str),
            flagged_rows["ObjectLabel"].astype(int),
        ))
        assert sorted(keys) == sorted(expected)

    def test_flagged_keys_returns_empty_when_columns_absent(self):
        data = pd.DataFrame({
            "Metadata_Plate": ["P1", "P1", "P2"],
            "Size_Area": [100.0, 200.0, 300.0],
            "input_severity": [0.0, 0.5, 0.6],
        })
        check = DummyQC(on="Size_Area", groupby=["Metadata_Plate"])
        check.analyze(data)

        assert check.flagged_keys() == []

    def test_flagged_keys_returns_empty_when_no_rows_flagged(self):
        data = _frame_with_severities([0.0, 0.02, 0.04, 0.04])
        check = DummyQC(on="Size_Area", groupby=["Metadata_ImageFile"])
        check.analyze(data)

        assert check.flagged_keys() == []


class TestResults:
    """``results()`` accessor returns the most recent analyze output."""

    def test_results_returns_latest_measurements(self):
        data = _frame_with_severities([0.01, 0.20])
        check = DummyQC(on="Size_Area", groupby=["Metadata_ImageFile"])
        out = check.analyze(data)

        results = check.results()

        pd.testing.assert_frame_equal(results, out)


class TestSetAnalyzerOverrides:
    """``_apply2group_func`` / ``show`` are intentionally not implemented."""

    def test_apply2group_func_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="_compute"):
            QualityCheck._apply2group_func(pd.DataFrame())

    def test_show_raises_not_implemented(self):
        check = DummyQC(on="Size_Area", groupby=["Metadata_ImageFile"])
        with pytest.raises(NotImplementedError, match="dash"):
            check.show()


class TestInstanceOverrides:
    """Constructor severity overrides win over class-level defaults."""

    def test_instance_severity_overrides_class_defaults(self):
        check = DummyQC(
            on="Size_Area",
            groupby=["Metadata_ImageFile"],
            severity_warn=0.5,
            severity_fail=0.9,
        )

        assert check.severity_warn == 0.5
        assert check.severity_fail == 0.9

        data = _frame_with_severities([0.3, 0.6, 0.95])
        out = check.analyze(data)

        statuses = out[check.status_col()].tolist()
        flags = out[check.flag_col()].tolist()
        assert statuses == ["pass", "warn", "fail"]
        assert flags == [False, False, True]


class TestInitSubclassDocstring:
    """``__init_subclass__`` appends the QC RST table to subclass docs."""

    def test_init_subclass_appends_qc_check_table_to_docstring(self):
        class FooQC(QualityCheck):
            """Original docstring."""

            name = "Foo"

            def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
                out = group.copy()
                out[self.severity_col()] = 0.0
                return out

        doc = FooQC.__doc__ or ""
        assert "QC_Foo_Flag" in doc
        assert "QC_Foo_Severity" in doc
        assert "QC_Foo_Status" in doc
