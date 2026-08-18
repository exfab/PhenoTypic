"""Tests for the :class:`QualityCheck` ABC contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from phenotypic.analysis.abc_._quality_check import QualityCheck
from phenotypic.schema import GENETIC, IMAGE


class DummyQC(QualityCheck):
    """Concrete QC that copies a caller-supplied metric column.

    Test fixtures pre-populate an ``input_metric`` column on the input
    frame; ``_compute`` mirrors it into ``QC_Dummy_Metric`` so the base
    class can derive ``Flag`` and ``Status``. This isolates the base-class
    metric-to-status machinery from any subclass computation logic.
    ``_HIGHER_IS_BAD`` is ``True`` so larger values are worse — the common
    direction exercised by the bulk of these tests.
    """

    name = "Dummy"
    _HIGHER_IS_BAD = True
    _measurement_infoclass = None

    def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
        out = group.copy()
        out[self.metric_col()] = group["input_metric"].astype(float)
        return out


def _frame_with_metrics(metrics: list[float]) -> pd.DataFrame:
    n = len(metrics)
    return pd.DataFrame({
        str(IMAGE.IMAGE_NAME): [f"img_{i // 2}.png" for i in range(n)],
        "Metadata_Strain": ["WT" if i % 2 == 0 else "KO" for i in range(n)],
        "Object_Label": list(range(1, n + 1)),
        "Size_Area": np.linspace(100.0, 200.0, n),
        "input_metric": metrics,
    })


class TestClassAttributes:
    """Field defaults and class-level flags declared by :class:`QualityCheck`.

    ``warn_threshold``/``fail_threshold`` are pydantic *instance* fields, so
    their base defaults live in ``model_fields`` rather than as bare class
    attributes; ``_exposes_agg_func`` remains a plain ``ClassVar``.
    """

    def test_threshold_field_defaults(self):
        assert QualityCheck.model_fields["warn_threshold"].default == 0.05
        assert QualityCheck.model_fields["fail_threshold"].default == 0.10

    def test_exposes_agg_func_default(self):
        assert QualityCheck._exposes_agg_func is False


class TestMetricToStatus:
    """Metric -> Flag/Status tri-state derivation (higher-is-bad)."""

    @pytest.mark.parametrize(
        ("metric", "expected_status", "expected_flag"),
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
    def test_metric_to_status_pass_warn_fail(
        self, metric: float, expected_status: str, expected_flag: bool
    ):
        data = _frame_with_metrics([metric])
        check = DummyQC(on="Size_Area", groupby=[str(IMAGE.IMAGE_NAME)])
        out = check.analyze(data)

        assert out[check.status_col()].iloc[0] == expected_status
        assert bool(out[check.flag_col()].iloc[0]) is expected_flag


class TestLowerIsBadDirection:
    """A lower-is-bad subclass inverts the threshold comparisons."""

    @pytest.mark.parametrize(
        ("metric", "expected_status", "expected_flag"),
        [
            (1.00, "pass", False),
            (0.80, "pass", False),
            (0.75, "warn", False),
            (0.60, "warn", False),
            (0.50, "fail", True),
            (0.10, "fail", True),
            (np.nan, "pass", False),
        ],
    )
    def test_lower_is_bad_status(
        self, metric: float, expected_status: str, expected_flag: bool
    ):
        class ScoreQC(QualityCheck):
            """Agreement-score QC where a smaller metric is worse."""

            name = "Score"
            _HIGHER_IS_BAD = False
            warn_threshold: float = 0.75
            fail_threshold: float = 0.50

            def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
                out = group.copy()
                out[self.metric_col()] = group["input_metric"].astype(float)
                return out

        data = _frame_with_metrics([metric])
        check = ScoreQC(on="Size_Area", groupby=[str(IMAGE.IMAGE_NAME)])
        out = check.analyze(data)

        assert out[check.status_col()].iloc[0] == expected_status
        assert bool(out[check.flag_col()].iloc[0]) is expected_flag


class TestColumnNameClassmethods:
    """Metric/flag/status column-name classmethods."""

    def test_metric_col_flag_col_status_col_classmethods(self):
        assert DummyQC.metric_col() == "QC_Dummy_Metric"
        assert DummyQC.flag_col() == "QC_Dummy_Flag"
        assert DummyQC.status_col() == "QC_Dummy_Status"


class TestAnalyzeValidation:
    """Input-column validation performed by :meth:`analyze`."""

    def test_analyze_raises_on_missing_groupby_column(self):
        data = _frame_with_metrics([0.0, 0.2])
        check = DummyQC(on="Size_Area", groupby=["Missing_Col"])

        with pytest.raises(KeyError):
            check.analyze(data)

    def test_analyze_raises_on_missing_on_column(self):
        data = _frame_with_metrics([0.0, 0.2])
        check = DummyQC(on="Missing_On", groupby=[str(IMAGE.IMAGE_NAME)])

        with pytest.raises(KeyError):
            check.analyze(data)


class TestAnalyzeShape:
    """Row preservation through :meth:`analyze`."""

    def test_analyze_never_drops_rows(self):
        metrics = [0.0, 0.05, 0.07, 0.15, 0.30, np.nan]
        data = _frame_with_metrics(metrics)
        check = DummyQC(on="Size_Area", groupby=[str(IMAGE.IMAGE_NAME)])

        out = check.analyze(data)

        assert len(out) == len(data)

    def test_flat_metadata_input_is_normalized_without_mutating_caller(self):
        """Canonical flat headers pass through without mutating the caller."""
        flat_strain = "Metadata_Strain"
        data = pd.DataFrame({
            flat_strain: ["WT", "WT"],
            "Size_Area": [10.0, 11.0],
            "input_metric": [0.01, 0.02],
        })

        result = DummyQC(on="Size_Area", groupby=[flat_strain]).analyze(data)

        assert list(data.columns) == [flat_strain, "Size_Area", "input_metric"]
        assert str(GENETIC.STRAIN) in result.columns
        assert result["Size_Area"].tolist() == [10.0, 11.0]


class TestMetadataReferenceCompatibility:
    """Metadata references normalize at configuration and filtering boundaries."""

    def test_criteria_accepts_flat_metadata_key(self):
        data = pd.DataFrame({str(GENETIC.STRAIN): ["WT", "KO"]})

        filtered = DummyQC._filter_by(data, {"Metadata_Strain": "WT"})

        assert filtered[str(GENETIC.STRAIN)].tolist() == ["WT"]

    def test_criteria_rejects_colliding_legacy_and_flat_keys(self):
        data = pd.DataFrame({str(GENETIC.STRAIN): ["WT"]})

        with pytest.raises(ValueError, match="normalize to the same"):
            DummyQC._filter_by(
                data,
                {"MetadataGenetic_Strain": "WT", str(GENETIC.STRAIN): "WT"},
            )

    @pytest.mark.parametrize("groupby", [None, 7, ["Plate", 1]])
    def test_bad_groupby_raises_pydantic_validation_error(self, groupby):
        with pytest.raises(ValidationError):
            DummyQC(on="Size_Area", groupby=groupby)

    def test_groupby_accepts_one_dimensional_numpy_array(self):
        check = DummyQC(
            on="Size_Area", groupby=np.array(["Metadata_Strain"])
        )

        assert check.groupby == [str(GENETIC.STRAIN)]

    @pytest.mark.parametrize(
        "groupby", [np.array("Metadata_Strain"), np.array([["Metadata_Strain"]])],
    )
    def test_nonvector_numpy_groupby_raises_pydantic_validation_error(self, groupby):
        with pytest.raises(ValidationError, match="one-dimensional"):
            DummyQC(on="Size_Area", groupby=groupby)


class TestSummary:
    """Per-group summary computed from the analyzed frame."""

    def test_summary_shape_and_columns(self):
        data = pd.DataFrame({
            str(IMAGE.IMAGE_NAME): [
                "img_0.png", "img_0.png", "img_0.png", "img_0.png",
                "img_1.png", "img_1.png", "img_1.png", "img_1.png",
            ],
            "Metadata_Strain": [
                "WT", "WT", "KO", "KO",
                "WT", "WT", "KO", "KO",
            ],
            "Object_Label": [1, 2, 3, 4, 5, 6, 7, 8],
            "Size_Area": [100.0] * 8,
            "input_metric": [0.02, 0.04, 0.15, 0.20, 0.06, 0.08, 0.30, 0.50],
        })
        check = DummyQC(
            on="Size_Area",
            groupby=[str(IMAGE.IMAGE_NAME), "Metadata_Strain"],
        )
        check.analyze(data)

        summary = check.summary()

        assert list(summary.columns) == [
            str(IMAGE.IMAGE_NAME),
            str(GENETIC.STRAIN),
            "qc_n_members",
            "qc_n_flagged",
            "qc_worst_metric",
            "qc_status",
        ]
        assert len(summary) == 4

    def test_summary_worst_metric_is_max_when_higher_is_bad(self):
        data = pd.DataFrame({
            str(IMAGE.IMAGE_NAME): ["g.png", "g.png", "g.png"],
            "Object_Label": [1, 2, 3],
            "Size_Area": [100.0] * 3,
            "input_metric": [0.02, 0.30, 0.15],
        })
        check = DummyQC(on="Size_Area", groupby=[str(IMAGE.IMAGE_NAME)])
        check.analyze(data)

        summary = check.summary().set_index(str(IMAGE.IMAGE_NAME))

        assert summary.loc["g.png", "qc_worst_metric"] == pytest.approx(0.30)

    def test_summary_status_is_worst_in_group(self):
        data = pd.DataFrame({
            str(IMAGE.IMAGE_NAME): [
                "fail_mix.png", "fail_mix.png",
                "warn_mix.png", "warn_mix.png",
                "all_pass.png", "all_pass.png",
            ],
            "Object_Label": [1, 2, 3, 4, 5, 6],
            "Size_Area": [100.0] * 6,
            "input_metric": [0.00, 0.30, 0.02, 0.07, 0.00, 0.01],
        })
        check = DummyQC(on="Size_Area", groupby=[str(IMAGE.IMAGE_NAME)])
        check.analyze(data)

        summary = check.summary().set_index(str(IMAGE.IMAGE_NAME))

        assert summary.loc["fail_mix.png", "qc_status"] == "fail"
        assert summary.loc["warn_mix.png", "qc_status"] == "warn"
        assert summary.loc["all_pass.png", "qc_status"] == "pass"


class TestFlaggedKeys:
    """``flagged_keys()`` hand-off to GUI curation."""

    def test_flagged_keys_returns_image_file_object_label_pairs(self):
        data = _frame_with_metrics([0.00, 0.20, 0.06, 0.50])
        check = DummyQC(on="Size_Area", groupby=[str(IMAGE.IMAGE_NAME)])
        check.analyze(data)

        keys = check.flagged_keys()

        flagged_rows = data[data["input_metric"] >= 0.10]
        expected = list(zip(
            flagged_rows[str(IMAGE.IMAGE_NAME)].astype(str),
            flagged_rows["Object_Label"].astype(int),
        ))
        assert sorted(keys) == sorted(expected)

    def test_flagged_keys_returns_empty_when_columns_absent(self):
        data = pd.DataFrame({
            "Metadata_Plate": ["P1", "P1", "P2"],
            "Size_Area": [100.0, 200.0, 300.0],
            "input_metric": [0.0, 0.5, 0.6],
        })
        check = DummyQC(on="Size_Area", groupby=["Metadata_Plate"])
        check.analyze(data)

        assert check.flagged_keys() == []

    def test_flagged_keys_returns_empty_when_no_rows_flagged(self):
        data = _frame_with_metrics([0.0, 0.02, 0.04, 0.04])
        check = DummyQC(on="Size_Area", groupby=[str(IMAGE.IMAGE_NAME)])
        check.analyze(data)

        assert check.flagged_keys() == []


class TestGroupMembers:
    """``group_members()`` maps group keys to member rows."""

    def test_group_members_maps_keys_to_image_label_value(self):
        data = _frame_with_metrics([0.00, 0.20, 0.06, 0.50])
        check = DummyQC(on="Size_Area", groupby=[str(IMAGE.IMAGE_NAME)])
        check.analyze(data)

        members = check.group_members()

        # Each key is always a tuple, even for a single groupby column.
        for key, rows in members.items():
            assert isinstance(key, tuple)
            for image_file, label, value in rows:
                assert isinstance(image_file, str)
                assert isinstance(label, int)
                assert isinstance(value, float)
        total_rows = sum(len(rows) for rows in members.values())
        assert total_rows == len(data)

    def test_group_members_empty_when_curation_columns_absent(self):
        data = pd.DataFrame({
            "Metadata_Plate": ["P1", "P1", "P2"],
            "Size_Area": [100.0, 200.0, 300.0],
            "input_metric": [0.0, 0.5, 0.6],
        })
        check = DummyQC(on="Size_Area", groupby=["Metadata_Plate"])
        check.analyze(data)

        assert check.group_members() == {}


class TestResults:
    """``results()`` accessor returns the most recent analyze output."""

    def test_results_returns_latest_measurements(self):
        data = _frame_with_metrics([0.01, 0.20])
        check = DummyQC(on="Size_Area", groupby=[str(IMAGE.IMAGE_NAME)])
        out = check.analyze(data)

        results = check.results()

        pd.testing.assert_frame_equal(results, out)


class TestSetAnalyzerOverrides:
    """``_apply2group_func`` / ``show`` are intentionally not implemented."""

    def test_apply2group_func_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="_compute"):
            QualityCheck._apply2group_func(pd.DataFrame())

    def test_show_raises_not_implemented(self):
        check = DummyQC(on="Size_Area", groupby=[str(IMAGE.IMAGE_NAME)])
        with pytest.raises(NotImplementedError, match=r"inspect\(\)"):
            check.show()


class TestThresholdOverrides:
    """Thresholds are instance fields: class defaults, per-instance overridable.

    ``warn_threshold``/``fail_threshold`` are pydantic instance fields. A
    subclass tunes the defaults by re-declaring them in its class body
    (as the concrete checks do), and any caller can override them per
    instance via constructor keywords.
    """

    def test_subclass_class_defaults(self):
        class TunedQC(QualityCheck):
            """QC with custom default thresholds."""

            name = "Tuned"
            _HIGHER_IS_BAD = True
            warn_threshold: float = 0.5
            fail_threshold: float = 0.9

            def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
                out = group.copy()
                out[self.metric_col()] = group["input_metric"].astype(float)
                return out

        check = TunedQC(on="Size_Area", groupby=[str(IMAGE.IMAGE_NAME)])
        assert check.warn_threshold == 0.5
        assert check.fail_threshold == 0.9

        data = _frame_with_metrics([0.3, 0.6, 0.95])
        out = check.analyze(data)

        statuses = out[check.status_col()].tolist()
        flags = out[check.flag_col()].tolist()
        assert statuses == ["pass", "warn", "fail"]
        assert flags == [False, False, True]

    def test_per_instance_override(self):
        check = DummyQC(
            on="Size_Area",
            groupby=[str(IMAGE.IMAGE_NAME)],
            warn_threshold=0.5,
            fail_threshold=0.9,
        )
        assert check.warn_threshold == 0.5
        assert check.fail_threshold == 0.9

        data = _frame_with_metrics([0.3, 0.6, 0.95])
        out = check.analyze(data)
        assert out[check.status_col()].tolist() == ["pass", "warn", "fail"]


class TestInitSubclassDocstring:
    """``__init_subclass__`` appends the QC RST table to subclass docs."""

    def test_init_subclass_appends_qc_check_table_to_docstring(self):
        class FooQC(QualityCheck):
            """Original docstring."""

            name = "Foo"
            _HIGHER_IS_BAD = True

            def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
                out = group.copy()
                out[self.metric_col()] = 0.0
                return out

        doc = FooQC.__doc__ or ""
        assert "QC_Foo_Flag" in doc
        assert "QC_Foo_Metric" in doc
        assert "QC_Foo_Status" in doc
