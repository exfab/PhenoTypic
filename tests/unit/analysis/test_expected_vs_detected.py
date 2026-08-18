"""Tests for the :class:`ExpectedVsDetectedCount` quality check."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from phenotypic.analysis.qc import ExpectedVsDetectedCount
from phenotypic.schema import IMAGE


def _make_96well_metadata(image_file: str = "plate1.png") -> pd.DataFrame:
    """Return a synthetic 96-well metadata frame with one row per well."""
    return pd.DataFrame({
        str(IMAGE.IMAGE_NAME): [image_file] * 96,
        "Object_Label": list(range(1, 97)),
    })


def _make_measurements(image_file: str, n_detected: int) -> pd.DataFrame:
    """Return a synthetic measurement frame with ``n_detected`` rows."""
    return pd.DataFrame({
        str(IMAGE.IMAGE_NAME): [image_file] * n_detected,
        "Object_Label": list(range(1, n_detected + 1)),
        "Size_Area": np.linspace(100.0, 200.0, n_detected),
    })


class TestMetricCalculation:
    """Metric computation across the pass/warn/fail bands."""

    def test_basic_match_zero_metric(self) -> None:
        metadata = _make_96well_metadata()
        measurements = _make_measurements("plate1.png", 96)
        chk = ExpectedVsDetectedCount(
            metadata=metadata, groupby=[str(IMAGE.IMAGE_NAME)]
        )

        result = chk.analyze(measurements)

        assert (result["QC_Count_Metric"] == 0.0).all()
        assert (result["QC_Count_Status"] == "pass").all()
        assert (~result["QC_Count_Flag"]).all()
        assert (result["QC_Count_Delta"] == 0).all()

    def test_missing_one_well_metric_below_warn(self) -> None:
        metadata = _make_96well_metadata()
        measurements = _make_measurements("plate1.png", 95)
        chk = ExpectedVsDetectedCount(
            metadata=metadata, groupby=[str(IMAGE.IMAGE_NAME)]
        )

        result = chk.analyze(measurements)

        metric = result["QC_Count_Metric"].iloc[0]
        assert math.isclose(metric, 1 / 96, rel_tol=1e-9)
        assert result["QC_Count_Delta"].iloc[0] == -1
        assert (result["QC_Count_Status"] == "pass").all()
        assert (~result["QC_Count_Flag"]).all()

    def test_missing_six_wells_metric_above_warn(self) -> None:
        metadata = _make_96well_metadata()
        measurements = _make_measurements("plate1.png", 90)
        chk = ExpectedVsDetectedCount(
            metadata=metadata, groupby=[str(IMAGE.IMAGE_NAME)]
        )

        result = chk.analyze(measurements)

        metric = result["QC_Count_Metric"].iloc[0]
        assert math.isclose(metric, 6 / 96, rel_tol=1e-9)
        assert (result["QC_Count_Status"] == "warn").all()
        assert (~result["QC_Count_Flag"]).all()

    def test_missing_many_wells_status_fail(self) -> None:
        metadata = _make_96well_metadata()
        measurements = _make_measurements("plate1.png", 80)
        chk = ExpectedVsDetectedCount(
            metadata=metadata, groupby=[str(IMAGE.IMAGE_NAME)]
        )

        result = chk.analyze(measurements)

        metric = result["QC_Count_Metric"].iloc[0]
        assert math.isclose(metric, 16 / 96, rel_tol=1e-9)
        assert (result["QC_Count_Status"] == "fail").all()
        assert (result["QC_Count_Flag"]).all()

    def test_extra_wells_positive_delta(self) -> None:
        metadata = _make_96well_metadata()
        measurements = _make_measurements("plate1.png", 100)
        chk = ExpectedVsDetectedCount(
            metadata=metadata, groupby=[str(IMAGE.IMAGE_NAME)]
        )

        result = chk.analyze(measurements)

        metric = result["QC_Count_Metric"].iloc[0]
        assert math.isclose(metric, 4 / 96, rel_tol=1e-9)
        assert result["QC_Count_Delta"].iloc[0] == 4
        assert (result["QC_Count_Status"] == "pass").all()


class TestUnmatchedGroups:
    """Behavior when a measurement group has no metadata counterpart."""

    def test_unmatched_group_metric_inf_and_recorded(self) -> None:
        metadata = _make_96well_metadata("plate1.png")
        measurements = _make_measurements("plate2.png", 10)
        chk = ExpectedVsDetectedCount(
            metadata=metadata, groupby=[str(IMAGE.IMAGE_NAME)]
        )

        result = chk.analyze(measurements)

        metric = result["QC_Count_Metric"].iloc[0]
        assert math.isinf(metric)
        assert (result["QC_Count_Status"] == "fail").all()
        assert (result["QC_Count_Flag"]).all()
        assert chk.unmatched_groups == [("plate2.png",)]

    def test_unmatched_groups_resets_on_reanalyze(self) -> None:
        metadata = _make_96well_metadata("plate1.png")
        first = _make_measurements("plate2.png", 5)
        second = _make_measurements("plate3.png", 8)
        chk = ExpectedVsDetectedCount(
            metadata=metadata, groupby=[str(IMAGE.IMAGE_NAME)]
        )

        chk.analyze(first)
        assert chk.unmatched_groups == [("plate2.png",)]

        chk.analyze(second)
        assert chk.unmatched_groups == [("plate3.png",)]


class TestMetadataValidation:
    """Constructor-time validation of the metadata argument."""

    def test_metadata_keyerror_on_missing_groupby_column(self) -> None:
        metadata = pd.DataFrame({
            str(IMAGE.IMAGE_NAME): ["plate1.png"],
            "Object_Label": [1],
        })

        with pytest.raises(KeyError, match="Missing_Col"):
            ExpectedVsDetectedCount(
                metadata=metadata, groupby=["Missing_Col"]
            )

    def test_metadata_accepts_csv_path(self, tmp_path) -> None:
        metadata = _make_96well_metadata()
        csv_path = tmp_path / "m.csv"
        metadata.to_csv(csv_path, index=False)

        chk = ExpectedVsDetectedCount(
            metadata=str(csv_path), groupby=[str(IMAGE.IMAGE_NAME)]
        )

        # ``metadata`` echoes the path verbatim; the resolved frame is on
        # the private ``_metadata`` slot.
        assert chk.metadata == str(csv_path)
        assert isinstance(chk._metadata, pd.DataFrame)
        assert len(chk._metadata) == 96
        assert str(IMAGE.IMAGE_NAME) in chk._metadata.columns

    def test_metadata_accepts_pathlib_path(self, tmp_path) -> None:
        metadata = _make_96well_metadata()
        csv_path = tmp_path / "m.csv"
        metadata.to_csv(csv_path, index=False)

        chk = ExpectedVsDetectedCount(
            metadata=csv_path, groupby=[str(IMAGE.IMAGE_NAME)]
        )

        # A pathlib.Path is normalized to a plain str for round-tripping.
        assert chk.metadata == str(csv_path)
        assert isinstance(chk._metadata, pd.DataFrame)
        assert len(chk._metadata) == 96

    def test_metadata_normalization_preserves_object_label(self) -> None:
        metadata = pd.DataFrame(
            {
                str(IMAGE.IMAGE_NAME): ["plate1.png"] * 2,
                "Object_Label": [1, 2],
                "Size_Area": [100.0, 101.0],
            }
        )

        chk = ExpectedVsDetectedCount(
            metadata=metadata, groupby=[str(IMAGE.IMAGE_NAME)]
        )

        assert list(chk._metadata.columns) == [
            str(IMAGE.IMAGE_NAME),
            "Object_Label",
            "Size_Area",
        ]
        assert chk._metadata["Object_Label"].tolist() == [1, 2]

    def test_bare_generic_layout_column_becomes_metadata_column(self) -> None:
        metadata = pd.DataFrame(
            {
                "Foo": ["a", "a", "b"],
                "Object_Label": [1, 2, 3],
            }
        )

        chk = ExpectedVsDetectedCount(metadata=metadata, groupby=["Metadata_Foo"])

        assert list(chk._metadata.columns) == ["Metadata_Foo", "Object_Label"]
        assert chk._metadata["Metadata_Foo"].tolist() == ["a", "a", "b"]

    def test_bare_unknown_groupby_normalizes_layout_and_measurement_frames(self) -> None:
        metadata = pd.DataFrame(
            {
                "plate": ["a", "a", "b"],
                "Object_Label": [1, 2, 3],
            }
        )
        measurements = pd.DataFrame(
            {
                "plate": ["a", "a", "b"],
                "Object_Label": [11, 12, 13],
            }
        )

        chk = ExpectedVsDetectedCount(metadata=metadata, groupby=["plate"])
        result = chk.analyze(measurements)

        assert chk.groupby == ["Metadata_plate"]
        assert list(chk._metadata.columns) == ["Metadata_plate", "Object_Label"]
        assert "Metadata_plate" in result.columns
        assert "plate" not in result.columns
        assert (result["QC_Count_Metric"] == 0.0).all()

    def test_measurement_join_normalization_preserves_unrelated_custom_columns(self) -> None:
        metadata = pd.DataFrame(
            {
                "plate": ["a", "a"],
                "Object_Label": [1, 2],
            }
        )
        measurements = pd.DataFrame(
            {
                "plate": ["a", "a"],
                "Object_Label": [11, 12],
                "custom_score": [0.2, 0.4],
                "note_text": ["first", "second"],
            }
        )
        original = measurements.copy(deep=True)

        result = ExpectedVsDetectedCount(
            metadata=metadata, groupby=["plate"]
        ).analyze(measurements)

        assert result["custom_score"].tolist() == [0.2, 0.4]
        assert result["note_text"].tolist() == ["first", "second"]
        assert "Metadata_custom_score" not in result.columns
        assert "Metadata_note_text" not in result.columns
        pd.testing.assert_frame_equal(measurements, original)

    def test_lower_snake_case_layout_column_becomes_metadata_column(self) -> None:
        metadata = pd.DataFrame(
            {
                "batch_id": ["a", "a", "b"],
                "Object_Label": [1, 2, 3],
            }
        )
        measurements = pd.DataFrame(
            {
                "batch_id": ["a", "a", "b"],
                "Object_Label": [11, 12, 13],
            }
        )

        chk = ExpectedVsDetectedCount(metadata=metadata, groupby=["batch_id"])
        result = chk.analyze(measurements)

        assert chk.groupby == ["Metadata_batch_id"]
        assert "Metadata_batch_id" in chk._metadata.columns
        assert "Metadata_batch_id" in result.columns
        assert (result["QC_Count_Metric"] == 0.0).all()

    def test_qualified_external_layout_column_stays_nonmetadata_end_to_end(self) -> None:
        metadata = pd.DataFrame(
            {
                "ExternalMeasure_BatchKey": ["a", "a", "b"],
                "Object_Label": [1, 2, 3],
            }
        )
        measurements = pd.DataFrame(
            {
                "ExternalMeasure_BatchKey": ["a", "a", "b"],
                "Object_Label": [11, 12, 13],
            }
        )

        chk = ExpectedVsDetectedCount(
            metadata=metadata, groupby=["ExternalMeasure_BatchKey"]
        )
        result = chk.analyze(measurements)

        assert list(chk._metadata.columns) == [
            "ExternalMeasure_BatchKey",
            "Object_Label",
        ]
        assert "Metadata_ExternalMeasure_BatchKey" not in chk._metadata.columns
        assert (result["QC_Count_Metric"] == 0.0).all()

    def test_metadata_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            ExpectedVsDetectedCount(
                metadata=str(tmp_path / "does_not_exist.csv"),
                groupby=[str(IMAGE.IMAGE_NAME)],
            )


class TestEmittedColumns:
    """Output column inventory and ``flagged_keys`` integration."""

    def test_emitted_columns_present(self) -> None:
        metadata = _make_96well_metadata()
        measurements = _make_measurements("plate1.png", 80)
        chk = ExpectedVsDetectedCount(
            metadata=metadata, groupby=[str(IMAGE.IMAGE_NAME)]
        )

        result = chk.analyze(measurements)

        expected = {
            "QC_Count_Detected",
            "QC_Count_Expected",
            "QC_Count_Delta",
            "QC_Count_Metric",
            "QC_Count_Flag",
            "QC_Count_Status",
        }
        assert expected.issubset(set(result.columns))

    def test_flagged_keys_returns_image_file_object_label_pairs(self) -> None:
        metadata = _make_96well_metadata()
        measurements = _make_measurements("plate1.png", 80)
        chk = ExpectedVsDetectedCount(
            metadata=metadata, groupby=[str(IMAGE.IMAGE_NAME)]
        )

        chk.analyze(measurements)
        keys = chk.flagged_keys()

        assert len(keys) == 80
        assert ("plate1.png", 1) in keys
        assert ("plate1.png", 80) in keys
        for image_file, label in keys:
            assert image_file == "plate1.png"
            assert isinstance(label, int)


class TestInspect:
    """Plotly figure rendering and pre-analyze guard."""

    def test_inspect_returns_plotly_figure(self) -> None:
        metadata = _make_96well_metadata()
        measurements = _make_measurements("plate1.png", 96)
        chk = ExpectedVsDetectedCount(
            metadata=metadata, groupby=[str(IMAGE.IMAGE_NAME)]
        )
        chk.analyze(measurements)

        fig = chk.inspect()

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_inspect_raises_before_analyze(self) -> None:
        metadata = _make_96well_metadata()
        chk = ExpectedVsDetectedCount(
            metadata=metadata, groupby=[str(IMAGE.IMAGE_NAME)]
        )

        with pytest.raises(RuntimeError, match="analyze"):
            chk.inspect()


class TestClassFlags:
    """Class-level opt-in flags read by ``OperationRegistry``."""

    def test_exposes_agg_func_is_false(self) -> None:
        assert ExpectedVsDetectedCount._exposes_agg_func is False

    def test_higher_is_bad_is_true(self) -> None:
        assert ExpectedVsDetectedCount._HIGHER_IS_BAD is True

    def test_metric_col_returns_metric_name(self) -> None:
        assert ExpectedVsDetectedCount.metric_col() == "QC_Count_Metric"

    def test_class_attributes_match_spec(self) -> None:
        metadata = _make_96well_metadata()
        chk = ExpectedVsDetectedCount(
            metadata=metadata, groupby=[str(IMAGE.IMAGE_NAME)]
        )
        assert ExpectedVsDetectedCount.name == "Count"
        assert chk.warn_threshold == 0.05
        assert chk.fail_threshold == 0.10

    def test_threshold_override_per_instance(self) -> None:
        metadata = _make_96well_metadata()
        chk = ExpectedVsDetectedCount(
            metadata=metadata,
            groupby=[str(IMAGE.IMAGE_NAME)],
            warn_threshold=0.02,
            fail_threshold=0.20,
        )
        assert chk.warn_threshold == 0.02
        assert chk.fail_threshold == 0.20
