"""Tests for the ``QUALITY_SE`` measurement-info enum.

Covers member values, category prefixing, header/label generation, and the
RST documentation surface inherited from :class:`MeasurementInfo`.
"""
from __future__ import annotations

from phenotypic.tools_.measurement_info._quality_se import QUALITY_SE


class TestMemberValues:
    def test_value_member(self) -> None:
        assert QUALITY_SE.VALUE.label == "Value"
        assert QUALITY_SE.VALUE.desc.startswith("Raw SE")

    def test_mean_member(self) -> None:
        assert QUALITY_SE.MEAN.label == "Mean"
        assert QUALITY_SE.MEAN.desc.startswith("Mean across replicates")

    def test_cv_member(self) -> None:
        assert QUALITY_SE.CV.label == "CV"
        assert QUALITY_SE.CV.desc.startswith("Coefficient of variation")

    def test_num_replicates_member(self) -> None:
        assert QUALITY_SE.NUM_REPLICATES.label == "NumReplicates"
        assert QUALITY_SE.NUM_REPLICATES.desc.startswith("Replicate count")


class TestCategory:
    def test_classmethod_returns_qc_se(self) -> None:
        assert QUALITY_SE.category() == "QC_SE"

    def test_instance_property_returns_qc_se(self) -> None:
        assert QUALITY_SE.VALUE.CATEGORY == "QC_SE"


class TestStringForm:
    def test_value_str_is_prefixed(self) -> None:
        assert str(QUALITY_SE.VALUE) == "QC_SE_Value"

    def test_num_replicates_str_is_prefixed(self) -> None:
        assert str(QUALITY_SE.NUM_REPLICATES) == "QC_SE_NumReplicates"


class TestLabelsAndHeaders:
    def test_get_labels_returns_unprefixed_labels_in_order(self) -> None:
        assert QUALITY_SE.get_labels() == ["Value", "Mean", "CV", "NumReplicates"]

    def test_get_headers_returns_prefixed_headers_in_order(self) -> None:
        assert QUALITY_SE.get_headers() == [
            "QC_SE_Value",
            "QC_SE_Mean",
            "QC_SE_CV",
            "QC_SE_NumReplicates",
        ]


class TestRstTable:
    def test_rst_table_contains_each_label(self) -> None:
        table = QUALITY_SE.rst_table()
        assert "Value" in table
        assert "Mean" in table
        assert "CV" in table
        assert "NumReplicates" in table

    def test_rst_table_contains_category_title(self) -> None:
        assert "QC_SE" in QUALITY_SE.rst_table()


class TestAppendRstToDoc:
    def test_appends_table_to_provided_docstring(self) -> None:
        result = QUALITY_SE.append_rst_to_doc("orig")
        assert result.startswith("orig")
        assert "QC_SE" in result
        assert "Value" in result
        assert "NumReplicates" in result
