"""Tests for the ``QUALITY_COUNT`` measurement-info enum."""
from __future__ import annotations

from phenotypic.tools_.measurement_info._quality_count import QUALITY_COUNT


class TestQualityCountMembers:
    def test_detected_label_and_desc(self) -> None:
        assert QUALITY_COUNT.DETECTED.label == "Detected"
        assert QUALITY_COUNT.DETECTED.desc.startswith("Detected")

    def test_expected_label_and_desc(self) -> None:
        assert QUALITY_COUNT.EXPECTED.label == "Expected"
        assert QUALITY_COUNT.EXPECTED.desc.startswith("Expected")

    def test_delta_label_and_desc(self) -> None:
        assert QUALITY_COUNT.DELTA.label == "Delta"
        assert QUALITY_COUNT.DELTA.desc.startswith("Detected")


class TestQualityCountCategory:
    def test_category_classmethod(self) -> None:
        assert QUALITY_COUNT.category() == "QC_Count"

    def test_category_instance_property(self) -> None:
        assert QUALITY_COUNT.DETECTED.CATEGORY == "QC_Count"


class TestQualityCountStringForm:
    def test_detected_str_is_prefixed(self) -> None:
        assert str(QUALITY_COUNT.DETECTED) == "QC_Count_Detected"

    def test_delta_str_is_prefixed(self) -> None:
        assert str(QUALITY_COUNT.DELTA) == "QC_Count_Delta"


class TestQualityCountCollections:
    def test_get_labels_order_and_contents(self) -> None:
        assert QUALITY_COUNT.get_labels() == ["Detected", "Expected", "Delta"]

    def test_get_headers_order_and_contents(self) -> None:
        assert QUALITY_COUNT.get_headers() == [
            "QC_Count_Detected",
            "QC_Count_Expected",
            "QC_Count_Delta",
        ]


class TestQualityCountRstTable:
    def test_rst_table_contains_labels_and_category(self) -> None:
        table = QUALITY_COUNT.rst_table()
        assert "Detected" in table
        assert "Expected" in table
        assert "Delta" in table
        assert "QC_Count" in table

    def test_append_rst_to_doc_preserves_original_and_appends_table(self) -> None:
        appended = QUALITY_COUNT.append_rst_to_doc("orig")
        assert appended.startswith("orig")
        assert "Detected" in appended
        assert "Expected" in appended
        assert "Delta" in appended
        assert "QC_Count" in appended
