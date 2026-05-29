"""Tests for the generic ``QUALITY_CHECK`` measurement info enum."""
from __future__ import annotations

from phenotypic.schema._quality_check import QUALITY_CHECK


class TestQualityCheckMembers:
    def test_flag_value(self) -> None:
        assert QUALITY_CHECK.FLAG.label == "Flag"
        assert QUALITY_CHECK.FLAG.desc.startswith("True")

    def test_severity_value(self) -> None:
        assert QUALITY_CHECK.SEVERITY.label == "Severity"
        assert QUALITY_CHECK.SEVERITY.desc.startswith("Normalized")

    def test_status_value(self) -> None:
        assert QUALITY_CHECK.STATUS.label == "Status"
        assert QUALITY_CHECK.STATUS.desc.startswith("Categorical")


class TestQualityCheckCategory:
    def test_category_classmethod(self) -> None:
        assert QUALITY_CHECK.category() == "QC"

    def test_category_property_on_member(self) -> None:
        assert QUALITY_CHECK.FLAG.CATEGORY == "QC"
        assert QUALITY_CHECK.SEVERITY.CATEGORY == "QC"
        assert QUALITY_CHECK.STATUS.CATEGORY == "QC"


class TestQualityCheckHeaders:
    def test_get_headers_uses_base_category(self) -> None:
        """``get_headers`` is inherited and uses the bare ``QC`` category.

        The per-check name substitution only applies to the RST docstring
        table rendered by :meth:`append_rst_to_doc`, not to the live
        enum values, so headers remain ``QC_Flag`` / ``QC_Severity`` /
        ``QC_Status``.
        """
        assert QUALITY_CHECK.get_headers() == ["QC_Flag", "QC_Severity", "QC_Status"]


class TestAppendRstToDocPlaceholder:
    def test_placeholder_columns_when_check_name_is_none(self) -> None:
        rendered = QUALITY_CHECK.append_rst_to_doc("orig")
        assert "QC_<name>_Flag" in rendered
        assert "QC_<name>_Severity" in rendered
        assert "QC_<name>_Status" in rendered

    def test_placeholder_preserved_when_check_name_explicit_none(self) -> None:
        rendered = QUALITY_CHECK.append_rst_to_doc("orig", check_name=None)
        assert "QC_<name>_Flag" in rendered
        assert "QC_<name>_Severity" in rendered
        assert "QC_<name>_Status" in rendered


class TestAppendRstToDocCount:
    def test_count_substitution_renders_full_column_names(self) -> None:
        rendered = QUALITY_CHECK.append_rst_to_doc("orig", check_name="Count")
        assert "QC_Count_Flag" in rendered
        assert "QC_Count_Severity" in rendered
        assert "QC_Count_Status" in rendered

    def test_count_does_not_leave_placeholder_behind(self) -> None:
        rendered = QUALITY_CHECK.append_rst_to_doc("orig", check_name="Count")
        assert "<name>" not in rendered


class TestAppendRstToDocSE:
    def test_se_substitution_renders_full_column_names(self) -> None:
        rendered = QUALITY_CHECK.append_rst_to_doc("orig", check_name="SE")
        assert "QC_SE_Flag" in rendered
        assert "QC_SE_Severity" in rendered
        assert "QC_SE_Status" in rendered


class TestAppendRstToDocStringInput:
    def test_string_input_preserves_original_docstring(self) -> None:
        rendered = QUALITY_CHECK.append_rst_to_doc("orig doc", check_name="Count")
        assert rendered.startswith("orig doc")
        assert "QC_Count_Flag" in rendered

    def test_string_input_separated_by_blank_line(self) -> None:
        rendered = QUALITY_CHECK.append_rst_to_doc("orig doc", check_name="Count")
        assert "\n\n" in rendered


class TestAppendRstToDocObjectInput:
    def test_object_with_docstring(self) -> None:
        class X:
            """original"""

        rendered = QUALITY_CHECK.append_rst_to_doc(X, check_name="Count")
        assert rendered.startswith("original")
        assert "QC_Count_Flag" in rendered

    def test_object_without_docstring_uses_empty_base(self) -> None:
        class Y:
            pass

        rendered = QUALITY_CHECK.append_rst_to_doc(Y, check_name="Count")
        assert "QC_Count_Flag" in rendered


class TestAppendRstToDocBackwardCompat:
    def test_no_check_name_keyword_still_renders_table(self) -> None:
        rendered = QUALITY_CHECK.append_rst_to_doc("doc")
        assert rendered.startswith("doc")
        assert "QC_<name>_Flag" in rendered
        assert "QC_<name>_Severity" in rendered
        assert "QC_<name>_Status" in rendered
