"""rst_table renders Biology/Image columns only when populated."""

from phenotypic.schema import Entry, MeasurementInfo


class _DescOnly(MeasurementInfo):
    @classmethod
    def metric_family(cls):
        return "DescOnly"

    A = Entry("A", "alpha")
    B = Entry("B", "beta")


class _WithBio(MeasurementInfo):
    @classmethod
    def metric_family(cls):
        return "WithBio"

    A = Entry("A", "alpha", bio_desc="grows")
    B = Entry("B", "beta")


class _WithImage(MeasurementInfo):
    @classmethod
    def metric_family(cls):
        return "WithImage"

    A = Entry("A", "alpha", image="shape/area.png")


class _WithRoles(MeasurementInfo):
    @classmethod
    def metric_family(cls):
        return "WithRoles"

    M = Entry("M", r"Ratio :math:`\frac{a}{b}` of two things.")
    X = Entry("X", "See :class:`Foo` for details.")


def test_desc_only_has_no_biology_or_image_columns():
    table = _DescOnly.rst_table()
    assert "Description" in table
    assert "Biology" not in table
    assert "Image" not in table
    assert "``A``" in table


def test_biology_column_appears_when_any_member_sets_bio_desc():
    table = _WithBio.rst_table()
    assert "Biology" in table
    assert "grows" in table
    assert "Image" not in table


def test_image_column_emits_directive_with_root_absolute_path():
    table = _WithImage.rst_table()
    assert "Image" in table
    assert ".. image:: /_static/measurements/shape/area.png" in table


def test_use_headers_renders_prefixed_value():
    assert "``DescOnly_A``" in _DescOnly.rst_table(use_headers=True)
    assert "``A``" in _DescOnly.rst_table(use_headers=False)


def test_math_role_preserved_in_cells():
    # content roles (LaTeX) must survive — otherwise formula descriptions render
    # as literal text in the rendered docs.
    table = _WithRoles.rst_table()
    assert r":math:`\frac{a}{b}`" in table


def test_xref_role_flattened_in_cells():
    # cross-reference roles resolve poorly in a list-table cell, so they are
    # flattened to inline literals.
    table = _WithRoles.rst_table()
    assert "``Foo``" in table
    assert ":class:`Foo`" not in table


def test_custom_description_header_is_honored():
    table = _DescOnly.rst_table(header=("Col", "Meaning"))
    assert "- Col" in table
    assert "- Meaning" in table


def test_categorized_table_has_a_categories_badge_column() -> None:
    from phenotypic.schema import SIZE

    table = SIZE.rst_table()
    assert "     - Categories" in table
    assert (
        ":bdg-ref-dark-line:`Starting Metrics <measurement-category-startingmetrics>`"
        in table
    )


def test_uncategorized_table_has_no_categories_column() -> None:
    from phenotypic.schema import SHAPE

    assert "Categories" not in SHAPE.rst_table()


def test_category_badges_is_empty_for_an_uncategorized_member() -> None:
    from phenotypic.schema import SHAPE, SIZE

    assert next(iter(SHAPE)).category_badges == ""
    assert SIZE.AREA.category_badges.count(":bdg-ref-") == 1


def test_quality_check_docs_render_with_category_column() -> None:
    # _render_info_table's second caller (schema/_quality_check.py) runs at
    # import time for every QualityCheck subclass; a row-shape mismatch there
    # makes phenotypic.analysis unimportable.
    from phenotypic.schema import QUALITY_CHECK

    doc = QUALITY_CHECK.append_rst_to_doc("Doc.", check_name="Count")
    assert doc.startswith("Doc.")
    assert ".. list-table:: Metric family: **QC_Count**" in doc


def test_analysis_package_imports() -> None:
    import importlib

    import phenotypic.analysis

    importlib.reload(phenotypic.analysis)
