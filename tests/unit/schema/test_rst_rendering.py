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
        ":bdg-ref-info-line:`Starting Metrics <measurement-category-startingmetrics>`"
        in table
    )


def _list_table_rows(table: str) -> list[list[str]]:
    """Split a rendered list-table into rows of cell text, header row first.

    A ``   * - `` line opens a row, a ``     -`` line opens a cell (an empty
    cell has no trailing space), and any other line continues the open cell.
    """
    rows: list[list[str]] = []
    for line in table.splitlines():
        if line.startswith("   * - "):
            rows.append([line[len("   * - "):]])
        elif rows and (line == "     -" or line.startswith("     - ")):
            rows[-1].append(line[len("     - "):])
        elif rows:
            rows[-1][-1] += "\n" + line
    return rows


def test_every_cell_sits_under_its_own_column_header() -> None:
    # SIZE.AREA carries a Type badge, a category, bio_desc and an image, so
    # every optional column is present and a cell emitted out of header order
    # lands under the wrong heading without changing any row's cell count.
    from phenotypic.schema import SIZE
    from phenotypic.schema._measurement_info import _rst_cell_text

    header, *rows = _list_table_rows(SIZE.rst_table())
    assert header == ["Name", "Description", "Type", "Categories", "Biology", "Image"]
    assert all(len(row) == len(header) for row in rows)

    (area,) = [row for row in rows if row[0] == f"``{SIZE.AREA.label}``"]
    assert area[header.index("Type")] == SIZE.AREA.use_badge
    assert area[header.index("Categories")] == SIZE.AREA.category_badges
    assert area[header.index("Biology")] == _rst_cell_text(SIZE.AREA.bio_desc)
    assert area[header.index("Image")].startswith(".. image::")


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


def test_analysis_package_imports_in_a_fresh_interpreter() -> None:
    # A fresh process, so QualityCheck.__init_subclass__ runs for every
    # concrete check regardless of what this session has already imported.
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import phenotypic.analysis"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_concrete_quality_check_docstring_carries_its_qc_table() -> None:
    from phenotypic.analysis import ICC

    assert f"Metric family: **QC_{ICC.name}**" in (ICC.__doc__ or "")
