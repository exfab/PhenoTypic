"""rst_table renders Biology/Image columns only when populated."""

from phenotypic.schema import Entry, MeasurementInfo


class _DescOnly(MeasurementInfo):
    @classmethod
    def category(cls):
        return "DescOnly"

    A = Entry("A", "alpha")
    B = Entry("B", "beta")


class _WithBio(MeasurementInfo):
    @classmethod
    def category(cls):
        return "WithBio"

    A = Entry("A", "alpha", bio_desc="grows")
    B = Entry("B", "beta")


class _WithImage(MeasurementInfo):
    @classmethod
    def category(cls):
        return "WithImage"

    A = Entry("A", "alpha", image="shape/area.png")


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
