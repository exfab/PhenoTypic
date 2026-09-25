"""The 0.20.0 size/shape change note renders from one hook in every doc surface."""

from __future__ import annotations

import phenotypic
from phenotypic.measure import MeasureShape, MeasureSize, MeasureTexture
from phenotypic.schema import SHAPE, SIZE, TEXTURE, Entry, MeasurementInfo

MARKER = ".. versionchanged:: 0.20.0"


def test_version_is_0_20_0():
    assert phenotypic.__version__ == "0.20.0"


def test_default_change_note_is_empty_and_leaves_docs_untouched():
    class _Plain(MeasurementInfo):
        @classmethod
        def category(cls):
            return "Plain"

        A = Entry("A", "alpha")

    assert _Plain.change_note() == ""
    assert _Plain.append_rst_to_doc("Doc.") == "Doc.\n\n" + _Plain.rst_table()


def test_size_and_shape_notes_carry_the_rename_table_and_the_trap():
    for info in (SIZE, SHAPE):
        note = info.change_note()
        assert note.startswith(MARKER)
        assert "``Shape_MaxRadius``" in note and "``Size_InscribedRadius``" in note
        assert "``Shape_MeanRadius``" in note and "``Shape_MeanBoundaryDist``" in note
        assert "Same name, different value" in note


def test_note_renders_above_the_table_in_measurer_docs():
    """Anchor on the table directive, not a column name: the note itself spells
    ``Size_Area``, so a header anchor would pass with the note below the table."""
    for measurer in (MeasureSize, MeasureShape):
        doc = measurer.__doc__
        assert MARKER in doc
        assert doc.index(MARKER) < doc.index(".. list-table::"), measurer


def test_note_renders_in_the_enum_docstrings():
    assert MARKER in SIZE.__doc__
    assert MARKER in SHAPE.__doc__


def test_unrelated_classes_carry_no_note():
    assert MARKER not in MeasureTexture.__doc__
    assert TEXTURE.change_note() == ""
