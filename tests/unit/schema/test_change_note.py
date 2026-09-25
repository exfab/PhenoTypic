"""The 0.20.0 size/shape change note renders from one hook in every doc surface."""

from __future__ import annotations

import inspect

import pytest

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
        for label in ("MinFeretDiameter", "MaxFeretDiameter"):
            assert f"``Shape_{label}``" in note and f"``Size_{label}``" in note, label
        assert "Same name, different value" in note


@pytest.mark.parametrize("info", [SIZE, SHAPE], ids=["SIZE", "SHAPE"])
def test_note_extends_the_trap_to_model_outputs(info):
    """Review MEDIUM-3. `metric_token` strips the category prefix, so a growth
    model fit on retired `Shape_MaxRadius` and one fit on `Size_MaxRadius` emit
    the same `<Model>_MaxRadius_*` header. The note must say so."""
    note = info.change_note()
    for label in ("MaxRadius", "MeanRadius", "MedianRadius"):
        assert f"``<Model>_{label}_*``" in note, label
    assert "share a name but not a meaning" in note


@pytest.mark.parametrize("info", [SIZE, SHAPE], ids=["SIZE", "SHAPE"])
def test_note_warns_that_a_pre_0_20_run_must_not_be_resumed(info):
    """Final review MEDIUM-1. Continuation identity carries no measurement
    revision, so resuming a pre-0.20.0 run reuses its finished stores with the
    retired names and the output mixes `Shape_Area` and `Size_Area` rows."""
    note = " ".join(info.change_note().split())
    assert "A run started before 0.20.0 must be re-run with ``--overwrite``, not resumed." in note
    assert "Resuming reuses the images it already finished, with their old column names." in note


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


@pytest.mark.parametrize("info", [SIZE, SHAPE], ids=["SIZE", "SHAPE"])
def test_enum_docstring_dedents_to_one_margin(info):
    """Review LOW-1. Appending the column-0 note to a docstring whose body keeps
    its 4-space source indent sets the common margin to 0, so `cleandoc` and
    Sphinx's `prepare_docstring` strip nothing and the API page renders the body
    as a block quote. After dedenting, the summary, the body and the directive
    all start at column 0, and the directive's content keeps its own indent.

    Mutation: restore `f"{SIZE.__doc__}\\n\\n{SIZE.change_note()}"` -> fails.
    """
    sphinx_docstrings = pytest.importorskip("sphinx.util.docstrings")
    for lines in (
        inspect.cleandoc(info.__doc__).splitlines(),
        sphinx_docstrings.prepare_docstring(info.__doc__),
    ):
        assert lines[1] == ""
        body = lines[2]
        assert body and body == body.lstrip(), body
        directive = lines.index(MARKER)
        assert lines[directive + 1].startswith("   ") and not lines[directive + 1].startswith("    ")


def test_unrelated_classes_carry_no_note():
    assert MARKER not in MeasureTexture.__doc__
    assert TEXTURE.change_note() == ""
