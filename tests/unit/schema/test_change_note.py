"""The 0.20.0 size/shape change note renders from one hook in every doc surface."""

from __future__ import annotations

import inspect

import pytest

from phenotypic.measure import (
    MeasureBounds,
    MeasureIntensity,
    MeasureShape,
    MeasureSize,
    MeasureTexture,
)
from phenotypic.schema import BBOX, INTENSITY, SHAPE, SIZE, TEXTURE, Entry, MeasurementInfo

MARKER = ".. versionchanged:: 0.20.0"


def test_default_change_note_is_empty_and_leaves_docs_untouched():
    class _Plain(MeasurementInfo):
        @classmethod
        def metric_family(cls):
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
    assert MARKER not in MeasureBounds.__doc__
    assert BBOX.change_note() == ""


# ------------------------------------------- single-channel normalisation (0.21.0)

SINGLE_CHANNEL = "single-channel"
SINGLE_CHANNEL_MARKER = ".. versionchanged:: 0.21.0"


def test_size_note_keeps_the_split_note_and_adds_integrated_intensity_units():
    """SIZE carries two changes from two releases: both notes render, split first."""
    note = " ".join(SIZE.change_note().split())
    assert note.count(MARKER) == 1
    assert note.count(SINGLE_CHANNEL_MARKER) == 1
    assert note.index(MARKER) < note.index(SINGLE_CHANNEL_MARKER)
    assert note.index("Shape_Area") < note.index(SINGLE_CHANNEL)
    assert "``Size_IntegratedIntensity``" in note
    assert "normalised units" in note


@pytest.mark.parametrize("info", [INTENSITY, TEXTURE], ids=["INTENSITY", "TEXTURE"])
def test_intensity_and_texture_notes_say_the_columns_are_now_produced(info):
    note = " ".join(info.change_note().split())
    assert note.startswith(SINGLE_CHANNEL_MARKER)
    assert SINGLE_CHANNEL in note
    assert "now produced" in note


_SINGLE_CHANNEL_INFOS = pytest.mark.parametrize(
    "info", [SIZE, SHAPE, INTENSITY, TEXTURE], ids=["SIZE", "SHAPE", "INTENSITY", "TEXTURE"]
)


@_SINGLE_CHANNEL_INFOS
def test_single_channel_note_says_detection_derived_columns_change(info):
    """Detectors now find colonies on single-channel scans, so every column
    measured on a detected object can change, not only the intensity ones."""
    note = " ".join(info.change_note().split())
    assert SINGLE_CHANNEL in note
    assert "segmentation" in note
    for prefix in ("``Shape_*``", "``Bbox_*``"):
        assert prefix in note, prefix


def test_shape_note_keeps_the_split_note_first():
    note = " ".join(SHAPE.change_note().split())
    assert note.count(MARKER) == 1
    assert note.count(SINGLE_CHANNEL_MARKER) == 1
    assert note.index("Shape_Area") < note.index(SINGLE_CHANNEL)


@_SINGLE_CHANNEL_INFOS
def test_single_channel_note_warns_against_resuming(info):
    """No work-id fence covers this change, so the note is the only guard."""
    note = " ".join(info.change_note().split())
    assert (
        "A single-channel run started before this change must be re-run with "
        "``--overwrite``, not resumed." in note
    )


@pytest.mark.parametrize(
    "measurer,info",
    [(MeasureIntensity, INTENSITY), (MeasureTexture, TEXTURE)],
    ids=["MeasureIntensity", "MeasureTexture"],
)
def test_single_channel_note_renders_in_measurer_and_enum_docs(measurer, info):
    for doc in (measurer.__doc__, info.__doc__):
        assert SINGLE_CHANNEL_MARKER in doc
        assert SINGLE_CHANNEL in doc
    assert measurer.__doc__.index(SINGLE_CHANNEL_MARKER) < measurer.__doc__.index(
        ".. list-table::"
    )
