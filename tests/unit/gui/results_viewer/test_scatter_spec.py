"""FigureSpec is a pure config object; plottable drops metadata phantoms."""

from __future__ import annotations

import polars as pl

from phenotypic.gui.results_viewer._scatter_tab._spec import FigureSpec, plottable


def test_plottable_drops_phantom_rows() -> None:
    df = pl.DataFrame(
        {
            "Object_Label": [1, 2, None, None],
            "QC_MetadataOnly": [False, False, True, True],
            "Shape_Area": [10.0, 20.0, None, None],
        }
    )
    out = plottable(df)
    assert out.height == 2
    assert out["Object_Label"].to_list() == [1, 2]


def test_plottable_is_a_no_op_without_the_curation_column() -> None:
    """A per-store table carries no QC_MetadataOnly; it must not crash."""
    df = pl.DataFrame({"Object_Label": [1, 2], "Shape_Area": [10.0, 20.0]})
    assert plottable(df).height == 2


def test_a_null_boolean_flag_keeps_its_row() -> None:
    """Null is "not known to be a phantom", so the colony is plotted.

    Without this the Boolean branch's ``fill_null(False)`` is unpinned:
    a null propagates through the negation and polars drops the row, so
    a real colony vanishes with nothing raising.
    """
    df = pl.DataFrame(
        {
            "Object_Label": [1, 2, 3],
            "QC_MetadataOnly": [False, True, None],
        }
    )
    assert plottable(df)["Object_Label"].to_list() == [1, 3]


def test_a_string_typed_flag_neither_raises_nor_hides_colonies() -> None:
    """Polars 1.41 cannot cast Utf8 to Boolean AT ALL.

    ``strict=False`` raises the same ``InvalidOperationError`` as
    ``strict=True`` -- measured by spike, against a plan that assumed the
    non-strict cast degraded to nulls. So any cast-based implementation
    500s the tab on a string-typed flag, and this test is what says so.

    It also pins the failure DIRECTION: ``"maybe"`` is unparseable and
    ``None`` is absent, and both keep their row. A malformed flag must
    show too many points, never silently hide real colonies.
    """
    df = pl.DataFrame(
        {
            "Object_Label": [1, 2, 3, 4],
            "QC_MetadataOnly": ["false", "true", "maybe", None],
        }
    )
    assert plottable(df)["Object_Label"].to_list() == [1, 3, 4]


def test_a_numeric_flag_reads_zero_as_kept_and_one_as_phantom() -> None:
    """A CSV round-trip can land the flag as Int64 rather than Boolean."""
    df = pl.DataFrame(
        {"Object_Label": [1, 2, 3], "QC_MetadataOnly": [0, 1, None]}
    )
    assert plottable(df)["Object_Label"].to_list() == [1, 3]


def test_an_unreadable_flag_plots_every_row_rather_than_raising() -> None:
    """The last resort, and the direction it must fail in.

    A dtype no branch recognises (here a list column) must not take the
    tab down and must not drop rows on a guess.
    """
    df = pl.DataFrame(
        {"Object_Label": [1, 2], "QC_MetadataOnly": [[1], [2]]}
    )
    assert plottable(df)["Object_Label"].to_list() == [1, 2]


def test_figure_spec_is_frozen() -> None:
    spec = FigureSpec(x_col="Metadata_FrameIndex", y_col="Shape_Area")
    assert spec.share_axes is True
    assert spec.hue_col is None
    try:
        spec.x_col = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("FigureSpec must be frozen")
