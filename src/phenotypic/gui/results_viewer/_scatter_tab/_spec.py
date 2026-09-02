"""The pure configuration object the Scatter tab's figures are built from.

Both destinations -- the on-screen ``dcc.Graph`` and the kaleido export --
consume one :class:`FigureSpec`, so the PDF cannot drift from the screen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

from phenotypic.schema import METADATA_MATCH

logger = logging.getLogger(__name__)

#: The curation column that marks a metadata-only row. Written by the CLI
#: into the measurements mirror; absent from per-store tables. Taken from
#: the schema rather than spelled here, so the column this predicate
#: depends on can only ever be the one the CLI writes.
CURATION_PHANTOM_COL: str = METADATA_MATCH.METADATA_ONLY.value

#: String spellings read as "this row is a phantom". Compared
#: lower-cased. Anything else -- including an empty string and an
#: unparseable token -- is read as "not a phantom", which keeps the row.
_TRUTHY_FLAG_VALUES: tuple[str, ...] = ("true", "t", "yes", "y", "1")


@dataclass(frozen=True)
class FigureSpec:
    """Every role and size that defines one Scatter figure.

    Args:
        x_col: Column plotted on X.
        y_col: Column plotted on Y.
        section_col: Column whose values become sections (PDF pages).
        row_col: Column whose values become facet rows.
        col_col: Column whose values become facet columns.
        hue_col: Column mapped to marker colour.
        shape_col: Column mapped to marker symbol.
        share_axes: Whether all facets share one X and Y range.
        show_removed: Whether curation-removed colonies render as grey x.
        sizes: Type sizes in px, keyed by role.
        marker_size: Marker area in points squared.
        marker_opacity: Marker alpha in ``[0, 1]``.
    """

    x_col: str
    y_col: str
    section_col: str | None = None
    row_col: str | None = None
    col_col: str | None = None
    hue_col: str | None = None
    shape_col: str | None = None
    share_axes: bool = True
    show_removed: bool = True
    sizes: dict[str, int] = field(
        default_factory=lambda: {
            "section": 14,
            "facet": 9,
            "axis": 8,
            "tick": 7,
            "legend": 8,
        }
    )
    marker_size: int = 6
    marker_opacity: float = 0.5


def _phantom_mask(dtype: pl.DataType) -> pl.Expr:
    """Build the "this row is a phantom" predicate for one flag dtype.

    Polars 1.41 does **not** support a ``Utf8 -> Boolean`` cast at all --
    ``strict=False`` raises the same ``InvalidOperationError`` as
    ``strict=True`` rather than yielding nulls, which is why the flag is
    read per dtype instead of cast. Measured by spike, against a plan that
    assumed the non-strict cast degraded gracefully.

    Every branch resolves null to ``False``, and an unrecognised dtype
    claims nothing, so a malformed flag renders too many points rather
    than silently hiding real colonies.

    Args:
        dtype: Dtype of the curation flag column.

    Returns:
        A boolean expression that is true for phantom rows.
    """
    col = pl.col(CURATION_PHANTOM_COL)
    if dtype == pl.Boolean:
        return col.fill_null(False)
    if dtype == pl.String:
        return (
            col.str.to_lowercase()
            .is_in(_TRUTHY_FLAG_VALUES)
            .fill_null(False)
        )
    if dtype.is_numeric():
        return (col != 0).fill_null(False)
    return pl.lit(False)


def plottable(df: pl.DataFrame) -> pl.DataFrame:
    """Drop metadata-only phantom rows, which cannot become points.

    A phantom has no ``Object_Label``, no coordinates and no crop. In the
    verification fixture 121 of 844 rows are phantoms; in the full run it
    is 117,415 of 231,229. The proportion varies, the rule does not.

    Args:
        df: A viewer frame, normally ``OutputRoot.master_df``.

    Returns:
        The subset that can be plotted. Returned unchanged when the frame
        carries no curation column, and when the flag cannot be read --
        the tab shows too much rather than raising or hiding colonies.
    """
    if CURATION_PHANTOM_COL not in df.columns:
        return df
    try:
        return df.filter(~_phantom_mask(df.schema[CURATION_PHANTOM_COL]))
    except Exception:
        logger.debug(
            "scatter: unreadable %s flag; plotting every row",
            CURATION_PHANTOM_COL,
            exc_info=True,
        )
        return df
