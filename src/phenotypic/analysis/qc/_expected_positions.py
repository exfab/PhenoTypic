"""Derive the expected colony positions from the frame being analyzed.

Two QC checks need to know how many colonies a group *should* have:
:class:`~phenotypic.analysis.qc.ExpectedVsDetectedCount`, whose whole metric is
that comparison, and the fitting-window check, which shows it as context. They
share this helper so the two can never disagree about what "expected" means.

**Why the frame can answer at all.** The CLI joins the user's ``--metadata``
table as the *left* frame, so every metadata row survives: a row that matched no
measured object becomes a **phantom** — measurement columns null, the
metadata-only flag true. When one metadata row names one position on one image,
the joined table is therefore a complete list of expected positions, phantoms
included. When it names something coarser (one row per image, or one grid
position applied to every image) it is not, and the guards below are what stop a
check from silently treating it as though it were.

QC checks and plotting code may read the metadata-only flag; analysis and post
ops may not (``_cli/CLAUDE.md``). This module is read by QC checks only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from phenotypic.schema import GRID, IMAGE, METADATA_MATCH, OBJECT

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["DEFAULT_POSITION_BY", "expected_positions", "numeric_order_key"]


def numeric_order_key(values: "pd.Series") -> "pd.Series":
    """Return *values* as numbers when every non-null one parses, else unchanged.

    **Why an index column cannot be sorted as it arrives.** A grid or frame
    index reaches a check in whatever dtype the run stored, and a CLI mirror
    has been seen to store ``Grid_RowNum`` and ``Grid_ColNum`` as text. A CSV
    with one stray non-numeric cell does the same. Sorted as text, ``"10"``
    lands before ``"2"``, so a triplet at columns 2, 6, 10 comes out
    ``(10, 2, 6)`` — right values, wrong order, and nothing fails. Casting to a
    number once, before anything is sorted, is what avoids it.

    **Only text is converted.** ``pandas.to_numeric`` will happily turn a
    ``datetime64`` column into int64 nanoseconds, which sorts correctly
    already and would publish 1.7e18 where the user's frame label was a
    timestamp — a conversion that fixes nothing and destroys the labels. The
    same goes for ``timedelta64`` and ``Period``. So the coercion is gated on
    the dtype being object or string rather than on "not numeric".

    A **categorical** is decoded to its categories' dtype first. A
    dictionary-encoded parquet column arrives that way — likely for a grid
    index, which has a dozen distinct values — and a categorical sorts by its
    *category* order, which for strings read back from parquet is the same
    lexicographic order this function exists to undo.

    A text column that does **not** parse comes back untouched, deliberately.
    A genuinely categorical axis (``"pre"``, ``"day1"``) keeps ordering the
    way it always has, and a numeric guard further down still gets to raise
    its own message rather than being handed a column of nulls this function
    made.

    Args:
        values: The index column an ordering is about to be built on.

    Returns:
        A numeric series, or *values* itself.
    """
    import pandas as pd

    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype(values.dtype.categories.dtype)
    if not (
        pd.api.types.is_object_dtype(values)
        or pd.api.types.is_string_dtype(values)
    ):
        return values
    numeric = pd.to_numeric(values, errors="coerce")
    if bool((numeric.isna() & values.notna()).any()):
        return values
    return numeric


def _blank_position_message(
    frame: "pd.DataFrame", blank: "pd.Series", columns: list[str]
) -> str:
    """Say why rows with a blank position cannot be expected positions.

    Two causes need two different fixes, so they get two sentences. A blank
    position on a row with no detected object (a phantom) means the metadata
    is at the wrong grain. A blank position on a *detected* object means an
    object was measured outside every grid cell. Without an object-label
    column the two cannot be told apart, and the message says only that rows
    are blank.

    Args:
        frame: The frame being analyzed.
        blank: Which of its rows carry a blank position value.
        columns: The position columns.

    Returns:
        The error message.
    """
    label = str(OBJECT.LABEL)
    if label not in frame.columns:
        return (
            f"expected_positions: {int(blank.sum())} row(s) carry a blank value "
            f"in one of the position columns {columns}. Expected positions can "
            "only be derived when every row names one position on one image. "
            "Pass per-position metadata, or supply expected counts explicitly."
        )
    detected = frame[label].notna()
    off_grid = int((blank & detected).sum())
    wrong_grain = int((blank & ~detected).sum())
    parts = []
    if wrong_grain:
        parts.append(
            f"{wrong_grain} row(s) with no detected object carry a blank value in "
            f"one of the position columns {columns}. That is the signature of "
            "metadata at the wrong grain — expected positions can only be derived "
            "from metadata with one row per position per image. Pass per-position "
            "metadata, or supply expected counts explicitly."
        )
    if off_grid:
        parts.append(
            f"{off_grid} detected object(s) carry a blank value in one of the "
            f"position columns {columns}: objects measured outside every grid "
            "cell (off-grid detections). They have no expected position to be "
            "counted against, so expected "
            "positions cannot be derived while they are in the frame. Drop them "
            "first (keep only rows whose position columns are all set), or pass "
            "a layout."
        )
    return "expected_positions: " + " ".join(parts)


def _default_position_by() -> tuple[str, ...]:
    """Return the identity of one expected colony position, per the schema."""
    return (
        str(IMAGE.IMAGE_NAME),
        str(GRID.ROW_NUM),
        str(GRID.COL_NUM),
    )


#: The identity of one expected colony: which image, and which grid cell on it.
#: Spelled through the schema rather than as literals, so a rename of a member
#: reaches here.
DEFAULT_POSITION_BY: tuple[str, ...] = _default_position_by()


def expected_positions(
    frame: "pd.DataFrame",
    position_by: Sequence[str] | None = None,
) -> "pd.DataFrame | None":
    """One row per expected colony position, or ``None`` when the frame cannot say.

    Expected positions are the **distinct** ``position_by`` tuples in *frame*.
    Two measured objects sharing one grid cell (a doublet) therefore count
    **once** toward expected and twice toward detected — which is exactly the
    divergence a count check exists to surface, so it must not be collapsed on
    the detected side too.

    A phantom row counts toward expected: it still carries its position. It is
    not a detection, because its ``Object_Label`` is null — the NaN-native test
    :meth:`ExpectedVsDetectedCount._detected_count` already uses, which needs no
    flag column and so stays correct on a frame that never went through the CLI.

    Two guards decide whether the answer is trustworthy, and both fail loudly
    rather than returning a number that means something else:

    * **Guard 1 — the frame must be a metadata left join.** Without the
      metadata-only flag column there are no phantoms, so every expected
      position would be one that was found, expected would equal detected, and
      every group would pass. ``None`` is returned so the caller can say what to
      do about it: ``ExpectedVsDetectedCount`` refuses with a pointer to
      ``metadata=``, ``FittingWindow`` reports a null expected count.
    * **Guard 2 — every row needs a position.** A blank ``position_by`` value on
      a row with no detected object is the signature of metadata at the wrong
      grain: per-*image* metadata's phantoms carry an image name and no grid
      cell. Dropping those rows would quietly shrink the denominator, so this
      raises instead. A blank value on a *detected* object is a different
      problem — an object measured outside every grid cell — and the error
      says so, since the fix is to drop those detections, not to change the
      metadata.

    Args:
        frame: The measurement frame being analyzed.
        position_by: Columns whose distinct tuples are the expected positions.
            Defaults to :data:`DEFAULT_POSITION_BY`.

    Returns:
        A copy of *frame* reduced to its first row per distinct ``position_by``
        tuple, with a fresh ``RangeIndex`` — so a caller may group it by its own
        keys and count rows. ``None`` when Guard 1 rejects the frame.

    Raises:
        ValueError: If ``position_by`` is empty, or if any row carries a blank
            value in a ``position_by`` column (Guard 2).
        KeyError: If a ``position_by`` column is absent from *frame*.

    Examples:
        Three rows over two grid cells — a doublet in cell ``(1, 1)`` and a
        phantom in ``(2, 1)`` — give **two** expected positions:

        >>> import pandas as pd
        >>> from phenotypic.analysis.qc._expected_positions import expected_positions
        >>> from phenotypic.schema import GRID, IMAGE, METADATA_MATCH, OBJECT
        >>> image, row, col = (
        ...     str(IMAGE.IMAGE_NAME), str(GRID.ROW_NUM), str(GRID.COL_NUM)
        ... )
        >>> frame = pd.DataFrame({
        ...     image: ["p1.png"] * 3,
        ...     row: [1, 1, 2],
        ...     col: [1, 1, 1],
        ...     str(OBJECT.LABEL): [1.0, 2.0, None],
        ...     str(METADATA_MATCH.METADATA_ONLY): [False, False, True],
        ... })
        >>> len(expected_positions(frame))
        2

        A frame with no metadata-only flag cannot say (Guard 1):

        >>> flag = str(METADATA_MATCH.METADATA_ONLY)
        >>> expected_positions(frame.drop(columns=[flag])) is None
        True

        A row with no grid cell is refused rather than dropped (Guard 2):

        >>> blank = frame.copy()
        >>> blank.loc[2, col] = None
        >>> expected_positions(blank)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: expected_positions: 1 row(s) with no detected object carry a blank value in ...
    """
    import pandas as pd

    columns = list(DEFAULT_POSITION_BY if position_by is None else position_by)
    if not columns:
        raise ValueError("expected_positions: position_by must name at least one column")

    if str(METADATA_MATCH.METADATA_ONLY) not in frame.columns:
        # Guard 1. Not an error: a notebook frame legitimately has no phantoms,
        # and only the caller knows whether that is fatal for it.
        return None

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"expected_positions: position columns absent from the frame: {missing}"
        )

    blank = pd.Series(False, index=frame.index)
    for column in columns:
        values = frame[column]
        column_blank = values.isna()
        if values.dtype == object or pd.api.types.is_string_dtype(values):
            # A whitespace-only cell is a blank position too: a CSV round-trip
            # turns an empty metadata field into "" rather than into NaN.
            column_blank = column_blank | (
                values.astype("string").str.strip().fillna("") == ""
            )
        blank = blank | column_blank
    if bool(blank.any()):
        # Guard 2.
        raise ValueError(_blank_position_message(frame, blank, columns))

    return frame.drop_duplicates(subset=columns).reset_index(drop=True)
