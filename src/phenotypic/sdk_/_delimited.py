"""One delimiter rule and one TSV writer for every tabular text file (spec Part B).

Generated tables are TSV: UTF-8, one header row, ``\\t`` separator, ``\\n`` line
ends, and ``"`` quoting for any field that contains a tab, a newline or a quote.
One library writes every file: :func:`write_tsv` converts a pandas frame to
polars first, so a table's bytes do not depend on which library a caller held
it in (pandas and polars disagree on datetimes and on empty strings).

User-supplied tables may be ``.csv`` or ``.tsv``; :func:`delimiter_for` picks the
delimiter from the suffix and refuses anything else, so a reader never guesses.

This module imports neither polars nor pandas at module scope: ``phenotypic.sdk_``
re-exports it eagerly, and ``import phenotypic`` must stay light.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Union

from ._atomic_io import atomic_write_with_writer

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import polars as pl

    from ._atomic_io import CommitGuard

#: Suffix of every tabular text file the CLI generates.
TSV_SUFFIX: Final[str] = ".tsv"

#: Suffix of the legacy generated tables and of comma-separated user inputs.
CSV_SUFFIX: Final[str] = ".csv"

#: Suffixes accepted for a user-supplied table (``--metadata``, reference
#: metadata, tune ground truth). Matched case-insensitively.
INPUT_TABLE_SUFFIXES: Final[frozenset[str]] = frozenset({CSV_SUFFIX, TSV_SUFFIX})

_DELIMITERS: Final[dict[str, str]] = {CSV_SUFFIX: ",", TSV_SUFFIX: "\t"}


def delimiter_for(path: Union[str, Path]) -> str:
    """Return the field delimiter a table at ``path`` uses, chosen by suffix.

    Args:
        path: A ``.csv`` or ``.tsv`` path. The suffix is matched
            case-insensitively.

    Returns:
        ``"\\t"`` for ``.tsv`` and ``","`` for ``.csv``.

    Raises:
        ValueError: For any other suffix, naming both allowed ones.

    Examples:
        >>> from pathlib import Path
        >>> from phenotypic.sdk_ import delimiter_for
        >>> delimiter_for(Path("plate_metadata.TSV")) == "\\t"
        True
    """
    suffix = Path(path).suffix.lower()
    try:
        return _DELIMITERS[suffix]
    except KeyError:
        raise ValueError(
            f"{Path(path).name!r} must be a .csv or .tsv table "
            f"(got suffix {Path(path).suffix!r})"
        ) from None


_DELIMITER_NAMES: Final[dict[str, tuple[str, str]]] = {
    ",": ("comma-separated", CSV_SUFFIX),
    "\t": ("tab-separated", TSV_SUFFIX),
}


def misdelimited_table_hint(path: Union[str, Path], columns: "list[str]") -> str:
    """A sentence naming the real delimiter of a table read with the wrong one.

    A comma-separated file named ``.tsv`` parses, as a single column whose
    name holds every comma; the reverse holds for a tab-separated ``.csv``.
    The refusal that follows (no join key) would otherwise point the user at
    their column names rather than at the suffix (review M6).

    Args:
        path: The table, read with :func:`delimiter_for`'s delimiter.
        columns: The header it parsed to.

    Returns:
        ``"<name> looks like a comma-separated table; name it .csv"`` (or the
        tab/``.tsv`` form), or ``""`` when the header gives no such sign.
    """
    if len(columns) != 1:
        return ""
    used = delimiter_for(path)
    for delimiter, (label, suffix) in _DELIMITER_NAMES.items():
        if delimiter != used and delimiter in columns[0]:
            return f"{Path(path).name} looks like a {label} table; name it {suffix}"
    return ""


def write_tsv(
    frame: "pl.DataFrame | pd.DataFrame",
    path: Union[str, Path],
    *,
    commit_guard: "CommitGuard | None" = None,
) -> None:
    """Atomically write ``frame`` to ``path`` as TSV.

    A concurrent reader sees the old file or the complete new one, never a
    partial write: the table goes to a ``.tsv.tmp`` sibling first and replaces
    ``path`` only after it is synced.

    The bytes are polars ``write_csv(separator="\\t")``'s whichever library
    holds ``frame``: a pandas frame is converted with ``pl.from_pandas`` (its
    index dropped, ``NaN`` written as null, i.e. an empty field) and written
    by polars, so the same table is the same file. A null is an empty field;
    an empty string is ``""``. A datetime is ISO 8601 with a ``T`` and only
    the fractional digits its value needs (0, 3, 6 or 9), plus ``+HH:MM`` when
    it carries a time zone, so the bytes depend on the value and not on the
    column's time unit (pandas holds nanoseconds, a Python ``datetime`` lands
    in polars as microseconds).

    Args:
        frame: A polars or pandas DataFrame. A pandas index is not written.
        path: Destination file. Its parent is created when missing.
        commit_guard: Optional publication guard; the replace happens only
            inside it (see :func:`phenotypic.sdk_._atomic_io.atomic_write_with_writer`).

    Raises:
        OSError: If the write or the rename fails; the temp file is removed.
    """
    atomic_write_with_writer(
        path,
        _tsv_writer(frame),
        commit_guard=commit_guard,
        temp_suffix=f"{TSV_SUFFIX}.tmp",
    )


#: Tables that stay ``.csv`` beside, or instead of, a ``.tsv``: the GUI's
#: curation write target (spec B9), the GUI-written error table and the user's
#: metadata snapshot. All three live directly in ``deliverables/``, so the
#: stale-CSV sweep refuses these names there; the same stem in a split folder
#: is an ordinary generated table.
_KEPT_CSV_STEMS: Final[frozenset[str]] = frozenset(
    {"measurements", "error_analysis", "metadata"}
)


def remove_stale_csv_sibling(tsv_path: Union[str, Path]) -> bool:
    """Unlink the pre-cutover ``.csv`` twin of a ``.tsv`` that was just published.

    Called only after the ``.tsv`` is on disk, so a failed publication never
    costs the older ``.csv``. Refuses (``ValueError``) a path directly in a
    ``deliverables/`` folder whose stem names a table that stays CSV:
    ``measurements`` (GUI curation writes it, spec B9), ``error_analysis``
    (GUI-written) and ``metadata`` (the user's snapshot). Those files live
    only there, so ``measurements_by_feature/metadata.tsv`` is an ordinary
    split whose ``.csv`` twin is swept (review M3).

    Args:
        tsv_path: The published ``.tsv`` file.

    Returns:
        ``True`` when a ``.csv`` sibling existed and was removed.

    Raises:
        ValueError: If ``tsv_path`` is not a ``.tsv`` or names a kept CSV.
    """
    from ._io_constants import DIR_DELIVERABLES  # _io_constants imports this module

    path = Path(tsv_path)
    if path.suffix != TSV_SUFFIX:
        raise ValueError(f"{path.name!r} is not a .tsv file")
    if path.parent.name == DIR_DELIVERABLES and path.stem in _KEPT_CSV_STEMS:
        raise ValueError(f"{path.stem}.csv is never swept; it stays CSV")
    if not path.is_file():
        return False
    stale = path.with_suffix(CSV_SUFFIX)
    if not stale.is_file():
        return False
    stale.unlink()
    return True


def _tsv_writer(frame: Any):
    """Return a ``writer(temp_path)`` callback for ``frame``, by duck type.

    Polars writes every file; a pandas frame is converted first (review M5).
    """
    if hasattr(frame, "to_csv") and not hasattr(frame, "write_csv"):  # pandas
        import polars as pl

        frame = pl.from_pandas(frame, include_index=False)
    if hasattr(frame, "write_csv"):  # polars
        frame = _datetimes_as_iso_text(frame)
        return lambda temp: frame.write_csv(temp, separator="\t")
    raise TypeError(
        f"write_tsv needs a polars or pandas DataFrame, got {type(frame).__name__}"
    )


#: ``chrono`` formats for a datetime column: ``%.f`` writes 0, 3, 6 or 9
#: fractional digits as the value needs, independent of the column's unit.
_NAIVE_DATETIME_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S%.f"
_ZONED_DATETIME_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S%.f%:z"


def _datetimes_as_iso_text(frame: "pl.DataFrame") -> "pl.DataFrame":
    """Render each datetime column as unit-independent ISO 8601 text."""
    import polars as pl

    exprs = [
        pl.col(name).dt.to_string(
            _ZONED_DATETIME_FORMAT if dtype.time_zone else _NAIVE_DATETIME_FORMAT
        )
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(exprs) if exprs else frame


def read_delimited(path: Union[str, Path], **pandas_kwargs: Any) -> "pd.DataFrame":
    """Read a ``.csv`` or ``.tsv`` table with the delimiter its suffix implies.

    Args:
        path: The table to read.
        **pandas_kwargs: Passed to :func:`pandas.read_csv`. ``sep`` is chosen
            by :func:`delimiter_for` and may not be given.

    Returns:
        The parsed table.

    Raises:
        ValueError: If the suffix is neither ``.csv`` nor ``.tsv``.
        TypeError: If ``sep`` or ``delimiter`` is passed.
    """
    import pandas as pd

    for name in ("sep", "delimiter"):
        if name in pandas_kwargs:
            raise TypeError(
                f"read_delimited chooses the delimiter from the suffix; drop {name}="
            )
    return pd.read_csv(path, sep=delimiter_for(path), **pandas_kwargs)
