"""One delimiter rule and one TSV writer for every tabular text file (spec Part B).

Generated tables are TSV: UTF-8, one header row, ``\\t`` separator, ``\\n`` line
ends, and ``"`` quoting for any field that contains a tab, a newline or a quote.
That is exactly what polars ``write_csv(separator="\\t")`` and pandas
``to_csv(sep="\\t", lineterminator="\\n")`` emit by default, so
:func:`write_tsv` only chooses the separator and the line end.

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


def write_tsv(frame: "pl.DataFrame | pd.DataFrame", path: Union[str, Path]) -> None:
    """Atomically write ``frame`` to ``path`` as TSV.

    A concurrent reader sees the old file or the complete new one, never a
    partial write: the table goes to a ``.tsv.tmp`` sibling first and replaces
    ``path`` only after it is synced.

    Args:
        frame: A polars or pandas DataFrame. A pandas index is not written.
        path: Destination file. Its parent is created when missing.

    Raises:
        OSError: If the write or the rename fails; the temp file is removed.
    """
    atomic_write_with_writer(
        path, _tsv_writer(frame), temp_suffix=f"{TSV_SUFFIX}.tmp"
    )


def _tsv_writer(frame: Any):
    """Return a ``writer(temp_path)`` callback for ``frame``, by duck type."""
    if hasattr(frame, "write_csv"):  # polars
        return lambda temp: frame.write_csv(temp, separator="\t")
    if hasattr(frame, "to_csv"):  # pandas
        return lambda temp: frame.to_csv(
            temp, sep="\t", index=False, lineterminator="\n", encoding="utf-8"
        )
    raise TypeError(
        f"write_tsv needs a polars or pandas DataFrame, got {type(frame).__name__}"
    )


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
