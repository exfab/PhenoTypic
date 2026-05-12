import re
import numpy as np

_WELL_PATTERN = re.compile(r"^\s*([A-Za-z]+)(\d+)\s*$")


def _row_letters_to_idx(letters: str) -> int:
    """Convert plate row letters to a 0-based row index.

    Uses bijective base-26 encoding matching the ANSI/SBS microplate standard:
    A=0, B=1, ..., Z=25, AA=26, AB=27, ..., AZ=51, BA=52, ...

    Args:
        letters: Row label string, e.g. ``"A"``, ``"H"``, ``"AA"``. Case-insensitive.

    Returns:
        0-based integer row index.
    """
    idx = 0
    for ch in letters.upper():
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _decode_one_well(label: str, n_rows: int, order: str) -> int:
    """Decode a single well label to a 0-based linear index.

    Args:
        label: Well identifier string, e.g. ``"A1"``, ``"B11"``, ``"AA3"``.
        n_rows: Number of rows in the plate layout.
        order: Indexing order; ``"column"`` for column-major,
            ``"row"`` for row-major.

    Returns:
        0-based linear index, or ``-1`` if the label does not match the
        expected ``<letters><digits>`` format.
    """
    m = _WELL_PATTERN.match(label)
    if m is None:
        return -1
    row_idx = _row_letters_to_idx(m.group(1))
    col_idx = int(m.group(2)) - 1
    if order == "column":
        return col_idx * n_rows + row_idx
    else:
        raise NotImplementedError(
                "Row-major indexing requires n_cols; pass order='column' or extend "
                "_decode_one_well to accept n_cols."
        )


def _to_str_list(well) -> list[str]:
    """Normalise heterogeneous array-like inputs to a flat list of strings.

    Accepts plain strings, Python lists, NumPy arrays, and any object
    exposing a ``.to_numpy()`` method (Polars Series, Pandas Series).
    Neither Polars nor Pandas is imported; compatibility is via duck-typing.

    Args:
        well: Scalar string or array-like of well label strings.

    Returns:
        Flat list of strings.
    """
    if isinstance(well, str):
        return [well]
    if hasattr(well, "to_numpy"):
        return well.to_numpy().astype(str).tolist()
    if isinstance(well, np.ndarray):
        return well.astype(str).tolist()
    return [str(w) for w in well]


def decode_well_position(
        well,
        n_rows: int = 8,
        order: str = "column",
) -> np.ndarray:
    """Decode microplate well identifiers to 0-based linear indices.

    Supports scalar strings, Python lists, NumPy arrays, Pandas Series, and
    Polars Series as input. Output is always a NumPy int32 array of the same
    length, or a scalar int32 for scalar input. The well position is distinct from
    the `GRID` labels. These are in relation to the actual original well plate, and
    may have an offset in comparison to the `GRID` positions from a `GridImage`

    Row encoding follows the ANSI/SBS microplate convention:
    single letters A-Z map to rows 0-25; double letters AA-AZ to 26-51, etc.
    This covers 96-well (A-H), 384-well (A-P), and 1536-well (A-AF) formats.

    Column-major index formula: ``(col - 1) * n_rows + row_idx`` (0-based).

    Args:
        well: Well label(s) of the form ``<letter(s)><digits>``, e.g. ``"A1"``,
            ``"B11"``, ``"AA3"``. Case-insensitive. Leading/trailing whitespace
            is stripped. Accepts ``str``, ``list[str]``, ``np.ndarray``,
            ``pandas.Series``, or ``polars.Series``.
        n_rows: Number of rows in the plate. Defaults to 8 (96-well).
            Use 16 for 384-well, 32 for 1536-well.
        order: Linear indexing order. ``"column"`` (default) uses column-major
            (Fortran) order. ``"row"`` is reserved but not yet implemented.

    Returns:
        np.ndarray of dtype int32 containing 0-based linear indices.
        Invalid or unparseable labels produce ``-1``. Returns a scalar
        int32 when a single string is passed.

    Raises:
        NotImplementedError: If ``order="row"`` is requested.

    Examples:
        >>> decode_well_position("A1")
        array(0, dtype=int32)
        >>> decode_well_position(["A1", "B1", "A2"], n_rows=8)
        array([ 0,  1,  8], dtype=int32)
        >>> decode_well_position(["A11", "H12"], n_rows=8)
        array([80, 95], dtype=int32)
    """
    scalar_input = isinstance(well, str)
    labels = _to_str_list(well)
    result = np.array(
            [_decode_one_well(label, n_rows, order) for label in labels],
            dtype=np.int32,
    )
    return result[0] if scalar_input else result
