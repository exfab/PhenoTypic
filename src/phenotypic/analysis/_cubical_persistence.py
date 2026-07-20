"""Analysis-only cubical persistence through the optional GUDHI dependency."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


Filtration = Literal["sublevel", "superlevel"]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PersistencePairsResult:
    """Cubical-persistence intervals and their top-cell representatives.

    Each tuple contains exactly two arrays, indexed by homology dimension
    ``0`` and ``1``. Regular pairs retain GUDHI's source order and essential
    pairs are appended. Essential intervals use ``(-1, -1)`` as their death
    coordinate.

    Attributes:
        birth_values: Birth intensities in the input image's coordinates.
        death_values: Death intensities, including signed infinity for
            essential intervals.
        lifetimes: Nonnegative persistence lifetimes.
        birth_cells: Birth top cells as ``(row, column)`` coordinates.
        death_cells: Death top cells, or ``(-1, -1)`` for essential intervals.
        essential_cells: Birth coordinates for essential intervals only.
        filtration: The selected ``"sublevel"`` or ``"superlevel"`` filtration.

    Note:
        The dataclass fields are frozen, but the NumPy arrays remain mutable.
    """

    birth_values: tuple[FloatArray, FloatArray]
    death_values: tuple[FloatArray, FloatArray]
    lifetimes: tuple[FloatArray, FloatArray]
    birth_cells: tuple[IntArray, IntArray]
    death_cells: tuple[IntArray, IntArray]
    essential_cells: tuple[IntArray, IntArray]
    filtration: Filtration


def _validate_image(image: np.ndarray) -> FloatArray:
    """Return a copied finite two-dimensional real image as ``float64``.

    Args:
        image: Candidate numeric image.

    Returns:
        Independent two-dimensional ``float64`` image.

    Raises:
        ValueError: If the input is not a nonempty, finite, real-valued numeric
            two-dimensional array.
    """
    try:
        array = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise ValueError("image must be convertible to a numeric 2-D array") from exc

    if array.ndim != 2:
        raise ValueError(f"image must be two-dimensional; got shape {array.shape}")
    if array.size == 0:
        raise ValueError("image must have no empty axes")
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError("image must be real-valued numeric data, not boolean")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"image must be numeric; got dtype {array.dtype}")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("image must be real-valued, not complex")
    if not np.isfinite(array).all():
        raise ValueError("image must contain only finite values")
    return np.array(array, dtype=np.float64, copy=True)


def _validate_filtration(filtration: object) -> Filtration:
    """Validate and narrow the filtration string."""
    if not isinstance(filtration, str):
        raise ValueError(
            "filtration must be exactly 'sublevel' or 'superlevel'; "
            f"got {filtration!r}"
        )
    if filtration == "sublevel":
        return "sublevel"
    if filtration == "superlevel":
        return "superlevel"
    raise ValueError(
        "filtration must be exactly 'sublevel' or 'superlevel'; "
        f"got {filtration!r}"
    )


def _validate_min_persistence(min_persistence: object) -> float:
    """Return a finite nonnegative real persistence threshold."""
    if isinstance(min_persistence, (bool, np.bool_)) or not isinstance(
        min_persistence, Real
    ):
        raise ValueError("min_persistence must be a real scalar other than bool")
    value = float(min_persistence)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("min_persistence must be finite and at least zero")
    return value


def _import_gudhi() -> Any:
    """Import GUDHI lazily with an actionable optional-dependency error."""
    try:
        return importlib.import_module("gudhi")
    except ImportError as exc:
        raise ImportError(
            "cubical_persistence requires the optional GUDHI dependency. "
            "Install PhenoTypic's topology extra with `uv sync --extra topology`."
        ) from exc


def _coface_coordinates(ids: IntArray, shape: tuple[int, int]) -> IntArray:
    """Convert Fortran-flat GUDHI top-cell IDs to row/column coordinates."""
    if ids.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    rows, columns = np.unravel_index(ids, shape, order="F")
    return np.column_stack((rows, columns)).astype(np.int64, copy=False)


def _ids_for_dimension(
    arrays: list[np.ndarray], dimension: int, columns: int
) -> IntArray:
    """Return one dimension's pair IDs with a stable empty shape."""
    if dimension >= len(arrays):
        return np.empty((0, columns), dtype=np.int64)
    return np.asarray(arrays[dimension], dtype=np.int64).reshape((-1, columns))


def _convert_dimension(
    *,
    dimension: int,
    regular: list[np.ndarray],
    essential: list[np.ndarray],
    flat_filtration_values: FloatArray,
    shape: tuple[int, int],
    filtration: Filtration,
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray, IntArray, IntArray]:
    """Convert one GUDHI coface group to the frozen public representation."""
    regular_ids = _ids_for_dimension(regular, dimension, 2)
    essential_ids = _ids_for_dimension(essential, dimension, 1).reshape((-1,))

    birth_filtration = flat_filtration_values[regular_ids[:, 0]]
    death_filtration = flat_filtration_values[regular_ids[:, 1]]
    essential_birth_filtration = flat_filtration_values[essential_ids]
    if filtration == "sublevel":
        regular_birth = birth_filtration
        regular_death = death_filtration
        essential_birth = essential_birth_filtration
        essential_death = np.full(essential_ids.size, np.inf, dtype=np.float64)
        regular_lifetime = regular_death - regular_birth
    else:
        regular_birth = -birth_filtration
        regular_death = -death_filtration
        essential_birth = -essential_birth_filtration
        essential_death = np.full(essential_ids.size, -np.inf, dtype=np.float64)
        regular_lifetime = regular_birth - regular_death

    birth_values = np.concatenate((regular_birth, essential_birth)).astype(
        np.float64, copy=False
    )
    death_values = np.concatenate((regular_death, essential_death)).astype(
        np.float64, copy=False
    )
    lifetimes = np.concatenate(
        (regular_lifetime, np.full(essential_ids.size, np.inf, dtype=np.float64))
    ).astype(np.float64, copy=False)

    regular_birth_cells = _coface_coordinates(regular_ids[:, 0], shape)
    regular_death_cells = _coface_coordinates(regular_ids[:, 1], shape)
    essential_cells = _coface_coordinates(essential_ids, shape)
    birth_cells = np.concatenate(
        (regular_birth_cells, essential_cells), axis=0
    ).astype(np.int64, copy=False)
    death_cells = np.concatenate(
        (
            regular_death_cells,
            np.full((essential_ids.size, 2), -1, dtype=np.int64),
        ),
        axis=0,
    ).astype(np.int64, copy=False)
    return (
        birth_values,
        death_values,
        lifetimes,
        birth_cells,
        death_cells,
        essential_cells,
    )


def cubical_persistence(
    image: np.ndarray,
    *,
    filtration: Filtration = "superlevel",
    min_persistence: float = 0.0,
) -> PersistencePairsResult:
    """Compute beta-0 and beta-1 persistence from image top cells.

    Pixels are closed, nonperiodic GUDHI top-dimensional cells. Consequently,
    foreground cells touching at a corner are connected. GUDHI receives the
    copied image for sublevel persistence and its negation for superlevel
    persistence. Returned values are always converted back to the original
    image intensity coordinates.

    Args:
        image: Nonempty, finite, real-valued numeric two-dimensional array.
        filtration: Either ``"sublevel"`` or ``"superlevel"``. Defaults to
            ``"superlevel"`` for bright structures.
        min_persistence: Finite nonnegative lifetime threshold. A finite class
            is retained only when its lifetime is strictly greater than this
            value. Essential classes are always retained.

    Returns:
        Persistence values and top-cell representatives for homology
        dimensions zero and one.

    Raises:
        ValueError: If an input or parameter violates the frozen contract.
        ImportError: If the optional GUDHI dependency is unavailable for a
            valid nonempty call.
    """
    source_image = _validate_image(image)
    selected_filtration = _validate_filtration(filtration)
    threshold = _validate_min_persistence(min_persistence)
    gudhi = _import_gudhi()

    filtration_values = (
        source_image if selected_filtration == "sublevel" else -source_image
    )
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=filtration_values)
    complex_.compute_persistence(
        homology_coeff_field=11,
        min_persistence=threshold,
    )
    regular_raw, essential_raw = complex_.cofaces_of_persistence_pairs()
    regular = list(regular_raw)
    essential = list(essential_raw)
    flat_filtration_values = filtration_values.ravel(order="F")

    dimension_zero = _convert_dimension(
        dimension=0,
        regular=regular,
        essential=essential,
        flat_filtration_values=flat_filtration_values,
        shape=source_image.shape,
        filtration=selected_filtration,
    )
    dimension_one = _convert_dimension(
        dimension=1,
        regular=regular,
        essential=essential,
        flat_filtration_values=flat_filtration_values,
        shape=source_image.shape,
        filtration=selected_filtration,
    )
    return PersistencePairsResult(
        birth_values=(dimension_zero[0], dimension_one[0]),
        death_values=(dimension_zero[1], dimension_one[1]),
        lifetimes=(dimension_zero[2], dimension_one[2]),
        birth_cells=(dimension_zero[3], dimension_one[3]),
        death_cells=(dimension_zero[4], dimension_one[4]),
        essential_cells=(dimension_zero[5], dimension_one[5]),
        filtration=selected_filtration,
    )
