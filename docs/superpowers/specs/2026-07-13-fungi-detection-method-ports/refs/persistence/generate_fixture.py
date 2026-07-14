"""Generate the pinned GUDHI 3.13.0 cubical-persistence fixture.

This is a source-oracle harness. It deliberately imports GUDHI and must never be
imported by production code or the standalone logic-validation script.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Literal

import gudhi
import numpy as np


REFERENCE_VERSION = "3.13.0"
HERE = Path(__file__).resolve().parent
FIXTURE_PATH = HERE / "fixture.json"


def _json_float(value: float) -> float | str:
    """Return a standards-compliant JSON representation of a float."""
    if np.isposinf(value):
        return "+inf"
    if np.isneginf(value):
        return "-inf"
    if np.isnan(value):
        raise AssertionError("The oracle emitted NaN")
    return float(value)


def _json_array(values: np.ndarray) -> list[Any]:
    """Convert an array, encoding non-finite float values as strings."""
    return np.vectorize(_json_float, otypes=[object])(values).tolist()


def _coordinates(ids: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Convert GUDHI's Fortran-flat top-cell IDs to row/column coordinates."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    rows, columns = np.unravel_index(ids, shape, order="F")
    return np.column_stack((rows, columns)).astype(np.int64, copy=False)


def _capture_case(
    image: np.ndarray,
    *,
    filtration: Literal["sublevel", "superlevel"],
    min_persistence: float,
) -> dict[str, Any]:
    """Capture every GUDHI output used by the frozen public contract."""
    source_image = np.asarray(image, dtype=np.float64)
    filtration_values = source_image if filtration == "sublevel" else -source_image
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=filtration_values)
    complex_.compute_persistence(
        homology_coeff_field=11,
        min_persistence=min_persistence,
    )
    regular, essential = complex_.cofaces_of_persistence_pairs()
    flat_values = filtration_values.ravel(order="F")

    public: dict[str, list[Any]] = {
        "birth_values": [],
        "death_values": [],
        "lifetimes": [],
        "birth_cells": [],
        "death_cells": [],
        "essential_cells": [],
    }
    raw_regular: list[list[list[int]]] = []
    raw_essential: list[list[int]] = []

    for dimension in range(2):
        regular_ids = (
            np.asarray(regular[dimension], dtype=np.int64).reshape((-1, 2))
            if dimension < len(regular)
            else np.empty((0, 2), dtype=np.int64)
        )
        essential_ids = (
            np.asarray(essential[dimension], dtype=np.int64).reshape((-1,))
            if dimension < len(essential)
            else np.empty((0,), dtype=np.int64)
        )
        raw_regular.append(regular_ids.tolist())
        raw_essential.append(essential_ids.tolist())

        birth_filtration = flat_values[regular_ids[:, 0]]
        death_filtration = flat_values[regular_ids[:, 1]]
        essential_birth_filtration = flat_values[essential_ids]
        if filtration == "sublevel":
            birth = birth_filtration
            death = death_filtration
            essential_birth = essential_birth_filtration
            essential_death = np.full(essential_ids.size, np.inf)
            lifetime = death - birth
        else:
            birth = -birth_filtration
            death = -death_filtration
            essential_birth = -essential_birth_filtration
            essential_death = np.full(essential_ids.size, -np.inf)
            lifetime = birth - death

        all_birth = np.concatenate((birth, essential_birth)).astype(np.float64)
        all_death = np.concatenate((death, essential_death)).astype(np.float64)
        all_lifetime = np.concatenate(
            (lifetime, np.full(essential_ids.size, np.inf))
        ).astype(np.float64)
        regular_birth_cells = _coordinates(regular_ids[:, 0], source_image.shape)
        regular_death_cells = _coordinates(regular_ids[:, 1], source_image.shape)
        essential_birth_cells = _coordinates(essential_ids, source_image.shape)
        all_birth_cells = np.concatenate(
            (regular_birth_cells, essential_birth_cells), axis=0
        )
        all_death_cells = np.concatenate(
            (
                regular_death_cells,
                np.full((essential_ids.size, 2), -1, dtype=np.int64),
            ),
            axis=0,
        )

        public["birth_values"].append(_json_array(all_birth))
        public["death_values"].append(_json_array(all_death))
        public["lifetimes"].append(_json_array(all_lifetime))
        public["birth_cells"].append(all_birth_cells.tolist())
        public["death_cells"].append(all_death_cells.tolist())
        public["essential_cells"].append(essential_birth_cells.tolist())

    intervals = []
    for dimension in range(2):
        interval_array = complex_.persistence_intervals_in_dimension(dimension)
        intervals.append(_json_array(interval_array))

    return {
        "input": source_image.tolist(),
        "filtration": filtration,
        "min_persistence": min_persistence,
        "filtration_values": filtration_values.tolist(),
        "source_visible": {
            "all_cells": complex_.all_cells().tolist(),
            "top_dimensional_cells": complex_.top_dimensional_cells().tolist(),
            "intervals_by_dimension": intervals,
            "regular_pair_coface_ids": raw_regular,
            "essential_coface_ids": raw_essential,
            "betti_numbers": complex_.betti_numbers(),
            "num_simplices": complex_.num_simplices(),
            "dimension": complex_.dimension(),
        },
        "public_contract": public,
    }


def generate_persistence_fixture() -> None:
    """Execute the pinned oracle and write a deterministic fixture."""
    if gudhi.__version__ != REFERENCE_VERSION:
        raise RuntimeError(
            f"Expected GUDHI {REFERENCE_VERSION}, found {gudhi.__version__}"
        )

    cases = {
        "single_sublevel": _capture_case(
            np.array([[2.5]]), filtration="sublevel", min_persistence=0.0
        ),
        "single_superlevel": _capture_case(
            np.array([[2.5]]), filtration="superlevel", min_persistence=0.0
        ),
        "four_peaks_superlevel": _capture_case(
            np.array([[5.0, 1.0, 4.0], [1.0, 0.0, 1.0], [3.0, 1.0, 2.0]]),
            filtration="superlevel",
            min_persistence=0.0,
        ),
        "four_peaks_threshold_equality": _capture_case(
            np.array([[5.0, 1.0, 4.0], [1.0, 0.0, 1.0], [3.0, 1.0, 2.0]]),
            filtration="superlevel",
            min_persistence=2.0,
        ),
        "one_hole_sublevel": _capture_case(
            np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]),
            filtration="sublevel",
            min_persistence=0.0,
        ),
        "diagonal_touch_superlevel": _capture_case(
            np.array([[0.0, 2.0], [2.0, 0.0]]),
            filtration="superlevel",
            min_persistence=0.0,
        ),
        "plateau_superlevel": _capture_case(
            np.array([[2.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            filtration="superlevel",
            min_persistence=0.0,
        ),
        "non_square_sublevel": _capture_case(
            np.array([[3.0, -1.0, 2.0], [0.0, 4.0, 1.0]]),
            filtration="sublevel",
            min_persistence=0.0,
        ),
    }
    payload = {
        "schema_version": 1,
        "oracle": {
            "package": "gudhi",
            "version": gudhi.__version__,
            "module_origin": "locally pinned CPython 3.12 macOS universal wheel",
            "module_filename": Path(gudhi.__file__).name,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "homology_coeff_field": 11,
            "cell_convention": "top-dimensional cells",
            "flat_order": "Fortran",
        },
        "cases": cases,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    FIXTURE_PATH.write_text(serialized, encoding="utf-8")
    print(f"wrote {FIXTURE_PATH}")
    print(f"sha256 {hashlib.sha256(serialized.encode()).hexdigest()}")


if __name__ == "__main__":
    generate_persistence_fixture()
