"""Contract tests for the GUDHI 3.13 cubical-persistence analysis adapter."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phenotypic.analysis._cubical_persistence import (
    PersistencePairsResult,
    cubical_persistence,
)


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = (
    ROOT
    / "docs/superpowers/specs/2026-07-13-fungi-detection-method-ports"
    / "refs/persistence/fixture.json"
)
REQUIRES_GUDHI = pytest.mark.skipif(
    find_spec("gudhi") is None,
    reason="golden cubical-persistence controls require the topology extra",
)


def _decode(values: list[float | str]) -> np.ndarray:
    """Decode the fixture's standards-compliant infinity strings."""
    decoded = []
    for value in values:
        if value == "+inf":
            decoded.append(np.inf)
        elif value == "-inf":
            decoded.append(-np.inf)
        else:
            decoded.append(float(value))
    return np.asarray(decoded, dtype=np.float64)


def _fixture() -> dict[str, Any]:
    """Load the pinned GUDHI 3.13 fixture."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _assert_case_matches_fixture(name: str) -> PersistencePairsResult:
    """Run one case and compare every public output exactly."""
    case = _fixture()["cases"][name]
    image = np.asarray(case["input"], dtype=np.float64)
    result = cubical_persistence(
        image,
        filtration=case["filtration"],
        min_persistence=case["min_persistence"],
    )
    expected = case["public_contract"]

    assert result.filtration == case["filtration"]
    for dimension in range(2):
        np.testing.assert_array_equal(
            result.birth_values[dimension],
            _decode(expected["birth_values"][dimension]),
        )
        np.testing.assert_array_equal(
            result.death_values[dimension],
            _decode(expected["death_values"][dimension]),
        )
        np.testing.assert_array_equal(
            result.lifetimes[dimension],
            _decode(expected["lifetimes"][dimension]),
        )
        np.testing.assert_array_equal(
            result.birth_cells[dimension],
            np.asarray(expected["birth_cells"][dimension], dtype=np.int64).reshape(
                (-1, 2)
            ),
        )
        np.testing.assert_array_equal(
            result.death_cells[dimension],
            np.asarray(expected["death_cells"][dimension], dtype=np.int64).reshape(
                (-1, 2)
            ),
        )
        np.testing.assert_array_equal(
            result.essential_cells[dimension],
            np.asarray(
                expected["essential_cells"][dimension], dtype=np.int64
            ).reshape((-1, 2)),
        )
    return result


@pytest.mark.parametrize(
    "name",
    [
        "single_sublevel",
        "single_superlevel",
        "four_peaks_superlevel",
        "four_peaks_threshold_equality",
        "one_hole_sublevel",
        "diagonal_touch_superlevel",
        "plateau_superlevel",
        "non_square_sublevel",
    ],
)
@REQUIRES_GUDHI
def test_every_public_output_matches_pinned_gudhi(name: str) -> None:
    """Golden control: all values and representative cells match the oracle."""
    _assert_case_matches_fixture(name)


def test_fixture_retains_every_source_visible_output() -> None:
    """The golden evidence must include all queried GUDHI intermediates."""
    fixture = _fixture()
    assert fixture["oracle"]["version"] == "3.13.0"
    assert fixture["oracle"]["homology_coeff_field"] == 11
    assert fixture["oracle"]["cell_convention"] == "top-dimensional cells"
    assert fixture["oracle"]["flat_order"] == "Fortran"
    expected_keys = {
        "all_cells",
        "top_dimensional_cells",
        "intervals_by_dimension",
        "regular_pair_coface_ids",
        "essential_coface_ids",
        "betti_numbers",
        "num_simplices",
        "dimension",
    }
    for case in fixture["cases"].values():
        assert set(case["source_visible"]) == expected_keys


@REQUIRES_GUDHI
def test_top_cell_hole_and_pair_cells_match_fixture() -> None:
    """Top-dimensional-cell construction retains the beta-1 ring interval."""
    result = _assert_case_matches_fixture("one_hole_sublevel")
    np.testing.assert_array_equal(result.birth_values[1], [0.0])
    np.testing.assert_array_equal(result.death_values[1], [2.0])
    np.testing.assert_array_equal(result.lifetimes[1], [2.0])
    np.testing.assert_array_equal(result.birth_cells[1], [[1, 2]])
    np.testing.assert_array_equal(result.death_cells[1], [[1, 1]])


@REQUIRES_GUDHI
def test_non_square_pair_ids_use_fortran_order() -> None:
    """Fortran-flat ID 2 in shape (2, 3) maps to row 0, column 1."""
    result = _assert_case_matches_fixture("non_square_sublevel")
    np.testing.assert_array_equal(result.essential_cells[0], [[0, 1]])


@REQUIRES_GUDHI
def test_diagonal_top_cells_share_a_vertex() -> None:
    """Diagonal top cells form one beta-0 class under the selected 8-connectivity."""
    result = _assert_case_matches_fixture("diagonal_touch_superlevel")
    np.testing.assert_array_equal(result.birth_values[0], [2.0])
    np.testing.assert_array_equal(result.death_values[0], [-np.inf])
    assert result.birth_values[1].size == 0


@REQUIRES_GUDHI
def test_superlevel_uses_original_intensity_coordinates() -> None:
    """Superlevel output reverses GUDHI's negated filtration values."""
    result = _assert_case_matches_fixture("four_peaks_superlevel")
    assert np.all(result.birth_values[0][:-1] >= result.death_values[0][:-1])
    np.testing.assert_array_equal(
        np.sort(result.lifetimes[0][np.isfinite(result.lifetimes[0])]),
        [1.0, 2.0, 3.0],
    )


@REQUIRES_GUDHI
def test_min_persistence_equality_is_excluded() -> None:
    """A finite class is retained only when lifetime is strictly above the threshold."""
    result = _assert_case_matches_fixture("four_peaks_threshold_equality")
    np.testing.assert_array_equal(result.birth_values[0], [4.0, 5.0])
    np.testing.assert_array_equal(result.death_values[0], [1.0, -np.inf])
    np.testing.assert_array_equal(result.lifetimes[0], [3.0, np.inf])


@REQUIRES_GUDHI
def test_essential_interval_sign_by_filtration() -> None:
    """Essential death is positive infinity for sublevel and negative for superlevel."""
    sublevel = _assert_case_matches_fixture("single_sublevel")
    superlevel = _assert_case_matches_fixture("single_superlevel")
    assert np.isposinf(sublevel.death_values[0][0])
    assert np.isneginf(superlevel.death_values[0][0])
    assert np.isposinf(sublevel.lifetimes[0][0])
    assert np.isposinf(superlevel.lifetimes[0][0])
    np.testing.assert_array_equal(sublevel.death_cells[0], [[-1, -1]])
    np.testing.assert_array_equal(superlevel.death_cells[0], [[-1, -1]])


@REQUIRES_GUDHI
def test_plateau_cells_match_pinned_gudhi() -> None:
    """Plateau representatives remain exact pinned-version drift evidence."""
    first = _assert_case_matches_fixture("plateau_superlevel")
    second = _assert_case_matches_fixture("plateau_superlevel")
    for left, right in zip(first.birth_cells, second.birth_cells, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(first.essential_cells[0], [[0, 1]])


@REQUIRES_GUDHI
def test_result_contains_all_pair_arrays_with_frozen_shapes_and_dtypes() -> None:
    """Output has exactly two dimensions and contract-pinned array dtypes/shapes."""
    result = _assert_case_matches_fixture("four_peaks_superlevel")
    assert isinstance(result, PersistencePairsResult)
    for values in (result.birth_values, result.death_values, result.lifetimes):
        assert len(values) == 2
        for array in values:
            assert array.dtype == np.dtype(np.float64)
            assert array.ndim == 1
    for cells in (result.birth_cells, result.death_cells, result.essential_cells):
        assert len(cells) == 2
        for array in cells:
            assert array.dtype == np.dtype(np.int64)
            assert array.ndim == 2
            assert array.shape[1] == 2


@REQUIRES_GUDHI
def test_result_dataclass_is_shallowly_frozen() -> None:
    """Fields are frozen while their documented NumPy arrays remain mutable."""
    result = _assert_case_matches_fixture("single_sublevel")
    with pytest.raises(FrozenInstanceError):
        result.filtration = "superlevel"  # type: ignore[misc]
    result.birth_values[0][0] = 7.0
    assert result.birth_values[0][0] == 7.0


@pytest.mark.parametrize(
    "invalid",
    [
        np.array(True),
        np.array([[True]]),
        np.array([[1.0 + 2.0j]]),
        np.array([[object()]], dtype=object),
        np.array([["1"]]),
        np.array(np.nan),
        np.array([1.0]),
        np.zeros((0, 2)),
        np.zeros((2, 0)),
        np.zeros((1, 1, 1)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[-np.inf]]),
    ],
)
def test_invalid_images_fail_before_optional_import(
    invalid: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All rejected image domains fail before resolving GUDHI."""
    module = importlib.import_module("phenotypic.analysis._cubical_persistence")

    def fail_import(name: str) -> Any:
        raise AssertionError(f"optional import reached for invalid image: {name}")

    monkeypatch.setattr(module.importlib, "import_module", fail_import)
    with pytest.raises(ValueError):
        module.cubical_persistence(invalid)


@pytest.mark.parametrize(
    "filtration", ["", "SUBLEVEL", "bright", None, 1, np.array(["sublevel"])]
)
def test_invalid_filtration_fails_before_optional_import(
    filtration: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only exact sublevel and superlevel strings are accepted."""
    module = importlib.import_module("phenotypic.analysis._cubical_persistence")
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected import of {name}"),
    )
    with pytest.raises(ValueError):
        module.cubical_persistence(np.ones((1, 1)), filtration=filtration)


@pytest.mark.parametrize(
    "minimum",
    [True, False, -1, -0.1, np.nan, np.inf, -np.inf, "0", None, np.array(0.0)],
)
def test_invalid_min_persistence_fails_before_optional_import(
    minimum: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistence thresholds must be finite nonnegative real non-booleans."""
    module = importlib.import_module("phenotypic.analysis._cubical_persistence")
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected import of {name}"),
    )
    with pytest.raises(ValueError):
        module.cubical_persistence(np.ones((1, 1)), min_persistence=minimum)


def test_analysis_import_does_not_import_gudhi() -> None:
    """The analysis package and private A11 module remain import-cheap."""
    script = """
import sys
import phenotypic.analysis
import phenotypic.analysis._cubical_persistence
assert 'gudhi' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_gudhi_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing optional dependency raises a call-time topology-extra error."""
    module = importlib.import_module("phenotypic.analysis._cubical_persistence")
    real_import = module.importlib.import_module

    def missing(name: str) -> Any:
        if name == "gudhi":
            raise ModuleNotFoundError("No module named 'gudhi'")
        return real_import(name)

    monkeypatch.setattr(module.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="topology"):
        module.cubical_persistence(np.ones((1, 1)))


def test_gudhi_receives_frozen_constructor_and_persistence_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter forwards copied top cells, field 11, and the exact threshold."""
    module = importlib.import_module("phenotypic.analysis._cubical_persistence")
    captured: dict[str, object] = {}

    class RecordingComplex:
        """Minimal source double recording the GUDHI call boundary."""

        def __init__(self, *, top_dimensional_cells: np.ndarray) -> None:
            captured["top_dimensional_cells"] = top_dimensional_cells

        def compute_persistence(self, **kwargs: object) -> None:
            captured["persistence_kwargs"] = kwargs

        def cofaces_of_persistence_pairs(
            self,
        ) -> tuple[list[np.ndarray], list[np.ndarray]]:
            return [np.empty((0, 2), dtype=np.int64)], [np.array([0])]

    class FakeGudhi:
        CubicalComplex = RecordingComplex

    monkeypatch.setattr(module, "_import_gudhi", lambda: FakeGudhi)
    image = np.array([[1, 2], [3, 4]], dtype=np.int16)
    result = module.cubical_persistence(
        image,
        min_persistence=np.float32(1.25),
    )
    received = captured["top_dimensional_cells"]
    assert isinstance(received, np.ndarray)
    assert received.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(received, -image.astype(np.float64))
    assert not np.shares_memory(received, image)
    assert captured["persistence_kwargs"] == {
        "homology_coeff_field": 11,
        "min_persistence": 1.25,
    }
    assert result.filtration == "superlevel"


def test_input_is_unchanged_and_result_does_not_alias_it() -> None:
    """A mutating external constructor cannot alter or alias the caller's image."""
    module = importlib.import_module("phenotypic.analysis._cubical_persistence")

    class MutatingComplex:
        """Minimal source double that mutates the received top-cell array."""

        def __init__(self, *, top_dimensional_cells: np.ndarray) -> None:
            top_dimensional_cells[...] = 99.0

        def compute_persistence(self, **kwargs: object) -> None:
            assert kwargs == {"homology_coeff_field": 11, "min_persistence": 0.0}

        def cofaces_of_persistence_pairs(
            self,
        ) -> tuple[list[np.ndarray], list[np.ndarray]]:
            return [np.empty((0, 2), dtype=np.int64)], [np.array([0])]

    class FakeGudhi:
        CubicalComplex = MutatingComplex

    image = np.array([[5.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    expected = image.copy()
    original_import = module._import_gudhi
    module._import_gudhi = lambda: FakeGudhi
    try:
        result = module.cubical_persistence(image, filtration="sublevel")
    finally:
        module._import_gudhi = original_import
    np.testing.assert_array_equal(image, expected)
    for array in result.birth_values + result.death_values + result.lifetimes:
        assert not np.shares_memory(array, image)


def test_a11_has_no_operation_or_reconstruction_surface() -> None:
    """G1 remains analysis-only and does not invent unsupported image reconstruction."""
    module = importlib.import_module("phenotypic.analysis._cubical_persistence")
    assert not hasattr(module, "persistence_denoise")
    assert not hasattr(module, "FocusEdgePersistenceDenoise")
    assert not (ROOT / "src/phenotypic/sdk_/reconnect/_persistence.py").exists()
    assert not (
        ROOT / "src/phenotypic/enhance/_focus_edge_persistence_denoise.py"
    ).exists()


MUTANTS = [
    pytest.param(
        "source_image if selected_filtration == \"sublevel\" else -source_image",
        "-source_image if selected_filtration == \"sublevel\" else source_image",
        "test_superlevel_uses_original_intensity_coordinates",
        id="reverse-filtration-sign",
    ),
    pytest.param(
        "CubicalComplex(top_dimensional_cells=filtration_values)",
        "CubicalComplex(vertices=filtration_values)",
        "test_top_cell_hole_and_pair_cells_match_fixture",
        id="construct-from-vertices",
    ),
    pytest.param(
        'np.unravel_index(ids, shape, order="F")',
        'np.unravel_index(ids, shape, order="C")',
        "test_non_square_pair_ids_use_fortran_order",
        id="decode-cofaces-in-c-order",
    ),
    pytest.param(
        "np.column_stack((rows, columns))",
        "np.column_stack((columns, rows))",
        "test_non_square_pair_ids_use_fortran_order",
        id="swap-row-column",
    ),
    pytest.param(
        "    return PersistencePairsResult(\n",
        "    dimension_zero, dimension_one = dimension_one, dimension_zero\n"
        "    return PersistencePairsResult(\n",
        "test_top_cell_hole_and_pair_cells_match_fixture",
        id="swap-homology-dimensions",
    ),
    pytest.param(
        "essential = list(essential_raw)",
        "essential = []",
        "test_essential_interval_sign_by_filtration",
        id="drop-essential-intervals",
    ),
    pytest.param(
        "min_persistence=threshold,",
        "min_persistence=np.nextafter(threshold, -np.inf),",
        "test_min_persistence_equality_is_excluded",
        id="make-threshold-inclusive",
    ),
    pytest.param(
        "regular_lifetime = regular_birth - regular_death",
        "regular_lifetime = regular_death - regular_birth",
        "test_superlevel_uses_original_intensity_coordinates",
        id="reverse-superlevel-lifetime",
    ),
    pytest.param(
        "essential_cells = _coface_coordinates(essential_ids, shape)",
        "essential_cells = np.zeros((essential_ids.size, 2), dtype=np.int64)",
        "test_plateau_cells_match_pinned_gudhi",
        id="invent-plateau-canonical-cell",
    ),
    pytest.param(
        "CubicalComplex(top_dimensional_cells=filtration_values)",
        "CubicalComplex(vertices=filtration_values)",
        "test_diagonal_top_cells_share_a_vertex",
        id="use-paper-four-connectivity-convention",
    ),
    pytest.param(
        "import importlib\n",
        'import importlib\n\nimportlib.import_module("gudhi")\n',
        "test_analysis_import_does_not_import_gudhi",
        id="eager-gudhi-import",
    ),
    pytest.param(
        "        raise ImportError(\n",
        "        return None  # type: ignore[return-value]\n        raise ImportError(\n",
        "test_missing_gudhi_is_actionable",
        id="swallow-missing-dependency",
    ),
    pytest.param(
        "return np.array(array, dtype=np.float64, copy=True)",
        "return np.asarray(array, dtype=np.float64)",
        "test_input_is_unchanged_and_result_does_not_alias_it",
        id="remove-input-copy",
    ),
    pytest.param(
        "    return PersistencePairsResult(\n",
        "    return source_image  # type: ignore[return-value]\n"
        "    return PersistencePairsResult(\n",
        "test_result_contains_all_pair_arrays_with_frozen_shapes_and_dtypes",
        id="return-image-instead-of-pairs",
    ),
    pytest.param(
        "\ndef cubical_persistence(\n",
        "\ndef persistence_denoise(image: np.ndarray) -> np.ndarray:\n"
        "    return image\n\n\ndef cubical_persistence(\n",
        "test_a11_has_no_operation_or_reconstruction_surface",
        id="add-unsupported-reconstruction",
    ),
]


@pytest.mark.parametrize(("needle", "replacement", "killing_test"), MUTANTS)
@REQUIRES_GUDHI
def test_required_mutants_are_killed(
    needle: str,
    replacement: str,
    killing_test: str,
    tmp_path: Path,
) -> None:
    """Apply each required mutant alone and prove its named test fails."""
    source_path = ROOT / "src/phenotypic/analysis/_cubical_persistence.py"
    source = source_path.read_text(encoding="utf-8")
    assert source.count(needle) == 1, f"mutation anchor drifted: {needle!r}"
    mutated = source.replace(needle, replacement, 1)
    compile(mutated, str(source_path), "exec")

    mutant_path = tmp_path / "_cubical_persistence_mutant.py"
    mutant_path.write_text(mutated, encoding="utf-8")
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """import importlib.util
import os
import sys

name = "phenotypic.analysis._cubical_persistence"
spec = importlib.util.spec_from_file_location(name, os.environ["A11_MUTANT_PATH"])
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["A11_MUTANT_PATH"] = str(mutant_path)
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(tmp_path), environment.get("PYTHONPATH", ""))
        if part
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__).resolve()),
            "-q",
            "-o",
            "addopts=",
            "-k",
            killing_test,
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    evidence = completed.stdout + completed.stderr
    assert completed.returncode == 1, evidence
    assert "1 failed" in evidence, evidence
    assert killing_test in evidence, evidence
