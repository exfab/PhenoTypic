"""Golden-equivalence tests for the pydantic v2 operation migration.

Each test re-runs one scenario from :mod:`tests.migration._scenarios`
against its frozen input and asserts the result is **bit-exact** equal
to the golden captured by ``scripts/capture_migration_goldens.py``:

* image operations -> ``numpy.testing.assert_array_equal`` on the
  resulting ``detect_mat`` / ``objmask`` / ``objmap`` arrays;
* measurers / post-measurement transforms / analyzers ->
  ``pandas.testing.assert_frame_equal(check_exact=True,
  check_dtype=True)``.

On the current, unmigrated code this suite must pass -- it compares the
code against goldens captured from that same code. After the pydantic
migration it proves the migration changed no numerical behavior.

The suite is parametrized so it can be sliced by subpackage::

    uv run pytest tests/migration -k detect      # just detect/
    uv run pytest tests/migration -k analysis    # just analyzers
    uv run pytest tests/migration -q             # everything

Each test id is the scenario id (e.g. ``detect.OtsuDetector`` or
``enhance.GaussianBlur.sigma4``), so ``-k`` filters on subpackage,
class name, or variant.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tests.migration._runner import (
    FrameGolden,
    ImageGolden,
    golden_path,
    load_golden,
    normalize_frame,
    run_scenario,
)
from tests.migration._scenarios import Scenario, build_scenarios

# Build the scenario list once at import time. ``build_scenarios`` does
# not touch the frozen inputs, so this is safe at collection time.
_SCENARIOS: list[Scenario] = build_scenarios()


def _scenario_id(scenario: Scenario) -> str:
    """Return the pytest parametrize id for a scenario.

    Args:
        scenario: The scenario to identify.

    Returns:
        The scenario's ``scenario_id`` slug.
    """
    return scenario.scenario_id


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=_scenario_id)
def test_operation_matches_golden(scenario: Scenario) -> None:
    """Assert a scenario reproduces its captured golden exactly.

    Args:
        scenario: The operation/analyzer scenario under test.
    """
    if scenario.structural_only:
        _assert_structural_golden(scenario)
        return

    golden = load_golden(scenario)
    result = run_scenario(scenario)

    if isinstance(golden, ImageGolden):
        assert isinstance(result, ImageGolden), (
            f"{scenario.scenario_id}: expected an image result"
        )
        _assert_image_equal(scenario, result, golden)
    elif isinstance(golden, FrameGolden):
        assert isinstance(result, FrameGolden), (
            f"{scenario.scenario_id}: expected a frame result"
        )
        _assert_frame_equal(scenario, result, golden)
    else:  # pragma: no cover - defensive
        pytest.fail(
            f"{scenario.scenario_id}: unknown golden type "
            f"{type(golden).__name__}"
        )


def _assert_image_equal(
    scenario: Scenario, result: ImageGolden, golden: ImageGolden
) -> None:
    """Assert a result's image arrays match the golden.

    Only the components the scenario captured are compared (an
    enhancer's golden carries just ``detect_mat``, a detector's just
    ``objmask`` + ``objmap``). The float ``detect_mat`` honors
    ``scenario.tolerance`` -- bit-exact via ``assert_array_equal`` when
    the tolerance is ``0``, otherwise ``assert_allclose`` -- while the
    integer ``objmask`` / ``objmap`` are always required to match
    exactly (label maps must be reproducible regardless of float
    jitter).

    Args:
        scenario: The scenario under test (supplies tolerance and the
            captured component set).
        result: Freshly computed image golden.
        golden: Persisted reference image golden.
    """
    sid = scenario.scenario_id
    assert set(result.arrays) == set(golden.arrays), (
        f"{sid}: component mismatch -- result has "
        f"{sorted(result.arrays)}, golden has {sorted(golden.arrays)}"
    )
    for name, golden_arr in golden.arrays.items():
        result_arr = result.arrays[name]
        if name == "detect_mat" and scenario.tolerance > 0:
            np.testing.assert_allclose(
                result_arr,
                golden_arr,
                atol=scenario.tolerance,
                rtol=0,
                err_msg=(
                    f"{sid}: {name} drifted beyond tolerance "
                    f"{scenario.tolerance}"
                ),
            )
        else:
            np.testing.assert_array_equal(
                result_arr,
                golden_arr,
                err_msg=f"{sid}: {name} drifted from golden",
            )


def _assert_frame_equal(
    scenario: Scenario, result: FrameGolden, golden: FrameGolden
) -> None:
    """Assert a result frame matches the golden frame.

    Both frames are :func:`normalize_frame`-d so a named/categorical
    index is materialized into columns and compared faithfully. With
    ``scenario.tolerance == 0`` the comparison is exact (values and
    dtypes); a positive tolerance relaxes float columns to that
    absolute tolerance.

    Args:
        scenario: The scenario under test (supplies the tolerance).
        result: Freshly computed frame golden.
        golden: Persisted reference frame golden.
    """
    fresh = normalize_frame(result.frame)
    exact = scenario.tolerance == 0
    try:
        pd.testing.assert_frame_equal(
            fresh,
            golden.frame,
            check_exact=exact,
            check_dtype=True,
            atol=scenario.tolerance,
            rtol=0,
        )
    except AssertionError as exc:  # noqa: BLE001 - re-raise with context
        raise AssertionError(
            f"{scenario.scenario_id}: result frame drifted from "
            f"golden\n{exc}"
        ) from exc


def _assert_structural_golden(scenario: Scenario) -> None:
    """Assert a structural-only golden exists and is well-formed.

    ``nn/`` model-backed detectors are not executed (they need model
    checkpoints). Their golden is a small JSON metadata file; this check
    confirms the file is present and self-consistent.

    Args:
        scenario: The structural-only scenario under test.
    """
    path = golden_path(scenario)
    assert path.exists(), (
        f"{scenario.scenario_id}: structural-only golden missing "
        f"({path}); run scripts/capture_migration_goldens.py"
    )
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta.get("structural_only") is True
    assert meta.get("class_name") == scenario.class_name


def test_scenario_registry_is_non_empty() -> None:
    """Sanity-check that scenario discovery produced a registry."""
    assert len(_SCENARIOS) > 100, (
        "expected 100+ scenarios; operation discovery likely broke"
    )


def test_every_subpackage_is_represented() -> None:
    """Sanity-check that every operation subpackage has scenarios."""
    subpackages = {s.subpackage for s in _SCENARIOS}
    expected = {
        "detect",
        "enhance",
        "refine",
        "correction",
        "grid",
        "post",
        "measure",
        "nn",
        "analysis",
    }
    missing = expected - subpackages
    assert not missing, f"subpackages with no scenarios: {missing}"
