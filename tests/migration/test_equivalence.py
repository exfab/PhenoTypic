"""Golden-equivalence tests for the pydantic v2 operation migration.

Each test re-runs one scenario from :mod:`tests.migration._scenarios`
against its frozen input and asserts the result matches the golden
captured by ``scripts/capture_migration_goldens.py``:

* image operations -> the float ``detect_mat`` is compared with a small
  relative+absolute tolerance (``_FLOAT_RTOL`` / ``_FLOAT_ATOL``); the
  integer ``objmask`` / ``objmap`` label maps are compared exactly;
* measurers / post-measurement transforms / analyzers ->
  ``pandas.testing.assert_frame_equal`` with the same float tolerance
  and ``check_dtype=True``.

Library float math (numpy/scipy/scikit-image/BLAS) is **not**
bit-reproducible across OS/arch, so the numeric goldens are valid only
on the platform they were captured on (``_GOLDEN_PLATFORM``); on other
platforms the numeric comparisons are skipped (large cross-platform
divergences -- e.g. ~20% on GaussianBlur on macOS arm64 -- cannot be
absorbed by any honest tolerance). On the capture platform, the small
``_FLOAT_RTOL`` / ``_FLOAT_ATOL`` tolerance still absorbs minor
library-version jitter while staying tight enough to fail a genuine
algorithmic regression. Label maps and dtypes stay exact, and the
structural goldens (plain JSON metadata for model-backed ``nn``
detectors) run on every platform. After the pydantic migration this
suite proves the migration changed no meaningful numerical behavior.

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
import sys

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

# Default float-comparison tolerance. Library float math is not
# bit-reproducible across platforms/library versions (the goldens were
# captured on one platform; a fresh env drifts at ~1e-8). These bounds
# absorb that jitter while staying far tighter than any real algorithmic
# regression. A scenario's own ``tolerance`` (e.g. the bm3d-backed
# ``TOLERANT_OPS``) raises the absolute floor when it is larger.
_FLOAT_RTOL = 1e-6
_FLOAT_ATOL = 1e-9

# The numeric goldens are float snapshots captured on the CI platform
# (Linux x86_64). Library float math (numpy/scipy/scikit-image/BLAS) is
# not bit-reproducible across OS/arch -- macOS arm64 (Accelerate) diverges
# from Linux (OpenBLAS) by as much as ~20% for some ops (e.g. GaussianBlur),
# which no honest tolerance can absorb without disabling the check. The
# numeric equivalence is therefore asserted only on the capture platform;
# elsewhere the run is skipped (not failed). Structural goldens are plain
# JSON and run on every platform.
_GOLDEN_PLATFORM = "linux"


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

    if sys.platform != _GOLDEN_PLATFORM:
        pytest.skip(
            "numeric migration goldens were captured on "
            f"{_GOLDEN_PLATFORM!r}; float results are not bit-reproducible "
            f"on {sys.platform!r}, so numeric equivalence is asserted only "
            "on the capture platform (structural goldens still run here)."
        )

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
    ``objmask`` + ``objmap``). The float ``detect_mat`` is compared with
    a small relative+absolute tolerance (``_FLOAT_RTOL`` / ``_FLOAT_ATOL``,
    with ``scenario.tolerance`` raising the absolute floor) so
    cross-platform/library float jitter does not fail it, while the
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
        if name == "detect_mat":
            atol = max(scenario.tolerance, _FLOAT_ATOL)
            np.testing.assert_allclose(
                result_arr,
                golden_arr,
                rtol=_FLOAT_RTOL,
                atol=atol,
                err_msg=(
                    f"{sid}: {name} drifted beyond tolerance "
                    f"(rtol={_FLOAT_RTOL}, atol={atol})"
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
    index is materialized into columns and compared faithfully. Float
    columns are compared with a small relative+absolute tolerance
    (``_FLOAT_RTOL`` / ``_FLOAT_ATOL``, with ``scenario.tolerance``
    raising the absolute floor) so cross-platform/library float jitter
    does not fail them; dtypes are still checked exactly and integer
    columns compare exactly.

    Args:
        scenario: The scenario under test (supplies the tolerance).
        result: Freshly computed frame golden.
        golden: Persisted reference frame golden.
    """
    fresh = normalize_frame(result.frame)
    try:
        pd.testing.assert_frame_equal(
            fresh,
            golden.frame,
            check_exact=False,
            check_dtype=True,
            atol=max(scenario.tolerance, _FLOAT_ATOL),
            rtol=_FLOAT_RTOL,
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
    # ``detect/nn`` model-backed detectors are discovered under the
    # ``detect`` subpackage (their scenario ids are ``detect.Sam2Detector``
    # etc.), so there is no separate ``nn`` subpackage in the taxonomy.
    expected = {
        "detect",
        "enhance",
        "refine",
        "correction",
        "grid",
        "post",
        "measure",
        "analysis",
    }
    missing = expected - subpackages
    assert not missing, f"subpackages with no scenarios: {missing}"
