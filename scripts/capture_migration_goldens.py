"""Capture golden outputs for the pydantic-migration equivalence suite.

This script runs every scenario from
:mod:`tests.migration._scenarios` against the matching frozen input and
dumps the result to ``tests/migration/_goldens/``. The goldens become
the immovable bit-exact reference that later proves the pydantic
migration changed no numerical behavior.

It must be run **before** any operation class is migrated -- the goldens
are captured from the current, unmigrated library.

Output artifacts per scenario:

* ``ImageOperation`` results -> ``<scenario_id>.npz`` carrying the
  resulting ``detect_mat``, ``objmask`` and ``objmap`` arrays.
* ``MeasureFeatures`` / ``PostMeasurement`` / analyzer results ->
  ``<scenario_id>.parquet`` carrying the result DataFrame.
* ``nn/`` model-backed detectors -> ``<scenario_id>.meta.json`` carrying
  only shape/dtype metadata (no model download is attempted).

Robustness: every scenario runs inside its own ``try/except``. The
script never aborts on a single failure; it prints a captured / skipped
/ failed summary at the end and exits non-zero only if a scenario
*failed* (skips -- e.g. ``UNCAPTURABLE`` classes -- do not fail the
run).

Usage::

    uv run python scripts/capture_migration_goldens.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the repository root importable so ``tests.migration`` resolves
# whether the script is run from the repo root or elsewhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.migration._inputs import (  # noqa: E402
    capture_frozen_inputs,
    load_frozen_input,
)
from tests.migration._runner import (  # noqa: E402
    GOLDENS_DIR,
    golden_path,
    run_scenario,
)
from tests.migration._scenarios import (  # noqa: E402
    UNCAPTURABLE,
    Scenario,
    build_scenarios,
    discover_operations,
)


def _capture_one(scenario: Scenario) -> tuple[str, str]:
    """Capture the golden for a single scenario.

    Args:
        scenario: The scenario to run and persist.

    Returns:
        A ``(status, detail)`` pair where ``status`` is ``"captured"``,
        ``"skipped"`` or ``"failed"`` and ``detail`` is a human-readable
        note.
    """
    if scenario.structural_only:
        # nn/ detectors: capture metadata only, no model download.
        meta = {
            "scenario_id": scenario.scenario_id,
            "class_name": scenario.class_name,
            "structural_only": True,
            "note": (
                "model-backed detector; goldens record metadata only, "
                "equivalence is structural"
            ),
        }
        path = golden_path(scenario)
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return "captured", "structural-only metadata"

    try:
        result = run_scenario(scenario)
    except Exception as exc:  # noqa: BLE001 - per-scenario isolation
        return "failed", f"{type(exc).__name__}: {exc}"

    try:
        result.save(golden_path(scenario))
    except Exception as exc:  # noqa: BLE001 - per-scenario isolation
        return "failed", f"save error: {type(exc).__name__}: {exc}"

    return "captured", result.summary


def main() -> int:
    """Capture frozen inputs and every scenario golden.

    Returns:
        Process exit code: ``0`` on success, ``1`` if any scenario
        failed.
    """
    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PhenoTypic pydantic-migration golden capture")
    print("=" * 72)

    # 1. Frozen inputs.
    print("\n[1/3] Capturing frozen inputs ...")
    written = capture_frozen_inputs()
    for name, path in written.items():
        print(f"  + {name:24s} -> {path.name}")

    # Sanity-check reconstruction round-trips.
    for name in written:
        load_frozen_input(name)
    print("  frozen inputs reconstruct cleanly")

    # 2. Discovery summary.
    print("\n[2/3] Discovered operations per subpackage:")
    discovered = discover_operations()
    op_total = 0
    for subpkg, classes in discovered.items():
        op_total += len(classes)
        print(f"  {subpkg:12s} {len(classes):3d}")
    print(f"  {'TOTAL ops':12s} {op_total:3d}  (+ 5 analyzers)")

    # 3. Per-scenario capture.
    scenarios = build_scenarios()
    print(f"\n[3/3] Capturing {len(scenarios)} scenario goldens ...")

    captured: list[str] = []
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    # UNCAPTURABLE classes never produce a scenario; record them.
    for class_name, reason in UNCAPTURABLE.items():
        skipped.append((class_name, reason))

    for scenario in scenarios:
        status, detail = _capture_one(scenario)
        if status == "captured":
            captured.append(scenario.scenario_id)
            print(f"  ok    {scenario.scenario_id}  ({detail})")
        elif status == "skipped":
            skipped.append((scenario.scenario_id, detail))
            print(f"  skip  {scenario.scenario_id}  ({detail})")
        else:
            failed.append((scenario.scenario_id, detail))
            print(f"  FAIL  {scenario.scenario_id}")
            print(f"        {detail}")

    # Summary.
    print("\n" + "=" * 72)
    print("CAPTURE SUMMARY")
    print("=" * 72)
    print(f"  captured : {len(captured)}")
    print(f"  skipped  : {len(skipped)}")
    print(f"  failed   : {len(failed)}")
    if skipped:
        print("\n  Skipped:")
        for name, reason in skipped:
            print(f"    - {name}: {reason}")
    if failed:
        print("\n  Failed:")
        for name, reason in failed:
            print(f"    - {name}: {reason}")
    print("=" * 72)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
