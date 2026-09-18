"""Print the Stage-2 legacy relocation as PATHS, before and after.

Evidence, not a verdict: `relocate_legacy_stage2_signal` returns a bool, and a
`True` from a half-migration is indistinguishable from a `True` from a correct
one. This prints every path and its existence on both sides of the call, so the
reader checks the tree rather than the return value.

Scratchpad, deliberately: this imports `phenotypic`, so it must NOT live under
docs/superpowers/logic_validation_scripts/ (that directory's contract is that
nothing in it imports the code under test).

Run:  uv run python <this file>
"""

from __future__ import annotations

import os
import shutil
import tempfile
import types
from pathlib import Path

import numpy as np

from phenotypic._cli._cli_stage2_token import (
    _legacy_stage2_raw_path,
    _legacy_stage2_token_path,
    detector_slot,
    relocate_legacy_stage2_signal,
    stage2_raw_path,
    stage2_result_replayable,
    stage2_token_path,
    write_stage2_raw,
    write_stage2_token,
)
import phenotypic._cli._cli_stage2_token as mod

SLOT = detector_slot(("CompositeDetector", "ops[0]"))
FAILURES: list[str] = []


def _plant_legacy(root: Path, *, value: int, with_token: bool = True) -> None:
    raw = _legacy_stage2_raw_path(root, "ds", "img")
    raw.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw, np.full((4, 4), value, dtype=np.uint16))
    if with_token:
        token = _legacy_stage2_token_path(root, "ds", "img")
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(
            '{"objmap_shape": [4, 4], "detector_duration_seconds": 1.5}',
            encoding="utf-8",
        )


def _show(root: Path, label: str) -> None:
    print(f"  {label}")
    for name, path in (
        ("legacy raw  ", _legacy_stage2_raw_path(root, "ds", "img")),
        ("legacy token", _legacy_stage2_token_path(root, "ds", "img")),
        ("slot   raw  ", stage2_raw_path(root, "ds", "img", SLOT)),
        ("slot   token", stage2_token_path(root, "ds", "img", SLOT)),
    ):
        mark = "EXISTS " if path.exists() else "absent "
        print(f"    {name}  {mark}  {path.relative_to(root)}")
    print(
        "    stage2_result_replayable(slot) ->",
        stage2_result_replayable(root, "ds", "img", SLOT),
    )


def _check(label: str, condition: bool) -> None:
    print(f"    [{'ok ' if condition else 'FAIL'}] {label}")
    if not condition:
        FAILURES.append(label)


def scenario_both_halves(root: Path) -> None:
    print("\n=== 1. Both halves present -> both relocate ===")
    _plant_legacy(root, value=7)
    _show(root, "BEFORE")
    moved = relocate_legacy_stage2_signal(root, "ds", "img", SLOT)
    print(f"  returned: {moved}")
    _show(root, "AFTER")
    _check("raw moved", stage2_raw_path(root, "ds", "img", SLOT).is_file())
    _check("token moved", stage2_token_path(root, "ds", "img", SLOT).is_file())
    _check(
        "legacy raw gone",
        not _legacy_stage2_raw_path(root, "ds", "img").exists(),
    )
    _check(
        "legacy token gone",
        not _legacy_stage2_token_path(root, "ds", "img").exists(),
    )
    _check(
        "replayable after relocation -- no GPU recompute",
        stage2_result_replayable(root, "ds", "img", SLOT),
    )
    _check(
        "raw content preserved (value 7)",
        int(np.load(stage2_raw_path(root, "ds", "img", SLOT)).max()) == 7,
    )


def scenario_order(root: Path) -> None:
    print("\n=== 2. Raw moves BEFORE the token (interrupted relocation) ===")
    _plant_legacy(root, value=7)
    _show(root, "BEFORE")

    attempted: list[str] = []
    real_replace = os.replace

    def _replace(src, dst):
        attempted.append(Path(dst).suffix)
        if Path(dst).suffix == ".json":
            raise OSError("interrupted between the halves")
        real_replace(src, dst)

    mod.os = types.SimpleNamespace(replace=_replace)
    try:
        relocate_legacy_stage2_signal(root, "ds", "img", SLOT)
    except OSError as exc:
        print(f"  raised (expected): {exc}")
    finally:
        mod.os = os

    print(f"  replace() destinations attempted, in order: {attempted}")
    _show(root, "AFTER (interrupted)")
    _check("raw attempted first", attempted == [".npy", ".json"])
    _check(
        "slot raw exists", stage2_raw_path(root, "ds", "img", SLOT).is_file()
    )
    _check(
        "slot token does NOT exist -- the only reachable intermediate state "
        "is 'no token, orphan raw'",
        not stage2_token_path(root, "ds", "img", SLOT).exists(),
    )
    _check(
        "not replayable, so Stage 2 recomputes rather than replaying a "
        "mask that is not there",
        not stage2_result_replayable(root, "ds", "img", SLOT),
    )


def scenario_current_wins(root: Path) -> None:
    print("\n=== 3. A current signal is never overwritten ===")
    write_stage2_raw(
        root, "ds", "img", np.full((4, 4), 3, dtype=np.uint16), SLOT
    )
    write_stage2_token(root, "ds", "img", SLOT, objmap_shape=(4, 4))
    _plant_legacy(root, value=9)
    _show(root, "BEFORE")
    moved = relocate_legacy_stage2_signal(root, "ds", "img", SLOT)
    print(f"  returned: {moved}")
    _show(root, "AFTER")
    _check("refused", moved is False)
    _check(
        "current raw still value 3, not the stale 9",
        int(np.load(stage2_raw_path(root, "ds", "img", SLOT)).max()) == 3,
    )


def scenario_multi_slot_refusal(root: Path) -> None:
    print("\n=== 4. Refuses when the caller declares more than one slot ===")
    _plant_legacy(root, value=7)
    _show(root, "BEFORE")
    moved = relocate_legacy_stage2_signal(
        root, "ds", "img", SLOT, slot_count=2
    )
    print(f"  returned: {moved}")
    _show(root, "AFTER")
    _check("refused", moved is False)
    _check(
        "nothing moved -- a per-image signal cannot be attributed to one of N "
        "detectors, and guessing would hand detector B detector A's mask",
        not stage2_raw_path(root, "ds", "img", SLOT).exists()
        and _legacy_stage2_raw_path(root, "ds", "img").exists(),
    )


def main() -> int:
    print(f"slot = {SLOT}")
    for scenario in (
        scenario_both_halves,
        scenario_order,
        scenario_current_wins,
        scenario_multi_slot_refusal,
    ):
        root = Path(tempfile.mkdtemp(prefix="reloc-"))
        try:
            scenario(root)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
