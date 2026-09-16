"""The Stage-2 signal is keyed by detector slot (nested-staging spec §4.4).

At ``N == 1`` slot keying is one extra directory level and no behavioural
difference on disk. It lands now so that supporting more than one
``GpuDetector`` is additive rather than a migration of the layout a
33,923-image run depends on.

The relocation tests are the ones that matter operationally: a run interrupted
before this change and resumed after it must **move** its signals rather than
recompute a full-dataset GPU sweep, and it must move **both halves** --
``stage2_result_replayable`` requires the token *and* the raw, so a
raw-only relocation recomputes anyway and leaves the tree half-migrated.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from phenotypic._cli import _cli_stage2_token as mod
from phenotypic._cli._cli_stage2_token import (
    detector_slot,
    find_stage2_token,
    load_stage2_raw,
    relocate_legacy_stage2_signal,
    stage2_raw_path,
    stage2_result_replayable,
    stage2_token_path,
    write_stage2_raw,
    write_stage2_token,
)
from phenotypic.sdk_ import DIR_STAGE2_DONE, progress_dir

_LEFT = ("CompositeDetector", "ops[0]")
_RIGHT = ("CompositeDetector", "ops[1]")


def _write_legacy_signal(
    root: Path, dataset: str, stem: str, *, value: int, token: bool = True
) -> tuple[Path, Path]:
    """Write a pre-slot-keying signal: the flat raw, and optionally its token."""
    raw_dir = progress_dir(root) / "stage2_raw" / dataset
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{stem}.npy"
    np.save(raw_path, np.full((4, 4), value, dtype=np.uint16))

    token_dir = progress_dir(root) / DIR_STAGE2_DONE / dataset
    token_path = token_dir / f"{stem}.json"
    if token:
        token_dir.mkdir(parents=True, exist_ok=True)
        token_path.write_text(
            '{"objmap_shape": [4, 4], "detector_duration_seconds": 1.5}',
            encoding="utf-8",
        )
    return raw_path, token_path


# ---------------------------------------------------------------------------
# The slot id
# ---------------------------------------------------------------------------


def test_slot_is_readable_and_collision_proof():
    a = detector_slot(_LEFT)
    b = detector_slot(_RIGHT)

    assert a != b
    assert "CompositeDetector" in a and "ops-0" in a  # debuggable by eye
    assert a.replace("-", "").replace("_", "").isalnum()  # filesystem-safe


def test_paths_that_sanitise_alike_still_differ():
    """``ops[0]`` and ``ops-0`` both sanitise to ``ops-0``; the hash separates them."""
    assert detector_slot(("A", "ops[0]")) != detector_slot(("A", "ops-0"))


def test_the_slot_is_deterministic_across_calls():
    assert detector_slot(_LEFT) == detector_slot(_LEFT)


def test_a_top_level_detector_has_a_one_segment_slot():
    slot = detector_slot(("FakeGpuDetector",))
    assert slot.startswith("FakeGpuDetector__")
    assert slot.count("__") == 1


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------


def test_the_slot_is_a_directory_level_between_dataset_and_stem(tmp_path):
    slot = detector_slot(_LEFT)

    assert stage2_raw_path(tmp_path, "ds", "img", slot) == (
        progress_dir(tmp_path) / "stage2_raw" / "ds" / slot / "img.npy"
    )
    assert stage2_token_path(tmp_path, "ds", "img", slot) == (
        progress_dir(tmp_path) / DIR_STAGE2_DONE / "ds" / slot / "img.json"
    )


def test_two_slots_round_trip_independently(tmp_path):
    left = detector_slot(_LEFT)
    right = detector_slot(_RIGHT)
    a = np.full((4, 4), 1, dtype=np.uint16)
    b = np.full((4, 4), 2, dtype=np.uint16)

    write_stage2_raw(tmp_path, "ds", "img", a, slot=left)
    write_stage2_raw(tmp_path, "ds", "img", b, slot=right)

    assert np.array_equal(load_stage2_raw(tmp_path, "ds", "img", slot=left), a)
    assert np.array_equal(load_stage2_raw(tmp_path, "ds", "img", slot=right), b)


def test_the_replayable_predicate_is_scoped_to_one_slot(tmp_path):
    """A token in one slot never satisfies another slot's replay gate.

    This is the whole point of the keying: at ``N > 1`` an unscoped predicate
    would let detector B's Stage 3 replay detector A's mask.
    """
    left = detector_slot(_LEFT)
    right = detector_slot(_RIGHT)

    write_stage2_raw(
        tmp_path, "ds", "img", np.zeros((4, 4), np.uint16), slot=left
    )
    write_stage2_token(tmp_path, "ds", "img", left, objmap_shape=(4, 4))

    assert stage2_result_replayable(tmp_path, "ds", "img", left) is True
    assert stage2_result_replayable(tmp_path, "ds", "img", right) is False


# ---------------------------------------------------------------------------
# Relocating a pre-slot-keying signal
# ---------------------------------------------------------------------------


def test_a_legacy_signal_is_relocated_rather_than_recomputed(tmp_path):
    """Avoids re-running a GPU sweep for images an interrupted run finished."""
    slot = detector_slot(_LEFT)
    legacy_raw, legacy_token = _write_legacy_signal(
        tmp_path, "ds", "img", value=7
    )

    assert relocate_legacy_stage2_signal(tmp_path, "ds", "img", slot) is True

    # BOTH halves moved -- the raw alone leaves the predicate False and the
    # GPU sweep happens anyway.
    assert stage2_raw_path(tmp_path, "ds", "img", slot).is_file()
    assert stage2_token_path(tmp_path, "ds", "img", slot).is_file()
    assert not legacy_raw.exists()
    assert not legacy_token.exists()

    assert stage2_result_replayable(tmp_path, "ds", "img", slot) is True
    assert load_stage2_raw(tmp_path, "ds", "img", slot=slot).max() == 7


def test_relocation_moves_the_raw_before_the_token(tmp_path, monkeypatch):
    """Interrupted mid-relocation must never leave a token with no raw.

    Mirrors the write order. A slot-keyed token with no slot-keyed raw is the
    one state ``stage2_result_replayable`` cannot express as "not done": Stage
    3 would replay into an uncaught ``FileNotFoundError`` and be recorded as a
    terminal scientific failure.

    ``os`` is replaced on the module only -- the real ``os`` module is never
    touched, so nothing else running in this process is affected.
    """
    slot = detector_slot(_LEFT)
    _write_legacy_signal(tmp_path, "ds", "img", value=7)

    import os as real_os

    attempted: list[str] = []

    def _replace(src, dst):
        attempted.append(Path(dst).suffix)
        if Path(dst).suffix == ".json":
            raise OSError("interrupted between the halves")
        real_os.replace(src, dst)

    monkeypatch.setattr(mod, "os", types.SimpleNamespace(replace=_replace))

    with pytest.raises(OSError):
        relocate_legacy_stage2_signal(tmp_path, "ds", "img", slot)

    assert attempted == [".npy", ".json"], "the raw must be attempted first"
    assert stage2_raw_path(tmp_path, "ds", "img", slot).is_file()
    assert not stage2_token_path(tmp_path, "ds", "img", slot).exists()


def test_a_raw_only_legacy_signal_still_relocates(tmp_path):
    """A run that died between the two writes: an orphan raw, no token."""
    slot = detector_slot(_LEFT)
    legacy_raw, legacy_token = _write_legacy_signal(
        tmp_path, "ds", "img", value=3, token=False
    )
    assert not legacy_token.exists()

    assert relocate_legacy_stage2_signal(tmp_path, "ds", "img", slot) is True
    assert stage2_raw_path(tmp_path, "ds", "img", slot).is_file()
    assert not legacy_raw.exists()
    # Still not replayable -- there was no token to move. Stage 2 re-runs,
    # which is correct; relocation does not invent a completion signal.
    assert stage2_result_replayable(tmp_path, "ds", "img", slot) is False


def test_relocation_never_overwrites_a_current_signal(tmp_path):
    slot = detector_slot(_LEFT)
    write_stage2_raw(
        tmp_path, "ds", "img", np.full((4, 4), 3, np.uint16), slot=slot
    )
    write_stage2_token(tmp_path, "ds", "img", slot, objmap_shape=(4, 4))
    legacy_raw, legacy_token = _write_legacy_signal(
        tmp_path, "ds", "img", value=9
    )

    assert relocate_legacy_stage2_signal(tmp_path, "ds", "img", slot) is False
    assert load_stage2_raw(tmp_path, "ds", "img", slot=slot).max() == 3
    # The stale legacy files are left alone rather than deleted: this helper
    # relocates, it does not garbage-collect.
    assert legacy_raw.exists() and legacy_token.exists()


def test_relocation_is_a_no_op_when_there_is_nothing_legacy(tmp_path):
    slot = detector_slot(_LEFT)
    assert relocate_legacy_stage2_signal(tmp_path, "ds", "img", slot) is False


def test_relocation_refuses_when_the_caller_declares_more_than_one_slot(tmp_path):
    """A per-image legacy signal cannot be attributed to one of N detectors.

    Guessing would hand detector B the mask detector A produced. Refusing
    costs a recompute.
    """
    slot = detector_slot(_LEFT)
    legacy_raw, legacy_token = _write_legacy_signal(
        tmp_path, "ds", "img", value=7
    )

    assert (
        relocate_legacy_stage2_signal(
            tmp_path, "ds", "img", slot, slot_count=2
        )
        is False
    )
    assert legacy_raw.exists() and legacy_token.exists()
    assert not stage2_raw_path(tmp_path, "ds", "img", slot).exists()


def test_relocation_does_not_reach_across_datasets(tmp_path):
    slot = detector_slot(_LEFT)
    _write_legacy_signal(tmp_path, "other", "img", value=7)

    assert relocate_legacy_stage2_signal(tmp_path, "ds", "img", slot) is False


# ---------------------------------------------------------------------------
# Slot-free discovery, for --mode migrate
# ---------------------------------------------------------------------------


def test_find_stage2_token_discovers_a_slot_keyed_token(tmp_path):
    """``--mode migrate`` has no pipeline and therefore no slot."""
    slot = detector_slot(_LEFT)
    written = write_stage2_token(tmp_path, "ds", "img", slot, objmap_shape=(4, 4))

    assert find_stage2_token(tmp_path, "ds", "img") == written


def test_find_stage2_token_still_finds_a_legacy_token(tmp_path):
    _, legacy_token = _write_legacy_signal(tmp_path, "ds", "img", value=7)

    assert find_stage2_token(tmp_path, "ds", "img") == legacy_token


def test_find_stage2_token_returns_none_for_an_image_with_no_token(tmp_path):
    write_stage2_token(
        tmp_path, "ds", "other", detector_slot(_LEFT), objmap_shape=(4, 4)
    )

    assert find_stage2_token(tmp_path, "ds", "img") is None
    assert find_stage2_token(tmp_path, "missing-ds", "img") is None
