"""The Stage-2 token replaces the .npy sidecar. It must be consumable.

Every helper here is keyed by **detector slot** as well as by image. These
tests use one fixed slot throughout: what they pin is the token/raw lifecycle,
not the keying. The keying itself -- two slots not colliding, the legacy
relocation -- is pinned in ``test_stage2_slot_keying.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phenotypic._cli._cli_stage2_token import (
    delete_stage2_raw,
    delete_stage2_token,
    detector_slot,
    load_stage2_raw,
    read_stage2_token,
    stage2_raw_path,
    stage2_result_replayable,
    stage2_token_exists,
    stage2_token_path,
    write_stage2_raw,
    write_stage2_token,
)
from phenotypic.sdk_ import progress_dir

_SLOT = detector_slot(("FakeGpuDetector",))


def test_token_lives_under_progress_not_in_the_store(tmp_path: Path) -> None:
    """Resume state lives where the rest of it already lives."""
    path = stage2_token_path(tmp_path, "ds", "img", _SLOT)
    assert path == (
        progress_dir(tmp_path) / "stage2_done" / "ds" / _SLOT / "img.json"
    )
    assert ".ome.zarr" not in str(path)


def test_token_path_separates_dataset_from_stem(tmp_path: Path) -> None:
    """Two images that differ only by which argument holds the name must differ."""
    assert stage2_token_path(tmp_path, "a", "b", _SLOT) != stage2_token_path(
        tmp_path, "b", "a", _SLOT
    )


def test_write_then_exists_then_delete(tmp_path: Path) -> None:
    assert stage2_token_exists(tmp_path, "ds", "img", _SLOT) is False
    written = write_stage2_token(
        tmp_path, "ds", "img", _SLOT, objmap_shape=(64, 48)
    )
    assert written == stage2_token_path(tmp_path, "ds", "img", _SLOT)
    assert stage2_token_exists(tmp_path, "ds", "img", _SLOT) is True
    delete_stage2_token(tmp_path, "ds", "img", _SLOT)
    assert stage2_token_exists(tmp_path, "ds", "img", _SLOT) is False


def test_deleting_one_images_token_leaves_another_images_alone(tmp_path: Path) -> None:
    write_stage2_token(tmp_path, "ds", "keep", _SLOT, objmap_shape=(4, 4))
    write_stage2_token(tmp_path, "ds", "drop", _SLOT, objmap_shape=(4, 4))
    delete_stage2_token(tmp_path, "ds", "drop", _SLOT)
    assert stage2_token_exists(tmp_path, "ds", "keep", _SLOT) is True


def test_delete_is_idempotent(tmp_path: Path) -> None:
    delete_stage2_token(tmp_path, "ds", "img", _SLOT)
    delete_stage2_token(tmp_path, "ds", "img", _SLOT)


def test_token_carries_the_objmap_shape_and_no_work_id(tmp_path: Path) -> None:
    """No `work_id` field (ledger FLOW-20).

    `stage2_detect_core` has no work_id parameter, so the field could only ever
    be None -- and a field that can only hold None gets misread as meaningful.
    The work-id check that matters reads `attributes.phenotypic.work_id` off the
    STORE (`staged_store_matches_work_id`), not the token.
    """
    write_stage2_token(
        tmp_path,
        "ds",
        "img",
        _SLOT,
        objmap_shape=(64, 48),
        detector_duration_seconds=1.25,
    )
    payload = read_stage2_token(tmp_path, "ds", "img", _SLOT)
    assert tuple(payload["objmap_shape"]) == (64, 48)
    assert payload["detector_duration_seconds"] == 1.25
    assert "work_id" not in payload
    # By exact key set: a field that can only ever be None must not creep back
    # under a different name either. The slot is a path component, not a
    # payload field, so it must not appear here either.
    assert set(payload) == {"detector_duration_seconds", "objmap_shape"}


def test_token_shape_is_the_shape_it_was_given_not_a_placeholder(
    tmp_path: Path,
) -> None:
    """(y, x) order is load-bearing; a non-square case is what proves it."""
    write_stage2_token(tmp_path, "ds", "tall", _SLOT, objmap_shape=(600, 800))
    assert read_stage2_token(tmp_path, "ds", "tall", _SLOT)["objmap_shape"] == [
        600,
        800,
    ]
    write_stage2_token(tmp_path, "ds", "wide", _SLOT, objmap_shape=(800, 600))
    assert read_stage2_token(tmp_path, "ds", "wide", _SLOT)["objmap_shape"] == [
        800,
        600,
    ]


def test_token_shape_survives_numpy_integers(tmp_path: Path) -> None:
    """The caller hands over `objmap.shape`, whose members are plain ints, but a
    derived shape can be numpy scalars -- which json.dumps refuses."""
    shape = tuple(np.array([12, 34], dtype=np.int64))
    write_stage2_token(tmp_path, "ds", "img", _SLOT, objmap_shape=shape)  # type: ignore[arg-type]
    assert read_stage2_token(tmp_path, "ds", "img", _SLOT)["objmap_shape"] == [
        12,
        34,
    ]


def test_token_is_written_atomically(tmp_path: Path, monkeypatch) -> None:
    seen: list[str] = []
    import phenotypic._cli._cli_stage2_token as module

    real = module.atomic_write_with_writer
    monkeypatch.setattr(
        module,
        "atomic_write_with_writer",
        lambda final, writer, *, commit_guard=None: (
            seen.append(str(final)),
            real(final, writer, commit_guard=commit_guard),
        )[1],
    )
    write_stage2_token(tmp_path, "ds", "img", _SLOT, objmap_shape=(2, 2))
    assert seen == [str(stage2_token_path(tmp_path, "ds", "img", _SLOT))]


def test_a_failed_token_write_leaves_no_token_and_no_debris(
    tmp_path: Path, monkeypatch
) -> None:
    """The property atomicity buys: never a half-written 'Stage 2 is done'.

    Asserting only that the helper was *called* would still pass if the payload
    were dumped straight onto the final path first.
    """
    import phenotypic._cli._cli_stage2_token as module

    monkeypatch.setattr(
        module, "json", type("_Boom", (), {"dumps": staticmethod(_raise_boom)})
    )
    with pytest.raises(OSError):
        write_stage2_token(tmp_path, "ds", "img", _SLOT, objmap_shape=(2, 2))
    assert stage2_token_exists(tmp_path, "ds", "img", _SLOT) is False
    assert (
        list(stage2_token_path(tmp_path, "ds", "img", _SLOT).parent.iterdir())
        == []
    )


def _raise_boom(*args, **kwargs):
    raise OSError("boom")


def test_token_is_valid_json(tmp_path: Path) -> None:
    write_stage2_token(tmp_path, "ds", "img", _SLOT, objmap_shape=(2, 2))
    json.loads(
        stage2_token_path(tmp_path, "ds", "img", _SLOT).read_text(
            encoding="utf-8"
        )
    )


def test_read_missing_token_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_stage2_token(tmp_path, "ds", "img", _SLOT)


# --- the retained raw array (D1) -------------------------------------------


def test_raw_array_lives_beside_the_token(tmp_path: Path) -> None:
    assert stage2_raw_path(tmp_path, "ds", "img", _SLOT) == (
        progress_dir(tmp_path) / "stage2_raw" / "ds" / _SLOT / "img.npy"
    )


def test_raw_array_round_trips_exactly(tmp_path: Path) -> None:
    """Stage 3 replays from this, so it must be bit-exact."""
    array = np.arange(64, dtype=np.uint16).reshape(8, 8)
    written = write_stage2_raw(tmp_path, "ds", "img", array, _SLOT)
    assert written == stage2_raw_path(tmp_path, "ds", "img", _SLOT)
    np.testing.assert_array_equal(
        load_stage2_raw(tmp_path, "ds", "img", _SLOT), array
    )
    assert load_stage2_raw(tmp_path, "ds", "img", _SLOT).dtype == array.dtype


def test_raw_array_is_stored_compressed(tmp_path: Path) -> None:
    """Every image's raw output is live at once across the Stage-2 barrier.

    Uncompressed, a 3140x5094 uint16 objmap is ~32 MB, so a 33,923-image run
    keeps ~1 TiB on shared storage until Stage 3 consumes it. A label image of
    flat regions compresses by orders of magnitude, so the file must not be a
    bare array: 4 MB of one label here is the compressible extreme, and a
    plain ``np.save`` would write all of it.
    """
    array = np.full((1024, 2048), 7, dtype=np.uint16)
    written = write_stage2_raw(tmp_path, "ds", "img", array, _SLOT)

    assert written.stat().st_size < array.nbytes // 100
    np.testing.assert_array_equal(
        load_stage2_raw(tmp_path, "ds", "img", _SLOT), array
    )


def test_a_precompression_raw_array_still_replays(tmp_path: Path) -> None:
    """A run interrupted across the upgrade must not re-infer on a GPU.

    Older builds wrote a bare ``.npy`` at this same path. Writing one by hand
    is the only way to prove the reader still accepts it; a file written by
    ``write_stage2_raw`` cannot fail this test whatever the reader does.
    """
    array = np.arange(12, dtype=np.uint16).reshape(3, 4)
    legacy = stage2_raw_path(tmp_path, "ds", "img", _SLOT)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    with open(legacy, "wb") as handle:
        np.save(handle, array)

    loaded = load_stage2_raw(tmp_path, "ds", "img", _SLOT)

    np.testing.assert_array_equal(loaded, array)
    assert loaded.dtype == array.dtype
    assert stage2_result_replayable(tmp_path, "ds", "img", _SLOT) is False, (
        "no token was written, so the pair is not replayable yet"
    )


def test_raw_array_keeps_a_wide_label_range(tmp_path: Path) -> None:
    """A silent uint8/int16 narrowing would renumber colonies, not fail."""
    array = np.array([[0, 1], [300, 70000]], dtype=np.uint32)
    write_stage2_raw(tmp_path, "ds", "img", array, _SLOT)
    loaded = load_stage2_raw(tmp_path, "ds", "img", _SLOT)
    np.testing.assert_array_equal(loaded, array)
    assert loaded.dtype == np.uint32
    assert int(loaded.max()) == 70000


def test_raw_array_for_one_image_does_not_overwrite_another(tmp_path: Path) -> None:
    write_stage2_raw(
        tmp_path, "ds", "a", np.full((2, 2), 1, dtype=np.uint16), _SLOT
    )
    write_stage2_raw(
        tmp_path, "ds", "b", np.full((2, 2), 2, dtype=np.uint16), _SLOT
    )
    assert int(load_stage2_raw(tmp_path, "ds", "a", _SLOT).max()) == 1
    assert int(load_stage2_raw(tmp_path, "ds", "b", _SLOT).max()) == 2


def test_raw_array_is_written_atomically(tmp_path: Path, monkeypatch) -> None:
    import phenotypic._cli._cli_stage2_token as module

    seen: list[str] = []
    real = module.atomic_write_with_writer
    monkeypatch.setattr(
        module,
        "atomic_write_with_writer",
        lambda final, writer, *, commit_guard=None: (
            seen.append(str(final)),
            real(final, writer, commit_guard=commit_guard),
        )[1],
    )
    module.write_stage2_raw(
        tmp_path, "ds", "img", np.zeros((2, 2), dtype=np.uint16), _SLOT
    )
    assert seen == [str(module.stage2_raw_path(tmp_path, "ds", "img", _SLOT))]


def test_a_failed_raw_write_never_replaces_a_good_one(
    tmp_path: Path, monkeypatch
) -> None:
    """A partial .npy that Stage 3 then replays is the failure D1 exists to stop."""
    import phenotypic._cli._cli_stage2_token as module

    good = np.arange(4, dtype=np.uint16).reshape(2, 2)
    write_stage2_raw(tmp_path, "ds", "img", good, _SLOT)

    # The writer compresses, so this patches savez_compressed, not save. A
    # patch on the wrong function makes the write SUCCEED and the assertion
    # below then compares the good array against itself.
    monkeypatch.setattr(module.np, "savez_compressed", _raise_boom)
    with pytest.raises(OSError):
        write_stage2_raw(
            tmp_path, "ds", "img", np.full((2, 2), 9, dtype=np.uint16), _SLOT
        )

    np.testing.assert_array_equal(
        load_stage2_raw(tmp_path, "ds", "img", _SLOT), good
    )


def test_load_missing_raw_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_stage2_raw(tmp_path, "ds", "img", _SLOT)


def test_raw_delete_is_idempotent(tmp_path: Path) -> None:
    delete_stage2_raw(tmp_path, "ds", "img", _SLOT)
    delete_stage2_raw(tmp_path, "ds", "img", _SLOT)


def test_consuming_the_token_does_not_consume_the_raw_array(tmp_path: Path) -> None:
    """They are deleted together by Stage 3, but only after Stage 3 succeeds.

    If deleting the token also dropped the raw array, the retry window the token
    exists to cover would have nothing left to replay from (OPEN-QUESTIONS D1).
    """
    array = np.arange(4, dtype=np.uint16).reshape(2, 2)
    write_stage2_raw(tmp_path, "ds", "img", array, _SLOT)
    write_stage2_token(tmp_path, "ds", "img", _SLOT, objmap_shape=(2, 2))

    delete_stage2_token(tmp_path, "ds", "img", _SLOT)

    np.testing.assert_array_equal(
        load_stage2_raw(tmp_path, "ds", "img", _SLOT), array
    )


def test_consuming_the_raw_array_does_not_consume_the_token(tmp_path: Path) -> None:
    write_stage2_raw(
        tmp_path, "ds", "img", np.zeros((2, 2), dtype=np.uint16), _SLOT
    )
    write_stage2_token(tmp_path, "ds", "img", _SLOT, objmap_shape=(2, 2))

    delete_stage2_raw(tmp_path, "ds", "img", _SLOT)

    assert stage2_token_exists(tmp_path, "ds", "img", _SLOT) is True
