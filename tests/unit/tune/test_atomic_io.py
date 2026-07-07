"""Atomic file writes — crash-safety for the tune output writers (B3).

``atomic_write_text`` / ``atomic_write_bytes`` (in ``phenotypic.sdk_``) and
``JournalStudyStore.to_parquet`` must each be all-or-nothing: a normal write
succeeds, and an exception raised mid-serialize leaves any pre-existing file
untouched and leaves no ``.tmp`` debris in the directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.sdk_ import atomic_write_bytes, atomic_write_text
from phenotypic.tune._study_store import JournalStudyStore, Trial


def _no_tmp_debris(directory: Path) -> bool:
    return not any(p.name.endswith(".tmp") for p in directory.iterdir())


# --- the tools_ helpers -------------------------------------------------------


def test_atomic_write_text_normal_write_succeeds(tmp_path):
    target = tmp_path / "marker.json"
    atomic_write_text(target, '{"k": 1}')
    assert target.read_text() == '{"k": 1}'
    assert _no_tmp_debris(tmp_path)


def test_atomic_write_text_creates_parent_dirs(tmp_path):
    target = tmp_path / "nested" / "deep" / "out.json"
    atomic_write_text(target, "hello")
    assert target.read_text() == "hello"


def test_atomic_write_bytes_normal_write_succeeds(tmp_path):
    target = tmp_path / "blob.bin"
    atomic_write_bytes(target, b"\x00\x01\x02")
    assert target.read_bytes() == b"\x00\x01\x02"
    assert _no_tmp_debris(tmp_path)


def test_atomic_write_text_failure_leaves_existing_file_intact(
    tmp_path, monkeypatch
):
    target = tmp_path / "marker.json"
    target.write_text("ORIGINAL")

    # Force the rename step to blow up *after* the temp file is written, so we
    # exercise the cleanup path with a pre-existing target in place.
    import phenotypic.sdk_._atomic_io as atomic_io

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(atomic_io.os, "replace", _boom)

    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(target, "NEW CONTENT")

    # The pre-existing file is untouched and no .tmp debris is left behind.
    assert target.read_text() == "ORIGINAL"
    assert _no_tmp_debris(tmp_path)


def test_atomic_write_text_failure_no_partial_file_when_target_absent(
    tmp_path, monkeypatch
):
    target = tmp_path / "marker.json"
    import phenotypic.sdk_._atomic_io as atomic_io

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(atomic_io.os, "replace", _boom)

    with pytest.raises(OSError):
        atomic_write_text(target, "NEW CONTENT")

    # No target was created (the replace never happened) and no .tmp lingers.
    assert not target.exists()
    assert _no_tmp_debris(tmp_path)


# --- JournalStudyStore.to_parquet atomicity -----------------------------------


def _journal_with_one_trial() -> JournalStudyStore:
    store = JournalStudyStore()
    store.append(
        Trial(
            number=0, params={"a": 1}, score=0.5, terms={"t": 0.5}, n_images=1
        )
    )
    return store


def test_to_parquet_normal_write_succeeds(tmp_path):
    path = tmp_path / "trials.parquet"
    _journal_with_one_trial().to_parquet(path)
    reloaded = JournalStudyStore.from_parquet(path)
    assert len(reloaded) == 1
    assert _no_tmp_debris(tmp_path)


def test_to_parquet_failure_leaves_existing_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "trials.parquet"
    # A valid pre-existing parquet (one trial).
    _journal_with_one_trial().to_parquet(path)
    original_bytes = path.read_bytes()

    # A *new* journal with two trials; force the atomic replace to fail so the
    # pre-existing single-trial parquet must survive untouched.
    bigger = _journal_with_one_trial()
    bigger.append(
        Trial(
            number=1, params={"a": 2}, score=0.9, terms={"t": 0.9}, n_images=1
        )
    )

    import phenotypic.sdk_._atomic_io as atomic_io

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(atomic_io.os, "replace", _boom)

    with pytest.raises(OSError, match="disk full"):
        bigger.to_parquet(path)

    # The original file is byte-for-byte intact and no .tmp debris remains.
    assert path.read_bytes() == original_bytes
    assert JournalStudyStore.from_parquet(path).__len__() == 1
    assert _no_tmp_debris(tmp_path)
