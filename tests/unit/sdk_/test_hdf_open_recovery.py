"""Tests for shared HDF writer open-recovery behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _hdf_for_path(path: Path):
    """Construct an HDF instance without opening a real file."""
    from phenotypic.sdk_.hdf_ import HDF

    hdf = object.__new__(HDF)
    hdf.filepath = path
    return hdf


class _FakeH5:
    """Tiny h5py.File stand-in with writable ``swmr_mode``."""

    def __init__(self) -> None:
        self.swmr_mode = False


def test_open_hdf_with_recovery_retries_lock_errors_with_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock errors should clear status flags, back off, then retry."""
    from phenotypic.sdk_ import hdf_ as hdf_mod

    path = tmp_path / "image.h5"
    path.touch()
    calls = 0
    sleep_calls: list[float] = []
    clear_calls: list[list[str]] = []
    handle = _FakeH5()

    def opener() -> _FakeH5:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise OSError("unable to lock file")
        return handle

    monkeypatch.setattr(hdf_mod.time, "sleep", sleep_calls.append)
    monkeypatch.setattr(
        hdf_mod.subprocess,
        "run",
        lambda cmd, **_kwargs: clear_calls.append(cmd)
        or SimpleNamespace(returncode=0, stderr=""),
    )

    opened = hdf_mod._open_hdf_with_recovery(
        path,
        opener,
        context="HDF5 file",
        lock_markers=hdf_mod._SAFE_WRITER_LOCK_MARKERS,
        clear_status=True,
        clear_force=False,
    )

    assert opened is handle
    assert calls == 3
    assert sleep_calls == [0.5, 1.0]
    assert clear_calls == [["h5clear", "-s", str(path)]] * 2


def test_safe_writer_runs_status_clear_between_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``safe_writer`` should run ``h5clear -s`` before retrying."""
    from phenotypic.sdk_ import hdf_ as hdf_mod

    path = tmp_path / "safe.h5"
    path.touch()
    hdf = _hdf_for_path(path)
    calls = 0
    clear_calls: list[list[str]] = []
    handle = _FakeH5()

    def fake_file(filepath, mode, libver):  # noqa: ANN001
        nonlocal calls
        assert Path(filepath) == path
        assert mode == "a"
        assert libver == "latest"
        calls += 1
        if calls == 1:
            raise OSError("file is already open")
        return handle

    monkeypatch.setattr(hdf_mod.h5py, "File", fake_file)
    monkeypatch.setattr(hdf_mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        hdf_mod.subprocess,
        "run",
        lambda cmd, **_kwargs: clear_calls.append(cmd)
        or SimpleNamespace(returncode=0, stderr=""),
    )

    assert hdf.safe_writer() is handle
    assert clear_calls == [["h5clear", "-s", str(path)]]


def test_swmr_writer_runs_status_and_force_clear_between_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``swmr_writer`` should run ``h5clear -s`` and ``h5clear -f``."""
    from phenotypic.sdk_ import hdf_ as hdf_mod

    path = tmp_path / "swmr.h5"
    path.touch()
    hdf = _hdf_for_path(path)
    calls = 0
    clear_calls: list[list[str]] = []
    handle = _FakeH5()

    def fake_file(filepath, mode, libver):  # noqa: ANN001
        nonlocal calls
        assert Path(filepath) == path
        assert mode == "a"
        assert libver == "latest"
        calls += 1
        if calls == 1:
            raise OSError("ring type mismatch")
        return handle

    monkeypatch.setattr(hdf_mod.h5py, "File", fake_file)
    monkeypatch.setattr(hdf_mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        hdf_mod.subprocess,
        "run",
        lambda cmd, **_kwargs: clear_calls.append(cmd)
        or SimpleNamespace(returncode=0, stderr=""),
    )

    opened = hdf.swmr_writer()

    assert opened is handle
    assert opened.swmr_mode is True
    assert clear_calls == [
        ["h5clear", "-s", str(path)],
        ["h5clear", "-f", str(path)],
    ]


def test_non_lock_oserror_raises_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unrelated OSErrors should not retry or run h5clear."""
    from phenotypic.sdk_ import hdf_ as hdf_mod

    path = tmp_path / "bad.h5"
    path.touch()
    clear_calls: list[list[str]] = []

    monkeypatch.setattr(hdf_mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        hdf_mod.subprocess,
        "run",
        lambda cmd, **_kwargs: clear_calls.append(cmd),
    )

    with pytest.raises(OSError, match="bad address"):
        hdf_mod._open_hdf_with_recovery(
            path,
            lambda: (_ for _ in ()).throw(OSError("bad address")),
            context="HDF5 file",
            lock_markers=hdf_mod._SAFE_WRITER_LOCK_MARKERS,
            clear_status=True,
            clear_force=False,
        )

    assert clear_calls == []


def test_final_lock_failure_raises_helpful_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final lock failure should keep the existing manual h5clear guidance."""
    from phenotypic.sdk_ import hdf_ as hdf_mod

    path = tmp_path / "locked.h5"
    path.touch()

    monkeypatch.setattr(hdf_mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        hdf_mod.subprocess,
        "run",
        lambda _cmd, **_kwargs: SimpleNamespace(returncode=0, stderr=""),
    )

    with pytest.raises(RuntimeError, match=r"h5clear -s .* && h5clear -f"):
        hdf_mod._open_hdf_with_recovery(
            path,
            lambda: (_ for _ in ()).throw(OSError("unable to lock file")),
            context="HDF5 file",
            lock_markers=hdf_mod._SAFE_WRITER_LOCK_MARKERS,
            clear_status=True,
            clear_force=False,
            max_retries=2,
            retry_delay=0.01,
        )
