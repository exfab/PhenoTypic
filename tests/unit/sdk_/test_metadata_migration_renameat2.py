"""The ``renameat2`` probe must never break importing the migration module.

``_metadata_migration`` is imported by ``phenotypic.sdk_``'s lazy re-exports,
and the unit-test conftest walks every package, so a probe that raises at
import time aborts the whole test session -- which is what happened on
Windows, where ``ctypes.CDLL(None)`` raises ``TypeError`` (not ``OSError``).
These tests reproduce that platform behaviour on any OS.
"""

from __future__ import annotations

import ctypes
import sys

import pytest

from phenotypic.sdk_ import _metadata_migration as migration


def _windows_cdll(name, *args, **kwargs):
    """What ``ctypes.CDLL`` does with ``None`` on Windows."""
    if name is None:
        raise TypeError("argument of type 'NoneType' is not iterable")
    raise AssertionError(f"unexpected CDLL({name!r})")


@pytest.mark.parametrize("platform", ["win32", "darwin", "cygwin"])
def test_the_probe_does_not_touch_libc_off_linux(
    monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    monkeypatch.setattr(ctypes, "CDLL", _windows_cdll)

    assert migration._load_renameat2(platform) is None


def test_a_missing_renameat2_fails_closed_at_use_not_at_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(migration, "_RENAMEAT2", None)

    with pytest.raises(RuntimeError, match="no-clobber rename is unsupported"):
        migration._libc_renameat2()


def test_a_libc_without_the_symbol_reads_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An old glibc or musl: the library loads, the symbol is absent."""
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: object())

    assert migration._load_renameat2("linux") is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux libc only")
def test_linux_still_finds_renameat2() -> None:
    """The guard must not switch the no-clobber rename off where it exists."""
    assert migration._load_renameat2() is not None
