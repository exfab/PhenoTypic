"""Reach a file through directories whose identity this process holds.

Both consumers of this module -- recompile transition recovery and the Browse
store route -- must prove that the bytes they read came from the directory they
validated, not from one swapped underneath them between the check and the open.
A path-based open cannot prove that; an open relative to a held descriptor or
handle can. This module is the one place that knows how.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable


class IdentityIoUnavailable(RuntimeError):
    """This platform cannot bind directory access to held identities."""


class IdentityRefused(ValueError):
    """A component, type, link count or identity failed the contract."""


@runtime_checkable
class HeldDirectory(Protocol):
    """One directory held open by identity for the life of the hold."""

    path: Path

    def child_directory(self, name: str) -> "HeldDirectory": ...

    def list_names(self) -> tuple[str, ...]: ...

    def read_regular_bytes(
        self, name: str, *, max_bytes: int | None = None
    ) -> bytes: ...

    def read_regular_with_stat(
        self, name: str, *, max_bytes: int | None = None
    ) -> tuple[bytes, os.stat_result]: ...

    def open_regular_stream(self, name: str) -> BinaryIO: ...

    def reverify(self) -> None: ...


def validate_component(name: str) -> str:
    """Return *name* if it is one canonical path component, else refuse."""
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or Path(name).name != name
    ):
        raise IdentityRefused(f"not a canonical path component: {name!r}")
    return name


def _select_backend():
    if os.name == "posix":
        from . import _identity_io_posix

        if _identity_io_posix.SUPPORTED:
            return _identity_io_posix
    elif os.name == "nt":
        from . import _identity_io_windows

        if _identity_io_windows.SUPPORTED:
            return _identity_io_windows
    return None


# Keep this call at the BOTTOM of the module. Each backend imports
# ``HeldDirectory``/``IdentityRefused``/``validate_component`` from here at its
# own module scope, so it may only use names defined ABOVE this line: when
# ``_select_backend()`` runs, this module is still partially initialized.
_BACKEND = _select_backend()


def identity_io_available() -> bool:
    """Whether this platform can hold directories by identity."""
    return _BACKEND is not None


def active_backend_name() -> str:
    """Return ``posix``, ``windows`` or ``unavailable``."""
    return "unavailable" if _BACKEND is None else _BACKEND.BACKEND_NAME


def open_identity_directory(
    path: Path,
) -> AbstractContextManager[HeldDirectory]:
    """Hold *path* by identity, refusing to follow any link on the way in.

    Args:
        path: Absolute directory path to hold.

    Returns:
        A context manager yielding the held directory. Every descendant reached
        through it is opened relative to a handle this process holds.

    Raises:
        IdentityIoUnavailable: If no backend supports this platform.
        FileNotFoundError: If *path* does not exist.
        IdentityRefused: If *path* is not a canonical, non-link directory.
    """
    if _BACKEND is None:
        raise IdentityIoUnavailable(
            "this platform cannot bind directory access to held identities"
        )
    return _BACKEND.open_identity_directory(Path(path))
