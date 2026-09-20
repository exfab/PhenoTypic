"""An in-memory NT-handle model for the Windows identity-IO backend.

The Windows backend cannot execute on a Linux CI node, so the refusal contract
would otherwise be proved on exactly one of the two backends that must satisfy
it. This model closes that gap by standing in for :class:`_CtypesWindowsApi`:
it answers the same questions Win32 answers, in the same shapes, while running
anywhere.

What it proves and what it does not:

* It **does** prove this backend's logic -- that every child is reached through
  a held handle, that the walk refuses what the spec says it refuses, and that
  identity is re-resolved rather than read back from the handle it was taken
  from. It inherits :class:`_MemoryWindowsApi`'s refusal of path-based child
  operations, so a backend that forgot to route through a held handle fails
  here rather than in production.
* It **never** proves Win32's behaviour. Every error code it raises and every
  ``.``/``..`` entry it reports is a transcription of the 2026-09-20 probe
  recorded in the design spec, not an observation made by this file.

Structure is modelled; bytes are not. The model is built by walking the real
tree once at construction, so ``adopt_descriptor`` and ``stream`` can hand back
real descriptors and real :class:`os.stat_result` values -- which is the whole
point of ``read_regular_with_stat``. The consequence is that the model is a
**snapshot**: a test that mutates the tree *after* the hold is taken cannot be
run against it, and those tests say so in their own docstrings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO

from phenotypic.sdk_._identity_io_windows import (
    _ERROR_ACCESS_DENIED,
    _ERROR_DIRECTORY,
)
from tests.unit.sdk_.test_windows_metadata_journal import _MemoryWindowsApi

#: The ``\\?\`` spelling ``_extended_length`` produces. The model strips it
#: rather than ignoring it, so the prefix itself is exercised on this lane.
_EXTENDED_PREFIX = "\\\\?\\"


def _path_key(path: Path) -> tuple[str, ...]:
    """Spell *path* the way :class:`_MemoryWindowsApi` keys its namespace."""
    return (path.anchor, *path.parts[1:])



def _true_link_count(path: Path) -> int:
    """Return *path*'s hard-link count, opening it when the OS requires it.

    Windows' no-follow ``stat`` takes a fast path that never opens the file
    and leaves ``st_nlink`` at ``0``. Seeding the model from that made every
    file look multi-linked: the five ordinary read tests were refused with
    "not a single-link file", while ``test_a_multi_link_file_is_refused``
    passed *for the wrong reason* -- the one shape a refusal test must never
    pass in. ``fstat`` on an open handle reports the real count on both
    platforms.
    """
    count = path.stat(follow_symlinks=False).st_nlink
    if count:
        return count
    with path.open("rb") as handle:
        return os.fstat(handle.fileno()).st_nlink or 1

class _MemoryIdentityWindowsApi(_MemoryWindowsApi):
    """The journal's handle model, extended for read-only identity I/O."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, expect_share_delete=True)
        self.real: dict[tuple[str, ...], Path] = {}
        self.link_counts: dict[tuple[str, ...], int] = {}
        # ``_MemoryWindowsApi.__init__`` registers every prefix of *root* as a
        # directory whether or not it exists, which would make a missing root
        # open successfully. ``self.real`` is the authority here instead, and
        # it only ever gains entries the scan actually found.
        if root.is_dir() and not root.is_symlink():
            self._scan(root, _path_key(root))

    def _scan(self, directory: Path, prefix: tuple[str, ...]) -> None:
        self.real[prefix] = directory
        self.directories.add(prefix)
        with os.scandir(directory) as entries:
            for entry in entries:
                key = prefix + (entry.name,)
                self.real[key] = Path(entry.path)
                if entry.is_symlink():
                    # A link is opened as itself, never followed: it is a
                    # member the backend must refuse after inspecting it, not
                    # an entry that disappears from the namespace.
                    self.reparse.add(key)
                    self.files[key] = b""
                    self.link_counts[key] = 1
                elif entry.is_dir(follow_symlinks=False):
                    self._scan(Path(entry.path), key)
                else:
                    self.files[key] = b""
                    self.link_counts[key] = _true_link_count(
                        Path(entry.path)
                    )

    def open_anchor(self, anchor: str, *, share_delete: bool) -> int:
        assert share_delete is self.expect_share_delete
        if not anchor.startswith(_EXTENDED_PREFIX):
            return super().open_anchor(anchor, share_delete=share_delete)
        key = _path_key(Path(anchor[len(_EXTENDED_PREFIX) :]))
        if key not in self.real:
            raise FileNotFoundError(anchor)
        return self._handle(key)

    def open_directory(
        self,
        parent: int,
        name: str,
        *,
        create: bool,
        share_delete: bool,
    ) -> int:
        self._relative_name(name)
        key = self.handles[parent] + (name,)
        if key in self.files:
            # ``FILE_DIRECTORY_FILE`` against a non-directory:
            # ``STATUS_NOT_A_DIRECTORY`` -> ``ERROR_DIRECTORY``.
            raise OSError(_ERROR_DIRECTORY, f"not a directory: {name}")
        return super().open_directory(
            parent, name, create=create, share_delete=share_delete
        )

    def open_regular_read(self, parent: int, name: str) -> int:
        self._relative_name(name)
        key = self.handles[parent] + (name,)
        if key in self.directories:
            # ``FILE_NON_DIRECTORY_FILE`` against a directory:
            # ``STATUS_FILE_IS_A_DIRECTORY``, which
            # ``RtlNtStatusToDosError`` folds onto ``ERROR_ACCESS_DENIED``.
            # Modelling the *collision* is the point -- it is what forces the
            # backend to disambiguate rather than assume.
            raise OSError(_ERROR_ACCESS_DENIED, f"is a directory: {name}")
        if key not in self.files:
            raise FileNotFoundError(name)
        return self._handle(key)

    def open_file(
        self,
        parent: int,
        name: str,
        *,
        create_new: bool,
        share_delete: bool,
    ) -> int:
        raise AssertionError(
            "identity I/O must reach a member through open_regular_read; "
            "open_file carries the journal's FILE_WRITE_DATA|"
            "FILE_WRITE_ATTRIBUTES|DELETE mask, which the probe confirmed a "
            "read-only store refuses with ERROR_ACCESS_DENIED"
        )

    def is_directory(self, handle: int) -> bool:
        return self.handles[handle] in self.directories

    def link_count(self, handle: int) -> int:
        return self.link_counts[self.handles[handle]]

    def file_size(self, handle: int) -> int:
        return self.real[self.handles[handle]].stat().st_size

    def list_names(self, handle: int) -> tuple[str, ...]:
        """Return child names, with the ``.`` and ``..`` Windows reports.

        The probe saw ``['.', '..', 'alpha.json', 'beta.json']`` from a real
        directory handle. Omitting them here would leave the backend's filter
        unexercised on this lane while the test asserting the filter still
        went green.
        """
        prefix = self.handles[handle]
        names = [".", ".."]
        for key in (*self.directories, *self.files):
            if len(key) == len(prefix) + 1 and key[:-1] == prefix:
                names.append(key[-1])
        return tuple(names)

    def adopt_descriptor(self, handle: int) -> int:
        """Hand back a real descriptor; ownership leaves the handle table."""
        path = self.real[self.handles.pop(handle)]
        return os.open(path, os.O_RDONLY)

    def stream(self, handle: int) -> BinaryIO:
        path = self.real[self.handles.pop(handle)]
        return open(path, "rb")


def memory_api_factory(path: Path) -> _MemoryIdentityWindowsApi:
    """Model the tree under *path* as it stands right now."""
    return _MemoryIdentityWindowsApi(Path(os.path.abspath(os.fspath(path))))
