"""POSIX backend: directory file descriptors with ``O_NOFOLLOW``."""

from __future__ import annotations

import errno
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from ._identity_io import HeldDirectory, IdentityRefused, validate_component

BACKEND_NAME = "posix"

# ``os.stat`` is required by ``reverify``, which re-resolves a child's name
# inside its held parent. The constant this replaces
# (``_cli_recompile_recovery.py:219-222``) also required mkdir/unlink/rename
# for a transition *writer* that no longer exists (`:246-250`); this facade is
# read-only, so those are deliberately dropped and `os.stat` deliberately kept.
SUPPORTED = (
    os.name == "posix"
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_NONBLOCK")
    and os.listdir in os.supports_fd
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
)

_DIR_FLAGS = (
    (os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW) if SUPPORTED else os.O_RDONLY
)
_FILE_FLAGS = (
    (os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW) if SUPPORTED else os.O_RDONLY
)


def _refuse_os_error(exc: OSError, subject: Path) -> "IdentityRefused":
    """Translate a no-follow open's refusal errno into the contract's error.

    ``O_NOFOLLOW`` on a symlink raises ``ELOOP``; ``O_DIRECTORY`` on a file
    raises ``ENOTDIR``. Both are refusals 2 and 3, not transport failures --
    the code this backend replaces mapped them the same way
    (``_cli_recompile_recovery.py:273-274``).

    ``OSError.errno`` is optional, and an ``OSError`` carrying none can never
    reach here -- every call site tests membership in ``_REFUSED_ERRNOS``
    first, which ``None`` fails. The branch exists so that a future caller
    which forgets that still gets a refusal naming the entry, rather than a
    ``TypeError`` raised from inside the error path.
    """
    if exc.errno is None:
        code: str | int = "no errno"
    else:
        code = errno.errorcode.get(exc.errno, exc.errno)
    return IdentityRefused(f"{code}: {subject}")


_REFUSED_ERRNOS = frozenset(
    {errno.ELOOP, errno.ENOTDIR, errno.EISDIR, errno.EMLINK}
)


class _PosixHeldDirectory:
    """A directory pinned by an open descriptor and its `st_dev`/`st_ino`.

    ``parent``/``name`` are how ``reverify`` re-resolves this directory: an
    identity read back from ``self._fd`` is the identity that fd was opened
    with and can never differ, so a check written that way cannot fail.
    """

    def __init__(
        self,
        path: Path,
        fd: int,
        owned: bool,  # retained for symmetry; _close always owns its own fd
        *,
        parent: "_PosixHeldDirectory | None" = None,
        name: str | None = None,
    ) -> None:
        self.path = path
        self._fd = fd
        self._owned = owned
        self._parent = parent
        self._name = name
        self._closed = False
        identity = os.fstat(fd)
        if not stat.S_ISDIR(identity.st_mode):
            raise IdentityRefused(f"not a directory: {path}")
        self._identity = (identity.st_dev, identity.st_ino)
        self._children: list[_PosixHeldDirectory] = []

    def child_directory(self, name: str) -> HeldDirectory:
        validate_component(name)
        try:
            child_fd = os.open(name, _DIR_FLAGS, dir_fd=self._fd)
        except OSError as exc:
            if exc.errno in _REFUSED_ERRNOS:
                raise _refuse_os_error(exc, self.path / name) from exc
            raise
        try:
            child = _PosixHeldDirectory(
                self.path / name, child_fd, owned=True, parent=self, name=name
            )
        except BaseException:
            os.close(child_fd)
            raise
        self._children.append(child)
        return child

    def list_names(self) -> tuple[str, ...]:
        self.reverify()
        return tuple(sorted(os.listdir(self._fd)))

    def _open_regular_fd(self, name: str) -> int:
        validate_component(name)
        try:
            file_fd = os.open(name, _FILE_FLAGS, dir_fd=self._fd)
        except OSError as exc:
            if exc.errno in _REFUSED_ERRNOS:
                raise _refuse_os_error(exc, self.path / name) from exc
            raise
        try:
            identity = os.fstat(file_fd)
            if not stat.S_ISREG(identity.st_mode):
                raise IdentityRefused(
                    f"not a regular file: {self.path / name}"
                )
            if identity.st_nlink != 1:
                raise IdentityRefused(
                    f"not a single-link file: {self.path / name}"
                )
        except BaseException:
            os.close(file_fd)
            raise
        return file_fd

    def read_regular_bytes(
        self, name: str, *, max_bytes: int | None = None
    ) -> bytes:
        file_fd = self._open_regular_fd(name)
        try:
            size = os.fstat(file_fd).st_size
            if max_bytes is not None and size > max_bytes:
                raise IdentityRefused(
                    f"{self.path / name} is larger than {max_bytes} bytes"
                )
            with os.fdopen(file_fd, "rb", closefd=False) as stream:
                payload = stream.read()
        finally:
            os.close(file_fd)
        self.reverify()
        return payload

    def read_regular_with_stat(
        self, name: str, *, max_bytes: int | None = None
    ) -> tuple[bytes, os.stat_result]:
        """Return the bytes and the stat of the same open file description.

        The torn-read comparison lives here rather than in the caller: the
        publication token folds ``st_mtime_ns``/``st_ctime_ns``/``st_ino`` and
        must be computed over one coherent read
        (``_io_constants.py:1946-1970``).
        """
        file_fd = self._open_regular_fd(name)
        try:
            before = os.fstat(file_fd)
            if max_bytes is not None and before.st_size > max_bytes:
                raise IdentityRefused(
                    f"{self.path / name} is larger than {max_bytes} bytes"
                )
            with os.fdopen(file_fd, "rb", closefd=False) as stream:
                payload = stream.read()
            after = os.fstat(file_fd)
        finally:
            os.close(file_fd)
        changed = (
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_ino,
        ) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_ino,
        )
        if changed or len(payload) != after.st_size:
            raise IdentityRefused(f"{self.path / name} changed during the read")
        return payload, after

    def open_regular_stream(self, name: str) -> BinaryIO:
        file_fd = self._open_regular_fd(name)
        try:
            stream = os.fdopen(file_fd, "rb")
        except BaseException:
            os.close(file_fd)
            raise
        return stream

    def reverify(self) -> None:
        """Re-resolve this directory's NAME and compare it to what is held.

        The stat result is only ever compared, never opened through, so this
        re-introduces no TOCTOU window: a hostile swap makes the check fail and
        can never make the facade open the wrong object, because every open
        still goes through the held descriptor.

        A child re-resolves one component inside its held parent, so no
        ancestor is re-walked. The root has no held parent and re-walks its
        whole prefix -- it catches its own rename, deletion or replacement, but
        not the swap of an ancestor. That asymmetry is deliberate and is why
        the spec's refusal 5 binds children.
        """
        if self._parent is None or self._name is None:
            current = os.lstat(self.path)
        else:
            current = os.stat(
                self._name, dir_fd=self._parent._fd, follow_symlinks=False
            )
        if (current.st_dev, current.st_ino) != self._identity:
            raise IdentityRefused(f"identity changed: {self.path}")

    def _close(self) -> None:
        """Close children leaf-first, then self. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for child in self._children:
            child._close()
        os.close(self._fd)


@contextmanager
def open_identity_directory(path: Path) -> Iterator[HeldDirectory]:
    """Hold *path* by identity through a no-follow directory descriptor."""
    root = Path(os.path.abspath(os.fspath(path)))
    fd = os.open(root, _DIR_FLAGS)
    held: _PosixHeldDirectory | None = None
    try:
        held = _PosixHeldDirectory(root, fd, owned=True)
        yield held
    finally:
        # In the ``finally``, not after the ``yield``: when the ``with`` body
        # raises, the generator resumes *at* the yield and anything below it is
        # skipped, leaking every child descriptor in ``_children``. In the
        # Browse route that is one leak per failing request -- the path most
        # exercised under load.
        if held is not None:
            held._close()
        else:
            os.close(fd)
