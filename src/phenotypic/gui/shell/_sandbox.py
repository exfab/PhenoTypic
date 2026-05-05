"""``SandboxRoot`` dataclass + safe-resolve helpers.

A ``SandboxRoot`` is a frozen-at-launch handle on the directory that the GUI is
allowed to read. Every path that comes from a request (sidebar URL, JSON-API
``?path=`` parameter, ``/runs/<rel>/...`` route) is run through
``SandboxRoot.resolve(...)`` before being touched on disk. Out-of-root paths
and symlinks pointing outside the sandbox raise ``ValueError``.

Single-user, in-process model — there is no auth gate. Cloud mode (multi-user,
per-session roots) is a non-goal for v1; see the TODO at the top.

TODO(cloud-deploy): when ``--mode=cloud`` ships, wire an auth gate via a Flask
``@before_request`` hook on every ``/sandbox/api/*`` and ``/runs/*`` route
(touch one place, not every Dash callback). The sandbox is currently
frozen-at-launch (single-user); cloud mode must make the root selectable
per session.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

__all__ = ["SandboxRoot"]


@dataclass(frozen=True)
class SandboxRoot:
    """Frozen-at-launch sandbox containment primitive.

    Attributes:
        root: Absolute, fully-resolved directory the GUI is allowed to read
            from. Symlinks in the path are resolved at construction time so
            ``contains`` / ``resolve`` comparisons are stable for the lifetime
            of the process.
    """

    root: Path

    @classmethod
    def from_path(cls, root: str | os.PathLike[str]) -> "SandboxRoot":
        """Construct a sandbox from a user-supplied root path.

        Resolves symlinks and verifies the root exists and is a directory.

        Args:
            root: A directory path. May be relative; will be resolved.

        Returns:
            A ``SandboxRoot`` with ``.root`` fully resolved and absolute.

        Raises:
            FileNotFoundError: If ``root`` does not exist.
            NotADirectoryError: If ``root`` exists but is not a directory.
            RuntimeError: If ``root`` resolution encounters a symlink loop
                (CPython's :meth:`Path.resolve` raises ``RuntimeError`` for
                cycles, not :class:`OSError`). The launcher catches and
                re-presents this as a startup error rather than a stack
                trace.
        """
        resolved = Path(root).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(
                f"sandbox root must be a directory: {resolved}"
            )
        return cls(root=resolved)

    def resolve(self, candidate: str | os.PathLike[str]) -> Path:
        """Resolve ``candidate`` to an absolute path inside the sandbox.

        ``candidate`` may be absolute or relative. Relative paths are joined
        against ``self.root``. Symlinks are followed; the resulting absolute
        path is verified to be inside ``self.root``. This catches both
        ``..``-style traversal and symlinks pointing outside the sandbox.

        Args:
            candidate: A path supplied by an untrusted source (URL, query
                string, JSON body).

        Returns:
            Absolute, fully-resolved ``Path`` guaranteed to be inside (or
            equal to) ``self.root``.

        Raises:
            ValueError: If ``candidate`` resolves outside ``self.root`` or
                follows a symlink that escapes ``self.root``.
        """
        cand = Path(candidate)
        joined = cand if cand.is_absolute() else self.root / cand
        # ``resolve(strict=False)`` is intentional: we want to apply the
        # same containment check whether or not the file currently exists,
        # otherwise probes for "does X exist?" leak whether the path traverses
        # out of root. ``strict=False`` resolves symlinks for path components
        # that do exist and leaves the rest as-is.
        resolved = joined.resolve(strict=False)
        if not self._contains(resolved):
            raise ValueError(
                f"path escapes sandbox root: {candidate!r} -> {resolved}"
            )
        return resolved

    def contains(self, candidate: str | os.PathLike[str]) -> bool:
        """Return ``True`` iff ``candidate`` resolves inside the sandbox.

        Convenience predicate that wraps ``resolve`` and swallows the
        ``ValueError`` raised on escapes. Useful for "should I render this
        at all?" checks where raising would be inconvenient.
        """
        try:
            self.resolve(candidate)
        except ValueError:
            return False
        return True

    def list_children(
        self,
        directory: str | os.PathLike[str] | None = None,
        *,
        include_hidden: bool = False,
        include_external_symlinks: bool = False,
    ) -> Iterator[Path]:
        """Yield direct children of ``directory`` (default: ``self.root``).

        Args:
            directory: Path to list. Defaults to the sandbox root. Validated
                via ``resolve`` first; out-of-root paths raise ``ValueError``.
            include_hidden: Yield dotfiles when ``True``. Default ``False``
                matches the sidebar's "Hidden files" toggle (off by default).
            include_external_symlinks: Yield symlinks whose targets escape
                ``self.root`` when ``True``. Default ``False`` matches the
                sidebar's "External symlinks" toggle. Non-symlink children
                are unaffected.

        Yields:
            Absolute ``Path`` objects, one per child. Symlinks are yielded
            unless they target outside the sandbox and the toggle is off.

        Raises:
            ValueError: If ``directory`` resolves outside the sandbox.
            PermissionError: If the directory cannot be listed; callers may
                choose to swallow this and surface the ``bad_perms`` capability
                badge instead.
        """
        target = self.resolve(directory) if directory is not None else self.root
        for child in target.iterdir():
            if not include_hidden and child.name.startswith("."):
                continue
            if child.is_symlink() and not include_external_symlinks:
                if not self._symlink_target_in_root(child):
                    continue
            yield child

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _contains(self, resolved: Path) -> bool:
        """Pure-comparison containment check on an already-resolved path."""
        try:
            resolved.relative_to(self.root)
        except ValueError:
            return False
        return True

    def _symlink_target_in_root(self, link: Path) -> bool:
        """Return ``True`` iff ``link`` is a symlink whose target is inside.

        Used by ``list_children`` to filter out external symlinks. Broken
        symlinks are evaluated against their unresolved target via
        ``Path.resolve(strict=False)`` — if the target string lands outside
        the sandbox, the link is filtered; if the target string would have
        landed inside the sandbox, the link is yielded even though it
        cannot currently be followed (the consumer surfaces the broken
        state via :func:`classify`).
        """
        try:
            target = link.resolve(strict=False)
        except (OSError, RuntimeError):
            # ``RuntimeError`` is raised by CPython on symlink cycles
            # (``"Symlink loop from..."``). Treat the link as external so
            # the caller filters it rather than crashing the render.
            return False
        return self._contains(target)
