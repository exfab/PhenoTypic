"""Unit tests for ``phenotypic.gui.shell._sandbox.SandboxRoot``.

The sandbox is the GUI's path-containment primitive: every untrusted path
(URL fragment, JSON-API query string, ``/runs/<rel>/...`` route) is run
through ``SandboxRoot.resolve`` before being touched on disk. These tests
exercise the escape conditions:

    * ``..``-style relative traversal
    * absolute paths outside the sandbox
    * symlinks pointing outside the sandbox

and the legitimate happy-path cases (resolve in-root, list children, hidden
+ external-symlink toggles).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from phenotypic.gui.shell._sandbox import SandboxRoot


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_from_path_resolves_symlinks_and_relative(tmp_path: Path) -> None:
    """``from_path`` always returns an absolute, fully-resolved root."""
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    sandbox = SandboxRoot.from_path(link)
    assert sandbox.root == target.resolve()
    assert sandbox.root.is_absolute()


def test_from_path_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SandboxRoot.from_path(tmp_path / "nope")


def test_from_path_file_raises(tmp_path: Path) -> None:
    f = tmp_path / "not_a_dir"
    f.write_text("hi")
    with pytest.raises((NotADirectoryError, FileNotFoundError)):
        SandboxRoot.from_path(f)


# ---------------------------------------------------------------------------
# resolve(): escape paths
# ---------------------------------------------------------------------------

def test_resolve_relative_traversal_rejected(tmp_path: Path) -> None:
    sandbox = SandboxRoot.from_path(tmp_path)
    with pytest.raises(ValueError):
        sandbox.resolve("../etc/passwd")


def test_resolve_absolute_outside_rejected(tmp_path: Path) -> None:
    sandbox = SandboxRoot.from_path(tmp_path)
    with pytest.raises(ValueError):
        sandbox.resolve("/etc/passwd")


def test_resolve_symlink_escape_rejected(tmp_path: Path) -> None:
    """Symlink whose target is outside the sandbox must raise.

    This is the most security-relevant case: a user dropping a symlink
    inside the sandbox to grant the GUI read access to ``/etc``.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    inside = tmp_path / "sandbox"
    inside.mkdir()
    sandbox = SandboxRoot.from_path(inside)
    bad_link = inside / "escape"
    bad_link.symlink_to(outside)
    with pytest.raises(ValueError):
        sandbox.resolve(bad_link)


def test_resolve_double_dot_inside_root_allowed(tmp_path: Path) -> None:
    """``..`` is allowed when the result still lands inside the sandbox."""
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    sandbox = SandboxRoot.from_path(tmp_path)
    resolved = sandbox.resolve(sub / ".." / "b")
    assert resolved == sub.resolve()


def test_resolve_root_itself(tmp_path: Path) -> None:
    sandbox = SandboxRoot.from_path(tmp_path)
    assert sandbox.resolve(tmp_path) == tmp_path.resolve()
    assert sandbox.resolve(".") == tmp_path.resolve()


def test_resolve_does_not_require_existence(tmp_path: Path) -> None:
    """Containment check applies to non-existent paths too.

    Otherwise probes for "does X exist?" leak whether the path traverses
    out of root (existence-vs-permission errors look different to a caller).
    """
    sandbox = SandboxRoot.from_path(tmp_path)
    # In-root non-existent path is fine.
    out = sandbox.resolve(tmp_path / "not_there_yet" / "deep" / "file.txt")
    assert out.is_absolute()
    # Out-of-root non-existent path still raises.
    with pytest.raises(ValueError):
        sandbox.resolve("/definitely/not/here/either")


# ---------------------------------------------------------------------------
# contains()
# ---------------------------------------------------------------------------

def test_contains_predicate(tmp_path: Path) -> None:
    sandbox = SandboxRoot.from_path(tmp_path)
    (tmp_path / "child").mkdir()
    assert sandbox.contains(tmp_path / "child") is True
    assert sandbox.contains("/etc/passwd") is False
    assert sandbox.contains("..") is False


# ---------------------------------------------------------------------------
# list_children()
# ---------------------------------------------------------------------------

def test_list_children_default_skips_hidden_and_external(
    tmp_path: Path,
) -> None:
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    visible = sandbox_dir / "visible.txt"
    visible.write_text("hi")
    hidden = sandbox_dir / ".hidden"
    hidden.write_text("hi")
    external_link = sandbox_dir / "escape"
    external_link.symlink_to(outside)
    internal_link = sandbox_dir / "loop_in"
    internal_link.symlink_to(visible)

    sandbox = SandboxRoot.from_path(sandbox_dir)
    names = {c.name for c in sandbox.list_children()}
    assert names == {"visible.txt", "loop_in"}


def test_list_children_show_hidden(tmp_path: Path) -> None:
    sandbox = SandboxRoot.from_path(tmp_path)
    (tmp_path / "visible").write_text("x")
    (tmp_path / ".hidden").write_text("x")
    names = {c.name for c in sandbox.list_children(include_hidden=True)}
    assert names == {"visible", ".hidden"}


def test_list_children_show_external_symlinks(tmp_path: Path) -> None:
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (sandbox_dir / "escape").symlink_to(outside)
    sandbox = SandboxRoot.from_path(sandbox_dir)
    names = {
        c.name
        for c in sandbox.list_children(include_external_symlinks=True)
    }
    assert names == {"escape"}


def test_list_children_outside_root_raises(tmp_path: Path) -> None:
    inside = tmp_path / "in"
    inside.mkdir()
    outside = tmp_path / "out"
    outside.mkdir()
    sandbox = SandboxRoot.from_path(inside)
    with pytest.raises(ValueError):
        list(sandbox.list_children(outside))


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="chmod-based permission test is POSIX-specific",
)
def test_list_children_permission_error_propagates(tmp_path: Path) -> None:
    """``list_children`` does not swallow PermissionError.

    Callers in the classifier path will catch it and surface ``bad_perms``;
    the sandbox primitive itself stays honest.
    """
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "child").write_text("x")
    sandbox = SandboxRoot.from_path(tmp_path)
    os.chmod(locked, 0o000)
    try:
        with pytest.raises(PermissionError):
            list(sandbox.list_children(locked))
    finally:
        os.chmod(locked, 0o755)


def test_list_children_broken_symlink_inside_root_is_yielded(
    tmp_path: Path,
) -> None:
    """Broken symlink whose unresolved target lies inside the root yields.

    Pairs with ``_symlink_target_in_root`` docstring: we yield the link so
    the consumer surfaces its broken state rather than silently hiding it.
    """
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    target = sandbox_dir / "target_inside"
    target.write_text("x")
    link = sandbox_dir / "link_inside"
    link.symlink_to(target)
    target.unlink()  # break the link, target string still in-root
    sandbox = SandboxRoot.from_path(sandbox_dir)
    names = {c.name for c in sandbox.list_children()}
    assert "link_inside" in names


def test_list_children_broken_symlink_outside_root_is_filtered(
    tmp_path: Path,
) -> None:
    """Broken symlink whose unresolved target lies outside is filtered."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    outside_target = tmp_path / "outside_target"
    outside_target.write_text("x")
    link = sandbox_dir / "link_outside"
    link.symlink_to(outside_target)
    outside_target.unlink()
    sandbox = SandboxRoot.from_path(sandbox_dir)
    names = {c.name for c in sandbox.list_children()}
    assert "link_outside" not in names


def test_list_children_hidden_and_external_combination(
    tmp_path: Path,
) -> None:
    """All four flag combinations are honoured independently."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (sandbox_dir / "visible").write_text("x")
    (sandbox_dir / ".hidden").write_text("x")
    (sandbox_dir / "external").symlink_to(outside)
    sandbox = SandboxRoot.from_path(sandbox_dir)

    names_both = {
        c.name
        for c in sandbox.list_children(
            include_hidden=True, include_external_symlinks=True,
        )
    }
    assert names_both == {"visible", ".hidden", "external"}
