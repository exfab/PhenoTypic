"""The refusal contract every identity-IO backend must satisfy."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from phenotypic.sdk_ import _identity_io


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "store" / "nested").mkdir(parents=True)
    (tmp_path / "store" / "root.json").write_bytes(b'{"ok": true}')
    (tmp_path / "store" / "nested" / "chunk").write_bytes(b"0123456789")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret").write_bytes(b"secret")
    return tmp_path


def test_a_held_directory_reads_its_own_regular_files(tree: Path) -> None:
    with _identity_io.open_identity_directory(tree / "store") as held:
        assert held.read_regular_bytes("root.json") == b'{"ok": true}'


def test_a_child_directory_stays_held(tree: Path) -> None:
    with _identity_io.open_identity_directory(tree / "store") as held:
        nested = held.child_directory("nested")
        assert nested.read_regular_bytes("chunk") == b"0123456789"


def test_listing_excludes_dot_entries(tree: Path) -> None:
    with _identity_io.open_identity_directory(tree / "store") as held:
        assert sorted(held.list_names()) == ["nested", "root.json"]


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b"])
def test_a_non_canonical_component_is_refused(tree: Path, name: str) -> None:
    """Refusal 1. Traversal must die at the component, not the resolved path."""
    with _identity_io.open_identity_directory(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.read_regular_bytes(name)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_a_symlinked_entry_is_refused(tree: Path) -> None:
    """Refusal 2. The link points outside; following it would leak `secret`."""
    link = tree / "store" / "escape.json"
    try:
        link.symlink_to(tree / "outside" / "secret")
    except (OSError, NotImplementedError):
        pytest.skip("this platform will not create a symlink here")
    with _identity_io.open_identity_directory(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.read_regular_bytes("escape.json")


def test_a_directory_is_refused_where_a_file_is_required(tree: Path) -> None:
    """Refusal 3."""
    with _identity_io.open_identity_directory(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.read_regular_bytes("nested")


def test_a_file_is_refused_where_a_directory_is_required(tree: Path) -> None:
    """Refusal 3, the other direction."""
    with _identity_io.open_identity_directory(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.child_directory("root.json")


def test_a_multi_link_file_is_refused(tree: Path) -> None:
    """Refusal 4. A hard link is a second name for authority bytes."""
    try:
        os.link(tree / "store" / "root.json", tree / "store" / "alias.json")
    except (OSError, NotImplementedError):
        pytest.skip("this filesystem will not create a hard link")
    with _identity_io.open_identity_directory(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.read_regular_bytes("alias.json")


def test_a_missing_entry_raises_file_not_found(tree: Path) -> None:
    """Refusal 6: absence is evidence, not failure; callers branch on it."""
    with _identity_io.open_identity_directory(tree / "store") as held:
        with pytest.raises(FileNotFoundError):
            held.read_regular_bytes("absent.json")


def test_max_bytes_bounds_a_read(tree: Path) -> None:
    with _identity_io.open_identity_directory(tree / "store") as held:
        with pytest.raises(_identity_io.IdentityRefused):
            held.read_regular_bytes("root.json", max_bytes=4)


def test_a_swapped_directory_fails_reverification(tree: Path) -> None:
    """Refusal 5. The held identity, not the path, is the authority."""
    with _identity_io.open_identity_directory(tree / "store") as held:
        nested = held.child_directory("nested")
        os.rename(tree / "store" / "nested", tree / "store" / "gone")
        (tree / "store" / "nested").mkdir()
        with pytest.raises(_identity_io.IdentityRefused):
            nested.reverify()


def test_a_missing_root_raises_file_not_found(tree: Path) -> None:
    with pytest.raises(FileNotFoundError):
        with _identity_io.open_identity_directory(tree / "absent"):
            pass
