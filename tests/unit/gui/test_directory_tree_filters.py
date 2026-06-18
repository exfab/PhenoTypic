"""Lock-in tests for ``directory_tree`` extension/file-selection filters.

These tests pin the public contract of
:func:`phenotypic.gui.builder._directory_browser.directory_tree`'s
``extensions`` and ``select_files`` keyword arguments, alongside the existing
path-safety guarantees (no hidden entries, no out-of-root symlink traversal,
parent link gated by the configured root).

The function returns a layout-only Dash tree, so every assertion walks the
returned :class:`html.Div` and inspects the pattern-matching ids of the
:class:`dbc.ListGroupItem` leaves rather than matching against the rendered
text/styling — that's the only stable contract callbacks consume.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Skip the whole module if the optional GUI deps are unavailable, mirroring the
# pattern used by ``test_optional_deps.py`` for the Dash builder surface.
dash = pytest.importorskip("dash")
dbc = pytest.importorskip("dash_bootstrap_components")

pytestmark = pytest.mark.skipif(
    dash is None or dbc is None,
    reason="dash / dash_bootstrap_components not installed",
)

from phenotypic.gui.builder._directory_browser import (  # noqa: E402
    IMAGE_EXTS,
    PIPELINE_EXTS,
    directory_tree,
)
from phenotypic.sdk_ import matches_any_suffix  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_item_ids(div: Any) -> List[Dict[str, Any]]:
    """Pull pattern-matching ids out of the rendered ListGroup.

    Walks the tree of Dash components looking for ids that are dicts whose
    ``"type"`` field starts with ``"dir-entry"`` (the convention used by
    ``directory_tree``). Resilient to wrapping changes — we only care about
    the leaf ids, not intermediate Divs/Rows.
    """
    items: List[Dict[str, Any]] = []

    def walk(node: Any) -> None:
        children = getattr(node, "children", None)
        if isinstance(children, list):
            for c in children:
                walk(c)
        elif children is not None:
            walk(children)
        nid = getattr(node, "id", None)
        if isinstance(nid, dict) and str(nid.get("type", "")).startswith("dir-entry"):
            items.append(nid)

    walk(div)
    return items


def _make_mixed_tree(tmp_path: Path) -> None:
    """Populate ``tmp_path`` with the canonical mixed fixture used below."""
    (tmp_path / "colony.tif").write_bytes(b"")
    (tmp_path / "colony.png").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("notes")
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")
    (tmp_path / "runs").mkdir()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_extensions_match_image_exts(tmp_path: Path) -> None:
    """Default ``extensions=IMAGE_EXTS`` surfaces images + dirs only."""
    _make_mixed_tree(tmp_path)

    tree = directory_tree(tmp_path)
    ids = _collect_item_ids(tree)

    kinds = {entry["kind"] for entry in ids}
    assert kinds == {"dir", "file"}

    file_paths = {Path(e["path"]).name for e in ids if e["kind"] == "file"}
    assert file_paths == {"colony.tif", "colony.png"}

    # Sanity: every image extension surfaced must come from IMAGE_EXTS.
    for entry in ids:
        if entry["kind"] == "file":
            assert Path(entry["path"]).suffix.lower() in IMAGE_EXTS


def test_explicit_pipeline_exts_filter(tmp_path: Path) -> None:
    """``extensions=PIPELINE_EXTS`` shows typed pipeline configs and legacy JSON."""
    _make_mixed_tree(tmp_path)
    (tmp_path / "pipeline.json").write_text("{}")
    (tmp_path / "recipe.json.pht-pipe").write_text("{}")
    (tmp_path / "tuning_spec.json.pht-tune").write_text("{}")
    # An extra .txt was already created by _make_mixed_tree; create no dup.

    tree = directory_tree(tmp_path, extensions=PIPELINE_EXTS)
    ids = _collect_item_ids(tree)

    file_paths = {Path(e["path"]).name for e in ids if e["kind"] == "file"}
    assert file_paths == {"pipeline.json", "recipe.json.pht-pipe"}

    # No tif/png/txt/csv files should appear.
    for entry in ids:
        if entry["kind"] == "file":
            assert matches_any_suffix(entry["path"], PIPELINE_EXTS)

    # Folder filtering is independent of the file-extension filter.
    dir_names = {Path(e["path"]).name for e in ids if e["kind"] == "dir"}
    assert "runs" in dir_names


def test_extensions_none_surfaces_all_files(tmp_path: Path) -> None:
    """``extensions=None`` surfaces every non-hidden file, dirs first then files alpha."""
    _make_mixed_tree(tmp_path)

    tree = directory_tree(tmp_path, extensions=None)
    ids = _collect_item_ids(tree)

    file_paths = {Path(e["path"]).name for e in ids if e["kind"] == "file"}
    assert file_paths == {"colony.tif", "colony.png", "notes.txt", "data.csv"}

    # Order contract: directories (alpha) first, then files (alpha).
    kinds_in_order = [e["kind"] for e in ids]
    # Every "dir" should come before every "file".
    last_dir_idx = max(
        (i for i, k in enumerate(kinds_in_order) if k == "dir"), default=-1
    )
    first_file_idx = min(
        (i for i, k in enumerate(kinds_in_order) if k == "file"),
        default=len(kinds_in_order),
    )
    assert last_dir_idx < first_file_idx

    # Within each kind, names sorted case-insensitive ascending.
    file_names_in_order = [
        Path(e["path"]).name for e in ids if e["kind"] == "file"
    ]
    assert file_names_in_order == sorted(
        file_names_in_order, key=lambda n: n.lower()
    )
    dir_names_in_order = [Path(e["path"]).name for e in ids if e["kind"] == "dir"]
    assert dir_names_in_order == sorted(
        dir_names_in_order, key=lambda n: n.lower()
    )


def test_select_files_false_hides_files_only(tmp_path: Path) -> None:
    """``select_files=False`` drops every file, regardless of extension."""
    _make_mixed_tree(tmp_path)
    (tmp_path / "pipeline.json").write_text("{}")

    tree = directory_tree(
        tmp_path, select_files=False, extensions=PIPELINE_EXTS
    )
    ids = _collect_item_ids(tree)

    kinds = {entry["kind"] for entry in ids}
    # Only directory entries; no file rows even though pipeline.json matches.
    assert "file" not in kinds
    assert kinds <= {"dir", "parent"}

    dir_names = {Path(e["path"]).name for e in ids if e["kind"] == "dir"}
    assert dir_names == {"runs"}


def test_id_type_propagates_to_items(tmp_path: Path) -> None:
    """Custom ``id_type`` is stamped on every item's id; default stays ``dir-entry``."""
    _make_mixed_tree(tmp_path)

    custom = directory_tree(tmp_path, id_type="dir-entry-json")
    custom_ids = _collect_item_ids(custom)
    assert custom_ids, "expected at least one entry to assert against"
    assert {entry["type"] for entry in custom_ids} == {"dir-entry-json"}

    default = directory_tree(tmp_path)
    default_ids = _collect_item_ids(default)
    assert default_ids, "expected at least one entry to assert against"
    assert {entry["type"] for entry in default_ids} == {"dir-entry"}


def test_parent_link_respects_root(tmp_path: Path) -> None:
    """Parent entry only appears when ``current`` is strictly below ``root``."""
    inner = tmp_path / "inner"
    inner.mkdir()

    # current == root -> no parent entry.
    tree_at_root = directory_tree(root=inner, current=inner)
    kinds_at_root = {e["kind"] for e in _collect_item_ids(tree_at_root)}
    assert "parent" not in kinds_at_root

    # current is a child of root -> exactly one parent entry pointing at root.
    tree_below = directory_tree(root=tmp_path, current=inner)
    parent_entries = [
        e for e in _collect_item_ids(tree_below) if e["kind"] == "parent"
    ]
    assert len(parent_entries) == 1
    assert Path(parent_entries[0]["path"]).resolve() == tmp_path.resolve()

    # Custom id_type also flows to the parent link.
    tree_custom = directory_tree(
        root=tmp_path, current=inner, id_type="dir-entry-image"
    )
    parent_custom = [
        e for e in _collect_item_ids(tree_custom) if e["kind"] == "parent"
    ]
    assert len(parent_custom) == 1
    assert parent_custom[0]["type"] == "dir-entry-image"


def test_hidden_entries_skipped(tmp_path: Path) -> None:
    """Names starting with ``.`` are filtered from both dir and file listings."""
    (tmp_path / ".hidden_dir").mkdir()
    (tmp_path / ".secret.json").write_text("{}")
    (tmp_path / "visible.json").write_text("{}")

    tree = directory_tree(tmp_path, extensions=PIPELINE_EXTS)
    ids = _collect_item_ids(tree)

    names = {Path(e["path"]).name for e in ids}
    assert names == {"visible.json"}
    for entry in ids:
        assert not Path(entry["path"]).name.startswith(".")


def test_symlink_above_root_excluded(tmp_path: Path) -> None:
    """Symlinks resolving outside ``root`` must not appear in the listing."""
    inner = tmp_path / "inner"
    inner.mkdir()
    target = tmp_path.parent  # one level above the configured root
    link = inner / "escape"
    try:
        os.symlink(target, link)
    except OSError:
        # Windows without SeCreateSymbolicLinkPrivilege, or other platform
        # limitation. Mirror the platform-skip approach in
        # tests/unit/gui/test_optional_deps.py.
        pytest.skip("symlink creation not permitted on this platform")

    tree = directory_tree(root=inner, current=inner)
    ids = _collect_item_ids(tree)
    names = {Path(e["path"]).name for e in ids}
    assert "escape" not in names
