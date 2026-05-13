"""Shared component-tree walking helpers for builder integration tests.

The Phase 4 inspector tests (wire-card + aux-section renderers) both
need to introspect Dash component trees: descend through every child,
locate a component by string-id or pattern-matching-id, and flatten
the descendant text content into a single string for substring asserts.
The four helpers below are byte-identical in both ``test_inspector_*``
files; hoisting them here keeps the test surface DRY without spreading
the same boilerplate to every future Phase 4+ integration test.

Why a fresh conftest (and not the existing
``tests/unit/gui/builder/conftest.py``)?  pytest only auto-discovers
``conftest.py`` modules in the test file's directory chain, and the
integration tests live in ``tests/integration/gui/builder/`` — a
sibling, not a descendant, of the unit-test directory.  A second
conftest at this layer is the canonical way to share fixtures across
``tests/integration/gui/builder/test_*.py``.
"""

from __future__ import annotations

from typing import Any, Iterable, List


def _walk(component: Any) -> Iterable[Any]:
    """Yield *component* and every descendant component recursively.

    Dash components carry their children on ``.children`` (a single
    component, a list, or a primitive); the walk descends only into
    component-shaped entries so primitives don't blow up the iteration.
    """

    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            if child is None or isinstance(child, (str, int, float, bool)):
                continue
            yield from _walk(child)
    elif not isinstance(children, (str, int, float, bool)):
        yield from _walk(children)


def _find_by_id(component: Any, target_id: Any) -> List[Any]:
    """Return every descendant whose ``id`` equals *target_id*.

    Handles both string-shaped ids (``id="my-store"``) and
    pattern-matching dict-shaped ids (``id={"type": "row", "index": 0}``).
    """

    hits: List[Any] = []
    for node in _walk(component):
        node_id = getattr(node, "id", None)
        if isinstance(target_id, dict):
            if isinstance(node_id, dict) and node_id == target_id:
                hits.append(node)
        elif node_id == target_id:
            hits.append(node)
    return hits


def _find_by_type_key(component: Any, type_key: str) -> List[Any]:
    """Return every descendant whose dict-shaped id carries ``type==type_key``."""

    hits: List[Any] = []
    for node in _walk(component):
        node_id = getattr(node, "id", None)
        if isinstance(node_id, dict) and node_id.get("type") == type_key:
            hits.append(node)
    return hits


def _collect_text(component: Any) -> str:
    """Flatten every string child / leaf into a single space-joined string."""

    parts: List[str] = []
    for node in _walk(component):
        children = getattr(node, "children", None)
        if isinstance(children, str):
            parts.append(children)
        elif isinstance(children, (list, tuple)):
            for child in children:
                if isinstance(child, str):
                    parts.append(child)
    return " ".join(parts)
