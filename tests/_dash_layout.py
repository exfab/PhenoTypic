"""Shared walkers over a built Dash layout and its registered callbacks.

Two phase-5 guards need the same non-obvious thing: the set of string ids a
built layout mounts, checked against the ids the registered callbacks write
to. The ``callback_map`` key parsing below is the part worth having in one
place -- it decodes an undocumented Dash-internal key format, and two copies
of it would drift the moment Dash changes that format, with only one of the
two tests updated.
"""

from __future__ import annotations

from typing import Any, Iterator


def walk_components(node: Any) -> Iterator[Any]:
    """Yield every component in a built layout tree."""
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from walk_components(child)
    elif children is not None:
        yield from walk_components(children)


def mounted_string_ids(app: Any) -> set[str]:
    """Return every string ``id`` the app's built layout carries."""
    layout = app.layout() if callable(app.layout) else app.layout
    return {
        node.id
        for node in walk_components(layout)
        if isinstance(getattr(node, "id", None), str)
    }


def dangling_callback_outputs(app: Any) -> set[tuple[str, str]]:
    """Callback Outputs naming a string id the built layout does not mount.

    Pattern-matching dict ids are excluded on purpose: a
    ``{"type": ..., "index": ALL}`` id legitimately matches zero components at
    layout time because the components it targets are created by other
    callbacks. Only string ids name a component the layout is expected to
    carry, so asserting on dict ids would false-positive on every wildcard
    callback.
    """
    mounted = mounted_string_ids(app)

    dangling: set[tuple[str, str]] = set()
    for key in app.callback_map:
        # The callback_map key encodes its outputs as
        # ``..<id>.<prop>...<id>.<prop>..`` with an optional ``@<hash>``
        # suffix on ``allow_duplicate`` outputs.
        for segment in key.strip(".").split("..."):
            segment = segment.strip(".").split("@", 1)[0]
            if "." not in segment or segment.startswith("{"):
                continue  # no property, or a pattern-matching dict id
            component_id, prop = segment.rsplit(".", 1)
            if component_id.startswith("{"):
                continue  # pattern-matching id -- may match zero components
            if component_id not in mounted:
                dangling.add((component_id, prop))
    return dangling


DANGLING_OUTPUT_MESSAGE = (
    "these callback Outputs name components the layout does not mount, so "
    "dash-renderer will throw and DISCARD the whole callback response "
    "(taking any co-outputs with it): "
)
