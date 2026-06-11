"""Pure callback-layer helpers shared by the colony and QC triage surfaces.

These are the Dash-free building blocks the colony-view and QC-review
callbacks both call so the two surfaces stay single-sourced. Keeping them
here (rather than re-importing across the two callback modules) means the
selection-fold + custom-add + bulk-mark semantics are unit-testable without
booting Dash, and a fix to one surface can never silently diverge from the
other.

The colony grid is the working template; the QC review gallery mirrors it
(spec decision C: within one tab the user selects on a single surface at a
time, so a SHARED selection store is fine — only the *order* store and the
*delta* store differ per surface).
"""

from __future__ import annotations

from typing import Any

from phenotypic.gui._shared.tiles import expand_range
from phenotypic.gui.results_viewer._filtered_state import (
    decode_removed_keys_payload,
)

__all__ = [
    "coerce_anchor",
    "selection_payload",
    "fold_selection_delta",
]


def coerce_anchor(payload: Any) -> tuple[str, int] | None:
    """Coerce a single ``[img, label]`` payload into a tuple, or ``None``.

    Args:
        payload: A 2-element ``[image_file, label]`` list/tuple (or anything
            else, which yields ``None``).

    Returns:
        ``(image_file, label)`` on success, ``None`` on a malformed payload.
    """
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        return None
    try:
        return (str(payload[0]), int(payload[1]))
    except (TypeError, ValueError):
        return None


def selection_payload(
    anchor: tuple[str, int] | None, selected: list[tuple[str, int]]
) -> dict[str, Any]:
    """Build the canonical selection-store payload (lists, not tuples).

    Args:
        anchor: The shift-range anchor key, or ``None``.
        selected: The currently-selected keys.

    Returns:
        ``{"anchor": [img, label] | None, "selected": list[[img, label]]}``,
        JSON-serialisable for a ``dcc.Store``.
    """
    return {
        "anchor": [anchor[0], anchor[1]] if anchor is not None else None,
        "selected": [[img, label] for img, label in selected],
    }


def fold_selection_delta(
    delta: Any,
    current_selection: Any,
    grid_order_payload: Any,
) -> dict[str, Any] | None:
    """Fold a JS-emitted click delta into the canonical selection payload.

    The pure core shared by the colony and QC selection-delta consumers
    (M1: QC-review selection parity). Both surfaces emit the SAME delta
    shape and write the SAME :data:`STORE_COLONY_SELECTION`; they differ
    only in which *order* store resolves shift-ranges.

    Delta shape: ``{"key": [image_file, label], "shift": bool, "ts": int}``.

    * Single toggle (no shift / no anchor): add or remove the key, set the
      anchor to the just-clicked key when it lands in the selection (else
      clear the anchor).
    * Shift+click with an anchor: union the current selection with the
      inclusive slice of ``grid_order`` between the anchor and the delta
      key, preserving the anchor. If either endpoint slipped out of the
      current grid (filter/axis changed since the anchor was captured),
      fall back to a single-toggle and drop the stale anchor.

    Args:
        delta: The JS-emitted delta payload.
        current_selection: The current ``STORE_COLONY_SELECTION`` value.
        grid_order_payload: The surface's row-major order store value (the
            colony grid order or the QC gallery order).

    Returns:
        The new selection payload, or ``None`` when the click is a no-op
        (malformed delta, or a same-value re-emission) — the caller maps
        ``None`` to ``dash.no_update`` so a no-op click doesn't re-fire the
        downstream visibility + render callbacks.
    """
    if not isinstance(delta, dict):
        return None
    delta_key = coerce_anchor(delta.get("key"))
    if delta_key is None:
        return None

    current = current_selection if isinstance(current_selection, dict) else {}
    current_anchor = coerce_anchor(current.get("anchor"))
    original_selected = set(decode_removed_keys_payload(current.get("selected")))
    grid_order = decode_removed_keys_payload(grid_order_payload)

    shift = bool(delta.get("shift"))

    if shift and current_anchor is not None:
        try:
            slice_keys = expand_range(grid_order, current_anchor, delta_key)
        except ValueError:
            # Anchor or target slipped out of the current grid (e.g. the
            # filter or axis changed since the anchor was captured). Fall
            # back to a single-toggle on the delta key, dropping the stale
            # anchor.
            working = set(original_selected)
            if delta_key in working:
                working.discard(delta_key)
                new_anchor: tuple[str, int] | None = None
            else:
                working.add(delta_key)
                new_anchor = delta_key
            new_selected = working
        else:
            new_selected = original_selected | set(slice_keys)
            new_anchor = current_anchor
    else:
        working = set(original_selected)
        if delta_key in working:
            working.discard(delta_key)
            new_anchor = None
        else:
            working.add(delta_key)
            new_anchor = delta_key
        new_selected = working

    # Suppress same-value re-emissions so a click that doesn't change the
    # selection (e.g. shift-click within an already-fully-covered range)
    # doesn't re-fire the downstream visibility + render callbacks.
    if new_anchor == current_anchor and new_selected == original_selected:
        return None
    return selection_payload(new_anchor, sorted(new_selected))
