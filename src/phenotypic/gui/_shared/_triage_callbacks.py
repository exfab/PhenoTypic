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

from phenotypic.gui._shared._radial import (
    RADIAL_CUSTOM_FOLDER_SENTINEL,
    RADIAL_RESTORE_SENTINEL,
)
from phenotypic.gui._shared.tiles import expand_range
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._filtered_state import (
    decode_removed_keys_payload,
)

__all__ = [
    "coerce_anchor",
    "selection_payload",
    "fold_selection_delta",
    "category_dropdown_options",
    "bulk_mark",
    "register_custom_category_safe",
    "decode_wedge_trigger",
    "apply_wedge_mark",
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


# ---------------------------------------------------------------------------
# Bulk-mark / custom-category helpers (shared by colony + QC bulk bars and
# the radial ＋ Add-custom submit callbacks).
# ---------------------------------------------------------------------------


def category_dropdown_options(categories: list[str]) -> list[dict[str, str]]:
    """Build the bulk-mark dropdown options from a category vocabulary.

    Pure + module-level so the option shape is unit-testable without booting
    Dash, and so the colony and QC bulk bars share one rendering. Each token
    renders with a human-friendly label (underscores → spaces, title-cased)
    while the ``value`` stays the bare token the mark callback feeds to
    :meth:`CurationLabels.mark_many`.

    Args:
        categories: Ordered category tokens (core enum labels then custom),
            typically from :meth:`CurationLabels.categories`.

    Returns:
        A list of ``{"label", "value"}`` dicts for a ``dcc.Dropdown``.
    """
    return [
        {"label": token.replace("_", " ").title(), "value": token}
        for token in categories
    ]


def bulk_mark(
    filtered: CurationLabels,
    selected: list[tuple[str, int]],
    category: str,
) -> list[list]:
    """Mark every selected colony with ``category`` and return the new payload.

    Module-level (not a callback closure) so the
    :meth:`CurationLabels.mutate_and_payload` contract — the action receives
    the state instance — is unit-testable without booting Dash, and so the
    colony and QC bulk bars share one batched save via
    :meth:`CurationLabels.mark_many`.

    Args:
        filtered: The shared :class:`CurationLabels`.
        selected: The ``(image_file, label)`` keys to mark.
        category: The category token to assign to every selected key.

    Returns:
        The updated removed-keys payload.
    """
    return filtered.mutate_and_payload(
        lambda state: state.mark_many(selected, category)
    )


def register_custom_category_safe(
    filtered: CurationLabels, name: str | None
) -> tuple[str | None, str]:
    """Register a custom category, returning ``(token_or_None, message)``.

    Module-level + pure so the radial custom-add submit callbacks (colony and
    QC) share one validation path and it is unit-testable without booting
    Dash. Sanitizes + registers via
    :meth:`CurationLabels.register_custom_category`, catching its
    ``ValueError`` (empty name / collision with a core token) and turning it
    into an inline message.

    Args:
        filtered: The shared :class:`CurationLabels`.
        name: The user-entered category name (``None`` / blank rejected).

    Returns:
        ``(token, message)`` where ``token`` is the registered bare token on
        success (``message`` is a short confirmation) or ``None`` on failure
        (``message`` is the reason to surface inline).
    """
    if not name or not name.strip():
        return None, "Enter a category name."
    try:
        token = filtered.register_custom_category(name)
    except ValueError as exc:
        return None, str(exc)
    return token, f"Added “{token}”."


# ---------------------------------------------------------------------------
# Radial wedge-click dispatch (shared by the colony + QC mark callbacks).
# ---------------------------------------------------------------------------


def decode_wedge_trigger(
    triggered_id: Any, triggered_list: list[dict[str, Any]]
) -> tuple[str, int, str] | None:
    """Decode a radial-wedge ``ALL``-pattern fire into ``(image_file, label, category)``.

    The mark callbacks (colony + QC) bind an ``ALL`` pattern over every wedge's
    ``n_clicks``; this folds their shared decode + guards into one Dash-free
    helper so the two callbacks become thin adapters. Returns ``None`` for any
    case the callback must NOT act on (the caller maps ``None`` to its own
    no-op — ``PreventUpdate`` for colony, ``no_update`` for QC):

    * a non-dict / missing ``triggered_id`` (no concrete wedge);
    * the ``ALL`` pattern's initial all-empty-``n_clicks`` fire at mount;
    * a malformed ``image_file`` / ``label`` / ``category``;
    * the ``Custom ▸`` folder placeholder
      (:data:`RADIAL_CUSTOM_FOLDER_SENTINEL`), which opens the folder rather
      than marking a category.

    Args:
        triggered_id: ``callback_context.triggered_id`` (the concrete wedge id
            dict, or ``None`` on the initial fire).
        triggered_list: ``callback_context.triggered`` — the per-input
            ``{"prop_id", "value"}`` records, used to detect the all-empty
            initial fire.

    Returns:
        ``(image_file, label, category)`` ready for :func:`apply_wedge_mark`,
        or ``None`` when the callback should no-op. ``category`` may be
        :data:`RADIAL_RESTORE_SENTINEL` (the caller passes it straight to
        :func:`apply_wedge_mark`, which restores).
    """
    if not triggered_id or not isinstance(triggered_id, dict):
        return None
    # Skip the initial empty-n_clicks fire from the ALL pattern.
    if not any(entry.get("value") for entry in triggered_list):
        return None

    raw_image_file = triggered_id.get("image_file")
    raw_label = triggered_id.get("label")
    raw_category = triggered_id.get("category")
    if raw_image_file is None or raw_label is None or raw_category is None:
        return None
    try:
        image_file = str(raw_image_file)
        label = int(raw_label)
    except (TypeError, ValueError):
        return None
    category = str(raw_category)

    # The custom-folder placeholder opens the folder (Task 7); it is not a
    # real category, so never mark on it.
    if category == RADIAL_CUSTOM_FOLDER_SENTINEL:
        return None
    return image_file, label, category


def apply_wedge_mark(
    filtered: CurationLabels, image_file: str, label: int, category: str
) -> list[list]:
    """Mark (or restore) one colony from a decoded wedge click; return the payload.

    The mark/restore dispatch shared by the colony + QC mark callbacks: a
    ``category`` equal to :data:`RADIAL_RESTORE_SENTINEL` clears the label
    (restore); any other token assigns it (a durable categorized removal).
    Both go through :meth:`CurationLabels.mutate_and_payload` so the mutation
    + payload emission happen under one lock.

    Args:
        filtered: The shared :class:`CurationLabels`.
        image_file: ``Metadata_ImageFile`` of the colony.
        label: ``Object_Label`` of the colony.
        category: The decoded wedge category, or
            :data:`RADIAL_RESTORE_SENTINEL` to clear it.

    Returns:
        The updated removed-keys payload (a list of ``[image_file, label]``
        pairs). The caller wraps it in a 1-tuple for the ``allow_duplicate``
        ``STORE_REMOVED_KEYS`` output (a Dash multi-mode artifact, kept at the
        callback layer).
    """
    if category == RADIAL_RESTORE_SENTINEL:
        return filtered.mutate_and_payload(
            lambda s: s.unmark(image_file, label)
        )
    return filtered.mutate_and_payload(
        lambda s: s.mark(image_file, label, category)
    )
