"""Pure cap/over-selection planning for the synced Compare strip (spec §7).

The Compare strip mounts one OpenSeadragon viewer per selected cell, each
holding a WebGL context; the browser ceiling (~16) forces a hard cap. This
module owns the **pure** cap/notice decision so it is unit-testable without a
browser, and so the over-cap notice text has a single source of truth that the
JS controller (``browse/_assets/timeline.js`` ``renderOverCapNotice``) mirrors
verbatim.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from phenotypic.gui._config import TIMELINE_COMPARE_CAP


@dataclass(frozen=True)
class ComparePlan:
    """The bounded plan for opening a synced Compare strip (spec §7).

    Attributes:
        shown: The refs that will be mounted (≤ ``cap``, in selection order).
        total: The full selection size (``len(refs)``).
        over_cap: ``True`` when ``total`` exceeded ``cap`` and ``shown`` was
            truncated to the first ``cap``.
        notice: The verbatim over-cap notice to display, or ``None``.
    """

    shown: tuple[object, ...]
    total: int
    over_cap: bool
    notice: str | None


def compare_selection_plan(
    refs: Sequence[object], *, cap: int = TIMELINE_COMPARE_CAP
) -> ComparePlan:
    """Bound a Compare-strip selection to ``cap`` viewers, never truncate silently.

    The synced Compare strip mounts one OSD viewer per ref, each holding a
    WebGL context; the browser ceiling (~16) forces a hard cap (spec §7/§12).
    When the selection exceeds ``cap`` this returns the first ``cap`` refs (by
    selection order) AND a visible notice so the user knows the rest were
    held back — never a silent drop.

    Args:
        refs: The selected cell refs, in selection order.
        cap: Maximum live viewers (defaults to :data:`TIMELINE_COMPARE_CAP`).

    Returns:
        A :class:`ComparePlan`. ``notice`` is non-``None`` only when over cap.
    """
    total = len(refs)
    if total <= cap:
        return ComparePlan(shown=tuple(refs), total=total, over_cap=False, notice=None)
    return ComparePlan(
        shown=tuple(refs[:cap]),
        total=total,
        over_cap=True,
        notice=f"Showing first {cap} of {total} — narrow the selection",
    )
