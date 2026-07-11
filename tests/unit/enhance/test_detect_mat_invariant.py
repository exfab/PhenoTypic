"""CI gate for the ``detect_mat`` [0, 1] invariant across every public enhancer.

``detect_mat`` is contractually bounded to [0, 1]. Enhancers write it, so a single
misbehaving enhancer (historically :class:`FocusEdgeLaplace`, whose signed Laplacian
response reached roughly [-1.52, +1.48]) silently breaks that contract for every
downstream detector and measurement. This module pins the invariant so it cannot
rot: every enhancer exported from ``phenotypic.enhance.__all__`` that constructs with
no required arguments must leave ``detect_mat`` inside [0, 1] after ``apply()`` on the
synthetic yeast plate.

A second test asserts that :class:`FocusEdgeLaplace` (``norm="rescale"``) preserves the
*bipolar* structure of the Laplacian rather than collapsing its negative lobe: a
``norm="clip"`` implementation would satisfy the [0, 1] gate above yet fail here.
"""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import enhance
from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import FocusEdgeLaplace


def _zero_arg_enhancers() -> list[tuple[str, object]]:
    """Construct every enhancer in ``enhance.__all__`` with no arguments.

    Deliberately performs **no** ``try/except`` around ``cls()``: an enhancer that
    cannot be constructed zero-arg must blow up test collection loudly, not be
    silently skipped. A silent skip would report the gate green while covering
    nothing.
    """
    return [(name, getattr(enhance, name)()) for name in enhance.__all__]


_ENHANCERS = _zero_arg_enhancers()


@pytest.mark.parametrize(
    ("name", "op"), _ENHANCERS, ids=[name for name, _ in _ENHANCERS]
)
def test_enhancer_keeps_detect_mat_in_unit_range(name: str, op: object) -> None:
    """Every zero-arg enhancer must leave ``detect_mat`` within [0, 1]."""
    out = op.apply(load_synth_yeast_plate()).detect_mat[:]
    mn, mx = float(out.min()), float(out.max())
    assert mn >= 0.0, f"{name} emits {mn:.4f} < 0"
    assert mx <= 1.0, f"{name} emits {mx:.4f} > 1"


def test_gate_covers_every_public_enhancer() -> None:
    """The gate must exercise the full ``__all__`` roster, with nothing skipped."""
    assert [name for name, _ in _ENHANCERS] == list(enhance.__all__)
    assert len(_ENHANCERS) == len(enhance.__all__)


def test_focus_edge_laplace_preserves_bipolar_structure() -> None:
    """``norm="rescale"`` keeps the signed Laplacian's full bipolar range in [0, 1].

    The raw Laplacian is signed: a large negative lobe (dark-side edge response),
    a flat near-zero background, and a large positive lobe. ``rescale`` maps that
    whole span onto [0, 1], so:

    - the extremes reach ~0.0 and ~1.0 (the response's min and max), and
    - the near-zero background lands at the interior zero-crossing (~0.5), leaving
      both lobes populated on either side of 0.5.

    A ``norm="clip"`` implementation would collapse the entire negative lobe to
    exactly 0.0, dragging the background/median to 0.0 — it passes the [0, 1] gate
    but fails this test.
    """
    out = FocusEdgeLaplace().apply(load_synth_yeast_plate()).detect_mat[:]
    mn, mx, median = float(out.min()), float(out.max()), float(np.median(out))

    # Extremes reach the unit endpoints.
    assert mn == pytest.approx(0.0, abs=1e-6), f"min {mn:.4f} did not reach ~0.0"
    assert mx == pytest.approx(1.0, abs=1e-6), f"max {mx:.4f} did not reach ~1.0"

    # The signed zero-crossing (flat background) maps to the interior, ~0.5. Under
    # clip it would collapse to ~0.0. A generous window keeps the anchor honest
    # without pinning the exact value.
    assert 0.4 < median < 0.6, (
        f"background/median {median:.4f} not near 0.5 -- negative lobe collapsed "
        f"(clip-style), bipolar structure lost"
    )

    # Both lobes survive on either side of the 0.5 midpoint.
    assert bool((out < 0.5).any()), "negative lobe collapsed"
    assert bool((out > 0.5).any()), "positive lobe collapsed"
