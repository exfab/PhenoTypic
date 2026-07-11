"""Cross-channel fusion of monogenic phase congruency.

Three colour channels, each run through the *unmodified* monogenic chain of
:mod:`_monogenic_kernels`, then combined. Adds no signal theory: every fusion rule here
is arithmetic over accumulators that module already produces.

Only :func:`fuse_l2` has a reference -- Shi et al., "Color edge detection based on the
fusion of monogenic phase congruency and color morphology", *Multimed. Tools Appl.* 78,
10701--10716 (2019), whose rule is a root-sum-of-squares over per-channel congruencies.
:func:`fuse_joint` and :func:`fuse_coherent` are ours, recorded as drift ``C7`` and ``C8``.

References:
    Kovesi, P. "Image features from phase congruency." *Videre* 1(3), 1--26 (1999).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ._monogenic_kernels import (
    EPSILON_MONOGENIC,
    MonogenicChannel,
    congruency_from_accumulators,
)


@dataclass(frozen=True)
class ColorPhaseResult:
    """Output of :func:`color_phase_congruency`.

    Attributes:
        pc: Fused congruency, **un-clipped**. ``joint`` and ``coherent`` land in ``[0, 1]``;
            ``l2`` lands in ``[0, ||w||]``. The caller clips at the write site, because the
            PFOM regression must see the paper's actual quantity (drift ``C3``).
        orientation: Feature orientation in radians, ``(-pi/2, pi/2]``, from the fused
            vector ``sum_i w_i * v_i``. ``0`` is a vertical edge. **Not exposed by**
            :class:`FocusEdgeColorPhase` -- drift ``C15``. Only ``coherent`` builds this
            vector natively; under ``joint`` (which sums scalar energies) and ``l2`` (which
            sums finished congruency maps) it is *not* what produced ``pc``.
        feature_type: Local weighted mean phase angle, ``[-pi/2, pi/2]``, from the same
            fused vector. ``0`` is a step edge, ``+pi/2`` a bright line, ``-pi/2`` a dark one.
        threshold: ``T_total = sum_i w_i * T_i`` (drift ``C10``). Under ``l2`` each channel
            applied its **own** ``T_i``, so this value is then *informational only* and does
            not appear anywhere in ``pc``.
        n_clamped: Pixels whose ``acos`` argument needed clipping into ``[-1, 1]``. Under
            ``l2`` the congruency runs once per channel, so this is the **sum over the
            three channels** and counts pixel-channel incidences, not pixels. Must be
            ``0`` either way: with non-negative weights ``E_total <= A_total`` holds
            analytically for every mode. Drift ``M1``.
    """

    pc: np.ndarray
    orientation: np.ndarray
    feature_type: np.ndarray
    threshold: float
    n_clamped: int


def _weighted_threshold(
        channels: Sequence[MonogenicChannel], weights: np.ndarray
) -> float:
    """``T_total = sum_i w_i * T_i`` (drift ``C10``).

    Split out of :func:`_weighted_scalars` because ``fuse_l2`` wants *only* this scalar:
    it thresholds each channel with its own ``T_i``, and reports ``T_total`` for the
    record. Calling the full triple there built ``A_total`` and ``A_max_total`` -- two
    image-sized arrays -- purely to discard them.
    """
    return float(sum(w * c.threshold for w, c in zip(weights, channels)))


def _weighted_scalars(
        channels: Sequence[MonogenicChannel], weights: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """``(A_total, T_total, A_max_total)`` -- the three 1-homogeneous denominators.

    ``T_total`` and ``A_max_total`` are inventions with no reference (drift ``C10``). They
    follow from ``joint`` existing at all, and being 1-homogeneous in ``w`` they leave the
    two-degrees-of-freedom argument intact.
    """
    a_total = sum(w * c.sum_amplitude for w, c in zip(weights, channels))
    t_total = _weighted_threshold(channels, weights)
    a_max = sum(w * c.max_amplitude for w, c in zip(weights, channels))
    return a_total, t_total, a_max


def _fused_vector(
        channels: Sequence[MonogenicChannel], weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``sum_i w_i * (even_i, h1_i, h2_i)``. Coherent's numerator; every mode's angles.

    **Hazard.** The odd pair's sign encodes edge polarity, so a boundary whose lightness
    falls as its yellowness rises has ``v_L`` and ``v_b`` pointing opposite and the sum
    cancels. The angles read off this vector are least reliable exactly where colour is
    doing the most work. Drift ``C15``.
    """
    v_even = sum(w * c.sum_even for w, c in zip(weights, channels))
    v_h1 = sum(w * c.sum_h1 for w, c in zip(weights, channels))
    v_h2 = sum(w * c.sum_h2 for w, c in zip(weights, channels))
    return v_even, v_h1, v_h2


def fuse_joint(
        channels: Sequence[MonogenicChannel],
        weights: np.ndarray,
        *,
        n_scale: int,
        cutoff: float,
        g: float,
        deviation_gain: float,
        epsilon: float,
) -> tuple[np.ndarray, float, int]:
    """Shared denominator, L1 energy: ``E_total = sum_i w_i * ||v_i||``.

    A channel's amplitude enters the denominator whether or not its structure is coherent,
    so a loud but incoherent channel -- grain, sensor noise, speckle -- **vetoes** the other
    channels' edges. That veto is the only mechanism by which colour is expected to help
    here, and :func:`fuse_l2` has no analogue of it. Unvalidated against any external
    reference; drift ``C7``.

    Note:
        The numerator's norm must match the denominator's. An **L2** numerator
        ``sqrt(sum_i (w_i E_i)**2)`` over this **L1** denominator annihilates a coherent
        three-channel edge exactly (response ``0.0`` at ``deviation_gain=1.5``) while a
        single-channel response passes at full strength -- inverting the very artefact the
        colour design set out to attack. ``fusion_algebra.py`` check 01.

    Args:
        channels: Three :class:`MonogenicChannel`, luminance first.
        weights: ``(3,)`` non-negative.
        n_scale: Number of scales the channels were built with.
        cutoff: Frequency-spread sigmoid centre.
        g: Frequency-spread sigmoid sharpness.
        deviation_gain: Scales the phase-deviation term.
        epsilon: Division guard.

    Returns:
        ``(pc, threshold, n_clamped)`` with ``pc`` un-clipped.
    """
    energy = sum(w * c.energy for w, c in zip(weights, channels))
    a_total, t_total, a_max = _weighted_scalars(channels, weights)
    pc, n_clamped = congruency_from_accumulators(
            energy, a_total, a_max, t_total, n_scale=n_scale, cutoff=cutoff, g=g,
            deviation_gain=deviation_gain, epsilon=epsilon,
    )
    return pc, t_total, n_clamped


def fuse_coherent(
        channels: Sequence[MonogenicChannel],
        weights: np.ndarray,
        *,
        n_scale: int,
        cutoff: float,
        g: float,
        deviation_gain: float,
        epsilon: float,
) -> tuple[np.ndarray, float, int]:
    """As :func:`fuse_joint`, but ``E_total = ||sum_i w_i * v_i||``.

    Cancels opposite-phase responses across channels. **This annihilates a genuine
    anti-correlated chromatic edge**, where lightness falls as yellowness rises -- not only
    the incoherent noise it is meant to suppress. Opt-in, never the default. No reference;
    drift ``C8``.

    Args:
        channels: Three :class:`MonogenicChannel`, luminance first.
        weights: ``(3,)`` non-negative.
        n_scale: Number of scales the channels were built with.
        cutoff: Frequency-spread sigmoid centre.
        g: Frequency-spread sigmoid sharpness.
        deviation_gain: Scales the phase-deviation term.
        epsilon: Division guard.

    Returns:
        ``(pc, threshold, n_clamped)`` with ``pc`` un-clipped.
    """
    v_even, v_h1, v_h2 = _fused_vector(channels, weights)
    # sqrt(a**2 + b**2 + c**2), never np.hypot: `hypot` appears in no reference and rounds
    # differently on ~21% of elements (21.4% on load_synth_yeast_plate's L*). Same
    # substitution the golden fixture cannot see. NB 4.5% is the *two*-component
    # feature_type figure and does not apply to this three-component norm.
    energy = np.sqrt(v_even ** 2 + v_h1 ** 2 + v_h2 ** 2)
    a_total, t_total, a_max = _weighted_scalars(channels, weights)
    pc, n_clamped = congruency_from_accumulators(
            energy, a_total, a_max, t_total, n_scale=n_scale, cutoff=cutoff, g=g,
            deviation_gain=deviation_gain, epsilon=epsilon,
    )
    return pc, t_total, n_clamped


def fuse_l2(
        channels: Sequence[MonogenicChannel],
        weights: np.ndarray,
        *,
        n_scale: int,
        cutoff: float,
        g: float,
        deviation_gain: float,
        epsilon: float,
) -> tuple[np.ndarray, float, int]:
    """CMPCM's rule: ``out = sqrt(sum_i (w_i * F_i)**2)`` over per-channel congruencies.

    Three independent detectors combined after the fact. **No cross-channel term reaches any
    denominator**, so incoherent chroma amplitude can never veto a spurious luminance edge.
    That absence is the whole reason :func:`fuse_joint` is the default.

    **Not divided by ``||w||``** -- the paper does not, and the PFOM regression must check the
    paper's actual quantity. Range ``[0, ||w||]``; the caller clips. Drift ``C3``.

    The returned threshold is ``sum_i w_i T_i`` for reporting only. Each channel has already
    applied its own ``T_i`` inside its own congruency; no fused threshold exists here.

    Args:
        channels: Three :class:`MonogenicChannel`, luminance first.
        weights: ``(3,)`` non-negative.
        n_scale: Number of scales the channels were built with.
        cutoff: Frequency-spread sigmoid centre.
        g: Frequency-spread sigmoid sharpness.
        deviation_gain: Scales the phase-deviation term.
        epsilon: Division guard.

    Returns:
        ``(pc, threshold, n_clamped)`` with ``pc`` un-clipped and possibly above ``1``.
    """
    total = np.zeros_like(channels[0].sum_amplitude)
    n_clamped = 0
    for w, channel in zip(weights, channels):
        pc_i, clamped_i = congruency_from_accumulators(
                channel.energy, channel.sum_amplitude, channel.max_amplitude,
                channel.threshold, n_scale=n_scale, cutoff=cutoff, g=g,
                deviation_gain=deviation_gain, epsilon=epsilon,
        )
        total = total + (w * pc_i) ** 2
        n_clamped += clamped_i

    # Only the scalar T_total is wanted here: no fused denominator exists under `l2`.
    t_total = _weighted_threshold(channels, weights)
    return np.sqrt(total), t_total, n_clamped


_DISPATCH: dict[str, Callable[..., tuple[np.ndarray, float, int]]] = {
    "joint": fuse_joint,
    "coherent": fuse_coherent,
    "l2": fuse_l2,
}

#: The three fusion rules -- ``("joint", "coherent", "l2")``. ``l2`` is CMPCM's; the other
#: two are ours. **Derived from** :data:`_DISPATCH` rather than spelled again, so the name
#: a caller is told to pass and the name that actually dispatches cannot diverge: a fourth
#: entry in the dict is a fourth entry here, and the rejection message below stays true.
#: ``sdk_.typing_.PhaseFusion`` is the third copy of these names and is pinned against this
#: tuple by ``test_focus_edge_color_phase.py``.
FUSIONS: tuple[str, ...] = tuple(_DISPATCH)


def color_phase_congruency(
        channels: Sequence[MonogenicChannel],
        weights: np.ndarray,
        *,
        fusion: str,
        n_scale: int,
        cutoff: float = 0.5,
        g: float = 10.0,
        deviation_gain: float = 1.5,
        epsilon: float = EPSILON_MONOGENIC,
) -> ColorPhaseResult:
    """Fuse three channels' monogenic accumulators into one congruency map.

    Args:
        channels: Exactly three :class:`MonogenicChannel`, in **luminance-first** order --
            ``lab`` gives ``(L*, a*, b*)`` and ``hsv`` gives ``(V, H, S)``. All three must
            share a shape.
        weights: ``(3,)`` finite and non-negative. ``weights[0]`` is pinned to ``1.0`` by
            the operation; the two chromatic axes carry the two real degrees of freedom.
        fusion: One of :data:`FUSIONS`.
        n_scale: Must match the ``n_scale`` used to build ``channels``. Passed through to
            ``spread_weight``'s ``n_scale - 1`` divisor. ``spread_weight`` itself does not
            validate; :func:`congruency_from_accumulators` raises below 2 (drift ``M9``).
        cutoff: Frequency-spread sigmoid centre.
        g: Frequency-spread sigmoid sharpness.
        deviation_gain: Scales the phase-deviation term.
        epsilon: Division guard, ``1e-4``.

    Returns:
        A :class:`ColorPhaseResult` whose ``pc`` is **un-clipped**.

    Raises:
        ValueError: If ``fusion`` is unknown; if ``channels`` or ``weights`` is not length
            3; if the channels' shapes disagree; if any weight is negative or non-finite; or
            (from :func:`congruency_from_accumulators`) if ``n_scale < 2``.

    Note:
        **Non-negative weights are load-bearing, not cosmetic.** ``n_clamped == 0`` rests on
        ``E_total <= A_total``, which follows from ``||sum_s v_is|| <= sum_s ||v_is||`` per
        channel and ``||sum_i w_i v_i|| <= sum_i w_i ||v_i||`` across them -- both of which
        need ``w_i >= 0``. A negative weight can push the ``acos`` argument outside
        ``[-1, 1]``, where drift ``M1``'s clamp -- written to absorb roundoff, and measured
        to be inert on every shipped image -- would silently start doing real work.
    """
    if fusion not in _DISPATCH:
        raise ValueError(f"fusion must be one of {FUSIONS}; got {fusion!r}.")

    weights = np.asarray(weights, dtype=np.float64)
    if len(channels) != 3 or weights.shape != (3,):
        raise ValueError(
                f"expected 3 channels and a (3,) weight vector; got {len(channels)} "
                f"channels and weights of shape {weights.shape}."
        )
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError(
                f"weights must be finite and non-negative; got {weights!r}. The "
                f"n_clamped == 0 invariant rests on E_total <= A_total, which requires it."
        )

    shapes = {c.sum_amplitude.shape for c in channels}
    if len(shapes) != 1:
        raise ValueError(f"all three channels must share a shape; got {sorted(shapes)}.")

    pc, threshold, n_clamped = _DISPATCH[fusion](
            channels, weights, n_scale=n_scale, cutoff=cutoff, g=g,
            deviation_gain=deviation_gain, epsilon=epsilon,
    )

    # Angles always come from the fused vector, in every mode. Under `joint` and `l2` that
    # vector is not what produced `pc`. They are returned, never exposed. Drift C15.
    v_even, v_h1, v_h2 = _fused_vector(channels, weights)
    orientation = np.arctan2(-v_h2, v_h1)
    orientation = np.where(orientation > np.pi / 2, orientation - np.pi, orientation)
    orientation = np.where(orientation <= -np.pi / 2, orientation + np.pi, orientation)
    feature_type = np.arctan2(v_even, np.sqrt(v_h1 ** 2 + v_h2 ** 2))

    return ColorPhaseResult(pc, orientation, feature_type, threshold, n_clamped)
