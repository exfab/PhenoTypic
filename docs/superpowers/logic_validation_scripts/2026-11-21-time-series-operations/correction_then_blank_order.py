"""When does "flat-field correct, then subtract the blank" equal the reverse order?

The time-series feasibility report (docs/superpowers/reports/2026-11-21-time-series-
operations/feasibility.md, section 10) chains a collection-fitted shading correction
(BaSiC-style) before ``SubtractBlank``. ``SubtractBlank`` refuses to follow any
correction today, because its blank is read raw. This script checks the tempting
shortcut, "subtract the raw blank first, then divide by the flat-field", against the
exact route, "correct both frames identically, then subtract", for a per-pixel affine
correction

    C_i(x) = (M_i(x) - D(x)) / S(x) - beta_i

with flat-field S > 0, dark-field D and a per-frame scalar baseline beta_i.

Claims re-derived here (each is asserted, and the script exits non-zero if one fails):

1. Single-channel (or channel-uniform S), linear detect mode, no baseline drift, and the
   ``brighter`` polarity with SubtractBlank's [0, 1] clip: the two orders agree to
   float precision once the shortcut's result is clipped again. The dark-field D
   cancels, so it never needs to be known.
2. Baseline drift (beta_t != beta_0) breaks that equality, because the shortcut clips
   before the drift term is removed.
3. A per-channel flat-field (S differs between R, G, B) breaks it under a weighted-gray
   projection when the colony's signal differs between channels (a pigmented colony):
   the shortcut then weights each channel's growth by its own S. For achromatic growth
   the weighting cancels exactly, which an earlier draft of this check got wrong; the
   ``achromatic`` case below pins that.
4. A nonlinear projection (a cube-root lightness, as CIE L* uses) breaks it even with
   a channel-uniform S and no drift.
5. The ``signed`` polarity (0.5 + target - blank) breaks it, because dividing by S also
   scales the 0.5 offset.

Conclusion the report draws: the shortcut is exact only in case 1. In every other case
the blank must receive the same correction as the target before subtraction.

Synthetic data only, fixed seed. Depends on the standard library and numpy; never
imports ``phenotypic``. Run: ``uv run --no-project --python <python with numpy> python
-I correction_then_blank_order.py``.
"""

from __future__ import annotations

import sys

import numpy as np

RNG_SEED = 20261121
SHAPE = (64, 64)
#: Any positive weights summing to 1 make the structural point; these are the
#: Rec. 709 luma weights, used here only as a plausible linear channel mix.
GRAY_WEIGHTS = np.array([0.2125, 0.7154, 0.0721])
TOLERANCE = 1e-6


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def _correct(raw: np.ndarray, flat: np.ndarray, dark: np.ndarray, beta: float) -> np.ndarray:
    return (raw - dark) / flat - beta


def _subtract(target: np.ndarray, blank: np.ndarray, polarity: str) -> np.ndarray:
    if polarity == "brighter":
        return _clip01(target - blank)
    if polarity == "signed":
        return _clip01(0.5 + (target - blank))
    raise ValueError(polarity)


def _scene(
    rng: np.random.Generator, channels: int, chromatic: bool
) -> tuple[np.ndarray, np.ndarray]:
    """A blank (agar only) and a later frame with colony growth, both true signal."""
    shape = SHAPE + ((channels,) if channels > 1 else ())
    agar = 0.30 + 0.02 * rng.standard_normal(shape)
    growth = np.zeros(shape)
    #: Pigmented growth: unequal per-channel signal (one value when single-channel).
    colour = 0.35 if channels == 1 else (np.full(channels, 0.35) if not chromatic
                                         else np.array([0.35, 0.25, 0.10]))
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]]
    for cy, cx in ((16, 16), (40, 44), (50, 12)):
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 < 49
        growth[disk] = colour
    return agar, agar + growth


def _shading(rng: np.random.Generator, channels: int, per_channel: bool) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[: SHAPE[0], : SHAPE[1]] / SHAPE[0]
    vignette = 1.0 - 0.35 * ((yy - 0.5) ** 2 + (xx - 0.5) ** 2)
    if channels == 1:
        return vignette, np.full(SHAPE, 0.02)
    if per_channel:
        flat = np.stack([vignette * k for k in (1.00, 0.90, 0.80)], axis=-1)
    else:
        flat = np.repeat(vignette[..., None], channels, axis=-1)
    return flat, np.full(SHAPE + (channels,), 0.02)


def _measure(true: np.ndarray, flat: np.ndarray, dark: np.ndarray, offset: float) -> np.ndarray:
    return true * flat + dark + offset


def _project(values: np.ndarray, mode: str) -> np.ndarray:
    if values.ndim == 2:
        return values
    gray = values @ GRAY_WEIGHTS
    if mode == "gray":
        return gray
    if mode == "lightness":
        return np.cbrt(np.clip(gray, 0.0, None))
    raise ValueError(mode)


def _max_gap(
    *, channels: int, per_channel: bool, mode: str, drift: float, polarity: str,
    chromatic: bool = True,
) -> float:
    rng = np.random.default_rng(RNG_SEED)
    blank_true, target_true = _scene(rng, channels, chromatic)
    flat, dark = _shading(rng, channels, per_channel)
    blank_raw = _measure(blank_true, flat, dark, 0.0)
    target_raw = _measure(target_true, flat, dark, drift * np.mean(flat))
    beta_blank, beta_target = 0.0, drift

    exact = _subtract(
        _project(_correct(target_raw, flat, dark, beta_target), mode),
        _project(_correct(blank_raw, flat, dark, beta_blank), mode),
        polarity,
    )
    flat_projected = _project(flat, "gray") if flat.ndim == 3 else flat
    shortcut = _clip01(
        _subtract(_project(target_raw, mode), _project(blank_raw, mode), polarity) / flat_projected
    )
    return float(np.max(np.abs(exact - shortcut)))


def check_order_claims() -> list[str]:
    failures: list[str] = []
    base = {"per_channel": False, "mode": "gray", "drift": 0.0, "polarity": "brighter"}
    cases = {
        "1 equal: 1 channel, linear, no drift, brighter": {**base, "channels": 1},
        "1 equal: RGB, uniform S, gray, no drift, brighter": {**base, "channels": 3},
        "2 differ: baseline drift": {**base, "channels": 1, "drift": 0.05},
        "3 differ: per-channel S under gray, pigmented": {
            **base, "channels": 3, "per_channel": True},
        "1 equal: per-channel S under gray, achromatic": {
            **base, "channels": 3, "per_channel": True, "chromatic": False},
        "4 differ: nonlinear lightness": {**base, "channels": 3, "mode": "lightness"},
        "5 differ: signed polarity": {**base, "channels": 1, "polarity": "signed"},
    }
    for label, kwargs in cases.items():
        gap = _max_gap(**kwargs)
        expect_equal = label.startswith("1 ")
        verdict = "equal" if gap <= TOLERANCE else "differ"
        print(f"{label:<52} max |exact - shortcut| = {gap:.3e}  -> {verdict}")
        if expect_equal and gap > TOLERANCE:
            failures.append(f"{label}: expected equality, gap {gap:.3e}")
        if not expect_equal and gap <= TOLERANCE:
            failures.append(f"{label}: expected a difference, gap {gap:.3e}")
    return failures


def run_all_checks() -> int:
    failures = check_order_claims()
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    print("OK" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run_all_checks())
