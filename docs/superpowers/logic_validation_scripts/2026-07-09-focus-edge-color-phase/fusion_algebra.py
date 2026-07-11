#!/usr/bin/env python3
"""Re-derive `color-phase-congruency.md`'s load-bearing numeric claims from scratch.

Depends only on the stdlib and numpy. Never imports `phenotypic`: the point is to check
the *spec*, not the implementation of it. Exits non-zero if any claim has stopped being
true.

Run:  uv run --no-project --with numpy python fusion_algebra.py
"""
import sys
import numpy as np

EPS = 1e-4
DEVIATION_GAIN = 1.5
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def _response(ratio: float, dg: float = DEVIATION_GAIN) -> float:
    return max(1.0 - dg * float(np.arccos(np.clip(ratio, -1.0, 1.0))), 0.0)


def check_01_l2_over_l1_annihilates_a_coherent_edge() -> None:
    """§3.1. An L2 numerator over an L1 denominator inverts the CA acceptance criterion.

    The expected responses are given to full precision, NOT to the four decimal places
    the spec table prints. `d(response)/d(ratio) = deviation_gain/sqrt(1 - ratio**2)`,
    which is `2.6520` at the third row's ratio -- so the printed 4-dp ratio `0.8247`,
    itself rounded by `3.4e-05`, moves the response by `9.0e-05`. Round-tripping the
    printed ratio back through the formula yields `0.0983`; the true value is `0.0982`.

    That is exactly how a wrong number entered this spec (see the retracted-claims table
    in `drift-register.md`). A four-decimal intermediate cannot determine a
    four-decimal result, and the tolerance here is tight enough to say so.
    """
    rows = [
        # label,       weights,                        firing,               ratio, response
        ("one only", np.ones(3), np.array([1.0, 0.0, 0.0]), 1.0, 1.0),
        ("all three", np.ones(3), np.ones(3), 0.5773502691896258, 0.0),
        ("real prior", np.array([0.804, 0.013, 0.183]), np.ones(3),
         0.824665992993527, 0.0982226557669601),
    ]
    ok = True
    for label, w, fires, want_ratio, want_resp in rows:
        e = a = fires
        ratio = float(np.sqrt(np.sum((w * e) ** 2)) / np.sum(w * a))
        resp = _response(ratio)
        ok &= abs(ratio - want_ratio) < 1e-12 and abs(resp - want_resp) < 1e-12
        print(f"      {label:12s} ratio={ratio:.6f} response={resp:.6f}")
        # The L1/L1 form must pass every row at full strength.
        ok &= abs(_response(float(np.sum(w * e) / np.sum(w * a))) - 1.0) < 1e-12
    # The load-bearing claim: the annihilation is EXACT, not merely small.
    ok &= _response(float(np.sqrt(3.0)) / 3.0) == 0.0
    check("01 l2-over-l1 annihilates a coherent edge", ok,
          "single-channel 1.000000, three-channel exactly 0.0 at deviation_gain=1.5")


def check_02_no_single_deviation_gain_reproduces_the_old_table() -> None:
    """The retracted §3.1 printed 0.0091 and 0.1425. Show they are mutually inconsistent."""
    dg_a = (1 - 0.0091) / float(np.arccos(0.5774))
    dg_b = (1 - 0.1425) / float(np.arccos(0.8247))
    check("02 the retracted response column is self-inconsistent",
          abs(dg_a - dg_b) > 0.3,
          f"row 2 needs dg={dg_a:.4f}, row 3 needs dg={dg_b:.4f}; shipped dg={DEVIATION_GAIN}")


def check_03_energy_never_exceeds_amplitude() -> None:
    """The `acos` argument stays in [-1, 1] for joint AND coherent. So n_clamped == 0.

    The bound is analytic, not empirical: `E_joint = sum_i w_i*||v_i|| <= sum_i w_i*A_i`
    because `||sum_s v_is|| <= sum_s ||v_is||` per channel, and
    `E_coherent = ||sum_i w_i*v_i|| <= sum_i w_i*||v_i|| = E_joint`. The draw only
    confirms it and reports how close the sampled maximum gets.

    The reported maxima are **sampled**, hence seed-dependent. Cite them with the seed
    or not at all.
    """
    rng = np.random.default_rng(20260709)
    worst_j = worst_c = -np.inf
    for _ in range(200_000):
        n_scale = int(rng.integers(2, 7))
        v = rng.normal(size=(3, n_scale, 3))
        w = np.array([1.0, rng.uniform(0, 8), rng.uniform(0, 8)])
        a = np.linalg.norm(v, axis=2).sum(axis=1)
        s = v.sum(axis=1)
        e = np.linalg.norm(s, axis=1)
        a_total = float((w * a).sum())
        worst_j = max(worst_j, float((w * e).sum()) / (a_total + EPS))
        worst_c = max(worst_c, float(np.linalg.norm((w[:, None] * s).sum(axis=0))) / (a_total + EPS))
    check("03 E_total <= A_total for joint and coherent", worst_j <= 1.0 and worst_c <= 1.0,
          f"max joint {worst_j:.6f}, max coherent {worst_c:.6f} "
          f"over 200000 draws at seed 20260709")


def check_04_epsilon_breaks_scale_invariance_through_the_sigmoid() -> None:
    """§4.2 / drift C17. The old '~1%' was wrong by two orders. Name the real culprit."""
    def joint(w, e, a, t, a_max, n_scale=4, cutoff=0.5, g=10.0):
        e_t, a_t = float((w * e).sum()), float((w * a).sum())
        t_t, m_t = float((w * t).sum()), float((w * a_max).sum())
        width = (a_t / (m_t + EPS) - 1.0) / (n_scale - 1)
        weight = 1.0 / (1.0 + np.exp(g * (cutoff - width)))
        return (weight * _response(e_t / (a_t + EPS))
                * max(e_t - t_t, 0.0) / (e_t + EPS))

    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(20_000):
        a = rng.uniform(0.5, 5.0, 3)
        e, t, a_max = a * rng.uniform(0.5, 1, 3), a * rng.uniform(0, 0.3, 3), a * rng.uniform(0.3, 0.9, 3)
        w = np.array([1.0, rng.uniform(0, 8), rng.uniform(0, 8)])
        base = joint(w, e, a, t, a_max)
        if base <= 1e-6:
            continue
        for c in (0.01, 100.0):
            worst = max(worst, abs(joint(c * w, e, a, t, a_max) - base) / base)
    check("04 eps breaks 1-homogeneity through A_max + eps, not E + eps", worst > 0.5,
          f"max relative change {worst * 100:.1f}% over c in [0.01, 100] at Lab L* amplitudes "
          f"-- the retracted claim was '~1%'")


if __name__ == "__main__":
    check_01_l2_over_l1_annihilates_a_coherent_edge()
    check_02_no_single_deviation_gain_reproduces_the_old_table()
    check_03_energy_never_exceeds_amplitude()
    check_04_epsilon_breaks_scale_invariance_through_the_sigmoid()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("4/4 checks passed.")
