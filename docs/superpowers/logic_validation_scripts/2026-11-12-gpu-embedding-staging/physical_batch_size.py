"""Re-derive the Stage-4 physical batch size rule (#282 spec 4.4, plan P0 T1).

Stdlib only; never imports ``phenotypic``. Exits non-zero on a failed check.

``m = min(batch_size, pow2floor((MEMORY_FRACTION * total - weights - RESERVE) / per_sample))``
with ``per_sample = dtype_bytes * (ACTIVATION_FACTOR * T * hidden + heads * T**2)``
and ``T = (crop_px // patch)**2 + 1 + registers``.

Assumptions (plan P0 T1, all stand):
- Model rows for dinov2 small/base/large and dinov3 small/base/large are filled in
  from memory (the plan gave tuples only for giant, huge_plus and 7b):
  UNVERIFIED UNTIL P2 T1 checks each row against the hub's config.json.
- Stdlib only (the plan allows numpy; none is needed).
- Memory is host-independent: total GPU bytes only, GiB = 1 << 30.
- Float32 crops are assumed in the sibling stage4_memory.py; 48 GiB means --mem=48G.

P2 T2 copies MEMORY_FRACTION, RESERVE_BYTES and ACTIVATION_FACTOR verbatim into
``src/``. P2 T1 verifies every model row below against the hub's config.json and
re-runs this script.
"""
from __future__ import annotations

import sys

MEMORY_FRACTION = 0.8
RESERVE_BYTES = 1 << 30
ACTIVATION_FACTOR = 12
BATCH_SIZE = 64
GIB = 1 << 30
# The spec's 0.8 stated again as its own literal, so the guard below does not
# reuse (and cannot be fooled by) the formula's MEMORY_FRACTION.
SPEC_MEMORY_FRACTION = 0.8

DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2}
CROPS = (112, 224, 336, 448)

# name: (params, hidden, heads, patch, registers)
MODELS = {
    "dinov2-small": (2.2e7, 384, 6, 14, 0),
    "dinov2-base": (8.6e7, 768, 12, 14, 0),
    "dinov2-large": (3.0e8, 1024, 16, 14, 0),
    "dinov2-giant": (1.1e9, 1536, 24, 14, 0),
    "dinov3-small": (2.1e7, 384, 6, 16, 4),
    "dinov3-base": (8.6e7, 768, 12, 16, 4),
    "dinov3-large": (3.0e8, 1024, 16, 16, 4),
    "dinov3-huge_plus": (8.4e8, 1280, 20, 16, 4),
    "dinov3-7b": (6.7e9, 4096, 32, 16, 4),
}

# name: total bytes
CLUSTER_GPUS = {
    "ada6000-48": 48 * GIB,
    "a100-40": 40 * GIB,  # not in the 48 GiB-and-up set; kept for monotonicity
    "a100-80": 80 * GIB,
    "h100-80": 80 * GIB,
    "blackwell6000-96": 96 * GIB,
}
GPUS = {**CLUSTER_GPUS, "p100-16": 16 * GIB}
BIG_GPUS = ("ada6000-48", "a100-80", "h100-80", "blackwell6000-96")


def tokens(crop_px: int, patch: int, registers: int) -> int:
    return (crop_px // patch) ** 2 + 1 + registers


def per_sample_bytes(model: str, crop_px: int, dtype: str) -> int:
    _, hidden, heads, patch, regs = MODELS[model]
    t = tokens(crop_px, patch, regs)
    return DTYPE_BYTES[dtype] * (ACTIVATION_FACTOR * t * hidden + heads * t * t)


def weights_bytes(model: str, dtype: str) -> float:
    return MODELS[model][0] * DTYPE_BYTES[dtype]


def pow2floor(x: float) -> int:
    if x < 1:
        return 0
    n = 1
    while n * 2 <= x:
        n *= 2
    return n


def physical_batch(model, total_bytes, crop_px, dtype, *, fraction=None, batch_size=BATCH_SIZE) -> int:
    """Total bytes only: never free memory. 0 means unsupported."""
    frac = MEMORY_FRACTION if fraction is None else fraction
    budget = frac * total_bytes - weights_bytes(model, dtype) - RESERVE_BYTES
    if budget <= 0:
        return 0
    return min(batch_size, pow2floor(budget / per_sample_bytes(model, crop_px, dtype)))


def guard_ok(model, total, crop, dtype, m) -> bool:
    """Budget identity: m * per_sample <= 0.8 * total - weights - RESERVE.

    Pow2floor slack and the batch cap make the weaker ``<= total`` form pass even
    when the fraction is 1.2, so the bound is checked against the spec's 0.8.
    """
    return (m * per_sample_bytes(model, crop, dtype)
            <= SPEC_MEMORY_FRACTION * total - weights_bytes(model, dtype) - RESERVE_BYTES)


FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    combos = [(mo, g, c, d) for mo in MODELS for g in GPUS for c in CROPS for d in DTYPE_BYTES]

    # 1
    unsupported, bad = [], []
    for mo, g, c, d in combos:
        m = physical_batch(mo, GPUS[g], c, d)
        if m == 0:
            unsupported.append((mo, g, c, d))
            continue
        if not (m & (m - 1) == 0 and 1 <= m <= BATCH_SIZE):
            bad.append((mo, g, c, d, m))
    check("1 m is a power of two in [1, batch_size] wherever supported", not bad, f"{len(combos)} combos, {len(unsupported)} unsupported")
    print("     unsupported set:", sorted({(mo, g, d) for mo, g, c, d in unsupported}))

    # 2
    missing = [(mo, g, c) for mo in MODELS for g in BIG_GPUS for c in CROPS
               if (mo, g, c, "bfloat16") in set(unsupported)]
    check("2 every model runs in bfloat16 at every crop on 48 GiB and up", not missing, str(missing[:3]))

    # 3
    w = weights_bytes("dinov3-7b", "float32")
    check("3 dinov3-7b float32 on the 16 GiB card is unsupported",
          physical_batch("dinov3-7b", GPUS["p100-16"], 224, "float32") == 0,
          f"weights {w / GIB:.2f} GiB")
    check("3b dinov3-7b float32 on a 40 GiB A100 fits (plan review I6)",
          physical_batch("dinov3-7b", GPUS["a100-40"], 224, "float32") >= 1,
          f"m={physical_batch('dinov3-7b', GPUS['a100-40'], 224, 'float32')}")

    # 4
    stable = all(physical_batch(mo, GPUS[g], c, d) == physical_batch(mo, GPUS[g], c, d) for mo, g, c, d in combos)
    import inspect
    params = list(inspect.signature(physical_batch).parameters)
    no_free = not any("free" in p for p in params)
    guard = all(guard_ok(mo, GPUS[g], c, d, physical_batch(mo, GPUS[g], c, d))
                for mo, g, c, d in combos if physical_batch(mo, GPUS[g], c, d) > 0)
    check("4 total bytes only, repeatable, and m*per_sample <= 0.8*total - weights - reserve", stable and no_free and guard)

    # 5
    order = sorted(GPUS, key=GPUS.get)
    mono = True
    for mo in MODELS:
        for c in CROPS:
            for d in DTYPE_BYTES:
                ms = [physical_batch(mo, GPUS[g], c, d) for g in order]
                mono &= ms == sorted(ms)
        for g in GPUS:
            for d in DTYPE_BYTES:
                ms = [physical_batch(mo, GPUS[g], c, d) for c in CROPS]
                mono &= ms == sorted(ms, reverse=True)
    check("5 monotone in GPU memory and in crop_px", mono)

    # 6
    m6 = physical_batch("dinov2-base", GPUS["ada6000-48"], 224, "float32")
    check("6 dinov2-base 224 float32 on Ada: m == 64", m6 == 64, f"m={m6}")

    # 7 self-test: a planted MEMORY_FRACTION = 1.2 must trip the guard.
    tripped = [(mo, g, c, d) for mo, g, c, d in combos
               if (m := physical_batch(mo, GPUS[g], c, d, fraction=1.2)) > 0 and not guard_ok(mo, GPUS[g], c, d, m)]
    check("7 planted MEMORY_FRACTION=1.2 makes the budget guard fail", bool(tripped), f"{len(tripped)} combos trip it")

    print("FAILED: " + ", ".join(FAILS) if FAILS else "all checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
