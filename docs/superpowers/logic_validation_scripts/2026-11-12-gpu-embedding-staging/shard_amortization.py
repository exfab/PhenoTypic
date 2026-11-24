"""Re-derive the Stage-4 shard amortisation threshold (#282 spec 2, plan P0 T1).

**Historical record: the preflight no longer uses this** (U13, user decision
2026-11-23). ``PF-EMBED-SHARD-AMORTIZE`` was dropped with its code, hint, docs
row and tests, because the P5 live smoke measured the cold load at 39.7-297.9 s
to the first commit, not the 750 s assumed here, and the warning fired on a run
whose load was under 5 min. Nothing in ``src/`` copies these constants any more;
the script is kept, and still runs, as the record of what O6 derived.

Stdlib only; never imports ``phenotypic``. P5 T3 copied COLD_LOAD_S,
WARM_GPU_S_PER_IMAGE and the threshold function into ``src/``; U13 removed
them again.

The warm times are the GPU-bound sum of spec section 2's objects (66 crops)
and 96-cell grid rows: base 0.72 + 0.99, large 1.05 + 1.48. The agar cell
(2.1 s) is a CPU resize and is left out. This is an efficiency threshold for
that pipeline shape, not a correctness one.
"""
from __future__ import annotations

import math
import sys

COLD_LOAD_S = 750.0
WARM_GPU_S_PER_IMAGE = {"base": 0.72 + 0.99, "large": 1.05 + 1.48}
# Quoted constants, as the plan writes them.
QUOTED = {"base": 1.71, "large": 2.53}


def amortize_min_images_per_shard(size: str) -> int:
    return math.ceil(COLD_LOAD_S / WARM_GPU_S_PER_IMAGE[size])


FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    check("0 warm times sum to the quoted constants",
          all(abs(WARM_GPU_S_PER_IMAGE[k] - v) < 1e-9 for k, v in QUOTED.items()),
          str(WARM_GPU_S_PER_IMAGE))
    base_n, large_n = amortize_min_images_per_shard("base"), amortize_min_images_per_shard("large")
    check("1 threshold base == 439 and large == 297", (base_n, large_n) == (439, 297),
          f"750/1.71={COLD_LOAD_S / 1.71:.3f}, 750/2.53={COLD_LOAD_S / 2.53:.3f}")
    ok2 = True
    for size, n in (("base", base_n), ("large", large_n)):
        w = WARM_GPU_S_PER_IMAGE[size]
        ok2 &= COLD_LOAD_S <= n * w                          # load <= GPU time
        ok2 &= COLD_LOAD_S <= (COLD_LOAD_S + n * w) / 2      # load <= half of total
        ok2 &= COLD_LOAD_S > (n - 1) * w                     # and n is the minimum
    check("2 at the threshold the load is at most half the shard time, and n-1 is not", ok2)
    per_shard = 33000 // 4
    check("3 33,000 images on 4 shards fire no warning",
          per_shard >= max(base_n, large_n), f"{per_shard} per shard vs {max(base_n, large_n)}")
    print("FAILED: " + ", ".join(FAILS) if FAILS else "all checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
