"""Re-derive the Stage-4 host-memory model (#282 spec 4.4, plan P0 T1).

Stdlib only; never imports ``phenotypic``. P5 T3 copies the constants into
``src/`` (PF-EMBED-RESOURCES) and P2 T3 copies MAX_IN_FLIGHT_OUTPUT_BYTES.

A loader worker holds one decoded plate: ``rgb`` at ``sample_bytes`` per
channel and ``gray``, ``detect_mat``, ``objmap`` at 4 bytes each (hence
``3 * sample_bytes + 12`` per pixel), plus the crops it cut and 1 GiB of
interpreter overhead. Crops are counted as float32 (the normalised form). JOB_MEM_BYTES is 48 GiB
(``--mem=48G``). Stdlib only.
"""
from __future__ import annotations

import sys

GIB = 1 << 30
LOADER_BASE_BYTES = 6 << 30  # spec section 2, peak RSS (about 6 GB)
JOB_MEM_BYTES = 48 * GIB     # --mem=48G in the gpu-baseline jobs


def loader_per_worker(h: int, w: int, sample_bytes: int, crop_bytes: int) -> int:
    return h * w * (3 * sample_bytes + 12) + crop_bytes + GIB


def encoder_output_bytes(h: int, w: int, patch: int, hidden: int) -> int:
    return (h // patch) * (w // patch) * hidden * 4


MAX_IN_FLIGHT_OUTPUT_BYTES = 2 * encoder_output_bytes(4012, 6016, 16, 4096)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    h, w, cpus = 4012, 6016, 8
    crops = 66 * 3 * 224 * 224 * 4
    u8 = LOADER_BASE_BYTES + (cpus - 1) * loader_per_worker(h, w, 1, crops)
    u16 = LOADER_BASE_BYTES + (cpus - 1) * loader_per_worker(h, w, 2, crops)
    check("1 8-CPU shard, uint8 plates, under 48 GiB", u8 < JOB_MEM_BYTES, f"{u8 / 1e9:.2f} GB")
    print(f"     uint16 case: {u16 / 1e9:.2f} GB ({'under' if u16 < JOB_MEM_BYTES else 'OVER'} 48 GiB)")
    check("1b 8-CPU shard, uint16 plates, under 48 GiB", u16 < JOB_MEM_BYTES)

    one = encoder_output_bytes(h, w, 16, 4096)
    check("2 7B masked_dense plate output is 1.54 GB +/- 1%", abs(one - 1.54e9) <= 0.01 * 1.54e9, f"{one} bytes")
    worst = LOADER_BASE_BYTES + MAX_IN_FLIGHT_OUTPUT_BYTES
    check("3 base + MAX_IN_FLIGHT_OUTPUT_BYTES under 48 GiB", worst < JOB_MEM_BYTES,
          f"MAX={MAX_IN_FLIGHT_OUTPUT_BYTES / 1e9:.3f} GB, total {worst / 1e9:.2f} GB")
    check("3b MAX_IN_FLIGHT_OUTPUT_BYTES is about 3.1 GB", abs(MAX_IN_FLIGHT_OUTPUT_BYTES - 3.1e9) <= 0.05 * 3.1e9)

    per_image = 1536 * 4096 * 4
    prefetched = per_image * 2 * (cpus - 1)
    check("4 a 1536-colony crop pipeline never reaches the bound",
          prefetched < MAX_IN_FLIGHT_OUTPUT_BYTES,
          f"{per_image / 1e6:.1f} MB/image x {2 * (cpus - 1)} = {prefetched / 1e6:.0f} MB")
    print("FAILED: " + ", ".join(FAILS) if FAILS else "all checks passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
