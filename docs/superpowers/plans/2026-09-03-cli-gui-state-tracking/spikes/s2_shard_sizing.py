"""S-2: how long does one aggregation shard task take?

Spec §8, §10, amended by D-A -- shard workers aggregate ONLY; the metadata backfill
they were also going to do is written at promote time instead.

Reports seconds per image and per shard for K in {1,4,16,64}, so P5 can size K from
a wall-clock target rather than a guess. It does not submit jobs; it times the
per-task body.

Deviation from the phase-0 draft, and why
-----------------------------------------
The draft sliced every K off the FRONT of the same sorted list::

    shard = tables[: max(len(tables) // k, 1)]

K=1 therefore read every table first and left the page cache warm for K=4, 16 and
64. The per-image cost those three reported would have been a warm-cache number
while K=1's was cold -- a systematic bias in the one quantity the whole S-2
formula consumes. The sweep would have looked like "per-image cost falls with
shard size" when it was only measuring cache residency.

This version walks a seeded shuffle of the table list with a cursor, so each K
reads tables no earlier K has touched. Every sweep row is a cold read. A final
row re-reads the first slice to report the warm/cold ratio explicitly, which is
what makes the cold numbers interpretable rather than merely conservative.

Sizing must use the COLD number: a real shard worker starts on a freshly
allocated node with nothing of this tree in its page cache.

K=1 asks for a shard of all N images but only the tables no earlier K consumed
remain, so it reports a smaller `images=` count. That is fine -- the formula
consumes `per_image`, not the shard's absolute wall-clock.

Usage:
    uv run python .../spikes/s2_shard_sizing.py <output_dir>
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

#: Fixed so a re-run reads the same slices in the same order.
SHUFFLE_SEED = 20260903


def _tables(output_dir: Path) -> list[Path]:
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH, results_dir

    return [
        store / MEASUREMENT_TABLE_RELATIVE_PATH
        for store in sorted(results_dir(output_dir).glob("*/zarr/*.ome.zarr"))
        if (store / MEASUREMENT_TABLE_RELATIVE_PATH).is_file()
    ]


def _shard_body(tables: list[Path]) -> int:
    """Exactly what one array task's per-image loop will do after D-A."""
    import polars as pl

    rows = 0
    for table in tables:
        rows += pl.read_parquet(table).height
    return rows


def _report(k: int, shard: list[Path], rows: int, elapsed: float, cache: str) -> None:
    print(
        f"K={k:3d} images={len(shard):5d} rows={rows:8d} "
        f"seconds={elapsed:8.3f} per_image={elapsed / len(shard):.4f} cache={cache}"
    )


def main() -> int:
    output_dir = Path(sys.argv[1]).resolve()
    tables = _tables(output_dir)
    print(f"n_tables={len(tables)}")
    if not tables:
        print("no measurement tables found -- wrong tree or wrong layout")
        return 1

    pool = list(tables)
    random.Random(SHUFFLE_SEED).shuffle(pool)

    cursor = 0
    first_slice: list[Path] = []
    # Ascending shard size, so the small shards get their cold read before the
    # large ones exhaust the pool.
    for k in (64, 16, 4, 1):
        want = max(len(tables) // k, 1)
        shard = pool[cursor : cursor + want]
        if not shard:
            print(f"K={k:3d} skipped -- no unread tables left")
            continue
        cursor += len(shard)

        t0 = time.perf_counter()
        rows = _shard_body(shard)
        elapsed = time.perf_counter() - t0
        _report(k, shard, rows, elapsed, "cold")

        if not first_slice:
            first_slice = shard

    # Same tables, second read: how much of the cold cost was the filesystem.
    t0 = time.perf_counter()
    rows = _shard_body(first_slice)
    elapsed = time.perf_counter() - t0
    _report(64, first_slice, rows, elapsed, "warm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
