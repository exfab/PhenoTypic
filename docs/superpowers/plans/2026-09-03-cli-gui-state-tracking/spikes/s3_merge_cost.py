"""S-3: can TASK_FINALIZE hold the shard merge in memory?

Spec §8, §10. Compares (a) polars concat of all N embedded tables, (b) a shard
write followed by a concat of the K shard parquets, and (c) a streaming
scan_parquet -> sink_parquet, on peak RSS and wall-clock. A projected peak RSS
above the finalizer's --mem means P5 needs the streaming merge.

Deviation from the phase-0 draft, and why
-----------------------------------------
`resource.getrusage().ru_maxrss` is a per-process HIGH-WATER MARK: it never
falls. The draft ran all three lanes in one process and printed ru_maxrss after
each, so every reading after the first was the running maximum -- and the first
lane (the full in-memory concat) held its result alive to the end for the
row-count assert. The streaming lane would have reported the in-memory lane's
peak as its own, which is precisely the number that decides IN-MEMORY vs
STREAMING. Read literally, the draft could only ever conclude that streaming
costs as much as the concat it was meant to be cheaper than.

Each lane therefore runs in its own subprocess here and reports its own
RUSAGE_SELF peak. Shards persist in the scratch dir between lanes, so the
ordering constraint (shard before merge/stream) still holds. The row-count
agreement the draft asserted in-process is now a comparison of the counts the
lanes report -- same check, no shared address space.

The streaming lane also falls back to a lazy `diagonal_relaxed` concat when a
uniform multi-path `scan_parquet` refuses heterogeneous shard schemas. That is
not defensive padding: the shipped aggregator carries the same fallback
(`_cli_parquet_agg.py:95`), so heterogeneity happens in this tree's lineage, and
without the fallback the cheapest lane would crash after the two expensive ones
had already been paid for.

Usage:
    uv run python .../spikes/s3_merge_cost.py <output_dir> <scratch_dir> [K]
"""

from __future__ import annotations

import re
import resource
import subprocess
import sys
import time
from pathlib import Path

LANES = ("direct", "shard", "merge", "stream")


def _peak_rss_mb() -> float:
    """This process's own high-water RSS, in MB (ru_maxrss is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _tables(output_dir: Path) -> list[Path]:
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH, results_dir

    return [
        store / MEASUREMENT_TABLE_RELATIVE_PATH
        for store in sorted(results_dir(output_dir).glob("*/zarr/*.ome.zarr"))
        if (store / MEASUREMENT_TABLE_RELATIVE_PATH).is_file()
    ]


def _shard_paths(scratch: Path) -> list[Path]:
    return sorted(scratch.glob("shard_*.parquet"))


def _emit(lane: str, rows: int, seconds: float, extra: str = "") -> None:
    """One machine-readable line the parent parses, plus context for a human."""
    print(
        f"RESULT lane={lane} rows={rows} seconds={seconds:.3f} "
        f"peak_rss_mb={_peak_rss_mb():.1f} {extra}".rstrip()
    )


def _run_lane(lane: str, output_dir: Path, scratch: Path, k: int) -> int:
    import polars as pl

    if lane == "direct":
        tables = _tables(output_dir)
        t0 = time.perf_counter()
        whole = pl.concat(
            [pl.read_parquet(t) for t in tables], how="diagonal_relaxed"
        )
        elapsed = time.perf_counter() - t0
        _emit("direct", whole.height, elapsed, f"cols={whole.width} n_tables={len(tables)}")
        return 0

    if lane == "shard":
        tables = _tables(output_dir)
        for stale in _shard_paths(scratch):
            stale.unlink()
        step = max(len(tables) // k, 1)
        rows = 0
        t0 = time.perf_counter()
        for i in range(k):
            chunk = tables[i * step : (i + 1) * step] if i < k - 1 else tables[i * step :]
            if not chunk:
                continue
            frame = pl.concat(
                [pl.read_parquet(t) for t in chunk], how="diagonal_relaxed"
            )
            rows += frame.height
            frame.write_parquet(scratch / f"shard_{i:04d}.parquet")
            del frame
        elapsed = time.perf_counter() - t0
        _emit("shard", rows, elapsed, f"n_shards={len(_shard_paths(scratch))}")
        return 0

    shards = _shard_paths(scratch)
    if not shards:
        print(f"RESULT lane={lane} ERROR no shard parquets in {scratch}")
        return 1

    if lane == "merge":
        t0 = time.perf_counter()
        merged = pl.concat(
            [pl.read_parquet(s) for s in shards], how="diagonal_relaxed"
        )
        elapsed = time.perf_counter() - t0
        _emit("merge", merged.height, elapsed, f"cols={merged.width}")
        return 0

    # lane == "stream"
    out = scratch / "streamed.parquet"
    out.unlink(missing_ok=True)
    paths = [str(s) for s in shards]
    t0 = time.perf_counter()
    try:
        pl.scan_parquet(paths).sink_parquet(out)
        mode = "uniform"
    except Exception as exc:
        # Heterogeneous shard schemas. This mirrors production rather than
        # padding against a hypothetical: the shipped aggregator carries the
        # same fallback at `_cli_parquet_agg.py:95` ("Uniform read failed;
        # falling back to diagonal_relaxed concat"), so a uniform read really
        # does get refused on trees in this lineage. Without it the cheapest
        # lane dies after the two expensive ones have already been paid for.
        print(f"uniform scan_parquet refused ({type(exc).__name__}: {exc}); "
              f"falling back to lazy diagonal_relaxed")
        out.unlink(missing_ok=True)
        pl.concat(
            [pl.scan_parquet(p) for p in paths], how="diagonal_relaxed"
        ).sink_parquet(out)
        mode = "diagonal_relaxed"
    elapsed = time.perf_counter() - t0
    # Count rows without materialising the frame we just avoided materialising.
    rows = pl.scan_parquet(out).select(pl.len()).collect().item()
    _emit("stream", rows, elapsed, f"mode={mode}")
    return 0


def _drive(output_dir: Path, scratch: Path, k: int) -> int:
    """Run each lane in a fresh process and collect its own peak RSS."""
    results: dict[str, dict[str, str]] = {}
    for lane in LANES:
        print(f"\n=== lane {lane} ===", flush=True)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--lane", lane,
             str(output_dir), str(scratch), str(k)],
            capture_output=True,
            text=True,
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            print(f"lane {lane} failed with exit {proc.returncode}")
            return proc.returncode
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                results[lane] = dict(
                    m.groups() for m in re.finditer(r"(\w+)=(\S+)", line)
                )

    print("\n=== summary ===")
    for lane in LANES:
        got = results.get(lane)
        if got:
            print(
                f"{lane:7s} rows={got.get('rows'):>9s} "
                f"seconds={got.get('seconds'):>9s} "
                f"peak_rss_mb={got.get('peak_rss_mb'):>9s}"
            )

    counts = {
        lane: results[lane]["rows"]
        for lane in ("direct", "merge", "stream")
        if lane in results
    }
    if len(set(counts.values())) == 1:
        print(f"row counts agree ({next(iter(counts.values()))})")
        return 0
    print(f"ROW COUNTS DISAGREE: {counts}")
    return 1


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--lane":
        lane = argv[1]
        return _run_lane(lane, Path(argv[2]).resolve(), Path(argv[3]).resolve(), int(argv[4]))

    output_dir = Path(argv[0]).resolve()
    scratch = Path(argv[1]).resolve()
    k = int(argv[2]) if len(argv) > 2 else 16
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"output_dir={output_dir}\nscratch={scratch}\nK={k}")
    return _drive(output_dir, scratch, k)


if __name__ == "__main__":
    raise SystemExit(main())
