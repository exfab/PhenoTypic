# Phase 0 — Spike gate

**Depends on:** nothing. **Blocks:** P1 (via S-5), P5 (via S-2 and S-3).

**Spec:** §10, as amended by [D-A and D-B](OPEN-QUESTIONS.md).

**This phase writes no production code.** It produces four measurements and a written
verdict for each.

### What changed from spec §10

| Spike | Spec §10 | Here |
|---|---|---|
| **S-1** hardlink re-promote | Gates §6.3; a bad result cascades into §6.4 and §7.4 | **Cut.** D-A writes per-store metadata at promote time, so there is no re-promote to measure and no §6.3 to gate. |
| **S-2** shard sizing | Chunk-sizing formula; whether backfill shares a task | **Kept**, minus the backfill half — shard workers aggregate only. |
| **S-3** merge cost | Whether `TASK_FINALIZE` holds the merge in memory | **Kept unchanged.** |
| **S-4** backfill locality | Gates the §8 DAG | **Kept, reduced.** The question moves from "can a *post-hoc* worker project metadata locally?" to "does the *per-image* projection at promote time match what a global join would attribute?" — still worth asking, because duplicate-key fan-out and metadata-only rows are where a local projection quietly diverges. |
| **S-5** cache cold-start | *(new)* | **Added by D-B.** Decides whether the verification cache needs an on-disk tier at all. |

**No P0 result can now invalidate the design.** D-A removed the mechanism S-1 was gating.
S-5 chooses between two implementations of one task; S-2 and S-3 choose parameters.

**Files:**
- Create: `.../spikes/s2_shard_sizing.py`
- Create: `.../spikes/s3_merge_cost.py`
- Create: `.../spikes/s4_metadata_projection_locality.py`
- Create: `.../spikes/s5_cache_cold_start.py`
- Create: `.../spikes/RESULTS.md`
- Create: `.../spikes/run_spikes.sbatch`

**These scripts import `phenotypic` and therefore do NOT go in
`docs/superpowers/logic_validation_scripts/`.** That directory's contract — nothing in it
imports the code under test — is what makes it an independent witness, and the contract is
directory-wide, not per file (project `CLAUDE.md`; spec §10).

**Interfaces:**
- Consumes: nothing from this plan.
- Produces: `spikes/RESULTS.md`, whose verdict lines are cited by P1 Task 3 (cache shape),
  P4 Task 5 (metadata projection) and P5 Tasks 1 and 4 (shard count, merge strategy).

---

## Step 0: the fixture tree

Every spike needs a **real** PhenoTypic output tree on **GPFS** (`/bigdata` or `/rhome`),
not a tmpfs fixture — the point is metadata cost on the shared filesystem.

- [ ] **Step 0.1: Find or build one**

```bash
find /bigdata/exfab/anguy344 /rhome/anguy344/bigdata_exfab -maxdepth 4 \
     -type d -name 'zarr' -path '*/results/*' 2>/dev/null | head -20
```

Prefer a tree with **≥ 200 stores**. If none exists, build one:

```bash
uv run python -m phenotypic \
  --input <a real plate image directory> \
  --output /bigdata/exfab/anguy344/spike-fixture \
  --pipeline <a pipeline.json with a detector and MeasureShape> \
  --metadata <a real metadata.csv>
```

Record in `RESULTS.md`: the tree path, `N_stores`, total bytes, whether a `metadata.csv`
is present, and `df -T <path>` (must report `gpfs`).

---

## S-5 — Cache cold start (gates P1's cache shape) — **runs first**

**Measures:** the wall-clock of one **cold** deep verification at realistic `N`, and how
much a warm in-process cache saves on the second call. Decides whether the on-disk tier
D-B deferred is needed.

**Why this is the one that runs first:** it is the only P0 result that changes what P1
builds, and P1 is the phase everything else depends on.

- [ ] **Step 1: Write the spike**

`spikes/s5_cache_cold_start.py`:

```python
"""S-5: does the verification cache need an on-disk tier?

Decision D-B (OPEN-QUESTIONS): audit S1 proposed a PROCESS-LEVEL cache; spec §9.1
escalated it to a file. Every cadence the audit measured -- the observer's 2s tick,
the viewer's 5-10s poll, OutputRoot.discover's double read, OutputMutationGuard's
double read -- repeats inside ONE long-lived process, which an in-memory cache
serves completely. On-disk buys only cold-start reuse ACROSS processes.

This measures the thing that decision turns on: how expensive is a cold deep
verification, and how often would a process actually pay it?

Reports:
  cold_deep_seconds   -- one full verification from a cold process
  warm_stat_seconds   -- the same answer with every artifact already stat-able
  hash_bytes          -- total bytes hashed on the cold pass

Read it as: if cold_deep_seconds is small enough that a fresh GUI launch, a CLI
resume, or a SLURM worker start can absorb it, the in-process cache is sufficient
and no new tracked artifact ships.

Usage:
    uv run python .../spikes/s5_cache_cold_start.py <output_dir>
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _marker_artifacts(output_dir: Path) -> list[tuple[Path, str]]:
    """Every (artifact, sha256) pair today's markers declare."""
    from phenotypic.sdk_ import progress_dir

    root = output_dir.resolve()
    pairs: list[tuple[Path, str]] = []
    for marker in (progress_dir(output_dir) / "image_complete").rglob("*.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for descriptor in (payload.get("artifacts") or {}).values():
            rel = descriptor.get("path")
            digest = descriptor.get("sha256")
            if isinstance(rel, str) and isinstance(digest, str):
                pairs.append((root / rel, digest))
    return pairs


def main() -> int:
    output_dir = Path(sys.argv[1]).resolve()
    pairs = _marker_artifacts(output_dir)
    print(f"n_artifacts={len(pairs)}")

    # Cold: hash every declared artifact, exactly as valid_image_success does.
    hashed = 0
    t0 = time.perf_counter()
    for artifact, _digest in pairs:
        try:
            with artifact.open("rb") as handle:
                h = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
                    hashed += len(chunk)
        except (OSError, IsADirectoryError):
            continue
    cold = time.perf_counter() - t0
    print(f"cold_deep_seconds={cold:.3f} hash_bytes={hashed} ({hashed / 1e9:.2f} GB)")

    # Warm: what a stat-tuple currency check costs instead.
    t0 = time.perf_counter()
    for artifact, _digest in pairs:
        try:
            st = artifact.stat()
            _ = (st.st_size, st.st_mtime_ns)
        except OSError:
            continue
    warm = time.perf_counter() - t0
    print(f"warm_stat_seconds={warm:.3f} speedup={cold / max(warm, 1e-9):.1f}x")

    # Extrapolate to 6,000 images at the observed per-artifact cost.
    per = cold / max(len(pairs), 1)
    print(f"projected_cold_seconds_at_6000_images={per * 6000 * 3:.1f}  "
          f"(3 artifacts/image: store root, measurements, overlay)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it on a compute node, on GPFS**

Compute work is a Slurm job — use the **`slurm-job`** skill:

```bash
srun -p short -c 4 --mem=16G -t 0:30:00 --pty bash
uv run python .../spikes/s5_cache_cold_start.py /bigdata/exfab/anguy344/spike-fixture
```

- [ ] **Step 3: Record the S-5 verdict**

Write exactly one of:

- **`S-5 IN-PROCESS SUFFICIENT`** — projected cold verification at N=6000 is under 30 s.
  A fresh GUI launch, a CLI resume, or a SLURM worker start absorbs it once; every
  subsequent call in that process is a stat sweep. **P1 Task 3 builds the in-process cache
  only. No new file ships.** This is the expected outcome and the one D-B prefers.
- **`S-5 ON-DISK TIER NEEDED`** — projected cold verification exceeds 30 s. P1 Task 3
  builds the in-process cache **plus** the `.phenotypic/verification_cache.json` tier from
  spec §9.1, with the full INV-CACHE mutation suite including the corrupt-JSON cases.
  Record the measured number that justifies the extra artifact **in the module docstring**,
  so a later reader can tell it was measured rather than assumed.

---

## S-2 — Shard sizing (gates P5's chunk formula)

**Measures:** per-image cost of one aggregation shard task's real work, at `K ∈ {1, 4, 16,
64}`, against the cluster's `MaxArraySize` / `MaxSubmitJobs`.

- [ ] **Step 4: Read the cluster's real limits**

```bash
scontrol show config | grep -E 'MaxArraySize|MaxJobCount|MaxSubmitJobs'
```

Record them. The plan assumes `MaxArraySize = 2500`, so the highest legal index is **2499**
(`--array=1-2500` is rejected) per the user's global `CLAUDE.md`. **If `scontrol` reports
something else, that value wins** and P5's formula uses it.

- [ ] **Step 5: Write and run `s2_shard_sizing.py`**

```python
"""S-2: how long does one aggregation shard task take?

Spec §8, §10, amended by D-A -- shard workers aggregate ONLY; the metadata backfill
they were also going to do is written at promote time instead.

Reports seconds per image and per shard for K in {1,4,16,64}, so P5 can size K from
a wall-clock target rather than a guess. It does not submit jobs; it times the
per-task body.

Usage:
    uv run python .../spikes/s2_shard_sizing.py <output_dir>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


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


def main() -> int:
    output_dir = Path(sys.argv[1]).resolve()
    tables = _tables(output_dir)
    print(f"n_tables={len(tables)}")

    for k in (1, 4, 16, 64):
        if k > len(tables):
            continue
        shard = tables[: max(len(tables) // k, 1)]
        t0 = time.perf_counter()
        rows = _shard_body(shard)
        elapsed = time.perf_counter() - t0
        print(
            f"K={k:3d} images={len(shard):4d} rows={rows:7d} "
            f"seconds={elapsed:7.3f} per_image={elapsed / len(shard):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Record the S-2 verdict**

Write the measured `seconds_per_image` and the resulting formula, instantiated:

```
K = clamp(ceil(N * seconds_per_image / 900), 1, min(MaxArraySize, MaxSubmitJobs) - 1)
```

The `- 1` is the reserved `TASK_FINALIZE` trigger entry — the project `CLAUDE.md` requires
every trigger entry to be counted when sizing chunks against `MaxArraySize`.

---

## S-3 — Merge cost (gates `TASK_FINALIZE`'s memory shape)

**Measures:** peak RSS and wall-clock merging `K` shard parquets versus a single-task
concat at `N ≈ 6000`, plus the streaming alternative.

- [ ] **Step 7: Write and run `s3_merge_cost.py`**

```python
"""S-3: can TASK_FINALIZE hold the shard merge in memory?

Spec §8, §10. Compares (a) polars concat of K shard parquets, (b) a single-task
concat of all N embedded tables, and (c) a streaming scan_parquet -> sink_parquet,
on peak RSS and wall-clock. A projected peak RSS above the finalizer's --mem means
P5 needs the streaming merge.

Usage:
    uv run python .../spikes/s3_merge_cost.py <output_dir> <scratch_dir> [K]
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> int:
    import polars as pl

    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH, results_dir

    output_dir = Path(sys.argv[1]).resolve()
    scratch = Path(sys.argv[2]).resolve()
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    scratch.mkdir(parents=True, exist_ok=True)

    tables = [
        s / MEASUREMENT_TABLE_RELATIVE_PATH
        for s in sorted(results_dir(output_dir).glob("*/zarr/*.ome.zarr"))
    ]
    tables = [t for t in tables if t.is_file()]
    print(f"n_tables={len(tables)} K={k} rss_start_mb={_peak_rss_mb():.1f}")

    t0 = time.perf_counter()
    whole = pl.concat([pl.read_parquet(t) for t in tables], how="diagonal_relaxed")
    t_direct = time.perf_counter() - t0
    print(
        f"direct_concat seconds={t_direct:.3f} rows={whole.height} "
        f"cols={whole.width} peak_rss_mb={_peak_rss_mb():.1f}"
    )

    shards: list[Path] = []
    step = max(len(tables) // k, 1)
    t0 = time.perf_counter()
    for i in range(k):
        chunk = tables[i * step : (i + 1) * step] if i < k - 1 else tables[i * step :]
        if not chunk:
            continue
        shard = scratch / f"shard_{i:04d}.parquet"
        pl.concat(
            [pl.read_parquet(t) for t in chunk], how="diagonal_relaxed"
        ).write_parquet(shard)
        shards.append(shard)
    t_shard = time.perf_counter() - t0

    t0 = time.perf_counter()
    merged = pl.concat([pl.read_parquet(s) for s in shards], how="diagonal_relaxed")
    t_merge = time.perf_counter() - t0
    print(
        f"shard_write seconds={t_shard:.3f}  merge seconds={t_merge:.3f} "
        f"rows={merged.height} peak_rss_mb={_peak_rss_mb():.1f}"
    )

    t0 = time.perf_counter()
    pl.scan_parquet([str(s) for s in shards]).sink_parquet(scratch / "streamed.parquet")
    t_stream = time.perf_counter() - t0
    print(f"streaming_merge seconds={t_stream:.3f} peak_rss_mb={_peak_rss_mb():.1f}")

    assert merged.height == whole.height, (
        f"shard merge lost rows: {merged.height} != {whole.height}"
    )
    print("row counts agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Record the S-3 verdict**

Extrapolate measured peak RSS to `N = 6000`. Write one of:

- **`S-3 IN-MEMORY`** — projected peak RSS under 32 GB. `TASK_FINALIZE` uses `pl.concat`;
  P5 Task 4 sets `--mem` to 2 × projected.
- **`S-3 STREAMING`** — projected peak RSS exceeds 32 GB, or `streaming_merge` is within
  1.5 × the in-memory merge. P5 Task 4 uses `pl.scan_parquet(...).sink_parquet(...)`.

---

## S-4 — Metadata projection locality (gates P4's promote-time write)

**Measures, reduced by D-A:** that the metadata rows one image's store should carry —
computed from that image's own measurements and `metadata.csv`, with no global frame —
equal the rows a global join would attribute to it, under **duplicate-key fan-out**,
**metadata-only rows**, and **partial keys**.

Under D-A this is asked at promote time, where the worker already holds the image's
measurements. That makes locality nearly structural — but "nearly" is why the spike still
runs: duplicate-key fan-out and metadata-only rows are exactly where a local projection
quietly diverges from a global one, and `prepare_embedded_measurement_table`
(`_embedded_measurement_tables.py:42`) already warns about both.

- [ ] **Step 9: Write `s4_metadata_projection_locality.py`**

```python
"""S-4: is the per-image metadata projection local?

Spec §8, §10, reduced by D-A. Asserts an EQUIVALENCE, not a duration: the metadata
rows a local projection selects for one image must equal the rows a global join
would attribute to that image, under fan-out, metadata-only rows, and partial keys.

If this fails, P4's promote-time write is wrong and the metadata table has to be
derived after the merge -- which under D-A means it cannot be written at promote
time at all, and D-A has to be revisited with the user.

Usage:
    uv run python .../spikes/s4_metadata_projection_locality.py <output_dir> <metadata.csv>
"""

from __future__ import annotations

import sys
from pathlib import Path


def _local_projection(table, metadata):
    """What the promote-time writer can compute from one image alone."""
    common = [c for c in metadata.columns if c in table.columns]
    if not common:
        return None
    keys = table.select(common).unique()
    return metadata.join(keys, on=common, how="semi").sort(common)


def _global_attribution(all_tables, metadata, index):
    """What a post-merge finalizer would attribute to image *index*."""
    import polars as pl

    table = all_tables[index]
    common = [c for c in metadata.columns if c in table.columns]
    if not common:
        return None
    merged = pl.concat(all_tables, how="diagonal_relaxed")
    joined = metadata.join(merged.select(common).unique(), on=common, how="semi")
    keys = table.select(common).unique()
    return joined.join(keys, on=common, how="semi").sort(common)


def main() -> int:
    import polars as pl

    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH, results_dir

    output_dir = Path(sys.argv[1]).resolve()
    metadata_csv = Path(sys.argv[2]).resolve()

    tables = [
        pl.read_parquet(s / MEASUREMENT_TABLE_RELATIVE_PATH)
        for s in sorted(results_dir(output_dir).glob("*/zarr/*.ome.zarr"))
        if (s / MEASUREMENT_TABLE_RELATIVE_PATH).is_file()
    ]
    base = pl.read_csv(metadata_csv)
    print(f"n_tables={len(tables)} metadata_rows={base.height}")

    variants = {
        "clean": base,
        "fanout": pl.concat([base, base.head(1), base.head(1)]),
        "metadata_only": pl.concat(
            [
                base,
                base.head(1).with_columns(
                    pl.lit("__ABSENT__").alias(base.columns[0])
                ),
            ]
        ),
        "partial_keys": base.drop(base.columns[-1]) if base.width > 1 else base,
    }

    failures = 0
    for name, metadata in variants.items():
        for i in range(min(len(tables), 25)):
            local = _local_projection(tables[i], metadata)
            glob = _global_attribution(tables, metadata, i)
            if local is None and glob is None:
                continue
            if local is None or glob is None or not local.equals(glob):
                failures += 1
                print(f"MISMATCH variant={name} image={i}")
                break
        else:
            print(f"variant={name}: local == global for every image checked")

    print()
    print("S-4 PASS" if failures == 0 else f"S-4 FAIL ({failures} variants)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 10: Record the S-4 verdict**

- **`S-4 PASS`** — P4 Task 5 writes `pht-metadata.parquet` at promote time as planned.
- **`S-4 FAIL`** — **stop and report to the user.** A local projection that diverges from a
  global one means D-A's promote-time write cannot be correct, and the decision needs
  revisiting. Name the failing variant and one concrete diverging row.

---

## Running and recording

- [ ] **Step 11: Run S-2, S-3 and S-4 as one Slurm job**

`spikes/run_spikes.sbatch` — fill in and submit via the **`slurm-job`** skill. The
constraints from the user's global `CLAUDE.md` that bind here:

- `-p short` (max `2:00:00`), default account — **no `--account=` flag at all.** An empty
  value is an invalid account and `sbatch` rejects the whole submission.
- **Always set `--mem` and `--time`.** `DefMemPerCPU` is 1 GB/CPU and the default is the
  usual cause of a silent OOM kill. S-3 needs at least 64 GB.
- `--output` must be on shared storage (`/bigdata` or `/rhome`), **never**
  `/scratch/<user>/<jobid>` — that is node-local and per-job; a job landing on another node
  fails with `ExitCode 0:53` and no log file at all.
- `sbatch --parsable` prints the error and returns an **empty** id on rejection. Verify the
  captured id matches `^[0-9]+$` and surface the raw output on failure.
- Submission ≠ start. Check `scontrol show job <id> | grep -E 'StartTime|Reason'`.

- [ ] **Step 12: Write `RESULTS.md`**

For each of S-2…S-5: the fixture tree and its size, the raw numbers, the verdict line, and
the decision that verdict licenses. Then report S-5 to the user before starting P1.

- [ ] **Step 13: Commit**

```bash
git add docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/spikes/
git commit -m "spike(state): measure cache cold start, shard sizing, merge cost, projection locality

Spec §10 as amended by D-A and D-B. S-1 (hardlink re-promote) is cut: D-A writes
per-store metadata at promote time, so there is no re-promote to measure."
```
