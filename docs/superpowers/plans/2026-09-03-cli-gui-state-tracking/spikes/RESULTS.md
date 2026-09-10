# Phase 0 spike results — S-2 and S-3

**Status: complete.** Job 28097634, `COMPLETED`, exit `0:0`, 4:14 elapsed. All numbers below are
measured.

Executed per [`phase-0-spike-gate.md`](../phase-0-spike-gate.md). S-1 and S-4 are cut, S-5
moved to P1's phase gate — none of the three was run, and this file records no verdict for
them.

**P0 gates P5 only.** Nothing here blocks P1–P4.

## The two verdicts

> ### `S-2 seconds_per_image = 0.026 (cold, GPFS). K = clamp(ceil(N × 0.026 / 900), 1, 2499). At N = 6000 this yields K = 1.`
>
> ### `S-3 IN-MEMORY` — peak RSS 2.5 GB measured at N = 6,529, against a 32 GB threshold. P5 Task 4 sets `--mem ≈ 8 GB`.

**Three findings that are not verdicts but change how P5 should read them:**

1. **`K = 1` at the design target.** One shard aggregates the whole run in a *measured* 169.6 s
   against a 900 s budget. The fan-out yields no wall-clock benefit until **N ≈ 34,600** — 5.8x
   the N the plan was designed around. It remains correctness machinery; it is not currently a
   performance optimisation. See the open question below.
2. **Streaming costs *more* memory here, not less** — 421.5 MB against `pl.concat`'s 314.6 MB,
   while running 2x faster. The usual reason to adopt streaming is absent. If memory ever does
   bind, the lever is **sharding** (8.2x cheaper than a flat concat), which P5 already builds.
3. **`mode=uniform`** — all 6,529 tables share a schema, so the `diagonal_relaxed` fallback
   never fired. The heterogeneity path that fallback exists for is **untested by this fixture.**

---

## The fixture tree

```
/bigdata/exfab/anguy344/projects/ucr_029_e_d_Maresca/data/results/2026-08-11
```

| Property | Value |
|---|---|
| Filesystem | `bigdata  gpfs` — `df -T` confirms GPFS, not tmpfs |
| Stores matching `results/*/zarr/*.ome.zarr` | **6,657** |
| Stores carrying `tables/measurements/table.parquet` | **6,529** |
| Measurement-table bytes | 552,413,013 B (527 MB), mean 82.6 KB/table |
| Tree size | ~1.29 TB (sampled: `du -sm` over 15 stores, mean 198.6 MB, range 178–209) |
| `deliverables/metadata.csv` | present, 31,433,735 B |

Already on disk as a cross-check on the merge: `master_measurements.parquet` 103 MB,
`measurements.parquet` 103 MB, `master_measurements.csv` 295 MB.

At **N = 6,657** the fixture sits *above* the N = 6000 target the verdicts extrapolate to,
so S-3 measures the real thing rather than projecting up to it.

### It is the only viable tree, and the near-misses are instructive

The phase doc's `find -maxdepth 4` returns nothing — the real layout needs depth 6. The full
survey:

| Stores | Tree |
|---|---|
| **6657** | `.../ucr_029_e_d_Maresca/data/results/2026-08-11/results/7-24-26_redo_full/zarr` |
| 111 | `/bigdata/exfab/anguy344/slurm_logs/e2e_slurm_27704648/.../zarr` |
| 111 | `/bigdata/exfab/anguy344/slurm_logs/e2e_slurm_27703237/.../zarr` |
| 111 | `/bigdata/exfab/anguy344/slurm_logs/e2e_slurm_27699274/.../zarr` |
| 111 | `/bigdata/exfab/anguy344/slurm_logs/e2e_slurm_27698143/.../zarr` |
| 36 | `.../2026-08-11-migration-test/results/7-24-26_redo_full/zarr` |
| 32 | `/bigdata/exfab/anguy344/slurm_logs/e2e_run_27697742/.../zarr` |
| 1 | `.../test-process-mode-cli/results/single_image/zarr` |
| 0 | `/bigdata/exfab/anguy344/slurm_logs/e2e_slurm_27703196/.../zarr` |
| 0 | `/bigdata/exfab/anguy344/slurm_logs/e2e_gpu_27698054/.../zarr` |

Only the Maresca tree clears the ≥200 bar. **The four 111-store trees carry zero
measurement tables** — they were viable-*looking* rather than viable, and a store count
alone would have suggested otherwise. Anything reading this file for a future fixture should
count `tables/measurements/table.parquet`, not `*.ome.zarr`; the obvious `find` returns four
candidates over 100 stores and every one of them is empty of the thing the spikes read.

**This bounds the result's generality, and P5 should treat it as a stated limit rather than
a footnote.** Every number below describes **one witness**: one dataset directory, ~82 KB mean
tables, this column set. A tree with many small datasets, much wider tables, or far more
objects per image could move both `seconds_per_image` and peak RSS with no code change.

State the limit precisely, because the imprecise version invites the wrong response. The claim
is **not** "we measured one tree and should have measured more". It is: **there was no second
witness available, and we did not decline to seek one.** Everything on `/bigdata/exfab/anguy344`
and `/rhome/anguy344/bigdata_exfab` was surveyed — nine candidate trees, four of which looked
viable on store count and carried zero measurement tables. A second fixture does not exist to
be used; it would have to be *built* by running a full pipeline, which is a decision for the
user and not a step inside a spike.

That distinction is what a later reader needs. A bounded measurement whose bound is known and
exhausted is worth acting on; the same numbers with an unexamined bound are not. If P5's
parameters ever look wrong in production, the first move is to build a second fixture with a
deliberately different shape — not to re-run this one.

**The run sharpened one specific bound.** S-3 reported `mode=uniform`: all 6,529 tables share a
schema, so the `diagonal_relaxed` fallback never executed. That fallback exists in production
precisely because real trees hit heterogeneity (`_cli_parquet_agg.py:95`), so a
deliberately-different second fixture should be **schema-heterogeneous** — that is the specific
gap these numbers leave, not a generic call for more data.

### API assumptions, verified against source before running anything

| Assumption | Verified |
|---|---|
| `progress_dir`, `results_dir`, `MEASUREMENT_TABLE_RELATIVE_PATH` exported from `phenotypic.sdk_` | `sdk_/__init__.py:559,579,375` |
| `results_dir(o)` = `<output>/results` | `sdk_/_io_constants.py:912` |
| `progress_dir(o)` = `<output>/.phenotypic/progress` | `sdk_/_io_constants.py:903` |
| `MEASUREMENT_TABLE_RELATIVE_PATH` = `tables/measurements/table.parquet` | `sdk_/ngff_.py:81` |
| glob `results/*/zarr/*.ome.zarr` matches the real layout | 6,657 hits by `ls` |
| `how="diagonal_relaxed"` matches production | `_cli_parquet_agg.py:110`, `_cli_chunk_writer.py:425,482` |

polars 1.41.2. **No API mismatch** — both deviations recorded below are methodology, not API.

---

## Cluster limits

```
$ scontrol show config | grep -E 'MaxArraySize|MaxJobCount|MaxSubmitJobs'
MaxArraySize            = 2500
MaxJobCount             = 50000
```

**`MaxSubmitJobs` does not appear in `scontrol show config` at all** — it is an association
limit, not a config one, and had to come from `sacctmgr`:

```
$ sacctmgr show assoc user=anguy344 format=Account,MaxSubmitJobs,GrpTRES,MaxJobs -P
Account|MaxSubmit|GrpTRES|MaxJobs
exfab|5000|cpu=32,mem=256G|
iwheeldonlab|5000|cpu=384,mem=1T|
preempt|5000|cpu=384,mem=1T|
...
```

So `MaxArraySize = 2500`, `MaxSubmitJobs = 5000`, and **`min(MaxArraySize, MaxSubmitJobs) =
2500`: MaxArraySize is the sole binding term and MaxSubmitJobs never will be here.** P5's
formula can state the constant rather than carry an inert `min()`.

Highest legal array index is **2499** — `--array=0-2499` is legal, `--array=1-2500` is
rejected.

> **Trap, recorded because the phase doc walks into it.** Its own command greps for
> `MaxSubmitJobs` against `scontrol show config`, which returns *nothing* for that term. A
> formula written as `min(MaxArraySize, MaxSubmitJobs)` over a shell variable populated that
> way becomes `min(2500, 0) = 0`, sizing every shard to zero — and it fails silently, since
> an empty grep is not an error. Take the value from `sacctmgr`, or hard-code 2500.

## Concurrency at the time of measurement

The fixture's `deliverables/` and `.phenotypic/` were last written **2026-09-03 17:16–19:30**,
so whether a job was actively writing it was checked rather than assumed.

A ~40-task array **was** running (`28095933`, `pht-cri_correct_...`, with a dependent
`pht-finalizer` pending), but `scontrol show job` resolves its `Command=` to
`.../AutoConvertRaw/staging/work_correct/20260904_005854_16527/out/` — **a different tree.**
The fixture was quiet *at the time of this survey* (2026-09-04); that is the honest claim,
not that it is quiet in general.

Two consequences:
- That array runs on **`Account=preempt`**, a separate 384-CPU pool from `iwheeldonlab`, so
  it does not draw on the budget this `short` job uses.
- The spikes only ever **read** the fixture (S-3's shards go to a separate scratch dir), so
  a concurrent writer would have meant noise and a possible spurious row-count mismatch —
  never damage to the tree.

---

## Two deviations from the phase doc's scripts

Both were found by reading the drafts before running them, and both are fixed in the
committed scripts. Neither is an API difference. The reasoning lives in each script's
docstring as well as here, because the failure mode of both is that someone "simplifies"
them back to the draft.

### S-2 — the K sweep measured page-cache residency, not shard size

The draft sliced every K off the front of one list:

```python
shard = tables[: max(len(tables) // k, 1)]
```

K=1 reads all 6,529 tables first, leaving all 527 MB warm in a 200 GB node's page cache; K=4,
16 and 64 then read subsets of what K=1 just warmed. The sweep would report per-image cost
*falling* with shard size — indistinguishable from a real economy of scale, and it would have
been written into P5's formula as one. This is the more dangerous of the two flaws precisely
because it yields a plausible number rather than an obviously broken one.

**Fixed** by walking a seeded shuffle with a cursor, so each K reads tables no earlier K
touched and every sweep row is a cold read, plus one extra row re-reading the first slice to
report the warm/cold ratio explicitly. Sizing uses the **cold** number, because a real shard
worker starts on a freshly allocated node with none of the tree cached. Consequence: K=1 gets
the ~4,387 tables that remain rather than all 6,529, which is fine — the formula consumes
`per_image`, not the shard's absolute wall-clock, and `images=` reports the true count.

**Measured evidence that this is real and not theoretical.** The smoke run on the 32-table
tree reads the *same* K=64 slice twice:

```
K= 64 images=    1 rows=      54 seconds=   0.303 per_image=0.3029 cache=cold
K= 64 images=    1 rows=      54 seconds=   0.003 per_image=0.0029 cache=warm
```

**104x.** On this filesystem the draft's warm rows would have produced a `seconds_per_image`
up to two orders of magnitude below the truth — in a formula that divides by it, the
difference between one shard and a hundred.

> **Do not size from the K=64 row.** At N=32 that rung reads a *single* table, so its
> `per_image` pays one-off process-warmup and first-touch costs that never amortise. The three
> rungs with real company agree closely (0.0330 / 0.0278 / 0.0290 s/image) while K=64 sits 10x
> higher at 0.3029. At N=6,529 the K=64 rung reads ~102 tables and stops being an outlier;
> if it still is in the full run, that is a finding rather than an artifact.

### S-3 — `ru_maxrss` is a high-water mark, so streaming inherited the concat's peak

The draft ran all three lanes in one process and printed
`resource.getrusage(RUSAGE_SELF).ru_maxrss` after each. That value **never falls**, and lane
(a) held its full in-memory result alive to the end for the row-count assert. So the
streaming lane would have reported the in-memory lane's peak as its own — and that is exactly
the number deciding `IN-MEMORY` vs `STREAMING`. Read literally, the lane that exists to be
cheaper was structurally incapable of measuring as cheaper: the check could only return one
answer.

**Fixed** by running each lane in its own subprocess reporting its own `RUSAGE_SELF` peak.
Shards persist in the scratch dir between lanes, so the ordering constraint still holds, and
the in-process assert becomes a comparison of the counts the lanes report — the same check
without a shared address space.

The streaming lane also falls back to a lazy `diagonal_relaxed` concat when a uniform
multi-path `scan_parquet` refuses heterogeneous shard schemas. That mirrors production
(`_cli_parquet_agg.py:95` logs the identical fallback), so it is not defensive padding; without
it the cheapest lane would crash *after* the two expensive lanes had been paid for.

Note the draft's direct-concat lane was sound as written — it runs first, so its reading is
clean. The verdict was salvageable for one lane of three.

**Measured evidence, from the same smoke run:**

```
direct  rows=      723 seconds=    0.117 peak_rss_mb=    499.6
shard   rows=      723 seconds=    0.130 peak_rss_mb=    502.0
merge   rows=      723 seconds=    0.041 peak_rss_mb=    204.0
stream  rows=      723 seconds=    0.036 peak_rss_mb=    229.0
row counts agree (723)
```

`merge` (204 MB) and `stream` (229 MB) both report **below** `direct` (499.6 MB) — which is
arithmetically impossible under the draft, because a shared-process high-water mark cannot
decrease. Those two lanes would have reported ≥499.6 MB regardless of what they used. **This
harness shape is what makes the S-3 verdict decidable at all**; without it the comparison has
one reachable outcome, and four ordinary-looking numbers would have hidden that.

### Run order is load-bearing: S-2 before S-3

Both scripts read the same 6,529 tables, and a compute node's page cache holds all 527 MB, so
whichever runs second runs warm. S-2's verdict is a sizing figure that must be cold; S-3's
verdict is peak RSS, which caching does not affect. So S-2 gets the cold cache, and **S-3's
wall-clock numbers below are warm-cache** — do not compare them against a cold measurement.

---

## Job execution

```
$ sbatch --parsable .../run_spikes.sbatch
28097634

$ scontrol show job 28097634 | grep -E 'StartTime|Reason|NodeList|JobState'   # mid-run
   JobState=RUNNING Reason=None Dependency=(null)
   StartTime=2026-09-04T01:55:16 EndTime=2026-09-04T03:40:16 Deadline=N/A
   NodeList=r42

$ sacct -j 28097634 --format=JobID,State,ExitCode,Elapsed,MaxRSS -P              # final
JobID|State|ExitCode|Elapsed|MaxRSS
28097634|COMPLETED|0:0|00:04:14|
28097634.batch|COMPLETED|0:0|00:04:14|3268572K

$ ls -la /bigdata/exfab/anguy344/slurm_logs/spikes_28097634.err
-rw-r----- 1 anguy344 exfab 0 Sep  4 01:55 .../spikes_28097634.err
```

**`COMPLETED`, exit `0:0`, 4:14 elapsed** — against a 1:45:00 wall, so the job finished on its
own rather than being truncated by the time limit. The `EndTime` above is the *limit*, not the
actual end.

The `.err` is **0 bytes and present**, not absent — `ls -la` rather than `cat`, because an empty
file and a missing one are indistinguishable to `cat` and mean opposite things here.

`-p short`, `-c 8`, `--mem=96G`, `-t 1:45:00`, default account (no `--account` flag).
The id matches `^[0-9]+$`, so this is a real submission rather than `--parsable`'s
empty-id-on-rejection.

**It started immediately** — `Reason=None`, not `Reason=Priority` — so no queue wait
distorts the wall-clocks. Submission and start are recorded separately here because they are
different events and only the second one licenses reading the timings.

> **Caveat for anyone weighing the wall-clocks: this measurement shared a node.** `r42` was
> also running tasks of the `pht-cri_correct` array (`28095933`, on the separate `preempt`
> pool) at the time. The job has its own cpuset, so correctness and peak RSS are unaffected,
> but if a wall-clock looks anomalous against the smoke ratios above, node contention is the
> first thing to check — not the code.

---

## S-2 — shard sizing

```
################ S-2 shard sizing (COLD cache) ################
n_tables=6529
K= 64 images=  102 rows=    2114 seconds=   3.263 per_image=0.0320 cache=cold
K= 16 images=  408 rows=    7898 seconds=  10.168 per_image=0.0249 cache=cold
K=  4 images= 1632 rows=   31564 seconds=  42.923 per_image=0.0263 cache=cold
K=  1 images= 4387 rows=   87022 seconds= 113.283 per_image=0.0258 cache=cold
K= 64 images=  102 rows=    2114 seconds=   0.146 per_image=0.0014 cache=warm
s2_exit=0
```

**stderr was 0 bytes** for the whole cold sweep (`ls -la` confirms the file exists and is
empty, rather than being absent). A cold read of 6,529 parquet files raising no polars warning
is itself evidence of schema uniformity in this tree — and it is the baseline that makes S-3's
`diagonal_relaxed` fallback a *finding* if it fires, rather than noise.

### `seconds_per_image ≈ 0.026`

The three rungs with real company agree within 5% (0.0249 / 0.0263 / 0.0258). **0.026 is the
number to quote.**

**The K=64 watch-item is resolved: it was process warmup, as suspected.** At 102 tables it is
0.0320 — about **25% high**, not the smoke's 10x. So the smoke's outlier was a single-table read
paying interpreter startup and first-touch costs, and it nearly vanishes once the rung has
company. Recorded as answered rather than dropped: the residual 25% is still a genuine
small-shard overhead signal, and it remains the honest floor argument for shard size — just a
far weaker one than the smoke implied.

**The page-cache fix holds at scale: cold 0.0320 vs warm 0.0014 on the identical slice — 23x.**
Lower than the smoke's 104x, because more tables amortise the fixed warmup, but still enormous.
Both ratios belong on the record: the effect is real, and its magnitude depends on N.

### The formula, instantiated — and it returns K = 1

```
K = clamp(ceil(N × seconds_per_image / 900), 1, min(MaxArraySize, MaxSubmitJobs) - 1)
K = clamp(ceil(6000 × 0.026 / 900), 1, 2499)
K = clamp(ceil(156 / 900), 1, 2499)
K = clamp(ceil(0.173),  1, 2499)
K = 1
```

**This barely needed extrapolating.** The four cold rungs read 102 + 408 + 1632 + 4387 = **6,529
tables — every table in the tree — in 169.6 s total.** So "one shard aggregates the entire run"
is a *measured* 169.6 s at N = 6,529, not a projection; scaled to the design target it is ~156 s
at N = 6000. Either way it is **under a fifth of the 900 s target.**

> ### `S-2 seconds_per_image = 0.026 (cold, GPFS). K = clamp(ceil(N × 0.026 / 900), 1, 2499). At N = 6000 this yields K = 1.`

`min(MaxArraySize, MaxSubmitJobs) - 1 = 2499`, a constant here — see the limits section for why
that term can never bind and why writing it as a `min()` over a shell variable is dangerous.

### What K = 1 does and does not license

State this plainly rather than softening it to "K is small":

- **It does not invalidate P5.** The fan-out also carries partial-failure semantics, the
  shard-completeness check (CAN-5), and the reserved `TASK_FINALIZE` trigger entry. None of those
  are wall-clock optimisations; all are correctness machinery that must work whether K is 1 or
  100.
- **It does mean P5's sharding is exercised only by its own tests, not by production need**, at
  present scale and shape. Name that as a risk: a K>1 path that never runs in practice is a path
  that rots, and its tests are the only thing standing between it and silent decay.
- **Crossover: K exceeds 1 only above N ≈ 34,600 images** (900 / 0.026 = 34,615; the 5% spread
  across rungs puts it in the 34,200–36,100 band). That is the single most useful line here for
  P5 — it is the point where the fan-out starts earning its keep, and it is **5.8x the N the plan
  was designed around.**

### Open question for the user — needs a decision, not a guess

Given a single shard finishes the design-target run in ~156 s against a 900 s budget: **is 900 s
the right target, or is the interesting N much larger than 6000?**

These have different consequences and P0 should not pick between them:
- If N ≈ 6000 is the real ceiling, the 900 s target is loose by ~6x and K = 1 is simply correct —
  P5's fan-out is insurance, and should be scoped and justified as insurance.
- If runs of 30k–100k images are anticipated, the crossover is reachable, K > 1 is real, and the
  900 s target deserves re-derivation from what actually needs to fit in a wall-clock window.

**Not answered here.** Deciding it inside a spike would convert a measurement into a design
choice the user never made.

<!-- raw stdout, the derived cold seconds_per_image, the instantiated formula, verdict -->

## S-3 — merge cost

```
################ S-3 merge cost (warm cache; RSS unaffected) ################
K=16

=== lane direct ===
RESULT lane=direct rows=128598 seconds=10.909 peak_rss_mb=2577.6 cols=148 n_tables=6529

=== lane shard ===
RESULT lane=shard rows=128598 seconds=11.170 peak_rss_mb=609.2 n_shards=16

=== lane merge ===
RESULT lane=merge rows=128598 seconds=0.473 peak_rss_mb=314.6 cols=148

=== lane stream ===
RESULT lane=stream rows=128598 seconds=0.228 peak_rss_mb=421.5 mode=uniform

=== summary ===
direct  rows=   128598 seconds=   10.909 peak_rss_mb=   2577.6
shard   rows=   128598 seconds=   11.170 peak_rss_mb=    609.2
merge   rows=   128598 seconds=    0.473 peak_rss_mb=    314.6
stream  rows=   128598 seconds=    0.228 peak_rss_mb=    421.5
row counts agree (128598)
s3_exit=0
```

Independent corroboration: `sacct` reports `MaxRSS=3268572K` ≈ **3.1 GB** for the batch step,
consistent with `direct`'s self-reported 2,577.6 MB plus the shell and `uv` wrapper. The
subprocess harness and the scheduler agree.

### The verdict is measured, not extrapolated

The phase doc says to *extrapolate* peak RSS to N = 6000. **No extrapolation was needed — the
fixture is N = 6,529, above the target.** Peak RSS is **2.5 GB against a 32 GB threshold**, a
12.7x margin. Scaling *down* to N = 6000 gives ~2.4 GB; the conclusion is identical and does not
depend on the scaling.

> ### `S-3 IN-MEMORY`
>
> Peak RSS 2,577.6 MB (2.5 GB) measured at N = 6,529, against a 32 GB threshold — a 12.7x
> margin. `TASK_FINALIZE` uses `pl.concat`. **P5 Task 4 sets `--mem` to 2x projected ≈ 8 GB**
> (2x the 2.5 GB peak, rounded up for the interpreter and headroom).

### The clauses disagreed — and the verdict survives not needing the tie-break

Exactly as anticipated before the numbers existed:

| Clause | Measured | Says |
|---|---|---|
| peak RSS < 32 GB | 2.5 GB — not close | `IN-MEMORY` |
| streaming within 1.5x merge | 0.228 s vs 0.473 s = **0.48x**, twice as fast | `STREAMING` |

The pre-dated precedence resolves it to `IN-MEMORY`. **But the verdict does not actually rest on
that precedence**, because of a result that inverts the usual argument for streaming:

**Streaming used MORE peak memory than the in-memory merge — 421.5 MB against 314.6 MB, 34%
more.** Streaming is normally adopted *because* it bounds memory. Here it does not: at this
scale `sink_parquet` costs a third more peak RSS than `pl.concat` while running 2x faster. So
clause 2 is not detecting "streaming is cheap enough to adopt for safety" — it is detecting a
speed gain bought *with* memory, the opposite of the trade the clause was written to catch.

Neither clause's underlying purpose is served by streaming here, so `IN-MEMORY` is right on its
own terms. A verdict that survives its own tie-break becoming unnecessary is worth more than one
that depends on it.

### The memory lever is sharding, not streaming

`direct` holds **2,577.6 MB** to produce a frame that `merge` produces in **314.6 MB** — the same
128,598 rows and 148 columns, **8.2x less memory.** The difference is not the data: it is that
`direct` holds 6,529 small DataFrames alive simultaneously before concatenating, while `merge`
holds 16. The peak is dominated by per-frame fragmentation overhead, not by row count.

**This is the practically useful finding in S-3, and it points at P5's own machinery.** If a
future tree ever does threaten the memory ceiling, the fix is not `sink_parquet` — which costs
*more* here — it is sharding, which P5 is already building. The shard lane peaks at 609.2 MB,
4.2x below `direct`, because it never holds more than 1/16 of the tree at once.

### Sharding is nearly free to keep

`direct` 10.909 s versus `shard` 11.170 s — **writing 16 shard parquets costs 2.4% over reading
everything once.** Combined with S-2's `K = 1`, that is the reassuring half of the fan-out story:
the sharding path is cheap to retain even while unused, so keeping it against a larger future N
costs almost nothing in wall-clock.

### `mode=uniform` — the fallback never fired

All 6,529 tables share one schema: the uniform multi-path `scan_parquet` succeeded and the
`diagonal_relaxed` fallback was never taken. Combined with **zero bytes on stderr across both
spikes**, this is positive evidence of schema uniformity in this tree rather than an absence of
evidence.

**It also bounds the result further, and P5 should know it.** That fallback exists in production
because real trees *did* hit heterogeneity (`_cli_parquet_agg.py:95`). This fixture cannot
exercise that path — so nothing here says what a heterogeneous tree costs to merge, and the
`diagonal_relaxed` branch remains covered only by its own tests.

### Clause precedence, agreed before the numbers were seen

The phase doc's S-3 verdict has two clauses and **no stated precedence**, and the smoke run
shows they will very likely disagree: `stream` came in at 0.036 s against `merge` at 0.041 s —
not merely within the 1.5x band, but *faster*. If peak RSS also lands in single-digit GB, then
clause one says `IN-MEMORY` and clause two says `STREAMING`.

Resolved by reading the clauses for their purpose rather than as two votes:

| Clause | What it is | Question it answers |
|---|---|---|
| peak RSS < 32 GB | **safety** property | Does `TASK_FINALIZE` OOM at N = 6000? |
| streaming within 1.5x | **cost** property | If we need streaming, is it affordable? |

**The wall-clock clause exists to stop us paying for streaming complexity we do not need. It
cannot also be a reason to adopt it.** So:

- **Peak RSS under 32 GB → `S-3 IN-MEMORY`, regardless of the ratio.**
- The ratio is still recorded, as a finding rather than a vote: *streaming is already at
  parity, so if a future tree pushes peak RSS over the threshold, switching costs nothing in
  wall-clock.* That is useful to P5 and is lost entirely if the ratio is read as a tie-break.
- If peak RSS **approaches** 32 GB, that is a different conversation and it changes P5 Task 4's
  `--mem` — escalate rather than decide here.

This precedence was fixed **before** the full-run numbers existed, so it cannot have been
chosen to fit them.
