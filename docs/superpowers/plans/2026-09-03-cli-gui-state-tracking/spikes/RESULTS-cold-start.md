# Cold-start spike results — which cheap start is cheapest (U-11)

Companion to [RESULTS.md](RESULTS.md), which covers the Phase 0 S-2/S-3 spikes. This one
answers a later question and was run separately, so it gets its own file rather than an
appendix under a title that names two other spikes.

**Script:** [`cold_start_options.py`](cold_start_options.py) ·
**Job:** [`cold_start_options.sbatch`](cold_start_options.sbatch)
**Run:** job `28142292`, node `x02`, `COMPLETED 0:0`, elapsed `00:02:34`, 2026-09-04.

---

## The question

U-11 asked which of two changes buys the cold GUI start, and the spec's decision table
records the answer but not the measurement behind the losing option:

| Option | What it does |
|---|---|
| **B** | drop the **overlay** from the completion proof and keep hashing the rest |
| **C** | replace the hash with a `(size, mtime_ns)` **stat** sweep, plus the on-disk cache tier |

**C shipped.** This is the evidence, and B's number is the part that was not written down.

## Why it had to be a Slurm job

The question is about a **cold GPFS client cache**, and the interactive node's cache was
fully warm — every earlier measurement in the session had walked the whole tree, so a repeat
there reads ~0.04 ms/stat and answers a different question. The client cache is **per node**,
so a fresh compute node is cold by construction. That was the only instrument available: the
caches cannot be dropped without root, and P0 established there is no second fixture tree of
this size ([`spike-fixture-trees`](RESULTS.md#the-fixture-tree)).

## The design fix that makes the numbers comparable

**The first attempt measured B and then C in one process over the same paths.** B ran against
a cold cache and C against the one B had just warmed, so C's advantage was partly just having
gone second — the same page-cache confound that forced S-2 to be rewritten.

The fix, borrowed from S-2: give each lane a **disjoint, seeded half** of the markers, so both
read paths the other never touched.

```python
random.Random(20260904).shuffle(markers)
half = len(markers) // 2
lanes = {"first": markers[:half], "second": markers[half:]}
```

Re-derived from the markers afterwards, the halves are near-identical in composition, which is
what licenses doubling a half to describe the tree:

```
markers total = 6657   first = 3328   second = 3329

first:  all3=9918  no_ov=6590  unparseable=0
      3262 x ('measurements', 'overlay', 'store')
        66 x ('overlay', 'store')

second: all3=9925  no_ov=6596  unparseable=0
      3267 x ('measurements', 'overlay', 'store')
        62 x ('overlay', 'store')
```

*(~1.9 % of markers carry no `measurements` artifact. Noted, not load-bearing here.)*

---

## Raw output

```
node=x02 job=28142292 start=2026-09-04T16:07:04-07:00
C lane=first  seconds=   18.5 files=9918 per_file_ms=1.9
B lane=second seconds=  128.2 files=6596 bytes=0.30GB per_file_ms=19.4
end=2026-09-04T16:09:34-07:00
```

### The two `files=` counts are not unequal halves — read them before comparing

`9918` and `6596` look like a botched split of one population. They are not. **The lanes
denominate different things**, and the script is right to:

- **C** stats `all3` — *every* artifact, 3 per marker → `9918`
- **B** hashes `no_ov` — artifacts *except* the overlay, 2 per marker → `6596`

Both counts reproduce exactly from the markers (table above). So `per_file_ms` compares
like with like — the cost of touching one file — but the printed line is **not** an option
comparison, because the two options do different amounts of work per image.

## The arithmetic, per marker

Per marker is the unit that matters: it is what a completion check costs for one image.

| | per file | files/marker | **per marker** | full tree (6,657) |
|---|---|---|---|---|
| **C** — stat all 3 | 1.87 ms | 3 | **5.56 ms** | **37.0 s** |
| **B** — hash 2, no overlay | 19.44 ms | 2 | **38.51 ms** | **256 s** |
| ratio | 10.4× | | **6.9×** | 6.9× |

**Everything past the first two columns is extrapolation from one cold half** — doubled, not
measured. Stated as such because the two halves are compositionally near-identical, not
because doubling is free.

### The extrapolation lands on an independently measured number

C's full-tree figure comes out at **37.0 s**, against the **~37 s** stat sweep measured
separately and cited in `design.md` U-11. Two cold measurements, different runs, different
halves, agreeing to the tenth of a second. That is corroboration of the shipped number, and
it is the strongest thing in this file.

---

## Verdict

Placed beside the existing figure for today's behaviour (hash every artifact, 73.6 GB →
**1403 s**):

| | cold, 6,657 images | vs today |
|---|---|---|
| **today** — hash all 3 | 1403 s | — |
| **B** — drop the overlay | ~256 s | 5.5× |
| **C** — stat sweep *(shipped)* | ~37 s | **38×** |

**C wins by ~6.9× over B**, and B was the option that looked nearly free.

### Why B is so much worse than its byte count predicts

B hashes **0.30 GB per half — 0.60 GB for the tree — and still costs 256 s.** Today's pass
hashes 73.6 GB. So dropping the overlay removes **~99 % of the bytes and only ~82 % of the
time.**

**Hashing a small file on GPFS costs ~10× a `stat` almost regardless of its size**, because
the price is the `open` + `read` round trip, not the bytes moved. B's remaining cost is
per-file syscalls on 13,186 small files.

That is the transferable result, and it is worth more than the ranking: **on this filesystem,
any per-file verification is dominated by file *count*, not volume.** An optimisation that
shrinks bytes without shrinking file touches will underdeliver, and B is the worked example —
99 % of the data removed, 5.5× returned.

It is also the reason the stat sweep is not merely "the fast option" but a different cost
class: it pays one metadata lookup per file and never opens anything.
