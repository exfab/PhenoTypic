# Spike gate findings — phase 0

Run 2026-08-27 against the landed CLI writer. Environment: node v24.16.0, npm 11.13.0,
Playwright chromium present, npm registry reachable, `zarr` 3.1.5.

## Task 0.1 — a store written by the real writer

Two stores, because the first turned out to be an invalid subject:

| Store | Source | Extent | Levels |
|---|---|---|---|
| `plate.ome.zarr` | `load_synth_yeast_plate()` | 800×600 | 2 |
| `big.ome.zarr` | synthetic `GridImage(nrows=32, ncols=48)` | **4000×3000** | **4** |

**The synth yeast plate cannot serve as the spike subject.** At 800×600 the chunk clamps to
the whole image (`chunk_shape [3,600,800]`), so it exercises neither the 1024² inner chunk
nor sharding. Spec §1.4's reference plate is 4000×3000; that is what the numbers below use.

### Backend §1.4 — every claim confirmed on the reference plate

```text
shape          [3, 3000, 4000]
shard shape    [3, 4096, 4096]      <- §1.4
inner chunk    [1, 1024, 1024]      <- §1.4
inner codecs   bytes, zstd          <- decision C1
key encoding   {"name": "default", "configuration": {"separator": "."}}
pyramid        levels 4, stop_px 512, downsample {image: mean, label: nearest}
```

`levels 4` at 4000 px matches `ceil(log2(4000/512)) + 1`, the ladder whose `floor`/`ceil`
boundary backend §1.3 records as having failed once.

### Q3 — does the `"."` separator round-trip? **ANSWERED: yes.**

```text
rgb/0/c.0.0.0     3-D  (channel, y, x)
gray/0/c.0.0      2-D  (y, x)
```

Confirms the plan's round-2 corrected literals (FLOW-13). One path segment per chunk key, as
backend §1.4 requires for Windows `MAX_PATH`.

---

## FINDING — spec §1.4's "verified" file-count table is wrong

§1.4 states as **verified** that 4 levels (auto) yields **40** files: 16 data + 24 metadata.
Measured on the reference plate:

```text
total 36  =  12 data  +  23 zarr.json  +  1 METADATA.ome.xml
```

**The gap is the objmap.** Zarr does not write a chunk whose contents equal `fill_value`, and
a Stage-1 objmap is all zeros — so the label levels contain **zero** chunk files, not four:

```text
rgb            L0=1 L1=1 L2=1 L3=1
gray           L0=1 L1=1 L2=1 L3=1
detect_mat     L0=1 L1=1 L2=1 L3=1
labels/objmap  L0=0 L1=0 L2=0 L3=0      <- sparse; the table assumed 4
```

Not a defect — sparse storage is correct and strictly better. But the table is presented as
verified and is off by 4 for **every image between Stage 1 and Stage 3**, which backend §3.3
guarantees carries a zeros objmap. The inode budget built on it (~40/image, 400k at 10k
images) is conservative by ~10%.

**Recorded for a backend-spec amendment. Not blocking, and it makes the format cheaper, not
dearer.**

---

## Task 0.2 — Range. The measurement that existed in no document.

One 16-byte ranged request for `big.ome.zarr/rgb/0/c.0.0.0`:

| Server | Code | Downloaded |
|---|---|---|
| `python -m http.server` | `200` | **36,045,031 B** (34.4 MiB) |
| Flask `send_file(conditional=True)` | `206` | **16 B** |

`SimpleHTTPRequestHandler` has **no Range support at all** — it ignores the header and sends
the whole file. This is exactly why round 2 split this step: an earlier draft served the
spike store with `http.server` and expected `206`, which is unreachable, and drove Q1/Q2/Q4
against that same server — measuring the no-Range regime throughout without recording a
single byte count.

### Shard amplification, measured

At 4000×3000 the level-0 `rgb` shard is a **single 34.4 MiB file**. A deck.gl tile fetch
wants one 1024² inner chunk ≈ 3.15 MB (`1024×1024×3×1`):

```text
34.4 MiB / 3.0 MiB  ≈  11.5x  amplification per tile, without Range
```

So `conditional=True` is not a nicety — it is the difference between a 3 MB and a 34 MB tile
fetch. That is the justification phase 1 needs, and the number §5.2's "accepted risk" was
missing.

Below the brief's 96 MB worst case only because the shard clamps to the array; a full
4096×4096 shard over a larger plate would reach it.

---

## Q1, Q2, Q4 — pending

Require a browser running Viv against a Range-capable server. The bundle build is phase 2's
deliverable and is being done now so these can be answered against the real artifact rather
than a CDN stand-in.

- **Q1** — does an unmodified vizarr/Viv resolve *our* `bioformats2raw.layout` series list?
- **Q2** — does `labels/objmap` attach to the primary series as a label layer?
- **Q4** — does the wasm zstd codec decode a **CLI-written** chunk (the actual read, not
  merely a successful registration)?

## §5.2 chunk-size decision — pending

Needs a cold-pan measurement over a real SSH tunnel at 1024² and 512². The governance is
explicit: the backend spec may be amended from 1024² to 512² **only** gated on a
measurement. Not yet taken.
