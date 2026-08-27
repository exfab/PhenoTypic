# Phase 0 — Spike gate

**Spec:** §2.1, §5.2. **Blocks:** every other phase.

**Deliverable:** a committed findings document answering four questions with evidence, plus
one **measured** number for chunk-size panning. Spec §2.1 is explicit that these are
"answered before the plan is written, not during it" — this plan is written, so they are
answered before phase 1 starts.

> **Why a gate and not a task.** Failure on questions 1 or 2 is not fatal; it moves work
> from "configure Viv" to "adapt Viv", which decision A already permits. But it changes the
> estimate for phases 2 and 3 materially, and an estimate discovered mid-implementation is
> a schedule failure rather than a technical one.

---

### Task 0.1: Build a real store to spike against

**Files:**
- Create: `docs/superpowers/plans/2026-08-26-viewer-viv-rebuild/spike/README.md`

- [ ] **Step 1: Write one store with the real CLI writer**

Not a hand-built fixture — the point is to test what the writer actually emits.

```bash
QT_QPA_PLATFORM=offscreen uv run python -c "
from phenotypic import load_synth_yeast_plate
img = load_synth_yeast_plate()
out = img.save2zarr('/tmp/spike/plate.ome.zarr')
print(out)
"
```
If `save2zarr`'s signature differs, read it first:
`uv run python -c "from phenotypic import Image; help(Image.save2zarr)"`.

- [ ] **Step 2: Record what was written**

```bash
find /tmp/spike/plate.ome.zarr -maxdepth 3 | sort
uv run python -m json.tool /tmp/spike/plate.ome.zarr/zarr.json | head -60
```
Capture both outputs into `spike/README.md`. The root `zarr.json`'s
`attributes.phenotypic` block is the contract every later phase reads; having its real
shape written down stops phases 2-4 guessing at key names.

- [ ] **Step 3: Confirm the four backend clauses this plan leans on**

From the captured JSON, assert by eye and record:
- `phenotypic.series` lists the named series; `OME/zarr.json` carries `series` in the same
  order with the primary first.
- `phenotypic.labels.objmap` records a **resolved path**.
- `phenotypic.pyramid` carries `levels`, `stop_px`, and `downsample`.
- chunk shape, shard shape, codec and chunk-key separator match backend §1.4
  (`(1,1024,1024)` / `(C,4096,4096)` / `zstd` / `"."`).

Any mismatch is a **backend** finding, not a Viv one — file it against the store spec.

---

### Task 0.2: Answer the four spike questions

**Files:**
- Modify: `spike/README.md`

- [ ] **Step 1: Serve the store over plain HTTP with Range**

```bash
cd /tmp/spike && uv run python -m http.server 8099 &
curl -s -I http://localhost:8099/plate.ome.zarr/zarr.json
curl -s -r 0-15 -o /dev/null -w '%{http_code}\n' \
  http://localhost:8099/plate.ome.zarr/rgb/0/c.0.0.0
```
Expected: `200` for the metadata, **`206`** for the ranged chunk read. A `200` on the
second means the server ignored `Range`; note it, because phase 1's whole job is a route
that does not.

- [ ] **Step 2: Q1 — does an unmodified vizarr/Viv resolve our series list?**

Open the store in an unmodified vizarr build pointed at
`http://localhost:8099/plate.ome.zarr`. Record: does the `bioformats2raw.layout` series
list resolve without patching?

Record the **answer and the evidence** (a screenshot path, or the console error verbatim).
"It worked" without evidence is not an answer this gate accepts.

- [ ] **Step 3: Q2 — does `labels/objmap` attach as a label layer?**

Same session. Record whether the label child attaches to the **primary** series
automatically, and whether it does so via the `ome.labels` list or by path convention.

If it resolves by convention (`rgb/labels/objmap`) rather than by reading
`phenotypic.labels.objmap`, that is a **finding that changes phase 3**: the façade must
resolve the path itself and hand Viv an explicit source, because backend §1.1 forbids
hard-coding it and a `gray`-primary store would break.

- [ ] **Step 4: Q3 — does the `"."` chunk-key separator round-trip?**

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  http://localhost:8099/plate.ome.zarr/rgb/0/c.0.0.0
```
Expected `200`/`206`. Then confirm the browser client requests that same flat key rather
than a nested `c/0/0/0` path — check the network panel. A client that nests will 404
against every chunk, and backend §1.4 makes `"."` mandatory store-wide for Windows
`MAX_PATH` reasons, so the **client** is what must adapt.

- [ ] **Step 5: Q4 — does the wasm zstd codec decode a CLI-written chunk?**

This is the one that must not be faked. Register `numcodecs.js`'s zstd via
`zarr.registry.set()` **before opening the store**, then read a real chunk and assert
pixel values against the same chunk read in Python:

```bash
uv run python -c "
import zarr, numpy as np
a = zarr.open_array('/tmp/spike/plate.ome.zarr/rgb/0', mode='r')
print(a.shape, a.dtype)
print(np.asarray(a[0, :4, :4]))
"
```
Record those 16 values and compare against what the browser decodes. Spec §5.1 is explicit:
the test is "the actual read", not "the codec registered".

- [ ] **Step 6: Commit the findings**

```bash
git add docs/superpowers/plans/2026-08-26-viewer-viv-rebuild/spike/
git commit -m "docs(viv): record the spike gate findings"
```

---

### Task 0.3: Measure the chunk-size risk (§5.2)

**Files:**
- Modify: `spike/README.md`

**Interfaces:**
- Produces: the number that either closes spec §5.2's accepted risk or triggers the
  1024² → 512² amendment against backend §1.4.

> Spec §5.2 records 1024² chunks as an **accepted risk, not a verified choice**: a deck.gl
> tile fetch pulls a whole inner chunk, ≈3 MB at 1024²×3×u8, and whether that pans
> acceptably over an SSH tunnel has not been measured. The fallback is cheap and named —
> moving to 512² is a pure §1.4 amendment with **no GUI rework**, because the client reads
> whatever `phenotypic.pyramid` describes.

- [ ] **Step 1: Record the per-tile arithmetic — no script**

> An earlier draft committed `tile_fetch_budget.py` here. **Dropped** (user ruling,
> 2026-08-26): it recomputed constants defined at the top of its own file, so its assertion
> could only fail if someone edited those constants, and it exited 0 regardless — it gated
> nothing. CLAUDE.md mandates a logic-validation script for "a numeric invariant a reader
> would otherwise take on faith"; `1024 × 1024 × 3` is not one. Contrast
> `logic_validation_scripts/2026-08-18-ome-zarr-image-store/ngff_store_geometry.py`, which
> re-derives pyramid level counts and shard divisibility and has already **refuted three
> claims** — that is what the rule is for.

Write into `spike/README.md`:

```text
A deck.gl tile fetch pulls a whole inner chunk.
  1024^2 x 3 channels x 1 byte = 3,145,728 B  ~= 3.0 MB per tile
  512^2  x 3 x 1               =   786,432 B  ~= 0.79 MB per tile
A 1920x1080 viewport at 1024^2 intersects ceil(1920/1024) x ceil(1080/1024)
  = 2 x 2 = 4 tiles  -> ~12 MB cold
                at 512^2: 4 x 3 = 12 tiles -> ~9.4 MB cold
```

The arithmetic is not the risk. **The latency is**, and only step 3 measures it.

- [ ] **Step 3: Measure the real thing over a tunnel**

The script gives the byte budget; only a measurement gives the latency. From a workstation:

```bash
ssh -L 8099:localhost:8099 <user>@cluster
```
then, against the tunnelled store, time a cold cross-plate pan and record: time-to-first-tile,
time-to-full-viewport, and observed throughput. Do it at both 1024² and — by rewriting the
spike store with `--pyramid-levels` unchanged but chunks at 512² — the fallback shape.

- [ ] **Step 4: Decide, and record the decision with its number**

Write into `spike/README.md` one of:

- **Keep 1024².** Record the measured full-viewport time and the threshold it cleared.
- **Amend to 512².** File a recorded amendment against
  `docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md` §1.4, carrying the
  measurement. Note it also shifts that spec's verified file-count table (16 / 40 / 132)
  and quadruples chunk count, so the table is amended in the same edit.

**An amendment without the number is out of order** — spec §5.2 grants this governance
"gated on a measurement" and nothing else. If the measurement cannot be taken (no tunnel
available), record that, keep 1024², and leave the risk open rather than closing it by
assertion.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-08-26-viewer-viv-rebuild/spike/
git commit -m "docs(viv): measure the per-tile fetch budget and settle chunk size"
```

---

### Gate

- [ ] **All four questions answered with evidence, and the chunk-size decision recorded
  with its measurement (or its explicit absence).** Report the findings before starting
  phase 1. If Q1 or Q2 came back "needs patching", say so in the report — phases 2 and 3
  grow, and that is the user's call to absorb or re-scope.
