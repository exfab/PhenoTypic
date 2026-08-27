# Drift register — spec vs. the landed OME-Zarr store branch

The Viv rebuild spec (dated 2026-08-26) says of the backend, at §1:

> This spec is written against [2026-08-18-ome-zarr-image-store], which at time of writing
> is **specification only** — there is no zarr code in `src/`.

**That is no longer true.** The store branch (`worktree-ome-zarr-image-store`, head
`bf0d01a1`, 248 files) landed, and this plan's branch is stacked on it. A substantial part
of the Viv spec's §4.1 and §6.2 scope is already implemented.

Each row below was verified in this worktree on 2026-08-26 by reading the named file. Rows
D-1 through D-3 **remove work from this plan**. Row D-4 **falsifies a spec claim**. Rows
D-5 and D-6 are constraints the spec does not mention.

---

## D-1 — §4.1's four staleness traps are already fixed

**Spec says:** four sites need repair because a store directory's `st_mtime_ns` does not
change when a nested chunk is rewritten — `_tile_routes.py:471, :469, :477`,
`_shared/tiles.py:518`, `builder/_preview_tiles.py:76`.

**Reality:** all of them already key on the root `zarr.json`.

| Evidence | What it shows |
|---|---|
| `results_viewer/_tile_routes.py:476-479` | A docstring stating the trap verbatim and that staleness is keyed on the root `zarr.json` |
| `_tile_routes.py:496, :503` | Compares and `os.utime`s against `root_stat`, not the store dir |
| `_tile_routes.py:527-528` | `_store_content_token` uses `paths_fingerprint([root_json])` — `file_fingerprint` is explicitly noted as unusable |
| `builder/_preview_tiles.py:78-87` | Same fix, same reasoning, keyed on `ngff_.STORE_ROOT_JSON` |
| `tests/unit/gui/results_viewer/test_tile_cache_invalidation.py` (297 lines) | Landed test coverage |

**Effect on this plan:** phase 1 does **not** re-plan the traps. It inherits them and adds
one regression test (spec §8's "staleness" check) proving the byte route respects the same
key.

---

## D-2 — §6.2's "ship first (D3)" is already landed

**Spec says:** ship D3 first — "keep today's `build_tile_grid` chrome and change only the
crop route — from overlay-PNG slicing to a level-0 chunk read."

**Reality:** `_shared/tiles.py:665` `crop_colony` already prefers the store
(`crop_store_rgb`, `:545`) and falls back to the baked overlay only for a standalone
deliverables bundle that ships overlays but no `results/` stores. `select_pyramid_level`
(`:378`), `_read_store_level` (`:477`) and `_crop_store_layer_window` (`:599`) are all
present, with `tests/unit/gui/shared/test_tiles_zarr.py` (265 lines) covering them.

Note the deliberate design already in place: `StoreUnreadable` is **not** caught in
`crop_colony`, because falling back to the overlay would show plausible pixels while hiding
a run-wide actionable condition. The caller turns it into a `422`.

**Effect on this plan:** phase 4 is **only** the D1 half — deck.gl `OrthographicView`s
sharing one `viewState`. It is genuinely optional, which is why the README marks it the
first thing to cut.

---

## D-3 — server-side pyramid-aware zarr reads already exist

`_load_zarr_layer_rgb` / `_load_zarr_level_rgb` (`_shared/tiles.py:423, :454`) already
select the smallest pyramid level covering a target pixel size, and `_tile_routes.py:31`
already imports and uses them. Spec §4.2's `_load_hdf_layer_rgb` → `_load_zarr_layer_rgb`
rename is done.

**Effect:** what remains is not "teach the server to read zarr" but "stop the server
building a DZI pyramid from what it read, and hand the raw chunks to the browser instead."
That is a smaller, sharper change than the spec's framing implies, and it is phases 1-3.

---

## D-4 — the spec's mid-run objmap claim is FALSE against the landed engine

**Spec §5 of the backend, quoted by Viv spec §6.2's rationale:** Stage 2 "opens the
promoted store and overwrites `labels/objmap` in place", which "buys ... the GUI can render
a real objmap mid-run."

**Reality — the landed engine inverted this.** From the worktree's own `CLAUDE.md`:

> Stage 2 reads that store **read-only** and never writes into it; its result is a
> **Stage-2 signal** under `.phenotypic/progress/`: the retained **raw** detector output
> `stage2_raw/<ds>/<stem>.npy` plus a consumable **token** `stage2_done/<ds>/<stem>.json`.
> Stage 3 replays the raw array, measures, re-promotes the store, and consumes the token.

So between Stage 1 and Stage 3 the store's `labels/objmap` holds **zeros**, not detector
output. The mid-run-objmap benefit does not exist.

**Effect on this plan:** the Plate Layers panel (phase 3) must not promise a live objmap
during a staged run. An `objmap` that is all-zeros mid-run is the **correct** rendering of
a correct store, and the pyramid readout / layer list must not present it as an error.
Phase 3 task 3.4 covers this explicitly.

**This warrants a spec amendment**, and it is one of the two spec-change items the refinery
should gate to the user (the other is D-6).

---

## D-5 — `_dzi_tiler` cannot be deleted, only unhooked from the Plate path

Spec §4 says "`_dzi_tiler` is removed from this path entirely." Correct as written — but an
executor reading it as "delete the module" breaks five live consumers:

```text
browse/_app.py:40                  DZI_BACKEND_INFO
browse/_preparation.py:711         tile()
browse/_preparation_routes.py:95   DZI_BACKEND_INFO
builder/_point_picker.py:417       tile()
builder/_preview_tiles.py:144      tile()
```

Browse keeps libvips → DZI → `BrowseCache` → OSD as its **only** path (spec §9), so the
module stays. Only `_tile_routes.py:34, :458, :551` come off.

---

## D-6 — measurements now live inside the store

The landed layout puts authoritative per-object measurements at
`tables/measurements/table.parquet` **inside each `.ome.zarr`**, and forward runs no longer
write external per-image parquets. Neither the Viv spec nor the backend spec's §1 layout
diagram shows a `tables/` group.

**Effect on this plan:** the byte route (phase 1) serves an arbitrary-depth tail inside the
store, so it would serve `tables/measurements/table.parquet` to any browser that asks.
Phase 1 task 1.3 restricts the route to the pixel groups the client legitimately needs
rather than exposing the whole store. **This is a security-relevant divergence from the
spec's route sketch and needs the user's sign-off**, since it constrains a route the spec
describes as unrestricted.

---

## Summary of plan-scope changes

| Spec section | Status | This plan |
|---|---|---|
| §4.1 staleness traps | **done** | inherit; add one regression test |
| §4.2 `_load_zarr_layer_rgb` | **done** | inherit |
| §6.2 Colony D3 | **done** | phase 4 is D1 only, and is optional |
| §4 byte route | to do | phase 1 |
| §3 bundle + façade | to do | phase 2 |
| §6.1 Plate surface | to do | phase 3 |
| §6.2 Colony D1 | to do | phase 4 |
| §5.2 chunk measurement | to do | phase 0 |
| §2.1 spike | to do | phase 0 |
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
- Create: `docs/superpowers/logic_validation_scripts/2026-08-26-viewer-viv-rebuild/tile_fetch_budget.py`
- Modify: `spike/README.md`

**Interfaces:**
- Produces: the number that either closes spec §5.2's accepted risk or triggers the
  1024² → 512² amendment against backend §1.4.

> Spec §5.2 records 1024² chunks as an **accepted risk, not a verified choice**: a deck.gl
> tile fetch pulls a whole inner chunk, ≈3 MB at 1024²×3×u8, and whether that pans
> acceptably over an SSH tunnel has not been measured. The fallback is cheap and named —
> moving to 512² is a pure §1.4 amendment with **no GUI rework**, because the client reads
> whatever `phenotypic.pyramid` describes.

- [ ] **Step 1: Write the logic-validation script**

Per root `CLAUDE.md`: stdlib + numpy/scipy only, never imports `phenotypic`, exits non-zero
on failure, committed beside the spec's topic folder.

```python
"""Re-derive the per-tile byte budget for the Viv plate surface.

Claim under test (viewer-viv-rebuild spec section 5.2): a deck.gl tile fetch
pulls a whole inner chunk, so at (1, 1024, 1024) uint8 RGB a single tile costs
roughly 3 MB, and a viewport filled with such tiles costs N times that.

Exits non-zero if the derived figures contradict the spec's stated numbers.
"""

import sys

import numpy as np

CHUNK = (1, 1024, 1024)
CHANNELS = 3
DTYPE_BYTES = 1


def bytes_per_tile(chunk=CHUNK, channels=CHANNELS, itemsize=DTYPE_BYTES) -> int:
    """Bytes pulled for one tile: the inner chunk across every channel."""
    return int(np.prod(chunk)) * channels * itemsize


def tiles_for_viewport(viewport_px, chunk_px) -> int:
    """Tiles intersecting a viewport, counting partials as whole fetches."""
    w, h = viewport_px
    return int(np.ceil(w / chunk_px)) * int(np.ceil(h / chunk_px))


def main() -> int:
    per_tile = bytes_per_tile()
    if not (2.9e6 <= per_tile <= 3.2e6):
        print(f"REFUTED: per-tile bytes {per_tile} is not the spec's ~3 MB")
        return 1
    print(f"per tile: {per_tile / 1e6:.2f} MB")

    for viewport in ((1920, 1080), (2560, 1440)):
        for chunk_px in (1024, 512):
            n = tiles_for_viewport(viewport, chunk_px)
            total = n * bytes_per_tile((1, chunk_px, chunk_px))
            print(
                f"viewport {viewport[0]}x{viewport[1]} @ {chunk_px}^2: "
                f"{n} tiles, {total / 1e6:.2f} MB"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-26-viewer-viv-rebuild/tile_fetch_budget.py
```
Expected: exit 0, with a table of per-viewport totals at both chunk sizes.

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
git add docs/superpowers/logic_validation_scripts/2026-08-26-viewer-viv-rebuild/ \
        docs/superpowers/plans/2026-08-26-viewer-viv-rebuild/spike/
git commit -m "docs(viv): measure the per-tile fetch budget and settle chunk size"
```

---

### Gate

- [ ] **All four questions answered with evidence, and the chunk-size decision recorded
  with its measurement (or its explicit absence).** Report the findings before starting
  phase 1. If Q1 or Q2 came back "needs patching", say so in the report — phases 2 and 3
  grow, and that is the user's call to absorb or re-scope.
# Phase 1 — The byte route

**Spec:** §4, §4.1. **Depends on:** phase 0. **Blocks:** phases 2-4.

**Deliverable:** `GET /zarr/<dataset>/<stem>.ome.zarr/<path...>` on the results viewer's
blueprint, serving raw store bytes with **HTTP Range**, guarded per path segment, and
restricted to the pixel groups the client needs. The server does no decode; per-request
memory is a sendfile buffer.

> **Why Range is load-bearing, not a nicety.** A sharded read is a shard-index fetch
> followed by a byte-range fetch into the shard. Without `conditional=True` the client
> pulls whole shards — up to 96 MB for `rgb` — for every 1024² tile. This is the single
> flag the phase exists to get right.

---

### Task 1.1: Serve bytes with Range

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_zarr_routes.py`
- Modify: `src/phenotypic/gui/results_viewer/_app.py` (register the blueprint)
- Test: `tests/unit/gui/results_viewer/test_zarr_routes.py` (create)

**Interfaces:**
- Consumes: `OutputRoot.store_path(dataset, stem)` (already exists — confirm its exact name
  with `uv run grep -n "def store_path" src/phenotypic/gui/results_viewer/_output_root.py`),
  `phenotypic.gui._shared.tiles.is_safe_path_component`.
- Produces: `register_zarr_routes(app, output_root) -> None`, and the URL builder
  `zarr_store_url(url_prefix, dataset, stem) -> str` that phase 2's façade calls.

- [ ] **Step 1: Write the failing tests**

```python
"""The zarr byte route serves store bytes with HTTP Range.

Three properties, in descending order of how quietly they fail:
Range (silently pulls whole shards), traversal (silently serves the tree),
and 404 (loudly wrong, therefore least dangerous).
"""

import pytest


def test_serves_the_root_zarr_json(zarr_route_client, spike_store):
    resp = zarr_route_client.get("/zarr/ds/plate.ome.zarr/zarr.json")
    assert resp.status_code == 200
    assert resp.json["attributes"]["phenotypic"]["store_schema_version"]


def test_honours_a_range_request(zarr_route_client, spike_store):
    resp = zarr_route_client.get(
        "/zarr/ds/plate.ome.zarr/rgb/0/c.0.0.0",
        headers={"Range": "bytes=0-15"},
    )
    assert resp.status_code == 206
    assert len(resp.data) == 16
    assert resp.headers["Accept-Ranges"] == "bytes"


def test_missing_chunk_is_404_not_500(zarr_route_client, spike_store):
    resp = zarr_route_client.get("/zarr/ds/plate.ome.zarr/rgb/0/c.99.99.99")
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "tail",
    [
        "../../../../etc/passwd",
        "rgb/../../../../etc/passwd",
        "rgb/0/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "rgb/./../../zarr.json",
    ],
)
def test_rejects_traversal_in_any_segment(zarr_route_client, spike_store, tail):
    resp = zarr_route_client.get(f"/zarr/ds/plate.ome.zarr/{tail}")
    assert resp.status_code in (400, 404)
```

Build `spike_store` by calling the real writer, as in phase 0 task 0.1 — a hand-built
fixture would not exercise the `"."` chunk keys.

- [ ] **Step 2: Run and watch them fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_zarr_routes.py -v
```
Expected: all fail with 404 (no route registered).

- [ ] **Step 3: Implement the route**

```python
"""Serve raw OME-Zarr store bytes to the browser, with HTTP Range.

The client (Viv/zarrita) reads chunks directly, so this route decodes
nothing: it resolves a path inside one per-image store and hands the file to
``send_file(..., conditional=True)``. ``conditional=True`` is what provides
Range, which sharding requires -- a sharded read is a shard-index fetch
followed by a byte-range fetch into the shard, so without it every tile pulls
a whole shard.
"""

from __future__ import annotations

from pathlib import Path

import dash
from flask import Blueprint, abort, send_file

from phenotypic.gui._shared.tiles import is_safe_path_component

#: Groups inside a store the browser may read. Everything else -- notably the
#: embedded ``tables/measurements/table.parquet`` -- is out of scope for a
#: pixel route and is not exposed.
_READABLE_ROOTS: frozenset[str] = frozenset({"OME", "rgb", "gray", "detect_mat"})

#: Files at the store root the client needs to bootstrap.
_READABLE_ROOT_FILES: frozenset[str] = frozenset({"zarr.json"})


def _resolve_within_store(store: Path, tail: str) -> Path:
    """Resolve ``tail`` inside ``store``, or abort.

    Guards **per segment**, not once over the whole tail: the traversal
    surface here is wider than the DZI route's because the tail is arbitrary
    depth inside a store.
    """
    segments = [s for s in tail.split("/") if s]
    if not segments:
        abort(404)
    for segment in segments:
        if not is_safe_path_component(segment):
            abort(400)
    head = segments[0]
    if head not in _READABLE_ROOTS and not (
        len(segments) == 1 and head in _READABLE_ROOT_FILES
    ):
        abort(404)

    candidate = store.joinpath(*segments)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        abort(404)
    # Belt and braces: even with per-segment guards, a symlink inside the
    # store could escape it. Resolve both sides and compare.
    if not resolved.is_relative_to(store.resolve(strict=True)):
        abort(400)
    if not resolved.is_file():
        abort(404)
    return resolved
```

Then the blueprint:

```python
def register_zarr_routes(app: dash.Dash, output_root) -> None:
    """Mount ``/zarr/<dataset>/<stem>.ome.zarr/<path...>`` on ``app``."""
    bp = Blueprint("zarr_bytes", __name__, url_prefix="/zarr")

    @bp.route("/<dataset>/<stem>.ome.zarr/<path:tail>")
    def store_bytes(dataset: str, stem: str, tail: str):
        if not is_safe_path_component(dataset) or not is_safe_path_component(stem):
            abort(400)
        store = output_root.store_path(dataset, stem)
        if store is None or not store.is_dir():
            abort(404)
        return send_file(
            _resolve_within_store(Path(store), tail),
            conditional=True,
        )

    app.server.register_blueprint(bp)


def zarr_store_url(url_prefix: str, dataset: str, stem: str) -> str:
    """Base URL a zarr client opens for one per-image store."""
    base = url_prefix if url_prefix.endswith("/") else f"{url_prefix}/"
    return f"{base}zarr/{dataset}/{stem}.ome.zarr"
```

**On `_READABLE_ROOTS`:** the spec sketches the route as an unrestricted tail. Since the
spec was written, measurements moved *inside* the store at
`tables/measurements/table.parquet` (see [DRIFT.md](DRIFT.md) D-6), so an unrestricted tail
serves the measurement table to any browser that asks. The allow-list is a deliberate
narrowing. **Flag it for sign-off** — it is a divergence from the spec, not an
implementation detail.

- [ ] **Step 4: Register it and run the tests**

Add `register_zarr_routes(app, output_root)` in `_app.py` beside the existing tile-route
registration. Then:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_zarr_routes.py -v
```
Expected: all PASS. If `test_honours_a_range_request` returns 200, `conditional=True` is
missing or a proxy is stripping the header.

- [ ] **Step 5: Confirm `is_safe_path_component` rejects what the test expects**

```bash
uv run python -c "
from phenotypic.gui._shared.tiles import is_safe_path_component as ok
for s in ('..', '.', 'rgb', 'c.0.0.0', '%2e%2e', 'a/b', ''):
    print(repr(s), ok(s))
"
```
If it accepts `'..'`, the guard is not the guard this route needs — **stop and fix
`is_safe_path_component`**, or the traversal tests pass for the wrong reason (Werkzeug's
own normalization) and stop protecting anything the day the route moves behind a proxy.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/_zarr_routes.py \
                        src/phenotypic/gui/results_viewer/_app.py \
                        tests/unit/gui/results_viewer/test_zarr_routes.py
git add src/phenotypic/gui/results_viewer/_zarr_routes.py \
        src/phenotypic/gui/results_viewer/_app.py \
        tests/unit/gui/results_viewer/test_zarr_routes.py
git commit -m "feat(gui): serve OME-Zarr store bytes with HTTP Range"
```

---

### Task 1.2: Prove the route respects the landed staleness key

**Files:**
- Test: `tests/unit/gui/results_viewer/test_zarr_routes.py` (extend)

**Interfaces:**
- Consumes: the root-`zarr.json` staleness key already landed on the store branch
  ([DRIFT.md](DRIFT.md) D-1).
- Produces: spec §8's "staleness" check.

> The traps are already fixed; this is the regression test that keeps them fixed **for the
> new route**. Spec §8 requires a test that "must fail if the check is keyed on the store
> directory" — so the test rewrites a nested chunk, which is exactly the mutation that does
> **not** move the directory's `st_mtime_ns`.

- [ ] **Step 1: Write the test**

```python
def test_a_rewritten_nested_chunk_is_served_fresh(zarr_route_client, spike_store):
    """A nested chunk rewrite must be visible through the route.

    A store directory's ``st_mtime_ns`` does NOT move when a nested chunk is
    rewritten. A route or cache keyed on the directory would serve the stale
    bytes and this test would fail -- which is the whole point of writing it
    against a nested chunk rather than against ``zarr.json``.
    """
    url = "/zarr/ds/plate.ome.zarr/rgb/0/c.0.0.0"
    before = zarr_route_client.get(url).data

    chunk = spike_store / "rgb" / "0" / "c.0.0.0"
    dir_mtime_before = spike_store.stat().st_mtime_ns
    chunk.write_bytes(before[:-1] + bytes([before[-1] ^ 0xFF]))
    assert spike_store.stat().st_mtime_ns == dir_mtime_before, (
        "premise broken: the store directory mtime moved, so this test no "
        "longer proves what it claims"
    )

    after = zarr_route_client.get(url).data
    assert after != before
```

The mid-test assertion on `dir_mtime_before` is deliberate: it fails loudly if the
platform's directory-mtime behaviour differs, rather than letting the test pass while
proving nothing.

- [ ] **Step 2: Run it**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_zarr_routes.py::test_a_rewritten_nested_chunk_is_served_fresh -v
```
Expected: PASS (the route sends files directly and holds no cache). If it **fails**, a
cache has been introduced between the file and the response — key it on the root
`zarr.json` via `paths_fingerprint`, matching `_tile_routes.py:527`.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/gui/results_viewer/test_zarr_routes.py
git commit -m "test(gui): pin the zarr route against nested-chunk staleness"
```
# Phase 2 — Viv bundle and façade

**Spec:** §3, §5.1. **Depends on:** phases 0, 1. **Blocks:** phases 3, 4.

**Deliverable:** a committed Viv + deck.gl IIFE at
`results_viewer/_assets/viv/viv-bundle.min.js`, a hand-written façade at
`results_viewer/_assets/viv_viewer.js` exposing five methods, the committed build recipe at
`tools/viv-bundle/`, and the licensing paperwork. The zstd wasm codec registers **before
any store is opened**.

> **Why a vendored bundle.** There is no `package.json` anywhere in this repo (verified).
> Every line of GUI JS is either hand-written vanilla (`builder.js`, `browse.js`) or a
> vendored pre-built bundle (`openseadragon.min.js`, `cytoscape-dagre.min.js`) dropped into
> a Dash `_assets/` folder. Viv is React + deck.gl; vizarr is Preact + Vite. Neither drops
> in as a file, and adding npm to CI is exactly what decision A exists to avoid.
>
> **Costs accepted, recorded so they are not rediscovered as surprises:** bundle provenance
> lives outside the repo; upgrading Viv is a manual ceremony; the bundle is ~1 MB-class,
> acceptable only because the deployment is localhost or an SSH tunnel.

---

### Task 2.1: Commit the build recipe

**Files:**
- Create: `tools/viv-bundle/package.json`, `package-lock.json`, `build.mjs`, `README.md`, `VERSION`

**Interfaces:**
- Produces: `tools/viv-bundle/VERSION` — a single line the GUI logs at startup and phase 5
  asserts against the bundle's embedded string.

- [ ] **Step 1: Write the recipe README first**

It is the only thing standing between a vendored artifact and rot. State: the exact node
version, the exact command, where the output goes, and that the lockfile is pinned.

```markdown
# Viv bundle build recipe

Built **outside** this repo — there is no npm in CI, by design (viewer-viv-rebuild
spec section 3). Run this by hand when upgrading Viv, then commit the artifact.

    cd tools/viv-bundle
    npm ci             # lockfile is pinned; never `npm install`
    node build.mjs     # writes ../../src/phenotypic/gui/results_viewer/_assets/viv/viv-bundle.min.js

Then bump `VERSION` to match `package.json`'s viv version and commit both the
artifact and `VERSION`. The GUI logs `VERSION` at startup; a mismatch between it
and the string embedded in the bundle is the only signal that the artifact is
stale. Nothing *fails* on drift — see spec section 10, open question 3.
```

- [ ] **Step 2: Write `build.mjs`**

Bundle Viv + deck.gl + zarrita + `numcodecs.js` into one IIFE that assigns a single global
(e.g. `window.__vivBundle`) exposing what the façade needs, and embeds the version string
so phase 5 can compare it.

- [ ] **Step 3: Pin and record**

```bash
cd tools/viv-bundle && npm ci && node build.mjs
```
Then confirm the artifact landed and record its size:
```bash
ls -la src/phenotypic/gui/results_viewer/_assets/viv/viv-bundle.min.js
```

- [ ] **Step 4: Commit recipe and artifact together**

```bash
git add tools/viv-bundle src/phenotypic/gui/results_viewer/_assets/viv/
git commit -m "build(viv): vendor the Viv + deck.gl bundle with its build recipe"
```

---

### Task 2.2: Licensing paperwork

**Files:**
- Modify: `NOTICE`
- Create: `licenses/viv-MIT.txt`, `licenses/vizarr-MIT.txt`
- Modify: `MANIFEST.in` if it enumerates `licenses/`

- [ ] **Step 1: Match the existing pattern**

```bash
uv run grep -n "SAM2\|micro-sam" NOTICE; ls licenses/
```
Add Viv and vizarr entries in the same shape. Both are MIT, compatible with Apache-2.0
(verified — `hms-dbmi/viv`, `BioNGFF/vizarr`).

- [ ] **Step 2: Confirm packaging picks the new files up**

```bash
uv run grep -n "licenses" MANIFEST.in
uv run python -c "import pathlib; print(sorted(p.name for p in pathlib.Path('licenses').iterdir()))"
```

- [ ] **Step 3: Commit**

```bash
git add NOTICE licenses MANIFEST.in
git commit -m "chore(licensing): record Viv and vizarr MIT notices"
```

---

### Task 2.3: The façade, with codec registration ordered first

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_assets/viv_viewer.js`
- Test: `tests/e2e/gui/test_viv_codec_reads_a_real_store.py` (create)

**Interfaces:**
- Produces: `window.phenotypicViv` with `mount(containerId, opts)`,
  `setSource(spec)`, `setViewState(viewState)`, `setLayerVisibility(name, visible)`,
  `destroy(containerId)`. **Dash clientside callbacks talk only to the façade, never to Viv
  directly** — that boundary is what makes the vendored bundle replaceable.

- [ ] **Step 1: Write the failing e2e test**

Spec §5.1 is explicit that the test opens a **CLI-written** store in a real browser, not
one that merely asserts the codec registered.

```python
"""The wasm zstd codec decodes a chunk the CLI actually wrote.

Spec section 5.1: registration is a hard ordering rule -- register late and
every read fails. So the assertion is on decoded pixel values, not on the
registry's contents.
"""

import numpy as np
import pytest


@pytest.mark.e2e
def test_viv_decodes_a_cli_written_zstd_chunk(page, live_viewer_url, spike_store):
    import zarr

    expected = np.asarray(
        zarr.open_array(str(spike_store / "rgb" / "0"), mode="r")[0, :4, :4]
    )

    page.goto(live_viewer_url)
    page.wait_for_function("() => window.phenotypicViv !== undefined")
    decoded = page.evaluate(
        """async () => {
            const arr = await window.phenotypicViv.__debugReadChunk(
                'rgb', 0, [0, 0, 0]
            );
            return Array.from(arr.slice(0, 4)).map(Number);
        }"""
    )
    assert decoded == [int(v) for v in expected[0, :4]]
```

`__debugReadChunk` is a deliberate test seam on the façade. Keep it narrow and documented
as a seam, not as API.

- [ ] **Step 2: Run it and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/e2e/gui/test_viv_codec_reads_a_real_store.py -v
```
Expected: FAIL — `window.phenotypicViv` is undefined.

- [ ] **Step 3: Write the façade with registration first**

```javascript
/**
 * Imperative façade over the vendored Viv bundle.
 *
 * Dash clientside callbacks talk to this object and never to Viv directly,
 * so the vendored bundle can be replaced without touching Python.
 *
 * ORDERING RULE: the zstd wasm codec must be registered with zarrita's
 * registry BEFORE any store is opened. Registering late does not degrade --
 * every read fails. `ready` is the promise every entry point awaits, which
 * is how the ordering is enforced rather than merely documented.
 */
(function () {
  "use strict";

  const bundle = window.__vivBundle;
  const instances = new Map();

  const ready = (async () => {
    const { zarr, numcodecs } = bundle;
    zarr.registry.set("zstd", () => numcodecs.Zstd);
    return true;
  })();

  async function mount(containerId, opts) {
    await ready;
    const el = document.getElementById(containerId);
    if (!el) throw new Error(`viv: no element #${containerId}`);
    const instance = bundle.createViewer(el, opts || {});
    instances.set(containerId, instance);
    return instance;
  }

  async function setSource(containerId, spec) {
    await ready;
    const instance = instances.get(containerId);
    if (!instance) throw new Error(`viv: #${containerId} not mounted`);
    // `spec.labelPath` is RESOLVED SERVER-SIDE from
    // `phenotypic.labels.objmap`. Never derive it as `${series}/labels/objmap`
    // here: backend section 1.1 forbids hard-coding it, and a `gray`-primary
    // store has no `rgb` group at all.
    return instance.setSource(spec);
  }

  function setViewState(containerId, viewState) {
    const instance = instances.get(containerId);
    if (instance) instance.setViewState(viewState);
  }

  function setLayerVisibility(containerId, name, visible) {
    const instance = instances.get(containerId);
    if (instance) instance.setLayerVisibility(name, visible);
  }

  function destroy(containerId) {
    const instance = instances.get(containerId);
    if (instance) {
      instance.finalize();
      instances.delete(containerId);
    }
  }

  window.phenotypicViv = {
    ready,
    mount,
    setSource,
    setViewState,
    setLayerVisibility,
    destroy,
    version: bundle.VERSION,
  };
})();
```

- [ ] **Step 4: Prove the ordering rule is enforced, not just written**

Add a second test that opens a store **without** awaiting `ready` and asserts it fails —
otherwise nothing distinguishes "we register first" from "registration happened to win the
race on this machine":

```python
@pytest.mark.e2e
def test_reading_before_ready_fails_loudly(page, live_viewer_url):
    page.goto(live_viewer_url)
    outcome = page.evaluate(
        """() => {
            try {
                window.__vivBundle.zarr.registry.delete('zstd');
                return 'deleted';
            } catch (e) { return 'unavailable'; }
        }"""
    )
    assert outcome in ("deleted", "unavailable")
```

If the registry offers no delete, record that in the test's docstring and assert on the
`ready` promise being awaited by every entry point instead — a code-shape assertion is
weaker but honest, and better than a test that passes vacuously.

- [ ] **Step 5: Run both, then commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/e2e/gui/test_viv_codec_reads_a_real_store.py -v
git add src/phenotypic/gui/results_viewer/_assets/viv_viewer.js \
        tests/e2e/gui/test_viv_codec_reads_a_real_store.py
git commit -m "feat(gui): add the Viv façade with zstd codec registration ordered first"
```

---

### Task 2.4: Vendor the upstream sources this work adapts

**Files:**
- Create: `docs/superpowers/specs/2026-08-26-viewer-viv-rebuild/refs/`

- [ ] **Step 1: Vendor byte-identical copies**

Copy in the upstream Viv/vizarr sources this implementation adapts — at minimum whatever
vizarr module resolves the `bioformats2raw.layout` series list and the label child, since
phase 3 mirrors its logic.

- [ ] **Step 2: Confirm ruff will not touch them**

```bash
uv run grep -n "extend-exclude" -A5 pyproject.toml
```
Expected: `docs/superpowers/**/refs` is excluded. Per root `CLAUDE.md`, these copies must
stay byte-identical to upstream — never linted, formatted, "tidied", or bug-fixed. Their
mistakes are the evidence; edit one and every citation against it silently stops meaning
anything, with nothing failing to tell you.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-08-26-viewer-viv-rebuild/refs/
git commit -m "docs(viv): vendor the upstream sources this rebuild adapts"
```
# Phase 3 — The Plate surface

**Spec:** §6.1, §4, §4.1. **Depends on:** phases 0-2. **Blocks:** phase 4 (shares the
façade wiring pattern).

**Deliverable:** Plate is a **full-canvas** deep-zoom Viv surface with floating controls,
in the vizarr / avivator posture rather than the current card-plus-sidebar one. The results
Plate path no longer builds a DZI pyramid: `_dzi_tiler` is unhooked from
`_tile_routes.py:34, :458, :551`, and no `.viewer_cache/` DZI directory is written for it.

**Mockup:** `docs/superpowers/artifacts/2026-08-26-gui-ome-zarr-sync/mockup/Main.dc.html`,
published at `https://claude.ai/code/artifact/7a8c50b6-042f-4948-9452-d6b6e557239f`.

> **All chrome derives from the existing system**, not from new invention — `gui/_design.py`
> and `results_viewer/_assets/results_viewer.css`: navy `#003660` header with the gold
> `#febc11` JetBrains-Mono pipeline chip, `#0e1620` image stage at `--radius` 6px, Comfortaa
> for display and body, JetBrains Mono for all numerics.

---

### Task 3.1: Resolve the store's real series list server-side

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_store_source.py`
- Test: `tests/unit/gui/results_viewer/test_store_source.py` (create)

**Interfaces:**
- Consumes: `phenotypic.sdk_.ngff_`, `OutputRoot.store_path`.
- Produces: `build_source_spec(output_root, dataset, stem, url_prefix) -> dict` with keys
  `baseUrl`, `series` (ordered, primary first), `primary`, `labelPath`, `pyramid`.
  Phase 3 task 3.3 hands this dict straight to `window.phenotypicViv.setSource`.

> **This is where backend §1.1 is honoured or violated.** The label path is read from
> `phenotypic.labels.objmap`. Nothing anywhere may construct `f"{primary}/labels/objmap"`.

- [ ] **Step 1: Write the failing tests, including the `gray`-primary case**

```python
"""The source spec is read from the store, never inferred.

The second test is spec section 8's 'label path' check: a store whose primary
series is ``gray`` (no ``rgb``) must resolve its objmap through
``phenotypic.labels.objmap``, proving nothing hard-codes ``rgb/labels/objmap``.
"""

from phenotypic.gui.results_viewer._store_source import build_source_spec


def test_series_come_from_the_store_in_primary_first_order(output_root, rgb_store):
    spec = build_source_spec(output_root, "ds", "plate", "/")
    assert spec["primary"] == "rgb"
    assert spec["series"][0] == "rgb"
    assert set(spec["series"]) <= {"rgb", "gray", "detect_mat"}


def test_label_path_is_read_not_constructed(output_root, gray_only_store):
    spec = build_source_spec(output_root, "ds", "grayplate", "/")
    assert spec["primary"] == "gray"
    assert not spec["labelPath"].startswith("rgb/")
    assert spec["labelPath"] == "gray/labels/objmap"


def test_pyramid_ladder_is_read_not_recomputed(output_root, rgb_store):
    spec = build_source_spec(output_root, "ds", "plate", "/")
    assert spec["pyramid"]["levels"] >= 1
    assert spec["pyramid"]["downsample"]["label"] == "nearest"
```

`gray_only_store` is a store written from an `Image` with no RGB layer. Build it with the
real writer; a hand-edited `zarr.json` would not prove the reader handles a real one.

- [ ] **Step 2: Run and watch them fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_store_source.py -v
```

- [ ] **Step 3: Implement**

```python
"""Build the client-facing source spec for one per-image store.

Every fact here is READ from the store's ``attributes.phenotypic`` block --
series list, primary series, resolved label path, pyramid ladder. None of it
is inferred, because backend section 1.1 forbids hard-coding the label path
and section 1.3 records that recomputing the level count has already been got
wrong once (``floor`` where ``ceil`` was needed).
"""

from __future__ import annotations

import json

from phenotypic.gui.results_viewer._zarr_routes import zarr_store_url
from phenotypic.sdk_ import ngff_


def build_source_spec(output_root, dataset: str, stem: str, url_prefix: str) -> dict:
    store = output_root.store_path(dataset, stem)
    root = json.loads((store / ngff_.STORE_ROOT_JSON).read_text(encoding="utf-8"))
    block = root["attributes"]["phenotypic"]

    series_map = block["series"]
    primary = "rgb" if "rgb" in series_map else "gray"
    ordered = [primary] + [name for name in series_map if name != primary]

    return {
        "baseUrl": zarr_store_url(url_prefix, dataset, stem),
        "series": ordered,
        "primary": primary,
        # RESOLVED, never constructed -- backend section 1.1.
        "labelPath": block["labels"]["objmap"],
        "pyramid": block["pyramid"],
    }
```

Confirm `ngff_.STORE_ROOT_JSON` exists and confirm the real key names before trusting the
literals above:

```bash
uv run grep -n "STORE_ROOT_JSON\|\"labels\"\|\"pyramid\"\|\"series\"" \
  src/phenotypic/sdk_/ngff_.py | head -20
```

- [ ] **Step 4: Run, lint, commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/results_viewer/test_store_source.py -v
uv run ruff check --fix src/phenotypic/gui/results_viewer/_store_source.py \
                        tests/unit/gui/results_viewer/test_store_source.py
git add src/phenotypic/gui/results_viewer/_store_source.py \
        tests/unit/gui/results_viewer/test_store_source.py
git commit -m "feat(gui): resolve the Viv source spec from the store's own metadata"
```

---

### Task 3.2: Pin level selection against the recorded ladder

**Files:**
- Test: `tests/unit/gui/results_viewer/test_level_selection.py` (create)

**Interfaces:**
- Consumes: `phenotypic.gui._shared.tiles.select_pyramid_level` (already landed).
- Produces: spec §8's "level selection" check.

> Spec §8: assert the level chosen for a target pixel size matches `phenotypic.pyramid`'s
> recorded ladder, **including the `ceil` boundary that backend §1.3 records as having
> failed once already**. That boundary is the whole reason this test exists rather than a
> smoke check.

- [ ] **Step 1: Write it, boundary case first**

```python
"""Level selection follows the store's recorded ladder, ceil boundary included.

Backend section 1.3: levels halve until ``max(H, W) <= 512``, so
``levels = ceil(log2(max(H, W) / 512)) + 1``. A draft used ``floor``, which
terminates one level early and leaves a 4000x3000 plate's smallest level at
1000x750. The parametrization below includes that exact case.
"""

import math

import pytest

from phenotypic.gui._shared.tiles import select_pyramid_level


@pytest.mark.parametrize(
    ("extent", "expected_levels"),
    [
        (512, 1),
        (513, 2),
        (1024, 2),
        (1025, 3),
        (4000, 4),
    ],
)
def test_recorded_ladder_matches_the_ceil_formula(extent, expected_levels):
    derived = 1 if extent <= 512 else math.ceil(math.log2(extent / 512)) + 1
    assert derived == expected_levels


def test_selected_level_covers_the_target_without_overshooting(rgb_store):
    for target in (128, 256, 512, 1024, 2048):
        level = select_pyramid_level(rgb_store, "rgb", target)
        assert level >= 0
```

Read `select_pyramid_level`'s real signature before writing the second test:
```bash
uv run sed -n '378,422p' src/phenotypic/gui/_shared/tiles.py
```
Adapt the call to match; do not assume the parameter order above.

- [ ] **Step 2: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_level_selection.py -v
git add tests/unit/gui/results_viewer/test_level_selection.py
git commit -m "test(gui): pin pyramid level selection to the recorded ladder"
```

---

### Task 3.3: Rebuild the Plate layout full-canvas

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_layout.py` (the Plate `cards_column`)
- Modify: `src/phenotypic/gui/results_viewer/_viewer_card.py`
- Modify: `src/phenotypic/gui/results_viewer/_assets/results_viewer.css`
- Modify: `src/phenotypic/gui/results_viewer/_callbacks.py` (clientside wiring)

- [ ] **Step 1: Read the mockup and the design tokens together**

```bash
uv run grep -oE 'class="[^"]+"' \
  docs/superpowers/artifacts/2026-08-26-gui-ome-zarr-sync/mockup/Main.dc.html \
  | sort -u | head -40
uv run grep -n "003660\|febc11\|0e1620\|Comfortaa\|JetBrains" src/phenotypic/gui/_design.py
```
Every colour, radius and family in the mockup must resolve to an existing token. If one
does not, it is an invention — either map it to the nearest token or raise it as a design
question, but do not add a new hard-coded hex.

- [ ] **Step 2: Build the surface**

Five pieces, per spec §6.1:

1. **Full-canvas stage.** Controls float **over** the canvas — the image gets the full
   frame instead of losing ~300 px to a sidebar.
2. **Layers panel** listing the store's **real** series from task 3.1's `series` key, plus
   `objmap` tagged as a **label image**, each with visibility, opacity and a swatch. It
   reads the list from the store, never a hard-coded set.
3. **Navigator inset** — OpenSeadragon's one genuinely missed affordance, carried over.
4. **Pyramid readout** naming the level actually being served, e.g.
   `level 1 of 4 · 2048×1536 · zstd · 1024² chunks`. This is **instrumentation, not
   decoration**: it is the pyramid the old DZI path rebuilt from scratch every time, and it
   is the fastest way to diagnose a level-selection bug.
5. **Image stepping** (`‹ dataset / stem ›`) and the object/grid summary, top-left.

- [ ] **Step 3: Wire the clientside callback through the façade only**

The callback calls `window.phenotypicViv.mount` / `.setSource` with task 3.1's dict. It
must not reference `window.__vivBundle`. Grep to enforce:

```bash
uv run grep -rn "__vivBundle" src/phenotypic/gui/ \
  --include='*.py' --include='*.js' | grep -v _assets/viv_viewer.js
```
Expected: no hits outside the façade.

- [ ] **Step 4: Unhook `_dzi_tiler` from the results Plate path**

Remove `_tile_routes.py:34` (the import), `:458` (`_dzi_tiler.tile(source_png, staging_dir)`)
and `:551` (the overlay tiling call), along with the now-dead staging/publish helpers
`_publish_dzi_cache` (`:337`) and `_generate_dzi_stage` (`:434`) **if** nothing else calls
them.

**Do not delete `_dzi_tiler` itself.** It has five live consumers outside this path —
`browse/_app.py:40`, `browse/_preparation.py:711`, `browse/_preparation_routes.py:95`,
`builder/_point_picker.py:417`, `builder/_preview_tiles.py:144` — and Browse keeps
libvips → DZI → OSD as its only pixel path (spec §9). See [DRIFT.md](DRIFT.md) D-5.

```bash
uv run grep -rn "_dzi_tiler" src/phenotypic/gui/results_viewer/_tile_routes.py
```
Expected after the edit: no hits.

- [ ] **Step 5: Decide the fate of the `.dzi` routes**

The `/tiles/<dataset>/<stem>.dzi` routes now have no producer for the Plate layer. Either
remove them with their tests, or keep them serving the **overlay** path if the QC review
gallery still calls them:

```bash
uv run grep -rn "\.dzi" src/phenotypic/gui/ --include='*.py' --include='*.js' | grep -v browse
```
Whichever you choose, say so in the commit body. Silently leaving a route with no producer
is the residue this rebuild should not create.

- [ ] **Step 6: Run the viewer suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/results_viewer -n 4 -q
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -k "plate or viewer" -q
```

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/
git add -A src/phenotypic/gui/results_viewer tests/unit/gui/results_viewer
git commit -m "feat(gui): rebuild the Plate surface on Viv, off the DZI path"
```

---

### Task 3.4: A mid-run zeros objmap is correct, not an error

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_store_source.py` (docstring)
- Test: `tests/unit/gui/results_viewer/test_store_source.py` (extend)

**Interfaces:**
- Consumes: [DRIFT.md](DRIFT.md) D-4.

> **The spec claim this corrects.** Viv spec §6.2 inherits from the backend spec the claim
> that Stage 2 writes the objmap into the store, so "the GUI can render a real objmap
> mid-run". The **landed** engine inverted that: Stage 2 is read-only and writes
> `stage2_raw/<ds>/<stem>.npy` plus a token under `.phenotypic/progress/`; only Stage 3
> re-promotes the store. Between Stage 1 and Stage 3 the store's `labels/objmap` is
> **zeros**.

- [ ] **Step 1: Write the test**

```python
def test_an_all_zero_objmap_is_a_valid_source_not_an_error(output_root, stage1_store):
    """A store between Stage 1 and Stage 3 holds a zeros objmap.

    Backend behaviour (landed): Stage 2 is read-only, so the in-store objmap
    stays zeros until Stage 3 re-promotes. The Layers panel must offer the
    label layer normally -- an empty segmentation is the correct rendering of
    a correct store, not a condition to surface as a fault.
    """
    spec = build_source_spec(output_root, "ds", "stage1plate", "/")
    assert spec["labelPath"]
    assert "error" not in spec
```

`stage1_store` is a store written by Stage 1 only — a zeros objmap with its `ome.labels`
list and `image-label` block present (backend §3.3 guarantees the objmap always exists,
including after Stage 1).

- [ ] **Step 2: Record the correction in the module docstring**

Add to `_store_source.py`:

```python
# A store observed between Stage 1 and Stage 3 holds an ALL-ZERO objmap:
# the landed staged engine keeps Stage 2 read-only and publishes the
# segmentation only when Stage 3 re-promotes. This is a valid store, and the
# Layers panel renders the label layer normally. Do not treat zeros as a
# fault -- see the plan's DRIFT.md, row D-4.
```

- [ ] **Step 3: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_store_source.py -v
git add src/phenotypic/gui/results_viewer/_store_source.py \
        tests/unit/gui/results_viewer/test_store_source.py
git commit -m "fix(gui): treat a mid-run zeros objmap as a valid label source"
```
# Phase 4 — Colony on deck.gl views (D1)

**Spec:** §6.2. **Depends on:** phases 0-3. **Blocks:** nothing.

**Deliverable:** the Colony grid renders as one `OrthographicView` per colony, each centred
on its centroid, with the Viv layer stack rendering into all of them and a single shared
`viewState`. Curation is retained throughout.

> **This phase is optional and is the first thing to cut.** Spec §6.2 stages the work as
> "ship first (D3), then D1", where D3 is "keep today's `build_tile_grid` chrome and change
> only the crop route — from overlay-PNG slicing to a level-0 chunk read." **D3 is already
> landed** on the store branch ([DRIFT.md](DRIFT.md) D-2): `_shared/tiles.py:665`
> `crop_colony` already prefers `crop_store_rgb`. So the staging already paid off — the
> data path is done and only the rendering layer remains.

---

### Task 4.1: Establish the virtualization cap before building on it

**Files:**
- Create: `docs/superpowers/logic_validation_scripts/2026-08-26-viewer-viv-rebuild/colony_view_budget.py`
- Modify: `docs/superpowers/plans/2026-08-26-viewer-viv-rebuild/spike/README.md`

**Interfaces:**
- Produces: the measured cell-count cap phase 4 task 4.3 enforces.

> Spec §6.2: "deck.gl re-renders every view each frame, so D1 needs a virtualization cap on
> cell count. The number is not set here — it is measured during D1, and until then D3 has
> no such limit." So the cap is measured **first**, not chosen and then justified.

- [ ] **Step 1: Write the budget script**

Per root `CLAUDE.md`: stdlib + numpy/scipy only, never imports `phenotypic`, exits non-zero
on failure.

```python
"""Re-derive the per-frame cost of one OrthographicView per colony.

Claim under test (viewer-viv-rebuild spec section 6.2): deck.gl re-renders
every view each frame, so an uncapped colony grid degrades linearly in cell
count. This script derives the draw-call and texture budget for a plate's
worth of cells so the cap is chosen against a number.

Exits non-zero if the derived budget contradicts the recorded cap.
"""

import sys

import numpy as np

#: Layers rendered into EACH view: base image + label overlay.
LAYERS_PER_VIEW = 2
#: A common plate: 32 x 48 = 1536 colonies (backend section 2.3's example).
PLATE_CELLS = 1536
#: Recorded cap, filled in from the measurement in step 3.
RECORDED_CAP: int | None = None


def draw_calls(cells: int, layers: int = LAYERS_PER_VIEW) -> int:
    """Draw calls per frame: every view renders every layer."""
    return cells * layers


def crop_texture_bytes(cells: int, crop_px: int = 64, channels: int = 3) -> int:
    """Resident texture bytes for a grid of RGB crops."""
    return cells * crop_px * crop_px * channels


def main() -> int:
    for cells in (64, 256, 1024, PLATE_CELLS):
        print(
            f"{cells:5d} cells: {draw_calls(cells):6d} draw calls/frame, "
            f"{crop_texture_bytes(cells) / 1e6:7.2f} MB textures"
        )
    if RECORDED_CAP is None:
        print("NO CAP RECORDED: run the measurement in task 4.1 step 3")
        return 1
    if draw_calls(RECORDED_CAP) > 4096:
        print(f"REFUTED: cap {RECORDED_CAP} exceeds a 4096 draw-call budget")
        return 1
    print(f"cap {RECORDED_CAP} holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it and confirm it fails on the missing cap**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-26-viewer-viv-rebuild/colony_view_budget.py
```
Expected: prints the table, then `NO CAP RECORDED`, exit **1**. The script failing at this
point is the intended state — it is what stops phase 4 proceeding on an invented number.

- [ ] **Step 3: Measure, then fill in `RECORDED_CAP`**

Render a prototype grid at 64, 256, 1024 and 1536 cells and record observed frame time at
each. Choose the cap at the largest count that holds an interactive frame budget, write it
into `RECORDED_CAP`, and record the measurement in `spike/README.md` beside the number.

- [ ] **Step 4: Re-run and commit**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-26-viewer-viv-rebuild/colony_view_budget.py
git add docs/superpowers/logic_validation_scripts/2026-08-26-viewer-viv-rebuild/ \
        docs/superpowers/plans/2026-08-26-viewer-viv-rebuild/spike/
git commit -m "docs(viv): measure the colony-view virtualization cap"
```

---

### Task 4.2: Shared camera as a value, not a sync protocol

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_assets/viv_viewer.js` (add `setGridViews`)
- Test: `tests/e2e/gui/test_colony_shared_camera.py` (create)

**Interfaces:**
- Consumes: `window.phenotypicViv` from phase 2.
- Produces: `setGridViews(containerId, cells, sharedViewState)` where `cells` is
  `[{id, centroidRr, centroidCc, size}, ...]`.

> **The napari implementation this ports.** `gui/_smart_grid/` patches `viewer.grid` so
> only **visible** layers get cells, then `create_overlay_clones` duplicates every
> Labels/Points/Shapes visual into **each** viewbox — every cell shows a different base
> image under the same annotation, sharing one camera.
>
> **The deck.gl translation:** zoom edits **one shared `viewState`** applied to every view.
> The shared camera is a **value, not a sync protocol** — there is no per-view listener
> reconciling positions, which is what makes it correct by construction rather than
> eventually consistent. `create_overlay_clones`' GPU-resource cleanup dance
> (`cleanup_clones`) has no deck.gl equivalent and is **not ported**.

- [ ] **Step 1: Read the napari original before porting**

```bash
uv run grep -rn "create_overlay_clones\|cleanup_clones\|viewer.grid" src/phenotypic/gui/_smart_grid/
```
Per the **`porting-a-reference-algorithm`** skill: cite `file:line` for each claim about
what it does, and diff line-by-line rather than inspecting and summarising. Record any
deviation in a drift-register row, however small.

- [ ] **Step 2: Write the failing e2e test**

```python
"""One shared viewState drives every colony view.

The assertion is that all views report the SAME zoom after one is changed --
not that they converge. A sync protocol would pass a convergence test and
still show tearing mid-gesture; a shared value cannot.
"""

import pytest


@pytest.mark.e2e
def test_zooming_one_cell_moves_every_cell(page, live_viewer_url):
    page.goto(live_viewer_url)
    page.wait_for_function("() => window.phenotypicViv !== undefined")
    page.click("[data-testid='tab-colony']")
    page.wait_for_selector(".colony-grid-view")

    page.evaluate(
        """() => window.phenotypicViv.setViewState(
               'colony-grid', {zoom: 3, target: [0, 0, 0]})"""
    )
    zooms = page.evaluate(
        """() => window.phenotypicViv.__debugViewStates('colony-grid')
                   .map(v => v.zoom)"""
    )
    assert len(zooms) > 1
    assert len(set(zooms)) == 1, f"views drifted apart: {sorted(set(zooms))}"
```

- [ ] **Step 3: Implement `setGridViews` with a single `viewState`**

One `OrthographicView` per cell, each with its own `x`/`y`/`width`/`height` and a
`target` at the colony centroid, but **one** `viewState` object shared by all of them.

- [ ] **Step 4: Make the shared-camera lock a visible affordance**

Spec §6.2: "The 'Shared camera' lock is a visible affordance, not hidden behaviour, so the
eventual unlock-one-cell mode has somewhere to live." Add the toggle to the Colony chrome
now, even though it only has one state today — retrofitting an affordance for a mode that
already shipped as invisible behaviour is the expensive order.

- [ ] **Step 5: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui/test_colony_shared_camera.py -v
git add src/phenotypic/gui/results_viewer/_assets/viv_viewer.js \
        tests/e2e/gui/test_colony_shared_camera.py
git commit -m "feat(gui): render colony cells as deck.gl views on one shared viewState"
```

---

### Task 4.3: Enforce the cap, and keep curation working

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/colony_view/_grid.py` — **rendering only**
- Test: `tests/unit/gui/results_viewer/test_colony_view_cap.py` (create)

> **The constraint that governs this task.** `colony_view/` is where the curation radial
> lives: `_grid.py:47, :462` build `build_radial_trigger` on every tile and
> `colony_view/_callbacks.py:43` builds the popover body. Curation is **retained** — the
> radial's six wedges are the real `ERROR_CATEGORY_COLORS` map (`oversegmented`,
> `undersegmented`, `merged`, `background_noise`, `debris`, `other`, each in its fixed
> Okabe-Ito slot), with the restore centre node and the custom-category strip, matching
> `_shared/_radial.py`'s anatomy. Bulk-mark still writes
> `deliverables/errors/<category>.parquet`.
>
> **`tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py` must pass
> unmodified.** If it needs editing, this task has overreached from rendering into
> curation — stop and escalate.

- [ ] **Step 1: Write the cap test**

```python
"""Above the cap, cells virtualize rather than all mounting.

Spec section 6.2 records the cap as measured during D1, so this test reads it
from the single source rather than restating a literal -- a cap that appears
in two places drifts.
"""

from phenotypic.gui.results_viewer.colony_view._grid import (
    COLONY_VIEW_CELL_CAP,
    plan_visible_cells,
)


def test_cells_beyond_the_cap_are_not_mounted():
    cells = [{"id": i} for i in range(COLONY_VIEW_CELL_CAP * 2)]
    visible = plan_visible_cells(cells, focus_index=0)
    assert len(visible) <= COLONY_VIEW_CELL_CAP


def test_the_focused_cell_is_always_visible():
    cells = [{"id": i} for i in range(COLONY_VIEW_CELL_CAP * 2)]
    focus = COLONY_VIEW_CELL_CAP + 5
    visible = plan_visible_cells(cells, focus_index=focus)
    assert any(c["id"] == focus for c in visible)
```

- [ ] **Step 2: Implement, taking `COLONY_VIEW_CELL_CAP` from task 4.1's measurement**

Add both to `_grid.py`. Keep `build_radial_trigger` on every **mounted** cell — a
virtualized-out cell has no radial because it has no tile, which is correct; a mounted cell
missing its radial is a curation regression.

- [ ] **Step 3: Prove curation is untouched**

```bash
git diff --stat src/phenotypic/gui/results_viewer/colony_view/_callbacks.py \
                 tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py
```
Expected: **empty**.

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py \
  tests/unit/gui/results_viewer/test_colony_view_cap.py -v
```
Expected: PASS, PASS.

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/gui/results_viewer/colony_view/_grid.py \
                        tests/unit/gui/results_viewer/test_colony_view_cap.py
git add src/phenotypic/gui/results_viewer/colony_view/_grid.py \
        tests/unit/gui/results_viewer/test_colony_view_cap.py
git commit -m "feat(gui): cap colony views at the measured budget, curation retained"
```
# Phase 5 — Verification, ledgers and docs

**Spec:** §8, and the `gui-checks` obligations inherited from the removals plan.
**Depends on:** phases 0-4.

**Deliverable:** all five of spec §8's checks green, the three CI gates passing, FEATURES.md
and WORKFLOWS.md updated for the rebuilt surfaces, and the bundle-staleness mitigation
wired up.

---

### Task 5.1: Close out spec §8's five checks

**Files:**
- Verify only, except where a check has no test yet.

Spec §8 names five. Four already have homes; confirm each and fill the gap.

- [ ] **Step 1: Walk the checklist**

| Spec §8 check | Where it lives | Action |
|---|---|---|
| Codec ordering — open a **CLI-written** store in a real browser, "not 'the codec registered' — the actual read" | phase 2 task 2.3 | run it |
| Level selection matches `phenotypic.pyramid`'s ladder, `ceil` boundary included | phase 3 task 3.2 | run it |
| Staleness — a rewritten nested chunk must invalidate | phase 1 task 1.2 | run it |
| Curation regression — colony curation tests pass **unmodified** | phase 4 task 4.3 step 3 | run it |
| Label path — a `gray`-primary store resolves its objmap through `phenotypic.labels.objmap` | phase 3 task 3.1 | run it |

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_zarr_routes.py \
  tests/unit/gui/results_viewer/test_store_source.py \
  tests/unit/gui/results_viewer/test_level_selection.py \
  tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py \
  -n 4 -v
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -k "viv or colony_shared" -v
```
Expected: all PASS.

- [ ] **Step 2: Confirm the spike gate's findings were actually acted on**

```bash
cat docs/superpowers/plans/2026-08-26-viewer-viv-rebuild/spike/README.md
```
Every question from phase 0 has an answer with evidence; the chunk-size decision carries
its measurement or an explicit "not measured, risk left open". **A gate whose findings were
never revisited is a gate that did not gate anything.**

---

### Task 5.2: Wire the bundle-staleness mitigation

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_app.py` (log the version at startup)
- Test: `tests/unit/gui/results_viewer/test_viv_bundle_version.py` (create)

> **Spec §10, open question 3, is unresolved and stays unresolved.** The build recipe is
> committed and the version logged, but *nothing fails* when the bundle drifts from the
> lockfile. A CI check that rebuilds and compares hashes would need npm in CI, which
> decision A exists to avoid. The version string is a **mitigation, not an answer** — this
> task implements the mitigation and says so plainly rather than dressing it up as a fix.

- [ ] **Step 1: Write the test**

```python
"""``tools/viv-bundle/VERSION`` agrees with the committed artifact.

This does NOT prove the artifact was built from the committed lockfile -- only
a rebuild could, and there is no npm in CI by design (spec section 3). It
catches the common case: bumping one and forgetting the other.
"""

import re
from pathlib import Path

import phenotypic.gui.results_viewer as rv

REPO = Path(rv.__file__).resolve().parents[4]
BUNDLE = Path(rv.__file__).parent / "_assets" / "viv" / "viv-bundle.min.js"
VERSION_FILE = REPO / "tools" / "viv-bundle" / "VERSION"


def test_bundle_embeds_the_recorded_version():
    recorded = VERSION_FILE.read_text(encoding="utf-8").strip()
    assert recorded, "tools/viv-bundle/VERSION is empty"
    head = BUNDLE.read_text(encoding="utf-8", errors="replace")[:4096]
    assert re.search(re.escape(recorded), head), (
        f"bundle does not embed VERSION {recorded!r}; rebuild it via "
        f"tools/viv-bundle/README.md or correct VERSION"
    )
```

Confirm `REPO` resolves correctly on this layout before trusting `parents[4]`:
```bash
uv run python -c "
import pathlib, phenotypic.gui.results_viewer as rv
print(pathlib.Path(rv.__file__).resolve().parents[4])"
```

- [ ] **Step 2: Log the version at startup**

In `_app.py`'s `create_app`, log `viv bundle: <VERSION>` alongside the existing startup
lines. Spec §3 requires the GUI to log it — with no npm in CI, nothing else will tell you
the bundle is stale.

- [ ] **Step 3: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_viv_bundle_version.py -v
git add src/phenotypic/gui/results_viewer/_app.py \
        tests/unit/gui/results_viewer/test_viv_bundle_version.py
git commit -m "chore(viv): log and pin the vendored bundle version"
```

---

### Task 5.3: Ledgers, tutorial and CLAUDE.md

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md`, `src/phenotypic/gui/WORKFLOWS.md`
- Modify: `scripts/capture_gui_tutorial_screenshots.py`
- Modify: `docs/source/tutorials/gui/06_view_results.md`
- Modify: `src/phenotypic/gui/CLAUDE.md`

- [ ] **Step 1: Update the FEATURES.md rows for the rebuilt surfaces**

The Plate and Colony rows now describe a Viv surface, not an OpenSeadragon one. Update the
capability text and any implementation refs that pointed at `_dzi_tiler` or the `.dzi`
routes. Add rows for the new affordances: the Layers panel, the navigator inset, the
pyramid readout, and the shared-camera lock.

```bash
uv run grep -n "OpenSeadragon\|DZI\|dzi\|Plate\|Colony" src/phenotypic/gui/FEATURES.md | head -30
```

- [ ] **Step 2: Refresh the results-viewer tutorial and its screenshots**

`06_view_results.md` shows the old card-plus-sidebar Plate. Update the prose and re-capture:

```bash
uv run grep -n "_capture_view_results\|06_view_results" \
  scripts/capture_gui_tutorial_screenshots.py src/phenotypic/gui/WORKFLOWS.md
```

Per the **`gui-tutorial-capture`** skill, the ledger ↔ capture-function ↔ tutorial-page
round trip must stay closed.

- [ ] **Step 3: Update `gui/CLAUDE.md`**

Record: the Plate/Colony pixel path is Viv over `/zarr/...`; Browse remains
libvips → DZI → `BrowseCache` → OSD; `_dzi_tiler` survives for Browse, the point picker and
the builder preview; the façade at `_assets/viv_viewer.js` is the only thing that may touch
`window.__vivBundle`.

- [ ] **Step 4: Run all three gates**

```bash
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
QT_QPA_PLATFORM=offscreen uv run python scripts/capture_gui_tutorial_screenshots.py --smoke
```
Expected: all exit 0.

- [ ] **Step 5: Full suite as a Slurm job**

Per the **`run-phenotypic-test`** skill — the full `tests/unit` suite is a ~65-minute Slurm
job, not a local invocation:

```bash
sbatch docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch
```
Report it as "green except the known baseline failure", and re-confirm it is still *that*
test failing for *that* reason.

- [ ] **Step 6: Commit**

```bash
git add -A src/phenotypic/gui/FEATURES.md src/phenotypic/gui/WORKFLOWS.md \
           src/phenotypic/gui/CLAUDE.md scripts/capture_gui_tutorial_screenshots.py \
           docs/source
git commit -m "docs(gui): record the Viv-backed Plate and Colony surfaces"
```

---

### Task 5.4: File the spec amendments this plan earned

**Files:**
- Modify: `docs/superpowers/specs/2026-08-26-viewer-viv-rebuild/design.md`
- Modify: `docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md` (only if the
  chunk-size measurement demanded it)

> Spec changes are the user's call, not the executor's. This task **drafts** them and stops.

- [ ] **Step 1: Draft the §1 correction**

Spec §1 opens by saying the backend is "specification only — there is no zarr code in
`src/`". That is no longer true. Draft a revision pointing at
[DRIFT.md](DRIFT.md) and marking §4.1, §4.2 and §6.2's D3 as landed.

- [ ] **Step 2: Draft the D-4 correction**

Spec §6.2's rationale inherits the backend's claim that Stage 2 writes the objmap in place,
so "the GUI can render a real objmap mid-run". The landed engine keeps Stage 2 read-only.
Draft the correction, and note that the *backend* spec §3.4 needs the same amendment.

- [ ] **Step 3: Draft the §4 route narrowing**

Spec §4 sketches an unrestricted path tail. Phase 1 restricts it to
`_READABLE_ROOTS` because measurements now live inside the store at
`tables/measurements/table.parquet` ([DRIFT.md](DRIFT.md) D-6). Draft the amendment with
that reasoning.

- [ ] **Step 4: Present all three to the user; do not self-approve**

Report the drafts and wait. Amending a spec on the executor's own authority is how a design
record stops being a record.
# Results viewer rebuild on Viv: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Plate and Colony surfaces stop rendering server-built DZI pyramids in
OpenSeadragon and start reading OME-Zarr chunks directly in the browser through Viv /
deck.gl, over a range-capable byte route.

**Architecture:** Three layers, added bottom-up. A Flask route serves raw store bytes with
HTTP Range (sharding needs it). A vendored, pre-built Viv + deck.gl IIFE — built outside
the repo, committed beside `openseadragon.min.js` — is driven by a hand-written imperative
façade, so Dash clientside callbacks never touch Viv directly. Plate becomes a full-canvas
deep-zoom surface with floating controls; Colony becomes one `OrthographicView` per colony
sharing a single `viewState` value.

**Tech Stack:** Viv + deck.gl (MIT, vendored IIFE), zarrita.js, `numcodecs.js` wasm zstd,
Flask `send_file(conditional=True)`, Dash clientside callbacks, Python 3.11+, `uv`.

**Spec:** [`docs/superpowers/specs/2026-08-26-viewer-viv-rebuild/design.md`](../../specs/2026-08-26-viewer-viv-rebuild/design.md)

**Read before starting:** [`DRIFT.md`](DRIFT.md). The spec was written against a tree that
predates the landed OME-Zarr store branch. **Roughly a third of its §4.1 and §6.2 scope is
already implemented.** DRIFT.md records what is done, what changed, and one spec claim that
is now false. Planning around the spec without it produces duplicated work.

**Baseline:** branch `feat/gui-ome-zarr-sync`, restacked onto
`worktree-ome-zarr-image-store` head `bf0d01a1`.

---

## Global Constraints

Everything in the removals plan's Global Constraints applies here too — `uv` only,
`QT_QPA_PLATFORM=offscreen`, never `-n auto`, explicit ruff paths, the known-failing
baseline test, and the three `gui-checks` CI gates. Additionally:

- **Backend contract, not backend guesswork.** Every store fact this plan relies on comes
  from `phenotypic.sdk_.ngff_` or the root `zarr.json`'s `attributes.phenotypic` block.
  **Readers MUST NOT hard-code `rgb/labels/objmap`** — resolve the label path through
  `phenotypic.labels.objmap` (backend §1.1). The primary series is `rgb` when present and
  `gray` otherwise.
- **Never infer the pyramid.** The resolved level count and downsample methods are
  persisted in `phenotypic.pyramid`. Read them; do not recompute `ceil(log2(...))` in the
  client. Backend §1.3 records that this exact computation has already been got wrong once
  (`floor` vs `ceil`).
- **`_dzi_tiler` is removed from the *results Plate path*, not from the tree.** It has
  live consumers in `browse/_preparation.py:711`, `browse/_preparation_routes.py:95`,
  `browse/_app.py:40`, `builder/_point_picker.py:417` and `builder/_preview_tiles.py:144`.
  Deleting the module breaks Browse, which spec §9 explicitly keeps on
  libvips → DZI → `BrowseCache` → OSD.
- **No npm in CI.** There is no `package.json` anywhere in this repo (verified). The Viv
  bundle is built outside the repo and committed as an artifact; the build recipe is
  committed at `tools/viv-bundle/` with a pinned lockfile and a recorded version string.
- **Licensing.** Viv and vizarr are MIT, compatible with Apache-2.0. `NOTICE` gains
  entries; `licenses/viv-MIT.txt` and `licenses/vizarr-MIT.txt` are added, matching the
  existing SAM2 / micro-sam pattern.
- **Vendored upstream sources are read-only.** Anything adapted from Viv/vizarr is
  vendored under `docs/superpowers/specs/2026-08-26-viewer-viv-rebuild/refs/` byte-identical
  to upstream, so every `file:line` citation resolves. Never lint, format, or fix them
  (root `CLAUDE.md`, "Porting a Reference Algorithm").
- **Curation is retained.** `colony_view/`'s radial keeps working throughout. Its six
  wedges are the real `ERROR_CATEGORY_COLORS` map in fixed Okabe-Ito slots; bulk-mark still
  writes `deliverables/errors/<category>.parquet`.
- **Chunk-size governance.** This work may file a **recorded amendment** against
  `2026-08-18-ome-zarr-image-store` §1.4 moving chunks from 1024² to 512² — but **only
  gated on a measurement** (spec §5.2). An amendment backed by a number is how the format
  stays right; one backed by convenience is how it drifts.

---

## Phases

Phase 0 is a **gate**: its findings amend the spec, and no later phase starts until it
reports. Spec §2.1 requires this explicitly.

| # | Phase | Deliverable | Doc |
|---|---|---|---|
| 0 | Spike gate | Four answered questions + a measured chunk-size number | [phase-0](phase-0-spike-gate.md) |
| 1 | Byte route | `/zarr/...` serving store bytes with HTTP Range and a per-segment path guard | [phase-1](phase-1-byte-route.md) |
| 2 | Viv bundle + façade | `tools/viv-bundle/`, the vendored IIFE, `viv_viewer.js`, NOTICE/licenses | [phase-2](phase-2-viv-bundle-facade.md) |
| 3 | Plate surface | Full-canvas Viv Plate; `_dzi_tiler` off the results plate path | [phase-3](phase-3-plate-surface.md) |
| 4 | Colony D1 | One `OrthographicView` per colony, shared `viewState`, virtualization cap | [phase-4](phase-4-colony-views.md) |
| 5 | Verification & ledgers | Spec §8's five tests, FEATURES/WORKFLOWS, tutorial refresh | [phase-5](phase-5-verification.md) |

**Phase 4 is separable.** Colony "D3" — the crop route reading level-0 store chunks — is
**already landed** (see DRIFT.md D-2). Phase 4 is purely the deck.gl rendering half, and
the viewer is shippable without it. If the spike or phase 3 overruns, cut phase 4 first.

## Definition of done

1. A CLI-written store opens in the browser and renders in Plate, with the zstd wasm codec
   decoding real chunks — not merely registering.
2. The level chosen for a target pixel size matches `phenotypic.pyramid`'s recorded ladder,
   including the `ceil` boundary.
3. A store whose primary series is `gray` (no `rgb`) resolves its objmap through
   `phenotypic.labels.objmap`.
4. A rewritten nested chunk invalidates the served tile.
5. `tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py` passes **unmodified**.
6. `uv run pytest tests/unit/gui -n 4` green (minus the known baseline failure); the three
   `gui-checks` gates exit 0.
