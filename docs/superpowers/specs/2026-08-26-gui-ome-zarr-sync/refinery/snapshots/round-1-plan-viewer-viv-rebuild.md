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

> **Correction, 2026-08-26 (round-1 review).** "All of them already key on the root
> `zarr.json`" is true for `_tile_routes` and `_preview_tiles` and **vacuous for the crop
> path**. `crop_colony` (`tiles.py:715`) still passes `os.stat(store).st_mtime_ns` — the
> store *directory* mtime, the exact key spec §4.1 flags as unsound — into `crop_store_rgb`,
> which immediately `del`s it (`:585-587`: "Accepted for caller/API compatibility; crop
> reads are windowed and not full-layer cached, so nothing keys on it").
>
> So there is **no staleness bug and no third cache** — but there is a live
> directory-mtime call sitting exactly where the spec says a trap lives. A reader auditing
> the fourth fix will find it and reasonably conclude the trap is open. Either drop the
> vestigial parameter, or leave it and rely on this note; do not "fix" a cache that does
> not exist.

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
# Phase 1 — The byte route and the shared resolver

**Spec:** §4, §4.0, §4.1. **Depends on:** phase 0. **Blocks:** phases 2-4, 6.

**Deliverable:** one `resolve_within_root` in `gui/_shared/tiles.py`, and
`GET /zarr/<dataset>/<stem>.ome.zarr/<token>/<path...>` on the results viewer's blueprint —
raw store bytes with **HTTP Range**, a per-store readable-root restriction enforced on the
**resolved** path, and a generation token that makes a torn read across a promote
structurally impossible.

> **Why Range is load-bearing, not a nicety.** A sharded read is a shard-index fetch
> followed by a byte-range fetch into the shard. Without `conditional=True` the client
> pulls whole shards — up to 96 MB — for every 1024² tile. This is the single flag the
> phase exists to get right.

> **Write the resolver once.** Phase 6 needs the same logic for the builder preview. A
> path-escape guard is a security primitive and spec §9.1 makes correctness binding; two
> copies drift **silently**, because each phase would test only its own copy. The two
> *routes* stay separate — they have genuinely different resolution and guard regimes
> (`OutputRoot.store_path` here, a session shape-check there) — but those regimes live
> outside the resolver.

---

### Task 1.1: The shared resolver

**Files:**
- Modify: `src/phenotypic/gui/_shared/tiles.py`
- Test: `tests/unit/gui/shared/test_resolve_within_root.py` (create)

**Interfaces:**
- Consumes: `is_safe_path_component` (`tiles.py:755`).
- Produces: `resolve_within_root(root: Path, tail: str, *, allowed_roots: frozenset[str] | None = None) -> Path`,
  raising `werkzeug.exceptions.BadRequest` / `NotFound`. Phase 6 task 6.1 calls it too.

- [ ] **Step 1: The guard is already verified — do not re-derive it**

`is_safe_path_component` was run against this route's inputs during plan refinement:

```text
'..' -> False   '.' -> False    '...' -> False   '.hidden' -> False
'a/b' -> False  'a\b' -> False  '%2e%2e' -> False  '' -> False
'rgb' -> True   'gray' -> True  'detect_mat' -> True  'OME' -> True
'zarr.json' -> True  'c.0.0.0' -> True  '0.0' -> True
'labels' -> True     'objmap' -> True   'tables' -> True
```

It rejects empty, any leading dot, `/`, `\`, literal `..`, then requires
`^[A-Za-z0-9._-]+$` (`_NAME_RE`, `:752`). Werkzeug 3.1.x does **not** normalise dot-segments
before routing, so the handler really does receive `../../../../etc/passwd` and the
traversal tests exercise the guard rather than passing on Werkzeug's behalf.

**Note `'tables' -> True`.** The guard does not keep the in-store measurements parquet off
the wire. The `allowed_roots` restriction does — and only if enforced correctly, which is
step 3.

- [ ] **Step 2: Write the failing tests**

```python
"""One resolver for every client-controlled path tail inside a store.

The third test is the one that matters and the one an earlier draft failed:
the restriction must bind the RESOLVED path, not the URL segments. Checking
the unresolved head lets a symlink inside a readable root escape it while
still passing containment.
"""

import pytest
from werkzeug.exceptions import BadRequest, NotFound

from phenotypic.gui._shared.tiles import resolve_within_root

ROOTS = frozenset({"OME", "rgb", "gray", "detect_mat"})


def test_resolves_a_file_inside_an_allowed_root(tmp_store):
    got = resolve_within_root(tmp_store, "rgb/0/c.0.0.0", allowed_roots=ROOTS)
    assert got == (tmp_store / "rgb" / "0" / "c.0.0.0").resolve()


def test_rejects_a_disallowed_root(tmp_store):
    with pytest.raises(NotFound):
        resolve_within_root(
            tmp_store, "tables/measurements/table.parquet", allowed_roots=ROOTS
        )


def test_a_symlink_into_a_disallowed_root_is_rejected(tmp_store):
    """The escape an unresolved head check misses.

    `rgb/sneak` passes a head check on segments[0] and resolves to a path
    still INSIDE the store, so containment passes too. Only testing the
    resolved path's first component catches it.
    """
    target = tmp_store / "tables" / "measurements" / "table.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"secret")
    (tmp_store / "rgb" / "sneak").symlink_to(target)

    with pytest.raises((NotFound, BadRequest)):
        resolve_within_root(tmp_store, "rgb/sneak", allowed_roots=ROOTS)


def test_label_group_under_a_series_resolves(tmp_store):
    """Only the FIRST resolved component is restricted, by design.

    Restricting every component would block `labels`, `objmap` and every
    level index, killing the label layer.
    """
    p = tmp_store / "rgb" / "labels" / "objmap" / "0"
    p.mkdir(parents=True, exist_ok=True)
    (p / "c.0.0").write_bytes(b"x")
    assert resolve_within_root(
        tmp_store, "rgb/labels/objmap/0/c.0.0", allowed_roots=ROOTS
    )


@pytest.mark.parametrize(
    "tail",
    [
        "../../../../etc/passwd",
        "rgb/../../../../etc/passwd",
        "rgb/0/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "rgb/./../../zarr.json",
        "",
    ],
)
def test_rejects_traversal_in_any_segment(tmp_store, tail):
    with pytest.raises((BadRequest, NotFound)):
        resolve_within_root(tmp_store, tail, allowed_roots=ROOTS)


def test_no_allow_list_permits_any_root(tmp_store):
    assert resolve_within_root(tmp_store, "zarr.json") is not None
```

- [ ] **Step 3: Implement — restriction AFTER resolution**

```python
def resolve_within_root(
    root: Path,
    tail: str,
    *,
    allowed_roots: frozenset[str] | None = None,
) -> Path:
    """Resolve a client-controlled ``tail`` to a file inside ``root``.

    The single path-escape guard for every route that serves bytes out of a
    store directory. Two properties are load-bearing and easy to get wrong:

    * Segments are validated INDIVIDUALLY. The traversal surface here is
      wider than a two-component route's because the tail is arbitrary depth.
    * ``allowed_roots`` is tested on the RESOLVED path, not on the URL
      segments. Testing the unresolved head lets a symlink inside a readable
      root (``<root>/rgb/x -> ../tables/measurements/table.parquet``) satisfy
      both the head check and containment, and the file is served.

    Only the FIRST resolved component is restricted. Restricting every
    component would reject ``labels``, ``objmap`` and every level index,
    which would kill the label layer.

    Args:
        root: Directory the result must live inside.
        tail: Client-controlled path, ``/``-separated.
        allowed_roots: First-component allow-list, or ``None`` for no
            restriction beyond containment.

    Returns:
        The resolved file path.

    Raises:
        BadRequest: A segment is unsafe, or the resolved path escapes ``root``.
        NotFound: The path does not exist, is not a file, or its first
            resolved component is not in ``allowed_roots``.
    """
    segments = [s for s in tail.split("/") if s]
    if not segments:
        raise NotFound()
    for segment in segments:
        if not is_safe_path_component(segment):
            raise BadRequest()

    root_resolved = root.resolve(strict=True)
    try:
        resolved = root.joinpath(*segments).resolve(strict=True)
    except (OSError, RuntimeError):
        raise NotFound() from None
    if not resolved.is_relative_to(root_resolved):
        raise BadRequest()
    if not resolved.is_file():
        raise NotFound()

    if allowed_roots is not None:
        rel = resolved.relative_to(root_resolved)
        head = rel.parts[0]
        if head not in allowed_roots and not (
            len(rel.parts) == 1 and head == ngff_.STORE_ROOT_JSON
        ):
            raise NotFound()
    return resolved
```

- [ ] **Step 4: Run, lint, commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/shared/test_resolve_within_root.py -v
uv run ruff check --fix src/phenotypic/gui/_shared/tiles.py tests/unit/gui/shared/test_resolve_within_root.py
git add src/phenotypic/gui/_shared/tiles.py tests/unit/gui/shared/test_resolve_within_root.py
git commit -m "feat(gui): add one path-escape resolver for store byte routes"
```

---

### Task 1.2: Per-store readable roots and the generation token

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_zarr_routes.py`
- Test: `tests/unit/gui/results_viewer/test_zarr_routes.py` (create)

**Interfaces:**
- Consumes: `resolve_within_root`, `OutputRoot.store_path` (`_output_root.py:495`, returns
  `Path | None`), `paths_fingerprint`, `ngff_.STORE_ROOT_JSON`.
- Produces: `register_zarr_routes(app, output_root)`,
  `zarr_store_url(url_prefix, dataset, stem, token)`,
  `readable_roots_for(store) -> frozenset[str]`, `store_generation_token(store) -> str`.
  Phase 6 calls the last two.

- [ ] **Step 1: Derive the readable set from the store, never a literal**

A fixed `{rgb, gray, detect_mat}` is wrong: `_write_store_part` appends `"original"` to
`series_names` whenever the image carries one (`_image_io_handler.py:1012-1014`), and that
list lands in `attributes.phenotypic.series`. A literal set makes the Layers panel list a
series the route 404s — the same hard-coding spec §1's label-path rule forbids, one layer
down.

```python
def readable_roots_for(store: Path) -> frozenset[str]:
    """First-path-components a pixel client may read from this store.

    Derived from the store's own ``attributes.phenotypic`` block, so a
    series the writer legitimately added (``original``) is readable without
    editing this function -- and ``tables/``, which holds the per-object
    measurement parquet, never is.
    """
    block = _readable_block(store)          # raises StoreUnreadable
    roots = set(block.get(PhenotypicAttr.SERIES, {}).keys())
    for label_path in block.get(PhenotypicAttr.LABELS, {}).values():
        roots.add(PurePosixPath(label_path).parts[0])
    roots.add("OME")
    return frozenset(roots)
```

Reuse the landed `_readable_block` (`_shared/tiles.py:304`) rather than a raw `json.loads`
— it raises `StoreUnreadable` on a schema-version mismatch, which is what keeps Plate and
Colony agreeing about a store this build cannot decode.

- [ ] **Step 2: The generation token**

`promote_store` (`sdk_/ngff_.py:1235-1300`) republishes by renaming the whole store
directory. The route resolves fresh per request and holds no handle, so without a token a
client can read `zarr.json` from promote *N* and chunks from *N+1*. Reuse the landed token
rather than inventing one:

```python
def store_generation_token(store: Path) -> str:
    """Short opaque token identifying one promote of ``store``.

    Same construction as ``_tile_routes._store_content_token``: the root
    ``zarr.json``'s content fingerprint AND its mtime. Neither alone is
    enough -- a rewrite can reproduce byte-identical metadata while the
    pixels underneath differ.
    """
    root_json = store / ngff_.STORE_ROOT_JSON
    digest = paths_fingerprint([root_json]).removeprefix("sha256:")[:16]
    return f"{digest}-{os.stat(root_json).st_mtime_ns}"
```

The token is a **path segment**, so a new promote yields a new base URL and mixing is
structurally impossible rather than merely unlikely. A stale token returns **409**, which
tells the client to re-read the source spec — a 404 would read as "chunk missing" and be
retried forever.

- [ ] **Step 3: Write the failing tests**

```python
def test_serves_the_root_zarr_json(zarr_route_client, spike_store, token):
    resp = zarr_route_client.get(f"/zarr/ds/plate.ome.zarr/{token}/zarr.json")
    assert resp.status_code == 200


def test_honours_a_range_request(zarr_route_client, spike_store, token):
    resp = zarr_route_client.get(
        f"/zarr/ds/plate.ome.zarr/{token}/rgb/0/c.0.0.0",
        headers={"Range": "bytes=0-15"},
    )
    assert resp.status_code == 206
    assert len(resp.data) == 16
    assert resp.headers["Accept-Ranges"] == "bytes"


def test_a_stale_token_is_409_not_404(zarr_route_client, spike_store, token):
    """A re-promote must not be served as a missing chunk.

    404 reads as 'this chunk does not exist' and the client retries; 409
    tells it to re-read the source spec, which is the actual remedy.
    """
    url = f"/zarr/ds/plate.ome.zarr/{token}/rgb/0/c.0.0.0"
    assert zarr_route_client.get(url).status_code in (200, 206)
    _repromote(spike_store)
    assert zarr_route_client.get(url).status_code == 409


def test_an_original_series_is_readable(zarr_route_client, store_with_original):
    """A store carrying `original` must not 404 on it.

    `_write_store_part` appends "original" to series_names when the image
    has one, so a hard-coded readable set breaks a legitimate store.
    """
    tok = store_generation_token(store_with_original)
    resp = zarr_route_client.get(
        f"/zarr/ds/orig.ome.zarr/{tok}/original/0/c.0.0.0"
    )
    assert resp.status_code in (200, 206)


def test_the_measurements_table_is_never_served(zarr_route_client, spike_store, token):
    resp = zarr_route_client.get(
        f"/zarr/ds/plate.ome.zarr/{token}/tables/measurements/table.parquet"
    )
    assert resp.status_code == 404
```

- [ ] **Step 4: Implement the route on the shared resolver**

```python
def register_zarr_routes(app: dash.Dash, output_root) -> None:
    bp = Blueprint("zarr_bytes", __name__, url_prefix="/zarr")

    @bp.route("/<dataset>/<stem>.ome.zarr/<token>/<path:tail>")
    def store_bytes(dataset: str, stem: str, token: str, tail: str):
        if not is_safe_path_component(dataset) or not is_safe_path_component(stem):
            abort(400)
        store = output_root.store_path(dataset, stem)
        if store is None or not store.is_dir():
            abort(404)
        if token != store_generation_token(store):
            abort(409)
        return send_file(
            resolve_within_root(
                store, tail, allowed_roots=readable_roots_for(store)
            ),
            conditional=True,
        )

    app.server.register_blueprint(bp)
```

- [ ] **Step 5: Run, lint, commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/results_viewer/test_zarr_routes.py -v
uv run ruff check --fix src/phenotypic/gui/results_viewer/_zarr_routes.py \
                        src/phenotypic/gui/results_viewer/_app.py \
                        tests/unit/gui/results_viewer/test_zarr_routes.py
git add src/phenotypic/gui/results_viewer tests/unit/gui/results_viewer
git commit -m "feat(gui): serve OME-Zarr bytes with Range, a per-store root set and a generation token"
```

---

### Task 1.3: What the route still exposes — record it for sign-off

**Files:**
- Modify: `docs/superpowers/plans/2026-08-26-viewer-viv-rebuild/spike/README.md`

> Spec §4.0 already records the narrowing. This task exists so the *residual* exposure is
> stated as a fact rather than left implied by "pixels only".

- [ ] **Step 1: Write down what a browser can read**

The root `zarr.json` is **mandatory** — the client bootstraps from it — and carries
`attributes.phenotypic.metadata`: the `protected`, `public` and `imported` sections plus
`work_id` (`sdk_/ngff_.py:559-583`). `OME/METADATA.ome.xml` carries the same `Metadata_*`
sections. The narrowing keeps `tables/measurements/table.parquet` off the wire; it does
**not** make the route metadata-free.

Combined with the no-authentication assumption in spec §9, that means: on the documented
Open OnDemand recipe (`--host 0.0.0.0`, `gui_hub.md:116, :124`), anything that can reach
the node's port can read a run's image metadata. This is **not new** — the existing DZI and
crop routes already serve pixels the same way — but it is now written down.

- [ ] **Step 2: No commit** — this rides on the spec §4.0 sign-off already recorded.
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
- Consumes: `_readable_block` and `ngff_.primary_series` (landed), `OutputRoot.store_path`
  (`_output_root.py:495`, returns `Path | None`), `store_generation_token` / `zarr_store_url`
  from phase 1.
- Produces: **`build_source_spec(store: Path, base_url: str) -> dict`** with keys
  `baseUrl`, `token`, `series` (ordered, primary first), `primary`, `labelPath` (**may be
  `None`**), `pyramid`, `measured`. Task 3.3 hands this dict to
  `window.phenotypicViv.setSource`; **phase 6 is the second caller** and is why the
  signature takes a store path rather than an `OutputRoot` — the builder preview has stores
  but no output root. Written at its final signature here so phase 6 adds a caller instead
  of refactoring this function's own work.

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


def test_series_come_from_the_store_in_primary_first_order(rgb_store):
    spec = build_source_spec(rgb_store, "/zarr/ds/plate.ome.zarr")
    assert spec["primary"] == "rgb"
    assert spec["series"][0] == "rgb"


def test_an_original_series_is_listed(store_with_original):
    """`_write_store_part` appends "original" when the image carries one.

    A spec that filtered the series list to a literal set would silently drop
    a layer the writer legitimately produced -- and the byte route's readable
    set is derived from this same list, so the two would disagree.
    """
    spec = build_source_spec(store_with_original, "/zarr/ds/orig.ome.zarr")
    assert "original" in spec["series"]


def test_label_path_is_read_not_constructed(gray_only_store):
    spec = build_source_spec(gray_only_store, "/zarr/ds/grayplate.ome.zarr")
    assert spec["primary"] == "gray"
    assert not spec["labelPath"].startswith("rgb/")
    assert spec["labelPath"] == "gray/labels/objmap"


def test_a_store_with_no_label_image_yields_no_label_path(label_less_store):
    """`labels` is omitted entirely, not emitted empty (ngff_.py:576-581).

    Most builder-preview stores are like this, because
    `save_intermediate_zarr` writes an objmap only when it is requested.
    """
    spec = build_source_spec(label_less_store, "/zarr/ds/prev.ome.zarr")
    assert spec["labelPath"] is None


def test_pyramid_ladder_is_read_not_recomputed(rgb_store):
    spec = build_source_spec(rgb_store, "/zarr/ds/plate.ome.zarr")
    assert spec["pyramid"]["levels"] >= 1
    assert spec["pyramid"]["downsample"]["label"] == "nearest"
```

**Six fixtures, none of which exist yet** — `rgb_store` is referenced across this phase and
is defined **nowhere** in `tests/` (the only grep hit is an unrelated *test name* at
`tests/unit/sdk_/test_ngff_validity.py:172`). Define all six in the results-viewer
`conftest.py`, every one built with the **real writer**; a hand-edited `zarr.json` would let
a test agree with a store no writer produces:

| Fixture | What it is |
|---|---|
| `rgb_store` | an ordinary finished run store, `rgb` primary, measured |
| `gray_only_store` | an `Image` with no RGB layer, so `gray` is primary |
| `store_with_original` | an image carrying an `original`, so `series` exceeds the three canonical names |
| `label_less_store` | `save_intermediate_zarr(layers=("gray",))` — the `labels` key absent entirely |
| `stage1_store` | Stage 1 only: zeros objmap present, no embedded measurements table |
| `store_at_extent(extent)` | factory writing at `(extent, extent * 3 // 4)`, for task 3.2 |

- [ ] **Step 2: Run and watch them fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/results_viewer/test_store_source.py -v
```

- [ ] **Step 3: Implement**

```python
"""Build the client-facing source spec for one per-image store.

Every fact here is READ from the store, never inferred -- series list,
primary series, resolved label path, pyramid ladder. None of it is
recomputed, because backend section 1.1 forbids hard-coding the label path
and section 1.3 records that re-deriving the level count has already been
got wrong once (``floor`` where ``ceil`` was needed).

Built on the LANDED resolvers rather than a second implementation of them:
``_readable_block`` raises ``StoreUnreadable`` on a schema-version mismatch,
which is what keeps Plate and Colony agreeing about a store this build
cannot decode -- ``crop_colony`` deliberately does not catch it
(``tiles.py:685-688``). A raw ``json.loads`` here would let Plate open a
store that Colony 422s on, with the two surfaces disagreeing about one
store.
"""

from __future__ import annotations

from pathlib import Path

from phenotypic.gui._shared.tiles import _readable_block
from phenotypic.gui.results_viewer._zarr_routes import (
    store_generation_token,
    zarr_store_url,
)
from phenotypic.sdk_ import ngff_


def build_source_spec(store: Path, base_url: str) -> dict:
    """Describe one store to the Viv facade.

    Takes a STORE PATH and a base URL rather than an ``OutputRoot``: the
    builder preview (phase 6) has stores but no output root, and it is the
    second caller. Written at its final signature here so phase 6 does not
    refactor this function's own work.
    """
    block = _readable_block(store)

    series_map = block.get(ngff_.PhenotypicAttr.SERIES, {})
    primary = ngff_.primary_series(list(series_map))
    ordered = [primary] + [name for name in series_map if name != primary]

    # `labels` is OMITTED ENTIRELY when the store carries no label image
    # (`ngff_.py:576-581`, ledger C3) -- and `save_intermediate_zarr` sets
    # `write_objmap = "objmap" in layers`, so MOST builder-preview stores
    # have no `labels` key at all. `block["labels"]` would KeyError on them.
    label_path = block.get(ngff_.PhenotypicAttr.LABELS, {}).get("objmap")

    # Absent `tables.measurements` means "not yet measured" -- the only
    # reliable way to tell a mid-run store (Stage 1 done, Stage 3 pending,
    # objmap all zeros) from a finished image whose detector found nothing.
    # See task 3.4.
    measured = ngff_.PhenotypicAttr.TABLES in block

    return {
        "baseUrl": base_url,
        "token": store_generation_token(store),
        "series": ordered,
        "primary": primary,
        "labelPath": label_path,      # may be None -- the facade must cope
        "pyramid": block[ngff_.PhenotypicAttr.PYRAMID],
        "measured": measured,
    }
```

Confirm the `PhenotypicAttr` member names and `primary_series`' signature before trusting
the references above:

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
>
> **First settle whether the thing under test still has a caller.** After step 4 of task
> 3.3, the server-side selection stack is orphaned: `select_pyramid_level` is called only by
> `_load_zarr_layer_rgb` (`tiles.py:473`), which is called only by `_tile_routes.py:31, :500`
> — the Plate DZI path this phase removes. Colony does **not** use it (`crop_store_rgb` is
> level-0 always, `tiles.py:566`). `_load_zarr_level_rgb` and `_store_level0_longest_edge`
> go the same way; `_store_content_token` is re-homed by phase 1's generation token.
>
> **So after this phase there is exactly one level-selection authority — the browser — and
> the server's is dead code.** Pinning dead code with a new test is worse than leaving it
> orphaned: it looks maintained. Choose, and say which in the commit:
>
> - **retire the stack with its tests** and assert the ladder against the *browser's*
>   choice in phase 5's level-selection check; or
> - **keep it** and state what still calls it.
>
> The invariant below is worth pinning either way — it is the arithmetic, not the caller.

- [ ] **Step 1: Write it against real stores — not against itself**

> **An earlier draft of this test asserted nothing about the code.** It computed
> `derived = 1 if extent <= 512 else ceil(log2(extent/512)) + 1` **in the test body** and
> compared it to its own parametrization; it imported `select_pyramid_level` and never
> called it. It would pass with `phenotypic` uninstalled, and could not fail if the
> implementation regressed to `floor` — the exact regression its own docstring named. Its
> companion `assert level >= 0` was unconditional, because `chosen` initialises to `0`
> (`_shared/tiles.py:414`).
>
> The five parametrized values themselves were **checked and are correct** (recomputed
> against `sdk_/ngff_.py:167-169` and `ngff_store_geometry.py` claim C1). Keep them; drive
> them against real stores.

```python
"""Level selection follows the store's RECORDED ladder, ceil boundary included.

Backend section 1.3: levels halve until ``max(H, W) <= 512``, so
``levels = ceil(log2(max(H, W) / 512)) + 1``. A draft used ``floor``, which
terminates one level early and leaves a 4000x3000 plate's smallest level at
1000x750. That regression is what these tests exist to catch, so every
assertion here runs against a store written by the real writer -- never
against a formula restated in the test body.
"""

import pytest

from phenotypic.gui._shared.tiles import select_pyramid_level


@pytest.mark.parametrize(
    ("extent", "expected_levels"),
    [(512, 1), (513, 2), (1024, 2), (1025, 3), (4000, 4)],
)
def test_the_written_store_records_the_ceil_ladder(store_at_extent, extent, expected_levels):
    """The STORE's recorded ladder, not a formula re-derived here.

    `floor` would give 512->1, 513->1, 1025->2, 4000->3 -- so 513, 1025 and
    4000 each fail under the regression, and 513/1025 are the ceil boundaries
    specifically.
    """
    store = store_at_extent(extent)
    block = _readable_block(store)
    assert block["pyramid"]["levels"] == expected_levels


def test_selected_level_is_the_coarsest_that_still_covers(store_at_extent):
    """Exercises `select_pyramid_level` itself, including the exact-edge case.

    `assert level >= 0` is vacuous -- `chosen` initialises to 0
    (`_shared/tiles.py:414`), so it holds even if every branch is wrong.
    Assert the contract in the docstring instead: the chosen level's longest
    edge is >= target, and the NEXT coarser level's is not.
    """
    store = store_at_extent(4000)
    shapes = _level_shapes(store, "rgb")          # [(4000,3000),(2000,1500),...]

    for target in (4000, 2000, 1024, 1000, 512, 256):
        level = select_pyramid_level(store, "rgb", target)
        assert max(shapes[level][-2:]) >= target, (
            f"level {level} does not cover {target}"
        )
        if level + 1 < len(shapes):
            assert max(shapes[level + 1][-2:]) < target, (
                f"level {level + 1} also covers {target}; {level} is not the coarsest"
            )


def test_a_target_landing_exactly_on_a_level_edge_picks_that_level(store_at_extent):
    """`>=` not `>` -- an exact match must not fall through to a finer level."""
    store = store_at_extent(4000)
    shapes = _level_shapes(store, "rgb")
    exact = max(shapes[1][-2:])                   # level 1's longest edge
    assert select_pyramid_level(store, "rgb", exact) == 1
```

**Define the fixtures — they do not exist.** `rgb_store` was referenced across phase 3 and
this task and is defined **nowhere** in `tests/` (the only grep hit is an unrelated *test
name* in `tests/unit/sdk_/test_ngff_validity.py:172`). Add to the results-viewer
`conftest.py`:

- `store_at_extent(extent) -> Path` — a factory writing a real store at
  `(extent, extent * 3 // 4)` via the CLI writer, so the recorded ladder is the writer's.
- `rgb_store`, `gray_only_store`, `label_less_store`, `store_with_original`, `stage1_store`
  — the five task 3.1 and 3.4 use.
- `_level_shapes(store, layer)` — read each level's shape from the store rather than
  computing it.

Build every one with the real writer. A hand-edited `zarr.json` would let the test agree
with a store no writer produces.

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

   **It must be read back from the façade, not computed server-side.** `build_source_spec`
   returns the *ladder*; the level in use is deck.gl's per-frame choice. A number computed
   from `select_pyramid_level` would report a level nobody rendered — worse than showing
   nothing, because a readout labelled "the level actually being served" will be trusted
   when diagnosing exactly the bug it is misreporting. Add an `onLevelChange` callback (or a
   `__debugViewStates`-style seam) to the façade in phase 2 and drive the readout from it.
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

- [ ] **Step 5: Remove the `/tiles/<dataset>/<stem>.dzi` routes — the answer is determinate**

Their consumers, enumerated during review:

```text
results_viewer/_assets/results_viewer.js:286-298   Plate      <- removed by this phase
timeline_view/_callbacks.py:266                    Timeline   <- DELETED by cycle 1
results_viewer/_assets/timeline.js:447-454         Timeline   <- DELETED by cycle 1
```

No `_qc_tab` or `_error_tab` reference exists, so the "keep them for the QC review gallery"
branch an earlier draft offered is **dead** — it would preserve a route for a consumer that
cycle 1 unmounts anyway. Remove the routes with their tests.

```bash
uv run grep -rn "\.dzi" src/phenotypic/gui/ --include='*.py' --include='*.js' | grep -v browse
```
**Run this after cycle 1 has landed.** Run before, it returns the timeline hits and argues
for the wrong answer. Treat it as confirmation of the table above, not as the decision.

**Do not delete `_tile_routes.py` as a module.** `builder/_preview_tiles.py:31` imports
`_TILE_NAME_RE` and `_json_error` from it, and `_validate` returns through `_json_error` —
so deleting it breaks the builder preview *and* phase 6's new route at import, in a
different sub-app from the one being edited. Either relocate those two symbols to
`_shared/` in this step, or leave the module in place with its routes removed.

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

> **Zeros is correct, but it is also ambiguous — and the ambiguity is the actionable half.**
> An all-zero objmap merges two states: (a) Stage 1 done, Stage 3 pending, and (b) a
> finished image whose detector produced **zero objects**, which is a real QC fault. Plate
> renders both identically, and so does Colony (no `master_df` rows either way). Nothing
> treats zeros as an error today, so no correctness change is needed — but a user cannot
> tell "still running" from "found nothing".
>
> **The discriminator is already in the store.** `attributes.phenotypic.tables.measurements`
> is written **only** when a measurement table was embedded
> (`_image_io_handler.py:1143-1167`), so its absence is a reliable "not yet measured" — and
> mid-run there is no `tables/` group at all. `build_source_spec` returns it as `measured`.

- [ ] **Step 1: Write the tests**

```python
def test_an_all_zero_objmap_is_a_valid_source_not_an_error(stage1_store):
    """A store between Stage 1 and Stage 3 holds a zeros objmap.

    Backend behaviour (landed): Stage 2 is read-only, so the in-store objmap
    stays zeros until Stage 3 re-promotes. The Layers panel must offer the
    label layer normally -- an empty segmentation is the correct rendering of
    a correct store, not a condition to surface as a fault.
    """
    spec = build_source_spec(stage1_store, "/zarr/ds/stage1plate.ome.zarr")
    assert spec["labelPath"]
    assert "error" not in spec


def test_a_mid_run_store_reports_itself_unmeasured(stage1_store):
    """"Measurement pending" and "found nothing" must be distinguishable."""
    assert build_source_spec(stage1_store, "/x")["measured"] is False


def test_a_finished_store_reports_itself_measured(rgb_store):
    assert build_source_spec(rgb_store, "/x")["measured"] is True
```

`stage1_store` is a store written by Stage 1 only — a zeros objmap with its `ome.labels`
list and `image-label` block present (backend §3.3 guarantees the objmap always exists,
including after Stage 1), and **no** embedded measurements table.

- [ ] **Step 1b: Surface it in the Layers panel**

When `measured` is false, label the objmap layer **"measurement pending"** rather than
leaving it looking like an empty result. One word of chrome; it is the difference between a
user waiting and a user filing a bug.

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

Per root `CLAUDE.md`: stdlib + numpy/scipy only (this one needs only the stdlib), never
imports `phenotypic`, exits non-zero on failure.

```python
"""Re-derive the per-frame cost of one OrthographicView per colony.

Claim under test (viewer-viv-rebuild spec section 6.2): deck.gl re-renders
every view each frame, so an uncapped colony grid degrades linearly in cell
count. This script derives the draw-call and texture budget for a plate's
worth of cells so the cap is chosen against a number.

Exits non-zero until the prototype measurement in task 4.1 step 3 has been
recorded, so no later task can proceed on an unmeasured cap.
"""

import sys

#: Layers rendered into EACH view: base image + label overlay.
LAYERS_PER_VIEW = 2
#: A common plate: 32 x 48 = 1536 colonies (backend section 2.3's example).
PLATE_CELLS = 1536
#: Measured cap, filled in from the prototype in step 3. None until measured.
RECORDED_CAP: int | None = None
#: Frame time, in ms, observed at RECORDED_CAP. Recorded beside the number so
#: the cap can be re-judged later without re-running the prototype blind.
RECORDED_FRAME_MS: float | None = None


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
    if RECORDED_CAP is None or RECORDED_FRAME_MS is None:
        print("NO MEASUREMENT: run the prototype in task 4.1 step 3")
        return 1
    print(
        f"cap {RECORDED_CAP} cells "
        f"({draw_calls(RECORDED_CAP)} draw calls, "
        f"{crop_texture_bytes(RECORDED_CAP) / 1e6:.1f} MB, "
        f"{RECORDED_FRAME_MS:.1f} ms/frame measured)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

> **This script fails closed on a missing measurement and asserts nothing else.** An earlier
> draft additionally refuted the cap against a `draw_calls(cap) > 4096` ceiling; that 4096
> was invented here and appears in no spec. **Dropped** (user ruling, 2026-08-26) — a
> measured cap validated against a guessed budget inverts the point. The script's whole job
> is to stop phase 4 proceeding on an unmeasured number, and to keep the measurement beside
> the number it justifies.
>
> This is the **only** surviving logic-validation script of the three originally proposed,
> kept because its number lands in shipped code as a behavioural cap.

- [ ] **Step 2: Run it and confirm it fails on the missing cap**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-26-viewer-viv-rebuild/colony_view_budget.py
```
Expected: prints the table, then `NO MEASUREMENT`, exit **1**. The script failing at this
point is the intended state — it is what stops phase 4 proceeding on an invented number.

- [ ] **Step 3: Measure, then fill in both constants**

Render a prototype grid at 64, 256, 1024 and 1536 cells and record observed frame time at
each. Choose the cap at the largest count that holds an interactive frame budget, write it
into `RECORDED_CAP` **and its measured frame time into `RECORDED_FRAME_MS`**, and record the
full table in `spike/README.md`.

The interactive budget is a judgement, not a spec number — spec §9.1 makes interactivity a
**target**, not a gate. Record what you chose and why; a later reader can re-judge the cap
from `RECORDED_FRAME_MS` without re-running the prototype.

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

> **The napari implementation this ports** (`gui/_smart_grid/`, 378 lines, read in full
> during plan refinement — every claim below carries its `file:line`):
>
> - **Visible-layers-only grid.** `install_smart_grid` (`_install.py:14-15`) shadows exactly
>   two methods on the grid *instance* (`:90-91`). The real predicate (`:64-70`) is
>   `layer.visible and not (_overlay_enabled and is_overlay_layer(layer))` — visible **AND
>   non-overlay**. The overlay exclusion is not a detail: it is what frees the
>   Labels/Points/Shapes layers from owning a cell so they can be cloned into every cell.
>   The two are **one coupled mechanism**, not a sequence. `_overlay_enabled` and
>   `_labels_enabled` are live user toggles (`_grid_popup.py:40-51`), so the reference has
>   two runtime modes.
>   **Not ported, declared:** the colony grid has no visibility mapping at all — napari's
>   cells are per-*layer*, colony cells are per-*region of one image*, so the concept has no
>   analogue. The two user toggles go with it.
> - **Detach + draw-order.** `_install.py:130-136` sets the original overlay visuals'
>   `node.parent = None`, because `patched_position` returns `(-1,-1)` and napari never
>   re-parents them — left attached they draw canvas-wide at `order=100`, on top of
>   everything. Clones then take `order = len(viewer.layers) + 10`
>   (`_overlay_visuals.py:33, :39`). The deck.gl analogue is layer order within each view's
>   stack; state it explicitly when building the stack rather than relying on insertion
>   order.
> - **Overlay cloning.** `create_overlay_clones` (`_overlay_visuals.py:20-42`) iterates
>   `canvas.grid_views × overlay_layers`, calls `create_vispy_layer(layer)` and parents each
>   clone to `viewbox.scene`. `is_overlay_layer` (`:16-18`) is exactly
>   `Labels | Points | Shapes`. Note it clones only **visible** overlay layers (`:26-28`) —
>   "every Labels/Points/Shapes visual" overstates it.
>
> **Correction — napari IS a sync protocol, and the deck.gl design is a declared
> deviation, not a faithful port.** An earlier draft of this plan (and spec §6.2's "sharing
> one camera") claimed the reference shares a camera *value* with "no per-view listener
> reconciling positions". **That is false**, verified in napari's source:
>
> ```text
> napari/_vispy/canvas.py:1121-1123   camera = VispyCamera(view, self.viewer.camera, self.viewer.dims)
>                                     self.grid_views.append(view); self.grid_cameras.append(camera)
> napari/_vispy/camera.py:50-56       self._camera.events.center.connect(self._on_center_change)
>                                     ... zoom / angles / perspective
> napari/_vispy/canvas.py:646-648     # sync all cameras
>                                     for camera in (self.camera, *self.grid_cameras):
>                                         camera.on_draw(event)
> ```
>
> One camera **model**, *N* `VispyCamera` objects, event-connected and re-reconciled on
> every draw — precisely the per-view listener protocol the draft said did not exist.
> (`_smart_grid/` itself contains no camera code; it patches grid geometry only. The
> behaviour is napari's, inherited.)
>
> **The deck.gl translation collapses that fan-out to a single `viewState`.** It is still
> the better design — a shared value cannot tear mid-gesture the way an event-reconciled
> set can — but it is a **deliberate simplification of the reference**, and per the
> `porting-a-reference-algorithm` skill it is recorded as a **declared deviation** requiring
> a user gate, not presented as fidelity. Do not argue "correct by construction" against the
> strawman the draft invented.
>
> **Why `cleanup_clones` is not ported — narrow claim only.** It
> (`_overlay_visuals.py:45-58`) calls `clone.close()` per clone then
> `canvas._scene_canvas.context.finish()` (a glFinish, so deletions land before
> reallocation), reclaiming GPU resources for *N overlay-layers × M viewboxes* separately
> instantiated vispy visuals. It is **not** a nicety in the reference: it runs at the top of
> every scenegraph rebuild (`_install.py:125`) against wholesale recreation at `:137`, and
> rebuilds fire on every visibility toggle, name change and layer insert/remove
> (`:157-183`). Without it those visuals leak per event.
>
> The narrow claim holds: deck.gl reconciles layers by id and finalizes them when dropped,
> and multi-view draws **one** layer instance per view rather than N instances, so there is
> no clone lifecycle to manage. **The broad claim does not.** The leak *class* survives —
> Viv's `TileLayer` texture cache grows with the union of all *N* views' visible tiles,
> bounded only by `maxCacheSize` / `maxCacheByteSize`, and **nothing in this phase sets,
> bounds, or tests it.** Task 4.1's cap bounds *cell count*, not *cached tile bytes*. Set an
> explicit cache bound when building the layer stack, and record its number alongside the
> cap.

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
    states = page.evaluate(
        """() => window.phenotypicViv.__debugViewStates('colony-grid')"""
    )
    zooms = [s.zoom for s in states]
    targets = [tuple(s.target[:2]) for s in states]

    assert len(zooms) > 1
    assert len(set(zooms)) == 1, f"zoom drifted apart: {sorted(set(zooms))}"

    # The complementary half, and the one that matters. Asserting only
    # "all zooms equal" is satisfied PERFECTLY by the bug this test exists
    # to catch: a single shared viewState gives every cell the same target,
    # so the grid renders one colony N times -- with identical zooms.
    assert len(set(targets)) == len(targets), (
        f"every cell is showing the same region: {targets[:4]}"
    )
```

- [ ] **Step 3: Implement `setGridViews` — `target` is per-view, `zoom` is shared**

> **Read this before writing the call.** An earlier draft said "one `OrthographicView` per
> cell, each with its own `x`/`y`/`width`/`height` and a `target` at the colony centroid,
> but **one** `viewState` object shared by all of them." **`target` does not live on a
> `View`.** A deck.gl `View` carries `id`/`x`/`y`/`width`/`height`; `target` and `zoom` live
> in the **viewState**. Built literally as written, every cell inherits the same `target`
> and the grid renders **the same colony N times**.

deck.gl's multi-view viewState is keyed by view id, which is exactly the split needed —
per-view `target`, shared `zoom`:

```javascript
// Views carry GEOMETRY only.
const views = cells.map((c, i) => new OrthographicView({
  id: `cell-${c.id}`, x: layout[i].x, y: layout[i].y,
  width: layout[i].w, height: layout[i].h,
}));

// viewState is keyed BY VIEW ID: target differs per cell, zoom is the
// shared value. This is the split the shared-camera lock actually locks --
// it constrains `zoom`, never `target`.
const viewState = Object.fromEntries(cells.map((c) => [
  `cell-${c.id}`,
  {target: [c.centroidCc, c.centroidRr, 0], zoom: shared.zoom},
]));
```

A "shared viewState object" and "a shared zoom across per-cell targets" are different
things, and only the second renders a grid.

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
| Level selection matches `phenotypic.pyramid`'s ladder, `ceil` boundary included | phase 3 task 3.2 | run it — and assert against the **browser's** choice if task 3.2 retired the server-side stack |
| Staleness — a rewritten nested chunk must invalidate | phase 1 task 1.2 | run it |
| Curation regression — colony curation tests pass **unmodified** | phase 4 task 4.3 step 3 | run it, **and run the three tests that actually prove the chain** (below) |
| Label path — a `gray`-primary store resolves its objmap through `phenotypic.labels.objmap` | phase 3 task 3.1 | run it |

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/shared/test_resolve_within_root.py \
  tests/unit/gui/results_viewer/test_zarr_routes.py \
  tests/unit/gui/results_viewer/test_store_source.py \
  tests/unit/gui/results_viewer/test_level_selection.py \
  tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py \
  -n 4 -v
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -k "viv or colony_shared or builder" -v
```

**The curation chain is not proved by `test_colony_callbacks_helpers.py`.** Its 15 tests
drive pure helpers against hand-built `ctx.triggered` dicts — it would pass unmodified while
phase 4's deck.gl rewrite removed the radial entirely. Run the three that do:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/gui/results_viewer/colony_view/test_grid.py::test_build_grid_tiles_carry_radial_trigger_not_old_remove_button \
  tests/integration/gui/test_triage_callbacks.py::test_colony_wedge_mark_writes_category_parquet_and_drops_mirror \
  tests/unit/cli/test_cli_error_outputs.py -v
```

The first is the **only** assertion anywhere that a cell carries `colony-radial-trigger`,
which is exactly what phase 4 endangers. It lives under `tests/gui/`, so no
`tests/unit/gui` invocation reaches it.

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
routes. Add rows for the new affordances: the Layers panel, the navigator inset and the
pyramid readout.

**Add the shared-camera lock row only if phase 4 actually landed.** Phase 4 is optional and
marked the first thing to cut; a `✅ shipping` row for it makes `check_features_md.py
--strict` resolve refs for an affordance that does not exist. If phase 4 was cut, either
omit the row or file it as `🔭 planned`.

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

Record: the Plate/Colony pixel path is Viv over `/zarr/...`, and the builder preview is Viv
over `/preview-zarr/...` (phase 6); Browse remains libvips → DZI → `BrowseCache` → OSD;
`_dzi_tiler` survives for **Browse and the point picker** — four consumers, not five, once
phase 6 has landed; `_tile_routes.py` survives as a module even with its `.dzi` routes gone,
because the builder imports `_TILE_NAME_RE` and `_json_error` from it; and the façade at
`_assets/viv_viewer.js` is the only thing that may touch `window.__vivBundle`.

**Run this task last, after phase 6.** Both phases edit the same four files
(`FEATURES.md`, `WORKFLOWS.md`, `gui/CLAUDE.md`, the capture script) through the same three
CI gates; doing it once is the point of folding phase 6 in.

- [ ] **Step 4: Run all three gates**

```bash
uv run python scripts/check_features_md.py --strict
uv run python scripts/check_workflows_md.py -v
QT_QPA_PLATFORM=offscreen uv run python scripts/capture_gui_tutorial_screenshots.py --skip-cli
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
# Phase 6 — Builder node preview on Viv

**Spec:** §7. **Depends on:** phases 1-3. **Not** blocked on phase 4.

**Deliverable:** the builder's per-node preview pane renders through the shared Viv façade,
over a session-scoped byte route, with `_dzi_tiler` off the preview path
(`_preview_tiles.py:30, :144`). The point picker is untouched.

> **Folded in from a retired cycle-3 spec** (`specs/2026-08-26-builder-preview-viv/`,
> withdrawn 2026-08-26). The split bought no parallelism — this work is blocked on phases
> 1-3 and reuses all three — while forcing phase 3 to write `build_source_spec` at a
> signature it would then refactor, parking phase 2's own asset-mount question in a separate
> document, and running the CI-gate ledger pass twice.

> **The data half already landed.** `_preview_cache.py:48, :255` write per-node
> `.ome.zarr` stores; `_preview_tiles.py:52-65` reads them via `Image.load_layer_zarr`;
> `:78-87` keys freshness on the root `zarr.json`. Decision E is implemented. **Do not
> rebuild it.** This phase is the client-side render swap only.

> **No scratch garbage-collection work.** `init_cache()` already calls `wipe_cache()` on the
> whole cache root at startup plus an `atexit` wipe (`_preview_cache.py:61-68`), and
> `wipe_scope` reclaims on fingerprint change. That is the stated policy (spec §7). An
> earlier draft specified a measured retention cap, oldest-first eviction and a startup
> sweep; **withdrawn** — the sweep duplicated `wipe_cache`, and the eviction policy
> generated the blank-pane bug its own second test existed to guard.

---

### Task 6.1: The preview byte route

**Files:**
- Create: `src/phenotypic/gui/builder/_preview_zarr_routes.py`
- Modify: `src/phenotypic/gui/builder/_preview_cache.py` (hash-keyed helpers)
- Modify: `src/phenotypic/gui/builder/_preview_tiles.py` (split `_validate`)
- Modify: `src/phenotypic/gui/builder/_app.py`
- Test: `tests/unit/gui/builder/test_preview_zarr_routes.py` (create)

**Interfaces:**
- Consumes: `resolve_within_root` from phase 1 task 1.1, `_validate_scope` (this task).
- Produces: `register_preview_zarr_routes(app)` and
  `preview_zarr_url(url_prefix, session_id, scope_hash, block_id, token)`.

- [ ] **Step 1: Split `_validate` into a channel-free core**

`_validate` (`_preview_tiles.py:107-116`) takes a channel this route does not have. Calling
it with a fabricated `"gray"` works only because `"gray" in _VALID_CHANNELS` makes that
clause constant-true — narrowing `_VALID_CHANNELS` would silently change an unrelated
route's guard. Extract:

```python
def _validate_scope(session_id, scope_hash, block_id) -> Optional[Response]:
    """Shape-validate the session/scope/block triple.

    NOT authentication. Nothing here binds the request to a session -- see
    the capability-URL note in step 2.
    """
    if (
        is_safe_path_component(session_id)
        and bool(_HASH_RE.match(scope_hash))
        and is_safe_path_component(block_id)
    ):
        return None
    return _json_error("invalid preview request", 404)
```

`_validate` becomes `_validate_scope(...) or (channel check)`.

- [ ] **Step 2: Add the hash-keyed cache helpers**

`read_manifest` (`_preview_cache.py:95`) and `scope_dir` (`:83`) key by `scope_path`, a
list; the route receives a hash. No reverse index is needed — `_scope_path` (`:78`) already
names the directory by the hash:

```python
def _scope_path_by_hash(session_id: str, scope_hash_hex: str) -> Path:
    return preview_cache_root() / session_id / scope_hash_hex


def read_manifest_by_hash(session_id: str, scope_hash_hex: str) -> Optional[dict]:
    path = _scope_path_by_hash(session_id, scope_hash_hex) / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
```

Refactor `_scope_path` to delegate, so there is one join.

- [ ] **Step 3: Resolve a block id through the REAL manifest shape**

The manifest is **not** a list of blocks. `_build_manifest` (`:257-263`) returns:

```python
{"version": …, "fingerprint": …, "fingerprint_inputs": …,
 "scope_key": …, "nodes": {block_id: {"store", "layers", "shape", "num_objects"}},
 "error": None}
```

`nodes` is a **dict keyed by block_id**. There is no `"blocks"` list and no `"block_id"`
field — an earlier draft of this plan read `manifest.get("blocks", [])` and would have
404'd every request. Mirror the landed DZI route, which is already correct
(`_preview_tiles.py:127-140`):

```python
def _store_for_block(session_id: str, scope_hash_hex: str, block_id: str) -> Path | None:
    manifest = pc.read_manifest_by_hash(session_id, scope_hash_hex)
    if not manifest:
        return None
    node = manifest.get("nodes", {}).get(block_id)
    if node is None:
        return None
    store = pc._scope_path_by_hash(session_id, scope_hash_hex) / node["store"]
    # The ROOT, not the directory: the promote writes it last, so an
    # interrupted write reads as ABSENT rather than partial.
    return store if (store / ngff_.STORE_ROOT_JSON).is_file() else None
```

- [ ] **Step 4: Write the route on the shared resolver**

```python
def register_preview_zarr_routes(app: dash.Dash) -> None:
    bp = Blueprint("preview_zarr", __name__, url_prefix="/preview-zarr")

    @bp.route("/<session_id>/<scope_hash>/<block_id>/<token>/<path:tail>")
    def preview_store_bytes(session_id, scope_hash, block_id, token, tail):
        invalid = _validate_scope(session_id, scope_hash, block_id)
        if invalid is not None:
            return invalid
        store = _store_for_block(session_id, scope_hash, block_id)
        if store is None:
            abort(404)
        if token != store_generation_token(store):
            abort(409)          # stale generation -- see task 6.1 step 5
        return send_file(
            resolve_within_root(store, tail, allowed_roots=readable_roots_for(store)),
            conditional=True,
        )

    app.server.register_blueprint(bp)
```

**On session scoping — write the tests against what is actually true.** `_validate_scope`
is a *shape* check. Nothing binds the request to a session. Isolation rests on `session_id`
being `uuid.uuid4().hex` (`builder/_callbacks.py:3662, :4083`) — 122 bits, unguessable —
carried in the URL path, where it reaches access logs, the OOD proxy's logs, browser
history and `Referer`. Spec §7 records this as an **accepted** capability-URL risk (user
ruling). So:

```python
def test_an_unissued_session_id_is_not_served(preview_client):
    """The property that actually holds: no sandbox, no bytes.

    NOT 'session A cannot present session B's id' -- it can, and the route
    serves it, exactly as the existing /preview-tiles/ route does. Asserting
    the stronger property would be asserting a binding that does not exist.
    """
    resp = preview_client.get(
        "/preview-zarr/" + "0" * 32 + "/" + "a" * 40 + "/blk/tok/zarr.json"
    )
    assert resp.status_code == 404
```

An earlier draft asserted session A's id against session B's *hash*, which 404s on a
manifest miss rather than on isolation — it would have passed with every isolation
mechanism deleted.

- [ ] **Step 5: Range, traversal and generation tests**

Reuse phase 1's shared parametrized traversal block over `resolve_within_root`; do not
duplicate it. Add the Range test and a generation test. **The chunk key is `c.0.0`, not
`0.0`** — `ngff_.py:409-412` sets `chunk_key_encoding` separator `"."` and a Zarr-v3
default key is `"c" + sep + indices`, so a 2-D `gray` level-0 chunk is `gray/0/c.0.0`
(phase 1's `rgb/0/c.0.0.0` is the 3-D form). With sharding, a small preview array is one
file on disk regardless.

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/builder/test_preview_zarr_routes.py -v
```

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix src/phenotypic/gui/builder/ tests/unit/gui/builder/
git add src/phenotypic/gui/builder tests/unit/gui/builder
git commit -m "feat(gui): serve builder preview store bytes with Range and a generation token"
```

---

### Task 6.2: Mount the shared bundle and swap the renderer

**Files:**
- Modify: `src/phenotypic/gui/builder/_app.py`
- Modify: `src/phenotypic/gui/builder/_preview_tiles.py` (remove `:30`, `:144`)
- Modify: `src/phenotypic/gui/builder/_preview_callbacks.py`, `assets/preview.js`
- Test: `tests/unit/gui/builder/test_shared_viv_asset.py`, `tests/e2e/gui/test_builder_preview_viv.py`

- [ ] **Step 1: Settle the asset-mount mechanism (spec §10 OQ4)**

```bash
uv run grep -n "assets_folder\|assets_url_path\|serve_locally" \
  src/phenotypic/gui/builder/_app.py src/phenotypic/gui/results_viewer/_app.py \
  src/phenotypic/gui/shell/_app.py
```
Prefer Dash's `assets_folder` / `assets_url_path` pointed at the results viewer's
`_assets/viv/`; fall back to a small Flask route serving the two files from that package
directory. Record which and why in the commit body — this closes OQ4.

- [ ] **Step 2: Assert one artifact, not two**

```python
def test_builder_does_not_carry_its_own_bundle_copy():
    stray = list((Path(builder_pkg.__file__).parent / "assets").rglob("viv-bundle*.js"))
    assert not stray, f"builder has its own bundle copy: {stray}"
```

- [ ] **Step 3: Build the preview source spec through the shared resolver**

Reuse phase 3 task 3.1's `build_source_spec`, which phase 3 already wrote at its
store-path-plus-base-URL signature **because this caller was in view when it was written**.

**`labelPath` is optional here and that is the common case.**
`build_phenotypic_attributes` omits the `labels` key entirely when `has_labels=False`
(`ngff_.py:576-581`), and `save_intermediate_zarr` sets `write_objmap = "objmap" in layers`
(`_image_io_handler.py:1244`) — so most preview stores have **no** `labels` key at all.
Phase 3 task 3.1 already makes `labelPath` optional; this is the caller that proves it.

- [ ] **Step 4: Swap `preview.js` onto the façade**

Replace the OpenSeadragon mount with `window.phenotypicViv.mount` / `.setSource`.

```bash
uv run grep -rn "__vivBundle" src/phenotypic/gui/builder/
```
Expected: no hits — the façade is the only thing that touches the bundle.

- [ ] **Step 5: Unhook `_dzi_tiler` from the preview path**

Remove `_preview_tiles.py:30` and `:144`. Then decide the fate of `stage_channel_png`
(`:74`) and `_channel_to_rgb_uint8` (`:52`): with Viv reading the store directly the
intermediate PNG has no consumer, **and neither does the root-`zarr.json` freshness key at
`:78-87`** — the property the retired spec named as its own test target. Retire the key
with the staging code, or say what still consumes it.

```bash
uv run grep -rn "stage_channel_png\|_channel_to_rgb_uint8\|preview_dzi_url" src/ tests/
```

**Do not delete `_dzi_tiler`, and do not delete `_tile_routes.py`.** The tiler keeps four
consumers; and `_preview_tiles.py:31` imports `_TILE_NAME_RE` and `_json_error` **from
`results_viewer/_tile_routes`**, which `_validate_scope` now returns through — so deleting
that module breaks the builder from a different sub-app. See the plan README's Global
Constraints.

- [ ] **Step 6: Prove the point picker is untouched**

```bash
git diff --stat src/phenotypic/gui/builder/_point_picker.py
QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/builder -k point_picker -q
```
Expected: **empty diff**, tests PASS. Spec §7 makes this the executable statement that the
picker stays on DZI.

- [ ] **Step 7: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/builder -n 4 -q
QT_QPA_PLATFORM=offscreen uv run pytest tests/e2e/gui -k builder -q
uv run ruff check --fix src/phenotypic/gui/builder/
git add -A src/phenotypic/gui/builder tests/unit/gui/builder tests/e2e/gui
git commit -m "feat(gui): render builder node previews through Viv, off the DZI path"
```

---

### Task 6.3: Record the tiler's real consumer set where a deleter will see it

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_dzi_tiler.py` (module docstring)

> An earlier draft added a test pinning the consumer set by text scan. Dropped: it is a
> tripwire, not a check — any legitimate future consumer fails it, and its own failure
> message says the remedy is to edit the test. A docstring reaches the person about to
> delete the module at the moment they open it, which a test in another directory does not.

- [ ] **Step 1: Add the consumer list to the module docstring**

```python
"""Deep-Zoom pyramid generation for the GUI's OpenSeadragon surfaces.

NOT DEAD CODE. Three specs each say the tiler is "removed from this path" --
the results Plate path, and the builder preview path. Read together they
suggest a module that can be deleted. It cannot. Live consumers:

    browse/_app.py:40                  DZI_BACKEND_INFO
    browse/_preparation.py:711         tile()
    browse/_preparation_routes.py:95   DZI_BACKEND_INFO
    builder/_point_picker.py:417       tile()

Browse keeps libvips -> DZI -> BrowseCache -> OSD as its ONLY pixel path
(viewer-viv-rebuild spec section 9), because it reads arbitrary source
images with no run behind them and so has no store to read. The point
picker picks points on a source image before any pipeline node has run.
"""
```

- [ ] **Step 2: Commit with the `gui/CLAUDE.md` edit in phase 5 task 5.3**

```bash
git add src/phenotypic/gui/results_viewer/_dzi_tiler.py
git commit -m "docs(gui): record _dzi_tiler's four remaining consumers on the module"
```
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
- **`_tile_routes.py` also survives as a module**, even after its `.dzi` routes go.
  `builder/_preview_tiles.py:31` imports `_TILE_NAME_RE` and `_json_error` from it, and
  `_validate` returns through `_json_error` — so deleting it breaks the builder preview *and*
  phase 6's new route, at import, in a different sub-app from the one being edited. Same
  shape as the `_dzi_tiler` misreading, one module over.
- **One path-escape guard, not two.** Phases 1 and 6 both resolve a client-controlled tail
  inside a root. That logic lives **once**, as `resolve_within_root` in `gui/_shared/`
  beside `is_safe_path_component`. A path-escape guard is a security primitive and
  correctness is binding (spec §9.1); two copies drift silently, because each phase would
  test only its own copy.
- **The readable-root restriction is enforced on the *resolved* path.** Checking the
  unresolved first URL segment leaves a symlink inside a readable root escaping the
  restriction while passing containment. Resolve, then test
  `resolved.relative_to(root).parts[0]`.
- **Never hard-code the readable series set.** `attributes.phenotypic.series` legitimately
  contains `original` (`_image_io_handler.py:1012-1014`). Derive the readable set per store
  from `series` + `labels`, or invert to a deny-list on `tables/`. A fixed
  `{rgb, gray, detect_mat}` makes the Layers panel list a series the route 404s — the same
  hard-coding the label-path rule forbids, one layer down.
- **`labelPath` is optional.** `build_phenotypic_attributes` **omits** the `labels` key when
  the store carries no label image (`sdk_/ngff_.py:576-581`, ledger C3), and most builder
  preview stores have none. `block["labels"]["objmap"]` `KeyError`s; use `.get`, as the
  landed code does (`tiles.py:485`, `_preview_cache.py:206`).
- **Byte-route URLs carry a generation token.** `promote_store` (`sdk_/ngff_.py:1235-1300`)
  republishes by renaming the whole store directory, and the routes resolve fresh per
  request holding no handle — so without a token a client can combine metadata from promote
  *N* with chunks from *N+1*. Harmless for a run-store re-promote; a decode error or
  plausible wrong pixels for a builder preview, where re-running a node changes extent.
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
| 0 | Spike gate | Four answered questions + a measured chunk-size decision | [phase-0](phase-0-spike-gate.md) |
| 1 | Byte route | `/zarr/...` with Range, a **shared** resolver, and a generation token | [phase-1](phase-1-byte-route.md) |
| 2 | Viv bundle + façade | `tools/viv-bundle/`, the vendored IIFE, `viv_viewer.js`, NOTICE/licenses | [phase-2](phase-2-viv-bundle-facade.md) |
| 3 | Plate surface | Full-canvas Viv Plate; `_dzi_tiler` off the results plate path | [phase-3](phase-3-plate-surface.md) |
| 4 | Colony D1 | One `OrthographicView` per colony, shared `viewState`, measured cap | [phase-4](phase-4-colony-views.md) |
| 5 | Verification & ledgers | Spec §8's checks, FEATURES/WORKFLOWS, tutorial refresh | [phase-5](phase-5-verification.md) |
| 6 | Builder preview | Preview byte route, shared asset mount, render swap | [phase-6](phase-6-builder-preview.md) |

**Phase 4 is separable and is the first thing to cut.** Colony "D3" — the crop route reading
level-0 store chunks — is **already landed** (DRIFT.md D-2), so phase 4 is purely the
deck.gl rendering half and the viewer ships without it.

**Phase 6 was a separate spec+plan cycle and was folded in on 2026-08-26.** It depends on
phases 1-3 and reuses all three, so the split bought no parallelism while forcing phase 3 to
write `build_source_spec` at a signature it would then refactor. It does **not** depend on
phase 4. Run phase 5 last, after 6, so the ledger pass happens once.

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
