# Tile-Surroundings Dimming (Spotlight Crops) — Design Spec

**Status:** Ready for implementation-plan
**Branch target:** `feature/smart-qc-gui`
**Author:** Design brainstorm (Claude + Alex)
**Date:** 2026-05-29

---

## Problem Statement

The results-viewer's two image-tiling surfaces render each colony as a
**centroid-centered, fixed-size crop** of the dataset's overlay PNG:

- the **colony-view grid** (`results_viewer/colony_view/_grid.py`), and
- the **QC Review gallery** (`results_viewer/_qc_tab/review/_callbacks.py`).

Both point their tile `<img>` tags at **one shared Flask crop route**
(`register_crop_route` in `gui/_shared/tiles.py`), which looks the colony
up in `OutputRoot.master_df`, opens the overlay PNG, and returns a
`size`×`size` crop centered on the colony's `Bbox_CenterRR/CC`.

On a **crowded plate** that fixed window pulls neighbouring colonies into
the crop. The target sits dead-center, but a reviewer cannot reliably
tell *which* colony in the tile is the one being measured — especially in
the QC Review detail gallery, where triage decisions (remove / restore)
ride on correctly identifying the target.

We already have everything needed to disambiguate it: the target's
bounding box (`Bbox_MinRR`, `Bbox_MaxRR`, `Bbox_MinCC`, `Bbox_MaxCC`) is
in `master_df`, and the crop geometry is known at render time. We can use
the bbox to **spotlight the target**: leave the bbox untouched and fade
everything outside it to black, so each tile reads as a black backdrop
with the measured colony lit up.

## Goals

- In every tile, keep the target's bounding box at full opacity and
  **blend the surrounding pixels toward black**, so the measured colony
  is unambiguous even on a crowded plate.
- Apply to **both** tiling surfaces through **one shared code path** (the
  crop route), so behaviour can never drift between them.
- Give the reviewer a **`−` / `+` strength stepper** (step `0.05`,
  **default on**) so the spotlight can be softened or turned off (down to
  `0.0` = today's full-context crop).
- Keep the change **server-side and cache-friendly**: strength rides as a
  `?dim=` query param on the crop URL, exactly like the existing `?size=`.

## Non-Goals

- **No mask-precise dimming.** v1 keeps the axis-aligned **bbox
  rectangle**; pixels of a neighbour that fall *inside* the target's bbox
  stay lit. (Pixel-precise object-mask dimming would need the labelled
  object map threaded into the crop route; explicitly out of scope.)
- **No soft feather.** The keep-region edge is a **hard rectangle**
  (chosen in brainstorm) — inside = 100%, outside = dimmed at full
  strength, crisp boundary.
- **No new measurement columns / schema changes** — we read existing
  `Bbox_*` columns.
- **No change to the overlay PNGs on disk** — dimming happens at
  crop-serve time only.

## Design Decisions (resolved in brainstorm)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Where it lives | **Shared crop route** (`gui/_shared/tiles.py`: `crop_overlay` + `register_crop_route`) | Both surfaces already serve tiles through it; one change covers both. |
| Keep region | **BBox rectangle** (axis-aligned, from `master_df`) | Needs only existing columns; no objmap threading. |
| Edge | **Hard rectangle** | Crisp, cheap, unambiguous about the target. |
| Dim target colour | **Black** `(0, 0, 0)` | "Add a black background to each tile" — matches the existing black `pad_value` so the out-of-image padding and the dimmed surroundings are one continuous black backdrop. |
| Strength control | **`−` / `+` stepper, step `0.05`, default on** | User preference (not a slider). Backed by a shared store; rides the crop URL as `?dim=`. |
| Scope | **Both surfaces** | Colony grid + QC Review gallery, via the shared route. |

## Proposed Design

### Where the effect is applied

The tile raster is produced once, server-side, in
`crop_overlay(png_path, center_rr, center_cc, size, …)`
(`gui/_shared/tiles.py`). That function already builds a `size`×`size`
RGB canvas and pastes the (edge-clamped) overlay region onto it. We add a
**post-paste dim pass**: blend every pixel **outside the target's bbox
rectangle** toward black by `dim_alpha`, leaving the bbox interior
untouched.

Because both surfaces fetch tiles through `register_crop_route` →
`crop_overlay`, this single change spotlights tiles in the colony grid
**and** the QC Review gallery.

> **Per-tile target.** Each tile's *own* colony is its target. The crop
> is centered on that colony's centroid and dimmed around that colony's
> bbox. There is no notion of a globally "active" tile here — every tile
> spotlights itself.

### Computing the keep-rectangle (bbox → crop-local pixels)

`crop_overlay` computes the crop window from an **unclamped origin**:

```
half     = size // 2
origin_r = round(center_rr) - half      # top of the size×size canvas, image coords
origin_c = round(center_cc) - half      # left of the canvas, image coords
```

The clamped source region is pasted back at the offset that re-aligns it
with this unclamped origin (existing `paste_x/paste_y` logic), so the
canvas is always addressed in the same `origin`-relative frame — **even
when the colony is near an image edge.** The bbox keep-rectangle in
canvas pixels is therefore:

```
keep_top    = clamp(round(Bbox_MinRR) - origin_r, 0, size)
keep_bottom = clamp(round(Bbox_MaxRR) - origin_r, 0, size)
keep_left   = clamp(round(Bbox_MinCC) - origin_c, 0, size)
keep_right  = clamp(round(Bbox_MaxCC) - origin_c, 0, size)
```

No separate edge-case path is needed: the same `origin` drives both the
paste and the keep-rect, so they always agree.

### The dim pass (hard rectangle, toward black)

Vectorised with NumPy (tiles are ≤ a few hundred px, so this is cheap and
the transient array is tiny):

```python
import numpy as np

def _dim_outside_bbox(canvas, keep, *, alpha, bg=TILE_DIM_RGB):
    """Blend pixels outside the keep-rectangle toward bg by alpha.

    canvas : (size, size, 3) uint8   the assembled crop
    keep   : (top, left, bottom, right) in canvas px
    alpha  : 0.0 (no dim) .. _DIM_MAX
    """
    if alpha <= 0.0:
        return canvas
    out = canvas.astype(np.float32)
    mask = np.ones(canvas.shape[:2], dtype=bool)
    t, l, b, r = keep
    if b > t and r > l:
        mask[t:b, l:r] = False            # keep-rect = not dimmed
    bgv = np.asarray(bg, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + bgv * alpha
    return out.astype(np.uint8)
```

`crop_overlay` converts its PIL canvas to a NumPy array, runs this, and
converts back before PNG-encoding. With `bg = (0, 0, 0)`,
`out = px * (1 - α)` — a straight darken toward black.

### `crop_overlay` signature change

```python
def crop_overlay(
    png_path, center_rr, center_cc, size,
    pad_value=(0, 0, 0),
    *,
    dim_alpha: float = 0.0,
    bbox: tuple[float, float, float, float] | None = None,  # (min_rr, max_rr, min_cc, max_cc)
) -> bytes:
```

- `dim_alpha <= 0` **or** `bbox is None` → byte-for-byte identical to
  today's output (regression guard — see *Testing*).
- Otherwise → assemble canvas, compute keep-rect from `bbox` + the
  origin, run `_dim_outside_bbox`, encode.

### Crop-route changes (`register_crop_route`)

1. **Pull the bbox columns.** Extend the existing master lookup
   `.select(["Bbox_CenterRR", "Bbox_CenterCC"])` to also select
   `Bbox_MinRR`, `Bbox_MaxRR`, `Bbox_MinCC`, `Bbox_MaxCC`. When any are
   absent (older outputs), pass `bbox=None` → no dimming (graceful
   degrade, never a 500).
2. **Parse `?dim`.** Float query param, **clamped** to
   `[TILE_DIM_MIN, TILE_DIM_MAX]`; **defaults to `0.0`** when omitted (so any
   legacy / cached URL without `dim` stays undimmed). Clamp rather than
   400 — a stray value should soften the spotlight, not break the tile.
3. Pass `dim_alpha` + `bbox` into `crop_overlay`.

The `?size=` validation, path-traversal guard, 400/404 paths, and the
`no-cache` response header are unchanged.

### Threading strength into the tile URLs

The shared `build_tile_cell` / `build_tile_grid` take a
`url_builder(dataset, image_file, label, crop_size) -> str` callable.
**Keep that protocol unchanged** — each surface binds the current alpha
via a partial so the shared `_shared/tiles.py` API does not move:

- **Colony grid** (`colony_view/_grid.py`): `_colony_crop_url` gains a
  keyword `dim_alpha` and appends `&dim={alpha}`. `build_grid` gains a
  `dim_alpha: float = 0.0` param, threaded through `_build_cell` →
  `build_tile_cell(url_builder=partial(_colony_crop_url, dim_alpha=…))`.
  The stack-popover crop URLs (`build_stack_popover_rows`) append the same
  `&dim=` for consistency.
- **QC Review** (`_qc_tab/review/_callbacks.py`): `_qc_crop_url` gains
  `dim_alpha` and appends `&dim={alpha}`. `_render_faceted_gallery` /
  `_render_detail` pass `build_tile_grid(url_builder=partial(_qc_crop_url,
  dim_alpha=…))`. (`build_tile_grid` already accepts `url_builder`, so no
  shared-API change here at all.)

### The `−` / `+` stepper + shared store

- **Shared store** `STORE_TILE_DIM_ALPHA` (declared in
  `results_viewer/_ids.py`, mounted once in the viewer layout),
  `storage_type="local"` so the preference survives a reload, default
  `TILE_DIM_DEFAULT` (**default on**). One store drives **both** surfaces
  so the spotlight strength is consistent across tabs (mirrors how the
  curated removal set is shared).
- **Control** — a compact `html.Div` in each toolbar:
  `[ − ]  dim 0.60  [ + ]`. A `−` button, a readout `Span`, a `+` button.
  Colony toolbar (`colony_view/_layout.py::_build_toolbar`) and QC Review
  toolbar (`_qc_tab/review/_layout.py::_build_toolbar`) each mount one
  instance with **distinct ids** (`COLONY_DIM_*`, `QC_REVIEW_DIM_*`).
- **Stepper callback(s)** — `−` / `+` clicks update `STORE_TILE_DIM_ALPHA`
  (`allow_duplicate=True`), stepping by `TILE_DIM_STEP` and clamping to
  `[TILE_DIM_MIN, TILE_DIM_MAX]`. The arithmetic lives in a **pure,
  module-level helper** (`step_dim_alpha(current, direction) -> float`)
  so it is unit-testable without Dash (per project memory on testable
  callback bodies). Each toolbar's readout `Span.children` is updated from
  the store.
- **Re-render on change** — add `Input(STORE_TILE_DIM_ALPHA, "data")` to
  the colony-grid render callback (`_render_colony_grid`) and the QC
  detail render callback (`_render_detail`), so the tiles rebuild with the
  new `&dim=` (same pattern as the existing tile-size slider input).

### New constants

- `gui/_design.py` (visual token): `TILE_DIM_RGB = (0, 0, 0)` — the
  blend-toward colour. (Kept in `_design.py` with the other palette
  values; it is a colour, and `_design.py` stays dash-free so the crop
  route can import it.)
- `gui/_config.py` (shared numeric policy, needed by **both** toolbars):
  `TILE_DIM_DEFAULT = 0.6`, `TILE_DIM_STEP = 0.05`,
  `TILE_DIM_MIN = 0.0`, `TILE_DIM_MAX = 0.9`.
- `gui/_shared/tiles.py` (route-side clamp bounds): reuse
  `TILE_DIM_MIN` / `TILE_DIM_MAX` from `_config.py` (single source of
  truth — the UI clamp and the route clamp must agree), mirroring how
  `_MIN_CROP_SIZE` / `_MAX_CROP_SIZE` already bound `?size=`.

## Phasing

### Phase 1 — Raster dim pass (P0, pure)
`_dim_outside_bbox` + keep-rect math + `crop_overlay` `dim_alpha`/`bbox`
params + `_design`/`_config` constants. Fully unit-testable, no Dash.

### Phase 2 — Crop route (P0)
`register_crop_route` pulls bbox columns, parses/clamps `?dim`, forwards
to `crop_overlay`. Route test for parse/clamp/default + graceful degrade
when bbox columns absent.

### Phase 3 — URL threading + stepper UI (P0)
`dim_alpha` partials in both surfaces' url builders; shared store; `−`/`+`
controls in both toolbars; render callbacks gain the store input.
`FEATURES.md` rows + live-browser verification.

## Testing

Unit (pure, no Dash):

- **Regression guard:** `crop_overlay(..., dim_alpha=0.0)` and
  `crop_overlay(..., bbox=None)` each return bytes identical to the
  pre-feature output.
- **Dim correctness:** synthetic all-white overlay, known centroid + bbox
  + size + `alpha` → inside-bbox pixels still `255`; outside pixels equal
  `round(255 * (1 - alpha))`.
- **Keep-rect geometry:** known `(center, bbox, size)` → expected
  `(top, left, bottom, right)`, including the **edge-clamped** case
  (colony near the image border) — assert the keep-rect still frames the
  bbox (no origin drift).
- **`step_dim_alpha`:** `+`/`−` step by `0.05`, clamp at `TILE_DIM_MIN` /
  `TILE_DIM_MAX`, float-safe (no `0.6000000001` drift — round to 2 dp).

Route / integration:

- `?dim` parse + clamp + default-`0.0`; lookup selects bbox columns;
  bbox-absent path degrades to undimmed (no 500); existing `?size`
  400/404 paths unchanged.
- Colony render and QC render thread the store alpha into tile URLs
  (`&dim=` present and equal to the store value).

Live GUI (required — callback-wiring bugs only surface on
`/_dash-update-component`, per project memory):

- Drive `−` / `+` on **both** the colony tab and the QC Review tab with
  the Playwright MCP; assert each tile `<img src>` `dim` param updates and
  the crop visibly darkens around the target; tail the viewer log for
  callback 500s.

## CI / docs gates (touches `src/phenotypic/gui/`)

- **`FEATURES.md` gate:** add a row per new affordance — the colony-tab
  dim stepper, the QC-Review dim stepper, and the shared
  `STORE_TILE_DIM_ALPHA` — each `✅ shipping` row carrying a `Test ref`
  (pre-commit validates this).
- **`WORKFLOWS.md`:** this enhances existing surfaces rather than adding a
  new end-to-end flow → no new workflow row / tutorial page expected;
  confirm `workflows-md-gate` stays green.
- **Screenshots:** the steppers add visible chrome — run
  `uv run python scripts/capture_gui_tutorial_screenshots.py` and commit
  the **full** regenerated set (do not cherry-pick the collateral churn).

## Risks & Open Questions

- **Neighbour inside the bbox.** An axis-aligned bbox keeps a neighbour
  lit if it overlaps the target's bounding box. Documented v1 limitation;
  the motivation for a future mask-precise upgrade.
- **Crop is centroid-centered, bbox may be slightly off-center.** Fine —
  the keep-rect is computed from the bbox min/max in the same origin
  frame, so it tracks the bbox wherever it sits in the canvas.
- **Default-on changes the first-load look.** Intended (the spotlight is
  the point); `−` to `0.0` restores today's full-context crop. The store
  default is the single knob if we want to ship off-by-default instead.
- **Cache keying.** Tile PNGs are keyed by URL; adding `?dim=` to the URL
  makes each strength its own cache entry. The route sends
  `Cache-Control: no-cache` today, so no stale-dim risk; the decoded
  *source* overlay LRU (`_load_overlay_rgb`) is upstream of the dim pass
  and unaffected.
- **Shared vs per-surface store.** Spec targets one shared store for a
  consistent cross-tab setting. If the dual-toolbar `allow_duplicate`
  wiring proves fiddly, a per-surface store is an acceptable fallback (two
  independent strengths) — note it in the plan if taken.

---

*Generated with Claude Code.*
