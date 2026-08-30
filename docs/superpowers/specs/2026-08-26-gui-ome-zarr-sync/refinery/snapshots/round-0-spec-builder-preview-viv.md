# Builder node preview on Viv

**Date:** 2026-08-26
**Status:** Draft
**Scope:** The builder's per-node pipeline preview — its pixel path, its cache lifecycle,
and the point picker that shares its tiler.

## Summary

The builder previews a pipeline against an **input image** and shows each node's output
layer. Viv-rebuild spec §7 (decision E) chose an **ephemeral store**: each preview node
writes its layer to a scratch `.ome.zarr` under the builder-tiles sandbox, reusing the CLI
writer, so the viewer reads it exactly like a real run.

**That half has already landed.** This spec covers only what remains.

| Piece | Decision E's plan | Actual state |
|---|---|---|
| Preview nodes write `.ome.zarr` | to build | **landed** — `_preview_cache.py:48` `BASE_STORE_NAME = "base_00.ome.zarr"`, `:255` `f"{i:02d}_{op_key}.ome.zarr"` |
| Preview reads the store | to build | **landed** — `_preview_tiles.py:52-65` via `Image.load_layer_zarr` |
| Freshness keyed on root `zarr.json` | to build | **landed** — `_preview_tiles.py:78-87` against `ngff_.STORE_ROOT_JSON` |
| **Browser renders the store directly** | implied | **remaining** — still `_dzi_tiler.tile` → DZI → OpenSeadragon (`_preview_tiles.py:144`) |
| **Scratch-store garbage collection** | named as a cost | **partially** — `wipe_cache` / `wipe_scope` exist; retention policy unstated |

So the remaining work is the **client-side render swap**: the same DZI → Viv change the
Plate surface makes in
[2026-08-26-viewer-viv-rebuild](../2026-08-26-viewer-viv-rebuild/design.md) §6.1, applied to
the builder's preview pane and reusing that spec's byte route, bundle and façade.

## Objective

A builder preview node renders its layer by reading zarr chunks in the browser, with no
server-side DZI pyramid built for it, and with the scratch stores garbage-collected under a
stated policy.

## Non-goals

- **No change to what the preview computes.** `_run_operations`' per-node execution, the
  scope/fingerprint model, and the manifest shape are untouched.
- **No change to the point picker's pixel path.** `builder/_point_picker.py:417` keeps
  `_dzi_tiler.tile`. It picks points on a *source image*, not on a pipeline node's store,
  so it has no store to read. See §4.
- **No pyramiding change.** Preview stores stay single-level where they are single-level
  today; pyramiding is mostly wasted at preview resolutions, which decision E already
  accepted as a cost.
- **`_dzi_tiler` is not deleted.** It keeps five consumers; see viv-rebuild §9 and this
  spec §4.

### Non-functional requirements

Inherited from [viv-rebuild §9.1](../2026-08-26-viewer-viv-rebuild/design.md), unchanged:
**correctness is binding; interactivity is a target.** Two consequences specific to the
builder:

- **Session isolation is a correctness requirement, not a performance one** (§2). A preview
  route that is fast and leaks one session's sandbox into another's has failed the binding
  requirement.
- **The scratch retention cap is a resource bound, not a speed target** (§3). It exists so a
  long authoring session does not grow without limit in bytes or inodes. Its number is
  measured (§6); until measured there is no cap to defend.

*Settled by the user, 2026-08-26.*

## Locked decisions

| # | Decision | Chosen | Note |
|---|---|---|---|
| A | Renderer | **Reuse the results viewer's Viv bundle and façade** | §1 |
| B | Byte route | **A second blueprint over the preview sandbox, not the results route** | §2 |
| C | Point picker | **Stays on DZI** | §4 |
| D | Scratch lifecycle | **Explicit retention policy, session-scoped** | §3 |
| — | Sequencing | **Cycle 3; blocked on the Viv rebuild's phases 1-3** | §5 |

## 1. Decision A — one bundle, two mounts

The builder is a separate Dash app (`gui/builder/`) with its own `assets/` folder, but the
Viv bundle is ~1 MB-class and committing a second copy would put two artifacts under one
build recipe, drifting independently.

**Chosen:** the builder serves the *same* vendored artifact. Either the shell's
`DispatcherMiddleware` exposes one shared asset path, or the builder's `assets/` carries a
symlink-equivalent registration pointing at the results viewer's copy.

The façade contract is unchanged: `mount`, `setSource`, `setViewState`,
`setLayerVisibility`, `destroy`, and the builder's clientside callbacks talk only to it.

**Cost accepted:** a coupling between two sub-apps that are otherwise independent. It is
the cheaper of the two couplings — the alternative couples them through a build recipe
neither app owns.

## 2. Decision B — a preview byte route

The results viewer's `/zarr/<dataset>/<stem>.ome.zarr/<path...>` route resolves through
`OutputRoot.store_path`, which is a **run output root**. Preview stores live under the
builder sandbox at `preview_cache_root()/<session>/<scope_hash>/<NN>_<op>.ome.zarr` and
have no `OutputRoot` at all.

**Chosen:** a second blueprint,
`/preview-zarr/<session_id>/<scope_hash>/<block_id>/<path...>`, mirroring the existing
`preview-tiles` route's shape and reusing its `_validate` guard for the first three
segments plus per-segment `is_safe_path_component` on the tail.

`send_file(..., conditional=True)` for HTTP Range, exactly as the results route — the
requirement comes from sharding, which is a property of the store format and not of which
app serves it.

**The session-scoping is a security property, not a URL convention.** The existing
`_validate` at `_preview_tiles.py:107` is what keeps one browser session out of another's
sandbox, and the new route inherits it rather than reimplementing it.

## 3. Decision D — scratch lifecycle

Decision E named "a scratch dir to garbage-collect" as an accepted cost but set no policy.
Left unstated, a long builder session accumulates one store per node per scope revision.

**Chosen:**

- `wipe_scope(session_id, scope_path)` already runs when a scope's fingerprint changes;
  that stays the primary reclamation point.
- Add a **session-exit sweep**: stores whose session id has no live Dash session are
  removed at builder startup.
- Add a **recorded cap** on retained scope revisions per session, enforced oldest-first.

The cap's number is measured, not chosen — one preview store's on-disk size times a
realistic node count is the budget, and §6 requires the measurement before the cap lands.

## 4. Decision C — the point picker stays on DZI

`builder/_point_picker.py` reuses `_dzi_tiler.tile` (`:417`), `_TILE_NAME_RE`, and the
mtime-compare manifest logic (`:547`). It operates on a **source image the user is picking
points on**, before any pipeline node has run. There is no `.ome.zarr` behind it, so there
is nothing for Viv to read.

Migrating it would mean writing a store *just to render a picker* — which is decision E's
reasoning applied where it does not pay, since the picker needs one image at one zoom, not
a pyramid.

**Consequence:** after this spec, `_dzi_tiler` still has four consumers —
`browse/_app.py:40`, `browse/_preparation.py:711`, `browse/_preparation_routes.py:95`, and
`builder/_point_picker.py:417`. It is not a deletion candidate and should stop being
described as one.

## 5. Sequencing

Cycle 3 of three. Blocked on viv-rebuild phases 1-3 (byte route, bundle + façade, Plate
surface), because it reuses all three. It is **not** blocked on viv-rebuild phase 4
(Colony), which is itself optional.

## 6. Testing

- **Range on the preview route.** A ranged chunk request returns `206`, not `200`.
- **Session isolation.** A request carrying session A's id must not reach session B's
  sandbox — asserted against a real second sandbox, not a crafted path.
- **Traversal.** Per-segment guard rejects `..` in any position of the tail.
- **Freshness survives the swap.** Re-running a node with changed parameters must change
  what the browser renders. The existing freshness key is the root `zarr.json`
  (`_preview_tiles.py:86`); the test rewrites a **nested chunk**, which does not move the
  store directory's `st_mtime_ns`, so a directory-keyed check would fail it.
- **Point picker unaffected.** Its existing tests pass **unmodified** — that is the
  executable statement of §4.
- **Scratch cap.** The measured cap is enforced oldest-first, and the focused scope is
  never evicted.

Per **`run-phenotypic-test`**: the full `tests/unit` suite is a ~65-minute Slurm job with
`QT_QPA_PLATFORM=offscreen` mandatory.

## 7. Open questions

1. **Shared-asset mechanism (§1).** Whether the shell's `DispatcherMiddleware` can serve
   one `_assets/viv/` to both sub-apps, or whether the builder needs its own registration
   pointing at the same file, is unverified. It is a packaging question, not a design one,
   and either answer satisfies decision A.
2. **The retention cap's number (§3).** Measured during implementation, not set here.
3. **Whether preview stores should pyramid at all.** Today they are single-level, which is
   right for a preview pane; if the preview pane grows a deep-zoom gesture, that changes.
   Out of scope, recorded so the decision is not made by accident.
