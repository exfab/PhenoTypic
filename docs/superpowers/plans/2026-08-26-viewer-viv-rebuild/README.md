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
over a **single dynamic `viewState` carrying `zoom`**, each view overriding only its own
`target` — one zoom, so it cannot drift.

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
- **Every Viv RENDERING test must launch `channel="chromium"` under `xvfb-run`.** Measured
  in the phase-0 spike: Playwright's default `chromium` launch uses
  `chromium_headless_shell`, which ships **no GL stack at all** —
  `canvas.getContext('webgl2')` returns `null`, and six flag combinations including
  `--enable-unsafe-swiftshader` and `--use-gl=angle --use-angle=swiftshader` do not change
  it. The full `chrome-linux64` build beside it does have one (`libEGL.so`, `libGLESv2.so`,
  `libvk_swiftshader.so`), and additionally needs an X display — bare headless still lost the
  GPU process to `BindToCurrentSequence failed`, and `xvfb-run -a` fixed it.
  **Decode tests need neither.** Without this a deck.gl test reports
  `Failed to create WebGL context` and zero painted pixels — a red that looks like a
  rendering bug and is not one. This constrains phases 3-5's e2e suite and the
  `gui-checks` e2e job.
- **Viv resolves nothing by convention — the resolver is mandatory, not defensive.**
  Phase-0 answered Q1 and Q2 **NO**: `@vivjs/loaders` never reads `OME/zarr.json`, and
  `loadOmeZarr` at a store ROOT fails with `Node not found: v2 array` because
  `loadMultiscales` falls back to `paths = ["0"]`. Pointed at a *resolved series* it works
  with nothing patched. Same for labels: Viv reads neither `ome.labels` nor
  `phenotypic.labels`, but loads the label group fine when handed the resolved path.
  **The upside: since Viv resolves nothing by convention, there is no `rgb/labels/objmap`
  hard-coding to fight — a `gray`-primary store works for free.**
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
| 6 | Builder preview | Preview byte route, shared asset mount, render swap | [phase-6](phase-6-builder-preview.md) |
| 5 | Verification & ledgers — **runs LAST** | Spec §8's checks, FEATURES/WORKFLOWS, tutorial refresh | [phase-5](phase-5-verification.md) |

**Phase 4 is separable and is the first thing to cut.** Colony "D3" — the crop route reading
level-0 store chunks — is **already landed** (DRIFT.md D-2), so phase 4 is purely the
deck.gl rendering half and the viewer ships without it.

**Phase 5 is listed after 6 deliberately: it runs LAST.** Its `-k "viv or colony_shared or builder"` e2e selector cannot pass until phase 6 has landed, and both phases edit the same four ledger files through the same three CI gates. The numbering is kept so ledger and commit references keep resolving.

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
4. A re-promote is refused under a stale token (409), and a rewritten nested chunk is
   served fresh **without** moving the token.
5. `tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py` passes **unmodified**.
6. `uv run pytest tests/unit/gui tests/gui -n 4` green (minus the known baseline failure);
   the three `gui-checks` gates exit 0. **`tests/gui` is not optional here** — browse and
   builder GUI tests live there, and a `tests/unit/gui`-only gate never reaches the code
   these phases edit.
