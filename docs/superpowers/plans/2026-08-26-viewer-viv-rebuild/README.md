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
