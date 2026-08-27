# Builder node preview on Viv: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The builder's per-node preview pane renders its layer by reading zarr chunks in
the browser through Viv, over a session-scoped byte route — with no server-side DZI
pyramid built for it, and with the scratch stores garbage-collected under a measured
retention policy.

**Architecture:** Reuse, not rebuild. The preview already writes and reads `.ome.zarr`
stores; only the render path is still DZI. So this plan adds a second byte blueprint over
the builder sandbox, mounts the results viewer's *existing* Viv bundle and façade in the
builder app, swaps the preview pane's renderer, and gives the scratch stores a lifecycle
they currently lack.

**Tech Stack:** Viv + deck.gl (the vendored bundle from the Viv rebuild), zarrita.js,
Flask `send_file(conditional=True)`, Dash clientside callbacks, Python 3.11+, `uv`.

**Spec:** [`docs/superpowers/specs/2026-08-26-builder-preview-viv/design.md`](../../specs/2026-08-26-builder-preview-viv/design.md)

**Blocked on:** [`2026-08-26-viewer-viv-rebuild`](../2026-08-26-viewer-viv-rebuild/README.md)
phases 1-3 — the byte route pattern, the bundle, and the façade. **Not** blocked on its
phase 4.

**Baseline:** branch `feat/gui-ome-zarr-sync`, restacked onto
`worktree-ome-zarr-image-store` head `bf0d01a1`.

---

## Global Constraints

Everything in the removals plan's Global Constraints applies — `uv` only,
`QT_QPA_PLATFORM=offscreen`, never `-n auto`, explicit ruff paths, the known-failing
baseline test, the three `gui-checks` CI gates. Additionally:

- **The data half is already landed. Do not rebuild it.** Verified in this worktree:
  `_preview_cache.py:48` (`BASE_STORE_NAME = "base_00.ome.zarr"`), `:255`
  (`f"{i:02d}_{op_key}.ome.zarr"`), `_preview_tiles.py:52-65` (`Image.load_layer_zarr`),
  `:78-87` (freshness on `ngff_.STORE_ROOT_JSON`). Viv-rebuild spec §7's decision E is
  **implemented**; this plan is the render swap only.
- **`_dzi_tiler` keeps four consumers after this plan** — `browse/_app.py:40`,
  `browse/_preparation.py:711`, `browse/_preparation_routes.py:95`, and
  `builder/_point_picker.py:417`. Only `_preview_tiles.py:30, :144` come off. The module is
  not a deletion candidate.
- **The point picker is out of scope** (spec §4). Its tests must pass **unmodified**; a
  diff in `tests/` touching the point picker means the plan overreached.
- **Session isolation is a security property.** `_preview_tiles.py:107` `_validate` is what
  keeps one browser session out of another's sandbox. The new route **reuses** it; it does
  not reimplement it.
- **One bundle, not two.** Committing a second ~1 MB copy of the Viv artifact into
  `builder/assets/` would put two artifacts under one build recipe, drifting
  independently. See spec §1 and phase 2.

---

## Phases

| # | Phase | Deliverable | Doc |
|---|---|---|---|
| 1 | Preview byte route | `/preview-zarr/...` with Range, session-scoped, traversal-guarded | [phase-1](phase-1-preview-byte-route.md) |
| 2 | Bundle reuse + render swap | Builder mounts the shared façade; preview pane renders through Viv; `_dzi_tiler` off the preview path | [phase-2](phase-2-render-swap.md) |
| 3 | Scratch lifecycle | Measured retention cap, session-exit sweep | [phase-3](phase-3-scratch-lifecycle.md) |
| 4 | Verification & ledgers | Spec §6's six checks, FEATURES/WORKFLOWS, tutorial | [phase-4](phase-4-verification.md) |

## Definition of done

1. A ranged chunk request to the preview route returns `206`.
2. A request carrying session A's id cannot reach session B's sandbox.
3. Re-running a node with changed parameters changes what the browser renders — proven by
   rewriting a **nested chunk**, which does not move the store directory's `st_mtime_ns`.
4. The point picker's tests pass **unmodified**.
5. The retention cap is enforced oldest-first and never evicts the focused scope.
6. `uv run pytest tests/unit/gui -n 4` green (minus the known baseline failure); the three
   `gui-checks` gates exit 0.
