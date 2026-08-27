# Context brief — GUI simplification, Viv rebuild, builder preview

**Purpose.** Discovery paid once, so reviewers start here and open source only to verify a
specific claim. Every fact below was verified in this worktree on 2026-08-26 by reading the
named file at the named line.

**Worktree:** `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/gui-ome-zarr-sync`
**Branch:** `feat/gui-ome-zarr-sync`, restacked onto `worktree-ome-zarr-image-store`
head `bf0d01a1`. **This restack is load-bearing** — before it, the worktree sat at the spec
commit `21a97d3f` and none of the landed zarr code was visible.

---

## 1. The three pairs under review

| # | Spec | Plan | Cycle |
|---|---|---|---|
| 1 | `specs/2026-08-26-gui-simplification-removals/design.md` | `plans/2026-08-26-gui-simplification-removals/` (README + 6 phases) | 1, unblocked |
| 2 | `specs/2026-08-26-viewer-viv-rebuild/design.md` | `plans/2026-08-26-viewer-viv-rebuild/` (README + DRIFT + 6 phases) | 2 |
| 3 | `specs/2026-08-26-builder-preview-viv/design.md` | `plans/2026-08-26-builder-preview-viv/` (README + 4 phases) | 3 |

Spec 3 was **written during planning** — viv-rebuild §7 carried decision E but explicitly
put it "out of scope for this spec's plan", leaving cycle 3 with no spec.

**Read `plans/2026-08-26-viewer-viv-rebuild/DRIFT.md` before reviewing pair 2.** It is the
single most important document in this set.

## 2. The central fact: the specs predate their own backend

Viv-rebuild spec §1 states:

> This spec is written against [2026-08-18-ome-zarr-image-store], which at time of writing
> is **specification only** — there is no zarr code in `src/`.

**That is false as of this branch.** The store branch landed 248 files including
`sdk_/ngff_.py` (1,506 lines) and `sdk_/_hdf_to_zarr.py`. Consequences, each verified:

| Spec clause | Reality |
|---|---|
| §4.1 four staleness traps to fix | **fixed** — `_tile_routes.py:476-479, :496, :503, :527`; `_preview_tiles.py:78-87` |
| §4.2 `_load_hdf_layer_rgb` → `_load_zarr_layer_rgb` | **done** — `_shared/tiles.py:423, :454`; imported at `_tile_routes.py:31` |
| §6.2 Colony "D3" crop re-sourcing | **done** — `_shared/tiles.py:665` `crop_colony` prefers `crop_store_rgb` (`:545`) |
| §7 decision E, preview writes `.ome.zarr` | **done** — `_preview_cache.py:48, :255`; `_preview_tiles.py:52-65` |
| §3.4 backend: Stage 2 writes objmap in store | **inverted** — see §3 below |

**Reviewers: do not raise "the plan omits §4.1" as a gap.** It is inherited, and DRIFT.md
records why. Do raise anything DRIFT.md gets wrong.

## 3. The falsified claim (spec-change candidate)

The backend spec §3.4 and viv-rebuild §6.2's rationale both say Stage 2 overwrites
`labels/objmap` in the promoted store, so "the GUI can render a real objmap mid-run".

The **landed** engine inverted this. From the worktree's own `CLAUDE.md`:

> Stage 2 reads that store **read-only** and never writes into it; its result is a
> **Stage-2 signal** under `.phenotypic/progress/`: the retained **raw** detector output
> `stage2_raw/<ds>/<stem>.npy` plus a consumable **token** `stage2_done/<ds>/<stem>.json`.
> Stage 3 replays the raw array, measures, re-promotes the store, and consumes the token.

So between Stage 1 and Stage 3 the in-store objmap is **zeros**. Viv plan phase 3 task 3.4
handles it; the spec text is not yet amended.

## 4. Architecture — GUI

Five Dash sub-apps mounted under one URL by Werkzeug `DispatcherMiddleware`
(`gui/shell/_app.py`), plus standalone entry points. Ledgers at
`src/phenotypic/gui/FEATURES.md` and `src/phenotypic/gui/WORKFLOWS.md` — **not** the repo
root, which is where both specs cite them from.

### Pixel paths today

| Surface | Path |
|---|---|
| Results Plate | store → `_load_zarr_layer_rgb` → PNG → `_dzi_tiler.tile` → DZI → OpenSeadragon |
| Results Colony crops | store → `crop_store_rgb` → PNG bytes (overlay fallback) |
| Builder node preview | per-node `.ome.zarr` → `Image.load_layer_zarr` → PNG → `_dzi_tiler.tile` → DZI → OSD |
| Builder point picker | source image → PNG → `_dzi_tiler.tile` → DZI → OSD |
| Browse | source image → libvips → DZI → `BrowseCache` → OSD |

So the **server already reads zarr**; what remains is that it then builds a DZI pyramid
from what it read. The rebuild deletes that step for two of the five.

### `_dzi_tiler` consumers — six today, four after all three plans

```text
results_viewer/_tile_routes.py:34, :458, :551   <- removed by viv plan phase 3
builder/_preview_tiles.py:30, :144              <- removed by preview plan phase 2
browse/_app.py:40                               <- stays (spec §9)
browse/_preparation.py:711                      <- stays
browse/_preparation_routes.py:95                <- stays
builder/_point_picker.py:417                    <- stays (preview spec §4)
```

Three specs each say the tiler is "removed from this path". Read together they suggest a
dead module. **It is not**, and preview plan phase 4 task 4.2 adds a test pinning the
consumer set.

### Key landed helpers reviewers will want

| Symbol | Location | Note |
|---|---|---|
| `select_pyramid_level` | `_shared/tiles.py:378` | reads the store's recorded ladder |
| `_read_store_level` | `_shared/tiles.py:477` | windowed; pulls covering shards only |
| `crop_store_rgb` | `_shared/tiles.py:545` | |
| `crop_colony` | `_shared/tiles.py:665` | store-first, overlay fallback; deliberately does **not** catch `StoreUnreadable` |
| `is_safe_path_component` | `_shared/tiles.py:755` | the path guard both new routes reuse |
| `_store_content_token` | `_tile_routes.py:~520` | `paths_fingerprint([root_json])` + mtime |
| `_validate` | `_preview_tiles.py:107` | session/scope/block guard the preview route reuses |
| `STORE_ROOT_JSON` | `sdk_/ngff_.py` | the freshness key for every store |

## 5. Store contract the client reads

Root `zarr.json` → `attributes.phenotypic`:

```json
{"store_schema_version": 3, "metadata_schema_version": 2,
 "image_class": "GridImage", "work_id": "…",
 "series": {"rgb": "rgb", "gray": "gray", "detect_mat": "detect_mat"},
 "labels": {"objmap": "rgb/labels/objmap"},
 "pyramid": {"levels": 4, "stop_px": 512,
             "downsample": {"image": "mean", "label": "nearest"}}}
```

Hard rules, both from backend §1.1/§1.3:

- **Never hard-code `rgb/labels/objmap`.** Read `phenotypic.labels.objmap`. Primary series
  is `rgb` when present, `gray` otherwise; a `gray`-primary store has no `rgb` group.
- **Never recompute the pyramid ladder.** `levels = ceil(log2(max(H,W)/512)) + 1`; a draft
  used `floor` and terminated one level early. Read `phenotypic.pyramid`.

Chunks `(1,1024,1024)`, shards `(C,4096,4096)`, codec `zstd`, chunk-key separator `"."`
(one path segment, `c.0.0.0`) — the separator is a Windows `MAX_PATH` measure and is
mandatory store-wide.

**Sharding is why HTTP Range is load-bearing:** a sharded read is a shard-index fetch then a
byte-range fetch. Without `conditional=True` every tile pulls a whole shard (up to 96 MB).

## 6. Store contents NOT in the spec's layout diagram

Measurements now live at `tables/measurements/table.parquet` **inside each store**, and
forward runs write no external per-image parquet. Neither spec's §1 layout shows a
`tables/` group. The Viv byte route serves an arbitrary-depth tail, so an unrestricted
route serves the measurement table to any browser. Viv plan phase 1 restricts it via
`_READABLE_ROOTS`; **that narrowing is flagged as needing user sign-off** and is
deliberately not treated as settled.

## 7. Non-functional requirements — settled by the user, round 0

- **Removals spec:** *no performance requirements.* Pure removal.
- **Viv rebuild + builder preview:** **correctness is binding; interactive-over-SSH is a
  target, not a gate.** User's words: "Interactive over ssh would be nice but correctness
  is most important."

For the precedence table: the interactivity target sits **above** tier 8 (unstated
performance) and **below** correctness, data integrity and reference faithfulness. A
simplicity argument costing *measured* interactivity is a real conflict; one costing
*speculated* interactivity is not.

## 8. Conventions that bind the plans

- **`uv` is the sole runner.** Never bare `python`/`pip`.
- **`QT_QPA_PLATFORM=offscreen` is mandatory** — without it pytest aborts at ~79% with no
  summary.
- **Never `pytest -n auto`** — `nproc` reports node cores, not the allocation's, and
  manufactures timeout failures. Explicit `-n 4`.
- **Full `tests/unit` is a ~65-minute Slurm job**, not a local invocation. Batch script at
  `plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`.
- **Known-failing baseline test, not caused by this work:**
  `tests/unit/cli/test_cli_terminal_failures.py::test_concurrent_process_appends_do_not_lose_records`
  — spawns 8 processes on a 4-core allocation.
- **`ruff check --fix` always with explicit paths.**
- **Three CI gates** in `.github/workflows/gui-checks.yml`: `features-md-gate` (touching
  `gui/` requires modifying FEATURES.md, then `check_features_md.py` and `--strict`),
  `workflows-md-gate` (`check_workflows_md.py -v` enforces the WORKFLOWS.md ↔ capture-fn ↔
  tutorial-page round trip), `smoke-capture`.
- **Vendored reference sources under `docs/superpowers/specs/*/refs/` are read-only** —
  byte-identical to upstream, never linted or fixed. `[tool.ruff] extend-exclude` enforces
  it for ruff; the rule binds regardless.

## 9. Verified line references

Every `file:line` in the **removals** spec was re-checked and holds, including
`_layout.py:65/66/72/74/609/610/615/656`, `_callbacks.py:113/114/115/116`,
`browse/_ids.py:47-49` (+64 `BROWSE_TL_*`), `browse/_callbacks.py:39/44/46/50`,
`browse/_app.py:33/84`, capture fns at `:1156/1246/1750/1810/1900/1947/2813`,
WORKFLOWS.md rows `:46/47/51/52/54/55/56`, tutorial pages 10/11/15/16/17/19/20.

The **Viv** spec's `_tile_routes.py` line references are **stale** — the store branch
rewrote that file. Reviewers should verify against the current file, not the spec.

## 10. Curation — the constraint that binds two plans

Removals spec §5 reverses an earlier read-only decision. The curation radial is mounted on
**Colony** (`colony_view/_grid.py:47, :462`; `colony_view/_callbacks.py:43`), not only on
QC/Error, so unmounting QC and Error does **not** take it with them. Colony survives, so the
radial survives.

Both the removals plan (phase 5/6) and the Viv plan (phase 4) therefore require
`tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py` to pass **unmodified**, and
treat a diff in that file as stop-and-escalate. `_shared/_radial.py` and
`_shared/_triage_callbacks.py` drop from two consumer surfaces to one; they are **not**
unmounted.

## 11. What has NOT been verified

Reviewers should treat these as open, not as established:

1. **vizarr/Viv resolves our series list and label child unpatched** — Viv spec §2.1,
   marked UNVERIFIED. Plan phase 0 is a gate for it.
2. **1024² chunks pan acceptably over an SSH tunnel** — §5.2, accepted risk. Plan phase 0
   task 0.3 measures it.
3. **The colony virtualization cap** — §6.2 says measured during D1. Plan phase 4 task 4.1
   measures it and fails closed until then.
4. **The preview scratch retention cap** — preview spec §3. Plan phase 3 task 3.1 same.
5. **Whether one `_assets/viv/` can serve both sub-apps** — preview spec §7 OQ1.
6. **Whether `is_safe_path_component` rejects `..`** — both plans assert it does and both
   include a step to check. Nobody has run it.
