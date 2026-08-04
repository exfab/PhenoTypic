# 2026-05-13 — PR #92 merge + GUI debug session

Working notes from merging PR #92 (DAG builder redesign, Phases 1–8)
into `builder-gui-redesign` and regenerating tutorial screenshots to
validate the result. The session ended mid-debug; this doc captures
state so a fresh cloud session can pick up.

## Branch state

- Local branch: `builder-gui-redesign`, 49 commits ahead of
  `origin/builder-gui-redesign`.
- Merge commit: `5388df75` —
  `Merge remote-tracking branch 'origin/claude/builder-dag-implementation-plan-XrxUm'`.
- Working tree: dirty (uncommitted GUI fixes, partial regenerated PNGs, capture-script
  tweak). See "Uncommitted changes" below.
- Remote PR: `exfab/PhenoTypic#92`, source branch
  `claude/builder-dag-implementation-plan-XrxUm`. The merge commit is local-only;
  nothing pushed yet.

## Merge resolution

Three conflicts, all "both branches added rows/sections in the same place":

| File                                          | Resolution                                                                                                                                                                                  |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `docs/source/tutorials/gui/index.md`          | Kept HEAD's QC (10) + heatmap (11). Renumbered PR's DAG tutorials 10/11/12 → **12/13/14**.                                                                                                  |
| `scripts/capture_gui_tutorial_screenshots.py` | Kept both branches' capture functions. Restored `page.close()` for `_capture_heatmap_exploration` that the conflict marker swallowed. Updated docstring path mentions to the new numbering. |
| `src/phenotypic/gui/WORKFLOWS.md`             | Kept both rows; updated Test-ref column for DAG rows to new file numbers.                                                                                                                   |

Also did `git mv` on the three PR-added tutorial pages so the filenames match the new
numbering. Cross-references inside each renamed page were updated.
`scripts/check_workflows_md.py` round-trip passes (14 ↔ 14 ↔ 14).

## Capture script validation outcome

User asked to re-run `uv run python scripts/capture_gui_tutorial_screenshots.py --force`
before pushing. **The capture script never completed** — it surfaced multiple
PR-introduced GUI bugs. Each failure required a code patch + restart.

### PR bugs found & patched on this branch (UNCOMMITTED)

All edits are in the working tree as of this writing.

1. **`dbc.Button` rejects `data-palette-class` kwarg** —
   `src/phenotypic/gui/builder/_layout.py:200-240` and `:284-326`. PR passes
   `draggable="true"` + `data-palette-class=...` as kwargs to `dbc.Button` (claims dbc
   forwards `**kwargs`, but dbc 2.0.4 enforces a whitelist).
   **Fix applied:** wrapped each button in `html.Div` that carries the data attribute.
   The clientside `palette_dnd.js` uses `closest("[data-palette-class]")` so the
   ancestor lookup finds the wrapper.

2. **`build_canvas` initial render calls legacy `build_canvas_elements`** —
   `src/phenotypic/gui/builder/_layout.py:1917-1921`. The DAG schema's
   `_DagBuilderScope` has `.blocks` not `.nodes`, so initial paint crashed with
   `AttributeError`. The dispatch callback already branches on `is_dag` correctly; only
   the first paint was wrong.
   **Fix applied:** call `build_canvas_elements_dag(scope, selected_block_id=...)`.

3. **`palette-add` click dispatches legacy `add_node`** —
   `src/phenotypic/gui/builder/_callbacks.py:2211-2240`. The DAG dispatcher only routes
   `block_create` through the drag-drop store (`STORE_PALETTE_DROP`); click handler
   still wrote `scope["nodes"].append(...)` which raises `KeyError: 'nodes'` on DAG
   state.
   **Fix applied:** dispatch `block_create` with `container_block_id=None` for the
   keyboard-fallback click path.

4. **`block_create` doesn't auto-wire new blocks** — `block_create` is by design
   wireless (user draws wires manually for drag-drop). Click-based / programmatic adds
   therefore produce stranded blocks that fail validation, blocking `Run preview`.
   **Fix applied (in `palette-add` handler):** after dispatching `block_create`,
   identify the scope's tail (last block in `scope.blocks` with no outgoing image edge)
   and dispatch `edge_create` from tail → new block with `target_port="in"`,
   `edge_kind="image"`. Drag-drop path remains unchanged.

5. **Initial cytoscape layout is `preset` but DAG elements are positionless** — relies
   on `cytoscape-dagre` extension that fails to register against the
   dash-cytoscape-bundled cytoscape (no `window.cytoscape` global). Without dagre,
   blocks pile at (0, 0).
   **Fix applied:** changed initial layout to `"breadthfirst"` (cytoscape core layout,
   no extension needed). Dagre still tries to run on first paint via `viewport_ops.js`;
   breadthfirst is the always-present fallback.

### Test-script fix applied (also uncommitted)

- `scripts/capture_gui_tutorial_screenshots.py:505-517` — `_capture_pick_points` used
  `window.cy || window._cy_instances[0]` to grab the cytoscape handle. The DAG redesign
  moved the handle to `window.phenoGetCy()`. Updated to use the new accessor.

## What still fails

The script now reaches `_capture_pick_points` step 4 (open the point-picker modal)
before timing out:

```
TimeoutError: Page.wait_for_selector: Timeout 15000ms exceeded.
Call log: waiting for locator("[data-testid=\"point-picker-osd-canvas\"]") to be visible
```

Visible in the regenerated `03_param_form.png`:

- ✅ Canvas shows 4 connected blocks (
  `InputImage → BlurGauss → OtsuDetector → ManualRefine`), breadthfirst layout
  positions them correctly.
- ✅ "0 ISSUES" badge — DAG validator green.
- ✅ Inspector renders the ManualRefine param form.
- ❌ "(no image loaded)" near the top — the test never clicks "Use synthetic plate", so
  preview runs without an image, no predecessor cache is produced, and the OSD canvas in
  the picker modal has nothing to render.
- ❌ Residual `KeyError: 'nodes'` toast labelled "Update failed" — some other dispatch
  path is still hitting `scope["nodes"]` on DAG state. Probably `tapNodeData` or another
  callback that wasn't updated for the DAG schema. Non-blocking for the current capture
  step but worth tracking.
- ⚠️  "Layout extension missing" banner — dagre extension didn't register. Cosmetic for
  breadthfirst fallback but the banner is in every screenshot.

The script has not yet reached:

- `_capture_pick_points` steps 4-8 (picker modal flow)
- `_capture_analysis`
- `_capture_aux_ports` (legacy)
- `_capture_qc_curation_loop` (HEAD)
- `_capture_heatmap_exploration` (HEAD)
- `_capture_aux_wire_in_dag` (PR)
- `_capture_wire_pipeline_as_aux` (PR)
- `_capture_fix_validation_issues` (PR)

The DAG-redesign tutorials (12/13/14) are the actual validation target — none of them
have been exercised yet.

## Successfully regenerated PNGs (uncommitted in working tree)

```
docs/source/_static/gui_images/build_pipeline/01_builder_empty.png
docs/source/_static/gui_images/file_explorer/01_sidebar_collapsed.png
docs/source/_static/gui_images/file_explorer/02_sidebar_expanded.png
docs/source/_static/gui_images/file_explorer/03_capability_badges.png
docs/source/_static/gui_images/file_explorer/04_sidebar_hidden.png
docs/source/_static/gui_images/pick_points/01_palette_with_badge.png
docs/source/_static/gui_images/pick_points/02_pipeline_with_selector.png
docs/source/_static/gui_images/pick_points/03_param_form.png
docs/source/_static/gui_images/run_local/01_run_console_form.png
docs/source/_static/gui_images/run_local/02_input_picker_modal.png
docs/source/_static/gui_images/run_local/03_recent_runs_panel.png
docs/source/_static/gui_images/run_slurm/01_slurm_mode.png
docs/source/_static/gui_images/setup/01_landing_page.png
docs/source/_static/gui_images/view_results/01_viewer_empty.png
```

14 PNGs total. The pick_points/03 includes the residual `KeyError: 'nodes'` toast — if
we commit it, expect a follow-up to re-shoot once that's fixed.

## Uncommitted changes (full inventory)

Outside `docs/source/_static/gui_images/`:

```
M scripts/capture_gui_tutorial_screenshots.py     # phenoGetCy fix (mine)
M src/phenotypic/gui/builder/_callbacks.py        # palette-add → block_create + auto-wire (mine)
M src/phenotypic/gui/builder/_layout.py           # dbc.Button wrappers + build_canvas DAG renderer + breadthfirst layout (mine)
M src/phenotypic/correction/_bayesshrink_corrector.py   # NOT MINE — pre-existing
M src/phenotypic/correction/_visushrink_corrector.py    # NOT MINE — pre-existing
M src/phenotypic/enhance/_bayesshrink_enhancer.py       # NOT MINE — pre-existing
M src/phenotypic/enhance/_bilateral_denoise.py          # NOT MINE — pre-existing
M src/phenotypic/enhance/_bm3d_denoiser.py              # NOT MINE — pre-existing
M src/phenotypic/enhance/_non_local_means.py            # NOT MINE — pre-existing
M src/phenotypic/enhance/_visushrink_enhancer.py        # NOT MINE — pre-existing
```

The `correction/` + `enhance/` modifications were sitting in the worktree before this
session (they extend GAT-mixin parameters in existing operations — see commit `d782349f`
for the pattern). They are **not part of the PR #92 merge and not part of my GUI fixes
** — leave them alone or stash before committing.

## Open task list (from TaskCreate)

1. **Patch `_capture_pick_points` to load synthetic plate** (in-progress) — add a click
   on the "Use synthetic plate" button before `Run preview` so the OSD canvas has image
   data.
2. **Resolve `KeyError: 'nodes'` toast** — find the remaining dispatch path still using
   `scope["nodes"]` on DAG state. Search `_callbacks.py` for `scope["nodes"]` reads not
   gated by an `is_dag` branch; the dispatch table at `_dispatch_state_update` (line ~
   1000+) is the most likely culprit (all the legacy kinds — `add_node`, `delete_node`,
   `reorder`, `edit_param`, `edit_label`, `drill_in`, etc. — still write to
   `scope["nodes"]` regardless of state shape). Either route DAG triggers to
   DAG-equivalent kinds, or short-circuit legacy kinds when state is DAG-shape.
3. **Run capture through aux_ports + DAG tutorials (10-14)** — the actual validation
   target. After (1) and (2), continue patching until the script completes or the
   captures look right.
4. **Commit GUI fixes + regenerated PNGs** — once captures complete (or we decide to
   stop), split into commits:
    - Commit A: GUI fixes (`_layout.py`, `_callbacks.py`) — title something like
      `fix(gui/builder): repair DAG dispatch + initial render after PR #92 merge`
    - Commit B: capture-script + regenerated PNGs
    - Show diff to user before pushing.

## Plan for cloud session

Suggested order:

1. **Re-confirm state.** `git status` to verify the uncommitted edits are still there.
   Run
   `uv run python -m phenotypic.gui --root docs/source/_static/gui_images/_dataset --port 8050`
   for ~10s to confirm the GUI still boots with the patches applied.
2. **Fix `KeyError: 'nodes'`.** Read `_callbacks.py:_dispatch_state_update` (the
   function starting around line 1000) — it has many `if kind == "add_node"` etc.
   branches that hardcode `scope["nodes"]`. For each one, decide: short-circuit when the
   state is DAG-shaped (return `out` unchanged), or route to the DAG-equivalent dispatch
   kind (`block_create`, `block_delete`, `edge_create`, `edge_delete`, `edit_param` —
   which the DAG renamed slightly). The `tapNodeData` callback is one likely caller; the
   inspector edit fields are another.
3. **Patch `_capture_pick_points`.** Add a
   `page.click("button:has-text('Use synthetic plate')")` (or whatever the selector
   resolves to — verify in the DOM) before line 524's `Run preview` click.
4. **Re-run the capture (no `--force`).** Watch for the next failure. Likely candidates:
    - `_capture_pick_points` 04-08 (picker modal) — may need further selector tweaks for
      DAG-redesigned modal.
    - `_capture_aux_ports` — entirely about the LEGACY popover-based aux flow. The PR
      replaced the popover with DAG wires. This test will not work against post-redesign
      chrome — needs either a full rewrite or to be retired in favor of the new
      `12_aux_wire_in_dag` capture.
    - `_capture_qc_curation_loop` + `_capture_heatmap_exploration` — HEAD-side, should
      work since they hit the viewer not the builder.
    - `_capture_aux_wire_in_dag` / `wire-pipeline-as-aux` / `fix-validation-issues` —
      these are the actual DAG tutorials. They use `cy.add(...)` + `phenoGetCy()`
      directly so they may "just work" once the underlying GUI is stable.
5. **Consider retiring `_capture_aux_ports`.** The legacy aux-ports flow no longer
   exists in the post-redesign builder. The PR's `12_aux_wire_in_dag` tutorial is the
   spiritual replacement. If the legacy capture can't reasonably be made to work, mark
   its row in `WORKFLOWS.md` as `🪦 retired` (or delete it) and update the tutorial
   index. Check whether `scripts/check_workflows_md.py` understands a "retired" status;
   if not, the row removal is the simpler path.
6. **Dagre extension banner.** Optional cleanup. The banner appears in every screenshot.
   Either get dagre to register (probably needs a small bootstrap script that exposes
   `window.cytoscape = cy.constructor` early so the vendored `cytoscape-dagre.min.js`
   finds a global to register against) or hide the banner when the breadthfirst fallback
   is sufficient.
7. **Commit + push.** Once the capture either completes or we accept partial coverage
   with documented gaps.

## Notes / gotchas

- `--force` regenerates the synthetic plate dataset (~+2 min). Once the dataset exists,
  omit `--force` for fast iterations.
- The dispatch chain has a `_dispatch_state_update(state, kind, payload)` helper. Always
  pass the JSON-shaped state dict (the deepcopy is internal). `block_create` payload
  requires `class_name`; `container_block_id=None` lands the block in the current scope
  root.
- `palette_dnd.js` writes `block_create` payloads to `STORE_PALETTE_DROP` on drag-drop.
  The keyboard-fallback path lives inside `fan_in_state_mutation`'s `palette-add`
  branch (now also dispatching `block_create` — see fix #3).
- Pyright surfaces many `Import "dash" could not be resolved` / `unknown import symbol`
  diagnostics — these are venv-path issues from the editor's Pyright not seeing `.venv`.
  Ignore unless they also fail in `uv run mypy`.
- `git diff scripts/capture_gui_tutorial_screenshots.py` will show my `phenoGetCy` patch
  alongside the merge-time changes — read the diff carefully when reviewing.
- The regenerated `pick_points/03_param_form.png` includes the `KeyError: 'nodes'` error
  toast in the top-right. If you ship this PNG without fixing (2) first, the tutorial
  will document a broken state. Either fix (2) first or re-shoot after.

## References

- Merge commit: `5388df75`
- PR #92 head: `cfa3d337` (
  `Phase 8 closure: wire DAG renderer into the live dispatch path`)
- PR description: see `gh pr view 92` — claims keyboard fallback exists ("the existing
  keyboard-fallback callback ... continues to resolve for users without drag-and-drop")
  but the click path wasn't actually wired to `block_create` until fix #3 above.
- Spec: `docs/superpowers/specs/2026-05-12-builder-dag-redesign-design.md` (§4.7 dagre,
  §4.8 keyboard fallback, §5.5 dispatch kinds, §5.6 wire drawing).
