# GUI simplification — removals

**Date:** 2026-08-26
**Status:** Draft
**Scope:** Deleting the two Timeline surfaces, unmounting the Tune sub-app and the
Heatmap / Error / QC tabs, and the CI-gated ledger work all three require.

## Summary

The GUI ships **Plate + Colony only**. Three user-visible surfaces come off, in two
different ways, and one is deliberately left alone:

| Surface | Action | Rationale |
|---|---|---|
| Results Timeline tab | **delete** | Superseded; no successor planned for it in its current form |
| Browse Timeline mode | **delete** | Same engine, same fate |
| `gui/_shared/timeline/` | **delete** | Dies with both of its consumers |
| Tune sub-app (`/tune/`) | **unmount** | Being spliced into a simplified tool later; code retained |
| Heatmap / Error / QC tabs | **unmount** | Same — full overhaul planned; code retained |
| **Curation radial (Colony)** | **untouched** | See §5 — this is a correction to an earlier decision |

This spec is **pure removal**. It adds no capability, depends on no unlanded work, and
in particular does **not** depend on
[2026-08-18-ome-zarr-image-store](../2026-08-18-ome-zarr-image-store/design.md).
It is deliberately first of three cycles (§8) because it is the only one that can execute
today.

## Locked decisions

Recorded from the decision walkthrough
(`docs/superpowers/artifacts/2026-08-26-gui-ome-zarr-sync/gui-design-walkthrough.html`):

1. **Delete vs unmount is a real distinction.** Deleted code leaves the tree. Unmounted
   code stays importable and tested-where-still-tested, but is unreachable from the UI:
   no dispatcher mount, no nav leaf, no `dbc.Tab`, and **no callback registration**.
2. **Unmounting means not registering.** `suppress_callback_exceptions=True` is the
   project default (`results_viewer/_app.py:144`), so a callback registered against an
   absent layout would silently never fire rather than crash. Relying on that would leave
   dead registrations behind; the register call is removed too (§4).
3. **The curation radial stays on Colony.** Superseded decision, §5.
4. **Ledger work is part of this change, not follow-up.** Three CI gates enforce it (§6).

## 1. What gets deleted

### 1.1 Source

```text
src/phenotypic/gui/_shared/timeline/                  5 modules
src/phenotypic/gui/results_viewer/timeline_view/      6 modules
src/phenotypic/gui/results_viewer/_assets/timeline.js
src/phenotypic/gui/results_viewer/_assets/timeline.css
src/phenotypic/gui/browse/_assets/timeline.js
src/phenotypic/gui/browse/_assets/timeline.css
src/phenotypic/gui/browse/_thumb_routes.py
src/phenotypic/gui/browse/_timeline_records.py
src/phenotypic/gui/browse/_capture_time.py
src/phenotypic/gui/browse/_plate_pattern.py
```

The last four are **verified timeline-only**, not shared browse infrastructure:

- `_thumb_routes.py` is a thin adapter over `_shared/timeline`'s
  `register_thumbnail_route` (its own module docstring: "Mount the Browse Timeline
  thumbnail route"), and imports `ThumbUnavailable` from the package being deleted.
- `_timeline_records.py` is imported only by `browse/_callbacks.py:50`.
- `_capture_time.read_capture_time` has **exactly one** call site,
  `browse/_callbacks.py:1531`, inside `_capture_time_of` feeding `build_browse_records`.
  It is *not* what powers the front metadata row's EXIF chip — that resolves through
  `browse/_metadata.py` — so deleting it does not regress Single mode.
- `_plate_pattern.py` is imported by `browse/_callbacks.py:46` and by
  `_timeline_records.py:17`, both of which are timeline paths.

### 1.2 Tests

```text
tests/gui/_shared/timeline/                           8 files
tests/gui/results_viewer/timeline_view/               7 files
tests/gui/browse/test_timeline_callbacks_helpers.py
tests/gui/browse/test_timeline_records.py
tests/gui/browse/test_capture_time.py
tests/gui/browse/test_plate_pattern.py
tests/integration/gui/test_timeline_thumb_url.py
tests/e2e/gui/test_results_timeline.py
tests/e2e/gui/test_browse_timeline.py
tests/e2e/gui/test_browse_compare_strip.py
```

`test_browse_compare_strip.py` is included deliberately: despite its name it drives the
Browse **Timeline** surface (`#browse-tl-compare-btn`, `window.__phenotypicTimeline`),
per its own module docstring.

### 1.3 Edits, not deletions

| File | Change |
|---|---|
| `results_viewer/_layout.py:74` | drop the `timeline_view` import; drop the `TAB_TIMELINE_ID` `dbc.Tab` (the 6th entry in the `dbc.Tabs` at ~`:622`) |
| `results_viewer/_callbacks.py:83, :116` | drop the import and the `_timeline_callbacks.register_callbacks(...)` call |
| `results_viewer/_app.py:82` | drop the `timeline_view` import (thumb-route registration) |
| `results_viewer/_ids.py:521` | drop `TAB_TIMELINE_ID` |
| `browse/_ids.py` | drop the ~70 `BROWSE_TL_*` / view-mode ids |
| `browse/_layout.py` | drop the view-mode toggle, `BROWSE_TIMELINE_BODY`, and every TL control; `BROWSE_SINGLE_BODY` becomes the only body and stops being conditional |
| `browse/_callbacks.py` | drop the TL callbacks and the four deleted-module imports (`:39, :44, :46, :50`) |
| `browse/_cache.py` | drop the timeline-thumb cache entries |
| `browse/_app.py` | drop the `_thumb_routes` registration |

**Note the asymmetry.** Browse's Single mode must survive with its behaviour *unchanged*;
the view-mode toggle disappearing means Single is no longer one of two modes but the whole
tab. Every J/K nav, filmstrip, keep-position, and preparation affordance stays.

## 2. What gets unmounted — Tune

Remove the mount and the nav leaf; leave `gui/tune/` on disk.

| File | Change |
|---|---|
| `shell/_app.py:606, :655, :667` | drop the tune app construction, the `MOUNT_TUNE` dispatcher entry, and the chrome wrap |
| `shell/_layout.py:72, :130, :140` | drop `SHELL_TAB_TUNE` from the imports, `_TAB_HREFS`, the label map, and `NAV_MODEL` |

`MOUNT_TUNE` and `TITLE_TUNE` **stay** in `_config.py` (`:235`, `:844`). They are
declarations, not registrations, and the retained code still references them; removing
them would be churn against a sub-app we intend to bring back.

`gui/tune/` keeps its own unit tests. Its **e2e** tests, which drive a mount that no longer
exists, are marked skip with a reason naming this spec rather than deleted — they are the
acceptance suite for the eventual re-mount.

## 3. What gets unmounted — Heatmap / Error / QC

| File | Change |
|---|---|
| `results_viewer/_layout.py:65, :66, :72` | drop the three `build_*_tab_body` imports |
| `results_viewer/_layout.py:609, :610, :615` | drop the three body constructions |
| `results_viewer/_layout.py:~634-648` | drop the QC, Heatmap and Error `dbc.Tab` entries |
| `results_viewer/_callbacks.py:70, :76, :79` | drop the three imports |
| `results_viewer/_callbacks.py:113, :114, :115` | drop `register_heatmap_callbacks` / `register_qc_callbacks` / `register_error_callbacks` |

`_heatmap_tab/`, `_error_tab/` and `_qc_tab/` stay on disk with their unit tests. Their
e2e tests are skip-marked as for Tune.

After this, `dbc.Tabs` holds **two** tabs and `active_tab=ids.TAB_PLATE_ID`
(`_layout.py:656`) needs no change.

## 4. Callback registration is part of unmounting

`register_callbacks` (`results_viewer/_callbacks.py:90`) dispatches to each module in a
fixed order. Leaving a `register_*` call for an unmounted tab would register callbacks
whose `Input`s resolve to nothing. Under `suppress_callback_exceptions=True` that is
**silent** — no error, no firing — which is exactly the failure mode this spec should not
leave behind. `register_qc_callbacks` additionally opens `deliverables/qc/qc.duckdb`
state on registration, so leaving it wired has a side effect beyond dead code.

The post-change dispatch list is:

```python
_layout.register_callbacks(app, output_root)
_filter_panel.register_callbacks(app, output_root, filtered_state)
_filter_offcanvas.register_filter_offcanvas_callbacks(app)
_viewer_card.register_callbacks(app, output_root)
_colony_callbacks.register_callbacks(app, output_root, filtered_state)
_register_clientside_callbacks(app)
```

## 5. Curation stays — a superseded decision

An earlier draft of this change had the viewer go **read-only**, on the stated premise
that "unmounting QC and Error takes the curation radial with them."

**That premise was false.** The radial is mounted on the Colony tab as well:
`colony_view/_grid.py:47, :462` builds `build_radial_trigger` on every tile, and
`colony_view/_callbacks.py:43` builds the popover body, with wedge-mark, per-tile category
badge, bulk-mark, and bulk-bar-visibility callbacks bound to `CurationLabels`.

Colony survives this simplification, so the radial survives with it unless deliberately
torn out. That inverts the economics the decision was made under: read-only was chosen as
the cheaper option and is in fact the more expensive one, requiring active deletion across
two modules plus orphaning `_curation_labels.py` (966 lines) and `CFG_FILTERED_STATE`.

**Resolution: `colony_view/` is not touched by this spec.** The viewer keeps its only
write path, `deliverables/errors/<category>.parquet` keeps a live producer, the CLI's
`reemit_error_deliverables` round-trip keeps its counterpart, and the overhauled QC tool
inherits a working curation layer.

Consequence for §3: `_shared/_radial.py` and `_shared/_triage_callbacks.py` are **not**
unmounted — they drop from two consumer surfaces to one.

## 6. Ledger and documentation obligations

Three CI gates in the `gui-checks` workflow make this part of the change, not follow-up:

- **`features-md-gate`** rejects any PR touching `gui/` without modifying
  `FEATURES.md`.

  > **Amended 2026-08-26 — anchor on headings, never line numbers.** An earlier revision
  > gave eight line ranges. **Every one was wrong**, by ≈ +7 past line ~400, and one was
  > actively dangerous: it cited `:372-394` for the Results Timeline rows, but `:372-377`
  > are **Colony curation rows** (`Colony radial lazy-populate`, `Custom folder + ＋ Add
  > custom`, `Bulk "Mark N as ▾" (colony)`, `Pixel layer toggle`) and the
  > `### Results Timeline tab` heading is at `:379`. Following it deleted four curation
  > rows — so §6 instructed a violation of **§5**, the clause this spec spends a whole
  > section defending. Line numbers move as sections are removed; headings do not.

  Retire the rows under these **headings**:

  | Surface | Heading |
  |---|---|
  | Browse Timeline | the timeline block under `## Browse tab (source image viewer)` |
  | Results Timeline tab | `### Results Timeline tab` |
  | Timeline shared engine + Compare-strip cap | the two rows whose Element names `gui/_shared/timeline/` |
  | Tune co-pilot | `` ## Tune co-pilot (`/tune/`) `` |
  | QC tab | `## QC tab` |
  | QC Review | `## QC Review sub-view` |
  | Heatmap tab | `## Heatmap tab` |
  | Error analysis tab | `## Error analysis tab` |
  Unmounted surfaces are **marked as unmounted with a pointer to this spec**, not deleted
  outright — the ledger's job is to describe what a user can reach, and "exists but
  unreachable" is a state it should carry. The status is **`⏸ unmounted`**, and the
  legend at `FEATURES.md:9-16` gains a row for it: the file currently documents four
  statuses (`🔭 planned`, `🚧 in progress`, `✅ shipping`, `🧪 internal`) and this makes
  five. `🧪 internal` is **not** reused — it means "retained internal/legacy coverage; not
  user-facing", which describes test coverage of internals, whereas an unmounted tab is
  user-facing by design and merely unreachable.
- **`workflows-md-gate`** enforces the WORKFLOWS.md ↔ capture-script ↔ tutorial-page
  round-trip. Retire rows `browse_timeline` (`:54`), `results_timeline` (`:55`),
  `tune_copilot` (`:56`), `qc_curation_loop` (`:46`), `heatmap_exploration` (`:47`),
  `qc_review` (`:51`), `error_analysis` (`:52`).
- **`smoke-capture`** runs `scripts/capture_gui_tutorial_screenshots.py`. Remove
  `_capture_browse_timeline` (`:1156`), `_capture_results_timeline` (`:1246`),
  `_capture_tune_copilot` (`:2813`), `_capture_qc_curation_loop` (`:1750`),
  `_capture_qc_review` (`:1810`), `_capture_heatmap_exploration` (`:1900`),
  `_capture_error_analysis` (`:1947`) and their call sites (`:671, :680-683, :2417,
  :2454`), plus the tune/results-timeline harness blocks at `:2381-2464` and `:2661`.

Tutorial pages to remove, with `docs/source/tutorials/gui/index.md` updated:
`10_qc_curation_loop.md`, `11_heatmap_exploration.md`, `15_qc_review.md`,
`16_tune_copilot.md`, `17_error_analysis.md`, `19_browse_timeline.md`,
`20_results_timeline.md`. Their committed PNGs go with them.

`gui/CLAUDE.md` needs the sub-app table cut to five mounts and the Error-analysis-tab
section marked unmounted.

## 7. Testing

The removal is verified by absence, which is the weakest kind of test, so three positive
checks carry the weight:

1. **Both apps import and build a layout.** `create_app()` for the hub and the standalone
   results viewer, asserting the results `dbc.Tabs` has exactly the two expected
   `tab_id`s. This is what catches a missed import of a deleted module.
2. **No dangling references.** A test asserting that no module under `src/phenotypic/gui/`
   imports `_shared.timeline`, `timeline_view`, `_thumb_routes`, `_timeline_records`,
   `_capture_time`, or `_plate_pattern`, and that `NAV_MODEL` contains no
   `SHELL_TAB_TUNE`.
3. **Colony curation still works.** The existing colony curation tests must pass
   **unmodified** — that is the executable statement of §5. If a test in
   `tests/unit/gui/results_viewer/test_colony_callbacks_helpers.py` needs editing, §5 has
   been violated.

Browse Single mode and the whole `colony_view` suite are regression surfaces: they must
pass unchanged. Run per the **`run-phenotypic-test`** skill — the full `tests/unit` suite
is a ~65-minute Slurm job, not a local invocation, and `QT_QPA_PLATFORM=offscreen` is
mandatory.

## 8. Sequencing

This is cycle 1 of 3:

1. **Removals** (this spec) — unblocked now; depends on nothing.
2. **Viewer rebuild on Viv** — see
   [2026-08-26-viewer-viv-rebuild](../2026-08-26-viewer-viv-rebuild/design.md);
   blocked on the OME-Zarr backend landing.
3. **Builder preview** — blocked on 2.

## 9. Non-goals

- No change to `colony_view/` (§5).
- No change to Browse Single mode's behaviour.
- No deletion of `gui/tune/`, `_heatmap_tab/`, `_error_tab/`, `_qc_tab/`.
- No change to the pixel path — plate tiles still come from `_dzi_tiler`. That is cycle 2.
- No `_config.py` constant removal.

### 9.1 Non-functional requirements

**None.** This is a pure removal: it adds no capability and no code path. Deleting work
makes app construction marginally cheaper, but no threshold is claimed and none should be
defended. Recorded explicitly so the reviewer panel's precedence table has an anchor rather
than an absence — a performance argument raised against this spec has nothing to appeal to.

*Settled by the user, 2026-08-26.*

## 10. Risks

| Risk | Mitigation |
|---|---|
| A browse helper is less timeline-only than it looks | §1.1 records the call-site evidence for each; `read_capture_time` was the one that needed checking and has a single call site |
| `_error_tab/` is deleted rather than retained | It is a **CLI dependency**: `_cli/_cli_error_outputs.py:81` imports `capture_error_source_fingerprints`, `compute_all_category_analysis` and `publish_error_analysis` from `results_viewer/_error_tab/_publication.py` on every finalize. Deleting it breaks CLI finalization, and **no GUI test catches it** |
| An unmounted tab's callbacks left registered fail silently | §4 removes the register calls; test 2 in §7 has no coverage for this, so it is a review-checklist item |
| e2e skip-marks rot into permanently dead tests | Each skip reason names this spec, so they are greppable when the surface returns |
| Deleting tutorial PNGs loses history | They are in git; the pages are recoverable with the surfaces |
