# Smart QC — design alternatives log

**Status:** brainstorming in progress (Component D / GUI).
**Companion mockups (persisted):** `.superpowers/brainstorm/60986-1780050693/content/`
(e.g. `qc-review-layout.html`).

This log records the alternatives considered during brainstorming so the
chosen path in the eventual design spec carries its rejected siblings with
it. The authoritative design lives in
`2026-05-29-smart-qc-*-design.md` (written at the end of brainstorming).

---

## D1 — QC review screen interaction model

**Chosen: A — Master–detail.** Persistent worst-first worklist sidebar on
the left (group key + metric + status badge, reviewed rows dimmed/checked),
the selected group's tile gallery + per-tile curation + actions on the
right. Top bar carries the module picker and the overall summary strip.
Picked for at-a-glance progress and free navigation around the queue.

**Alternates (kept for later, not discarded):**

- **B — Filmstrip focus.** A thin horizontal queue strip (worst→best
  chips, reviewed dimmed, current outlined) above a full-width gallery.
  Trade-off: gives the group's tiles the most screen room and keeps the
  queue glanceable, but the queue carries less per-group detail than a
  sidebar (no inline metric/key text). Revisit if plates have many
  colonies per group and tile real estate becomes the bottleneck.
- **C — Queue-driven wizard.** One group front-and-center, large
  "N of M · K flagged" progress, prev / mark-reviewed-&-next / skip; the
  full worklist hidden in a collapsible drawer. Trade-off: best for fast
  heads-down triage with minimal chrome, but hides progress/structure and
  makes non-linear navigation a second-class action. Revisit if users ask
  for a "just walk me through them" focus mode — could ship later as a
  toggle on top of A.

A mode toggle (A ⇄ C) is a plausible future enhancement: same data, same
recompute, different chrome.

---

## Earlier decision forks (Components A–C)

Recorded for the spec's "alternatives considered" section.

- **QC result contract.** Chosen: raw headline **metric** per check +
  `_HIGHER_IS_BAD` class flag setting threshold/sort direction; the
  normalized `severity` abstraction is dropped. Rejected: keeping
  `severity` as the universal sort key with the flag as mere display
  metadata; and a hybrid carrying both numbers.
- **Persisted QC artifact.** Chosen: compact long-format `qc/` dir
  (`qc_summary.parquet`, `qc_members.parquet`, `qc_config.json`), leaving
  `measurements.parquet` untouched so the after-each-group recompute is
  cheap. Rejected: broadcasting QC columns onto `measurements.parquet`;
  and a hybrid that also mirrors flag columns there.
- **QC config home.** Chosen: a new `qc` section inside `pipeline.json`
  (sibling to `operations`/`post`), read by the CLI at finalize. Rejected:
  a neutral top-level `qc/qc_config.json`; and keeping the viewer-owned
  `.viewer_cache/qc_recipe.json`. Consequence to design around:
  `pipeline.json` is CLI-written with a staleness guard, so the QC tab
  must do a scoped read-modify-write of only the `qc` array + a one-time
  migration of any existing sidecar.
- **Tile-imaging extraction scope.** Chosen: extract a real
  `gui/_shared/tiles.py` (crop_overlay, overlay cache, safe-path, generic
  cell/grid builder, crop-route factory) and refactor `colony_view` to
  consume it — one implementation, two consumers. Rejected: extracting
  only the pure pieces and leaving colony_view's grid duplicated; and
  importing colony_view internals directly from the QC UI.

---

## Component D decisions (resolved)

- **Surface placement.** Chosen: one **QC tab** with a Configure | Review
  segmented toggle. Rejected: a separate top-level "QC Review" tab; and a
  launch-from-card full-screen overlay.
- **Recompute granularity.** Chosen: recompute **per group** (on finishing
  a group, if changes were made), not on every tile click — matches the
  "recalculate after each group" requirement.
- **Re-sort behavior.** Chosen: **freeze queue order, update in place** +
  a manual "↻ Re-sort" button. Rejected: auto re-sort + auto-advance; and
  a resolved-section hybrid (kept as a possible later enhancement).
- **Review-progress tracking.** Chosen: `qc/review_state.json`,
  **per-module** (keyed by check `instance_id`), reset by a CLI
  recompile/remeasure ("a different run"), preserved by the GUI's
  in-session recompute. Rejected: viewer-cache placement; and a single
  global reviewed set across all modules.
- **Curation mechanism.** Reuse the colony-view `FilteredMeasurements`
  removal store so edits are consistent across the viewer.

## Still open (deferred to implementation planning)

- Home for `run_qc` / migrated `QcRecipe` (`_cli/_cli_qc.py` vs a neutral
  `phenotypic/qc/`).
- ICC KEPT in v1 with **subject=Metadata_Time, rater=Metadata_Replicate**
  (loud missing-axis via unmatched_groups). Test data:
  src/phenotypic/data/meas/all_meas.csv. **Documented refinement (deferred):**
  subject=Metadata_StrainID snapshot and/or per-timepoint binning (sounder).
  [Heavy decision churn: time-as-subject → cut for a missing time column →
  reinstated → snapshot-StrainID → finally KEPT on the green time-based default
  per the user's "keep subject=Metadata_Time (green now)" call. The sounder
  StrainID/per-timepoint model is the v2 refinement.]
- Default thresholds, tuned against real plates.
