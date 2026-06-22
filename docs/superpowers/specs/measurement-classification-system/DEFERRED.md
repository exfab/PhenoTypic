# Measurement Classification — Deferred Work

Follow-ups intentionally left out of the initial feature (PR #163, branch
`metric-classification-system`). The shipped scope is the conceptual framework +
the `schema/` integration (tier/kind classification, coverage gate, rst Use
column, docs page). Items below were scoped, deliberately excluded, and recorded
here so they aren't lost.

See the design doc (`2026-06-18-measurement-classification-system-design.md`,
§9 scope / §10 integration) for the full rationale.

---

## 1. GUI tier badging (largest deferral)

Surface the tier/kind classification in the results-viewer GUI so users see, at
the point of choosing a measurement, whether a column is a reportable phenotype
or a discriminative feature.

**Prerequisite — the header→classification bridge (build this first).** The GUI
works in column-header strings read from the master/mirror parquet; it has no
`MeasurementInfo` objects. A reusable lookup is needed:

```python
classify_header("Shape_Area")    # -> (kind="primary", tier=1, use_label="Direct phenotype (Tier 1)")
classify_header("Metadata_Time") # -> None  (non-schema column → no badge)
```

Mirror the existing `util/_measurement_outputs._measurement_descriptions()`
(`header → desc`, scans `schema.__all__`). It is reusable beyond the GUI (CLI
`--help`, README generator, exports) and unit-testable without Dash. **Lean
toward a public `phenotypic.schema` helper** since it is schema knowledge.

**Surfaces, ranked by value:**

| Surface | File | Why |
|---|---|---|
| Heatmap measurement picker ("color" dropdown) | `gui/results_viewer/_heatmap_tab/_layout.py` (`color_options`) | Where the user chooses what to visualize — highest value |
| Filter-panel column dropdown | `gui/results_viewer/_filter_panel.py` (`_column_options`) | Tier shown when filtering; pairs with a future "filter by tier" |
| Colony view / measurement tooltips | `gui/results_viewer/colony_view/`, qc/error tabs | Per-value tier chip |
| Trust-contract legend / help popover | new, shared | Explains what each tier licenses (mirrors the docs page) |

**Visual treatment + Dash constraint:** badge = small colored chip + tooltip with
the one-line trust-contract text; tier colors belong in `gui/_design.py` (palette
in `DESIGN.md`) — 3 tier hues + a neutral for identity/quality/derived-without-tier.
`dcc.Dropdown` option labels are plain strings by default, so rich chips inside
options need a custom/clientside renderer. Scope ladder:

- **Minimal** — text-prefixed tier label in the heatmap picker (`"Shape_Area · Tier 1"`) + a static legend. No new component.
- **Medium** — colored chips + tooltips via a small custom/clientside option renderer; filter dropdown too; legend popover.
- **Full** — group the picker by tier (Dash has no native optgroups — needs a custom component or disabled-header separators; reuse `util.split_measurements` / `_producer_column_groups` re-keyed by tier), a "show only Tier 1" toggle, and a tier column in measurement tables.

**CI gate cost (why this is its own effort, per `gui/CLAUDE.md`):**

- `FEATURES.md` — every new affordance (badge, legend, toggle) needs a row with a `Test ref`; the `features-md-gate` blocks GUI PRs that don't update it.
- `WORKFLOWS.md` — only for new end-to-end flows. A badge is an affordance (skippable); a "filter/group by tier" feature crosses into it (needs a `_capture_<id>` + tutorial page).
- Screenshots — any visible chrome change requires re-running `scripts/capture_gui_tutorial_screenshots.py` and committing the *full* regenerated set.
- Tests — Dash callback/component tests for the `Test ref`; Playwright e2e is `ci_flaky`-aware.

**Recommended sequencing:** (1) bridge helper + unit tests (gate-free, reusable);
(2) minimal heatmap-picker badge + legend; (3) defer grouping / tier-filter /
table columns.

**Open decisions for a brainstorm:** bridge home (schema vs util); derived columns
show their resolved tier (consistent with the rst badge — yes); unclassified/
metadata columns get a neutral "Identity" chip vs no badge; flat-label-prefix vs
custom grouped dropdown; exact tier→color mapping in `_design.py`.

This warrants its own brainstorm → spec → plan rather than a bolt-on.

---

## 2. Reconcile the `Shape_Area` / `Size_Area` duplication

`SHAPE.AREA` (`Shape_Area`, regionprops `current_props.area` in
`measure/_measure_shape.py`) and `SIZE.AREA` (`Size_Area`, `_calculate_sum` in
`measure/_measure_size.py`) are near-duplicate pixel-area measurements computed by
two operations. The classification feature left this as-is — both are simply
classified Tier 1. Reconciling (deprecating one, or clarifying their distinct
roles) is a separate cleanup with output-column and downstream-consumer impact.

---

## 3. Full Path 2 de-straddle (column renames) — explicitly NOT taken

The alternative integration (encode tier purely in the type hierarchy) would
require every measurement enum to be tier-uniform, forcing:

- moving size-magnitude members out of `SHAPE` into `SIZE` (`Shape_Area` →
  `Size_Area`, etc.), and
- relocating growth-fit knobs (`lambda`/`beta`/`Kmax`) into `MODEL_METRICS`.

Rejected for the initial feature: `Shape_Area` alone appears as a literal in ~38
files and collides with the existing `Size_Area`; this is a ~1-week migration with
golden churn vs. the chosen Path 3 (~1–2 days). Documented here only as the
considered-and-declined alternative; revisit only if a structural/type-level
grouping becomes worth the migration.

---

## 4. Minor review nits (deferred, low value)

From the per-task and whole-branch reviews, accepted as Minor:

- `test_tier2_primary_enums` / `test_tier3_primary_enums` assert `resolved_tier`
  but not `resolved_kind` — effectively moot because the coverage gate asserts
  kind for every member. Add kind assertions for parity only if convenient.
- `_USE_LABELS` is typed `dict[tuple[int | None, str], str]`; no `None`-tier key
  is exercised — tighten to `tuple[int, str]` if no `None`-tier entry is added.
- Tests were appended to `test_classification.py` in reverse task order
  (cosmetic).

---

## 5. Extend classification beyond the schema (opportunistic)

The `classify_header` bridge (item 1) also unlocks: a tier/use column in the CLI
`deliverables/README.md` generator, and tier annotations in CLI `--help` /
measurement listings. Low priority; bundle with whoever needs the bridge first.

---

## Out of observed scope (not owned by this feature)

The docs build surfaced pre-existing docutils ERRORs in unrelated docstrings
(`analysis/_log_growth_model.py` LogGrowthModel indentation,
`measure/_measure_color.py` MeasureColor indentation,
`refine/_extract_colony_core.py` undefined substitution) and autodoc import
warnings (`Image` color properties, `GridImage.show_overlay`). These pre-date and
are unrelated to the classification feature. The stale-autosummary-stub build
abort was fixed separately (PR #164). Listed here only so a future docs pass knows
they exist.
