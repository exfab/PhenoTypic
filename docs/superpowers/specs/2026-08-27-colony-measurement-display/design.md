# Colony cards display a chosen measurement

**Date:** 2026-08-27
**Status:** Draft
**Scope:** The Colony tab's tile cards, a new per-image measurement route, and the
embedded-table reader it needs.

## Summary

The Colony grid shows crops and nothing else. Every colony already carries ~130 measured
values in its image's store, and none of them are visible without leaving the viewer for the
exported CSV.

This adds a **column picker** to the Colony tab. Choosing a column
(`Shape_Area`, `Intensity_MeanIntensity`, …) makes each colony card **show that colony's
value as text** and **tint the card** on a continuous scale across the values in view — so a
plate's spatial structure (edge effects, contamination gradients, a dead quadrant) is legible
at a glance, and the exact number is right there on the card.

## Objective

A user picks a measurement column and reads both the value and its distribution off the
Colony grid, without exporting anything.

## Non-goals

- **No Plate-surface overlay.** Colouring objmap *pixels* by measurement is a different
  feature with a different cost; this one reuses the existing card chrome.
- **No new measurement.** Only columns the CLI already wrote are displayed.
- **No editing.** Display only; curation writes are untouched.
- **No cross-image aggregation.** Values come from each image's own embedded table.
- **No change to the `/zarr/` byte route's readable roots.** See §2.

### Non-functional requirements

**Correctness is binding; responsiveness is a target.** Inherited from
[viewer-viv-rebuild §9.1](../2026-08-26-viewer-viv-rebuild/design.md). Specific to this
feature: a value shown against the wrong colony is worse than no value at all, so the
`Object_Label` join is the property to protect.

## 1. Where the column list comes from — the store, not the parquet

The store's root already enumerates its own columns:

```text
attributes.phenotypic.tables.measurements.measurement_columns
  -> ["Metadata_Dataset", "Shape_Area", ..., "Grid_ColMajorIdx"]     (~130)
attributes.phenotypic.tables.measurements.target
  -> {"column": "Object_Label", "path": "rgb/labels/objmap"}
```

So the picker is populated **without opening the parquet**, and `target.column` names the
join key rather than assuming it. Verified against a real migrated run
(`ucr_029_e_d_Maresca/.../2026-08-11-migration-test`).

**A store may carry no `tables` descriptor at all** — a `--mode process` run never measures,
and older or migrated stores may have none. That is a **normal** state, not a pending one:
the picker is simply empty and the cards render as they do today. It must **never** be
reported as "measurement pending" — see the retraction note in
`results_viewer/_store_source.py`.

## 2. Values reach the browser as JSON, one column at a time

```text
GET /measurements/<dataset>/<stem>?column=<name>
 -> {"column": "Shape_Area", "values": {"1": 412.0, "2": 388.5, ...},
     "min": 12.0, "max": 512.0, "n": 96}
```

**Why not serve `tables/` on the existing byte route.** That route is unauthenticated and the
documented Open OnDemand recipe is `--host 0.0.0.0` on a shared cluster
(`gui_hub.md:116, :124`). `readable_roots_for` deliberately excludes `tables/`, and a
round-2 security review rated the first version's bypass **Major**. Opening it would reverse
that decision, ship ~130 columns to display one, and put a parquet reader in the browser
bundle.

The JSON route keeps the narrowing intact and is **strictly smaller**: the real table is
71 KB for ~96 colonies × 130 columns; one column is ~2 KB.

**`column` is validated against the store's own `measurement_columns`.** A name not in that
list is a **400**, not a parquet read — so the parameter cannot be used to probe the
filesystem or to fish for columns a store does not have.

Error contract, matching `/zarr/`: absent store or no `tables` descriptor → **404**;
unreadable store → **422**; unknown column → **400**.

## 3. The card shows the value and is tinted by it

`build_tile_cell` (`_shared/tiles.py:1087`) already receives `label` — which **is**
`Object_Label`, the join key — alongside `dataset` and `image_file`. So the card needs one
new optional input and no new identity plumbing.

- **Text.** The value renders in the card's existing chrome, JetBrains Mono per
  `DESIGN.md`, formatted to a fixed precision so a grid of numbers stays scannable.
- **Tint.** The card's frame/background takes a colour from a continuous scale over the
  **values currently in view**, not the column's global range — the grid is already filtered,
  and rescaling to what is shown is what makes a gradient visible.
- **A colony with no row in the table** (label absent) renders untinted with no text. It is
  not an error: post-ops can remove objects after measurement.
- **A legend** names the column and its min/max, or the tint means nothing.

**Curation chrome is unchanged.** The radial trigger, the selection checkbox and the
remove/restore button keep their positions and behaviour; the tint is a background, not a
replacement for the selected-state outline. Spec §5 of the removals design protects that
layer and this feature does not touch it.

## 4. The reader `sdk_` is missing

`sdk_` has `write_embedded_measurement_table` and `replace_embedded_measurement_table`
(`_measurement_tables.py:86, :242`) but **no reader**. This adds one:

```python
read_embedded_measurement_column(store: Path, column: str) -> dict[int, float]
```

reading `ngff_.MEASUREMENT_TABLE_RELATIVE_PATH` and projecting **one** column plus the
target column. Reading one column of a parquet does not read the file's other 129 — that is
the format's whole point, and it is why the route can afford to be per-request.

## 5. Testing

- **The join is correct.** Against a **real migrated store**, assert a known
  `Object_Label` maps to that colony's value in the source parquet. A value on the wrong
  card is the failure this feature must not have.
- **Column validation.** A column absent from `measurement_columns` yields 400 and **does
  not** open the parquet.
- **No `tables` descriptor** → 404 from the route, and an empty picker in the layout — **not**
  a "pending" caption.
- **Missing label** renders untinted with no text, and does not raise.
- **Scale is over values in view**, not the global range — pin it, because the alternative
  passes every smoke test and produces a uniformly-coloured grid on filtered data.
- **Curation regression.** `test_colony_callbacks_helpers.py` passes unmodified, and
  `test_grid.py::test_build_grid_tiles_carry_radial_trigger_not_old_remove_button` still
  passes — a tinted card must still carry its radial.

Per **`run-phenotypic-test`**: `QT_QPA_PLATFORM=offscreen`, never `-n auto`.

## 6. Open questions

1. **Scale choice.** Sequential (viridis-like) suits magnitude; diverging suits deviation
   from a control. Sequential is assumed; a diverging option is a later addition.
2. **Precision.** Fixed significant figures vs per-column formatting. `Shape_Area` and
   `ColorLab_DeltaE2000MedianFromMedoid` want different displays; a single rule is assumed
   until it reads badly.
3. **`ColorLab_MedoidColorHex` is a string column.** Non-numeric columns cannot be scaled.
   Assumed: they are offered as text-only, or filtered out of the picker. Not settled.
