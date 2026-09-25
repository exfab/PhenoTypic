# Feret move review (`3171781b`, `4bbdfe45`)

**Date:** 2026-09-25 · **Scope:** the post-PR follow-up that moves `Shape_Min/MaxFeretDiameter`
to `Size_Min/MaxFeretDiameter` (PR #248, branch `claude/size-measures-consolidation`).
**Baseline:** `81d19ec66` (the branch's merge base with `origin/main`).
**Out of scope, already verified by the lead:** the 172 focused tests, the two mutations, the
main-vs-tip differential on 552 objects, the golden-file scope, and the running full regression.

## Verdict

The code move is correct. The helper is byte-identical to main's, and the hull feeding it
comes from the same input and the same Qhull options. Degenerate objects produce NaN.
**No code consumer was missed.**

The open findings are all in prose: the PR description, the resume hand-off, the schema
guide, and one measurer docstring still describe the pre-move state.

| Severity | Count |
|---|---|
| CRITICAL / HIGH | 0 |
| MEDIUM | 2 |
| LOW | 3 |
| INFO | 5 |

---

## Findings

### M1 (MEDIUM): the PR #248 description contradicts the Feret move in five places

**Where:** `gh pr view 248 -R exfab/PhenoTypic --json body` (remote head is still `bec1822b`,
so the two Feret commits are not pushed yet).

**Evidence:**

- **Summary, `MeasureSize` bullet.** "emits `Area`, `IntegratedIntensity`, `Perimeter`,
  `ConvexArea` …, `BboxArea`, `Major/MinorAxisLength`, and a five-member radius family".
  The Feret diameters are missing.
- **Summary, `MeasureShape` bullet.** "keeps `Circularity`, … `Orientation` and
  `Min/MaxFeretDiameter`". This is now false.
- **Breaking change: rename table.** It has no `Shape_MinFeretDiameter` or
  `Shape_MaxFeretDiameter` row. Spec §6 says "It goes in the PR description", so this table
  is the user-facing migration map for the break.
- **Decisions (spec §10).** "Size columns are **removed** from Shape, with no aliases.
  Feret stays in Shape." Spec §10 now records the reversal.
- **Verification.** "Final full regression at `13821bba`" and "3 newly passing: the
  recaptured Shape/Intensity/KeepSectionLargest goldens" describe the tree before
  `3171781b`. The Size golden has since changed as well (`4bbdfe45`).

**Fix:** when the commits are pushed, edit the body:

1. Add `MinFeretDiameter`/`MaxFeretDiameter` to the `MeasureSize` bullet.
2. Remove them from the `MeasureShape` bullet.
3. Extend the table's "`Size_` + same label" row with `Shape_MinFeretDiameter` and
   `Shape_MaxFeretDiameter`, to match spec §6.
4. Replace "Feret stays in Shape" with "Feret moved to Size (post-PR user decision,
   2026-09-25)".
5. Refresh the Verification block once the running regression lands.

### M2 (MEDIUM): `RESUME.md` still lists "Feret stays in Shape" as a do-not-re-litigate user decision

**Where:** `docs/superpowers/plans/2026-09-24-size-measures-consolidation/RESUME.md:40-41`
and `:51`.

**Evidence:**

- `:40-41` reads "`MeasureShape` keeps the form descriptors (… Orientation,
  Min/MaxFeretDiameter)".
- `:51` sits in the table headed "Decisions the user made (do not re-litigate)" and reads
  "What moves | Radii, perimeter, hull and box areas, ellipse axes. **Feret stays in Shape**".

This file is the hand-off a resumed or newly dispatched agent reads first. With that
framing, a later agent would treat the current code as a violation of a user decision, and
might "fix" it back. The spec was updated to strike the old decision; `RESUME.md` was not.

**Fix:**

- At `:40-41`, move `Min/MaxFeretDiameter` from the Shape list to the Size list.
- At `:51`, strike the old decision in the same way spec §1 and §10 do: "~~Feret stays in
  Shape~~ **Reversed 2026-09-25: Feret moved to Size**".

### L1 (LOW): `schema/CLAUDE.md` still names `SHAPE` as the example straddler, two paragraphs above saying no primary enum straddles

**Where:** `src/phenotypic/schema/CLAUDE.md:69` against `:79-81`.

**Evidence:**

- `:69` reads "**Straddlers** (enums whose members span tiers, e.g. `SHAPE`) subclass the
  neutral `PrimaryMeasure`/`DerivedMeasure` …".
- `:80` reads "No primary enum straddles since 0.20.0".

The commit rewrote the old example paragraph but left this one untouched.

**Fix:** change the example at `:69` to one that still straddles, e.g. "e.g. the growth
models: `LOG_GROWTH_MODEL` spans Tier-1 kinetics and Quality diagnostics". Alternatively,
drop "e.g. `SHAPE`" altogether.

### L2 (LOW): the new `schema/CLAUDE.md` sentence about derived enums is inaccurate

**Where:** `src/phenotypic/schema/CLAUDE.md:85-86`. The sentence is new in `3171781b`.

**Evidence:** it reads "Derived enums (the growth models) still tag every member with
`Entry(tier=1, derivation_type=...)`". That does not hold for `LOG_GROWTH_MODEL`:

- `LAM`, `BETA` and `K_MAX` (`src/phenotypic/schema/_log_growth_model.py:34-49`) carry
  `derivation_type="diagnostic"` and **no** `tier`. They resolve to Quality
  (`tests/unit/schema/test_classification.py` `test_derived_growth_models_and_edge_correction`).
- `EDGE_CORRECTION` is also a `DerivedMeasure`, and it defers its tier through
  `"normalization"`.

**Fix:** reword to "Derived enums (the growth models) still straddle: each member carries an
`Entry(derivation_type=...)`, and the parameterization members also carry `tier=1`."

### L3 (LOW): the `MeasureTexture` docstring still sends readers to `MeasureShape` for Feret diameters

**Where:** `src/phenotypic/measure/_measure_texture.py:81-82`.

**Evidence:** it reads "- :class:`MeasureShape` for geometric morphology metrics
(circularity, Feret diameters) that complement texture." Main's `_measure_size.py:42` had
the same kind of cross-reference, and this commit updated it (`_measure_size.py:67-68`).
This one was missed. It is the only `src/` hit for "feret" outside the moved code and the
schema.

**Fix:** "(circularity, solidity, eccentricity)". Optionally, add a
`:class:`MeasureSize`` line for the calipers.

---

### I1 (INFO): `SHAPE` is now structurally a `DescriptiveTrait`

**Where:**

- `src/phenotypic/schema/_shape.py:8`, `:26-28`;
- `src/phenotypic/schema/_tiers.py:6-8`, `:90`;
- `src/phenotypic/schema/CLAUDE.md:65`.

**Evidence:** `SHAPE(PrimaryMeasure)` overrides `tier()` to return 2 and has no per-member
`tier` override. That is exactly what `DescriptiveTrait` (`_tiers.py:105-110`) provides.
Meanwhile three docs still describe `PrimaryMeasure` as the base for straddlers:

- `_tiers.py:90`: "(used by straddlers)";
- `_tiers.py:6-8`: "Straddling enums subclass the neutral parent";
- `CLAUDE.md:65`: "(straddler base)".

Since this change, `PrimaryMeasure`'s only direct subclass besides the three tier bases is
`SHAPE`, and `SHAPE` no longer straddles.

**Option:** re-parent to `class SHAPE(DescriptiveTrait)` and delete the `tier()` override.
Resolved tiers are unchanged, and `test_shape_is_uniformly_tier2_…` still passes. The
`CLAUDE.md:79-86` paragraph would then shrink. Keeping the current form is also defensible
as the minimal change. Either way, `CLAUDE.md:81` should keep matching the code.

### I2 (INFO): spec §1 still omits the Feret diameters from the size magnitudes it lists

**Where:** `docs/superpowers/specs/2026-09-24-size-measures-consolidation/design.md:14-15`.

**Evidence:** it reads "`MeasureShape` currently mixes **size magnitudes** (area, perimeter,
radii, axis lengths, hull and box areas, all tagged `tier=1`) …". On main the Feret
diameters were also `tier=1` size magnitudes in SHAPE. §1's Non-goals, §3, §5, §6 and §10
are all updated. The Objective paragraph is the one place that still frames Feret as
out of scope.

**Fix (optional):** add "Feret diameters" to the list.

### I3 (INFO): ragged line wrap in the `MeasureSize` docstring

**Where:** `src/phenotypic/measure/_measure_size.py:27-29`.

**Evidence:** the insertion left "… and four radii (median,\n    mean, robust mean and
maximum)\n    measured from one center, …". The short middle line is left over from
the edit. It renders fine in Sphinx, but it reads oddly in source and in `help()`.

**Fix:** reflow the paragraph.

### I4 (INFO): the Size table on the biological-meaning page has no Feret row

**Where:** `docs/source/explanation/measurement_metrics_biological_meaning.md:20-32`.

**Evidence:** Feret was never in that page's Shape table on main either (`git grep -i feret
81d19ec66 -- docs/source` finds nothing), so nothing regressed. But the page now presents
MeasureSize's table as the size reference, and the calipers are size columns.

**Fix (optional):** add a row such as "MinFeretDiameter / MaxFeretDiameter | pixels |
Narrowest / widest caliper width across all orientations; a large Max/Min ratio flags an
elongated colony".

### I5 (INFO): the classification test's docstring claims more than it asserts

**Where:** `tests/unit/schema/test_classification.py:121-134`.

**Evidence:**

- The docstring says the calipers resolve to tier 1 "through SIZE(DirectPhenotype) without
  an Entry tag". The assertions at `:133-134` check only `(resolved_kind, resolved_tier)`,
  so a leftover `Entry(..., tier=1)` on the SIZE members would still pass.
- The same check is already implied by `test_tier1_primary_enums` (`:151-156`), which
  asserts tier 1 for every SIZE member.

**Fix (optional):** assert the member's `Entry` carries no explicit tier. Otherwise, drop
"without an Entry tag" from the docstring.

---

## Verified clean

### Helper is byte-identical to main

`awk`-extracted `_calculate_feret_diameters` (51 lines, decorator excluded) from
`git show 81d19ec66:src/phenotypic/measure/_measure_shape.py` and from the tip's
`_measure_size.py`. `diff` reports them identical. `@staticmethod` is preserved on both.

### Same hull input as main

- **Main** (`_measure_shape.py:166-191` at `81d19ec66`) used
  `ConvexHull(current_props.coords)`, with `warnings.filterwarnings("ignore", message="Qhull")`,
  `except QhullError: None`, and then `coords[convex_hull.vertices]`.
- **Tip:** `convex_hull_area(props.coords)` (`_object_geometry.py:22-41`) does the same
  build, filter and exception handling. `_measure_size.py:272-279` passes
  `props.coords[hull.vertices]`.
- The loop runs over `image.objects.props` in both versions, so row alignment is unchanged.

### ConvexArea is unchanged by the hull reuse

`hull_area` is `float(hull.volume)`, or `nan` on failure. It is exactly the old
`convex_hull_area(props.coords)[1]`.

### `hull is None` gives NaN Feret

The measurement arrays are `np.full(n_objects, np.nan)` for every SIZE member
(`_measure_size.py:253-256`), and the Feret assignment sits inside `if hull is not None`. A
successful 2-D hull has at least 3 vertices, so the helper's `< 2` guard is unreachable, as
it was on main. The degenerate test (`test_measure_size.py:172-185`) asserts NaN for both
Feret columns.

### `MeasureShape` after the removal

- It still uses `np`, `convex_hull_area` and `object_edt`; there are no dead imports.
- Solidity keeps its own `if hull is not None` guard.

### Consumer sweep: no missed consumer

**Case-insensitive grep for `feret|caliper`.** `git grep -niI` over tracked files, plus
`git grep -a FeretDiameter HEAD` for binaries. Every hit outside `docs/superpowers/`
history falls in one of these groups:

- the moved code and schema;
- the change note;
- the updated tests;
- `_image_handler.py:560`, skimage's unrelated `feret_diameter_max` docstring;
- L3 above.

**Nothing found in any of these:**

- `scripts/`;
- `docs/source/`, including notebooks;
- GUI defaults, `FEATURES.md` and `WORKFLOWS.md`;
- `tune/`, `qc` and `sdk_/`;
- the bundled CSVs (`data/meas/*.csv`, which had no Feret column on main either);
- pipeline JSON fixtures;
- `RemoveByFeature(feature="MeasureShape", value="…Feret…")`-style bare-label lookups.

**Enum-member references.** `FERET` enum members are referenced only in the updated code
and tests, plus historical plan and report files.

**Goldens.** The only goldens containing Feret columns are
`tests/migration/_goldens/measure.MeasureSize.parquet` and
`tests/unit/measure/_golden/size_consolidation_baseline.parquet` (main's baseline). On main,
only `measure.MeasureShape.parquet` carried them, so no other golden needed recapture.

**Pipelines that lose or gain columns.**

- All 7 prefabs that run `MeasureShape` also run `MeasureSize`, so none silently loses the
  calipers.
- `_spimager_pipeline.py` (`MeasureSize` only) gains two columns, which is additive.
- `scripts/bench_*_drift.py` run `MeasureShape` alone and simply drop two columns. They do
  not name them.

**Prefix lists.** `analysis/_error_cutoffs.py:33-41` covers both `Size_` and `Shape_`.

### Straddler dependencies

`git grep -i straddl` over the whole repo, including `docs/source` and `scripts`:

- No docs badge builder or coverage gate needs a primary straddler.
- The only straddler tests (`test_classification.py:72-81`) build synthetic enums.
- `docs/source/_extensions`, `scripts`, `_gui` and `qc` never read `resolved_tier` or
  `tier()`.
- The explanation page `measurement_classification_system.md` makes no per-enum claim about
  `SHAPE`.

### Change-note RST table

The new rows fit the column widths:

- `` ``Shape_MinFeretDiameter`` `` is 26 characters, within the 30-character column;
- `` ``Size_MinFeretDiameter`` `` is 25 characters, within the 32-character column.

`test_change_note.py` asserts both pairs are present in the SIZE and SHAPE notes.

### Consistency with the spec and docstrings

These all agree with the code:

- spec §1 Non-goals, §3.1 rows, §3.2, §5 `MeasureShape` bullet, §6 table and §10 note and
  row;
- the SIZE and SHAPE docstrings;
- the `MeasureSize` and `MeasureShape` docstrings;
- `schema/CLAUDE.md:181`, whose `get_headers()` example matches the enum order
  (Circularity, Eccentricity);
- the `diff_migration_scenarios.py` and `size_rename.pl` mappings.

The exceptions are the items listed above.
