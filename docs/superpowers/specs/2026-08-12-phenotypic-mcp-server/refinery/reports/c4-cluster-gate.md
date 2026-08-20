# C4 cluster gate — Tasks 10a/10b/10c, 11, 12

**Scope:** `git diff 1df13f334..HEAD -- src/ tests/` on `feat/mcp-server` @ `7c1b92849`
(commits `4e6385c06`, `0b4e8ea9c`, `b1a8bb045`, `c6dbcccdd`, `115d2efae`, `340fb3d87`,
`7c1b92849`; `0654500e0` is docs-only and excluded).
**Method:** read + **20 runtime mutants**. Each mutant is a plausible wrong implementation
injected through a pytest plugin (`pytest_configure`, before test-module import), so no
file in the repo was edited. Baseline: 47 passed across the three new files in 2.0s.

---

## Verdict

**C4 is not sound as committed.** The implementation is close to right and the test
suite is unusually strong — **19 of 20 mutants killed**, and every one of the lead's
four named claims verified *positively*. But three defects survive it, two of them in
the agent-facing projection that is the whole point of the cluster:

1. **`constraints` is silently empty for 13 parameters** whose bound sits on an `anyOf`
   branch (`float | None` + `gt=0`). C4's own code, and inconsistent with its two
   sibling helpers in the same file.
2. **14 of 143 catalog entries advertise a phantom `required` parameter named `data`**
   — including 6 of the 11 measurers. Root cause is **pre-existing** (also live on
   `main`), but `7c1b92849`'s commit message rationalized it as correct rather than
   flagging it.
3. **`derive_columns` under-reports by 15 columns** — `measure()` always appends an
   `Object_Label` / `Bbox_*` / `Grid_*` info block that the derivation never sees — and
   the test named "matches a real measure run" asserts only one direction, so it cannot
   see the gap.

Plus one coverage hole: dropping `phenotypic.grid` or `phenotypic.correction` from
discovery is **invisible** to the whole of `tests/unit/services` plus the GUI registry
tests.

---

## The four named claims — all verified positively

| # | Claim | Result |
|---|---|---|
| 1 | 10a asserts **order**, not membership | **Confirmed.** `test_one_shared_module_list` (`tests/unit/services/test_catalog_reconciliation.py:42`) is `tuple(PHENOTYPIC_CLASS_MODULES) == _SUBMODULES_BEFORE_THE_LIFT`. Mutant `order` (move `phenotypic.detect.nn` from index 9 → index 1, membership unchanged) → **1 failed**, and *only* that test. Exactly the mutation a membership assertion would miss. |
| 2 | 10b genuinely derives from the constant | **Confirmed.** Mutant `discover_hardcoded` (a byte-identical literal inside `discover()`, constant untouched) → `test_discover_derives_from_the_shared_constant` fails. The loader side is separately pinned: mutant-free, `test_loader_resolves_through_the_constant` empties the constant and asserts resolution returns `None`. Both consumers are wired and both are proven wired. |
| 3 | 10c's lazy-module guard | **Confirmed twice.** Mutant `eager_only` (drop the `__all__` getattr walk) → 2 failed; mutant `unguarded_getattr` (walk present, `try/except` removed) → `test_a_failing_lazy_export_is_guarded_at_getattr_time` fails. The guard is at getattr level and the test proves it must be. |
| 4 | Task 12's `header_scheme()` dispatch — **is a texture measurer actually exercised?** | **Yes, decisively.** Mutant `blanket_get_headers` (`lambda info_cls, owner: list(info_cls.get_headers())`) → **6 failed**, including `test_texture_headers_expand_per_scale` and `test_texture_scheme_reaches_get_headers_with_the_scale`. A subtler mutant, `texture_ignores_scale` (expand only `scale[0]`), also dies. The 130-column assertion is real: `MeasureTexture(scale=[5,10])` is constructed and driven end to end. |

### Claim 5 — the two fix commits

- **`340fb3d87` (broken tune install) — fallout from C4's own change, correctly
  handled.** Before 10b, `discover()` imported only `phenotypic.abc_` bases.
  10b resolves `Scorer` and `StrategyConfig` from `phenotypic.tune.*`, which newly
  couples *every* registry consumer — the Dash builder palette included — to `tune`
  importing cleanly. The guard (`registry.py:337-347`) is at the right level (the
  per-module guard in `discover` cannot help; the failure is in resolving the base
  class). Mutant `unguarded_tune` kills
  `test_a_broken_tune_install_costs_only_the_tuning_categories`. Nothing to report
  against `main`.
- **`7c1b92849` (alias resolution) — a defect in C4's own new file, but with a
  pre-existing twin.** `_property_for` (`catalog.py:186-213`) is correct and mutant
  `no_alias` kills two tests. **However** the same bug shape is live on `main` at
  `src/phenotypic/tune/_search_space/_infer.py:174` — see *Pre-existing, report
  separately* below.

---

## Blockers

### B1 — `constraints` drops every bound that lives on an `anyOf` branch

`src/phenotypic/_services/catalog.py:164-166`

```python
def _param_constraints(prop: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in prop.items() if k not in _NON_CONSTRAINT_KEYS}
```

It reads **only the top level**. Its two siblings in the same file, `_param_type`
(`:139`) and `_param_choices` (`:157`), both go through `_property_branches(prop)` —
so this is an oversight, not a decision.

A field declared `float | None = Field(None, gt=0)` publishes `{"anyOf": [{"type":
"number", "exclusiveMinimum": 0}, {"type": "null"}], "default": None, ...}`. The
top level carries no constraint keyword, so the projection reports `constraints: {}`.

A sweep over all 143 registered classes finds **13 such parameters**:

```
FilFinderDetector.branch_threshold_px   exclusiveMinimum: 0.0   -> {}
ManualGridPointDetector.coord2          minItems/maxItems: 2    -> {}
BayesShrinkEnhancer.gat_scale_factor    exclusiveMinimum: 0     -> {}
EnhanceBlockMatch.gat_scale_factor      exclusiveMinimum: 0     -> {}
LocalEdgeDenoise.gat_scale_factor       exclusiveMinimum: 0     -> {}
NonLocalMeansDenoiser.gat_scale_factor  exclusiveMinimum: 0     -> {}
VisuShrinkEnhancer.gat_scale_factor     exclusiveMinimum: 0     -> {}
BayesShrinkCorrector.gat_scale_factor   exclusiveMinimum: 0     -> {}
VisuShrinkCorrector.gat_scale_factor    exclusiveMinimum: 0     -> {}
Sam2.checkpoint                         format: path            -> {}
Insid3Detector.reference_image          format: path            -> {}
Insid3Detector.reference_mask           format: path            -> {}
ReferenceFreeScorer.gt_masks_source     format: path            -> {}
```

**Why the tests miss it:** `test_constraints_are_json_schema_keywords`
(`test_operation_descriptor.py:17`) uses `FlattenIllumination.sigma`, a **non-optional**
`float` whose `exclusiveMinimum` is at the top level. The test proves the *keyword
spelling* — a real correction — but says nothing about *where the projection looks*.

**Impact:** the agent is told `gat_scale_factor` is an unconstrained number; it must be
`> 0`. `constraints` existing at all is the reason the descriptor is more than a schema
dump.

**Fix:** fall back to the non-`null` branches.

```python
def _param_constraints(prop: Dict[str, Any]) -> Dict[str, Any]:
    merged = {k: v for k, v in prop.items() if k not in _NON_CONSTRAINT_KEYS}
    if merged:
        return merged
    for branch in _property_branches(prop):
        if branch.get("type") == "null":
            continue
        merged.update(
            {k: v for k, v in branch.items() if k not in _NON_CONSTRAINT_KEYS}
        )
    return merged
```

Add a test on `BayesShrinkEnhancer.gat_scale_factor` (`== {"exclusiveMinimum": 0}`)
**and** a sweep asserting no parameter reports `{}` while a non-null branch declares a
constraint keyword — otherwise the next optional-with-bound field reopens the hole one
operation at a time, which is exactly the argument `7c1b92849` made for its own sweep.

---

### B2 — 14 catalog entries advertise a phantom **required** parameter `data`

Root cause: `src/phenotypic/_services/registry.py:481`

```python
model_fields = getattr(cls, "model_fields", None)
if model_fields:                       # <-- falsy for a ZERO-field model
    return self._extract_parameters_from_model_fields(cls, model_fields)
return self._extract_parameters_from_signature(cls)
```

For a pydantic operation with **no fields**, `model_fields == {}` is falsy, so extraction
falls through to `inspect.signature(cls.__init__)`, which on a pydantic model is
`(self, /, **data: Any)`. The `**data` catch-all is registered as a parameter named
`data` — and `_extract_parameters_from_signature` skips only `self`, `args`, `kwargs`,
not the actual VAR_KEYWORD name.

What the agent receives for `describe_operation("MeasureSize")`:

```json
{"name": "data", "type": null, "default": null, "required": true,
 "description": null, "constraints": {}, "choices": null, ...}
```

`MeasureSize` takes **no** parameters. `MeasureSize(data=None)` raises
`ValidationError: Extra inputs are not permitted`. So the descriptor both invents a
required parameter and hides the true "this takes nothing" contract.

Affected (14 of 143, ~10% of the catalog, incl. 6 of 11 measurers):
`SecondaryOtsuDetector`, `TriangleDetector`, `MeasureBounds`, `MeasureGridSpread`,
`MeasureIntensity`, `MeasureNeighborDist`, `MeasureShape`, `MeasureSize`,
`FocusEdgeSobel`, `ImageInverter`, `GridOversizedObjectRemover`, `KeepNearestCenter`,
`KeepSectionLargest`, `ReduceSectionsByLine`.

**Provenance:** the line is unchanged by C4 and is live on `origin/main` as
`src/phenotypic/gui/_operation_registry.py:353`. So the *bug* is pre-existing and should
also be reported against `main` (the GUI builder renders the same phantom field today).
It is listed as a C4 blocker because C4 is what makes it agent-facing, and because
`7c1b92849`'s commit message explicitly saw these params and called the behaviour
correct:

> Params the schema omits entirely (`data`, read off a stale `__init__` signature rather
> than a pydantic field) still project empty — correctly, as the schema says nothing
> about them.

Projecting empty is fine; projecting `required: true` for a parameter that does not
exist is not.

**Fix (one line, registry):**

```python
if model_fields is not None:
```

Both C4 test sweeps deliberately step around this: `test_every_param_with_a_default_reports_it`
(`test_operation_descriptor.py:258`) does `if pname not in info.cls.model_fields: continue`.
Add a test that no descriptor reports a parameter absent from `model_fields` for a
pydantic class.

---

### B3 — `derive_columns` omits the always-emitted info block

`src/phenotypic/_services/catalog.py:491-530` walks only `pipeline._meas`. But
`ImagePipeline.measure()` always appends a per-object info block — its own docstring
(`_core/_pipeline_parts/_image_pipeline_core.py:1095`) says so:

> then the per-object image-info block (``Object_Label`` followed by the ``Bbox_*`` /
> ``Grid_*`` geometry columns) last.

Measured on a real run (`OtsuDetector` + `MeasureSize` + `MeasureShape`,
`load_synth_yeast_plate()`):

- `derive_columns(pipe)` → **19** columns
- `pipe.measure(img).columns` → **38** columns
- absent from the derivation: `Object_Label`, `Bbox_CenterRR/CC`, `Bbox_MinRR/CC`,
  `Bbox_MaxRR/CC`, `Bbox_IntensityWeightedCenterRR/CC`,
  `Bbox_DistWeightedCenterRR/CC`, `Grid_RowNum`, `Grid_ColNum`, `Grid_RowMajorIdx`,
  `Grid_ColMajorIdx` (**15**), plus the four `Metadata_*` provenance columns (a
  defensible omission).

**Why the test misses it — this is the cluster's one genuine false green.**
`test_derive_columns_matches_a_real_measure_run` (`test_column_derivation.py:109`) is
named as "the end-to-end anchor", but asserts only

```python
missing = [c for c in derived if c not in measured.columns]
assert not missing
```

i.e. **derived ⊆ measured**. It cannot detect under-reporting. Its guard against
vacuousness (`len(derived) == 65 + len(derive_columns(...))`) calls `derive_columns`
again, so it is self-referential and equally blind.

**Impact:** Phase 2A's `produces_columns` is documented as this function's caller. An
agent asking which columns a pipeline yields is told 19 when the answer is 38.
`Bbox_*` in particular is load-bearing downstream — the results viewer's curation
labels key off `Bbox_CenterRR/CC` (`gui/results_viewer/_curation_labels.py:42-43`).

**Fix — pick one and say which:**
- *(preferred)* append the info-block headers, with the `Grid_*` subset gated on the
  target image type, and add the converse assertion
  `set(measured.columns) - set(derived) ⊆ {Metadata_*}`; or
- keep the narrow contract, but state the exclusion in the `derive_columns` docstring
  **and** carry it into Phase 2A's `produces_columns` description, so the agent is not
  told a partial list is complete.

Either way the anchor test needs the second direction; as written a future regression
that halves the derivation still passes it.

---

## Non-blocking findings

### N1 — dropping `phenotypic.grid` or `phenotypic.correction` from discovery is invisible

`test_the_pre_existing_categories_are_untouched` (`test_catalog_reconciliation.py:104`)
is the anti-regression guard for the `discover()` rewrite — "Rewiring `discover` must
not drop anything it already found" — but names only `BlurGauss`, `OtsuDetector`,
`MeasureSize`, `EdgeCorrector`. Two of the eight original walks have no representative.

Mutant `grid_missing` (`_discovery_targets` drops `phenotypic.grid`): **205 passed** across
`tests/unit/services` + `tests/unit/gui/test_operation_registry.py` +
`tests/unit/gui/test_param_forms.py`. Mutant `correction_missing` (8 operations) —
**205 passed** on the same wider run.

Note `EdgeCorrector`, the one name that *sounds* like it covers Corrector, is the
`analysis` edge-correction class (category `Edge Correction`), not a
`phenotypic.correction` operation — so the `Corrector` category's 8 operations have no
representative at all.

By contrast `post_missing`, `analysis_missing` and `nn_missing` are all caught. Fix:
assert the category → count map, or add one representative per original category
(`GridCropper`-family for Grid, a `Corrector` name for Corrector).

### N2 — the one surviving mutant is benign

`default_from_field_not_schema` (read `default` from `ParamInfo.default` rather than the
schema property) **survives** all 47 tests. It is near-equivalent: a sweep finds the two
sources differ for exactly 5 parameters, all tuple-vs-list
(`FocusEdgeSato.sigmas` `(1, 2, 3)` vs `[1, 2, 3]`, etc.), and `json.dumps` renders both
as the same JSON array. The current implementation (schema side) is the better choice —
it is already in JSON form — but nothing observable distinguishes them, so this is not a
coverage gap worth closing.

### N3 — new registry categories reach the GUI's aux-accepts list

`discover()` now registers `Prefab` (7), `Scorer` (4) and `Strategy` (3). The builder
palette is category-whitelisted (`gui/builder/_layout.py:181-188`) so those do **not**
appear as palette buttons — good. But `_aux_accepts_for_param`
(`gui/builder/_layout.py:1060-1064`) emits *every* registered name when a parameter's
annotation is `ImageOperation` itself, and none of those 14 classes is an
`ImageOperation`. The over-broad branch is pre-existing (it already emitted Measure /
Post / analysis names); C4 widens it by 14. `tests/unit/gui` + `tests/gui/builder`:
**1364 passed**, so nothing breaks — flagged as a latent surface, not a regression.

Also user-visible: the builder's Detector palette gains `MicroSamDetector`, `Sam2`,
`Sam3`, `DinoSam2Detector`. That looks intended (they are real detectors previously
missing), but it is a GUI change shipped with no `FEATURES.md` / tutorial note.

### N4 — catalog/loader disagree on duplicate names (latent)

The loader is **first-match** over `PHENOTYPIC_CLASS_MODULES`
(`_serializable_pipeline.py:669-676`); `_discover_from_module` does
`self._operations[name] = op_info` (`registry.py:449`), i.e. **last-match**. So for a
duplicated class name the two consumers would resolve to different classes — the exact
failure the shared constant is meant to end, and the docstring at `registry.py:200-206`
asserts they agree. A scan across all 13 modules finds **0 duplicates today**, so this
is latent. A one-line test (`no class name resolves to two distinct classes across the
module list`) would keep it that way.

### N5 — no environment exercises the real `skipped_imports` path

`skipped_imports` is `{}` in this environment: `torch` is not installed, yet
`MicroSamDetector`, `Sam2`, `Sam3` and `DinoSam2Detector` all register (their heavy
deps are deferred below class definition), and `discover()` costs **0.06 s** with no
DL stack imported. The degradation path is therefore only ever exercised through
monkeypatched `ImportError`s. That is adequate, and worth noting as the reason the
`__all__` walk carries no import-cost regression — a concern the design invites.

---

## Pre-existing, report separately (not C4's work)

1. **`registry.py:481` / `origin/main` `gui/_operation_registry.py:353`** — the zero-field
   fallthrough behind **B2**. Affects the GUI builder's parameter forms today for the
   same 14 operations.
2. **`tune/_search_space/_infer.py:174`** — `_schema_description` does
   `props.get(field_name, {})` with the **Python** name, the same alias hole `7c1b92849`
   fixed in `catalog.py`. Verified: `_schema_description(RemoveGridOutliers(),
   "cutoff_multiplier")` returns `""`; the description is published under
   `stddev_multiplier`. Cosmetic (knob descriptions only), but it is the identical bug
   class and the fix is the same `_alias_keys` walk.
3. **`_cli_readme_generator.py:140-235`** under-reports texture columns — already noted
   as out of scope by the plan; C4 correctly did not model on it.

---

## Mutation log

Harness: a pytest plugin patching the implementation in `pytest_configure` (before test
modules import), selected by `MUT=<name>`. No repo file was modified.
Target: the three new test files (47 tests) unless noted.

| Mutant | What it breaks | Result |
|---|---|---|
| `order` | reorder `detect.nn` in the constant (membership unchanged) | **killed** (1) — only `test_one_shared_module_list` |
| `discover_hardcoded` | `discover()` keeps its own identical literal | **killed** (1) |
| `eager_only` | drop the `__all__` getattr walk | **killed** (2) |
| `unguarded_getattr` | `__all__` walk without `try/except` | **killed** (1) |
| `unguarded_tune` | resolve `Scorer`/`StrategyConfig` unguarded | **killed** (1) |
| `prefab_missing` | no target for `phenotypic.prefab` | **killed** (2) |
| `post_missing` | no target for `phenotypic.post` | **killed** (1) |
| `analysis_missing` | no target for `phenotypic.analysis` | **killed** (1) |
| `nn_missing` | no target for `phenotypic.detect.nn` | **killed** (3) |
| `grid_missing` | no target for `phenotypic.grid` | **SURVIVED** (also survives 205-test wider run) → **N1** |
| `correction_missing` | no target for `phenotypic.correction` | **SURVIVED** (also survives the 205-test wider run) → **N1** |
| `blanket_get_headers` | `get_headers()` with no dispatch | **killed** (6) |
| `texture_ignores_scale` | expand only `scale[0]` | **killed** (2) |
| `class_level_infoclasses` | read infoclasses off the class, not the instance | **killed** (1) |
| `no_dedup` | drop de-duplication in `derive_columns` | **killed** (1) |
| `no_alias` | `properties.get(param_name, {})` | **killed** (2) |
| `default_from_field_not_schema` | `default` from `ParamInfo`, not the schema | **SURVIVED** — benign, → **N2** |
| `layers_imageoperation_guard` | guard on `ImageOperation` not `BaseOperation` | **killed** (1) |
| `naive_sentence_split` | `(?<=[.!?])` without the trailing `\s` | **killed** (1) |
| `verbose_ignored` | always return the full description | **killed** (2) |
| `constraints_passthrough` | no non-constraint-key filter | **killed** (2) |
| `choices_toplevel_only` | read `enum` only at top level | **killed** (1) |
| `no_ndarray_flag` | never report `ndarray` | **killed** (1) |
| `list_ops_no_category_filter` | ignore the `category` argument | **killed** (1) |

