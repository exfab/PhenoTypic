# Phase 4.5 — Robust-eval Prerequisites — EXPANDED TDD PLAN (held for execution)

> Untracked working artifact (like RESUME.md / PHASE5_EXPANDED_PLAN.md). Produced
> by the code-architect expansion 2026-06-05. Slots BEFORE Phase 5 (task #15
> blocks #10). Build with isolated-worktree Opus agents, chunk-by-chunk.

## DECISIONS (user, 2026-06-05)

- **Suspicious flag (OQ#2) = cheap score-vs-Count-floor.** `suspicious = (score ≥
  suspicious_score_floor) and (terms["Count"] ≤ suspicious_count_floor)` — read
  from already-computed aggregates; ~0 cost; thresholds config-exposed +
  fact-checked. Plus the mandatory qc §7 gaming regression test.
- **CLI flags (OQ#5) = ADD override flags now.** Add `--held-out-fraction` and
  `--cv-group` (and `--gap-margin`?) to the `run` subcommand, threading to
  `HeldOutConfig` overrides; group key still auto-infers from the scorer's
  `groupby[0]` when not given.
- **Per-trial gap (OQ#1) = Option A (CONFIRMED 2026-06-05, research-backed).**
  Per-trial `Trial.gap` = relative across-plate **dispersion** (the sklearn
  `std_test_score` analogue — Cawley & Talbot 2010 / Varma & Simon 2006 establish
  the held-out is for the *selected* model, not per-candidate; no framework runs a
  fresh held-out per config). The **winner** gets the one true held-out
  generalization gap in `generalization.json`. Per-trial IQR is a **relative
  shortlist risk flag**, NOT the generalization claim (variance can also reflect
  small folds/heterogeneity — documented caveat). Journal stays immutable.
  → Chunk C populates `gap` from dispersion; Chunk D writes the winner held-out
  gap to `generalization.json` (does NOT overwrite `Trial.gap`).
- **Settled defaults (not OQs):** gap-margins (`rel=0.15`, `abs=0.05`), held-out
  fraction (`0.2`), `min_heldout_plates=6`, `min_stability_n=4` → conservative
  config fields + `# TODO: review (unverified)` + a later `fact-checker` pass.
  Deliverable shape: new `deliverables/generalization.json`; `splits/split.json`
  at the **output-dir root** (resume-state, not a deliverable).
- **Spec deviation (recorded in DEFERRED-WORK):** the §8 data-poor fallback uses a
  **calibration-stability estimate** (CV deferred), marked
  `{"cv_deferred": true, "estimate": "calibration_stability"}` in
  `generalization.json`. DEFER: group-aware CV (StratifiedGroupKFold/LOGO),
  metadata-stratified pruning rungs, the incremental per-image-frame cache.

## Verified ground-truth (live anchors)

- `tune/_evaluation/_evaluator.py`: `EvaluationResult(frozen)` L71 (`score`,
  `terms`, `n_images`, `objectives=None`, `failed`, `pruned`); `Evaluator(frozen)`
  L101 (`stability_weight=0.5`, `failure_score=0.0`, `rung_floor=6`,
  `rung_factor=3`, `min_rungs=2`); `evaluate(base, scorer, params, images, *,
  channel)` L150 — the held-out pass re-uses this on the held-out list, single
  full pass, no channel; `_robust_aggregate` L54 (`np.percentile [75,25]`);
  `_project_finalize` L26.
- `tune/_study_store.py`: `Trial(frozen)` L22; `_COLUMNS` L133; `to_dataframe`
  L138; `from_parquet` L168; `_parse_objectives` L194 (the back-compat pattern).
- `tune/_study/_optuna_store.py`: `_ATTR_*` L28-36; `append` L95/L117; `_to_trial`
  L148/L180 (defensive `attrs.get`).
- `tune/_engine.py`: `optimize(images)` L48; appends `Trial` L94; `best()` return
  L117; `best_pipeline()` L41 — held-out attaches AFTER the loop, winner only.
- `tune/_tune_cli/_run.py`: `run_tuning` L172; `_load_images` L51 (where split is
  created/persisted/resumed); deliverable writers L234/267/272. Live flags:
  `--strategy --n-trials --screen --storage-url --slurm` (NO held-out/cv/stability
  flag today).
- `tools_/_io_constants.py`: filenames L144-182; `DIR_DELIVERABLES` L345;
  helper pattern `best_pipeline_path` L525, `pareto_best_pipeline_path` L564.
- `tune/_scoring/_qc_scorer.py`: `QCScorer` L45 Count-only; `_threshold_anchored`
  L39 (metric=inf→0, 0→1) — the suspicious heuristic reads `terms["Count"]`.
- `_core/_image_parts/_image_handler.py`: `Image.name` L194 → dataset-identity +
  deterministic split ordering (by name, reproducible cross-process).
- Locks: `test_grid_golden_manifest.py` asserts `total_pipelines==6` (schema-
  independent); `test_grid_byte_compat_lock.py` (verify it doesn't hash the
  trials-parquet column set — append new cols LAST if it does);
  `test_lazy_import_lock.py` (all new code numpy-only, optuna-free).

## Chunks (strict A → (B ∥ C) → D → E; orchestrator owns the seam files)

| Chunk | Owns | Files | Deps |
|---|---|---|---|
| **A. Schema + back-compat seam** | `gap: Optional[float]=None` + `suspicious: bool=False` on `EvaluationResult`+`Trial`; journal round-trip (scalar `gap` col + `suspicious` bool col, appended LAST; `_parse_optional_float`); Optuna `pheno_gap`/`pheno_suspicious` user_attrs; `DIR_SPLITS`+`split_assignment_path`(root)+`generalization_path`(deliverable). | `_evaluator.py`,`_study_store.py`,`_optuna_store.py`,`_io_constants.py` | — |
| **B. Split + determinism** | `_split.py` (new): `_dataset_identity` (sha256 sorted names), `_split_subseed` (`SeedSequence([seed, dataset_int]).spawn`), `derive_split` (3-tier: group-whole→within-group→none), `Split` dataclass, `write/read/resolve_split` (root `splits/split.json`, resume-reuses); `HeldOutConfig` on `TuningSpec` + group-key auto-infer from `QCScorer.check.groupby[0]`. | `tune/_evaluation/_split.py`(new),`_spec.py`,`_io_constants.py` | A |
| **C. Per-trial gap + suspicious** | `_per_trial_dispersion` (relative IQR of primary term) → `EvaluationResult.gap` [PENDING OQ#1]; `_is_suspicious(score,terms,n_images,*,score_floor,count_floor)` → `.suspicious`; thresholds on a config model. | `_evaluator.py`,`_spec.py` | A |
| **D. Held-out pass + generalization.json** | `_generalization.py`(new): `GeneralizationReport`, `compute_generalization_gap(cal,heldout,*,rel,abs_floor)` (relative+floor, flag), `run_held_out(spec,winner_params,heldout_imgs,cal_score)` (reuse `evaluate`, report-only, data-poor fallback); engine wires it post-loop on the winner; CLI writes `generalization_path`. | `tune/_evaluation/_generalization.py`(new),`_engine.py`,`_tune_cli/_run.py` | A,B,C |
| **E. CLI flags + gaming regression + gate** | `--held-out-fraction`/`--cv-group` (+ maybe `--gap-margin`) → `HeldOutConfig`; qc §7 gaming regression test (under-detect scores strictly lower — verify `QCScorer` already passes); `DEFERRED-WORK.md`; whole-phase gate. | `__main__.py`,`_tune_cli/_run.py`,tests,`DEFERRED-WORK.md` | A–D |

## Per-chunk TDD tasks (condensed — full detail in the 2026-06-05 architect transcript)

**A1** `EvaluationResult.gap`/`.suspicious` defaults · **A2** `Trial` fields +
journal round-trip + legacy-parquet-loads test (mirror
`test_study_store_objectives.py`; `gap` = scalar nullable float column,
`suspicious` = bool col, appended LAST; add `_parse_optional_float`) · **A3**
Optuna round-trip (gated; `pheno_gap`/`pheno_suspicious`) · **A4** io-constants
`DIR_SPLITS`+`splits_dir`(root)+`split_assignment_path`+`generalization_path`(deliverable).

**B1** `_dataset_identity` order-independent + changes-with-added-plate;
`_split_subseed` deterministic · **B2** `derive_split` 3-tier (group/within_group/
none) deterministic · **B3** `write/read/resolve_split` + resume-reuses-persisted
(pass a *different* seed on 2nd call → identical partition) · **B4**
`HeldOutConfig` on `TuningSpec` (back-compat default) + group-key auto-infer.

**C1** `_per_trial_dispersion` = relative IQR of primary term → `gap` [PENDING
OQ#1: Option A=this dispersion; Option B=true per-trial held-out] · **C2**
`_is_suspicious` (score≥floor AND Count≤floor) → `.suspicious`; faithful→False;
degenerate→True.

**D1** `compute_generalization_gap` relative+floor+flag; data-poor fallback
(`cv_deferred`, calibration-stability, warning); within-group caveat · **D2**
engine runs held-out on the winner only; winner pick unchanged whether flagged ·
**D3** CLI `run_tuning` resolves+persists split, passes calibration/held-out
subsets to engine, writes `generalization.json`; resume reuses split.

**E1** qc §7 gaming regression (verify QCScorer passes; lock it) · **E2** CLI
`--held-out-fraction`/`--cv-group` parse→`HeldOutConfig` · **E3** `DEFERRED-WORK.md`
+ deviation record.

## Back-compat (mirror the `objectives` sidecar exactly)
Nullable fields w/ safe defaults; parquet columns appended LAST + defensive
`row.get`/`_parse_optional_float`; Optuna new `user_attrs` restored via
`attrs.get(default)`; `TuningSpec.held_out=HeldOutConfig()` default so legacy
`tuning_spec.json` loads; golden lock untouched (verify byte-compat lock doesn't
hash trials-parquet columns).

## Determinism
Master seed = `spec.strategy.seed` (persisted in `tuning_spec.json`); sub-seed =
`SeedSequence([master_seed, int(sha256(sorted names)[:16],16)]).spawn(1)[0]`;
split persisted to root `splits/split.json` `{kind,group_key,calibration[names],
held_out[names],seed_entropy,dataset_identity,within_group_caveat}`;
`resolve_split` returns persisted verbatim (ignores passed seed) → resume
reproduces identical study; selection by image `.name` (cross-process stable).

## Numeric defaults (conservative config fields; fact-check posture)
λ=0.5 (spec, shipped) · held_out_fraction=0.2 (TODO) · min_heldout_plates=6 (TODO)
· min_stability_n=4 (TODO) · gap_margin_relative=0.15 (TODO) ·
gap_margin_absolute=0.05 (TODO) · suspicious_score_floor=0.7 (TODO) ·
suspicious_count_floor=0.3 (TODO). Cross-ref `imageproc-tuning-bounds-references`
memory; run a `fact-checker` per §6.5 before finalizing.

## Whole-phase gate (all -n 8)
Phase-4.5 suite + full `tests/unit/tune` regression + standing locks
(`test_lazy_import_lock`, `test_grid_golden_manifest`, `test_grid_byte_compat_lock`,
`test_study_store_objectives`, `test_tuning_spec`) + `test_io_constants` + `mypy
src/phenotypic/tune` + `ruff` + doctests on `_split.py`/`_generalization.py`.
