# Tune Engine — Deferred Work

> Tracks scope intentionally **shipped-but-not-completed** by the param-sweep-redesign build,
> per decisions made during execution planning (2026-06-03). Each item ships its *machinery* so
> the seams exist, but completion is deferred. Convert each to an issue/PR when picked up.

---

## 1. Ground-truth validation of supervised + reference-free scorers  — *deferred*

**Decision:** *Ship machinery, defer GT validation.*

- **Built (Phase 3 / Phase 4):** `ReferenceFreeScorer` + its meta-validation gate
  (`meta_validate(gt_images, grid)`, cached pass/fail, Spearman ρ≥0.7 enable / ≥0.8 unattended,
  fail-safe to `QCScorer`), and `SupervisedScorer` with a **path-configured GT loader**
  (`gt_masks_source: Path | None`, name→mask mapping, mirroring `QCScorer`'s metadata-path pattern
  so it round-trips).
- **Deferred:** actually **validating** these against real annotated plates — running the
  meta-validation correlation on a real GT set, and checking supervised metric values against
  human annotations. v1 tests cover only construction, serialization round-trip, term-shape, and
  `availability()` tiering — **not** numeric correctness vs. ground truth.
- **Unblock when:** an annotated calibration set (≥3–5 plates) exists. Then: run the meta-validation
  gate end-to-end, confirm the abstain/enable thresholds, and validate metric values.

## 2. Additional supervised metric families  — *deferred*

**Decision:** *Minimal v1: region overlap + count error only.*

- **Built (Phase 4 v1):** region metrics **Dice / IoU** + counting **count MAE**. Matching strategy:
  per-grid-cell on `GridImage`; IoU τ=0.5 greedy unique-match fallback. `availability()` reports the
  runnable tier given the GT modality present.
- **Deferred:**
  - **Partition:** ARI, VI (matching-free, touching-colony safe).
  - **Instance:** PQ (SQ×RQ), SEG, AJI — need instance-label GT.
  - **Boundary:** Hausdorff/HD95, NSD.
  - **Verify-before-implement (flagged unverified in `supervised-scorers.md`):** AJI+, Mahalanobis,
    Boundary-F θ — do **not** implement until their formulas are verified against the literature.
- **Unblock when:** GT validation (item 1) is in place and a metric beyond region/count is needed;
  pin τ / symmetrization / ICC-form / aggregation per the spec when each family is added.

## 3. Operation tuning-annotation coverage — remaining families  — ✅ *RESOLVED (annotation pass complete)*

**Decision:** *Annotate `detect/` + `enhance/` now; the rest soon.*  ⚠️ **User flagged: needed soon.**

**Status (resolved):** the annotation pass now covers **all five** families. `ANNOTATED_MODULES`
(`tests/unit/tune/_annotation_introspect.py`) was extended to `detect + enhance + refine + grid +
correction`, every numeric-tunable field in `refine/`, `grid/`, and `correction/` was annotated with a
`TuneSpec` search window or `TuneSpec(tunable=False)`, and the remaining `detect/`/`enhance/` allowlist
was worked through in the same sweep. Coverage is now **231/243 (95.1%)**. All four guards (`⊆`
invariant, apply-time backstop, back-compat, coverage) pass.

**Deliberately NOT windowed (12 allowlist entries — no fabricated bounds):** every shipped `TuneSpec`
window is grounded in the field's docstring "Typical range" or a documented library/house default. For
the 12 fields with **no defensible documented range**, no guessed window was shipped — they stay
un-annotated on the allowlist so the engine's Tier-2 heuristic flags them (`needs_review=True`) rather
than asserting a fabricated bound:

> `FocusEdge{Frangi.alpha, Frangi.gamma, Hessian.alpha, Meijering.alpha}` (ridge-filter shape params),
> `StructureSmoothing.{alpha, C}` (structure-tensor diffusion constants),
> `BayesShrinkEnhancer.sigma` / `VisuShrinkEnhancer.sigma` (auto-estimated wavelet noise σ),
> `GridAlignmentRefiner.peak_prominence` / `RefineBySineFit.peak_prominence` (signal-derived),
> `ColorCheckerProfile.{median_filter_size, stddev_mag_threshold}` (border-detection thresholds).

> **Follow-up (separate PR — values, not structure):** after the operation docstrings are enriched and
> fact-checked, set these 12 `TuneSpec` windows from verified sources (and re-check the "med"-confidence
> windows that *were* shipped — `ChanVese.{lambda1,lambda2}`, `MadHysteresis.min_size`,
> `ColorCheckerProfile.{min_swatch_area_frac,core_fraction,ridge_lambda,outlier_sigma}`, the
> Bayes/Visu *corrector* σ, `StructureSmoothing.{sigma,num_iter}`, the `min_peak_distance`s, etc.).

- **Built (annotations workstream v1):** `TuneSpec(...)` search hints + `Field(ge=, le=)` validity
  bounds on **`src/phenotypic/detect/`** and **`src/phenotypic/enhance/`** fields, with the
  `⊆` invariant test + apply-time backstop (for validator-enforced bounds, which `model_fields`
  metadata can't see) + a shrinking coverage allowlist.
- **Done (this pass):** the same annotation pass over **`src/phenotypic/refine/`**,
  **`src/phenotypic/grid/`**, and **`src/phenotypic/correction/`**, plus the residual
  `detect/`+`enhance/` allowlist. `infer_search_space` now resolves these families' annotated fields via
  Tier-1 (`source="tune_spec"`, `needs_review=False`) instead of the unbounded heuristic, so the
  `--auto-space` autonomy gate (`proposal.needs_review`) is no longer conservatively tripped by their
  numeric fields.
- **Migration rule (carried forward):** converted a `field_validator`→`Field(...)` **only** for a bare
  scalar bound (4 refine mergers' positivity guards → `Field(gt=0)`; 3 new detect bounds where no
  validator existed); kept normalizing/conditional validators in place. Back-compat `pipeline.json`
  fixtures still load. Note: migrating a custom-message validator to `Field` changes the
  `ValidationError` text (tests asserting the old message were updated to the pydantic phrasing).
- **Gating:** coverage check is **advisory** (warns on shrink) until ≥70% of numeric fields across the
  annotated families are covered, then **hard-gates**. Now hard-gating at 95.1%. The `ADVISORY_UNTIL_
  COVERAGE = 0.70` floor was left as-is: the subset/stale allowlist tests are the true no-regression
  ratchet, and a high % floor would conflict with the allowlist escape hatch for future new ops.

## 4. Spec-deferred (v1 scope caps — informational)

- **Two-level nested presence** (`conditional_on` chains of depth >1): `search-space-inference.md §6`
  caps presence-wrapping at top-level positions (depth = 1). Lifting to a two-level chain is deferred
  to a future version.
- **Dash 6c nested-op editing:** the space-editor edits **flat + presence** knobs only in v1; nested-op
  knobs are shown read-only ("edit in the source `tuning_spec.json`"). Full nested editing → v2.

---

## Postgres for distributed Optuna (tooling provided — wiring is Phase 2/6)

- **PhenoTypic is backend-agnostic (decoupled 2026-06-05):** it does **not** parse any specific
  server's handshake files. SQLite-WAL is unsafe on NFS/Lustre, so distributed SLURM array studies
  use a **generic, user-defined Postgres URL**. A user-space PostgreSQL Slurm job (e.g.
  `~/util/postgres_server/`) is **one example** of standing up a server — documented in Phase 6, not
  a PhenoTypic dependency. (The earlier `read_pg_connection_info()` helper + `_study/_pg.py` were
  removed — coupling to one user's util.)
- **Phase 2 wiring:** `OptunaConfig.storage_url` / `--storage-url` / `PHENOTYPIC_TUNE_STORAGE_URL`,
  using a **password-less** `postgresql+psycopg://USER@HOST:54399/DB` scheme (psycopg3); the password
  is resolved by **libpq** from `~/.pgpass` / `$PGPASSWORD` (standard PostgreSQL — never in argv, the
  shell, or the generated worker script). Postgres integration tests are gated behind
  `PHENOTYPIC_TEST_PG_URL` / `@pytest.mark.postgres`; the default suite uses local SQLite so CI stays
  hermetic. Local single-node runs use SQLite-WAL.
- **Phase 6 docs:** `tune_distributed_hpcc.md` documents the why + the generic `--storage-url` +
  `~/.pgpass` flow (with `~/util/postgres_server/` shown as one way to get a server).

---

## Postgres connection ergonomics — service file / `PG*` env (a future version)

**Decision (2026-06-05):** *Ship the explicit password-less `--storage-url`; defer the
"don't retype the address" conveniences to a future version.*

- **Why the address is required today:** `--storage-url` is the connection **target** (driver +
  host + port + **dbname** + user); `~/.pgpass` is **password-only**, keyed by `host:port:db:user`
  (and our entries use `db=*`, so it doesn't even pin a database). **Verified live:** an empty
  conninfo with no target defaults to the local unix socket and fails — `.pgpass` alone gives libpq
  no server to reach. So the target must come from the URL (or `PG*` env, or a service file). This is
  correct + standard; not a flaw.
- **Future ergonomics (no core code needed — standard libpq, already flows through our stack):**
  - **`~/.pg_service.conf`** named service: define `[tune]` (host/port/dbname/user) once, then
    `--storage-url "postgresql+psycopg://?service=tune"`. **Verified 2026-06-05** that this connects
    end-to-end through `psycopg → SQLAlchemy → optuna RDBStorage` with **zero code change** (target
    from the service file, password from `~/.pgpass`, nothing in argv/history/repo).
  - **`PG*` env** (`PGHOST`/`PGPORT`/`PGDATABASE`/`PGUSER`) + a bare `postgresql+psycopg://` URL —
    also verified (empty conninfo + `PG*` env connects, password from `~/.pgpass`).
- **Deferred (future version):** (1) **document** all three forms in the Phase 6
  `tune_distributed_hpcc.md` (full password-less URL · `?service=tune` · `PG*` env), with `.pgpass`
  for the secret; (2) *optional* CLI sugar — e.g. a `--pg-service NAME` flag that builds the
  `?service=NAME` URL — so a user never types host/port/db at all.
- **Unblock when:** a CLI-ergonomics polish pass, or users ask to stop retyping the address. Until
  then `?service=…` already works for anyone who wants it.

---

## Phase 4.5 robust-eval — held-out generalization (deferrals + one deviation)

> Shipped (4.5 part 1 + part 2): the reproducible calibration/held-out **split**
> (3-tier: whole-group / within-group / data-poor skip; numpy-only, cross-process
> + resume stable), the per-trial **calibration-dispersion `gap`** + **suspicious**
> under-detection flag, the **report-only held-out pass** on the winner, and the
> user-facing `deliverables/generalization.json` verdict (both-thresholds overfit
> gate, the 3-tier report, the `--held-out-fraction` / `--cv-group` CLI overrides).
> The following are intentionally deferred.

### (a) Full group-aware cross-validation  — *deferred*

**Decision:** *Ship single held-out split; defer multi-fold group-aware CV.*

- **Built:** a **single** calibration/held-out partition — one whole group held
  out (`kind="group"`), or one within-group slice (`kind="within_group"`).
- **Deferred:** proper **k-fold** group-aware CV — `StratifiedGroupKFold` /
  leave-one-group-out — rotating every group through held-out and averaging the
  generalization estimate across folds (lower-variance than a single split).
- **Unblock when:** enough groups exist that single-split variance is the
  limiting factor on the verdict's reliability.

### (b) Metadata-stratified pruning rungs  — *deferred*

**Decision:** *Ship id-sorted deterministic rungs; defer stratification.*

- **Built:** the ASHA-style fidelity ladder evaluates a candidate over a
  **deterministic, id-sorted** subset of calibration plates (`Evaluator._rung_sizes`).
- **Deferred:** **metadata-stratified** rungs — each rung a representative
  stratified sample across the grouping column, so an early rung does not
  over-represent one batch and mis-rank a candidate before the full pass.
- **Unblock when:** rung-induced ranking instability is observed on a
  multi-batch calibration set.

### (c) Incremental per-image `measure()`-frame cache  — *deferred*

**Decision:** *Ship per-rung memoization within one trial; defer a cross-trial cache.*

- **Built:** within one `Evaluator.evaluate` each image is measured **once**
  (memoized across rungs).
- **Deferred:** an **incremental cross-trial cache** of per-image `measure()`
  frames keyed by `(image, candidate-params-affecting-detection)`, so two trials
  sharing a measurement-equivalent sub-pipeline reuse the frame instead of
  re-measuring. (Memory-bounded — images are large; the project's accuracy-over-
  speed bias means this only pays once measurement dominates the trial budget.)
- **Unblock when:** profiling shows `measure()` re-computation, not scoring,
  dominates wall-clock.

### (d) §8 data-poor "CV-estimate" → **substituted** by a calibration-stability estimate — *deviation*

**Decision:** *For `kind="none"` (data-poor), report a calibration-stability
proxy instead of a cross-validation estimate.* ⚠️ **Deviation from the spec's §8.**

- **Spec §8:** a data-poor run (too few plates to reserve a held-out set) would
  fall back to a **cross-validation** estimate of the generalization gap.
- **Shipped instead:** the data-poor branch of `run_held_out` writes
  `generalization.json` with `estimate="calibration_stability"`,
  **`cv_deferred=true`**, `gap=null`, `flagged=false`, and carries the winner's
  per-trial **calibration dispersion** (`Trial.gap`, the relative across-plate
  IQR of the primary term) as the `calibration_stability` proxy, plus a
  "no untouched held-out — calibration-stability estimate (CV deferred)" warning.
- **Why:** a real CV estimate needs the multi-fold machinery deferred in (a); the
  stability proxy is a cheap, already-computed honesty signal that ships now. The
  `cv_deferred=true` flag is the explicit marker to swap in CV when (a) lands.
- **Unblock when:** (a) ships — replace the proxy with the k-fold CV estimate and
  set `cv_deferred=false`.

### (e) QC batch-panel / geometric-fusion / Count-floor  — *approximated, deferred*

**Decision:** *Approximate the gaming defense via a score-vs-Count-floor heuristic.*

- **Built:** the **`suspicious`** flag (`EvaluationResult.suspicious` /
  `Trial.suspicious`) approximates the qc gaming defense: a **high** finalized
  `score` paired with a **low** aggregated `Count` term (the "great score on
  under-detection" signature) is flagged for review. The
  `test_qc_gaming_regression` lock additionally pins that a faithful detection
  scores strictly higher than an under-detecting one.
- **Deferred:** the full qc design — a **batch panel** of per-plate diagnostics,
  **geometric fusion** of multiple count/quality signals, and a hard **Count
  floor** that rejects (not just flags) a degenerate under-detecting candidate.
- **Unblock when:** the single score-vs-Count-floor heuristic proves
  insufficient on a real gaming case (a candidate slips past the flag), or a
  hard reject (not a review flag) is wanted in the search loop.
