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

## 3. Operation tuning-annotation coverage — remaining families  — *high-priority deferred*

**Decision:** *Annotate `detect/` + `enhance/` now; the rest soon.*  ⚠️ **User flagged: needed soon.**

- **Built (annotations workstream v1):** `TuneSpec(...)` search hints + `Field(ge=, le=)` validity
  bounds on **`src/phenotypic/detect/`** and **`src/phenotypic/enhance/`** fields, with the
  `⊆` invariant test + apply-time backstop (for validator-enforced bounds, which `model_fields`
  metadata can't see) + a shrinking coverage allowlist.
- **Deferred (do next):** the same annotation pass over **`src/phenotypic/refine/`**,
  **`src/phenotypic/grid/`**, and **`src/phenotypic/correction/`**. Until then,
  `infer_search_space` flags those families' numeric fields as `needs_review=True`, keeping the
  `--auto-space` autonomy gate (`proposal.needs_review`) conservative for pipelines that use them.
- **Migration rule (carry forward):** convert a `field_validator`→`Field(...)` **only** for a bare
  scalar bound; keep normalizing/conditional validators in place (split when both exist). Back-compat
  `pipeline.json` fixtures must still load after any `Field` tightening.
- **Gating:** coverage check is **advisory** (warns on shrink) until ≥70% of numeric fields across the
  annotated families are covered, then **hard-gates**. Re-evaluate the threshold as families land.

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
