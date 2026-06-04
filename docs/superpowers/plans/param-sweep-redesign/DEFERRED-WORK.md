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

- **Server tooling exists:** `~/util/postgres_server/` — a user-space PostgreSQL 18.4 server
  (conda env `pg`) run as a Slurm job (`sbatch pgserver.sh`), writing its node address to
  `connection_info.txt` and the superuser password to `pgpassword.txt` (port 54399). SQLite-WAL is
  unsafe on NFS/Lustre, so distributed SLURM array studies use this Postgres backend.
- **Phase 2 wiring:** `OptunaConfig.storage_url` / `--storage-url` / `PHENOTYPIC_TUNE_STORAGE_URL`,
  using the `postgresql+psycopg://USER:PW@NODE:54399/DB` scheme (psycopg3) + a
  `read_pg_connection_info()` helper that parses `connection_info.txt`/`pgpassword.txt`. Postgres
  integration tests are gated behind `PHENOTYPIC_TEST_PG_URL` / `@pytest.mark.postgres`; the default
  suite uses local SQLite so CI stays hermetic. Local single-node runs use SQLite-WAL.
- **Phase 6 docs:** `tune_distributed_hpcc.md` documents the why + the launch/read-address/wire-in flow.
