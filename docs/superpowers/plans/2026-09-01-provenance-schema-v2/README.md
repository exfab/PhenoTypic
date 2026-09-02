# Provenance schema v2 implementation plan

> **For agentic workers:** use the subagent-driven-development and
> test-driven-development workflows. Commit after each task.

**Goal:** Preserve every pipeline application across process, full CLI, staged,
and programmatic workflows while adding explicit, GPFS-efficient schema-v1
migration.

**Spec:**
[`design.md`](../../specs/2026-09-01-provenance-schema-v2/design.md)

## Global constraints

- Use `uv` exclusively. Write each behavioral test first and observe the
  expected failure before production code.
- Keep store schema version 3. New v2 applications require a non-empty version;
  null is a migrate-only legacy allowance.
- Persist basenames, never directory paths. Do not fabricate legacy values.
- Keep `Image.provenance` flattened and immutable; operation sequences remain
  global and retry offsets application-local.
- Reuse migration leases, lifecycle, dispatcher, and fencing. Do not introduce
  a scheduler sidecar or a custom Slurm wrapper.
- Migration inventory must not descend into Zarr chunks; each store worker gets
  one root read and writes only when upgrading v1.
- Run Ruff only on explicit changed paths. Preserve unrelated worktree changes.
- Full verification runs through Slurm with grouped, massively parallel unit
  suites. Run an independent whole-branch review before completion.

## Tasks

### Task 1: Core schema and application ownership

- [ ] Add failing tests for the v2 shape, strict validation, immutable flattened
  reads, global operation sequencing, application-local retry truncation,
  process/full/programmatic ownership, repeated pipeline applications, and v1
  mutation refusal.
- [ ] Implement normalized journal/application helpers and replace root-level
  mutation. Capture exact import/input basenames and installed versions.
- [ ] Update operation/pipeline wrappers and full/process/staged call sites so
  one invocation owns one application and continuation reopens its checkpoint.
- [ ] Run focused core and CLI provenance tests, then commit.

### Task 2: Store I/O and chained application preservation

- [ ] Add failing tests for v2 store round trips and recursive process
  sanitization across multiple applications.
- [ ] Update store readers/writers and pipeline identity setup so process output
  can feed full CLI and browse readers without losing application boundaries.
- [ ] Add the process -> full -> exported-store integration regression and
  staged single-application coverage. Run focused tests, then commit.

### Task 3: Local migration and target classification

- [ ] Add failing tests for v1-to-v2 conversion, version/filename recovery and
  null fallback, idempotence, malformed/future refusal, dry-run, active-output
  refusal, `--delete-sources`, and full/store/process-tree autodetection.
- [ ] Implement the one-read atomic root upgrader, reuse existing layout
  predicates, integrate full-run marker recertification, and add
  `provenance_upgraded` to results/reports.
- [ ] Reuse local joblib parallelism and prove inventory never scans chunks.
  Run focused migration tests, then commit.

### Task 4: Migration Slurm topology

- [ ] Add failing tests for typed provenance-only tasks, versioned manifest/
  worker/status evidence, store-array -> seal -> finalizer ordering, external
  direct-store control roots, and absence of standalone sidecar submissions.
- [ ] Extend the existing migration dispatcher and fencing with an explicit
  provenance-only topology while retaining the full-run chain.
- [ ] Run focused Slurm generation/worker/finalizer tests, then commit.

### Task 5: Documentation, instructions, and verification

- [ ] Add the user migration guide under `docs/source/how_to/pages/`, including
  local/Slurm commands and the exact null-version convention.
- [ ] Update `CLAUDE.md` (the root `AGENTS.md` symlink target) without replacing
  the symlink: only migrated legacy applications may have a null version, and
  consumers must not substitute the migration release.
- [ ] Supersede the process-mode spec's one-hop rule and update Zarr storage
  documentation.
- [ ] Run explicit-path Ruff, focused tests, mypy for affected modules, and
  `graphify update .`; commit documentation and final fixes.
- [ ] Submit full grouped unit verification through the Slurm workflow, inspect
  terminal evidence, request independent whole-branch review, fix findings,
  and rerun affected verification.
