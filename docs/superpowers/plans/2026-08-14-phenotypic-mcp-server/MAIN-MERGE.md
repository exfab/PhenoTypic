# Merging main into `feat/mcp-server` — plan and spec audit

**Decided 2026-08-18.** Main has moved **6 commits / 82 files / +8,976 lines**
past this branch's point (`c847373c8`). Two of the six are substantive, not
cosmetic, and both reach into what this project is building on.

## What moved

| Commit | What |
|---|---|
| `379acee4` | **feat(cli): crash-safe incremental continuation** — `--resume` replaced by automatic continuation |
| `3057fbe0` + `1d8eec75` | **feat: flatten metadata namespace** |
| `068155e3` + `61590277` | **docs: require schema-owned metadata checks** — string-prefix metadata detection is now forbidden |
| `9e4159a3` | Merge of the resume rework |

## Collision surface — 5 of the files Phase 1 refactors

| File | Main's change | Phase 1's involvement |
|---|---|---|
| `_cli/_cli_slurm_array_scripts.py` | +56 / −3 | **C3 (Task 9)** extracts `build_array_script_spec` from it |
| `sdk_/_io_constants.py` | +49 / −9 | C1 moved `IMAGE_EXTS` in; C2 moved `tune_presets_dir` + 3 `SANDBOX_*` in |
| `gui/shell/_runs_registry.py` | +29 | C1 promoted → `_services/runs.py` |
| `gui/run_console/_state.py` | 9 / −9 | C1 promoted → `_services/argv.py` |
| `gui/tune/_space.py` | 2 / −2 | C2 split into pure + view halves |

Also changed and relevant later: `tune/score/*` (six files — C6 and Phase 2B),
`sdk_/_metadata_helpers.py` (+686), `sdk_/_metadata_migration.py` (+2516 new).

## DECISION 1 — merge after C3, before the Phase 1a gate

Not mid-cluster, and not deferred to the end of Phase 1.

- **Not now:** C3 is mid-extraction on a file main changed. Stopping it discards
  work and it would have to reconcile either way.
- **Not later:** C4/C5/C6 touch `sdk_/_io_constants.py` and `tune/score/*`, both
  of which main changed. The conflict surface grows with every cluster, and
  later clusters would be written against stale code.
- **At the 1a boundary** the whole phase's suite is the check on whether the
  merge was resolved correctly — 1786 passing tests plus the purity gates,
  rather than one cluster's subset.

**Order:** C3 completes → merge `origin/main` → resolve → full suite green →
Phase 1a simplify pass → Phase 1a exit gate → re-sync exfab.

Expect real conflicts in `_io_constants.py` (two Phase 1 additions vs main's
+49/−9) and in `_cli_slurm_array_scripts.py` (C3's extraction vs main's +56/−3).
Resolve toward **main's** version of shared code and re-apply the Phase 1 move
on top, rather than the reverse — main is the trunk everything else merges into.

## DECISION 2 — audit the spec against new main, after the merge

The spec was written against `c847373c8` and now describes CLI behaviour that
has changed. **Do this before Phase 2's task documents are written**, and record
findings the way DR1–DR5 were, in `review-findings.md`.

Known suspects, to confirm rather than assume:

1. **§5.4 `deploy_start`** takes `resume`, `retry_failures`, `restart`, and
   pre-validates via `validate_resume_compatibility`. If `--resume` is gone in
   favour of automatic continuation, **that entire argument contract is stale**,
   and with it §6.2's `resume_incompatible` and `scheduler_jobs_active` codes.
2. **§5.5 `deploy_status`** and §3's measurement projection assume the current
   `Metadata_*` namespace. The flatten-metadata commit changes it.
3. **The new schema-owned metadata rule** — "never `startswith('Metadata_')`,
   prefix splitting, or category-name comparison" — binds on any Phase 2 tool
   that classifies columns. §3.1's `catalog_measurements` and §5.5's
   `QC_MetadataOnly` handling both need checking against it.
4. **`tune/score/*` changed** — §4.1's `scorers_available` reports availability
   per scorer, and C6/Phase 2B are written against those classes.

The audit is a task, not a note: it produces drift-register rows with `file:line`
evidence, exactly as the original spec verification did.

---

## INCOMING: the OME-Zarr image store (not yet on main)

Flagged by the user 2026-08-18. Branch `worktree-ome-zarr-image-store` @ `21a97d3f`
— three **docs-only** commits on top of `9e4159a3`; spec at
`docs/superpowers/specs/2026-08-18-ome-zarr-image-store/design.md`. No code yet,
so nothing to merge — but it invalidates more of the MCP spec than the resume
rework does, and the audit must cover it.

What it changes:

- **The per-image HDF5 file is replaced by OME-Zarr (NGFF 0.5).** Legacy `.h5`
  becomes reachable only through an explicit new mode.
- **A new `--mode migrate`**, joining `full`/`measure`/`process`/`recompile`.
  `--mode recompile` *stops* rewriting `deliverables/metadata.csv`; that moves
  into `--mode migrate`.
- **The "dead HDF DataFrame layer" is retired.**

### Consequences for the MCP spec — to confirm in the audit, not assume

| Spec site | Why it is at risk |
|---|---|
| **§5.4 `deploy_plan` / `deploy_start`** | `mode` is enumerated `"full" \| "measure" \| "process"`. A fourth and fifth mode exist (`migrate`, `recompile`), and `migrate` is the only path to legacy data |
| **§2.3 workspace layout** | Documents `results/<dataset>/{hdf,measurements}/`. The `hdf` half becomes zarr |
| **§5.5 `deploy_status`** | Reads per-image HDF as the unit of progress |
| **§5.4 staged-GPU description** | **Most exposed.** The MCP spec describes Stage 2 writing a per-image `.npy` **sidecar**, and the OME-Zarr spec says that sidecar exists *only* as "a workaround for HDF's read-only-while-open constraint". Remove the constraint and the sidecar — and the three-stage resume contract built on it — may simply not exist. The MCP spec states it as settled fact |
| **§7 P6 subset staging** | Materializes symlink trees of image *files*. A per-image zarr store is a directory, not a file; symlinking still works but the "flat vs nested" reasoning needs re-checking |

**Sequencing implication.** The MCP spec's deploy surface (§5) and Phase 2C are
now downstream of a storage change that is still being designed. Phase 1 and
Phase 2A/2B do not touch it — they are catalog, pipeline, tune, and campaign
work. **2C should not be written until the OME-Zarr design settles**, or it will
be written against a storage layout and a mode list that are about to change.
