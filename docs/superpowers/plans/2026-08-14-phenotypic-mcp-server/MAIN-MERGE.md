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

---

## `deploy_plan` is no longer a `W0` call — a spec defect the merge created

Found by C3 while re-applying Task 9 on the merged file; both halves verified
independently at `_cli_slurm_array_scripts.py:388-401`.

### 1. Building a spec reads every input image, twice

```python
work_id, _ = work_id_for_image(config, dataset.name, image_path)   # :388  hashes image + pipeline
identity_rows.append((work_id, file_sha256(image_path), uuid4().hex))  # :389  hashes the image AGAIN
... f"{shlex.quote(file_sha256(config.pipeline_json))}"            # :401
```

`file_sha256` streams the whole file in 1 MiB chunks. So building the spec for an
N-image chunk costs **2N full image reads plus N pipeline reads**. Nothing is
written — the purity guarantee Task 9 exists for still holds, and the purity test
correctly passes — but the *cost* model does not.

**§1.5 defines `W0` as "pure computation over metadata; **no image I/O**", and
§5.3 classifies `deploy_plan` as `W0`.** That is now false. A plan over a
480-image dataset reads 960 images. Under §1.5's routing table `W0` takes no
`LocalComputeSlot` and runs inline on the event loop — so as specified, a single
`deploy_plan` would block every other subagent's calls for the duration of a
full-dataset hash. This is the same failure `run_in_executor` exists to prevent
for `W1`, and §5.5 already had to carve out `detail: "results"` for exactly this
reason.

**Options, none free:** reclassify `deploy_plan` as `W1` (takes the slot, bounded
by a timeout); keep it `W0` but force it through the executor as §5.5 does; or
have the builder accept precomputed identity rows so a preview can skip hashing.
The third is the only one that keeps a plan genuinely cheap, and it interacts
with the OME-Zarr work, which is redesigning what a per-image identity even is.

### 2. The spec is nondeterministic — a preview can never be byte-identical

Every `ATTEMPT_IDS` entry is a fresh `uuid4().hex` (`:389`), regenerated at
submit time. So the script `deploy_plan` shows and the script `deploy_start`
submits **cannot** match byte-for-byte, and §5.3's `sbatch_preview` implies they
do.

C3 handled this correctly in the tests rather than papering over it: byte
equality with **only** the `ATTEMPT_IDS` block masked, plus a second guard that
builds twice and asserts the renders *differ* and match once masked (so the mask
cannot quietly grow), plus a structural consumption proof. Mutation M6 — making a
second field per-call random — fails both, which is what keeps the mask honest.

**Phase 2C must choose:** present the preview as "modulo attempt ids", or thread
attempt ids into the builder so a plan and its submission share them. The second
is what makes a `plan_token` meaningful — §5.4 binds the token to an
`argv_digest`, and a digest over a nondeterministic render is not a binding.

**Both items go in the spec audit**, and both are further reasons Phase 2C waits.

---

## C8/C9 — does JournalStorage survive a realistic fleet? YES, measured

Asked by the user 2026-08-18 before deciding P1's sequencing, because their
filamentous-fungi pipelines cost **~30 min per evaluation** and the answer
decides whether distributed tuning is required for v1 rather than optional.

C7 (job 27468703) had only shown 4 workers × 15 trials = 60 appends, as `srun`
tasks in one allocation. That is not a fleet.

| Run | Shape | Result |
|---|---|---|
| **C8** (job 27555140) | SLURM **array**, 8 tasks × 50 trials = 400 | **400 persisted intact** — but SLURM packed all 8 onto **one node** (c05), so this measures a single GPFS client. `--require-distinct-nodes` caught it rather than letting it be reported as a cross-node pass |
| **C9** (job 27555152) | `srun -N8 -n8`, 8 **distinct** nodes, 50 trials each = 400 | **400 trials persisted intact across r44–r51.** verify rc=0 |

**Conclusion: journal storage supports concurrent writes from a realistic
multi-node fleet.** GPFS is not incidental to this — a job id boundary is
invisible to the filesystem; what matters is 8 separate GPFS clients, which C9
had and C8 did not.

**One measured caveat, and it is why this conclusion is workload-specific.**
C9's log carried optuna's own warning: *"taking longer than 10.0 seconds to
acquire the lock file … Retrying"*. So `JournalFileSymlinkLock` **is** genuinely
contended — refining C7's "no discrimination" reading: the lock does real work,
GPFS would merely also have serialized those appends without it.

The rate is what settles it:

| | Appends/s |
|---|---|
| C9 (400 trials in ~20 s) | **~20** — produced the warning |
| A 30-min-per-evaluation fleet, 8 workers | **~0.004** |

**~4,500× headroom.** For expensive evaluations the contention is irrelevant, which
matches C6's earlier throughput finding by a different route. **A fast pipeline
(≈1 s/evaluation) at 32 workers would approach ~32 appends/s — above what was
just measured to warn.** Record that limit rather than generalising this result.
