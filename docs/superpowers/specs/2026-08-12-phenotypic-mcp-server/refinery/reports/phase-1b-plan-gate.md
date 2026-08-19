# Phase 1b — pre-dispatch plan gate

**Reviewer:** `execute-plan-orchestration:plan-reviewer` (Opus, high effort)
**Date:** 2026-08-19
**Tree:** `/bigdata/iwheeldonlab/anguy344/PhenoTypic`
**Reviewed at:** `8a7a49bb8`; re-verified against `0a246827f` (HEAD)
**Scope:** `execution.md` clustering, `phase-1b-engine-prerequisites.md` Tasks 10–20,
with Tasks 19 and 20 treated as unreviewed.
**Method:** every claim in the task bodies checked against shipped source. Analysis
only — no file in `src/` was modified.

---

## Verdict

**Not safe to dispatch C8 or C9 as written.** Each contains a central factual claim
that is false against the code, and in C9's case the fix as specified introduces a
new failure mode that the stated acceptance test cannot catch.

- **C4** — dispatchable after M-1 (a one-paragraph restatement of Task 10a).
- **C5, C6** — dispatchable; C6 needs its `Files` block corrected (B-1a).
- **C7** — dispatchable after C6, but split it (see *Cluster validity*).
- **C8** — blocked on B-5 … B-9 plus a user ruling on GEN-18.
- **C9** — blocked on B-2 … B-4.

`phase-1b-engine-prerequisites.md` is **unchanged** between `8a7a49bb8` and HEAD,
so every Task 19 / Task 20 finding below stands verbatim against the current tree.

---

## Already resolved at HEAD (recorded so nobody fixes them twice)

Two findings from this review were independently corrected in `0a246827f` and
`fd19be922` while the review was in progress:

| Was | Now |
|---|---|
| execution.md claimed C8 has zero file overlap with every cluster | Corrected — the C8 ∥ C6 collision on `_services/argv.py` is named, with a pairwise table and the "gate and merge C8 before C6" sequencing constraint |
| execution.md blocked C8 pending the OME-Zarr impact review | Lifted — input images stay ordinary files |

**But the root cause of the first is still live — see B-1a.**

---

## Blockers

### B-1a. Task 16's `Files` block still omits `_services/argv.py`

`phase-1b-engine-prerequisites.md:833-836` lists only `tune/__main__.py`,
`tune/_tune_cli/_run.py`, and the test. The instruction to edit
`_services/argv.py` lives only in the MANDATORY CORRECTIONS section
(`phase-1b:103`).

Verified: `src/phenotypic/gui/tune/_run_argv.py` is a 15-line re-export shim;
`tune_run_tail` is at `_services/argv.py:417` and `tune_run_argv` at `:513`, and
both emit `--slurm-partition`, `--slurm-mem`, `--slurm-time` and a bare `--slurm`
— exactly what Task 16 rewrites.

This matters beyond tidiness. execution.md's corrected text now recommends the
check that caught the overlap: *"parse every task's `Files` block, intersect per
cluster ... Worth re-running whenever a task is added."* **That check will still
miss this overlap**, because the `Files` block it parses does not name the file.
The symptom was fixed; the input that produced it was not.

**Fix:** add `- Modify: src/phenotypic/_services/argv.py` to Task 16's `Files`
block.

### B-2. Task 20's central claim is false — four methods nest, not one, and the partial fix deadlocks

The defect is real as described: `_services/runs.py:316` takes `self._lock`, and
`:330` enters `exclusive_path_lock` while still holding it. But *"the inversion is
local to `allocate`"* is wrong.

| Method | `self._lock` | file lock |
|---|---|---|
| `allocate` | `:316` | `:330` |
| `compare_and_set` | `:444` | `:510` → `_persist_candidate_if_current_locked` `:1363` |
| `publish_if_current_generation` | `:540` | `:544` |
| `observe_local_exit` | `:569` | `:599` → `:1363` |

Three consequences:

1. **`compare_and_set` is the hot path** — every status poll. Fixing only
   `allocate` leaves the stall the task exists to remove.
2. **The partial fix is worse than the bug.** Today all four take
   `self._lock → file lock`: one consistent order, so contention serializes but
   never deadlocks. Invert only `allocate` and thread A holds `file_lock(X)`
   waiting on `self._lock`, while thread B in `compare_and_set` holds `self._lock`
   waiting on `file_lock(X)`. Classic ABBA, bounded only by the 30 s timeout,
   after which `ArtifactLockTimeout` propagates out of `compare_and_set` — a
   method documented at `:441-443` as returning `False` on failure, never raising.
   A stall becomes an exception on a path that previously succeeded. The
   triggering sequence is launch-then-poll on one output dir, i.e. the normal one.
3. **The stated acceptance test passes against the incomplete fix**, because it
   contends `allocate` only. It is a false green by construction — the exact
   failure class the cluster gates exist to catch.

**Fix:** enumerate all four sites; invert all four or none; add an acceptance case
that contends `compare_and_set` specifically. Warn the implementer not to route
`allocate` through `_persist_candidate_if_current_locked` — that helper takes the
file lock itself (`:1363`) and would self-deadlock under an inverted `allocate`.

### B-3. Task 20 misidentifies which lock the `_locked` helpers require

`_persist_record_locked` (`:1324`, docstring *"while holding the lock"*) is called
from `register()` at `:360` and `_commit_mutation_locked` at `:1380` with **only
`self._lock` held and no file lock**. The `_locked` suffix means `self._lock`
throughout this class. Only `_assert_output_claimable_locked` (`:1076`, *"while
ownership lock is held"*) means the file lock.

The task states both assume the file lock. An implementer trusting that will move
`_persist_record_locked` outside `self._lock` and silently break its four callers.

### B-4. `publish_if_current_generation` documents its nesting as an intentional contract

`runs.py:526-528`: *"The registry lock and output owner lock remain held through
`publisher`. Callers should prepare the complete payload first and perform only
the final atomic write inside the callback."*

Inverting that changes a documented invariant with an external callback inside the
critical section. Per execution.md's own gate rule — *"Any finding that conflicts
with a design decision stops the line and comes back to the user"* — this needs a
ruling before C9 dispatches.

### B-5. Task 19: "the parsing is not new" is false; `load_staged_manifest` cannot read a P8 manifest

`_cli_staged_slurm_worker.py:422` does take `--manifest`, but it feeds
`load_staged_manifest` (`_cli/_cli_staged_orchestration.py:238`), which requires
`{"version": 2|3, "images": [...]}` where each entry constructs
`StagedManifestEntry(dataset, image_name, stem, input_path, work_id,
relative_image_path, attempt_id)` (`:56-64`). `StagedManifestEntry(**entry)` raises
on a bare path list.

The P8 artifact is a different thing: spec `05-deploy-and-slurm.md:473` defines it
as `.phenotypic-mcp/plans/<token>.images`, bound by a **content** digest — a plain
image list. The spec hedges (*"Precedent exists and is **probably** reusable"*,
`07-prerequisites.md:460`); the plan hardened the hedge into a fact.

**Fix:** state the `.images` format in Task 19 and write a small dedicated reader.
Drop the reuse claim. The risk is an implementer adopting the staged schema for the
public flag, at which point the spec's `.images` artifact and
`image_manifest_digest` no longer describe what the CLI reads.

### B-6. GEN-18 is an open item escalated to the user, not a settled ruling

`refinery/ledger.md:892` heads the section **"Still open, going to the user"**;
GEN-18 sits at `:894`. Folding it into C8 hands that decision to an implementer.

### B-7. GEN-18's premise needs correcting, and the natural fix double-emits three flags

Three of the four flags **do** have emitters — in `gui/`, not `_services/`:

- `gui/run_console/_slurm.py:177-194` — `_slurm_argv_extension` emits repeated
  `--slurm k=v`
- `gui/run_console/_slurm.py:206-209` — emits `--gpu-slurm` and `--gpu-shards`
- `_build_subprocess_argv` (`:197-210`) composes `to_argv(state)` at `:203` with
  those at `:204-209`

Only `--restart` is genuinely unemittable anywhere, and it has no
`RunConsoleState` field.

Two independent sources say the exclusion is deliberate: `to_argv`'s docstring
(`_services/argv.py:330-332`) — *"`--slurm k=v` pairs are not added here;
SLURM-specific argv extension is the SLURM runner's responsibility"* — and spec
§5.4 (`05-deploy-and-slurm.md:483-484`), which defines `argv_digest` as *"the
rendered argv list ... as produced by `to_argv` **plus the profile's `--slurm`
pairs**"*, i.e. two sources by design.

So inlining them into `to_argv` contradicts both, and makes `_build_subprocess_argv`
emit each flag **twice** on the shipped GUI SLURM path. That duplicate is invisible
in the `_services/argv.py` diff — both halves look correct in isolation — and it
moves `argv_digest`, the consent-carrying value C8 exists to protect.

**Fix:** *promote* `_slurm_argv_extension` and the two GPU blocks into
`_services/argv.py` as a separate composition function, leaving `_slurm.py` calling
the promoted one — the Phase-1a T8 shim idiom. That satisfies GEN-18 ("a `_services`
emitter exists") without touching `to_argv`'s contract or §5.4. Only `--restart`
needs genuinely new behaviour. See *Design call* below for the general form.

### B-8. Task 19 defers a decision that has exactly one safe answer

The task says to *"Decide and state whether `--input` becomes conditional or is
passed alongside the manifest"*, presenting it as a free choice. It is not.

`work_id_for_image` (`_cli/_cli_failure_tracker.py:179-186`):

```python
if config.input_path.is_file():
    relative_path = Path(image_path.name)
else:
    relative_path = image_path.relative_to(config.input_path)
```

Point `--input` at the manifest file and every work ID becomes basename-only —
which **collides across datasets** for identically-named images
(`plate1/img001.tiff` and `plate2/img001.tiff` both reduce to `img001.tiff`) and
diverges from the same images' IDs in a parent-directory run. The C3+merge gate
already established that `EXPECTED_WORK_IDS` and `EXPECTED_INPUT_SHA256S` have
**zero test coverage** (execution.md, PHASE 1a section), so nothing catches it.

**Fix:** pre-decide before dispatch — the manifest is passed *alongside*
`--input <parent>`; `--input` stays required and stays the parent directory — and
add a test that work IDs are identical between a manifest run and a parent run over
the same images.

### B-9. C8's two-file scope is insufficient; resume is the gap

**What genuinely does not need changing** (traced end to end, and this part of the
two-file claim holds up):

- The manifest can substitute at the single scan point `phenotypicCLI.py:1653`.
- `full_dataset_inventory` is derived from `datasets` immediately after at
  `:1741-1744`, so it inherits the subset for free.
- The SLURM array script passes each image individually as `${CURRENT_IMAGE}` with
  `--input-root` for identity (`_cli/_cli_slurm_array_scripts.py:250-265`) rather
  than re-scanning. **The execution strategies and the array script builder need no
  changes.**

**Where it breaks:** `validate_resume_compatibility`
(`_cli/_cli_state_management.py:299-302`) compares `state.input_path !=
config.input_path` and nothing else about the image set. Under B-8's correct
answer, two *different* manifests under the same parent are resume-compatible —
and PhenoTypic auto-resumes by design (CLAUDE.md: *"Run the same command again
after an interruption... there is no `--resume` flag"*). The drift USER-26 was
adopted to prevent returns on the resume path.

**This got sharper at HEAD.** `89b966536` added to §5.4's pre-submit block: *"The
server imports and calls `validate_resume_compatibility` directly rather than
re-enumerating its field list."* So if the manifest does not participate in that
function, the server's own pre-submit drift check is blind to manifest drift — the
guard and the gap are now the same code path.

Answering *"no, the manifest does not participate"* satisfies the task's literal
wording and defeats its purpose. Making it participate needs `ExecutionConfig`
(`_cli/_cli_types.py:95-100`) and `ProcessingState` save/load
(`_cli/_cli_state_management.py:57,156,194`).

**Realistic C8 file set:** `phenotypicCLI.py`, `_services/argv.py`,
`_cli/_cli_types.py`, `_cli/_cli_state_management.py`, `gui/run_console/_slurm.py`.
Still one coherent intent, still one agent — but the plan should say five files,
not two, and execution.md's *"T19's two-file scope stands"* (added in `0a246827f`)
should be corrected along with it.

---

## Major

### M-1. Task 10a's `Interfaces` block is unachievable at 10a and untested

The block promises `PHENOTYPIC_CLASS_MODULES` as *"the single source **both
consumers** read"*, and 10a's scope row says *"both consumers read it. **Zero
behaviour change.**"* Both cannot hold:

- `discover()` (`_services/registry.py:190-231`) consumes 7
  `(module, category, base_class)` triples plus `_discover_analyzers(analysis_module)`.
  It cannot read a bare module-name list until 10b supplies categories and base
  classes.
- If it could, adding the constant's other five entries (`prefab`, `detect.nn`,
  `tune`, `tune.score`, `tune.strategy`) is by definition a behaviour change.
- `test_one_shared_module_list` asserts only membership in the constant, so it
  passes on a pure lift and never checks the second consumer.

**Fix:** 10a = lift the literal, one reader, plus an assertion that the constant
equals the old literal *in order* (resolution is first-match). Move "both
consumers" to 10b with an explicit assertion that `discover()` derives from it.

### M-2. Citation drift — five instances, one dangerous

**The dangerous one:** Task 20 cites *"spins up to 30 s before raising
`FileLockTimeout` (`_cli/_cli_file_locking.py:50`)"*. But `runs.py:72` imports from
`phenotypic.sdk_._file_locking`; the exception is `ArtifactLockTimeout`
(`sdk_/_file_locking.py:17`), the 30 s default is `:25`, the spin loops are
`:69-78`. `_cli/_cli_file_locking.py:50` *does* define a class named
`FileLockTimeout` — so the citation looks right, is wrong, and an acceptance test
written to catch `FileLockTimeout` never fires.

Harmless but worth one pass, since implementers `sed -n` these:

| Task | Cited | Actual |
|---|---|---|
| 20 | `runs.py:317` | `:316` (`def allocate` at `:288`) |
| 10 | `registry.py:198-205` | `:201-207` |
| 16 | `_run.py:798-804` | `:797-805` |
| 19 | `argv.py:326-380` | `to_argv` runs `:326-414` |

Accurate as written: Task 19's `phenotypicCLI.py:922-931` and `argv.py:53-97`.

### M-3. A third "C7" namespace

execution.md flags the C8/C9 collision with `MAIN-MERGE.md`'s SLURM labels but
misses that `README.md:253` uses **C7** for the cross-node JournalStorage
validation job, while execution.md uses C7 for the P1 cluster. Rename one.

### M-4. Stale instruction to C4

execution.md tells C4 to *"Fix the header claim in `phase-1b` as part of C4's
commit"*. That header was already corrected —
`phase-1b-engine-prerequisites.md:18-25` now reads *"These tasks are NOT mutually
independent"*. C4's agent will hunt for a claim that no longer exists.

---

## Design call — should C8 absorb a general fix?

**Yes to one structural fix; no to re-plumbing every flag.**

The named-list approach is already leaking. GEN-18 names three flags plus P8's
fourth. `--gpu-workers-per-gpu` (`phenotypicCLI.py:1005`) is a fifth it never
mentions. `--durable-writes` will be a sixth. Counting the top-level option block
(`phenotypicCLI.py:905-1145`): roughly 30 public options exist and `to_argv` emits
12 of them. The gap is ~18 flags, not four, and enumerating it by hand is how
`--gpu-workers-per-gpu` got missed in a list written specifically to catch this.

But "make `to_argv` emit everything" is the wrong general fix — it contradicts
§5.4's two-source definition of `argv_digest` and double-emits on the GUI path
(B-7).

**Recommended shape, and it stays inside one cluster:**

1. **One composition point in `_services/argv.py`.** Promote
   `_slurm_argv_extension` and the GPU blocks out of `gui/run_console/_slurm.py`
   (B-7), so every argv the server or the GUI renders is composed from `_services`
   functions and nothing renders flags in `gui/`.
2. **Convert the flag list into a tested invariant.** Add a coverage test that
   enumerates the CLI's top-level options and asserts each is either emittable from
   `_services/argv.py` or on an explicit deny-list with a one-line reason. That is
   the same shape as the existing tune annotation-coverage gate and the
   `FEATURES.md` ledger gate — both already precedents in this repo.
3. **Add only what v1 needs to the state object:** the manifest field, and
   `--restart`. Everything else stays deny-listed until a tool actually needs it.

The gate is what makes this durable: `--durable-writes` then fails the coverage
test the day it lands, instead of being discovered by a sixth reviewer. Without it,
C8 fixes a snapshot of a list that keeps growing.

**Scope impact:** this is one extra file (`gui/run_console/_slurm.py`, already
required by B-7) plus one test. It does not push C8 past what one agent holds.

---

## Cluster validity, ordering, gates

**Shapes.** C4, C5, C6, C8, C9 each satisfy the rule — one intent, one reviewable
diff, self-verifiable in a pass. C4 is the largest after B5 grew it; keep it as one
cluster but enforce the plan's own *"do not start 10b/10c until 10a is committed"*,
since 10a is the only part with no behavioural risk.

**C7 — split it.** It spans five files (`tune/_tune_cli/_run.py`, `_worker.py`,
`_optuna_store.py`, `strategy/_optuna.py`, `gui/tune/_callbacks.py`) **and** two
distinct intents: storage **dispatch** (a mechanical five-site idiom, one
reviewable diff) versus storage **failure semantics** (B1's retry predicate,
B2/B3's Monitor safety, B4's compaction — each a behavioural judgment call needing
its own test and its own reasoning). One agent can hold either. Holding both gives
the gate a diff where a mechanical rename and a concurrency-semantics change are
interleaved, which is where reviewer attention degrades.

- **C7a** = five construction sites + backend dispatch
- **C7b** = B1–B4

Both stay after C6.

**Ordering.** `C4 → C6 → C7 → C5` is still correct — C5 needs C4's constant (T14
adds `"phenotypic.subset"` to it), C7 needs C6's `_run.py`. Tasks 19/20 do not
change that. HEAD's added constraint (gate and merge C8 before C6 is dispatched) is
right. C9 ∥ anything is safe: `_services/runs.py` appears in no other cluster's
file set.

**Gates.** Placement is right in principle — cluster tests plus ruff plus mypy
before the reviewer, reviewer on that cluster's diff only. Three additions:

1. **C9's gate must explicitly check the other three nesting sites.** A reviewer
   reading only the `allocate` diff will confirm a fix that is incomplete by
   construction, and the stated acceptance test agrees with them (B-2).
2. **C8's gate must check that no flag is emitted twice** against
   `_build_subprocess_argv` (`gui/run_console/_slurm.py:197-210`). The duplicate is
   invisible in the `_services/argv.py` diff alone (B-7).
3. **C8's gate must check work-ID stability** between a manifest run and a parent
   run (B-8). Nothing in the existing suite covers `EXPECTED_WORK_IDS`.

---

## Verified sound — checked as suspected defects, and they are not

Recorded so these are not re-investigated:

- **T16's "one profile, both engines" works despite two key spellings.**
  `format_sbatch_directives` (`sdk_/slurm/_sbatch.py:133`) strips the `slurm_`
  prefix, so `partition` and `slurm_partition` both reach `--partition`.
  `--slurm slurm_account=exfab` reaches `--account`, which is T16's stated purpose.
- **The 13-entry module order** at `_serializable_pipeline.py:645-660` matches the
  MANDATORY CORRECTIONS exactly, including `detect.nn` in tenth position.
- **`discover()`'s "eight modules"** is accurate: 7 triples (`registry.py:222-231`)
  plus `_discover_analyzers`.
- **Test directory claims** are right: `tests/unit/subset` does not exist (T14
  correctly creates it); `tests/unit/sdk_` does (T13's correction is right).
- **The tune CLI is argparse** (`tune/__main__.py:17,38`), `--slurm` at `:88`;
  `parse_slurm_args` (`_cli/_cli_utils.py:336`) does raise `click.BadParameter`, so
  B3's wrap-into-`parser.error` correction is right.
- **Task 19's remaining claims all verify:** `-i/--input` is a single `click.Path`
  with no `multiple=True` (`phenotypicCLI.py:922-931`); `to_argv` raises when
  `input_dir` is unset (`argv.py:352-359`) and emits `--input` unconditionally
  (`:369-370`); `RunConsoleState` carries `gpu_slurm_args` and `gpu_shards`
  (`:97-98`) that `to_argv` never reads; `advanced_args` is a closed recognised set
  (`:42-44`) whose unknown keys are dropped; `input_path` is compared by literal
  string equality with no normalization (`_cli_state_management.py:299-302`).
- **Task 20's defect is real** as described, notwithstanding B-2/B-3/B-4 about its
  scope and its fix.

---

## What must change before dispatch

| Cluster | Required first |
|---|---|
| **C4** | M-1 (restate 10a's `Interfaces` and its test), M-4 |
| **C5** | nothing |
| **C6** | B-1a (add `_services/argv.py` to Task 16's `Files` block) |
| **C7** | split into C7a / C7b |
| **C8** | user ruling on GEN-18 (B-6); correct the emitter premise and adopt promotion over inlining (B-7); pre-decide manifest-alongside-`--input` (B-8); restate scope as five files incl. resume (B-9); drop the `load_staged_manifest` reuse claim and state the `.images` format (B-5) |
| **C9** | rewrite around all four nesting sites and the ABBA hazard (B-2); fix the `_locked` convention (B-3); user ruling on `publish_if_current_generation`'s documented contract (B-4); fix the module/exception citation (M-2) |
| **execution.md** | C7 name collision (M-3); correct *"T19's two-file scope stands"* per B-9 |
