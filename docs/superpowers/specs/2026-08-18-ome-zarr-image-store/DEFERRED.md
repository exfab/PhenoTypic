# Deferred work — OME-Zarr per-image store

Known weaknesses and follow-up work identified while implementing and verifying
[`design.md`](design.md), **deliberately not fixed in that change**. Each entry
records what is wrong, how it was found, why it was left, and what fixing it
would involve.

Nothing here is a regression introduced by the OME-Zarr change. These are
pre-existing gaps it surfaced, or weaknesses in machinery it touched but did not
own.

---

## D-1. A blocked Stage-2 submission loops controllers indefinitely, silently

**Severity:** medium — wastes shared cluster capacity, and the operator's only
symptom is a job that never appears.

**What happens.** When the staged SLURM engine reaches Stage 2 and the
submission is *rejected by the scheduler* — as opposed to running and failing —
the controller records the attempt as `"status": "blocked"` in
`.phenotypic/progress/slurm_jobs.jsonl`, exits **1**, and its pre-armed
successor starts and does exactly the same thing. This repeats without bound.

Observed 2026-08-21 on run `e2e_slurm_27698143`: **~15 successive controllers
over roughly an hour**, each living ~90 s. Stage 1 had completed correctly
(111/111 stores published); the run simply never advanced.

**Why the existing safety valve does not catch it.** The design terminalises a
stuck run via *"one unchanged retryable-set round is retried; a second unchanged
round terminalizes the remainder and advances to Stage 3"*. That counts rounds
that **ran**. A submission blocked by the scheduler never becomes a round, so
`zero_progress_rounds` stayed at **0** across every cycle and the valve never
engaged. The guard and the failure are on different axes.

**Why it was hard to diagnose.** The controller job logs contain nothing between
`Start Time` and `Exit Code: 1` — no reason, no resource line, no scheduler
message. The `blocked` status is visible only by reading the append-only ledger
directly. From `squeue` the run looks idle; from the output directory it looks
half-finished.

**How it was triggered** (an operator error, but the *loop* is the defect):
`--gpu-slurm` inherits and deltas over the `--slurm` CPU profile, so a
`slurm_constraint=broadwell|cascade|rome|milan|genoa` on the CPU profile is
inherited by Stage 2 unless explicitly overridden. `exfab`'s only node
(`gpu12`) advertises `gpu_latest` and nothing else, so the inherited constraint
is **permanently unsatisfiable** — the correct fix for the run is
`--gpu-slurm "slurm_constraint=gpu_latest"`, which is why the production
`run_phenotypic.sh` carries that flag. It is an override, not a redundancy.

**Suggested fix.** In the controller: treat *N consecutive blocked submissions
of the same token* as terminal (`N = 2` matches the existing round rule), fail
the run with a clear message, and log the scheduler's rejection text plus the
resource block that was requested. A submission that cannot be satisfied should
fail fast and loudly, not spin.

**Worth considering alongside it:** a submit-time preflight that rejects a GPU
profile whose `constraint` no node in the target partition can satisfy. That
turns an hour of silent looping into an error at the moment of submission.

**Not fixed here** because it is Phase 3 staged-controller machinery, unrelated
to storage format, and changing terminalisation semantics deserves its own
change with its own tests.

---

## D-8. A cancelled or crashed staged run permanently blocks its own continuation

**Severity:** high — the output directory of an interrupted run becomes
unusable, and continuation is the headline resume feature of the staged engine.

**What happens.** `--mode full` against an existing output directory exits 1
with:

```
Error: Cannot continue, restart, or overwrite while SLURM jobs are active:
  27699330, 27699335, 27700219, 27700283, 27700336, 27700464, 27700465
```

Every one of those jobs had **already finished** (six `COMPLETED`, one
`FAILED`). None was in the queue. The message names the jobs, so it reads as a
safety guard doing its job — it is not.

**Mechanism**, `_cli_staged_orchestration.py`:

```python
def scheduler_job_is_active(job_id: str) -> bool | None:
    """Return active/inactive, or ``None`` when SLURM cannot answer."""
    ...
    if result.returncode != 0:
        return None            # <- "squeue said no such job" lands here
    ...

def active_ledger_job_ids(output_dir: Path) -> list[str]:
    ...
    if scheduler_job_is_active(job_id) is not False   # <- unknown counts as ACTIVE
```

`squeue --noheader --jobs <finished-id> --format=%T` exits **1** with
`slurm_load_jobs error: Invalid job id specified` — verified. That is the
**normal** result for a job that has left the queue, not a scheduler failure,
but the code conflates the two: `returncode != 0` → `None` → `None is not
False` → **active**.

**Why runs usually work anyway.** A `terminal` ledger row pops the key, so a
job that recorded its own completion is excluded before the scheduler is ever
consulted. The trap is entries left at `submitted`/`recovered` with **no
terminal row** — a run stopped by `scancel`, a node failure, or a controller
killed mid-flight. Those ids never pop, always read as unknown, and block the
directory **forever**.

**Blast radius.** The guard covers *continue, restart, and overwrite* alike, so
there is no in-CLI escape: the run cannot be resumed, restarted, or written
over. Recovery requires hand-editing `.phenotypic/progress/slurm_jobs.jsonl` or
deleting run state, neither of which is documented.

**Suggested fix.** Distinguish "the scheduler says this job does not exist"
from "the scheduler could not be reached":

- Treat `Invalid job id specified` (or `squeue` exit 1 with empty stdout) as
  **not active** — the job is gone.
- Fall back to `sacct -j <id> --format=State` before concluding "unknown";
  `sacct` answers correctly for finished jobs and is the authority for terminal
  state.
- Reserve `None` for a genuine scheduler outage (`FileNotFoundError`, timeout),
  and consider whether even that should block *restart* and *overwrite*, as
  opposed to *continue*.

**Discovered** 2026-08-21 while running Task 7.4 Step 3, attempting to continue
a run whose chain had been cancelled with `scancel`. Cancelling through the CLI
fences the epoch via `deactivate_orchestration` and is likely unaffected — but a
node failure is not a CLI action, and neither is `scancel`, which is what the
`slurm-job` skill tells operators to use.

---

## D-9. `--wait` never returns when a run terminalises incomplete

**Severity:** medium — turns a successful run into a job that is killed at
walltime and reported as `TIMEOUT`.

**What happens.** A staged SLURM run in which **even one image fails
terminally** finishes all of its real work — Stage 3 completes for every other
image, the finalizer runs, and `deliverables/` is fully published — but the
run-level completion marker is never written, because the run's phase is
`terminal_incomplete` rather than complete. `--wait` monitors that marker, so it
blocks forever.

**Observed** 2026-08-22 on run `e2e_slurm_27704648`: 110 of 111 images
processed, 1 failed with a genuine scientific error (`NoObjectsError` — SAM2
detected nothing on that plate). The finalizer job `27705890` finished
`COMPLETED` in 46 s and published `dashboard.html`,
`master_measurements.{csv,parquet}`, `measurements.{csv,parquet}`,
`measurements_by_feature/`, `overlays/`, and `pipeline.json.pht-pipe`;
`master_measurements.parquet` holds **2,345 colony rows across 110 images in 136
columns**. Thirty minutes later, with **zero** phenotypic jobs in the queue,
`--wait` was still blocking and had to be killed.

**Why it matters.** A single unreadable plate in a batch is ordinary, not
exceptional — and per-image isolation exists precisely so one bad image does not
sink a run. But any wrapper that uses `--wait` then burns its entire walltime
and exits `TIMEOUT`, so a run that succeeded for 110 of 111 images is reported
as a failed job. Operators reading `sacct` see a failure; the deliverables on
disk say otherwise.

**Suggested fix.** `--wait` should return when the run reaches **any terminal
phase**, not only `complete`. On `terminal_incomplete` it should exit non-zero
with a summary — how many images succeeded, how many terminalised, and where the
per-image failures are recorded — so the operator gets the distinction between
"nothing worked" and "110 of 111 worked" from the exit path rather than by
reading the event log.

**Related:** the completion marker's absence is *correct*; the aggregate should
not claim completeness when an image is missing. It is `--wait`'s treatment of
that state as "keep waiting" that is wrong.

---

## D-2. `sphinx-build -W` cannot pass: ~651 residual RST warnings

**Severity:** low — cosmetic, but it keeps a real gate switched off.

The docs build had **24,565** warnings and had never been run by any phase of
this project. **23,934 of them (97%) were one bug**, fixed here (`e6877122`):
five class docstrings wrote a Google-style `Attributes:` section whose body
begins with `None`, which napoleon reads as declaring an attribute *named*
`None`, making every autodoc'd docstring that returns `None` an ambiguous
cross-reference.

**Remaining: ~651**, all ordinary RST, in files this change never touched — 372
`Bullet list ends without a blank line`, 51 inline-strong, 51 emphasis, 14
`autosummary.import_cycle`, 8 autodoc import failures, 4 short title
underlines.

Until they are cleared, `-W` cannot be a CI gate, which is why the Phase 6
criterion was amended (user ruling, 2026-08-20) to **no-regression against the
phase base** rather than "succeeds". The build is now legible enough for the
cleanup to be worth doing; before this change it was not.

**Not fixed here** by user direction: out of scope for a storage refactor,
touching many unrelated files.

---

## D-3. `tests/migration/test_equivalence.py` — 57 stale goldens

**Severity:** medium — a real regression detector is switched off.

57 of its 341 tests fail. The failures are **structural, not numeric**: the
goldens expect `LogGrowthModel_r` while the code emits `LogGrowthModel_Area_r`.
Analysis headers became metric-qualified in `67cfa259`, an intentional and
documented convention change; the goldens were last captured at
`e2a91078`/`f3fa28b3`, before it, and were never re-taken.

The suite is the pydantic-v2 migration's regression detector and has **silently
rotted**, because `tests/migration` is not in `testpaths`
(`pyproject.toml:218`) and nothing runs it.

**Fix:** re-capture the goldens with
`scripts/capture_migration_goldens.py` on the capture platform, and add
`tests/migration` to `testpaths` (or a nightly lane) so it cannot rot again.

**Not fixed here** because it belongs to whoever owns the header convention, and
none of the failing modules (denoisers, refiners, analysis, measure) are touched
by this change. The Phase 6/7 gates name the two migration files that *are*
relevant and run those explicitly.

---

## D-4. Orphaned test lanes: `tests/e2e` and `tests/migration`

`testpaths` (`pyproject.toml:218`) is
`["tests/unit", "tests/smoke", "tests/integration", "tests/gui"]`. Both
`tests/e2e` (211 Playwright tests) and `tests/migration` are excluded, so
neither runs in the default lane or in CI.

This is how D-3 went unnoticed. It is also why three `FilFinderDetector` smoke
failures — an absent `topology` extra — were first discovered at the Phase 2
gate rather than years earlier: **a gate that has never run is not a gate.**

**Fix:** give each orphan lane an explicit CI job with its own dependencies
(`tests/e2e` needs a browser; `tests/migration` needs current goldens), rather
than folding them into the default lane where they would be permanently red.

---

## D-5. Stale saved pipelines in project configs

All three `ucr_029_e_d_Maresca` pipeline configs fail to load against current
`phenotypic`: `UCR_029_E_D_Maresca_v3.json` names `StableDenoise`, and
`.pht.json` / `_v2.json` name `LowCircularityRemover`. Neither class exists —
casualties of the verb-first operation rename (`e2a91078`).

`ImagePipeline.from_json` raises
`AttributeError: Class '<name>' not found in phenotypic namespace`.

This is **outside this repo** (project configs, not library code), but it means
a saved pipeline is not reproducible across an operation rename, which is worth
a decision: either a rename-alias table in deserialization, or an explicit
"pipeline written for phenotypic X, this is Y" error naming the rename. The
current message tells a user the class is missing, not that it was renamed.

`UCR_029_E_D_Maresca_v14.json.pht-pipe` loads correctly and is what the Task 7.4
Step 3 verification uses.

---

## D-6. Two small API surprises found while scripting a real run

Neither is a defect, both cost time and are cheap to smooth:

- **`ImagePipeline.to_json(filepath)` returned the JSON string without writing
  the file.** The docstring says *"Optional path to save the JSON. If None,
  returns JSON string."* Worth verifying the write path, or making the
  documented behaviour explicit.
- **`GridImage(path)` raises** `ValueError: Input must be a NumPy array, Image
  instance. Got <class 'str'>`. The constructor takes an array; reading from a
  path is `GridImage.imread(path)`. The error is accurate but does not name the
  method the caller wants.

---

## D-7. `--image-manifest` is documented but absent on this branch

Root `CLAUDE.md` documents `--image-manifest <file>` for processing an approved
image subset. It does not exist on `worktree-ome-zarr-image-store` — it lands in
`9e6b4d2e` / `4de5b751` on another line of development.

Not a defect in either branch; noted because the documentation and the code are
reachable from the same checkout and disagree, which cost a submitted job. When
the branches converge, verify the flag and its docs land together.
