# Phase E adherence review: cluster checks, output location, GUI Validate

- **Reviewer:** independent reviewer (Claude), analysis only. I did not write this code.
- **Commit reviewed:** `origin/claude/modest-mccarthy-jz0ylw` at `26bf6e3`, checked out
  detached. Scope is Phase E: `8738e39` (Tasks 13 and 14) and `5177ae5` (Task 15). The later
  commits `f508094`, `af1ef98`, `aaa96e2`, `d3bd6ee` and `26bf6e3` were diffed against
  `8738e39`; none changes a Phase E function (`git diff 8738e39 HEAD --
  src/phenotypic/_cli/_cli_preflight.py src/phenotypic/phenotypicCLI.py` touches only the
  module, license, bit-depth, metadata and preload code of earlier phases).
- **Read:** spec `docs/superpowers/specs/2026-09-24-cli-preflight/design.md` (§0, §6, §9,
  §11, §12, §13, decisions D6, D9, D13, D14, open question 3, the R and D dispositions),
  plan Tasks 13 to 15 and the review-gate section, `slurm-behavior.md`,
  `slurm_behavior_probe.sh`, `phase-gates.md`, `phase-c-adherence.md`,
  `phase-d-adherence.md`, root `CLAUDE.md`, `src/phenotypic/_cli/CLAUDE.md`.
- **Slurm evidence.** There is no Slurm client here, so no claim below was observed on a
  cluster. Where a claim rests on Slurm's behavior I read Slurm's own source and manual
  pages at `SchedMD/slurm` `master`, fetched 2026-09-25 into the session scratchpad:
  `src/slurmctld/job_scheduler.c`, `job_mgr.c`, `node_scheduler.c`, `proc_req.c`,
  `src/sbatch/sbatch.c`, `src/sinfo/sinfo.c`, `src/sinfo/opts.c`,
  `src/common/slurm_errno.c`, `doc/man/man1/sbatch.1`, `doc/man/man5/slurm.conf.5`. Line
  numbers cited as `slurm:<file>:<line>` refer to that snapshot. The installed Slurm on a
  target cluster may be older; the behaviors cited have been stable for many releases, but
  that is inferred, not verified.
- **Method:** code reading against the spec; probes that drive the real check functions
  with a scripted scheduler (`verify_phase_e.py`, `verify_process_gpu.py` in the session
  scratchpad, not in the repository); 36 revert proofs, each applied by a scratch runner
  that restored the original bytes, followed by `git checkout -- <file>` for every touched
  file and an empty `git status --short`.
- **Test commands:** `QT_QPA_PLATFORM=offscreen uv run pytest <files> -q --no-header -o
  addopts= -m "not slow" -p no:randomly` with at most `-n 4`. The full suite was not run.
  The process runs as root (`id -u` is 0), which matters for one test (E10).

## Verdict

**Changes required.** The Phase E checks are implemented largely as §6 and §9 describe,
every scheduler call is read-only and bounded by a timeout, nothing is submitted by either
the CLI preflight or the GUI Validate, the entry points stay lazy, and most hunks are
pinned by a test that fails when the hunk is reverted. However, the preflight can refuse
runs that would succeed, which violates the rule that governs the whole change (§0,
"Severity follows reach"; D3):

- E1 (Blocking): `PF-SBATCH-REJECTED` treats every nonzero `sbatch --test-only` exit as a
  configuration rejection. Slurm's will-run test excludes DOWN and DRAINED nodes, so a
  valid job for a partition whose nodes are drained (maintenance, a small private or GPU
  partition) fails `--test-only` although the real submission is accepted and queues.
  Transient controller errors outside the four listed patterns are also errors.
- E2 (Major): for a GPU pipeline in `--mode process` on SLURM, the preflight tests the
  CPU profile without `--gpus-per-node`, while `AutonomousSLURMStrategy` submits with it.
  Any site policy that requires a GPU request on a GPU partition rejects the test script
  and not the real one.
- E3 (Major): the partition-time and GPU-partition checks read only the
  `slurm_partition` key. The GUI (the path Task 15 adds) and the GUI tutorial spell it
  `partition`. The GPU check then never runs for GUI runs, the time check compares against
  the wrong partition, and a mixed spelling produces a verified false refusal.
- E4 (Major, inherited): `PF-SLURM-LIMIT` is driven by the smallest
  `MaxSubmitJobsPerUser` over every QoS on the cluster, including QoS the user never
  submits under.

Nine revert proofs survive (see the table), one of them only because the test that pins
`PF-OUTPUT-UNWRITABLE` skips as root.

## Findings

### Blocking

#### E1. `sbatch --test-only` failures that are not rejections refuse the run

**Evidence.**

- `check_slurm_profiles` classifies a nonzero exit as `PF-SBATCH-REJECTED`, severity
  `error`, unless stderr contains one of four fragments
  (`src/phenotypic/_cli/_cli_preflight.py:591-596`, `:713-722`).
- What `--test-only` computes (verified by reading Slurm source):
  - `sbatch --test-only` calls `slurm_job_will_run` and exits 1 on any error, printing
    `allocation failure: <strerror>` (`slurm:sbatch.c:283-294`).
  - The controller runs `job_allocate(..., will_run=true, ...)`, which returns
    `job_start_data(...)` (`slurm:proc_req.c:2841-2846`, `slurm:job_mgr.c:4385-4392`).
  - `job_start_data` restricts the candidate nodes to those that are neither DOWN nor
    DRAINED: `/* Only consider nodes that are not DOWN or DRAINED */ bit_and(avail_bitmap,
    avail_node_bitmap);` (`slurm:job_scheduler.c:4632-4633`). If the select test then
    fails, the result is `ESLURM_REQUESTED_NODE_CONFIG_UNAVAILABLE`
    (`slurm:job_scheduler.c:4721-4723`), whose text is "Requested node configuration is
    not available" (`slurm:slurm_errno.c:277`).
  - A real submission of the same job evaluates *all* nodes of the partition for
    "runnable ever" and only the available ones for "runnable now"; nodes that exist but
    are down give `ESLURM_NODE_NOT_AVAIL` (`slurm:node_scheduler.c:2114-2140`), which
    `job_allocate` lists as "Non-fatal error, but job can't be scheduled right now": the
    job stays queued (`slurm:job_mgr.c:4485-4523`). The same list makes
    `ESLURM_REQUESTED_PART_CONFIG_UNAVAILABLE` non-fatal when `EnforcePartLimits=NO`
    (`:4482-4492`), while will-run returns it as a failure (`slurm:job_scheduler.c:4578-4581`,
    `:4597-4598`).
  - For a real submission, `sbatch` retries on a full job queue (`ESLURM_MAX_JOB_COUNT`),
    busy nodes or ports, and `EAGAIN` (`slurm:sbatch.c:296-317`). The `--test-only` branch
    has no retry and exits 1.
- Verified by probe (`verify_phase_e.py`, P6): each of these stderr lines yields
  `PF-SBATCH-REJECTED error`: "Requested node configuration is not available",
  "Zero Bytes were transmitted or received", "Communication connection failure",
  "Invalid authentication credential", "MaxJobCount limit reached", "Resource temporarily
  unavailable", "Requested partition configuration not available now". The first is the
  drained-partition case above; the next three are controller or authentication faults
  (`slurm:slurm_errno.c:84`, `:1200`, `:1207`); the last three are conditions the real
  submission retries on or queues through.
- The spec's own framing assumed that anything other than a clear rejection is a warning
  (§6; `slurm-behavior.md` "every `sbatch --test-only` outcome that is not a clear
  rejection ... is the `PF-SBATCH-UNAVAILABLE` warning"). The code inverts that: it allows
  four known-transient strings and calls everything else a rejection. The probe script
  has no case that would expose this (no drained-node case).

**Impact.** During a maintenance drain, or whenever every node of a small partition (a
lab's private partition, a GPU partition with a few nodes) is drained or down, every
`--slurm` run to that partition, CLI or GUI Validate, is refused with exit 1, although
submitting it would queue it to start after the maintenance. That is the common "submit
before the maintenance window" workflow. Transient controller or `munge` faults also
refuse the run, contrary to §6's "a transient controller fault says nothing about the
configuration". The only escape is `--skip-validation`, which also drops every other
check.

**Fix.** Classify by message, not by exclusion. Keep `PF-SBATCH-REJECTED` as an error only
for messages that name a configuration fault and that a real submission also rejects
(invalid partition, account, QoS, `--time`, node count, unrecognized option, and the job
submit plugin's own message). Treat "Requested node configuration is not available" and
"Requested partition configuration not available now" as a warning that says the
request cannot start on the currently available nodes and may be unsatisfiable. Add
"Zero Bytes were transmitted or received", "Communication connection failure",
"Protocol authentication error", "Invalid authentication credential", "MaxJobCount limit
reached" and "Resource temporarily unavailable" to `SBATCH_COMMUNICATION_PATTERNS`. Add a
drained-partition case to `slurm_behavior_probe.sh` (for example, `--nodelist=<a drained
node>`) and a unit test for each new classification.

### Major

#### E2. A process-mode GPU run on SLURM is tested with a profile it does not submit

**Evidence.**

- `_slurm_profiles` returns the CPU profile, plus the GPU profile only when
  `_staged_slurm_run` holds, which requires `mode == "full"`
  (`_cli_preflight.py:626-652`).
- On SLURM, `uses_staged_gpu_strategy` is false whenever `process_only_layer` is set
  (`_cli_execution_strategies.py:1335-1336`), so a GPU pipeline in `--mode process` runs
  through `AutonomousSLURMStrategy`, which adds `slurm_gpus_per_node=1` to the profile it
  submits (`_cli_execution_strategies.py:906-913`) and runs its own GRES check (`:918-930`).
- Verified by probe (`verify_process_gpu.py`): for `--layer objmap` and `--layer gray` with
  `FakeGpuDetector`, one profile is tested, it carries no `--gpus-per-node`, and
  `check_gpu_partition` returns nothing.

**Impact.** Two ways. (a) A false refusal where a site's job submit plugin rejects jobs on
a GPU partition that request no GPU (a common site policy; inferred, not verified on a
cluster): the test script is rejected, the real script would be accepted. (b) The check
misses exactly the rejections it exists to catch for this run (an unsatisfiable GPU
request on that partition), and `PF-GPU-PARTITION` is skipped, so the strategy's check
fires only after state is written.

**Fix.** Derive the profiles from the same predicate the strategy uses: when the run is
SLURM, not `measure`, and `pipeline_requires_gpu` holds but the staged path is not taken,
test the CPU profile with the strategy's `slurm_gpus_per_node` default applied, and run
`check_gpu_partition` on it. Best is one function, shared with
`AutonomousSLURMStrategy`, that returns the profile it will submit.

#### E3. Partition checks read only `slurm_partition`; the GUI sends `partition`

**Evidence.**

- `check_partition_time` takes partition names from `profile.get("slurm_partition")` only
  (`_cli_preflight.py:801`); `check_gpu_partition` does the same (`:876`), as does the
  strategy's GRES check (`_cli_execution_strategies.py:918`).
- `format_sbatch_directives` strips `slurm_` from every key
  (`sdk_/slurm/_sbatch.py:135`), so `partition=x` and `slurm_partition=x` both render
  `#SBATCH --partition=x`, and `parse_slurm_args` keeps keys as typed
  (`_cli/_cli_utils.py:358-373`).
- The GUI emits `--slurm partition=...`, `time=...`, `mem=...`, `gpus=...`
  (`_gui/run_console/_slurm.py:50-56`, `:178-194`), and the GUI tutorial teaches that
  spelling (`docs/source/tutorials/gui/05_run_slurm.md:7`, `:18-23`), while the CLI docs
  teach `slurm_partition=` (`docs/source/how_to/pages/gpu_detection_setup.md:601`).
- `resolve_stage_slurm_args` merges the two profiles by key (`_cli_staged_slurm.py:129`),
  so mixed spellings keep both keys.
- Verified by probe (`verify_phase_e.py`):
  - P1: `--slurm slurm_partition=batch --gpu-slurm partition=gpu` renders
    `#SBATCH --partition=batch` followed by `#SBATCH --partition=gpu`; `sbatch` applies
    `#SBATCH` lines in order, so the job goes to `gpu` (inferred from `sbatch.1:19-23` and
    option handling, not observed). `check_gpu_partition` asks about `batch`, gets
    `(null)`, and returns `PF-GPU-PARTITION error`: a false refusal.
  - P2: with the GUI spelling `partition=gpu-short time=1-00:00:00`, the only scheduler
    call is `scontrol show partition` (the default-partition lookup), so the 24-hour
    request is compared against the default partition's 7-day limit and no finding is
    produced, although `gpu-short` allows 4 hours and `EnforcePartLimits=ALL` would
    reject the job. With `partition=cpu-only` on a staged GPU run, `check_gpu_partition`
    makes no call and returns nothing.

**Impact.** Task 15's purpose (Validate checks what Run submits) is only partly met: for
every GUI run the GPU-partition check is skipped and the time check reads the wrong
partition. A CLI user who mixes the two spellings, which the two sets of docs invite, is
refused a valid staged GPU run.

**Fix.** Resolve the effective partition the way `sbatch` will: the last of
`slurm_partition` and `partition` in the rendered directive order (read it back from
`format_sbatch_directives` output, which is the one formatter). Apply the same resolution
for time (`slurm_time`, `time`) and GPUs (`slurm_gpus_per_node`, `gpus_per_node`,
`gpus`, `gres`). Add GUI-spelling and mixed-spelling tests.

#### E4. `PF-SLURM-LIMIT` inherits a cluster-wide minimum `MaxSubmitJobs`

**Evidence.**

- `check_staged_slurm_limits` refuses when `staged_slurm_limit_errors` returns a message
  (`_cli_preflight.py:843-859`), fed by `get_slurm_max_submit_jobs()`.
- That function runs `sacctmgr show qos format=MaxSubmitJobsPerUser -P`, which lists
  every QoS on the cluster, and returns the minimum positive value together with the
  user's association limits (`sdk_/slurm/_config.py:88-125`). Its docstring calls this
  conservative because the job's QoS cannot be inferred.
- A cluster with, for example, a `debug` QoS limited to 2 submitted jobs per user
  therefore yields `max_submit = 2`, and every staged GPU run is refused with "SLURM
  MaxSubmitJobs must be at least 3" (`_cli_staged_slurm.py:96-100`), whatever QoS the
  user's jobs actually run under; a limit of 4 caps `--gpu-shards` at 2. (Inferred from
  the code and from `sacctmgr` listing all QoS; not observed on a cluster.)
- This is pre-existing: `StagedSlurmStrategy` refused the same runs before Phase E
  (`git show 8738e39 -- src/phenotypic/_cli/_cli_staged_slurm.py`). Phase E moves the
  refusal earlier and gives it a finding code.

**Impact.** Not a new refusal of a run the product would otherwise complete, since the
strategy refuses it later, but the preflight now reports a wrong reason as an
authoritative error, and "conservative" is the wrong direction for a value used to
refuse. The task brief asks specifically about wrong `MaxSubmitJobs` reads.

**Fix.** Resolve the QoS the job will use (the profile's `qos`/`slurm_qos`, else the
association's `DefaultQOS`: `sacctmgr show assoc where user=$USER format=DefaultQOS,QOS
-P`) and read only that QoS's `MaxSubmitJobsPerUser`. Where the QoS cannot be resolved,
make `PF-SLURM-LIMIT` a warning and let the strategy decide.

### Minor

#### E5. The GRES check reads a 20-character, truncated `sinfo` field

**Evidence.** `partition_gres_error` runs `sinfo -p <p> --Format=gres --noheader` and
tests `"gpu" in stdout` (`sdk_/slurm/_config.py:312-326`). `sinfo --Format` fields without
an explicit size are truncated to 20 characters (`slurm:opts.c:903-924`, `*field_size =
20`; `sinfo.1:376-381`). Verified by probe (P5): a node whose GRES string begins
`shard:a100:64(S:0-1),gpu:a100:4(S:0-1)` prints as `shard:a100:64(S:0-1)` and the GPU
stage is refused with "has no GPUs". Whether sites order GRES that way is inferred. The
same read existed before in `AutonomousSLURMStrategy`; Phase E extends it to the staged
path as an error.

**Fix.** Request an untruncated field (`--Format=gres:0`, or `-o %G`, whose default width
is not truncating), and match a `gpu` GRES token rather than a substring.

#### E6. `sinfo` exits 0 for an unknown partition, so the F18 fix is not reached

**Evidence.** `sinfo` loads every partition and filters locally; there is no error path
for a name that matches nothing (`slurm:sinfo.c:350-460`, `slurm:opts.c:257-264`), so
`sinfo -p does-not-exist` exits 0 with empty output (inferred from source; probe case 6
would confirm). `partition_gres_error` then reports "partition 'nope' has no GPUs (sinfo
gres: '')" (verified by probe P4), which is the misreport F18 names. The unit test
`test_an_unknown_partition_is_not_reported_as_having_no_gpus` scripts a nonzero exit and
an "Invalid partition name" stderr that `sinfo` does not produce
(`tests/unit/cli/test_preflight_slurm_checks.py:231-240`). The refusal itself is
correct, because such a run fails at submission, so this is a wrong message rather than a
false refusal. The same empty output could also come from a site that hides node
information from users (`PrivateData=nodes`, `slurm.conf.5:3677-3679`), where an error
would be a false refusal (inferred).

**Fix.** Treat empty output as "sinfo lists no partition named ..." and, when `sbatch
--test-only` has already accepted the profile, as a warning. Replace the test's scripted
nonzero exit with the empty-output shape and cite the probe once it runs.

#### E7. Partition-time wording and partition-list semantics

**Evidence.**

- The tightest `MaxTime` of a partition list is always used (`_cli_preflight.py:819`).
  Slurm documents that under `EnforcePartLimits=ANY` a job is accepted if it satisfies
  any one partition, and under `NO` a job remains queued only when it exceeds the limits
  of *all* requested partitions (`slurm.conf.5:1511-1531`, verified). Only `ALL` matches
  the tightest rule. Probe P3: `slurm_partition=a,b` with 24 hours, `a` 2 hours, `b` 3
  days, `EnforcePartLimits=ANY` warns that the job "would be be rejected at submission",
  although it is accepted and runs in `b`.
- The message is ungrammatical for two of three outcomes: "would be be rejected" and
  "would be pend or be rejected" (`_cli_preflight.py:824-831`, verified by P3).
  `test_enforced_limits_change_the_wording` checks only the substring `"rejected"`.
- With no partition named, the default partition is assumed (`:807-808`), but a
  submission honors `SBATCH_PARTITION` from the environment the preflight itself passes
  to `sbatch`, and a site job submit plugin may route by time or resources (inferred).
- Verified as correct: a QoS with `Flags=PartitionTimeLimit` does override the partition
  `MaxTime` (`slurm:job_mgr.c:6547-6556`), so keeping the finding a warning is right, and
  `EnforcePartLimits` defaults to `NO` (`slurm.conf.5:1530`).

**Fix.** Use the loosest limit unless `EnforcePartLimits=ALL`; honor `SBATCH_PARTITION`;
fix the grammar (`outcome` values should not repeat "be") and assert the full sentence in
the test.

#### E8. `PF-NODE-LOCAL` misclassifies some mounts

**Evidence.** Verified by probe (P8) against `_filesystem_type`
(`_cli_preflight.py:1306-1319`):

- Over-mounts: when one mount point appears twice, the kernel resolves to the later
  entry, but `depth > best[0]` keeps the first. `/dev/sdb1 /scratch xfs` followed by
  `gpfs0 /scratch gpfs` classifies `/scratch/run` as `xfs` (a false warning).
- Only `\040` is unescaped (`:1314`); `/proc/self/mounts` also octal-escapes tab
  (`\011`), newline (`\012`) and backslash (`\134`). A mount at `/data<TAB>tab` falls
  through to the root filesystem's type.
- A login node that is also the NFS server sees its exported `/home` as `xfs` or `ext4`,
  while every compute node sees `nfs`; the warning then names storage that workers can
  see (inferred; common on small clusters).
- `overlay`, `ramfs`, `devtmpfs`, `ext2` and `ext3` are node-local in the code
  (`:1292-1294`), beyond §9's list. `overlay` is also the root of a container image, where
  a path baked into the image is visible to workers running the same image. This is a
  silent deviation from §9.

All of these are warnings, so none refuses a run.

**Fix.** Let a later entry at equal depth win (`>=`), decode every octal escape
(`re.sub(r"\\(\d{3})", lambda m: chr(int(m.group(1), 8)), field)`), record the
`overlay` addition in §9, and say in the hint that a login node which exports the path
over NFS can trigger the warning.

#### E9. The dry-run preview shows only the CPU profile

**Evidence.** `_display_slurm_config(config.slurm_args)` is the only caller
(`_cli/_cli_interactive.py:194`). For a staged GPU run the GPU profile, which is the one
with the automatic `--gpus-per-node=1`, is never previewed, nor is the GPU request
`AutonomousSLURMStrategy` adds (E2). F26 asks that "the preview is the text that will be
submitted"; for GPU runs it is not.

**Fix.** Preview every profile `_slurm_profiles` returns (after E2's fix), with labels.

#### E10. Test gaps behind the surviving mutations

**Evidence.** See the revert-proof table. In summary:

- M9: removing the `Default=YES` filter survives, because in
  `test_no_partition_uses_the_default_partition` the default partition is also the
  tightest one.
- M20: removing the `os.access` check survives here because
  `test_an_unwritable_ancestor_is_an_error` skips as root. A scratch probe that patches
  `os.access` fails under the same revert, so the logic is right; but any root CI or
  container run leaves `PF-OUTPUT-UNWRITABLE` unpinned.
- M13 and M23b: the `<=` versus `<` boundaries of the time and space comparisons are not
  pinned.
- M28b: dropping `overlay` from the node-local set survives (see E8 on whether it
  belongs).
- M26b: letting the later equal-depth mount win survives, and is the fix E8 recommends.
- M18: dropping the canonicalization of `--gpu-slurm` time survives; harmless, because
  `format_sbatch_directives` re-parses the value.
- The e2e fake scheduler answers every preflight command permissively
  (`tests/e2e/gui/test_run_console_fake_slurm.py`, the `sinfo` and `scontrol` branches
  added in `5177ae5`), so no browser test covers a Validate that reports a cluster
  finding.

**Fix.** Make the default-partition test's default partition the looser one; replace the
root skip with a patched `os.access` (as the probe does) or add such a test beside it;
add boundary cases; add a Validate test whose fake `sbatch --test-only` exits 1.

### Nit

#### E11. The Slurm behavior record can be partly closed from the documentation

`slurm-behavior.md` says open question 3 stays open because Slurm's documentation was
unreachable. Three of its points are settled by the manual pages: `--test-only`
"Validate[s] the batch script ... No job is actually submitted" (`sbatch.1:2491-2494`);
`sbatch` reads the script from standard input when no file is named (`sbatch.1:15-17`);
`EnforcePartLimits` defaults to `NO` (`slurm.conf.5:1525-1530`). The probe script should
gain the drained-node case (E1), a long-GRES case (E5), and an `sinfo` exit-status case
that prints stdout length (E6). Cluster confirmation is still needed for the installed
version.

#### E12. SLURM-mode Validate with an empty SLURM form validates a local run

With `state.mode == "slurm"` and no SLURM fields, `_build_subprocess_argv` adds no
`--slurm`, so the CLI dry run is local (`is_slurm_mode` is false, `_cli_types.py:223-227`)
and Validate succeeds, while Run refuses "SLURM submission requires a non-empty SLURM
configuration" (`_gui/run_console/_slurm.py:770-773`). Validate could refuse the same way.

#### E13. Crash isolation and default runner

`check_output_location` computes writability, space and node-local findings in one
function (`_cli_preflight.py:1329-1411`); an exception in the mount code (for example a
symlink loop in `Path.resolve`) discards an already computed `PF-OUTPUT-UNWRITABLE` error
and reports only `PF-CHECK-CRASHED`. `partition_gres_error`'s default runner catches
`FileNotFoundError` and `TimeoutExpired` but not other `OSError`
(`sdk_/slurm/_config.py:306-316`), which the strategy does not catch either. Split the
output check into three registered checks; catch `OSError` in the runner.

## Revert-proof table

Runner: a scratch script applies one exact-string mutation, runs the listed tests with
`uv run pytest <tests> -q --no-header -o addopts= -m "not slow" -p no:randomly -p
no:cacheprovider` (M17: `tests/unit/cli -n 4 -k "staged or slurm"`), and restores the
original bytes. Afterwards `git checkout -- src/phenotypic/_cli/_cli_preflight.py
src/phenotypic/sdk_/slurm/_config.py src/phenotypic/phenotypicCLI.py
src/phenotypic/_cli/_cli_interactive.py src/phenotypic/_cli/_cli_staged_slurm.py
src/phenotypic/_gui/run_console/_callbacks.py` was run and `git status --short` was
empty. Baseline of the Phase E files: 45 passed, 1 skipped.

| # | Hunk reverted or mutated | Test file | Result | Failing test(s) |
|---|---|---|---|---|
| M1 | `check_slurm_profiles` tests no profile | slurm checks | 8 failed | script, rejection, 4 controller-fault cases, staged, timeout |
| M2 | no `#!/bin/bash` line | slurm checks | 1 failed | `test_the_test_script_is_well_formed_and_uses_the_submission_environment` |
| M3 | `env=None` instead of `sbatch_submission_environment()` | slurm checks | 1 failed | same |
| M4 | communication patterns ignored | slurm checks | 4 failed | `test_a_controller_fault_is_only_a_warning[*]` |
| M5 | `TimeoutExpired` not caught | slurm checks | 1 failed | `test_missing_sbatch_and_a_timeout_are_warnings` |
| M6 | GPU profile not tested | slurm checks | 1 failed | `test_a_staged_gpu_run_tests_both_profiles` |
| M7 | rejection severity `warning` | slurm checks | 1 failed | `test_a_rejected_profile_is_an_error_carrying_stderr` |
| M8 | tightest limit becomes loosest | slurm checks | 1 failed | `test_a_partition_list_uses_the_tightest_limit` |
| M9 | `Default=YES` filter removed | slurm checks | **survived** | none (E10) |
| M10 | wording ignores `EnforcePartLimits` | slurm checks | 1 failed | `test_enforced_limits_change_the_wording` |
| M11 | time finding severity `error` | slurm checks | 1 failed | `test_time_over_the_partition_limit_warns_with_the_live_setting` |
| M12 | days dropped from durations | slurm checks | 2 failed | `test_slurm_durations_parse[2-00:00:00]`, `[1-12:00:00]` |
| M13 | `minutes <= limit` becomes `<` | slurm checks | **survived** | none |
| M14 | staged limits not reported | slurm checks | 1 failed | `test_the_staged_limits_are_checked_before_submission` |
| M15 | GPU partition finding dropped | slurm checks | 2 failed | unknown-partition and CPU-partition tests |
| M16 | `sinfo` return code ignored | slurm checks | 1 failed | `test_an_unknown_partition_is_not_reported_as_having_no_gpus` |
| M17 | strategy ignores `staged_slurm_limit_errors` | `tests/unit/cli -k "staged or slurm"` | 1 failed (1914 passed) | `test_staged_slurm_scripts.py::test_strategy_rejects_max_submit_limit_below_three` |
| M18 | `--gpu-slurm` time not canonicalized | slurm checks | **survived** | none (harmless) |
| M18b | `--gpu-slurm` startup parse removed | slurm checks | 1 failed | `test_gpu_slurm_time_is_parsed_at_startup_even_with_skip_validation` |
| M19 | old hand-written preview restored | slurm checks | 1 failed | `test_the_dry_run_preview_prints_the_real_directives` |
| M20 | `os.access` check removed | output location | **survived as root** | test skips; scratch probe with patched `os.access`: 1 passed before, 1 failed after |
| M21 | nearest existing ancestor not used | output location | 5 failed | writable-new-output and space tests |
| M22 | space check in every mode | output location | 2 failed | `test_the_space_heuristic_runs_in_full_mode_only[process]`, `[measure]` |
| M23 | space never warns | output location | 1 failed | `test_less_space_than_the_inputs_warns_as_a_heuristic` |
| M23b | `free < total` becomes `<=` | output location | **survived** | none |
| M24 | node-local check on local runs | output location | 6 failed | incl. `test_a_local_run_does_not_check_mounts` |
| M25 | `JoinMetadata` tables skipped | output location | 1 failed | `test_a_join_metadata_table_on_node_local_storage_is_named` |
| M26 | shortest mount wins | output location | 8 failed | shared-filesystem and node-local cases |
| M26b | later equal-depth mount wins | output location | **survived** | none (the E8 fix) |
| M27 | `\040` not unescaped | output location | 2 failed | space-path case, `test_the_longest_mount_wins` |
| M28 | `xfs` not node-local | output location | 2 failed | `xfs` cases |
| M28b | `overlay` not node-local | output location | **survived** | none |
| M28c | `--input` not checked | output location | 2 failed | input cases |
| M28d | `--metadata` not checked | output location | 1 failed | metadata case |
| M29 | Validate uses the local argv | GUI callbacks | 1 failed | `test_slurm_validate_forwards_the_run_cluster_options` |
| M30 | Validate always uses the SLURM argv | GUI callbacks | 1 failed | `test_local_validate_carries_no_cluster_options` |

Test file abbreviations: "slurm checks" is `tests/unit/cli/test_preflight_slurm_checks.py`;
"output location" is `tests/unit/cli/test_preflight_output_location.py`; "GUI callbacks"
is the two Validate tests in `tests/integration/gui/test_run_console_callbacks.py`.

## Regression spot checks

- Phase E files plus the two GUI Validate tests: **45 passed, 1 skipped** (the skip is the
  root-only writability test).
- `tests/unit/cli -k "slurm or staged or interactive or preflight or dry_run"`,
  `tests/unit/sdk_/test_slurm_time.py`, `tests/unit/gui/run_console` (`-n 4`): **2236
  passed, 2 skipped**.
- `tests/unit/ci/test_startup_imports.py`, `tests/unit/ci/test_deferred_imports.py`,
  `tests/unit/test_docs_preflight_codes.py`, `tests/unit/cli/test_cli_preflight_core.py`
  (`-n 4`): **253 passed**. The new code keeps every import inside the function that uses
  it (`_cli_preflight.py:601`, `:618`, `:662`, `:694-696`, `:790`, `:838`, `:1339`, `:1382`;
  `_cli_interactive.py:63-65`).
- `uv run ruff check` on the seven changed source and test files: all checks passed.
- The e2e fake-SLURM file was not run (no browser here); `phase-gates.md` records 20
  passed at the Phase E gate.

## Verified as correct

- **Nothing is submitted.** The preflight runs only `sbatch --test-only`, `scontrol show
  config`, `scontrol show partition`, `sinfo` and (through the existing getters)
  `sacctmgr show` (`_cli_preflight.py:606-623`, `:700-705`, `:770`, `:803-806`, `:836-840`,
  `:880-883`). `--test-only` exits before the submission loop (`slurm:sbatch.c:283-294`)
  and the controller purges the will-run job record before any scheduling or accounting
  of a submission (`slurm:job_mgr.c:4385-4392`), so no job is queued and no submit limit
  is consumed. No scheduler sidecar job and no parallel job is introduced, so the array
  trigger routing rule in `_cli/CLAUDE.md` is untouched.
- **GUI Validate submits nothing.** It still starts `python -m phenotypic ... --dry-run`
  through `LocalRunner.start` (`_gui/run_console/_callbacks.py:1965-1993`), now with
  `_build_subprocess_argv` in SLURM mode (`:320-337`), the same builder `submit_slurm`
  uses (`_slurm.py:779`), and with the same inherited environment. The CLI exits for
  `--dry-run` at `phenotypicCLI.py:2731-2739`, before `create_execution_strategy`
  (`:3394`).
- **Script and environment parity (spec §6, R15).** `render_test_script` uses the one
  formatter, adds the shebang, logs to `/dev/null`, and is piped on stdin with
  `env=sbatch_submission_environment()` and a 30 s timeout (`_cli_preflight.py:655-705`),
  exactly as `submit_script` passes its environment (`sdk_/slurm/_sbatch.py:228-235`).
  The real submission adds only `--parsable`, `--export=ALL`, `--dependency` and
  `--array` on the command line and `--array` in the script, none of which makes the test
  stricter than the submission.
- **Time handling.** `EnforcePartLimits` is read live and a QoS override is respected by
  keeping the finding a warning (D13; `slurm:job_mgr.c:6547-6556`).
  `parse_slurm_duration_minutes` accepts every form `scontrol` prints (`UNLIMITED`,
  `D-HH:MM:SS`, `HH:MM:SS`) and the other `sbatch` forms (verified by probe P7: `1-0`,
  `2-12`, `1-00:30`, `90`, `5:00`).
- **`--gpu-slurm` time at startup (F16).** Parsed and canonicalized before
  `ExecutionConfig` and outside `--skip-validation` (`phenotypicCLI.py:2156-2178`), pinned
  by M18b. Note, not a finding of this phase: `parse_slurm_time` refuses Slurm forms such
  as `1-0`, `5:00` and `1:00:00` for both `--slurm` and `--gpu-slurm`
  (`sdk_/slurm/_sbatch.py:22-25`); before Phase E the GPU value failed the same way at
  script rendering, so no run that used to succeed is now refused.
- **Shared limit function (F18).** `staged_slurm_limit_errors` returns the two existing
  messages verbatim and the strategy raises its first message (`_cli_staged_slurm.py:78-108`,
  `:598-602`); pinned by M14 and M17.
- **Output writability semantics.** `os.access(anchor, W_OK | X_OK)` on the nearest
  existing ancestor (`_cli_preflight.py:1343-1350`) uses the kernel's `access(2)`, which
  on Linux consults POSIX ACLs and, on NFS, asks the server (so root squash is reflected);
  a denial therefore predicts a failing `mkdir`. I found no false refusal here short of
  exotic cases (NFSv2 UID mapping, FUSE filesystems that do not implement `access`).
  Inferred from the `access(2)` contract, not tested on those filesystems.
- **Space heuristic.** Warning only, `full` mode only, labelled a heuristic, hint names
  GPFS quotas (`_cli_preflight.py:1351-1366`, `HINTS["PF-OUTPUT-SPACE"]`), as §9 and R21
  require.
- **Node-local coverage (F29, R30).** `--output`, `--input`, `--pipeline`, `--metadata`
  and every in-scope `JoinMetadata` table are examined on SLURM runs only, by mount type
  rather than name, and the check skips when `/proc/self/mounts` is unreadable
  (`_cli_preflight.py:1368-1402`); pinned by M24, M25, M28c, M28d.
- **`partition_gres_error` return code first (F18)** and the strategy delegating to it
  (`_cli_execution_strategies.py:918-930`), pinned by M16; see E6 for why real `sinfo`
  rarely takes that branch.
- **Task 15 scope.** Local Validate is unchanged; no chrome changed, so `FEATURES.md` and
  `WORKFLOWS.md` need no edit; the e2e submit/cancel fixture now uses a real TIFF, which
  the plan's "fixtures are real images" constraint requires.
