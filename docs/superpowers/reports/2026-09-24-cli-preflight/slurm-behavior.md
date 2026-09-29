# Slurm behavior the run preflight relies on (plan Task 13 Step 1)

**Status: NOT RUN.** The implementation environment for this change (a cloud
container, 2026-09-24) has no Slurm client: `sbatch`, `scontrol` and `sinfo` are not
on `PATH`. Following plan Task 13 Step 1, the step is recorded as not run and spec
open question 3 stays open.

The design does not depend on the answers. `PF-TIME-OVER-PARTITION` is a warning
whose wording follows the **live** `EnforcePartLimits` read at run time, and only an
`sbatch --test-only` failure whose message names a configuration fault is an error;
every other outcome is the `PF-SBATCH-UNAVAILABLE` warning.

## Settled from Slurm's source and manual pages (Phase E review, E1, E5-E7, E11)

Read at `SchedMD/slurm` `master` on 2026-09-25 by the Phase E reviewer and, for the
strings the code matches, re-read when fixing. The installed version on a target cluster
may differ; the probe below confirms it there.

- `--test-only` validates the script and submits no job (`sbatch.1`, `--test-only`). It
  exits 1 on any will-run failure, printing `allocation failure: <message>`
  (`sbatch.c:283-291`), and has no retry, where a real submission retries a full queue
  or a busy controller.
- The will-run test considers only nodes that are not DOWN or DRAINED
  (`job_scheduler.c:4632-4633`) and then reports "Requested node configuration is not
  available"; a real submission of the same job is accepted and queues
  (`job_mgr.c:4485-4523`). **This is why `--test-only` failures other than a named
  configuration fault are warnings** (review E1).
- The error strings the code treats as configuration faults are `slurm_errno.c:221`
  (invalid partition), `:415` (account), `:427` (QoS) and `:521` (GRES).
- Input environment variables override options set in the batch script (`sbatch.1`,
  INPUT ENVIRONMENT VARIABLES, NOTE), so `SBATCH_PARTITION` and `SBATCH_TIMELIMIT` win.
- `EnforcePartLimits` (`slurm.conf.5`): `ALL` requires every requested partition's
  limits; `ANY` accepts a job that satisfies one; `NO`, the default, keeps a job queued
  only when it exceeds all of them.
- `sinfo --Format` fields without a size are truncated to 20 characters (`opts.c`), and
  `sinfo` has no error path for a partition name that matches nothing, so it exits 0 with
  empty output (inferred from `sinfo.c`; case 6 confirms).

To close the question, run on a login node of the target cluster:

```bash
bash docs/superpowers/plans/2026-09-24-cli-preflight/slurm_behavior_probe.sh <cpu-partition> [<gpu-partition>] [<drained-node>]
```

and record its output here. It submits no job. What each case settles:

| # | Case | Question it answers |
|---|---|---|
| 1 | valid request | `--test-only` reads the script from stdin and exits 0 |
| 2 | time far above `MaxTime` | rejected, or accepted to pend (`EnforcePartLimits=NO`) |
| 3 | unknown partition | rejected with a message the finding can quote |
| 4 | GPU request on a CPU partition | rejected by `--test-only`, making the GRES check redundant |
| 5 | misspelled option | rejected, and whether the message contains "unrecognized option", which is what makes it an error rather than a warning |
| 6 | `sinfo` on an unknown partition | its exit status and output length; the GRES check treats empty output as unknown (E6) |
| 7 | `scontrol show partition` | the `MaxTime` / `Default` fields the time check parses |
| 8 | GPU partition GRES | whether `--Format=gres` truncates a GPU entry that `-o %G` shows (E5) |
| 9 | a job only a drained node can run | that `--test-only` fails where a real submission would queue (E1); needs a drained node name |
