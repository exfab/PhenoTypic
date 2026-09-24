# Slurm behavior the run preflight relies on (plan Task 13 Step 1)

**Status: NOT RUN.** The implementation environment for this change (a cloud
container, 2026-09-24) has no Slurm client: `sbatch`, `scontrol` and `sinfo` are not
on `PATH`. Following plan Task 13 Step 1, the step is recorded as not run and spec
open question 3 stays open.

The design does not depend on the answers. `PF-TIME-OVER-PARTITION` is a warning
whose wording follows the **live** `EnforcePartLimits` read at run time, and every
`sbatch --test-only` outcome that is not a clear rejection (no `sbatch`, a timeout, a
controller-communication error) is the `PF-SBATCH-UNAVAILABLE` warning.

To close the question, run on a login node of the target cluster:

```bash
bash docs/superpowers/plans/2026-09-24-cli-preflight/slurm_behavior_probe.sh <cpu-partition> [<gpu-partition>]
```

and record its output here. It submits no job. What each case settles:

| # | Case | Question it answers |
|---|---|---|
| 1 | valid request | `--test-only` reads the script from stdin and exits 0 |
| 2 | time far above `MaxTime` | rejected, or accepted to pend (`EnforcePartLimits=NO`) |
| 3 | unknown partition | rejected with a message the finding can quote |
| 4 | GPU request on a CPU partition | rejected by `--test-only`, making the GRES check redundant |
| 5 | misspelled option | rejected (the case that motivated `--test-only` over an allow-list) |
| 6 | `sinfo` on an unknown partition | its exit status, which the GRES check now reads (F18) |
| 7 | `scontrol show partition` | the `MaxTime` / `Default` fields the time check parses |
