# RESUME BRIEF — written 2026-08-17 17:03, for the 03:00 wake-up

The Slurm job hosting the session was restarted; three agents were killed
mid-flight. This file is the authoritative state so nothing is reconstructed
from memory.

## Where the work lives

**LIVE:  `/bigdata/iwheeldonlab/anguy344/PhenoTypic`**  branch `feat/mcp-server`
**STALE: `/bigdata/exfab/anguy344/PhenoTypic`**        branch `feat/mcp-server` @ `7bc9b6d25`

The repo was moved off `exfab` when that fileset hit its 36T quota. `exfab` has
space again but the live work stays on `iwheeldonlab` — see the merge-back step.

Always export before any `uv` command (the default cache is on a quota-limited
fileset):

    export UV_CACHE_DIR=/bigdata/iwheeldonlab/anguy344/.uv-cache
    uv run --no-sync <cmd>

## Cluster C1 — COMPLETE, unreviewed

HEAD `1292a946b`, working tree clean. Seven implementation commits:

    af0c8596e  T1    import-purity gate
    4fad8f0f2  T2    IMAGE_EXTS -> sdk_
    adc32c925  T3    operation registry + shim
    2cce6c625  T4    SandboxRoot
    be2afc66d  T2.5  lazy shell/tune package __init__s   (the B1 fix)
    3dae8ef69  T5    RunRegistry + LocalRunner
    1292a946b  T8    to_argv + RunConsoleState

Measured green before the restart:
  tests/unit/services            36 passed, 1 xfailed
  tests/unit/gui + integration   1727 passed, 3 skipped

The 1 xfail is deliberate and `strict=True`: `gui.tune._space` imports dash
directly at `_space.py:33-34`, so only Task 6's split can fix it. It will
XPASS -> FAIL when C2 lands, forcing the marker's removal. Do not "fix" it.

## Agents killed by the restart — redeploy in this order

1. **`C1-gate-review`** — the cluster gate. Was running when killed; produced
   nothing. **Redeploy first.** Reviews the combined C1 diff
   (`git diff 7bc9b6d25..1292a946b -- src tests`) for false greens (mutate each
   new test and confirm it fails), shim completeness (derive required names from
   the code — two were already missed this way), scope leak on the five
   `git mv` tasks, and the actual exported Interfaces of each `_services` module.
2. **`C1-promotion`** — finished its cluster; only redeploy if the gate returns
   blockers it should fix.
3. **`plan-reviewer`** — DONE. Delivered before the restart; findings are in
   `review-findings.md`. Do not redeploy.

## After the gate clears

1. **Merge back to exfab** (user instruction). `7bc9b6d25` is an ancestor of
   `1292a946b`, so this is a fast-forward, not a merge. First clear these stale
   untracked leftovers in the exfab repo, which will otherwise block the
   checkout — all are superseded by committed versions in the live repo:
       src/phenotypic/_services/     tests/unit/services/
       plus an uncommitted docs/.../execution.md
   Confirm each is superseded before removing, and report what was removed.
2. **Dispatch C2** (Tasks 6 + 7 — the `_space.py` pure/view split, then folding
   four modules into `_services/tune_spec.py`). Note T8 already landed in C1, so
   B2's ordering requirement is satisfied.

## Open items not blocking C2

- B4, B5, B8, B9 in `review-findings.md` — task-content defects to fix before
  their clusters (C4, C6) run. B5 splits Task 10 into 10a/10b/10c.
- Three decisions already taken and recorded: I8 (cluster-boundary reviewers),
  B3 (`--slurm` becomes `action="append"`), B7 (finalize re-loads images).
- HPCC ticket still unsent: snapshot `1786876141` pins ~10TB of deleted data.

## Orchestrator discipline

**Never `git add -A` while an implementation agent is running** — it already
swept an agent's staged rename into a docs commit (incident X1). Stage explicit
paths under `docs/superpowers/plans/` only.
