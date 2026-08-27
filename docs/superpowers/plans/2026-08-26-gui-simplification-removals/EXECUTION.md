# Execution — cluster-and-isolate

Derived from each task's `Files` / `Interfaces` blocks. A version-controlled view;
regenerate it when a task's `Files` block changes.

**Worktree:** `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/gui-ome-zarr-sync`
**Baseline:** `feat/gui-ome-zarr-sync`, restacked onto `worktree-ome-zarr-image-store`
head `bf0d01a1`. Nothing in this plan is implemented yet.

## Pre-dispatch gate — discharged

The skill's pre-dispatch `plan-reviewer` gate is **satisfied by the refinery**, not skipped.
Two rounds put this plan through `general-reviewer`, `data-flow-reviewer`,
`simplicity-reviewer`, a security specialist and `algorithm-fidelity`, plus an independent
resolution verifier — strictly more coverage than one `plan-reviewer` pass, and every
Critical is closed. Provenance:
[`refinery/ledger.md`](../../specs/2026-08-26-gui-ome-zarr-sync/refinery/ledger.md).

Carried in as known, and **not** re-litigated during execution:
~10 Minors (ledger "Open at end of round 2"). None blocks a cluster.

## The DAG

```text
1.1 ──► 1.2 ──► 1.3 ──► 1.4 ─┐
 (test)  (del)  (repair) (ledger)
                              ├──► 2.1 ─► 2.2 ─► 2.3 ─► 2.4b ─► 2.5 ─► 2.4/2.6 ─┐
                              │     (pin) (del) (strip) (tests) (engine) (ledger) │
                              │                                                   ├──► 6
                              ├──────────────► 4.1 ─► 4.2 ─► 4.3 ────────────────┤   (verify)
                              │                (src) (skips) (ledger)             │
                              └──────────────► 5.1 ─► 5.2 ─► 5.3 ────────────────┘
                                               (src) (skips) (ledger)
```

**Shared-file edges — why this is mostly sequential:**

| Files | Touched by | Consequence |
|---|---|---|
| `results_viewer/_layout.py`, `_callbacks.py` | phases **1 and 5** | 5 cannot run beside 1 |
| `tests/unit/gui/results_viewer/test_layout_tab_shape.py` | 1 creates, 1 edits, **5 edits again** | same |
| `FEATURES.md`, `WORKFLOWS.md`, capture script, `tutorials/gui/index.md` | **1, 2, 4, 5 — all of them** | see below |
| `browse/*`, `_shared/timeline/` | phase 2 only | isolated |
| `shell/*` | phase 4 only | isolated |

**The ledger layer cannot parallelize, and that is the binding constraint.** All four
phases *delete sections* from `FEATURES.md`, and every deletion shifts the line numbers
below it. Two agents editing it in separate worktrees would not merely conflict — they
would each compute anchors against a file the other is renumbering. This is the same
hazard GEN-6 was rated Critical for, promoted from a stale citation to a live race.

**The heading-anchoring fix is what makes sequential execution safe in any order.** Because
every ledger instruction now anchors on `##`/`###` headings rather than line numbers, a
phase's edit stays correct no matter how many sections earlier phases removed.

## Clusters

| # | Cluster | Tasks | Shape | Model | Why |
|---|---|---|---|---|---|
| **A** | Results Timeline — delete + repair | 1.1, 1.2, 1.3 | Keystone + Sweep + **Seam** | Opus, high | 1.2 leaves the tree unimportable and 1.3 makes it whole; they **must be one diff**. 1.1 creates the test 1.3 edits. |
| **B** | Results Timeline — ledger | 1.4 | Sweep | Opus, high | Consistency-critical: this is the cluster that deletes curation rows if it uses line numbers. Not delegated to a mid-tier model despite being mechanical. |
| **C** | Browse Timeline — strip | 2.1, 2.2, 2.3, 2.4b | Keystone + Sweep + Seam | Opus, high | Same atomicity: 2.2 deletes, 2.3 repairs, 2.4b fixes the five test files. 64 ids + a large callbacks file — at the edge of one reviewable diff, but splitting it leaves the tree broken across a gate. |
| **D** | Shared timeline engine — delete | 2.5 | **Seam** | Opus, high | Tiny and risky: legal only once *both* consumers are gone. Isolated for its own gate precisely because it is small — risk ≠ size. |
| **E** | Browse — ledger | 2.4, 2.6 | Sweep | Opus, high | As B. |
| **F** | Tune — unmount | 4.1, 4.2, 4.3 | Keystone + Leaf + Sweep | Sonnet, medium | Smallest phase, no deletions, `_config.py` guarded by an explicit empty-diff check. Introduces `⏸ unmounted` + its legend row. |
| **G** | Heatmap / Error / QC — unmount | 5.1, 5.2, 5.3 | Keystone + Leaf + Sweep | Opus, high | Shares `_layout.py`/`_callbacks.py` with A. `_error_tab/` is a **CLI dependency** — a delete here breaks CLI finalize with no GUI test to catch it. |
| **H** | Verification & docs | 6.1, 6.3, 6.4 | Seam | Opus, high | Task 6.2 is a cut stub; nothing to do. |

**No parallel-worktree candidates.** Every cluster shares either `results_viewer/` or the
ledger with another. Attempting fan-out here buys nothing and risks the renumbering race.

## Gates

- **Per cluster (light):** read the diff, run the cluster's own tests + `ruff` on changed
  paths. Surface any design-level open question **to the user** before the next cluster.
- **Per phase (deep):** after A+B, C+D+E, F, G — dispatch a fresh reviewer over the
  phase's combined diff (Opus). Use `implementation-test-reviewer` where the phase added
  tests (A, C, H): it checks the tests can actually **fail**, not merely pass.
- **End:** one `code-simplifier` pass (quality only, no behaviour change), apply, then the
  regression suite for affected areas.

## Test invocation — binding for every cluster

```bash
QT_QPA_PLATFORM=offscreen uv run pytest <paths> -n 4 -q
```

- `QT_QPA_PLATFORM=offscreen` is **mandatory** — without it the interpreter aborts at ~79%
  with no summary.
- **Never `-n auto`** — `nproc` reports the node's cores, not the allocation's, and
  manufactures timeout failures.
- **`tests/gui` is not optional.** Browse GUI tests and the colony-view package live there;
  a `tests/unit/gui`-only run never reaches what clusters C, D, G and H touch.
- **Known-failing baseline, not caused by this work:**
  `tests/unit/cli/test_cli_terminal_failures.py::test_concurrent_process_appends_do_not_lose_records`
  — spawns 8 processes on a 4-core allocation. Report "green except this one", and
  re-confirm it is still *that* test failing for *that* reason.
- The full `tests/unit` suite is a ~65-minute Slurm job, not a local invocation
  (`plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`). Per-cluster runs are
  scoped; the full suite runs once, at the end.

## Order

`A → B → [deep review] → C → D → E → [deep review] → F → [review] → G → [deep review] → H → [simplify] → full suite`

Phase 4 (F) is independent and could move earlier; it is placed after the browse work only
so the two `FEATURES.md`-heavy phases are not adjacent.

---

## Execution log

### Phase 1 — Results Timeline: **COMPLETE**

| Cluster | Commits | Outcome |
|---|---|---|
| A (tasks 1.1-1.3) | `fb56e004`, `fe74d832` | 472 tests green; scope boundary held — zero ledger edits |
| B (task 1.4) | `fae46da1` | 19 rows + heading, 1 WORKFLOWS row, tutorial + PNGs, 231 capture-script lines |
| review fix | `936492ef` | two orphaned tests removed |

**Gates at close:** `check_features_md.py` and `--strict` exit 0 (434 rows, 363 shipping);
`check_workflows_md.py -v` exit 0 (19 workflows, 19 dispatched). Colony curation rows **4,
unchanged**. `gui/_shared/timeline/` rows **2**, left for phase 2.

**The gate is red between a phase's code commit and its ledger commit** — cluster A deletes
the tests that cluster B's rows still reference. Expected, and the reason the phase is the
PR unit rather than the commit (GEN-11).

#### Findings execution produced that two review rounds did not

1. **A plan enumeration gap, now fixed** (`936492ef`).
   `tests/unit/gui/test_capture_tutorial_script.py:20,41` monkeypatched
   `RESULTS_TIMELINE_OUTPUT_DIR`, removed by cluster B — both `AttributeError` on the
   default lane. Task 1.4 prescribed grepping `_capture_results_timeline`; these tests name
   only the **seed** symbols, which do not match that pattern. **The orchestrator's gate
   missed it too**, having scoped to `tests/unit/gui/results_viewer`.
   **Lesson: a removal's blast radius is not enumerable by grepping the removed surface's
   name.** Later phases run the whole `tests/unit/gui tests/gui tests/integration/gui` lane
   at their gate, not a scoped subset.

2. **Phase 5 must NOT delete `TAB_QC_ID` / `TAB_HEATMAP_ID` / `TAB_ERROR_ID`** — recorded
   in phase 5's preamble. Phase 1 deleted `TAB_TIMELINE_ID`, making the analogy inviting and
   wrong: that tab was *deleted*, these are *unmounted*, and the retained packages still read
   their ids (`_error_tab/_callbacks.py:442`).

3. **Two Minors deliberately left alone.** `VIEWER_THUMB_URL_SEGMENT` is now orphaned in
   `_config.py`, but spec §9 Non-goals forbids `_config.py` constant removal. A stale
   docstring at `colony_view/_grid.py:153` cites the deleted timeline surface — tidying it
   would put a diff in `colony_view/`, which spec §5 protects and phase 5's guard asserts is
   byte-unchanged. **The guard is worth more than the tidy.**

#### An orchestration mistake worth not repeating

The full-lane suite was run **while cluster C was mid-edit**, producing 87 failures that all
cascaded from `browse/` being in its deliberately-unimportable window between tasks 2.2 and
2.3. Nothing was broken; the measurement was taken at the wrong moment. **Clusters share one
worktree and run sequentially — verification happens at the gate, never beside a running
cluster.**

### Phase 2 — Browse Timeline + shared engine: **COMPLETE**

| Cluster | Commits | Outcome |
|---|---|---|
| C (2.1-2.3, 2.4b) | `19735350`, `0e20fb4f` | 64 ids, 5,106 deletions; **7** broken test files, not the 5 the plan listed |
| D (2.5, 2.6) | `2c9f85c8`, `41ace3b4` | engine + 8 test files gone; 2 ledger rows |
| E (2.4) | `37c31013` | 19 rows retired, 1 **repointed**, tutorial + capture fn |

**Gates at close:** all three exit 0 — 408 feature rows / 339 shipping; 18 workflows, 18
dispatched. Full GUI lane **2356 passed, 3 skipped, 0 failed**. Curation rows **4**.

#### The `_preparation.py` race — adjudicated, change is CORRECT

Cluster C modified a file the plan said not to touch, and was right to. It hoisted
`_mark_ready` out of the publication lock because `complete_event` was being set while the
lock was held, so `BrowseCache.clear()` — which takes that lock with `timeout=0.0` and
`continue`s on `ArtifactLockTimeout` (`_cache.py:300-312`) — would be refused and **silently
skip the entry**. The timeline-thumb `rmtree` at the top of `clear()` had been masking it;
deleting that dead code exposed it.

> **Correction (phase-2 review).** The rmtree masked it by **timing, not semantics** —
> ~1 ms of syscalls executing before the entry loop, i.e. an accidental sleep. A bare
> `time.sleep(0.002)` at the top of `clear()` reproduces the mask exactly. The commit
> message's causal story is right in effect and wrong in mechanism; a reader should not
> infer the timeline thumbs mattered.

The obvious objection is that hoisting a completion signal out of a lock trades one race for
another: there is now a window where the entry is published and the lock released but
`complete_event` is not yet set, in which a concurrent `clear()` could `rmtree` the entry and
leave `_mark_ready` marking a deleted one ready.

**That window is not reachable in production.** Traced:

- `clear(*, protected=frozenset())` defaults to protecting **nothing** — so the objection is
  well-founded on the signature alone.
- But the **only** production caller is `_preparation_routes.py:153`, and it passes
  `protected = set(self.manager.protected_keys())` plus pinned selections plus
  `current_revision`.
- `protected_keys()` (`_preparation.py:310-317`) returns `set(self._requesters)` ∪ pinned ∪
  `_active_key`. A key under active preparation **necessarily has requesters** — otherwise
  `_cancelled()` (`:642`) is true and the work aborts before publishing.
- Therefore the in-flight key is always in `protected`, and `clear()` skips it for the whole
  window. The remaining bare `cache.clear()` calls are in tests only.

Verdict: the race C fixed was real; the race the fix could introduce is closed by
`protected_keys()` at the only call site that matters. **No change required.**

Recorded because the reasoning is not visible from the diff, and the next person to read
that hoist will ask exactly this question.

#### Phase-2 review findings applied

- **Major, fixed** (`browse.js`): ~190 lines of dead Browse Timeline JS, never touched by
  the phase. **The phase-1 lesson, repeating structurally:** the prescribed greps were
  Python-symbol-shaped (`BROWSE_TL_*`) and the JS spells the same ids kebab-case
  (`browse-tl-grid`), so no grep for the removed surface's Python names could reach it.
  A test was also **pinning** the dead code — narrowed with it.
- **Minor, not fixed, recorded:** the hoisted-signal invariant holds at **one of five**
  terminal-signal sites; `_mark_preview_only` / `_mark_cancelled` still set
  `complete_event` under the lock, and the preview path reproduces the original bug in
  23/30 trials. No production consequence (`protected_keys()` covers those keys too), but
  the comment at `_preparation.py:634-638` reads as a general rule the file does not
  follow. Either hoist the other four or narrow the comment — **carried to the simplify
  pass**, not done here, because it is a behaviour change outside this phase's scope.
- **Minor, not fixed:** `_config.py:740-741` says the thumb segments are "mounted by
  `register_thumbnail_route`", a function that no longer exists. Spec §9 forbids
  `_config.py` constant removal; the stale comment is inside that carve-out.

### The heading-anchoring fix, vindicated concretely

Measured before phase 1, and again after phases 1-2 landed:

| Section | Spec said | Before phase 1 | **Now** | Drift |
|---|---|---|---|---|
| `## QC tab` | 587 | 594 | **546** | −48 |
| `## QC Review sub-view` | 617 | 624 | **576** | −48 |
| `## Heatmap tab` | 658 | 665 | **617** | −48 |
| `## Error analysis tab` | 677 | 684 | **636** | −48 |

The spec's numbers were already wrong by ≈ +7 when written. Phases 1 and 2 then deleted
~48 lines of rows **above** these sections, so every one of them has moved again — and will
move once more when phase 4 retires the Tune block, which sits above all four.

**Had the plan kept line numbers, cluster G would now be editing sections ~48 lines off
target** — landing inside `## Analysis sub-app` and `## Cross-cutting infrastructure`
instead. GEN-6 was rated Critical on the strength of one bad range; the real hazard is that
every range in a multi-phase removal is a moving target by construction.

This is also why the ledger layer could not parallelize (see the DAG): concurrent agents
would each be computing anchors against a file the other was renumbering.
