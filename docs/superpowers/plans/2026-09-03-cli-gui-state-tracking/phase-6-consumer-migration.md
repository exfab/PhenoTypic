# Phase 6 — Consumer migration and the deletions

**Depends on:** P1–P5. **Blocks:** P7.

**Spec:** §11 (consumer migration), §11.1 (~1,400 lines deleted), §11.2 (folded in).

**Goal:** every consumer of the nine evidence sources calls `resolve_run_state` instead,
and the machinery they used is deleted. This is the phase that pays for the previous five.

> ### ⚠ `--mode recompile` is unsupported on `--metadata` trees for the whole of this phase
>
> **Known, ruled on, and owned — not a bug to re-diagnose.** From P4 until
> [P7 Task 5 Step 1e](phase-7-migrate-mode.md), `--mode recompile` **raises** on any
> forward tree built with `--metadata`. Reproduced end to end against the shipped CLI:
>
> ```
> forward run with --metadata: exit=0   store tables: ['measurements', 'metadata']
> --mode recompile on that tree: exit=1  RuntimeError: Cannot recompile an inverted store
> ```
>
> **Why:** P4 Task 2 makes every `--metadata` run write both tables; recompile still
> builds with the pre-inversion producer, which would silently un-invert the store.
> `_refuse_inverted_store` (`_cli_recompile_tables.py:87`) stops that. The guard is
> correct; the repoint that makes it unnecessary is scheduled, not done.
>
> **What this means for you in this phase.** Metadata-free trees recompile normally. If
> you need a `--metadata` tree recompiled to verify something here, you cannot get it —
> plan the verification around a metadata-free tree, or defer it past P7. A P4 pre-flight
> scan makes the refusal whole-run rather than per-store, so a mixed tree is refused
> before anything is rewritten.
>
> **Do not repoint it here.** The trigger is not the schema gate — an inverted store is a
> *forward* tree, so arming does nothing for it — and the real work is extending
> recompile's write transaction to bind and atomically commit two tables. That belongs
> with the phase that owns the retirement, and user ruling (2026-09-08) put it in P7.

**The deletions are the deliverable.** A task here that migrates a consumer without
deleting what it replaced has not finished — the failure mode this whole change addresses
is nine sources that each closed a real hole and none of which was ever removed.

> **The first draft of this phase migrated the GUI and forgot the CLI (CAN-8).** Spec §9's
> caller table names two CLI depths — "CLI finalize, before publishing proofs → deep" and
> "CLI resume, deriving the worklist → deep, cache-assisted" — and §9.2's entire headline
> scenario (10 images added to 6,000) **is** the resume worklist. Migrating only the GUI
> leaves the O(N)-hashing readers on every CLI path, so the double walk is never removed,
> §11's last row ("split: readers → `sdk_/_run_state.py`") is not delivered, and **two
> completion predicates ship permanently** — the exact drift hazard this phase cites when
> deleting `_latest_event_states`. Task 0 fixes that, and it goes first because everything
> else in the phase assumes the split has happened.

---

## Deletion ledger

Track these in the phase's final commit body. Spec §11.1 estimates ~1,400 lines.

| # | Delete | Where | Task |
|---|---|---|---|
| 1 | `classify_output_consistency`, `OutputCompletionEvidence`, `inspect_output_consistency`, `OutputConsistencyReport` | `gui/results_viewer/_output_consistency.py` (617 lines, whole file) | 2 |
| 2 | `RunRegistry._processing_state_conflict`, `_publication_evidence_conflict`, `_orchestration_state_conflict` | `gui/shell/_runs_registry.py:1087,1202,1264` | 4 |
| 3 | `_local_completion_evidence_conflict`'s 8-branch tree | `_runs_registry.py:591` | 5 |
| 4 | `_latest_event_states` | `_runs_registry.py:1172` | 4 |
| 5 | `_read_status_from_manifest`, `_manifest_is_complete` | `_runs_registry.py`, `_slurm_observer.py` | 4, 6 |
| 5b | `local_manifest_completion_problem` — **the third manifest consumer** (M6) | `_cli_gui_lifecycle.py:41-65`, gating `publish_run_completion_evidence` at `:130-135` | 4 |
| 6 | `DashboardManifestKey.VERSION` — written as `3` at one site, read at **zero** | `_dashboard/_manifest_builder.py:766` | 7 |
| 7 | `sdk_/monitor_slurm_jobs.py` — zero importers in `src/` or `tests/`, **but `sdk_/CLAUDE.md` names it**; delete the doc reference in the same commit or it is stranded | whole file (241 lines) | 7 |
| 8 | `browse/_source_render.py`'s `browse_cache_base` / `cache_png_path` / `init_cache` / `wipe_cache` — zero production callers. ⚠ **The obvious check refutes this and is wrong:** `gui/builder/_preview_cache.py` **defines its own** `init_cache`/`wipe_cache`, so `grep -rln init_cache` returns four files. Verify with `grep -rn "_source_render import"` — nothing imports these four from here | `_source_render.py:35-38` (the `__all__` entries) | 7 |
| 9 | ⚠ **CLAIM REFUTED — measure before deleting anything here.** *"Eight zero-caller resolvers"* is false. Measured at `99cc068c`: of eleven `resolve_*` in `_io_constants`, **nine have `src/` consumers** (`resolve_processing_state_path` 13 files, `resolve_manifest_json_path` 6, `resolve_event_log_path` 6, `resolve_tuning_spec_path` 6, `resolve_pipeline_config_path` 4, `resolve_progress_dir` 3, and three more). The only two without a `src/` caller are `resolve_best_pipeline_path` (a test pins its legacy fallback) and `resolve_qc_dir` (**documented as live in `_cli/CLAUDE.md:425` and `gui/CLAUDE.md:286`**). **Zero are safe to delete as stated.** The `:2107` citation is also dead — it now lands mid-docstring, +70 lines from P5 `b8ef480f`. | `_io_constants.py:1111-2371` | 7 |
| 10 | Every `_legacy_*` helper and `resolve_*` fallback on the hot path | across `_cli` | P7 (they **move into** migrate, not away) |

> **M6: `manifest.json` is still evidence after P6 unless this site is converted.**
> `local_manifest_completion_problem` branches on `DashboardManifestKey.COMPLETED`, `FAILED`
> and `TOTAL_IMAGES` (`completed != total`, `failed != 0`) and **gates run-proof
> publication**. It is in neither `_output_consistency.py` nor `_slurm_observer.py`, so the
> round-1 note claiming every manifest-count reader lives in those two files was wrong.
>
> **Which half of the "sole publisher" sentence Task 4 relies on, and it is not the
> obvious one.** Root `CLAUDE.md` says the dependent finalizer is *"the sole publisher of
> aggregated outputs **and** the completion marker"*. After P5 those halves have
> **different truth conditions**: the completion-marker half holds unconditionally (the
> forward flow never publishes the run proof — `phenotypicCLI.py:2437` is guarded by
> `process_only_layer is not None` and `:3821` is `--mode recompile`; only `_run_finalize`
> publishes it), while the aggregated-outputs half is **false under `--wait`**, where
> `AutonomousSLURMStrategy` never sets `remote_managed` and the CLI aggregates in-process at
> `phenotypicCLI.py:2977`. **Task 4's deletions rest on the completion-marker half and are
> therefore safe** — but a reader who follows the citation to that sentence gets the other
> half unless this says which one.
>
> This does **not** disturb U-5 — that ruling was about `RunState`-mediated consumers, and
> this one is not — but it makes two plan claims false unless fixed: P7 Task 6's register
> says *"`manifest.json` as evidence"* was deleted, and §4.2 demotes it. **Convert this
> site**, or state in the register that the GUI-local publication path is the one surviving
> manifest consumer, and why it is allowed to be.

Items 1–9 land here. **Item 10 is P7's** — spec §11.1 says legacy paths move *into*
`--mode migrate`, and deleting them before migrate can read them would strand every
existing tree.

---

## Task 0: Split `_cli_completion.py` and migrate the CLI's own readers (CAN-8)

**Files:**
- Modify: `src/phenotypic/_cli/_cli_completion.py` — readers out, writers stay
  — ⚠ **re-grep before trusting any line number here.** P2 Task 4 renamed a
  writer parameter in this file; see below.
> **All citations below RE-DERIVED at `99cc068c`. The previous set was 2-for-13**, and
> three of the misses were caused by P5 itself — see the callout under this list.

- Modify: `src/phenotypic/phenotypicCLI.py` — completion readers at **`:2432,2436`**
  (`current_run_is_complete`, process-only branch), **`:2497-2516`** (a dense five-reader
  block: `current_aggregate_is_current`, `current_success_counts`, `valid_run_completion`),
  **`:2964,2966`** (`current_success_counts`), **`:3816,3820`**
  (`current_run_is_complete`, recompile). The old citation named five lines; there are
  **four clusters over ten call lines**, and `:2497-2516` appeared in neither.
- Modify: `src/phenotypic/_cli/_cli_checkpoint_handler.py:303,305,354,362,413,415`
  (was `291,348,401`; **≈ +12, P5 `eadf0fdf`** added the fan-out block to `_run_finalize`)
- Modify: `src/phenotypic/_cli/_cli_recompile_worker.py:672,676,679,689` (was `643,653`)
- Modify: `src/phenotypic/_cli/_cli_gui_lifecycle.py:90` — **still correct**
- Modify: `src/phenotypic/_cli/_dashboard/_manifest_builder.py:729` — **still correct**
- Modify: `src/phenotypic/sdk_/_hdf_to_zarr.py:742` (was `728`; **+14, P5 `eadf0fdf`**
  added the `source_work_ids` rationale)
- Modify: `src/phenotypic/_cli/_cli_staged_resume.py:238,240,448` — `valid_image_success`,
  not the three above (was `203-213`)
- Modify: `src/phenotypic/_cli/_cli_migrate.py:88-89` — **still correct**

> ### ⚠ P5 moved three of these, and nothing could have caught it
>
> `_cli_checkpoint_handler.py` (+12), `sdk_/_hdf_to_zarr.py` (+14) and
> `_io_constants.py` (+70, ledger item 9) all shifted under P5's own edits. P5 and P6
> name **different files** in their `Files:` blocks, so `dag.py`'s veto table reports no
> conflict — correctly, because a `Files:` block is an index of write targets and
> "this file grew" is not one. Register entry 59.
>
> **A hit is not evidence of a live citation, only of a coincidence not yet checked.**
> The three marked *still correct* above were re-derived, not assumed.
- Test: `tests/unit/cli/test_completion_split.py` *(new)*

> ### ⚠ Line numbers in this file have drifted twice — regenerate, do not trust
>
> **CORRECTED after the P2 gate: P2 Task 4 renamed nothing.** This callout used to
> say *"P2 Task 4 renamed `scheduler_epoch` on `publish_image_success`"* and used
> that as the reason line numbers had moved. §5.1's collapse was **withdrawn**
> (`design.md:323-345`, user-ruled) — `publish_image_success` still takes
> `lifecycle_epoch`, and there is no rename to shift anything.
>
> The false premise sat inside the callout telling you not to trust `file:line`,
> so a reader who obeyed it and regenerated would find no rename and be left
> deciding which half of a paragraph about trustworthiness to believe.
>
> **The warning still holds; only its reason changed.** Lines in
> `_cli_completion.py` moved for other work — P2 Task 3 shifted
> `phenotypicCLI.py`, and the P2 gate's six fixes plus the shared-helper
> increment rewrote `valid_image_success` outright.
>
> That is the second drift: this task's own file list was already short by three files and
> wrong about the invocation count (gen-r4 N-1/N-2, open three rounds), and P2 Task 3
> moved lines in `phenotypicCLI.py` besides. **Regenerate the greps rather than trusting
> any `file:line` written here** — the commands are in Step 1 and Step 3, and this plan's
> most repeated defect is a citation that was true when written.
>
> The *claims* have held every time they were checked. It is the line numbers that move.

> ### ⚠ A deleted comment in the middle of your call sites
>
> Your `:2394` and `:2428` sites sandwich `phenotypicCLI.py:2422`, which **P2 Task 3
> rewrites** — and the four lines above it were a justification that Task 3 makes false:
>
> > *"Every compatible invocation owns a fresh machine-state epoch, including a no-work
> > reconciliation. This fences workers left by a killed local attempt and prevents
> > historical started events from remaining active forever."*
>
> That was true of a `uuid4()` generation. It is **not** true of a content-derived one —
> D3's whole point is that the generation is a function of content, so it is deliberately
> *not* fresh per invocation. And the fencing the comment claims as its purpose is now
> `restart_epoch`'s job, built in P2 Task 1.
>
> **Why this is a note and not a code conflict.** The statements do not overlap: your sites
> are reader imports migrating to `resolve_run_state`; `:2422` is a minting expression.
> Neither rewrites the other, and nothing you do to those imports changes what `:2422`
> should be. Your line numbers will shift, but they need re-verifying regardless.
>
> **The hazard is reasoning, not merging.** Someone migrating readers around this block
> while the stale comment is still present would conclude — reasonably, from the comment —
> that a fresh epoch per invocation is intended, and might preserve or reintroduce it.
>
> **The conclusion it hides, which the plan never states:** a resume mints the **same**
> generation and does **not** bump the restart epoch. A resume is not a restart; only
> `--restart` is. If a resume bumped, every resume would fence its own in-flight workers —
> the precise failure D5 exists to prevent, and the opposite of what a resume is for.
>
> P2 Task 3 rewrites the comment, so if it has landed there is nothing here to avoid. This
> note exists for the case where someone reads this task first.

> **The file list was short and the count has now been wrong three times.**
> Ten across four → **13 across 6** (measured `c9d1fbfc`) → **20 invocations across 9
> files** (re-measured `ef436461`, 2026-09-09):
>
> ```
> _cli_completion.py            5     phenotypicCLI.py                5
> _cli_checkpoint_handler.py    3     _cli_recompile_worker.py        2
> _cli_gui_lifecycle.py         1     _dashboard/_manifest_builder.py 1
> sdk_/_hdf_to_zarr.py          1     gui/run_console/_slurm_observer.py 1
> gui/shell/_runs_registry.py   1
> ```
>
> **And "the thirteen CLI call sites" in Step 3's title is wrong twice over:** the count is
> 20, and **two of the nine files are GUI**, not CLI — `_slurm_observer.py` and
> `_runs_registry.py`, which Tasks 4 and 6 migrate. A title that says *CLI* invites Step 3
> to convert six files and call it complete. The earlier figure was measured, and then
> quoted across four subsequent commits without re-measurement. `_cli_checkpoint_handler.py` (3 — the in-array
> `__PHENOTYPIC_CHECKPOINT__` dispatch), `_cli_recompile_worker.py` (2) and
> `_cli_gui_lifecycle.py` (1) were named nowhere in this task. P4 and P5 touch two of those
> files but for other reasons — P4 rewrites `_cli_recompile_worker.py:764` only, and P5's
> publisher table marks `_cli_checkpoint_handler.py` **not** a publisher — so nothing else
> in the plan removes these reads. Regenerate the list rather than trusting it:
>
> ```bash
> grep -rn 'current_run_is_complete\|current_success_counts\|current_aggregate_is_current' \
>   src/phenotypic --include=*.py | grep -v _cli_completion.py
> ```

**This task goes first.** Every later task assumes `resolve_run_state` is the only
completion predicate; while a second one survives on the CLI side, the phase's premise is
false.

- [ ] **Step 1: Write the test that keeps the split split**

```python
def test_only_one_completion_predicate_survives():
    """CAN-8 / §11's last row. Two parsers of one question drift -- this phase
    deletes _latest_event_states for exactly that reason, and would ship a new
    instance of it on the CLI side."""
    import subprocess

    from pathlib import Path

    # Scoped to src/phenotypic/_cli + sdk_, NOT all of src/ (gen-r4 N-1). The GUI's three
    # holders -- _runs_registry.py, _slurm_observer.py, and _output_consistency.py -- are
    # migrated by Tasks 1-6 of this phase, so a whole-tree grep here is red by construction
    # at the end of Task 0. The whole-tree assertion is Task 7's, where it can pass.
    # AST, not grep. A text search matches DOCSTRINGS and COMMENTS, and
    # `_cli_completion.py`'s own prose references these names -- so a grep
    # version goes red after the deletion and gets "fixed" by editing prose to
    # satisfy a search, which is the wrong repair. Assert on CALL SITES.
    # `tests/unit/sdk_/test_run_state_layering.py` is the precedent in-tree.
    import ast

    root = Path(__file__).resolve().parents[3] / "src" / "phenotypic"
    retired = {
        "current_run_is_complete",
        "current_success_counts",
        "current_aggregate_is_current",
    }
    hits = []
    for path in [*(root / "_cli").rglob("*.py"),
                 *(root / "sdk_").rglob("*.py"),
                 root / "phenotypicCLI.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Call):
                func = node.func
                name = (func.id if isinstance(func, ast.Name)
                        else func.attr if isinstance(func, ast.Attribute) else None)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in retired:
                        hits.append(f"{path}:{node.lineno} imports {alias.name}")
            if name in retired:
                hits.append(f"{path}:{node.lineno} calls {name}")
    assert not hits, "the old O(N)-hashing readers survive CLI-side:\n" + "\n".join(hits)


def test_no_migrated_reader_gained_a_second_definition():
    """A deletion guard that counts ZERO of the old name cannot see a NEW copy
    under the same name in another module.

    Step 2 says so itself -- *"nothing in this phase would catch a duplicate
    landing"* -- because both greps search only the three deleted predicate
    names. A second `valid_image_success` inside `_run_state.py` passes every
    other gate in this plan, and it is the exact defect this phase exists to
    remove: two parsers of one question.

    Counts DEFINITIONS, not occurrences, so a docstring naming the function is
    not a hit and cannot be "fixed" by editing prose.
    """
    import ast
    from collections import defaultdict
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "src" / "phenotypic"
    watched = {
        "valid_image_success",
        "valid_run_completion",
        "valid_aggregate_snapshot",
        "current_success_inventory",
        "run_proof",
        "run_proof_is_current",
    }
    seen = defaultdict(list)
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in watched:
                    seen[node.name].append(f"{path}:{node.lineno}")

    duplicated = {n: w for n, w in seen.items() if len(w) > 1}
    assert not duplicated, f"a migrated reader has two definitions: {duplicated}"


def test_the_resume_worklist_uses_the_cache_assisted_path():
    """§9's caller table, row 2 -- and §9.2's headline scenario IS this call.

    Asserted on CALL NODES, not on substrings, and not by scoping to a
    function.

    The first draft was `assert "resolve_run_state" in inspect.getsource(
    phenotypicCLI)` -- a substring test over a 4,000-line module, satisfied by
    a comment, an unused import, or a docstring ABOUT the migration.

    **The first correction scoped it to `_prepare_incremental_startup` and was
    also wrong**: that function is `:494-525`, thirty-one lines, and contains
    no completion reader at all. The readers are at `:2497-2516`, inside
    `phenotypic_cli` itself -- which is the 4,000-line command, so scoping to
    the enclosing function buys nothing over scoping to the module. **The axis
    that works here is node type, not location.**
    """
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[3]
        / "src" / "phenotypic" / "phenotypicCLI.py"
    ).read_text(encoding="utf-8")
    called: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)

    assert "resolve_run_state" in called, (
        "phenotypicCLI never CALLS resolve_run_state -- §9.2's headline "
        "scenario still re-hashes every marker"
    )
    assert not called & {"current_success_counts", "valid_run_completion"}, (
        "a retired O(N) reader is still called from the CLI"
    )
```

- [ ] **Step 2: Move six readers — and DELETE three (flow-r4 M4)**

> **This step and Step 1's test contradicted each other for three rounds.** Step 1 asserts
> the three predicate names appear nowhere; this step listed those same three among the
> symbols it *moves into* `sdk_/_run_state.py`. A moved symbol still greps. One of the two
> had to be wrong, and it is this one — settled by P1's own contract rather than by
> weakening the test:
>
> `sdk_/__init__.py` exports `resolve_run_state`, `run_identity`,
> `assert_identity_current` and `clear_verification_cache` (P1 Task 1's table). The
> `current_*` trio is **not** in that list and was never meant to be a second public
> predicate — the whole phase premise is that there is one. They are not moved. They are
> **subsumed**, because `RunState` already carries what each returned:
>
> | Deleted | Replaced by |
> |---|---|
> | `current_run_is_complete(d)` | ⚠ **TRI-STATE — two questions, see below.** Not this expression alone. |
> | `current_success_counts(d)` | ⚠ **three questions under one name — see the split below.** Not `diagnostics`. |
> | `current_aggregate_is_current(d)` | **`_run_proof_covers_current_inventory`'s clause 2** (`_run_state.py:1164`), reached through `resolve_run_state` — its docstring: *"Clause 2 is the **five** comparisons `current_aggregate_is_current` makes today, not the one an earlier draft kept (CAN-4)."* Private, one internal caller, and it takes already-loaded `config`/`identity`/`images`, so it is **not** callable directly the way the pair above is |
>
> With that, Step 1's scoped grep and Task 7's unrestricted one are both satisfiable, and
> the test's message ("the old O(N)-hashing readers survive") means what it says.
>
> ### ⚠ `current_run_is_complete` is TRI-STATE and its replacement is a bool
>
> ```python
> def current_run_is_complete(output_dir: Path) -> bool | None:
>     """Return marker-derived current completion, or ``None`` for legacy state."""
> ```
>
> `resolve_run_state(d).completion == "complete"` is a **bool**. It cannot carry the third
> arm, and **four sites branch on `is False` while treating `None` as *not* False.**
> Converting the row as written changes behaviour at two of them:
>
> | Site | Today, on a legacy (`None`) tree | After a naive conversion |
> |---|---|---|
> | Branch | Today, on a legacy (`None`) tree | After a naive conversion |
> |---|---|---|
> | `_cli_gui_lifecycle.py:91` — `is False` | not taken | **taken → raises** `"Cannot publish GUI local completion…"` |
> | `_cli_gui_lifecycle.py:96` — `is None and generation is None` | `return False` | unreachable — `:91` raised first |
> | `_cli_gui_lifecycle.py:98` — `is None` | manifest read | unreachable — `:91` raised first |
> | `_cli_checkpoint_handler.py:363` — `is False` | not taken | **taken → `deactivate_orchestration(…, "terminal_incomplete")`** |
> | `_cli_checkpoint_handler.py:365` — `else` | `mark_staged_complete` | not taken |
> | `_cli_checkpoint_handler.py:367` — `is True` | not taken (no run proof) | not taken |
>
> **Keyed by branch condition, not by value.** `_cli_gui_lifecycle` has **two `None` arms
> and one `False` arm** — three branches on the tri-state — and the two `None` arms differ
> from each other on `generation`. A summary saying "three `None` arms" sends a reader
> looking for a third `None` test that does not exist, and hides that `None` already
> produces two different behaviours.
>
> `_cli_checkpoint_handler` is the sharper case and the strongest argument for the split:
> the conversion does not make it stricter, it calls **the opposite function**. A tree that
> is marked staged-complete today has its orchestration deactivated as terminally
> incomplete instead.
>
> The other two (`_dashboard/_manifest_builder.py:729`, `gui/shell/_runs_registry.py:597`)
> and the three `is True` sites convert safely. **Right for three call sites, wrong for
> four** — which is why it survived three rounds: the name matches on both sides and the
> majority of sites are fine.
>
> ### The split: two questions with different costs
>
> Not a missing contract. A second row collapsing a shape, and the fix is the same as the
> other two — ask both questions explicitly:
>
> | Question | Cost | Target |
> |---|---|---|
> | *is this run complete?* | **O(N)** — what the migration exists to make cheap | `resolve_run_state(d).completion == "complete"` |
> | *is this a legacy state?* | **O(1)** — one JSON field, `success_markers_required` | read it directly; **not** a hashing reader, so the migration does not target it |
>
> ```python
> legacy = not json.loads(resolve_processing_state_path(d).read_text(encoding="utf-8")
>                         )["config"].get("success_markers_required", False)
> complete = resolve_run_state(d).completion == "complete"
> # is False -> (not legacy) and not complete    is None -> legacy
> # is True  -> (not legacy) and complete
> ```
>
> **No `_cli` import, including from the GUI.** `resolve_processing_state_path` is in
> `sdk_/_io_constants.py` and already exported — thirteen `src/` consumers, the most-used
> resolver in the ledger item this plan refutes. So `_runs_registry.py` gets its legacy
> signal through `sdk_` and P6's GUI-decoupling goal is unharmed.
>
> **⛔ Do NOT fold the two reads into a helper returning a tri-state.** That recreates
> `current_run_is_complete` under a new name in a new place — what
> `test_no_migrated_reader_gained_a_second_definition` exists to catch, and what §4.2
> demotes. Two named locals at each site.
>
> **⛔ Do NOT route the legacy read through `_schema_shape`.** It reads
> `success_markers_required` **zero** times, it is gated behind `SCHEMA_GATE_ARMED` which is
> `False` until **P7 Task 5 arms it** — so a P6 dependency on it would change behaviour when
> P7 lands, silently, at a distance.
>
> **This unblocks both stalled rows**: `current_success_counts`' `is not None` arm is the
> identical read, so one O(1) field answers both.

> ### ⚠ `current_success_counts` is THREE questions, and `diagnostics` answers none of them
>
> The row above used to send it to `resolve_run_state(d).diagnostics`. Every one of its
> eight surviving callers **branches** on the result — and `RunDiagnostics`' own docstring
> forbids exactly that, giving the reason:
>
> > *"Counts derived from `images`. **Nothing branches on these** (§4.2, §9). One-line
> > projections over `ImageState.verdict`, **not cached counts of a collection the caller
> > already holds.**"*
>
> The second sentence is the ruling: a caller holding a `RunState` already holds `images`,
> so `diagnostics` exists for **display**, not for decisions. Sending eight deciding callers
> there either reintroduces count-based branching under a new name — the thing §4.2 demotes
> and this change exists to end — or is simply the wrong target. It is the wrong target.
>
> | Question actually asked | Sites | Target |
> |---|---|---|
> | **`is not None`** — *is this a legacy state?* `current_success_counts`' own docstring: *"`None` identifies a **legacy state that does not require general image success markers**."* Not a count question at all | `_cli_recompile_worker.py:676`, `_slurm_observer.py:1317` | a **schema-shape** predicate — see the caveat below |
> | **`counts[0] > 0`** — *is anything verified yet?* | `phenotypicCLI.py:2502`, `:2966`, `_cli_checkpoint_handler.py:305`, `sdk_/_hdf_to_zarr.py:729` | a projection over `state.images`, which is what the `RunDiagnostics` docstring points at instead of itself |
> | **`successful != total`** — *is everything verified?* | inside `current_run_is_complete` only | **dies with its caller** |
>
> ⚠ **The schema-shape target does not exist yet, and this was checked rather than
> assumed.** `sdk_/_schema_shape.py` exports `requires_conversion`,
> `describe_conversion_advisory`, `describe_required_conversion`, `ConversionVerdict`,
> `SCHEMA_GATE_ARMED`, `STATE_SCHEMA_VERSION` — and **never reads
> `success_markers_required`**. It is also **not re-exported from `sdk_/__init__.py`**, so
> today's consumers reach it as `phenotypic.sdk_._schema_shape` (`_cli_schema_gate.py:30`),
> a private module path.
>
> `requires_conversion` asks *"is this tree in the old shape?"* by directory and version
> signals, which is adjacent to but **not the same as** *"does this state require success
> markers?"*. So the first row names the right **kind** of home and not an existing
> function. Settle that before converting those two sites; it is a P1-contract question, not
> something to close by adding a predicate here.
>
> **Step 2's stop rule does not fire.** Each of the three questions has a target or dies;
> what had no equivalent was the *conflation*. A row collapsing three questions into one
> name is a mis-shaped table, the same defect as a 1→1 mapping over a 1→2 tree one row up.

> ### ⚠ Corrected: three of these six are **already in `sdk_`** and cannot be moved
>
> Written before P1 existed, this list says six functions *move* into `sdk_/_run_state.py`.
> P1 shipped, and `_run_state.py` **already re-derives all six** under different names. Its
> own module docstring says so, at `_run_state.py:22-23`: *"the record and proof readers
> below re-derive what `valid_image_success`, `valid_aggregate_snapshot` and
> `valid_run_completion` decide today."*
>
> **Every line number below was re-derived at `ef436461`; the previous set was 0-for-11,
> drifting +32 to +342. And one entry named a function that does not exist.**
>
> | `_cli_completion.py` | Already in `_run_state.py` | |
> |---|---|---|
> | `valid_image_success` (`:288`) | `_verify_image` (`:682`) | |
> | `valid_run_completion` (`:1253`) | **`run_proof` (`:1002`) + `run_proof_is_current` (`:1032`)** — a **pair**, see below | |
> | `valid_aggregate_snapshot` (`:1100`) | `_valid_aggregate_proof` (`:1111`) | |
> | `current_success_inventory` (`:538`), `_walk_current_success` (`:585`), `_current_success_work_ids` (`:733`) | `_resolve_images` (`:1385`) + `_accepted_inventory` (`:828`) | |
>
> ### ⚠ `_valid_run_proof` does not exist, and the mapping is 1→2
>
> The previous table sent `valid_run_completion` to `_valid_run_proof` (`:771`).
> **`grep -rn "_valid_run_proof" src/` returns nothing**; the only tree match is the
> *prose* inside the test name `test_a_live_worker_does_not_mask_a_valid_run_proof`, which
> is why a grep for it looks almost-successful.
>
> The real counterpart is a **pair**, and `run_proof_is_current`'s own docstring names it:
> *"The other half of `run_proof`, and the comparisons
> `_cli_completion.valid_run_completion` makes once it has the marker: `inventory_digest`,
> `scientific_config_digest`, `finalization_input_digest` and -- since U-4 replaced the
> opaque `publication_id` -- `source_set_digest`."*
>
> The split is deliberate. `run_proof` (`:1002`) is **structural validity only** —
> `version`, `status`, `finalizer_succeeded` — and its docstring gives the reason: forcing
> a caller that only asks *"is this file a run proof at all?"* to load processing state
> *"is what pushed four readers into open-coding this predicate, and every one of those
> four dropped the `version` check on the way."*
>
> **Migrating to `run_proof` alone silently drops every digest comparison.** That is the
> case Step 2's stop rule exists for, and a 1→1 table over a 1→2 tree is how it hides.
>
> ### Two conjuncts stay at the CALLER, on purpose
>
> Both are documented decisions; record them at each converted site rather than
> rediscovering them:
>
> | Conjunct | Why it is not in `sdk_` |
> |---|---|
> | `current_run_is_complete` | *"a separate conjunct kept at the caller on purpose: it is O(N) in images, this is O(1), and a GUI surface polling every five seconds wants to ask the cheap question."* |
> | `success_markers_required` | *"the caller's policy. `valid_run_completion` waives them for legacy state"* |
>
> So the migrated form is **`run_proof(d) and run_proof_is_current(d)`**, plus those two
> retained locally. Folding either into `sdk_` reverses a documented decision and puts an
> O(N) walk back on a five-second GUI poll.
>
> **So this is a deletion, not a move — and one of the three cannot be moved at all.**
> `valid_run_completion` calls `load_processing_state` from `_cli_state_management`
> (`_cli_completion.py:1255`, the third line of its body). Carrying it into `sdk_` verbatim fails
> `test_neither_module_ever_names_the_cli_package`, which walks the AST for lazy
> in-function imports precisely so this cannot slip through. The `sdk_` replacement is
> 30 lines of structural validation against `RUN_PROOF_VERSION` (`run_proof`, `:1002-1031`;
> re-measured — the row said 12); the original is ~40 that
> reach back into CLI state. They are not the same function and the rewrite already happened.
>
> **What this task actually does:** confirm each `_run_state.py` reader covers its
> `_cli_completion.py` counterpart's behaviour, convert the call sites, delete the originals.
> If a behaviour has no `sdk_` equivalent, **stop** — same rule as the deleted trio below:
> that is a gap in P1's contract, not a licence to copy a second implementation into `sdk_`.
>
> **Nothing in this phase would catch a duplicate landing.** Step 1's scoped grep and Task 7's
> unrestricted one both search for the three *deleted predicate* names only. A second
> `valid_image_success` inside `_run_state.py` passes every gate in this plan.
>
> ### And the migration Step 3 does not size: `valid_image_success` has 14 external call sites
>
> **All three figures re-measured at `ef436461` (2026-09-09); all three were wrong, and the
> first correction fixed only the one that carried the words "measured this turn".** The two
> beside it were derived from the same command family and inherited its authority without
> inheriting its check.
>
> ```bash
> grep -rn "valid_image_success(" src/ | grep -v "def valid_image_success" | grep -v ">>>" | wc -l
> # 17   total   (was stated as 19)
> grep -rn "valid_image_success(" src/ | grep -v "_cli_completion.py" | grep -v ">>>" | wc -l
> # 14   outside its own file   (was stated as 15)
> grep -rn "valid_image_success(" src/ | grep -v "_cli_completion.py" | grep -v ">>>" \
>   | cut -d: -f1 | sort -u | wc -l
> # 10   distinct modules   (was stated as 11)
> ```
>
> 14 call sites across 10 modules.
>
> > **⚠ The export list this argument used to rest on was wrong, and wrong in the direction
> > that mattered.** It read *"`_run_state.__all__` exports **nine names, none of them a
> > per-image validator**"* and listed nine. Measured at `ef436461`: **sixteen**, and the
> > seven it omitted include **`run_proof` and `run_proof_is_current`** — the exact pair the
> > corrected mapping above sends `valid_run_completion` to — plus
> > **`staged_image_is_complete(output_dir, dataset, image_stem)`**, which *is* a per-image
> > reader, so the "none of them" clause was false as well.
> >
> > Full list: `ImageState`, `RunDiagnostics`, `RunIdentity`, `RunState`,
> > `accepted_finalization_digests`, `assert_identity_current`, `clear_verification_cache`,
> > `fenced_artifact_path`, `finalization_input_digest`, `finalization_input_object`,
> > `marker_rejection`, `resolve_run_state`, `run_identity`, `run_proof`,
> > `run_proof_is_current`, `staged_image_is_complete`.
> >
> > The sizing argument below still stands — none of the sixteen answers
> > `valid_image_success`'s question — but it must be made against the real list, because a
> > list missing the migration's own targets is the one a reader would use to conclude the
> > migration has nowhere to go.
>
> "Moves into `sdk_` and becomes private there" leaves
> all 14 with nothing to call, and Step 3 converts thirteen *other* call sites (the deleted
> trio's). **Decide the replacement for these 14 explicitly before deleting anything:** each
> is a per-image question, and the only exported answer is `resolve_run_state(...).images`,
> which is a whole-run walk. Calling it once per image is the O(N²) that §9's note about a
> per-task reader already warns of — so most of these 14 want the *record* reader in
> `sdk_/_image_record.py`, not `_run_state`. Name which, per site, in Step 3's table.
>
> ### `sdk_/_hdf_to_zarr.py` has FIVE `_cli` import statements, not one

> Re-derived at `1573a4e1`: **five statements** (`:605`, `:714`, `:719`, `:777`, `:778`)
> drawing from **two** modules — `_cli_completion` three times, `_cli_state_management`
> twice. The heading said four; an earlier report of this correction said "five across
> three modules", which was also wrong on the module count. Both are recorded because the
> figure has now been stated wrongly twice in a row, by two people, in an argument whose
> entire point is that this file is under-counted.
>
> Step 3's table gives it *"1 — migration's own progress read"*. On disk:
>
> | Line | Imports | Fate |
> |---|---|---|
> | `:605` | `ARTIFACT_KIND_STORE`, `SUCCESS_MARKER_VERSION`, `_artifact_descriptor` | P3 renames `SUCCESS_MARKER_VERSION` → `RECORD_VERSION`; `_artifact_descriptor` is private |
> | `:714` | `_current_success_work_ids`, `current_success_counts`, `publish_aggregate_snapshot` | **three, not two** — P5 `eadf0fdf` added the first when `source_work_ids` became required. Two of the three are deleted by this task |
> | `:719`, `:778` | `load_processing_state` | the progress read Step 3 counted (was cited `:718`, `:762`) |
> | `:777` | `valid_image_success` | the 14-call-site problem above (was cited `:761`) |
>
> This file is not incidental: P7 Task 6 Step 3 records that the HDF→Zarr migrator **is
> itself a producer of the record schema (CAN-7)**, not a stage that runs before one. It
> breaks on a deleted function, a privatised one, and a renamed constant, and Step 3 sizes
> it at one line.
>
> ### Why nothing fails today
>
> `test_neither_module_ever_names_the_cli_package` walks **four named modules** —
> `_state_types`, `_verification_cache`, `_schema_shape`, `_run_state` — not the `sdk_`
> package. `_hdf_to_zarr.py`, `generate_report.py` and `monitor_slurm_jobs.py` all import
> `phenotypic._cli` and all pass. That scoping is deliberate and correct for P1 (the test
> defends the readers it names, and says why each was added), but it means **INV-LAYER as a
> package-wide claim is untested**, and this task is where that gap becomes load-bearing.
> Either widen the walk to the package with the three known importers listed as explicit,
> dated exemptions, or stop describing the invariant as `sdk_`-wide. Do not leave it stated
> broadly and tested narrowly.

**Deleted** out of `_cli_completion.py` — their behaviour already lives in
`sdk_/_run_state.py`, per the correction above:
`current_success_inventory`, `_walk_current_success`, `_current_success_work_ids`,
`valid_aggregate_snapshot`, `valid_run_completion`, `valid_image_success`.

**Deleted** outright, per the table above: `current_success_counts`,
`current_aggregate_is_current`, `current_run_is_complete`. Step 3 converts every call site,
so nothing is left calling them; if a call site turns out to have no `RunState` equivalent,
**stop** — that is a gap in P1's contract, not a licence to keep the function.

Staying CLI-side because they **write**: `publish_image_success`,
`publish_aggregate_snapshot`, `publish_run_completion_evidence`, `image_data_artifact`,
`refresh_success_markers_after_metadata_migration`, `authorized_measurement_sources`
(a reader, but of run-authorization for the writer path — keep it beside its caller and say
so in a comment).

**INV-LAYER still binds.** The moved readers must not reach back into `_cli`; that is why
P1 Task 4 established the plain-JSON state read. If a mover needs
`load_processing_state`, it has taken the wrong function. The **record** reader they call is
in `sdk_/_image_record.py`, which P3 puts there for exactly this reason (N-3).

> **The move silently drops a relocation the readers depend on (M3).**
> `load_processing_state` calls `migrate_legacy_machine_state(output_dir)` on **every** read
> (`_cli_state_management.py:106`) — relocating `progress/`, `processing_state.json` and
> `processing_events.log` from the output root into `.phenotypic/`
> (`sdk_/_io_constants.py:1006-1052`). The readers being moved use the **non-resolving**
> `progress_dir` (`:903-909`; see `_cli_completion.py:27,785` and
> `image_completion_marker_path`), so on a **pre-relocation** tree they work today *only
> because the state read relocated first.*
>
> Removing the trigger without replacing it makes those readers silently find nothing on
> such a tree — an empty inventory, which is a *valid* result. Two options, and the second is
> better: have the moved readers use the **resolving** path helpers, so relocation stops
> being a precondition; or have P1's `requires_conversion` classify a pre-relocation tree as
> `CONVERT` and let migrate own the move. Decide in this task, and state which — a read path
> that depends on a write side effect is the thing this phase exists to remove.

- [ ] **Step 3: Convert the thirteen CLI call sites**

Each becomes one `resolve_run_state(output_dir, depth="deep")` — `deep` on the CLI, per
§9's table, because the CLI publishes proofs and derives worklists and must not act on a
stat-only answer. Re-grep before editing; these line numbers are from `c9d1fbfc`.

Three of the six files carry a call site the earlier draft of this task never named, and two
of them are the ones a resume actually runs through — so convert by grep output, not by the
list:

| File | Invocations | What the call gates |
|---|---|---|
| `phenotypicCLI.py` | 5 | startup counts, aggregate currency, the two early-exit checks |
| `_cli_checkpoint_handler.py` | 3 | the in-array `__PHENOTYPIC_CHECKPOINT__` dispatch (gen-r4 N-1) |
| `_cli_recompile_worker.py` | 2 | whether recompile may skip re-derivation (gen-r4 N-2) |
| `_cli_gui_lifecycle.py` | 1 | gates `publish_run_completion_evidence` — see item 5b above |
| `_dashboard/_manifest_builder.py` | 1 | the manifest's completion field |
| `sdk_/_hdf_to_zarr.py` | 1 | migration's own progress read |

All thirteen are `deep`, including `_cli_checkpoint_handler.py`'s three. Check that
conclusion rather than assuming it, because the file's name invites the opposite one: it is
the `__PHENOTYPIC_CHECKPOINT__` handler, which sounds per-task and is not. The trigger is a
**reserved entry in the array task list** (root `CLAUDE.md`, *SLURM array auxiliary work*) —
one index, not one per image — and all three call sites sit on a publication path:
`:291` decides whether an empty aggregate is a `RuntimeError` or a legitimate
terminal-incomplete close; `:348` chooses between `deactivate_orchestration` and
`mark_staged_complete`, then gates `publish_run_completion_evidence`; `:401` gates the
completion marker itself. That is §9's first row — *CLI finalize, before publishing proofs →
`deep`* — not its two `shallow` rows, both of which are read-only pollers. No `deep` read
here is per-image, so no O(N²) arises.

> **§9's table has no row for a worker process.** Its six rows are two CLI paths, two
> binding/guard paths, and two pollers. If a *genuine* per-task reader turns up during this
> task — one that runs once per array index — it has no assigned depth and this plan must
> not invent one: raise it, because `deep` there is O(N) per task on the exact walk this
> change exists to make cheap. None of the thirteen is such a reader.

- [ ] **Step 4: Confirm the double walk is gone**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli -q
```

Then count: a single completion query must walk the images **once**. Instrument
`valid_image_success` with a counter in a throwaway patch, run one `--mode full` resume
over a 6-image fixture, and assert the count equals 6 rather than 12. That doubling is
audit §4's finding, and it is the thing §9.2's number depends on.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic tests/unit/cli/test_completion_split.py
git commit -m "refactor: split _cli_completion.py -- readers to sdk_, writers stay

Spec §11's last row, §9's two CLI depth rows. Ten CLI call sites moved onto
resolve_run_state; the double walk (audit §4) is gone. A test now fails if a second
completion predicate reappears."
```

---

> ## ⚠ Before Tasks 1 and 6: the poll cadence spec §9 assigns is not achievable
>
> **User ruling (2026-09-04): the pollers gate on a cheap sentinel and call
> `resolve_run_state` only when it moves.**
>
> Spec §9 assigns the GUI snapshot poll (5–10 s) and the SLURM observer tick (2 s) to
> `depth="shallow"`. Measured on a real 6,657-image tree on GPFS, **shallow has a ~150 s
> floor** — about 20,000 `stat` calls at a measured **7.63 ms each**. The 2 s tick is 75×
> short of what one poll costs.
>
> > ### ⚠ MEASURED AND REFUTED: a `stat` here is 0.012 ms, not 7.63 ms
> >
> > **Neither input has provenance.** `grep -rn "7\.63\|20,000\|20000"` across this
> > change's entire plan and spec directories returns **one hit: the line above.** A figure
> > labelled *"a measured 7.63 ms"* with no measurement on record.
> >
> > Measured at `6987f646` on GPFS, over 2,000 real files under the subset bed:
> >
> > ```
> > os.stat                     0.012 ms   <- 620x faster than the figure used
> > read + parse a marker JSON  0.346 ms
> > sha256 an overlay PNG      67.277 ms
> > ```
> >
> > **The internal arithmetic is sound and the input is not.** 20,000 x 7.63 ms = 152.6 s,
> > which does give 76.3x ("75x"), 1403/152.6 = 9.19x ("~9x"), and 2 x 7.63 = 15.3 ms
> > ("~15 ms"). Every derived figure follows correctly from a number that is wrong by
> > two-and-a-half orders of magnitude — which is why none of them looked suspicious.
> >
> > At the measured rate, **20,000 stats cost 0.24 s**, which *fits inside a 2 s tick* with
> > room to spare rather than exceeding it 75-fold. **Task 6's quantitative case inverts.**
> >
> > **And 7.63 ms is not a mismeasurement of a `stat` — it is the wrong operation.** It is
> > within an order of magnitude of hashing, not of stat-ing. So the paragraph below,
> > *"an on-disk cache tier does not help here — it saves the hashing, not the stat-ing"*,
> > is arguing against the cache using **the hashing cost as if it were the stat cost**. If
> > the floor is really 0.24 s, the cache tier is exactly what removes the expensive part
> > and the "irreducible floor" claim does not hold.
> >
> > **What this does NOT settle:** whether Task 6 should still happen. A sentinel gate is
> > defensible on other grounds — 20,000 stats per tick is wasteful even at 0.24 s, and the
> > *hashing* cost is real and large (67 ms per overlay PNG). What is refuted is the
> > specific case as argued. **Re-derive the rationale before executing Task 6**, and state
> > which operation each figure measures.
>
> **This is not a regression this change introduces.** Today those pollers call the full
> predicate and pay the ~1403 s path; the change makes it ~9× cheaper and still not cheap
> enough. The cadence was already unachievable — the change only makes it visible.
>
> **And the floor is irreducible for a correct answer.** The currency check must `stat` every
> artifact to detect change; that *is* the cost, and no cache removes it. An on-disk cache
> tier does not help here — it saves the hashing, not the stat-ing.
>
> ### The mechanism: gate on what the writers already maintain
>
> Run-level files already exist whose mtime moves whenever anything happens, so **no new
> tracked artifact is required** — which matters in a change whose thesis is removing them:
>
> ```
> .phenotypic/processing_events.log        appended on every image completion
> .phenotypic/processing_state.json        rewritten on every save_processing_state
> .phenotypic/aggregate_publication.json   rewritten on publication
> ```
>
> A poller stats **two files (~15 ms)** and calls `resolve_run_state` only when one has
> moved. Cadence preserved for the common case, which is *nothing changed*.
>
> ### Two things this must not become
>
> **A sentinel that moves constantly is not a gate.** During an active run the event log is
> appended continuously, so the sentinel fires on every tick and the poller is back to paying
> ~150 s at 2 s intervals — worse than today, because it now also stats the sentinel. **The
> full verification needs its own minimum interval behind the sentinel**, independent of the
> tick. State that interval explicitly rather than leaving it implicit in a debounce.
>
> **A sentinel is evidence that something changed, never evidence of what.** It gates the
> call; it does not answer the question. Nothing may branch on the sentinel's value, and no
> verdict may be derived from it — that would make it a tenth evidence source, which is what
> this change exists to eliminate. It is a *trigger*, and the register in P7 Task 6 should say
> so under its own heading rather than in the tracked-state table.

> ## ⚠ Task 0b: the viewer starts before verification finishes (U-11)
>
> **User ruling: `OutputRoot.discover` comes off the startup path.** It is a `deep`, binding
> call made synchronously at `results_viewer/__main__.py:94`, so today a cold tree hangs the
> GUI for as long as verification takes — measured at **1403 s** on a 6,657-image tree, and
> ~37 s once P2's on-disk tier lands. Neither is a startup budget.
>
> **Most of the scaffolding exists.** `discover` already accepts `cancellation`
> (`OutputDiscoveryCancellation`, documented thread-safe and cooperative) and
> `progress_callback` (phase updates). What is synchronous is the **call site**, not the
> function. That is the smallest version of this change: run it on a thread, feed the
> existing callback into the view, and let the viewer open first.
>
> ### The safety property this must not break
>
> **This whole change exists to stop unverified state being read as verified.** A viewer
> that renders before verification completes is doing exactly that, so the pre-verification
> state must be **visibly distinct** — not a spinner that resolves into content a user cannot
> tell apart from a verified answer afterwards.
>
> Concretely: show what is on disk, say verification is in flight, and make the transition
> observable. **A test must pin that the in-flight state is distinguishable**, or the first
> refactor that "simplifies the loading state" reintroduces exactly the confusion the nine
> evidence sources caused.
>
> ### And it composes with the sentinel
>
> The poll gating decided above and this are the same idea at two timescales: **do not pay
> for a verdict until something has changed, and never block on paying for one.** Together
> they mean a cold GUI opens immediately, verifies behind the view, and thereafter only
> re-verifies when a sentinel moves. Build them in that order — async discover first, since
> the sentinel is pointless while startup still blocks.

## Task 1: `_snapshot_status.py` — 101 lines to ~30

**Files:**
- Modify: `src/phenotypic/gui/_snapshot_status.py`
- Test: `tests/unit/gui/test_snapshot_status.py`

`snapshot_refresh_status` currently branches on `inspect_output_consistency` plus
`snapshot_is_current()` plus `refresh_state_is_current()` — the last of which full-content
SHA-256s seven files on every 5–10 s tick (`_output_root.py:559`).

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize(
    "completion,refresh_supported,expected_color",
    [
        ("complete",   True,  "success"),
        ("incomplete", True,  "warning"),
        ("failed",     True,  "danger"),
        ("active",     True,  "warning"),
        ("complete",   False, "success"),
    ],
)
def test_the_badge_is_a_pure_function_of_completion(
    completion, refresh_supported, expected_color, fake_output_root
):
    """§11: ~30 lines mapping `completion` -> badge, replacing two fingerprints and
    a full re-hash of 7 files per poll."""
    from phenotypic.gui._snapshot_status import snapshot_refresh_status

    fake_output_root.run_state = _state(completion=completion)
    _label, color, _disabled = snapshot_refresh_status(
        fake_output_root, refresh_supported=refresh_supported
    )
    assert color == expected_color


def test_no_badge_refresh_hashes_a_deliverable(fake_output_root, monkeypatch):
    """The 5-10s tick currently full-content SHA-256s measurements.parquet,
    measurements.csv, pipeline.json, curation_labels.parquet, custom_categories.json,
    qc.duckdb and review_state.json. Per tab."""
    import hashlib

    calls = {"n": 0}
    real = hashlib.sha256
    monkeypatch.setattr(
        hashlib, "sha256", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1]
    )
    snapshot_refresh_status(fake_output_root, refresh_supported=True)
    assert calls["n"] == 0
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`snapshot_refresh_status` takes `resolve_run_state(output_root.layout.output_root,
depth="shallow")` and maps `completion` → `(label, color, disabled)`. Delete
`_completion_evidence_status` entirely.

**But `completion` alone cannot replace it (CAN-18).** The function answers **two**
questions today (`_snapshot_status.py:17-63`): run activity/completion, *and* whether the
**bound in-memory snapshot** still matches disk — "Current" versus "Changed on disk"
(`:38-44`, `:55-62`). `completion` answers only the first. A re-finalize over an unchanged
inventory rewrites `measurements.parquet` while `completion` stays `complete`, so a badge
driven by `completion` alone reads **"Current" over a stale snapshot** — and Task 3 then
deletes `refresh_state_is_current` and `consumed_state_fingerprint`, the two things that
answered the second question.

Keep both axes. The badge is a function of `(completion, snapshot_is_current)`:

```python
def test_a_refinalize_over_an_unchanged_inventory_shows_changed_on_disk(bound_output_root):
    """CAN-18. completion stays `complete` across a re-finalize, because the
    inventory did not change -- but the mirror the viewer is holding is now stale."""
    _refinalize(bound_output_root.root)          # rewrites measurements.parquet
    label, color, _ = snapshot_refresh_status(bound_output_root, refresh_supported=True)
    assert color == "danger" and "Changed" in label
```

The parametrized badge test therefore takes both inputs, not just `completion`, and the
`("complete", True, "success")` row asserts `snapshot_is_current` is also true.

**What replaces the deleted fingerprint** is the verification cache's stat sweep over the
deliverables the viewer actually consumed — the same `(size, mtime_ns)` tuples P1 already
records, not a second full-content hash. That is the win: the question survives, the
7-file SHA-256 per tick does not.

**Fold in audit S2 while here** — it is the same function and the same tick. `OutputRoot`'s
frozen `consumed_state_fingerprint` (`_output_root.py:882`) and `CurationLabels`'
self-updating `_source_fingerprint` (`_curation_labels.py:760`) hash overlapping path sets
with different lifecycles, so **marking one colony makes the viewer report its own write as
external drift** — the badge flips to `"Changed on disk"` / `danger`. Exclude GUI-owned
mutable paths from the snapshot fingerprint, the way `snapshot_is_current()` already
deliberately does (`_output_root.py:545-548`). Add a test that a curation click leaves the
badge `success`.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_snapshot_status.py src/phenotypic/gui/results_viewer/_output_root.py \
        tests/unit/gui/test_snapshot_status.py
git commit -m "refactor(gui): the snapshot badge becomes a map over RunState.completion

Spec §11. Also fixes audit S2: marking one colony no longer makes the viewer report
its own write as external drift, and the tick stops re-hashing seven deliverables."
```

---

## Task 2: Delete `_output_consistency.py`

**Files:**
- Delete: `src/phenotypic/gui/results_viewer/_output_consistency.py` (617 lines)
- Modify: every importer
- Test: `tests/unit/gui/`

- [ ] **Step 1: Find every importer**

```bash
grep -rn '_output_consistency\|inspect_output_consistency\|classify_output_consistency\|OutputConsistencyReport\|OutputCompletionEvidence' src/ tests/
```

Expected from the pre-flight count: 4 in `src/`, 3 in `tests/`.

- [ ] **Step 2: Add the regression guard that `contradictory` cannot come back**

> ⚠ **Not a failing test — it passes today and always has.** `Completion` is
> already exactly `{"complete", "incomplete", "failed", "active"}`
> (`sdk_/_state_types.py:28`; re-exported at `_run_state.py:72`), because P1
> defined it with four values and `contradictory` was never among them. As a
> **regression** guard it is worth keeping and fires on a fifth literal. As a
> red-green step it cannot fail, so do not run it expecting red and do not
> treat its green as evidence that Task 2 did anything.

```python
def test_contradictory_is_not_a_state_any_more():
    """Spec §4.3: `contradictory` is DELETED as a state.

    It exists today only because derived counts are cross-checked against each
    other. Once counts stop being evidence, two authorities cannot disagree: there
    is exactly one path to each verdict. This test is the thing that stops it coming
    back -- it was the source of 'run flagged read-only for a reason the user cannot
    act on', which is the user-visible bug this whole change is for.
    """
    import typing

    from phenotypic.sdk_._run_state import Completion

    assert set(typing.get_args(Completion)) == {
        "complete", "incomplete", "failed", "active"
    }


def test_no_module_still_imports_the_deleted_classifier():
    import subprocess

    # ABSOLUTE, from __file__ -- as this document's own greps at Task 0 Step 1
    # and Task 7 Step 4 already do. A relative "src/" greps $CWD/src, which
    # under the sharded regression harness (or any invocation not rooted at the
    # repo) does not exist -- and an empty result is this assertion's PASS
    # condition. The broken form and the correct one were in one document.
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "src"
    assert root.is_dir(), f"grep root does not exist: {root}"
    hits = subprocess.run(
        ["grep", "-rn", "_output_consistency", str(root)],
        capture_output=True, text=True,
    ).stdout
    assert not hits, f"dangling importers of the deleted classifier:\n{hits}"
```

- [ ] **Step 3: Migrate the callers, then delete the file**

- `_snapshot_status.py` — done in Task 1.
- `OutputRoot.discover` (`_output_root.py:178`) — `resolve_run_state(depth="deep")`,
  **one** call replacing the current double read.
- The processing-inventory cache's `cache_reusable` (`report.state == "coherent"`) becomes
  `state.completion == "complete"`.

Then `git rm src/phenotypic/gui/results_viewer/_output_consistency.py`.

### Two predicates that are NOT `completion` in disguise (CAN-17)

The first draft replaced both with one-line `completion` tests. Neither is equivalent, and
**both errors fail open in the dangerous direction.**

**`core_readable`** (`_output_consistency.py:109-114`) is
`not marker_authority_required or aggregate_marker_valid`. Two cases the proposed
`completion in {"complete","incomplete"} and a valid aggregate proof` gets wrong:

- a **legacy** tree is core-readable today *with no aggregate proof at all* — the first
  disjunct carries it;
- an **`active`** output with a valid proof **is** core-readable, and `active` is excluded.

This is the predicate the live-run test `skipif` asks. A false `False` **skips** tests
rather than failing them, and a skip is invisible in a summary line — so this error hides
itself. Keep the disjunction:

```python
def core_readable(state: RunState) -> bool:
    """Whether the canonical aggregate bytes are authorized to read.

    NOT `completion`. A legacy tree with no aggregate proof is readable, and so is
    an ACTIVE run whose previous finalization published one. Both are excluded by a
    naive completion test, and because this gates a `skipif`, the failure is a
    silent skip rather than a red test (CAN-17).
    """
    return not state.marker_authority_required or state.aggregate_proof_valid
```

**`is_read_only`** is `state != "coherent"` (`:93-96`), so today `incomplete` prohibits
mutation. The proposed `completion != "active"` would make **every `incomplete` output
GUI-mutable** — a widening with no spec authority; §4.3 says only that `incomplete` is
"safe to read, safe to resume", which is not "safe to write". Keep the current meaning:
mutation requires `complete`.

```python
def test_an_incomplete_output_is_not_mutable(fake_output_root):
    """CAN-17. `is_read_only` is `state != coherent` today. `completion != "active"`
    would silently grant write access to every incomplete output."""
    fake_output_root.run_state = _state(completion="incomplete")
    with pytest.raises(RuntimeError):
        OutputMutationGuard(fake_output_root).require_mutable()


def test_an_active_run_with_a_valid_proof_is_still_core_readable(fake_output_root):
    fake_output_root.run_state = _state(completion="active", aggregate_proof_valid=True)
    assert core_readable(fake_output_root.run_state)
```

Grep `core_readable` in `tests/` and migrate each site deliberately.

- [ ] **Step 4: Run the GUI suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui tests/gui -q
```

`tests/gui` **is** in `testpaths`; `tests/e2e` is not and needs `PLAYWRIGHT=1`.

- [ ] **Step 5: Commit**

```bash
git rm src/phenotypic/gui/results_viewer/_output_consistency.py
git add -A src tests
git commit -m "refactor(gui): delete _output_consistency.py -- 617 lines, 9 sources, 23 rules

Spec §4.3, §11. `contradictory` is gone as a reachable state, and a test now stops
it coming back. Callers use resolve_run_state(depth=...)."
```

---

## Task 3: `OutputRoot` currency — one shallow verification

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_output_root.py:542,559,882`
- Test: `tests/unit/gui/results_viewer/`

- [ ] **Step 1: Write the failing test**

```python
def test_one_currency_check_replaces_two(fake_output_root):
    """§11: snapshot_is_current() + refresh_state_is_current() -> one shallow
    verification. Two overlapping fingerprints with different lifecycles is audit
    S2, and the fix is one owner, not two better-synchronised ones."""
    from phenotypic.gui.results_viewer._output_root import OutputRoot

    assert not hasattr(OutputRoot, "refresh_state_is_current")
    assert not hasattr(OutputRoot, "consumed_state_fingerprint")


def test_a_chmod_does_not_report_changed_on_disk(bound_output_root):
    """Audit S3, at the consumer. _inventory_is_current compares st_ctime_ns
    (_processing_inventory.py:462), which moves on chmod, chown, hardlink and
    rsync -a -- all routine on a shared HPC filesystem, and each one makes the whole
    binding report 'Changed on disk'."""
    for path in bound_output_root.root.rglob("*.parquet"):
        path.chmod(0o644)
    assert bound_output_root.is_current()
```

- [ ] **Step 2: Implement**

`snapshot_is_current()` delegates to `resolve_run_state(depth="shallow")`.
`refresh_state_is_current()` and `consumed_state_fingerprint` are deleted. Drop
`st_ctime_ns` from `_inventory_is_current` (`_processing_inventory.py:462`) — audit S3.

**Confirm no test depends on ctime-sensitivity before dropping it**
(`grep -rn 'ctime' tests/`), as audit S3 asks.

- [ ] **Step 3: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui tests/gui -q
git add -A src/phenotypic/gui tests
git commit -m "refactor(gui): one currency check, and drop ctime from the inventory sweep

Spec §11, audit S2 and S3. A chmod on GPFS no longer makes a binding report
'Changed on disk'."
```

---

## Task 4: `RunRegistry` claimability — ~200 lines to one call

> ⚠ **The "248 lines" this heading used to carry could not be reproduced.** Measured at
> `ef436461`, the four members Task 4 deletes are `_processing_state_conflict` (83),
> `_publication_evidence_conflict` (60), `_orchestration_state_conflict` (32) and
> `_latest_event_states` (28) — **203**; adding `_read_status_from_manifest` (63) gives
> **266**. Neither is 248, and `_local_completion_evidence_conflict` (104) belongs to
> Task 5. The heading now says ~200 for the four; **re-measure before quoting a deletion
> total in the phase's final commit body**, which is where the ledger says these land.

**Files:**
- Modify: `src/phenotypic/gui/shell/_runs_registry.py:1087,1172,1202,1264`
- Test: `tests/unit/gui/shell/`

> ### ⚠ Task 0 left you an unmade decision in this file — `_runs_registry.py:597-610`
>
> P6 Task 0 privatised `current_run_is_complete`, and **a deletion's blast radius is not
> its task's file list**: this file imported it, so Task 0 converted the call to keep the
> tree importing. It used `resolve_run_state(...)`'s **`deep` default**.
>
> **§9's caller table gives GUI pollers `shallow`, and this is a poller.** That is not a
> regression — `deep` costs exactly what `current_run_is_complete` cost here, the same
> O(N) marker walk — but it is *the cost §9 exists to remove*, now wearing a line that
> looks correct. Task 0 marked it rather than resolving it, because choosing the depth
> needs this task's context.
>
> **Decide it here.** The site also preserves the retired predicate's **tri-state**: it
> branches on `is False` and must not treat a legacy tree as incomplete, so whatever depth
> you choose, the `None` arm stays (see Task 0's tri-state callout).
>
> Recorded in the task text and not only at the call site, because a `Files:` block is an
> index of write targets and cannot represent *"a decision is waiting for you here"* —
> which is register entry 59's whole subject.

- [ ] **Step 1: Write the failing tests**

```python
def test_claimability_is_one_resolve_call(fake_registry):
    """§11: three conflict predicates -> one resolve_run_state call."""
    from phenotypic.gui.shell import _runs_registry as reg

    for gone in (
        "_processing_state_conflict",
        "_publication_evidence_conflict",
        "_orchestration_state_conflict",
        "_latest_event_states",
        "_read_status_from_manifest",
    ):
        assert not hasattr(reg.RunRegistry, gone) and not hasattr(reg, gone), gone


def test_the_event_log_is_replayed_at_most_once(bound_registry, monkeypatch):
    """Audit S5 / §11: _latest_event_states reimplements aggregate_state_from_events
    with different semantics (stage demotion, no inventory fence). Two parsers of one
    append-only log will drift."""
    calls = {"n": 0}
    _count_event_log_reads(monkeypatch, calls)
    bound_registry.claimability("some-output")
    assert calls["n"] <= 1
```

- [ ] **Step 2: Implement**

Replace all three conflict predicates with `resolve_run_state(output_dir, depth="shallow")`
and a `completion`-based decision. Delete `_latest_event_states` and
`_read_status_from_manifest`.

**Where does the stage-demotion rule go?** Audit S5 proposes folding it into the CLI
aggregator as an option. Under §4.2 the event log is no longer evidence, so the demotion
has no consumer — verify that with a grep before deleting rather than after, and if
something still reads it, fold it in rather than dropping it silently.

- [ ] **Step 3: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui tests/gui -q
git add -A src/phenotypic/gui tests
git commit -m "refactor(gui): RunRegistry claimability becomes one resolve_run_state call

Spec §11, audit S5. Three conflict predicates and the second event-log replay are
deleted; one append-only log now has one parser."
```

---

## Task 5: `RunRegistry` local exit, plus DEFERRED D-2

**Files:**
- Modify: `src/phenotypic/gui/shell/_runs_registry.py:591,1058,1306`
- Test: `tests/unit/gui/shell/`

**This is not an optional fold-in — Q2 rule 2 requires it (CAN-24).**

The first draft justified pulling DEFERRED D-2 in as "cheap and adjacent". Round 1 showed
it is a **correctness requirement of the verdict ladder**. Spec §4.1 makes
`gui_launch_owner.json` one of the three liveness authorities, and Q2 rule 2 reads it.
Audit S7 **[verified]**: nothing in the codebase ever deletes or repairs that record, and
`rehydrate_from_sandbox` downgrades it in memory only (`_runs_registry.py:773`,
`persist=False`). So a SIGKILLed GUI pins `status: "running"` **forever**, and rule 2 is
unsound as written — it reports `active` for a run nothing is working on.

The ladder's obligation therefore lives in **P1 Task 5** (added there: a verdict-matrix row
asserting a dead `pid` does not yield `active`). The **repair** lives here, where
`_assert_output_claimable_locked` is rewritten. Both are required; neither substitutes for
the other.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_eight_branch_refusal_tree_becomes_advisories(fake_registry):
    """§11: '8-branch refusal tree -> resolve_run_state(deep); refusals become
    advisories.' A refusal the user cannot act on is the bug; an advisory they can
    read is the fix."""
    state = fake_registry.local_exit_state("some-output")
    assert state.completion in {"complete", "incomplete", "failed", "active"}
    assert isinstance(state.advisories, tuple)


def test_a_sigkilled_gui_does_not_lock_the_output_forever(tmp_path):
    """DEFERRED D-2 / audit S7 [verified]: nothing in the codebase ever deletes or
    repairs gui_launch_owner.json. A SIGKILLed GUI leaves status: 'running';
    rehydrate_from_sandbox downgrades it IN MEMORY ONLY (_runs_registry.py:773,
    persist=False), and _assert_output_claimable_locked then refuses the output
    forever, with no UI affordance to clear it.

    The record already stores pid and started_at. Use them."""
    _write_owner_record(tmp_path, status="running", pid=_a_dead_pid(), started_at="2020-01-01")
    registry = _registry(tmp_path)
    registry.assert_output_claimable(tmp_path)   # must not raise


def test_a_live_owner_still_refuses_the_claim(tmp_path):
    """The liveness check must not become a rubber stamp -- an owner whose process
    is alive still owns the output."""
    import os

    import pytest

    _write_owner_record(tmp_path, status="running", pid=os.getpid(), started_at="2026-09-03")
    with pytest.raises(RuntimeError):
        _registry(tmp_path).assert_output_claimable(tmp_path)
```

- [ ] **Step 2: Implement**

`_local_completion_evidence_conflict`'s eight refusal strings become advisories on the
`RunState`. `_assert_output_claimable_locked` gains a liveness check: an owner record whose
`status` is non-terminal but whose `pid` is not alive is downgraded **and persisted** — the
`persist=False` at `_runs_registry.py:773` is precisely what makes today's downgrade
useless.

Use `os.kill(pid, 0)` guarded for `ProcessLookupError` / `PermissionError`; a `pid` that
has been recycled is a real but bounded risk, and `started_at` bounds it further — treat a
record older than the boot time as dead regardless.

- [ ] **Step 3: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui tests/gui -q
git add -A src/phenotypic/gui tests
git commit -m "refactor(gui): local-exit refusals become advisories; repair a stale owner record

Spec §11 plus DEFERRED D-2, folded in deliberately: this task rewrites the exact
predicate that caused the permanent dead-end, and under the Q2 ladder a stale owner
record masks incomplete as active. The record already stored pid and started_at;
nothing read them."
```

---

## Task 6: SLURM observer — call sites only

> ### ⚠ BEFORE YOU START: the observer's poll loop got more expensive in P3, by design
>
> **If you are here because the observer feels slow, this is why — it is not a bug you
> introduced and not one to fix in this task.**
>
> P3 collapsed the stage-3 marker into the per-image record, so
> `stage3_completion_exists` went from a bare `is_file()` to `read_image_record` +
> `json.loads` + two dict lookups (`_cli_staged_resume.py:135-160`).
>
> The spec's efficiency argument for that collapse — *"one JSON read replaces one read plus
> three `is_file()` probes across three directory trees"* (`design.md:578`) — **is true of
> the per-image decision point it describes, which was already reading a marker.** It says
> nothing about the callers that were doing *zero* reads, and there are three, all
> whole-inventory:
>
> | Caller | Scope | Short-circuits? |
> |---|---|---|
> | `gui/run_console/_slurm_observer.py:1338-1354` `_all_stage3_markers_exist` | every image of every dataset in `job_metadata.json`, **on the polling path** | yes, on first miss |
> | `_cli_staged_orchestration.py:264-289` `completed_inventory_images` | per dataset, per poll | no |
> | `_cli_staged_controller.py:76-98` retryable/terminal split | per controller round | no |
>
> On a 6,000-image run on GPFS a poll that was 6,000 `stat()`s is now 6,000
> `open`/`read`/`close`/`json.loads`. **Not measured** — flagged from reading, and recorded
> because the observer is user-facing and the cost is invisible in the spec's framing.
>
> **The collapse is the design; do not undo it here.** If it needs addressing, the shape is
> a cached or batched inventory read at these three call sites, and it is its own change
> with its own measurement — not a revision of §6.1.


**Files:**
- Modify: `src/phenotypic/gui/run_console/_slurm_observer.py:536,909,1312`
- Test: `tests/unit/gui/run_console/`

**Scope discipline is the whole of this task.** Spec §2.2 and DEFERRED D-1 put the
observer's decision tree, its 30-second reconciliation grace window and its `squeue`/`sacct`
state ranking **out of scope**. Only ~185 of its lines are filesystem-derived; the rest is
scheduler domain, it is the least testable code in the GUI, and its failure mode ("run
stuck in `reconciling`") is directly user-visible.

**Change exactly three things:**

1. the two `_cli_completion` call sites (`current_success_counts`, `valid_run_completion`) →
   one `resolve_run_state(depth="shallow")`
2. `_all_stage3_markers_exist` → reads `stages.stage3` from the P3 record
3. `_manifest_is_complete` → deleted

- [ ] **Step 1: Write the failing test**

```python
def test_the_observer_tick_does_not_hash_anything(fake_observer, monkeypatch):
    """Audit §4: the 2-second daemon tick currently runs valid_run_completion ->
    current_run_is_complete -> current_success_counts -> _walk_current_success, which
    calls valid_image_success once per image, each re-hashing the embedded
    measurements parquet AND the overlay PNG. Then current_aggregate_is_current walks
    it all AGAIN. On a 10,000-image run that is ~2-3 x 10^4 file hashes every two
    seconds."""
    import hashlib

    calls = {"n": 0}
    _count_sha256(monkeypatch, calls)
    fake_observer.tick()
    assert calls["n"] <= 8


def test_the_decision_tree_is_untouched():
    """Spec §2.2, DEFERRED D-1. This test exists to make scope creep fail CI rather
    than fail review."""
    import inspect

    from phenotypic.gui.run_console._slurm_observer import SlurmLifecycleObserver

    source = inspect.getsource(SlurmLifecycleObserver._observe_record)
    assert "resolve_run_state" in source
    assert "_manifest_is_complete" not in source
    # The grace window and squeue/sacct ranking stay exactly as they are.
    assert "GRACE" in source or "grace" in source
```

- [ ] **Step 2: Implement, run, commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/run_console tests/gui -q
git add -A src/phenotypic/gui tests
git commit -m "refactor(gui): the observer tick asks resolve_run_state once

Spec §11, scoped by §2.2 and DEFERRED D-1: two call sites and the Stage-3 probe
move; the decision tree, grace window and scheduler polling are untouched. The 2s
tick stops walking every image twice and hashing every artifact."
```

---

## Task 7: The pure deletions, and §11.2's fold-ins

**Files:** as listed in the deletion ledger, items 6–9.

Each of these is **evidence-backed** — the audit verified the caller counts by hand. Re-run
each grep before deleting; a claim from 2026-09-03 is not a claim about the tree you are
editing.

- [ ] **Step 1: Verify each claim, then delete**

```bash
grep -rn 'monitor_slurm_jobs'        src/ tests/    # expect 0 outside the file itself
grep -rn 'DashboardManifestKey.VERSION' src/ tests/ # expect 1 write, 0 reads
grep -rn 'browse_cache_base\|cache_png_path\|init_cache\|wipe_cache' src/ tests/
grep -rn 'read_run_manifest\|load_master_measurements\|resolve_best_pipeline_path\|resolve_qc_dir\|recompile_status_dir\|chunk_parquet_path\|checkpoint_lock_path\|chunk_manifest_path' src/ tests/
```

**If a grep disagrees with the audit, stop and record it** — either the tree moved or the
audit was wrong, and both matter more than the deletion.

Two of the eight `_io_constants` resolvers "claim in their docstrings to replace inline
blocks that still exist" (S21). For those, route the inline block through the helper and
keep it, or delete both. Deleting the helper while the inline block survives is the worse
of the three outcomes.

- [ ] **Step 2: Fold in §11.2 — inside files already being rewritten**

Only these. Everything else in DEFERRED's churn table stays deferred:

- Hand-joined `.phenotypic/aggregate_publication.json` in the GUI → use
  `aggregate_publication_marker_path()` (audit S8, `_output_consistency.py:380` — the file
  Task 2 deleted, so this is now wherever its caller moved).
- The ~17 shadow state filenames into `_io_constants` (S9). Two of them —
  `staged_orchestration.json` and `staged_finalization_complete.json` — are **double-spelled
  across the CLI/GUI boundary**; those two are the ones that matter.
- `DIR_PROGRESS` at the two literal sites in `phenotypicCLI.py:839,841`, in the same file
  that already imports and uses it at `:943` (S15).
- The recompile `task_manifest.json` and `job_metadata.json` naive writers made atomic
  (S11) — these **are** polled by concurrently launched SLURM workers, including the
  unlocked write at `phenotypicCLI.py:3385`.

- [ ] **Step 3: Delete the three tests that assert on text, not behaviour (CAN-31)**

`_canonical_digest`'s collapse **moved to P1 Task 4** (CAN-29): hoisting a pure function
into `sdk_` up front is less total work than adding a third copy plus a keeper test here and
then deleting both. Nothing to do for it in this phase.

Instead, delete three tests this plan proposed that cannot fail for the right reason:

| Test | Why it goes |
|---|---|
| `test_run_state_exports_no_writer` (P1 T1) | asserts `__all__` name **prefixes**. A writer called `record_stage` or `persist_x` passes, and the prefix list is itself a tracked list needing sync with naming fashion. |
| `test_no_module_still_imports_the_deleted_classifier` (P6 T2) | a CWD-relative `subprocess` grep. A deleted module with a live importer is an `ImportError` the suite already raises; the relative path makes this pass or fail on where pytest was invoked. |
| `test_the_decision_tree_is_untouched` (P6 T6) | `"GRACE" in source`. Fails on a harmless rename, passes if you gut `_observe_record` and leave the word in a comment. Its goal — "make scope creep fail CI rather than fail review" — is a review concern, and this is the weakest possible enforcement of it. |

**Keep the INV-LAYER AST test exactly as written.** It is structural, it can fail, and P1
Task 1 Step 6 proves both the module-scope and lazy-in-function forms trip it. If a
write-side structural guarantee is still wanted for `_run_state.py`, check the AST for
`open(..., "w")`, `Path.write_*` and `os.replace` — a real check, where the prefix list was
a spelling check.

The two advisory tests (`any("migrate" in advisory)`, `any("metadata" in advisory)`)
substring-match human prose, which makes advisory wording a de-facto API. Give advisories a
small closed set of codes plus optional detail, and assert on the code.

> ### While you are here: one advisory is written and can never be shown
>
> **Test-review finding 5.** `describe_conversion_advisory`'s `UNREADABLE_STATE` branch
> (`sdk_/_schema_shape.py:415-421`) composes a careful reader-facing message and **no input
> reaches it**. Its only caller is `_advisories`, invoked at `resolve_run_state`'s tail — but
> `resolve_run_state` returns early whenever `_read_state_config` yields `None`, and *every*
> payload that makes `_classify` return `UNREADABLE_STATE` also makes `_read_state_config`
> return `None`. The reviewer checked the one shape that might have escaped (a valid object
> with `version`/`datasets` but no `config`): still an early return, and `_classify` calls
> that `CONVERT` anyway.
>
> **Do not just delete the branch.** The early return hardcodes:
>
> > *"No readable processing state under this directory, so it has no run identity and no
> > completion to establish."*
>
> and that sentence is **wrong for half the cases it covers**. It conflates two situations a
> user must tell apart:
>
> | Situation | What the user has | What they should be told |
> |---|---|---|
> | No state file at all | a fresh or empty directory | the generic sentence — correct today |
> | State file present but corrupt | truncated JSON, `null`, `[]`, `""` | *"…is not readable as a state file… **Repair or remove that file; conversion cannot recover one it cannot read.**"* |
>
> Telling someone whose `processing_state.json` is truncated that there is "no readable
> processing state under this directory" suggests **absence**, not damage — so they go looking
> for a missing file instead of repairing the one they have. The better message already
> exists, fully written; it is simply shadowed.
>
> **The fix is at the early return, not in `_schema_shape.py`:** consult
> `describe_conversion_advisory(output_dir)` there and use its result when non-`None`, keeping
> the generic sentence as the fallback for genuine absence. Four lines, and it makes seven
> lines of already-written text reachable.
>
> **It lands here rather than in P2 because `_run_state.py` was under active edit when this
> was found**, and this step is already reworking advisory shape — a codes-plus-detail scheme
> has to decide what code an unreadable state file carries, which is the same question. Give
> it a test that distinguishes the two situations by code, and a mutation: collapsing the two
> back into one message must go red, or the distinction decays the first time someone
> simplifies the branch.

- [ ] **Step 4: The whole-tree predicate assertion — and commit**

Task 0's gate test was scoped to `_cli` + `sdk_` because the GUI's holders were still live
at that point (gen-r4 N-1). Tasks 1–6 have now migrated them, so the unrestricted form can
finally pass. Add it here, in the same file, beside the scoped one:

```python
def test_no_completion_predicate_survives_anywhere():
    """The unrestricted form of test_only_one_completion_predicate_survives.
    Scoped to _cli + sdk_ in Task 0 because gui/ had not been migrated yet; by the
    end of this phase the whole tree must be clean."""
    import subprocess

    from pathlib import Path

    src = Path(__file__).resolve().parents[3] / "src"
    hits = subprocess.run(
        ["grep", "-rn",
         "current_run_is_complete\\|current_success_counts\\|current_aggregate_is_current",
         str(src)],
        capture_output=True, text=True,
    ).stdout.strip()
    assert not hits, f"a completion predicate survives:\n{hits}"
```

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_completion_split.py -q
git add -A src/phenotypic tests
git commit -m "refactor: delete the nine-source completion machinery

Spec §11.1, §11.2. Net -<N> lines.

Deleted:
  _output_consistency.py                        -617
  RunRegistry three claimability predicates     -<n>
  _local_completion_evidence_conflict tree      -<n>
  _latest_event_states, _read_status_from_manifest, _manifest_is_complete  -<n>
  sdk_/monitor_slurm_jobs.py (0 importers)      -241
  browse/_source_render.py dead cache API       -<n>
  eight zero-caller _io_constants resolvers     -<n>
  DashboardManifestKey.VERSION (1 write, 0 reads) -<n>

Folded in (§11.2): aggregate_publication_marker_path at the GUI site; 17 shadow
filenames into _io_constants, including the two double-spelled across the CLI/GUI
boundary; DIR_PROGRESS at phenotypicCLI.py:839,841; the four naive control-manifest
writers made atomic.

Three text-asserting tests removed (CAN-31). The completion-predicate gate is now
unrestricted -- Task 0 could only scope it to _cli + sdk_ because gui/ had not been
migrated yet.

Every caller count was re-grepped before deletion, not taken from the audit."
```

- [ ] **Step 5: Phase gate — the full suite**

```bash
uv run mypy src/phenotypic
uv run ruff check --fix <every path this phase touched>
```

Then the full `tests/unit` **and** `tests/gui` suites, as a Slurm job via the
**`run-phenotypic-test`** and **`slurm-job`** skills. This is the phase most likely to break
something distant.

Compare against the recorded baseline: four failures are known pre-existing, three of which
fail only on compute nodes. **A fifth failure is this phase's**, not the baseline's.

---

## Task 7b: Convert the GUI's master readers to P4's v1/v2 helper

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_output_root.py:320`,
  `_curation_labels.py:417`, `:763`, `_error_tab/_publication.py:125`,
  `_qc_tab/review/_data.py:84`
- Test: alongside each module's existing suite

**Added by user ruling (2026-09-06), which split this work between P4 and P6.** P4 changes
the master's shape — after §7.3's inversion it carries **intrinsic identity only**, with
user metadata living in the mirror — and creates the helper that tells the two shapes apart.
**P4 owns the helper and the `sdk_` readers; P6 owns these GUI readers**, because P6 owns
this surface and is already rewriting them. The ruling is recorded in
[`phase-4-finalize-run.md`](phase-4-finalize-run.md) Task 3 Step 6; this task is its other
half, and it exists because a split written down in only one plan is how the second half
never happens.

**The helper, created in P4** (`sdk_/_master_io.py`):

```python
# V1/V2 MASTER DISCRIMINATION -- DELETE WHEN: no run predating the P4 inversion is
# still readable, i.e. every master in the wild was written by finalize_run's
# post-inversion path. A v1 master carries Metadata_* user-metadata columns
# because the join happened per-image; a v2 master does not, because the join
# moved to finalization. Nothing else distinguishes them, and nothing stamps them.
def master_carries_user_metadata(frame: "pl.DataFrame") -> bool: ...
```

**There is no schema stamp** — that was cut by user ruling, because the master already
self-describes and a stamp would be a second home for a fact the file carries. Do not add
one here, and do not write a reader that expects one.

- [ ] **Step 1: Establish which of the five readers actually branch**

Most do not. Curation keys on dataset / image / object-label — all intrinsic — so it should
be unaffected; `_processing_inventory.py:202,373` only add the path to an existence/mtime
inventory and never read the frame at all. **The readers that matter are those that filter
or group on a `Metadata_*` column**, which on a v2 master return **empty rather than
raising** — §7.3's "one genuinely dangerous failure mode in §7".

```bash
grep -rn 'Metadata_' src/phenotypic/gui/results_viewer/_output_root.py \
  src/phenotypic/gui/results_viewer/_curation_labels.py \
  src/phenotypic/gui/results_viewer/_error_tab/_publication.py \
  src/phenotypic/gui/results_viewer/_qc_tab/review/_data.py
```

**Expected: a non-empty list of candidate sites, each classified in the plan before any
edit** — "branches on user metadata" or "intrinsic only, no change". A reader you convert
without needing to is churn; one you skip because it *looked* intrinsic is the failure this
task exists to prevent.

- [ ] **Step 2: Convert only those, and test both shapes**

For each converting reader, add a test that feeds it **a v1 master and a v2 master** and
asserts the outcomes differ in the intended way — not that the column sets differ, which is
true by construction and proves nothing about behaviour.

- [ ] **Step 3: Verify no reader still assumes the v1 shape**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui tests/gui -q
```

**Expected: 0 failures and a non-zero collected count for each path.** A bare PASS does not
distinguish "ran and passed" from "collected nothing" — see P4 Task 2 Step 4 for why that
distinction is load-bearing in this change.

> **One overlap to check before deleting anything here:** P4 Task 4 Step 4 deletes
> `load_master_measurements` as part of D8, and Task 7's ledger above greps for the same
> symbol. If P4 landed first the grep returns 0 and that ledger item is already discharged —
> which is the correct outcome, not a missing deletion. Confirm which, rather than
> re-deleting or recording it as skipped.

---

## Task 8: Record what the GUI tracks, in `gui/CLAUDE.md`

**Files:**
- Modify: `src/phenotypic/gui/CLAUDE.md`

**This task is not optional and is not a docs-polish afterthought.** The change deletes
nine evidence sources and four classifiers from the GUI; a module guide that still
describes them is worse than no guide, because the next reader will trust it. The
distinction that matters and is nowhere written down today is **what the GUI *owns* versus
what it merely *reads*** — and the change moves that line.

- [ ] **Step 1: Add a "State the GUI tracks" section**

Place it after `### Flask app.server.config keys` (`gui/CLAUDE.md:176`), which already
enumerates the process-wide singletons. Three tables, and the split between them is the
content:

**(a) GUI-owned durable state — the GUI is the writer.**

| Artifact | Path | Written by | Read back by | Notes |
|---|---|---|---|---|
| Launch ownership | `.phenotypic/gui_launch_owner.json` | `_persist_record_locked` (`shell/_runs_registry.py:1306`) | the CLI's freshness guard, and `resolve_run_state` rule 2 | **A §4.1 liveness authority.** Carries `pid` + `started_at`; P6 Task 5 added the liveness check that makes rule 2 sound. |
| Curation labels | `deliverables/qc/curation_labels.parquet` | `_curation_labels.py` | the CLI re-emits from it | GUI is the primary writer; keyed on intrinsic identity, so §7's inversion does not touch it |
| Custom categories | `deliverables/qc/custom_categories.json` | GUI only | GUI only | — |
| Review state | `deliverables/qc/review_state.json` | GUI | GUI | **The CLI deletes it at finalize** (`_cli_output_manager.py:1238`) — a fresh run resets review progress |
| Verified rows | `deliverables/verified.parquet` | GUI only | GUI only | finalize never writes it |
| Error categories | `deliverables/errors/<category>.parquet` | **both** | both | dual-owned, documented |

**(b) GUI-owned ephemeral state — dies with the process or the tab.** The 136 `dcc.Store`s
(only 35 declare `storage_type`), the 14 `dcc.Interval`s, the `app.server.config`
singletons already listed above, and the sandbox caches. State the rule the audit found
and P6 did not change: **one bound output per process, shared by every browser tab.**

**(c) State the GUI reads and must never write.** Everything under `.phenotypic/` except
the owner record. Say it once, plainly, and name the one function that answers questions
about it: `resolve_run_state(output_dir, depth=...)`.

- [ ] **Step 2: Delete what is no longer true**

Grep `gui/CLAUDE.md` for the deleted machinery and remove or rewrite each mention:

```bash
grep -n 'consistency\|coherent\|contradictory\|manifest\|_output_consistency\|snapshot_is_current\|refresh_state_is_current' src/phenotypic/gui/CLAUDE.md
```

The four-state vocabulary (`coherent`/`active`/`incomplete`/`contradictory`) is gone;
`contradictory` no longer exists at all. Replace with the four verdicts and a pointer to
`resolve_run_state`.

- [ ] **Step 3: State the import rule that replaces the 25-symbol seam**

Audit §7: the GUI imports 25 private `phenotypic._cli` symbols across 9 modules, and that
is how the O(N)-hashing completion predicate ended up on a 2-second timer. After this
change the rule is: **the GUI imports readers from `phenotypic.sdk_`, never from
`phenotypic._cli`.** List the remaining legitimate `_cli` imports (the SLURM lifecycle
helpers the observer needs, per DEFERRED D-1) so a reader can tell the survivors from a
regression.

- [ ] **Step 4: Verify every claim you just wrote**

Each path and function name gets a `grep`. A module guide asserting a `file:line` that
moved is the failure this whole change is about.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/CLAUDE.md
git commit -m "docs(gui): record what state the GUI owns, and what it only reads

The change deletes nine evidence sources and four classifiers; the module guide
still described them. Adds the owned/ephemeral/read-only split, which was nowhere
written down, and the sdk_-not-_cli import rule that replaces the 25-symbol seam."
```
