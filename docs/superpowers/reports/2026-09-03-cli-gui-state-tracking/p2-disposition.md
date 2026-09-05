# P2 gate — finding disposition

**What this is.** One row for every finding raised by P2's three independent
gate reviewers, each closed as exactly one of **FIXED**, **WITHDRAWN**,
**DEFERRED**, or **OPEN**. It exists so that no finding is closed by silence —
the failure mode where a report is read, most of it acted on, and the remainder
disappears because nobody wrote down that it was left.

**38 findings, 39 rows.** `REUSE-F12` occupies two rows on purpose; see its
entry.

**Sources.** `p2-implementation-review.md` (13), `p2-spec-adherence.md` (13),
`p2-check-reuse.md` (12). Dispositions were established by reading the working
tree and the branch history, not by recalling the conversation — several rows
below correct what I believed before checking.

---

## Read this first: the three reports collide on finding IDs

**Two of the three number their findings `F1`…`Fn`.** `p2-implementation-review`
has F1–F13 and `p2-check-reuse` has F1–F12, so a bare "F2" or "F10" names two
different findings, and the two happen to be **live at the same time with
opposite dispositions**:

| Bare ID | `p2-implementation-review` | `p2-check-reuse` |
|---|---|---|
| **F2** | `mint_run_identity` returns an identity no reader can reproduce | `inventory_digest` has two producers that cannot agree |
| **F10** | the coarse-`mtime` false-current window is now persistent | `_cli_completion` re-spells its own marker check |

This is not hypothetical. Commit `fcf99a19`'s subject line reads *"F2 takes
option (a), F10 is withdrawn"* and **names neither report**. Both halves resolve
only by inspection: "option (a)" is a phrase that appears in `p2-check-reuse`'s
F2 and nowhere in the implementation review, and the withdrawal's body argues
about `mtime` granularity, which is the implementation review's F10. So the
commit's F2 and its F10 are **from different reports** — a reader who assumes
one report gets one of the two wrong.

**This file therefore prefixes every ID**: `IMPL-`, `SPEC-`, `REUSE-`. Later
phases should do the same, and a future gate should not issue a second `F`
series.

---

## `p2-implementation-review.md` — 13 findings

| ID | Finding | Disposition | Evidence |
|---|---|---|---|
| **IMPL-F1** | `--overwrite` deletes the restart counter *after* the identity was minted from it — a live SLURM run never reported `active` for its whole lifetime | **FIXED** `41acbf0c` | Mint moved below the overwrite block. Reverting it re-breaks `test_run_identity.py`'s restart-epoch fence cases; the guard `not resume and not restart and not measure_only` preserves dominance. |
| **IMPL-F2** | `mint_run_identity` returns a `RunIdentity` no reader can reproduce | **FIXED** `fcf99a19` | Same defect as `REUSE-F2`/`SPEC-E1`, resolved once by option (a): `inventory_digest=""`, reader-owned, `_inventory_digest_for` deleted. `test_the_minted_inventory_digest_is_empty_and_that_is_the_contract` + its mutation. |
| **IMPL-F3** | `_metadata_digest_for`'s stated guarantee fails on the commonest path (continuation without `--metadata`) | **FIXED** `41acbf0c` | Guarded fallback to `metadata_csv_deliverable_path`. Reverting it makes §7.4's late-metadata fence fire on every continuation. |
| **IMPL-F4** | the metadata agreement test restated the computation it was testing | **FIXED** `41acbf0c` | Test now drives `_snapshot_metadata_csv` across three arms, including the continuation case that used to fail. |
| **IMPL-F5** | `clear_verification_cache`'s docstring claims a wiring that does not exist | **FIXED** `41acbf0c` (docstring only) | Same finding as `SPEC-C4`. Wiring deliberately **not** added — an unruled behaviour change, and unnecessary since `clear_machine_state` removes tier 2 with the directory. |
| **IMPL-F6** | `_schema_shape`'s "three of the five signals" note is stale | **OPEN** | Same finding as `SPEC-C3`. **Verified still stale**: `_schema_shape.py:76` says *"does not yet write `restart_epoch`, so three of the five signals"*; `create_initial_state` writes it, so two fire. Benign (`SCHEMA_GATE_ARMED` is `False`) but it is a claim about what a check does, in the docstring of the flag gating both consumers. |
| **IMPL-F7** | the recompile-worker equality removal cites a supplier that does not supply | **OPEN — and it was closed by silence** | Named in neither the fixed list nor the deferred list of `41acbf0c`. **Verified still wrong**: `_cli_recompile_worker.py:236` cites `phenotypicCLI.py:3464` as passing both values; `_wait_for_recompile_finalizer_status` (`phenotypicCLI.py:3608-3616`) takes **no `attempt_id` parameter**. The line number has since drifted too — the call is now at `:3473`. See *Dissents*. |
| **IMPL-F8** | a `--restart --dry-run` burns an epoch and creates `.phenotypic/` | **FIXED as documented** `41acbf0c` | Deliberately a comment, not a change: monotonic so the cost is bounded, `--restart` has already written via `clear_machine_state`, and a conditional bump would put an exception on a counter whose whole value is being unconditional. |
| **IMPL-F9** | the resume comment's second justification has no replacement | **DEFERRED** | Named in `41acbf0c` as touching text `IMPL-F2`'s resolution may move. That resolution has now landed, so this is **unblocked** and owed a disposition by the next phase that edits the resume path. |
| **IMPL-F10** | the coarse-`mtime` false-current window is now persistent | **WITHDRAWN** `fcf99a19` | The mechanism is real; the precondition does not occur. No path writes a tracked artifact twice within a pass, and GPFS `/bigdata` measured 0/200 same-size back-to-back writes sharing an `mtime_ns`. What survives is one line in `_verification_cache.py` recording *why* `(size, mtime_ns)` suffices. **See *Dissents* — I am not disputing the withdrawal, but its supporting claim is incomplete.** |
| **IMPL-F11** | `schema_version` check satisfied by `true` (`True == 1`) | **FIXED** `41acbf0c` | `_is_plain_int` guard; the module already rejected `bool` in three other places, so the omission was inconsistent rather than reasoned. |
| **IMPL-F12** | `_cli_identity`'s import-weight rationale does not match its imports | **DEFERRED** | Named in `41acbf0c`. Unblocked by `IMPL-F2`'s resolution; owed by the next phase editing that module. |
| **IMPL-F13** | three spellings of "finalization input digest" coexist | **OPEN — and `REUSE-F4` sharpened it rather than closing it** | Deferred pending `IMPL-F2`, which has now landed. **Verified**: `_cli_identity.py:318-325` still hand-builds the versioned dict inline, which is now a **fourth** producer, because `REUSE-F4` created `sdk_._run_state.finalization_input_digest(config)` as the canonical one. The mint cannot call it as-is — it holds an `ExecutionConfig` (attributes) and the helper takes a state `config` block (`.get`) — so the fix is an adapter or a written reason, not a rename. |

---

## `p2-spec-adherence.md` — 13 findings

Category **A** is empty by the reviewer's own count, so the 13 are B×4, C×4,
D×4, E×1.

| ID | Finding | Disposition | Evidence |
|---|---|---|---|
| **SPEC-B1** | `test_a_new_image_does_NOT_mint_a_new_generation` did not land | **OPEN** | **Verified absent**: `grep -rn` over `tests/` returns nothing. No commit dispositions it. |
| **SPEC-B2** | `test_a_metadata_edit_does_NOT_mint_a_new_generation` did not land | **OPEN** | **Verified absent.** Same. |
| **SPEC-B3** | the migrated-tree gate test was replaced by a weaker one, and the stronger one would fail | **OPEN** | The reviewer's own analysis shows this is a **phase-ordering conflict in the plan**, not an implementer's omission: removing `datasets.{completed,failed}` is §4.2's job in P3+, and until then `requires_conversion` returns `CONVERT` on signal 3. Its recommended disposition — amend `phase-2-identity-schema.md:616`, re-point two code comments at signal 4 only, move the assertion to P7 Task 5 — is **not yet done**. |
| **SPEC-B4** | `test_a_stale_slurm_worker_cannot_publish` did not land, and the guard has no test at all | **OPEN** | **Verified absent.** The reviewer notes the guard itself is untested, which makes this the most consequential of the three missing tests. |
| **SPEC-C1** | the `scheduler_epoch` collapse and the `publish_image_success` rename were not done | **WITHDRAWN** (requirement), user-ruled | `design.md:323-345` amends §5.1 to *"the five-token collapse is achievable ZERO times"*, with a writer-by-writer table; register entry 19. Task 4 Step 4's other branch was **done** (`9a364dc4`, the self-comparison deletion). One residual recommendation is unactioned — see *Still owed*. |
| **SPEC-C2** | the `per_image_config_digest` rename stopped short in the four places a reader looks up the formula | **FIXED** `e1010b8e` | Formula sites in `design.md` and `phase-7-migrate-mode.md` updated; the §5.4 row now states why the second component is *not* `scientific_config_digest`. Register entry 14. |
| **SPEC-C3** | `_schema_shape.py:74-78` is now false, and P2 made it false | **OPEN** | Duplicate of `IMPL-F6`; **verified still stale**. Two reviewers reaching the same site independently is the strongest signal in the pair, and it is still unfixed. |
| **SPEC-C4** | `_verification_cache.py:553` claims a wiring P2 did not do | **FIXED** `41acbf0c` | Duplicate of `IMPL-F5`. Note the reviewer's point worth preserving: the effect is nil for a *different* reason than the docstring named — a restart moves the generation, which moves `RunIdentity.digest()`, so tier 1 is **unreachable** rather than cleared. |
| **SPEC-D1** | `VERIFICATION_CACHE_VERSION` and a seventh fall-through case | **WITHDRAWN** | Reviewer's own disposition: §9.1 constrains *what must* fall through, not *what may not*, and every added cause moves toward `deep`. Tested with a mutation. |
| **SPEC-D2** | `derive_processing_generation` not in §5.2's surface | **WITHDRAWN** | Latitude: §5.1 constrains the formula, U-7 forces a config-free entry point for migrate. The mapping-vs-`‖` digest choice is inside that latitude and has never been on disk. |
| **SPEC-D3** | the mint-once guard is an instance attribute | **WITHDRAWN** | Latitude: the plan specifies the guard and its `RuntimeError`, not the mechanism. No serialization of `ExecutionConfig` exists that could leak the flag. Tested with a mutation. |
| **SPEC-D4** | the SLURM lifecycle record grew a `restart_epoch` field | **WITHDRAWN** | A *forced* choice, not a free one: rule 2 requires an authority to report work in flight for the **current identity**, which is uncomputable without it. Tested in both directions, writer and fence separately. |
| **SPEC-E1** | `_inventory_digest_for` returns a constant on the default path and disagrees with the other producer | **FIXED** `fcf99a19` | The same defect as `IMPL-F2` and `REUSE-F2`; three reviewers, three routes, one site. Deleted rather than repaired. |

---

## `p2-check-reuse.md` — 12 findings, 13 rows

| ID | Finding | Disposition | Evidence |
|---|---|---|---|
| **REUSE-F1** | "is the run proof valid?" has seven implementations; four omit the version check | **FIXED** `4b7ddeba` | `run_proof(output_dir)` (the strict one) and `run_proof_is_current(output_dir)` exported from `sdk_/_run_state.py`. `test_run_proof_refuses_a_version_it_cannot_interpret` fails if the version check is dropped. **Call sites not migrated — P6 Task 7.** |
| **REUSE-F2** | `inventory_digest` has two producers that cannot agree on any input | **FIXED** `fcf99a19` | User-ruled option (a). See `IMPL-F2` / `SPEC-E1`. |
| **REUSE-F3** | per-image record validity has five readers, two provably divergent, and the keeper test excludes exactly that input | **FIXED** `4b7ddeba` | `marker_rejection` exported; `valid_image_success` reads it and keeps its `bool` signature. `test_the_sdk_reader_agrees_with_the_cli_validator[migrated-provenance]` — **added and run red before the fix**, green after. |
| **REUSE-F4** | `finalization_input_digest` has four spellings; reader tolerates two, validator one | **FIXED** `4b7ddeba` | `finalization_input_digest(config)` (publishers) and `accepted_finalization_digests(config)` (validators) as a pair. `test_the_publishers_digest_is_one_of_the_validators_accepted_ones` carries a P4 tripwire. **See `IMPL-F13`** — this created the canonical producer that the minter still does not use. |
| **REUSE-F5** | `RunIdentity.digest()` hand-rolls `canonical_digest` with `ensure_ascii` defaulted | **FIXED** (pre-`4b7ddeba` increment) | `digest()` is now `canonical_digest(payload)`. DF-19: every proof on disk was written with `ensure_ascii=False`, so the hand-rolled copy was the deviation. Mutation in `p2_task1`. |
| **REUSE-F6** | the fenced identity field set is enumerated twice with nothing keeping them in step | **FIXED** (same increment) | `IDENTITY_DIGEST_FIELDS` moved to `_state_types.py`; `digest()` derives from it. Derived, not asserted-equal — `test_the_fenced_field_set_has_exactly_one_home` + mutation. |
| **REUSE-F7** | the stated reason `_inventory_digest_for` cannot be shared does not exist | **WITHDRAWN — dissolved by `REUSE-F2`** | Probe 1 showed the cited import cycle is absent (closure of 10 modules, containing neither). Moot once the function was deleted; no import was needed. |
| **REUSE-F8** | "does this artifact still match its descriptor?" has three copies | **FIXED** `4b7ddeba` | `fenced_artifact_path` exported and read by `valid_image_success`. `test_fenced_artifact_path_dispatches_on_kind`. |
| **REUSE-F9** | `valid_aggregate_snapshot` never reads `kind` | **DEFERRED — P6 Task 7** | The helper that fixes it exists (`REUSE-F8`) and `run_proof_is_current` already routes through the sdk aggregate reader. `valid_aggregate_snapshot` itself is unmigrated. Unreachable today: `required_outputs` is always four deliverable files. |
| **REUSE-F10** | `_cli_completion` re-spells its own marker check 130 lines from the original | **DEFERRED — phase owning marker refresh** | Ruled 2026-09-05. "Subsumed by F3" is a claim about the *rejection predicate*; rewiring `refresh_success_markers_after_metadata_migration` changes what it **does** — it would begin refreshing migrated markers it currently rejects — on a path with no coverage. |
| **REUSE-F11** | a second worklist definition, with no callers | **FIXED** `4b7ddeba` | `get_remaining_images` **deleted**, with a comment where it stood. `test_the_dead_second_worklist_definition_is_gone`. Deletion not relocation: it derives from `datasets.{completed,failed}`, which §4.2 removes. |
| **REUSE-F12a** | the staged completeness guard exists three times; one copy stats a file forward runs never write | **FIXED (helper)** `4b7ddeba` | `staged_image_is_complete(...)` in `sdk_/_run_state.py`, embedded-table variant, both flags required keywords. Two tests pin that the legacy parquet does **not** satisfy it. **Call sites not migrated — P6 Task 7.** |
| **REUSE-F12b** | `stage3_markers_required` read with two different defaults at three sites | **DEFERRED — staged SLURM engine's phase** | Built, then **reverted**, because the report's premise is false: `_cli_staged_slurm.py:412` writes the *controller config*, but the two `False`-defaulting readers consult the *orchestration state*, which `initialize_orchestration` never writes the key into. Flipping to `True` would make `completed_inventory_images` stop recognising restart's own completed work. Two tests encode the convention — `:555` leaves the key unset and expects the parquet branch, `:581` sets it to `True` explicitly. The real fix is an explicit parameter on `initialize_orchestration`; that is a behaviour change in a subsystem P2 is not reviewing. |

---

## What remains open

Nine items, none of them a code defect that ships today.

| ID | Owner | Why it cannot close here |
|---|---|---|
| `IMPL-F6` = `SPEC-C3` | phase editing `_schema_shape` | A stale count in a docstring; benign while `SCHEMA_GATE_ARMED` is `False`. |
| `IMPL-F7` | phase editing recompile | A wrong citation in a docstring that tells a reader not to re-derive. |
| `IMPL-F9`, `IMPL-F12` | next phase editing those files | Unblocked by `IMPL-F2`; both are prose. |
| `IMPL-F13` | phase that can adapt the minter | Needs an adapter or a written reason, not a rename. |
| `SPEC-B1`, `SPEC-B2`, `SPEC-B4` | P3 | Three planned tests that did not land. `B4`'s guard has **no test at all**. |
| `SPEC-B3` | P3 (code) / now (docs) | Phase-ordering conflict; the doc half could be amended immediately. |
| `REUSE-F9`, `REUSE-F12a` call sites | P6 Task 7 | Destinations built; migration is that task by design. |
| `REUSE-F10`, `REUSE-F12b` | owning phases | Both are behaviour changes outside P2's remit. |

### Still owed from a resolved finding

`SPEC-C1` is withdrawn, but its reviewer made a recommendation that was not
actioned: §5.1's amendment gives `lifecycle_epoch` only the
runtime-mode-dependence argument, when it is **also an on-disk key** —
`publish_image_success` writes `"lifecycle_epoch"` into every image marker
(`_cli_completion.py:234`). That is the same persistence argument that pins
`slurm_generation`, and a stronger one, since it survives any future change to
the value's meaning. One line, and it closes the row against the obvious
rebuttal (*"then narrow the value"*).

---

## Dissents, and one finding this exercise produced

**I agree with every ruling.** Two carry a qualification, and building this
table turned up one thing nobody had raised.

### 1. `IMPL-F7` was closed by silence, which is what this file exists to catch

It appears in neither list in `41acbf0c` — not in the six fixed, not in the four
named as deferred. The commit accounts for twelve of thirteen findings and the
missing one is invisible precisely because the message is otherwise thorough. It
is also still wrong on disk, and its line number has drifted since, so a reader
following it now lands on neither the cited function nor the right line.

### 2. `IMPL-F10`'s withdrawal is right; its supporting claim is incomplete

The withdrawal is correct and I am not disputing it — the `mtime`-granularity
mechanism really is dissolved by "no path writes twice in a pass."

But the commit's supporting sentence says *"The only in-place rewrite of a
tracked artifact is `replace_embedded_measurement_table` at
`_cli_migrate_image.py:281`, under `--mode migrate`."* **There are three call
sites**, and the third is not migrate: `_cli_output_manager.py:1997`
(`replace_image_store_measurements`), reached from
`_cli_process_single.py:439` on the **`--mode measure`** path.

This does not revive `IMPL-F10` — `--mode measure` is a separate invocation,
hours apart, so `mtime` moves. It matters for a *different* invariant, the one
that makes root-only fencing sound. `_cli/CLAUDE.md` states it as: *"a **store**
is fingerprinted by its root `zarr.json` alone … the root is written **last** …
and nothing writes into the store after publication."*

**Verified, by reading `_measurement_tables.py:283-289`:** when the descriptor is
unchanged, `replace_embedded_measurement_table` calls `_write_validated_parquet`
on the payload and **returns without rewriting the root**. And the descriptor is
built from `schema_version`, `type`, `format`, `path`, `measurement_columns` and
`target` — **no row count and no content digest** — so a re-measure producing the
same columns takes that path by construction.

**Not verified, and it decides the severity:** whether the `--mode measure` path
re-publishes the image marker afterwards. If it does, the stale fence is
unreachable in practice and this is a documentation defect. If it does not, a
marker's store descriptor can certify a table that has since changed.

I am recording it as a **question with the verified half separated from the
unverified half**, rather than as a defect, because the alternative is the exact
error `fcf99a19` was written to correct: a claim about what a check cannot catch,
carried forward without establishing that the triggering condition occurs.

### 3. Two reviewers found the same site independently, twice

Worth recording because it is evidence about the *sites*, not about the
reviewers. `IMPL-F5`/`SPEC-C4` (the `clear_verification_cache` docstring) and
`IMPL-F6`/`SPEC-C3` (the `_schema_shape` count) are each one defect found twice
from different directions — and the second pair is **still open**. A duplicate
finding that survives triage is a stronger signal than a singleton that gets
fixed.

---

## Provenance

Dispositions were established from the working tree at `4b7ddeba` and from
`git log main..HEAD`, not from the review conversation. Where memory and the
tree disagreed, the tree won: I had `IMPL-F10` recorded as *deferred* and it is
**withdrawn**, and I had `IMPL-F7` recorded as *fixed* and it is **open**. Both
corrections came from reading `fcf99a19`'s and `41acbf0c`'s messages and then
checking the cited source, which is the only reason this file is worth more than
the recollection it replaces.
