# Phase 1 test review — can these tests fail, and for the reason their names claim?

**Change:** `cli-gui-state-tracking`, Phase 1 (`sdk_/_run_state.py` and the run-state SDK).
**Reviewed tree:** `git diff 6902124e~1..17f144ef` — 24 files, +6613/−55, twelve commits.
**Review date:** 2026-09-04. **Analysis only** — no source or test file was edited by this review.
**Spec:** [`../../specs/2026-09-03-cli-gui-state-tracking/design.md`](../../specs/2026-09-03-cli-gui-state-tracking/design.md) (§0 authoritative).
**Plan:** [`../../plans/2026-09-03-cli-gui-state-tracking/phase-1-run-state-sdk.md`](../../plans/2026-09-03-cli-gui-state-tracking/phase-1-run-state-sdk.md).

The question asked was **not** "do these tests pass". It was: *can each test fail, and does it
fail for the reason its name claims?* Phase 1 had already run 36 mutations (12 over
`_verification_cache.py`, 24 over the readers) and confirmed them all RED, and had committed a
coverage gate. This review looked for **what that work could not see**: tests whose mutation
proves something narrower than the name, fixture builders that could make a whole suite vacuous,
declared controls that nothing verifies, and branches no mutation targets.

Line citations are against the reviewed tree unless a finding says otherwise; where a fix has
since moved a line, the HEAD line is given as well.

---

## Status at time of writing

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | The committed mutation gate covers 20 of 88 tests, and cannot say so | High | **Open** — informs P2's first item |
| 2 | `_live_authority`'s two liveness predicates had no negative control | High | **Fixed** `c296d2eb` |
| 3 | `and not escalated` in the depth report is unpinned | Medium | **Open** — handed to the agent owning `test_verification_cache.py` |
| 4a | `_source_set_binding`'s post-U-4 branch was dead and would ship unproven in P4 | Medium-High | **Fixed** `cfca15a5` |
| 4b | The `publication_id` aggregate→run binding is unpinned | Medium | **Open** |
| 5 | `describe_conversion_advisory`'s `UNREADABLE_STATE` branch is unreachable | Low | **Open** |
| 6 | `test_image_state_stages_carry_no_backfilled_key` is a tautology | Low | **Open** |
| 7 | Five smaller unexercised paths | Low | **Open** |

Finding 1 also generalised beyond the harnesses: applying the same question to the phase's own
shard collector found that it detected an *unparseable* shard and had no notion of an *absent*
one, so a 48-shard array aggregated with four tasks still `RUNNING` reported
`shards=44 tests=9967 REGRESSIONS (0)` against an 11,106-test baseline — 1,139 tests silently
missing, and a failure count dropping 81→21 reading as an improvement. Fixed at `29e5ecc8`; the
complete run is 0 regressions over 48 shards / 11,244 tests.

---

## 1. The committed mutation gate covers one suite of three, and reports green over the other two

**Open.** Severity: high — this is the finding that hides the others.

`check_mutation_coverage.py` globs `mutation_harnesses/*.py`. Exactly one harness is committed:
`p1_task3_verification_cache.py`, with `SUITE = "tests/unit/sdk_/test_verification_cache.py"`
(harness `:59`) and `TARGET = src/phenotypic/sdk_/_verification_cache.py` (`:58`). So
`COVERAGE_OK=True` asserts only that the 20 tests in the cache suite are each claimed by some
mutation. It is silent about everything else in the phase.

| Suite | Tests | Claimed by any mutation, committed **or** scratchpad |
|---|---|---|
| `tests/unit/sdk_/test_verification_cache.py` | 20 | **20** |
| `tests/unit/sdk_/test_run_state.py` | 44 | **21** |
| `tests/unit/cli/test_schema_gate.py` | 22 | **0** |
| `tests/unit/sdk_/test_run_state_layering.py` | 2 | **0** |

The 24 reader mutations exist only in the session scratchpad
(`mutation_harness_p1c3.py`, `mutate_schema_gate.py`), which the harness README itself names as
the failure mode: *"A harness living in a scratchpad is invisible to it, so 'the checker is
green' does not mean 'every anchor in this change is validated'. … If you are mutating a target
whose harness is not in this directory, that target is unwatched. Commit the harness."*
Session scratchpads do not survive the session.

**Tests the scratchpad reader harness does claim** (parsed from its `MUTATIONS` table): the five
`test_each_of_rule_ones_comparisons_is_load_bearing` params, `test_clause_one_is_load_bearing`,
`test_a_live_worker_does_not_mask_a_valid_run_proof`,
`test_an_active_run_outranks_a_stale_terminal_failure`,
`test_a_store_built_against_older_metadata_is_an_advisory`,
`test_the_unavailable_fence_is_surfaced_as_an_advisory`,
`test_an_unconverted_h5_is_an_advisory_and_never_a_gate`,
`test_a_dead_gui_owner_does_not_pin_an_unfinished_run_at_active`,
`test_an_unmarked_record_is_still_fenced_by_work_id`,
`test_shallow_reuse_is_independent_of_the_image_count`,
`test_shallow_with_a_cold_cache_equals_deep`,
`test_a_new_image_escalates_the_whole_resolution`,
`test_a_pre_markers_tree_is_incomplete_and_never_raises`,
`test_a_process_run_reads_complete`,
`test_assert_identity_current_names_the_field_that_changed`,
`test_scheduler_epoch_and_owner_generation_are_not_in_the_digest`,
`test_the_verdict_matrix[terminal-failure]`,
`test_a_superseded_failure_does_not_make_the_run_failed`, plus the three handover tests.

**Never claimed by anything**, and therefore the input list for P2's harness:

- `test_the_sdk_reader_agrees_with_the_cli_validator` — **put this first.** INV-LAYER forces
  `_run_state` to re-derive what `valid_image_success` decides, and this parametrized test over
  four tamperings is the *only* thing keeping the two implementations in step until P6 Task 7
  deletes the CLI copy. Note its four cases (`untouched`, `marker-gone`, `overlay-rewritten`,
  `store-root-rewritten`) do not reach the branches most likely to diverge: a wrong marker
  `version`, a wrong `dataset`, an unknown artifact `kind`, or a descriptor path escaping the
  root.
- `test_the_marker_schema_constants_have_exactly_one_home` — the guard on the constant
  relocation that makes a false `complete` impossible.
- `test_the_verdict_matrix`'s other five params (`untouched`, `missing-marker`, `no-run-proof`,
  `unreadable-proof`, `unreadable-state`).
- `test_a_real_change_after_publication_invalidates_completion` (all three params),
  `test_a_fully_republished_rolling_input_reads_complete`.
- `test_a_process_run_still_detects_a_pipeline_edit`,
  `test_a_process_run_detects_a_changed_export_layer`.
- `test_a_migrated_record_is_accepted_on_artifact_validity_alone`.
- `test_a_live_worker_over_an_unfinished_run_reads_active`,
  `test_a_live_gui_owner_reads_active`,
  `test_a_dead_gui_owner_does_not_pin_the_verdict_at_active`.
- `test_a_warm_shallow_pass_reports_the_same_advisories`,
  `test_the_schema_advisory_is_gated_and_never_a_lie`,
  `test_an_unverified_image_says_why`,
  `test_an_empty_directory_is_incomplete_and_never_raises`.
- Every identity-reader test (`test_run_identity_*`, `test_assert_identity_current_accepts_*`,
  `test_assert_identity_current_raises_when_there_is_no_state`,
  `test_finalization_input_*`, `test_each_digest_token_moves_the_digest`).
- `test_the_demoted_sources_live_only_under_diagnostics`,
  `test_image_state_stages_carry_no_backfilled_key`.
- The whole of `test_schema_gate.py` and `test_run_state_layering.py`.

Two declared controls are correctly excluded rather than unclaimed:
`test_a_clean_tree_carries_no_advisories` and
`test_a_matching_metadata_snapshot_raises_no_advisory`. Both are real controls — I verified each
fails if the corresponding advisory fires spuriously. Note the committed harness declares no
`CONTROLS` tuple at all (its suite is fully claimed), so the mechanism the README documents is
currently unexercised; P2's harness will be the first to use it.

**Recommendation:** commit a `p1_run_state.py` harness with the 24 scratchpad mutations plus new
ones for the list above, and one for `test_schema_gate.py`. Until then, treat `COVERAGE_OK=True`
as scoped to the cache suite and say so wherever the number is quoted.

---

## 2. `_live_authority`'s two liveness predicates had no negative control

**Fixed at `c296d2eb`.** Severity: high. The suite was completely blind to it.

`src/phenotypic/sdk_/_run_state.py:732-735` (HEAD `:733-736`):

```python
if lifecycle is not None and lifecycle.get("active") is True:
    return slurm_lifecycle_path(output_dir).name
...
if owner is not None and owner.get("status") in _OWNER_STATUSES_IN_FLIGHT:
```

Every liveness test drove `_live_authority` **positively only**:

- `_mark_slurm_lifecycle_active` (`test_run_state.py:362`) calls `initialize_slurm_lifecycle`,
  which writes `"active": True` (`_cli_slurm_lifecycle.py:125`). No test ever created a
  lifecycle record with `active: False`.
- `_write_owner_record` (`test_run_state.py:401`) was called three times, always with
  `status="running"` (`:679`, `:703`, `:715`). `"submitting"` was never exercised, and no
  terminal status ever was.

**Neither state is hypothetical.** `active: False` is what finalize and clear write
(`_cli_slurm_lifecycle.py:661,679`) — it is the state of *every SLURM run that has ended*. And
the registry writes terminal statuses (`"complete"`, `"failed"`, `"cancelled"`) into
`gui_launch_owner.json` while the GUI process that owns the recorded pid is **still alive**,
because it is the long-running Dash server.

**Failure scenario.** Widen `:732` to `lifecycle is not None`. A SLURM run that finished with two
images failed leaves `active: False` on disk. Rule 1 cannot fire (clause 1 fails — not every
accepted image has a valid proof), rule 2 now does, and the run reports **`active` forever**: a
real failure masked by a worker that is not there, and nothing repairs the record.

The owner half is the same shape *without a pid to disbelieve*. Drop the status membership test
and keep `_process_is_alive`: `pid_exists` is `True` for the live GUI, so every incomplete output
that GUI ever launched reads `active`. `test_a_live_gui_owner_reads_active` passes (it wants
`active`), `test_a_dead_gui_owner_does_not_pin_an_unfinished_run_at_active` passes (dead pid),
and `test_a_dead_gui_owner_does_not_pin_the_verdict_at_active` passes (rule 1 wins). This is
CAN-24's failure exactly one field over from where CAN-24 was fixed.

**Resolution.** Both predicates widened at once —
`lifecycle.get("active") is True` → `lifecycle is not None`, and
`owner.get("status") in IN_FLIGHT` → `owner.get("status")`. The two new controls
`test_a_finished_lifecycle_record_is_not_a_live_authority` and
`test_a_terminal_owner_status_is_not_a_live_authority` go red; **all 58 pre-existing tests stay
green.**

---

## 3. `and not escalated` in the depth report is unpinned — deleting it breaks no test

**Open**, handed to the agent that owns `test_verification_cache.py`. Severity: medium.

`_run_state.py:1086-1090` (HEAD `:1101-1105`):

```python
performed: Depth = (
    "shallow"
    if requested_depth == "shallow" and warm is not None and not escalated
    else "deep"
)
```

Every test that asserts `.depth` falls into one of two buckets, and neither reaches the third
term:

- **`warm is None`** → `"deep"` either way: `test_a_new_image_escalates_the_whole_resolution`
  (`test_run_state.py:1242` at HEAD), `test_shallow_with_a_cold_cache_equals_deep` (`:1254`),
  `test_clear_scoped_to_one_output_does_not_clear_another_end_to_end`
  (`test_verification_cache.py:621`), `test_an_identity_change_forces_reverification` (`:645`).
- **fully current cache** → `"shallow"` either way:
  `test_shallow_reuse_is_independent_of_the_image_count` (`test_run_state.py:1228`),
  `test_a_warm_shallow_pass_reports_the_same_advisories` (`:1272`),
  `test_a_warm_cache_is_actually_used` (`test_verification_cache.py:584`).

The only test that reaches a **partial** escalation under an unchanged identity is
`test_a_tampered_artifact_falls_through_even_with_a_warm_cache`
(`test_verification_cache.py:537`) — warm cache, identity unchanged, image `a`'s entry no longer
current, `b`'s still current. It asserts `after.completion != "complete"` (`:564`) and never
looks at `after.depth`. One line closes it:

```python
after = resolve_run_state(complete_run, depth="shallow")
assert after.completion != "complete"
assert after.depth == "deep", "a partial cache miss must still report the depth performed"
```

**Why no other test can reach it.** The inventory is folded into `RunIdentity.digest()`
(`_state_types.py:72-83`), so "a new image" can only ever present as a whole-cache drop.
`test_a_new_image_escalates_the_whole_resolution`'s name describes a per-image miss the test does
not exercise; what it actually proves is the identity fence, which
`test_an_identity_change_forces_reverification` already proves.

**Blast radius.** `RunState.depth` would report `"shallow"` after a pass that deep-verified some
images. That understates rather than overstates authority, so it is not an INV-VERDICT breach —
but `depth` is documented as what a caller reads to know whether the answer is authoritative,
and it would be lying in the direction that makes a consumer skip a re-check it needs.

**Note for whoever picks this up.** U-11 landed an on-disk cache tier (`eb80c7b3`) after this
review. `escalated` now has a second consumer at HEAD `:1099` — `if escalated:
persist_states(...)` — so it is no longer dead if the `performed` term is removed. The
`performed` term itself is still unpinned at HEAD (confirmed against the four `.depth`
assertions above), and the new tier-2 write gating is outside this review's scope and has not
been reviewed here.

---

## 4. `_source_set_binding`: two distinct gaps

`_run_state.py:928-938` (HEAD `:929-939`):

```python
if "source_set_digest" in proof:          # the post-U-4 shape
    return proof
aggregate = _valid_aggregate_proof(output_dir)
if aggregate is None:
    return None
if proof.get("publication_id") != aggregate.get("publication_id"):
    return None
return aggregate
```

### 4a. The post-U-4 branch was dead, and P4 would have shipped it unproven

**Fixed at `cfca15a5`.** Severity: medium-high.

`publish_run_completion_evidence` (`_cli_completion.py:1008-1035`) writes `publication_id` and no
`source_set_digest`, so in Phase 1 **only the legacy branch has ever executed**. `grep -rn
"source_set_digest"` over `test_run_state.py`, `test_verification_cache.py` and
`tests/_output_layout.py` returned only aggregate-proof falsifications — nothing ever put the key
in a run proof. Phase-4 Step 7 (`phase-4-finalize-run.md:1086-1091`) publishes `source_set_digest`
**and** `source_image_count` into both proofs, at which point the modern branch fires for the
first time ever, in the phase that ships it.

**Failure scenario.** Rule 1 reads *both* fields off whichever proof the binding returns. A P4
that publishes the digest without the count makes `binding.get("source_image_count")` `None`, the
arity check compares `None` to an int, rule 1 stops firing, and **every full run reads
`incomplete` forever.** That is N-4's shape — the exact failure this dual read exists to prevent —
and the plan's claim of "no window in which the two comparisons silently stop being made" was the
one claim in rule 1 that nothing checked.

**Resolution.** `test_a_post_u4_run_proof_binds_without_the_aggregate` stamps both fields into the
run proof and asserts `complete`. One subtlety is worth recording, because the first draft of the
test would have proved nothing: stamping `source_set_digest` while leaving the aggregate binding
intact still reaches `complete` **through the legacy branch**, so the test would have passed with
line 931 deleted. The committed version severs `publication_id` first, so `complete` is reachable
only through the modern branch. Proved by deleting the branch — exactly that test fails, the
other 60 stay green.

### 4b. The `publication_id` aggregate→run binding is unpinned

**Open.** Severity: medium.

No test in the change touches `publication_id` on either proof. Delete the two lines

```python
if proof.get("publication_id") != aggregate.get("publication_id"):
    return None
```

and the suite stays green — including `test_each_of_rule_ones_comparisons_is_load_bearing`, whose
five params falsify `inventory_digest`, `scientific_config_digest` and
`finalization_input_digest` on the run proof and `source_set_digest`/`source_image_count` on the
aggregate proof, but never the field that binds the two documents to each other.

**What the check does.** It ties a run proof to *the specific aggregate publication it was minted
against*. `publish_run_completion_evidence` copies `aggregate.get("publication_id")` into the run
proof at publish time (`_cli_completion.py:1021-1023`), and `valid_run_completion` compares the
same pair (`:1105`). Without it, a stale run proof sitting beside a freshly republished aggregate
proof borrows the new aggregate's source set — the run proof certifies a publication that no
longer exists, and rule 1 accepts it.

**Cheapest pin.** Add one param to the existing parametrization at `test_run_state.py:760-786`,
reusing the helper already there:

```python
pytest.param(
    _falsify_run_proof("publication_id", "deadbeef"),
    "publication_id",
    id="publication_id",
),
```

`_valid_run_proof` checks only `version`, `status` and `finalizer_succeeded`
(`_run_state.py:748-756`), so the falsified field survives to the comparison under test — the
same mechanism that makes the other five params work.

Note this pin has a shelf life: U-4 cuts `publication_id`, and once P4 lands the modern branch the
binding is stated directly by `source_set_digest` in the run proof. Until then the legacy branch
is what runs against every real tree, and its binding check is the unguarded part of it.

---

## 5. `describe_conversion_advisory`'s `UNREADABLE_STATE` branch cannot be reached through its only caller

**Open.** Severity: low — dead user-facing text, not a wrong answer.

`_schema_shape.py:415-421` composes a careful reader-facing message for an unreadable state file:
*"This output's processing state (…) is not readable as a state file, so its completion cannot be
established. What is displayed comes from the artifacts on disk. …"*

Its only caller is `_run_state._advisories` (`_run_state.py:967`), invoked at
`resolve_run_state`'s tail (`:1186`, HEAD `:1203`). But `resolve_run_state` returns early at
`:1140-1150` (HEAD `:1157-1167`) whenever `_read_state_config` yields `None`. Every payload that
makes `_classify` return `UNREADABLE_STATE` — the whole `_UNREADABLE_STATE_PAYLOADS` list at
`test_schema_gate.py:355`: `"{truncated"`, `"null"`, `"[]"`, `'{"config": {}}'`, `""` — also
makes `_read_state_config` return `None`, so the reader emits the generic *"No readable
processing state under this directory"* string instead and that branch never runs.

I checked the one shape that might have escaped: a valid JSON object carrying `version`/`datasets`
but **no `config` key**. `_read_state_config` still returns `None` (config is not a dict) → early
return; `_classify` returns `CONVERT` on absent `work_ids`, not `UNREADABLE_STATE`. So there is no
input that reaches it.

`test_the_gui_reports_rather_than_refuses` (`test_schema_gate.py:790`) uses a readable
markers-era tree and does not touch it either. Either delete the branch or give the reader a path
to it; today it is seven lines of message that can never be shown, with no test that could
notice.

---

## 6. `test_image_state_stages_carry_no_backfilled_key` is a tautology

**Open.** Severity: low.

`tests/unit/sdk_/test_run_state.py:84`. The test constructs `ImageState(..., stages={"measured":
{...}})` by hand and asserts `"backfilled" not in state.stages`. `ImageState` is a frozen
dataclass with no `__post_init__` (`_state_types.py:86-107`), so the assertion is over a literal
the test itself wrote three lines earlier. It cannot fail for any change to `_run_state.py`.

The property its name claims — that D-A's removal of the backfill stage holds in the code that
*builds* stages — lives in `_verify_image` (`_run_state.py:598-611`), which the test never calls.
The real form goes through the reader:

```python
state = resolve_run_state(complete_run, depth="deep")
assert all("backfilled" not in image.stages for image in state.images.values())
```

Its neighbour `test_the_demoted_sources_live_only_under_diagnostics` (`:56`) is fine by contrast —
it pins actual `dataclasses.fields` sets and fails if U-5's dropped trio returns.

---

## 7. Smaller unexercised paths

**All open.** Severity: low, but two of them are P2's landing zone.

- **`restart_epoch`'s reader is entirely unexercised.** `_identity_from` (`:264-271`) guards
  non-int values and excludes `bool` explicitly, but `write_processing_state` never writes the
  key, so every fixture takes the `else 0` fallback. `test_each_digest_token_moves_the_digest`
  (`:221`) uses `dataclasses.replace(identity, restart_epoch=7)`, which exercises `digest()`, not
  the parse. **D4's one added tracked value lands in P2 on a reader path no test has run.**
- **`_scheduler_epoch`'s v1 fallback is untested.** `:213-215` falls back from `generation` to
  `epoch`; no fixture writes a v1 lifecycle record, so only the modern spelling is covered.
- **`test_every_convert_verdict_is_dischargeable_by_one_migrate` is `@pytest.mark.skip`**
  (`test_schema_gate.py:883`), correctly and with a P7 removal note. Consequence worth recording:
  the file header's claim that the two shape lists are *"single sources of truth, shared by the
  classification tests and the discharge test"* currently buys nothing, since only the
  classification half runs. INV-DISCHARGEABLE's discharge half is unproven until P7 Task 5
  removes the mark.
- **`test_migrate_mode_is_never_refused_by_the_gate`** (`:707`) asserts only `"cannot read this
  output" not in result.output`, with no exit-code or positive assertion. It is sound today
  because `migrate_only` short-circuits the gate at `phenotypicCLI.py:1671`, after every earlier
  `UsageError` its argv could trigger — but it would go green just as readily if a future edit
  made migrate fail earlier for an unrelated reason.
- **INV-LAYER's AST walk has two holes.**
  `test_run_state_layering.py::test_neither_module_ever_names_the_cli_package` misses
  `from phenotypic import _cli` (`node.module == "phenotypic"` fails the
  `startswith("phenotypic._cli")` test) and any `importlib.import_module`. Low probability,
  one-line fix. Its sibling `test_run_state_exports_no_writer` checks only `__all__`, not module
  contents — correct for the stated invariant (the exported surface), worth knowing is all.

---

## Areas checked and found clean

Reported as clean rather than padded, because a clean area confirmed is worth more than a list of
maybes.

### The four "claimed under another test's mutation" handover claims are real

Each was traced rather than accepted:

- **`test_a_forged_entry_cannot_manufacture_complete`** under mutation #2 (drop the empty
  `stat_tuples` guard). With the guard gone, `all()` over an empty map is `True`, the forged
  entries are "current", and the resolver reuses their `verdict="verified"`. I confirmed the
  fixture genuinely flips: on `build_incomplete_run`, clause 2 is *already* satisfied — the
  aggregate proof still carries both images' `source_set_digest` and count, the inventory,
  pipeline and finalization digests are untouched — so clause 1 is the only thing holding the
  verdict, and forged verdicts remove it. `incomplete` → `complete`. RED.
- **`test_a_tampered_artifact_falls_through_even_with_a_warm_cache`** under mutation #12 (drop
  the stat comparison). The appended overlay changes both size and `mtime_ns`; with no
  comparison the warm entry stays current, the deep pass is skipped, and a tampered run reports
  `complete`. RED.
- **`test_clear_scoped_to_one_output_does_not_clear_another_end_to_end`** under mutation #5
  (unscoped clear). Clearing `a` empties `b`, so `b`'s shallow pass escalates and reports
  `"deep"`. RED.
- **`test_an_identity_change_forces_reverification`** under mutation #1 (`cached_states` ignores
  `identity_digest`). After `bump_scientific_config_digest`, the stale map is returned, nothing
  on disk changed so every entry is current, and `performed` is `"shallow"`. Both assertions in
  the test fail. RED.

All 20 tests in the cache suite are claimed; that coverage number is honest.

### `tests/_output_layout.py`'s builders are not vacuous

This was the highest-leverage thing to check — a `build_complete_run` that produced something not
actually complete would make a whole suite vacuous while every mutation over the source still
went red, and no mutation can detect a fixture that lies.

It does not hand-write the format under test at any point. It promotes through
`Image.save2zarr` with a real `PreparedEmbeddedMeasurementTable`, writes a real PNG overlay, then
calls the production publishers in the contract's own order: `publish_image_success` →
`save_processing_state` → `write_master`/`write_measurements_mirror` →
`publish_aggregate_snapshot` → `publish_run_completion_evidence`.

**The failure mode is blocked structurally, not only by assertion.**
`publish_run_completion_evidence` has a degenerate early branch (`_cli_completion.py:975-995`)
taken when `current_run_is_complete` is `None`, the state is missing, or
`success_markers_required` is false. That branch writes `{"schema_version": 1, …}` with **no
`version` key**, and `_valid_run_proof` (`_run_state.py:748-750`) rejects it outright. A tree that
only *looked* built would therefore read `incomplete`. Two independent tests assert it reads
`complete`: `test_the_verdict_matrix[untouched]` (`:594`) and
`test_the_schema_advisory_is_gated_and_never_a_lie`'s `disarmed.completion == "complete"`
(`:568`). Completeness is a checked fact, not a name.

I also confirmed the gate those publishers pass through is **marker-derived, not state-derived**:
`current_run_is_complete` → `current_aggregate_is_current` → `_current_success_work_ids` →
`valid_image_success` per work id (`_cli_completion.py:683-703`). So the builder's
`DatasetState(initial_images=…)` with no `completed` key cannot be what makes it pass — the
fixture cannot cheat by writing the derived sets. And `build_incomplete_run` removes the
**marker**, not the artifacts, which is the state a kill between promoting a store and publishing
its proof actually leaves.

One residual, low severity and not a defect today: both fixture images get byte-identical pixels
(`np.zeros((8,8,3))`) and byte-identical overlays (`pixels[:4,:4,0] = 255`). Artifact paths come
from the marker's own descriptors, so a stem mix-up in *artifact* resolution cannot hide there —
but `_store_metadata_snapshot` (`_run_state.py:493`) **recomputes** the store path from
`(dataset, image_stem)`, and that one is covered only because
`test_a_store_built_against_older_metadata_is_an_advisory` stamps the divergent snapshot onto
stem `a` alone. Worth knowing if anyone later makes the fixture symmetric in that dimension too.

### The dual-shape rule-1 **compatibility** path is the one under test

The question as posed had it backwards, so stating it plainly: today's publisher writes neither
`source_set_digest` nor `source_image_count` in the run proof, so **every fixture in the change
falls through into the aggregate-proof fallback**. The legacy branch is exercised by all five
`test_each_of_rule_ones_comparisons_is_load_bearing` params,
`test_clause_one_is_load_bearing`, and every `complete` assertion in the suite. It is in good
shape as a path. Its one unguarded *comparison* is the `publication_id` binding — finding 4b.

The **modern** branch was the untested one — finding 4a, now fixed. So the risk here was entirely
forward-facing and materialised in P4, not in P1.

### Rule 1's isolation tests really do isolate

- **`test_clause_one_is_load_bearing`** (`:727`) is caught by clause 1 *alone*. After removing
  marker `b` and re-pointing the aggregate's `source_set_digest`/`source_image_count` at the
  survivor, `verified` is `["work-a"]` and the binding matches it exactly — all five of clause
  2's comparisons pass, and only "every accepted image has a valid proof" notices the run is not
  done. This is the U-2 restoration doing real work.
- **Each of the five falsification params** is caught by exactly one comparison, with no
  neighbour absorbing it. `_valid_run_proof` does not re-validate the falsified fields, so each
  falsification survives to the comparison under test.
- **`test_a_real_change_after_publication_invalidates_completion[late-metadata]`** genuinely
  isolates §7.4's guarantee: `bump_metadata_snapshot_digest` leaves `work_ids`, the pipeline
  digest and every marker untouched, so the finalization comparison is the only thing that can
  notice.

### `canonical_digest`'s hoist has no drift

Checked specifically because a default mismatch here would make every run read `incomplete`
forever. `_finalization_inputs` (`_run_state.py:168-173`), `publish_aggregate_snapshot`
(`_cli_completion.py:906-914`) and `current_aggregate_is_current` (`:729-735`) agree
field-for-field — including `include_dataset_column` read with **no default** on all three sides,
and `no_qc` defaulting to `False` on all three. The reader's two-spelling acceptance
(`_accepted_finalization_digests`, `:817-841`) is sound: both digests are functions of exactly the
same three values, so a change to any of them moves both.

### Layering

`_state_types` ← `_verification_cache` ← `_run_state` holds with no edge back. The lazy
`from .ngff_ import …` imports inside `_fenced_artifact_path` and `_store_metadata_snapshot` stay
within `sdk_`, so they are not INV-LAYER violations.

### Already corrected during the review

`_schema_shape.py`'s `SCHEMA_GATE_ARMED` docstring described a `_cli_schema_gate` re-export that
does not exist and instructed future test authors to patch a module whose
`test_the_arming_flag_has_one_source` forbids binding the name. Corrected at `bc4e6449` while
this review was in progress; the current text at `:87-100` is right.

---

## Method note

Two things about this review are worth reusing.

**The question was "can it fail", not "does it pass".** Every finding above came from asking what
a specific mutation would leave green, then checking whether any assertion in the tree could
observe it. Findings 2, 3 and 4b are all of one shape: a predicate with a positive control and no
negative one. That shape is invisible to a green suite and invisible to a coverage gate, and it
is what a mutation harness exists to catch — which is why finding 1 ranks first.

**A finding that lives only in a message dies with the session.** This report exists because the
same conclusion applied to the review itself. It also applied to the phase's tooling twice: a
harness in a scratchpad is not a committed guard, and a shard collector that counts what arrived
is not a collector that knows what is missing.
