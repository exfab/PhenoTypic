# P4 implementation-test review — can these tests fail?

**Scope:** `06809fbc` (T1+T2), `858ce22e` (T3), `8a6d3ae0` (T4), `6b3cf2c9` (T5).
**Question asked:** not whether P4's tests pass, but whether each would go red if the
behaviour it names regressed.
**Method:** read the three new test files and every production symbol they name, then for
each assertion ask what establishes that the code reached the point being asserted about.
Nothing in this review was run; every mutation below is written so the orchestrator can
run it and see the predicted colour.

## What I measured

| File | `def test_` | cases after parametrization |
|---|---|---|
| `tests/unit/cli/test_embedded_table_inversion.py` | 16 | 17 (`producer` ×2) |
| `tests/unit/cli/test_finalize_run.py` | 24 | 26 (`mode` ×3) |
| `tests/integration/cli/test_promote_time_metadata.py` | 3 | 3 |

46 cases. `tests/integration` **is** in `testpaths` (`pyproject.toml:220`), so T5 runs in the
normal lane — that was worth checking and it is fine.

I did not re-run the suite and quote no pass/fail totals of my own.

## Summary

The standing rule — *establish, then assert* — is applied unusually well. Nearly every
negative and equality in the three new files is preceded by a guard, most guards carry a
message saying why they exist, and several are genuinely load-bearing
(`test_stores_with_mixed_metadata_snapshots_do_not_abort_finalization`'s
`len(digests) == 2`; `test_a_heterogeneous_master_loses_no_measured_row`'s
`ragged_rows.height > 0`; `test_the_legacy_arm_still_prefers_its_dataset_aggregate`'s two
preconditions). The rebuild of
`test_finalizer_does_not_publish_after_master_parquet_failure` onto the write *attempt*
(`blocked == [master_path]`) is a real positive control and strictly stronger than the
sentinel it replaced.

**The remaining holes are not in the assertions. They are in the wiring.** Four of the six
findings below are the same shape: a guard whose *behaviour* is proved by a direct unit call,
and whose *reachability* is proved by nothing. That shape passes the standing rule — the
assertion is guarded, the fixture produced the thing — while leaving the production
behaviour it was written for unprotected. It is the natural next defect after the standing
rule closes the first one, and it is what a P4 mutation harness would have caught.

| # | Severity | Finding |
|---|---|---|
| 1 | **HIGH** | The "no schema stamp" falsifier builds two identical arms |
| 2 | **HIGH** | `refuse_mixed_measurement_authority`: both production call sites uncovered |
| 3 | **HIGH** | The legacy leg of `_image_authority_shapes` — added by this phase — is never taken with its version |
| 4 | **HIGH** | `_refuse_inverted_store` is wired by an order-blind substring test |
| 5 | MEDIUM | `_join_ragged_key_groups`' `unjoinable` branch and its `inner` path are untested |
| 6 | MEDIUM | `_structural_key_patterns` orders the mirror through an unordered `.unique()` |
| 7 | MEDIUM | The phase's own prescribed INV-INPUTS mutation is unrecorded |
| 8 | MEDIUM | P4 shipped no mutation harness, so none of these 46 cases is in the coverage checker |
| 9 | LOW | INV-PROVEN's only gate compares mtimes, not content |
| 10 | LOW | `master_carries_user_metadata` has no production consumer |
| 11 | LOW | Three smaller items (control in a sibling; determinism claim; misnamed fixture) |

---

## HIGH 1 — the designated falsifier for a HARD-STOP ruling compares a run against itself

`tests/unit/cli/test_finalize_run.py:1345`
`test_a_v1_metadata_free_master_is_indistinguishable_from_v2_and_that_is_harmless`

```python
v2.mkdir()
_publish_successful_images(v2, snapshot=None)
assert finalize_run(v2, dataset_names=[DATASET]) is not None

v1.mkdir()
_publish_successful_images(v1, snapshot=None)
assert finalize_run(v1, dataset_names=[DATASET]) is not None
...
assert outcomes_v1 == outcomes_v2
```

**The two arms are the same code over the same inputs.** Same fixture, same producer, same
`finalize_run`, no snapshot in either. `outcomes_v1 == outcomes_v2` is true by construction
and can only fail on nondeterminism. There is no v1 in this test.

This matters more than an ordinary vacuous equality because of what it is for. The plan
(`phase-4-finalize-run.md`, Step 6b) makes this test the falsifier for the user's ruling that
**no `master_schema_version` stamp is minted**, and states the consequence in terms: *"If
v1-no-metadata and v2 turn out BEHAVIOURALLY DISTINGUISHABLE … the ruling flips … **That
raises the state-artifact count, which is a HARD STOP**."* A test that cannot distinguish
them confirms the ruling by tautology.

The docstring is candid about the mechanism — *"With no snapshot the two producers agree by
construction, which is precisely the premise under test"* — but that sentence is the reason
the test cannot fail, not a defence of it. If the arms agree by construction, the property is
settled by argument and does not need a comparison; dressing the argument as a comparison is
what makes it read as evidence.

**Mutations that should redden it and will not:**

1. Invert the predicate — `src/phenotypic/sdk_/_master_io.py:94`,
   `return bool(...)` → `return not bool(...)`. Both outcome dicts change identically.
   *(This one also reddens `test_the_master_carries_no_user_metadata:650` and
   `test_master_carries_user_metadata_reads_ownership_not_the_prefix:1010`, so run it and
   check the falsifier is **not** among the failures — that is the finding.)*
2. Make the fourth reader always fail: in `_reader_outcomes`
   (`test_finalize_run.py:1333`), replace the `normalize_viewer_frame(frame)` call with
   `raise RuntimeError("x")`. Both arms record the identical string; still green. Only
   `_EXPECTED_READER_COUNT` and `assert outcomes_v1` can fire, and neither is about v1 vs v2.

**What would make it real.** The pre-inversion producer still ships —
`prepare_embedded_measurement_table` is retained by design (`06809fbc`), so a genuine v1
master is constructible: build the v1 stores through it and aggregate those. If that master
then turns out byte-identical to v2's under `snapshot=None`, assert *that* — `_master_bytes(v1)
== _master_bytes(v2)` — which is a claim that can fail, rather than an equality of two runs of
the same function. Failing that, rename the test to what it does pin (the reader count, and
that no P4 reader raises on a v2 master) and move the ruling's justification into prose.

## HIGH 2 — `refuse_mixed_measurement_authority` is proved to refuse, and never proved to be reached

`test_finalize_run.py:1401` `test_mixed_embedded_and_legacy_authority_is_still_refused` calls
the function directly, with both halves-alone controls. That part is good.

Its two production call sites are covered by nothing:

- `src/phenotypic/_cli/_cli_finalize_run.py:89` — the authorized arm of
  `select_measurement_sources`.
- `src/phenotypic/_cli/_cli_recompile_worker.py:809` — `_run_post_master_steps`, over the
  finalizer task's `measurement_sources`.

Line 89 executes on every authorized finalize, but always over a homogeneous list, so it can
never raise in any existing test. No fixture builds an authorized tree that also carries a
legacy external Parquet in its source set; no test drives the recompile finalizer task with a
mixed `measurement_sources`.

This is the surviving half of the retired `_consistent_embedded_join_keys`, and H6's entire
argument is that retiring the mixed-**generation** guard must not retire the
mixed-**authority** one. The function survived; its reachability did not get a test.

**Mutations (each should redden something, and will not):**

1. Delete `_cli_finalize_run.py:89`.
2. Delete `_cli_recompile_worker.py:807-811` (the `if measurement_sources is not None:` block).

Run `tests/unit/cli` and `tests/unit/sdk_` after each. Green under both is the finding.

**Fix:** one test per site. For (1), publish two images the ordinary way, then add a legacy
`results/<ds>/measurements/<stem>.parquet` to the authorized set — via a record whose
`artifacts["measurements"]` points at it — and assert `finalize_run` raises. For (2), hand
`_run_post_master_steps` a task dict whose `measurement_sources` mixes the two shapes.

## HIGH 3 — this phase added a legacy compatibility leg and nothing ever takes it

`8a6d3ae0` introduces `_image_authority_shapes` (`_cli_recompile_recovery.py:51`), which pairs
each payload shape with the version that shape must carry:

```python
return (
    (image_record_path(...), RECORD_VERSION),          # 1
    (image_completion_marker_path(...), SUCCESS_MARKER_VERSION),  # 2
)
```

The docstring names precisely what the pairing prevents, and `_marker_allows_table_transition`
(`:893-900`) repeats it: *"checking one shape against the other's number returns `False`
silently — which is exactly how a path-only repoint would have disabled table-authority repair
with nothing failing."*

**Every recompile fixture is a forward record tree.** `tests/unit/cli/conftest.py:346`
records that the branch which used to write an `image_complete/` marker stopped at P3's clean
break, and `test_image_record.py:799` is an explicit guard that no forward path creates that
directory. So `image_authority_payload` — consumed with its version at
`_cli_recompile_recovery.py:581`/`:745` (into `_marker_allows_table_transition`), `:490`, and
`_cli_recompile_slurm_scripts.py:557-568` — only ever returns the record leg.

The one place the legacy leg *is* exercised is `image_authority_**path**`, by
`tests/unit/cli/test_cli_recompile.py:270`
`test_crosslinked_accepted_authority_keeps_generic_recovery_error`, which writes a bare legacy
marker. That test reaches the path and never the version.

**Mutations, none of which redden anything:**

1. Reverse the tuple in `_image_authority_shapes` (legacy first). A forward tree still resolves
   the record, because the marker does not exist.
2. Change the legacy entry's version to `RECORD_VERSION` — i.e. reintroduce exactly the
   shape-blind check the docstring says is impossible.
3. Delete the legacy entry entirely.

**Why this is HIGH rather than housekeeping.** The plan's ⚠ RULED block (2026-09-06) chose
*both shapes, each on its own predicate* over a record-only repoint specifically because a
record-only version mismatch *"would make legacy trees lose overlay and table-authority repair
for the whole P4→P7 window, and silently — those functions return `None`/`False` on a version
mismatch rather than raising."* The ruling was implemented; the failure mode it was chosen to
avoid is exactly what mutation (2) reintroduces, and the suite would not notice.

**Fix:** one recompile test over a legacy tree — write an `image_completion_marker_path`
payload carrying `"version": SUCCESS_MARKER_VERSION` and no record, then assert the recovery
path accepts it (and, with `"version": RECORD_VERSION`, refuses it). That single fixture
covers the pairing at all four consuming sites.

## HIGH 4 — the guard's call site is pinned by a substring, which is blind to where it sits

`tests/unit/cli/test_embedded_table_inversion.py:584`:

```python
source = inspect.getsource(_cli_recompile_tables._replace_and_republish_table)
assert "_refuse_inverted_store(store_path)" in source
```

The test's own docstring is right that a guard defined and never called reads as coverage.
A substring in a function body is a weaker claim than it looks: it says the text occurs, not
that it runs, and not that it runs *first*.

In the shipped code the placement is correct — `_cli_recompile_tables.py:153`, before
`exclusive_path_lock` and before any write. Nothing tests that it stays there.

**Mutations that leave all three guard tests green while restoring the silent un-inversion:**

1. Move `_refuse_inverted_store(store_path)` from `:153` to the last line of the
   `with exclusive_path_lock(...)` block. The store is rewritten, its metadata table dropped
   and its measurements re-joined, *then* the guard raises. Substring present → green.
2. `if False:  # noqa` above the call. Substring present → green.

There is also no end-to-end test of the guard. `_run_mode(output_dir, "recompile")`
(`test_finalize_run.py:448-457`) is the only test that drives
`recompile_embedded_measurement_tables`, and it builds its tree with `snapshot=None`
(deliberately, for the byte-identity comparison), so those stores declare no
`tables.metadata` and the guard never fires on any real recompile.

**Fix:** replace the substring test with a behavioural one — build an inverted store, run
`recompile_embedded_measurement_tables` over it, assert `pytest.raises(RuntimeError, match=
"inverted store")` **and** that the store's `tables/metadata/pht-metadata.parquet` and root
`metadata_table` block are byte-unchanged afterwards. The second half is what mutation (1)
breaks and the substring cannot see. Both delete with the recompile repoint, as the current
three do.

## MEDIUM 5 — the ragged join's row-preserving branch is the one branch no test reaches

`src/phenotypic/_cli/_cli_output_manager.py:223-236`:

```python
if not usable:
    logger.warning("%d measurement row(s) carry none of the join columns %s ...")
    unjoinable.append(group)
    continue
```

This branch exists *because* `_join_ragged_key_groups`' stated invariant is that no measured
row is lost. `test_a_heterogeneous_master_loses_no_measured_row` (`test_finalize_run.py:783`)
is a good test of the ragged path — its guards are real and its four assertions each catch a
distinct half of the defect — but its fixture gives image `b` `Metadata_Well` and withholds
only `Grid_RowNum`, so `usable` is never empty. A third image carrying neither key reaches
this branch, and nothing builds one.

**Mutation:** `unjoinable.append(group)` → `continue` (drop the rows). Suite green; the
invariant the function was written for is broken for exactly the case it names.

Two adjacent gaps from the same fixture shape:

- **`how="inner"` on a ragged frame is untested.** `parts.extend(... unjoinable ...)`
  (`:270`) sits inside `if how == "left":`, so under `inner` the unjoinable rows disappear
  with no warning path distinguishing them from a deliberate drop. `join_metadata(how=
  "inner")` has callers outside finalization.
- **`if not parts: return df.clear().with_columns(...).clear()`** (`:274`) — the
  all-groups-empty case — is unreached.

**Fix:** extend `_measurements` with an `include_well=False, extra_columns=[]` variant and add
a third image to the ragged fixture that carries neither key; assert its rows survive into the
mirror with null metadata and are **not** flagged `QC_MetadataOnly`.

## MEDIUM 6 — the ragged path's output order is set by an unordered `.unique()`

`_cli_output_manager.py:146`:

```python
return df.select(masks).unique().rows()
```

`.unique()` is called without `maintain_order`, so the returned pattern list has no specified
order. That list drives the iteration in `_join_ragged_key_groups`, hence the order of
`matched`, hence the concat order of the mirror's measured rows.

No test would see it. `test_a_heterogeneous_master_loses_no_measured_row` asserts on
`set(...)`, `null_count()` and heights; `test_every_mode_produces_a_byte_identical_master`
compares **masters**, which carry no join at all and never touch this code. The
`test_the_ragged_path_is_reached_only_by_a_ragged_frame` predicate test asserts
`len(...) == 2` for the ragged case, deliberately not the order.

**Mutation:** in `_join_ragged_key_groups`, iterate `reversed(patterns)`. Suite green; two
runs of the same ragged input can now produce mirrors that differ by row order.

This does not contradict the phase's determinism claim — that claim is about stores and the
master — but a user diffing two `measurements.parquet` files from identical inputs would see
it, and the ordinary (non-ragged) path *does* guarantee order via `maintain_order="left"`.
Adding `maintain_order=True` to the `.unique()` costs nothing and makes the guarantee uniform.

## MEDIUM 7 — the phase's one prescribed INV-INPUTS mutation is not recorded as run

`phase-4-finalize-run.md` Step 5 withdraws sub-step (1) (it presupposed a "fix" the KEEP-the-arm
ruling eliminated) and keeps (2), saying in terms: *"The mutation in (2) is therefore the only
INV-INPUTS proof available."*

> (2) Mutate step 1 to prefer `_dataset_aggregated.parquet` **on the authorized arm too**.
> Confirm `test_finalize_run_ignores_every_stale_intermediate` goes red. Remove the mutation.

Grepping all four commit bodies for `mutation`, `INV-INPUTS` and `ignores_every_stale` returns
one hit: `858ce22e`'s *"Step 5's mutation recipe is struck with the contradiction named."*
That is accurate about (1) and reads as covering the whole step. **(2) is unrecorded.**

**Mutation to run now** — in `src/phenotypic/_cli/_cli_finalize_run.py`
`select_measurement_sources`, immediately before the `return authorized_sources, True` at
`:90`, insert a preference for `<output>/results/<ds>/measurements/_dataset_aggregated.parquet`
when it exists. Expected: `test_finalize_run_ignores_every_stale_intermediate` red on
`"GHOST.tif" not in master["Metadata_ImageName"]`. If it stays green, INV-INPUTS' only gate is
not a gate.

## MEDIUM 8 — P4 shipped no mutation harness, so none of these 46 cases is in the checker

`docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/mutation_harnesses/` holds
`p1_task3_verification_cache.py`, `p2_task0_disk_verification_cache.py`,
`p2_task1_restart_epoch.py` and `check_mutation_coverage.py`. Each harness binds **one suite**,
and the checker reports a test name from another file as a typo — so `test_finalize_run.py`,
`test_embedded_table_inversion.py` and `test_promote_time_metadata.py` are not merely unproved,
they are outside the coverage report entirely. Running the checker today will not say P4 is
uncovered; it will not mention P4.

P3's substitute — a 1152-cell equivalence gate — was a defensible alternative because it was a
different mechanism aimed at the same property. P4 has neither. The mutations that *are*
recorded (in `8a6d3ae0`: breaking the shard glob, removing D8's return-None gate) all belong to
one test, `test_finalizer_does_not_publish_after_master_parquet_failure`.

The harness README states the decay mechanism this leaves open: *"An unproved test looks
exactly like a proved one from the outside … discovered to have been guarding nothing only
when someone finally breaks the code it named and the suite stays green."* Findings 1–5 above
are four instances of exactly that, found by reading. A `p4_finalize_run.py` harness carrying
the mutations in this report would have found them mechanically, and would keep finding
regressions of them through P5–P7, which all build on this surface.

**Recommendation:** ship one, seeded with the mutations named here. It is the cheapest way to
convert this review into something that stays true.

## LOW 9 — INV-PROVEN's only gate compares mtimes, not content

`test_finalize_run.py:711` `test_finalize_run_writes_no_byte_into_a_proven_store` snapshots
`{path: st_mtime_ns}` before and after. Its guards are correct and its message is honest.

Two things a per-file content digest would catch that this does not: a write that lands inside
the filesystem's timestamp granularity, and a write followed by an `os.utime` restore. Neither
is likely from an accidental regression, but the test is named as INV-PROVEN's *only* gate and
a digest costs the same to write.

**Mutation to size it:** add a line to `finalize_run` that re-writes the store's root
`zarr.json` with byte-identical content, and see whether the test reddens on this filesystem.
If it does, this stays LOW and can be left; if it does not, it is the same class as the three
defects this phase already found.

## LOW 10 — `master_carries_user_metadata` has no production consumer

`grep -rn "master_carries_user_metadata\|user_metadata_headers" src/` returns the definition in
`sdk_/_master_io.py` and the re-export at `sdk_/__init__.py:275,613`. Nothing branches on it.

The helper's own behaviour *is* tested, and adequately — `test_master_carries_user_metadata_
reads_ownership_not_the_prefix` (`test_finalize_run.py:976`) has a positive case, a negative
case, and pins the known limit. So this is not a false green.

What is untested is the claim the module leads with: *"The one genuinely dangerous failure
mode in §7 is a reader that filters or groups a master on a user-metadata column: against a v2
master that returns **empty** rather than raising. This predicate is what such a reader
branches on."* No reader branches on it yet — that is P6's step, by the plan's own division —
so nothing in P4 demonstrates that any reader is protected. Worth recording so P6 does not
inherit the helper as "already covered".

## LOW 11 — three smaller items

**(a) `test_retry_rejects_stale_or_unbound_prior_table_evidence`'s control lives in a sibling.**
`test_cli_recompile_slurm.py:2052`. Both param arms crash `_republish_table_marker`, perturb
evidence, and assert `pytest.raises(RuntimeError, match="measurement authority")`. Nothing
*in the test* shows the unperturbed crashed tree recovers, so the raise could be attributable
to the crash alone. **The control does exist** —
`test_retry_schedules_table_replaced_before_marker_publish_crash` (`:1697-1795`) uses the same
crash point and asserts `build_recompile_tasks` succeeds — so this is not vacuous. But a
reader of the rejection test cannot see it. A one-line cross-reference in the docstring fixes
it; a third param case (`"unperturbed"`, asserting no raise) would fix it structurally.

**(b) The `maintain_order="left"` determinism claim is weakly covered.** `06809fbc` claims the
semi join at `_embedded_measurement_tables.py:157-162` makes *"two identical runs write
byte-identical metadata tables."* The only order assertion anywhere is
`test_duplicate_metadata_keys_preserve_fan_out`'s three-row
`["WT-a", "WT-b", "MUT"]`, and no test compares two runs' `pht-metadata.parquet` bytes.
**Mutation:** drop `maintain_order="left"` from the join and see whether that three-row
assertion catches it — a three-row frame may well come back in order anyway, in which case the
claim has no guard at all.

**(c) `test_the_recompile_guard_still_accepts_a_pre_inversion_store` does not build a
pre-inversion store.** `test_embedded_table_inversion.py:559` builds a *post*-inversion store
with `snapshot=None`. The guard keys on the presence of `tables.metadata`, so the two are
equivalent *for this guard* — but the name asserts a property of the fixture that the fixture
does not have. This is the same defect the phase corrected two files over, when it renamed
`test_nothing_writes_into_a_promoted_store` to `test_promote_store_replaces_rather_than_merges`
for naming more than it proved. Rename to
`test_the_recompile_guard_accepts_a_store_with_no_metadata_table`.

**(d)** `_restore_join_key_dtypes`' warning fallback (`_embedded_measurement_tables.py:36-43`)
is unreached by any test — a dtype that cannot be restored is logged and the column silently
kept as string.

---

## What P4 got right, and should not be lost in a fix pass

Worth recording because the corrections above must not weaken these:

- **`test_finalizer_does_not_publish_after_master_parquet_failure`'s rebuild.** Moving the
  control from a column read out of the master to `blocked == [master_path]` — the write
  *attempt*, recorded by the fault injector — is strictly stronger than the sentinel it
  replaced, and catches the empty-shard case no reading of the master can reach. The commit
  records outcomes written down before the run, both directions.
- **The three preconditions in `test_an_authorized_metadata_run_does_not_lose_the_join`**
  (`test_finalize_run.py:1035`): authority exists, it covers every image, and the store is
  genuinely inverted. Each on its own makes the finding unfalsifiable, and all three are
  asserted.
- **`test_the_legacy_arm_still_prefers_its_dataset_aggregate`** asserts the *opposite* of the
  plan's test body and says why, with both preconditions checked. Pinning the ruling over the
  stale draft is the right call and the docstring makes it auditable.
- **T5's third-party join** (`test_promote_time_metadata.py`) imports no `phenotypic` code in
  the assertion path and drives the real CLI. That is the only construction that can
  substantiate "self-describing", and it is built correctly — including the `keys` non-empty
  guard that stops the merge degenerating into a cross product.
- **The `test_ngff_promote.py` rename** (`test_nothing_writes_into_a_promoted_store` →
  `test_promote_store_replaces_rather_than_merges`) fixes a test whose name asserted more than
  its body across three dependent subsystems. Item 11(c) above is the one remaining instance
  of that same pattern.

## Suggested order of work

1. Findings 2, 3, 4 — one test each, all small, all closing a reachability hole on a guard
   that already exists. These are the ones where a regression is currently invisible.
2. Finding 7 — run the prescribed mutation and record the outcome; it is a five-minute check
   of the phase's headline invariant.
3. Finding 1 — decide whether the v1/v2 property is proven by construction (then say so in
   prose and rename the test) or by comparison (then build a real v1 arm through the retained
   producer). Either is fine; the current shape is neither.
4. Finding 5 — extend the ragged fixture with a third, keyless image.
5. Finding 8 — write `p4_finalize_run.py` seeded with every mutation named above, so findings
   1–7 stay closed through P5–P7.
6. Findings 6, 9, 10, 11 — cheap, and each removes an inference a later reader would otherwise
   have to make.
