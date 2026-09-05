# P2 implementation & test review — `cli-gui-state-tracking`

**Scope.** `git diff 29965f56~1..HEAD -- src/ tests/` (15 commits, 19 files,
~3,200 insertions): the on-disk verification-cache tier, `restart_epoch` and
the liveness-identity fence, the content-derived `processing_generation`, and
the tests that go with them.

**Method.** Code read as the authority; every claim below resolves to a
`file:line` in the worktree. No source or test was edited. No command with a
side effect was run — the findings are static reads, so nothing here rests on
an unexecuted test run. Where a claim would need execution to settle, it is
labelled as such.

**Verdict in one line.** The three named load-bearing mechanisms are, in
themselves, correctly built — but **two of them are wired to inputs the same
function later destroys or never sees**, and the one test written to catch
exactly that class of drift does not constrain the side it names. One
high-severity ordering defect (F1), one high-severity contract defect (F2),
three medium findings, eight low ones, and five coverage gaps. The test suite
is, on the whole, unusually strong: I found no test in the ~1,700 new lines
that is structurally incapable of failing. I did find one that cannot fail for
the reason its docstring gives (F4).

---

## Critical / high confidence

### F1 — `--overwrite` deletes the restart counter *after* the identity has been minted from it

`phenotypicCLI.py:2167` mints the identity, reading `restart_epoch.json`.
`phenotypicCLI.py:2183`, twenty lines later and inside the same straight-line
block, runs `shutil.rmtree(output_dir)` — which removes
`.phenotypic/restart_epoch.json` along with everything else.

The mint comment (`phenotypicCLI.py:2163-2166`) reasons carefully about the
one destructive operation *above* it (`clear_machine_state`, which preserves
the counter by `_PRESERVED_ON_RESTART`, `_io_constants.py:1193-1195`) and does
not consider the destructive operation *below* it. `--restart` and
`--overwrite` are mutually exclusive (`phenotypicCLI.py:1782-1784`), so
`restart=False` on this path and the mint only **reads** — which is why nothing
crashes and why the divergence is silent.

Result: `state.config["restart_epoch"]` is the pre-overwrite counter value
while the counter file is gone (reads as `0`,
`_cli_identity.py:310-313`). Two consequences, both real:

**(a) The liveness fence is inverted for the whole life of the run.**
`initialize_slurm_lifecycle` stamps the record with
`read_restart_epoch(output_dir)` (`_cli_slurm_lifecycle.py:143`) — i.e. `0`.
`_live_authority` requires `_record_restart_epoch(lifecycle) >=
identity.restart_epoch` (`_run_state.py:766`), where `identity.restart_epoch`
comes from `state.config` (`_run_state.py:261-272`). `0 >= 1` is false, so a
genuinely running SLURM job **never** reports `active`; `resolve_run_state`
falls through to `failed`/`incomplete` for the entire run. Degrading away from
`active` is INV-VERDICT's safe direction, so nothing raises — the GUI simply
shows a live run as not-live, permanently.

**(b) The next `--restart` does not fence.** After the overwrite run,
`restart_epoch.json` is absent. A subsequent `--restart` reads `0`, bumps to
`1` (`_cli_identity.py:360`) — the same value the overwrite run recorded — and
`derive_processing_generation` therefore mints the **identical** generation,
because pipeline and per-image config are unchanged. Pre-restart workers pass
the event-log generation fence, which is precisely the failure D5/§14 exist to
prevent.

Reproduction (no code change needed): run once, `--restart` once (counter → 1),
then run `--output <same> --overwrite`, then `--restart` again.

**Fix.** Mint after the overwrite branch — move `identity =
mint_run_identity(...)` below `phenotypicCLI.py:2194`, or have the overwrite
branch re-establish the counter deliberately. Moving it down is the smaller
change and keeps the "dominates both resume sites" property, since the resume
sites are all below 2194 and the overwrite branch is guarded by
`not config.resume and not restart and not measure_only`.

**Confidence:** high. Purely an ordering read; the two line numbers are in the
same block and there is no branch between them that skips the rmtree for a
directory whose counter is non-zero.

---

### F2 — `mint_run_identity` returns a `RunIdentity` that no reader can ever reproduce

`RunIdentity.digest()` folds five fields (`_state_types.py:71-78`). The mint
supplies all five (`_cli_identity.py:265-289`), but **two of them are computed
by rules that differ from the reader's, and three of the five are never read by
any caller.**

| Field | Minted (`_cli_identity.py`) | Read back (`_run_state._identity_from`) | Agree? |
|---|---|---|---|
| `processing_generation` | derived, written to state | `config["processing_generation"]` | yes |
| `restart_epoch` | derived, written to state | `config["restart_epoch"]` | yes (except F1) |
| `inventory_digest` | `canonical_digest(image_manifest_digest)` (`:124-141, :275`) | `canonical_digest(config["work_ids"])` (`_run_state.py:276`) | **never** |
| `scientific_config_digest` | `pipeline_sha256 or ""` (`:280`) | `str(config["pipeline_sha256"] or "")` | yes |
| `finalization_input_digest` | `_metadata_digest_for(config)` (`:281-288`) | `config["metadata_sha256"]` | **often not — see F3** |

`inventory_digest` is the sharp one. The mint's value is the digest of the
**image-manifest fingerprint** — the `--image-manifest` subset token, which
lands in state under the *different* key `image_manifest_digest`
(`_cli_state_management.py:250`). It is `None` for the overwhelming majority of
runs, so the minted `inventory_digest` is a constant
(`canonical_digest(None)`), while the reader's is a digest over the whole
`work_ids` map. The two are never equal for any run.

Today this is inert: `phenotypicCLI.py` consumes only
`identity.processing_generation` (`:2446, :2729, :2772`) and
`identity.restart_epoch` (`:2448, :2730`), and `create_initial_state` consumes
the same two (`_cli_state_management.py:256, 261`). Nothing calls
`.digest()` on a minted identity and nothing passes one to
`assert_identity_current`.

That is exactly what makes it dangerous. The function's name, its return type
and its docstring ("Mint the identity of a new or resumed invocation") all
promise an object interchangeable with `run_identity(output_dir)`. The first
caller that treats it that way — a P4/P7 site keying the verification cache off
the minted identity, or calling `assert_identity_current(output_dir, identity)`
to check nothing moved during the run — gets a **100% mismatch rate**: a
permanent cache cold-start, or a hard `RuntimeError` on every run. Nothing in
the suite would catch it, because no test compares the minted identity with the
read-back one.

**Fix, in order of preference.**
1. Make the mint produce the reader's `inventory_digest`. Not possible at line
   2167 — `work_ids` is not computed until `phenotypicCLI.py:2756-2764` — which
   is itself the argument for (2).
2. Stop returning the three unconsumed fields. Have `mint_run_identity` return
   only what it can compute correctly (`processing_generation`,
   `restart_epoch`) as a small named tuple / dataclass, and let the readers own
   `RunIdentity` construction. This makes the read/write asymmetry §5.2 is
   built on visible in the types.
3. If the `RunIdentity` return type must stay, document at the return site that
   the object is **not** digest-comparable with `run_identity()`, and add a test
   asserting the two digests differ, so the constraint is pinned rather than
   discovered.

**Confidence:** high on the mechanism (two different expressions, both read
directly). Medium on severity, because the consequence is latent — I could find
no current consumer.

---

## Medium confidence

### F3 — `_metadata_digest_for`'s stated guarantee fails on the commonest path

`_cli_identity.py:104-121` states the constraint explicitly: it recomputes
`metadata_sha256` because the snapshot copier stamps it only after state
creation, and *"a different computation here would make the minted
`finalization_input_digest` disagree with the one every later reader derives
from the state, and §7.4's late-metadata guarantee would fire on every run
instead of only on an actual edit."*

The computation is identical (`sha256` over the source CSV's raw bytes,
matching `phenotypicCLI.py:471` and `file_sha256`'s flat-file branch,
`_cli_failure_tracker.py:186-188`). **The inputs are not.**

* At mint time (`phenotypicCLI.py:2167`), `config.metadata_csv` is the user's
  `--metadata` path, or `None`.
* `_prepare_incremental_startup` runs **later** (`:2331`, `:2385`) and
  reassigns `config.metadata_csv = _snapshot_metadata_csv(...)`
  (`phenotypicCLI.py:509-511`), which returns the **existing
  `deliverables/metadata.csv` snapshot** when no source was supplied
  (`phenotypicCLI.py:461-462`).
* The state is then stamped from that reassigned value:
  `state.config["metadata_sha256"] = file_sha256(config.metadata_csv)`
  (`phenotypicCLI.py:2749-2752`).

So on any continuation that does **not** re-pass `--metadata` — the default way
a run is continued — the mint sees `None` and records
`canonical_digest({... "metadata_sha256": None ...})`, while the state records
the snapshot's real digest. The two disagree on every such invocation, which is
precisely the "fires on every run instead of on a real edit" failure the
docstring says it is preventing.

Currently harmless only because of F2 (nothing reads the minted digest). Fix:
resolve the snapshot before minting, or have `_metadata_digest_for` fall back
to `metadata_csv_deliverable_path(config.output_dir)` when
`config.metadata_csv is None`.

**Confidence:** high on the divergence; medium on impact, latent for the same
reason as F2.

---

### F4 — the metadata agreement test does not constrain the CLI side

`tests/unit/cli/test_run_identity.py:487-527`. The docstring is explicit about
its purpose: *"A docstring saying 'keep these identical' is prevention with no
detection. This is the detection: compute both ways over one input and require
them equal."* And it is careful about tautology: *"Spelled out rather than
imported, because the point is that two independently written computations
agree — importing the other one would make this test tautological."*

But the "CLI side" it spells out is a **hand-copy of the same single
expression**, not the CLI's actual code path:

```python
cli_side = hashlib.sha256(metadata.read_bytes()).hexdigest()
assert _metadata_digest_for(config) == cli_side
```

`_snapshot_metadata_csv` is never called. `phenotypicCLI` is never imported.
The test therefore fails only if `_metadata_digest_for` itself changes — the
side that *is* documented and *is* under one reader's eye. Every drift it names
is invisible to it:

* the snapshot copier switching to hashing the destination, or a normalised
  frame, or `file_fingerprint`'s `"sha256:<hex>"` spelling → still green;
* the *input* divergence that already exists (F3) → still green.

**Fix.** Drive both writers over one file and compare the artefacts:

```python
config = make_exec_config(..., metadata_csv=metadata, output_dir=root)
minted = mint_run_identity(config, restart=False)
_snapshot_metadata_csv(root, metadata)            # the CLI's own copier
stamped = json.loads(resolve_processing_state_path(root).read_text())["config"]
assert stamped["metadata_sha256"] == _metadata_digest_for(config)
```

and add the no-`--metadata`-with-existing-snapshot case, which is the one that
currently fails.

**Confidence:** high. The test body is five lines and imports nothing from the
CLI.

---

### F5 — `clear_verification_cache`'s docstring claims a wiring that does not exist, and its own next paragraph contradicts it

`_verification_cache.py:553-554`: *"Rule 4 of Task 3. **P2 wires the scoped form
to `clear_machine_state`**, so discarding a run's tracked state also discards
what was derived from it."*

`clear_machine_state` (`_io_constants.py:1203-1255`) contains no call to
`clear_verification_cache`, and `grep -rn clear_verification_cache src/`
returns only the definition, two re-exports and the import in `_run_state.py`.
There is no caller in `src/` at all — every call site is a test.

Lines 556-562 of the same docstring then say the opposite, correctly:
*"**Tier 1 only, and deliberately.** This function touches no file … A caller
that clears only this one and then resolves shallowly will still be served from
disk."* The two statements cannot both be true.

The wiring is probably not wanted — `_io_constants` is imported *by*
`_verification_cache`, so calling back would be a cycle — and it is not needed,
because a restart bumps the epoch and therefore the identity digest, so tier-1
entries under the old digest are unusable anyway. Behaviourally I could find no
path where the missing wiring produces a wrong answer. **Delete the sentence at
:553-554**; leave :556-562, which is the accurate account.

**Confidence:** high (a grep result), low severity (documentation only).

---

## Low confidence / maintainability

### F6 — `_schema_shape`'s "three of the five signals" note is now stale

`_schema_shape.py:74-77` explains why `SCHEMA_GATE_ARMED` is `False`: *"the
forward path still writes `image_complete/`, still writes
`datasets.<ds>.completed`, and **does not yet write `restart_epoch`**, so three
of the five signals below fire on a tree the running build has just written."*

This phase makes the forward path write `restart_epoch`
(`_cli_state_management.py:261`, `phenotypicCLI.py:2730`) — the change's own
comment at `_cli_state_management.py:257-260` says so. Signal 4
(`_schema_shape.py:272-278`) no longer fires on a current tree, so the count is
now two, not three. The commit that fixed the writer should have updated the
sentence that justified its absence. Nothing enforces the count, so nothing
went red.

### F7 — the recompile-worker equality removal: one cited supplier does not supply

`_cli_recompile_worker.py:225-256` removes `if slurm_generation != attempt_id:
raise` and justifies it by enumerating suppliers. Two of the three citations
hold:

* `_cli_recompile_slurm_scripts.py:292-293` — `slurm_generation=attempt_id,
  attempt_id=attempt_id` ✓
* the manifest writer at `:170, :182, :211, :256` — `task["slurm_generation"] =
  attempt_id` ✓ (four sites, as stated)
* `phenotypicCLI.py:3464` — **not a supplier.** That line calls
  `_wait_for_recompile_finalizer_status(..., slurm_generation=attempt_id)`,
  which takes no `attempt_id` parameter and never reaches
  `_assert_worker_generation`. The only caller of `_assert_worker_generation`
  is `run_recompile_task` (`:114`), whose only invoker is the generated sbatch
  script.

The conclusion (the check was unreachable) is nonetheless correct: the script
generator emits both options from the same `attempt_id`
(`_cli_recompile_slurm_scripts.py:689-695`). But a docstring that says *"Audit
§11.1; confirmed against the tree in P2 Task 4 before removal"* should not carry
a citation that does not resolve — it is the one sentence a later reader will
rely on instead of re-deriving. The removal also ships with **no test**, and
nothing now pins the invariant "every supplier passes one value into both", so
a future supplier passing two distinct values is accepted silently.

### F8 — a `--restart --dry-run` burns an epoch and creates `.phenotypic/`

`mint_run_identity(config, restart=True)` at `phenotypicCLI.py:2167` calls
`bump_restart_epoch`, which `mkdir`s `.phenotypic/` and persists the counter
(`_cli_identity.py:358-361`), **before** the dry-run exit at
`phenotypicCLI.py:2300-2302` and before input scanning can fail. Epochs are
monotonic so a burned one costs only a generation change, and `--restart` has
already run `clear_machine_state` by then, so the destructive precedent is
pre-existing. Still: a `--dry-run` that writes tracked state into a directory is
worth a line of comment at minimum, and `mint_run_identity` is the only
`--dry-run`-reachable writer this change adds.

### F9 — the resume comment's second justification has no replacement

The comment deleted at `phenotypicCLI.py:2431` gave two reasons for the
per-invocation uuid: *"fences workers left by a killed local attempt"* **and**
*"prevents historical started events from remaining active forever."* The
replacement (`:2434-2444`) answers the first with `restart_epoch` and does not
mention the second. With a stable generation, a resume now counts prior
`started` events — pinned deliberately by
`test_a_resume_counts_events_from_its_own_generation`
(`tests/unit/cli/test_run_identity.py:723`).

I traced `in_progress` and it is display-only: `_cli_types.py:50` feeds
`_dashboard/_manifest_builder.py:341, 672-711` and nothing else. A killed
image's `started` event also self-heals, because the image is not in
`completed | failed` and is therefore reprocessed. So the residue is a stale
`active` count in the dashboard for the window between a kill and the retry
finishing. Low, but the second justification deserves an explicit "and this one
is display-only, so it is accepted" rather than silence.

### F10 — the coarse-`mtime` false-current window is now persistent

`entry_is_still_current` fences on `(st_size, st_mtime_ns)` with `ctime_ns`
deliberately excluded (`_verification_cache.py:284-297`, audit S3). A rewrite
of identical size within one filesystem mtime tick reads as current. Under tier
1 that window closed when the process exited; under tier 2 it survives on disk
until the identity changes. The reasoning for excluding `ctime_ns` (chmod,
`rsync -a`, hardlink on GPFS) is sound and I am not proposing to reverse it —
but the module docstring's honest list of what the on-disk tier weakens
(`:37-66`) enumerates *older build*, *mid-write*, *another user* and does not
mention *durability of the stat-granularity window*. It belongs there.

### F11 — `schema_version` check is satisfied by `true`

`_verification_cache.py:446`: `if document.get(_SCHEMA_VERSION_KEY) !=
VERIFICATION_CACHE_VERSION: return None`. `VERIFICATION_CACHE_VERSION` is `1`
and `True == 1` in Python, so a document with `"schema_version": true` passes.
The module already type-guards `bool` in two other places for exactly this
reason (`_is_plain_int`, `:402-404`; `_record_restart_epoch`,
`_run_state.py:795-797`; `read_restart_epoch`, `_cli_identity.py:326`), so the
omission is inconsistent rather than reasoned. Impact is nil under the
acknowledged forgery model — a forger who can write the file can write `1` —
but the fix is `_is_plain_int(v) and v == VERIFICATION_CACHE_VERSION` and it
costs nothing.

### F12 — `_cli_identity`'s import-weight rationale does not match its imports

`_cli_identity.py:255-257` defers `_cli_staged_resume` with *"this module is
imported BY `_cli_slurm_lifecycle`, so it stays import-light at module scope to
keep that edge acyclic."* Two reasons are conflated — weight and acyclicity —
and the module then imports `..sdk_._run_state` at module scope
(`_cli_identity.py:37`) for a single integer constant
(`FINALIZATION_INPUT_SCHEMA_VERSION`), pulling in the whole reader stack. Either
the weight argument applies to both or to neither. Similarly,
`_inventory_digest_for`'s docstring (`:124-133`) asserts a cycle *"the moment
`create_initial_state` takes a `RunIdentity`"* — but `_cli_state_management`
imports `RunIdentity` only under `TYPE_CHECKING` (`:30-31`), so no runtime edge
was created. The duplication may still be the right call; the stated reason is
not the one that holds.

### F13 — three spellings of "finalization input digest" now coexist

* `_cli_completion.py:905-913` (proof-side, written to disk):
  `{metadata_sha256, include_dataset_column, no_qc}` — **no `schema_version`**.
* `_run_state._finalization_inputs` (`:169-174`): the same three **plus**
  `schema_version`.
* `mint_run_identity` (`_cli_identity.py:281-288`): aligns with the reader.

The 1-vs-2 mismatch predates this phase and the mint picked the right side of
it. But the mint is now a third site computing a value under a name that
already means two things on disk, and `_cli_identity`'s A/B block
(`:52-89`) explains at length why `scientific_config_digest` must not be
unified while saying nothing about this one. One sentence there would save the
next reader the same forty minutes.

---

## Test suite assessment

**The strong parts, stated plainly so the weak ones are readable in contrast.**
I looked specifically for the failure mode the brief names — a test asserting a
property of a literal it wrote, or matching a string that cannot match — and
found none in the new code. The suite does several things well:

* **Paired directional tests.** `test_a_record_without_an_epoch_still_counts_on_
  an_unrestarted_run` / `..._is_fenced_on_a_restarted_run`
  (`test_run_identity.py:891, 917`) pin the *direction* of the `0` default,
  not merely its existence; the second docstring correctly notes that
  `sys.maxsize` satisfies the first alone. Same shape for the event-log fence
  pair and for `test_a_restart_moves_the_generation_but_not_any_work_id`
  (`test_run_identity.py:553`), which is the only test that can distinguish "nothing changed" from
  "everything changed".
* **A named control.** `test_a_current_authority_still_reports_the_run_active`
  (`test_run_identity.py:858`) exists solely so the four refusal tests cannot all pass against an
  implementation that refuses everything. That is the right instinct and it is
  rare.
* **`test_a_cold_process_reuses_the_persisted_tier`**
  (`test_verification_cache_disk.py:564`) is the test that would notice a tier
  2 written and never read — the failure every corruption test in that file
  passes through unharmed. Naming it as such in its own docstring is correct.
* **`test_a_fully_warm_shallow_pass_does_not_rewrite_the_file`** (`test_verification_cache_disk.py:582`) uses
  a witness key rather than mtime, which is the right choice on a
  coarse-granularity filesystem and is explained as such.
* **`test_migrate_mode_is_never_refused_by_the_gate`**
  (`test_schema_gate.py:793`) replaces a negative string assertion with a spy
  on `run_migrate` — and the spy works, because `handle_migrate_mode` calls
  `run_migrate` by global lookup within `_cli_migrate`
  (`_cli_migrate.py:2343`), which `monkeypatch.setattr(_cli_migrate,
  "run_migrate", ...)` intercepts. The `assert reached` is a genuine positive.
* **`test_neither_module_ever_names_the_cli_package`**
  (`test_run_state_layering.py:70`): I hand-checked `_absolute_module` against
  every level. `level=2, module='_cli'` → `phenotypic._cli` ✓; `level=2,
  module=None` → `phenotypic`, with `_cli` caught via `node.names` ✓;
  `level=1` → `phenotypic.sdk_.<mod>` ✓; `from phenotypic import _cli` ✓. The
  old `startswith("._cli")` prefix genuinely could never match, as the
  docstring says.

**Weaknesses.**

1. **F4 above** — the only outright false green in the new tests.
2. **`test_measure_mints_the_identity_a_full_run_would`**
   (`test_run_identity.py:601`) overclaims. It builds two configs that differ
   only in `measure_only`, so it proves exactly one thing: `measure_only` is not
   folded into `processing_configuration_digest`. It does **not** establish
   DF-16's "measure runs under the same identity as full", which additionally
   requires `include_dataset_column`, `overlay_alpha` and `save_overlays` to
   match (`_cli_failure_tracker.py:228-233`). A measure run invoked with a
   different `--overlay-alpha` than the full run mints a different generation
   and, because measure never rewrites state
   (`phenotypicCLI.py:2680`), will fail `_local_epoch_ownership`'s check
   (`_cli_execution_strategies.py:94-103`) with *"Local lifecycle epoch is
   stale"*. That is not a regression — measure's `uuid4()` guaranteed the same
   mismatch before — but the docstring reads as if the parity now holds
   unconditionally, and it does not.
3. **`test_the_proof_side_digest_is_a_different_value_entirely`** (`test_run_identity.py:158`)
   inspects `processing_configuration_digest_from_values.__code__.co_varnames`
   for the absence of `"pipeline_sha256"`. It can fail (adding such a parameter
   turns it red), but `co_varnames` includes locals as well as parameters, and
   an equivalent component added under any other name — `pipeline_digest`,
   `pipeline_identity` — sails through. The docstring's claim that "the two
   cannot be substituted for one another" is stronger than what a name check
   buys.
4. **Two `chmod`-gated tests** (`test_a_failed_write_raises_rather_than_
   returning_quietly`, `test_persisting_into_a_read_only_output_is_not_an_
   error`) both guard with `if os.access(dir, os.W_OK): pytest.skip(...)`, which
   is the right shape. Flagged only because this repository already has a
   recorded history of chmod-based tests behaving differently on GPFS than on
   local disk; on a filesystem where `os.access` and the actual write disagree
   these will fail rather than skip. Not a defect — a thing to remember when
   one of them goes red on a compute node.
5. **Unverifiable claims in docstrings.** `test_migrate_mode_is_never_refused_
   by_the_gate` states *"1 failed, 56 passed, 10 skipped"* for a mutation, and
   `_verification_cache.py:5-8` states 1403 s vs ~37 s at N=6,657. Both are
   plausible and neither is checkable from the tree. Recording a measured number
   is better than not recording it; I note only that a reader cannot confirm
   them, and that per CLAUDE.md's `logic_validation_scripts/` rule a numeric
   invariant a design rests on is meant to have a runnable witness.

---

## Coverage gaps

Ranked by what they would have caught.

1. **Nothing pins `state.config["restart_epoch"] == read_restart_epoch(output_dir)`.**
   This is the invariant `_live_authority` (`_run_state.py:766`) is built on:
   one side comes from the state, the other from the counter file, and F1 is
   exactly what happens when they part. Every fence test in
   `test_run_identity.py` sets `config.restart_epoch` **by hand**
   (`_set_config_restart_epoch`, `:66-77`) and separately publishes a lifecycle
   record, so no test ever observes the two produced by the same invocation.
   Add one that drives the CLI (or `mint_run_identity` +
   `initialize_slurm_lifecycle` in sequence) and asserts equality.
2. **No `--overwrite` test at all** in the identity suite — the path that
   creates F1.
3. **No end-to-end test that a minted identity is (or is not) comparable with
   `run_identity(output_dir)`** — the gap behind F2. Even an
   `assert mint.digest() != run_identity(out).digest()` with an explanatory
   message would convert a trap into a documented constraint.
4. **`_metadata_digest_for` is never tested against the actual snapshot
   writer** (F4), and the no-`--metadata`-with-existing-snapshot case (F3) is
   untested in either direction.
5. **The recompile-worker equality removal has no test** (F7). A test asserting
   that the generated script emits `--slurm-generation` and `--attempt-id` from
   one value would pin the invariant the removal now relies on.

Two smaller ones: nothing exercises `derive_processing_generation` with a
negative or very large `restart_epoch` (the reader clamps negatives to `0`,
`_cli_identity.py:326`, but the *deriver* accepts any `int`, so a state file
hand-edited to `-1` and a fresh mint at `0` produce different generations from
the same tree); and `persist_states`' `TypeError` branch is covered
(`test_an_unserializable_stage_value_is_not_an_error`) while its `ValueError`
branch — a `stages` body that is not a mapping, so `dict()` raises — is not.

---

## Answers to the four questions in the brief

**1. Does `_metadata_digest_for`'s agreement test constrain both sides?**
No. It re-implements the same one-line expression it is testing and never
touches `_snapshot_metadata_csv` or `phenotypicCLI` (F4). And the drift it was
written to prevent has already occurred, on the input rather than the algorithm
(F3).

**2. Is any `mint_run_identity` component accepted and not folded into the
digest?** All three `derive_processing_generation` components are folded and
each is pinned individually by `test_every_component_moves_the_generation`.
The larger problem is the inverse: `mint_run_identity` computes three
`RunIdentity` fields that **are** folded into `RunIdentity.digest()` and that
**no caller reads**, two of them by rules that do not match the readers' (F2).

**3. Does the mint point dominate both call paths, and does `clear_machine_state`
invalidate the value read below it?** Dominance: **yes**, verified. `identity =
mint_run_identity(...)` is at 8-space indent (`phenotypicCLI.py:2167`),
unconditional, with no alternate entry into the block; both the
incremental-reconciliation site (`:2446`) and the main state block (`:2729`,
`:2772`) are below it in the same linear flow. `clear_machine_state`: **no** —
`_PRESERVED_ON_RESTART` (`_io_constants.py:1193-1195`) keeps
`restart_epoch.json`, the sweep skips it (`:1234-1235`), and
`test_restart_epoch_survives_clear_machine_state` pins it. But the operation
**below** the mint does invalidate it (F1), and that is the one nobody checked.

**4. Is there a path where the on-disk tier does not degrade to a deep pass?**
I could not find one. I traced: missing file, directory-in-place-of-file,
undecodable bytes, malformed JSON, non-dict document, wrong schema version,
wrong identity, non-str work_id, every field type in `_entry_from_json`,
`bool`-as-int, empty `stat_tuples`, moved/deleted/non-regular fenced paths,
read-only output, absent `.phenotypic/`, unserializable stage bodies — all
return `None`/`False` and fall through. The one structural risk is not a
missing degrade but a **laundering** path: when `escalated` is true,
`persist_states` (`_run_state.py:1153-1154`) rewrites the *whole* entry map,
including entries this process took from tier 2 and never re-verified, under
the current build's `schema_version`. That is safe **only** because a rules
change is accompanied by a version bump, which is a discipline rather than a
mechanism — and `VERIFICATION_CACHE_VERSION`'s own comment
(`_io_constants.py:695-703`) says so. It is worth stating in
`persist_states`' docstring that it re-blesses entries it did not verify;
today only the module preamble hints at it.

Two smaller notes on the same question. Negative verdicts cannot be laundered:
`_verify_image`'s failure branch returns `stat_tuples={}`
(`_run_state.py:640`), which is permanently non-current and is dropped by
`persist_states`' `if entry.stat_tuples` filter — so a `failed`/`unverified`
image is re-verified on every pass, which is the behaviour a run that might yet
succeed needs. And `_fenced_artifact_path` correctly returns a store's root
`zarr.json` rather than the store directory (`_run_state.py:458-466`), which is
what keeps `entry_is_still_current`'s fail-closed-on-directories rule from
turning every store-bearing image into a permanent cache miss.
