# Phase 2 — spec-adherence review

**Reviewer:** independent (did not write this code). **Question asked:** *is this what we
said we would build, all of it?* — not *is it correct?*

**Range reviewed:** `29965f56~1..HEAD` (31 commits, `29965f56` → `23574266`), restricted to
`src/` and `tests/` for the diff and extended to `docs/` where a document is the thing that
drifted. 19 files, +3204 / −86.

**Authorities read:** `design.md` §0 (authoritative over later sections), §3 D3–D7, §4.2–4.3,
§5.1–§5.5, §9.1–§9.2, §11.1, §12, §13, §14; `phase-2-identity-schema.md` in full;
`EXECUTION.md`'s *"Spec-adherence reviewer — the brief"*; `document-drift.md`'s 20 entries;
`spec-adherence.md` (P1's gate, for its four open items and its *For the next reviewer* list).

**Method.** Every citation below was regenerated against the working tree, not copied from a
plan document. Where I claim a symbol has no test or no consumer, that is a `grep` over
`src/` and `tests/` run for this review, not an inference from a file list.

---

## Summary

| Category | Count | Highest severity |
|---|---|---|
| **A** — specified, not implemented | 0 | — |
| **B** — planned, not done | 4 | **B-3** (a planned assertion replaced by a weaker one that hides a live gate failure) |
| **C** — implemented, but differs | 4 | **C-2** (a user-ruled rename applied to the code and to one spec section, left standing in the four places a reader looks up the formula — including the P7 register) |
| **D** — implemented, never specified | 4 | none — all four are latitude; see each disposition |
| **E** — implemented as a placeholder | 1 | **E-1** (`_inventory_digest_for` returns a constant on the default path, and disagrees with the other producer of the same field) |

**E-1 and B-1/B-2 are one defect seen from two sides**, and together they are the most
expensive thing here: `mint_run_identity` populates seven `RunIdentity` fields, two of them
are consumed in P2, and of the five that are not, **two are both unspecifiable-from-the-code
and untested**. A placeholder that nothing downstream reads yet is exactly the shape category
E exists for.

**What the phase got right, stated plainly** (detail in *Satisfied obligations* below): the
content-derived generation, `restart_epoch` and its preserve set, rule 2's first half, the
on-disk verification tier with all six of §9.1's fall-through cases, §14's stale-worker fence
pinned as a pair, `_assert_worker_generation`'s §11.1 deletion, and P1's open **B-1**. The
mutation harnesses (`p2_task0_disk_verification_cache.py`, `p2_task1_restart_epoch.py`) carry
33 mutations between them and are unusually honest about which claims they had to withdraw.

---

## Category A — specified, not implemented

**Empty.** Every §5 requirement this phase claims has code behind it. The two candidates I
chased and discharged:

- **§5.1's `scheduler_epoch` absorption** (`design.md:314`) has no code. It is not category A
  because §0's own amendment (`design.md:323-345`, user-ruled) withdraws the requirement. See
  **C-1**.
- **§5.2's `mint_run_identity` shown under `from phenotypic.sdk_`** (`design.md:363-371`) is
  not exported from `sdk_` — `sdk_/__init__.py` gains only `RESTART_EPOCH_JSON`,
  `VERIFICATION_CACHE_JSON`, `VERIFICATION_CACHE_VERSION`, `restart_epoch_path`,
  `verification_cache_path`. Not category A: the same block annotates the symbol `CLI only`,
  and §5.2's prose (`design.md:373-375`) argues for exactly the placement shipped — *"`sdk_/_run_state.py`
  exports only readers, so the GUI cannot reach a `publish_*` function"*. The two halves of
  §5.2 cannot both hold; the implementation satisfies the one the section argues for, and
  INV-LAYER's AST walk (`tests/unit/sdk_/test_run_state_layering.py:56-144`) enforces it.
  The import block is a fourth instance of the drift register's shape #1 and should be
  corrected in the same pass as **C-2**.

---

## Category B — planned, not done

### B-3 — the migrated-tree gate test was replaced by a weaker one, and the stronger one would fail · **highest severity in this category**

**Plan** (`phase-2-identity-schema.md:641-648`):

```python
def test_a_freshly_migrated_tree_is_not_refused_by_the_next_full_run(tmp_path):
    ...
    assert requires_conversion(output) is None
```

**Landed** (`tests/unit/cli/test_run_identity.py:286-303`),
`test_a_migrated_tree_records_a_restart_epoch`:

```python
    assert _state_config(root)["restart_epoch"] == 0
```

The field, not the verdict. And the verdict assertion would fail today, because **the third
half of the same plan step did not land**. `phase-2-identity-schema.md:616-617` says:

> Bring it to the v3 schema: content-derived generation, `restart_epoch: 0`, **no
> `datasets.{completed,failed,started}`**.

`_ensure_migration_processing_state` still builds populated `DatasetState`s
(`_cli_migrate.py:647-657`), and `save_processing_state` writes `completed`/`failed`
unconditionally for every dataset (`_cli_state_management.py:82-88`). So a freshly migrated
tree carries `datasets.<ds>.completed`, and `requires_conversion` returns `CONVERT` on
**signal 3** (`sdk_/_schema_shape.py:252-260`) before it ever reaches signal 4 at `:271-277`.

**Two shipped comments assert the consequence that does not follow.**

- `_cli_migrate.py:711-713`: *"P1's `requires_conversion` signal 4 is the ABSENCE of this key.
  A freshly migrated tree without it is refused by the very next `--mode full`, which is the
  gate firing on its own migrator."* True about signal 4; the refusal it implies is removed
  is not, because signal 3 is still live.
- `_cli_state_management.py:257-260`: *"until this line existed the schema gate fired on every
  tree the current build wrote — the gate armed against its own writer."* Still fires, on
  signals 1 and 3.

**Mitigating, and it is real:** `SCHEMA_GATE_ARMED` is `False` (`_schema_shape.py:65-78`), so
nothing surfaces today, and removing `datasets.{completed,failed}` is §4.2's job in P3+ — it
could not have been done in P2 for the migrator alone without doing it for the forward writer
too. **That makes it a phase-ordering conflict in the plan, not an implementer's omission —
which is precisely why it belongs in the drift register and is not there.** As written, a P7
agent reading `phase-2-identity-schema.md:616` will believe the migrator already emits a
signal-3-clean state.

**Recommended disposition:** amend `phase-2-identity-schema.md:616` to say the derived-set
removal is P3's, re-point the two code comments at what actually changed (signal 4 only), and
move `test_a_freshly_migrated_tree_is_not_refused_by_the_next_full_run` to the phase that can
make it pass. It is INV-DISCHARGEABLE's migrate half, which
`test_schema_gate.py:1013-1023` already defers to P7 Task 5 with a correctly-formed skip —
that is where this assertion belongs.

### B-1 — `test_a_new_image_does_NOT_mint_a_new_generation` did not land

**Plan** `phase-2-identity-schema.md:515-528`:

```python
    assert after.processing_generation == before.processing_generation
    assert after.inventory_digest != before.inventory_digest
```

Not in the tree. `grep -rn "inventory_digest" tests/unit/cli/test_run_identity.py` returns one
hit, inside a docstring at `:496`. No test anywhere asserts the value of
`RunIdentity.inventory_digest` as produced by `mint_run_identity`, and the mutation harness
`p2_task1_restart_epoch.py` contains no mutation naming it.

The D7 half of the plan's assertion **is** covered, twice, at the primitive
(`test_run_identity.py:210-235`, the parametrized component test) and at the migrator
(`:254-283`). What is uncovered is the second line — and that second line is the one that
would have caught **E-1**.

### B-2 — `test_a_metadata_edit_does_NOT_mint_a_new_generation` did not land

**Plan** `phase-2-identity-schema.md:531-543`:

```python
    assert after.processing_generation == before.processing_generation
    assert after.finalization_input_digest != before.finalization_input_digest
```

Not in the tree. `test_the_two_metadata_digests_agree` (`test_run_identity.py:487-529`) is a
different and genuinely valuable test — it pins `_metadata_digest_for` against the CLI's own
computation — but it stops at the helper. The four-key object `mint_run_identity` builds at
`_cli_identity.py:281-288` is never asserted, so nothing detects a wrong
`FINALIZATION_INPUT_SCHEMA_VERSION`, a dropped `no_qc`, or a key spelled differently from
`_run_state._finalization_inputs`' (`_run_state.py:161-174`) — the reader that every later
comparison goes through.

This one has a live consumer waiting: §7.4's late-metadata guarantee (`design.md:666`) and
rule 1's fifth comparison (`_run_state.py:892-940`). The gate brief singles out exactly this
risk for `metadata_sha256` and calls the agreement test *"category E's concern"* — the agreement
test was written for the **helper** and not for the **object**.

### B-4 — `test_a_stale_slurm_worker_cannot_publish` did not land, and the guard it targets has no test at all

**Plan** `phase-2-identity-schema.md:774-790`, written *after* CAN-15 diagnosed the first
draft as targeting a mechanism that is not there, and corrected to target the one that is.

`grep -rn "stale SLURM lifecycle" src/ tests/` returns exactly one line:
`src/phenotypic/_cli/_cli_completion.py:191`. The guard at `_cli_completion.py:181-192` —
`SLURM_JOB_ID` set, a lifecycle record present, generation mismatched or not active → raise —
is untested anywhere in the tree.

Not covered by §5.1's amendment: the amendment withdraws the **rename**, and this test is
about the guard's behaviour, which the rename never touched. Cheap to add and it needs both
preconditions the plan spells out, including the real artifact (`publish_image_success`
resolves every artifact `strict=True`, `_cli_completion.py:196`).

---

## Category C — implemented, but differs

### C-2 — the `per_image_config_digest` rename stopped one section short, in the four places a reader looks up the formula · **highest severity in this category**

**Disposition: row 2 — the code is right and the documents are now wrong.** Do not touch the
code. The ruling exists (user, P2 Task 3), the evidence exists (drift register entry 14), and
`design.md:422-453` is amended. What is missing is the rest of the sweep.

The rename's whole point is that **`scientific_config_digest` names a different value** — the
pipeline file's bytes, written into every proof (`_cli_completion.py:914,1020,1087`) and read
back at `_run_state.py:277`. After the rename, these still spell the per-image digest with the
proof-side name:

| Where | Text | Reads as, post-rename |
|---|---|---|
| `design.md:310` | `` `sha256(pipeline_sha256 ‖ scientific_config_digest ‖ restart_epoch)` `` | `sha256(pipeline_sha256 ‖ pipeline_sha256 ‖ restart_epoch)` — **the one place a reader looks up the formula** |
| `design.md:24` (D-C, §0) | *"`scientific_config_digest` **is** `processing_configuration_digest`, verbatim"* | now **false**; `per_image_config_digest` is (`_cli_identity.py:90`) |
| `design.md:25`, `:387` (U-12) | *"`include_dataset_column` is in `scientific_config_digest` (`_cli_failure_tracker.py:230`)"* | names the per-image digest by the proof-side name, at the citation that proves it is the per-image one |
| `phase-7-migrate-mode.md:1373` | `` `processing_generation` | `sha256(pipeline_sha256 ‖ scientific_config_digest ‖ restart_epoch)` `` | **the P7 token register** — this change's stated deliverable |
| `phase-2-identity-schema.md:60, 392, 404-416, 434, 554` | the whole of Task 2, still under the old name | a reader executing the plan writes the collision back |

The register's own conclusion from entry 14 is *"when a name is introduced that already means
something elsewhere in the same system, the cheap moment to fix it is the commit that
introduced it. Deferring converts a naming problem into a migration problem."* The rename was
taken at that cheap moment for the **code**; four documents were left at the expensive one,
and one of them is the register that will document the shipped tokens.

The tree does defend the code side: `test_the_proof_side_digest_is_a_different_value_entirely`
(`test_run_identity.py:158-183`) and `test_the_minted_proof_side_digest_is_the_pipeline_digest`
(`:398-424`), each with a mutation. Nothing defends the documents, which is the register's
entire thesis.

### C-1 — the `scheduler_epoch` collapse and the `publish_image_success` rename were not done

**Disposition: row 1 — legitimate, with one qualifier.** Plan `phase-2-identity-schema.md`
File Structure rows 8 and 9 (`_cli_slurm_lifecycle.py:78` → `scheduler_epoch`;
`_cli_completion.py:163` takes `scheduler_epoch`) and Task 4 Step 4 (`:802-805`) were not
implemented. Verified: `grep -rn "lifecycle_epoch" src/phenotypic/` returns 40 hits across 12
modules, and `publish_image_success` still takes `lifecycle_epoch` (`_cli_completion.py:173`).

The requirement was withdrawn: `design.md:323-345` amends §5.1 to *"the five-token collapse is
achievable ZERO times"*, user-ruled, with a writer-by-writer table; register entry 19 records
both the finding and the orchestrator's own error inside it. The rejected alternative
(read-both-keys shims) is named with its reason.

**The qualifier, and it is why I am not calling this row 3.** The rule's first row asks for an
*experiment*. What is recorded is a **code audit**, not a measurement — five citations checked
against the shipped tree. For a claim of the form *"this name is a public format read by three
GUI sites"*, an audit is the correct instrument and a benchmark would answer nothing; the
audit is checkable, was checked, was found wrong once and corrected before the ruling landed.
That is evidence in the sense the rule means. Recording it as such matters, because a later
reader comparing this to U-11's 1403 s-vs-37 s measurement will otherwise conclude the bar
moved.

**One supporting fact the amendment does not cite, and should.** `lifecycle_epoch`'s row gives
only the runtime-mode-dependence argument (`_authoritative_lifecycle_epoch`,
`_cli_process_single.py:115`). It is *also* an on-disk key — `publish_image_success` writes
`"lifecycle_epoch"` into every image marker at `_cli_completion.py:234` — which is the same
persistence argument that pins `slurm_generation`, and a stronger one than mode-dependence
since it survives any future change to the value's meaning. Adding it costs one line and
closes the row against the obvious rebuttal (*"then narrow the value"*).

**Not done and correctly so:** Task 4 Step 4's *"keep the staged `epoch` and recompile
`attempt_id` as diagnostic fields written under the collapsed name"* presupposes the collapsed
name. It follows the withdrawal.

**Task 4 Step 4's other branch was resolved correctly.** `_assert_worker_generation`'s
`slurm_generation != attempt_id` was deleted (`_cli_recompile_worker.py:225-256`, commit
`9a364dc4`), which is what §11.1 (`design.md:884-885`) lists. The plan required either the
deletion *or* a `_cli/CLAUDE.md` register line; the deletion means no line is owed.

### C-3 — `_schema_shape.py:74-78` is now false, and P2 made it false

**Disposition: row 2 — amend the document.** `SCHEMA_GATE_ARMED`'s docstring says:

> the forward path still writes `image_complete/`, still writes `datasets.<ds>.completed`, and
> **does not yet write `restart_epoch`, so three of the five signals below fire** on a tree the
> running build has just written.

`create_initial_state` now writes it (`_cli_state_management.py:261`), so **two** of five fire.
The sentence was true at P1 and is stale as of this phase. Not in the register.

Benign in effect — the flag stays `False` either way — but it is the register's shape #2, *a
claim about what a check does*, in the docstring of the flag that gates both consumers.

### C-4 — `_verification_cache.py:553` claims a wiring P2 did not do

**Disposition: row 2 — amend the document.** `clear_verification_cache`'s docstring, in a
paragraph P2 rewrote:

> Rule 4 of Task 3. **P2 wires the scoped form to `clear_machine_state`**, so discarding a
> run's tracked state also discards what was derived from it.

It does not. `grep -rn "clear_verification_cache" src/phenotypic/` returns only the definition
(`_verification_cache.py:550`), two re-exports (`sdk_/__init__.py:271,419`;
`_run_state.py:74,101`) and comments. `clear_machine_state` (`_io_constants.py:1203-1243`)
never calls it.

**Effect is nil, for a reason worth writing down rather than relying on:** a restart bumps the
epoch, which moves `processing_generation`, which moves `RunIdentity.digest()`, and
`cached_states` refuses a non-exact identity match wholesale
(`_verification_cache.py:199-206`). Tier 1 is therefore *unreachable* after a restart rather
than *cleared*. That is a correct outcome reached by a different mechanism than the docstring
names — which is exactly the distinction this docstring is otherwise very careful about
elsewhere (the same paragraph correctly says `clear_machine_state` removes tier 2 *"by removing
every child of `.phenotypic/`"*, i.e. passively).

---

## Category D — implemented, never specified

All four are **latitude, not drift**, and none owes an experiment. I state for each what the
spec constrains, and why the thing chosen is outside it.

### D-1 — `VERIFICATION_CACHE_VERSION` and a seventh fall-through case

`_io_constants.py:695-701`; enforced at `_verification_cache.py:418-419`. §9.1
(`design.md:826-828`) enumerates **six** ways a shallow pass must fall through to deep; a
schema-version mismatch is not among them. §9.1 constrains *what must fall through*, not *what
may not* — and its invariant (`design.md:830-834`) permits any additional cause, since every
one of them moves toward `deep`. Not drift. Tested
(`test_a_different_schema_version_is_refused`, `test_verification_cache_disk.py:384-403`) with
a mutation. The docstring's instruction to bump on a *rules* change, not only a *shape* change,
is the right rule and is stated where an editor will read it.

### D-2 — `derive_processing_generation`

`_cli_identity.py:144-197`, exported in `__all__`. Not in §5.2's function surface
(`design.md:363-371`) nor the plan's Interfaces block (`phase-2-identity-schema.md:47-70`).
§5.1 (`design.md:310`) constrains the **formula**, which this implements; U-7
(`design.md:33`) rules that migrate builds no `ExecutionConfig`, and `_cli_migrate.py:706-710`
is the caller that forces a config-free entry point. The spec constrains the formula and the
layer; it is silent on decomposition. Not drift.

One implementation choice inside that latitude is worth a reviewer's eye and is *not* a
finding: the components are digested as a **mapping**, not the `‖` concatenation §5.1's
notation implies. Self-describing and extension-safe, argued at `:165-167`, and no value it
produces has ever been on disk (the sites it replaces all wrote `uuid4()`), so there is nothing
to invalidate.

### D-3 — the mint-once guard is an attribute set on the `ExecutionConfig` instance

`_cli_identity.py:96-101, 238-244, 264`. The plan specifies the guard and its `RuntimeError`
(`phase-2-identity-schema.md:568-582`); the mechanism is unspecified. `ExecutionConfig` is a
plain mutable dataclass (`_cli_types.py:95-96`) with no `__slots__`, and I found no
`asdict`/`vars`/`__dict__` serialization of it in `_cli/` or `phenotypicCLI.py`, so the flag
cannot leak into a state file. Not drift. Tested (`test_run_identity.py:336-347`) with a
mutation.

### D-4 — the SLURM lifecycle record grew a `restart_epoch` field

`_cli_slurm_lifecycle.py:110-146`. Added during execution and recorded in the plan
(`phase-2-identity-schema.md:225-235`) with the rejected alternative — a `restart_epoch=`
parameter — and its reason: a caller could stamp an epoch that was never current, so the fence
would assert the caller's belief rather than the run's state.

§5.1's `restart_epoch` row (`design.md:312`) constrains it as *"monotonic int; preserved by
`clear_machine_state`"* and says nothing about a liveness record carrying it; §4.1
(`design.md:267`) lists `slurm_lifecycle.json` as a liveness authority without fixing its
fields. But `README.md:113-114`'s rule 2 requires an authority to report work in flight **for
the current identity**, which is uncomputable without this field — so this is a *forced*
choice, not a free one. Not drift.

Tested in both directions and at the writer independently of the fence
(`test_run_identity.py:817-855` for the writer, `:858-947` for the four fence cases,
`:950-976` for the deliberate GUI-owner asymmetry), each with a mutation, including one whose
first version was found not to model the bug at all and is documented as such
(`p2_task1_restart_epoch.py:411-431`).

---

## Category E — the placeholder sweep

### The mechanical sweep — clean

Run over the ten changed `src/` files and the nine changed `tests/` files:

- `NotImplementedError|TODO|FIXME|XXX|placeholder|for now|stub` — **zero hits** in `src/`.
- Bodies that are only `...` or `pass` — two hits, `_cli_slurm_lifecycle.py:810` and
  `_io_constants.py:1137`, both outside this range's hunks (pre-existing `except: pass`
  degrade paths).
- Skipped or xfailed tests — four hits, all legitimate:
  - `test_run_identity.py:806` and `test_verification_cache_disk.py:442`: runtime
    `pytest.skip` guarding a `chmod`-based permission test against a root runner, with the
    condition named.
  - `test_cli_v2.py:1798`: pre-existing platform `skipif`.
  - `test_schema_gate.py:1013-1019`: `@pytest.mark.skip` naming both the condition
    (*"`--mode migrate` does not yet convert `.phenotypic/`"*) and the phase that removes it
    (*"P7 Task 5 removes this mark; it is that phase's gate"*) — the correct form the gate
    definition cites, and pre-existing to this range.

### Every symbol the plan names, and what its body computes

| Symbol | What it computes |
|---|---|
| `mint_run_identity` (`_cli_identity.py:200`) | Refuses a second call on the same config; requires an `output_dir`; bumps or reads the restart epoch; digests the pipeline file when present; returns a `RunIdentity` with five derived fields and two `None`s. |
| `read_restart_epoch` (`:292`) | Reads `.phenotypic/restart_epoch.json`; returns its `restart_epoch` when it is a non-`bool`, non-negative `int`; returns `0` for every other outcome including OSError and malformed JSON. |
| `bump_restart_epoch` (`:331`) | Creates `.phenotypic/`, reads the current value, atomically writes value+1, returns it; propagates `OSError`. |
| `per_image_config_digest` (`:90`) | A module-level **alias** of `processing_configuration_digest` — no body, which is the point (`is`-checked at `test_run_identity.py:155`). |
| `derive_processing_generation` (`:144`) | `canonical_digest` of `{pipeline_sha256, per_image_config_digest, restart_epoch}` with `None` normalized to `""`. |
| `_metadata_digest_for` (`:104`) | sha256 over `config.metadata_csv`'s raw bytes, or `None` when the path is absent or not a file. |
| **`_inventory_digest_for` (`:124`)** | **`canonical_digest` of `config.image_manifest_digest`, else of the manifest file's digest, else of `None` — a constant whenever `--image-manifest` was not passed. See E-1.** |
| `load_persisted_states` (`_verification_cache.py:407`) | Reads and fully type-validates the on-disk document; returns a read-only `work_id → CachedVerification` map, or `None` for any of file-missing / unreadable / unparseable / non-object / wrong schema version / wrong identity / one malformed entry. |
| `persist_states` (`:464`) | Declines when `.phenotypic/` does not exist; otherwise atomically writes `{schema_version, identity_digest, entries}`, omitting entries with no stat tuples; returns `False` for any `OSError`/`TypeError`/`ValueError`. |
| `warm_states` (`:525`) | Returns tier 1 if present, else tier 2, else `None`. |
| `_entry_to_json` / `_entry_from_json` (`:316` / `:339`) | Render one entry without its `work_id` key / rebuild it with every field type-checked, `bool` rejected as an `int`, and any deviation returning `None`. |
| `_record_restart_epoch` (`_run_state.py:781`) | Returns the record's `restart_epoch` when it is a non-`bool` `int`, else `0` — degrading *downward*, which fences a doubtful authority. |
| `_live_authority` (`_run_state.py:710`) | Returns the lifecycle filename when the record is `active` **and** its epoch ≥ the identity's; else the GUI owner filename when its status is in flight and its pid is alive; else `None`. |
| `initialize_slurm_lifecycle` (`_cli_slurm_lifecycle.py:107`) | Under the lock: returns a standing active record unmodified; otherwise writes a new record stamped with the epoch read at that moment. |
| `create_initial_state` (`_cli_state_management.py:183`) | Unchanged body plus two config keys taken from a now-**required keyword-only** identity. |
| `_assert_worker_generation` (`_cli_recompile_worker.py:226`) | Requires both arguments non-empty, then `assert_generation_active`; the self-comparison is gone. |

No body here is unsummarisable without repeating its signature, save one.

### E-1 — `_inventory_digest_for` returns a constant on the default path, and disagrees with the other producer of the same field

**The body** (`_cli_identity.py:124-141`):

```python
    digest = getattr(config, "image_manifest_digest", None)
    if digest is None:
        manifest = getattr(config, "image_manifest", None)
        if manifest is not None:
            from ._cli_directory_scanner import image_manifest_digest
            digest = image_manifest_digest(manifest)
    return canonical_digest(digest)
```

`image_manifest_digest` is set at exactly one site, `phenotypicCLI.py:2011`, guarded by
`if image_manifest is not None`, and `image_manifest` is the optional `--image-manifest` file
(`_cli_types.py:170,173`). **On any run that does not pass `--image-manifest` — the default —
both lookups are `None` and this returns `canonical_digest(None)`, the same 64 characters for
every run, forever.**

**What the spec constrains, and whether this is the thing being chosen.** §5.3
(`design.md:406`) defines `inventory_digest` as the answer to *"Did the accepted **scope**
change?"*, and §5.3's following paragraph (`design.md:410-416`) makes the `inventory`/`source_set`
split load-bearing: *"if ten new images arrive and three fail … `inventory_digest` has changed,
so the run is no longer complete for its current scope."* D7 (`design.md:255`) restates it:
*"generation fences configuration; `inventory_digest` fences scope."* Rule 1 spells out the
consequence of losing it — *"without `inventory_digest` a new image under a rolling input never
invalidates completion"* (`_run_state.py:908-910`), and enforces it at `_run_state.py:936`.

That is not silence. The spec constrains **what the digest must be able to distinguish**, and a
constant distinguishes nothing. This is category E shape #1 — *returns a constant where the
spec requires derivation* — reached not by writing `return ""` but by deriving from an input
that is absent on the default path.

**The second half, and the one that makes it expensive.** `RunIdentity` now has **two**
producers that compute this field differently:

| Producer | `inventory_digest` |
|---|---|
| `_identity_from` — the P1 reader (`_run_state.py:276`) | `canonical_digest(config.get("work_ids", {}))` — the accepted inventory, which *does* move when an image arrives |
| `mint_run_identity` — the P2 writer (`_cli_identity.py:275`) | `canonical_digest(<manifest digest or None>)` |

These can never agree. `_IDENTITY_DIGEST_FIELDS` (`_run_state.py:290-296`) folds
`inventory_digest` into `RunIdentity.digest()`, which is the verification cache's key
(`_verification_cache.py:199`) and what `assert_identity_current` (`_run_state.py:299`)
compares. So the first phase that hands a **minted** identity to either — a cache lookup, a
binding check, a proof write — gets a permanent mismatch: every cache entry cold, or
`assert_identity_current` raising on a run nothing changed.

**Why nothing has failed.** P2 consumes exactly two of the seven minted fields — `grep -n
"identity\." src/phenotypic/phenotypicCLI.py` returns `processing_generation` at `:2446, 2729,
2772` and `restart_epoch` at `:2449, 2730`, and nothing else. The other five are computed and
discarded. That is the definition the gate brief gives: *"by construction nothing downstream
will fail until something depends on the behaviour that was never written."*

**And B-1 is the test that would have caught it.** The plan's
`test_a_new_image_does_NOT_mint_a_new_generation` asserts `after.inventory_digest !=
before.inventory_digest` after adding an image to the input. Against the shipped
implementation that assertion is **false** — which is presumably why the test did not land.
A planned test that cannot pass is a finding about the code or about the plan; either way it
belongs in the register, and it is not there. Compare register entry 17, where exactly this
situation (a plan test that no correct implementation could satisfy) was diagnosed, ruled and
recorded — that is the precedent this one should have followed.

**Recommended disposition — orchestrator's call, and I would not guess it.** Three coherent
answers, in the order I would consider them:

1. **Point the minter at the same source as the reader.** `mint_run_identity` runs after
   inventory resolution (`phenotypicCLI.py:2011` precedes `:2166`), but `work_ids` is not on
   the `ExecutionConfig` — so this needs the scanned manifest threaded in, and is not a
   one-liner.
2. **Declare the field not-yet-minted.** Leave it `""` with a comment naming the phase that
   fills it, and add a keeper test that the two producers agree once both are real. Honest,
   cheap, and turns a silent disagreement into a visible gap.
3. **Keep it, and add the cross-producer agreement test now** — the same instrument
   `test_the_two_metadata_digests_agree` (`test_run_identity.py:487`) applies to
   `metadata_sha256`. It would fail today, which is the point.

What is *not* available is leaving it as it stands, because the two producers' disagreement is
invisible, is folded into an identity digest, and gets more expensive with every phase built
on top of it.

---

## Satisfied obligations — checked, and listed because a reviewer who names only the gaps has not shown they looked

| Requirement | Where | Evidence in the tree |
|---|---|---|
| D3 — content-derived generation | `design.md:250` | `_cli_identity.py:144-197`; five minting sites converted (`_cli_state_management.py:253`, `phenotypicCLI.py:2445, 2729, 2772`, `_cli_migrate.py:706`); `grep -rn uuid4 src/phenotypic/_cli src/phenotypic/phenotypicCLI.py` returns **nothing**; `test_run_identity.py:189, 218, 241, 350, 370, 382, 459` |
| D4 — one tracked counter, preserved by `clear_machine_state` | `design.md:251` | `_io_constants.py:1198-1200, 1235`; `test_run_identity.py:753` + mutation |
| D5 — `--restart` still reuses surviving stores | `design.md:252` | `test_a_restart_moves_the_generation_but_not_any_work_id` (`test_run_identity.py:553-598`) asserts both halves in one test, with a mutation that models the leak as a `work_id` field addition |
| D7 — `inventory_digest` out of the generation | `design.md:255` | `_cli_identity.py:158-163, 191-197`; migrator's pre-existing violation fixed (`_cli_migrate.py:691-710`), `test_run_identity.py:254-283` + mutation |
| §5.1 `restart_epoch` row | `design.md:312` | `_io_constants.py:686-693, 973-983`; `_cli_identity.py:292-362` |
| §5.4 — one definition, two uses | `design.md:455-460` | `per_image_config_digest is processing_configuration_digest` (`_cli_identity.py:90`), `is`-checked at `test_run_identity.py:155`, mutated to a wrapper in `p2_task1_restart_epoch.py:61-72` |
| §5.4's corrected field table | `design.md:466-472` | matches `processing_configuration_digest_from_values`' branch structure as read this review; the D-C amendment's flat claim is superseded, and register entry 11 records why |
| §5.5 — the versioned finalization object | `design.md:503-511` | `_cli_identity.py:281-288` carries exactly those four keys (shape only — value untested, **B-2**) |
| §9.1's six fall-through cases | `design.md:826-828` | all six, each with a test and a mutation: `test_verification_cache_disk.py:606, 626, 648, 667, 678, 689` end-to-end and `:253-403` at the loader |
| §9.1 best-effort / never an error | `design.md:836-838` | `persist_states` returns `False`; `test_verification_cache_disk.py:427-449, 451-487, 489-506` |
| §9.1 `clear_machine_state` deletes it | `design.md:838` | passively, via the everything-else branch; `test_clear_machine_state_deletes_the_persisted_cache` (`:781`) + mutation 20 |
| U-11's two module-docstring obligations | `design.md:143-155` | tenth-artifact-is-a-cache at `_verification_cache.py:29-36`; the weakened *"by some process"* guarantee, spelled out in three cases, at `:38-64` |
| §11.1 — `_assert_worker_generation`'s dead comparison deleted | `design.md:884-885` | `_cli_recompile_worker.py:225-256`, with a docstring saying why it must not be reinstated |
| §14 — stale-worker test | `design.md:944-945` | `test_run_identity.py:701-721`, **paired** with `:723-751` so an implementation that counts everything or nothing fails one of them; two mutations, one positive-prediction |
| §14 — every test proved able to fail | `design.md:948-949` | 33 mutations across `p2_task0_disk_verification_cache.py` and `p2_task1_restart_epoch.py`, three of them documented as over-claims that were withdrawn after running |
| §13 rule 1 — the cache can only cause re-verification | `design.md:925-926` | `test_a_forged_persisted_cache_cannot_manufacture_complete` (`test_verification_cache_disk.py:698`) |
| P1 gate finding **B-1** (open) | `spec-adherence.md:198` | closed: `_build_modern_process` (`test_schema_gate.py:240-274`) and `test_the_modern_process_shape_converts_on_the_marker_tree_and_nothing_else` (`:428-471`), which distinguishes emptying the marker directory from removing it |
| P1 gate, test-review finding 7 | `test_schema_gate.py:799-849` | the migrate-exemption test now spies `run_migrate` instead of asserting a string's absence, with the deferred can-fail proof discharged (commit `21a84177`) |
| INV-LAYER's four holes | register entry 15 | `test_run_state_layering.py:43-144` now resolves relative imports, checks imported names, and catches `import_module`/`__import__` literals |

---

## Two small items, neither worth a numbered finding

- **`phase-2-identity-schema.md:674` points at "3b"**, a task with no section in the plan. The
  work exists — `test_run_identity.py:530` heads a section *"D5 and mode parity (3b)"*, commit
  `578147f9` — but a reader following the pointer finds nothing. The register's shape #1, one
  more time, in the document that catalogues it.
- **`design.md:311` still carries the `publication_id` row** that U-4 cut, three lines below
  the `AMENDED (U-4): five tokens` note at `:305`. Pre-dates this phase; folds into the same
  §5.1 pass as **C-2**.

---

## Ordered recommendations

1. **E-1** — decide `inventory_digest`'s minting rule and make the two producers agree, or
   declare the field not-yet-minted with a keeper test. This is the only item here that gets
   more expensive with each phase.
2. **B-3** — correct `phase-2-identity-schema.md:616` and the two code comments
   (`_cli_migrate.py:711`, `_cli_state_management.py:257`), and move the `requires_conversion`
   assertion to P7 Task 5 beside the skip that already defers its sibling.
3. **C-2** — one sweep for `scientific_config_digest`: `design.md:24, 25, 310, 387`,
   `phase-2-identity-schema.md:60, 392, 404-416, 434, 554`, `phase-7-migrate-mode.md:1373`.
   The last is the P7 register and matters most.
4. **B-1, B-2** — add the two dropped mint-level tests, with B-1 written against whatever E-1
   resolves to.
5. **B-4** — add `test_a_stale_slurm_worker_cannot_publish`; the guard has no test at all.
6. **C-1** — add the on-disk-key fact (`_cli_completion.py:234`) to §5.1's `lifecycle_epoch`
   row, and note in the register that entry 19's evidence is an audit rather than a
   measurement, so the U-11 comparison is not misread.
7. **C-3, C-4** — two docstring corrections (`_schema_shape.py:74-78`,
   `_verification_cache.py:553`); register both, since both are the shapes the register says
   are the most dangerous.

## For the next reviewer

- P3 is the first phase that can consume a minted `inventory_digest` or
  `finalization_input_digest`. If E-1 is still open when P3's proofs land, check
  `_run_state.py:936` and `_verification_cache.py:199` against whatever the minter produces
  **before** believing a green suite — nothing in P2's tests covers that pairing.
- `_PRESERVED_ON_RESTART` (`_io_constants.py:1198`) is a set later phases are explicitly
  invited to grow (P7 adds `legacy-v2/`). It now carries a membership rule and a worked
  exclusion in its docstring. Check any addition against that rule and against
  `test_clear_machine_state_deletes_the_persisted_cache`, which lives in a suite an editor of
  the constant will not have open.
- P6 Task 0 converts `_cli_completion.py`'s call sites. Its `lifecycle_epoch` parameter is
  **staying** (C-1); a P6 agent working from the P2 plan's File Structure row 9 will expect it
  to have been renamed already.
