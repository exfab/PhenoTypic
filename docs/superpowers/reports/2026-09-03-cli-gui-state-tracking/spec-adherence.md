# Phase 1 — spec-adherence review

**Change:** `cli-gui-state-tracking`
**Phase:** P1 — `sdk_/_run_state.py`, the one reader
**Question asked:** not *"is the code correct?"* (that is the implementation/test
reviewer's gate) but *"is it all of what was specified?"* The failure this review
exists to catch is a phase that is green, correct, and missing a third of its scope.

---

## Revision reviewed — read this before acting on any finding

**Reviewed at `17f144ef`**, over the twelve-commit range `6902124e~1..17f144ef`
(24 files, +6,613/−55). Every `file:line` citation below resolves against that
revision.

**The tree has since moved, and some of it moves under this report.** At the time
of writing there are eight newer commits and a substantial uncommitted working
tree:

| | |
|---|---|
| Newer commits | `7d3e0fc0`, `bc4e6449`, `1e6cc200`, `c296d2eb`, `29e5ecc8`, `eb80c7b3`, `cfca15a5`, `275ceff1` |
| Uncommitted | `_verification_cache.py` (+353), `_run_state.py` (+35), `_io_constants.py` (+31), `sdk_/__init__.py` (+6), `test_verification_cache.py` (+21), and an untracked `tests/unit/sdk_/test_verification_cache_disk.py` |

Two of those change premises this review was briefed on:

- **`eb80c7b3` — U-11 ships the on-disk verification-cache tier.** My brief listed
  *"`VERIFICATION_CACHE_JSON` / `verification_cache_path()` do not exist, D-B
  deferred them"* as known-and-deliberate, and I verified their absence at
  `17f144ef`. That is no longer the design. **Anything below that reasons from
  "the cache is in-process only" is scoped to `17f144ef` and must be re-read
  against U-11.** In particular, P1 Task 3 Step 8's on-disk conditions — the six
  JSON-corruption mutation cases, the `try/except OSError` swallow, and the
  invariant weakening the plan required be stated out loud in the module docstring
  (*"an in-process entry can only have been written by a deep pass in this
  process; an on-disk entry cannot"*) — become live obligations that this review
  did not assess.
- **`cfca15a5`** touches rule 1's post-U-4 branch, which is finding C-3's subject
  matter.

I have **not** re-reviewed the tree at HEAD. A second spec-adherence pass is
warranted once the U-11 work settles.

---

## Verification method

By reading the diff and the source, never the checkboxes. All 44 checkboxes in
`phase-1-run-state-sdk.md` were still `- [ ]` at `17f144ef`, so the plan's own
ledger carries no signal in either direction. Greps were regenerated rather than
taken from the plan's file lists — this plan's recurring defect across four review
rounds was *a reader in a file nobody named*, six separate instances.

### Measured, not relayed

Four-file selection covering everything P1 added or changed:

```
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/sdk_/test_run_state.py tests/unit/sdk_/test_verification_cache.py \
  tests/unit/sdk_/test_run_state_layering.py tests/unit/cli/test_schema_gate.py \
  -p no:randomly -q

133 passed, 9 skipped in 6.25s
```

All four collect; zero errors; no planned test is a collection failure or an error
wearing the costume of a delivered function. The 9 skips are one parametrized
test, `test_every_convert_verdict_is_dischargeable_by_one_migrate`, over the nine
entries of `_EVERY_CONVERTIBLE_SHAPE`, with P7 Task 5 named in its reason string.

Full phase gate:

```
shards=48  tests=11244  failed=81  errors=0
REGRESSIONS (0)
```

All 81 failures pre-existing, in `migration`/`tune`/`smoke`. Test count up 138
from the 11,106 baseline — P1's own additions.

**Scope note.** `133` (four files) and `3051` (the commit body's
`tests/unit/sdk_` + `tests/unit/cli`) and `11244` (whole tree, 48 shards) are
three different scopes. This report cites the four-file number for P1's own
suites and the 48-shard number for regression. Neither is the other's total.

**Also note: 88 test *functions* produce 142 *cases*.** The gap is
parametrization — the verdict matrix (6), the two rule-1 comparison families
(5 + 3), the two shape registries (9 + 5), the malformed-payload set (5). A count
of functions and a count of cases will not match, and the mismatch is not
evidence of anything.

### What a green suite can and cannot settle

This is why the spec-adherence gate and the implementation/test gate are separate
agents rather than one:

> A passing suite can only move Category B, and only in one direction. It can
> reveal that a planned test does not run. It cannot retire a Category A, C or D
> finding — those are **absences and mismatches**, and a passing suite is silent
> about both. Code that was never written passes every test that was never
> written to exercise it, and a module in the wrong package passes every test
> that imports it from the right one.

The 48-shard result is therefore load-bearing for exactly one claim in this
report: that P1's "moves no consumers" contract held in practice as well as by
inspection.

---

## The phase's headline claim is true

P1 states it *"moves **no consumers**"* — `_output_consistency.py`, `RunRegistry`,
the SLURM observer and `_snapshot_status.py` keep working exactly as they do
today. Verified rather than assumed: `grep -rn "resolve_run_state|run_identity|RunState"`
across `src/` returns hits only inside `sdk_/_run_state.py`, `_state_types.py`,
`sdk_/__init__.py`, and docstrings. No production consumer calls the new reader.
The one new call site in `phenotypicCLI.py:445` is Task 3b's specified gate
wiring, and it is inert. `REGRESSIONS (0)` over 48 shards is the behavioural half
of the same claim.

This matters because it is the property that makes P1's correctness establishable
in isolation, and it is what P6 depends on.

---

## Findings

Twelve findings. Three are resolved; nine remain open. Categories are the brief's:
**A** specified-not-implemented, **B** planned-not-done, **C** implemented-but-differs,
**D** implemented-never-specified.

### Summary

| # | Cat | Finding | Status |
|---|---|---|---|
| A-1 | A | Rule 2's `for the current identity` fence has no code, no test, no owner | **Resolved** `7d3e0fc0` |
| B-1 | B | Modern `--mode process` shape: docstring row, no test | Open |
| B-2 | B | `test_a_converted_tree_is_accepted` did not land | **Resolved** `1e6cc200` (armed half); migrate half is P7's |
| B-3 | B | 44 plan checkboxes, none ticked | Open (process) |
| C-1 | C | `sdk_/_schema_shape.py` is a 421-line module no plan document names | Open — recommendation below |
| C-2 | C | `_schema_shape.py:87-94` documents the rejected design and walks P3 into the trap | **Resolved** `bc4e6449` |
| C-3 | C | `_run_state.py` is a second home for the marker/proof formats; P6 Task 7 Step 2's premise is now false | Open |
| C-4 | C | Task 6's headline test renamed and restructured, unrecorded | Open (benign) |
| C-5 | C | `clear_verification_cache` re-exported through `_run_state.__all__` | Open (benign) |
| D-1 | D | Seven constants + two helpers relocated into `sdk_/_io_constants.py`; 8 new public names | Open |
| D-2 | D | `check_mutation_coverage.py` has two capabilities with no committed consumer | Open |
| D-3 | D | Test volume far beyond plan | Open (not a defect) |

---

### Category A — specified, not implemented

**One finding. This is the category the review exists for, and it was nearly empty
— which is the right outcome for a phase this heavily reviewed.**

#### A-1 — Rule 2's identity fence has no code, no test, and no owner · RESOLVED `7d3e0fc0`

`README.md:113-114` and `OPEN-QUESTIONS.md` Q2 rule 2 both read: *"else a liveness
authority reports work in flight **for the current identity**, and that authority
is itself live → `active`"*.

The second half shipped. `_run_state.py:746-762` probes the GUI owner's pid, and
`:731-733` documents why the SLURM fence is taken at face value (DEFERRED D-1
keeps the observer's decision tree out of this change). Negative controls for both
predicates landed later in `c296d2eb`.

**The first half did not.** `_live_authority` (`_run_state.py:709`) takes only
`output_dir` and never compares a generation against `identity.scheduler_epoch` or
`identity.restart_epoch`.

**Why it is vacuous now and stops being vacuous in P2.** `identity.scheduler_epoch`
is read from `slurm_lifecycle.json` (`_run_state.py:210`) and `_live_authority`
reads `active` from that same file, so a fence would compare a value with itself.
`resolve_run_state(output_dir)` also accepts no caller-held identity to compare
against, so the qualifier is not expressible in the signature the plan itself
specifies. It becomes real the moment P2 writes `restart_epoch` and a lifecycle
record can predate the identity.

**The failure it admits** (added by the lead when actioning this): `--restart`
mints a new epoch; a worker from the previous epoch is still draining; its record
still says `active`; rule 2 fires; the run reports alive on the strength of a
worker the restart abandoned. **A stale authority outranking a valid verdict** —
the same shape as rule-2-over-3, which P1 has a test for, in the one direction P1
could not construct.

`grep restart_epoch` across `phase-2-identity-schema.md` near *fence/rule 2/liveness*
returned exactly one hit at review time — a test assertion **string**, not an
obligation. An obligation named in the README, half-built, and owned by no task
disappears by default; that is the class of defect this whole change exists to
remove.

---

### Category B — planned, not done

#### B-1 — The modern `--mode process` shape has a docstring row and no test · OPEN

P7 Task 1 Step 3c states it literally: *"Each gets a row in `requires_conversion`'s
docstring **and a test**."* The row shipped at `_schema_shape.py:325-328`. No
builder exists: `_EVERY_CONVERTIBLE_SHAPE` (`tests/unit/cli/test_schema_gate.py:269-279`)
holds nine shapes and `_EVERY_CURRENT_SHAPE` (`:335-341`) five; none is a modern
process tree.

Low severity **today** — such a tree trips signal 1 exactly as `markers-era` does,
so a test would assert nothing the suite does not already cover. The row's value is
entirely in the future: it is the shape that flips from `CONVERT` to `None` once P3
converts its records, and the flip is what needs a witness. A row with no test is
a claim about post-P3 behaviour that nothing will check when post-P3 arrives.

#### B-2 — `test_a_converted_tree_is_accepted` did not land · RESOLVED `1e6cc200`

Named in P7 Task 1 Step 1. It was not among the 9 skips either, so it had not
landed as a deferred test — it simply was not there. Nearest coverage was
`test_every_current_shape_is_not_an_unconverted_tree` over the `converted` shape
(`test_schema_gate.py:389`, builder `:303`), which is predicate-level only.

**The resolution was to split it, which is better than either deferring or
noting it.** The test has two halves:

- `_invoke_cli(mode="migrate")` genuinely needs P7 and is honestly deferred.
- `_invoke_cli(mode="full")` on an already-converted tree — *the gate is armed and
  must stay silent* — **needs no migrate at all, and nothing tested it.**
  `test_a_current_tree_is_not_refused_while_the_gate_is_unarmed` (`:725`) calls
  `_refuse_unmigrated_output` directly but with `SCHEMA_GATE_ARMED` false, so it
  proves inertness, not correct wiring.

The uncovered false-negative — an armed gate refusing a tree that needs nothing —
is INV-DISCHARGEABLE's other half at the CLI layer, and it is the half that
strands a user behind a refusal escapable only by `--overwrite`. It was buildable
in the tree as it stood, using the same `monkeypatch.setattr(_schema_shape,
"SCHEMA_GATE_ARMED", True)` that `test_the_gui_reports_rather_than_refuses`
(`:790`) already uses.

#### B-3 — The plan's checkbox ledger carries no signal · OPEN (process)

All 44 boxes in `phase-1-run-state-sdk.md` still `- [ ]`, none ticked, at a point
where the phase is complete and gated. Not evidence of anything undone — I verified
delivery by diff — but it means the ledger cannot serve as the audit trail it was
written to be, and the next phase's reviewer will have to redo the same diff walk.

**Category B is otherwise clean.** Every other test the plan names by name landed,
including all three Task-5-inherited cache tests (`test_verification_cache.py:500,
537, 561`), the amendment's four required restructurings
(`test_a_live_worker_over_an_unfinished_run_reads_active`,
`test_clause_one_is_load_bearing`, the two-family comparison proof at
`test_run_state.py:760` and `:808`, and the dead-owner positive control at `:710`),
and `_run_process_mode` implemented exactly as the amendment directed —
`build_complete_run(tmp_path, process_only_layer=...)` at
`tests/_output_layout.py:374-378`. `tests/_output_layout.py` is purely additive
(385 insertions, 0 deletions), so no existing fixture consumer was disturbed. The
`canonical_digest` hoist is complete: no third copy, no surviving CLI copy, and
the one-shot keeper test correctly deleted in its own commit as the plan directed.

---

### Category C — implemented, but differs

#### C-1 — The detection predicate lives in a module no plan document names · OPEN

Task 3b and P7 Task 1 both specify *"Create: `src/phenotypic/_cli/_cli_schema_gate.py`"*.
What shipped is a 421-line **`src/phenotypic/sdk_/_schema_shape.py`** holding
`requires_conversion`, `describe_required_conversion`, `describe_conversion_advisory`,
`ConversionVerdict`, `STATE_SCHEMA_VERSION` and `SCHEMA_GATE_ARMED`, with
`_cli_schema_gate.py` reduced to 88 lines of re-export plus the one `click`-raising
function. `grep -rn "_schema_shape" docs/superpowers/` returned **nothing**.

**The move is correct.** INV-LAYER forced it once `resolve_run_state` needed the
same detection for §4.3's advisory and could not import `phenotypic._cli`; a second
copy would let the GUI's advisory and the CLI's refusal disagree about whether a
tree needs migrating, which is CAN-4's shape exactly. It is argued at
`_schema_shape.py:1-31`. The split is also mechanically clean: `_cli_schema_gate`
imported nothing from `phenotypic._cli`, so INV-LAYER's line was already drawn and
only the `click` consumer belonged on the writer side.

**The defect is the unrecorded consequence.** `phase-3-per-image-record.md` cited
`_cli_schema_gate.py:216-224` for signal 3 — in an amendment written during this
same execution — while signal 3 is `_schema_shape.py:246-254`. That citation has
since been fixed. Import lines elsewhere (P2, P7) resolve through the re-export and
were never broken.

**Recommendation — record it in P1's file table, not P2/P3.** The two serve
different readers and only one is still unserved:

- P2/P3 are where someone *touches* the flag, and those references are now correct.
  A reader arriving there has a working pointer.
- **P1's table is the provenance record** — the answer to *"what does this phase's
  boundary consist of, and why?"* That is what a later auditor reconstructs the
  phase from, and it is precisely what my C-1 finding says is incomplete. A module
  recorded only at its point of use reads as incidental; recorded in the phase that
  created it, the INV-LAYER reasoning travels with it.
- There is a third place worth a one-line pointer, and it is the one with teeth:
  **P6 Task 7 Step 2**, where the module boundary becomes load-bearing again — see
  C-3.

#### C-2 — The arming note documented the rejected design · RESOLVED `bc4e6449`

`_schema_shape.py:87-94` read: *"`_cli_schema_gate` re-exports this as its own
module-level binding and `refuse_unconverted_schema` reads *that*, so a test arming
the refusal patches `_cli_schema_gate`."* False three ways:

- `_cli_schema_gate.py:81` reads `_schema_shape.SCHEMA_GATE_ARMED` **through the
  module** and binds no copy.
- `_cli_schema_gate.py:64-71` says so explicitly, and warns that a re-export
  *"would still read correctly while being **inert under monkeypatch**"*.
- `test_the_arming_flag_has_one_source` (`test_schema_gate.py:834-885`) asserts
  `not hasattr(_cli_schema_gate, "SCHEMA_GATE_ARMED")` — structurally, over the
  AST, catching `Assign`, `AnnAssign` **and** `ImportFrom`, because a re-export is
  how the copy comes back.

Compounding it, `phase-3-per-image-record.md`'s arming instruction named
`_cli_schema_gate.SCHEMA_GATE_ARMED`, a symbol that does not exist and is asserted
absent. **A P3 implementer following either text writes the flag into
`_cli_schema_gate.py`, fails `test_the_arming_flag_has_one_source`, and — if they
delete that test to get past it — ships an arming flag that changes nothing.**

This was the highest-value finding in the review, and the reason is worth keeping:
the 133 green cases were entirely consistent with the trap being fully armed and
untriggered. No test result could have surfaced it. Only reading the prose against
the code could.

#### C-3 — `_run_state.py` is a second home for the marker and proof formats, and P6 Task 7 Step 2's premise is now false · OPEN

Planned at *"~340 lines. The four public readers."* Shipped at **1,198**. The
excess re-implements `_cli_completion`'s validators:

| In `_run_state.py` | Re-derives |
|---|---|
| `_marker_rejection` (`:479`), `_fenced_artifact_path` (`:420`) | `valid_image_success` |
| `_valid_aggregate_proof` (`:784`) | `valid_aggregate_snapshot` |
| `_valid_run_proof` (`:770`) | `valid_run_completion` |

**Structurally forced**, and the implementation knows it: INV-LAYER blocks the
import, Task 1's "Consumes" block lists none of them, and the duplication is
disclosed at `_run_state.py:21-31` with two real mitigations — constants shared
through `sdk_/_io_constants` and pinned by
`test_the_marker_schema_constants_have_exactly_one_home` (`test_run_state.py:1085`),
and `test_the_sdk_reader_agrees_with_the_cli_validator` (`:1060`) comparing both
implementations image-by-image over a real tree and four tamperings.

But it is a second home for a format, in a change whose §0 records **six** defects
of exactly that shape across three review rounds, and the plan's file structure did
not anticipate it.

**The forward consequence, and the actionable half of this finding:**
`phase-6-consumer-migration.md` Task 7 Step 2 plans to **move**
`valid_image_success`, `valid_aggregate_snapshot` and `valid_run_completion` into
`sdk_/_run_state.py`. **They cannot be moved into a module that already contains
independent implementations of all three.** That step is now a reconciliation of
two implementations, not a move, and no document says so.

This is the same failure mode as A-1 in a different costume: a planned step whose
premise quietly stopped being true, with nothing that fails to announce it. P6's
own Step 1 test greps for the three predicate names and asserts they appear
nowhere — which a *merge* satisfies and a *move* does not, so the test will pass
either way and the ambiguity survives.

#### C-4 — Task 6's headline test was renamed and restructured · OPEN (benign)

`test_shallow_after_deep_does_not_re_hash_artifacts` shipped as
`test_shallow_reuse_is_independent_of_the_image_count` (`test_run_state.py:1184`),
replacing the plan's fixed `assert calls["n"] <= 8` with a two-tree asymptotic
comparison (`stems=("a","b")` vs six stems: a warm shallow pass must cost the same
on both, a deep pass must cost more on the larger).

**Strictly better** — a fixed bound cannot separate "constant" from "small" and
would need re-tuning every time the run-level proof gains an artifact. Recorded
only because it is an unrecorded rename of a plan-named test, which is exactly the
shape Category B is meant to catch when the substitution *isn't* an improvement. A
reviewer grepping the plan's test names against the suite gets a miss here and has
to reconstruct why.

#### C-5 — `clear_verification_cache` re-exported through `_run_state.__all__` · OPEN (benign)

At `_run_state.py:100`, where the plan's Interfaces block places it on
`_verification_cache` and only `sdk_/__init__.py` was to export it. Justified at
`:88-93` on the same grounds as the four dataclasses — spec §5.2 declares the
public surface as `_run_state`, and the module split below it is a cycle-breaking
mechanism rather than an interface change.

Worth knowing that it survives `test_run_state_exports_no_writer` because "clear"
is not among the six forbidden prefixes (`publish`, `write`, `mint`, `append`,
`save`, `delete`). That test is the **sole** enforcer of the "readers only" half of
INV-LAYER, it already cannot catch a writer named `record_stage` (CAN-31), and P6
Task 7 Step 3 deletes it. After P6, nothing structural keeps writers out of this
module.

---

### Category D — implemented, never specified

#### D-1 — Constant relocation across three CLI modules · OPEN

Task 1 Step 3 authorizes exactly two additions to `_io_constants.py`:
`DIR_IMAGE_RECORDS` and `image_record_path()`. Also shipped:

- `SUCCESS_MARKER_VERSION`, `ARTIFACT_KIND_FILE`, `ARTIFACT_KIND_STORE`,
  `AGGREGATE_PROOF_VERSION`, `RUN_PROOF_VERSION` — moved out of
  `_cli_completion.py`, which now imports and re-exports them so every existing
  importer is unchanged.
- `SLURM_LIFECYCLE_JSON` and `slurm_lifecycle_path()` — moved out of
  `_cli_slurm_lifecycle.py`, whose `lifecycle_state_path` now delegates (`:82`).
- Eight new names in `sdk_/__init__.__all__`.

Same INV-LAYER justification as C-1 and C-3, behaviour-preserving, and pinned
structurally by `test_the_marker_schema_constants_have_exactly_one_home`, which
asserts `_cli_completion` *assigns* none of the five rather than merely that the
values are equal — the right guard, since equality passes on the day of a
"tidying" re-declaration and drifts afterwards.

Reported as real public-surface growth on `phenotypic.sdk_` that no task
authorized, not as a mistake. The reason it belongs in this category rather than
being waved through: a version number that gates a **completion verdict** is the
one duplication class that can silently manufacture a false `complete`, so where
it lives is a design decision, and design decisions in this change are supposed to
be written down before they are made.

#### D-2 — `check_mutation_coverage.py` has two capabilities with no committed consumer · OPEN

`CONTROLS` (`:36`, `:87-93`) and `TARGETS` (`:108-127`), added by `b567e9c9` and
`a0c3002c`, exist for the `test_run_state.py` harness that was not committed. The
only committed harness, `p1_task3_verification_cache.py`, declares neither —
singular `TARGET` at `:58`, no controls.

I am not reopening the harness deferral, which is on the known-and-deliberate list
with its reason recorded. The finding is narrower and is the deferral's reverse
side: the committed gate script now **advertises more coverage than it can
exercise**. Against the harness README's own rule — *"A harness in a scratchpad
leaves its target unwatched. Commit the harness."* — the gate currently validates
12 mutations over one 20-case suite while the phase's 58-case suite is unwatched by
it. A reader running `check_mutation_coverage.py`, seeing `COVERAGE_OK=True`, and
concluding P1's mutation coverage is complete would be wrong, and the script's own
README is the only thing that says so.

#### D-3 — Test volume far beyond plan · OPEN (not a defect)

88 test functions / 142 cases against roughly 30 named in the plan; 12 mutations
against Task 3's specified 5, with the extra seven justified in `5286f039`'s body
(*"Five would have left eight of the tests unproved, and an unproved test is
decoration"*).

Recorded so the ratio is visible, because it is the shape of this phase:
**over-delivered on tests, under-delivered on keeping the plan's forward references
true.** Every open finding above is an instance of that second half — A-1, B-1,
C-1, C-3 and D-2 are all cases where the code moved and a document did not.

---

## Empty categories

None. All four categories produced findings. Category A produced only one, and it
was structurally unimplementable rather than forgotten, which for a phase with four
adversarial review rounds behind it is the expected and correct result.

---

## Ordered recommendations

Three are done. The remainder, in the order I would take them:

1. **C-3** — amend `phase-6-consumer-migration.md` Task 7 Step 2 from *"move three
   validators"* to *"reconcile three validators against the P1 implementations"*,
   and say which implementation wins. Highest remaining risk: a P6 agent reading it
   literally will either duplicate a third time or delete the wrong copy, and P6's
   own grep test passes either way.
2. **C-1** — add `sdk_/_schema_shape.py` to P1's file table with the INV-LAYER
   reasoning, plus a one-line pointer from P6 Task 7 Step 2.
3. **B-1** — add a modern-process-tree builder to `_EVERY_CONVERTIBLE_SHAPE`. Its
   value is the post-P3 flip to `None`, so it wants to exist before P3, not after.
4. **D-2** — either commit the `test_run_state.py` harness or record in the harness
   README that P1's largest suite is deliberately unwatched, so `COVERAGE_OK=True`
   cannot be misread.
5. **C-4, C-5, D-1, B-3** — documentation and ledger hygiene; no behavioural risk.

## For the next reviewer

Re-run this gate against HEAD once U-11's on-disk tier settles. The specific
obligations to check, none of which existed at `17f144ef`:

- P1 Task 3 Step 8's six on-disk corruption cases — `truncated`, `null`,
  `wrong-type`, `binary-garbage`, `deleted`, and the unwritable-cache-directory
  case that must return `depth="deep"` rather than raising — each proved able to
  fail.
- `store_verification_cache` wrapping `atomic_write_json` in `try/except OSError`
  and swallowing, per spec §9.1's *"best-effort … must never turn an unwritable
  output into an error"*.
- The invariant weakening stated **in the module docstring**, not inherited
  silently from wording written for the in-process case: *an in-process entry can
  only have been written by a deep pass in this process; an on-disk entry cannot.*
  The plan is explicit that this must be said out loud.
- Whether `clear_machine_state` deletes the new file (spec §9.1), and whether that
  landed in P1 or was deferred to P2 alongside `clear_verification_cache`.
