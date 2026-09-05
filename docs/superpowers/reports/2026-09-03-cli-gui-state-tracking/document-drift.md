# Document drift found during execution

**Scope:** every case, found while executing this change, where a document or a test asserted
something the tree did not support. Written during execution rather than after, because the
list is only cheap to keep while the evidence is in hand.

**Why this file exists.** The change itself is a claim about *state that is tracked* versus
*state that is checked*. This is the same defect one level up: prose that is tracked by
nobody and checked by nothing. Every entry below was found by **reading**, never by a gate —
that is the property they share, and it is what makes the class expensive.

Kept because P7 Task 6's register is the deliverable these all point at. A register written
without knowing how its predecessors failed will fail the same way.

---

## The three kinds, which are not equally expensive

The distinction is due to the P2 cluster agent, and it matters when deciding what to do
about each.

| Kind | What it means | Cost |
|---|---|---|
| **Stale** | True when written; the code moved underneath it | Low. Expected, and the fix is mechanical. |
| **Never true** | Not true at the moment of writing | Medium. Someone asserted rather than checked. |
| **Wrong while correcting** | An amendment, whose job is to fix a claim, is itself wrong about it | **Highest.** It carries the authority of a correction, so a reader who checks *is* likely to stop there. |

---

## The register

| # | Document | Claim | Kind | Resolved |
|---|---|---|---|---|
| 1 | P1 T3b, P7 T1 | *"Create `_cli/_cli_schema_gate.py`"* — no document named `sdk_/_schema_shape.py`, where the detection actually lives | stale | `c29167bb` |
| 2 | P6 T0 Step 2 | *"move six readers into `sdk_/_run_state.py`"* — P1 already re-derives all six there, and `valid_run_completion` **cannot** be moved: it imports `_cli_state_management`, which INV-LAYER's AST test fails on | stale | `c29167bb` |
| 3 | P6 T0 Step 3 | `sdk_/_hdf_to_zarr.py` listed at **1** invocation; it has **four** `_cli` import statements — a deleted function, a privatised one, a renamed constant, and the progress read | never true | `c29167bb` |
| 4 | design.md §0 (U-11), P2 T0 | *"P7 Task 6's register lists it under cache"* — no such heading existed | never true | `c29167bb` |
| 5 | P2 T0 | *"add to the same `_PRESERVED_ON_RESTART` test Task 1 already touches"* — neither the constant nor the test existed; Task 1 creates them, and Task 0 runs first | never true | `fb745abf` |
| 6 | `_io_constants.py:683-685` | *"unlike `restart_epoch.json`, **which that function preserves**"* — `clear_machine_state` preserved one name; the file did not exist | never true | `d2f5f3ab` |
| 7 | `_verification_cache.py:124-130` | *"deletes every child **bar the terminal-failure journal**"* (true) then *"the preserve set that `restart_epoch.json` is in"* (a set with a name that did not exist) — self-contradictory two sentences apart | never true | `d2f5f3ab` |
| 8 | P2 T1 | *"P6 Task 0's call-site conversion is the only external caller"* of `_live_authority` — there is **no** external caller; one module-private site, and `grep 'live_authority'` over phase-6 returns nothing | never true | `d2f5f3ab` |
| 9 | `EXECUTION.md` | *"Derived by `scratchpad/dag.py` — regenerate rather than trust it"* — the generator lived in a session scratchpad and went with it, so the veto table could not be regenerated and the instruction to distrust it could not be followed | never true | `2bfa1006` |
| 10 | `_schema_shape.py` docstring | table headed *"each with a test"*; the modern `--mode process` row had none | never true | `5e03635b` |
| 11 | **design.md §0, D-C** | The amendment correcting §5.4's field list states flatly that three fields are in `work_id`. The code branches: present for full/measure, **absent for process**. Right for two modes of three, with a drifted citation | **wrong while correcting** | `b2e7a4b9` |
| 12 | harness `README.md` | *"the coverage checker would have caught the missing `pytest.param`"* — it strips parametrized names to their stem (`:185`, `:214`) and structurally cannot | never true | `29965f56` |
| 13 | harness `README.md` | *"the run structurally could not have found"* the upward-degrade hole — it would have, one freeze cycle later; the structural claim belongs to `COVERAGE_OK`, not the run | never true | `d2f5f3ab` |
| 14 | design.md §5.3 vs §5.4 | **`scientific_config_digest` names two different values.** §5.3's table asks *"did the **pipeline** change?"* — the pipeline file's bytes, which is what the proofs write (`_cli_completion.py:914,1020,1087`). §5.4 calls it *"the per-image digest already folded into `work_id`"* — a payload containing no pipeline bytes. Adjacent sections, one name, two values | never true | user-ruled; renamed |
| 15 | `test_run_state_layering.py` | INV-LAYER's walk checked `module.startswith(("phenotypic._cli", "._cli"))`. `ast` strips a relative import's dots into `level`, so **`"._cli"` could never match anything** — and `from .._cli import x`, the natural relative violation from `sdk_`, passed. Four holes, not the two the review named | never true | `45af0a81` |
| 16 | `test_run_state.py` | `test_image_state_stages_carry_no_backfilled_key` asserted a property of a literal the test wrote three lines earlier; no source change could redden it. And forbidding a key in a deliberately **open** map argued against the design it claimed to protect | never true | `74d75f3c` |

Two of the sixteen (12, 13) are the orchestrator's own, and both are claims about **what a
check verifies** — written into the file whose stated job is being trustworthy about exactly
that. Both were caught by the agent whose work they described.

### Entry 14 is the most expensive of the set, and it nearly shipped

One name, two values, in **adjacent sections of the spec** — and the code has always matched
§5.3. Task 2 then shipped an alias binding that name to §5.4's value, so the collision
reached the tree.

**Why it survived four review rounds:** both readings are individually correct. §5.3
describes what the proofs write; §5.4 describes what the generation folds in. D-C ruled on
§5.4 and was right *about §5.4*. Nothing was false in isolation — the defect only exists
across the two sections, which is exactly the shape a section-scoped reviewer cannot see.

**The trap it set.** A later reader meeting one name for two values assumes a bug, and the
obvious repair is to make the proofs use the other value. That **rewrites the digest in every
aggregate and run proof on disk**, so every existing complete run reads `incomplete` until
re-finalized — a silent migration wearing the costume of a rename, apparently endorsed by an
approved amendment.

**Resolved by renaming the value that has no on-disk representation**, on the reasoning that
only one of the two *can* be renamed safely. Taken now rather than deferred because the new
name was one commit old with one call site; P3–P7 build on it, after which the cost of the
rename rises steeply and every intervening reader has to be told about the collision.

**The rule this yields:** when a name is introduced that already means something elsewhere in
the same system, the cheap moment to fix it is the commit that introduced it. Deferring
converts a naming problem into a migration problem.

---

## A fourth kind, kept OUT of the register above — and why that matters

**D7 is not in the table, and declining to put it there is the finding.**

D7 says: *"`inventory_digest` stays out of the generation digest."* The migrator has
folded the inventory into `processing_generation` since **`dd18d9c7` (2026-08-26)** —
`git merge-base --is-ancestor` confirms it is a direct ancestor of `c9d1fbfc`
(2026-09-03), the commit that created `design.md` and D7 with it. So the code predates
the rule by eight days, and has never satisfied it.

That looks like a register entry and is not one:

- **Not stale.** Stale means *true when written, then the code moved*. D7 was never
  satisfied — the violation predates the rule.
- **Not never-true.** That means *the author asserted something untrue about the tree*.
  **D7 asserts nothing about the tree.** Its text is prescriptive — a rule in a
  decisions table, with a rationale — so there is nothing in it that is true or false
  about `_cli_migrate.py`.

Every entry in the register above is **a document wrong about the code**. This is
**code wrong about a document**, and the document is correct. Folding it in to make the
tally larger would put the first non-falsehood in a table of falsehoods, and the kinds
are only worth having if they are trustworthy — which is the same standard this whole
change applies to its verdicts.

| Kind | What it means | Cost |
|---|---|---|
| **Rule written without checking compliance** | A document states a requirement; nobody verified the shipped tree already met it. The document is not wrong — the code is | Medium, and **misattributed**. Invisible until someone implements the rule, and the failure then surfaces in whatever phase touches it, blamed on that phase. |

**The misattribution is the whole cost, and here it is concrete.** The plan warns that
leaving this unfixed makes P5's rolling-input matrix fail on any migrated tree, where
*"the failure looks like a bug in P5 rather than an unrevised writer"*. Every new image
under a rolling input would change the generation, reset live progress, and fence
in-flight workers — exactly what D7 exists to prevent, on every migrated tree today.

**The procedural lesson:** a new rule in a decisions table needs a compliance check
against the existing tree at the moment it is written, not at the moment someone
implements it. Nothing in the four review rounds asked *"does the shipped code already
satisfy this?"* — the rounds reviewed the plan against the spec, and both were new.

### The same kind, one level up: a rule APPLIED without checking its precondition

**Second instance, 2026-09-05, and it is the orchestrator's.** The user gave a standing
rule — *spec drift is acceptable only where the alternative was experimentally validated
and the decision recorded.* Within minutes the orchestrator told the P2 agent that three
of its implementation decisions fell in that rule's most severe row and owed experiments.

**The rule's precondition was never checked.** It governs *"the spec said X and the code
does Y"*. `mint_run_identity` appears in the spec **once**, as a signature and a layer
constraint, and the three decisions sit outside anything it says. They were latitude, not
drift.

The agent **accepted the instruction on the orchestrator's authority** and had planned to
write all three into its commits as documented-but-unmeasured deviations — three
non-findings entering the gate's most severe category, from a rule that had been correct
when stated and wrong when applied.

| | Stated vs applied |
|---|---|
| **D7** (above) | a rule **stated** without checking the tree already complied |
| **This** | a rule **applied** without checking its precondition held |

Both are a requirement meeting a reality nobody looked at. The second is faster to make
and faster to spread, because an instruction carries authority the moment it is sent and
the recipient has no reason to re-derive it.

**What caught it:** writing the rule down. Recording the scope boundary in `EXECUTION.md`
forced the question *"what does the spec actually say here?"*, which the instruction had
skipped. **The artifact caught the author** — which is the argument for writing rules
into files rather than into messages, and it is the same argument this register makes
about prose that nothing checks.

**And the correction was itself imprecise.** *"The spec says nothing about how
`metadata_sha256` reaches the identity"* is true but invites a false check —
`metadata_sha256` appears five times. The agent caught that too. The final form is the
test now in the gate: **what does the spec constrain, and is that the thing being
chosen?** — with the instruction to cite the sections checked *including the satisfied
ones*, because a claim that something is unconstrained needs evidence exactly as much as
a claim that it is.

---

## What the pattern says

**Nothing failed, and nothing could have.** Not one entry would have been caught by a test,
a linter, a type check, or the coverage gate. That is not incidental — it is the definition
of the class. Prose is not executed, so a false sentence has no failure mode until a human
acts on it.

**The two shapes that recur:**

1. **A pointer to a name that does not exist yet** (5, 6, 7, 9). Almost always written while
   holding a future state in mind — the author is describing the system they are reasoning
   about rather than the one on disk. It comes true one task later, which is the most
   forgiving version and still the same error.
2. **A claim about what a check does** (10, 12, 13). The most dangerous, because it converts
   a green gate into false assurance. Ask of every one: *what would this have looked like if
   it had failed?*

**The mitigation that actually worked**, and it is not "be careful": **prefer a pointer to a
real symbol over a restatement.** A pointer to a symbol that exists is the one form of these
claims that cannot be false — an import error or a failed grep catches it. That is why the
`_PRESERVED_ON_RESTART` fix is one home and three pointers rather than four statements, and
why the consolidation was folded into the task that *creates* the symbol rather than left as
a follow-up.

**And one thing that found defects nothing else did:** writing down what a mutation run
*should* do, before running it. It caught two mis-claimed mutations and a real coverage hole
against a suite where all 323 tests passed. See the harness
[README](../../plans/2026-09-03-cli-gui-state-tracking/mutation_harnesses/README.md).
