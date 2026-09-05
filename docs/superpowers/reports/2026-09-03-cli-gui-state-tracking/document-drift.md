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
