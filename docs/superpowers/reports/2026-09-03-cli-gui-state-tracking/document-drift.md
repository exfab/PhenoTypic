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
| **True but incomplete** | Every statement is correct; a **consequence** is omitted. Not a falsehood, and the only kind here that is not | Medium, and **invisible to every check in this change** — nothing can fail for a sentence that was not written. Found only by tracing a mechanism to its consumer. |

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
| 17 | P2 T3 Step 5 | parametrizes `["full","measure","process"]` and asserts each mints the identity `full` does. Unreachable: the digest branches mutually exclusively on `process_only_layer`, so process and full digest different payloads. No DF-16-satisfying implementation could make them equal without deleting a field | never true | `578147f9` |
| 18 | P2 T3 Step 7 | *"38 tests"* for `-k 'restart or resume'`. `-k` matches the whole **node id**, so it selects every test in a class or module whose name contains either word — **451**, not 38 | never true | `578147f9` |
| 19 | **design.md §5.1** | *"`scheduler_epoch` absorbs `slurm_generation`, staged `epoch`, `lifecycle_epoch`, `execution_epoch`, and recompile's `attempt_id`."* **Zero of the five can be renamed** — see below | never true | user-ruled |
| 20 | **commit `3220a740`** | Says five minting sites became content-derived and `uuid4` is gone — all true. Omits that a generation stable across resumes stops `aggregate_state_from_events` excluding prior events, so **a resumed run now counts history it previously discarded** | **true but incomplete** | annotated, user-ruled |

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

### Entry 19, and the distinction that produced it: OWNERSHIP is not PERSISTENCE

§5.1's line is the largest single reduction the spec claims — five identity tokens into
one. Checked writer by writer against the shipped code, **none of the five can be
renamed**, for four different reasons:

| Token | Why not |
|---|---|
| `slurm_generation` | an **on-disk key** in `job_metadata.json` (`_cli_execution_strategies.py:1059`) and the recompile manifest, read by `_cli_checkpoint_handler.py:169,208`, `_cli_recompile_slurm_scripts.py:251`, and **three GUI sites** (`gui/run_console/_slurm.py:244,290`, `_slurm_observer.py:436`) |
| recompile `attempt_id` | *is* `slurm_generation` by value — one variable passed into both parameters, then asserted equal — and is pinned by that token's persistence |
| `lifecycle_epoch` | **mode-dependent at runtime**: `_authoritative_lifecycle_epoch()` returns the scheduler generation under SLURM and the *processing generation* locally. `scheduler_epoch` is **narrower than the value**, which is a worse defect than the vagueness it would fix |
| `execution_epoch` | a **proof field** — renaming rewrites keys in every aggregate and run proof on disk |
| staged `epoch` | its own writer, its own lifetime |

**The general form, and it is the reason this took two passes to get right:**

> **Ownership says who may change the value. Persistence says who else can still read the
> name.** A token can have exactly one writer and still be a public format.

Those two properties read as one, which is why *"one writer, scheduler-owned"* felt like
it meant *"safe to rename"*. The cluster agent applied the on-disk test to
`execution_epoch`, did **not** apply it to `slurm_generation`, and marked row 1
collapsible. **The orchestrator then carried that row into a table put in front of the
user for a ruling, without applying to row 1 the test it had just read in row 4.** The
agent caught its own error while the ruling was in flight.

**Both halves belong in the register.** The first is a missed check; the second is an
unverified claim propagating through an intermediary who had the disproof in the same
table. It is the second time this phase an unchecked assertion of the orchestrator's
reached the user's decision, and the second time the cluster agent caught it.

**Disposition, user-ruled:** §5.1 is amended to record the collapse as unachievable, with
these four reasons cited. The alternative — renaming with read-both-keys shims in every
reader — was rejected because **dual-key support is more state to keep in sync, not
less**, and a change whose stated purpose is reducing tracked state would have ended by
adding some.

What survives is smaller and real: `_assert_worker_generation`'s
`slurm_generation != attempt_id` compares one value with itself, so it is a **dead
comparison to delete** rather than a token to collapse — no name, no key, no behaviour
changes, and it removes the thing that made the pair look like two values.

---

### Entry 20 — a behaviour change that shipped inside a true commit message

**The only entry here that is not a falsehood, which is why it needed a new kind.**

`3220a740` converted the resume path's `uuid4()` to the content-derived generation. Its
message is accurate throughout: five sites converted, the import reduced to `UUID`, and
the `:2422` comment quoted and refuted as a *justification*. Nothing in it is wrong.

What it does not say is the consequence, two files away:

> `aggregate_state_from_events` (`_cli_update_state.py:337-347`) ignores events tagged with
> a **different** generation, and `load_processing_state` has always passed the current one.
> A fresh `uuid4()` per invocation therefore excluded **every prior event on every resume**.
> Making the generation stable means a resume now **counts** that history.

| | before `3220a740` | after |
|---|---|---|
| restart | prior events excluded | excluded — §14, unchanged |
| **resume** | prior events **excluded** | prior events **counted** |

**Why it was missed, and the lesson.** The deleted comment said a fresh epoch *"fences
workers left by a killed local attempt"*. That was checked and found false **as a
justification** for minting a uuid — which it was. It was simultaneously **accurate as a
description of what the uuid did**. Only the first was verified.

> **Refuting a justification does not refute the description it rests on.** When deleting
> a comment that explains *why* something is done, trace what it says the thing *does* to
> the consumer that observes it.

**Assessed after the fact, and the risk direction is safe.** The work list is
`processed = completed | failed` (`_cli_update_state.py:496-497`) — **`started` is not in
it**. So a stale `started` from a killed worker drops its image out of `completed` and the
image is **reprocessed**; it can never be wrongly skipped. Failure direction is extra work,
never lost work, which is INV-VERDICT's direction. The merge point's own comment says
*"prefer event log as source of truth"*, so the pre-`3220a740` behaviour was silently
defeating the stated design on every resume — this is a latent bug fixed, not one
introduced. §4.2 deletes these derived sets entirely by P6, bounding the window.

**User ruling:** the change is **accepted as shipped**, and the commit is **annotated
rather than amended**. Amending would rewrite six subsequent SHAs and break committed
citations — manufacturing, inside the artifacts built to catch dangling references,
exactly the defect this register tracks. Nine SHA citations exist in `docs/superpowers/`
today.

**Still owed:** the test must pin the **pair** — a restart excludes prior events (§14,
unchanged, unpinned today) and a resume **includes** them (new since `3220a740`, unpinned,
and the half a regression would silently reverse).

---

## The two P2-gate rulings, and one of them was not a defect

### F2 — `inventory_digest` is reader-owned. Option (a). User-ruled 2026-09-05.

`mint_run_identity` returns `inventory_digest=""`, documented as *populated by the reader,
empty at mint*; `assert_identity_current` skips empty tokens; **`_inventory_digest_for` is
deleted**, not repaired.

**What made this obvious rather than a compromise** was counting the uses. The field is
`canonical_digest(work_ids)` — a pure function of data already in `processing_state.json` —
and it is computed that way in **four** places that all agree: the reader
(`_run_state.py:276`) and the three proof writers (`_cli_completion.py:904,1012,1086`).
Only the minter disagreed, and it disagreed by digesting a 64-char hex string that is
`None` by default, so the field meant to answer *"did the accepted scope change?"* answered
**"no"** unconditionally.

**So the minter never needed to carry it.** It is derived from state, the minter runs before
state exists, and anyone needing it computes it live from disk. `_run_state.py:384` already
sets `inventory_digest=""` for the unidentified case, so the empty form has precedent.

### F10 — NOT A DEFECT. Withdrawn 2026-09-05.

Reported as: the on-disk cache made a `(size, mtime_ns)` staleness window *persistent*, so a
file rewritten within one filesystem tick at the same size keeps a stale "verified" verdict
forever rather than until the process exits.

**The mechanism is real; the precondition does not occur, and nobody checked it before
building a decision on it.** Two independent reasons:

| Requirement | Reality |
|---|---|
| the same artifact written **twice** | no code path does this within a pass — each worker owns one image and writes its artifacts once; the only in-place rewrite of a tracked artifact is `replace_embedded_measurement_table` (`_cli_migrate_image.py:281`), which runs under `--mode migrate`, a separate invocation |
| **within one mtime tick** | measured on GPFS `/bigdata`, where runs live: **0 of 200** back-to-back same-size writes shared an `mtime_ns` (delta ~81 µs). Node-local scratch is 181/200 at ~1 ms — but run output may never live there |

**The correction is the orchestrator's**, and it is the session's recurring error in a new
costume: *a claim about what a check cannot catch, carried forward without establishing that
the triggering condition exists.* A decision menu was built for it, and the measurement
taken was of granularity — the half that was not load-bearing. The user asked the question
that dissolved it: *"each job owns one image, so what would the second write be?"*

**What survives** is one line in `_verification_cache.py` recording *why* `(size,
mtime_ns)` is sufficient — no path rewrites a tracked artifact within a pass, GPFS resolves
to ~81 µs measured — so the next reader gets the reasoning rather than re-deriving it. That
is documentation of a sound decision, not of a hole. It also states the condition under
which it would stop being true: output on a filesystem with coarser granularity.

---

### Entry 21 — A TOOL'S OUTPUT IS A SAMPLE TOO. 2026-09-05.

**The sixth instance of the session's recurring error, and the first where the sampler was a
tool rather than a person** — which is why it is worth its own entry rather than a line in
entry 20's tally.

A mutation harness ABORTed with a drifted-anchor error naming three anchors in
`_verification_cache.py`. The orchestrator concluded the anchors had genuinely moved, and
said so in writing: *"real drift — confirmed by the harness's own ABORT, not by my
sampling."* The P2 agent accepted it and queued a re-anchor as the one edit owed before
anything else.

**The ABORT was a sample.** A harness reads the tree when it starts, and it inherits exactly
the precondition its operator has: it is meaningful only if nothing else holds the tree. That
run started while a second harness — chained into the same shell — was mid-mutation on
`_verification_cache.py`, the only target the two shared for those anchors. It read a
mutated file and correctly refused to proceed.

**The refusal was right; the conclusion drawn from it was not.** Checked afterwards on a
still tree, with the harness's own ownership rule (an anchor must match *exactly once in
exactly one target* — stricter than summing `count(old)` across targets):

| | |
|---|---|
| `p2_task0_disk_verification_cache.py` | 3 targets, 21 mutations — precondition **passes** |
| `p2_task1_restart_epoch.py` | 9 targets, 35 mutations — precondition **passes** |
| the three named anchors | present, **exactly once each**, all in `_verification_cache.py` |

Mechanically it could not have been the suspected cause either: the F10 paragraph went into
the **module docstring**; the three anchors are inside `persist_states`, several hundred
lines below.

**Why this instance is the instructive one.** *"The tool said so, not me"* feels like
independent evidence and is not. A log written during a collision describes the collision.
The rule the session had already adopted — check the precondition before hashing or running
`git status` — was **too narrow**: it must also run before *reading a tool's output*.

**What it would have cost.** A wrong re-anchor is invisible. The harness would still pass,
against anchors that no longer name the code they were written for — a green gate pinned to
the wrong lines, which is entry 10's shape reached by a different road. It was caught only
because the agent verified the precondition instead of trusting a confirmed-looking abort,
after being told by the orchestrator that the drift was real.

### The same defect in the tooling built to detect it

Found in the same hour, while running the precondition check the entry above prescribes.

1. **`pgrep -af "a\|b"`** — `\|` is BRE alternation, but `pgrep` uses ERE, where it is a
   **literal pipe**. The pattern matched nothing, ever, so the check returned *"no harness
   running"* unconditionally. Four conclusions were drawn from it, including suspicion of a
   subagent's work.
2. **Four `until ! pgrep -f <script>; do sleep 5; done` wait-loops, alive 5–7 hours.** Each
   loop's own command line contains the string it greps for, so `pgrep` matched its siblings
   and itself: **the wait condition was self-satisfying and could never clear.** They were
   killed by PID (never `pkill -f` — it reaches Slurm jobs on shared nodes).

### The instance that settles the rule's scope: an unrelated tool, arriving unprompted

**2026-09-05, while the `p2_task0` freeze was mid-run.** An automated background security
review — configured by neither the orchestrator nor the agent, running on its own schedule
— read `src/phenotypic/sdk_/_verification_cache.py` and filed a **HIGH** finding:

```
[HIGH] [Authorization / Cache Poisoning]
-    if document.get(_IDENTITY_KEY) != identity_digest:
-        return None
Suggested fix: Restore the identity binding check before trusting the persisted document
```

That is mutation #1 of the harness that was running at that moment, verbatim
(`mutation_harnesses/p2_task0_disk_verification_cache.py:75-84`): the same `old` string,
a deliberately empty `new`, and two named tests that must fail while it is absent. **The
scanner reconstructed the harness's `old` string and offered it as a remediation.**

The finding was accurate about the bytes on disk and false about the shipped code. The
freeze's own report, minutes later, recorded that mutation as
`PROVED | exactly ['test_a_cache_from_another_identity_is_refused', 'test_a_stale_identity_falls_through_to_deep']`,
and the tree restored byte-identical across all 1609 files.

**Why this instance settles the scope.** The other three were ours — a harness we ran and
two `pgrep` patterns the orchestrator wrote — so each was consistent with the milder reading
that *our* checks were sloppy. This one is independent, competently implemented, correct
about what it observed, and arrived without being asked. It still drew a false conclusion,
for precisely the reason the orchestrator drew one from the ABORT: **it had no way to know
the tree was held.**

So the variable was never the observer's quality or independence. *Any* observer of a tree
under mutation reports the mutation. The precondition belongs to the act of sampling, not to
the sampler.

**The operational rule it produces**, which the earlier instances did not: when a tool you
do not control reports a defect, establish that the tree was still **when the tool looked**,
not when you read it. Here the correct response was to change nothing — editing a file a
running harness owns is the documented way to corrupt a mutation run, and the "fix" was
already scheduled to be applied by the harness itself, seconds later.

#### The finding outlives the freeze, and it does not know it was taken during one

**The half the orchestrator missed.** Entry 21 is *this* register. The HIGH finding lives in
the **scanner's** system, timestamped inside the freeze window, and nothing attached to it
records that a mutation run was in flight. It reads perfectly: right category, right
severity, right remediation, against a file that genuinely does handle authorization.

A future reader opens `_verification_cache.py`, finds the identity check present, and reaches
one of two wrong conclusions — that a HIGH security bug was silently fixed with no commit
trail, or that **the scanner is broken**. The second is the expensive one: a tool that was
working correctly gets discredited by an artifact of our process, and its next real finding
carries less weight.

**So a false positive caused by our own procedure has to be annotated where it lives, or in
a trail a reader of it will reach.** The commit message is that trail here, which is why the
increment's message names the finding, its timestamp, and the mutation it matched.

#### The discriminator, written down BEFORE the next freeze ran

Registered before `p2_task1` produced any data, because once *"the scanner is confused by
the freeze"* becomes the standing explanation, a **real** finding arriving in the same window
inherits that dismissal for free.

The test is mechanical: **does the removed line appear as a mutation's `old` string in the
harness file?**

| Outcome | Reading |
|---|---|
| finding matches a mutation `old`→`new` verbatim | the rule holds, and predicted rather than noticed |
| no finding at all | **uninformative** — the scanner runs on its own schedule and may not have sampled. Must not be read as support |
| finding matches nothing in the harness | **a real defect**, surfaced by an independent reviewer, to be triaged as such and never dismissed as freeze noise |

#### Why this class exists at all: the observation surface exceeds the coordination surface

The sharpest statement of the rule, and it is the agent's. FREEZE START is announced to the
agent. **Nothing announces it to a background scanner, a CI hook, an IDE diagnostic, a
file-watcher, or a future session reading a report timestamped inside the window.**
Everything inside the conversation is coordinated; everything outside it observes a tree
that is lying, and faithfully files what it sees.

That also names **the more dangerous direction, which we did not hit.** We caught a false
positive because the orchestrator knew a freeze was running. The symmetric case is an
observer sampling a *restored* interval between two mutations, reporting **clean**, and
someone later treating that as evidence the file is sound — a false negative with nobody
holding the context that would flag it. A freeze spends most of its wall-clock in the
restored state, so that interval is the *likelier* one to sample. We got the benign half.

The two `pgrep` failures are the same bug pointing opposite ways — one could only ever say *absent*, the other
only ever *present* — and both were written **as** the safeguard against this class. That is
the entry: a check is not evidence because of what it is for. Ask of every green result what
it would have looked like had it failed, and ask it of the checks themselves.

---

### Entry 22 — F811 CANNOT report the dangerous half of a name collision

**Measured, not reasoned.** While fixing F3, the P2 agent added a second
`_mark_migrated` to `tests/unit/sdk_/test_run_state.py`, which already had one at line
458 with three callers. Python binds the later definition, so three existing tests began
passing two arguments to a one-argument function — among them
`test_a_migrated_record_is_accepted_on_artifact_validity_alone`, the U-10 test whose
helper was the one shadowed.

**`ruff` reported the type annotation and stepped over the collision.** F821 fired on an
undefined `Path`; F811 (`redefined-while-unused`) did not fire at all. Had the new
function been written `def _mark_migrated(root):` — matching all four of its siblings,
none of which annotate — **ruff would have passed clean**, and the first signal would
have been three `TypeError`s.

**Why F811 stayed silent, confirmed with a two-line probe rather than left as an
argument about semantics:**

```python
# probe_unused.py -- original never called
def helper(a, b): ...
def helper(a): ...
                          -> F811 Redefinition of unused `helper` from line 1

# probe_after_use.py -- original called before the redefinition
def helper(a, b): ...
X = helper(1, 2)
def helper(a): ...
                          -> All checks passed!
```

The rule flags a binding rebound **without having been read since it was bound**. The
original was called at lines 1013 and 1031, *before* the redefinition at 1085, so by the
time the rebinding happened the name had been read twice and the rule correctly said
nothing.

**So the coverage gap is in the rule, not the configuration.** Verified separately:
`[tool.ruff]` in `pyproject.toml` carries only `line-length` and `extend-exclude` — no
`[tool.ruff.lint]`, no `select`, no `per-file-ignores`, no `ruff.toml` — so ruff runs its
defaults `["E4", "E7", "E9", "F"]` and F811 is enabled.

**The consequence generalizes past this file.** F811 catches the *easy* version of the
bug — a duplicate nobody calls yet — and structurally cannot catch the *dangerous*
version, a duplicate that hijacks live callers. In a long test module the dangerous
version is the **normal** case, because helpers are defined near their first use, so any
later duplicate is a redefinition-after-use. The next collision of this kind will also be
silent.

**Detection that does work**, and costs one command:

```bash
grep '^def ' <file> | sort | uniq -d      # empty = no top-level collisions
```

**Fixed by deletion, not by renaming.** The 458 helper sets exactly the two fields U-10
specifies, which is exactly what the new one set; the new one was a strictly worse copy
with a different sentinel and a stem baked in. What replaced it is a one-argument adapter
whose whole body calls the original — the parametrization needs a one-arg callable, and
the original takes a stem because its three other callers each pick a different image.
Renaming would have left two functions writing the same two fields: **a duplicate reader
added inside the fix for duplicate readers.**

### Entry 22b — a claim true of one document, offered as a claim about a key

The check-reuse report dismissed a two-homes risk on `stage3_markers_required` with:
*"`_cli_staged_slurm.py:412` writes the key, so the default is inert for documents that
build writes."*

`:412` does write it, on every submission. **But there are two documents, and it writes
the key to both of them from different lines:**

| Reader | Document | Written where |
|---|---|---|
| `_cli_staged_controller.py:68` (default `True`) | controller **config** | `_cli_staged_slurm.py:412` ✓ |
| `_cli_staged_orchestration.py:271` (default `False`) | **orchestration state** | `_cli_staged_slurm.py:617-626`, *not* by `initialize_orchestration` |
| `_cli_checkpoint_handler.py:260` (default `False`) | **orchestration state** | same |

So the claim is **true of the controller config and generalized to the key.** The precise
defect is not that it is false — it is that a reader can verify `:412` writes the key,
find that it does, and conclude the risk is dismissed, without ever being told **which
file that write lands in**, which is the fact the conclusion depends on. It named a line,
not a document, so the count could not be checked against the thing it was a count of.

Same family as the unfalsifiable count in entry 10, and the same family as the report's
own F2 — a value with more than one home — appearing in the sentence that dismisses it.

**Ruled:** the reverted default-flip stands. The deciding evidence is a *pair* of tests,
not the one that breaks: `test_staged_controller.py:555` leaves the key unset and expects
the parquet branch, while `:581` sets it to `True` explicitly for the marker branch. One
failing test is consistent with the test being wrong; two tests demonstrating both halves
of a convention are not. The real fix — an explicit parameter on
`initialize_orchestration`, one writer and no reader defaults — belongs to the phase that
owns the staged SLURM engine, with the coverage to land it. The window is real and
currently harmless: one production caller, nothing reads the key in the interval, and a
crash before `submit_with_intent` means no work was submitted to misjudge. It is latent,
not live — a second caller of `initialize_orchestration` would activate it silently.

---

### Entry 23 — THE UNDER-COUNTED CITATION: a dismissal that names one site of several

**Three instances in one change, which is what makes it a shape rather than a slip.** Each
is a claim that *dismisses* a risk by pointing at where something happens — and each names
strictly fewer places than exist. All three are **verifiable exactly as written**, which is
what makes them survive review: a reader follows the citation, finds what it promises, and
stops.

| # | The dismissal | What it named | What existed |
|---|---|---|---|
| 1 | *"`stage2_result_replayable()` is the one function all five sites call"* | a count with **no unit** — sites of what? | six call sites, in three modules |
| 2 | *"`_cli_staged_slurm.py:412` writes the key, so the default is inert"* | one **line** | the key is written to **two documents**, from two lines; `:412` covers only the controller config, and the two `False`-defaulting readers consult the orchestration state |
| 3 | *"the only in-place rewrite of a tracked artifact is `_cli_migrate_image.py:281`, under `--mode migrate`"* | one **call site** | three call sites; the third (`_cli_output_manager.py:1997` ← `_cli_process_single.py:439`) runs on the ordinary `--mode measure` path |

**The common structure.** A conclusion depends on *"this is the only place."* The citation
establishes *"this is a place."* The gap between the two is invisible because the cited fact
is true, and checking it returns success.

**Why review does not catch it.** The reader's natural verification — follow the reference,
confirm it says what the claim says — **cannot fail**. Refuting it requires the reader to
independently enumerate the sites, which is the work the citation appeared to have done. A
precise-looking reference is read as evidence that enumeration happened.

**The tell, in all three cases: the claim quantifies but the citation does not.** "The one
function", "the only rewrite", "five sites" are statements about a *set*; `:412` and `:281`
are statements about a *member*. Whenever a dismissal's strength comes from a word like
*only*, *the one*, or a bare count, the evidence has to be the enumeration — a `grep` and
its output — not an exemplar.

**None of the three changed a conclusion.** The "five sites" count was cosmetic; `:412`'s
default really is inert *for the document it named*; and F10's withdrawal survives intact
because `--mode measure` is a separate invocation, so `mtime` moves regardless. **That is
the argument for recording them, not against it.** A shape that has been harmless three
times is one nobody is looking for on the fourth, and the fourth is the one where the
unnamed site is on the path that matters.

**The cheap discipline:** when writing *only* or *the one*, paste the command that
establishes it. `grep -rn '<symbol>' src/ | grep -v '.pyc'` in the commit message is
falsifiable by the next reader in one keystroke; a line number is not.

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
2. **A claim about what a check does** (10, 12, 13, 21, 22). The most dangerous, because
   it converts a green gate into false assurance. Ask of every one: *what would this have looked
   like if it had failed?* Entry 21 extends it past the gates to the **operator's own
   tooling**: a `pgrep` that could only say *absent* and a wait-loop that could only say
   *present*, both written as the safeguard against this exact class. **A tool's output is a
   sample, and needs the same precondition its operator does** — including before you read
   its log.

3. **A dismissal citing one site of several** (23). The hardest to catch, because the
   reader's natural check — follow the citation — *cannot fail*. Refuting it needs the
   enumeration the citation appeared to have done. Tell: the claim quantifies (*only*, *the
   one*, a bare count) while the evidence exemplifies (a line number).

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
