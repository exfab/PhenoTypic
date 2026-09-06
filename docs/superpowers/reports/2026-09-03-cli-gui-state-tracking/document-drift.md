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

#### The fix-side variant, which is worse

**A fourth instance, and the first where the under-count was in a FIX rather than a
dismissal.** Reporting the `--mode measure` repair, an agent wrote *"the defect was purely
that two callers handed it the wrong path"*. Counted properly, the recompile path alone had
**six** orphaned production readers, plus one in `phenotypicCLI.py` that neither party had.

Its own diagnosis:

> My "two call sites" was the count of readers **I happened to touch**, presented as the
> count that *needed* touching.

**The fix-side version is more dangerous than the dismissal-side version**, for a reason
worth stating: a dismissal invites scrutiny — a reader who doubts *"this is the only place"*
goes looking. A completed fix does not. "I changed the two sites" reads as a report of work
done, and nobody re-derives the denominator of a job someone says is finished.

**It also propagated into a ruling.** The orchestrator had approved fixing that defect
in-phase partly *because* "the fix is two call sites, not a refactor" — a premise supplied by
the same under-count. The ruling was right for a different reason than the one given, which
nobody would have discovered had the agent not corrected itself.

**What caught it:** an instruction to classify a *sample* before fixing. The first sample
happened to be fixture-shaped, so the sample lied too. What settled it was measuring the
**population** — one command asking, for each of 119 failures, whether its traceback contains
a `src/` frame:

```
PROD   32     ->  31 real regressions + 1 non-regression (a gate correctly RAISING)
TEST  1275
```

Note the 32nd. The proxy (`src/` frame ⇒ production regression) was checked rather than
trusted, and it failed once out of 32 — a gate whose `src/` frame is the behaviour under
test. **A count that gets reported as the thing it proxies for is how the original
under-count happened, one direction over.**

#### The mechanism behind most of these: identical lines in different functions

**Named by the agent that kept hitting it**, and it explains why under-counts recur even
when the person counting is careful:

> Structurally identical lines, different call paths; **I pattern-matched on the line I had
> already read.**

`_cli_migrate_image.py` computes `sha256(task.marker_path.read_bytes())` in `_dry_run_result`
and again in `migrate_image_task`. Same expression, same variable, different functions,
different reasons. Having read and classified the first, **the second reads as
already-handled** — recognition of the *line* substitutes for enumeration of the *sites*.

That is what produced "four sites" when there were five in one file, and it is the same
mechanism as reading a fact stated twice in a long document and updating one instance.

**The counter is mechanical and it is the only one that works here:** enumerate by
`grep -n`, and classify every hit including the ones that look like duplicates of hits you
have already classified. **A hit that looks familiar is the dangerous kind**, because
familiarity is exactly the signal that suppresses the second look.

*(And it applies to your own edits, not only to counts you are handed: the agent repointed
four sites, reported four as complete, and the fifth was in the file it had just finished
editing. Re-derive the population of your own diff.)*

**The cheap discipline:** when writing *only* or *the one*, paste the command that
establishes it. `grep -rn '<symbol>' src/ | grep -v '.pyc'` in the commit message is
falsifiable by the next reader in one keystroke; a line number is not.

---

### Entry 24 — A PLAN STEP SUPERSEDED BY A LATER RULING, still written as an instruction

**A new kind, and the most dangerous one in the register, because a plan is not read — it is
followed.**

`phase-2-identity-schema.md:515-528` specifies the body of `SPEC-B1`'s missing test:

```python
assert after.inventory_digest != before.inventory_digest   # on the MINTED identity
```

Written literally today, that compares `""` with `""` and **fails against correct code**.
`REUSE-F2` was ruled option (a) — `inventory_digest` is reader-owned, the minter no longer
carries the field, and `_inventory_digest_for` was deleted. The plan step was correct when
written and was superseded by a decision taken afterwards. **Nothing propagated the ruling
back into the plan.**

**Why this is worse than a stale docstring.** Every other entry here misleads a *reader*,
who can check. This one instructs an *implementer*, who has been told to make it pass. The
failure mode is not confusion — it is an implementer meeting a red assertion they were
directed to write, against code that is right, and concluding the **code** is wrong. The
plan supplies both the false expectation and the authority to act on it.

The near-miss was avoided only because the implementer stopped to ask why a documented
assertion compared a field the ruling had deleted, rather than treating the plan as
authoritative. **A literal execution of the plan would have produced either a failing test
or a reverted fix.**

**Where the proof went instead.** Scope change is now proved at the reader, through
`run_identity()`, which is also where `REUSE-F2` put the value's only producer. That has a
second benefit the plan's version lacked: it does not restate `canonical_digest(work_ids)`
in the test, which is exactly the duplication `IMPL-F4` was filed for. **The superseded step
would have re-introduced a defect the same gate was fixing.**

**The mechanical gap.** A ruling that changes a design updates the spec and the code, and
`EXECUTION.md` requires a drift-register row. Nothing requires re-reading the *plan tasks
not yet executed* to see which the ruling invalidated. The plan is downstream of every
ruling and is treated as upstream of every task.

**The rule:** when a ruling changes a value's ownership, lifetime, or existence, grep the
unexecuted plan for that value's name before closing the ruling. Here that is one command —
`grep -rn inventory_digest docs/superpowers/plans/…` — and it would have found `:515-528`
the day `REUSE-F2` was decided.

*(The reviewer's note that this assertion "would have caught E-1" is true and now moot: E-1
was fixed by deleting the value the assertion would have checked.)*

#### The harder half: a ruling that changes NOTHING in the code propagates furthest

**Searching for the rest of the damage found a set nineteen times larger, from the ruling
nobody thought of as a ruling.**

`REUSE-F2` was *enacted* — a field was emptied, a function deleted — and it cost exactly one
plan casualty (`:515-528`), because an enacted change leaves a symbol to grep for.

§5.1's five-token collapse was **withdrawn**. Nothing in the code changed. And it left
**nineteen** stale plan sites across seven unexecuted documents, counted independently by
both the agent and the orchestrator:

| File | sites | |
|---|---|---|
| `phase-3-per-image-record.md` | **8** | the next phase to execute |
| `phase-5-fanout.md` | 5 | |
| `README.md` | 2 | |
| `phase-4`, `phase-6`, `phase-7`, `EXECUTION` | 1 each | |

*(The 13 in `phase-2` and 6 in `design.md` are not casualties — those documents record the
withdrawal, so the token appears legitimately.)*

**`phase-6-consumer-migration.md:81` is entry 24's shape in its purest form:**

> *"P2 Task 4 renamed `scheduler_epoch` on `publish_image_success` … the rename shifts line
> numbers under every citation this task makes into that file."*

A false premise stated as accomplished fact, used to justify a warning about line drift —
**inside the callout telling the reader not to trust `file:line`.** An implementer
regenerating those greps finds no rename and must decide which half of the paragraph to
believe.

**Why the withdrawal propagated further than the enactment.** A ruling recorded as *"do
nothing"* does not feel like a change, so nobody greps for what assumed it. But every plan
step written before it was written *expecting* it — and those steps are now instructions to
build a thing that was cancelled. **The absence of a code diff is exactly what makes the
plan diff invisible.**

**So the rule from entry 24 is too narrow as first written.** It is not *"when a ruling
changes a value's ownership, lifetime, or existence, grep the unexecuted plan."* It is:

> **Every ruling gets the grep — especially the ones that change nothing.** A withdrawal
> leaves no symbol to search for in the code, so the plan is the *only* place its
> consequences are written down, and the only place they can rot.

*(A third family surfaced in the same sweep: `SPEC-C2`'s rename was applied to two files of
three. The fix script listed three and sliced `edits[:2]`, and the third entry's anchor had
`old == new` — two independent reasons to miss it, under a list that reads as covering all
three. One residual site was also line-wrapped, so the obvious grep did not match it.)*

---

### Entry 25 — A CORRECTION THAT NAMES THE TRAP, THEN STEPS IN IT

**Found at P3's open, before a line was written, and the plan had already described the
failure it was about to cause.**

`phase-3-per-image-record.md` states how many modules Task 1 creates in two places, and they
disagree:

```
:72   | Create sdk_/_image_record.py      | Readers and shared vocabulary...
:73   | Create _cli/_cli_image_record.py  | Writers only...
:252  - Create: src/phenotypic/_cli/_cli_image_record.py      <- Task 1's Files block: ONE
```

Task 1's own test snippets import `phenotypic.sdk_._image_record` **eight times**. So an
implementer following the Files block creates one module, hits `ModuleNotFoundError`, and
takes the cheapest repair: move the readers into `_cli`. That re-creates a layering deadlock
which does not surface until P6 needs `valid_image_success` in `sdk_` parsing a record whose
parser now lives in `_cli` — four phases later.

**The plan warns against exactly this, 180 lines above the block that causes it:**

> *"An earlier revision changed this table and left twelve snippets importing readers from
> `_cli` — an implementer resolving the resulting `ImportError` by majority would put them
> back in `_cli`, re-creating the exact deadlock this split prevents, and discovering it four
> phases later."*

**The warning and the trap are the same edit's two halves.** A reviewer noticed the hazard,
wrote a precise callout about it, and the revision that carried the callout left a second
block stating the wrong count. The document now contains its own postmortem, filed in
advance, one screen away from the defect.

**Why this is not just "a stale block".** Every other stale-document entry here is a claim
that *decayed*. This one was authored **alongside a correct account of why it is dangerous**,
which means proofreading against intent could not have caught it: the intent is stated, and
correct, and adjacent. Only checking the two counts against each other finds it.

**The tell, and it is cheap:** a document that has been revised states its load-bearing
facts more than once, and revision updates one site. **Before trusting a block, ask whether
the document says the same thing somewhere else** — the contradiction is invisible to a
reader who finds the first statement and stops, and it is *more* likely in a carefully
revised document, not less.

*(Same revision's other residue: three of five test snippets carry column-0 imports inside
function bodies — a syntax error if copied. The signature of a machine edit applied without
parsing its own output, which is also entry 25's provenance.)*

---

### Entry 26 — A SCHEMA IS NOT A RENAME: two discriminators for repointing a reader

**Both found while repointing readers after P3's clean break, and both matter for P4-P7,
which repoint the rest.**

#### 1. Path sites are mechanical; VERSION sites are a migration

Seven production readers were orphaned when the publisher stopped writing
`image_complete/`. Five read `work_id` and `artifacts`, which the record carries with
identical shape — pure path repoints. One (`recompile_store_lock_path`) never opens the
file at all, deriving a lock name by `.with_suffix()`, so the break does not reach it and
repointing it **moves a lock**, which is a concurrency change, not a read change.

**The seventh decides everything:**

```
_cli_recompile_slurm_scripts.py:569   if marker.get("version") != SUCCESS_MARKER_VERSION
_cli_recompile_recovery.py:782        (same shape)

SUCCESS_MARKER_VERSION = 2      (_io_constants.py:728)
RECORD_VERSION         = 1      (_image_record.py:64)
```

A path-only repoint makes **every record fail that check** — and those functions return
`None`/`False` on a bad marker, so the outcome is not an error but *"this image has no valid
authority"*. **Silent-and-wrong, wearing a completed fix's clothes.**

> **The record is not a marker with a different name. It is a different schema with its own
> version line.** A site that resolves a *path* is mechanical. A site that also asserts a
> *version* is a migration, and the constant must move with the path.

That discriminator is settleable by `grep` rather than judgement, which is what makes it
usable at scale — and it is why doing the five mechanically while deferring the version pair
would have been worse than doing nothing: a half-migrated authority path whose un-migrated
half fails **closed**, and quietly.

#### 2. A fixture's INTENT and its MECHANISM diverge silently

The same sweep, in test fixtures, produced the opposite lesson — and a blanket
find-and-replace would have been wrong in both directions at once.

**`repoint_marker_at_hdf` reads the RECORD and writes the LEGACY MARKER** — two different
paths, deliberately, because its job is turning current publisher output into the shape a
genuine legacy tree carries. It also had to start **deleting the record**: a tree carrying
both shapes is not a legacy tree, and distinguishing those is the schema gate's entire
purpose. A path substitution would have built a tree that is legacy by one signal and
current by another — an input the gate is *entitled* to classify either way, used to test
the gate.

**`strip_completion_evidence` removed only `image_complete/`**, which *was* every per-image
publication when it was written. After the clean break, a "markerless" tree from that helper
would still have been **fully certified by its records**. The helper's name describes an
intent; its body encoded a mechanism; the mechanism moved and the name did not.

> **A helper named for an intent goes wrong silently when the mechanism moves.** The name
> still reads true, the body still runs, and the tree it produces is no longer the tree the
> name promises.

#### The sharpest sub-case: which NOUN does "legacy" modify?

**Five right answers emerged under one grep pattern** (`reads image_completion_marker_path`),
and the fifth is the one a careful implementer still gets wrong:

| Site | Correct action |
|---|---|
| the ordinary fixtures | **repoint** to the record |
| `repoint_marker_at_hdf` | read record, write marker, **delete** the record |
| `strip_completion_evidence` | delete **both** trees |
| `test_migration_republishes_state` | **no change** — the migrator rewrites the legacy marker in place |
| `legacy_file_marker` | **repoint, but keep the descriptor** — see below |

`legacy_file_marker` was first classified as "no change", by reading *legacy* as naming the
**file's location**. It names the **descriptor's shape** — a v1 descriptor carrying no
`kind`, which must still read as `"file"`. That property now lives behind the record, so
leaving the fixture on `image_complete/` makes its test return `False` **for the trivial
reason instead of the interesting one: passing while testing nothing.**

> **A fixture whose name describes an intent can survive a mechanism change while quietly
> ceasing to test that intent.** The name is not stale — it is *correct about a different
> noun* than the one the repoint concerns.

**In this codebase "legacy" modifies at least four different nouns** — a file's location, a
descriptor's shape, an image format (`.h5`), and a publisher era — and they call for opposite
actions. Before repointing anything whose name carries the word, establish which one.

**And not every "version" test is about the same version.**
`test_marker_version_is_bumped` asserts `SUCCESS_MARKER_VERSION >= 2` about the **legacy**
constant, which still exists and still guards the retained-`.h5` case. A sweep that treats
"version" as one subject collapses two.

**The combined rule for any repointing sweep:** ask of each site *which shape is it
building*, not *which path is it reading*. Those have different answers, and only the first
one is the question.

---

### Entry 27 — COVERAGE DECAY BY REMOTE CHANGE: an axis dies, the suite stays green

**The first entry here that is not about a document at all.** Everything above is prose that
stopped being true. This is a *test* that stopped testing what it says it tests, without
anyone editing it.

`tests/unit/cli/conftest.py`'s `ArtifactWorld._write_success_marker` has two branches. The
`parquet` branch calls the real `publish_image_success` and therefore followed P3's clean
break automatically. **The else branch hand-writes a legacy `image_complete/` marker**, and
its comment states its purpose:

> *"No artifact to describe: write the marker anyway, pointing at the parquet that is not
> there. `valid_image_success` then returns False — identically in both worlds — which is the
> **stale marker** case."*

After the break, `valid_image_success` reads `images/`. That hand-written marker is never
consulted. **It still returns False — but because the record is ABSENT, not because a marker
is STALE.** The stale-marker path is exercised nowhere in the harness.

**Nothing reports it, because the test passes.** And the axis goes inert: wherever
`parquet=False`, both values of `success` produce "no record" and therefore the same verdict.
`ARTIFACTS` is `product([False, True], repeat=5)`, so 16 of 32 combinations have
`parquet=False` — the axis is **dead in 192 of the parity suite's 384 cases**, at full green.

**The citation that makes it damning.** `test_staged_resume_parity.py:26-32` exists to
prevent this, on this axis:

> *"The FIFTH axis is load-bearing… Without this axis that branch is never exercised —
> `valid_image_success` returns False in both worlds — and the parity test passes while
> production breaks."*

Someone hit this once, added the axis, and wrote down why. **A source change three files away
then half-removed it, and the warning could not fire because the warning is prose in the file
that was not edited.**

#### Why this is its own shape

A stale document misleads a reader who consults it. A decayed test **withdraws a guarantee
nobody asked for again** — the suite's green is unchanged, the diff is empty, and the loss
is invisible at every point where someone might look.

> **A test's coverage is a property of the production paths it reaches, not of its own text.
> So it can stop covering without changing.** Any test whose fixture hand-builds an artifact
> the production writer also builds is holding a copy of a contract, and the copy does not
> follow the original.

**The detection that works** is the one the agent used: for each axis, assert that *some*
group of otherwise-identical cases holds more than one verdict. An axis that no longer moves
any outcome is either a fixture bug or a real finding, and it names itself. That check was
written for a different harness in the same phase and would have caught this one on the day
it broke.

**And the fix has the same trap one level down.** Restoring the axis means hand-writing a
*record* that is rejected — but rejected **for staleness rather than for shape**. A record
that fails for the wrong reason restores the axis's appearance and not its content, and the
second instance would be harder to find because someone would already have fixed it. So the
fix must assert on `record_rejection`'s **reason string**, never on the boolean.

---

### Entry 28 — A PROXY THAT DEGRADES INSIDE ITS OWN USE CASE

**The best instance of this register's central class, and it invalidated a conclusion both
parties had already acted on.**

To separate "test churn" from "production regression" across 119 failures, the agent asked of
each traceback: **does it contain a `src/phenotypic/…: in` frame?** A sound proxy, applied to
the population rather than a sample, and it correctly caught its own one exception — a 32nd
`PROD` hit that was a gate legitimately *raising*.

It reported **79 of 79 migrate failures as `TEST`-frame**, and both parties concluded migrate
needed no behaviour change. It then swept migrate's fixtures on that basis.

**`--mode migrate` was in fact totally broken.** `_cli_migrate_manifest.py:392` builds
`task.marker_path` from `image_completion_marker_path`; `publish_image_success` now returns
the *record* path; and `_cli_migrate_image.py:580` compares them and raises when they differ.
The guard fires on the first image of every tree.

#### Why the proxy could not see it

**The migrate unit tests die in their fixtures — reading a marker — before execution ever
reaches `:580`.** So the traceback contains only test frames, truthfully, and the production
defect beneath is never exercised.

> **A frame check tells you where a test DIED, not whether the code beneath it is sound.**
> It is **sound as a LOWER bound on production regressions and unsound as an UPPER bound** --
> and it was presented, and acted on, as both.

**The consequence for anything built on the count.** P3's 24 `xfail` markers were derived
from that upper bound, so **24 is a floor, not a total.** A test that dies in its fixture and
would *then* hit deferred production code is invisible to the check: repointing the fixture
moves the death downstream into the deferred region, turning a `TEST` classification into a
`PROD` failure outside the marked set. `test_embedded_measurement_recompile.py:97` is exactly
that case.

**The true deferred set is knowable only by re-running after the fixture sweep.** Any
classification of failures taken *before* fixtures are repaired is provisional by
construction.

And the failure is not random: a fixture that reads an artifact the change just moved dies
*early*, in exactly the population where the change is most likely to have broken production
code. **The proxy degrades hardest precisely where it is being relied upon** — it
under-reports production defects in proportion to how thoroughly the change broke the
fixtures.

#### The two misses were one miss

The only test reaching `:580` runs full `--mode migrate` and lives in `tests/integration`,
which was outside the measured population (see the population-ownership rule in
`EXECUTION.md`). So the narrow lane did not merely omit two files: **it produced a confident
classification that was an artifact of where execution stopped.** The gap and the regression
are one defect seen twice.

#### What to do instead

- **Fix fixtures first, then re-measure.** A frame check is only meaningful over a suite whose
  fixtures reach the code. Classifying before repairing inverts the dependency.
- **Ask of any proxy: in what conditions does it degrade, and am I in them?** "Does the
  traceback reach `src/`?" is sound when tests run and vacuous when they abort early — and
  aborting early is what a schema change causes.
- **An end-to-end test is not redundant with unit tests of its parts.** It is the only thing
  that reaches the code *after* the fixtures, and it is the first thing a narrow lane drops.

*(The agent revised its own conclusion unprompted and said plainly that "79 of 79 are
`TEST`-frame" had been true and had not meant what both parties took it to mean. The
correction is the reason the defect was found before the commit rather than after.)*

---

### Entry 29 — A LIST COPIED FROM THE ADJACENT VOCABULARY. 2026-09-05.

**The spec's demotion list is not a garbled version of the right list. It is verbatim the
*wrong* list — a different, neighbouring closed set that shares two of its three members.**

`design.md:275` demotes `processing_state.datasets.{completed, failed, started}`. Compare
what the writer actually wrote (`_cli_state_management.py` at HEAD, `:83-88`) against what
the event log actually carries (`_cli_update_state.py:237`):

| key | in `datasets.<ds>` | an event `status` | spec demotes | P3 drops |
|---|---|---|---|---|
| `completed` | ✅ | ✅ | ✅ | ✅ |
| `failed` | ✅ | ✅ | ✅ | ✅ |
| `errors` | ✅ | ❌ | **❌ omitted** | ✅ |
| `initial_images` | ✅ | ❌ | **❌ omitted** | **❌ kept** |
| `started` | **❌ never written** | ✅ | **✅ demoted** | n/a |

`{completed, failed, started}` is exactly the event-status set. The two dict keys the spec
omits — `errors` and `initial_images` — are exactly the two with **no** event counterpart.
That is not a typo distribution; it is the signature of copying one closed set where the
other belonged.

#### There are THREE overlapping sets here, not two

Found while checking the correction itself, and it explains why the collision was so easy to
make:

| set | `completed` | `failed` | `started` | written where |
|---|---|---|---|---|
| event-log `status` | ✅ | ✅ | ✅ | `processing_events.log` |
| `DashboardManifestKey` | ✅ | ✅ | ✅ | `manifest.json` (`_manifest_builder.py:780`) |
| `ProcessingStateKey` | ✅ | ✅ | **constant only, no writer** | `processing_state.json` |

**`started` is a live, correct key in two of the three files** — which is exactly why writing
it in a sentence about the third feels right and reads right. `_io_constants.py` declares
`STARTED` twice, at `:2391` and `:2453`, because two different classes legitimately need it;
only the second has nothing behind it.

So the discriminator cannot be *"is this key real?"* — it is real three times over. It can
only be *"which file is this sentence about, and does that file's writer emit it?"*

#### Why this one is worse than a wrong name

The usual drift (entries 5, 6, 7, 9) points at a name that does not exist, so the reader's
first grep refutes it. **This one survives the grep.** `ProcessingStateKey.STARTED` is a
real constant at `_io_constants.py:2452`, so a reader checking "is `started` a thing?" gets
*yes*. The false part is not the name — it is the **container**: `started` has never once
been written into `datasets.<ds>`, and its only reference in the entire tree is
`test_io_constants.py:827` asserting its own spelling against itself.

> **When two closed sets share most of their members, naming a member is not evidence you
> named the right set.** The check that discriminates is *where is it written*, never *does
> it exist*.

#### The cost it nearly imposed

An implementer following §4.2 literally drops three keys and keeps `errors` — the opposite
of correct on both counts. The P3 agent instead followed a task callout that said **four**,
dropped all four, and destroyed `initial_images`, which no event can reconstruct. So the
spec and the callout were wrong in *opposite directions*, and the intersection of "what both
documents agree on" was the only safe subset. Neither document alone gets you there; the
writer and the load path do.

**The corrective is the same one entry 23 reached from the other side:** the authority for
what a file contains is the code that writes it. Here that is nine lines, and reading them
settles in seconds a question two documents disagreed about.

*(Recorded at the team lead's instruction after the lead independently traced the load path.
The lead's summary said "there is no `started` key"; the constant does exist, and the
sharper claim — no writer has ever put it in this file — is the one that survives the grep a
reader will run.)*

---

### Entry 30 — A SAMPLE ORDERED BY THE WRONG THING. 2026-09-05.

**The eighth sampling error of this change, and the only one where the whole population was
addressable in a single command that was already half-typed.**

57 `tests/migration/test_equivalence.py` goldens failed in gate lane 2. The disposition
offered three legs — scope, coupling, and magnitude — and named the third **"the decisive
one"**:

> `Max absolute difference among violations: 2.5933718981185905e-06 /
> 5.124068070544441e-08 / 2.8299792897057332e-08` — **Machine-precision jitter, not
> behaviour.**

The population, read across all 48 shard logs and sorted numerically:

```
7 values at 1e-08 ... 2.4e-05, then
0.0013848   0.0015385   0.0039216   1.0   1.5156933

mismatched elements: 15% .. 37% .. 54% .. 71.4% .. 99.5% x2 .. 100%
```

**A 100% mismatch at 1.5157 is a different array.** The distribution is bimodal; the claim
described one mode.

#### The mechanism, which is narrower than "it sampled"

The read was `grep ... | head -12` **on one shard log**. `head` is **position-ordered**. The
claim was about **magnitude**. The 1e-08 cluster happened to sit at the top of that file, so
the sample looked homogeneous — which is precisely what made it persuasive and what removed
the impulse to look further.

> **A sample ordered by anything other than the quantity you are claiming about is not a
> sample of that quantity.** It is a sample of file layout.

The correct read differs by one flag and one glob: `grep -h ... gate_*.log | sort -g`. Not a
tool gap and not a pattern insight — the population was already addressable in the command
that was written.

#### The tell was in the sentence that made the claim

The author wrote *"the decisive one."* **That label is the trigger condition for the
population check**, not a licence to skip it: a leg carrying that much weight is the one that
must rest on everything. The same author had applied that rule twice that hour — to the
orchestrator's `13` and to its own `17` — and not here.

#### Why the disposition survived anyway, and this is the transferable part

Scope (all seven operation subpackages) and coupling (zero P3 files under any operation
package) were **independent** of magnitude, and were presented as three separate lines rather
than braided into one argument. So a refuted third leg cost the argument its strongest-
sounding support and **changed none of its conclusion**.

> **An argument assembled from independent legs degrades gracefully; one resting on a single
> "decisive" leg fails completely.** The structural choice is worth more than the care taken
> on any one leg — it is what makes being wrong survivable.

#### What the correction actually changed — severity, not action

The disposition was identical either way: leave the 57 red, do not mark them, **do not
regenerate them**. What moved was what the finding *means*:

| | Reported | Actual |
|---|---|---|
| Character | machine-precision jitter | substantive drift, up to a wholly different array |
| Why unseen | — | `tests/migration` is **outside `testpaths`**; it entered only via the sbatch's `SCOPE=full` |
| Filed as | noise | an undetected behaviour change nobody has run |

**The wrong version was the one that buries it.** A finding filed as noise is closed, and a
suite outside `testpaths` has no second chance to raise it. It also inverts the strength of
the do-not-regenerate argument: at 1e-08 regenerating is merely wrong; at a 100% mismatch it
would bless something real and unexamined.

*(The author verified the refutation rather than accepting it, reproduced the numbers, and
identified its own error more precisely than the refutation had — that the ordering, not the
sample size, was the defect. That distinction is the entry.)*

---

### Entry 31 — THE REMEDY EXHIBITS THE DEFECT IT WAS WRITTEN TO PREVENT. 2026-09-05.

Entry 28 cost this change a production regression: a narrow lane dropped `tests/integration`,
which was precisely where `--mode migrate`'s total breakage lived. The remedy written into
`EXECUTION.md` was a rule — *derive the gate's scope from the diff, never from intuition* —
with a one-line recipe to make it mechanical:

```bash
git show --stat --format= <sha> | grep 'src/' | xargs -n1 dirname | sort -u
```

**The recipe returns an empty scope on exactly the commits that matter.** `git show --stat`
abbreviates long paths — `.../_core/_pipeline_parts/_image_pipeline_core.py` — so the
`src/phenotypic` prefix is elided and `grep 'src/'` matches nothing. Deep packages are the
ones that get truncated. An empty scope is **the narrowest possible lane**: a phase gate that
runs no `src/` suite at all.

> The tool introduced to prevent a too-narrow gate silently produces the narrowest one.

#### It also fails the other way, from the same bug

On a commit whose `src/` lines are *not* truncated, `xargs` splits the `|` separator and the
`+++--` counts into their own arguments, `dirname` obliges, and the output gains a bare `.` —
**the entire repository**. So one bug yields an empty scope on some commits and a whole-repo
scope on others, and neither is the answer.

#### Why it survived being "checked"

**It was verified against P3's own commit, where it printed the two correct packages.** That
commit has **nine** truncated stat lines; all nine happened to be under `docs/` and `tests/`,
so the `src/` answer survived by luck. The check was real, the output was right, and the
recipe was broken — a green result over the one case that does not exercise the defect.

The corrected form, checked against both a deep-path commit and a merge:

```bash
git diff --name-only <base>..<head> | grep '^src/' | xargs -n1 dirname | sort -u
```

A **range**, because a phase is several commits and `--name-only` returns nothing for a merge
commit by default; `^src/` anchored, because the path is now complete.

#### The transferable part

Entry 28's lesson was *ask in what conditions a proxy degrades, and whether you are in them*.
This is that question aimed one level up: **a remedy is a check too, and it inherits the
obligation.** The instinct that a fix needs no verification because it is the fix is what
carried this one into the document as the mechanical, no-judgment-required version of a rule
whose whole purpose was to remove judgment from scoping.

*(Found by a session reading `EXECUTION.md` cold, and confirmed here by running both forms
against `c8eeafba` — mine printed a `dirname: missing operand` error and nothing else; the
corrected one printed `src/phenotypic/_core/_pipeline_parts`.)*

---

### Entry 32 — A TRIPWIRE THAT FIRES ON THE FIX, WHEN NOTHING SCHEDULES THE FIX. 2026-09-05.

P3 deferred arming the schema gate to P7 under a user ruling, and guarded the deferral the
way this change guards every deferral — two `xfail(strict=True)` markers that become
**failures** the moment the flag is armed, forcing their own removal. The markers name their
owner:

> *"P7 Task 5 Step 1b arms the gate and turns this green."*

**P7 Task 5 Step 1b exists. It renames legacy trees (CAN-12).** It has nothing to do with the
flag, and `SCHEMA_GATE_ARMED` appeared **zero** times in the entire P7 plan. An implementer
working that plan top to bottom would rename the trees, finish the phase, and never touch the
flag.

#### Why the guard cannot report this

A strict `xfail` is self-cleaning **only if something eventually does the thing**. It fires on
the *fix*. With a fictional owner there is no fix, so the marker sits at XFAIL — which is a
**passing** state — indefinitely, and every gate reports green.

> **The mechanism that prevents a stale deferral cannot detect an unscheduled one.** Those are
> different failures, and the tripwire only covers the first.

Worse, it looks *especially* healthy: the reason string is specific, cites a real phase, a
real task and a real step, and would survive any check that the pointer resolves. It is a
**pointer to a name that exists** — normally the one form of claim that cannot be false
(see the mitigation note at the end of this register) — pointing at the wrong thing.

#### The two-sided repair

Neither half is sufficient alone:

- **The claim** — P3's plan still carried `### This task must ARM the schema gate` as a live
  instruction, 4 months of context after the ruling withdrew it. Struck through and marked
  ⛔ SUPERSEDED, with the derivation kept (the *reasoning* stayed correct; only the
  assignment moved).
- **The work** — P7 gained a real **Step 1d**, existing for this and nothing else, carrying
  the flip, the no-re-export warning, the two marker names to delete, and an instruction to
  **re-derive the signal count rather than trust the docstring** that had by then gone stale
  three times.

#### The general form

> A deferral needs **three** things, and this change had been shipping two: a **tripwire**
> that fails when the debt is paid, a **reason** that survives being read cold — and a
> **step in the receiving plan that will actually be executed.** Without the third, the first
> two document a debt that nobody is scheduled to pay.

*(Found by the P3 spec-adherence reviewer, which checked whether the ruling had propagated
rather than whether it had been recorded. It had been recorded — in the code, accurately, with
the honest cost stated. It had propagated nowhere else.)*

---

### Entry 33 — A QUIESCENCE CHECK THAT COULD ONLY SAY "QUIET". 2026-09-05.

**Entry 21's defect, in a different tool, committed by the same operator who wrote entry 21.**

Before snapshotting the tree for a gate, and again before running a probe while an
implementing cluster was working, the orchestrator asked whether anything had changed
recently:

```bash
find src tests -name '*.py' -newermt '-3 hours'    # "nothing -- tree is still"
find src tests -name '*.py' -newermt '-4 minutes'  # "nothing -- safe to probe"
```

Both printed nothing. **Neither could have printed anything.** `find` here is `bfs`, which
rejects that timestamp form outright:

```
bfs: error: ... -newermt "-4 minutes" -print
bfs: error: Invalid timestamp.
```

The error goes to **stderr**; stdout is empty. So the check's failure output and its
"all quiet" output are the same empty stream — and the empty stream is the one that
**authorises the next action**.

#### What it cost

The second call was immediately followed by a probe against `_cli_completion.py`, which had
been modified **fifty seconds earlier** by a cluster mid-edit on it. The probe raised
`NameError: DIR_IMAGE_RECORDS is not defined` — the import had landed at `:20`, the use at
`:885`, and the read fell between them. That traceback was very nearly written up as a
defect in a review report.

The first call is worse, because nothing failed: **"no edits in the last 3 hours" was
reported as an established fact and was never a measurement.** Only the SHA-256 snapshot
taken beside it was real evidence — which is the one reason the gate lane it guarded is
still trustworthy.

#### How it was caught, and why that generalises

Not by suspicion. By an **independent measurement that contradicted it**: a `stat` showing
`_cli_completion.py` modified 50 seconds ago, next to a "nothing changed in 4 minutes"
result. Two readings of the same question disagreed, and only then was the predicate
itself tested — against a file already known to be recent, which is a **known-positive
control** and costs exactly one command.

> **When a check's NEGATIVE result authorises an action, prove it can produce a POSITIVE.**
> Ask it something you already know the answer to. A predicate that has never been seen to
> say "yes" has not been shown to be a predicate.

#### The pattern across 21, 28 and 33

| | The check | Could it report the bad case? |
|---|---|---|
| **21** | `pgrep -af "a\|b"` — BRE alternation to an ERE matcher | No. Only ever "absent". |
| **21** | `until ! pgrep -f <script>` — self-matching wait loop | No. Only ever "present". Ran 5-7 h. |
| **28** | "does the traceback reach `src/`?" | Not when the fixture dies first. |
| **33** | `find -newermt '-N minutes'` under `bfs` | No. Errors to stderr, empty stdout. |

All four are **operator tooling**, not product code; all four fail **silently**; and three of
the four fail in the direction that says *proceed*. The register's standing question — *what
would this have looked like if it had failed?* — is hardest to apply to the throwaway command
you type on the way to the real work, because it does not feel like a check. It is one, and
its output is load-bearing.

---

### Entry 34 — AN `xfail` CAN TAKE A SECOND ASSERTION OFFLINE, SILENTLY. 2026-09-05.

Auditing 35 deferred `xfail(strict=True)` marks, one failed in the opposite direction from
the other 34 — `Failed: DID NOT RAISE`, a guard expected to fire that did not. Flagged for a
separate P4 instruction on the grounds that its success criterion is inverted: the repoint
must make the error *appear*, not disappear.

**That understated it.** `test_cli_recompile_slurm.py:2862-2905`:

```python
with pytest.raises(SlurmGenerationInactiveError):
    _run_overlay_task(..., {"restore_marker_authority": True, ...}, ...)

assert not overlay.exists()          # <- the property the test is NAMED for
```

When the block does not raise, `pytest.raises` fails at the `with` **exit**, so the line below
it **never executes.** The test is `test_stale_slurm_overlay_worker_does_not_publish_rendered
_bytes`, and *whether a stale worker publishes rendered bytes is currently checked by nothing*
— not by this test, not by any other. The strict marker means no one sees that.

> **An `xfail` says "expected to fail." It does not say "and everything after the first
> failure is now unverified."** A marker applied for one known reason quietly takes every
> later assertion in that test out of service, including ones motivated by something else
> entirely.

The consequence for the receiving phase is precise: **restoring the raise alone would leave
the no-publish property exactly as unverified as it is today, and would look fixed.** So the
P4 instruction asserts both halves.

#### The general form

A deferral marker's blast radius is the **whole test body after the failure point**, not the
one behaviour named in its `reason`. Before marking, ask what *else* that test asserts — and
if it is more than one property, either split it or say in the reason which assertions the
marker is also suspending. Neither had been done here across 35 marks.

#### And a fabricated identifier, in the same exchange

The class is `SlurmGenerationInactiveError`. The traceback line was truncated by the
orchestrator's own extraction script — `re.search(..., ".{0,70}")` — to
`...SlurmGenera`, and the orchestrator then wrote **`SlurmGenerationError`** in the brief:
a plausible completion of a truncated string, and a name that **does not exist in the
codebase**. A P4 implementer grepping it would have found nothing.

> **A truncated identifier completed from context is a fabricated identifier.** It differs
> from the sampling errors elsewhere in this register only in what got sampled — there, a
> subset of values; here, a prefix of a name. Both produced something plausible, and
> plausible is the failure mode: a mangled name looks wrong and gets checked, while a
> well-formed wrong one gets grepped once and quietly abandoned.

*(Caught by the reviewer reading the source rather than the brief. The truncation was in a
throwaway analysis script — operator tooling again, as in entries 21 and 33.)*

---

### Entry 35 — A JUSTIFICATION TRUE OF THE PATH IT DESCRIBES, READ AS A PROPERTY OF THE CHANGE. 2026-09-05.

The spec justifies collapsing the stage-3 marker into the record with a cost argument
(`design.md:578`):

> *"one JSON read replaces one read plus three `is_file()` probes across three directory
> trees."*

**Every word is true, and it is true only of the per-image decision point** — which was
already reading a marker, so the extra read is genuinely free there.

`stage3_completion_exists` has **three whole-inventory callers that previously did zero
reads**: the SLURM observer's polling path, the orchestration inventory, and the
controller's retryable/terminal split. For those, a bare `is_file()` became
`open`/`read`/`close`/`json.loads`. On a 6,000-image GPFS run the observer's poll goes from
6,000 stats to 6,000 parses.

#### The shape

The sentence is a **claim about one call site, positioned as the rationale for the
change**. Nothing in it is false; the defect is scope, and scope is the part a reader
supplies for themselves. Compare the `initial_images` wipe earlier in this change: a plan
callout said *"stop writing four keys"* over a justification true of **three**. Same error,
opposite direction — there the scope was too wide, here too narrow — and both were invisible
because the sentence read as complete.

> **A justification is a measurement of the case it names.** When it appears as the reason
> for a change, ask which *other* call sites the change touches, and whether the argument
> survives them. It usually has not been asked.

#### Why this one is not a bug and is still worth an entry

The collapse **is** the design, and no fix is proposed. The cost is real, unmeasured, and
sits under a user-facing surface — so the failure mode is not a wrong answer but a future
engineer meeting a mystery slowdown with a spec paragraph that says the change made things
cheaper. It is now written into P6 Task 6, which owns the observer, as a ⚠ block before the
first step rather than filed only here.

*(Found by the P3 implementation reviewer, which read the spec's justification against the
call graph instead of against the function it describes.)*

---

### Entry 36 — THREE LEVELS OF GREEN-BY-CONSTRUCTION, IN ONE TEST. 2026-09-05.

**The register's central mechanism, found three times in a row inside successive fixes for
itself.** Each layer was caught only because someone looked one level below the last fix.

**Level 1 — the original gate (review finding B7).** P4's plan proved INV-INPUTS by adding a
`_dataset_aggregated.parquet` fast path and asserting a test failed. But the fast path is on
the **legacy** discovery arm only, and the test ran on a **forward** tree. The gate was green
whether or not the violating arm survived.

**Level 2 — the fix for B7 (finding NEW-1).** The repair added a dedicated legacy-arm test.
Its fixture claimed to force the legacy arm via *"no processing state, or
`success_markers_required` false"* -- but `authorized_measurement_sources` delegates **both**
conditions to `_sources_without_state` (`_cli_completion.py:932-935`), which returns `None`
only when **neither** progress tree holds a single `*/*.json` (`:889-895`). The fixture
*published two images*, so it got a mapping back and took the authorized arm. **The
replacement gate was green for the same reason the original was.**

**Level 3 — below the fix for the fix.** Reaching the legacy arm is necessary and **not
sufficient**: the arm must also *prefer* the poisoned aggregate. `discover_measurement_sources`
skips it when `_aggregate_needs_image_name_recovery` is true and individual Parquets exist
(`_measurement_sources.py:161-167`). Measured by execution:

```
_image_name_column(['Metadata_ImageFile']) -> None            # -> needs_recovery -> SKIPPED
_image_name_column(['Metadata_ImageName']) -> Metadata_ImageName
metadata_member_for_header('Metadata_ImageFile') -> None
```

The plan's poison frame carried only `Metadata_ImageFile`. **The aggregate would have been
skipped, and the test would have passed having chosen nothing.**

#### The distinction that was missing at every level

> **Reaching a branch is not the same as the branch doing the thing under test.**

Level 1 failed to reach the *arm*. Level 2 reached the arm but not via the *route* it claimed.
Level 3 reached the arm by the right route and then failed the arm's own *precondition* for
doing the work. Three different senses of "the test exercises the code", and a fixture can
satisfy any two while failing the third.

The operational form, which is narrower than "ask what a green result would look like if it
had failed":

> **State the precondition of every layer between the fixture and the assertion, and assert
> each one before the act.** Not "this fixture forces the legacy arm" but: *no payload under
> progress/* → `authorized_measurement_sources` returns `None` → legacy arm → aggregate
> exists AND carries a valid `Metadata_ImageName` → the aggregate is preferred → the poison is
> read. Five links; the plan asserted one.

#### Why three rounds of review found three levels rather than one

Each reviewer looked at what the previous fix *claimed*, and checked the layer that claim
named. Nobody had reason to look below it until the layer above was sound. **That is not a
failure of the reviewers -- it is a property of layered preconditions**, and the only defence
is to enumerate the chain up front rather than discover it one round at a time.

*(Level 3 was found by the implementing agent while applying the fix for level 2, and
confirmed here by executing the two predicates rather than reading them.)*

---

### Entry 37 — THE RULE HELD IN THE DELIVERABLE AND NOT IN THE INSTRUMENT. 2026-09-06.

P4's plan carries a standing rule, written after entry 36:

> **Every assertion of a negative, or of an equality, must be preceded by an assertion that
> the fixture produced the thing whose absence or equality is being claimed.**

The implementing cluster applied it **fourteen times** in the test file it shipped —
correctly, including cases nobody had marked. It then wrote a throwaway probe to settle
whether `finalize_post_master_outputs` silently drops metadata, and the probe asserted an
absence with nothing behind it:

```
No measurement files provided to aggregate.
authorized_measurement_sources -> 0 authorized sources
_consistent_embedded_join_keys -> None        <- not (), because nothing was aggregated
Metadata_Strain in master: False              <- the "finding", over an empty fixture
  exit: 0
```

It **did** carry a guard — `exit 2` if it landed on the legacy path — so the author was
thinking about false results. The guard covered the failure mode they had in mind and not
the one that happened. An empty source set produced the same printed output as a confirmed
hazard, and the script reported success.

#### What is new here, against entries 21, 31, 33 and 34

Those are all *operator tooling failing silently*. This one adds the distribution:

> **The discipline was applied to the artifact and not to the instrument measuring it — by
> the same author, in the same hour, having just written the rule fourteen times.**

The deliverable gets the standard because it is the deliverable. The probe does not, because
it is "just a check" — and the check is what the decision rests on. Entry 33 said the
throwaway command does not feel like a check; this says the exemption survives even when
the author has the rule fully in mind and has been executing it flawlessly minutes earlier.

**So it is not a knowledge problem and cannot be fixed by writing the rule down again.** The
operational form has to attach to the *moment*, not the artifact:

> **A script written to settle a question is a test of that question.** Before reading its
> output, ask what it prints when the fixture fails to materialise — and if that is the same
> thing it prints when the answer is "yes", it has not been run yet.

*(The hazard the probe was meant to settle remains unmeasured and was handed to the next
task as an explicit unverified prediction, rather than being carried as established. That
disposition is the part that worked.)*

---

### Entry 38 — THE FAILING LINE NUMBER IS A POSITIVE CLAIM ABOUT EVERY LINE ABOVE IT. 2026-09-06.

A test reported `assert not True` at `:517`. The orchestrator read it as *"the master carries
user metadata"*, concluded the unconditional join was landing **before the master write**,
and told the implementer the master/mirror boundary had collapsed -- adding a warning against
making the tests pass by adjusting them rather than the placement.

The test's assertions:

```
515  assert "Metadata_Strain" not in master.columns
516  assert "Metadata_Strain" in mirror.columns
517  assert not master_carries_user_metadata(master)      <- reported failure
```

**pytest stops at the first failure. A failure at `:517` is a statement that 515 and 516
passed** -- the master has no user metadata column, the mirror does. The boundary was intact,
and the diagnosis was refuted by the very output the orchestrator had pasted into its own
message one turn earlier.

#### The mechanism, which is not sampling

Every other misreading in this register came from looking at **part** of the evidence. Here
the whole of it was present, in hand, and quoted. The failure was not extracting a fact the
output states implicitly:

> **In a sequence of assertions, the failing line is a positive claim about all of them
> above it.** A traceback does not only say what broke; it certifies everything that ran
> first.

That inverts how a failure is usually read. The eye goes to the assertion that failed and
treats the rest of the function as unexamined context, when the lines above it are the
strongest evidence available -- they are *passing assertions*, executed on the same fixture,
seconds earlier.

#### The compounding error

Having mis-diagnosed, the orchestrator then warned the implementer *"these four tests would
be very easy to make green by adjusting the tests instead of the placement."* The warning was
sound in general and, applied here, was pressure toward defending a defect that did not
exist. **A confident wrong diagnosis attached to a correct principle is worse than either
alone**, because the principle makes the diagnosis harder to contradict.

The implementer contradicted it anyway, with the line numbers, and was right on all three
counts -- an unowned column in a fixture, a key-format bug, and a pre-existing ordering
defect that predates the phase.

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
2. **A claim about what a check does** (10, 12, 13, 21, 22, 27, 28, 30, 31, 32, 33, 34, 36, 37). The most dangerous, because
   it converts a green gate into false assurance. Entry 27 is its limit case: no claim was
   made and none went stale — a *test* silently stopped covering what its own comment says it
   covers, because a path it depended on moved three files away. Ask of every one: *what would this have looked
   like if it had failed?* Entry 21 extends it past the gates to the **operator's own
   tooling**: a `pgrep` that could only say *absent* and a wait-loop that could only say
   *present*, both written as the safeguard against this exact class. **A tool's output is a
   sample, and needs the same precondition its operator does** — including before you read
   its log.

3. **A plan step superseded by a ruling** (24). The only kind that instructs rather than
   informs, so its victim is an implementer who has been told to make it pass — against code
   that is correct. Rulings propagate to the spec and the code; nothing re-reads the
   *unexecuted plan tasks* they invalidate.
4. **A dismissal citing one site of several** (23). The hardest to catch, because the
   reader's natural check — follow the citation — *cannot fail*. Refuting it needs the
   enumeration the citation appeared to have done. Tell: the claim quantifies (*only*, *the
   one*, a bare count) while the evidence exemplifies (a line number).
5. **A list copied from the adjacent vocabulary** (29). Shares shape 4's defence — the
   reader's natural check cannot fail, because every name in it is real — but the error is
   in the *container*, not the name. It appears wherever two closed sets overlap: statuses
   versus keys, labels versus headers, stages versus modes. Tell: the list's members are
   exactly some *other* set's members, and the keys it omits are exactly the ones that other
   set never had.

### The single mechanism behind most of this register

**Named by the P3 agent after hitting it three times in one phase, and it is a better
statement of the register's theme than the question this document opens with.**

> **Presence and reachability differ by a branch.** Something is *there*, so it is assumed to
> be *doing work*, and nothing checks the gap.

Its three instances, all P3, all green throughout:

| Finding | Present | Not reaching |
|---|---|---|
| **F6** | `_image_record.py` exists *because* of INV-LAYER | the layering test's `_MODULES` tuple never listed it |
| **F7** | the `success` axis is enumerated in 384 cases | inert in 192 of them — both values produced "no record" |
| the triage count | the `record_rejection` assertion is written into every world build | reached in 96 of 384 — the other branch never sees it |

And the same mechanism, restated, covers most of what precedes it here: a **guard whose scope
is enumerated rather than derived** (F6, and the harness `TARGETS` list); a **proxy that
terminates early** (entry 28's frame check, present in every traceback, reaching past the
fixture in none); a **helper named for an intent** whose mechanism moved beneath it
(`strip_completion_evidence`); a **constant still spelled correctly** for a noun that is no
longer the one at issue (`legacy_file_marker`).

**In every case the artifact is present, correct-looking, and not doing the job its name
claims** — and the suite is green, because green measures what ran, not what was reachable.

**The operational form** is narrower and more checkable than *"ask what a green result would
look like if it had failed"*, which is the question this register opens with and which
requires imagination:

> **For every guard, axis, assertion or watched-module list: what fraction of the cases does
> it actually reach?** If the answer is "all of them" and you have not measured it, that is
> the assumption to test first.

The P3 gate's `test_every_axis_changes_at_least_one_outcome` is exactly that check made
executable — it groups by every axis but one and requires some group to hold more than one
verdict, so an axis that reaches nothing names itself. **It is the one mitigation here that
generalises to all four shapes above.**

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
