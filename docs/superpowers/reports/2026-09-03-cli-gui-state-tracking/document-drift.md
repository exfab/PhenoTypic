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

**Counting this register takes two queries, and one of them nests.** Entries run in a
single sequence across two shapes: **rows in the table below** (1-20) and **`### Entry N`
sections after it** (14, 19 and 20 expand their own rows; 21 onward are narrative-only,
plus one sub-entry, 22b). **Two further numbered tables are not part of this sequence**:
the dismissals (1/2/3) inside entry 23's body, and the broken-mutation table (1/2) inside
entry 44's.

**No literal counts appear below, and their absence is deliberate — see entry 53.** Every
number this header has ever carried went stale, including one that went stale *in the edit
that corrected the previous one*, because a count of this file is invalidated by the act of
writing the count into this file. What follows says why each naive query is wrong
**structurally**, which does not rot:

* `grep -cE '^\| [0-9]+ \|'` **over-counts.** It matches the register's own rows *plus*
  every row of the two nested tables.
* `grep -cE '^### Entry [0-9]+'` **under-counts.** Entries 1-13 and 15-18 have no narrative
  section at all, so they are invisible to it — and it folds `22b` into `22`.

Run the command below; do not read a total out of this prose.

**Use this, and read its limit below. Two earlier commands here were wrong.**

```bash
{ grep -oE '^### Entry [0-9]+[a-z]?' document-drift.md
  awk '/^## The register$/{f=1;next} /^## /{f=0} f' document-drift.md \
    | grep -oE '^\| [0-9]+ \|'
} | grep -oE '[0-9]+' | sort -n | uniq
```

Two things it fixes. The `awk` **scopes table rows to the register section**, so no nested
table can reach them. The `[a-z]?` makes suffixed headings visible: without it `### Entry
22b` matches as plain `22` and folds into entry 22 silently -- and 22b **exists today**, so
that was a live blind spot rather than a hypothetical one.

**Read the whole list, not just the last line.** It prints gapless `1..N`, and gaplessness
is the check worth running: it catches a skipped or duplicated number, which is the failure
this header exists to prevent, and no `tail -1` can perform it.

**Its limit, stated because the output does not show it.** It answers *"what number comes
next"*. It does **not** answer *"how many entries are there"*: 22 and 22b are two entries
sharing one number, and `uniq` folds them, so the true count is one higher than the list is
long. A future author reaching for this to get a total will be off by one -- which is this
file's own subject, one level up.

The command that stood here until 2026-09-08 also matched `^| N |`, so it counted rows of
the two nested tables as entries. It returned the right answer only because the register
happened to be the longest numbered sequence in the file -- see **entry 45**, and **entry
40**, which is the same defect in the same apparatus.

Entry 40 records what that cost. **Before quoting a total from this file, run the command.**


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

### Entry 39 — A SENTINEL RETARGETED ONTO AN OBJECT THAT NO LONGER CONTAINS IT. 2026-09-08.

`test_finalizer_does_not_publish_after_master_parquet_failure` was rewritten by P4 Task 4
for D8's inversion -- the master CSV was required and the Parquet best-effort; now the
Parquet is the master, so its failure must stop finalization. The rewrite is **right about
the inversion**. It is wrong about the object it retargeted the test's sentinel onto.

Before:

```python
assert pl.read_csv(master_measurements_csv_path(output_dir))[
    "Size_Area"
].to_list() == [999999]
```

After:

```python
assert 999999 not in pl.read_parquet(master_path)["Size_Area"].to_list()
```

**`Size_Area` was real in the first and cannot exist in the second**, and neither fact is
about the column. Pre-D8, `_write_master_outputs_from_shards` wrote the master CSV *from
the shards* before attempting the Parquet, so that CSV **was** the shard concat -- and the
fixture's shard (`_write_parquet(..., [999999])`) has exactly one column, `Size_Area`.
Post-D8 the CSV is deleted and the Parquet write is the one the test blocks, so the only
master on disk afterwards is the fixture's own, produced by a pipeline whose measurers are
`MeasureShape`/`MeasureIntensity`/`MeasureTexture`/`MeasureColor` and which therefore
carries no `Size_*` column at all. `ColumnNotFoundError`, not a failed assertion.

**Kind: wrong while correcting.** The amendment carries the authority of a fix, is correct
about the thing it set out to fix, and is wrong about a name it carried across unexamined.

#### The generalization

> **A column reference is a claim about a specific frame. Retargeting the read retargets
> the claim.** When an edit changes *which file* an assertion opens, every name inside it
> has to be re-checked against the new file -- even though not one character of the name
> changed, which is exactly why the diff does not look like a place to check.

Root `CLAUDE.md` already carries this rule ("Ask the schema for the spelling, then assert
the column is in the frame ... Spellability and presence are different questions, and only
presence is a property of the run") and uses **`Size_Area` as its own worked example**. The
rule was documented, in this repository, against this column, and was not applied.

#### The half a gate caught, and the half it could not

This is the first entry in the register that a gate reported, so it is worth being exact
about which part the gate saw. It saw the loud half: the column does not exist, the test
errors, the suite is red.

**It could not have seen the larger half, which is that the sign flipped.** `== [999999]`
was a *positive control*: it proved the shards had been merged and written. `not in` is a
negative, and negatives of that shape are satisfied by a run that merged nothing at all --
an empty shard glob, a frame that came back `None`, a finalizer that raised earlier. Had
the rewrite happened to name a column the fixture *does* emit, the test would have been
green, and green for a run that never reached the write it exists to describe.

So the summary above -- *"Nothing failed, and nothing could have"* -- still holds of the
register-relevant defect here. The two halves share one cause and are recorded together.

D8 leaves no successfully-written artifact carrying the shard concat, so the control could
not be restored where it was. It was rebuilt on the one thing the inversion does leave
behind: the write **attempt**, recorded by the fault injector and asserted before the two
"nothing changed" equalities. That assertion fails for the empty-shard case, which no
reading of the master can detect.

---

### Entry 40 — A REGISTER WHOSE OWN SIZE NO SINGLE QUERY RETURNS. 2026-09-08.

The brief handing P4 cluster 4.2 to a fresh agent said this register kept **"23 entries"**.
It keeps 38. Nobody had counted; the number came from `grep -c '^| [0-9]'`, which spans the
register table *and* the three-row dismissals table nested inside entry 23's body.

The reply proposed **21** as the next index, from
`git show cd0ffb82 -- document-drift.md | grep '^+| [0-9]'` returning nothing — reasoning
that a commit message saying *"and drift entry 21"* had named an entry it never wrote, and
flagging that as a register candidate in its own right.

**It wrote one.** The grep was shaped for table rows; entry 21 is a narrative section:

```
$ git show cd0ffb82 -- document-drift.md | grep -E '^\+### Entry'
+### Entry 21 — A TOOL'S OUTPUT IS A SAMPLE TOO. 2026-09-05.
```

`cd0ffb82`'s message is accurate. Appending at 21 would have **duplicated a live entry** —
the one whose subject is that a tool's output is a sample.

#### The measurement, and the coincidence that hid it

| Query | Returns | What it actually counts |
|---|---|---|
| `grep -cE '^\| [0-9]+ \|'` | **23** | 20 register rows **+ 3 nested dismissal rows** |
| `grep -cE '^### Entry [0-9]+'` | **23** | only the sections; entries 1-13 and 15-18 have none |
| max index across both shapes | **39** | the answer |

Two structurally unrelated wrong queries returning the same number is luck, not
corroboration — the second only reached 23 once entry 39 was appended. Had either party
cross-checked the other's 23 with their own grep, it would have agreed, and both would have
been wrong together.

#### Why this is not drift in a claim

No sentence in this register is false. Every entry is correct, and so is `cd0ffb82`'s
message. The defect is in the **instrument**: a document that collects claims about totals,
whose own total is not obtainable by any query its structure invites, and which offers two
plausible queries that both mislead.

That is the register's most common pattern — *a claim about what a check does* — one level
up, as **a claim about what a count counts**. Nearest kind: **true but incomplete**. The
omitted consequence is that the file's size cannot be read off it, so anyone who needs the
number derives it, and derives it wrong.

#### Both directions of the error appeared, which is the useful part

The two failures are opposites and neither is carelessness:

* **Over-count by nesting** (23 for 20) — a query that reaches past the structure it meant
  to sample, into a table that belongs to an entry rather than to the register.
* **Under-count by shape** (0 for 1) — a query that samples the right region in the wrong
  form, and reports absence for something present.

The second is the dangerous one, because absence reads as a finding. It produced a
confident, well-evidenced, wrong hypothesis — *a commit claimed an entry it never wrote* —
which is exactly the shape entry 38 records under a different mechanism. **A `grep` that
returns nothing is a claim about the pattern, not about the file.**

#### Remedy, applied

The header of `## The register` now states the two shapes, names the nested table, and
carries the command that returns the highest index. It deliberately does **not** record the
count itself: a number written into a file that grows is the same defect one turn later.

---

### Entry 41 — A COMMENT THAT ASSERTS ITS OWN COMMIT'S CO-CHANGE. 2026-09-08.

`_cli_completion.py:238-242`, written by P3 (`1cc6740c`) to justify D1's clean break:

> D1 is a CLEAN BREAK: the record replaces `image_complete/`, and nothing dual-writes. A
> tree carrying `image_complete/` and no `images/` is a legacy tree, which `--mode migrate`
> converts and every writing mode now refuses — **which is why `SCHEMA_GATE_ARMED` flips in
> this same commit.** A dual write would leave the gate unable to tell the two shapes apart.

```
$ grep -n 'SCHEMA_GATE_ARMED: bool' src/phenotypic/sdk_/_schema_shape.py
153:SCHEMA_GATE_ARMED: bool = False
$ git log --oneline -S 'SCHEMA_GATE_ARMED: bool' -- src/phenotypic/sdk_/_schema_shape.py
17f144ef feat(sdk): resolve_run_state -- the one reader, completing Phase 1
```

The flag's value was last changed in **P1**, and is still `False`. `1cc6740c` did not touch
it. **Kind: never true** — the sentence is about the commit that contains it, so unlike a
pointer to a name that arrives one task later, there is no later moment at which it becomes
true. It was false when written and false in the same breath.

Three other comments get it right — `_cli_recompile_recovery.py:71`, `_cli_finalize_run.py:94`
and `_cli_migrate.py:719` all name P7 Task 5 Step 1d as the arming site. One comment out of
four, and the outlier is the one that states it as accomplished rather than scheduled.

#### What it would have cost

Not the clean break, which is correct — it is justified by P7 arming the gate *later*, not
by this commit. The damage is downstream of the tense.

**An armed gate refuses legacy trees before they reach any writing mode. That is precisely
the condition under which `_image_authority_shapes`' second arm is dead code** — and that
function's own docstring says so, naming the same P7 step as its deletion trigger. So a
reader who believes this comment is told, four files away, that a live and load-bearing
legacy arm is unreachable. P4 repointed five production sites onto exactly that pairing; a
maintainer acting on the false half would have deleted the arm as unreachable and made
`--mode recompile` silently non-functional on the legacy trees it exists to rescue.

The falsehood is one clause. What it contradicts is a whole compatibility surface.

#### The mechanism: a ruling that revised one site and not the other

The disarming was deliberate. `tests/unit/cli/test_image_record.py:769-778` records it in
detail — a test that had opened with `assert SCHEMA_GATE_ARMED is True` was renamed and its
guard removed, because *"the ruling that disarmed the gate for P3 would have failed that
guard and renamed nothing."* That site was revised carefully, with reasoning. This one, in
a different file, was not touched.

> **A ruling that changes a plan has to be applied to every sentence written under the old
> plan, and nothing enumerates those sentences.** The elaborate revision at one site is not
> evidence the sweep happened; it is evidence that whoever revised it was looking at that
> file.

#### Resolved

The comment is corrected in place, in the same commit as this entry: it now states the flag
is `False`, cites the three sites that name P7, and says explicitly that the legacy arm is
live because of it — so the next reader meets the compensating code rather than a reason to
delete it.

---

### Entry 42 — A DELETION'S BLAST RADIUS IS NOT THE SET OF SITES THAT NAME IT. 2026-09-08.

D8 deleted `master_measurements_csv_path`. P4 Task 4 Step 4 prescribes the sweep that finds
what breaks:

```bash
grep -rn "MASTER_MEASUREMENTS_CSV\|master_measurements_csv_path\|load_master_measurements\|master_csv" src/
```

and the same step counts the test side (`phase-4-finalize-run.md:2314`, reporting Q6):
*"ten test files reference `master_measurements_csv_path` — **verified exactly ten**."*
Both are correct. Both are also **blind to the largest affected cluster in the change**,
which contains not one occurrence of any deleted name:

```python
result = aggregate_measurements(...)      # returns the master path
master = pd.read_csv(result)              # <- 11 of these, in one class
```

`aggregate_measurements`' return **type** changed from a CSV path to a Parquet path. Every
caller holding that value is affected, and not one of them mentions a symbol the grep can
see. The execution bore this out exactly: `tests/unit/cli/test_cli_v2.py` shows the four
sites that *spell* `master_measurements_csv_path(...)` correctly repointed, and all eleven
that spell `result` untouched — one file, one editing session, a clean split along
grep-ability.

> **A symbol deletion's blast radius is not the set of sites that name the symbol. It is
> every site that holds its value.** A return value is an alias, and an alias carries no
> name for the grep the deletion invites you to run.

#### It propagated into two scope claims, neither of them careless

| Claim | Basis | Missed |
|---|---|---|
| *"8 failures"* | the two files that had been run | a 9th site, `test_embedded_measurement_recompile.py:97` |
| *"the affected suites"* | the three files known to be red | these 11, in a file neither party had run |

Both were honest reports of what had been executed. Neither was the change's blast radius,
and the gap between those two things is the entry. **The same root as entry 40:** a total
asserted from a query that covers part of its subject. There it was a register's own size;
here it is a deletion's reach. In both cases the query was well-formed and the answer was
right about what it measured.

#### What actually closes it

Not a better grep — no lexical query can see an alias. Two things do:

1. **Run the suite that exercises the function, not the suite that names the symbol.** The
   eleven surfaced the moment `tests/unit/cli` was run whole instead of by named file.
2. **Treat a changed return type as a changed contract and enumerate the callers**, which
   is a different question from enumerating the references. Here that is one line and the
   set is closed: `grep -c 'pd.read_csv(result)'` returns **11**, matching the 11 failures
   exactly.

**Kind: true but incomplete.** The Step 4 grep and Q6's "exactly ten" are both accurate
about what they measured. The omitted consequence is that their shape cannot reach a
value-carrying call site — and nothing fails to say so, because a grep that misses a site
returns a smaller number, not an error.

#### Audited, not assumed

Before repointing, each of the eleven was checked for the entry-39 defect — a positive
control silently lost when a read is retargeted. **None had one.** Nine assert master
*content* (row counts, plate sets, column presence) and those claims are format-independent;
the row-level claims in that class live on the **mirror**, which is genuinely CSV and was
never affected. A mechanical format swap is the right fix here, which is only knowable
by having asked.

One pre-existing gap surfaced while checking: `test_aggregate_measurements_no_dataset_column`
asserted a bare negative with nothing establishing the frame had content, while its own
sibling `..._metadata_no_common_columns` does it correctly. Fixed in passing, and **not**
given a register entry — it is a missing assertion, not a false claim, and inflating the
register with those would cost it the property that makes it worth reading.

---

### Entry 43 — A DEBT STATED CORRECTLY, IN A FORM NOTHING CAN SCHEDULE. 2026-09-08.

**Kind: true but incomplete.** Same root as 40 and 42, one level out: not a claim about
what a check measured, but a claim about *when work will happen*, written so that nothing
can act on it.

`_refuse_inverted_store` (`_cli_recompile_tables.py:87`) is a compatibility guard P4
added. Its retirement condition, at `:90-93`, is accurate:

> *"Delete it when the recompile repoint lands — that is, when
> `recompile_embedded_measurement_tables` builds its payload with `prepare_image_tables`."*

Every word is true. Nothing can act on it. It names a **code condition**, and no phase
gate, test, or review step ever asks "is this condition now satisfiable?" Compare the two
sibling arms the same phase added, which say *"DELETE WHEN: the schema gate is armed
(P7 Task 5 Step 1d)"* — a **scheduled event**, tied to a step that exists and that
someone will execute.

**The consequence was a shipped mode refusing its ordinary case.** `--mode recompile`
raises on every forward tree built with `--metadata`, reproduced end to end:
`store tables: ['measurements','metadata']` then `exit=1`. The guard is the right
loud-over-silent call; it was simply never made unnecessary. The plan **had** scheduled
the repoint (`phase-4-finalize-run.md:232`, `:346` — *"→ `PreparedImageTables`. **This is
the crash**"*), but the producer was correctly retained for migrate, so the isinstance
check the plan expected to fire never fired, the guard went in instead, and the schedule
silently lost its subject.

**Why no gate caught it.** The phase's own §7.4 test,
`test_every_mode_produces_a_byte_identical_master` (`test_finalize_run.py:474`), runs its
`recompile` arm with **no metadata snapshot** — deliberately, and for a *correct* reason
about the master's shape (`:437-439`). The side effect is that the headline claim about
recompile is established on the only tree shape where recompile still works. A correct
local decision removed the phase's one chance to see it.

**And the guard had a destructive variant.** It is checked per store, inside the loop.
On a uniform `--metadata` tree it fails on the first store having written nothing; on a
**mixed** tree it rewrites the un-inverted stores first, then aborts — the mixed
generations the function's own docstring warns about. Ordering within a store was right;
scope across stores was not.

**Resolved 2026-09-08 by user ruling:** document the limitation where a user meets it,
add a whole-run pre-flight scan so the refusal cannot be partial, and schedule the
repoint as **P7 Task 5 Step 1e** — a step, with a number, in the phase that owns the
other retirements. Deliberately *not* folded into Step 1d: arming the schema gate does
nothing for an inverted store, because an inverted store is a **forward** tree. Tying it
there would have produced a fourth instance of this same defect — a trigger that cannot
fire for the thing attached to it.

**The rule.** A retirement condition must name **an event someone will execute**, not a
state someone might notice. "Delete when X is true" schedules nothing. "Delete at
P7 Task 5 Step 1e" is work.

*This entry may deserve its own kind rather than sitting under "true but incomplete" —
the register's four kinds are all about claims regarding the tree, and this is a claim
about the future. Left as-is rather than minting a fifth kind unilaterally; a reviewer
should decide.*

### Entry 44 — A MUTATION THAT LOOKS DIFFERENT AND BEHAVES IDENTICALLY. 2026-09-08.

**Kind: never true** — of the mutation's own claim about itself. Not drift in a document;
drift between what a mutation is labelled and what it does. Recorded here because the
register's subject is claims nothing checks, and **a mutation is the one artifact whose
entire purpose is to be checked, which is what makes a broken one so quiet.**

Two occurred in this change, four days apart, and both passed review because the mutated
line plainly differs from the original.

| # | Labelled | Actually did | Detected by |
|---|---|---|---|
| 1 | *"an existing active fence is RE-DATED"* | set a key on the in-memory dict the early return returns; **nothing writes it**, and the fence reads the file | `NOT PROVED` against a correct test |
| 2 | *"the scan becomes a per-store guard"* (M2) | moved the call inside the loop, but the scan takes the **whole** `authorized` mapping, so iteration one still scans everything and raises before any write | `PROVED`-shaped **pass**, which read as the test being vacuous |

**The second is the dangerous polarity.** A broken mutation that comes back `NOT PROVED`
sends you to look at a test; you find the test is fine and eventually find the mutation.
A broken mutation that **passes** tells you the test is worthless — and the author had
pre-registered exactly that reading (*"if M2 passes, my test is not testing ordering"*).
Accepting it would have deleted a working ordering test on the strength of a mutation
that could not have failed.

**The generalisable form**, due to the author of the mutation that failed:

> A mutation that narrows a function's **input** to something the function's own
> signature already tolerates is a no-op. The scope that has to change is the
> **caller's**, not the callee's.

`_refuse_inverted_stores_before_any_write({table_path: dataset})` is the mutation that
works, because it changes what the caller *asks about*. Moving the call site alone changes
only when a whole-tree question is asked, and the answer to a whole-tree question does not
depend on when you ask it.

**The rule this yields, and it is the counterpart to the one this register already
carries about checks:** *what would this have looked like if it had failed?* has a twin
for mutations — **what would this have looked like if it had done nothing?** If a
mutation's pass and its no-op are the same observation, it proves nothing either way, and
it will sit in a harness reading as a proved row indefinitely.

**Operationally:** for each mutation, state the expected red **and** the mechanism by
which the mutated code reaches the assertion differently. A mutation whose mechanism
cannot be stated in one sentence is usually not modelling the bug it names.

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
2. **A claim about what a check does** (10, 12, 13, 21, 22, 27, 28, 30, 31, 32, 33, 34, 36, 37, 40). The most dangerous, because
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

---

### Entry 45 — THE REGISTER'S OWN MEASURING COMMAND COUNTED OTHER TABLES' ROWS. 2026-09-08.

**Kind: never true**, and it is entry 40's defect in the apparatus entry 40 created to
prevent it. Recorded first among this batch because it is the one the next author uses
before writing anything else.

The header carried:

```bash
grep -oE '^(\| [0-9]+ \||### Entry [0-9]+)' document-drift.md \
  | grep -oE '[0-9]+' | sort -n | tail -1
```

It returns **44**, which is correct. It is also not measuring what it claims to. The `^| N |`
alternation matches numbered rows in **every** table in the document, and there are two
besides the register:

| Lines | Table | Values |
|---|---|---|
| 54-73 | the register's own rows | 1-20 |
| 590-592 | the dismissals, inside entry 23 | 1, 2, 3 |
| 1991-1992 | the broken-mutation table, inside entry 44 | 1, 2 |

Twenty-five matched rows for twenty register rows. **It returns the right answer only
because the register happens to be the longest numbered sequence in the file.** The first
time any nested table reaches a row number above the register's highest entry, the command
silently reports that other table's row and the next author numbers a duplicate.

**And there is a second blind spot, live rather than latent.** `### Entry [0-9]+` has no
right boundary, so it matches `### Entry 22` inside **`### Entry 22b`** (line 558) and folds
two entries into one number. That is true of the document today. Add `### Entry 44b`
tomorrow and neither the maximum nor the count moves. **The first replacement written for
this entry fixed the nested-table half and reproduced this half unchanged** -- the third
drift in this apparatus, inside the correction for the second.

**Two corrections to how this was first diagnosed, both worth keeping.**

The lines at 590-592 were read as a claim-vs-evidence table. They are the **dismissals**,
and the header already accounted for them — *"a third numbered table … lives inside entry
23's body"*. The genuinely unaccounted one is at **1991-1992**: entry 44's own M1/M2 table,
added on 2026-09-08. So the header's enumeration of nested tables went stale **the moment
entry 44 landed**, in the same commit, and neither the author nor the reviewer updated it.

That sharpens the entry rather than softening it. The register's counting apparatus has now
drifted **twice** — once in the command (this entry) and once in the prose that enumerates
what the command must exclude — and both drifts are in the machinery whose entire subject is
counting this file correctly.

**The fix, in the header where the command lives, not only here.** A register row that names
a broken command without replacing it leaves the next author with the broken one, which is
how entry 40 stayed live long enough to bite twice:

```bash
grep -oE '^### Entry [0-9]+' document-drift.md | grep -oE '[0-9]+' | sort -n | tail -1
```

The `awk` scopes table rows to the register section so no nested table reaches them, and
`[a-z]?` makes `22b` visible. **Read the full list rather than `tail -1`:** it is gapless
`1..N`, and gaplessness catches the skipped-or-duplicated number this header exists to
prevent, which no maximum can. The command answers *"what number comes next"* and **not**
*"how many entries"* — `uniq` folds 22 with 22b — and that limit is stated in the header,
because the output does not show it.

**The general form.** A counting command is a claim about a document's structure, and it
keeps being true only while that structure holds. This one was written against a file with
one numbered table and survived two more being added underneath it. *Ask of a counting
command what else its pattern matches* — not whether its answer looks right, because a wrong
pattern and a right answer coexist comfortably for as long as the wrong matches stay small.

---

### Entry 46 — A STATIC CHECK'S GREEN IS A STATEMENT ABOUT ITS OWN QUESTION. 2026-09-08.

**Kind: rule written without checking compliance.** Three instances inside one task, each a
tool reporting accurately on a question narrower than the one being asked of it — and in
every case the gap is invisible from the output.

| Check | Accurate about | Silent about | Evidence |
|---|---|---|---|
| `mypy` over `src/` | every shipped caller of a newly-required parameter | every caller under `tests/` | 2 production sites found; **8 test sites** missed, producing 15 failures and 5 errors |
| `ruff` F821 | whether a name is *bound* | whether `from X import Y` *resolves* | `All checks passed!` on a file where pytest then raised three `ImportError`s — same file, same batch |
| a `pytest` summary line | what was *classified* | what was *executed* | `566 deselected, 1 error in 11.82s`, accurate, on a suite that never started |

**flow-r4 specified the first as the mechanism** — *"Making it required puts the type checker
on the job of finding all three — which is the only mechanism here that cannot be
forgotten"* — and asserted its reach without stating what it does not reach. It **is** the
right mechanism: making `publish_aggregate_snapshot`'s `source_work_ids` required is how
`sdk_/_hdf_to_zarr.py:732`'s genuinely-different case surfaced (it wants the live set,
because migrate re-certifies deliverables over a tree it has just rewritten and there is no
master in that call for a proof to describe) instead of being papered over with the forward
path's answer. But *"green mypy means the change is complete"* is false, and the test call
sites outnumbered the production ones **4:1**.

The second is the tightest evidence in the register for this shape, because both signals
came from one batch over one file. Ruff's F821 is a **scope** analysis: `from X import Y`
binds `Y` unconditionally, whatever `X` contains. Resolving it means importing `X`, which a
linter does not do.

**The third is the nastiest, because no number in it is zero.** `566 deselected` is *true* —
those tests were deselected by `-m "not slow"` during the collection that then aborted. A
gate grepping for `failed` finds none; a gate parsing `N passed` finds nothing to compare.
Both read as clean. The only word carrying *nothing ran* is `Interrupted`.

**Operational rule, and it corrects a weaker one proposed first.** The harness guard is
`p4_finalize_run.py`'s **green-baseline refusal**, not a non-zero-executed-count check. The
executed-count guard would not have caught the `ImportError` case — twelve tests passed, so
the suite plainly executed. A green-baseline refusal catches that case and the
collection-abort case both.

---

### Entry 47 — A FILE-TABLE ROW CONTRADICTED BY A CONTRACT OLDER THAN THE PLAN. 2026-09-08.

**Kind: never true.** `phase-5-fanout.md`'s File Structure table:

> **Modify** `src/phenotypic/_cli/_cli_slurm_array_scripts.py:30` | Add the finalize trigger
> beside `_CHECKPOINT_SENTINEL` and `_MANIFEST_SENTINEL`.

Those sentinels are inserted into the **image-processing** array's entry list. The module's
own comment at `:28` says so — *"Sentinel value inserted into the image list"* — and
`_cli/CLAUDE.md:104-107` forbids the instruction in as many words:

> *"It also does not convert a terminal `afterany` finalizer into an array entry: a finalizer
> runs after the array becomes terminal and is not a parallel sidecar."*

That sentence predates the plan, which settles *never true* rather than *stale*.

**It is independently impossible**, contract aside: array indices run concurrently, so
aggregation of image *i*'s table cannot be ordered after image *i* within one array. No
arrangement of the row's instruction aggregates a complete run.

**The plan contradicted itself two sections later and nobody noticed.** Task 2 Step 1's own
test describes `[{"task_type": "measurements"} × K, {"task_type": "finalize"}]` — a
**task-dict manifest**, recompile's shape — while the file table describes a **bash
string-sentinel list**. Two incompatible designs in one task, and the plan even says *"The
shape already exists"* while pointing at the one its file table rejects.

**Resolved by ruling:** the fan-out is its own dependent array. `task_indices=[0]` on the
existing `pht-finalizer` becomes `range(K+1)`; line 30 is untouched. Pinned by
`test_the_image_array_entry_list_carries_no_finalize_token` and
`test_only_two_sentinels_are_defined_for_the_image_array`, so the rejected reading is a
guard rather than a decision in prose.

---

### Entry 48 — A MISCHARACTERISED MODULE, AND AN INFERENCE THAT INVERTS WITH IT. 2026-09-08.

**Kind: never true.** `phase-5-fanout.md` Task 2 Step 3:

> `_cli_checkpoint_handler.py` is the in-array `__PHENOTYPIC_CHECKPOINT__` dispatch — **the
> SLURM fan-out path itself** — […] So the shard-completeness check, as first written, does
> not reach the path most able to publish over a short master.

`__PHENOTYPIC_CHECKPOINT__` dispatches to `phenotypic._cli._cli_chunk_writer`
(`_cli_slurm_array_scripts.py:325-333`, consumed at `:353-357`) — byte-identical at the
plan-edit commit `ea2eb130` and at the merge-base `72f68b9b`, so never true rather than
stale. `_cli_checkpoint_handler`'s in-array half is `__PHENOTYPIC_MANIFEST__` →
`--checkpoint-type manifest` → `_run_manifest`, which publishes nothing.

**The citation being right is what let the characterisation survive.** Both publish sites the
plan names (`:354`, `:442`) are exactly where it says. They are inside `_run_finalize`,
reached only via `--checkpoint-type finalize` — the dependent `pht-finalizer`. A reader who
checks the citation finds what it promises and stops. Same shape as entry 23.

**What inverts.** The plan's conclusion describes the terminal `afterany` finalizer the
contract explicitly **permits** and that P5 replaces. So CAN-19's question is not *"what do
we do about a second publisher"* but *"this is the job we are replacing"*: keep it, make it
index K, and no site changes its publication behaviour. Confirmed by enumerating
`_run_finalize` — 165 lines, 29 calls, and already the run proof's publisher twice over, so
minting a new `TASK_FINALIZE` that published would have **created** the second publisher this
change exists to remove, through the fix for it.

**The seven-row table's value was in proving that, not in authorising the change it was
expected to authorise.** A reader who finds a seven-row table and a one-line outcome should
be told the table is why the outcome is safe.

---

### Entry 49 — P4 MOVED THE TREE UNDER FOUR SETS OF PLAN CITATIONS AT ONCE. 2026-09-08.

**Kind: stale**, uniformly — every one was correct at `ea2eb130`, the commit that last wrote
flow-r4 into `phase-5-fanout.md`. Established with `git show` against that commit and the
merge-base, not inferred.

| Plan claim | At `ea2eb130` | At `e15117e7` |
|---|---|---|
| *"three call sites in shipped code"* for `publish_aggregate_snapshot` | true | **two**: `_cli_finalize_run.py:435`, `sdk_/_hdf_to_zarr.py:732` |
| seven-site rows 1-2 (`_cli_output_manager.py:1540`, `_cli_recompile_worker.py:652`) | both real | both removed by P4; `_cli_recompile_worker.py:683` now carries a comment saying so |
| `phenotypicCLI.py:2395`, `:3726` | correct | `:2437`, `:3820` |
| `_cli_migrate.py:1220` | correct | `:1263` |
| `_cli_completion.py:905`, `:921`, `:922`, `:736-745` | correct, incl. `_canonical_digest` | `:1043`, `:1062`, `:1063`, `:789-797`; the name is now `canonical_digest` |
| `_cli_recompile_slurm_scripts.py:146`, `:198`, `:339` | — | `:145`, `:194`, `:334` |

**The seven-site table is also under-enumerated**, which is a different defect from being
stale: `_cli_sentinel.py:163` (deprecated but shipped) and `_cli_migrate.py:1044` both reach
`aggregate_measurements` and appear in no row.

**The count mattered and the conclusion did not.** flow-r4 argued the parameter must be
**required** because *"there are three call sites and only one of them is `finalize_run`"*.
At two sites, one of which **is** `finalize_run`, the arithmetic is wrong and the conclusion
holds unchanged — `sdk_/_hdf_to_zarr.py:732` is still a non-`finalize_run` caller, so an
optional parameter would still have left the defect alive there. Worth separating: an
argument can rot in its evidence and stand in its conclusion, and only re-checking tells you
which half moved.

---

### Entry 50 — A SYNCHRONISATION STEP THE PLAN NEEDED, CITED, AND NEVER BUDGETED. 2026-09-08.

**Kind: true but incomplete.** Task 2 Step 4 tells the implementer to reuse recompile's
`"expected_non_finalizer_tasks": len(tasks)` key, and says what for:

> **K comes from the task payload**, written at planning time. […] Reuse that key's shape
> rather than inventing one.

Every word true. Omitted: in recompile the same key drives a **blocking wait** —
`_wait_for_non_finalizer_statuses` (`_cli_recompile_worker.py:702-718`), a 5 s poll against a
deadline — and without it the check being specified cannot work at all.

Within one SLURM array, index K has no ordering relation to `0..K-1`; the scheduler starts
them together. So `TASK_FINALIZE` would find an incomplete shard set and raise CAN-5's
refusal on a run where **nothing is wrong** — not intermittently, but on essentially every
run.

**Why this is the invisible kind.** Nothing can fail for a sentence that was not written. The
key is cited and the precedent named; a reader who follows the citation lands in a file where
the wait is three functions away and never mentioned.

**And the measured K makes it worse.** S-2 puts K = 1 at the design target, so the array is
**two tasks** — the shape where a race is least likely to be caught by eye, and most likely
to be written off as flakiness if it ever is.

---

### Entry 51 — THE PLAN PRESCRIBED A WEAKER INSTRUMENT THAN THE TREE ALREADY USED. TWICE. 2026-09-08.

**Kind: never true**, of the implied claim that the prescribed instrument tests what it
names. Its own entry because it is a *shape*, and the second instance was found only because
the first had been.

**1. `fake_sbatch` versus the mandated chokepoint.** Task 2 Step 1 drafts a fixture capturing
raw `sbatch` argv. No such fixture exists; the convention is `unittest.mock.patch` on the
submission helper (`test_cli_recompile_slurm.py:96-113`). Argv-level is not merely
unidiomatic, it is **weaker**: it goes green on code that bypassed the drip-feed dispatcher
and shelled out — the failure `_cli/CLAUDE.md:120-124` records as having already happened
(*"eager submission is what caused the `AssocMaxSubmitJobLimit` failures"*). The chokepoint
catches the sidecar **and** the bypass; argv catches only the sidecar, and the bypass is the
one with a history.

**2. Two Task 1 tests that could not fail.**

- `test_the_finalize_trigger_is_counted_against_the_array_bound` used
  `n_images=1_000_000, seconds_per_image=1.0`, whose unclamped target is **1112** against a
  bound of 2499. `assert k <= 2499` was green with the clamp idle, and identically green
  against a `shard_count` containing no clamp at all.
- `test_max_array_size_caps_the_index_not_the_task_count` asserted
  `array_spec(k) == f"0-{k}"` — the implementation restated as its own expectation, true for
  every K including wrong ones.

**The near-miss, recorded because it is the mechanism.** Correcting (1), the first
replacement written patched `submit_slurm_script_chain` and then **called it directly** —
proving only that the test called the function it had just patched. Caught before review. A
prescription's weakness propagates into its correction unless the correction states what it
is *for*, which is why both replacement docstrings now carry their rejected alternatives.

**General form:** a plan that specifies an instrument as well as a subject has doubled what
needs checking, and the instrument is the half nobody re-derives. Ask of a prescribed test
the question this register asks of a check — *what would this have looked like if the code
were wrong?*

---

### Entry 52 — A JUSTIFICATION OFFERED BY A REVIEWER, ALMOST ACCEPTED BY ITS AUTHOR. 2026-09-08.

**Kind: rule written without checking compliance — caught during manufacture rather than
after it shipped.** Every other instance of this kind in the register was found downstream of
the sentence; this one was refused at the moment it was offered, which is the only reason
there is anything to record.

`_cli_finalize_fanout.py` imports `aggregation_shard_dir` inside function bodies rather than
at module scope. Three test call sites then imported it *from that module*, where it is not
an attribute, and the reviewer proposed two dispositions — repoint the tests, or add a
module-scope re-export — with this reasoning attached:

> *"the function-local imports look deliberate (import-cycle avoidance) […] but you wrote the
> module; if the local imports were **not** cycle-driven, say so."*

**They were not.** They were the file's dominant style plus a wish to keep the pure sizing
functions cheap to import. There is no cycle, and none was ever checked for.

Had the hypothesis been accepted — and it was flattering, and it was easy — the record would
carry *"cycle avoidance"* as the justification for a shape that is merely house style, and
the next person to touch the module would preserve a constraint that never existed. A
manufactured constraint is harder to remove than a real one, because removing it looks like
risking a cycle.

**The mechanism, which generalises past this file:** *an offered explanation is evidence
about the offerer, not about the code.* The reviewer's hypothesis was an inference from the
shape of the file, not a reading of it, and inference dressed as attribution is
indistinguishable from a finding once it is written down.

**The disposition did not depend on the answer**, which is the part that makes this cheap to
get right. The reviewer's own argument — a re-export makes the CLI module a **second address
for an `sdk_` symbol**, the same defect as the two `measurement_shards` paths that forced the
`aggregation_shards` rename three commits earlier — settles it without reference to cycles at
all. When a decision holds under both answers, resist supplying the answer.

---

### Entry 53 — A COUNT OF THIS FILE, INVALIDATED BY BEING WRITTEN INTO THIS FILE. 2026-09-08.

**Kind: wrong while correcting** — the kind this register's own table marks **Highest**
cost, because it carries the authority of a correction. It occurred **inside entry 45's
fix**, which is the edit whose entire subject is that this file's counting apparatus keeps
drifting.

Entry 45's rewrite of the header stated two counts as present-tense fact:

> `grep -cE '^\| [0-9]+ \|'` gives **25** … and `grep -cE '^### Entry [0-9]+'` gives **28**

Both were measured, and both were correct **when measured**. Then eight entries (45-52) were
appended to the file the measurement was about. Measured after:

```
grep -cE '^\| [0-9]+ \|'       -> 25     still correct
grep -cE '^### Entry [0-9]+'   -> 36     was written as 28
git show HEAD:…                -> 28     correct for the pre-append file
```

Eight new `### Entry N` headings is exactly the difference.

**Why this is not simply "a stale number", and the distinction is the entry.** `25` survived
and `28` did not, and nothing about how they were written distinguished them. The eight new
entries added no `^| N |` rows, so the first count was unaffected; they added eight
headings, so the second measured **its own publication**. The mechanism is narrower than
staleness:

> **A count of a file is invalidated by the act of writing the count into that file.** It is
> not that the number aged — it was false the moment the sentence containing it was saved,
> and it could not have been otherwise.

**The instruction against it was nine lines below and was obeyed once.** The header ends
*"Before quoting a total from this file, run the command."* The command was run — that is
where both numbers came from. What no instruction covered is that running it *before* the
append does not license quoting it *after*, and only one of the two numbers cared.

**This is the fourth drift in the same apparatus**, and the sequence is worth stating
because the trend is the finding: the original command (entry 40) → the command again
(entry 45) → the nested-table prose that said what the command must exclude (entry 45) →
this count, in the fix for all three. Each correction was written by someone who had just
finished reading about the previous one.

**The fix is removal, not a fresh number.** Correcting `28` to `36` would have been false
again the instant this entry was appended — 37 — which is the defect reproducing itself a
fifth time under the guise of repair. So the header now carries **no literal counts at
all**. It states why each naive query is wrong *structurally* — one over-counts because it
matches nested tables, one under-counts because entries 1-13 and 15-18 have no narrative
section — and structural claims do not rot. The only number in that section is produced by
running the command.

**The general form, for any document that describes itself:** a self-referential measurement
has no correct literal value, only a correct method. Where a count must appear, it is a
worked example of the method and must be labelled as one; where it appears as a fact, it is
false on save. The test is mechanical — *would appending to this file change this number?*
If yes, the number does not belong in the file.

---

### Entry 54 — A FAILURE MESSAGE THAT NAMES A CAUSE IT CANNOT OBSERVE. 2026-09-08.

**Kind: never true.** Pre-existing, not this change's to fix, and recorded because it is a
variant the register has no instance of yet.

`test_transition_fifo_evidence_is_rejected_without_blocking` fails under `-n auto` with:

```
Failed: FIFO receipt blocked recovery discovery
```

Nothing blocked. What the test measures is:

```
assert ready.is_file(), "FIFO recovery probe did not finish importing"   # PASSED
process.wait(timeout=1.0)                                                # TimeoutExpired
```

— *"the subprocess did not exit within 1.0 s"*. It cannot distinguish a blocking `open()` on
the FIFO from an interpreter descheduled under 16-way xdist load, and it reports the first
with certainty.

**The variant.** The register already carries *a check accurate about what it measured and
silent about what it could not reach* (10, 12, 13, 21, 27, 28, 30-34, 36, 37, 40, 46). Every
one of those under-reports. This one **over-reports**: the message names a **cause**, not a
symptom, so a reader who trusts it is pointed at FIFO handling and at
`_cli_recompile_recovery` — the wrong subsystem entirely. An under-reporting check wastes a
reader's time; an over-reporting one spends it in the wrong file.

**And it is invisible where anyone would look.** Serial runs never trip the 1.0 s window, so
the defect appears only under parallel load, which is where a reader is least likely to
suspect the assertion and most likely to suspect their own change.

**It nearly did exactly that.** The first hypothesis when this fired was that P5 Task 2
caused it, and the hypothesis was *specific* rather than vague: `_cli_completion` is a Task 2
file and **is** in the probe's import chain — 459 modules, 4.4 s cold. What refuted it was
going to the assertion rather than to the plausibility: `ready` is written **after** the
import completes, and `ready.is_file()` passed. So the import finished inside its 15 s budget
and import cost — the only channel by which a Task 2 file could reach this test — is
excluded. Three independent legs, none load-bearing alone: the import gate passed; it fails
only under load (2 of 3 loaded runs, 0 of 1 serial); and the diff touches neither
`_cli_recompile_recovery` nor any FIFO handling.

**The rule.** A message may name only what the check observed. *"The process did not exit
within 1.0 s"* is true and diagnostic; *"a FIFO blocked recovery discovery"* is an inference
presented as a measurement, and it survives precisely because it sounds like a finding. The
register's standing question — *what would this have looked like if it had failed?* — needs
its complement here: **what else would have looked exactly like this?**

**Owed, not fixed:** the fix is a message naming the timeout, and a timeout budget that
survives xdist load or a marker that excludes the test from it. Out of P5's scope.

---

### Entry 55 — THE HOUSE PATTERN WAS DOCUMENTED, AND NEW CODE REACHED PAST IT. 2026-09-08.

**Kind: rule written without checking compliance** — where the rule already existed, with
its reasoning recorded at the call site, and was never consulted. The cost was not a style
inconsistency: it was a **ten-minute hang that no gate in this plan could have reported.**

`run_local_aggregation_fanout` used `ProcessPoolExecutor`. On Linux that defaults to
**fork**, and `fork()` copies only the calling thread, so any lock held by another thread at
that instant is copied in the **held** state into a child where nothing releases it.
Measured at the ten-minute mark:

```
pid 3719060  pytest parent   state=S   threads=82   13 open pipes
pid 3727680  child           state=S   threads=1
pid 3727689  child           state=S   threads=1
```

Eighty-two threads — numpy/OpenMP pools, polars, pytest's own capture machinery holding
locks around I/O. Two children, both parked at start.

**The pattern that would have prevented it is three files away and explains itself.**
`_cli_overlay_rendering.py:165-180` is the same function shape:
`resolve_local_worker_count` → a `workers == 1` short-circuit → a **`ThreadPoolExecutor`**.
The new code already used the same helper and the same short-circuit *with the same
justification*, and diverged only at the executor. And the choice is documented, at
`phenotypicCLI.py:3156-3160`:

> *"Threading rather than multiprocessing because the heavy ops … all release the GIL, and
> per-image memory is large enough that fan-out to processes risks exhausting RAM."*

Both halves carry to aggregation shards, the second harder: polars releases the GIL and is
internally multithreaded, so processes buy nothing threads do not already have, while S-3
measured 2.5 GB peak RSS for N=6,529 in **one** process — N processes each materialising a
slice is precisely the multiplication that comment warns about.

**The search that would have found it was one `grep`**, on the helper the new code was
already calling. Before this change the project contained exactly **one**
`ProcessPoolExecutor` (`detect/_filfinder_detector.py:104`); every other pool in `src/`,
seven of them, is a `ThreadPoolExecutor`.

## The second finding: a hang is the absence of a statement

Recorded here rather than separately because it is what made the first one expensive.

A failing gate says `failed`. An aborted collection says `Interrupted` (entry 46). **A hang
says nothing** — no pass, no fail, no summary line — so every check that greps for `failed`
or parses `N passed` reads it as *no failures*. It cost ten minutes here; on a Slurm shard it
costs the whole wall clock and surfaces as a scheduler timeout with no failing test name.

**It defeats the guard settled two entries earlier.** Entry 46 adopted
`p4_finalize_run.py`'s **green-baseline refusal** as covering both a red baseline and an
empty collection. It covers neither case here: that refusal is evaluated on a run that
*completed*. **Two guards were specified — one against a wrong answer, one against an absent
answer — and neither covers an answer that never arrives.** Every gate in this change needs a
wall-clock bound as well as a verdict check.

**Three details of the fix, because only the first is the fix.**

1. **`ThreadPoolExecutor`**, matching the house pattern. The deadlock disappears rather than
   being configured around: there is no fork, so there is no inherited-lock hazard.
   `spawn` would also have worked, by paying a whole extra mechanism to avoid a hazard one
   can simply not create.
2. **A per-shard timeout naming the shard**, which converts a recurrence from a hang into a
   failure with a name. Not the fix — a correctness fix that leaves the failure mode silent
   has fixed one instance and none of the class.
3. **`pool.shutdown(wait=False, cancel_futures=True)` in a `finally`, not a `with` block.**
   A stuck *thread* cannot be killed, and `with` performs `shutdown(wait=True)` on exit —
   which would block on it forever and turn the timeout in (2) straight back into the hang
   it exists to prevent. The thread version has this hazard and the process version did not,
   so porting the fix without porting this would have preserved the symptom while appearing
   to remove it.

**The general form:** *a gate's contract includes terminating.* The register's standing
questions are now three — *what would this have looked like if it had failed?* (27), *what
else would have looked exactly like this?* (54), and **what would this have looked like if
it had never answered?** — because that last outcome is indistinguishable from success to
every output-parsing gate ever written.

**And the smaller lesson, which is the cheaper one:** before reaching for a more powerful
primitive, grep for the helper you are already calling. The pattern, its rationale, and the
reason not to use processes were all sitting at the other call site of
`resolve_local_worker_count`.

---

### Entry 56 — A DOCSTRING THAT JUSTIFIED THE DEFECT IT INTRODUCED. 2026-09-09.

**Kind: never true.** `shard_sources` assigned sources to shards by **stride**
(`index % K == shard_id`) and defended the choice in its own docstring:

> *"strided rather than blocked so an uneven tail does not land entirely on one worker."*

That is true of **naive** blocking — `chunk = ceil(n/K)`, remainder piled on the last shard.
It is false of a **balanced** block split (the first `n % K` shards take one extra), whose
spread is identical to the stride's. Measured across `(n, K)` of (3,2), (10,2), (10,3),
(10,4), (7,3) and (6529,8):

```
balance spread:  stride <= 1,  block <= 1        (identical)
merge ordered:   stride False, block True        (every case)
```

**So the stride bought nothing and cost an invariant.** The finalizer merges shards by sorted
filename, so a non-contiguous assignment makes the master's row order a function of K:

```
ten sources, merged in shard order
  K=1  ->  abcdefghij      K=3  ->  adgjbehcfi
  K=2  ->  acegibdfhj      K=4  ->  aeibfjcgdh
```

Two runs of identical data producing different masters, with `source_set_digest` certifying a
byte sequence that depended on the worker count.

**It was not a local-only defect, and the phase structure hid that.** The SLURM path merges by
the same sorted glob and used the same `shard_sources`, so it had the bug identically —
latent only because `shard_count` returns 1 below N ~= 34,600, which is above anything run so
far. It was introduced in Task 1/2 and found by Task 3's test, which was the first instrument
sensitive enough to see it. **A defect confined to one path by accident of scale is not a
defect of that path.**

**The register has no instance of this shape: two accurate statements whose conjunction is
false.** Every other entry here is a single claim that is wrong. Here, both ends were
individually defensible and each documented itself correctly:

| Site | Said | True? |
|---|---|---|
| `shard_sources` | strided assignment, balanced across workers | balanced: **yes** |
| `collect_shard_paths` | *"a re-run of identical inputs produces byte-identical master bytes"* | at a **fixed K**: yes |

`collect_shard_paths` is *true but incomplete* in the register's own taxonomy: it states the
determinism that holds and is silent on the axis along which it fails — identical inputs at a
**different K**. And that unmentioned axis is exactly the guarantee Task 3's test existed to
check. A reader auditing byte-identity would have read that sentence, found it accurate, and
stopped.

**Neither file was the place a reader would look for the other's assumption**, which is why
reading either alone could not find it. That is the mechanism, and it dictated the fix:
all three sites — the assignment, the ordering, and the concatenation in
`_cli_finalize_run.py` — now **name the dependency they rest on and point at each other**.
`collect_shard_paths` says in terms that its ordering means nothing if the assignment stops
being contiguous.

**Three further things made it nearly invisible, and the third is the transferable one.**

1. It contradicted the precedent this phase claims to generalise. Recompile's `_chunk_paths`
   (`_cli_recompile_slurm_scripts.py:662`) slices **contiguously**. The stride diverged from
   the very pattern it was written to reuse — the same shape as entry 55, one task later.
2. The docstring made the wrong choice look considered. A bare `%` would have invited the
   question; a stated rationale closes it. **A justification is not evidence, and a wrong one
   is worse than none** — it converts a reviewer's question into a settled matter.
3. **The end-to-end test caught it by luck of fixture size.** `test_local_fanout_produces_a_
   byte_identical_master` parametrizes `njobs in {1, 2, 8}` over **three** sources. `njobs=8`
   clamps to K=3, where the stride degenerates to one source per shard and *is* sorted;
   `njobs=1` is trivially sorted. So it passed at both ends and failed only in the middle. A
   fixture of two sources would have made `[2]` and `[8]` identical and hidden which knob
   mattered; a fixture of four would have failed both and looked like a different bug.

**The fix is structural, not restorative**, and the distinction was the ruling. The
alternative — keep the stride, sort after merging — also produces a correct master, but it
puts the guarantee in the **merge step**, so any other route that merges shards reintroduces
the bug. And another route exists: recompile hands its own shards to the same
`build_master_frame`. A balanced contiguous split makes *"shard order equals sorted order"*
true **by construction**, in the decomposition, where every merger inherits it.

**And the replacement test does not depend on fixture luck.** A pure-function grid over
`(n_sources, shards)` — 30 cases — asserts merged-order-equals-sorted-order, no drops or
duplicates, and a size spread of at most 1. The end-to-end test stays, but it is no longer
the only thing standing between this invariant and silence.

---

### Entry 57 — A PAGER THAT DESTROYED THE EXIT STATUS IT WAS ADDED TO READ. 2026-09-09.

**Kind: never true**, of what the shell idiom claims to report. It occurred **while running
the gate for entry 55**, whose subject is checks that report something other than what they
appear to.

Entry 55 established that every gate in this change needs a wall-clock bound, so command 1
was issued as:

```bash
timeout 300 uv run pytest … ; echo "exit=$?"
```

Run with a pager appended to see the output, it became:

```bash
timeout 300 uv run pytest … | tail -12 ; echo "exit=$?"
```

which reported **`exit=0`** while pytest had failed. `$?` after a pipeline is the **last**
command's status — `tail`'s — and `tail` succeeds whatever it is fed.

**The failure is silent and inverted.** It does not garble the status, it replaces it with a
success. The un-piped form reports `exit=1` correctly, so the check was sound as specified
and destroyed by a change made *to inspect its output* — the least suspicious kind of edit
there is.

**It is entry 21's shape at one more remove.** Entry 21 recorded an operator's own tooling
producing a sample that could only say one thing. This is the operator's tooling producing a
sample that says the **opposite** thing, introduced by a habit — piping to a pager — that has
no relationship to the property under test and no visible effect on the output.

**The rule, and it is mechanical:** `$?` reports the pipeline's last stage. If a status is
being read, nothing may follow the command; capture to a file and read the file instead, or
use `PIPESTATUS[0]`. Concretely, for any gate in this change:

```bash
timeout 300 <cmd> > /tmp/gate.log 2>&1; echo "exit=$?"; tail -12 /tmp/gate.log
```

The general form belongs beside entry 55's: a bound on a gate is only as good as the reading
of it, and **the reading is itself a check that can be wrong** — most easily by an edit whose
purpose is convenience rather than measurement.

---

### Entry 58 — THE INVOCATION DEGRADED AT THE HANDOVER, IN THE HALF NOBODY WAS BEING CAREFUL ABOUT. 2026-09-09.

**Kind: true but incomplete** — of a command sent as a reproducible artifact, which was
complete as a command and incomplete as an *invocation*.

Command 3 was sent as an exact string, deliberately, so it could be run verbatim. Run that
way it returned **`exit=134`** — SIGABRT:

```
........................................  [ 69%]
<no summary line, ever>
Extension modules: … PyQt6.QtCore, PyQt6.QtGui, PyQt6.QtWidgets, PyQt6.QtTest …
QT_QPA_PLATFORM=<unset>
```

Re-run identically with `QT_QPA_PLATFORM=offscreen`, it gave `5244 passed`. `tests/CLAUDE.md`
documents the failure precisely — *"a missing `QT_QPA_PLATFORM=offscreen` aborts the
interpreter at 79% with no summary"*.

**The mechanism is the round-trip protocol, not Qt.** The earlier run of the same file set
was green because the operator's shell carried `export QT_QPA_PLATFORM=offscreen`. When the
command was standardised into a message as *the* artifact, the environment did not travel
with it — **an environment is not part of a command string.** The invocation degraded exactly
at the handover, and the part that degraded was invisible in the part being scrutinised. Both
parties were reviewing the command; neither was reviewing the shell.

## Three no-verdict modes in two days, and each defeats a different guard

This is the third time in this change that a gate produced **no verdict**, and the set is
worth stating together because no one bound catches them all:

| Mode | What it emits | Which guard misses it | Why |
|---|---|---|---|
| Hang (entry 55) | nothing, forever | the green-baseline refusal | that refusal is evaluated on a run that **completed** |
| `$?` after a pipe (entry 57) | `exit=0` on a failure | any exit-status check | the status read is the pager's |
| SIGABRT (this entry) | partial dots, no summary | **`timeout`** | it did not hang, it **died** — 134, not 124 |

The `timeout 900` bound was correct and did not fire, because there was nothing to time out.

**The generalisation, and it is the actionable part:** all three guards are **negative** — they
detect the absence of a bad outcome. A **positive** check catches all three at once:

> Assert the gate stated what it did. Grep the summary line for a **non-zero pass count**,
> in addition to the exit status and the wall-clock bound.

A hang emits no summary; a piped failure emits a summary that disagrees with `exit=0`; a
SIGABRT emits no summary. One positive assertion covers what three negative ones do not,
which is the same lesson as entry 27's *"what would this have looked like if it had failed?"*
turned around: **ask what it would have looked like if it had succeeded, and require that.**

## Two operational rules

1. **`QT_QPA_PLATFORM=offscreen` goes inside any command naming `tests/gui`**, not in the
   shell that runs it. `regression_shard.sbatch` already exports it, which is why the phase
   gate would not have hit this — the defect is specific to the ad-hoc commands traded in
   messages, which are the ones with no file to carry an environment.
2. **A command sent for someone else to run is an invocation, not a command.** If it depends
   on an environment variable, a working directory, or a module load, those are part of the
   artifact and belong in the string.

---

### Entry 59 — A RENAME IS INVISIBLE TO EVERY OVERLAP CHECK THIS PLAN HAS. 2026-09-09.

**Kind: stale** in its symptom and **rule written without checking compliance** in its cause,
and the second is why it gets an entry of its own rather than joining 49.

`phase-5-fanout.md` Task 4's code block reads:

```python
from phenotypic.sdk_ import master_measurements_parquet_path, measurement_shard_dir
```

```
def measurement_shard_dir   ->  NOT IN THE TREE
def aggregation_shard_dir   ->  sdk_/_io_constants.py:2096
```

**Task 1 of the same phase renamed it** — commit `b8ef480f`, whose message is *"one name that
means one thing."* Transcribing Task 4's snippet produces an `ImportError` at collection.

**Entry 49 is a phase boundary; this is four tasks apart inside one document**, written by
the author of the rename, in a file that author had open. That is the difference worth
recording.

## Why nothing could have caught it

The plan's own machinery reasons about **files**, and a rename is a fact about **symbols**.

| Check | What it inspects | Sees a rename? |
|---|---|---|
| `dag.py`'s veto table | `Files:` blocks — path strings | **no** — Task 1 lists `sdk_/_io_constants.py`, Task 4 lists `tests/unit/cli/test_finalize_fanout.py`; no conflict, correctly |
| cluster assignment | shared files | **no** — same reason |
| `ruff` F821 | whether a name is *bound* | **no** — `from X import Y` binds `Y` regardless of whether it resolves (entry 46) |
| `mypy` over `src/` | shipped callers | **no** — the stale name is in a plan document, not in `src/` |

So a rename in Task 1 silently invalidated code in Task 4 and **every gate this change owns
reported green**, each correctly answering its own question. The rename was even reviewed —
it was ruled on explicitly, and the ruling included carrying the new name to the other
cluster's Task 4 *text*. What was not done was carrying it to the **snippet inside the phase
document the renamer was editing**.

## The rule

> **When a plan contains code, renaming a symbol is a plan edit as well as a source edit.**
> `Files:` blocks make a plan's *file* dependencies checkable and its *symbol* dependencies
> invisible, so the symbol half has no mechanism and must be done by hand at the moment of
> the rename — not deferred to whoever transcribes the snippet, who will meet it as an
> `ImportError` with no context.

The mechanical form, cheap enough to be worth doing every time:

```bash
grep -rn '<old_symbol>' docs/superpowers/plans/<change>/ docs/superpowers/specs/<change>/
```

**And treat every plan code block as pseudocode against a tree that has moved.** Task 4's
block also calls `_write_current_epoch_shards` and `_publish_six_successful_images`, neither
of which exists. A snippet in a plan is a statement of intent that was true when written; it
is not code, and the register now has two entries (49, 59) whose whole content is that
distinction being forgotten.

## The same instrument, a second blind spot — and it caught its own author within the hour

The `Files:` block cannot represent a **symbol**, which is the finding above. It also cannot
represent **work that produces no file**, and that one was demonstrated immediately.

Task 4's `Files:` block names exactly one path, `tests/unit/cli/test_finalize_fanout.py`. On
the strength of that, cluster 5.2 was described to the implementer as *"test-only, no sbatch,
no real dataset, no `slurm-job` routing"* — a correction to an earlier, more accurate
statement. **Task 4 Step 4 is "Phase gate — a real SLURM run", and Step 3 is a code change.**
Neither appears in the `Files:` block, because a submitted job produces no file the plan
tracks and Step 3's target is `src/`.

The implementer had begun sizing a `slurm-job` submission and stopped on the strength of the
correction; the retraction arrived before it cost anything, which is luck rather than a
property of the process.

**So the instrument has two blind spots of the same shape**, and the second was walked into
by the person who had just finished writing a paragraph about the first:

| `Files:` block cannot see | Consequence |
|---|---|
| a **symbol** rename | Task 1 invalidated Task 4's snippet; every gate green |
| **work producing no file** | a phase gate requiring a real SLURM run read as test-only |

**The rule generalises past both:** *a `Files:` block is an index of write targets, and scope
is not a function of write targets.* Reading scope off it is sound only for work whose entire
footprint is files that already exist. Everything else — renames, submissions, deletions,
schema arming, anything measured rather than written — is invisible there **by construction,
not by omission**, so no amount of care in maintaining the block would help. Read the task's
Steps, which are prose and therefore have to be read rather than queried.

---

### Entry 60 — A CORRECTION THAT MOVED THE VACUITY INSTEAD OF REMOVING IT. 2026-09-09.

**Kind: wrong while correcting** — Highest cost by this register's own table, because it
carries the authority of a fix. **Second instance today**, the first being the
`### Entry [0-9]+` boundary reproduced inside the correction for the nested-table match
(entry 45).

`phase-5-fanout.md` Task 4's `test_a_prior_epochs_shards_are_never_merged` carries a
docstring diagnosing its own predecessor:

> *"CAN-5: the first draft of this test called `finalize_run` with **NO** `shard_paths` — the
> local concat path, which never looks at a shard directory at all. It proved nothing about
> the only path where a prior epoch's shards could be merged. Pass the current epoch's shards
> explicitly, with the stale directory present."*

The diagnosis is exact. The prescribed fix is:

```python
finalize_run(tmp_path, dataset_names=["plate"], shard_paths=current)
assert "GHOST.tif" not in master["Metadata_ImageFile"].to_list()
```

**Which is vacuous for a different reason.** The ghost is planted under `"old-epoch"` and
`shard_paths=current` is passed **explicitly**, so the function never consults a shard
directory at all — it reads exactly the list it was handed. The test asserts that
`finalize_run` uses its own argument. It would pass against an implementation with no epoch
namespacing whatsoever.

**The vacuity moved from one axis to another:** the first draft never reached the shard path;
the correction reaches it but supplies the answer as a parameter. Neither exercises the
derivation — `resolve_finalizer_shard_inputs` deriving a shard set **from an epoch** — which
is the only mechanism by which a prior epoch's shards could ever be reached.

## Why this one is cheap to catch and was not caught

The tell is in the test's own body, one line apart: the ghost is written to a path built from
`"old-epoch"`, and the thing under test is handed a list built from `"new-epoch"`. **A test
whose fixture and whose subject are connected by nothing cannot be testing their
relationship.** Asking *what would this have looked like if the code were wrong?* — the
register's standing question, and it is quoted in the file this snippet lives in — answers
immediately: identical.

## Disposition, which is not "write it correctly"

The subject was **already covered**, by the right mechanism on each path, before this test
was considered:

| Test | Covers |
|---|---|
| `test_shards_are_namespaced_by_scheduler_epoch` | the SLURM half — distinct epochs give distinct directories, and `collect_shard_paths` globs one |
| `test_fanout_start_empties_a_prior_invocations_shards` | the local half, at `epoch=None`, where the namespace **cannot** carry it and clearing does |

Those two split the guarantee along the axis the design actually splits it on: namespacing on
SLURM, clearing locally. The plan's third test adds a case on a covered subject, and adds it
in the one form that asserts nothing.

**The general form, and it is the reason this is an entry rather than a deleted line:** a
correction inherits its predecessor's *frame*. The first draft asked "does the merge see the
ghost?" and answered on the wrong path; the correction kept the question and fixed the path,
without noticing that supplying `shard_paths` had made the question unaskable. **Ask what the
corrected test now depends on, not whether it addresses the stated defect** — those come
apart precisely when the correction is narrow enough to look obviously right.

---

### Entry 61 — A GREEN GATE IS A STATEMENT ABOUT THE WORKING TREE, NOT ABOUT THE COMMIT. 2026-09-09.

**Kind: true but incomplete** — of what a passing gate licenses. Distinct from entry 58,
which is an *invocation* losing part of itself at a handover; here the invocation was
complete and correct, and the **artifact** was a strict subset of what it verified.

`test_finalize_fanout.py` was committed at `3ae28b89` carrying **8 references** to
`finalizer_memory_advisory`, while `_cli_finalize_fanout.py` — which defines it — was left
uncommitted. **That commit could not pass its own tests.**

## The diagnostic detail, which is the whole entry

The gate run *before* the commit and the gate run *after* both reported exactly **70
passed**. The count did not move, and could not have: the advisory tests were already in the
file and already passing, because the **working tree** had the function. The commit was
partial; the verification was not wrong about anything it measured.

> A green gate is a statement about the working tree. It says nothing about whether the
> commit being made is self-consistent, and the two come apart precisely when staging is
> partial.

**Every guard built during this change watches the run, and all four were satisfied:**

| Guard | Entry | Saw this? |
|---|---|---|
| wall-clock bound | 55 | no — the run terminated normally |
| redirect before reading `$?` | 57 | no — the status was read correctly, and was 0 |
| environment inside the invocation | 58 | no — the invocation was complete |
| positive non-zero pass count | 58 | no — 70, correctly |

They are all checks on **the act of measuring**. None of them can observe that the thing
being committed is smaller than the thing that was measured, because that is not a property
of the measurement at all.

## The check that does catch it is different in kind

Verify **the commit**, not the run — resolve the symbols across `HEAD` rather than across the
working tree:

```bash
git show HEAD:<test_file>   | grep -c '<symbol>'   # references
git show HEAD:<source_file> | grep -c 'def <symbol>'  # definition
```

That is what diagnosed it and what would have prevented it. Cheaply generalised: **after
staging, ask whether the staged set is closed under the imports it introduces.**

## The proximate cause is one the process already forbids

`git status` and `git add` ran in a single block, and the file was extended between the gate
run and the commit — staging a file mid-edit, which is the discipline this change has
enforced all day, broken by the person enforcing it. That half is ordinary and would not
merit an entry.

**What merits the entry is that the ordinary mistake produced no signal.** A partial commit
is normally caught by the next test run — but only if the next run happens against a fresh
checkout. Against a working tree that still holds the missing file, it is invisible
indefinitely, and surfaces later as a bisect failure or a broken CI on a clean clone, far
from its cause.

Fixed by `--amend` rather than a follow-up: `3ae28b89` was never pushed, and a
bisect-hostile commit left in history is worse than a rewritten one.

---

### Entry 62 — A PRECONDITION ASSERTED ON A VARIABLE THAT CANNOT VARY. 2026-09-09.

**Kind: never true**, of the guard's implicit claim that it can fail. Recorded because of
where it happened: **specified, implemented, and reviewed inside a conversation whose subject
was guards that are green by construction**, in a script whose own header cites entry 56.

The P5 gate asserts *"exactly one finalizer role"*, which is valid only for a single-chunk
run — `AutonomousSLURMStrategy` records `job_ids[1]` as `role="finalizer"` **only** when
`has_initial_dispatcher` is false (`_cli_execution_strategies.py:1142`). Above one chunk that
row is recorded as `dispatcher` and no finalizer row exists at all. The precondition asked
for, and written, was:

```python
chunks = [r for r in roles if "chunk" in r]
if len(chunks) != 1:
    failures.append("the fixture grew past one chunk …")
```

**It can never fire.** The strategy records **only** `chunk-0` (`:1131-1135`), because later
chunk ids are assigned by the drip-feed dispatcher after each chunk completes — the comment
at `:1146-1150` says so. So `len(chunks)` is 1 whether the run chunked or not.

The correct discriminator is the **`dispatcher` role**, which appears exactly when
`has_initial_dispatcher` is true — the same fact, read from the side that varies.

## Why this one is worth an entry rather than a diff

**The reasoning was right and the variable was wrong.** Both parties correctly derived that
chunking invalidates the finalizer assertion, correctly located the branch, and then guarded
it by counting a quantity the same source lines say is always 1. Knowing the mechanism did
not prevent asserting on the wrong side of it.

**And the specify/implement split hid it.** One person asked for `len(chunks) == 1`; the
other implemented exactly that and reported it done; the flaw surfaced only on a third
reading. Neither step was careless — the implementer had no reason to doubt a precondition
handed down with its justification attached, and the specifier had no reason to re-derive an
instruction already accepted. **A specified guard arrives with its own authority**, which is
the same property entry 45's *wrong while correcting* has and the same reason it is expensive.

**The general form**, and it is the counterpart to this register's standing questions: for a
guard, ask not only *what would this look like if the code were wrong?* but **what input
would make this fire at all?** If the answer cannot be stated concretely, the guard is
decorative. Here the answer was *"a value the writer never writes"*.

This is the second vacuous guard in this phase — the first was Task 1's clamp test, whose
input never reached the clamp. Both were preconditions whose author reasoned correctly about
the mechanism and then asserted on a quantity that could not move.

---

### Entry 63 — AN EDIT BUILT FROM A STALE READ, CAUGHT ONLY BY ITS ANCHOR ASSERTION. 2026-09-09.

**Kind: never true** — of the edit's implicit claim that it is applying to the text it was
composed against. A near-miss rather than a defect that shipped, recorded because the thing
that stopped it was cheap, mechanical, and easily omitted.

While reviewing the P5 gate script, a reviewer read the file, composed a one-line move of its
`--array` observation block, and ran the edit. The anchor assertion aborted:

```
assert text.count(tail) == 1     ->     0
```

The file had been revised in the interval, in a **stronger** form than the edit being
composed — `nullglob` so an unmatched pattern is an empty array, an asserted script *count*
rather than mere existence, `ls -la` on failure, and a hard exit. The edit would have
replaced newer, better work with an older, weaker version **and then submitted it**, in a
session that had been enforcing "never edit a tree someone else is working in" all day.

## Why the anchor assertion is the whole of the defence

An edit built from a stale read is **indistinguishable from a correct edit** until it lands:
same author, same intent, same shape, and it applies cleanly if the anchor still happens to
match somewhere. Nothing about the act signals staleness. A `sed -i` with no count check
would have succeeded silently and reported success.

That is why the discipline is *assert the anchor count **before** writing*, not verify the
result after:

```python
n = text.count(old)
assert n == 1, f"anchor matched {n} times, refusing"
```

Both failure modes are covered by the same line. `0` means the text moved under you — this
entry. `>1` means the anchor is not unique — the failure entry 44's neighbourhood records,
where a 12-space anchor matched a 16-space copy in another target and silently disabled a
harness for two phases. **One assertion, two distinct disasters**, which is unusual value for
a single line.

## The pattern across this change

Every scripted edit in this phase asserted its anchor count, and the practice has now caught:
a four-way ambiguous import anchor, a would-be revert of newer work, and several
not-unique-enough anchors refused before writing. **None of those was found by review**; all
were found by a line that costs nothing and runs every time.

The generalisation is narrow and worth stating plainly: **a text edit is a claim about the
current contents of a file, and the only cheap way to check that claim is to count.**

---

### Entry 64 — A WALL-CLOCK QUOTED ACROSS A DECADE OF SILICON. 2026-09-09.

**Kind: true but incomplete** — of what a timing measurement is a measurement *of*.

A per-image cost was measured at **165 s** and used to size the P5 gate's wall. The gate ran
at **5.5 min/image — 2.0x slower**. Two explanations were offered and both were wrong:

| Offered | By | Wrong because |
|---|---|---|
| *"`GridImage` is heavier; `Image` will be faster, so the estimate is safe"* | the implementer | the pipeline config is not what dominates |
| *"per-task CPU differs, 16 vs 4"* | the reviewer | also config; also not what dominates |

Measured:

```
scontrol show node r31   ryzen,amd,milan          CPUTot=256   <- where 165 s was measured
scontrol show node c07   amd,abu_dhabi            CPUTot=64    <- where the gate landed

sinfo -p short -o %f  ->  abu_dhabi  broadwell  cascade  genoa  milan  rome  ryzen  amd  intel
```

**`short` spans nine feature classes**, from `abu_dhabi` (Opteron 6300, ~2012) to `genoa`
(Zen 4, ~2022). The measurement was taken on Zen 3 and spent on Opteron-era silicon. It was
not an over- or under-estimate of the same quantity; **it was a measurement of a different
machine**, quoted as though the allocation were the variable.

> A wall-clock figure is a property of a **(code, data, node)** triple. On a heterogeneous
> partition the third term is not yours to choose, so a timing quoted for a `short` job needs
> either the node class stated or a worst-case margin.

## Both explanations were config-shaped, and that is the interesting part

Neither party said anything false. `GridImage` *is* heavier than `Image`; 16 CPUs *are* more
than 4. Two people reasoned carefully about the terms they had varied and neither noticed the
term that had moved on its own — the scheduler chose it, silently, and nothing in either
invocation mentions a node.

This is entry 30's sample-ordering shape one level out: **the variable you did not set is
still a variable.**

## The answer was in the fixture's own state file, printed hours earlier

```json
"slurm_args": {"slurm_partition": "intel", "slurm_cpus_per_task": 6, "mem_gb": 16,
               "slurm_time": "01:30:00",
               "slurm_constraint": "broadwell|cascade|rome|milan|genoa"}
```

**The production run of this exact data pinned node generation and excluded `abu_dhabi`.**
That block was read aloud during this same work — to extract `image_type`, `nrows`, `ncols`,
`overlay_alpha` for continuation identity — and `slurm_constraint` was read past, because the
question in hand was identity rather than performance.

That is the third time in this phase that the answer was already in this fixture's own
recorded state or README and was skipped because it was filed under a different question: the
`.tiff` extension trap, the two mismatching pipeline digests, and now the node constraint.
**A fixture's recorded configuration is a set of preconditions on everything that touches it,
not a description of one past run** — and the failure mode is not carelessness but
*relevance filtering*: each field was read by someone looking for something else.

**Operationally:** a gate on `short` should carry `slurm_constraint` when its timing matters,
or state that it does not. This one did not, and its budget survived only because the inner
array is one image per task with ~3x headroom.

---

### Entry 65 — A TEST THAT FOUND A REAL GAP FOR A REASON THAT WAS FALSE. 2026-09-09.

**Kind: never true**, of the test's stated reason for being able to detect what it detected.
The gap is real, the detection was real, and **the mechanism named in the test's own docstring
had nothing to do with either.**

`test_an_unready_file_is_not_accepted_into_the_inventory` was written to run **without**
`--skip-validation`, with this justification:

> *"validation is the mechanism that keeps a half-written file out of the inventory … a
> version of this test that inherited that flag would pass on a build with no admission check
> at all."*

The reasoning is sound and the premise is false. Measured:

```
phenotypicCLI.py:2260   if not config.skip_validation:
    Step 1  validate_execution_config(config)   <- config only
    Step 2  validate_pipeline(...)              <- pipeline loading
help text:  "Skip pipeline validation (for advanced users)"
```

It never opens an input image. The test would have failed **identically** with the flag on,
so the opt-out could not change the outcome — entry 62's question (*what input would make
this fire?*) applied to a flag rather than an assertion, and the third instance of that shape
in this phase.

**It was reviewed and agreed to before it ran.** The author reasoned it out, the reviewer read
the reasoning, agreed with it, and neither checked what the flag gates. A justification that
is *internally* coherent recruits agreement without ever being tested against the code.

## The substantive finding is better than the one that was looked for

There is no admission check to skip, and its absence is **the composition of two deliberate,
individually-correct decisions** rather than an omission:

| Decision | Where | Why it is right alone |
|---|---|---|
| candidates are tested **by name, never by opening** | `_cli_directory_scanner.py:28-32` | *"reading a root `zarr.json` per entry would cost an open per file at 10k-image scale"*; unreadable input is left to fail *"later, loudly, in `imread`"* |
| `total` counts **every image the state claims** | `_cli_completion.py:710-730` | *"a run with a failed image reports `successful < total`"*, and `current_run_is_complete` requires equality |

Composed: a file admitted while still being copied can **never** succeed, `total` never
shrinks, and the run is **permanently incomplete**. The first decision defers the cost of
readability to processing; the second makes processing failure terminal for the whole run.
Each docstring is correct and neither mentions the other.

That is entry 56's shape — *two accurate statements whose conjunction is false* — now with a
third instance and a sharper form: here the two are not merely uncoordinated, they are
**both justified in writing, by different authors, against different cost models.**

## Disposition, and why not the obvious one

`xfail(strict=True)`, with the composition in the reason. **Not** implemented in P5:
admission checking is new product behaviour with its own failure modes — what counts as
unready? a size check races the writer; an `imread` probe reinstates exactly the per-file open
the scanner's docstring rejects on cost — and it belongs to whoever owns input handling.
`strict=True` so the day the behaviour appears, the marker's staleness is a failure rather
than a silently passing xpass.

**The transferable half is not the gap.** It is that the test would have found it either way:
a correct assertion detects a real defect regardless of whether its author understood why it
could. The corollary is uncomfortable and worth stating — **a passing test's stated rationale
is not evidence that the rationale is true**, and this one only got audited because the test
failed.

---

### Entry 66 — THE GATE'S CENTRAL ASSERTION NEVER TOUCHED THE THING IT NAMED. 2026-09-09.

**Kind: never true**, of what the assertion claimed to compare. **The most expensive instance
in this register**: it survived four author reads and three reviewer reads, was found only
because an *unrelated* string-parsing bug turned the run red, and every other assertion in
the gate passed.

The P5 Task 4 gate exists to establish one thing no unit test can: that the SLURM fan-out's
master is byte-identical to a local `--njobs 1` run. It reported:

```
masters byte-identical (194342 bytes)
```

**It compared a local fan-out to a local fan-out.**

## The chain, each link measured

```
_cli_types.py:259            remote_managed: bool = False
_cli_staged_slurm.py:680,760 remote_managed=True        <- the STAGED path only
AutonomousSLURMStrategy      never sets it
phenotypicCLI.py:2879        if results.remote_managed:  -> early exit NOT taken
phenotypicCLI.py:2977        aggregate_master_csv(..., njobs=config.n_jobs)
phenotypicCLI.py:1350        --njobs default = -1
```

So under `--wait` the ordinary SLURM path **aggregates in the submitting process**, and
`resolve_local_worker_count(-1, 2)` caps at the work count:

| arm | what actually ran | K |
|---|---|---|
| "SLURM" | in-process **local** fan-out | 2 |
| local | in-process local fan-out | 1 |

A real result — a genuine end-to-end confirmation of Task 3's K-independence through the real
CLI — and already covered by `test_local_fanout_produces_a_byte_identical_master`. The
dependent finalizer array ran **twelve seconds later**, wrote its own master, and was examined
by nothing; the `afterany` cleanup then deleted the tree while it was still writing.

## Three assumptions, each individually reasonable

1. **`--wait` waits for the run.** It waits for the image *chunks*. The dependent finalizer is
   submitted `afterany` and starts afterwards, so every assertion ran on a pre-finalization
   tree.
2. **A SLURM run does not aggregate locally.** True for the staged path, which is where
   `remote_managed` is set, and false for the ordinary one.
3. **A default flag is neutral.** `--njobs -1` is inert for a scheduler run — except that the
   in-process fall-through made it the *only* thing choosing K.

None is careless. Their conjunction made a gate that could not fail its own central claim.

## What made it invisible, and it is the transferable part

**Every guard this change built watches the measurement; none watches the subject.** The
wall-clock bound (55), the redirect before `$?` (57), the environment in the invocation (58),
the non-zero pass count (58) — all four were satisfied. The gate ran, terminated, reported,
and its numbers were true. It was measuring the wrong tree.

> A green assertion proves the *comparison* was performed. It says nothing about whether the
> operands are the things the assertion names, and nothing in the output can tell you.

The rule that follows is the one the fix implements: **assert the subject changed.** The gate
now records the master's mtime before the wait and requires it to have moved — so "we waited
and then read a master" can no longer be satisfied by the stale in-process one. That is
entry 58's positive-assertion rule pointed at the operand instead of the outcome.

## Disposition

Five fixes, none of them "move the assertion later": wait on the finalizer by job id from the
ledger; **assert the master's mtime moved**; treat an unexpanded `sacct` form (`12345_[0-1]`)
as *not yet dispatched* and retry rather than parsing it as an index set; keep the cleanup
edge but make it safe by not exiting until the submitted chain is terminal; and pin
`--njobs 1` on both arms so the only difference is the scheduler.

**The K=1 topology claim stands**, and was never at risk: `scontrol` showed
`ArrayTaskId=0, ArrayTaskId=1` on the live finalizer and `sacct` showed both terminal. That
went through the scheduler, not through the master.

## The `--wait` documentation gap, and what is NOT wrong

P5 amended `CLAUDE.md`'s *"the dependent finalizer is the sole publisher of aggregated outputs
and the completion marker"*. That sentence is **conditioned on `--wait` being absent** and is
true in that condition; the amendment inherits the condition and needed no retraction.

What it was, was **silent about the `--wait` case** — *true but incomplete* — and that silence
mattered more after P5 than before, because the finalizer now merges shards while the
in-process path concatenates. Investigated and found benign: both take
`.aggregate_publication.lock` (`_cli_output_manager.py:1528`) so they serialize, both read the
same authorized sources, and the master is a pure function of those. **And the
completion-marker half holds unconditionally** — `phenotypicCLI.py:2437` is guarded by
`if config.process_only_layer is not None` and `:3821` is `--mode recompile`, so neither is
reachable from the forward `--wait` path; only `_run_finalize` publishes it. Corrected by
addition, not rewrite.

**And the benign verdict is itself dated, which is the last trap in this entry.** *"They write
the same bytes"* is true **now** and was **false three commits ago**. The in-process path fans
out at K = worker count; the finalizer merges the scheduler's shards at its own K. Until
`87f933cb` made `shard_sources` a contiguous split, merge order depended on K — so two writers
holding the same lock over the same sources produced **different bytes**. The gate's own run is
the demonstration: local K=2 agreed with local K=1 only because the contiguous split had
already landed.

So the redundancy is safe *because* Task 3 put the ordering guarantee in the **decomposition**
rather than in the merge — the argument made for choosing contiguous over sort-after-merge,
*"where every merger inherits it"*. **The second merger was one nobody knew existed at the
time.** A design choice defended on a general principle turned out to be load-bearing for a
specific consumer discovered three commits later, and a revert would break it with nothing
named nearby to object.

---

### Entry 67 — A FENCE THAT COMPARES A VALUE TO ITSELF. 2026-09-09.

**Kind: never true.** Spec §11's consumer table gives `RunRegistry` local exit as
*"8-branch refusal tree → `resolve_run_state(deep)`; refusals become advisories"*, and
§11.1 lists *"`_local_completion_evidence_conflict`'s 8-branch tree"* under **Deleted**.
The replacement named cannot answer the question the site asks, and never could.

The site's question is **generation-fenced**: did *this GUI launch generation* publish?
The real fence is `_runs_registry.py:718`:

```python
        if marker.get("generation") != str(record.generation):
```

compared against the completion marker the CLI stamps at publication time
(`_cli_completion.py:1279-1280`, `"gui_record_generation"` / `"generation"`).

`RunState` does not carry that value. `_run_state.py:1021-1029` reads the marker's
`version`, `status` and `finalizer_succeeded` and nothing else; `run_proof_is_current`
compares the three digests plus `source_set_digest`; `_advisories` (`:1310-1347`) emits
only the schema-shape conversion note. In the whole of `sdk_`, `gui_record_generation`
exists **only as a key constant** — `_io_constants.py:2432,2462` — read by nothing in
the verdict path.

## The near-miss that makes it survivable

`RunIdentity.owner_generation` looks exactly like the missing fence. It is not:

```
_run_state.py:266   _owner_generation()  reads gui_launch_owner_path(...)["generation"]
_runs_registry:1380 _persist_record_locked WRITES "generation": str(record.generation)
                                           into that same file
```

So `state.identity.owner_generation == str(record.generation)` is the registry reading
back its own write and calling it evidence. **A fence that compares a value to itself is
not a weak fence — it is not a fence.** Converting the site would have accepted a
*previous* launch's proof as this launch's success, which two existing tests already
guard: `test_observe_local_zero_exit_rejects_preexisting_complete_manifest` and
`test_cross_timezone_future_manifest_cannot_satisfy_new_generation`.

**Never true rather than stale**: `RunState` has never carried the marker's generation,
so the row could not have worked on any tree at any point in this change.

## Two smaller errors in the same row

* **Twelve branches, not eight.** Counted mechanically in
  `_local_completion_evidence_conflict`: twelve refusal returns plus two `return None`.
  P6 Task 0's `_all_accepted_images_succeeded` conversion added the schema-3 arm's three.
  A change sized against "eight" is sized against a number that was already wrong.
* **`deep` where every neighbouring row says shallow.** A deep pass on every local
  process exit is a full artifact re-read. Affordable once per exit, and the opposite of
  the row directly above it.

## What would catch this class, and it is not "does the field exist"

Anyone checking *"does `RunState` carry a generation?"* finds `owner_generation` and
stops. The check that discriminates is one level further in: **who wrote the value the
field holds.** A field populated from a file the asking component itself writes cannot
be evidence for that component, however well it is named.

## Disposition

The fence belongs in `RunState`, and that is P1/P7 work — it cannot be done from inside
`_runs_registry.py`, which is why this row has sat unbuildable across two tasks. Two
options, both one-sided changes to `sdk_/_run_state.py`:

1. carry the marker's `gui_record_generation` on `RunIdentity`, beside `owner_generation`
   and outside `digest()` for the same reason `owner_generation` is; or
2. expose it as its own reader, next to `run_proof_is_current`.

Until then the registry's tree stays. P6 Task 5 shipped its D-2 half only.

---

### Entry 68 — A FIX AND ITS OWN REGRESSION IN ONE TASK. 2026-09-09.

**Kind: true but incomplete** — of P6 Task 5 Step 2, which is correct about what to
change and silent about what that change collides with two hundred lines above it.

Task 5 has two halves against one file. **Half one** (DEFERRED D-2) releases an owner
record whose process is provably dead, so a SIGKILLed GUI stops refusing its output
forever. **Half two** was to stop `observe_local_exit` forcing `status = "failed"` when
publication evidence cannot be verified.

The only honest replacement status is `"unknown"` — and `"unknown"` is **nonterminal**
(`run_status_is_nonterminal`: in `_RUN_STATUSES`, not in `_TERMINAL_STATUSES`). The
claim path has two guards, not one:

```python
_runs_registry.py:325   for existing in self._records.values():      # in-memory
                            if ... existing.status not in _TERMINAL_STATUSES:
                                raise RuntimeError(...)
_runs_registry.py:334   with exclusive_path_lock(...):               # durable
                            self._assert_output_claimable_locked(...)
```

Half one's repair lives in the **second**. The first runs before the lock and never
consults it. So half two would have made every unverifiable local exit block re-launch
for the rest of the session, with no release path — **the exact defect half one exists
to remove, reintroduced by its own task.**

## Why the sentence that welds them is the mechanism

The two halves share a task because they share a file, and the plan's justification
welds them into one clause: *"a refusal the user cannot act on is the bug; an advisory
they can read is the fix."* That is **true of half one** — the claim refusal genuinely
had no UI affordance to clear it — and it *reads* as describing half two. A reader who
accepts the sentence inherits the collision without ever seeing a second claim.

## No test would have caught it

Both halves' tests pass. Half one's fixtures construct a **fresh** `RunRegistry`, whose
`_records` is empty, so the in-memory guard never fires in any of them. The collision
appears only in a session that ran the failing run *and then retried it* — which no unit
test in this file does, because each builds its own registry.

**The guard that would have caught it is skipped by every fixture that constructs a new
object.** That is the transferable half: a guard reached only through accumulated
in-process state is invisible to a suite whose fixtures are all fresh.

## Disposition

Task 5 ships half one. **Half two is withdrawn, not deferred** — under Entry 67 there is
no correct version of it at this site, and the priced alternative is refused on the
record so that a later reader does not find a costed proposal with no verdict and
re-price it.

The alternative was: split the twelve refusals into *contradicts success* (stays
`failed`) and *cannot prove success* (becomes `unknown`), and teach the in-memory guard
the same liveness predicate so `unknown` stops blocking. **Refused** on three counts —
it flips eight existing tests, each carrying a deliberate docstring, so it is eight
*contract* changes to argue rather than eight fixtures to update; the split is policy
neither the spec nor the plan makes, invented inside the task that just proved the
spec's row for this consumer *never true*; and the user loses signal, because a run
that genuinely failed to publish would read `unknown`.

**The disposition for the underlying complaint is better text, not a different
status.** Each of the twelve strings already names the exact path whose evidence is
missing, unreadable or foreign. If a specific one proves unactionable in use, rewrite
that string. Two items owed, both recorded rather than done:

* **`_process_is_alive` to `sdk_`'s public surface.** The registry imports it privately
  from `phenotypic.sdk_._run_state` so that the ladder and the claim check cannot
  disagree about one pid. One line, held only because `sdk_/__init__.py` took two new
  public names today already.
* **`rehydrate_from_sandbox` persisting its downgrade.** Today it is `persist=False`
  (`:796`), which is why the in-memory downgrade never reaches the durable claim check.
  Persisting it requires `exclusive_path_lock` — the boot walk holds no lock and would
  otherwise race another GUI's `allocate`. Naming the lock requirement is the point of
  recording it.

---

### Entry 69 — A CORRECT VERDICT REACHED THROUGH AN ARGUMENT THAT DID NOT REACH IT. 2026-09-09.

**Kind: never true** — of the framing, not of the verdict. Recorded because the verdict
was right, which is exactly what makes this one hard to notice.

Executing P6 Task 5, I put a choice to the lead: **(a)** convert the local-exit site to
`resolve_run_state`, or **(b)** surface the refusal as an advisory without one. I framed
P6 Task 0's comment at `_runs_registry.py:606-617` as settling it against (a). The lead
accepted the framing and ruled (b).

**The framing was wrong.** Task 0's comment objects to swapping *one predicate* for
`.completion` **while keeping the tree below it** — its stated reason is that the branch
below would become dead code. Spec §11 asks for something else: delete the tree
entirely. Those are different operations, and Task 0's argument does not reach the
second.

The verdict survived anyway. (a) is unbuildable — for the reason in Entry 67, an
inexpressible generation fence, which **neither of us had at the time**.

## Why a right answer from a wrong argument is worth an entry

Had the lead ruled (a) on the framing offered, the objection given would not have been
the reason it failed. The real reason would have surfaced as two red tests *after* the
code was written, and the recorded rationale would have pointed at dead code rather than
at a missing field.

**A correct verdict reached through the wrong argument is indistinguishable from a lucky
one, and it fails the moment the argument is reused.** The next task that cites "Task 0
settled this" would inherit a scope the comment never had.

## What caught it, and it was not a check

Going to the spec — after (b) turned out to be *already implemented*, leaving nothing to
build and therefore nothing to justify. The collapse of the recommended path is what
forced a second look at the rejected one. No gate, no test, and no review step was
involved; it was the same "found by reading" property this whole register documents.

## The transferable rule

**When you quote an existing decision as settling a new question, check that the new
question is the same *operation* the decision was about.** A recorded objection is scoped
to what its author was looking at, and a comment left in code carries no marker for how
far its authority extends. This one was two sentences long and its scope was one
predicate; it was read as covering a whole function.

---

### Entry 70 — TWO INDEXES CONSULTED IN PLACE OF THE THINGS THEY INDEX. 2026-09-09.

**Kind: never true**, twice, of two different indexes — and the pair is the entry, because
neither half is interesting alone and together they are one shape.

**Half one — a stale read.** After a context compaction, the Read tool returned the
**git-HEAD** version of `src/phenotypic/gui/_snapshot_status.py` (101 lines, with
`_completion_evidence_status` intact) while disk held a 128-line rewrite by another
agent. No error, no staleness marker. `git diff` and `wc -l` through Bash both showed the
new file; the harness's file-state cache had survived the boundary and the disk had not.

It was caught **incidentally**: a grep for badge label strings, run to check whether
renaming one would break another agent's test, returned hits in a test file this session
had not written. Without that unrelated grep, the next act would have been to
re-implement an already-implemented file, and the second write would have silently
reverted whatever the first got right.

**Half two — an unwitnessed agent.** The lead dispatched two agents onto one task, having
consulted `dag.py`'s parallelism veto. `dag.py` derives that veto from the plan's `Files:`
blocks: it is an index of **the plan**, and says nothing about which agents are live. The
correction of the first collision arrived after the second agent had already implemented
the task.

## The shape they share

An index was consulted in place of the thing it indexes, and **neither index reported
staleness, because neither was observing.** A file cache is not the file. A file veto
derived from a plan is not a statement about a session.

## Why the first half is the expensive one

The tool that failed is the one whose entire job is to report the state of a file. Once
it can be silently wrong, **"I read the file" stops being evidence about the file** — and
every downstream judgement built on that read inherits the defect with nothing to mark
it. This is the same failure the change itself is about, one level up: state that is
*tracked* (a cache) standing in for state that is *checked* (a stat).

The second half is cheaper only because a person noticed within the hour.

## Disposition

* After a compaction, use a Bash read for **anything about to be edited**. Reserve the
  cached reader for orientation.
* An anchor assertion (`assert text.count(old) == 1`) catches the *edit* case and is
  already established practice here — see Entry 63. It does **not** catch this case,
  because composing a new file from scratch has no anchor to assert. The defence for
  writing is different in kind from the defence for editing, and only one of them
  existed.
* A file veto is a property of a plan. Whatever answers "which agents are live" has to be
  a property of the session, and today nothing is.

---

### Entry 71 — A CITATION THAT RESOLVED, ATTACHED TO A CLAIM ABOUT THE WRONG IDENTIFIER. 2026-09-09.

**Kind: wrong while correcting** — the highest-cost kind, and earned here: the
paragraph exists to correct a fence that compares a value to itself, and names the wrong
field while doing it.

P6 Task 8's new `gui/CLAUDE.md` section said the completion marker is the one the CLI
*"stamps with `gui_record_generation` at publication time, and
`RunRegistry._local_completion_evidence_conflict` is the only thing that reads it
(`shell/_runs_registry.py:718`)"*.

`:718` reads a different key. The marker carries **two**, written two lines apart
(`_cli_completion.py:1279-1280`):

```python
        "gui_record_generation": gui_record_generation,          # exact; None off-GUI
        "generation": gui_record_generation or execution_epoch,  # falls back
```

The fence compares `generation` — the **fallback** field. And `gui_record_generation` is
read by **nothing** in `src/`: every other occurrence is a writer argument or the
dashboard manifest, which is a different artifact.

## Why the citation check passed

`:718` is the right line, in the right function, in the right file. **A `file:line`
citation verifies a location; the claim attached to it can still be about the wrong
identifier at that location.** A check that resolves the citation cleanly never re-reads
the field name inside it.

So a citation check and a claim check are **two checks, not one** — and Task 8's own Step
4, *"each path and function name gets a `grep`"*, mandates only the first. It caught a
different error in the same pass (a symbol table listing 18 while its prose said 19) and
could not have caught this one.

## The variable is isolated: same author, same hour, different source

Entry 67 makes the same statement **correctly**, naming both keys. It was written with
the greps still on screen. The `gui/CLAUDE.md` paragraph was written later, from memory
of the concept — and the concept's name *is* `gui_record_generation`, because that is
what the design calls the idea.

**The wrong name is the right name for the thing one level up.** That is what lets this
class survive review: the identifier is not arbitrarily wrong, it is the name of the
concept the field implements, so a reader checking *"is there such a field?"* finds one
and stops. The two artifacts differed in nothing except whether the author was reading or
recalling.

## What the correction bought beyond accuracy

Being forced to say *which* field is read produced a claim the original paragraph did not
contain. A SLURM launch has no GUI generation, so `generation` holds `execution_epoch`,
the comparison against the registry record fails, and **"a scheduler launch cannot
satisfy a GUI generation" falls out of the fallback** rather than needing a rule of its
own. The imprecise version carried that behaviour as an unexplained assertion; the
precise version explains it.

## Disposition

Both edits applied, and the closing instruction rewritten: what `RunState` lacks is a
**reader** for either field, not the value — both are present on disk. As written it read
as though the exact field were missing from the marker.

The generalisation, for Step-4-style verification anywhere in this change: **verify the
identifier, not only the address.** For a claim shaped *"X reads Y at F:N"*, resolving
`F:N` proves only that `F:N` exists and concerns X. Reading the line is what proves Y.

---

### Entry 72 — A CHECK THAT RESOLVES AN ADDRESS AND NEVER READS WHAT IS AT IT. 2026-09-09.

**Kind: never true** — of the confidence a green result licenses. Three instances in one
hour, at three layers, and the entry is the shape rather than any of them.

**Instance 1 — the ledger gate.** `scripts/check_features_md.py` validates that every
`✅ shipping` row's `Test ref` **resolves**: the file exists and defines a test of that
name (`:164-180`). It does not run it. So `[features-check] OK (473 feature rows, 294
shipping)` is compatible with a shipping row pointing at a **failing** test — which is the
live state right now: P6-T1's Task 2 row cites
`test_mutation_guard.py::test_inconsistent_results_layout_keeps_views_and_disables_mutations`,
one of the failures that task is fixing. Both of us ran that gate green in the same hour
and neither run was evidence about the row's subject.

**Instance 2 — a `file:line` citation.** Entry 71: `:718` resolved, in the right function,
in the right file, and the claim attached to it named the wrong key.

**Instance 3 — a probe's own precondition.** The first curation-fence probe printed
`aggregate_proof_is_current = True` with three `match=True` lines and never reached the
curation those lines were supposed to bracket. The measurement resolved; it was a
before-vs-before comparison printed twice.

## The shape

Every check has a **subject** and an **address**, and it is cheap to verify the address
and expensive to verify the subject — so checks drift toward the cheap half and keep
reporting in the vocabulary of the expensive one. `check_features_md.py` says `OK` and
means *"the refs resolve"*; the reader hears *"the features work"*. A citation says
`file.py:718` and means *"this line exists"*; the reader hears *"this claim is true"*.

**A green address check is not weak evidence about the subject. It is no evidence about
the subject**, and its danger is proportional to how much it looks like the check you
wanted.

## What distinguishes this from Entry 45

Entry 45 (and its repeats) is **enumerating a class against a proxy**: the population
measured is not the population meant. This is different — the population is right and each
member is checked, but the *predicate* applied to each member is a weaker one than the
claim being made.

The two are easy to conflate and worth keeping apart, because the fixes differ. A proxy
error is fixed by re-scoping the enumeration. An address/subject error cannot be fixed by
scoping at all: the check has to do more work, or the claim has to be weakened to what
the check actually establishes.

Both happened today within the hour. A grep for `tests/….py::name` across FEATURES.md
returned five apparently-dead refs which turned out to be prose in the file's history
sections — the gate was right and the *measurement* was reading rows it was not scoped to.
That one is Entry 45. The three above are not.

## Disposition

Not "make the gate run the tests" — a ledger gate that executed 294 tests would be the
suite, and it is deliberately cheap so it can run on every PR.

**Rename what it reports.** A gate whose output says `refs resolve` cannot be misread as
`features pass`, and the two-line change costs nothing. The general form: **a check should
report in the vocabulary of what it verified, never of what it was written for.**

> **Applied.** The success line is now `refs RESOLVE (not run)` plus a breakdown, and
> writing it surfaced a distinction the single number had hidden: a shipping row is
> verified at **one of three strengths** — a named test found by `def` grep (452 rows), a
> file-only ref with no `::test` (4), or `n/a (manual)`, which this gate verifies not at
> all (6). Reporting them as one `294 shipping` was the address/subject gap *and* a
> collapse of three populations into one count. Only the first was noticed before the
> rename was attempted, which is an argument for doing the cheap fix rather than only
> filing it.

For citations, the rule from Entry 71 stands and generalises here: for a claim shaped
*"X reads Y at F:N"*, resolving `F:N` proves only that `F:N` exists. Reading the line is
what proves Y.

For probes, the rule the second version of the curation probe now enforces: **an arm
asserts its own precondition before measuring.** An arm that silently ran under the other
condition produces a clean-looking number for a question nobody asked.

---

### Entry 73 — CURATING A RUN MAKES IT UNDISCOVERABLE, AND S2's FIX WAS NEVER CARRIED DOWN. 2026-09-09.

**Kind: true but incomplete** — of audit S2's disposition. S2 established that GUI-owned
mutable state must be excluded from a currency check, because its writer carries its own
guard and comparing it against a frozen fingerprint makes the viewer report *its own
writes* as external drift. That was applied to `snapshot_is_current()` and **not carried
to the aggregate proof, which fences the same kind of file.** Nothing was false; a
consequence was not followed.

**The defect, measured end to end on a tree built by production publishers.** Marking one
colony in the results viewer makes that run refuse to open, permanently, with no remedy
in the message.

```
ARM FENCED  (success_markers_required == True)   -- as published
  core_readable (before) = True     discover (before) = OK
  curating 'a' object 1 as 'oversegmented'  -> curation WRITTEN
  core_readable (after)  = False
      master_parquet        match=True   1593 -> 1593
      measurements_csv      match=False    89 -> 74
      measurements_parquet  match=False  1593 -> 1570
  discover (after) = ValueError: Core aggregate files are not authorized by a
                     valid aggregate publication marker
```

The chain: `publish_aggregate_snapshot` fences three artifacts by size + sha256
(`_cli_completion.py:1119-1122`) — the master **and both mirror files**.
`CurationLabels._write_curated_mirror` rewrites two of the three
(`_curation_labels.py:846,848`) and republishes nothing. `publish_aggregate_snapshot` has
exactly two call sites, `_cli_finalize_run.py:484` and `sdk_/_hdf_to_zarr.py:746`, neither
reachable from the GUI, and there is no recertify path under `src/phenotypic/gui/`.

**Pre-existing, not introduced by this change.** The retired classifier computed
`core_readable` the same way and `discover` raised at the same site with the same string.
What is new is only that P6-T1's fixture is the first to set `success_markers_required`,
so it is the first to reach the predicate's second disjunct at all.

## The method note, which is the transferable half

The probe ran **two arms differing only in `success_markers_required`**, and that is what
turned a demonstration into a measurement:

| | byte damage | `aggregate_proof_is_current` | `core_readable` | `discover` |
|---|---|---|---|---|
| FENCED | csv + mirror parquet | **False** | **False** | raises |
| UNFENCED | csv + mirror parquet | **False** | **True** | OK |

Identical damage, opposite outcomes. The proof breaks in **both** arms, so *the proof
breaking is not the deciding term* — the divergence is entirely
`state_requires_success_markers`. A single-arm probe would have established "the fence
raises" and left "would a legacy tree also break?" open, which is the question that says
how many existing trees are affected.

**A control arm turns "X happens" into "X is caused by Y", and those have different
consequences.** The first licenses a fix; only the second says who is affected.

Two negatives worth recording, because both were live hypotheses:

* **`_publish_if_current` does not save this.** Its CAS guard refuses a write when a
  curation source changed; nothing had changed, so it wrote. `curation WRITTEN` in both
  arms retires the "undocumented guard prevents it" hypothesis as *dead*, not unobserved.
* **`master_parquet` is untouched in both arms.** The damage is exactly the two artifacts
  the GUI is designed to rewrite — which is the whole shape of the finding rather than
  incidental.

## Severity is higher than "an exception is raised"

* **The sidebar classifier consults neither `core_readable` nor the proof**
  (`shell/_classifier.py`), so the directory still presents as viewer-openable. The
  affordance says open; the open fails.
* **The hub** turns it into an HTTP 400 carrying the raw string
  (`shell/_routes.py:376-381`).
* **The standalone launcher does not catch it at all** — `results_viewer/__main__.py:94`
  lets the `ValueError` escape, so `python -m phenotypic.gui.results_viewer` fails to
  start on a curated run.
* **The message names no remedy**, unlike its neighbour in the same function, which ends
  *"Re-run `python -m phenotypic` with the current version to regenerate the master."*
* **The remedy that does exist costs the user work.** Re-running finalization republishes
  the proof over current bytes and rewrites the mirror from the master, replacing the
  curated bytes. `curation_labels.parquet` is not fenced and survives, so the labels are
  not lost — but the curated mirror is regenerated uncurated, and the run is inaccessible
  until the user runs the CLI again.

## Disposition

Argued separately and at length; the short form is that the aggregate proof has **two
consumers asking different questions** — `resolve_run_state` asks *"did the CLI's
finalization publish a complete set?"* and `core_readable` asks *"are these bytes safe to
read now?"* — and one artifact is answering both. The mirror's dual ownership is what
makes the difference visible. **After curation the claim the fence makes about the mirror
is false, and no re-issued certificate can make it true**, which is why republishing from
the GUI is the wrong shape.

**Owed engineering item, its own task, red-first.** It touches
`sdk_/_run_state.py:1131-1133` and `_cli_completion.py:1119-1122`, both live under other
agents while P6 lands.

## The probe's own two versions, and why 71, 72 and 73 are one thing

The first version of this probe had **an arm whose label could be read two ways**:
`force_schema3` was the parameter, so the arm printed `force success_markers_required =
False` meaning *forcing disabled*, which reads equally well as *forced to False*. The
logic was right — that arm was designed not to force, and did not — but a label with two
readings is not a labelling problem. It meant the run could not say **which condition it
had measured**, and it was read as the wrong one within minutes of being produced.

Had `discover` not failed first for an unrelated reason, that arm would have printed a
clean-looking result under a condition nobody could confirm. The second version names the
arms for the state they run under and **asserts its own precondition** —
`state_requires_success_markers(root)` compared against the arm's own claim — before
measuring anything.

**That is the same defect as entries 71 and 72, and stating it once is worth more than
three separate lessons:**

| | Verified | Reported as |
|---|---|---|
| **71** — a citation | the line exists at `F:N` | the claim about the identifier at `F:N` is true |
| **72** — a ledger gate | 294 refs resolve | 294 features work |
| **73** — a probe arm | a measurement ran | a measurement of *this condition* ran |

Each verified something adjacent to its claim and reported in the claim's vocabulary. The
common repair is not more checking; it is **making the report say what was actually
established** — read the identifier, print `refs resolve`, assert the arm's condition. A
check, a citation and an experiment are the same object here: something that licenses a
conclusion, and can license the wrong one for free by describing itself generously.

---

### Entry 74 — AN "UNUSED IMPORT" IS NOT DECIDABLE FROM THE MODULE THAT HOLDS IT. 2026-09-09.

**Kind: never true** — of the claim an orphaned-import pass makes. Recorded as its own
entry rather than folded into 72 because **a third coordinate slipped**, and the repair is
different.

P6 Task 7 deleted `browse/_source_render.py`'s ephemeral cache and, with it, the
`tempfile` import that served it. Module-locally that import was genuinely orphaned: after
the cache went, nothing in the file referenced the name. Deleting it broke **15 tests**,
which reached the name through the module object:

```python
from phenotypic.gui.browse import _source_render as sr
monkeypatch.setattr(sr, "tempfile", ...)
```

The repaired fixture states the mechanism better than a summary would
(`tests/gui/browse/test_tile_routes.py:39-43`):

> *"This used to read `sr.tempfile`, which worked only because that module imported
> `tempfile` for its own ephemeral cache. P6 Task 7 deleted that cache and the import with
> it, and every fixture reaching the module's attribute broke — a consumer with no import
> edge, which no import-graph walk can see."*

## The fact underneath, which is a property of Python and not of this codebase

**An `import` statement binds a module attribute, and that attribute is part of the
module's public surface whether or not the module uses it.** `sr.tempfile` is a legitimate
reference to a name the module never mentions again. So *"is this import orphaned?"* has
**no module-local answer** — it is a whole-program question, and the program includes the
test suite.

That is why three separate checks missed it, per the executing agent's report, and they
missed it for one reason: the consumer names the target through **an alias plus a string**,
so neither the target's qualified name nor its import edge ever appears.

* an **import-graph AST walk** sees import edges; a `setattr` consumer has none;
* a **`_source_render.tempfile` grep** cannot match, because the test aliases the module
  to `sr`;
* a **name-based target grep** is defeated by the same alias.

## Why this is a third axis and not another face of 72

Entry 73's closing collapsed 71, 72 and 73 into one claim: *a check, a citation and an
experiment are the same object, and can license the wrong conclusion for free by
describing themselves generously.* This one is that family, but the slip is elsewhere:

| | The check established | The claim was | Slipped | Repair |
|---|---|---|---|---|
| **45** | P of the wrong population | P of the right one | **population** | re-scope the enumeration |
| **71 / 72** | a weak P of the right thing | a strong Q of it | **predicate** | do more work, or shrink the claim |
| **74** | P **within one module** | P **across the program** | **scope** | widen the universe |

Folding 74 into 72 would blur exactly the distinction 72 was written to draw. The test is
the repair: a population error is fixed by re-scoping, a predicate error by strengthening
the check or weakening the claim, and **neither of those fixes this one.** A better AST
walk over `_source_render.py` returns the same answer, correctly, forever.

## Disposition

**Treat deleting an import as a public-API change when anything uses the module as a
namespace.** Concretely, an orphaned-import pass needs either:

* a whole-program reference check that includes aliased attribute access and **string**
  targets (`getattr`/`setattr`/`monkeypatch.setattr`), which is what defeats name-based
  greps; or
* a gate that **runs** the affected tests rather than reasoning statically about them —
  which is the honest option, because the static version is trying to decide a
  whole-program property from one file.

The cheap rule that would have caught it with no tooling: **before deleting an import,
grep the test suite for the module's aliases, not for the module's name.** The alias is
where the reference lives.

And the diagnostic worth keeping: the fixture's comment now records *why* it patches
`tempfile` directly. A test that reaches through a module attribute is depending on an
implementation detail of that module; saying so at the point of use is what stops the next
deletion re-breaking it — the deletion was correct, and the coupling was the defect.

## A second instance of row 3, one layer out: the baseline of a differential test

A failure was attributed to "pre-existing, proved against `820d4133`" and briefed that way
twice. `820d4133` **is** the commit that introduces the change under suspicion
(`feat(gui): P6 Tasks 1, 3, 4, 6`), so "fails at `820d4133`" is equally consistent with
*predates Task 4* and *caused by Task 4*. Run at its parent, the test passes: Task 4 caused
it.

The baseline was chosen by **position** — *the commit before the failures started
appearing* — rather than by the **property** the claim needed it to have: *does not contain
Task 4*. Those two coincided for the other two suspects and diverged for the third, and the
run printed the same green either way.

**This is row 3 of the table above, at a different layer**, and the repair is identical —
which is the test for whether something is a new kind. My own probe split its arms on a
flag it assumed rather than asserted; this split a history on a commit whose contents were
assumed rather than asserted. Both produce a measurement of a condition nobody confirmed.

| | The condition assumed | What asserts it |
|---|---|---|
| probe arm | `success_markers_required` is unset in this fixture | read it back and compare |
| differential baseline | this commit does not contain the change | `git log -1 --format=%s`, or diff it for the file |

**Name the property the baseline must have, then assert the baseline has it.** A commit's
position in history is not that property; it is a heuristic that correlates with it, and
the correlation is exactly what breaks at a boundary commit — the one place a differential
test is most often pointed.

Filed here rather than as its own entry, and rather than in 74, because 74's coordinate is
*scope* and its repair is widening the universe. Nothing here needs a wider universe. It
needs the same one sentence as the probe arm: **say what condition you are measuring under,
and check.**

---

### Entry 75 — a fixture that blanks fields cannot build a tree from before those fields existed

**Where:** `tests/unit/sdk_/_migration_fixtures.py::make_markerless`, and every test standing
on it. **Found:** P7 Task 2b, while building the v0.17.3 floor shape it turned out not to be.

`make_markerless` is named for the pre-markers era and is not from it. It takes a **modern**
`build_completed_run` and blanks two things:

- `success_markers_required = False` — **present and falsey**, where the floor shape has the
  key **absent**;
- it leaves `work_ids` **in place**, content-derived, where the floor shape has no such key
  at all — the concept did not exist.

**The general form, which is the part worth carrying:** absence and emptiness are different
shapes, and every detector in this change keys on **absence**. `requires_conversion`'s
signal 5 fires on `work_ids` being *absent*; signal 4 on `restart_epoch` being *absent*. A
fixture that sets a field to a falsey value clears the signal while describing a tree that
never existed, so a suite built on it tests a shape the world does not contain.

**What it cost, measured rather than supposed.** `_configured_work_id`
(`_cli_migrate_image.py:126`) has two arms: a lookup that hits when `work_ids` carries the
stem, and a fall-through to the synthetic `_migration_work_id`. MIG-10 named the
fall-through as the blind spot. Because `make_markerless` *retains* `work_ids`, every test
reaching that function has exercised the **hit** arm — so the arm MIG-10 flagged has been
untested since it was flagged, **by the fixture that looked like it covered it**.

**The naming hazard is the multiplier.** "Markerless" reads as *the shape from before
markers* to everyone who does not open it, which is everyone who is using it to avoid
building that shape themselves. A fixture whose name asserts a provenance its body does not
produce is worse than an unnamed one: it answers the question a reader would otherwise ask.

**Repair.** Build the absent-key shape by `pop`, never by assignment:

```python
config.pop("work_ids", None)              # absent -- the shape signal 5 detects
config.pop("success_markers_required", None)
```

and assert the absence *before* the code under test runs, so a fixture that quietly stops
producing the shape fails in its own test rather than in the conclusions drawn from it.

**Relation to 74.** 74's coordinate is *scope* — a consumer outside the universe searched.
This one's is *shape* — a fixture inside the universe that is not the thing it is named for.
Different repair: 74 widens the search, this one changes how the input is built. Same
underlying move, though, and it is the same one as the probe arm and the differential
baseline above: **state the property the input must have, then assert the input has it.**
Three coordinates now — scope, condition, shape — and one sentence repairs all three.

---

### Entry 76 — a citation measured in the wrong coordinate system

**Where:** three citation-shaped errors across one session, in briefs and plan text.
**Found:** each time by someone re-resolving the symbol against the file, never by a check.

The register already carries two citation failures. This is a third axis, and the three are
worth stating together because the repair differs for each:

| Axis | The failure | What catches it |
|---|---|---|
| **Existence** | the cited line does not exist / is far off | print the line |
| **Identifier** | the line exists and carries a *different* symbol than the claim names | print the line **and read it** |
| **Coordinate system** | every number is present, plausible, and wrong by a constant | re-resolve by symbol against the **whole file** |

The third is the nastiest, because nothing is missing. A representative production:

```bash
sed -n '/^def publish_image_success/,/^def [a-z]/p' file | grep -nE 'publish_image_record'
```

**`grep -n` numbers the stream it is given, not the file.** After a `sed` extract every
number is an offset *within the extract*. The output looks exactly like file lines — same
shape, same plausibility, no gap to notice — and is uniformly wrong by the extract's start
offset. Reported as `:83-84` and `:90`; the real lines were `:254` and `:261`.

Contrast the other two productions from the same session, which share the underlying move:

- `grep -rn "deactivate_generation("` — the `(` excluded every wrapper call, returning **4**
  where the symbol returns **18**. A *spelling* measured, an *enumeration* reported.
- `grep -rn "def _run_finalize" … | head -2` — a truncation applied to a result whose size
  had not been established, cutting the one hit that mattered.

**One sentence covers all three: measure the quantity you are about to report.** Stream
position is not file position; a spelling is not a symbol; the first two hits are not the
hit set.

**Mechanical repairs, in order of how often they are needed here:**

1. `grep -n` on the **file**, never after a `sed` extract. If an extract is wanted for
   reading, take the citation from a separate whole-file grep.
2. Search the **symbol**, not a spelling that includes punctuation — `deactivate_generation`,
   not `deactivate_generation(` — since wrappers, aliases and re-exports all fail the
   punctuated form.
3. `| wc -l` before `| head`. A truncation is only safe once the size is known.

**Why this keeps landing.** All three were produced *while the author was writing about this
very failure mode* — the enumeration error appears in the same message that documents it.
Knowing the rule is not the control; the control is the instrument. That is the same
conclusion as the three coordinates in Entry 75, arriving from the citation side rather than
the fixture side: **state the property you are about to assert, then use an instrument that
measures that property.**


---

### Entry 77 — two fitted chains agreeing is close to no evidence

**Where:** P7 Task 2b, a resume-path test expected to fail that passed. **Found:** by the
executing agent, about **its own** reasoning, after that reasoning had been independently
corroborated.

A test was run expecting failure. It passed. A four-link chain was then built explaining
why the pass was consistent with the bug still being present. A second party built the same
chain from the same four files and reached the same conclusion — which read as strong
confirmation.

**It is close to no confirmation at all.** Both chains were constructed **after** seeing
that green, so each is *fitted to* the observation rather than *predictive of* it. Two
explanations converging is evidence only when the convergence is independent, and reading
the same four files in a different order is not independence — it is one derivation
performed twice.

## What separates this from its two neighbours

The register already carries two agreement failures, and the repairs are all different,
which is what makes this a third rather than a restatement:

| | The agreement | Why it was worthless | Repair |
|---|---|---|---|
| **40** | two unrelated queries returned the same count | both measured the wrong population; cross-checking would have *agreed* and both been wrong | measure the right population |
| **65** | a reviewer agreed with a justification | it was internally coherent and never tested against the code | test it against the code |
| **77** | two chains explained one result | both were built **from** that result | predict a result **not yet observed** |

**65 is the near neighbour and the contrast is the point.** Its failure was *not consulting
the evidence*; this one's failure is consulting it **first and then explaining it**. So 65's
repair is unavailable here — "test it against the code" is exactly what both parties did,
and it did not help, because the code is what the explanation was fitted to. An explanation
cannot be tested against the observation that produced it.

**An explanation built after the observation must earn its keep by predicting a second
one.**

## The defence that worked, and why it belongs in this entry rather than its own

The follow-up test came back **inconclusive** — it failed at a continuation refusal and
never reached the record comparison the chain was about. The executing agent had named that
outcome **in advance** as *inconclusive, not confirmation*.

That advance naming is the only thing that held, and the reason is specific: **the failure
was in the predicted direction.** A fitted chain predicts a direction — that is nearly all a
fitted chain does — so a directional match is exactly what one produces whether or not it is
true. Without the outcome named beforehand it would have been read as confirming the chain,
and the chain would have been confirmed by the observation it was built from, one step
removed.

**A false positive that agrees with you is the one nobody re-examines.**

It is filed here rather than as Entry 78 on the separateness test this register has been
using — *does the repair differ?* At first pass it looks like it does: 77's repair is a
novel prediction, the defence's is an outcome table written before the run. **But a
prediction with no stated falsifier is not a prediction.** Naming what each outcome would
mean, including the ones that mean nothing, is not a second repair — it is what the first
repair requires in order to be real. Apart, 77 is aspirational and the defence is
unmotivated; together they are one procedure.

## Why this one was catchable at all, which is the least obvious part

It was diagnosed by the agent whose reasoning it was, *after* corroboration — and
corroboration is precisely the signal that stops people looking. Entries 40 and 65 were both
caught by an outside party re-deriving the claim. This one had to be caught from the inside,
because from the outside it looked like two independent analyses agreeing, which is the
shape of a result nobody re-opens.

The transferable form is uncomfortable and worth stating plainly: **agreement is the point
at which to ask what would have had to be different for you both to be wrong.** If the
answer is "nothing we looked at", the agreement is a property of the looking, not of the
subject.

---

### Entry 76a (addendum) — the write variant is the same error with consequences instead of output

Entry 76 named three productions of one instrument failure: `grep -n` after a `sed` extract,
a punctuated spelling searched instead of a symbol, and `| head -2` on an unmeasured result.
All three **printed** a wrong answer. There is a fourth production that **applies** one, and
it is the same error:

```python
start = next(i for i, l in enumerate(lines) if l.startswith("@pytest.mark.xfail("))
```

That file held **three** `xfail` markers. `next(...)` took the first, the span ran to the
following `@pytest.mark.parametrize`, and the write removed **224 lines and four tests** —
`test_a_legacy_tree_is_refused_now_that_the_gate_is_armed`,
`test_the_arming_flag_has_one_source`,
`test_the_gate_is_armed_exactly_when_the_forward_path_stops_writing_markers` and
`test_the_gui_reports_rather_than_refuses`.

**`next(...)` is `head -1` wearing different clothes.** Both silently select one of N without
telling you N. So is `str.replace(old, new)` with no count assertion, and so is `sed -i`
without `/g` reasoning. The list is not the point; the shape is: **a selector that cannot
express "I expected exactly one" will not tell you when there were more.**

**What makes the write variant worse is not the instrument, it is the absence of a reader.**
A wrong line number gets read by a human who may notice it looks odd. A wrong span gets
applied, and the only thing standing between it and lost work is whether someone checks
afterwards. Nothing failed here: the file still parsed, the suite still collected, and the
four deletions were invisible until a `git diff --stat` was read.

**The repair is the one already in 76, applied one step earlier — to the anchor rather than
to the citation:**

```python
assert t.count(old) == 1        # before ANY edit derived from a search
```

Two further notes, both from how this one was caught rather than from how it was made:

- **`git diff --stat` after a scripted edit is the cheap check.** `24 insertions(+), 6
  deletions(-)` was the expected shape; `21 insertions(+), 224 deletions(-)` was not, and the
  difference is visible in one line without reading any code.
- **Verify a restore against the source of truth, not against the restorer.** The AST count
  said "44 defs, none lost", which is what a *correct* restore looks like and also what a
  restore that silently kept a stale copy would look like. The independent check that
  settled it was a second party resolving the four names and the def count from `HEAD`.

**This is the second time in one session that a defence recorded in this register was not
applied by the person who wrote it up** — 76 was written hours before its author used
`next(...)` on an uncounted pattern. That is not carelessness and treating it as such
predicts the wrong fix. It is that **the register is read at review time and the instrument
is chosen at typing time**, and only one of those is a moment of deliberation. The repair
that works is the one small enough to become reflex: `assert count == 1`, every time, before
the edit.

**A fourth production, and the one that settles what kind of failure this is.** While
routing the fix for a defect *diagnosed by counting*, the corrector counted
`republish_aggregate`'s `return False` sites with a `sed` range that overran into the next
function, and reported the range's count (six) as the function's (four) — correcting
someone who was right. Same instrument, same coordinate-system error as 76's opening
example, produced by the person applying the discipline, in the message applying it.

That is the argument against reading any of this as inattention. It has now been produced
by both participants, in messages *about* it, in both directions of correction. **A failure
that survives knowing about it, caring about it, and actively looking for it in someone
else is not an attention failure — it is a tooling default.** `grep -n` numbers the stream
it is handed; `sed -n 'a,bp'` yields whatever the range contains and not what the symbol
spans; `next(...)` returns one of N silently. None of them can express *"I expected exactly
one"* or *"I expected this to be the whole function"*, so the expectation lives only in the
author's head, where it cannot fail.

The repairs are therefore all shaped the same way — make the expectation something the
command can contradict:

```bash
grep -n 'pattern' file            # not: sed -n 'a,bp' file | grep -n
awk '/^def name/,/^def /' file    # symbol-bounded, not line-bounded
grep -c 'pattern' file            # before any head/next/first-match
```

```python
assert text.count(anchor) == 1    # before any edit derived from a search
```


---

### Entry 78 — a topology with no equivalence test cannot detect a class, and nothing on the page looks wrong

Every other entry in this register was found by **reading**. That shared property is stated
at the top of the file, and it is what makes the class expensive. This entry is here because
it is the complement: a defect class that reading does not reach, because there is nothing
written down to be wrong about.

`--mode migrate` has two execution paths (local, SLURM) crossed with two target kinds (full
run, provenance-only). Four combinations. Nobody had enumerated them, so nobody had noticed
that `migrate_machine_state` was wired into one.

**The full-run topology had a local-vs-SLURM equivalence test**
(`test_local_and_synchronous_slurm_migration_publish_equivalent_runs`), and it earned its
keep the moment the converter was wired into the local driver: the SLURM tree kept the
**forward** run's `processing_generation`, minted with a real `per_image_config` digest,
while the local tree re-derived it under migrate's inputs. One field, exact comparison,
caught immediately.

**The provenance-only topology had no such test**, and carried the identical gap. It was
found by asking what else had the first gap's shape — not by anything failing.

#### What writing the missing test then cost, and returned

The absence was not a coverage gap in a *behaviour*. It was a gap in the ability to
**detect**, and the difference is the whole entry: a behaviour gap leaves a wrong statement
somewhere a reader can find; a detection gap leaves nothing at all. The cost of it was one
bug found by luck rather than by a gate.

Writing the test turned three things up, and they arrived by three different routes:

- **Detected directly.** Its first execution failed on the SLURM planner refusing a tree
  with zero stores — which is what a pre-markers process tree legitimately *is*, since
  OME-Zarr postdates the vintage the kind exists to convert.
- **Forced.** That failure made a question unavoidable that had been askable for weeks:
  whether MIG-11's mint-from-outputs arm was reachable at all. It was not.
  `execute_provenance_migration` iterates `target.stores` and never reads `target.outputs`,
  so the local arm did nothing, reported `provenance_upgraded=0`, and **exited 0** — a
  silent successful no-op on the exact tree the arm exists for.
- **Found by asking its shape a second time.** Filling the first two gaps created an
  asymmetry that had not existed before, and interrogating *that* turned up a defect in the
  fix shipped one round earlier: the seal-stage call ran for `direct_store` too, whose
  lifecycle state is a hashed sibling **by contract**, so it would have written
  `.phenotypic/` inside the store. It no-ops on a bare store today only because every arm
  happens to find nothing — the accident that stops being true without anything failing.

#### The two things worth carrying into P7 Task 6's register

**A register lists what is tracked. It cannot list what nothing checks.** Every entry above
this one describes a statement that was wrong; this one describes a *question nobody asked*.
The register's own failure mode is one level up from drift: not a stale row, but a row that
was never written because the axis it belongs to was never enumerated. `xfail(strict=True)`
is the instrument that survives this, because a mark that stops being true fails loudly —
which is also why a strict mark must never carry a **superseded** reason, or the instrument
becomes the failure it prevents.

**"Compare approximately" is the wrong default for identity.** The exact comparison is what
caught the first gap, and the pressure to relax it came on hardware-variance grounds that do
not reach the field: `derive_processing_generation` digests a pipeline hash, a per-image
config digest, and a restart epoch — no paths, no timestamps, no measurements. Two runs on
different nodes produce identical digests. **Comparing approximately would have hidden a
missing conversion, not absorbed variance.** The rule that generalises: tolerance belongs to
measurement outputs, never to configuration identity, and a test that tolerates a difference
it cannot name is worse than one that fails.

#### The corollary about a test's claim

When the second half of that test finally ran against the fixes, the two arms could no
longer be equivalent — one of them had become a deliberate refusal, because the provenance
chain has no vocabulary for a storeless tree (its seal barriers store statuses and its
finalizer reports upgrade counts, so the only real work would be invisible in its own
terminal report). The test's **claim** was rewritten to name the divergence; its rigour was
not relaxed. It now asserts that the refusal precedes any write, which is a stronger
property than the equivalence it replaced: a refused submission that already converted half
a tree is the worst of both arms.

"The test found something" and "the test's claim was wrong" are different, and only the
second licenses a rewrite.

---

### Entry 76b — when the right operation is not expressible, discipline cannot reach it

**Found by P6-T1**, repairing a `--dry-run` guard in P7 Task 5, and written up here at
its suggestion rather than by it — it declined to append to 76a unasked, on the grounds
that a shared register gets contended exactly that way. The formulation below is its; the
placement is mine.

**A separate entry rather than an addendum to 76a, by 76a's own test.** The two look like
one kind and have **different repairs**, which is the test that separates them here.

The guard needed to know whether a call site passed `dry_run=True` or `dry_run=False`. Its
helper returned **line numbers**. With a list of integers in hand, `min()` was available,
looked correct, and was wrong — and the correct partition was **not representable at all**,
because the information distinguishing the two calls is not in their positions. Returning
`ast.Call` nodes did not make the right answer easier to write. **It made the wrong one
impossible to write.**

| | 76a | 76b |
|---|---|---|
| Coordinate | which instrument you **reach for** | which operations the data shape makes **available** |
| The right move was | available, and not taken | **not available** |
| Repair | `assert count == 1` before acting — a check | change what the function **returns** — a type |
| Why the other repair fails | — | `assert len(candidates) == 1` passes: both integers are legitimate line numbers, and neither carries `dry_run` |

That last row is why this is not 76a with more feeling. Someone who had internalised 76a
perfectly, and who asserted their selection was unique before acting on it, **would still
have written `min()`** — because the assertion would have held and the answer would still
have been wrong. Discipline cannot reach a distinction the data does not carry.

**The general form:** *before hardening a selection, ask whether the value being selected
from carries the property you are selecting on.* If it does not, no amount of care at the
call site helps, and the fix is upstream — return the richer thing.

**And the guard's other half belongs here too.** `_literal_dry_run` **raises** on a call it
cannot classify rather than defaulting, *"because that is how a guard starts agreeing with
whatever it is shown."* That is the same move as `strict=True` on an `xfail`: refuse the
default that quietly accepts. A guard with a fallback is a guard with an opinion about
inputs it does not understand, and the opinion is always "fine".

**Relation to the rest.** 75 is *shape* — a fixture that satisfies a check without
describing a tree. 76 is *coordinate system* — a measurement reported in the wrong units.
76a is *selection* — one of N taken silently. **76b is expressiveness** — the operation you
needed was never on the menu. Four coordinates now, and only the first three are repaired
by being careful.


---

### Entry 79 — the answer was right and the cost was wrong, so no assertion over outputs could see it

**Found by the gate**, not by reading — the second entry here with that provenance, and the
first found by a gate rather than by the absence of one. Entry 78 is the complement of this
file's opening claim in one direction (nothing written down to be wrong about); this is the
complement in the other (something checkable, checked, and caught by a machine).

`unprojectable_stores` enumerated stores with `results.rglob(f"*{STORE_SUFFIX}")`. Every
test over that function passed, and would have kept passing forever, because **the returned
tuple was correct**. `rglob` descends *into* each matched store — roughly 400k `stat` calls
at 10k images — and then filters back to exactly the list a two-level `glob` produces. The
walk is invisible in the result by construction.

`test_no_recursive_glob_for_stores` exists for precisely this, and its docstring says why in
one line worth quoting: *"an assertion about results cannot see cost."*

#### Why this is a fifth coordinate and not a face of 76b

76b is about an operation that was **not on the menu**: the right answer could not be
written, so care at the call site could not reach it. Here the right operation was on the
menu, trivially available, and one character shorter. What was unavailable is **the
observation**. The defect is not in what the code computes; it is in what it *does on the
way*, and the entire instrument family this change relies on — assert the return value,
compare the tuple, diff the tree — is blind to that by design, not by oversight.

| | 76b | 79 |
|---|---|---|
| What was missing | the right **operation** | the right **observable** |
| A perfect implementer would have | still got it wrong | got it right, and been unable to prove it stayed right |
| Repair | change the return **type** | add an instrument in a **different modality** — read the source, not the result |
| What a passing behavioural test proves | nothing about the choice | nothing about the cost, *and it never will* |

The general form: **when correctness and cost are separable, a suite that only asserts
correctness cannot regress-protect cost, and its greenness is not evidence either way.** The
instrument has to change modality — here, an AST/regex sweep over the source. That is why
the invariant test matches the f-string form deliberately, so that the function which once
used the pattern cannot exempt itself.

#### The companion finding, which is this register's ordinary kind

Repairing the walk turned up a **never true** row in the repairer's own test.
`test_a_store_with_no_measurement_descriptor_is_named_not_raised` claimed:

> *"The second store is the co-witness: a function that returned every store it saw, or
> none, would pass a one-store version."*

**The body built one store.** The docstring described a discriminator that was never
constructed — the claim written first, the fixture never catching up.

It was not decorative. The tuple equality could catch *"returned none"*, but with a single
store nothing could catch *"returned every store it walked past"* — which is exactly the
failure mode a rewrite of the enumeration introduces, missing precisely where someone had
just been editing. The repair builds the second store the docstring always claimed (an
**unreadable** one, which must be absent because that is a different fault with its own
reporting), so the equality now discriminates and the `(OSError, ValueError)` arm has a
witness. The docstring records that it overclaimed, which is the honest form.

**The pairing is the point.** A cost defect that no behavioural test can see was sitting
behind a behavioural test whose stated discriminator did not exist. Neither would have
found the other, and the gate found only the first.

---

### Entry 80 — "migrated records carry `provenance: migrated`" is true of two writers and false of the third

**Kind: true but incomplete.** Found while writing the contributor-guide page
that replaces `_cli/CLAUDE.md`'s state tables — i.e. by having to state the rule
completely for a reader who could not check it, which is a different exercise
from checking it.

`_cli/CLAUDE.md` said:

> *"Migrated records carry `provenance: "migrated"`, and `record_rejection`
> skips the `work_id` comparison for exactly those."*

Every clause is correct. What it omits is that **`--mode migrate` has three
record-writing paths and only two of them stamp it**:

| Path | Publisher | Provenance |
|---|---|---|
| Converting an existing marker | `_cli_migrate_state.py:350` | `migrated` |
| Minting from outputs (MIG-11) | `_cli_migrate_state.py:1080` | `migrated` |
| Migrating the image artifact | `_cli_migrate_image.py:589` | **`forward`** |

The third goes through `publish_image_success`, which has **no `provenance`
parameter** and so takes `publish_image_record`'s `PROVENANCE_FORWARD` default.
Verified by signature, not by inference.

#### Why the omission is load-bearing rather than pedantic

A markerless legacy tree — the case the continuation refusal exists for — is
converted by exactly the third path. **So on precisely the trees where you most
want to ask "was this migrated?", the field says `forward`.** A reader who
believed the sentence would reach for provenance, find it, and get the wrong
answer on the only trees that matter.

`_output_was_migrated`'s docstring already said this, in `phenotypicCLI.py`,
where nobody comparing it against `_cli/CLAUDE.md` would look. **Two documents
were each locally consistent and jointly misleading**, which is what makes this
kind expensive: neither is wrong on its own page, so neither review catches it.

#### The general form

*A rule stated over a mode is wrong whenever the mode has more than one code
path.* "Migrate writes X" is a claim about a **mode**; the truth was a claim
about a **publisher**. The repair is not more caveats but changing the subject
of the sentence — the table above is keyed by path, and a fourth path added
tomorrow forces a fourth row rather than silently joining a majority.

**And it is why the disposition is a page, not a paragraph.** This register's
running complaint is that prose nobody checks drifts; the answer adopted here is
a single reference with the consumer table in it and an explicit instruction to
update it in the same change, with `_cli/CLAUDE.md` reduced to a pointer.
**Two copies of a table are two things to keep true.** The pointer cannot
disagree with the page, because it does not restate it.
