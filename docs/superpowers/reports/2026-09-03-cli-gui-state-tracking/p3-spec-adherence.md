# P3 spec adherence — does what was built match what the spec and plan said?

**Subject:** commit `1cc6740c` (with `9480dd5b`), phase 3 of `cli-gui-state-tracking`.
**Base:** `e74bf706`. **Reviewer remit:** spec/plan ↔ code, plus category E.
**Analysis only. This file is the only write.**

Verified against the P3 diff (`git diff --name-only e74bf706..1cc6740c -- src` →
14 files), the shipped tests, and the four source documents named in the brief.

---

## Verdict in one paragraph

**The five known divergences are all real, and four of the five are recorded
accurately somewhere in the tree.** The code and the test suite carry the P3
rulings better than the plan documents do: the disarm ruling is written down
with its honest cost, the equivalence gate's derivation was genuinely rewritten
rather than digit-patched, and the deferrals carry strict tripwires. The
failures are concentrated in one place and have one shape — **the disarm ruling
propagated to the code and to one docstring paragraph, and to nothing else.**
It left an instruction standing in P3's plan, no instruction at all in the plan
of the phase that now owns it, two false claims in source that the same commit
falsified, and — the expensive one — an inverted signal count in the exact
docstring paragraph a P7 implementer will read to decide whether arming is
safe. Two further items are undocumented drift the brief did not name: a
plan-mandated `work_id` merge fence that was not built and whose absence is
recorded nowhere, and one unconverted read-back site in `_cli_migrate.py` whose
sibling and validator were both repointed.

Findings are labelled with the brief's categories: **(a)** recorded and
justified, **(b)** recorded but inaccurately, **(c)** undocumented drift.

---

## 1. The schema gate ships disarmed

### 1.1 The ruling itself — **(a) recorded and justified**

`src/phenotypic/sdk_/_schema_shape.py:113-143` records the reversal in full: that
P3 armed it and un-armed it again, that the binding constraint moved from *"arm
with the publisher"* to *"arm with dischargeability"*, that
`_hdf_to_zarr._republish_image_marker` rewrites the legacy marker so signal 1
fires on a tree migrate has just finished, and that the armed error message would
name migrate as the remedy for a verdict migrate cannot discharge. That is
INV-DISCHARGEABLE reasoning, stated on the escape hatch itself.

**The wrong claim you asked about does not survive.** `_schema_shape.py:135-143`
says the opposite, explicitly and unprompted:

> *"It is **not** a clean return to pre-P3 behaviour. Pre-P3 the publisher *and*
> the reader were both on `image_complete/`, so a legacy tree resumed correctly.
> Now the reader is on the record and nothing stops a legacy tree from entering
> a writing mode, so `valid_image_success` is false for every image and the run
> **reprocesses from source** instead of resuming — which on a migrated archive
> whose inputs are gone is a failure rather than a waste. That is worse than
> pre-P3 and better than the refusal loop, which is the trade the ruling makes."*

A grep for `pre-P3`, `restores exactly`, `exactly the pre` across `docs/`, `src/`
and `tests/` returns no surviving instance of the wrong framing. Nothing to fix.

**The arming obligation is mechanically enforced, which is the right shape.**
`tests/unit/cli/test_schema_gate.py:930-940` puts `xfail(strict=True)` on
`test_the_gate_is_armed_exactly_when_the_forward_path_stops_writing_markers`
rather than weakening its assertion, with the reason naming P7 Task 5 Step 1b.
When P7 arms the flag the test xpasses, strict turns that into a failure, and the
marker is removed because the suite says so. This is EXECUTION.md's category-E
rule 3 satisfied exactly.

### 1.2 P3's plan still instructs the implementer to arm — **(c)**

- **Plan:** `phase-3-per-image-record.md:722` — a callout headed
  *"### This task must ARM the schema gate, in the same commit"*, running to
  `:763`, including `:742-743` *"this task arms it. Set `SCHEMA_GATE_ARMED = True`
  in the same commit that moves the publisher off `image_complete/`."*
- **Code:** `_schema_shape.py:144` — `SCHEMA_GATE_ARMED: bool = False`.
- **What differs:** the plan directs an action the user's ruling reversed.

The plan file *was* edited in `1cc6740c` (39 lines, `git show 1cc6740c --
docs/.../phase-3-per-image-record.md`) — but only for the `_COMBOS` count. The
arming callout was not touched, and the drift register has no row for it.

This is **drift-register entry 24's shape exactly** — *a plan step superseded by
a later ruling, still written as an instruction* — and it fails entry 24's own
remedy, restated in that entry as **"Every ruling gets the grep — especially the
ones that change nothing."** Here the ruling *was* enacted in code, so the grep
target existed (`SCHEMA_GATE_ARMED`) and returns `:724`, `:743`, `:749` in this
very file. One command would have found it.

Severity is lower than entry 24's original only because P3 is executed; it rises
again the moment anyone re-reads P3's plan to reconstruct why the gate is off.

### 1.3 P7's plan — the phase that now owns arming — says nothing about it — **(c)**

- **Code:** `_schema_shape.py:130-134` designates the arming commit:
  *"that lands in **P7 Task 5 Step 1b** … Arm here in that commit, not before."*
- **Plan:** `grep -n "SCHEMA_GATE_ARMED\|arming\|armed" phase-7-migrate-mode.md`
  returns **nothing relevant** — the four hits are `--mode process` "arm" in the
  sense of a code branch. Task 5 Step 1b (`:1168-1216`) covers the `legacy-v2/`
  rename, `_PRESERVED_ON_RESTART`, `--revert` and MIG-13 collision, and never
  mentions the flag.

So the obligation to arm exists in exactly two places, both of which a P7
implementer reaches only by accident: a comment inside a constant's docstring,
and an `xfail` reason. The plan they will follow does not carry it.

The tripwire in §1.1 *does* close the loop — if P7 lands without arming, nothing
fails, but if P7 arms, the test forces the marker's removal. The gap is the other
direction: **nothing makes P7 arm.** A P7 that ships the `legacy-v2/` rename and
stops leaves the tree permanently in the "reprocesses instead of resuming" state
§1.1 calls *worse than pre-P3*, with a green suite.

### 1.4 Two source claims the same commit falsified — **(b)**

**`src/phenotypic/_cli/_cli_completion.py:236-240`:**

> *"A tree carrying `image_complete/` and no `images/` is a legacy tree, which
> `--mode migrate` converts and every writing mode now refuses — which is why
> `SCHEMA_GATE_ARMED` flips in this same commit."*

Both clauses are false as shipped. No writing mode refuses such a tree
(`_cli_schema_gate.py:81` returns early on the unarmed flag), and the flag did
not flip. This sits at the top of the very function P3 rewrote.

**`src/phenotypic/sdk_/_schema_shape.py:67-71`**, the opening of the flag's own
docstring:

> *"Whether a `CONVERT` verdict may be **surfaced**. **Flipped to `True` by P3
> Task 2**, in the same commit that makes `publish_image_success` write the
> consolidated record — see `test_the_gate_is_armed_exactly_when_the_forward_
> path_stops_writing_the_legacy_marker`, which fails the moment those two
> disagree."*

Three errors in one sentence, in a docstring that is corrected 47 lines lower:

1. *"Flipped to `True` by P3 Task 2"* — contradicted by `:114` (*"STILL DISARMED
   after P3"*) and by `:144`.
2. The cited test name does not exist. The real name is
   `test_the_gate_is_armed_exactly_when_the_forward_path_stops_writing_markers`
   (`tests/unit/cli/test_schema_gate.py:941`) — no `_the_legacy_marker` suffix.
   The register's own closing advice is *"prefer a pointer to a real symbol over
   a restatement"*; this is a restatement dressed as a pointer.
3. *"which fails the moment those two disagree"* — it no longer fails. It is
   `xfail(strict=True)`, so on disagreement it reports **xfail**, which is a
   pass. This is the register's shape 2 (*a claim about what a check does*)
   applied to the check that guards this flag.

This is **entry 25's tell inside a single docstring**: *"a document that has been
revised states its load-bearing facts more than once, and revision updates one
site."* A reader who reads the first paragraph and stops concludes the gate is
armed and the coupling is enforced. Both are false.

### 1.5 The signal count is inverted, in the paragraph P7's decision rests on — **(b), highest severity in this report**

`src/phenotypic/sdk_/_schema_shape.py:73-77`:

> *"It is `False` today because the legacy shape and the **current** shape still
> overlap: the forward path writes `image_complete/` and writes
> `datasets.<ds>.completed`, so **two** of the five signals below fire on a tree
> the running build has just written."*

**Both halves were made false by this commit.**

| Claim | Falsified by |
|---|---|
| "the forward path writes `image_complete/`" | D1's clean break. `_cli_completion.py:248-256` now returns `publish_image_record`; `test_schema_gate.py:970-985` measures `writes_legacy_marker` as `False`. |
| "writes `datasets.<ds>.completed`" | `_cli_state_management.py:81-109` — `completed`, `failed` and `errors` were removed from the per-dataset entry in this commit. Signal 3 (`_schema_shape.py:299-301`) keys on `"completed" in entry`. |

So **zero** of five signals fire on a tree this build wrote, not two — and the
change ships its own proof: `tests/unit/cli/test_image_record.py:650-683`,
`test_a_tree_this_build_wrote_needs_no_conversion`, asserts
`requires_conversion(root) is None` over `build_complete_run`. That test's own
docstring names its purpose as *"the standing evidence that arming is safe for
forward-written trees, which is what P7 Task 5 Step 1b needs before it flips the
flag."*

The docstring 580 lines away says the opposite, and it is the paragraph a P7
implementer opens the constant to read. The docstring's forward-looking sentence
is wrong too: *"Signal 3 … is §4.2's to remove in P3+, and the count drops to one
when it does"* — P3 removed signal 3's evidence **and** signal 1's, so the count
dropped to zero.

**This is the third recurrence of the same defect in the same docstring.** The
paragraph immediately below (`:79-85`) narrates the first two — *"It said three
until P2, and P2 is what made that false … The count is not decoration — it is
the claim that decides whether arming this flag would refuse trees the current
build wrote, so a stale one is a wrong answer to the only question a reader opens
this docstring to ask (gate IMPL-F6 / SPEC-C3, found independently by two
reviewers)."* The commit that made the count wrong a third time is the commit
that quotes that warning.

**Disposition:** row 2 of EXECUTION.md's table — *the code is right and the
document is now wrong.* Amend the docstring; do not touch the code.

---

## 2. Signal 1 fires on directory existence — **(a), and not a divergence**

- **Plan:** `phase-7-migrate-mode.md:162-164` specifies the shape detection as
  *"1. `.phenotypic/progress/image_complete/` exists — 2. `stage3_complete/`
  exists. **NOT `stage2_done/`** (U-9)"*.
- **Code:** `_schema_shape.py:275-283` — a loop over
  `(DIR_IMAGE_COMPLETE, _DIR_STAGE3_COMPLETE)` testing `.is_dir()`, with a
  comment carrying the U-9 exclusion and the `legacy-v2/` note verbatim.

The code matches the plan text as it now stands, and P3 did not change this
logic — `git show 1cc6740c -- src/phenotypic/sdk_/_schema_shape.py` touches only
the flag's docstring and `_DIR_STAGE3_COMPLETE`'s. If an earlier discriminator
existed it is no longer the plan of record, so there is nothing unrecorded here.

**One thing was correctly recorded as a loss rather than a simplification.**
`_DIR_STAGE3_COMPLETE`'s new docstring (`:146-161`) states that
`test_the_stage3_directory_name_matches_the_writer` was deleted with the writer
it was pinned against, that no replacement can exist (it could only compare the
constant to itself), that a wrong value fails **silently and in the dangerous
direction** — signal 1 stops firing and a legacy staged tree is neither converted
nor refused — and that the remaining ground truth is P7's migrate tests. That is
a gap named as a gap, which is the correct disposition.

**By contrast, `DIR_IMAGE_COMPLETE` (`sdk_/_io_constants.py:669`) carries no
comment at all** — a bare `Final[str]` under `DIR_PROGRESS`'s docstring. After
P3 nothing writes it and its only remaining roles are signal 1 and the deferred
readers. A reader meeting it has no way to learn either fact, and the careful
treatment three files over shows the standard the project holds itself to. Minor,
and cheapest to fix beside the `DIR_IMAGE_RECORDS` note already at `:672-675`.

---

## 3. `_COMBOS` is 1152, not 384 — derivation rewritten, **but the document now contradicts itself**

### 3.1 The derivation was genuinely rewritten — **(a)**

`phase-3-per-image-record.md:1087-1110` (rewritten in `1cc6740c`) does not patch
a digit. It states the new product `9 store/table × 2 × 2 × 2 × 4 layers × 2 × 2
= 1152`, names each of the three gaps found at execution with the branch each
one unblocks (`:248-257` for the measurement table, `:235-238` for the fifth
store state, `:220-224` for the objmap split), explains why the store/table axis
is **coupled rather than crossed** (9 combinations, not 10, because the table
lives inside the store), cites the repo's own prior instance of the same defect
(`test_staged_resume_parity.py:26-32`), and records that the licensed
evidence-based reduction was measured and **not taken**. Step 4 (`:1148`) asserts
the same number.

The arithmetic checks against the shipped code:
`tests/unit/cli/test_staged_resume_equivalence.py:120-165` — five store states,
`_STORE_TABLE_COMBOS` collapsing to nine, four `_LAYER_STATES`, and an eight-axis
`_Key` (`:170`). 9 × 8 × 4 × 4 = 1152. The commit message's
`9 store/table × 8 signal × 4 layer × 2 markers × 2 work-id` is the same product
regrouped.

### 3.2 The Step 1 snippet still derives 384 — **(b)**

Eighty lines above the corrected prose, Step 1's code block is unchanged:

```
:1007   _STORE_STATES = ["absent", "stage1_only", "matching_work_id", "mismatched_work_id"]
:1008   _LAYERS = [None, "objmap", "rgb"]
:1010   _COMBOS = [ ...seven axes, no table axis... ]
:1024   #: The key is the full seven-axis tuple, matching _COMBOS.
```

Four stores × 3 layers × 2⁴ = **384**, and a **seven**-axis key against the
shipped **eight**. The CAN-16 comment above it (`:990-1006`) still lists the
measurement table in prose and omits it from the product — the exact defect
`:1092-1095` was written to record.

This is precisely the check the brief asked for, and it fails: *a reader who
checks the arithmetic against the snippet concludes the gate is wrong.* It is
also **entry 25 reproduced one edit later** — *"a document that has been revised
states its load-bearing facts more than once, and revision updates one site"* —
in the same file where entry 25 was discovered. The fix is to update the Step 1
snippet's three lines to the shipped axes, or to replace it with a pointer to
`test_staged_resume_equivalence.py:120-165` (a real symbol, which cannot rot the
same way).

**Nothing about the gate itself is wrong.** The commit message's added guarantee
— both captures verify their subject by content hash, so *"the collapse preserved
every decision"* and *"the file swap never happened"* cannot print the same
success message — is the right property for a freeze harness, and the measured
inertness of the `valid_image_success` axis (called 576/1152, true in 0) is
reported as a measurement rather than an argument.

---

## 4. Four deferred consumers with tripwires — **three of four, and the fourth needs no tripwire**

Enumerated by `grep -rn "UNTIL_P" tests/ --include=*.py`:

| Consumer | Marker | Owning phase named? | Strict? |
|---|---|---|---|
| `--mode recompile`, non-SLURM | `test_cli_recompile.py:53` | yes, P4 | yes |
| `--mode recompile`, SLURM | `test_cli_recompile_slurm.py:51` | yes, P4 | yes |
| `refresh_success_markers_after_metadata_migration` | `test_cli_completion_store.py:468` | yes, P7 (REUSE-F10) | yes |
| `_hdf_to_zarr._republish_image_marker` | `test_migration_republishes_state.py:52` | yes, P7 (U-10) | yes |
| **the 2 GUI readers** | **none** | named only in a comment | — |
| the arming coupling | `test_schema_gate.py:930` | yes, P7 Task 5 Step 1b | yes |

**The four that exist are exemplary** and satisfy category E rule 3 in its strong
form. Each reason states the *mechanism* of the deferral, not just its existence
— `test_cli_completion_store.py:437-467` is the model: it names the two-part fix
(path **and** `SUCCESS_MARKER_VERSION` → `RECORD_VERSION`, because repointing
only the path leaves every record failing the version check and `continue`-ing),
states that the function has zero production callers so nothing user-facing is
broken meanwhile, names the dangerous failure *direction*, and records that the
third of its three tests **had to be given a positive control** before it would
fail for the real reason rather than staying vacuously green.

**The GUI entry is the inaccuracy — (b).** The commit message says *"Four
deferred consumers keep `xfail(strict=True)` tripwires … 7 recompile readers
(P4), **2 GUI readers (P6)**, the refresh bridge and the migrator (P7)."* The GUI
readers have no tripwire. They are named once, in a comment at
`tests/unit/cli/test_cli_completion_store.py:437-438`.

**The absence is defensible; the claim is not.** The two readers are
`gui/run_console/_slurm_observer.py:29` (importing `stage3_completion_exists`
from `_cli`) and its use at `:1349`. P3 kept the function *name* and repointed
its *body* onto `stages.stage3`, exactly as
`phase-3-per-image-record.md` Task 3 Step 3 instructed (*"Keep the function
names — the SLURM observer imports `stage3_completion_exists`, and renaming it
is P6's job, not this task's"*). So the GUI reader is not broken and there is
nothing for a tripwire to catch. What is deferred is the *layering* fix (the GUI
still reaches into `phenotypic._cli`), which is P6 Task 6's scope and is not a
P3 regression.

**Nothing was deferred silently.** Every deferral I could find is recorded with
its owning phase. The one correction needed is to the commit message's count —
three tripwires, not four, and the GUI item is a layering deferral rather than a
broken-reader deferral.

---

## 5. `task.marker_path` keeps its legacy meaning — documented at the read-backs, **with one site missed**

### 5.1 The repointing is documented where it happens — **(a)**

The distinction is stated at the sites that matter, each time naming the hazard:

- `_cli_migrate_manifest.py:1021-1026` — *"`task.marker_path` is the LEGACY
  marker the task reads as input, and after P3's clean break the two are
  different files — agreeing on the wrong one binds the seal silently."*
- `_cli_migrate_image.py:641-648` — records that comparing against the task field
  *"made this guard fire unconditionally after P3's clean break — `--mode
  migrate` raised on the first image of every tree"*, and that the guard is kept
  because it caught exactly what it exists to catch.
- `_cli_migrate_image.py:645-654` — *"this value becomes
  `MigrationImageResult.marker_digest`, which the controller re-derives and
  compares at six sites … **All seven must digest the same file** or the seal
  binds the wrong one — silently, if they agree on the wrong file, and loudly if
  they disagree."*
- `_cli_migrate_image.py:718-739` (`_current_marker_digest`) — names it as the
  read-back that **guards a deletion**, and that the digest comparison is
  load-bearing *for the first time since the break*.

`git show 1cc6740c -- src/phenotypic/_cli/_cli_migrate_manifest.py` confirms six
`task.marker_path.read_bytes()` → `image_record_path(...)` conversions.

### 5.2 The dataclass field itself is undocumented — **(c), minor**

`_cli_migrate_manifest.py:56-66`: the class docstring is *"One migration target
and its canonical artifact paths."* After P3 that sentence is misleading for one
of its seven fields — `marker_path` is the only one naming a **legacy input**
rather than a current canonical artifact. A P7 reader meets the field before they
meet any of the four comments above. One line on the field closes it.

### 5.3 `_cli_migrate.py:976` is an unconverted read-back — **(c)**

`_cli_migrate.py` **is not in P3's diff** (`git diff --name-only
e74bf706..1cc6740c -- src` — 14 files, not including it), yet it contains a
read-back of the same value:

```
_cli_migrate.py:969   def _retained_reclaim_result(output_dir, task, result) -> ReclaimResult:
_cli_migrate.py:976       marker_digest = hashlib.sha256(task.marker_path.read_bytes()).hexdigest()
```

That value becomes `ReclaimResult.marker_digest`. Its **sibling producer** and
its **validator** were both repointed in P3:

| Role | Site | Digests |
|---|---|---|
| producer, clean path | `_cli_migrate_image.py:776` → `_current_marker_digest` (`:742-746`) | the **record** |
| producer, retained path | `_cli_migrate.py:976` | the **legacy marker** |
| validator, both paths | `_cli_migrate_manifest.py:1453-1474` (repointed in this commit) | the **record** |

`_validate_reclaim_result` requires `result.marker_digest == current_marker_digest`
unless `retained_after_unclean_image`, which needs **both** to be `""`
(`:1463-1468`). On a migrating tree where the image was published, the record
exists (non-empty) and the legacy marker exists — `_hdf_to_zarr._republish_image_marker`
still writes it, which is what the P7 migrator tripwire is about — so the two
digests differ and the comparison at `:1471-1474` appends *"reclaim result marker
digest does not match current bytes"*.

**Reachability:** `_retained_reclaim_result` is called from `_cli_migrate.py:1872`
(unclean image seal) and `:1885` (reclaim raised), both under `--delete-sources`,
and from `_cli_migrate_worker.py:705`. The resulting `ValueError` is caught at
`_cli_migrate.py:1899` and recorded as a reclaim failure, so it **fails in the
safe direction** — sources are retained — but attributes the retention to a
digest mismatch that is an artifact of the split repoint rather than to the
condition that actually caused it.

**Why nothing caught it — the register's own named mechanism.** The two tests
that exercise this function (`test_cli_migrate_authority.py:744`, `:791`) plant
`hdf_path` and `measurement_path` and **neither a legacy marker nor a record**.
So `task.marker_path.read_bytes()` raises `OSError` → `marker_digest = ""`, the
record is absent → `current_marker_digest = ""`, `retained_after_unclean_image`
is True, and the branch that would disagree is never entered. That is
*"presence and reachability differ by a branch"* — the mechanism the drift
register names as the single thread behind most of its entries — reappearing in
the phase that named it.

**Also note the plan predicted the boundary and the boundary is where it broke.**
The consumer table (`phase-3-per-image-record.md`, *Every consumer of the marker
surface*) lists ``_cli_migrate.py`, `_cli_migrate_image.py` | 4 | migrate's own
publisher (P7)`` — one row for two files, deferred as a unit. Execution
converted `_cli_migrate_image.py` and `_cli_migrate_manifest.py` (it had to: the
migrator calls `publish_image_success`, which moved) and left `_cli_migrate.py`
deferred. The pair was split, and the two halves now digest different files. **No
tripwire covers this** — the migrator's `xfail` is scoped to
`_republish_image_marker`.

**Recommended verification** (I did not run it; commands with side effects are
yours):

```
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_cli_migrate_authority.py -q
```

and, to make the branch reachable, a variant of
`test_reclaim_noop_records_missing_marker_without_deleting_sources` that plants
**both** `image_record_path(run, ds, stem)` and `task.marker_path` with different
bytes before calling `_retained_reclaim_result`. I expect
`publish_migration_reclaim_status` to raise *"reclaim result marker digest does
not match current bytes"*. If it does, the fix is one line at `_cli_migrate.py:976`
plus that test; if it does not, I have mis-read `retained_after_unclean_image`
and the finding reduces to 5.2.

---

## 6. Undocumented drift the brief did not name

### 6.1 The `work_id` merge fence was not built — **(c)**

**Plan:** `phase-3-per-image-record.md`, Task 1 Step 3, rule 4 (the rule's own
heading is *"Merging is fenced on `work_id`, and `consume_stage` is
idempotent"*), concluding:

> *"Fence the merge on **`work_id`**: entries from a record whose `work_id`
> differs are not merged forward."*

**Code:** neither writer fences.

- `_cli_image_record.py:158-159` — `merged = _existing_stages(...)` then
  `merged.update(...)`. `_existing_stages` (`:62-76`) takes no `work_id` and
  compares nothing.
- `_cli_image_record.py:216-227` — `record_stage` does
  `record = dict(existing)`, carrying the prior record's `stages` **and its
  `artifacts` and `work_id`** forward wholesale, then merges one stage in.

`consume_stage`'s idempotency (rule 4's other half) **is** implemented and
documented (`:230-273`). The fence is not, and there is no test, no comment, no
`EXECUTION.md` ruling and no drift-register row explaining the omission.

**Actual exposure is narrower than the rule implies, which is probably why it
slipped.** `record_rejection` (`sdk_/_image_record.py:180-183`) compares
`work_id` at *read* time, so a record carrying a stale `work_id` cannot certify
an image. And the plan's own rule-4 discussion argues at length that the stale-
`stage2` hazard is already covered by FLOW-40's raw-presence branch
(`_cli_staged_resume.py:279-283`) and by the store's `work_id` check. What the
missing fence permits is narrower: `publish_image_record` stamps the **current**
`work_id` onto a record whose `stage1`/`stage2` entries came from a superseded
work id, producing a record that reads as internally consistent and is not.

**Disposition: row 3 of EXECUTION.md's table** — a plausible rationale exists,
but there is no experiment and no ruling, so *"neither the reviewer nor the
orchestrator may ratify a deviation after the fact by finding the reasoning
persuasive."* This one is the user's call: either build the fence, or record the
argument above as the ruling and amend the plan.

### 6.2 `stage2_done/` and `stage2_raw/` appear in no register heading — forward risk, P7-owned

U-9 makes both trees permanent. P7 Task 6's register
(`phase-7-migrate-mode.md:1347-1394`) has four headings — tracked state, content
proofs, derived, and *"neither tracked nor derived"* — and neither tree appears
under any of them. `grep -n "stage2_done\|stage2_raw"` over that file returns
three hits, all in unrelated prose (`:163`, `:436`, `:505`).

Not a P3 finding — P3 correctly retained them per U-9 and correctly moved
`DIR_STAGE2_DONE` into `_io_constants.py:686` with a docstring explaining why its
treatment differs from `stage3_complete/`'s. Flagged so P7 does not author a
register whose *"Four. If a fifth appears, that is a design regression"* claim is
silently incomplete: the stage-2 token is a per-image file that
`classify_staged_image` **branches on**, which by the HARD STOP's own test 1
makes it look like tracked state until the register says why it is not.

---

## 7. Standing rules — compliance

### State-artifact budget (HARD STOP): **within budget, and P3 net-decreases it**

| Change | Effect |
|---|---|
| `image_complete/` — writer removed (D1 clean break) | −1 written tree |
| `stage3_complete/` — writer removed | −1 written tree |
| `images/` (`DIR_IMAGE_RECORDS`, `_io_constants.py:674`) | +1, and it is content proof #1 in the register's (b), which already budgeted it |
| `processing_state.datasets.{completed,failed,errors}` | −3 keys, and (c) explicitly lists these as *deleted and must not come back* |
| `stage2_done/`, `stage2_raw/` | retained, unchanged (U-9) |

Net: two written trees and three state keys removed, one added and already
budgeted. Tracked state is unchanged at four —
`config.work_ids` / `terminal_failures.jsonl` / lifecycle+ownership /
`restart_epoch.json`. No fix in this phase raised the count.

The one field P3 adds that something branches on is `provenance` inside the
record (`record_rejection` reads it via `record_provenance`). It is spec-mandated
(U-10, `design.md` §0) and P7 Task 6 Step 1 already requires documenting it, so
it is accounted for rather than a fifth state.

### INV-DISCHARGEABLE: **honoured, and it is the reason for the phase's largest divergence**

The disarm ruling exists precisely because arming would emit a verdict migrate
cannot discharge (§1.1). Signal 3's evidence was removed writer-side rather than
by dropping the signal, which is the fix the plan required
(`_cli_state_management.py:81-109`) and which closes the
migrate → run → refused loop at its source.

**One correction to the plan's callout was made during execution and is
recorded**, and it is the right kind of correction: the callout said *"stop
writing the four keys"* over a justification (*"the reader re-aggregates from the
event log"*) true of only three. `initial_images` is the accepted inventory —
tracked state #1 — and is not derivable from the event log. The comment at
`_cli_state_management.py:88-103` states this, names the callout as the source of
the error, and `test_image_record.py:469-478` asserts the inventory survives the
round trip.

### INV-VERDICT: **honoured**

Every reader degrades downward. `read_image_record` (`sdk_/_image_record.py:78-97`)
maps absent, truncated, non-object and `OSError` all to `None`;
`_existing_stages` (`_cli_image_record.py:62-76`) maps unreadable to `{}` with
the reason stated (*"a stage is then re-run rather than skipped on the strength
of a file nobody could parse"*); `consume_stage` returns `False` rather than
raising. `record_provenance` (`:95-121`) defaults to the **strict** reading, so a
writer that forgets the field produces a fenced record — the direction U-10
requires. §6.1's missing merge fence is the one place a verdict could improve
without verification, and it is bounded by `record_rejection`'s read-time
comparison.

### All reusable state-tracking checks live in `sdk_`: **satisfied**

`sdk_/_image_record.py` holds the readers and the whole vocabulary; writers are
in `_cli/_cli_image_record.py`; the module docstring states the split is forced
by INV-LAYER + P6 Task 0 rather than stylistic. `record_rejection` is *"the
single implementation of per-image record validity"* and is called by both
`_cli_completion.valid_image_success` and `_run_state`'s deep path — the merge
gate finding IMPL-F3 achieved, preserved. `ARTIFACT_KIND_*` was correctly **not**
moved (eight existing importers from `_io_constants`), while `_STAGE_MEASURED`
and `_PROVENANCE_MIGRATED` **were** removed from `_run_state.py` — the same rule
producing opposite answers, with the reasoning recorded in `9480dd5b`'s message.

INV-LAYER's AST walk covers the new module:
`tests/unit/sdk_/test_run_state_layering.py:54` — `Path(image_record.__file__)`
is in `_MODULES`. (Drift register F6 records this as having been missed and
fixed; it is present now.)

`marker_rejection` is retained in `_run_state.py` with an explicit ⚠ block saying
it has **no caller in `src/` as of P3 and is not dead code**, naming REUSE-F10
and P7's migrator as the two readers that should be calling it — which is rule 4
of the shared-helper policy (*a restatement that ships anyway carries the reason
and the name of the definition it mirrors*) applied to a helper with no callers
rather than a duplicate.

---

## 8. Category E — the placeholder sweep

Run over the 14 `src/` files in `e74bf706..1cc6740c`:

```
grep -rn "NotImplementedError\|TODO\|FIXME\|XXX\|placeholder\|for now\| stub" <files>   → no hits
grep -rn "^\s*\.\.\.\s*$\|^\s*pass\s*$" <files>                                          → no hits
```

Every symbol the plan named exists with a body I can summarise in one sentence.
The four shapes:

1. **Constant where derivation is required** — none. The `return False` sites in
   `_cli_image_record.py:260,263` are `consume_stage`'s documented absent/unreadable
   arms, and `return True` at `:273` follows an actual write.
2. **Guard with no consequence** — none new. The nearest thing is
   `_cli_schema_gate.py:81`'s early return on the unarmed flag, which is the
   *subject* of §1 rather than a placeholder: it is a ruled, marked, tripwired
   gap, and EXECUTION.md's category-E rule 3 names exactly that distinction.
3. **Skipped/xfailed test with no removal note** — none. All six `xfail` markers
   (§4) carry the condition and the owning phase, and all are `strict=True`, which
   is stronger than rule 3 requires.
4. **Parameter accepted and never read** — none found.
   `publish_image_record`'s `provenance`, `source_provenance`, `pre_replace` and
   `commit_guard` are each read at `:159-176`.

**But categories E and (b) are not disjoint, and this phase's placeholders are
in prose.** §1.4 and §1.5 are the E-shaped defects of this phase: text that
*reads* as specification and constrains nothing, or worse, constrains the wrong
thing. `_schema_shape.py:67-71` reads as a statement of the flag's value and
the coupling that enforces it, and is wrong about both. `:73-77` reads as the
measured precondition for arming and inverts it. Neither has a failure mode until
a P7 implementer acts on it — which is the register's own closing observation
that *"prose is not executed, so a false sentence has no failure mode until a
human acts on it."*

---

## 9. Findings, ordered

| # | Location | Category | Severity |
|---|---|---|---|
| 1 | `sdk_/_schema_shape.py:73-77` — "two of the five signals fire" is now **zero**; both cited writers were stopped by this commit; contradicted by `test_image_record.py:650`, which exists to be P7's arming evidence | **(b)** | **High** — inverts the precondition P7 Task 5 Step 1b decides on; third recurrence of IMPL-F6/SPEC-C3 in this docstring |
| 2 | `_cli_migrate.py:976` — `_retained_reclaim_result` still digests `task.marker_path` while its sibling producer and its validator were repointed to `image_record_path`; existing tests exercise only the both-absent branch | **(c)** | **High** — a split repoint across a deferred/converted boundary; fails safe but reports a false cause |
| 3 | `sdk_/_schema_shape.py:67-71` — "Flipped to `True` by P3 Task 2", a test name that does not exist, and "which fails" of an `xfail(strict=True)`; contradicted 47 lines below | **(b)** | Medium-high — first paragraph of the flag's own docstring |
| 4 | `_cli_image_record.py:158-159, 216-227` — plan Task 1 rule 4's `work_id` merge fence not implemented; no test, comment, ruling or register row | **(c)** | Medium — bounded by `record_rejection`'s read-time compare; disposition is the user's (row 3) |
| 5 | `phase-3-per-image-record.md:722-763` — "This task must ARM the schema gate" still standing as an instruction after the reversal | **(c)** | Medium — entry-24 shape, and it fails entry 24's own "every ruling gets the grep" |
| 6 | `phase-7-migrate-mode.md` Task 5 Step 1b — no arming instruction anywhere in the phase the code designates as the owner | **(c)** | Medium — nothing makes P7 arm; the tripwire only fires if it does |
| 7 | `phase-3-per-image-record.md:1007-1008, 1024` — Step 1 snippet still derives 384 and a seven-axis key, 80 lines above the corrected 1152 derivation | **(b)** | Medium — the exact reader-checks-the-arithmetic failure the brief asked about |
| 8 | `_cli_completion.py:238-239` — "every writing mode now refuses" and "`SCHEMA_GATE_ARMED` flips in this same commit", both false as shipped | **(b)** | Low-medium |
| 9 | commit `1cc6740c` message — "Four deferred consumers keep `xfail(strict=True)` tripwires"; three do, and the GUI item is a layering deferral with nothing to trip | **(b)** | Low |
| 10 | `_cli_migrate_manifest.py:56-66` — `marker_path`'s legacy-input meaning absent from the dataclass a P7 reader meets first | **(c)** | Low |
| 11 | `sdk_/_io_constants.py:669` — `DIR_IMAGE_COMPLETE` has no comment; nothing writes it after P3 and its only roles are signal 1 and the deferred readers | **(c)** | Low |
| 12 | `phase-7-migrate-mode.md:1347-1394` — `stage2_done/` and `stage2_raw/` in no register heading, though `classify_staged_image` branches on the token | forward risk, P7-owned | Low |

**No finding contradicts the spec's design.** Every item is either a document
that did not follow a ruling the code did follow (1, 3, 5, 6, 7, 8, 9, 10, 11),
one plan rule not built with no decision recorded (4), or one call site the
deferred/converted split missed (2). The record schema, the reader/writer layer
split, the clean break, the equivalence gate, the artifact budget and the two
INV- invariants are all as specified.

---

## What this phase did that is worth carrying forward

Stated because the disposition table above is one-sided by construction, and
three of these are the reason the findings are as few as they are.

- **The four production regressions in the commit message were found and fixed
  inside the phase, and each is named with its mechanism** — the `--mode measure`
  guard on a marker nothing writes, the same guard on recompile, migrate raising
  on the first image of every tree, and the `initial_images` wipe caused by a
  plan callout whose justification covered three keys of four. None was caught by
  a failing test, and the last one is a plan defect corrected in code with the
  plan's error named rather than silently worked around.
- **Both equivalence captures verify their subject by content hash.** Without it,
  *"the collapse preserved every decision"* and *"the file swap never happened"*
  print the same success message. That is the register's *"what would this have
  looked like if it had failed?"* made executable.
- **The one hybrid risk in the capture was measured, not argued** —
  `valid_image_success` called in 576/1152 cells and true in 0, so the frozen
  baseline is inert where it could have been contaminated.
- **A vacuous green was found and given a positive control**
  (`test_cli_completion_store.py:456-464`): a bridge that examines nothing
  returns 0 exactly as one that examines a store and finds it unchanged, and
  *"a vacuous green between two reds reporting the same cause is worse than a
  third red."*
- **`_DIR_STAGE3_COMPLETE`'s lost anchor test is recorded as a gap rather than a
  simplification**, with the failure direction named and the remaining ground
  truth identified.
