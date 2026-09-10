# P2 gate — state-check reuse

**Question:** is every state-tracking check reached through a reusable helper, or
do some reimplement it inline?

**Reviewer:** independent (did not write this code). Scope is *reuse of the
checking logic*, not correctness and not spec adherence — those are covered by
`p2-implementation-review.md` and `p2-spec-adherence.md`.

**Worktree:** `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/cli-gui-state-tracking`,
branch `cli-gui-state-tracking`, HEAD `e1010b8e`.

---

## Answer

**No.** All six questions have at least one check answered in more than one
place. Four of those sets **already disagree on inputs available today** — F1
(a crashed finalizer, or a v1 proof), F2 (every input), F5 (any non-ASCII dataset
name), F12 (any forward staged run with `staged_stage3_markers=False`). Two more
are on a schedule: F3 fires when P7 writes its first migrated marker, F4 when P4
bumps the aggregate publisher.

One duplication is genuinely well-managed and is worth naming as the
counterexample: the staged path's `stage2_result_replayable` probe really is
single-sourced across six call sites in three modules, which is the *opposite*
result to F7's cycle claim and shows the discipline is achievable here.

The result is not empty and it is not cosmetic. The single sharpest instance:

> On a tree whose finalizer crashed between publishing the aggregate proof and
> rewriting the run proof, `gui/shell/_runs_registry.py:612-616` reports the run
> **complete** and `_cli_completion.valid_run_completion` (which
> `gui/results_viewer/_output_consistency.py:381` calls) reports it **not
> complete**. Two GUI surfaces, one tree, one moment, opposite answers — because
> the registry open-codes a two-field predicate where the shared validator checks
> seven.

A structural note that frames everything below: **`resolve_run_state` has zero
production callers at P2.** Every consumer still goes through the CLI validators
(`_output_consistency.py:375-381`, `_runs_registry.py:595`,
`_slurm_observer.py:1313-1319`, `_cli_gui_lifecycle.py:88-90`, and eleven more).
So P2 has added a tenth answer to "is this run done?" rather than replaced the
nine. That is what the plan says P6 Task 7 is for, and I am not filing it as a
defect — but it does mean the duplications below are **live in both directions**
for the whole interval between now and P6, and the findings are ranked on that
basis.

Four claims were additionally checked mechanically; **all four confirmed, nothing
struck** (see [Verification status](#verification-status)). One of them —
`_inventory_digest_for`'s cited import cycle — turned out not to exist, which
promoted [F7](#f7) from Medium to High: a duplication justified by a constraint
that is not there is a duplication nobody chose.

### Findings by severity

| # | Question | Severity | Can the duplicates disagree? |
|---|---|---|---|
| [F1](#f1) | 1 — run complete | **Blocking** | Yes, on trees that exist today |
| [F2](#f2) | 6 — `inventory_digest` | **Blocking** | Yes, on every input (they cannot ever agree) |
| [F3](#f3) | 3 — per-image record valid | **High** | Yes, on the shape P7 is scheduled to write |
| [F4](#f4) | 6 — `finalization_input_digest` | **High** | Yes, from P4's writer bump onward |
| [F7](#f7) | (justification) | **High** | The cited blocker does not exist — probe 1 |
| [F5](#f5) | 2 — identity current | Medium | Yes, on any non-ASCII dataset name — probe 3 |
| [F6](#f6) | 2 — identity current | Medium | Not today; nothing keeps them in step — probe 4 |
| [F12](#f12) | 5 — worklist | Medium | Yes — one of three copies stats a different file |
| [F8](#f8) | 4 — content proof | Low | No (equivalent), but three copies |
| [F9](#f9) | 4 — content proof | Low | Only if an aggregate output became a store |
| [F10](#f10) | 3 — per-image record valid | Low | No (identical), 130 lines apart in one file |
| [F11](#f11) | 5 — worklist | Low | Dead code, but it is a second definition |

### Coverage of the six questions in the brief

| # | Question | Sites found | Verdict | Where |
|---|---|---|---|---|
| 1 | Is this run complete? | 7 proof readers + 13 completion callers | **Duplicated, divergent** | [F1](#f1), [F4](#f4) |
| 2 | Is this identity current? | 2 enumerations, 1 hand-rolled digest, 1 open-coded generation fence | **Duplicated, latent divergence** | [F5](#f5), [F6](#f6) |
| 3 | Is this per-image record valid? | **5 readers** | **Duplicated, provably divergent** | [F3](#f3), [F10](#f10) |
| 4 | Is the content proof intact? | 3 copies | Duplicated, equivalent today | [F8](#f8), [F9](#f9) |
| 5 | Which images remain? | 6 probe calls (shared) + 3 guard copies + 1 dead | **Probe shared; guard divergent** | [F12](#f12), [F11](#f11) |
| 6 | Every digest the identity is built from | 5 families | **2 divergent, 3 clean** | [F2](#f2), [F4](#f4); clean ones under *no finding* |

**All six are now fully enumerated.** Question 5 was the last one open and it
resolved in two halves worth stating separately, because they point in opposite
directions:

* the `stage2_result_replayable` single-sourcing claim in `_cli/CLAUDE.md` is
  **true** — six calls across three modules, and nothing open-codes the two-half
  check. Its count is wrong (five named roles, six calls) and its unit is
  undefined, but the substance holds. This is the *opposite* result to
  [F7](#f7)'s cycle claim, and worth recording as such: a documented
  single-sourcing claim that survives checking.
* underneath it sits a **second, unshared** completeness guard with three
  variants, one of which stats a file forward runs never write — [F12](#f12).

So question 5 is not clean, but it fails for a reason nobody had written down,
rather than at the place the documentation invited scrutiny.

---

## Method

1. Enumerated every site that *decides* each of the six questions, by grepping
   for the on-disk keys each decision branches on (`finalizer_succeeded`,
   `SUCCESS_MARKER_VERSION`, `RUN_PROOF_VERSION`, `inventory_digest`,
   `work_ids`, `success_markers_required`, …) rather than for helper names — a
   grep for helper names finds only the sites that already reuse.
2. For each set, diffed the predicates clause by clause and looked for an input
   on which they return different answers.
3. Checked the tests cited as keeping duplicated implementations in step, to see
   whether they cover the inputs on which the implementations differ.
4. Verified, rather than accepted, the one documented "this cannot be shared"
   claim (`_inventory_digest_for`'s import cycle).

### A caution on the citations

Line numbers were re-pinned against the working tree immediately before writing.
`_cli_identity.py` and `_verification_cache.py` both changed on disk *during* the
review, and six gate fixes have landed since — including one that moved
`mint_run_identity` in `phenotypicCLI.py` by ~42 lines.

**Where a symbol name is unambiguous, this report cites the symbol.** Line
numbers appear only where the claim is about a specific expression rather than a
function, and they should be treated as of the moment of writing. None of the
landed fixes touches `inventory_digest`'s two producers, `RunIdentity.digest()`'s
`ensure_ascii`, or the two enumerations of the fenced fields, so no finding below
is affected — but a reader following a `phenotypicCLI.py` or `_cli_identity.py`
line number should expect drift and navigate by symbol.

---

<a name="f1"></a>
## F1 — BLOCKING. "Is the run proof valid?" has seven implementations; four omit
## the version check, and two of those are primary reads

`RUN_PROOF_VERSION` is **2**. It was already `2` as a bare literal before this
branch (`git show 17f144ef -- src/phenotypic/_cli/_cli_completion.py`, hunk at
`publish_run_completion_evidence`: `- "version": 2` → `+ "version":
RUN_PROOF_VERSION`). So **version-1 run proofs exist on trees written by an
earlier release**, and `_cli/CLAUDE.md` states the policy for them: "a version
mismatch invalidates rather than migrates".

The seven readers:

| # | Site | version? | status | finalizer_succeeded | digests |
|---|---|---|---|---|---|
| 1 | `sdk_/_run_state.py:825-836` `_valid_run_proof` | **yes** (`:830`) | `== "complete"` | `is not True` | via `_run_proof_covers_current_inventory:892` |
| 2 | `_cli/_cli_completion.py:1060-1104` `valid_run_completion` | **yes** (`:1080`) | `!= "complete"` (`:1069`) | `is True` (`:1078`, legacy arm only) | four, `:1085-1103` |
| 3 | `gui/shell/_runs_registry.py:612-616` | **no** | `!= "complete"` | `is not True` | **none** |
| 4 | `gui/shell/_runs_registry.py:685-689` | **no** | `!= "complete"` | `is not True` | none (legacy arm; consistent with #2's legacy arm) |
| 5 | `gui/results_viewer/_output_consistency.py:387-391` | **no** | `== "complete"` | `is True` | none |
| 6 | `gui/run_console/_slurm_observer.py:1263,1291-1292` | **no** | **five spellings** | `bool(...)` | none |
| 7 | `_cli/_cli_checkpoint_handler.py:388-393` | **no** | `== "complete"` | `is True` | none |

**The disagreement that is reachable today, without any version bump.**
`_runs_registry._local_completion_evidence_conflict` (`:594-620`) is the gate on
whether a zero-exit local run may be recorded complete. Its flow:

```
:597   marker_complete = current_run_is_complete(record.output_dir)
:603   if marker_complete is True:
:612       if not isinstance(marker, dict) or (
:613           marker.get("status") != "complete"
:614           or marker.get("finalizer_succeeded") is not True
:615       ):
:616           return "local completion marker is missing successful publication status"
:620       return None          # <- no conflict: publish complete
```

`current_run_is_complete` (`_cli_completion.py:750-766`) never reads the run
completion marker at all — it walks per-image markers and then calls
`current_aggregate_is_current`, which validates the **aggregate** proof. So on
this path the run proof's `version`, `inventory_digest`,
`scientific_config_digest`, `finalization_input_digest` and `publication_id` are
**never checked by anything**.

`valid_run_completion:1085-1103` checks all five. Concretely, a tree where the
finalizer published a fresh aggregate proof and then died before rewriting the
run proof — a window the idempotence short-circuit at
`_cli_completion.py:1039-1055` exists precisely because of — has a current
aggregate and a stale run proof whose `publication_id` no longer matches. Then:

* `current_run_is_complete` → **True** (the aggregate is current)
* `_runs_registry.py:612-616` → no conflict → the shell marks the run
  **complete**
* `valid_run_completion` → **None** on the `publication_id` comparison
  (`:1098,:1102`)
* `_output_consistency.py:381,435` → `completion_marker_valid=False`; the
  results viewer reports the run's completion evidence absent

Same tree, same instant, two GUI surfaces, opposite answers. A v1 run proof
produces the same split by a different clause.

`_slurm_observer` (#6) is loose along three independent axes at once — it accepts
`{"complete", "completed", "success", "succeeded", "ok"}` (`:1263`) where every
other reader accepts only `"complete"`, and uses `bool(marker.get(...))` (`:1292`)
where every other reader uses `is True`. It is *gated* by `valid_run_completion`
before it returns `"complete"` (`:1317-1319`), so it cannot manufacture a false
complete; its failure mode is the opposite one — a marker it accepts and the
shared validator rejects leaves the run wedged in `"reconciling"` with no
diagnostic naming the version.

**Proposed helper.** One predicate, in `sdk_/_run_state.py` beside
`_valid_run_proof`, exported and imported by the CLI half (the direction
INV-LAYER permits):

```python
def run_proof(output_dir: Path) -> Mapping[str, object] | None:
    """Return the structurally valid run proof, or None.

    Structural validity only -- version, status, finalizer_succeeded. Whether
    the proof still *covers* the current inventory is a separate question with
    a separate function, because the two have different callers: a caller
    asking "is this file a run proof at all?" must not be forced to load
    processing state.
    """
```

`valid_run_completion` keeps its digest comparisons but obtains the marker from
this function; `_runs_registry.py:611-616`, `_output_consistency.py:387-391`,
`_slurm_observer.py:1291-1292` and `_cli_checkpoint_handler.py:388-393` call it
instead of re-reading and re-branching. The two `_runs_registry` sites and the
observer additionally need the digest half — they should call
`valid_run_completion` outright rather than a second structural check.

---

<a name="f2"></a>
## F2 — BLOCKING. `inventory_digest` has two producers that cannot agree on any
## input, and the minting side's value has no consumer

This is the instance the brief named. It is worse than "they compute it
differently": they compute **values in different domains**, so they can never be
equal.

| Site | Expression | What it digests |
|---|---|---|
| `sdk_/_run_state.py:276` | `canonical_digest(config.get("work_ids", {}))` | the nested `{dataset: {image: work_id}}` **map** |
| `_cli/_cli_completion.py:740,904,1012,1086` | `canonical_digest(work_ids)` | the same map — agrees with `_run_state` |
| `_cli/_cli_identity.py:148-165` `_inventory_digest_for` | `canonical_digest(<manifest digest str>)` | a single 64-char **hex string** |

`canonical_digest` of a mapping and `canonical_digest` of a string are digests of
different JSON documents. `RunIdentity.inventory_digest` is in
`_IDENTITY_DIGEST_FIELDS` (`_run_state.py:290-296`), so it is compared by
`assert_identity_current` (`:326-332`) and folded into `RunIdentity.digest()`
(`_state_types.py:75`), which keys the verification cache
(`_run_state.py:1124,1147,1154`).

**Why it has not exploded yet, and why that is the finding.** Two facts, both
verified:

1. `assert_identity_current` has **no production caller** — every use is in
   `tests/unit/sdk_/test_run_state.py`, and those construct the identity via
   `run_identity(...)` (the reader), never via `mint_run_identity`.
2. Of the seven fields `mint_run_identity` returns, only `processing_generation`
   and `restart_epoch` are ever read. `phenotypicCLI.py` touches `identity` at
   `:2446`, `:2449`, `:2744`, `:2745`, `:2761`, `:2787` — all
   `.processing_generation` or `.restart_epoch`. `create_initial_state`
   (`_cli_state_management.py:183-266`) persists those two — at `:256` and
   `:261` — and none of the three digest fields.

So `_inventory_digest_for`, and the inline `finalization_input_digest` at
`_cli_identity.py:296-303`, and `scientific_config_digest` at `:295`, are **dead
computations that also disagree with the reader**. The moment any later phase
threads a minted identity into `assert_identity_current` — which is the
documented plan — every call raises `inventory_digest changed: expected <digest
of a hex string>, found <digest of a map>` on a run whose configuration never
moved. Because the two are structurally incapable of agreeing, this cannot be
caught by any equality test that happens to be written later against a
round-tripped identity; it needs a mint-then-assert test, which does not exist.

**Proposed helper.** The shared definition is the *accepted inventory*, and both
halves already have it — `work_ids`. One function, in `sdk_/_run_state.py`
(a pure reader, no `_cli` dependency, so both sides may import it):

```python
def inventory_digest(work_ids: Mapping[str, Mapping[str, str]] | None) -> str:
    """Digest the accepted inventory: {dataset: {image_name: work_id}}.

    THE input is the work-id map, never the image-manifest digest. The manifest
    digest answers "did the input scope change?"; this answers "which images did
    this run accept?", and they are different questions with different
    lifecycles -- an image can enter the manifest and be rejected by the scanner.
    """
```

`_run_state._identity_from:276` and `_cli_completion.py:740,904,1012,1086` call
it (a no-op change in value). `mint_run_identity` calls it too — but it must be
given the `work_ids` map, which does not exist at mint time. That is the real
shape of this defect, and it forces a choice the next phase has to make
explicitly:

* **(a)** `mint_run_identity` returns `inventory_digest=""` and the field is
  documented as "populated by the reader, empty at mint", with
  `assert_identity_current` skipping empty tokens; or
* **(b)** minting moves to after `state.config["work_ids"]` is built
  (`phenotypicCLI.py:2769-2777`), which is a real reordering because
  `identity.processing_generation` is consumed at `:2446` and `:2744`, before
  that point; or
* **(c)** `RunIdentity` is split into the mint-time subset and the read-time
  full identity.

I recommend **(a)**: it is the only one that does not reorder the entry point,
and it makes the current situation honest instead of silently wrong. Whichever is
chosen, `_inventory_digest_for` should be **deleted**, not repaired — its value
is not an inventory digest under any definition either half uses.

---

<a name="f3"></a>
## F3 — HIGH. Per-image record validity has five readers, two of which provably
## disagree, and the test cited as keeping them in step excludes exactly that input

`sdk_/_run_state.py:21-31` states the duplication openly and names its keeper
test. The duplication is real and, on one input, so is the divergence.

The five readers:

1. `_cli_completion.valid_image_success:256-303` — the CLI authority, ~20 call
   sites (`_cli_staged_strategy.py:98,434`, `_cli_staged_resume.py:212,438`,
   `_cli_staged_slurm_worker.py:252,364`, `_cli_state_management.py:463`,
   `_cli_recompile_tables.py:148`, `_cli_recompile_recovery.py:395`,
   `_cli_failure_tracker.py:368`, `_dashboard/_manifest_builder.py:617,662`,
   `sdk_/_hdf_to_zarr.py:790`, `_cli_migrate_image.py:222`,
   `_cli_migrate.py:1440`, `_cli_completion.py:695,798,831`)
2. `sdk_/_run_state._marker_rejection:480-508` + `_fenced_artifact_path:421-477`
3. `_cli_completion.refresh_success_markers_after_metadata_migration:400-410`
   (see [F10](#f10))
4. `_cli_recompile_recovery._marker_allows_table_transition:772-800`
5. `_cli_recompile_slurm_scripts.py:568-582`

**The divergence.** `_run_state.py:502-504`:

```python
if marker.get("provenance") != _PROVENANCE_MIGRATED:
    if marker.get("work_id") != work_id:
        return "marker was written for a different work_id"
```

`_cli_completion.py:269` compares `work_id` **unconditionally**. So for a marker
carrying `provenance: "migrated"` and a non-matching `work_id` — U-10's shape,
which `--mode migrate` is scheduled to write in P7 — the sdk reader **accepts**
and the CLI validator **rejects**. Readers 3, 4 and 5 also reject
(`:403`, `:782-788`, `:569-576` all compare `work_id` unconditionally).

The consequence when P7 lands: on a migrated tree, `resolve_run_state` reports
`complete` while every CLI resume path treats the same images as unprocessed and
reprocesses them — the migration's entire purpose, silently undone, with the GUI
insisting it is done.

**The keeper test does not cover it.** `test_the_sdk_reader_agrees_with_the_cli_validator`
(`tests/unit/sdk_/test_run_state.py:1085-1116`) is parametrized over exactly four
tamperings (`:1085-1093`): `untouched`, `marker-gone`, `overlay-rewritten`,
`store-root-rewritten`. The fixture that produces the divergent shape,
`_mark_migrated` (`:458-475`), sets both `provenance: "migrated"` **and** a
non-matching `work_id` — precisely the input on which the two implementations
differ — and it is not in the list. The test asserting the two agree therefore
cannot fail on the one input where they do not.

This is not a hole to be plugged by adding `_mark_migrated` to the
parametrization: doing so would make the test **fail**, correctly, because the
two really do differ by design. The fix is to make them one implementation.

**Proposed helper.** `_marker_rejection` is the better of the two — it returns a
sentence rather than a bool, which is what makes `ImageState.reason` useful — and
it already lives on the importable side of INV-LAYER. Promote it:

```python
# sdk_/_run_state.py, exported
def marker_rejection(
    marker: Mapping[str, object],
    *,
    work_id: str,
    dataset: str,
    image_stem: str,
) -> str | None:
    """Return why this marker cannot certify this image, or None."""
```

`valid_image_success` becomes `read marker → marker_rejection(...) is None →
every artifact still matches disk`, keeping its `bool` signature so its ~20
callers are untouched. Readers 4 and 5 keep their extra clauses (the recompile
paths legitimately check more) but obtain the shared clauses from this function.

**Until that lands**, the P6 Task 7 note in `_run_state.py:26` should say
explicitly that the two implementations are *known* to differ on the migrated
shape and that P7 must not ship before P6 Task 7 — otherwise the ordering
dependency exists only in this report.

---

<a name="f4"></a>
## F4 — HIGH. `finalization_input_digest` has four spellings; the sdk reader
## tolerates two of them and the CLI validator tolerates one

| Site | Spelling |
|---|---|
| `sdk_/_run_state.py:161-174` `_finalization_inputs` | **versioned** — `schema_version` + three values |
| `sdk_/_run_state.py:864-889` `_accepted_finalization_digests` | **both** — versioned *and* unversioned, deliberately |
| `_cli/_cli_identity.py:296-303` (mint) | **versioned**, but as an inline dict literal, not via `_finalization_inputs` |
| `_cli/_cli_completion.py:905-913` (aggregate publisher) | **unversioned** |
| `_cli/_cli_completion.py:729-737` (`current_aggregate_is_current`) | **unversioned only** |
| `_run_state.py:944-948` / `_cli_completion.py:1016-1018,1091-1093` | `{"process_only_layer": ...}` — a fourth shape, spelled twice each side |

`_accepted_finalization_digests`' docstring (`:869-876`) says P4 bumps the
publishers and drops the unversioned spelling from the accepted set. It does not
mention that `current_aggregate_is_current:729-737` computes **only** the
unversioned spelling and compares it for equality (`:741`). When P4 bumps
`publish_aggregate_snapshot:905` to the versioned form,
`current_aggregate_is_current` returns `False` for every run — and it is the
gate inside `current_run_is_complete:766`, which is what
`_runs_registry.py:597`, `_cli_gui_lifecycle.py:90`,
`_cli_checkpoint_handler.py:348,401`, `_cli_recompile_worker.py:677`,
`_dashboard/_manifest_builder.py:729` and `phenotypicCLI.py:2418,3781` all call.
Every completion surface in the product returns "not complete" on the day P4
lands, and `resolve_run_state` — tolerant of both — returns "complete". The
tolerance was added to one of the two implementations and not the other.

**Proposed helper.** `finalization_input_object` is already public
(`_run_state.py:177`, in `__all__`). Two additions beside it:

```python
def finalization_input_digest(config: Mapping[str, object]) -> str:
    """The digest a NEW proof carries. Versioned, one spelling."""

def accepted_finalization_digests(config: Mapping[str, object]) -> frozenset[str]:
    """Every spelling a proof already on disk may legitimately carry."""
```

Rule: **publishers call the first, validators call the second.**
`_cli_completion.py:905-913` and `_cli_identity.py:296-303` become calls to the
first; `_cli_completion.py:729-737,741` and `_run_state.py:950-954` become calls
to the second. That is what makes P4's bump a one-line change in one file
instead of a coordinated edit across two modules that nothing pins together.

The `{"process_only_layer": ...}` shape should be the same pair's `process`
branch rather than a literal repeated at four sites.

---

<a name="f5"></a>
## F5 — MEDIUM. `RunIdentity.digest()` hand-rolls `canonical_digest`, and
## differs on the one flag `_digests.py` calls load-bearing

`sdk_/_digests.py` exists to remove exactly this class of duplication — its
docstring (`:1-10`) records hoisting two copies into one "so that nothing could
disagree", and `:31-35` singles out `ensure_ascii=False` as **load-bearing
(ledger DF-19)**, because flipping it "would make `canonical_digest` disagree
with itself on any non-ASCII dataset name and invalidate every proof written by
the other half of the code".

`_state_types.py:79-83` then writes:

```python
return hashlib.sha256(
    json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()
```

— `canonical_digest`'s body with `ensure_ascii` left at its `True` default. A
fourth copy, differing in precisely the parameter the hoist's own docstring
identifies as the dangerous one.

**The concrete triggering input.** This is not hypothetical and it does not need
a code change to fire — it needs a filename.

`state.config["work_ids"]` is keyed by **dataset name**, which is a directory
name off `--input`, and by **image filename** (`phenotypicCLI.py`, the
`work_ids` comprehension: `{dataset.name: {image.name: ...}}`). A dataset
directory named `plaque-café/`, `Ölproben/`, `2026-05_Müller/`, or any of the
accented, umlauted or non-Latin names that occur routinely on a shared research
filesystem puts a non-ASCII character into `work_ids`, hence into
`canonical_digest(work_ids)`, hence into `RunIdentity.inventory_digest`, hence
into the payload `digest()` serializes.

At that point:

* `canonical_digest` (`ensure_ascii=False`) emits the literal character
* `RunIdentity.digest()` (`ensure_ascii` defaulted to `True`) emits `\uXXXX`

— two different byte strings, two different SHA-256s, for the same identity. The
verification cache is keyed on `digest()` (`_run_state.py:1124,1147,1154`), so
the effect is a permanently cold cache for every run under such a name: the
per-image deep pass the cache exists to avoid (audit's 1403 s at N=6,657, versus
~37 s warm) is paid on **every** GUI poll and **every** observer tick, forever,
with nothing failing and nothing logged. A performance cliff that looks like the
filesystem being slow.

Severity stays Medium only because the consequence is cost rather than a wrong
verdict — `digest()` is a cache key, not a fence. If any later phase folds
`digest()` into a comparison that gates a verdict, this becomes Blocking without
anything about it changing.

**Proposed fix.** `_state_types.py` imports `_digests` (both are leaves;
`_digests` imports only stdlib, so no cycle) and `digest()` becomes
`return canonical_digest(payload)`.

---

<a name="f6"></a>
## F6 — MEDIUM. The set of fenced identity fields is enumerated twice, and
## nothing keeps the two in step

* `_run_state.py:290-296` `_IDENTITY_DIGEST_FIELDS` — the tuple
  `assert_identity_current:326` iterates
* `_state_types.py:72-78` — the same five names, spelled again as dict keys

They agree today. `_run_state.py:284-289`'s comment says "Comparing exactly these
is what makes 'the identity is current' mean the same thing as 'the cache entry
may stand'" — which is true only while the two lists are identical, and nothing
enforces that. Adding a sixth fenced token to one and not the other splits the
two meanings silently: `assert_identity_current` would fence on a token the cache
key ignores (so a stale cache entry survives an identity change), or the reverse
(so the cache is discarded on a change nothing hard-errors on).

**Proposed fix.** `_IDENTITY_DIGEST_FIELDS` moves to `_state_types.py` beside the
dataclass, and `digest()` builds its payload from it:

```python
IDENTITY_DIGEST_FIELDS: Final[tuple[str, ...]] = (...)

def digest(self) -> str:
    return canonical_digest(
        {name: getattr(self, name) for name in IDENTITY_DIGEST_FIELDS}
    )
```

`_run_state` imports the name it already re-exports the types from. This folds
[F5](#f5) in at the same time. Note the value of `digest()` is **unchanged** by
this refactor only if F5's `ensure_ascii` change is made simultaneously — do them
in one commit, or the cache silently cold-starts once.

**What pins them today: nothing.** Probe 4 confirms the two lists are equal right
now, which is the entire strength of the current arrangement — an equality that
holds by attention rather than by construction. Deriving one from the other is
the fix; a test asserting they are equal is **not**, and should not be offered as
a substitute. The regression that matters here is someone adding a sixth token to
`_IDENTITY_DIGEST_FIELDS` and not to `digest()`, and an equality test would catch
that — but so would deriving, and deriving makes the class of bug unrepresentable
instead of merely detected. This is the same reasoning
`test_the_marker_schema_constants_have_exactly_one_home`
(`tests/unit/sdk_/test_run_state.py:1119-1171`) already applies to the marker
constants: it asserts *structurally* that `_cli_completion` assigns none of those
names, on the stated grounds that "equality alone would not catch the regression
that matters". Apply the same standard here.

This is the cheapest of the four probe-confirmed findings to fix and the easiest
to regress.

---

<a name="f7"></a>
## F7 — HIGH. The stated reason `_inventory_digest_for` cannot be shared does not
## exist. **Just import it.**

> Promoted from Medium on the strength of probe 1. The severity is not about the
> two lines of restated code; it is about the disposition. A duplication
> justified by a constraint that is not there is **a duplication nobody chose** —
> the author believed they had no option. The standing rule is that a *genuine*
> constraint means the shared definition needs a home neither module owns; **no
> constraint means just import it**, and this is the second case.

`_cli_identity.py:151-156`:

> Mirrors `_cli_state_management._image_manifest_digest_for`, and is
> **deliberately not imported from it**: `_cli_slurm_lifecycle` imports this
> module, so a module-level import of `_cli_state_management` here would close a
> cycle the moment `create_initial_state` takes a `RunIdentity`.

Two things are wrong with this.

**The cycle does not exist.** The module-level import closure of
`_cli_state_management` is `{_cli_types, _cli_update_state, _cli_file_locking,
_stages, _cli_directory_scanner, _cli_staged_resume, _cli_process_only,
_cli_stage2_token, _cli_failure_tracker}` — it contains neither `_cli_identity`
nor `_cli_slurm_lifecycle`. (`_cli_state_management.py:16-31`,
`_cli_directory_scanner.py:23`, `_cli_staged_resume.py:27-34`,
`_cli_process_only.py:29`, `_cli_update_state.py:31-33`; `_cli_process_only` and
`_cli_stage2_token` and `_stages` and `_cli_file_locking` and `_cli_types`
introduce no further local edges.) A module-level
`from ._cli_state_management import _image_manifest_digest_for` in
`_cli_identity` is legal today. `create_initial_state` already takes a
`RunIdentity` (`_cli_state_management.py:183-189`) and imports it under
`TYPE_CHECKING` (`:30-31`), so the "the moment it takes one" condition has
already occurred without producing a cycle. **Probe 1 confirms this
mechanically**: the closure has 10 members and contains neither module.

**It does not mirror it either.** `_image_manifest_digest_for`
(`_cli_state_management.py:36-44`) returns the manifest digest string or `None`;
`_inventory_digest_for` returns `canonical_digest(...)` of that. They differ by a
digest. "Only the two-line attribute lookup is restated" is not accurate — the
restatement adds the wrapping that makes [F2](#f2)'s divergence.

**Proposed fix.** Delete `_inventory_digest_for` per [F2](#f2). If a
manifest-digest lookup is still wanted at mint time for a *different* field, call
`_image_manifest_digest_for` directly and drop the docstring paragraph. The
general lesson, and the reason this is Medium rather than Low: a false
"cannot-share" justification is more durable than the duplication it excuses —
the next reader takes it on faith, as four review rounds did.

Where a helper genuinely cannot be shared across INV-LAYER, this repo already has
the right pattern and should follow it: `sdk_/_schema_shape.py` puts the pure
predicate in `sdk_` and leaves only the `click.UsageError` in
`_cli/_cli_schema_gate.py` (`_schema_shape.py:1-31`,
`_cli_schema_gate.py:10-22`), with a test asserting the CLI module re-declares
nothing. That is the shape every fix in this report should take: **the shared
definition goes in a module neither half owns.**

---

<a name="f8"></a>
## F8 — LOW. "Does this artifact still match its descriptor?" has three copies

* `sdk_/_run_state._fenced_artifact_path:421-477` — handles `file` and `store`
* `_cli_completion.valid_image_success:277-300` (+ `_store_artifact_matches:100-112`)
  — handles both
* `_cli_completion.valid_aggregate_snapshot:940-954` — file only

I diffed these clause by clause. **For the `file` kind they are behaviourally
equivalent**: `_stat_tuple`'s `stat.S_ISREG` check and `artifact.is_file()` agree;
`_digest_file` and `_sha256` are the same streamed SHA-256; both compare `size`
then `sha256`; both treat an escaping path as a rejection. For the `store` kind
`f"sha256:{_digest_file(root_json)}"` (`_run_state.py:463`) and
`file_fingerprint(root_json)` (`_cli_completion.py:112`, defined at
`_io_constants.py:168-181`) produce the same `"sha256:<hex>"` string.

So this is a tidiness note, not a defect — but it is three copies of the
containment-plus-size-plus-digest walk, and the `store`/`file` dispatch is the
part most likely to gain a third `kind`. Fold into the promoted
`marker_rejection` from [F3](#f3) by exporting `_fenced_artifact_path` alongside
it.

---

<a name="f9"></a>
## F9 — LOW. `valid_aggregate_snapshot` never reads `kind`

`_cli_completion.py:941-954` iterates `required_outputs` descriptors and applies
the file predicate unconditionally — no `descriptor.get("kind", ...)` branch,
unlike its sibling at `:286-300`. `sdk_/_valid_aggregate_proof:858-860` goes
through `_fenced_artifact_path`, which does dispatch on `kind`.

Unreachable today: `required_outputs` is always the four deliverable files
(`:881-886`). If an aggregate output ever became a store, the sdk would accept it
and the CLI would reject it. Fixed for free by [F8](#f8).

---

<a name="f10"></a>
## F10 — LOW. `_cli_completion` re-spells its own marker-identity check 130
## lines from the original

`refresh_success_markers_after_metadata_migration:400-410`:

```python
if (
    not isinstance(marker, dict)
    or marker.get("version") != SUCCESS_MARKER_VERSION
    or marker.get("work_id") != work_id
    or marker.get("dataset") != dataset
    or marker.get("image_stem") != stem
):
    continue
artifacts = marker.get("artifacts")
if not isinstance(artifacts, dict) or not artifacts:
    continue
```

is `valid_image_success:267-276` verbatim, in the same file. Identical today, so
it cannot disagree — but it is the cheapest possible fix in this report and it is
the same clause set [F3](#f3) is promoting anyway. Both call
`marker_rejection(...) is None`.

---

<a name="f11"></a>
## F11 — LOW. A second worklist definition, with no callers

`_cli_update_state.get_remaining_images:482-497` derives the remaining-image set
from `dataset_state.completed | dataset_state.failed`. `grep -rn
"get_remaining_images\b" src/ tests/` returns **only its own definition** — no
callers anywhere.

The live worklist is `_cli_state_management.get_remaining_images_for_datasets:402`,
which goes through `valid_image_success` (`:463`). The two answer the same
question from different evidence: markers versus
`processing_state.datasets.{completed,failed}` — and `_run_state.py:10-13` records
that spec §4.2 **demotes the event log out of the evidence set and deletes those
very fields from the file**. So this function is a second definition sourced from
evidence the change is removing. Delete it in P6 Task 7 rather than leaving a
plausible-looking helper that would silently return "everything remains" once the
fields are gone.

**One genuine second answer that is not dead**:
`_cli_staged_strategy._terminal_output_exists:95-135` tries `valid_image_success`
first (`:98-104`) and, on `False`, falls through to an independent acceptance
path — `stage3_completion_exists AND staged_store_matches_work_id AND
measurement table is a file` (`:108-117`) — which then *mints* a marker
(`:125-132`). That is deliberate staged-resume recovery, not an accidental
duplicate, and I am not filing it as a defect. It is worth noting only because it
is the one place where "this image is done" is answered `True` for an image
`valid_image_success` answered `False` for, and any future consolidation has to
preserve it.

---

<a name="f12"></a>
## F12 — MEDIUM. The staged worklist's completeness guard exists three times,
## and one copy checks a different file — the one a forward run never writes

This is what closing question 5 turned up, and it is the finding the
`stage2_result_replayable` claim was hiding rather than causing.

**First, the claim itself. It holds in substance and is wrong in its count.**
`_cli/CLAUDE.md:57` says `stage2_result_replayable()` "is the one function all
five sites call", and the function's own docstring
(`_cli_stage2_token.stage2_result_replayable`) enumerates five roles: the local
strategy's Stage-2 filter, its Stage-3 gate, its `--layer objmap` gate, the SLURM
shard worker's candidate filter, and the recovery controller's already-done skip.

There are **six** calls: `_cli_staged_strategy.py:216,273,442`,
`_cli_staged_slurm_worker.py:248,438`, `_cli_staged_controller.py:84`. The
omitted one is the SLURM worker's **Stage-3 gate** (`:438`) — the docstring names
only that worker's candidate filter.

And critically, **no site open-codes the two-half check.** The only direct
`stage2_token_exists` / `stage2_raw_path` uses outside the defining module are
`_cli_staged_resume.py:228` and `:281`, both inside `classify_staged_image`, both
deliberate and both documented (ledger FLOW-40, with an explicit comment at
`:268-278` explaining why the raw check must *not* be folded into `stage2_done`).
So unlike [F7](#f7)'s cycle, this single-sourcing claim is **true**; it is the
count and the unit that are loose. Worth one sentence of correction, not a
finding — except that the doc says "sites" without saying sites *of what*, so the
next reader cannot check it either. **An unfalsifiable single-sourcing claim is
worth as little as a false one**, and this one should say "six call sites in three
modules" so that a grep either confirms it or does not.

**The actual finding.** Underneath that correctly-shared probe sits a *second*
completeness guard that is **not** shared, and it has three variants:

| Site | Guard | File it stats |
|---|---|---|
| `_cli_staged_controller.py:86-88` | `resume and not markers_required and parquet.is_file()` | `results/<ds>/measurements/<stem>.parquet` — **legacy** |
| `_cli_staged_slurm_worker.py:260-266` | `resume and not markers_required and (…).is_file()` | `<store>/tables/measurements/table.parquet` — **embedded** |
| `_cli_staged_resume.py:260-266` (`classify_staged_image`) | `process_only_layer is None and not markers_required and (…).is_file() and not stage2_done` | **embedded** — and **no `resume` conjunct** |

The controller stats a different file from the other two. `_cli/CLAUDE.md`'s
output-layout section says so in as many words:

> "Forward, staged Stage 3, and measure runs do not create
> `results/<ds>/measurements/<stem>.parquet`; that directory is **legacy
> migration input only**."

So the controller's third disjunct is **dead on every forward staged run** and
fires only on a migrated tree, while the worker's and classifier's equivalent
disjunct fires normally.

**Why that is load-bearing rather than a dead branch.** This guard is reached
only when `markers_required` is false — and in that configuration
`stage3_completion_exists` is false *by design*, because the Stage-3 marker is
what `staged_stage3_markers=False` switches off. The parquet check is therefore
**the only completeness signal that exists** in that mode, and the two sides of
the SLURM loop read different files for it.

The concrete sequence, with `resume=True, markers_required=False`, for an image
whose Stage 3 embedded its table:

1. controller `_reclassify` — `stage2_result_replayable` false, `stage3_completion_exists`
   false, legacy parquet absent (forward run) → falls through → store is a valid
   Stage-1 store → **retryable**, submitted in the next round
2. shard worker candidate filter — embedded table present → **excluded from
   candidates**, nothing runs
3. round ends having done no work; the controller reclassifies to the same set

`_cli/CLAUDE.md` describes what happens then: "One unchanged retryable-set round
is retried; a second unchanged round terminalizes the remainder." So a **finished
image is terminalized after two wasted SLURM array rounds** — not a wrong
scientific result, but a wrong worklist, two spurious rounds, and a terminal
classification for an image that was complete.

**A second defect on the same key, found alongside.** `stage3_markers_required`
is read by three sites with **two different defaults**:

* `_cli_staged_controller.py:68` — `config.get("stage3_markers_required", True)`
* `_cli_staged_orchestration.py:271` — `state.get("stage3_markers_required", False)`
* `_cli_checkpoint_handler.py:260` — `orchestration.get("stage3_markers_required", False)`

`_cli_staged_slurm.py:412` writes the key, so the default is inert for documents
that build writes. On a document from an older build, or any path that does not
set it, the controller reads **True** and the other two read **False** — one key,
three readers, opposite answers. That is the [F6](#f6) pattern (a value spelled
independently in more than one place) applied to a default rather than to a field
list, and it is cheaper to fix.

**Proposed helper.** One predicate in `sdk_`, taking the mode explicitly, so the
three sites cannot differ on either the file or the default:

```python
# sdk_/_run_state.py
def staged_image_is_complete(
    output_dir: Path,
    dataset: str,
    image_stem: str,
    *,
    markers_required: bool,
) -> bool:
    """Whether a staged image needs no further stage.

    THE measurement table is the embedded one inside the store. The legacy
    `results/<ds>/measurements/<stem>.parquet` is migration input and is never
    written by a forward staged run, so statting it is a guard that cannot fire
    on the runs it guards.
    """
```

**Which implementation to keep: the embedded-table one** — the worker's and the
classifier's. The controller's legacy path is not a variant to be reconciled, it
is a stale path that predates the OME-Zarr cutover. And **the default for
`markers_required` should be `True`** (the controller's), because it is the
conservative one: `True` means "require the marker", which routes a doubtful
image back through a stage rather than declaring it complete on a file stat.

The `resume` conjunct is a real difference and should be preserved, not
normalized away: the classifier is called on paths where `resume` is not
meaningful. Make it a parameter rather than dropping it.

---

## Questions with a clean answer — no finding

**Question 2, the lifecycle/liveness reads.** `_run_state._scheduler_epoch:211-216`
restates `load_slurm_lifecycle`'s `generation`-falls-back-to-`epoch` rule
(`_cli_slurm_lifecycle.py:175`), and `_slurm_observer.py:1256` spells it a third
time. Two lines, identical semantics, genuinely blocked by INV-LAYER in one
direction. Not worth a finding on its own; it should ride along with whatever
moves the lifecycle *reader* into `sdk_` if anything ever does.

**`metadata_sha256`.** I was going to file this: `_metadata_digest_for` returned
`None` on any continuation that did not re-pass `--metadata`, while the state
recorded the snapshot's real digest, so the two disagreed on every such
invocation. **It was fixed on disk during this review** — `_metadata_digest_for`
now falls back to `metadata_csv_deliverable_path(config.output_dir)`, guarded so
measure/process runs (which skip the snapshot) do not invent one. Reporting as
resolved.

This is the corroboration recorded under *Verification status*: the correctness
reviewer reached the same site as its gate finding F3, from the opposite
direction, at the same time. Neither reviewer saw the other's work. One residual
nit: the docstring still anchors on `phenotypicCLI.py:471`,
but the value that actually lands in `state.config` comes from
`file_sha256(config.metadata_csv)` at `phenotypicCLI.py:2749-2753`. Those agree
in value (`file_sha256`'s non-directory branch, `_cli_failure_tracker.py:188`, is
a plain streamed SHA-256), so it is a wrong citation, not a wrong computation.

**`scientific_config_digest` / `pipeline_sha256`.** Four producers —
`pipeline_content_digest` (`_cli_staged_resume.py:76-78`), used by
`create_initial_state:245-249`, `phenotypicCLI.py:2717`, `mint_run_identity`
(`_cli_identity.py:275-277`) and `validate_resume_compatibility:326`; plus
`_cli_migrate._file_sha256:557-565` at `_cli_migrate.py:707,715`. All are
`sha256(file bytes).hexdigest()`. They **agree on every input**. A spelling
duplication only.

**`_digests.canonical_digest`.** The hoist is correct and the docstring's account
of it is accurate — it removed two copies rather than adding a third, took the
wider annotation, and records why `ensure_ascii=False` is not negotiable. This is
the pattern the rest of the report is asking for. ([F5](#f5) is the one place a
fourth copy escaped it.)

**`_walk_current_success`.** `_cli_completion.py:535-591` is the single traversal
behind both `current_success_inventory:488` and `current_success_counts:660`,
with the docstring stating exactly why (`:540-542`). Correct.

**`_schema_shape`.** One detection, two audiences, one home, one arming flag with
one binding, and `test_the_arming_flag_has_one_source` asserting structurally that
the CLI module re-declares nothing (`_schema_shape.py:87-100`). This is the
template.

---

## Specification: the helpers to build

**Home: `sdk_`. Every one of them, with no re-export shim.**

There are **three** consumers, not two — CLI, GUI, and the `sdk_` readers — and
`sdk_` is the only layer all three may import. The GUI reaches into
`phenotypic._cli` today with 25 private imports across 9 modules, which is the
audit finding this change exists to remove; homing shared logic in `_cli/` would
make that reach *correct* and entrench it.

Probe 2 is the case that proves the layer matters rather than merely tidies:
`inventory_digest`'s two producers straddle the INV-LAYER line — one in
`sdk_/_run_state.py`, one in `_cli/_cli_identity.py` — so a definition both need
**cannot** live in `_cli/` without inverting the layer and failing
`tests/unit/sdk_/test_run_state_layering.py`.

**And [F1](#f1) is the argument for this, so read them together.** The runs
registry open-codes a two-field predicate *because* the shared validator lives on
the wrong side of a boundary the GUI cannot cross. The duplication is not
carelessness; it is what a layer violation looks like once someone declines to
commit it. Move the definition to `sdk_` and the open-coding has no reason to
exist.

### The table

For each: the signature, the `sdk_` module, and **which existing implementation
is the correct one to keep**. The last column is the one that matters — whoever
builds this will otherwise keep whichever is easiest to call.

| # | Signature | `sdk_` module | Which implementation to keep, and why |
|---|---|---|---|
| **F1** | `run_proof(output_dir) -> Mapping \| None` | `_run_state.py` | **The strict one** — `_valid_run_proof`, which checks `version`. The four version-less readers are wrong, not merely lax: `_cli/CLAUDE.md` states the policy as "a version mismatch invalidates rather than migrates", so a reader that skips the check is certifying a proof this build cannot interpret. Keep `valid_run_completion`'s digest comparisons as a **separate** `run_proof_is_current(output_dir)`; the two have different callers and forcing a structural check to load processing state is what pushed callers into open-coding in the first place. |
| **F2** | `inventory_digest(work_ids) -> str` | `_run_state.py` | **`canonical_digest(work_ids)`** — the reader's and the publishers' shared spelling, already on disk in every aggregate and run proof. `_inventory_digest_for` is **deleted, not reconciled**: its value is a digest of the manifest-digest string, which is not an inventory digest under any definition either half uses, and adopting it would rewrite every proof on disk. |
| **F3** | `marker_rejection(marker, *, work_id, dataset, image_stem) -> str \| None` | `_run_state.py` (export the existing private `_marker_rejection`) | **The sdk one, including its `provenance: "migrated"` branch.** It returns a sentence rather than a bool, which is what makes `ImageState.reason` answer "which images are missing, and why" without a re-run; and the migrated branch is a *ruling* (U-10), not an oversight — a pre-markers tree never had a `work_id` to match, so the CLI's unconditional comparison would reject every migrated image. `valid_image_success` keeps its `bool` signature over the top so its ~20 callers are untouched. |
| **F4** | `finalization_input_digest(config)` and `accepted_finalization_digests(config)` | `_run_state.py` | **Both, as a pair, and the split is the point.** Publishers call the first (versioned, one spelling); validators call the second (every spelling a proof already on disk may carry). Keeping only the strict one breaks existing trees; keeping only the tolerant one means a publisher can emit a spelling no validator was told about. The current tolerance in `_accepted_finalization_digests` is right and `current_aggregate_is_current`'s single-spelling comparison is the one to drop. |
| **F5** | `RunIdentity.digest()` → `canonical_digest(payload)` | `_state_types.py` | **`canonical_digest`**, i.e. `ensure_ascii=False`. Not a coin-flip: DF-19 records that every proof on disk was written that way, so the hand-rolled `ensure_ascii=True` copy is the deviation. Fix F5 and F6 **in one commit** or the cache silently cold-starts once. |
| **F6** | `IDENTITY_DIGEST_FIELDS`; `digest()` derives from it | `_state_types.py` | **`_IDENTITY_DIGEST_FIELDS`'s five**, moved down beside the dataclass. Derive rather than assert equality — the same standard `test_the_marker_schema_constants_have_exactly_one_home` already applies to the marker constants. |
| **F8/F9** | export `_fenced_artifact_path(output_root, descriptor) -> str \| None` | `_run_state.py` | **The sdk one** — it is the only copy that dispatches on `kind` *and* returns the path to fence rather than a bool, which is what the verification cache needs. The two CLI copies are equivalent for `file` today; `valid_aggregate_snapshot`'s is missing the `kind` branch outright. |
| **F12** | `staged_image_is_complete(output_dir, dataset, image_stem, *, markers_required, resume)` | `_run_state.py` | **The embedded-table variant** (worker + classifier), and **`markers_required` defaults to `True`** (the controller's). The controller's legacy `results/<ds>/measurements/<stem>.parquet` is not a variant to reconcile — it is a pre-OME-Zarr path that forward runs never write. `True` is the conservative default: it routes a doubtful image back through a stage rather than declaring it complete on a file stat. Keep `resume` as a parameter; the classifier is called where it is not meaningful. |
| **F11** | delete `get_remaining_images` | `_cli/_cli_update_state.py` | **Deletion, not relocation.** It derives from `datasets.{completed,failed}`, which spec §4.2 removes from the state file. Moving dead code into a new shared module would launder it as current. |

### Dispositions that are not helpers

* **[F7](#f7) is "just import it".** Probe 1 shows the cited blocker does not
  exist, so this is not a case needing a neutral home. Best outcome is that
  [F2](#f2) deletes the function entirely.
* **[F10](#f10)** is subsumed by F3 — the same clause set, already in the table.
* **[F2](#f2) forces a design choice**, not a drop-in: `mint_run_identity` has no
  `work_ids` at mint time. Three options and my recommendation — **(a)** mint an
  empty `inventory_digest`, have `assert_identity_current` skip empty tokens,
  document the field as reader-populated — are in [F2](#f2). This is the decision
  the user is holding, and no choice of module changes it.
* **[F12](#f12)'s second half** — `stage3_markers_required` read with default
  `True` at one site and `False` at two — is a one-line fix at each site and
  needs no helper. Standardize on `True`.
* **Documentation, not code:** `_cli/CLAUDE.md:57` and
  `stage2_result_replayable`'s docstring should say "six call sites in three
  modules" so the claim is checkable by grep. An unfalsifiable single-sourcing
  claim is worth as little as a false one.

### Constraints observed

**No proposed helper adds a tracked-state artifact.** Checked row by row rather
than assumed: every one is a pure function over bytes already on disk, or a
deletion. Nothing introduces a file, key or field that anything branches on. The
four tracked states are untouched and the count does not move. The closest call
is F1's `run_proof`, which reads the existing run completion marker and writes
nothing.

**No call-site migration is proposed here.** That is P6 Task 7. This section is
the destination and the signatures only — a gate reviewer refactoring the tree it
is reviewing would invalidate its own review.

**Two dispositions that are not helper proposals**, and should not be turned into
ones:

* **F7 is "just import it".** Probe 1 shows the cited blocker does not exist, so
  this is not a case needing a neutral home. `_cli_identity` imports
  `_image_manifest_digest_for` from `_cli_state_management`, or — better, per F2
  — `_inventory_digest_for` is deleted outright because its value is not an
  inventory digest under any definition either half uses.
* **F2 forces a design choice the next phase must make explicitly**, not a helper
  that can be dropped in: `mint_run_identity` has no `work_ids` to digest at mint
  time. The three options and my recommendation (**(a)** — mint an empty
  `inventory_digest`, have `assert_identity_current` skip empty tokens, document
  the field as reader-populated) are in [F2](#f2). This is the decision the user
  is holding, and no choice of module changes it.

**No proposed helper adds a tracked-state artifact.** Every one is a pure
function over bytes already on disk, or the deletion of dead code. Nothing here
introduces a new file, key or field that anything branches on — the four tracked
states are untouched, and the count does not move. I checked this against each
row rather than assuming it; the one that came closest is F1's `run_proof`, and
it reads the existing run completion marker and writes nothing.

**Scope:** this section is the destination and the signatures. The migration into
`state_tracking.py` is implementation work for a later cluster — a gate reviewer
refactoring the tree it is reviewing would invalidate its own review.

---

## Verification status

Four claims were checked mechanically as well as by reading
(`scratchpad/p2_reuse_probe.py`, read-only, run by the lead). **All four
confirmed; nothing struck.**

```
PROBE 1: module-level import closure of _cli_state_management
  size=10
  closure=['_cli_directory_scanner', '_cli_failure_tracker', '_cli_file_locking',
           '_cli_process_only', '_cli_stage2_token', '_cli_staged_resume',
           '_cli_state_management', '_cli_types', '_cli_update_state', '_stages']
  contains _cli_identity        : False
  contains _cli_slurm_lifecycle : False
  VERDICT: cycle claim is FALSE (no cycle -- the import is legal today)

PROBE 2: the two inventory_digest producers
  _run_state.py:276     canonical_digest(config['work_ids']) = 68887577c0afee1b...
  _cli_identity.py      canonical_digest(manifest_digest)    = 12ab9e1136729d02...
  equal: False
  VERDICT: THEY CANNOT AGREE -- different domains (nested map vs one hex string)

PROBE 3: RunIdentity.digest() vs canonical_digest on the same payload
  RunIdentity.digest()            = a1f85d136923e3f1...
  canonical_digest(same payload)  = c723340426762d50...
  equal: False
  ascii-only control (both should agree): True

PROBE 4: the two enumerations of the fenced fields
  _run_state._IDENTITY_DIGEST_FIELDS = [finalization_input_digest, inventory_digest,
                                        processing_generation, restart_epoch,
                                        scientific_config_digest]
  RunIdentity.digest() payload keys  = [same five]
  equal today: True
```

Probe 3's **ASCII control is what makes it a finding rather than a coincidence**:
the two agree on ASCII and diverge otherwise, which is the signature of a
dropped `ensure_ascii=False` rather than of two unrelated computations.

Everything else in this report was established by reading the code at the site
cited. Line numbers were re-pinned against the working tree after
`_cli_identity.py` and `_verification_cache.py` changed mid-review; **where a
symbol name is unambiguous this report cites the symbol rather than the line**,
because six gate fixes have landed since and this plan's most repeated defect is
a citation that was true when written.

### Corroboration worth recording

The `_metadata_digest_for` divergence noted under *no finding* below was found
independently and simultaneously from two directions — by the correctness
reviewer as its gate finding F3, and by this review as a digest-producer
mismatch. Two reviewers with different questions converging on the same site is
evidence about the site, not about either reviewer's method, and it is the
strongest signal in this report that the digest-producer family is where the
duplication actually bites.
