# Phase 1b phase review — the seams between C4, C5, C6, C7a, C7b, C8

**Scope:** `git diff 1df13f334..HEAD -- src/ tests/` on `feat/mcp-server` @ `564d5e4c`
(57 files, +12347 / −313).
**Method:** read + live probes + cross-cluster mutation testing in an isolated
`git archive` copy under scratch with a `PYTHONPATH` override. The shared working
tree was never modified (`git status` clean throughout).
**Not** a re-run of the per-cluster gates. Every finding below is about an
*interaction* between two clusters, or about a property no single cluster owned.

---

## Verdict

**Phase 1b is sound as a whole.** The six pieces compose, and the seams are
clean in the places the phase review was most worried about: no later commit
weakened an earlier one's guarantee on any of the three shared files, no two
clusters compute the same identity differently, and every cross-cluster
mutation I injected was killed by an existing test.

One cross-cluster **blocker**: a safety property the spec states as a hard CLI
error (**USER-33**, `--sample` excludes `--image-manifest`) fell between C8's
task list and the spec commits that landed in the same phase, and nothing
implements it. Reproduced through the CLI.

One **false green the per-cluster gates could not see**: the whole
`finalize_distributed_study` suite (C6) runs against a storage backend that
C7a made unreachable for `--slurm`. Repointed at a real `journal://` study, all
17 tests pass — so this is a fixture-fidelity gap, not a defect.

The digest family is **consistent in what it identifies** but carries **two
spellings**, and the boundary between them is documented in one place and
missing in the one that matters for Phase 2.

---

## A. Cross-cluster blocker

### X1 — `--sample` and `--image-manifest` compose silently; the spec requires a hard CLI error

`src/phenotypic/phenotypicCLI.py:1685-1697` (manifest narrows the scan) then
`:1775-1783` (`get_sample_datasets` thins what survived). No guard between them.

Spec `05-deploy-and-slurm.md:644-649` (USER-33) states the failure verbatim and
the required remedy:

> `--sample N` applies *after* the manifest, so a human could approve 312 images
> and have 20 run with the manifest and its digest both unchanged. … so the
> combination is a hard error **at the CLI** rather than an ordering subtlety for
> an agent to reason about.

`06-errors-limits-testing.md:51` carries the matching error code
`sample_excludes_manifest`.

**Reproduced.** `--image-manifest` naming 3 images plus `--sample 1`:

```
exit_code: 0
✓ Image manifest …/plan.images selected 3 of 3 image(s)
│   Sample Mode        1 images per dataset                            │
```

The dry-run returns before sampling applies, so the plan block still shows 3;
on the real path `get_sample_datasets` reduces it. No `UsageError`, no warning.

**Why this is a seam and not a C8 miss.** The USER-33 ruling reached the
defining sections *inside this phase* (`7fd22a22`, `3c7a350d`), after C8's task
list was written — `phase-1b-engine-prerequisites.md`'s five-item breakdown for
the manifest task has no item for it, and no other plan task carries it. So no
agent was assigned the guard, and each half looks correct alone.

**And two clusters now assert incompatible things about `--sample`.** In
`tests/unit/services/test_argv_coverage.py`, the deny-list entry for `--layer`
gives as its reason *"v1 deploy is always the full pipeline (spec 05 §5.3 cut
mode, layer **and sample**)"* — while `--sample` is emitted by `to_argv`
(`_services/argv.py:439-441`) and is pinned as one of "the seventeen the service
tier is expected to reach" by `test_named_flags_stay_emittable`. The gate
actively defends the reachability of a flag whose own deny-list neighbour cites
the section that cut it.

**Consequence for Phase 2.** `image_manifest_digest` binds the approved set into
the plan token; the token re-derives and compares it at `deploy_start`; and a
`--sample` on the same command line makes all of that decorative. The property
currently rests entirely on a not-yet-written tool never populating
`advanced_args["sample"]`.

**Fix.** A `click.UsageError` beside the existing
`--image-manifest requires --input` guard (`phenotypicCLI.py:1248-1252`), plus a
`CliRunner` test. Decide separately whether `--sample` should stay emittable
from `to_argv` at all, or move to `_DENIED` with §5.3 as its reason — the
current split is the contradiction.

---

## B. False greens the per-cluster gates missed

### X2 — the finalize suite tests a backend a `--slurm` run cannot have

`tests/unit/tune/test_distributed_finalize.py:134-146`, `_build_study`:

> Mirrors exactly what ``run_tuning --slurm`` leaves behind …

`:162` then builds `storage_url = f"sqlite:///{io.tune_cache_study_db_path(out)}"`
and the marker records `"slurm": True`.

Since C7a, `run_tuning --slurm` **cannot** leave a SQLite URL:
`_validate_slurm_request` (`_run.py:805-812`) refuses an explicitly named SQLite
URL outright, and the `--slurm` default is `journal:///…/journal.log`
(`_default_journal_url`, `_run.py:104-127`). So every one of the 17 tests for the
function that exists *solely* to finalize a `--slurm` run exercises a state the
CLI forbids.

C6's gate ran before C7a landed and explicitly flagged the concurrent edit
("it changes the default `--slurm` storage URL, which is an input to
`finalize_distributed_study`"); C7a's gate looked at storage, not at finalize.
Neither could see this.

**The behaviour is correct.** I repointed `_build_study` at a real
`journal://` study in an isolated copy and ran the file unchanged otherwise:

```
17 passed in 40.89s
```

So this is coverage, not a defect — but nothing in the suite would catch a
journal-specific finalize regression, and the fixture's central docstring claim
is false as written. **Fix:** parametrize `_build_study` over both URLs, or
switch it to journal and keep one sqlite lane for the local path.

---

## C. The digest family

### X3 — one identity model, two spellings, and the warning is in the wrong place

| Function | Returns | Cluster |
|---|---|---|
| `bytes_fingerprint`, `file_fingerprint`, `paths_fingerprint` | `"sha256:<hex>"` | pre-existing |
| `directory_digest` (`sdk_/_io_constants.py:288`) | `"sha256:<hex>"` | C5 |
| `subset_digest` (`_services/staging.py:146`) | `"sha256:<hex>"` | C5 |
| `image_manifest_digest` (`_cli/_cli_directory_scanner.py:188`) | **bare hex** | C8 |
| `pipeline_content_digest` (`_cli/_cli_staged_resume.py`) | **bare hex** | pre-existing |
| `argv_digest` | **does not exist yet** | Phase 2 |

**Semantics are consistent and non-overlapping.** No pair claims the same
identity and computes it differently: `directory_digest` is a parent's file
*inventory* (`name, size, mtime_ns`, contents deliberately unread),
`subset_digest` is a chosen image set keyed on `(relative_path, content
fingerprint)` per image, `image_manifest_digest` is one artifact's exact bytes.
Each choice is argued in its own docstring and the arguments are mutually
consistent — notably C5's decision to hash *contents* in `subset_digest`
precisely because `directory_digest`'s `mtime_ns` does not survive a preserving
copy on this cluster's gpfs. That is the answer to "does any pair disagree": no.

**The spelling split is real and lands in one field.** The spec's own plan-token
record (`05-deploy-and-slurm.md:432-441`) shows every digest prefixed:

```
"pipeline_digest":"sha256:9c1e…", "subset_digest":"sha256:77b2…",
"argv_digest":"sha256:4b0a…"
```

…and `image_manifest_digest` is in that same record and in that same binding
table. The implementation returns bare hex, and that is **pinned by two tests**
(`test_digest_is_the_file_bytes`, `test_the_recorded_digest_is_the_manifest_content_digest`);
mutating it to the prefixed form fails exactly those two. So this is a
deliberate implementation choice that contradicts the spec's own example, and
the two will be string-compared across the tool boundary at `deploy_start`.

`directory_digest`'s docstring carries a **"Digest-format warning"** for exactly
this hazard (`_io_constants.py:311-316`) — but it names only
`pipeline_content_digest`. It does not name `image_manifest_digest`, which is
the bare-hex digest that actually sits beside `subset_digest` in the token.

**Third consumer, same field name, no normalizer.** C8's resume guard stores it
as `state.config["image_manifest_digest"]`
(`_cli_state_management.py:233`, `:315-345`), and the spec has `deploy_start`
re-derive and compare against the token's field of the same name. Two producers,
one name, two spellings.

**Related, pre-existing, and directly in `argv_digest`'s path.** There are
already two functions named `_command_digest` writing the *same*
`RunRegistry.command_digest` field with different inputs and different spellings:

- `gui/run_console/_callbacks.py:501-508` — `f"sha256:{…}"` over the serialized
  `RunConsoleState` JSON
- `gui/tune/_deploy.py:21-23` — bare hex over `"\0".join(argv)`

Only the second matches the spec's `argv_digest` definition ("the SHA-256 of the
rendered argv list, joined with `\0`"), and it is on the *tune* path, while
C8's `to_subprocess_argv` — the function whose docstring says "Spec §5.4 digests
exactly this list" — feeds the *deploy* path, whose digest hashes a state
payload instead. Nothing in Phase 1b broke this; Phase 2 will have to pick one.

**Fix (cheap, now):** extend `directory_digest`'s digest-format warning to name
`image_manifest_digest`, and add the reciprocal note to
`image_manifest_digest`. Decide the token spelling before Phase 2 writes
`argv_digest`, not after.

---

## D. Shared-file interference — none found

All three shared files were checked for a later edit weakening an earlier
guarantee. None does.

### `_services/argv.py` — C8, then C6

C8's coverage gate walks the AST of exactly three functions
(`to_argv`, `slurm_argv_extension`, `to_subprocess_argv`), so C6's four new tune
builders in the same module are quarantined by construction, and
`test_the_emitting_functions_all_exist` fails loudly if a rename empties the
walk. The `(32, 17, 15)` count lock is unmoved and still accurate — verified by
running the file. The gate is *blind* to `phenotypic.tune`'s argparse surface,
which the C6 gate already recorded; that remains open (see X4, and the P9 item).

### `_optuna_store.py` — C7a, then the T18 fix, then C7b

C7b's edits (`450c91d1`) are additive: the SQLite path correction in
`backing_file_for_url` and a carve-out note on `_JOURNAL_SIZE_WARN_BYTES`.
`best()`, `_to_trial`, and the terminal-trial filtering the T18 fix
(`68d9551f`) installed are untouched. The `StudyStore` protocol gained
`terminal_trials()` with a stated contract, and **both** implementations conform
(`JournalStudyStore.terminal_trials` returns `trials` with the reason spelled
out). `_require_terminal_study`'s refusal message even cites the journal
backend's heartbeat-free semantics by name — the C6↔C7a documentation seam is
closed, not merely absent.

### `_serializable_pipeline.py` — C4, then C5

**Mutation-proven.** Removing `"phenotypic.subset"` from
`PHENOTYPIC_CLASS_MODULES` in an isolated copy fails three tests:

```
FAILED tests/unit/services/test_catalog_reconciliation.py::test_the_selector_subpackage_is_appended_not_interleaved
FAILED tests/unit/subset/test_selectors.py::test_selectors_resolve_by_bare_class_name
FAILED tests/unit/subset/test_selectors.py::test_selectors_round_trip_through_class_and_params
3 failed, 232 passed
```

C5's append is genuinely append-only, so C4's first-match resolution order is
unchanged, and nothing leaks the other way — the live catalog builds 143
entries across 14 categories with **zero** `Subset*` rows and
`skipped_imports == {}`.

---

## E. Does the end-to-end flow compose?

Catalog (C4) → pipeline → subset (C5) → staging (C5) → distributed tune
(C7a/C7b) → finalize (C6) → deploy with a manifest (C8).

**Where it holds:**

- **Path spelling is one convention throughout.** `ImageRef.relative_path`,
  `subset_digest`'s terms, the staged `nested/` layout, the `.images` manifest
  entries, and `work_id_for_image` are all *parent-relative POSIX*. `to_argv`
  always emits `--input` alongside `--image-manifest` for exactly this reason
  (docstring, `argv.py:376-380`), so a manifest run and a whole-parent run agree
  on work IDs — the property C8 proved by test.
- **Storage identity survives the C7a→C6 handoff.** `_open_finished_store`
  prefers the marker's `storage_url` over re-deriving one, and `run_tuning`
  writes a non-null resolved URL into the marker above the submission branch, so
  finalize opens the URL the fleet actually opened.
- **The staging namespace is deliberately out of `--restart`'s reach.**
  `SERVER_SCRATCH_DIR` is documented as placed outside `runs/` and the parent so
  restart/overwrite semantics cannot reach parent images through a staging tree.

**Where the chain is not yet joined (expected — the tool layer is Phase 2, but
worth pinning before it is written):**

- **Nothing in `src/` calls `stage_subset`**, and nothing converts a
  `SubsetSelection` into a `SubsetToStage`. The two halves of C5 do not
  type-connect: `SubsetSelection.images` is `tuple[str, ...]` (relative paths
  only), `SubsetToStage.images` is `tuple[ImageRef, ...]` (paths *and* relative
  paths). The Phase-2 bridge must re-derive `ImageRef.path` as
  `parent / relative_path`, which is sound for a nested parent and needs a test
  for the flat one, where `_dataset_of` returns `""` and staging substitutes
  `parent.name`.
- **`finalize_distributed_study` still has no CLI or GUI caller** — confirmed;
  the only non-test reference in `src/` is a docstring cross-link at
  `_run.py:951`.

---

## F. Non-blocking findings

### X4 — `build_tune_command` cannot carry C6's `slurm_args`, so Task 16's objective is unreachable from the service tier

Task 16 exists because *"`account`, `qos`, `cpus_per_task`, and `gpus_per_node`
cannot reach a tune fleet at all"*, and on UCR HPCC `--account` is mandatory for
`exfab` and `preempt`. The CLI half works — verified:

```
{'account': 'exfab'}       -> #SBATCH --account=exfab
{'slurm_account': 'exfab'} -> #SBATCH --account=exfab
{'qos': 'high'}            -> #SBATCH --qos=high
```

But `build_tune_command` (`_services/tune_spec.py:554-576`) has **no
`slurm_args` parameter** and both its `tune_run_tail` calls omit it. That is the
validated composition point the GUI uses and the plan designates for the service
tier, so `slurm_account=exfab` reaches a fleet only through a direct
`tune_run_tail` / `tune_run_argv` call. One parameter and one passthrough.

### X5 — confirmed still open: `build_tune_command` marks `--slurm --screen` valid (C6 gate N1)

Reproduced. With a valid strategy and slurm+screen both set, the semantic tail
ends `['--slurm', '--screen']` and `issues` names nothing about the
incompatibility; C6's `_validate_slurm_request` then hard-refuses at run time.
A silent drop became a post-launch crash with no preflight. One line in
`build_tune_command`.

### X6 — confirmed still open: duplicate `#SBATCH --mem` (C6 gate N2)

Reproduced exactly:

```
merge_slurm_args({"mem_gb": 16}, partition=None, mem="8G", …)
  -> {'slurm_mem': '8G', 'mem_gb': 16}

#SBATCH --mem=8G
#SBATCH --mem=16G
```

`_canonical_slurm_key` collapses `mem`→`slurm_mem` and `partition`→`slurm_partition`
but leaves `mem_gb` untouched, so a non-canonical alias survives as a second key.
The documented "the more specific spelling wins" precedence holds only because
Slurm honours the last directive **and** `merge_slurm_args` happens to insert
explicit pairs after the sugar. Worth making a real rule rather than an ordering
coincidence, and worth refusing (or collapsing) a duplicate directive.

### X7 — `_open_store`'s docstring is stale after C7a, on a path C6 calls

`_run.py:528-530` still describes "the 3-way fallback (env var > local
`study.db`)". `_resolve_storage_url` is now a documented **4-way** fallback whose
default is `slurm`-dependent, and `_open_store` never passes `slurm=`.
`_open_finished_store` (C6) is correct today only because the marker always
carries a non-null URL for an Optuna run; if it ever did not, finalize would
fall back to `sqlite:///…/study.db` for a fleet run whose study is a journal and
report an empty study. An assertion or a comment beats the current accident.

### X8 — finalize opens the study with `create=True`, so a read-only operation writes

`_open_store` never passes `create=False`, so `finalize_distributed_study` goes
through `optuna.create_study(load_if_exists=True)`. On the journal backend,
`build_optuna_storage` does `journal_path.parent.mkdir(parents=True,
exist_ok=True)` and the store then creates a new study record. Verified:

```
before: False
after create=True: True   trials: 0
create=False raised: FileNotFoundError …/other/journal.log
```

C7b built `require_existing_backing_store` for exactly this and wired it to the
`create=False` path; finalize does not take it. A run directory whose journal was
never written or was removed gets a fresh empty journal and a "the study recorded
no successful trial" warning instead of a `FileNotFoundError` naming the missing
study.

### X9 — `directory_digest`'s walk change is behaviour-preserving on this interpreter

`bf1e729e` swapped `Path.rglob` for `os.walk(onerror=…)` and claimed the two
"agree on symlinked subdirectories under 3.12". Checked on the installed
interpreter (CPython 3.12.10): a symlinked subdirectory is listed by both and
descended by neither, so a symlinked dataset directory contributes nothing to the
digest under either implementation. The claim holds; noting only that the
resulting property (a symlinked dataset dir digests as absent) is unstated in the
docstring and is reachable here, where `/rhome` ↔ `/bigdata` aliasing and
symlinked staging trees are ordinary practice.

---

## G. Environment and evidence

- **`optuna` 4.9.0 is installed and the tune tests really ran.** `-rs` reports
  `991 passed, 2 skipped` across `tests/unit/tune` +
  `tests/unit/cli/test_cli_image_manifest.py` + `test_cli_state_management.py`;
  both skips are the Postgres lanes (`$PHENOTYPIC_TEST_PG_URL`). No
  `skipif(not _OPTUNA)` lane was silently green.
- `tests/unit/services` + `tests/unit/subset` +
  `tests/unit/sdk_/test_directory_digest.py`: **248 passed**, 0 skipped.
- `QT_QPA_PLATFORM=offscreen` set for every run; no Qt abort.
- All mutation testing ran in `git archive HEAD | tar -x -C <scratch>` with
  `PYTHONPATH=<scratch>/iso/src`. The shared tree was never edited.

---

## H. Known-open items — confirmed still open

| Item | Status |
|---|---|
| `finalize_distributed_study` has no CLI or GUI caller | **Confirmed.** Only reference in `src/` is a docstring cross-link (`_run.py:951`) |
| `fail_stale_trials` silently no-ops under journal storage | **Confirmed and well documented** in `build_optuna_storage`'s docstring, including the precise cost (zombie inflates the raw count; cannot win, cannot consume budget) |
| `--nrows`/`--ncols` not emittable from the tune service tier (spec P9) | **Confirmed.** They *are* emitted by `to_argv` for deploy; `tune_run_tail` has no parameter for them, and the argv coverage gate is structurally blind to the argparse surface |
| Task 20 / `RunRegistry` lock ordering withdrawn | Confirmed; no lock-ordering change in the diff |
