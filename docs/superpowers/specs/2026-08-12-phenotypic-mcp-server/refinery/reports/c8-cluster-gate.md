# C8 cluster gate — `--image-manifest`, argv promotion, resume participation

**Verdict: NOT yet safe to merge into `feat/mcp-server`. Two blockers, both on the
approval-gate path, both cheap to fix.**

Everything else about this cluster is good work: the reader fails closed on every
malformed-input probe I could construct, work-ID equivalence holds and is proved
by a test that measures the load-bearing property rather than assuming it, the
resume guard genuinely round-trips, the argv coverage gate is derived from the
live Click object (it *would* fail the day `--durable-writes` lands), and the
double-emit `Counter` test covers all three flag families. The implementer's own
correction of the work-ID false green was right, and its four killed-mutation
claims replicate.

- Worktree: `/bigdata/iwheeldonlab/anguy344/PhenoTypic-worktrees/c8-manifest` @ `94d0616a4`
- Diff reviewed: `git diff 1df13f334..HEAD` — 14 files, +1442
- Baseline: `uv run --no-sync pytest -q tests/unit/cli/test_cli_image_manifest.py
  tests/unit/services/ tests/unit/gui/run_console/` → **277 passed in 72.17s**

---

## Blockers

### B1 — The CLI's manifest wiring has zero test coverage; deleting it is a surviving mutation

`src/phenotypic/phenotypicCLI.py:1683-1697` is the *only* place a manifest ever
narrows real work. `apply_image_manifest` is exhaustively unit-tested in
isolation, but **nothing invokes the CLI with `--input` + `--image-manifest` and
asserts what got selected.** The one CLI-surface test
(`tests/unit/cli/test_cli_image_manifest.py:592`, `test_the_flag_requires_input`)
exits at the `UsageError` guard and never reaches the scan.

Verified by mutation, not by reading. Patching the CLI's bound name to a no-op —
i.e. the manifest is accepted, echoed about, and then ignored, and the run
processes the entire parent directory:

```python
phenotypicCLI.apply_image_manifest = lambda mapping, manifest, input_path: mapping
```

```
### mut1
tests/unit/cli/test_cli_state_management.py ........
============================== 30 passed in 0.24s ==============================
```

**The mutation survives.** That is the literal failure mode this cluster exists
to prevent — a human approves N images, the run processes every image under the
parent — and the suite is green through it. The implementation is correct by
reading; the gate protecting it is not there.

**Fix.** One `CliRunner` test on a small two-dataset tree with `--dry-run`,
asserting the plan names exactly the manifest's images and not the others. The
`--dry-run` path (`phenotypicCLI.py:1766-1772`) runs after the manifest is
applied and before any image is opened, so it needs no real TIFFs — the existing
`collision_tree` fixture's byte-content files are enough. Asserting the
`"selected N of M image(s)"` echo alone would *not* kill this mutant cleanly;
assert the selected set.

### B2 — A symlink alias in the input tree silently selects more images than the manifest names

`src/phenotypic/_cli/_cli_directory_scanner.py:335-340` selects by
resolved-path **set membership**:

```python
selected.add(resolved)          # :331
...
kept = [p for p in image_paths if _resolved(p) in selected]   # :337
```

`_resolved` follows symlinks. If two entries of the scan resolve to the same real
file — a symlinked image, or a symlinked dataset directory — a manifest naming
**one** of them selects **all** of its aliases. Reproduced against the real code:

```
SCANNED: {'plate1': ['copy.tiff', 'img001.tiff', 'img002.tiff']}   # copy.tiff -> img001.tiff
ONE ENTRY -> {'plate1': ['copy.tiff', 'img001.tiff']} count: 2
```

A one-line manifest produces two units of work with two distinct work IDs (the
relative paths differ, so they are genuinely two computations, two HDFs, two
measure passes). Nothing notices: the duplicate guard fires only when *the
manifest* names both spellings, and the echo prints `selected 2 of 3` without
ever comparing against `len(entries)`.

This is over-selection on the irreversible full-dataset deploy path, and
symlinked image trees are ordinary staging practice on this cluster.

**Fix (both halves, they are three lines each).**

1. Carry `(dataset, path)` identities out of the entry loop instead of
   re-deriving membership — `scanned[resolved]` already holds exactly the pair —
   then rebuild `filtered` from those pairs, re-sorted into scan order.
2. Keep a total invariant regardless: after filtering,
   `sum(len(v) for v in filtered.values()) == len(entries)` or raise
   `ImageManifestError`. That is the assertion that makes "the count approved is
   the count that runs" a checked property rather than an emergent one, and it
   holds under any future aliasing this code does not anticipate.

A test: a manifest naming one image in a tree containing a symlink to it selects
one image.

---

## False greens found

### FG1 — `test_the_recorded_digest_is_the_manifest_content_digest` is tautological

`tests/unit/cli/test_cli_image_manifest.py:541-561` asserts
`state.config["image_manifest_digest"] == image_manifest_digest(manifest)`. Both
sides call the same function, so it agrees under any definition of the digest.
Replacing `image_manifest_digest` with a digest of the *resolved image set*
(mut3) left this test green:

```
### mut3
FAILED ...::test_digest_is_the_file_bytes
FAILED ...::test_resume_refuses_when_the_manifest_has_gone_missing
========================= 2 failed, 28 passed in 1.02s =========================
```

The mutant was killed — by `test_digest_is_the_file_bytes:143`, which anchors
against `hashlib.sha256(first.read_bytes())` — so the *suite* is sound. But the
test whose docstring claims to pin the cross-side contract ("the server binds
this same number into the plan token (§5.4)") proves nothing on its own.
**Non-blocking**, since the anchor exists; worth one line changing the
right-hand side to `hashlib.sha256(manifest.read_bytes()).hexdigest()` so the
test that names the contract is the test that checks it.

---

## Mutation battery — what I ran, and what it found

Mutations were applied at runtime through pytest `-p` plugins that rebind the
target symbol in `pytest_configure`, so **no file in the worktree was edited**.
Scripts: `<scratchpad>/mut/mut{1..9}.py`.

| # | Mutation | Result | Killed by |
|---|---|---|---|
| 1 | `phenotypicCLI.apply_image_manifest` → identity (CLI ignores the manifest) | **SURVIVED** | — see **B1** |
| 2 | `apply_image_manifest` deduplicates instead of erroring | killed | `test_a_repeated_entry_is_an_error` |
| 3 | `image_manifest_digest` digests the resolved image *set*, not the file bytes | killed | `test_digest_is_the_file_bytes`, `test_resume_refuses_when_the_manifest_has_gone_missing` |
| 4 | `_image_manifest_digest_for` always returns `None` (resume guard neutered) | killed (5 tests) | incl. `test_a_state_predating_the_flag_...` |
| 5 | `to_argv` inlines the `--slurm` pairs (the double-emit regression) | killed (4 tests) | **`test_the_shipped_slurm_path_emits_each_flag_exactly_once`** |
| 6 | `_SLURM_DIRECT_KEYS` reordered (moves `argv_digest`) | killed | `test_live_cancellation_hold_profile_reaches_cli_argv` — **pre-existing**, not a C8 test |
| 7 | `work_id_for_image` always uses the bare basename | killed | both work-ID tests |
| 8 | `to_argv` drops the `--image-manifest` branch | killed | `test_to_argv_passes_the_image_manifest_alongside_input` |
| 9 | empty manifest reads as "process everything" | killed | `test_reader_refuses_an_empty_manifest` |

One nuance worth recording: mutation 6 is killed by
`test_live_cancellation_hold_profile_reaches_cli_argv`
(`tests/unit/gui/run_console/test_slurm.py:415-428`), which **predates this
cluster**. The promoted `_SLURM_DIRECT_KEYS` order therefore has a guard, but it
is an inherited one — and the `slurm_argv_extension` doctest that also documents
the order never executes, because `--doctest-modules` is not in `addopts`
(`pyproject.toml`). If that pre-existing test is ever narrowed, the order
protecting `argv_digest` loses its only real check.

**The implementer's mutation claims replicate.** Mutation 7 in particular is the
one it reported fixing — the basename-always mutant that passed the id-only
version of the work-ID test. It is now killed by *both* work-ID tests, and the
kill comes from the relative-path assertions
(`test_cli_image_manifest.py:346-353`), which is the property that actually
distinguishes parent-rooted identity from basename-only. The correction to the
plan's rationale is also right: `compute_work_id`
(`src/phenotypic/_cli/_cli_failure_tracker.py:154-174`) hashes `dataset` as its
own field, so the plan's claimed cross-dataset collision does not occur. The
test docstring now says so explicitly.

---

## Attack surfaces, one by one

### 1. The `.images` contract — fails closed on every probe

Probed against the real reader (`<scratchpad>/probe.py`):

| Probe | Behaviour | Verdict |
|---|---|---|
| CRLF line endings | `line.strip()` removes `\r`; entries parse clean | correct |
| trailing newline / blank lines / `#` comments / indented comments | ignored | correct |
| UTF-8 BOM | entry keeps `﻿`, fails the scan lookup → `ImageManifestError` | fails closed (message is confusing — see N1) |
| non-UTF-8 bytes | `ImageManifestError: is not valid UTF-8 text` | correct |
| `..` traversal (`../in/plate1/img001.tiff`) | resolves back into the tree, accepted; resolving *out* of it is refused as unknown | correct |
| absolute path outside `--input` | `ImageManifestError: not one of the images found under --input` | correct |
| absolute vs relative spelling of the same image | identical result (`test_absolute_and_relative_entries_name_the_same_image`) | correct |
| empty file / comment-only file | refused, never "everything" | correct |
| duplicate entry | refused, never deduplicated | correct |
| unknown / nonexistent entry | refused, naming the entry | correct |
| **symlink alias inside the tree** | **one entry selects two images** | **B2** |

Unicode normalization (NFC vs NFD) is not applied on either side. An NFD entry
against an NFC filename fails the lookup and is refused — the safe direction.
Worth one line in the format docstring; not a defect.

**Does the digest the CLI records equal what the server binds?** Yes, subject to
the server using the same definition. `image_manifest_digest`
(`_cli_directory_scanner.py:188-211`) is SHA-256 over the file's raw bytes,
streamed, no normalization — so any edit to the approved artifact, including one
that resolves to the same image set, invalidates the token. That is the
conservative direction and matches spec 05 §5.4's wording. Nothing in this
cluster can enforce the server half; that binding belongs to the server task.

One narrow TOCTOU: the manifest is read at `phenotypicCLI.py:1685` and re-read
for its digest at `create_initial_state` (`phenotypicCLI.py:2178` →
`_cli_state_management.py:233`). A racing edit between the two makes the recorded
digest describe a file that is not the one applied. Microscopic window, and the
resume guard catches the drift on the next invocation. Noted, not filed.

### 2. Work-ID equivalence — holds, and is proved properly

`work_id_for_image` (`_cli_failure_tracker.py:177-205`) reads only
`config.input_path`, `dataset`, the image path, the pipeline fingerprint and
`processing_configuration_digest`. `image_manifest` appears in none of them, and
`test_the_manifest_stays_out_of_the_processing_configuration_digest`
(`test_cli_image_manifest.py:563-589`) pins that. Since the manifest narrows the
scan *within* `--input` and dataset names come from the same scan keys, a
manifest run and a parent run produce byte-identical work IDs. Confirmed by
mutation 7 rather than by reading.

### 3. The double-emit regression — the `Counter` test does its job

`tests/unit/gui/run_console/test_slurm.py:434-491` counts `--`-prefixed tokens on
the **shipped** path (`_build_subprocess_argv`, the GUI's alias) with all three
families populated: `--slurm == 6`, `--gpu-slurm == 2`, `--gpu-shards == 1`, plus
a catch-all "nothing else appears twice". A duplicated `--slurm` would read 12; a
duplicated `--gpu-slurm` would read 4; `--gpu-shards` is covered by the exact
count *and* the catch-all. All three families are covered, answering the question
as asked. Value-asserting tests would not have caught any of them, which the
docstring correctly explains.

The load-bearing `_SLURM_DIRECT_KEYS` **order** is separately pinned by the
pre-existing `slurm_pairs == [...]` assertion at
`tests/unit/gui/run_console/test_slurm.py:422-428`, so the reordering mutation
has a killer even though `--doctest-modules` is not enabled and the
`slurm_argv_extension` doctest never executes.

### 4. The coverage gate — genuinely derived, would fail on the day a flag lands

`tests/unit/services/test_argv_coverage.py:57-68` enumerates from
`phenotypic_cli.params` — the live Click object, not a source scan, not a
hardcoded list — so a `@click.option` added anywhere in the ~240-line block is
picked up with no edit to the test. The emitting side is an **AST** walk over
three named functions (`:71-92`), which correctly refuses to count a flag
mentioned in prose or assembled from a variable.

The gate is unusually well-defended for a gate:

- `test_the_emitting_functions_all_exist` stops a rename from silently emptying
  the walk and passing vacuously — the classic way this shape of test dies.
- `test_the_deny_list_names_only_real_options` kills stale excuses.
- `test_nothing_is_both_emitted_and_denied` stops a flag from being excused and
  emitted at once.
- `test_every_emitted_flag_is_a_real_cli_option` catches a typo on the emitting
  side.
- `test_named_flags_stay_emittable` prevents a silent demotion from emitted to
  denied.
- `test_the_counts_are_what_the_plan_recorded` locks `(32, 17, 15)` — and
  `17 + 15 == 32`, so every option is accounted for with no slack.

Every deny-list entry carries a substantive reason, and each one I checked is
accurate. I could not construct a mutation that adds a CLI flag and leaves this
green.

### 5. Resume — rejects a changed manifest, and the digest round-trips

`validate_resume_compatibility` (`_cli_state_management.py:318-346`) compares the
saved `image_manifest_digest` against a freshly computed one, using `.get` rather
than the tolerant `key not in → skip` idiom used further down. That choice is
what makes "absent" mean "no manifest" instead of "unknown, skip the check", and
it is directly tested
(`test_a_state_predating_the_flag_resumes_without_one_and_refuses_with_one`).
Mutation 4 (guard always sees `None`) kills five tests.

Both CLI call sites `sys.exit(1)` on incompatibility
(`phenotypicCLI.py:1595-1601`, `:1827-1838`), so the failure mode is refusal, not
a silent whole-parent run. The unreadable-manifest branch returns
`(False, message)` rather than letting `OSError` escape a function documented as
returning a message — reachable, since the server reaps a terminal plan token's
`.images` file — and is tested.

The round-trip through `ProcessingState` save/load is exercised indirectly (the
key is written by `create_initial_state` and read back through
`state.config.get`); the digest is a plain string in the same dict as
`pipeline_digest`, which has its own save/load coverage.

### 6. Back-compat — the GUI path is unchanged

`_slurm_argv_extension = slurm_argv_extension` and
`_build_subprocess_argv = to_subprocess_argv` are rebindings, not wrappers, and
`test_slurm_emitters_are_one_object_each` asserts identity while
`test_the_gui_module_no_longer_defines_the_emitters_or_their_key_order` asserts
by AST that no parallel copy (including `_SLURM_DIRECT_KEYS`) was left behind.
The pairing is right: identity alone would be satisfied by a rebinding to an
equal object, and the AST check alone would miss a wrapper.

The promoted `to_subprocess_argv` gained a keyword-only `python` parameter with a
`sys.executable` default, so the GUI's single-positional calls are unaffected.
`state_from_controls` (`gui/run_console/_state.py:87-110`) takes neither
`restart` nor `image_manifest`, so every state the GUI actually builds keeps both
at their defaults and renders exactly the argv it rendered before. The one
behaviour change — `dict` → `Mapping` for `extra` — is a widening.

---

## Non-blocking findings

### N1 — `--restart` turns the manifest drift guard off, and this cluster is what makes it reachable

`continuing` is computed at `phenotypicCLI.py:1486-1491` as
`not restart and not overwrite and ...`, so `--restart` sets `config.resume =
False` and `validate_resume_compatibility` — the entire manifest digest guard —
never runs. `--restart` also deliberately **keeps** `results/`, `deliverables/`
and `qc/` (`:1605-1613`).

Consequence on the MCP path: `--restart` against the same output directory with a
*different* manifest is accepted with no drift check, and the aggregated
deliverables then mix the previous approved set's outputs with the new one. The
new run's *compute* is exactly what the new token approved — so this is an
output-provenance problem, not an over-spend one — but the deliverables no longer
correspond to any single approval.

This is worth flagging here specifically because `--restart` becomes reachable
from `RunConsoleState` **in this cluster** (Task 19 part 3: *"Only `--restart` is
genuinely new behaviour"*), and the coverage gate's deny-list reasons about
`--restart` purely on the destructive-deletion axis
(`test_argv_coverage.py:47`) — which is the wrong axis for this failure.

Not a C8 implementation bug; it is a spec question for the server task
(*may a deploy carry `--restart`, and if so must it require a fresh output
root?*). Route it there rather than patching it here.

### N2 — `--sample` silently thins the approved set

`--sample N` is applied at `phenotypicCLI.py:1775-1783`, **after** the manifest
narrows the scan, so `--image-manifest` + `--sample` processes fewer images than
the manifest names, with no warning. Both flags are emittable from
`_services/argv.py`, so the combination is reachable from the server tier. Spec
05 §5.3 reportedly cuts `sample` from v1 deploy — if that holds, this is inert;
if it does not, the two flags should be mutually exclusive at the `UsageError`
guard alongside the existing `--image-manifest requires --input` check.

### N3 — The BOM refusal message is misleading

A UTF-8 BOM survives `read_text(encoding="utf-8")` and lands inside the first
entry, so the failure surfaces as *"names '﻿plate1/img001.tiff', which is
not one of the images found under --input"* rather than as an encoding problem.
Fails closed, so it is cosmetic — but a human debugging a server-written manifest
would chase the wrong thing. `encoding="utf-8-sig"` fixes it, and does not touch
the digest (which is over raw bytes, unchanged).

### N4 — Unicode normalization is unspecified

Neither side applies NFC/NFD normalization, so an NFD entry against an NFC
filename is refused as unknown. That is the safe direction and needs no code
change, but the `read_image_manifest` format docstring
(`_cli_directory_scanner.py:214-243`) enumerates the contract precisely enough
that the omission reads as an oversight. One line.

### N5 — Placement of the reader in `_cli_directory_scanner.py` is right

The plan named five files; the implementation touched six, putting
`read_image_manifest` / `apply_image_manifest` / `image_manifest_digest` /
`ImageManifestError` in `_cli/_cli_directory_scanner.py`. **This is the correct
call**, and the implementer should have said so rather than leaving it unflagged:

- `apply_image_manifest` consumes `scan_directory_structure`'s output and returns
  the same shape. It is a filter over the scan, and the scanner is the tier that
  owns "what images does `--input` contain".
- The module is cheap to import — `hashlib`, `pathlib`, `sdk_.constants_`,
  `_cli_types` — so the server tier importing it costs nothing.
- Spec 05 §5.4 already has the server importing `validate_resume_compatibility`
  from `_cli/_cli_state_management.py`, so a server → `_cli` import is sanctioned
  precedent, not a new layering violation.

The one debatable piece is `image_manifest_digest`, which is a pure SHA-256 with
no dependency on scanning; a server task importing
`_cli._cli_directory_scanner` solely for it is mildly odd. Not worth moving.

### N6 — Plan citation drift is real but minor

Spot-checked five citations against `1df13f334`, the base commit:

| Plan citation | Actual at base | |
|---|---|---|
| `work_id_for_image` `_cli_failure_tracker.py:179-186` | `def` at `:177`, cited body at `:181-186` | off by 2 |
| `_slurm_argv_extension` `gui/run_console/_slurm.py:177-194` | `def` at `:177` | exact |
| `validate_resume_compatibility` `_cli_state_management.py:299-302` | input-path check at `:300-301` | off by 1 |
| `ExecutionConfig` `_cli_types.py:95-100` | `class` at `:95`, core paths `:98-100` | exact |
| `--gpu-workers-per-gpu` `phenotypicCLI.py:1005` | decorator `:1004`, flag `:1006` | off by 1 |
| `-i/--input` `phenotypicCLI.py:922-931` | exact | exact |

Nothing that would mislead an implementer. Notably **not** the Task 20 failure
mode (a citation that resolves to a real-looking wrong symbol).

The plan's *substantive* error — "colliding across datasets for identically named
images" — was caught and corrected by the implementer, and the correction is
right: `compute_work_id` hashes `dataset` as its own field, so no collision
occurs. The disqualifying harm is divergence from the parent run, which the
corrected test asserts.

### N7 — `_emitted_flags` walks only `ast.FunctionDef`

`test_argv_coverage.py:80-92` skips `ast.AsyncFunctionDef` and does not descend
into nested helper functions defined inside the three emitters. Neither exists
today. Cosmetic; noted only because the gate's value is that it cannot be
outgrown quietly.

---

## Checks run

| Check | Result |
|---|---|
| `pytest tests/unit/cli/test_cli_image_manifest.py tests/unit/services/ tests/unit/gui/run_console/` | **277 passed**, 72.17s |
| `mypy src/phenotypic`, filtered to the six touched files | **zero errors** — no new errors introduced |
| Mutation battery, 9 mutations | 8 killed, **1 survived** (B1) |
| Edge-case probe of the `.images` reader | all fail-closed except B2 |
| Plan citation spot-check, 6 citations | ≤2 lines drift, none misleading |

## What must happen before merge

1. **B1** — one `CliRunner` `--dry-run` test that kills the "CLI ignores the
   manifest" mutant.
2. **B2** — make the selection carry entry identity, plus the
   `len(selected) == len(entries)` invariant, plus a symlink test.

Both are small. Nothing in the design needs revisiting, and the argv promotion,
the coverage gate, and the resume participation are all ready as they stand.
