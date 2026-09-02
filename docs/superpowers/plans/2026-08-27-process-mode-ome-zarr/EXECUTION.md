# Execution — cluster-and-isolate orchestration

Derived from the per-task **Files** / **Interfaces** blocks in the phase files.
This is a version-controlled view of them, not a separate source of truth: if a
task's Files block changes, regenerate this.

## 1. Dependency DAG

```
        T1 ──┬─→ T2                     (guard needs the omission to exist)
             ├─→ T6                     (imread fixtures use write_image_class)
             ├─→ T7                     (writer passes write_image_class=False)
             ├─→ T9                     (writer passes consolidate=True)
             ├─→ T10a                   (digest fixtures build stores)
             └─→ T10b                   (scanner fixtures build stores)

        T3                              (independent)
        T4 ──→ T7                       (writer passes basename_only=True)
        T5 ──→ T6 ──→ T10b              (resolver → imread → scanner integration)

        T7 ──┬─→ T8a ──→ T8b            (core param → worker option → user CLI)
             └─→ T9                     (consolidation switches on the writer)

        T10a ─→ T10b                    (file_sha256 must survive a directory
                                         before the scanner can hand it one)

        T7, T8b, T10b ──→ T11
```

## 2. Shared files — the parallelism constraint

| File | Tasks |
|---|---|
| `src/phenotypic/sdk_/ngff_.py` | T1, T3, T5 |
| `src/phenotypic/_core/_image_parts/_image_io_handler.py` | T1, T2, T6 |
| `src/phenotypic/_core/_provenance.py` | T4 |
| `src/phenotypic/sdk_/typing_.py` | T7, T8b |
| `src/phenotypic/_cli/_cli_process_only.py` | T7, T8a, T9, T10a |
| `src/phenotypic/_cli/_cli_process_single.py` | T8a, T8b |
| `src/phenotypic/_cli/_cli_failure_tracker.py` | T8b, T10a |
| `src/phenotypic/phenotypicCLI.py`, `_cli_types.py`, `_cli_execution_strategies.py`, `_cli_staged_{strategy,resume}.py`, `_cli_slurm_array_scripts.py`, `_cli_state_management.py` | T8b |
| `src/phenotypic/_cli/_cli_directory_scanner.py` | T10b |
| `CLAUDE.md`, `logic_validation_scripts/…` | T11 |

Two hot files (`ngff_.py`, `_image_io_handler.py`) are touched by five of the
first six tasks, which is what forces Phases 1–2 to run sequentially. In Phase 3
the hot file becomes `_cli_process_only.py`, touched by four tasks.

**T10a now shares `_cli_process_only.py` with the Phase-3 cluster** (it fixes
`process_only_output_path`'s degenerate relative path, which T7 rewrote). That
is the one change to the parallelism picture from the pre-review plan, and it is
why C6 no longer runs beside C4 — see §5.

## 3. Shapes

| Task | Shape | Why |
|---|---|---|
| T1 `write_image_class` + `consolidate` | **Seam** | Widens a *required* parameter to optional and threads two keywords through three call layers. Every store write in the codebase passes through it. Small, risky. |
| T2 `load_zarr` guard | Seam | Behaviour change on a public read path. Inseparable from T1 — its fixture needs T1's flag. |
| T3 `omero.rdefs` | Leaf | ~5 lines, but conformance-gated. |
| T4 provenance basename | Leaf | One function, one new keyword, isolated file. |
| T5 `read_ngff_image_spec` | **Keystone** | Novel core logic — the whole projection rule, ~250 lines and 24 tests. The largest single piece. |
| T6 `imread` store branch | Seam | The public dispatch point changes shape. Thin, but it is *the* wiring. |
| T7 process-only zarr writer | **Keystone** | Produces the artifact; novel; the spec's centre. |
| T8a `resolve_process_format` | Leaf | One pure function, two refusals, plus the worker's option. |
| T8b user-facing `--process-format` | **Sweep + Seam** | Ten files. Flips the default output format, and this is where the AutoConvertRaw hazard lands. Wide but shallow: the same keyword at eleven sites, plus the continuation digest. |
| T9 consolidation | Leaf | One keyword on T7's call, plus its regression tests. |
| T10a store work identity | Seam | One function body, but four call sites depend on it and all four die without it. |
| T10b scanner | **Seam** | One file, two traps that both yield plausible wrong results, a duplicated dry-run path, and a subtle monkeypatched test. |
| T11 validation + docs | Sweep + Leaf | Two files plus a standalone derivation script. |

## 4. Clusters

| # | Tasks | Intent | Model | Effort |
|---|---|---|---|---|
| **C1** | T1, T2 | Make a store able to omit `image_class` and to consolidate its part, and make `load_zarr` refuse a store that is not a bundle | Opus 5 | high |
| **C2** | T3, T4 | Two independent primitives: the render-model field and the provenance basename | Sonnet 5 | medium |
| **C3** | T5, T6 | The read path end to end — pure projector, resolver, `imread` branch | Opus 5 | high |
| **C4** | T7, T9 | The writer and its consolidation, one file, one intent | Opus 5 | high |
| **C5** | T8a, T8b | The CLI surface and the default flip, worker then user-facing | Opus 5 | high |
| **C6** | T10a, T10b | A store as a first-class input: its digest, its work ID, and the scanner that finds it | Opus 5 | high |
| **C7** | T11 | Validation script and documentation | Sonnet 5 | medium |

**Why T3+T4 are one cluster and not folded into C1:** they share no file with
each other, but T3 touches `ngff_.py`, which C1 also edits — so they cannot run
beside C1 anyway. Bundling the two Leaves into one sequential dispatch costs one
gate instead of two and keeps C1's diff focused on the risky widening.

**Why C3 merges a Keystone with a Seam:** T6 is ~60 lines that do nothing but
call T5. Splitting them would hand the second agent an interface it did not
design and cannot see the reasoning for. Combined they are ~350 source lines and
33 tests — the upper edge of one reviewable diff, and verifiable in one pytest
run. **This is the largest library cluster; if its diff comes back
unreviewable, split at T6 and re-gate.**

**Why C5 holds both halves of the option:** T8a is a pure function plus a worker
flag, and T8b is the ten-file sweep that makes it reachable. Landing T8a alone
would merge a resolver nothing calls from a user-facing path — reviewable, but
not verifiable, since the only end-to-end proof of T8a is a command T8b enables.
They are one intent and one gate. **C5 is the widest diff in the run**; if the
ten-file sweep swamps review, split at T8b's Step 5 (the seven call sites) and
re-gate, but do not split T8a from T8b.

**Why C6 gained T10a:** `file_sha256` raising `IsADirectoryError` is not a
scanner detail — it is the precondition that makes the scanner's output usable
at all. Splitting them would merge a scanner that finds stores and a runner that
dies on them, which is a worse state than either end.

**C6 keeps its parallelism.** A first pass put
`process_only_output_path`'s degenerate-relative-path fix in T10a, which would
have made C6 touch `_cli_process_only.py` and forced the whole run sequential.
That fix moved to **T7**, which rewrites the function wholesale and is its
natural owner. C6 now touches only `_cli_failure_tracker.py` and
`_cli_directory_scanner.py`, neither of which any other cluster reads or
writes, so it runs beside C4 as decided.

## 5. Order and parallelism

```
C1 → C2 → C3 → C4 → C6 → C5 → C7
```

**No cluster runs in parallel any more.** The pre-review plan ran C6 beside C4
on the grounds that `_cli_directory_scanner.py` is touched by nothing else. That
is still true of T10b, but T10a — added to C6 by the review — edits
`_cli_process_only.py`, which is C4's own file. A parallel worktree would
produce two divergent versions of `process_only_output_path` and a merge
conflict in the exact function whose correctness this design turns on.

The cost is wall-clock, not correctness, and it is recoverable: if the run needs
the parallelism back, split T10a into C4 (it is a ten-line change to a function
C4 already rewrites) and run T10b alone in its own worktree. Do that
deliberately, not by drift.

**C5 is deliberately late.** Nothing depends on T8b: T11 needs T7, T8b, and
T10b, and T10b needs neither CLI task. Deferring C5 to just before C7 buys the
maximum amount of time for the AutoConvertRaw pin to hold, and if the run stops
before it, **nothing user-visible has changed** — the default `--mode process`
output is still a flat TIFF, because T7's `process_format` parameter defaults to
`"tiff"` until T8b wires the option to a user.

**C6 must precede C5 rather than follow it**, which is a change from the
pre-review order. T10a's `file_sha256` branch is what lets a store-input run
derive a work ID at all, and T8b puts `process_format` into that same work ID
via `processing_configuration_digest_from_values`. Landing T8b first means
touching the digest twice.

C7 stays last: it documents `--process-format` and the closed loop, and its
end-to-end exit criterion runs the real command twice.

## 6. Gates

| When | Gate | Model |
|---|---|---|
| Before any dispatch | `plan-reviewer` over spec + plan — API claims, runnability, test validity, ordering, unnecessary complexity | Opus 5 |
| After each cluster | Light: read the diff, run the cluster's tests + `ruff` + `mypy` on changed paths | orchestrator |
| After C3 | Deep: `implementation-test-reviewer` over the combined C1+C2+C3 diff — the library phase, and the phase that adds the most tests | Opus 5 |
| After C5 | Deep: `implementation-test-reviewer` over the combined C4+C6+C5 diff — the CLI phase, and the one that changes a production command's output | Opus 5 |
| After C7 | `code-simplifier` over the whole branch diff (quality only, no behaviour change), apply, then re-run `tests/unit` | Opus 5 |

Never review with a model weaker than the one that implemented. C2 and C7 are
Sonnet-implemented and Opus-reviewed, which satisfies that.

**One gate specific to this run.** After C5, before the deep review, run

```bash
uv run python -m phenotypic --mode process --layer rgb --dry-run \
    --input <tree of tiffs> --output /tmp/probe --pipeline <pipeline.json>
```

and confirm the sample paths it prints end in `.ome.zarr`. That single line
exercises the one thing eleven separate call-site edits can each silently get
wrong: `phenotypicCLI.py:954` is the dry-run path, it takes the format from
`config`, and it defaults to `"tiff"` if the wiring missed it.

## 7. C5's external blocker — CLEARED 2026-08-27

**Resolved.** AutoConvertRaw `HEAD` moved `63c4657` → **`3df608d`**
(branch `rsync-transport`), tree clean, with the pin in the committed
`pyproject.toml`. Verified directly from this session, not taken on report.
C5 is safe to execute.

The pin as landed: `vendor/phenotypic-0.18.0-py3-none-any.whl` (tracked,
10.6 MB, sha256 `dbafeb7f…`), referenced from `[tool.uv.sources]` and
`uv.lock:1204`. The stale "the pin lives in .venv only" note in
`src/config.sh` was replaced, and the wheel-over-git-ref reasoning is recorded
in both the commit message and a `pyproject.toml` comment so a future reader
does not "fix" it to a tag.

**One residual:** `rsync-transport` is 27 commits ahead of its remote
(pre-existing, not introduced by the pin), so the pin is durable **locally
only**. That is what protects the machine ACR actually runs on, which is the
machine that matters here.

**C5 stays late anyway.** The ordering was never only about the pin: it costs
nothing, and it keeps the one task that changes a production command's output at
the end of the run where a late problem is cheapest to abandon.

The original hazard, retained because it explains the ordering:

- **The pin is staged, not committed.** ACR `HEAD` is still `63c4657`; the
  change exists only in the working tree (`pyproject.toml`, `uv.lock`,
  `vendor/`). A `git checkout -- .` or `git stash` there silently restores the
  hazard.
- **The pin is a vendored wheel**, not a tag or SHA:
  `vendor/phenotypic-0.18.0-py3-none-any.whl`, sha256 `dbafeb7f…`, referenced
  from `[tool.uv.sources]` and `uv.lock:1204`. A git pin was not available —
  there is a `v0.18.1` tag but no `v0.18.0`, PhenoTypic's version is
  `dynamic`, and the installed `direct_url.json` records no revision, so the
  installed commit is not recoverable with confidence.
- **ACR previously pointed at this very repository.** Both files resolved to
  `/bigdata/exfab/anguy344/PhenoTypic` — the checkout this branch lives in.
- **ACR is running right now**, with a live `pht-cri_correct_*` Slurm array. It
  never invokes `uv` at runtime, so config edits cannot reach a running job; the
  risk is strictly a future manual `uv sync`.
- **Reaping is not the only break.** `worker_push.sh:161,163,199,254` assume one
  file per key (`[[ -f …tiff ]]`, `stat -c '%s'`, `find -type f`), so a
  directory store breaks *publishing* too — and those `find -type f` sweeps
  would descend into every store.

**Unblock condition (met):** ACR `HEAD` past `63c4657` with the pin in it.
