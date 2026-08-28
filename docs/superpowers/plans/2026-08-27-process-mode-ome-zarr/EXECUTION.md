# Execution — cluster-and-isolate orchestration

Derived from the per-task **Files** / **Interfaces** blocks in the phase files.
This is a version-controlled view of them, not a separate source of truth: if a
task's Files block changes, regenerate this.

## 1. Dependency DAG

```
        T1 ──┬─→ T2                     (guard needs the omission to exist)
             ├─→ T6                     (imread fixtures use write_image_class)
             ├─→ T7                     (writer passes write_image_class=False)
             └─→ T10                    (scanner fixtures build stores)

        T3                              (independent)
        T4 ──→ T7                       (writer passes basename_only=True)
        T5 ──→ T6 ──→ T10               (resolver → imread → scanner integration)

        T7 ──┬─→ T8                     (CLI feeds process_format into the core)
             ├─→ T9                     (consolidation wraps the zarr branch)
             └─→ T11
        T10 ─→ T11
```

## 2. Shared files — the parallelism constraint

| File | Tasks |
|---|---|
| `src/phenotypic/sdk_/ngff_.py` | T1, T3, T5 |
| `src/phenotypic/_core/_image_parts/_image_io_handler.py` | T1, T2, T6 |
| `src/phenotypic/_core/_provenance.py` | T4 |
| `src/phenotypic/_cli/_cli_process_only.py` | T7, T8, T9 |
| `src/phenotypic/_cli/_cli_process_single.py` | T8 |
| `src/phenotypic/_cli/_cli_directory_scanner.py` | T10 |
| `CLAUDE.md`, `_cli_readme_generator.py`, `logic_validation_scripts/…` | T11 |

Two hot files (`ngff_.py`, `_image_io_handler.py`) are touched by five of the
first six tasks, which is what forces Phases 1–2 to run sequentially.

## 3. Shapes

| Task | Shape | Why |
|---|---|---|
| T1 `write_image_class` | **Seam** | Widens a *required* parameter to optional and threads it through three call layers. Every store write in the codebase passes through it; mypy fallout is plausible. Small, risky. |
| T2 `load_zarr` guard | Seam | Behaviour change on a public read path. Inseparable from T1 — its fixture needs T1's flag. |
| T3 `omero.rdefs` | Leaf | ~5 lines, but conformance-gated. |
| T4 provenance basename | Leaf | One function, one new keyword, isolated file. |
| T5 `read_ngff_image_spec` | **Keystone** | Novel core logic — the whole projection rule, ~250 lines and 19 tests. The largest single piece. |
| T6 `imread` store branch | Seam | The public dispatch point changes shape. Thin, but it is *the* wiring. |
| T7 process-only zarr writer | **Keystone** | Produces the artifact; novel; the spec's centre. |
| T8 `--process-format` | **Seam** | Flips the default output format. This is where the AutoConvertRaw hazard lands. |
| T9 consolidation | Leaf | ~20 lines wrapping T7's branch, same file. |
| T10 scanner | **Seam** | One file, but two traps that both yield plausible wrong results, and a subtle monkeypatched test. |
| T11 validation + docs | Sweep + Leaf | Three files plus a standalone derivation script. |

## 4. Clusters

| # | Tasks | Intent | Model | Effort |
|---|---|---|---|---|
| **C1** | T1, T2 | Make a store able to omit `image_class`, and make `load_zarr` refuse one that does | Opus 5 | high |
| **C2** | T3, T4 | Two independent primitives: the render-model field and the provenance basename | Sonnet 5 | medium |
| **C3** | T5, T6 | The read path end to end — pure projector, resolver, `imread` branch | Opus 5 | high |
| **C4** | T7, T9 | The writer and its consolidation, one file, one intent | Opus 5 | high |
| **C5** | T8 | The CLI surface and the default flip | Opus 5 | high |
| **C6** | T10 | The input scanner | Opus 5 | high |
| **C7** | T11 | Validation script and documentation | Sonnet 5 | medium |

**Why T3+T4 are one cluster and not folded into C1:** they share no file with
each other, but T3 touches `ngff_.py`, which C1 also edits — so they cannot run
beside C1 anyway. Bundling the two Leaves into one sequential dispatch costs one
gate instead of two and keeps C1's diff focused on the risky widening.

**Why C3 merges a Keystone with a Seam:** T6 is ~60 lines that do nothing but
call T5. Splitting them would hand the second agent an interface it did not
design and cannot see the reasoning for. Combined they are ~350 source lines and
26 tests — the upper edge of one reviewable diff, and verifiable in one pytest
run. **This is the largest cluster; if its diff comes back unreviewable, split
at T6 and re-gate.**

**Why C5 is alone despite being small:** it is the task that changes what a
working production command emits. It earns its own gate.

## 5. Order and parallelism

```
C1 → C2 → C3 → ┬─ C4 ──┬─→ C5 → C7
               └─ C6 ──┘        ▲
                                └── blocked on an external pin (§7)
```

**C5 is deliberately last-but-one, not adjacent to C4.** Nothing depends on T8:
T11 needs T7 and T10, neither of which needs the CLI flag. Deferring C5 to just
before C7 buys the maximum amount of time for the AutoConvertRaw pin to land,
and if it never does, the run stops with 9 of 11 tasks merged and **nothing
user-visible changed** — the default `--mode process` output is still a flat
TIFF, because T7's `process_format` parameter defaults to `"tiff"` until T8
wires the option.

C7 must stay after C5: it documents `--process-format`, and
`tests/unit/test_docs_staged_cli.py` checks documented CLI flags against the
real click options, so documenting an unwired flag fails that gate.

**C6 is the one parallel-worktree candidate.** It touches
`_cli_directory_scanner.py`, which no other cluster reads or writes, and its
dependencies (T1, T6) are satisfied once C3 lands. Everything else shares a hot
file with its neighbour.

Running C6 beside C4→C5 is optional. The gain is real (C4→C5 is the longest
remaining stretch) and the merge risk is low (one file, no shared imports), but
sequential is the safe default on a stacked branch.

## 6. Gates

| When | Gate | Model |
|---|---|---|
| Before any dispatch | `plan-reviewer` over spec + plan — API claims, runnability, test validity, ordering, unnecessary complexity | Opus 5 |
| After each cluster | Light: read the diff, run the cluster's tests + `ruff` + `mypy` on changed paths | orchestrator |
| After C3 | Deep: `implementation-test-reviewer` over the combined C1+C2+C3 diff — the library phase, and the phase that adds the most tests | Opus 5 |
| After C6 | Deep: `implementation-test-reviewer` over the combined C4+C5+C6 diff — the CLI phase | Opus 5 |
| After C7 | `code-simplifier` over the whole branch diff (quality only, no behaviour change), apply, then re-run `tests/unit` | Opus 5 |

Never review with a model weaker than the one that implemented. C2 and C7 are
Sonnet-implemented and Opus-reviewed, which satisfies that.

## 7. C5 is blocked on an external pin

**Do not execute C5 until AutoConvertRaw's PhenoTypic pin is committed.**
Confirmed 2026-08-27 by a session working on that repo:

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

**Unblock condition:** ACR `HEAD` has moved past `63c4657` with the pin in it.
That is the user's call on their own production checkout, not something this
session or the peer session can decide.
