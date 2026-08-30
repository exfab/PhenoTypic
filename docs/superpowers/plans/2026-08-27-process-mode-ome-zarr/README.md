# Process-mode OME-Zarr Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--mode process` writes a single-series OME-Zarr store per image
carrying the operations that produced it, and `Image.imread` reads that store —
or any third-party OME-Zarr — as plain pixels.

**Architecture:** Four writer primitives land first (omit `image_class` and
consolidate inside the `.part`, guard `load_zarr`, add `omero.rdefs`, record the
pipeline basename), because everything downstream depends on their signatures.
Then a pure NGFF resolver in `sdk_` and the `imread` branch that consumes it.
Then the CLI writer, the `--process-format` option, and the option's journey out
to the command a user actually runs. Then the store-aware input identity, the
input scanner, and the logic-validation script. No new dependency is introduced.

**Tech Stack:** Python 3.11–3.12, `zarr>=3.0` (Zarr format v3), NGFF 0.5,
`jsonschema` for conformance, `click` for the CLI, `pytest` (+`pytest-xdist`).

**Spec:** [`docs/superpowers/specs/2026-08-27-process-mode-ome-zarr/design.md`](../../specs/2026-08-27-process-mode-ome-zarr/design.md)

**Branch:** `process-mode-ome-zarr`, stacked on `worktree-ome-zarr-image-store`.
Worktree: `.worktrees/process-mode-ome-zarr`.

## Global Constraints

Every task's requirements implicitly include this section. Values are copied
verbatim from the spec and from `CLAUDE.md`.

- **`uv` is the sole package manager and runner.** Never bare `python` or `pip`.
  Run tests as `uv run pytest …`, type checks as `uv run mypy src/phenotypic`.
- **`uv run ruff check --fix <paths you changed>`** — always pass explicit paths.
  A bare `ruff check --fix` rewrites the whole repo.
- **Do not edit anything under `docs/superpowers/specs/*/refs/`.** Vendored
  upstream sources are read-only evidence and must stay byte-identical.
- **Google-style docstrings** on every new public function, with a runnable
  doctest example where the function is user-facing.
- **Operations and models are pydantic v2, keyword-only.** No positional args.
- **`STORE_SUFFIX` is `".ome.zarr"`** (`ngff_.py:59`). Never derive a store's
  stem with `Path.stem` — use `store_stem`, imported from the public
  `phenotypic.sdk_` re-export (`sdk_/__init__.py:229`, defined at
  `_io_constants.py:1531`), never from `phenotypic.sdk_._io_constants`. It
  exists precisely because `Path("img.ome.zarr").stem` is `"img.ome"`. It
  **raises** `ValueError` on a path that does not end in `STORE_SUFFIX`
  (`_io_constants.py:1554-1555`), deliberately — so every caller that may see
  either a flat file or a store must branch on the suffix rather than calling it
  unconditionally.
- **`STORE_SCHEMA_VERSION` is `3`** (`ngff_.py:56`) and is gated **by value**,
  not presence, by `ngff_.require_readable_store` (`:626`).
- **NGFF axis rule (0.5 §2.4):** axes MUST contain 2 or 3 of `type:space`, MAY
  contain one `time` and MAY contain one `channel`, ordered time → channel →
  space. `rgb` is `("c","y","x")` and planar `(3,H,W)`; everything else is
  `("y","x")` and `(H,W)`.
- **`omero` is emitted completely or not at all**, and never on a float dtype or
  a label group. Do not change that guard (`ngff_.build_omero`, `ngff_.py:776-777`).
- **Only `rgb` and `gray` can be a store's sole series.** `_write_store_part`
  calls `ngff_.primary_series(series_names)` unconditionally
  (`_image_io_handler.py:1018`) and `build_phenotypic_attributes` calls it again
  (`ngff_.py:539`); that function accepts only `rgb` or `gray`
  (`ngff_.py:471-474`). `_save_store(series=("detect_mat",))` therefore raises
  `ValueError: no primary series among ['detect_mat']` — verified by execution.
  Do not widen `primary_series` to work around this; `detect_mat` and `objmap`
  are refused at the CLI instead (spec §5.3).
- **Pyramid depth is `ngff_.pyramid_level_count(h, w)`** — a pure function of the
  level-0 shape. There is no user-facing level knob; do not add one.
- **`imread` must never call `require_readable_store`.** It raises `KeyError` on
  a store with no `phenotypic` block, which is the normal case for third-party
  input and the exact case `imread` exists to serve (spec §4.6).
- **Refuse, never silently project.** A store `imread` cannot map onto the 2-D
  image model raises; it never quietly takes index 0 of an axis it does not
  understand (spec decision 13).
- **Commit after every task.** Conventional-commit subjects (`feat:`, `fix:`,
  `refactor:`, `test:`, `docs:`).

## Tasks

| Phase | Tasks | File |
|---|---|---|
| 1. Writer primitives | 1–4 | [`phase-1-writer-primitives.md`](phase-1-writer-primitives.md) |
| 2. Read path | 5–6 | [`phase-2-read-path.md`](phase-2-read-path.md) |
| 3. CLI process mode | 7, 8a, 8b, 9 | [`phase-3-cli-process.md`](phase-3-cli-process.md) |
| 4. Scanner and validation | 10a, 10b, 11 | [`phase-4-scanner-validation.md`](phase-4-scanner-validation.md) |

Subagent clustering, model assignment, gates, and the dependency DAG:
[`EXECUTION.md`](EXECUTION.md).

1. `write_image_class` and `consolidate` threading — process stores omit the
   key, and the writer can consolidate inside the `.part` before promoting
2. `load_zarr` guard — refuse a store that is not a run bundle
3. `omero.rdefs.model` on integer series
4. Pipeline **basename** in the provenance journal
5. `ngff_.read_ngff_image_spec` — the projection resolver
6. `Image.imread` store branch
7. Process-only zarr writer + provenance init
8a. `resolve_process_format` and the worker's `--process-format`
8b. `--process-format` on the user-facing CLI, threaded to every consumer
9. Consolidated metadata on a process-mode store
10a. Store-aware input digest and work identity
10b. Input scanner learns stores
11. Logic-validation script + documentation

**Dependency order.** Read `A → B` as "A must land before B".

- 1 → 2. The guard is only reachable once a store can omit the key.
- 1 → 7. The writer passes `write_image_class=False`.
- 1 → 9. Consolidation is a `_save_store` parameter, added in 1 and switched on
  in 9.
- 3 and 4 are independent of everything and of each other.
- 4 → 7. The writer passes `basename_only=True`.
- 5 → 6 → 10b. Resolver, then `imread`, then the scanner whose fixtures read
  back what it finds.
- 7 → 8a → 8b. The core takes the parameter, the worker resolves it, the
  user-facing CLI supplies it.
- 7 → 9.
- 10a → 10b. A store input raises `IsADirectoryError` in `file_sha256` until
  10a lands, so every scanner-fed run would die at work-ID derivation.
- 7, 8b, 10b → 11. The docs task documents a wired flag and a working loop.

**Nothing here is parallelisable as clustered**, and that is a change from an
earlier draft that offered `{3, 4} ∥ {5 → 6}`. Task 3 and Task 5 both edit
`ngff_.py`, and Task 10a edits `_cli_process_only.py`, which Tasks 7, 8a, and 9
also edit. Every adjacent pair shares a hot file. `EXECUTION.md` §5 records the
one place the parallelism is recoverable and what it costs.
