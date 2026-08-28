# Process-mode OME-Zarr Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--mode process` writes a single-series OME-Zarr store per image
carrying the operations that produced it, and `Image.imread` reads that store —
or any third-party OME-Zarr — as plain pixels.

**Architecture:** Four writer primitives land first (omit `image_class`, guard
`load_zarr`, add `omero.rdefs`, record the pipeline basename), because
everything downstream depends on their signatures. Then a pure NGFF resolver in
`sdk_` and the `imread` branch that consumes it. Then the CLI writer and its
`--process-format` option. Then the input scanner and the logic-validation
script. No new dependency is introduced.

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
  stem with `Path.stem` — use `sdk_.store_stem` (`_io_constants.py:1531`), which
  exists precisely because `Path("img.ome.zarr").stem` is `"img.ome"`.
- **`STORE_SCHEMA_VERSION` is `3`** (`ngff_.py:56`) and is gated **by value**,
  not presence, by `ngff_.require_readable_store` (`:626`).
- **NGFF axis rule (0.5 §2.4):** axes MUST contain 2 or 3 of `type:space`, MAY
  contain one `time` and MAY contain one `channel`, ordered time → channel →
  space. `rgb` is `("c","y","x")` and planar `(3,H,W)`; everything else is
  `("y","x")` and `(H,W)`.
- **`omero` is emitted completely or not at all**, and never on a float dtype or
  a label group. Do not change that guard (`ngff_.build_omero:777`).
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
| 3. CLI process mode | 7–9 | [`phase-3-cli-process.md`](phase-3-cli-process.md) |
| 4. Scanner and validation | 10–11 | [`phase-4-scanner-validation.md`](phase-4-scanner-validation.md) |

1. `write_image_class` threading — process stores omit the key
2. `load_zarr` guard — refuse a store that is not a run bundle
3. `omero.rdefs.model` on integer series
4. Pipeline **basename** in the provenance journal
5. `ngff_.read_ngff_image_spec` — the projection resolver
6. `Image.imread` store branch
7. Process-only zarr writer + provenance init
8. `--process-format` CLI option, layer-dependent default, objmap guard
9. Consolidated metadata on a process-mode store
10. Input scanner learns stores
11. Logic-validation script + documentation

**Dependency order.** 1 → 2 (the guard needs the omission to be reachable).
3 and 4 are independent of everything and of each other. 5 → 6. 7 needs 1 and 4.
8 → 7. 9 → 7. 10 needs 6. 11 needs 7 and 10.

Parallelisable after Phase 1: {3, 4} ∥ {5 → 6}. Phase 3 must follow 1 and 4.
