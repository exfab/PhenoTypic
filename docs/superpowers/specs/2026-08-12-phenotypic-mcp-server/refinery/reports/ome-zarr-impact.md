# OME-Zarr impact on the PhenoTypic MCP spec + plan

**Date:** 2026-08-19
**Subject:** does the FINAL OME-Zarr image-store spec + plan invalidate, contradict, or add
work to the MCP server spec (`docs/superpowers/specs/2026-08-12-phenotypic-mcp-server/`) and
its plan (`docs/superpowers/plans/2026-08-14-phenotypic-mcp-server/`)?
**OME-Zarr source:** branch `worktree-ome-zarr-image-store` @ `f3564336` in the separate
clone at `/bigdata/exfab/anguy344/PhenoTypic`. Docs only — **no code exists yet**.
All citations below are `path:line` against that branch unless marked `[MCP]`.

---

## VERDICT: CHANGES NEEDED

Four documentation edits (one structural, three prose) plus one new pre-check on the
irreversible deploy path. **Three of the five originally predicted risk sites are refuted**,
including the one an earlier reviewer called "most exposed". **Site 6 is NOT at risk** —
see below, it is answered first because it gates a dispatch decision.

---

## Site 6 (answered first) — §7 P8 / USER-26 `image_manifest`: **NOT AT RISK**

**A line in `.phenotypic-mcp/plans/<token>.images` is a filesystem path to one input image
file — a `.tif`, `.png`, or raw file. It is not a `.zarr` directory and not a path-plus-array-index.
Task 19 is scoped correctly.**

Rests on **settled design**, on three independent grounds:

1. **OME-Zarr explicitly refuses zarr as pipeline input.** `design.md:1076` — Non-goals:
   *"Ingesting third-party OME-Zarr as pipeline input (the projection is write-only)."*
   The store is an **output** artifact only.
2. **The store lives under the output root, never the input root.** `design.md:164` /
   `design.md:1093` (OQ4, "**Confirmed**"): the path is
   `results/<dataset>/zarr/<stem>.ome.zarr/` — i.e. under `--output`, inside the run
   directory. `--input` is untouched by the entire change.
3. **The input scanner is not in any task's file list.** `phase-3-cli-staged.md:1525-1537`
   (Task 3.6) renames only `scan_hdf_outputs` (`_cli_directory_scanner.py:173`, glob at
   `:217`) → `scan_store_outputs`. `scan_directory_structure` (`_cli_directory_scanner.py:46`),
   which is the function MCP uses for dataset discovery
   (`10-subsets-and-promotion.md:356` `[MCP]`), appears in **no** phase document.

Corollary: **OME-Zarr does not change `scan_directory_structure`.** Verified by grepping all
eight phase docs plus `design.md` for the symbol — zero hits.

Also verified for Task 19's sake: OME-Zarr touches **no file under `_services/`**.
`RunConsoleState` and `to_argv` (`_services/argv.py`) are named in no phase document, so
P8's manifest field and argv branch are unobstructed. The only interaction is textual —
OME-Zarr Task 3.7 edits the `phenotypicCLI.py` option block "beside `--mode` at line 942"
(`phase-3-cli-staged.md:1617`), which is the same block P8 adds its option to. That is a
rebase, not a design conflict.

---

## Site 7 — CONC-18, the completion marker: **NOT AT RISK**, and the ledger rule is what saves it

**Yes, the final design defines completion markers, and they are named files — but MCP names
none of them, which is exactly why nothing breaks.**

Three distinct markers, and OME-Zarr changes only the first:

| Marker | Kind | Changed by OME-Zarr? |
|---|---|---|
| Per-image success marker (`_cli_completion.py`) | **Named file, versioned JSON** | **Yes — content schema** |
| `run_completion.json` (ordinary runs) | Named file | **No** |
| `staged_finalization_complete.json` (staged runs) | Named file | **No** |

The per-image change is `phase-3-cli-staged.md:1780` — *"Task 3.8: Per-image completion
markers must describe a store, not a file"*. `publish_image_success`'s `_sha256` opens its
argument as a file (`:1799` — *"`IsADirectoryError` on a store -- UNCAUGHT"*) and
`valid_image_success`'s `not artifact.is_file()` is False for every store. The fix is a
`kind`-tagged descriptor — `:1818`,
`{"path": <relative>, "kind": "store", "sha256": file_fingerprint(store / "zarr.json")}` —
and `SUCCESS_MARKER_VERSION` bumped `1 → 2` (`:1814`, `:1847`; the live constant is `1` at
`src/phenotypic/_cli/_cli_completion.py:26`). Five `"hdf"` artifact declarations move:
`phenotypicCLI.py:400`, `_cli_staged_slurm_worker.py:332` and `:382`,
`_cli_process_single.py:640`, `_cli_execution_strategies.py:167`.

**MCP is insulated.** `grep -rn 'success_marker|SUCCESS_MARKER|publish_image_success|valid_image_success'`
over all 11 MCP spec docs returns **zero hits**. `05-deploy-and-slurm.md:690` `[MCP]` names
only `run_completion.json` and `staged_finalization_complete.json`, neither of which OME-Zarr
touches (`git grep` for both symbols across the whole OME-Zarr spec+plan: no matches). The MCP
spec's resume/continuation description remains true. **Keep CONC-18's ruling** — stating the
contract as "the engine's completion marker" rather than a named file is precisely what made
a marker-schema change a no-op here.

**Status caveat:** the underlying finding is filed as `OPEN-QUESTIONS.md:314-316` (**D2**,
status `OPEN — needs a new task`), but the remedy is fully costed as Phase 3 Task 3.8, so
treat it as settled design with a stale status label.

---

## Sites 1-5

### 1. §5.4 `deploy_plan`/`deploy_start` `mode` — **ALREADY MOOT for the enum, CONFIRMED AT RISK for a different reason**

The enum problem is dissolved by USER-8 before OME-Zarr could cause it.
`05-deploy-and-slurm.md:328` `[MCP]`: *"`mode`, `layer` and `sample` are deliberately absent
… they are the spec's largest coupling to a storage redesign that is adding `--mode migrate`
and changing what `--mode recompile` does."* Both predicted changes are real —
`design.md:821` (*"`migrate` joins `{full, measure, recompile, process}` in the existing
`--mode` choice list"*) and `design.md:75-80` (recompile stops rewriting legacy headers, keeps
reading them) — but MCP hardcodes `--mode full`, so neither reaches a tool argument.

**The cut creates a new problem.** `phase-5-migrate.md:1261` (Task 5.7) applies a migration
refusal to *"**every mode that consumes results**, not `recompile` alone"* (`:1299`), with
`migrate` itself exempt. Its own test asserts `exit_code != 0` for `--mode full` on a
half-migrated tree. That is a fresh `sys.exit(1)` **inside the subprocess `deploy_start`
launches** — the opaque-exit failure `05-deploy-and-slurm.md:622` `[MCP]` pre-checks
`validate_resume_compatibility` to prevent. And because `mode` is cut, the server has **no
way to emit the remedy**: `--mode migrate` is unreachable from any tool.

Settled design (Task 5.7 is fully written with tests). Independently found already as ledger
`FLOW-14` and `GEN-4`, both `open · spec-change`.

### 2. §2.3 workspace layout — **CONFIRMED AT RISK**

Settled design. `design.md:164` and `design.md:1093` (OQ4 — *"**Confirmed** as
`results/<dataset>/zarr/<stem>.ome.zarr/`"*); plan `README.md:63` — *"Path:
`results/<dataset>/zarr/<stem>.ome.zarr/`. Never hand-join `f"{stem}.ome.zarr"`; always go
through `zarr_store_path(output_dir, dataset, stem)`."*
`02-state-and-identity.md:160` `[MCP]` documents `results/<dataset>/{hdf,measurements}/`.

### 3. §5.5 `deploy_status` progress unit — **NOT AT RISK. The premise is false.**

MCP never reads per-image HDF as the progress unit. `05-deploy-and-slurm.md:686` `[MCP]`:
*"**`manifest.json` is the designed polling surface for progress**"*, with counts derived
from `_manifest_builder.py:632,646`. The real builder —
`src/phenotypic/_cli/_dashboard/_manifest_builder.py` — contains **zero** `.h5`/`hdf`
references; it is format-agnostic. `git grep` over the entire OME-Zarr spec + plan for
`manifest.json|manifest_builder|run_completion|staged_finalization_complete` returns
**nothing**. Settled: matches ledger `FLOW-13`.

### 4. §5.4 staged-GPU `.npy` sidecar — **NOT AT RISK. Refuted twice over.**

First, **the MCP spec never describes the sidecar.** `grep -rn 'sidecar'` over all 11 spec
docs: one hit, `04-tune-integration.md:365` `[MCP]`, an unrelated tune export. The MCP
staged description (`05-deploy-and-slurm.md:651-665` `[MCP]` — three stages, epoch-fenced
controller, Stage 2 as a GPU array, `compute.gpu_profile` → `--gpu-slurm`) survives verbatim:
plan `README.md:16` — *"The CLI's staged-GPU engine keeps its three-stage shape."*

Second, **the earlier reviewer's premise is itself false in the final version.** The `.npy`
does not disappear; it moves. `design.md:562-563`: *"Stage 2 writes its raw detector output
to `.phenotypic/progress/stage2_raw/` and then a **consumable Stage-2 token**. It does **not**
open the promoted store."* The in-store label write was withdrawn entirely by user ruling
(locked decision #4, `design.md:41-44`, ledger GEN-37), and the raw array is *retained*
because Stage 3 re-promotes over the store's own objmap and so cannot use it as replay input
(`README.md:293`, decision **D1**).

### 5. §7 P6 subset staging vs symlinks — **NOT AT RISK**

P6 symlinks **input** images. Same grounds as Site 6: `design.md:1076` (input ingestion is a
non-goal) and `design.md:164` (the store lives under the output root). The `flat/` + `nested/`
reasoning at `10-subsets-and-promotion.md:367-390` `[MCP]` holds unchanged.

---

## Settled design vs open question — per finding

| Site | Finding rests on | Where |
|---|---|---|
| 1 (mode) | **Settled design** | `design.md:821`, `phase-5-migrate.md:1261,1299` (Task 5.7, fully costed) |
| 2 (layout) | **Settled design** | `design.md:164`, `:1093` (OQ4 "Confirmed"), `README.md:63` |
| 3 (status) | **Settled** (absence of any change) | no match in any OME-Zarr doc |
| 4 (sidecar) | **Settled design** | `design.md:41-44` (locked decision #4), `:562-563`, `README.md:16` |
| 5 (P6) | **Settled design** | `design.md:1076` (non-goal), `:164` |
| 6 (image_manifest) | **Settled design** | `design.md:1076`, `:164`; `phase-3-cli-staged.md:1525-1537` |
| 7 (completion marker) | **Settled design, stale status label** | remedy at `phase-3-cli-staged.md:1780-1847`; filed `OPEN-QUESTIONS.md:314-316` as `OPEN` |

**OME-Zarr's own live open questions** are `G1/P19` (dead `build_multiscales(resolution=)`
parameter), `G2/P18` (OME-XML failure fallback), `G3/P22` (`long_path` coverage), `G5`
(chunk-key separator not asserted store-wide), `D9`, `D10` — `OPEN-QUESTIONS.md:622-630`.
**None is a surface MCP depends on.**

`D9` (`--mode migrate`'s `metadata.csv` rewrite vs `metadata_sha256`) is the only near miss,
and it is **already settled against MCP's interest** in the authoritative document:
`design.md` §5.2 item 3 and the withdrawn supersession at `design.md:83-107` state that
migrate **never touches** `deliverables/metadata.csv`; a canonical *view* is emitted as
`metadata.canonical.csv` and no `metadata.original.csv` is created. MCP depends on neither
`metadata_sha256` nor `finalization_input_digest` (grep: zero hits across all 11 docs); its
only contact is passing `metadata_csv` through as a CLI argument
(`05-deploy-and-slurm.md:327` `[MCP]`).

> **Ambiguity in the OME-Zarr artifacts, flagged not resolved.** `D9`'s status is
> **contradictory across its own documents**: `OPEN-QUESTIONS.md:422` says `**Status:** OPEN`
> and `:629` carries it in the "Still open" table, while `design.md:83-107` records it as
> settled by user ruling and plan `README.md:315` states *"Nothing is still undecided."*
> `design.md` is the authoritative artifact and the ruling is explicit, so the MCP-facing
> consequence (metadata.csv untouched) is safe either way — but the stale rows should be
> relayed upstream. Ledger `FLOW-16` already relays this class of defect.

---

## Numbered changes

1. **`02-state-and-identity.md:160`** — change `results/<dataset>/{hdf,measurements}/` to
   `results/<dataset>/{zarr,measurements}/`, noting that `zarr/` holds `<stem>.ome.zarr/`
   **directories** resolved via `zarr_store_path()`.
   *Why:* settled design, `design.md:164` + OQ4 `design.md:1093`. Structural, not cosmetic —
   §2.3 is the layout the server's path helpers are specified against, and the section's own
   rule is that the server never hand-joins a filename.

2. **`03-tool-catalog.md:600`** — *"Passing an `output_dir` writes each snapshot to HDF5 and
   sets the dict value to `None`"* → writes each snapshot to an OME-Zarr store, via
   `save_intermediate_zarr`.
   *Why:* `design.md` §3.1 — *"`save_intermediate_zarr` replaces `save_intermediate_layers`
   … three live callers in `_image_pipeline_core.py`"*. MCP behaviour is unaffected (it
   passes `output_dir=None`); only the sentence becomes false.

3. **`05-deploy-and-slurm.md:606`** — *"it destroys `deliverables/`, every per-image HDF"* →
   per-image store. Cosmetic.

4. **`06-errors-limits-testing.md:173`** — *"Destroying measurements, HDFs, and human QC
   curation"* → stores. Cosmetic.

5. **NEW WORK — §5.4 gains a migration pre-check, and must state the remedy.**
   Add `datasets_needing_migration(output_dir)` (OME-Zarr Task 5.7, landing in
   `sdk_/_io_constants.py`) to the same pre-submit block that already calls
   `validate_resume_compatibility` (`05-deploy-and-slurm.md:622-626`), returning a structured
   `code` instead of an opaque non-zero subprocess exit.
   *Why:* `phase-5-migrate.md:1299` applies the refusal to every result-consuming mode,
   including `--mode full`, which is the only mode MCP emits. This bites on `resume`,
   `restart`, or re-deploy against a `runs/<name>` created before the OME-Zarr upgrade — a
   half-migrated tree is the *expected* state after any interruption, since migration is
   resumable.
   **Second half, do not skip:** because USER-8 cut `mode`, the server cannot offer the fix.
   §5.4 must say what the remedy is — either surface the shell command
   (`python -m phenotypic --mode migrate --output <run>`) in the error, or re-admit `migrate`
   as a narrow exception to the cut. This is a fresh refusal on the irreversible path with no
   in-server escape; it should be decided explicitly, not left implicit.
   Reconciles ledger `FLOW-14` + `GEN-4`, both already `open · spec-change`.

**Optional, accuracy only:**

- `02-state-and-identity.md:165` — the `.phenotypic/progress/{…}` enumeration gains
  `stage2_done/<dataset>/<stem>.json` and `stage2_raw/<dataset>/<stem>.npy`
  (`design.md:562-575`).
- `05-deploy-and-slurm.md:611`'s `restart` sentence (*"`clear_machine_state` wipes only
  `.phenotypic/`"*) becomes **more** true, not less. Today `--restart` also globs
  `results/*/objmap/*.npy` via `clear_stage2_sidecars` (`phenotypicCLI.py:1590`), which
  `phase-3-cli-staged.md:1351-1355` deletes as a permanent no-op once the raw arrays move
  under `.phenotypic/`. No edit strictly required.

---

## Dependencies to record, not changes to make

- **`--durable-writes`** (`phase-3-cli-staged.md:1614`, Task 3.7) is a new tri-state
  top-level CLI flag that auto-detects SLURM. `to_argv`/`RunConsoleState` cannot emit it —
  but the SLURM default is `on`, which is what MCP deploys want, so this is an optional
  enhancement, not a break. OME-Zarr touches no `_services/` file, so MCP's already-promoted
  P2 tier is safe.
- **Merge coordination:** MCP P8's manifest option and OME-Zarr Task 3.7 both edit the
  `phenotypicCLI.py` option block at line 942. Textual collision only.
- **Python floor** moves to `>=3.11, <3.13`, Python 3.10 dropped (locked decision #3,
  `design.md:37-38`; the `<3.13` cap is caused by `mahotas`, not zarr). The MCP spec states
  no floor; `fastmcp` 3.x compatibility with 3.11/3.12 is unverified — the live tail of
  ledger `FLOW-14`.

---

## Sequencing

**OME-Zarr is settled enough to write MCP Phase 2C.** Every MCP-facing surface is locked
design: store path (OQ4), mode list, migration predicate, three-stage GPU shape,
format-agnostic manifest, completion-marker shape. What is still moving is internal to
OME-Zarr — the OME projection's resolution unit, an XML fallback, long-path hygiene, one
missing test. Ledger `GEN-12` already narrows the blanket "2C waits for OME-Zarr": three of
2C's five tools are format-independent. The residual coupling is **change 5 alone**, and it
is specifiable now from Task 5.7's finished text.

**No ordering constraint on MCP Phase 1b (Tasks 10-20).** OME-Zarr's DAG is
Phase 0 → 1 → 2 → {3, 4, 5} → 6 → 7 (`README.md` Phase DAG), touching `sdk_/ngff_.py`,
`_core/_image_parts/`, `_cli/_cli_staged_*`, `gui/_shared/tiles.py`, and `_cli_migrate.py`.
No `_services/` file appears in any phase's file table. OME-Zarr has **no code yet** while
MCP Phase 1a has landed, so MCP is currently ahead.
