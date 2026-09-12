# RESUME — lazy startup (follow-up inside PR #218)

**This file is scaffolding for a paused change. Delete it in Task 8, before the branch is finished.**

Branch `refactor/private-gui` → PR #218. The private-GUI work in that PR is complete and
reviewed; what follows is a **second change riding in the same PR**: making `import phenotypic`,
the CLI and the GUI hub start lazily.

Read these two first — they are the binding authority, in this order:

1. `docs/superpowers/specs/2026-09-11-lazy-startup/spec.md` — decisions D1–D6, the measured
   baseline, behaviours B1–B4, acceptance criteria AC1–AC11, non-goals, and **Amendment A
   (P1–P14)**, which supersedes the body wherever they disagree.
2. `docs/superpowers/plans/2026-09-11-lazy-startup/plan.md` — Tasks 0–8 with the exact code to
   write, Global Constraints, and Shared facts.

The execution ledger is `.superpowers/sdd/plan/progress.md` (git-ignored). **It is the recovery
map**: every task's commit, every ruling, and every open minor is there. Trust it and `git log`
over any recollection.

---

## Where this stands

**Tasks 0–4 are complete and committed. Tasks 5–8 are not started.**

| Task | What it did | Commit |
|---|---|---|
| 0 | Baseline: timings, mypy/ruff finding sets, builder e2e | `9803376f8` |
| — | Plan review applied (5 Critical defects) | `d105397d0` |
| 1 | `HEAVY_STARTUP_MODULES`, `DEFERRED_RUNTIME_MODULES`, `load_runtime_dependencies()`, the probe helper | `e30cd2e68`, `b429da66d` |
| 2 | Lazy `__init__` for `phenotypic`, `sdk_`, `abc_`; fixed the three cycles the eager order hid | `f685e272c` |
| — | Preload guard made able to fail (it was a false green) | `2ed2a1df4` |
| 3 | `colour`, `h5py`, `matplotlib`, `plotly` moved to point of use in the image accessors | `13466a667` |
| 4 | `numba`, `cv2`, `bm3d`, `mahotas` and the plotting libraries moved to point of use in operations | `31c9e3c1c` |
| — | **Phase 1 gate**: independent review, its three findings applied and mutation-proved | this commit |

**The Phase 1 gate passed.** The independent review
(`docs/superpowers/reports/2026-09-11-lazy-startup/phase1-review.md`) returned *Ready for Phase 2*
with no Critical findings and no live defect in the 34 deferral sites. Its three Important findings
were all gaps in the **guard set** rather than bugs in the code, and all three are now closed:

- A deferred library could be re-added at the top of an operation module and **pass every one of
  the 70 existing guards** — proved by mutation. Closed by tier 5
  (`test_operation_subpackage_loads_no_deferred_runtime_library`), which asserts the whole deferred
  set is absent from each of 13 subpackages, with `phenotypic.correction`'s deliberate `colour`
  load asserted by equality rather than merely allowed.
- The watched sets were unpinned, so deleting a name from both tuples retired a guard silently.
  Closed by `test_the_watched_sets_are_the_ones_the_spec_names`.
- The per-site checker only proved that *listed* functions import their name. Closed by
  `test_no_runtime_use_of_a_deferred_name_escapes_its_local_import`, walking every runtime use's
  scope chain for a binding.

Each was proved able to fail by reintroducing the exact defect it guards. The guard suite is now
**215 passed** (was 160).

### What it bought, measured

- `import phenotypic`: **~1.57 s → ~4.5 ms**, and **~3300 modules → 71**.
- `from phenotypic import Image` no longer loads `colour`, `h5py`, `matplotlib.pyplot` or `plotly`.
- All five guard tiers are green; the startup guard suite is **215 passed**.

Full default lanes at the gate: **4 failed, 12588 passed, 33 skipped, 23 xfailed** in 11m17s. The
four failures are the pre-existing autocrlf digest failures listed below — verified by running the
same file in the pre-change baseline worktree, where the same four fail.

The CLI and GUI numbers are **not** in yet — those are Tasks 5 and 6. The after-measurements
that AC1–AC3 are scored against are Task 7's job, and must be taken on an otherwise idle
machine.

---

## What is left

Each task's full text is in the plan; `scripts/task-brief` (in the subagent-driven-development
skill) extracts one task to its own file for a fresh implementer.

- **Task 5 — CLI (9 steps).** `_CLI_RUNTIME_IMPORTS` + `_load_cli_runtime()` in
  `phenotypicCLI.py`, using `globals().setdefault` so the 27 existing
  `mock.patch("phenotypic.phenotypicCLI.<name>")` sites still win, plus a module `__getattr__`
  and a `TYPE_CHECKING` block. **Step 5b** is the optional-extras handling the user asked for:
  `DEFERRED_OPTIONAL_MODULES` is skipped unless `importlib.util.find_spec` finds it, and its
  guard test builds the expected set from the project's own dependency specifiers, excluding
  any that carry an environment marker.
- **Task 6 — GUI (11 steps).** PEP 562 lazy `__getattr__` on `phenotypic._gui.shell`; the
  builder is built on the first `/builder/` request and never released; the run console stays
  eager, because constructing it starts the SLURM observer. **A change under
  `src/phenotypic/_gui/` requires `src/phenotypic/_gui/FEATURES.md` modified in the same PR** —
  Task 6 does it, and CI rejects the PR otherwise.
- **Phase 2 gate** — independent review of Tasks 5–6 combined.
- **Task 7 — docs, the M1–M7 mutation proofs, and the after-measurements.**
- **Task 8 — final whole-branch review, `/simplify`, full regression (including e2e and a
  Sphinx build), then finish the branch.** Delete this file there.

---

## Rulings already made — do not relitigate

These are recorded in full, with their reasons, in the ledger. Summarised:

1. **ruff's baseline for this change is 25 findings in `src/phenotypic`**, not the 65 the spec
   and plan quote. 65 came from the private-gui gate's wider scope. The gates compare finding
   *sets*, not counts.
2. **Per-task reviews were dropped for Tasks 2–7**; the three independent phase gates stay.
   (This is what the user asked for; the per-task reviews were the controller's SDD default.)
3. **Tier 4 was left red at Task 3's commit and closed in Task 4**, rather than editing
   `_core/_pipeline_parts/_image_pipeline_core.py` to force it green early. The plan was
   corrected in `f71788d9a`.
4. **The preload covers only the eight hard-dependency libraries.** Optional extras and
   marker-bearing dependencies are excluded *by construction*, not by a list someone maintains.

5. **The probe deliberately does not strip `PYTHONPATH`/`PYTHONWARNINGS`.** The Phase 1 reviewer
   judged the stripping unnecessary: a shadowing `PYTHONPATH` fails the tier assertions and
   `-W error` makes the child exit non-zero, so both surface as a red guard rather than a silent
   pass. The controller had stripped them; that was reverted, and the comment at `_STRIPPED_ENV`
   records why they are absent.
6. **`_require_plotly()` runs before `import plotly.express`** in `_plotly_imshow` — a deliberate
   exception to the plan's "import first after the docstring" rule, because an import above the
   guard raises `ModuleNotFoundError` first and makes the curated message unreachable. Commented
   at the site.

## Known-open minor findings

None blocking. Two deliberate non-fixes are recorded as rulings 5 and 6 above. The Phase 1
reviewer's remaining optional polish (M5's second half — moving `MeasureBounds` past an early
return that only matters when `num_objects == 0`) was declined; the ledger says why.

---

## Running the checks

The startup guards are fast and are the right per-step instrument:

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest \
  tests/unit/ci/test_startup_imports.py tests/unit/ci/test_deferred_imports.py \
  -o addopts= -m "not slow" -p no:cacheprovider -n 4
```

Use the **`run-phenotypic-test`** skill before any wider run: four traps in this repo produce a
*wrong answer* rather than a slow one. Never `-x` for a measurement, always an explicit `-n`,
always `-o addopts=` for a run whose output goes to a file (and re-add `-m "not slow"`, which
`-o addopts=` drops).

**Pre-existing failures, not caused by this change** — verify any red test against the baseline
worktree `/tmp/pht-lazy-base` before attributing it:

- `tests/unit/test_ngff_schema_fixtures.py::test_schema_matches_recorded_digest[*]` — autocrlf
  rewrites the vendored schema bytes.
- `tests/e2e/gui/test_builder_preview_viv.py` under `-n` — a shared preview cache root; it
  passes alone.

### Static analysis

Compare finding *sets* against the baseline, never counts:

```bash
uv run mypy --no-color-output src/phenotypic > /tmp/after.txt
uv run python docs/superpowers/logic_validation_scripts/2026-09-10-private-gui-module/compare_findings.py \
  mypy /tmp/lazy-mypy-before.txt /tmp/after.txt
```

Current result: **mypy 418 = 418, ruff 25 = 25, zero new, zero gone**, both exit 0.

**The comparer used to be able to lie.** Handed colourised or default-`full`-format input it
parsed zero findings from both sides, printed "0 findings", and exited 0 — indistinguishable from
a clean run, in a gate the private-gui change also relied on. It now strips ANSI before filtering,
normalises a line number quoted inside a message (`already defined on line 644`), and **exits 2**
when an input file is neither empty nor a recognised all-clear. If you see exit 2, regenerate the
file rather than working around it.

---

## Constraints that bind anyone resuming this

From the plan's Global Constraints and the project's own rules:

- **`uv` only** — never bare `python` or `pip`. `ruff check --fix` **only with explicit paths**;
  bare, it rewrites the whole repo.
- **Never edit** `sdk_/reconnect/_tensor_voting.py`, `sdk_/branch_pathfinding/_dijkstra_kernels.py`,
  `sdk_/colourspace.py`, `sdk_/hdf_.py`, or anything under `docs/superpowers/**/refs/`.
- **No numeric change**: no algorithm, constant, public name or `__all__` list may move.
- **Stage by explicit path.** Never `git add -A`, never `git commit -a`, never `git stash`.
- **A test that cannot run must fail, not skip**, and a guard must be *proven* able to fail —
  by reintroducing the bug it guards. Two false greens have already been caught in this change
  that way; assume a third is waiting.
- The working tree is CRLF (`core.autocrlf=true`). `git diff --numstat` must never show a
  whole-file line-ending flip.
- `docs/superpowers/artifacts/2026-08-28-hyphae-detection/ALGORITHM_DECISIONS.md` carries a
  pre-existing unrelated modification. **Leave it alone**; do not stage it.

### The point-of-use import rule

Tasks 3–6 all apply the same edit shape. Delete the module-level heavy import; add the name to
a `TYPE_CHECKING` block if an annotation still needs it; add a local import as the **first
statement after the using function's docstring**. The plan's Shared facts section has the worked
example — and a warning not to trust `probes/post_change_closure.py`, whose static analysis
caused two of the five Critical defects in the plan review.
