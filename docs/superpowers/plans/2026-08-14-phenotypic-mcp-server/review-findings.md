# Plan review — findings register

**Reviewer:** independent `plan-reviewer` agent, dispatched 2026-08-14, reported
2026-08-17. All seven brief areas covered; nothing skipped.

**Verification:** every citation below was checked against the code. B1 was
re-measured independently by the orchestrator before any action was taken — see
the transcript block in B1. Treat unverified reviewer claims as claims until a
task's implementer confirms them.

**Status key:** ☐ open · ☑ fixed · ◐ decision needed from the user

---

## Blockers

### ☑ B1 — Phase 1a's central premise was wrong: the leak is the eager `__init__.py`, not the modules

`phase-1a:25-31` scoped the eager `gui/shell/__init__.py` and `gui/tune/__init__.py`
out as "deferred cleanup, not a prerequisite". That is the root cause, not a side
issue. Re-measured, one subprocess per module:

```
phenotypic.gui.shell._sandbox      ['dash','dash_bootstrap_components','flask','werkzeug']
phenotypic.gui.shell._classifier   [same]
phenotypic.gui.tune._space         [same]
phenotypic.gui.run_console._state  [same]
phenotypic.gui._config             CLEAN
phenotypic.gui._operation_registry CLEAN
```

`gui/shell/__init__.py:17-20` eagerly imports `_app`, `_launcher`, `_sandbox`,
`_session`. So `_runs_registry.py:59`'s `from phenotypic.gui.shell._classifier
import classify` pulls Dash in transitively.

**Consequences:** Task 5 cannot pass the Task 1 gate. Task 2 is necessary but
**not sufficient** — the dependency Task 5 declares is real, but for a different
mechanism than the plan states. Task 7 is under-scoped by five modules
(`_setup_authoring.py:20-28` reaches `gui._config`, `gui.shell._metadata_context`,
`gui.shell._sandbox` incl. privates, `gui.shell._source_context`, `gui.tune._space`;
`_source_context.py:23` → `_classifier`; `_command.py:12-16` → `_sandbox`,
`_run_argv`; `_validation.py:7` → `gui.tune._domain_editor`).

**FIX: new Task 2.5** — make both `__init__.py` files lazy using the `__getattr__`
pattern `gui/__init__.py:31` and `gui/run_console/__init__.py:25` already use.
Ordered before Task 5. Added to `phase-1a`.

### ☑ B2 — Task 7 is ordered before Task 8 but depends on it

`gui/tune/_command.py:13-17` imports `tune_run_argv`, `tune_run_argv_from_tail`,
`tune_run_tail` from `gui.tune._run_argv`, which **Task 8** promotes. In the stated
order, `_services/tune_spec.py` imports `phenotypic.gui.tune._run_argv`, whose
package `__init__.py:19` eagerly imports `._app` → dash, and the purity gate fails.

**FIX:** swap — Task 8 runs before Task 7. Recorded in `execution.md`; C2's brief
must state it.

### ☐ B3 — Task 16's tests target Click; the tune CLI is argparse, and `--slurm` is taken

`tune/__main__.py:38` is `_build_parser() -> argparse.ArgumentParser`, exposing
`main(argv)` and `_run_command(args)`. There is no `cli` object, so every
`CliRunner().invoke(...)` test in Task 16 is unrunnable.

Worse: `--slurm` is **already** `action="store_true"` (`__main__.py:88-92`) and is
the flag that *enables* fleet submission (`slurm=args.slurm`, `:216`). It cannot
also be repeatable `key=value`. The plan's tests pass `--slurm slurm_account=exfab`
and would never enable SLURM mode at all.

Also `parse_slurm_args` raises `click.BadParameter`, which under argparse surfaces
as an unhandled traceback rather than a usage error.

**◐ DECISION NEEDED:** either keep `--slurm` boolean and add
`--slurm-arg KEY=VALUE` (`action="append"`), or make `--slurm` `action="append"`
with "present implies submit" and drop the boolean. Then rewrite the tests against
`main([...])` / `_build_parser().parse_args(...)`, and wrap `BadParameter` into
`parser.error(...)`.

### ☐ B4 — Task 16's two tests demand opposite implementations

Step 3 says to merge the four legacy flags **inside** `_submit_slurm_fleet`, but
`test_legacy_flags_still_work` monkeypatches that function and asserts
`captured["slurm_args"]["slurm_partition"] == "batch"` — i.e. that the merge already
happened **above** the call boundary. Both cannot hold.

**Recommended:** merge in `run_tuning`; have `_submit_slurm_fleet` take one
`slurm_args` dict, deleting the four `slurm_*` params (`_run.py:733-736`) and the
`if ... is not None` chain (`:798-804`). Note this changes `run_tuning`'s signature
(`:483-504`) and its call site (`:594-609`) — the plan must say so.

### ☐ B5 — Task 10 is three tasks, and its sketch violates its own instruction

`discover()` (`_operation_registry.py:188-233`) is **not** eight symmetric walks: it
is seven `(module, category, base_class)` triples through `_discover_from_module`
(`:281`, filtered `issubclass(obj, base_class)` where `base_class: Type[ImageOperation]`),
**plus** `analysis` through a separate `_discover_analyzers` (`:238`) walking the
`SetAnalyzer` hierarchy. The new modules have neither category nor base class:
`FilamentousFungiPipeline(PrefabPipeline)` is not an `ImageOperation`; the scorers
are on the scorer hierarchy. **All three assertions in the task's tests fail after
the stated change.**

Two further defects, each fatal as written:

- **`detect.nn` stays invisible regardless.** `_discover_from_module` uses
  `inspect.getmembers(module, inspect.isclass)`, which reads `dir(module)`.
  `detect/nn/__init__.py:37-63` is a module-level `__getattr__` lazy loader **with
  no `__dir__`**; `MicroSamDetector` is in `__all__` (`:65-75`) but never in the
  module dict until touched. Adding the module to a list changes nothing — it needs
  an `__all__`-driven getattr walk, and then the proposed per-module
  `try/except ImportError` sits at the wrong level, since the failure lands at
  getattr time inside the heavy imports.
- **The proposed tuple reorders `detect.nn`.** Real order
  (`_serializable_pipeline.py:645-658`) puts it tenth; the plan's tuple
  (`phase-1b:110-124`) puts it second — three lines after instructing the
  implementer to preserve order because resolution is first-match.

**FIX:** split into **10a** (lift the constant, both consumers read it, zero
behaviour change), **10b** (categories/base classes for prefabs, scorers,
strategies), **10c** (`__all__` walk for lazy modules).

### ☑ B6 — Two more dependencies beyond the ones already fixed

The earlier independence fix covered T15/T16/T18 sharing `_run.py` and T14→T10. It
missed:

- **11 → 10.** `describe_operation(name)` resolves a name to a class via
  `get_registry()`, and its own docstring promises "Operation, **scorer, or
  strategy** class name" — only reachable after Task 10's reconciliation.
- **12 → 11.** Task 12 modifies `_services/catalog.py`, the file Task 11 *creates*.

**Real DAG:** `10 → 11 → 12` and `10 → 14 → 17`; 13/15/16/18 otherwise free modulo
the `_run.py` contention. **Two chains of three, not eight parallel tasks.** The P3
cluster must not be staffed as parallel work. Recorded in `execution.md`.

### ◐ B7 — Task 18 cannot execute step 4 as specified

`_finalize_generalization` (`_run.py:907-914`) takes
`(winner, spec, output_dir, split: Split, images: list, images_by_name: dict)` —
the last three being **loaded `GridImage` objects** from `_load_images`.
`finalize_distributed_study(output_dir, *, force=False)` has none of them. It would
have to re-scan and re-read the entire calibration set (recoverable from the run
marker at `:585-591`) — minutes of I/O and a materially different task — or omit
step 4, in which case spec §7 P7's "writes the four artifact groups in the existing
order" is wrong and so is the plan.

**DECISION NEEDED:** re-load images, or drop step 4 and correct the spec.
Either way add `assert len(order) == 4` — `order.index(...) == 2` passes even if
step 4 is silently dropped.

### ☐ B8 — Task 9's `test_generator_and_builder_agree` cannot pass

The spec embeds `output_dir`-derived absolute paths:
`log_dir = logs_dir(output_dir)/"slurm"/dataset.name` and
`log_path = log_dir/f"{dataset.name}_%A_%a.log"` (`_cli_slurm_array_scripts.py:199-201`).
The test renders from `out_a` and `out_b`, so the `#SBATCH --output` lines differ
and the equality always fails. **Fix:** use one `output_dir` for both and take the
tree digest before the builder call — which the sibling test already does correctly.

Second defect: Step 3's sketch is `def generate_array_job_script(*, output_dir, **kwargs)`.
The real signature is
`generate_array_job_script(dataset, array_indices, config, output_dir, chunk_id=0, checkpoint_interval=None, is_last_chunk=False)`
(`:116-124`), called **positionally** from `_cli_slurm_array_scripts.py:484`,
`tests/unit/cli/test_cli_slurm_array.py:209,258,276,384`,
`tests/unit/cli/test_slurm_process_only_scripts.py:85,100,169,172`, and
`tests/unit/cli/test_cli_v2.py:1678,1727`. Keyword-only conversion breaks ten call
sites for no reason — keep the signature; only the new builder need be
keyword-friendly.

### ☐ B9 — Task 15's tests don't match `run_tuning`'s signature, and the guard is misplaced

`run_tuning(spec, images, output_dir, *, ...)` (`:483-504`) — **`images` is a
required positional.** All three tests omit it, so they raise `TypeError` instead of
exercising the assertion. `--slurm` also requires `spec_path` and `images_dir`
(`_validate_slurm_request`, `:664-681`), and there is
`assert effective_storage_url is not None` at `:594`. And
`test_screen_alone_still_works` as written runs a real local tuning study — not a
unit test.

**Placement:** the plan raises at `:593`, but `_write_run_marker` already ran at
`:585-591`, so a refused run leaves artifacts behind. The correct home is
`_validate_slurm_request` — whose own docstring says "Reject unsupported SLURM
combinations **before any run artifact is written**" — called at `:563`. Add a
`screen: bool` parameter to it.

---

## Improvements

| # | Finding |
|---|---|
| **I1a** | **False green.** Task 14's `test_expected_vs_detected_keeps_its_shipped_field_name` — `metadata` is a *required* field (`_expected_vs_detected.py:208`), so `ExpectedVsDetectedCount(expected_counts_csv="x.csv")` raises for **missing metadata**, and would still raise if someone added `expected_counts_csv` as an alias — the exact change it claims to guard. Replace with `assert "expected_counts_csv" not in ExpectedVsDetectedCount.model_fields` plus `assert "metadata" in ...`. |
| **I1b** | Task 12's `== 130` silently decides `derive_columns` returns measurement columns **only** — no index, no `Metadata_*`. Arithmetic verified (13 labels × (4 angles + 1 avg) = 65/scale × 2 = 130), but the contract is undeclared. State it in the Interfaces block or Phase 2A discovers it by breaking the test. |
| **I2** | Task 3's `test_discovery_stays_lazy` is a flake generator: `importlib.reload` rebinds `_services.registry.get_registry` to a new object while the shim holds the old one, so later `assert shim is canonical` fails. Definition order saves it today, but Tasks 4/5/7 all append to that same file and the standing preference is `pytest -n auto`. Assert laziness in a subprocess instead. Also: four tasks editing one test file serializes supposedly independent work. |
| **I3** | Task 6's `assert "phenotypic.gui" not in inspect.getsource(...)` is a source-text grep, not a purity check. `_space.py:29-42` has a `TYPE_CHECKING` import of `TuneRunRoot` — runtime-pure, textually fails, forcing deletion of a correct annotation. Use the Task 1 runtime gate. |
| **I4** | Task 18: (a) "`_finalize_*` have no call sites outside `run_tuning`" is true in `src/` but **false overall** — `tests/unit/tune/test_run_tuning_pareto.py:158,168` calls `_finalize_pareto_outputs`; moving the functions breaks it. (b) The monkeypatch pattern only works if `_finalize.py` does `from ._run import ...` and calls them as module globals — say so. (c) Prose says `best_params.json` is written "last"; it is **third of four**. The test correctly asserts index 2; only the prose is wrong. |
| **I5** | Task 18's test hand-joins paths the Global Constraints forbid: `out / "trials.parquet"` (`trials_parquet_path` exists at `_io_constants.py:1186`) and `/ "finalize_in_progress"` (needs a new helper). |
| **I6** | Task 1's gate is narrower than its claim: `FORBIDDEN` misses `dash_cytoscape`, `dash_ag_grid`, `plotly`; and `pkgutil.iter_modules` is non-recursive, so a future `_services/<subpkg>/` is unguarded. Two one-line fixes on the phase's single load-bearing invariant. |
| **I7** | Task 11's first-sentence rule is untested where it is hard: a naive `desc.split(".")[0]` passes every given assertion and mangles descriptions containing decimals. Add a leading-decimal case or specify a splitter requiring whitespace/EOS. |
| **◐ I8** | **18 per-task reviewers is more machinery than the problem needs.** Nine tasks are `git mv` + shim, collectively guarded by one purity gate and an unchanged GUI suite — for those the shim-identity test *is* the review. Reviewer proposes five reviewers at cluster boundaries (after 3–5, 6–9, 10–12, 14, 18) with no identifiable loss of coverage. **This contradicts the user's explicit instruction ("independent reviewer after each feature addition") and is therefore the user's call, not the reviewer's.** |

---

## What the review confirmed as sound

Stated briefly, because it is evidence the plan's analysis was mostly right:

- **DR1 is correct and better than the spec.** `gui/_config.py` measured clean;
  `IMAGE_EXTS` at `:429`; pushing it to `sdk_` rather than leaving it in
  `gui/_config` is the right call. (One drift: `_classifier` is `:36`, not `:34`.)
- **P4's bug is real exactly as described** — `if slurm: return` at `:593` precedes
  `if screen:` at `:623`.
- **Task 12 is the best-evidenced task in the plan** — `TEXTURE.get_headers` has no
  default `scale`; `MeasureColor(include_XYZ=True)` really yields three infoclasses
  vs two. All executed.
- **Task 17's premises are exact** — non-recursive `iterdir`, dataset names from one
  subdirectory level, single `click.Path` with no `multiple=True`. No single-layout
  design works. (Drift: `phenotypicCLI.py` is at `src/phenotypic/`, not under `_cli/`.)
- **Task 4 is clean**; its apparent leak today is entirely B1.
- **Task 6's split works** — every `ids.` reference sits inside a view function; all
  six line numbers exact.
- **Task 8's "to_argv cannot travel alone" is right**, though its own purity test
  fails until B1 is resolved.
- **Task 11's factual claims all check out.**
- **Spec coverage: no gap.** Every P2–P7 item has an owning task.

---

## Sequencing consequences

1. **Resolve B1 before Task 5**, or C1 ends on a red purity gate the plan forbids
   weakening.
2. **Swap Tasks 7 and 8** (B2).
3. **Do not staff the P3 cluster as parallel work** — B6 makes `10 → 11 → 12` a chain.
4. B3/B4 must be settled before C6; B5 before C4; B7 before C6's Task 18.
