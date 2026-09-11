# Whole-branch implementation and test review — `refactor/private-gui`

**Range:** `0117572d9..ba45f8643` (18 commits) · **Reviewer role:** final whole-branch gate (read-only) · **Date:** 2026-09-10

**Inputs read:**
- `final-review-residual.diff`, all 2258 lines;
- `final-review-mechanical-files.txt`, all non-rename entries plus samples;
- `final-review-carry.md`;
- the spec, including Addendum A;
- the Task 6, Task 7 and Task 8 reviews and reports;
- `acceptance.md`.

Task 8's hunk (`_seed_error_triage_labels`) is also under concurrent review. I read it and found no defect, but defer to that review for its detail.

## Verdict

**Ready to merge.** Critical: 0 · Important: 0 · Minor: 6.

- No defect sits between tasks that breaks users or CI, and I found no test that cannot fail.
- Every guard added or changed on this branch was mutation-checked or probed: by me for AC5 and AC8, and by Task 7's logged mutants for AC9.
- All six Minor findings are cheap. Three of them, plus three carry items, are worth folding into one polish commit before merge (see Carry triage). None blocks the merge.

## Strengths

- **The rename left no half-renamed runtime paths.**
  - AC4 grep: `phenotypic.gui` / `phenotypic/gui` matches only `tests/unit/gui/test_private_package.py:4,21,22` outside `docs/superpowers/`. The control shows 614 matches inside `docs/superpowers/`.
  - Path-joined forms: the only `"phenotypic" / "gui"` hits left are the viewer cache root (`_output_root.py:1044`) and its pin (`test_viewer_cache_ownership.py:33`), both of which must stay.
  - Hub strings: no `python -m phenotypic._gui` hub string and no `"phenotypic._gui"` launch literal survive. The pattern does match the sub-app launcher forms, 20 of them.
- **Must-not-change paths are byte-identical to `main`.** Each check below compares `main` with HEAD:
  - `_output_root.py:1044` (`"phenotypic" / "gui"`);
  - `run_console/_slurm.py:438` and `_slurm_observer.py:908` (`"logs" / "gui"`);
  - `run_console/_callbacks.py:795` (`'gui' in path.parts`);
  - `phenotypicCLI.py:948` (`gui_logs.name != "gui"`);
  - `_config.py:380` (`".phenotypic-gui"`) and `_config.py:1037` (`"phenotypic-gui"`);
  - `_launcher.py:135` (`prog="phenotypic-gui"`).

  A mutated-form grep (`logs/_gui`, `phenotypic-_gui`, `tests/unit/_gui`, `tutorials/_gui`, …) is empty. The per-file `"gui"` literal diff shows only the planned package-path removals.
- **Packaging and CI moved together.**
  - `pyproject.toml:253-255` holds the `_gui/**` globs and `:145` the entry point.
  - `package-integrity.ci.yml:88-89` imports `phenotypic._gui.shell._layout` and runs `phenotypic-gui --help`.
  - `gui-checks.yml` has path filters at 29-54 and the FEATURES gate at 96-100.
  - The `.pre-commit-config.yaml` `files:` regexes point at `_gui`.
  - `scripts/check_features_md.py:31` and `check_workflows_md.py:41` point at `_gui`; both gates pass (294 shipping rows; 14/14 workflows).
  - Every Dash `assets_folder` is `__file__`-relative, so none can be orphaned by the move.
- **AC5 fails loudly in all three places.** I ran each lookup against an empty scripts directory by patching `sysconfig.get_path`:
  - `tests/integration/gui/test_console_script.py` raises `Failed`;
  - `tests/e2e/gui/conftest.py` raises `RuntimeError`, not `Skipped`;
  - `scripts/capture_gui_tutorial_screenshots.py` raises `RuntimeError`.

  In the real environment all three resolve `.venv/bin/phenotypic-gui`. `main()` calls `boot_gui` with no surrounding `except` (script lines 2115-2119), so the capture job exits non-zero.
- **The AC8 browser guard is real.** I re-ran the mutants myself against temporary copies of the manifest and workflow; the Task 6 review had only the pasted output.

  | Case | Result |
  |---|---|
  | M1: `gui-browser` gets `playwright=false` | FAIL |
  | M2: the splitter module is also owned by a non-browser shard | FAIL |
  | M2b: the `playwright` key is deleted | FAIL |
  | M3: the Chromium install line is removed | FAIL |
  | M4: the install is gated on `false` | FAIL |
  | M5: the detector is blind (so the anchor must fire) | FAIL |
  | Controls: real manifest, real workflow, workflow bytes rewritten as CRLF | pass |

  The detector finds exactly the four `*_browser.py` modules named in A1.
- **The completed-run fixture fixes the cause, and its tests can fail.**
  - `publish_complete_run_over_outputs` writes the evidence chain the resolver reads. It doesn't loosen a product guard or add a skip.
  - `/tmp/task7-mutants.log` shows M1-M5 killing exactly the targeted tests.
  - The tests run real `OutputRoot.discover` and `output_mutations_disabled` with no mocks, and prove byte stability with SHA-256 digests.

## Critical

None.

## Important

None.

## Minor

### N1. A pulled checkout keeps `src/phenotypic/gui/__pycache__/`, so the old package still resolves

- **Where:** `tests/unit/gui/test_private_package.py:22`; spec AC1 and D2.
- **What happens:** Git removes the tracked files under `src/phenotypic/gui/` but leaves untracked `__pycache__/` directories. Any existing clone, or any Known-risk worktree, that ever imported the GUI keeps a `gui/` directory, and Python resolves it as a namespace package.
- **Reproduced in a scratch repo** (`/tmp/fr_pyc.*`): check out the pre-move commit, create `pkg/gui/__pycache__/m.pyc`, then check out the move. `find_spec('pkg.gui')` returns `ModuleSpec(name='pkg.gui', loader=None, submodule_search_locations=_NamespacePath([...pkg/gui]))`. A fresh commit made in place, with no checkout, returns `None`. This is why this checkout and CI are clean (`ls src/phenotypic/gui` → no such directory).
- **Failure scenario:** After merge, a contributor pulls `main` and runs the unit suite.
  - `test_public_gui_import_path_is_gone` fails with a bare `assert ModuleSpec(...) is None`, which gives no hint that it's a stale-cache artefact.
  - `import phenotypic.gui` succeeds as an empty namespace, so old user code fails later with `AttributeError` instead of the `ModuleNotFoundError` D2's hard break promises. This only affects source checkouts, not wheel users.
- **Fix:**
  - Give the assertion a diagnosis:

    ```python
    spec = importlib.util.find_spec("phenotypic.gui")
    assert spec is None, (
        f"phenotypic.gui still resolves ({spec.submodule_search_locations}); a checkout "
        "that predates the move keeps src/phenotypic/gui/__pycache__ -- delete src/phenotypic/gui/"
    )
    ```
  - Add a one-line "after pulling, `rm -rf src/phenotypic/gui`" note to the PR description.

### N2. `_gui/_shared/tiles.py:583` cites the wrong `CLAUDE.md` line after Task 1's header

- **Where:** `src/phenotypic/_gui/_shared/tiles.py:583` says `_gui/CLAUDE.md:58 -- "Never recompute the pyramid…"`.
- **Cause:** Task 1 added a 6-line "Private package" block at the top of `_gui/CLAUDE.md`. That moved the cited rule from line 58 (`main`) to line 64 (HEAD). The mechanical substitution rewrote the path but kept `:58`.
- **Impact:** A reader who follows the citation lands in unrelated text. This is exactly the `file:line` drift the project's citation discipline exists to prevent.
  - I checked every other `file:line` citation into the four files whose line counts changed (`_gui/CLAUDE.md`, `_launcher.py`, `_gui/__init__.py`, `FEATURES.md`). This is the only one.
  - I also checked `contrib_guide/tracked_state.md`'s line citations: they point at files whose edits kept line counts, so they are still accurate.
- **Fix:** Cite `_gui/CLAUDE.md:64`, or cite the rule by name without a line number.

### N3. AC4's "no tracked file" claim has an unrecorded binary exception: `.testmondata`

- **Where:** `.testmondata` is tracked on `main` (last touched in `579f80c58`) and is unchanged by this branch.
- **Evidence:** `git grep -F 'phenotypic.gui' -- ':!docs/superpowers/'` prints `Binary file .testmondata matches`. The AC4 row in `acceptance.md` lists only `test_private_package.py`, so its grep must have skipped binaries.
- **Impact:** None at runtime. `run-pytest.yml` no longer uses testmon (`test_pr_workflow_uses_complete_shards_without_testmon`), but the record is literally inaccurate, and the stale SQLite file embeds the old module paths.
- **Fix:** Pick one:
  - record the exception in `acceptance.md`; or
  - better, as a follow-up: `git rm --cached .testmondata` and add it to `.gitignore`. That is a separate change, so it isn't needed on this branch.

### N4. The heatmap fixture derives `total_images` from the frame it seeds, which makes the count guard a no-op there

- **Where:** `tests/e2e/gui/test_heatmap_tab.py:127-130`: `total_images=df.select(_DATASET_COLUMN, str(IMAGE.IMAGE_NAME)).unique().height`.
- **Why:** The helper's count check (`tests/_output_layout.py:644-650`) exists "so a fixture cannot drift silently". When the declared count is computed from the same frame the helper reads, the check can never fire for this caller.
- **Mitigation:** The module is skipped at module level as an unmounted surface, and the parametrised frames really do list only `_IMAGES[0]` (lines 534, 734, 917). The derivation is honest about what the master holds; it just isn't a check.
- **Fix:** When the surface is remounted, pass the literal per-parametrisation count (1 for single-image frames, `len(_IMAGES)` otherwise), or note that the guard is intentionally bypassed.

### N5. The e2e recovery hint uninstalls the `docs` group (carry m2-1, confirmed)

- **Where:** `tests/e2e/gui/conftest.py:241-245` and `scripts/capture_gui_tutorial_screenshots.py:549-563`. Both say `` `uv sync --group dev --group test-qt --all-extras` ``.
- **Failure scenario:** A contributor's hub-script lookup fails, and they paste the hint. `uv sync` with an explicit selection removes packages outside that selection, so the `docs` group (Sphinx) is uninstalled. Root `CLAUDE.md`'s full dev environment includes `--group docs`; CI's selection (`gui-checks.yml:192,248`) doesn't need it.
- **Fix:** Say "run `uv sync` with your usual groups (CI uses `--group dev --group test-qt --all-extras`)", or append `--group docs`. The Task 8 implementer's own full sync used `--group dev --group test-qt --group docs --all-extras`.

### N6. `gui-checks.yml` path filters omit `tests/_output_layout.py`, which every e2e completed-run fixture now depends on

- **Where:** `.github/workflows/gui-checks.yml:29-54`.
- **Why it matters:** Task 7 made `tests/e2e/gui/conftest.py:49` and all nine mutation-capable fixtures depend on `publish_complete_run_over_outputs`. A PR that changes only `tests/_output_layout.py`, a shared helper with many importers, does not trigger the e2e job, so AC10's consumers are not re-proven.
- **Mitigation:**
  - Eight e2e modules already imported `tests._output_layout` on `main`, so the gap predates this branch.
  - `tests/gui/results_viewer/test_complete_run_fixture.py` runs on every PR in the `gui-browser` shard, since `run-pytest.yml` has no path filter. The helper's own contract stays guarded.
- **Fix:** Add `'tests/_output_layout.py'` to both `paths:` lists. This could go on this branch or in a follow-up.

## Carry triage

| ID | Decision | Reason |
|---|---|---|
| m1-1 | Keep deferred | The installed entry point is exercised end to end by `tests/integration/gui/test_console_script.py`. It runs the scripts-dir binary with `--help`, asserts `--url-prefix`, and fails on a missing script (probed). The pyproject substring pin is secondary. |
| m2-1 | **Fix on this branch** | Confirmed as N5. Two string literals; the current hint actively removes the docs group. |
| m2-2 | Keep deferred | Every documented and CI workflow uses a uv venv, where `sysconfig.get_path("scripts")` is `.venv/bin` (probed), and a miss raises. |
| m2-3 | Keep deferred | Pre-existing, CI sets `PLAYWRIGHT=1`, and it is documented in `tests/CLAUDE.md`. Changing it is outside this spec. |
| m3-1 | Keep deferred | Report-only miscounts, no code effect. |
| m3-2 / m7-4 | Keep deferred | None of the warnings is attributable to the branch. The surface's 29 warnings equal `main`'s baseline (`baseline.md:15`: `3009 passed … 29 warnings`); Task 1's 35 was transient, and Task 6 removed the `\.` SyntaxWarning. The four guard files emit 0 warnings under `-W default`, and the shard guard's whole-suite AST sweep passes under `-W error::SyntaxWarning`. |
| m6-1 | **Fix on this branch** | One token (`filename=str(path)`) at `tests/unit/ci/test_pytest_shard_manifest.py:73`. The guard parses every sharded test module, so a future SyntaxWarning in any of them would otherwise be reported as `<unknown>` inside this test. That is the trace Task 6 had to chase by hand. |
| m6-2 | Keep deferred | No instances: Task 6's AST scan of conftests and grep for string requests were both empty, and all wrapper fixtures live in unsharded `tests/e2e`. |
| m6-3 | Keep deferred | Safe-side false failure only. The assertion message can ride along with m6-1 if that commit is made. |
| m7-1 | **Fix on this branch** | Two lines (`assert images.with_columns(stem).select(dataset, stem).is_unique()`-style, or a set-size check) close a silent-drift hole in a contract the helper states in its own docstring. No current fixture triggers it. |
| m7-2 | Keep deferred | Plan-mandated, pytest never runs under `-O`, and `test_a_miscounted_fixture_fails_loudly` pins `AssertionError`. |
| m7-3 | Keep deferred | The ordering half of the rationale survives in `_build_sandbox`'s comment (conftest.py:182-183). The dropped half concerns a legacy-manifest shadow the manifest-only helper still avoids by writing through `manifest_json_path`. |

**Ruled items:** I found no new evidence against any ruling.
- **Duplicated lookup:** all three copies fail loudly (probed).
- **FEATURES row:** the entry-points rows at `FEATURES.md:778-779` are correct.
- **Screenshots:** no rendered chrome changed.
- **Criterion 4 exception:** holds.
- **Trailers, `test_builder_preview_viv`, `Metadata_StrainID`:** not code findings here.

**Out-of-scope follow-ups to relay to the user:**
- all seven items in `final-review-carry.md`;
- N3's `.testmondata` untracking;
- the `docs/superpowers/plans/2026-09-01-results-scatter-tab/verify_scatter_fixture.py` and `2026-09-03-cli-gui-state-tracking/curation_fence_probe.py` executables. They still import `phenotypic.gui`, so they no longer run. D4 covers them as historical plan artefacts, and nothing outside `docs/superpowers/` references them.

## Checks run

Each check is listed as the named risk, the check performed, and the result.

- **AC4, residual old names.**
  - Check: `git grep -F 'phenotypic.gui'` and `-F 'phenotypic/gui'` outside `docs/superpowers/`. Control: 614 matches inside `docs/superpowers/`.
  - Result: only `test_private_package.py`, plus binary `.testmondata` (N3).
- **Half-renamed path joins and launch strings.**
  - Check: regexes for `"phenotypic" / "gui"`, `joinpath("gui")`, `python -m phenotypic._gui` with no sub-app, and `"phenotypic._gui"` launch literals. Controls: `"phenotypic" / "_gui"` matches 3 files; `-m phenotypic._gui.` matches 20 sub-app forms.
  - Result: only the Must-not-change cache root and its test.
- **Must-not-change runtime paths.**
  - Check: `main` vs HEAD line comparison at the seven sites named in the spec; a mutated-form grep; a per-file `"gui"` literal count diff.
  - Result: identical; mutated forms absent; removals are the planned package paths only.
- **Packaging, CI filters, pre-commit, `assets_folder`, `importlib` literals.**
  - Check: grep of `pyproject.toml`, `.github/`, `.pre-commit-config.yaml`, and `assets_folder` / `find_spec` / `import_module` under `_gui`.
  - Result: all repointed or `__file__`-relative; no `MANIFEST.in` / CODEOWNERS / coverage references exist.
- **Deleted generators referenced by docs or CI.**
  - Check: grep for `generate_dispatch_reference|generate_validation_reference|_reference_generator|api_reference/gui` outside `docs/superpowers/`.
  - Result: none.
- **Seam, Task 1 rename × Task 2 `sysconfig` lookup.**
  - Check: probed all three lookups (real env resolves `.venv/bin/phenotypic-gui`; empty scripts dir → `Failed` / `RuntimeError` / `RuntimeError`). Every CI job that runs them (`gui-checks.yml:192,248`, `run-pytest.yml:127`, `run-pytest-full.yml:81,117,152`) runs `uv sync`, and `[tool.uv]` does not set `package = false`.
  - Result: the console script exists wherever these checks run.
- **Seam, Task 7 top-level `tests._output_layout` import in the e2e conftest.**
  - Check: `tests/__init__.py`, `tests/e2e/__init__.py` and `tests/e2e/gui/__init__.py` are tracked, and the helper's module-level imports are stdlib plus `phenotypic.sdk_`. `tests/e2e` is not in `testpaths`.
  - Result: no new collection cost or failure ahead of the PLAYWRIGHT skip.
- **Seam, Task 6 guard × real `.github/pytest-shards.json` / `run-pytest.yml`.**
  - Check: the mutation table in Strengths, plus a CRLF control.
  - Result: all mutants fail, controls pass, and `read_text` newline translation makes the regex CRLF-safe.
- **Seam, Task 3 docs × code.**
  - Check: `--url-prefix` is in `phenotypic-gui --help` (asserted by `test_phenotypic_gui_help_succeeds`); exactly five `__main__.py` under `_gui` (`git ls-files`) match `CLAUDE.md:241`; no `_gui/__main__.py`.
  - Check: the `python -m phenotypic.gui.sweep` line removed from `getting_started.rst` was already stale (`git ls-tree 0117572d9` has no `sweep`).
  - Check: the AC6 grep of README and `docs/source` for `python -m phenotypic._?gui` / `phenotypic._gui` / "standalone launch" / `--output-root` finds only contributor-guide source-file paths in `tracked_state.md`.
  - Result: docs and code agree.
- **AC5, other hub launches.**
  - Check: every `Popen` / `sys.executable` in `tests/e2e`, `tests/integration/gui` and `tests/gui`.
  - Result: two start Xvfb, and the two tune modules boot the unmounted `_gui.tune` sub-app via `create_app`, not the hub. The capture script's other boots are sub-app launchers (lines 1826, 1917).
- **`file:line` drift from line-count changes.** Grep for citations into `_gui/CLAUDE.md`, `_launcher.py`, `_gui/__init__.py` and `FEATURES.md` on `main` and HEAD found one stale citation (N2).
- **Live executables under `docs/superpowers/`.**
  - Check: grep of `*.py` / `*.sh` / `*.sbatch` for old names, and of references to them from outside.
  - Result: only the historical plan probes (D4); the live `run_unit_suite.sbatch` does not name the package.
- **Test integrity, `test_private_package.py`.**
  - Check: namespace-leftover probe (N1) shows the guard can fail; ruff clean; passes.
  - Result: sound. The pyproject substring is m1-1, deferred.
- **Test integrity, `test_complete_run_fixture.py`.**
  - Check: read all five tests; `/tmp/task7-mutants.log` M1-M5; stem derivation. The helper's `Path(image).stem` agrees with `source_image_stem` (`sdk_/_io_constants.py:1808-1824`) for every non-`.ome.zarr` name, and all e2e image lists are `plate_00N.tif`.
  - Result: sound. The latent duplicate-stem case is m7-1.
- **Test integrity, `test_console_script.py`.**
  - Check: no module-level skip or `importorskip`; `pytest.fail` on a missing script (probed); `sys` is still used (line 72).
  - Result: sound.
- **Lint and warnings (AC7 / m3-2).**
  - Check: `ruff check` on the seven materially changed Python files, `main` vs HEAD: 0 → 0 each; both new files clean.
  - Check: focused run `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest tests/unit/gui/test_private_package.py tests/unit/ci/test_pytest_shard_manifest.py tests/gui/results_viewer/test_complete_run_fixture.py tests/integration/gui/test_console_script.py -q -o addopts= -p no:cacheprovider -W default` → `21 passed`, no warnings.
  - Check: `tests/unit/ci/test_pytest_shard_manifest.py -W error::SyntaxWarning` → `4 passed`.
  - Result: no new lint findings or warnings.
- **AC9-AC11 evidence.**
  - AC9: Task 7 mutant log present.
  - AC10: Task 7 e2e and `ci_flaky` logs reviewed there. The vacuity for the three module-skipped surfaces is already recorded as a carry follow-up.
  - AC11: Task 8 harness GREEN (`authorized images: 3; marker source_image_count: 3; source_set_digest matches authorized set: True`) and a full capture reaching `exit=0`. `_current_success_work_ids(OUTPUT_DIR, state.config.get("work_ids", {}))` is the same call `sdk_/_hdf_to_zarr.py:779-781` makes.
- **Mechanical split sanity.**
  - Check: sampled `tools/viv-bundle/build.mjs`, `NOTICE`, `tests/unit/gui/test_viewer_cache_ownership.py`, `.github/workflows/gui-checks.yml`, `.pre-commit-config.yaml`, `.claude/skills/gui-tutorial-capture/SKILL.md` and `src/phenotypic/_assets/__init__.py`.
  - Result: every changed line is a pure `gui → _gui` substitution, and the cache-root assertion is untouched.
