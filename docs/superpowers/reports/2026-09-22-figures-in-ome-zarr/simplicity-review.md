# Simplicity / YAGNI review: per-image figures in the OME-Zarr store

- **Reviewed:** `docs/superpowers/specs/2026-09-22-figures-in-ome-zarr/design.md` and
  `docs/superpowers/plans/2026-09-22-figures-in-ome-zarr/plan.md`, at commit `e3df9c42`.
- **Reviewer role:** the simplicity counterweight. This is analysis only. It proposes
  cuts and adds nothing.
- **Label key:** **SPEC CHALLENGE** means the cut changes spec content the user
  approved, so it goes to the user. A cut without that label changes only the plan, and
  the orchestrator can apply it.

## Codebase facts the cuts rely on (checked by grep and read)

- **No shipped figure uses matplotlib.**
  - 27 of the 28 `@figure(` declarations in `src/` say `backend="plotly"`. The
    remaining one passes `backend=` dynamically.
  - All five shipped `PlotImage` providers draw with Plotly: `PlotDiagnostics`,
    `PlotDetectModes`, `MeasureSymZones`, `MeasureOrientationZones` and
    `_OrientationZonesReport`.
  - No product code in `src/phenotypic` produces SVG.
- **No test uses `requires_kaleido_chrome` today** (`tests/unit/cli/_kaleido_utils.py:29`).
  Task 3 brings it back into use for a single test case.
- **`replace_image_tables` has exactly one production call chain.**
  - Measure mode (`_cli_process_single.py:442`) calls
    `OutputManager.replace_image_store_measurements`, which calls
    `replace_image_tables` (`_cli_output_manager.py:1979`).
  - The private helper `_rewrite_store_tables` is shared with
    `replace_embedded_measurement_table`, which is the migrate path
    (`_measurement_tables.py:743,806`).
- **Nothing in `src/` reads a plot `manifest.json`.** That includes its `renderers`
  field and its per-page `metadata`. Only tests and people read these files.
- **Binding ids are already checked for collisions after sanitization.** The check
  applies `safe_path_component` and then casefold (`plotting/_pipeline/_bindings.py:231-248`).
- **Page keys are unique but not safe as filenames** (`abc_/plotting/_output.py:69-72`).
- **The existing writer does two things the copy-out would drop or duplicate.**
  - `publish_plot_output` takes `exclusive_path_lock(directory / ".publication.lock")`
    (`_writer.py:314`).
  - `_publish_plot_output_locked` already holds the manifest-commit block
    (`_writer.py:455-473`). Task 7 writes that block again.

---

## Ranked cuts

### 1. Drop `svg` from the closed format set (**SPEC CHALLENGE**, largest payoff)

**(a) What is removed:**
- **All of Task 1:** the probe script, installing Chrome on the HPCC node, and recording
  the outcome.
- **Every "outcome A / outcome B" branch:**
  - Task 2: the extra parametrize case and the comment on `STORE_FORMATS["svg"]`.
  - Task 3: `_serialize_svg`, the `pin_svg_ids` export and the Chrome parametrize list.
  - Task 8: `chrome_formats = {"png", "svg"}`.
  - Task 11 Step 1: the paragraph for the spec §2 gate.
- **The matplotlib SVG serializer** (the `hashsalt` and `Date` handling) and its
  cross-process determinism case.
- **The svg rows** in the media-type table and in `custom_plotter.md`.

Today, Tasks 2 and 3 cannot be written in final form until the Task 1 probe runs on a
node that has Chrome. After this cut, no part of the plan depends on an experiment's
result.

**(b) Why the spec still holds:**
- The objective is that a dashboard can show an image's figures from the store alone.
  `plotly-json` covers interactive display and `png` covers static display.
- Every shipped figure is Plotly.
- The spec's Non-goals already say the declaration "is shaped so [a format] can be
  added without a break". Adding a member to a closed `Literal` is additive, so SVG can
  return unchanged when a real figure needs it.
- No row in the decisions table covers SVG. SVG is supported only by the §2 serializer
  table and the probe gate.

**(c) Risk:** an external author who wants vector output from a matplotlib figure
cannot ask for it until SVG is added back. No shipped provider is affected, and there
is no correctness risk.

**Smaller fallback (also a SPEC CHALLENGE):** keep matplotlib `svg` and rule Plotly
`svg` unsupported without running the probe.
- The plan's code already defaults to outcome B.
- So this still removes Task 1 and every A/B branch, but keeps `_serialize_svg` for
  matplotlib.
- It departs from §2's rule that the choice is "decided by a gate, not by judgement".
  That departure costs nothing, because outcome B is the cautious result anyway.

---

### 2. Drop `html` from the store set (**SPEC CHALLENGE**)

**(a) What is removed:**
- `_serialize_html`, and the HTML use of `_stable_token`.
- The test `test_html_names_its_div_after_the_binding_and_page_not_a_uuid`, and the
  `html` cross-process case.
- The validation case `("mpl", ("html",), ...)`.
- The "the CDN means viewing needs a network" caveat in §2 and in the docs.
- The Task 3 Step 7 mutation, and the first mutation in Task 10 Step 1 (the plan itself
  says the test "does not notice" it).

**(b) Why the spec still holds:**
- `plotly-json` is lossless and interactive, and it needs no Chrome.
- A dashboard needs plotly.js to render it, and every dashboard that can show Plotly
  figures already loads plotly.js.
- The stored HTML is **not self-contained** by design, because it loads Plotly from the
  CDN. It gives the objective no capability it lacks.
- Its only advantage over `plotly-json` is opening in a browser with a double-click.
  `deliverables/` already offers that through the HTML the copy-out generates (§3), so
  deliverables stay browsable without it.

**(c) Risk:** a consumer that can only embed `<iframe src=...>`, and cannot run
`Plotly.newPlot(json)`, loses a no-code path. For a "downstream dashboard", which needs
JavaScript anyway, the risk is low.

**Supporting correctness finding.** This cut also removes a real conflict in Task 7's
`_publish_binding`. When a page stores both `plotly-json` and `html`, the function
writes `<stem>.html` twice:
- The branch for `entry["format"] == "html"` writes the stored copy verbatim, which
  loads Plotly from the CDN.
- `_write_html_from_json` writes a second `<stem>.html` that points at the hoisted
  bundle.

The later write wins, and which one comes later depends on the order of the author's
`store` declaration. `files["html"]` is overwritten without any warning. Spec §3 says
both "stored files verbatim" and "additionally writes `<…>.html` from each stored
plotly-json" but gives no rule for when both apply.

**If `html` stays, the plan needs an explicit rule.** Either skip the generated HTML
when `html` is stored, or give the two files different names. That rule is more code,
which is one more reason to make this cut.

---

### 3. Drop the Chrome-dependent determinism case and the CI Chrome lane (**SPEC CHALLENGE**)

**(a) What is removed:**
- **Task 3 Steps 1–2:**
  - the `PHENOTYPIC_REQUIRE_CHROME` environment variable in `_kaleido_utils.py`;
  - the `"chrome"` key on every entry of `.github/pytest-shards.json`;
  - the `plotly_get_chrome -y` workflow step and the workflow `env:` line;
  - the test `test_the_chrome_lane_installs_chrome_and_makes_its_marker_strict`.
- **The test** `test_a_chrome_serializer_is_stable_across_processes`.
- In total, the plan stops touching four files outside `src/` and `tests/unit/plotting`.

**(b) Why the spec still holds:**
- With cut 1 (or its fallback), the only serializer that needs Chrome is Plotly `png`.
  Its body is `pio.to_image(figure, format="png")`, and PhenoTypic changes nothing about
  it.
- The test would therefore check that **Kaleido** is deterministic. The spec's
  Background section has already measured this ("yes, per environment").
- The one PhenoTypic behaviour on this path does not need Chrome to test. That
  behaviour is that when Chrome is absent, a `PlotBackendUnavailable` failure is
  recorded for that format. Task 5's
  `test_a_declared_png_without_chrome_fails_that_format_only` pins it.
- **No default figure declares `png` for Plotly.** The §4 contract that process-mode
  stores are byte-identical only reaches this path when an author opts in.
- The §5 row saying "the plan must name a lane" exists only because of this test.
  Removing the test removes that requirement.

**(c) Risk:**
- If a future Kaleido release became random within one environment, authors who opt
  into Plotly `png` would not be warned.
- Against that, keeping the lane adds its own risk. `plotly_get_chrome -y` downloads
  Chrome on every `plots-post-viz` CI run. That shard gains a network dependency and a
  new source of flaky failures it does not have today.

**If the user keeps this test,** the setup can be smaller and still meet §5.
- Put `"chrome": true` on the one shard entry that needs it, and read the key with a
  default of false. There is no need to add it to every shard.
- The manifest test then only has to assert that one lane carries the key.

---

### 4. Drop `rebuild_figures` and make `figures` a required keyword on the replace path (plan-only)

**(a) What is removed:**
- The `rebuild_figures: bool = False` parameter on `replace_image_tables` and on
  `OutputManager.replace_image_store_measurements`.
- Its docstring paragraph and the `if rebuild_figures:` branch.
- `test_a_table_only_replace_leaves_figures_untouched`, which tests a mode no caller
  would use.

**What replaces it:**
- `figures: StoredFigures | None` becomes a **keyword-only argument with no default**,
  and the function always rebuilds. Passing `None` removes the figures.
- The two existing test callers pass `figures=None`: `tests/unit/cli/conftest.py` and
  `tests/unit/cli/test_embedded_table_inversion.py`.

**(b) Why the spec still holds:**
- The §3 measure-mode rule still holds: "the whole `figures/` group is rebuilt from the
  current pipeline". The only production caller always rebuilds.
- The migrate path is untouched. `replace_embedded_measurement_table` still calls
  `_rewrite_store_tables(..., clear_figures=False)`.
- **`clear_figures` stays on the private helper**, because migrate shares that helper.
  Its `shutil.rmtree` stays exactly as planned.
- That `rmtree` is the hard-link guard and it is load-bearing. The figure files copied
  into the part are hard links into the live store. Without the `rmtree`, the new
  figures would be written through those links into the live store. It is **not** part
  of this cut.

**(c) Risk:** a future caller that only wants to replace tables must say explicitly
what happens to figures. That is the purpose of making the keyword required. Today two
parameters can express a meaningless state: `figures=<value>, rebuild_figures=False`
silently ignores the figures. One required parameter cannot express it.

---

### 5. Shrink Task 8 (the preflight) and fold it into Task 5 (plan-only)

**(a) What is removed:**
- The new `image_store_ids` list, the new warning sentence, the change to the early
  return, and the docstring addition.
- In their place, one rule inside the existing loop: a `PlotImage` binding counts toward
  `plotly_ids` only when its declared `store` includes `png`. Otherwise it is skipped.
- The test becomes two asserts in `test_backends.py`, written as part of Task 5.

**(b) Why the spec still holds:**
- Nothing in the spec asks for a new preflight message. Neither §5 nor the Blast radius
  section mentions the preflight. The plan's self-review maps Task 8 to "Blast radius:
  preflight semantics", but no such row exists in the spec.
- What must not happen is a misleading "will publish HTML only, without PNG" warning for
  default image plots. The one-rule filter prevents that.
- A declared PNG that fails is already recorded per format, in the descriptor and in
  `.failures.jsonl`.

**(c) Risk:** for an image plot that opts into Plotly PNG, the warning uses the existing
aggregate wording ("HTML only, without PNG") instead of wording specific to the store.
The difference is cosmetic.

---

### 6. Reuse the writer's manifest commit instead of writing it again in copy-out (plan-only)

**(a) What is removed:** `_store_copyout._write_manifest`. It repeats the block at
`_writer.py:455-473`: write a temporary file, enter `_guarded_commit`, call
`os.replace`, clean up.

**What replaces it:**
- Extract `_commit_manifest(directory, manifest, *, publication_guard, commit_guard)`
  in `_writer.py`, and call it from both places.
- This is the same kind of extraction the plan already does for `unique_page_stems`, and
  it does not change behaviour.

**(b) Why the spec still holds:** the §3 requirement is unchanged: "multi-page → … +
manifest v2".

**(c) Risk:** none beyond the extraction itself, and the existing manifest tests cover
that.

**A related drift to fix at the same time.**
- The copy-out computes `renderers` as an *outcome*: "png is available when any page
  carries a PNG".
- `_writer.py:429-434` documents that `renderers` answers a *capability* question, and
  it warns explicitly against reading it as an outcome.
- No code in `src/` reads `renderers`. The simplest honest option is to state only what
  the copy-out knows: `{"html": "available"}` when any Plotly page exists.
- The alternative is to reuse the writer's computation. Either way, do not invent a
  third meaning.

---

### 7. Keep one entry point for building figures (**SPEC CHALLENGE**, naming only, trivial)

**(a) What is removed:**
- The delegate method `PlotCoordinator.build_image_figures` and its `TYPE_CHECKING`
  import.
- The spec clarification that Task 11 plans to add about it.
- All four call sites call `build_image_figures(pipeline, image)` directly. Process mode
  already does.

**(b) Why the spec still holds:** the behaviour of §3 step 1 is unchanged. Only the
spelling `PlotCoordinator.build_image_figures(image)` in §3 changes, and Task 11 edits
that sentence anyway. The coordinator keeps `publish_store_figures`, which does need its
`plots_base` and guards.

**(c) Risk:** none.

---

### 8. Do not re-export the `_image_figures` names from `phenotypic.sdk_` (plan-only)

**(a) What is removed:** Task 4 Step 4's re-export of eight names plus `FIGURES_GROUP`
into the public `phenotypic.sdk_` namespace. Every consumer in the plan, in `src/` and
in tests, imports from `phenotypic.sdk_._image_figures` directly.

**(b) Why the spec still holds:**
- The external contract is the JSON descriptor (§1), plus the stdlib-only snippet in
  `zarr_storage.md`. It is not a Python API.
- The project rule is that only `__init__` exports are public, so leaving these names
  out keeps the public API from growing when nothing calls it.

**(c) Risk:** this departs from the `write_image_tables` precedent. That name is
exported, but it too is only imported through its private path
(`_image_io_handler.py:1383`). The re-export can be added later without breaking
anything.

---

### 9. Tests that duplicate another test's guarantee (plan-only)

| Cut | Already guaranteed by | Saves |
|---|---|---|
| Task 9 `test_full_mode_without_image_bindings_writes_no_figures` | Task 5 `test_no_image_binding_builds_nothing` (build returns `None`) plus Task 6 `test_no_figures_means_no_key_and_no_group` (given `None`, the writer writes no key and no group). The wiring only passes the value through. | One full pipeline run |
| Task 3 `test_plotly_png_without_chrome_raises_backend_unavailable` | Task 5 `test_a_declared_png_without_chrome_fails_that_format_only`. It asserts `error.startswith("PlotBackendUnavailable: ")`, which can only pass if the serializer raises that exception. | One test |
| Task 10 Step 3, the "absence case" (a store without `figures` migrates and gains none) | Every existing test in `test_cli_provenance_migration.py` already uses a pre-feature store, and the plan does not touch migrate code. | One test |
| Task 10 Step 1, first mutation (a uuid div id, which "the test does not notice") | Nothing. The plan admits it proves nothing. Keep the second mutation (a timestamp in `plotly-json`). | One step |

**Keep the migrate test for a store that *carries* figures.** Migrate could rebuild
`attributes.phenotypic` from a whitelist and drop the `figures` key. That is a real risk
and nothing else tests it.

**Optional merge.**
- Two tests run process mode on a `MeasureSymZones` binding: Task 9's
  `test_process_mode_carries_figures_only_in_a_store[zarr]` and Task 10's
  byte-identical test.
- Add `assert not (first / "tables").exists()` to the byte-identical test.
- Task 9 can then keep only its `tiff` case. That saves one process run and loses no
  coverage.

---

## Task structure if cuts 1, 2, 3 and 5 are accepted

**The plan goes from 12 tasks to 9.**
- Task 1 is deleted.
- Task 3 loses Steps 1–2. Its remaining serializer module handles Plotly `plotly-json`
  and Plotly/matplotlib `png`, about 30 lines. That is small enough to fold into Task 5,
  next to its only caller.
- Task 8 folds into Task 5.
- Tasks 2, 4, 5, 6, 7, 9, 10, 11 and 12 remain.

**The closed set becomes `{"plotly-json", "png"}`.**
- `STORE_FORMATS`, `default_store_formats` and `resolve_store_formats` keep their
  current shape.
- The `backends` check still refuses `plotly-json` for matplotlib.

## Considered and not recommended for cutting

- **Per-format failure detail (§1).**
  - This is a decision on record.
  - The loop over formats exists anyway, so the extra detail costs about 10 lines:
    omitting an empty page, omitting an empty binding, and `_located`.
  - Part of the objective is that a dashboard can tell "the PNG failed but the JSON is
    fine" apart from "the figure failed".
- **`StoredFigureBinding.directory`.**
  - It looks derivable from `binding_id`. But deriving it inside `sdk_` would mean
    importing `plotting._writer.safe_path_component`, which inverts the layering.
  - Binding ids are already checked for collisions after sanitization, so the field is
    safe as it stands.
- **Extracting `unique_page_stems`.** The copy-out needs it to reproduce today's
  manifest layout, and reusing it in the store costs nothing.
- **Root-last ordering, `sha256` in the descriptor with verification on copy-out, and
  the measure-mode `rmtree` before rewriting.** All are load-bearing.
  - Root-last ordering is the publication protocol.
  - `sha256` is the only integrity check available to a consumer that reads only the
    store.
  - The `rmtree` is the only thing stopping a write through a hard link into the live
    store.
- **The cache-parity test (Task 10 Step 2).** It is the test most likely to find a real
  provider bug. It is also the evidence behind the claim that a stored figure is
  self-contained.
- **Plotly `png` in the set.** Dropping it would leave no way to get a Plotly PNG into
  `deliverables/`, which would lose something users can do today.
- **The `FigureAdapter.close` test.** It guards memory, and images are large.

## Incidental findings (outside the simplicity remit; for the general and data-flow reviewers)

- **The copy-out takes no directory lock.**
  - Today the multi-page path goes through `publish_plot_output`, which holds
    `exclusive_path_lock(directory/.publication.lock)`.
  - `_store_copyout._publish_binding` writes pages and the manifest without any lock.
  - Each image has its own directory, so this is probably harmless within one run. But
    the plan drops a guarantee without saying so.
- **The stored `html` + `plotly-json` collision**, described under cut 2.
