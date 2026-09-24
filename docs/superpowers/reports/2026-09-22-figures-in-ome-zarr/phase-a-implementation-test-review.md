# Phase A implementation and test review: per-image figures in the OME-Zarr store

- **Reviewed:** `git diff e8679593..5476af60` (Tasks 1–5: d4bd5f4e, ee194084, 151dfcac,
  b575300c, 5476af60), checked against `specs/2026-09-22-figures-in-ome-zarr/design.md`
  (including its Revision section) and `plans/2026-09-22-figures-in-ome-zarr/plan.md`
  Tasks 1–5.
- **Method:** I read every changed source and test file in full, plus the code they call:
  `_writer.py`, `_coordinator.py`, `_bindings.py`, `ngff_.promote_store`/`fsync_tree`,
  `_rewrite_store_tables`, `_write_store_part`, `atomic_write_json` and
  `_consolidate_store_part`. The orchestrator ran one probe for me (output quoted
  under **Probe** below). I did not run any test or mutation myself.
- **Severity key:** BLOCKER means it must be fixed before Phase B builds on it. MAJOR
  means it should be fixed in the Phase A simplify/fix pass. MINOR means fix it when
  convenient, or record it as a known limitation.

## Summary

| Severity | Count |
|---|---|
| BLOCKER | 0 |
| MAJOR | 2 |
| MINOR | 13 |

- **MAJOR-1:** The page-metadata check is weaker than the writers that consume it. A
  measure-mode rewrite fails the image, and a `NaN` makes the store root non-standard
  JSON. Both were confirmed by the probe.
- **MAJOR-2:** Copy-out takes its layout and its manifest `failed` list from the pages
  that *survived* in the store. A failed page disappears from the manifest, a
  multi-page binding can publish flat, and an all-failed binding leaves the previous
  manifest in place. Confirmed by the probe.

## Probe (run by the orchestrator, output quoted verbatim)

```
P1 failed: () bindings: ['Mixed']
P1 full save2zarr OK
P1 measure rewrite FAILED: TypeError '<' not supported between instances of 'str' and 'int'
P2 failed: () bindings: ['NanMeta']
P2 full save2zarr OK
P2 strict JSON parse of root FAILED: ValueError non-standard JSON constant NaN
P2 process zarr write OK
DefaultPlusBad bindings: ['DefaultPlusBad'] failed: [('bad', None)]
DefaultPlusBad files: ['.failures.jsonl', '.failures.lock', '.plotlyjs.lock', 'DefaultPlusBad/ds/img-39263368a266.html', 'DefaultPlusBad/ds/img-39263368a266.plotly.json', 'plotly.min.js']
TwoOneBad bindings: ['TwoOneBad'] failed: [('bad', None)]
TwoOneBad files: [..., 'TwoOneBad/ds/img-39263368a266/good.html', 'TwoOneBad/ds/img-39263368a266/good.plotly.json', 'TwoOneBad/ds/img-39263368a266/manifest.json', 'plotly.min.js']
TwoOneBad manifest pages: ['good'] manifest failed: []
Empty bindings: [] failed: []
Empty files: []
```

What each case built:

- **Mixed:** one page with `metadata={1: "x", "b": 2}`.
- **NanMeta:** one page with `metadata={"v": float("nan")}`.
- **DefaultPlusBad:** pages `default` (Plotly) and `bad` (`object()`).
- **TwoOneBad:** pages `good` and `bad`.
- **Empty:** `PlotOutput(pages=())`.

---

## MAJOR

### MAJOR-1: The metadata check is weaker than the writers that consume it

**Evidence:**

- `src/phenotypic/plotting/_pipeline/_store_figures.py:131-134`: `metadata =
  dict(page.metadata)` followed by `json.dumps(metadata)`, with no `sort_keys` and
  default `allow_nan=True`.
- The measure-mode rewrite writes the root through `atomic_write_json`
  (`sdk_/_atomic_io.py:209`), which defaults to `sort_keys=True`.
- The full-mode and process-mode root is written by `_write_group_json`
  (`_image_io_handler.py:976`), with `default=str` and NaN allowed.

**Two confirmed failure modes:**

1. **Mixed key types fail the image in measure mode (probe P1).**
   - The build accepts the page, because `json.dumps({1: "x", "b": 2})` succeeds, and
     the full-mode save succeeds.
   - `replace_image_tables` then raises `TypeError: '<' not supported between
     instances of 'str' and 'int'` while sorting the root. That raise happens inside the
     store transaction, so the image fails.
   - This violates the global constraint "no figure error may fail an image". It is
     exactly the case the code comment at `:132-133` claims to prevent. The comment
     accounts for `default=` but not for `sort_keys`.
   - Such metadata violates `PlotPage.metadata`'s annotation
     (`Mapping[str, scalar]`, `abc_/plotting/_output.py:53`), but nothing enforces that
     annotation.
2. **A NaN value makes the root `zarr.json` non-standard JSON (probe P2).**
   - `float("nan")` is inside the declared value type (`float`). The store writes a
     literal `NaN`.
   - Python and zarr-python accept it, so the process write still reports OK. A strict
     parser (a browser `JSON.parse`, i.e. the downstream dashboard this spec exists
     for, or the Viv/zarrita path the results viewer uses) rejects the **whole root**,
     not just the figure.
   - This turns a figure-level value into a store-level unreadability.

**Scope:** No shipped `PlotImage` provider sets page metadata today (grep:
`PlotPage(` appears only in `_output.py:16` and `_plot_meas_time_series.py:138`, a
`PlotMeas`). So this is latent, but it sits on the author-facing surface.

**Fix (one line):** normalise instead of checking.
`metadata = json.loads(json.dumps(dict(page.metadata), allow_nan=False, sort_keys=True))`.

- It refuses NaN/Inf and mixed or unsortable keys as a page failure.
- It stores exactly what a reader will read back: int keys become str, tuples become
  lists.
- It keeps full-mode and measure-mode descriptors identical in content.

**Test gap:** spec §5 row "metadata not JSON-native … the store still publishes (all
modes, including the measure-mode root rewrite)" is covered only on the build side
(`test_store_figures_build.py:58`, `np.int64`). No test drives a built page through
`replace_image_tables`, which is how this slipped. Add P1 and P2 as tests. P2 should
assert that the root parses with `parse_constant` raising.

### MAJOR-2: Copy-out derives layout and manifest `failed` from surviving pages only

**Evidence:**

- `src/phenotypic/plotting/_pipeline/_store_copyout.py:143-150`: `flat = len(pages) ==
  1 and pages[0]["key"] == "default"`, where `pages` is the descriptor's pages. Those
  are the pages that stored at least one file.
- `:237-239`: manifest `failed` is filled only for descriptor pages whose files could
  not be copied.
- `:252`: store failures are consulted only for `partial`, and only for keys that are
  still in `pages`.
- `:114`: a binding absent from `bindings` is never passed to `_publish_binding`.

Compare the retired writer:

- `_coordinator.py` `_publish_image_value` decides flat versus directory from **all**
  pages of the `PlotOutput`.
- `_writer.py` `_publish_plot_output_locked` lists every page that produced nothing in
  `failed`.

**Consequences:**

1. **A whole-page store failure vanishes from the manifest (probe TwoOneBad).**
   - Manifest: `pages: ['good'], failed: []`.
   - The `bad` page is recorded only in `.failures.jsonl`. The manifest, which is the
     per-directory authoritative record, no longer says the page existed.
   - Spec §3: "outcomes stay in `files`, `partial` … and `failed`". The plan (Task 5,
     "`failed` lists pages that had no file copied") narrowed this, so the gap
     originates in the plan.
2. **The layout flips with failures (probe DefaultPlusBad).**
   - A two-page output whose non-`default` page failed is published flat as
     `<stem>.{html,plotly.json}`, not as `<stem>/default.*` with a manifest.
   - On the next run, if both pages succeed, it publishes to the directory, and the flat
     files from the earlier run are orphaned beside it with nothing marking them stale.
3. **An all-pages-failed multi-page binding leaves a stale manifest (code reading;
   UNVERIFIED by probe).**
   - The binding is absent from `bindings`, so `_publish_binding` never runs, and the
     previous run's `manifest.json` stays authoritative, listing pages from a run that
     no longer describes this image.
   - The old writer would have committed `pages: [], failed: [...]`.

**Fix, with no descriptor change:** the descriptor's `failed` entries already carry
the page key.

- For each binding, take the union of its published page keys and the non-null `page`
  values in its `failed` entries.
- Decide `flat` from that union (flat iff the union is exactly `{"default"}`).
- Add a `failed` manifest row for every failed key that is absent from `pages`.
- Call `_publish_binding` for a binding that has page-level failures but no published
  pages. It then commits an empty-pages manifest for the directory case, and changes
  nothing for the flat case (keeping "a page that published nothing keeps its previous
  files").
- A binding-level failure (`page: null`) keeps today's behaviour: a record only, no
  manifest.

**Tests to add:** the three cases above, asserting on manifest contents and on the flat
versus directory path.

---

## MINOR: implementation

### MINOR-1 (open point a): `replace_image_tables` public signature break

- **Evidence:** `sdk_/__init__.py:37,460` re-exports it, and
  `sdk_/_measurement_tables.py:716` makes `figures` a required keyword.
- **Rating: MINOR.** The only production caller is
  `OutputManager.replace_image_store_measurements`, and the break was planned (plan
  line 83, 1301).
- **Why the naive fix is unsafe:** the parameter's only non-`None` type,
  `StoredFigures`, lives in a private module. A public caller can therefore only pass
  `None`, and `None` **deletes** the store's figures. An external caller who "fixes" the
  `TypeError` by adding `figures=None` silently strips figures. Giving it a default of
  `None` would do that to *every* external caller.
- **Recommended:** a module-level sentinel default, e.g. `figures=KEEP_FIGURES`, that
  maps to `clear_figures=False` and carries the group across byte-for-byte. The public
  signature stays compatible, and removal stays an explicit act. If you keep the
  required keyword, add a CHANGELOG line.

### MINOR-2 (open point b): `read_image_figures_descriptor` raises instead of returning `None`

- **Evidence:** `sdk_/_image_figures.py:151-158` goes through
  `ngff_.read_phenotypic_attributes` (`ngff_.py:653-667`). That raises `KeyError` when
  the root has no `phenotypic` block and `FileNotFoundError` when there is no root. The
  docstring says it returns "`None` when it has none".
- **Rating: MINOR.** The only caller today, copy-out, wraps it in a broad `except`
  and records `<store>` (`_store_copyout.py:99-107`). Copy-out only runs on stores this
  run promoted, which always have the block. The function is the natural entry point
  for the dashboard reader the spec targets, and a third-party store is the common
  input there.
- **Recommended:** read `read_root_attributes(...).get(ROOT, {})`, and document
  `FileNotFoundError` for a missing root.

### MINOR-3 (open point c): the preflight wording is wrong for a PNG-only image plot

- **Evidence:** `_backends.py:204` puts any Plotly image plot that declares `png` into
  `plotly_ids`, which is reported as "Plotly plots will publish HTML only, without PNG"
  (`:227`).
- **Rating: MINOR,** but worse than a wording nit.
  - That sentence is accurate for `store=("plotly-json", "png")`: deliverables get
    `.plotly.json` and `.html`.
  - For `store=("png",)` it is **false**. Without Chrome the page stores no file, the
    binding is absent, and copy-out publishes nothing at all, not even HTML
    (`test_a_binding_whose_every_page_failed_is_absent` pins that store side).
- **Recommended:** a separate list for image plots, e.g. "N image plots declare a PNG
  that will be recorded as failed: …". Say "will publish nothing" when `plotly-json` is
  not also declared.
- **Test gap:** `test_backends.py` `test_image_plots_need_chrome_only_when_they_declare_png`
  asserts only that the id is in the line, so no wording is pinned.

### MINOR-4 (open point d): a failed HTML generation is recorded as `format="plotly-json"`

- **Evidence:** `_store_copyout.py:223-236`. `_write_html_from_json` runs inside the
  `plotly-json` entry's `try`, and its exception is recorded with
  `fmt=entry.get("format")`.
- **Rating: MINOR.**
  - The `.failures.jsonl` `format` field is spec'd (§3) as the *store* format a failure
    concerns. Here the stored JSON was verified and copied successfully.
  - A triager filtering `format == "plotly-json"` to find store corruption gets a false
    hit, and nothing distinguishes the two cases.
  - The manifest side is correct (the page is published, and `partial` carries the
    error).
- **Recommended:** give the HTML step its own `try` and record it as
  `fmt="html"` (document that `format` may also name a deliverable rendering), or record
  it with no `fmt`.
- **Test gap:** `test_a_failed_html_generation_is_partial_on_a_published_page` asserts
  only `r["page"]`, so neither choice is pinned.

### MINOR-5: an empty `PlotOutput` makes a binding vanish silently

- **Evidence:** `_store_figures.py:77` appends a binding only `if pages:`. If
  `inspect()` returns `PlotOutput(pages=())`, `normalize_plot_output` passes it through
  (`_output.py:12-13`), and the binding ends up in neither `bindings` nor `failed`
  (probe **Empty**).
- Spec §3: "absence means 'not configured'". This is the same defect class the
  implementation fixed for `None` (plan-review minor 4), reached by a different route.
- The old writer committed an empty manifest here, because `len(pages) != 1` sends it
  down the directory path.
- **Recommended:** treat zero pages exactly like `None`: one `page: null, format: null`
  failure.

### MINOR-6: a figure write error now fails the store, and with it the image

- **Evidence:** `sdk_/_image_figures.py:96-108` runs inside `_write_store_part`
  (`_image_io_handler.py:1406-1412`), and its exceptions abort the whole store
  transaction.
- The build boundary covers rendering only, so filesystem errors on figure files are
  store errors. That is consistent with tables. Two sources are figure-specific:
  - `safe_path_component` (`_writer.py:95-114`) does not reject Windows reserved names.
    A binding id `CON`/`aux`/`nul` is a valid binding (`_bindings.py:92-100`) and would
    fail every store on Windows.
  - `write_bytes` does not go through `ngff_.long_path`, so a long page key near
    `MAX_PATH` can fail the store.
- Under the retired writer both of these were best-effort deliverable failures.
- **UNVERIFIED:** not exercised on Windows.
- **Recommended:** reject reserved device names in `safe_path_component` (it helps
  deliverables too), and use `ngff_.long_path` for the figure file writes.

### MINOR-7: copy-out trusts descriptor paths and ignores `schema_version`

- **Evidence:** `_store_copyout.py:206` reads `store / entry["path"]` with no check
  that the path stays under `store/figures/`. The descriptor's `schema_version` is never
  checked.
- **Rating: MINOR** (hardening). Stores are self-produced, and the `sha256` must still
  match, so the risk is low.
- A future schema-2 descriptor would currently be interpreted as schema 1 without any
  warning.
- **Recommended:**
  - Resolve the path and require `is_relative_to(store / "figures")`.
  - Skip, and record, a descriptor whose `schema_version` is not 1.

### MINOR-8: the manifest `class` disagrees with the failure-record class

- **Evidence:** `_store_copyout.py:178` writes `"class": binding.get("class",
  binding_id)`, taken from the descriptor. The records at `:77`/`:97-98` prefer the
  pipeline's class.
- For a published binding the two are the same value unless a class was renamed between
  the store write and the copy-out (measure mode on an old store). This is cosmetic.
- **Recommended:** use `classes[binding_id]`, which is already the merged map.

### MINOR-9: interim state must not ship without Task 6

These are not bugs at the phase boundary. Record them in the Phase B gate:

- `_cli_process_single.py:446` passes `figures=None`. After Task 6 wires the full-mode
  store write, a measure-mode run on a figure-carrying store would **delete** its
  figures until this line is replaced.
- The preflight change (`_backends.py:200-212`) already assumes copy-out semantics.
  Meanwhile `emit_image`, still live, renders PNGs whenever Chrome is present, and on a
  Chrome-less run it now does so without any preflight warning.

---

## MINOR: test validity

### MINOR-10: a collision-safety assertion that passes vacuously

- **Evidence:** `test_store_figures_build.py:127-128`: `assert mpl_name.endswith(".png")
  and mpl_name != "A-b.png"`.
- With `unique_page_stems` replaced by plain `safe_path_component` (no case-folded
  collision handling), the mpl page's name is `"a-b.png"`. That still passes, because
  `"a-b.png" != "A-b.png"`.
- **Fix:** `assert re.fullmatch(r"a-b-[0-9a-f]{8}\.png", mpl_name)`, or
  `mpl_name.casefold() != "a-b.png"`.

### MINOR-11: the pipeline-class test cannot tell class from id

- **Evidence:** `test_store_copyout.py:212-218`. The binding id defaults to the class
  name (`Explodes`), so a regression that falls back to `binding_id` (instead of
  `plot_classes`) still yields `plot_class == "Explodes"`.
- **Fix:** bind with an explicit id that differs from the class name.

### MINOR-12: the spec §5 metadata row is unguarded in measure mode

- See MAJOR-1. The build-side test exists; the "including the measure-mode root
  rewrite" half has no test.
- This may be intended for Task 7, but MAJOR-1 shows it is not only a wiring concern.

### MINOR-13: test-helper temp directories leak

- **Evidence:** `tests/unit/plotting/_store_fixtures.py:185` calls `mkdtemp(...,
  dir=output_root.parent)` and never removes the result. The directory lands in pytest's
  per-session base directory, so it is bounded by pytest's retention. This is hygiene
  only.
- **Fix:** use a `tmp_path_factory` fixture, or clean up in a `finally`.

---

## Checked and found correct

These are recorded so the next phase does not re-check them.

- **`PlotPublicationBlocked` is never swallowed.**
  - Build: `_store_figures.py`, at the binding, page and format levels.
  - Copy-out: the read, the early guard, the per-file S11 unlink-and-reraise,
    `_remove_leftovers` through `_guarded_commit`, and `_commit_manifest`.
  - `record_plot_failure` never raises, so every `except Exception` that records is
    safe.
  - The guard is asked before the first record and before the read-failure record. Both
    are pinned by tests that fail if the early check is removed
    (`test_a_refused_guard_propagates_before_anything_is_written`,
    `test_an_unreadable_store_asks_the_guard_before_recording`).
- **Hard-link safety in `_rewrite_store_tables`.**
  - `clear_figures=True` removes the part's hard-linked copies before `write_bytes`
    (`_measurement_tables.py:691-696`).
  - The held-descriptor test (`test_image_figures_store.py:97-110`) is a genuine
    guard: without the `rmtree`, `write_bytes` opens the linked inode with `O_TRUNC`
    and the held fd reads the new bytes.
  - The inode test proves that `copytree` actually links, so the held-fd test is not
    vacuous.
  - Group documents go through `atomic_write_json` (replace), so they are safe either
    way.
  - The migrate path (`clear_figures=False`) carries figures across untouched, and that
    is pinned.
- **Root-last ordering.**
  - Full and process mode: figures are written at `_image_io_handler.py:1406-1412`,
    before the root is built and written, and before `_consolidate_store_part`, so the
    consolidated metadata includes `figures` and `figures/<binding>` (pinned).
  - Measure mode: written in `populate`, before the root. The promote-time spy asserts
    that the hashes match at the promote.
- **Durability.** `fsync_tree` walks every file under the part (`ngff_.py:1700-1724`),
  so figure files are covered under `--durable-writes` with no extra code.
- **Binding-directory collisions** are impossible: `normalize_plot_bindings` refuses
  ids that collide after case-folded sanitization (`_bindings.py:233-248`). Page stems
  are made unique by `unique_page_stems`, and the two extensions never collide with each
  other or with `zarr.json`.
- **Lazy imports, by reading (UNVERIFIED by running the guards).**
  - `abc_/plotting/_store_formats.py` is stdlib-only.
  - Every plotly/matplotlib/zarr import in the new code is function-local.
  - The `_image_figures` imports in `_image_io_handler.py`, `_measurement_tables.py`,
    `_cli_output_manager.py` and `_cli_process_only.py` are under `TYPE_CHECKING` or
    function-local.
  - `plotting/_pipeline/__init__.py` does not import `_store_copyout` or
    `_store_figures`.
- **Determinism.**
  - The serializers are as spec'd (`to_json()` removes uids by default).
  - Error addresses are masked (`_store_figures.py` `_ADDRESS`).
  - Descriptor order follows the pipeline, page and declared-format order.
  - Page stems do not depend on which pages failed, because they are computed over all
    pages before failure.
  - The cross-process test uses two subprocesses with different `PYTHONHASHSEED`.
    Store-level byte identity is Task 7's row.
- **Decisions under review (all accepted):**
  - S11 parity on a refused guard.
  - Copy-out errors joining `partial`.
  - The descriptor-class fallback.
  - The guard before the read-failure record.
  - The preflight judging image plots by a declared `png`. The rule is right; only the
    wording is wrong (MINOR-3).
