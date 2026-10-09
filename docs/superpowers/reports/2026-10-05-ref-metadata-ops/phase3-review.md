# Phase 3 (GUI) review — ref-metadata-ops

**Scope.** `git diff c56d1e1ac..f61627e68 -- src/phenotypic/_gui tests`: Cluster F
(`9f78c30de`, builder: picker, preview context, fingerprint/revision, RefColumn
dropdowns) and Cluster G (`f61627e68`, run console: `rc-reference-metadata-required`
alert, Run gate, launch-seam refusal, FEATURES rows). HEAD at time of writing is
`29af4d3f2`, which adds screenshots only.

**Reviewer.** implementation-test-reviewer (opus), analysis only.

## Summary

The GUI half is sound on its main paths, and I verified that by running code, not
only by reading it.

- **Nested-scope preview works.** A `SubtractBlank` inside a nested `ImagePipeline`
  previews correctly against the picked table. Probe output:
  - root error and inner error are both `None`;
  - the inner image is named `t01` after its store round-trip;
  - `detect_mat` max is `0.39216`, the expected `100/255`.
- **The callback wiring matches the spec.** The probe dumped `app.callback_map`:
  - `set_reference_metadata` writes the state, inspector and status, from
    `input-reference-metadata.value`;
  - `update_run_disabled` takes `rc-reference-metadata-required.is_open` as an input;
  - the fan-in subscribes to `param-column-scalar`.
- **The Phase 3 tests pass:** 30 passed in 2.81s.
- **No path writes the table path into a pipeline.** `to_pipeline_dag` ignores it.
  The one other export, the raw `builder-state.json` download, is session state by
  design.

There are no high-severity defects. There are three medium findings:

- an unknown-class pipeline that leaves the run console's alert stale (verified);
- a fresh-state reset that drops the session's table while the picker still shows it;
- preview/CLI divergence when the table has more than one dataset.

There are also eight low findings. The largest test weakness is that every new
callback is tested through a helper or `__wrapped__`, so nothing pins the Dash wiring.
The probe shows the wiring is correct today.

**Verdict: ready after fixes** — M1 and M2 are small. M3 can be a hint rather than
a fix.

## Assessment of the two known open edges

### Edge 1: nested-scope preview image name and root

**Correct.** The chain is as follows.

- **The image name survives.**
  - `compute_scope` loads a nested scope's input with `load_image_from_store`
    (`_preview_cache.py:392`).
  - `_load_from_store` restores the `protected` metadata section verbatim
    (`_image_io_handler.py:1783-1793`).
  - `Image.name` reads `protected[IMAGE_NAME]` (`_image_handler.py:197`).
  - So the lookup key is still the source stem, and the probe printed `t01`.
- **The root is right.**
  - `image_path` is threaded unchanged through the recursion (`_preview_cache.py:344`).
  - `preview_reference_context(reference, image_path)` (`_preview_cache.py:399`)
    therefore roots blanks beside the original image, not beside the scope store.
- **The fingerprint inherits correctly.** Only the root fingerprint carries
  `reference_identity` (`:354`). Nested scopes inherit it through `parent_fp`.
- **The parent pass runs inside the context too.** The parent scope's apply also
  runs the container, and so its `SubtractBlank`, inside the context. The root
  manifest error is `None`.

Caveats, filed below:

- M3: no `dataset=` narrowing.
- L4: the `"<synthetic>"` sentinel roots at the current working directory.

### Edge 2: dropdown refresh lag

**Not reproducible for the pick action itself.** `set_reference_metadata` returns
the inspector from `_render_views(state)` in the same response
(`_callbacks.py:4086-4106`, `4738`). `_render_views` renders
`build_linear_side_loader`, which is exactly what `INSPECTOR_CONTENT` mounts
(`_callbacks.py:4032`). The probe confirms the output wiring. So the dropdowns appear
in the response to the pick, not on the next render.

The lag does exist in three adjacent cases:

- **(a) The table changes on disk.** For example, a column is added.
  - `_columns_for` is keyed on mtime, so the next render picks the change up.
  - Nothing triggers that render.
  - Re-entering the same path does not fire the callback either, because
    `Input(..., "value")` does not change.
  - The status line also keeps the old `N rows · M columns` text.
- **(b) The table is deleted.** The dropdown silently reverts to free text while the
  status line still reports the table as loaded.
- **(c) M2 below.**

**Minimal fix for (a) and (b):** add `Input(ids.INPUT_REFERENCE_METADATA, "n_submit")`
to `set_reference_metadata`. Pressing Enter in the field then re-validates and
re-renders even when the value is unchanged. `dbc.Input` exposes `n_submit`. The
callback body needs no change.

## Findings

### Medium

**M1. An unknown-class pipeline crashes the alert callback and leaves the previous
pipeline's alert and Run state in place.**

*Where.* `run_console/_callbacks.py:301` (`except (OSError, ValueError, TypeError)`)
and `_callbacks.py:1842`.

*Mechanism.* `ImagePipeline.from_json` documents `ImportError` and `AttributeError`.
`UnknownOperationClassError` subclasses `AttributeError`. The probe confirmed the
leak: `UNKNOWN raised: UnknownOperationClassError(AttributeError) ...`.

*Failure scenario.*
1. The user selects a `SubtractBlank` pipeline. The alert opens and Run is disabled.
2. The user then selects a pipeline with a custom op whose module is not in
   `PHENOTYPIC_PRELOAD_MODULES`.
3. `show_reference_metadata_requirement` raises a server-side callback error.
4. The alert keeps the old pipeline's message (*"This pipeline reads reference
   metadata (sb reads Metadata_BlankImage)…"*), and Run stays disabled.

The reverse also happens: going from an ordinary pipeline to an unknown one keeps
Run enabled. That direction is harmless, because the CLI reports the unknown class.

*Pre-existing.* `_staged_gpu_capability` (`:271`) has the same catch set.
`click_action` already toasts the `AttributeError` from it before the new check is
reached, so the launch seam is not newly broken.

*Fix.* In `reference_metadata_requirement`, widen the catch to
`(OSError, ValueError, TypeError, AttributeError, ImportError)`. The docstring
already says an unreadable pipeline "is not reported here: the CLI's pipeline
validation owns that message". Do the same in `_staged_gpu_capability`, after its
`UnstageableGpuDetectorError` clause.

**M2. `start_new_builder_state` drops the session's table.**

*Where.* `builder/_callbacks.py:4712-4730`; `new_state = BuilderState()` is at `:4720`.

*Mechanism.* Every other state-replacing path carries `reference_metadata_path`:
- `_state_replacement_payload` for Load JSON and prefabs;
- the stale-breadcrumb rebuild in `_render_views`.

This reset path does not. It also takes no `State(STORE_BUILDER_STATE)`.

*Failure scenario.*
1. The user picks `blank_map.csv`. The status reads "blank_map.csv · 12 rows · 3 columns".
2. The user hits "start new state" from the unsupported-state panel.
3. The input and status still show the table, but the state has `None`.
4. RefColumn params render as free text.
5. The preview fails with *"…none is active… Pick a table in the builder's Reference
   metadata field"*. That message contradicts the visible field.

*Fix.* Add `State(ids.STORE_BUILDER_STATE, "data")`, and set
`new_state.reference_metadata_path = _session_reference_path(state_data)`, mirroring
`click_json_entry` (`:6953`).

**M3. The preview does not narrow by dataset, so it diverges from the CLI on tables
with more than one dataset.**

*Where.* `builder/_reference_metadata.py:122-124`.

*Mechanism.*
- The CLI plans each dataset with `context.narrow(dataset=dataset.name, image_root=dataset.input_dir)`
  (`_cli_reference.py:167`).
- `Dataset.name` is the input directory name (`_cli_types.py:30`).
- The preview builds `ReferenceContext(path, image_root=root)` with no `dataset`.
- Without a dataset, `_key_columns()` keys on `Metadata_ImageName` alone.

*Failure scenario.*
1. A blank map covers `plateA/t01` → `t00` and `plateB/t01` → `t00b`, with a
   `Metadata_Dataset` column.
2. The preview of `images/plateA/t01.tif` raises `ReferenceLookupError(reason="ambiguous")`.
3. The same table runs cleanly from the CLI.

This is exactly Review Focus 4's case, and the preview gives the opposite verdict
to the run.

*Fix (minimal).* When the table has `Metadata_Dataset`, pass
`dataset=Path(image_path).parent.name` to `ReferenceContext`. That matches the CLI's
naming for the common layout, where the preview image sits in its dataset directory.
If that is judged too clever, append a hint to the ambiguous message instead, in
`reference_error_message`.

### Low

**L1. Table edits on disk are not observed, and the status line goes stale.** See
Edge 2 (a) and (b). Fix: add `n_submit` as an Input.

**L2. The preview cache and revision ignore the blank image's identity.**

*Where.* `_preview_cache.py:352-355`; `_callbacks.py:3617-3637`.

*Mechanism.* The table is content-hashed, but neither of these enters the
fingerprint or `_pipeline_revision`:
- the resolved blank file (`t00.tif`);
- the image-root listing.

*Failure scenario.* The user re-exports `t00.tif` with a corrected exposure, or adds
a second `t00.png` that makes the CLI refuse as ambiguous. The cached node preview
is then returned as fresh (`compute_scope` hits at `:363-366`), and "Preview
complete" stays current.

*Context.* This is consistent with existing semantics: `_source_identity` keys the
source image by path only (`:196-200`). Hence Low.

*Fix, if wanted.* Append `context.reference_image_digest(name)` for the blank of the
previewed image to the root `fingerprint_inputs`. Or document that the preview is
keyed by the table, not the blank.

A related point: `_file_sha256` is keyed on `(path, mtime_ns, size)`. A same-size
rewrite within one mtime tick hits the cache. That is negligible on GPFS at
nanosecond resolution.

**L3. The `_pipeline_reference_columns` cache is keyed on `(path, mtime_ns)` only.**

*Where.* `run_console/_callbacks.py:276`.

*Failure scenarios.* A pipeline file replaced under the same mtime gives a stale
gate in both directions. Ways this happens:
- `cp -p` or `rsync -t` from a file that shares the mtime;
- filesystems with coarse mtime resolution.

The two directions differ in cost:
- **Permissive.** The gate wrongly passes, and the CLI's `PF-REF-NO-TABLE` catches
  it.
- **Restrictive.** Run is disabled for a pipeline that no longer needs a table, with
  no recovery short of touching the file or restarting the server.

*Fix.* Key on `(path, st_mtime_ns, st_size)`. This matches `_file_sha256`, and the
spec's own reference-image LRU, which keys on `(resolved_path, st_mtime_ns, st_size, …)`.

**L4. The `"<synthetic>"` sentinel roots blank resolution at the current working
directory.**

*Where.* `_reference_metadata.py:122`.

*Mechanism.*
- `_load_preview_image` treats `SYNTHETIC_SENTINEL` (`"<synthetic>"`) as "synthetic plate".
- `preview_reference_context` instead computes `Path("<synthetic>").parent`, which is
  `Path(".")`.
- Blanks then resolve against the server's working directory.

Current UI paths store the real synthetic-plate path (`_callbacks.py:7243`), so the
sentinel only arrives from older sessions. A server started in an image directory
could then silently pick up a file there.

*Fix.* Treat `image_path in (None, "", SYNTHETIC_SENTINEL)` as "no root".

**L5. `reference_error_message` has two issues.**

*Where.* `_reference_metadata.py:150`.

1. **It ignores `__suppress_context__`.** It follows `__cause__ or __context__`. An
   implicit `__context__` that holds a `ReferenceContextError`, which was caught and
   handled before an unrelated failure, would replace the real error in the toast.
   Fix: follow `__context__` only when `not current.__suppress_context__`, which is
   the traceback module's rule.
2. **It strips the op that failed.** It drops the
   `[SubtractBlank] (step i/n, key=…)` prefix. With two reference ops, a
   `ReferenceLookupError` ("No row … for image 't01'") does not say which one.
   Fix: keep the outermost `RuntimeError`'s bracketed step prefix, if present, in
   front of the innermost message.

**L6. The free-text table path is not confined like the builder's other file
pickers.**

*Where.* `_layout.py:4141-4153`; `_reference_metadata.py:30-54`.

*Mechanism.* Load Image, Save and Load JSON all browse within `CFG_IMAGE_ROOT`. This
input instead accepts any server-readable path, relative paths included (resolved
against the server's working directory). It returns the path's header names, row
count, and polars parse-error text to the browser.

*Severity.* Low for a single-user tool. Worth a sentence in the docs for shared Open
OnDemand deployments.

*Fix, if wanted.* Resolve the path through `SandboxRoot(image_root)`, as
`_browse_seed_from_source` does (`:3603-3611`), or add a browse button over the
existing modal browser.

**L7. A bare-header column value is shown as stale although it would resolve.**

*Where.* `_param_forms.py:383-396`.

*Mechanism.* `ReferenceContext._resolve_column` accepts `BlankImage` for
`Metadata_BlankImage`. The dropdown only checks literal membership in `columns`,
which are normalized. A block saved with `blank_column="BlankImage"` therefore:
- renders as `previously: "BlankImage" (not in reference_metadata file)`;
- shows an empty select.

The run would succeed.

*Fix.* In the builder's `param_form` seeding (`_param_form.py:133-144`), map a
`reference_metadata` value through `ensure_metadata_prefix` when only the prefixed
spelling is among the columns.

**L8. `state_from_json` does not type-check `reference_metadata_path`.**

*Where.* `_state.py:1672`.

*Mechanism.* A non-string value raises `TypeError` in `reference_columns_provider`
(`Path(123)`). The provider catches only `(OSError, ReferenceTableError)`, so every
inspector render fails. `_session_reference_path` already validates the value, but
`state_from_json` does not. The only source is client-tampered store data.

*Fix.* `reference_metadata_path=v if isinstance(v := data.get(...), str) and v else None`.

## Observations (no change requested)

- **Mount-triggered write of the default column.**
  - Dash fires a callback whose input is inserted by *another* callback, even with
    `prevent_initial_call=True`, unless the output is inserted alongside it. The
    codebase relies on this at `_callbacks.py:4431-4437`.
  - So the select that `set_reference_metadata` mounts fires the fan-in once with
    its seeded default.
  - `_handle_param_edit` then writes `blank_column="Metadata_BlankImage"` explicitly
    into `block.params`.
  - Semantically this is a no-op: same value, and the pipeline JSON merely becomes
    explicit. The `None` guard at `:8168` correctly stops a stale value from being
    written as `None`.
  - Medium confidence, not exercised in a browser.
- **Over-gating on continuation.** A full-mode continuation whose output already has
  `deliverables/metadata.csv` would be accepted by the CLI. The console still
  requires a CSV. This follows spec §5.3 ("Run is disabled until it is set"), and the
  console only ever emits `--mode full` (`_state.py:557`), so the FEATURES row's
  `--mode` caveat is accurate.
- **The run-console path checks hold.**
  - `form_state["metadata_csv"]` holding whitespace or a non-existent path cannot
    reach the gate. `metadata_csv` is produced only by `resolve_metadata_csv`, which
    is sandbox- and fingerprint-checked (`_callbacks.py:734-740`, `:457`), and
    yields `None` for both. The gate then holds.
  - The interaction with the staged-GPU refusal is correct: two independent alerts,
    and either one disables Run.
- **Lazy imports are clean.** `_reference_metadata.py` imports only the stdlib at
  module level. Polars and `_core` load inside functions, and only when a table is
  set. The run console adds `functools`.

## Test gaps

The tests that exist are good: they would fail if the behaviour regressed. They pass
because the helpers are correct. The gaps are about wiring and about paths that
neither cluster covered.

**T1. Pin the Dash wiring.** Today's tests call `_reference_metadata_pick`,
`_handle_param_edit` and the run-console callbacks via `__wrapped__`, so all of these
mutations survive:
- deleting the `param-column-scalar` Input (`_callbacks.py:4221-4224`);
- deleting its membership in the dispatch set (`:4449`);
- deleting `Input(RC_REFERENCE_METADATA_REQUIRED, "is_open")` from
  `update_run_disabled` (the test passes `is_open` positionally).

Add to `tests/unit/gui/builder/test_reference_metadata.py`:

```python
def _spec(app, name):
    return next(
        (key, spec) for key, spec in app.callback_map.items()
        if getattr(spec.get("callback"), "__wrapped__", None) is not None
        and spec["callback"].__wrapped__.__name__ == name
    )

def test_picker_and_dropdown_are_wired(tmp_path):
    from phenotypic._gui.builder._app import create_app
    app = create_app(image_root=tmp_path)
    key, spec = _spec(app, "set_reference_metadata")
    assert spec["inputs"] == [{"id": "input-reference-metadata", "property": "value"}]
    assert "inspector-content.children" in key and "store-builder-state.data" in key
    _, fan_in = _spec(app, "fan_in_state_mutation")
    assert any(isinstance(i["id"], str) and "param-column-scalar" in i["id"]
               or isinstance(i["id"], dict) and i["id"].get("type") == "param-column-scalar"
               for i in fan_in["inputs"])
```

Add the run-console counterpart to `test_reference_requirement.py`: assert that
`{"id": "rc-reference-metadata-required", "property": "is_open"}` is in
`update_run_disabled`'s inputs. Also assert that the `"param-column-scalar"`
membership routes to `_handle_param_edit`, by driving
`fan_in_state_mutation.__wrapped__` with a monkeypatched `ctx` whose
`triggered_id`/`triggered` name a column-scalar id.

**T2. Add a positive control at the launch seam.** Use a reference pipeline with
metadata included and an acknowledged preflight. `click_action` should get past the
seam to the (monkeypatched) submitter. Without this, a regression where
`state.metadata_csv` is always `None` at the seam (for example, through
`recheck_metadata_selection`) would make reference pipelines unrunnable with every
test green. Put it in `tests/integration/gui/test_run_console_callbacks.py` beside
`test_run_action_refuses_a_reference_pipeline_without_a_metadata_table`, reusing
`_guard_action_controls` with a metadata payload and `"include"`.

**T3. Unknown-class pipeline (M1).** This test is red today.

```python
def test_unknown_class_pipeline_is_not_blocked(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(ImagePipeline(ops={"sb": SubtractBlank()}).to_json()
                    .replace("SubtractBlank", "NoSuchOp"), encoding="utf-8")
    assert reference_metadata_requirement(str(path), None) is None
```

**T4. Nested-scope preview (Edge 1).** Port
`/scratch/.../phase3_probe.py::probe_nested_scope` into
`tests/gui/builder/test_preview_nested_integration.py`, beside `_nested_state()`. It
builds a container with an inner `SubtractBlank`, picks a table, and asserts on
`compute_scope(..., [container.block_id], image, …)`:
- `error is None`;
- the inner store's image `.name == "t01"`;
- `detect_mat.max() == pytest.approx(100/255, abs=1e-6)`.

**T5. Assert values in `test_node_preview_runs_subtract_blank_against_the_picked_table`.**
It asserts only `error is None`. Load the `SubtractBlank` node's store and assert
`detect_mat` max ≈ `100/255` and min `0`. The tolerance should be one float32 ulp:
the inputs are exact uint8 values divided by 255. That proves the subtraction ran
against `t00`, not that something merely ran.

**T6. Fresh-state reset carries the table (M2).** Drive
`start_new_builder_state.__wrapped__([1], state_data)` after the fix. Assert that
`state_from_json(result[0]).reference_metadata_path == path`.

**T7. Load JSON and prefab carry the table through their callbacks.**
`test_loading_a_pipeline_keeps_the_picked_table` tests `_state_replacement_payload`
directly, so deleting `reference_metadata_path=_session_reference_path(state_data)`
at `:6995`/`:7073` survives. Drive `click_prefab_card.__wrapped__([1], state_data)`
with a monkeypatched `ctx.triggered_id`.

**T8. Revision follows table content.** The existing test proves that
`reference_identity` follows content, and that `_pipeline_revision` follows *set vs
unset*. Nothing proves `_pipeline_revision` changes when the same path's content
changes. Add a test that:
1. computes the revision;
2. appends a row;
3. calls `os.utime` to bump the mtime;
4. asserts the revision differs.

**T9. Cache key (L3).** Add a test that:
1. writes the reference pipeline;
2. computes the requirement;
3. overwrites the file with an ordinary pipeline, using
   `os.utime(path, ns=(old_atime, old_mtime))`;
4. asserts the requirement is `None`.

It is red until `st_size` joins the key. Pick pipelines whose JSON sizes differ.

## Verdict

**Ready after fixes.**

- **Before merge:** M1 (a one-line catch widening) and M2 (one `State` plus one
  assignment), with T1, T2 and T3.
- **Before end-of-change docs:** M3. Either the `dataset=` narrowing or a hint.
- **Optional:** L1–L8 and T4–T9. They are cheap, and T4/T5 lock in the nested-scope
  behaviour verified here.
