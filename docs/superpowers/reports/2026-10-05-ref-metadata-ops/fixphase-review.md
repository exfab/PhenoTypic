# Fix-phase review: `ref-metadata-ops`, pre-PR fixes

Scope: `git diff e75f97fdb..HEAD -- src tests scripts`, which covers commits `c167ae03b`, `fb19f1048`, `538daed42` and `4bafc49b1`.
Reviewer: fix-phase deep reviewer. Analysis only. No source or test file was edited.

Evidence used:
- I read every changed hunk, plus the code each hunk calls into: the loaders, the controller, the lifecycle ledger, the preflight walker and the GUI launcher.
- The lead ran one read-only probe for me on 2026-10-06. Its output is quoted verbatim where it is used (P1–P5).
- Every finding records where it came from:
  - **[branch]**: introduced by these commits.
  - **[unmasked]**: the defect existed before, and these commits made it reachable.
  - **[pre-existing]**: the defect existed before and these commits did not change it.

Each provenance label was checked against the hunks in `e75f97fdb..HEAD`.

## Summary

- **No high-severity defect.** The shipped behaviour matches the user's decisions: "narrow or refuse", "load as written + warn", and "refuse while live".
- **Pinning is complete.** All three staged cores, and the local objmap export, receive the run's single per-image pin on every production call path. No other caller of the staged cores exists in `src/`. A `None` digest under a reference manifest is refused at every stage. A legacy direct-test tuple (`_entry`) also yields `digest=None`, so it is refused rather than run unpinned.
- **A run cannot refuse itself.** Every scheduled job runs `_cli_sentinel`, the staged controller or a worker module, never `phenotypic_cli`. The GUI launches a fresh `python -m phenotypic` before anything is submitted.
- **Three medium findings:**
  - One mutant survivor judged equivalent is not equivalent, which leaves a test gap on F1 (M1).
  - The fail-closed "unknown" refusal gives the user no remedy (M2).
  - A stale-pinned image wastes Stage-2 GPU rounds on SLURM (M3).
- **Eight low findings.** Several are behaviour changes the probe confirmed.

---

## Medium

### M1. "Unpin Stage 3's measure on its own" is not an equivalent mutant [test gap on branch code]

`src/phenotypic/_cli/_cli_staged_workers.py:619-629` opens two contexts in turn: `worker_reference_context(pin)` around `replay_pipeline.apply`, then a second one around `replay_pipeline.measure`. Each context checks the pin once, on entry.

**Failure scenario under the mutant.** Suppose the measure context is unpinned (`pin=None`) and a re-plan lands between the apply context's exit and the measure context's entry:
- The apply has already passed its check.
- The measure then runs under the new manifest's narrowed context.
- The measurements are published under the old `work_id`.

That is exactly the F1 defect: the image is applied and measured under one plan, then certified under another. The suite does not kill this mutant because every test re-plans **before** the core is entered, so the pinned apply context refuses first.

**The reverse mutant is equivalent.** If apply is unpinned and measure is pinned, the measure context refuses before anything is published. That survivor can stay classified as equivalent.

**Fix.** Add this test (and keep the code as it is):

```python
def test_stage3_measure_refuses_a_replan_between_apply_and_measure(run_inputs, monkeypatch):
    from phenotypic._cli import _cli_staged_slurm_worker as worker
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    pipeline = _staged_pipeline(run_inputs, monkeypatch, "stage1")
    out = run_inputs[0] / "slurm_out"
    _, _, manifest = _slurm_setup(run_inputs, pipeline, out)
    worker.run_stage1_step(pipeline, out, "Image", manifest, 0, ".tiff")
    worker.run_stage2_shard(pipeline, out, "Image", manifest, 0, 1)
    real_apply = ImagePipeline.apply
    def apply_then_replan(self, image, *a, **k):
        result = real_apply(self, image, *a, **k)
        _replan(out, "plate1", "t01")          # after apply's context check
        return result
    monkeypatch.setattr(ImagePipeline, "apply", apply_then_replan)
    with pytest.raises(_cli_reference.ReferencePlanStaleError, match="t01"):
        worker.run_stage3_step(pipeline, out, "Image", manifest, 0, ".tiff")
    assert not image_record_path(out, "plate1", "t01").exists()
```

The monkeypatch must be installed only around Stage 3, as shown. If `ReplayDetector`'s pipeline subclasses `ImagePipeline` with its own `apply`, patch the class `build_replay_pipeline` returns instead.

### M2. The `unknown (...)` refusal gives no actionable remedy, and nothing overrides it [branch]

**Where.**
- `src/phenotypic/_cli/_cli_staged_orchestration.py:740-752` builds the entry.
- `src/phenotypic/phenotypicCLI.py:789-810` prints it.

**What the user sees.** A user on a host where `squeue` is missing or hung (a workstation mounting the output, or a login node during a slurmctld outage) receives:

```
Error: Cannot continue, restart, overwrite or recompile while SLURM jobs are active:
unknown (unresolved submissions <uuid>:chunk-1; the scheduler could not be queried).
Wait for them to finish or cancel them (scancel), then run the command again.
```

**Why that message fails the user.**
- It says jobs "are active", which is unknown.
- It says to `scancel`, but no job ID is given.
- Waiting does not help, because the scheduler will never answer from that host.
- No flag overrides the check.
- The same host also reports every ledgered `submitted` job as active: `scheduler_job_is_active` returns `None` on `FileNotFoundError`, at `_cli_staged_orchestration.py:672-673`. The message then lists job IDs that may be long finished, with nothing to show they are unverified. This last part is **[pre-existing]** for forward modes; this branch adds it for recompile.

**Timeout cost.** One query costs at most 30 s, which is acceptable. The underlying `active_ledger_job_ids` is not bounded that way:
- It walks **every generation** in the ledger.
- It issues one `squeue --jobs` per job, plus one `sacct` per job slurmctld has forgotten, each with a 30 s timeout.
- Under a hung controller, that costs N×30 s before the refusal appears.

The walk itself is **[pre-existing]**. It now also gates recompile (L5).

**Fix.** Split "live" from "unknown" in the return type, for example `(live_ids, unknown_reasons)`, and print unknown entries in their own sentence, e.g.:

> "The scheduler could not be queried from this host, so PhenoTypic cannot tell whether submission(s) X are running. Run the same command on a cluster node where `squeue` works. To find the job by hand: `squeue --format='%i|%k' | grep 'phenotypic:<gen>:<token>'`."

Optionally, also name the recovery for a lifecycle that is known to be dead. No such recovery is documented today, so a user's only escape is hand-editing `slurm_lifecycle.json`.

### M3. A stale-pinned image triggers futile Stage-2 rounds on SLURM [branch]

**What happens.** `run_stage2_shard` (`_cli_staged_slurm_worker.py:332-357`) swallows the per-image `ReferencePlanStaleError`, and no terminal record is written. Then:
1. `_classify_stage2` (`_cli_staged_controller.py:58-107`) sees a valid Stage-1 store, no Stage-2 signal and no terminal record, so it lists the image as **retryable**.
2. The controller submits another GPU round. The manifest pin cannot change within a run, so that round refuses again.
3. Only after two zero-progress rounds (`_cli_staged_controller.py:158`) does the chain move to Stage 3. There, `run_stage3_step` emits a missing prereq and exits 1 for that task.

**Impact.** The image is not lost: the next invocation re-plans it, its new `work_id` misses the stored store, and Stage 1 clears its downstream artifacts. The run does pay up to two extra GPU queue waits and model loads, plus a failed Stage-3 array task.

**How often.** Rarely in practice. With the live-run guard in place, a mid-run re-plan needs two submitters to race.

**Fix.** Make the controller classify a stale-pinned entry as not retryable for this run. In `_classify_stage2`, before `retryable.append(entry)`:

```python
from ._cli_reference import reference_digest_for
if reference_digest_for(output_dir, entry.dataset, entry.stem) != entry.reference_digest:
    terminal.append(entry)   # stale for this run; the next invocation re-plans it
    continue
```

Pin it with a test in which the image is re-plan-stale and `_classify_stage2` returns it under `terminal`, not under `retryable`.

---

## Low

### L1. Aliasing between a user's array and `Image._data.rgb` is now silently mutable [unmasked]

`Image(arr=a)` with `uint8`/`uint16` `a` stores `a` itself. This aliasing is **pre-existing**: the pass-through dates from before the branch.

**What changed.** Before `c167ae03b`, `ImageRGB._subject_arr` flipped the **base** array read-only. That base was the caller's own array, so after any whole-layer read (`vmax`, `normed`, `show`, `np.asarray(img.rgb)`), the caller's array refused writes. This side effect was itself a bug. Now the caller's array stays writable, and a write desynchronises `rgb` from `gray`/`detect_mat`.

Probe (P1):

```
alias True
rgb follows user write 200 gray 0.3921568691730499 vs 0.3921568691730499 user writeable True
```

The pixel's `rgb` became 200 while its `gray` is still the value derived from 100.

**Fix (out of scope to change semantics here).** Either:
- copy in `_handle_array_input` for the pass-through dtypes, which costs memory; or
- document in `_core/CLAUDE.md` and the `Image` docstring that the array is adopted by reference and the caller must not mutate it.

At minimum, add a test that pins whichever behaviour is chosen.

### L2. An integer or all-NaN stored gray in an RGB store loads silently [branch]

`_restore_stored_gray` (`_image_io_handler.py:1833`) casts to float32 and calls `_warn_if_stored_gray_outside_unit_range` (`:1859`). That helper returns early for any non-float dtype (`:1861`), and `nanmin`/`nanmax` skip NaN.

Before this branch, the public setter's assertion refused both cases. Now:

Probe (P2):

```
int gray warns 0 gray 200.0
nan gray warns ['All-NaN slice encountered', 'All-NaN slice encountered'] nan
```

So a uint8 gray of 200 loads as 200.0, with no PhenoTypic warning. An all-NaN gray loads with only numpy's RuntimeWarning.

How likely: an RGB image's gray has always been float from `rgb2gray`, so this is improbable data. The behaviour is still unpinned.

**Fix.** Do either of these, then add a test:
- Normalise an integer gray with `normalize_integer_matrix`, as the gray-only path does.
- Warn on any non-float dtype and on non-finite values (`if not np.isfinite(matrix).all(): warn`).

### L3. Integer RGB width depends on content, not on the source [branch, per user decision]

Probe (P4): two `int32` frames, both holding the pixel value 200, read 257× apart. The only difference is that one frame has a single brighter pixel.

```
dark 8 0.7843137383460999 bright 16 0.0030518043786287308
```

**Why it matters.** This is the agreed "narrow or refuse" rule, and single-channel input already behaved this way. The case to watch is a run that mixes frames in which `SubtractBlank` sees a dark blank narrowed to 8 bits and plates narrowed to 16 bits: the blank would then be on the wrong scale. Real 16-bit scans rarely stay at or below 255. No warning is emitted, and `test_other_integer_rgb_raises_no_unknown_dtype_warning` forbids one.

**Fix (documentation).** Name `--bit-depth`/`bit_depth=` as the way to pin the width for a dataset of non-`uint8`/`uint16` integer images:
- in the `_core/CLAUDE.md` gray bullet;
- in the CLI `--bit-depth` help.

### L4. "the declared 8-bit range" is printed when nothing was declared [branch, for RGB]

`bit_depth` is sticky once it has been inferred. `_as_unsigned_array` (`_image_data_manager.py:379`, `:396-404`) cannot tell an explicit value from an inferred one.

Probe (P5):

```
sticky refused: RGB int64 image has values in [1000, 1000], which do not fit the declared 8-bit range [0, 255]. ...
```

Here the 8 bits were inferred from the first `int64` frame by `Image(arr=...)`. `set_image` on a 16-bit-valued frame is then refused and called "declared". This branch extended the existing single-channel behaviour to RGB.

**Fix.** Track whether `bit_depth` was explicit (e.g. `self._bit_depth_explicit`). If it was not, either re-infer on `set_image` or say "the 8-bit range inferred from this image's first array".

### L5. The live-run guard costs sequential subprocess calls across all generations, and now gates recompile too [pre-existing cost; branch extends it to recompile]

`active_ledger_job_ids` (`_cli_staged_orchestration.py:687-704`) reads the ledger without a generation filter. For each unterminated job it runs `squeue --jobs <id>` and, once slurmctld has forgotten the job, `sacct -j <id>`, each with a 30 s timeout.

Ordinary chunk, dispatcher and finalizer jobs never get a `terminal` row. A long-lived output with a stuck `active: true` fence therefore pays about two subprocesses per historical job on every invocation. The cost is zero whenever the fence is closed, which covers normal completion and cancellation.

**Fix (follow-up).** Either:
- batch into one `squeue --jobs a,b,c` call and one `sacct -j a,b,c` call; or
- restrict to the current lifecycle generation, as `live_slurm_job_ids` already does for intents.

The restriction is sound: a new generation can only start after this guard passed for the old one.

### L6. The manifest key was added without a version bump [branch]

`StagedManifestEntry.reference_digest` adds a key while `_MANIFEST_VERSION` stays `3` (`_cli_staged_orchestration.py:51`).

Backward compatibility is correct: old entries load as `None`. Forward compatibility is not: an older reader does `StagedManifestEntry(**entry)` and fails with `TypeError: unexpected keyword 'reference_digest'` rather than the clean "Unsupported staged manifest version". This happens only if the install is downgraded during a live run.

**Fix.** Either bump the version to 4 (and accept 2–4), or have the loader drop unknown keys.

### L7. The tutorial capture still reuses cached output that lacks the metadata join [branch, partial fix]

`scripts/capture_gui_tutorial_screenshots.py:277` skips the CLI whenever `deliverables/master_measurements.parquet` exists. A cached output from before `538daed42`, produced without `--metadata`, is therefore reused, and the analysis panel stays empty unless `--force` is passed.

**Fix.** Also require `deliverables/metadata.csv`, the startup snapshot that `--metadata` writes:

```python
deliv = OUTPUT_DIR / "deliverables"
if (deliv / "master_measurements.parquet").exists() and (deliv / "metadata.csv").exists():
```

Check whether the regeneration job now running passes `--force` before trusting its screenshots.

### L8. Narrowing runs before the shape check, which mislabels unsupported shapes [branch]

`_as_unsigned_array` now runs for every integer array before `_guess_image_format` (`_image_data_manager.py:344-345`, `:382-385`). As a result:
- A 4-D or 2-channel integer stack gets an error labelled "Single-channel" or "2-channel" about value ranges, instead of the "unsupported number of dimensions/channels" error.
- An oversized array is copied by `astype` before it is refused.

**Fix.** Call `_guess_image_format(arr)` first. Alternatively, label by `arr.ndim` and bail out early for `ndim not in (2, 3)`.

---

## Answers to the lead's specific questions

**Narrowing and existing uint8/uint16 inputs.** These are unchanged.
- `_as_unsigned_array` returns `uint8`/`uint16` untouched at `:376-377`, including when an explicit `bit_depth` disagrees, as before.
- `_infer_bit_depth` sees the same dtype.
- `_retain_original` copies `rgb[:]`, as before.
- RGBA: `uint8`/`uint16` pass through to `rgba2rgb`, as before. Note that `rgba2rgb` yields float rgb, which is **[pre-existing]**.
- `GridImage` construction and crops (`_restore_crop_of` → `_restore_array`) see the same dtypes.
- Loaders pass stored `uint8`/`uint16` through.
- Byte identity is pinned by `test_uint8_and_uint16_rgb_are_byte_identical`.

**Scalar indexing and array keys.** No change for any key that yields an array:
- `_read_only` flags only `np.ndarray` results, exactly as before. That covers slices, boolean masks, fancy indexing (copies, flagged as before) and `...`-style keys that give 0-d arrays.
- `objects[i]` goes through `Image.__getitem__` → `_restore_crop_of`, which copies.
- `objmap`/`objmask` getters never set flags, so they are unaffected.

**Removing the read-only flag on the stored array.** Nothing in `src/` relied on `_data.rgb` being non-writeable:
- `grep writeable` finds only view-level flags.
- `ColorDenoise` and the wavelet correctors replace `_data.rgb` rather than writing into it.
- `ImageHandler.__setitem__` (`_image_handler.py:142`) writes into it in place, and it previously failed after a whole-layer read. That failure was a bug.

The only consequence is L1.

**Staged entry points.** The only callers of the three cores are `StagedGpuStrategy` and `_cli_staged_slurm_worker`, and both pass pins. The process-mode objmap export is local-only, because the staged SLURM path rejects process layers other than `None` (`_cli_execution_strategies.py:1362`), and it is pinned. `_terminal_output_exists` (`_cli_staged_strategy.py:122`) still computes a fresh `work_id`, but uses it consistently: it publishes only when the store's stamped `work_id` equals that fresh one. It is not a hole.

**Old-manifest compatibility.** Correct: `None` matches a run with no reference manifest, and is refused under one. It is pinned by `test_a_staged_manifest_written_before_the_digest_field_still_loads` and `test_an_old_shape_entry_is_refused_under_a_reference_plan`. See L6 for forward compatibility.

**The swallowing in `run_stage2_shard`.** The image is retried by the next invocation and is not lost. Within the run it costs futile rounds (M3).

**Can a run refuse itself?** No. See the Summary.

**`squeue` unavailable.** See M2. The 30 s timeout per query is acceptable. The unbounded per-job walk is L5.

**Precedence of recompile's usage errors.** The guard now sits:
- after the `--input`/`--dry-run` rejections, `_refuse_unmigrated_output` and the "output directory does not exist" UsageError;
- before `_snapshot_metadata_csv`.

The only precedence that changed: a snapshot failure (`ClickException`) now loses to the live-run refusal. That is the desired order, because the snapshot is a write and recompile previously made it over a live run.

**Preflight `_requirements_of`.** Correct for every requirement check, because a table-only node genuinely needs no grid, RGB or weights. One related gap is **[pre-existing]**: an analyzer cannot declare `modules`, so `PF-MISSING-MODULE` cannot see an analyzer's lazy import (e.g. a `ModelFitter` that needs an optional package). If any analyzer gains an optional dependency, consider a `preflight_requirements` hook on the analyzer ABCs.

**The `epoch=None` assertion.** It is vacuous on the SLURM path: `_record_terminal_scientific_failure` returns `False` whenever `epoch is None` (`_cli_staged_slurm_worker.py:86-91`). It matters little, for two reasons:
- The stale-error classification lives in the shared cores' `except (MemoryError, ReferencePlanStaleError): raise` clauses.
- The local parametrized test, whose recorder needs no epoch, kills a mutant that wraps the error.

The SLURM recorder's own `isinstance` gate is what goes unguarded by these tests (T2).

---

## Test gaps (exact tests to add)

- **T1.** The test in M1: re-plan between Stage 3's apply and measure.
- **T2.** Make the SLURM-worker refusal tests non-vacuous. Initialize a real epoch, e.g. `epoch = initialize_orchestration(out, ...)`, or whatever `test_staged_slurm_*` already uses to create an active epoch. Pass `epoch=epoch` to `run_stage1_step`, `run_stage2_shard` and `run_stage3_step` in `test_the_staged_slurm_workers_refuse_an_image_replanned_after_submission`, so that `_no_terminal_failures(out)` can actually fail.
- **T3.** `blocked` status. Parametrize `intent_only_output` over `"intent"` and `"blocked"`. A mutant reducing `{"intent", "blocked"}` to `{"intent"}` at `_cli_staged_orchestration.py:735` survives today.
- **T4.** Generation filter. Write an older generation `gen-old` with an unresolved `intent` row, then initialize lifecycle `gen-live` with a clean ledger. Patch `query_scheduler_comments` to `pytest.fail` and assert the forward `--dry-run` exits 0. Dropping `epoch=` from the `read_job_ledger` call at `:727-730` survives today.
- **T5.** Drop `raising=False` from every `query_scheduler_comments` monkeypatch (`test_cli_v2.py:1416, 1450, 1587, 1632, 1664, 1691`). The name is imported into `_cli_staged_orchestration` today. If that import ever moves, `raising=False` turns the `pytest.fail` patches into silent no-ops and lets a real `squeue` run.
- **T6.** The whole-layer RGB view stays read-only:
  ```python
  img = Image(arr=_rgb_plate())
  view = np.asarray(img.rgb)
  assert not view.flags.writeable and img._data.rgb.flags.writeable
  with pytest.raises(ValueError, match="read-only"):
      view[0, 0, 0] = 1
  ```
  A mutant returning `self._root_image._data.rgb` from `_subject_arr` survives `test_accessor_scalar_indexing.py` today.
- **T7.** Pin L2's chosen behaviour: an integer gray and an all-NaN gray passed to `_restore_stored_gray` must each warn or normalise.
- **T8.** A ledger with no lifecycle file (legacy staged tree) holding a resolved `intent → submitted` pair plus one unresolved `intent`, with the scheduler unavailable. Assert that the refusal says "unknown" and does not crash on `lifecycle is None` (`:729`).
- **T9.** For M3: `_classify_stage2` on an entry whose `reference_digest` no longer matches the manifest must return it in `terminal`.

---

## Verdict

**Ready after fixes.**
- **Before the PR:** M1 (the test), M2 (the refusal message) and T5. Each is small.
- **May be follow-ups:** M3, L5 and L6.
- **Documentation only:** L1, L3 and L7. They need a sentence each or a one-line guard.

None of these is a correctness defect in the shipped code paths.

**Counts:** High 0 · Medium 3 · Low 8 · Test gaps 9.
