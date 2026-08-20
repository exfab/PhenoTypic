# Open questions raised during planning

The spec's own §10 records "none blocking". These are questions the **plan** surfaced by
resolving the spec against the actual code — either gaps the spec does not cover, or places
where the spec's statement and the code disagree. Each says what the plan currently assumes,
so implementation is not blocked on any of them; a different answer means editing the named
task.

**Status key:** `OPEN` — needs a decision. `ASSUMED` — plan proceeds on a stated
assumption. `RESOLVED` — decided, with the decision recorded.

---

## P1 — `image-label.colors` goes stale after Stage 2, breaking conformance mid-run

**Status:** RESOLVED — **option (2): emit the background entry only.** Re-graded from
data-integrity to conformance/interop first (see D12: nothing in PhenoTypic reads `colors`),
then decided. `build_image_label()` now takes **no arguments**, so it cannot depend on array
contents and cannot go stale. Applied in Phase 1 Task 1.4; the ~60 KB per-plate JSON the
spec's OQ9 budgeted for disappears with it.

Spec §2.3 requires `image-label.colors` to carry **one entry per unique label value**, and
§7 requires **every written store** to validate against `label.schema`. Spec §3.4 has
Stage 2 overwrite `labels/objmap` **in place**, without re-promoting.

Those two cannot both hold. After Stage 1 the objmap is zeros, so `colors` has exactly one
entry (background). After Stage 2 the array holds up to ~1536 distinct labels while the
group's `zarr.json` still says one colour. Any conformance check run against a mid-run store
— and the spec asks the GUI to render exactly that store — sees a `colors` list that does
not describe the array.

Three ways out, none free:

1. **Stage 2 also rewrites the label group's `zarr.json`.** Cheapest, but it makes Stage 2 a
   two-file write with no atomicity between them, in a step the spec deliberately defined as
   an intermediate.
2. **Relax `colors` to background-only** and drop the per-value requirement. Loses the
   viewer-friendly palette that motivated it, and needs a re-read of whether `label.schema`
   actually requires exhaustiveness or only well-formedness.
3. **Stage 2 promotes after all**, which contradicts locked decision #4.

**Plan currently assumes (1)**, implemented inside `write_objmap_in_place`
(Phase 3, Task 3.3). `test_stage2_drops_a_token_and_the_objmap_is_readable` and
`test_stage1_store_conforms` need a Stage-2 conformance sibling either way.

---

## P2 — `omero.window` bounds are wrong for `detect_mat`

**Status:** RESOLVED — **omit `omero` from `detect_mat` entirely.** NGFF makes `omero`
conditional and the whole-or-nothing rule is per group, so `rgb` and `gray` keep their
blocks and `detect_mat` simply has none. No wrong window can be emitted, and `detect_mat`
is a derived analysis layer no viewer has a meaningful default rendering for. Applied in
Phase 1 Task 1.4 (`build_omero` returns `{}` for `detect_mat`); supersedes spec §2.2.

Spec §2.2 fixes `max`/`end` at `2**bit_depth - 1` for every series. `detect_mat` is a
**float** detection matrix, typically in `[0, 1]` (the spec's own §10 notes it is float64
and 96 MB). A window of `[0, 65535]` over data in `[0, 1]` renders as a black image in every
viewer that honours `omero`.

Options: emit `{"min": 0, "max": 1, "start": 0, "end": 1}` for float series; or compute the
window from the actual array min/max; or omit `omero` from `detect_mat` entirely (NGFF makes
it conditional, and the whole-or-nothing rule is per-group).

**Plan currently assumes** the literal spec text (`2**bit_depth - 1` everywhere), which is
almost certainly wrong for `detect_mat`. Phase 1 Task 1.4's `build_omero` is the single
place to change. Flagged rather than silently "fixed" because it is a spec statement, not an
oversight in the plan.

---

## P3 — Changing `--pyramid-levels` between runs produces a mixed-geometry tree

**Status:** RESOLVED by **descoping the lever**. `--pyramid-levels` is not implemented; the
pyramid depth is `pyramid_level_count(h, w)`, a pure function of the level-0 shape. With no
user lever, two stores in one tree cannot disagree — so `valid_staged_store` needs no level
check, a resumed run cannot produce mixed geometry, and the tile-request crash this question
described is unreachable. `resolve_pyramid_levels` is removed from Phase 1; Phase 3 Task 3.7
is now `--durable-writes` only. A single-level store stays reachable internally via the
private `levels=` argument for builder node previews. The lever can land later as its own
change; spec §1.3 should record it as deferred.

`valid_staged_store` (§3.6) checks only **level-0** extents. A resumed run with a different
`--pyramid-levels` therefore leaves every already-written store at the old level count while
new ones use the new one. Nothing detects it, and the GUI's `select_pyramid_level` reads
`phenotypic.pyramid.levels` per store, so it will not crash — it will just serve tiles at
inconsistent resolutions.

Options: record the resolved level count in the run manifest and refuse a resume that
changes it; or add level count to `valid_staged_store` so a mismatched store reclassifies to
`stage1`; or accept mixed geometry and document it.

**Plan currently assumes** mixed geometry is accepted and undocumented, because the spec says
nothing. Phase 3 Task 3.4 is where a validity change would go; Phase 5's
`migrate_run_hdf_to_zarr` has the same question for `--njobs`-parallel conversions.

---

## P4 — Stage 2 must rewrite **every** objmap pyramid level, which the spec does not say

**Status:** ASSUMED (plan is stricter than the spec).

Spec §3.4 says Stage 2 "overwrites `labels/objmap` in place" without saying at which levels.
Rewriting only level 0 leaves levels 1..n holding Stage 1's zeros, which the GUI then serves
as a blank overlay for any zoomed-out view — silently wrong, never an error.

**Plan assumes** all levels are rewritten (`write_objmap_in_place`, Phase 1/3), and
`test_stage2_rewrites_every_pyramid_level_of_the_objmap` pins it. This should be folded back
into the spec's §3.4 rather than living only in the plan.

---

## P5 — `save_intermediate_layers` has **five** call sites, not three

**Status:** RESOLVED (plan uses the verified count).

Spec §3.1 says `save_intermediate_layers` "has three live callers in
`_image_pipeline_core.py` and two in tests". Verified against the code: that file has
**five** relevant calls — `save_intermediate_layers` at lines 1024, 1046, 1052, **and
`save2hdf5` at lines 1021 and 1042**. The two `save2hdf5` calls are the same builder-preview
path and must move together, or node previews write HDF into a zarr tree.

Phase 2 Task 2.4 covers all five. The spec's §3.1 sentence should be corrected.

---

## P6 — The builder DAG manifest's `"hdf"` key is a GUI-visible contract change

**Status:** ASSUMED.

`gui/builder/_preview_cache.py` writes a per-node manifest with a `"hdf"` key
(lines 208, 212, 217) that `_preview_cache.py:284-286` and `_preview_tiles.py:124` read
back. Renaming the per-node artifact from `base_00.h5` to `base_00.ome.zarr` changes an
on-disk contract the spec's §3.1 mention of `save_intermediate_zarr` does not discuss.

**Plan assumes** the key becomes `"store"` and `MANIFEST_VERSION` is bumped so a stale
manifest is rebuilt rather than misread (Phase 2, Task 2.4). Alternative: keep the key name
`"hdf"` to avoid the version bump — rejected, because a key named `hdf` holding a zarr path
is exactly the kind of lie that costs an afternoon later.

---

## P7 — Stage 3 deletes the Stage-2 signal only when `work_id is None`

**Status:** RESOLVED (plan preserves the code's behaviour, not the spec's sentence).

Spec §3.5 says Stage 3 "writes the completion marker and **deletes the Stage-2 token**".
In the code (`_cli_staged_workers.py:250-258`) both actions sit inside `if work_id is None:`;
on the work-id path the SLURM worker publishes and deletes instead
(`_cli_staged_slurm_worker.py:409`).

**Plan preserves the guard verbatim** (Phase 3, Task 3.3). Making the deletion unconditional
would double-delete against the SLURM worker and change resume classification — which is
precisely what Task 3.4's differential test exists to catch. The spec sentence should gain
the qualifier.

---

## P8 — `jsonschema` is not a declared dependency

**Status:** RESOLVED.

Spec §7 forbids a conformance check that skips on a missing dependency, and §6 rules out
`ome-zarr-models`. But `jsonschema` appears nowhere in `pyproject.toml` — it is available
today only transitively. A transitive dependency can vanish on any lock refresh, and the
check would then be unrunnable, which §7 says must fail rather than skip.

**Plan declares it** in the test dependency group and pins that with
`test_jsonschema_is_declared_not_transitive` (Phase 0, Task 0.1).

---

## P9 — `save_array2hdf5`'s "eight live call sites" are removed by this change

**Status:** ASSUMED.

Spec §5.4 keeps `save_array2hdf5` on the grounds that it "has eight live call sites". Those
sites are in `_image_io_handler._save_image2hdfgroup` and `save_intermediate_layers` — both
of which **this change deletes** (Phase 2 Task 2.4, Phase 6 Task 6.2). After Phase 6 the
count is zero.

**Plan keeps it anyway**, for a different and explicit reason: `tests/fixtures/legacy_hdf/_generate.py`
(Phase 5, Task 5.1) needs an HDF writer to rebuild the migration golden fixtures after the
production writer is gone. That reason is recorded in Phase 6 Task 6.1's keeper table. If the
fixtures are instead frozen as committed bytes with no regeneration path, `save_array2hdf5`
can go too — but then the fixtures can never be extended.

---

## P10 — Two GUI/SDK call sites are missing from the spec's affected-module table

**Status:** RESOLVED (plan covers them).

Spec §4.4 lists 24 files. Two real call sites are not among them:

- `sdk_/_io_constants.py:2063` — `BundleLayout.hdf_path`, the accessor
  `OutputRoot.hdf_path` delegates to. **Note the trap:** it checks `is_file()`. A copy-paste
  port to a store returns `None` for every image, silently disabling every full-res GUI
  read, with no error anywhere. Phase 2 Task 2.1 uses `is_dir()` and pins it by test.
- `gui/results_viewer/_output_root.py:1146-1152` — an `("hdf", hdf_path)` pair built for the
  output-consistency report. Phase 4 Task 4.1 ports it.

---

## P11 — Chunk and shard shapes for pyramid levels smaller than one chunk

**Status:** ASSUMED.

Spec §1.4 fixes chunks at `(1024, 1024)` and shards at `(4096, 4096)` but does not say what a
`257 × 2` level gets. Zarr rejects a chunk larger than the array in some code paths, and the
sharding codec requires exact divisibility regardless.

**Plan assumes** both are clamped to the level's own extent, with the shard then rounded down
to an exact multiple of the clamped chunk (Phase 1, Task 1.2). This means the small levels
are single-chunk, single-shard — which is what the file-count table in §1.4 already implies
(8 files per additional level, flat across plate sizes), so the assumption is consistent with
the committed validation script. Worth confirming the script's `data_files`/`metadata_files`
functions agree with the clamped policy rather than with an unclamped one.

---

## P12 — `--pyramid-levels` and `--durable-writes` are not wired into the CLI argument list

**Status:** RESOLVED — **Phase 3 Task 3.7** added.

§1.3 introduces `--pyramid-levels auto|N` and §3.7 requires `--durable-writes /
--no-durable-writes`, but neither appears in §5's interface section or anywhere else that
enumerates CLI flags, and `phenotypicCLI.py`'s option block is untouched by the spec. Both
would therefore have shipped unimplemented.

**Resolved** by adding Phase 3 Task 3.7, which creates both options, validates
`--pyramid-levels` through `resolve_pyramid_levels` so the CLI and writer cannot disagree,
makes `--durable-writes` genuinely tri-state (unset = auto-detect; a plain `is_flag` would
collapse that to "off" and silently lose the SLURM detection), rejects both on `recompile`
and `migrate`, and documents that builder node previews are always single-level and ignore
`--pyramid-levels`.

The spec should gain these two flags in its §5 interface enumeration.

---

# Round 1 — data-flow review findings (D1–D16)

Raised by an independent data-flow review that traced five flows (staged write, resume
state, metadata, GUI reads, migration) against the real code. **Every claim reproduced
below was independently re-verified in this worktree before being recorded here**; the
verification method is stated per item. Findings the review raised that I could not
reproduce are marked as such.

## D1 — Stage 3 is not idempotent under retry, and a retry can delete a colony

**Status:** RESOLVED — **option (1): keep the raw Stage-2 array outside the store.**
Stage 2 retains its raw detector output at
`.phenotypic/progress/stage2_raw/<ds>/<stem>.npy`, written *before* the token so a crash
between them just re-runs Stage 2; Stage 3 replays from it and consumes both. Restores
today's exact idempotency with no new NGFF surface. Applied in Phase 3 Tasks 3.2 and 3.3,
with `test_stage3_is_idempotent_under_retry` and a `staged_run_with_border_colony` fixture
that makes the test non-vacuous (without a border-touching colony `drop_frame_background`
returns early and a second pass is a harmless no-op).

The full analysis follows, retained because it is the reasoning behind the decision.

**Verified** by reading `abc_/_gpu_detector.py:242-249` and
`_core/_image_parts/accessors/_objmap_accessor.py:498-509`.

Today Stage 3's input is the **raw** detector output, preserved in the `.npy` sidecar
(`_cli_staged_workers.py:210`), which survives Stage 3's re-save. A Stage 3 killed and
re-run replays from the same raw input and produces an identical result, any number of
times.

The plan replaces that input with the store's own objmap (Phase 3 Task 3.3) and then
re-promotes the store over it — so the raw detector output is destroyed the moment Stage 3
first succeeds. The retry window is real: `save_image_store` lands at
`_cli_staged_workers.py:225`, but the completion marker is not written until `:251`, with
`save_overlay` (`:239`) and `PlotCoordinator.emit_image` (`:243`) in between. A timeout
anywhere in that span leaves store re-promoted, parquet written, token present, marker
absent — which `classify_staged_image` (`_cli_staged_resume.py:233`) reads as `"stage3"`.

On the second pass, `_write_object_output` runs again on already-refined labels:

```python
if self.output_kind == "instance":
    image.objmap[:] = result.astype(np.uint16)
    if self.drop_frame_background:
        image.objmap.drop_frame_background()
```

`drop_frame_background` zeroes the label owning the **plurality of border pixels**, after
`border = border[border > 0]` excludes the already-zeroed background. So on the second pass
the plurality falls to whichever **real colony** touches the frame most — and that colony is
deleted. Silently, no error, once per retry. `post_pipeline.apply` also runs twice, which is
harmless for a size filter and not for erosion, border refiners, or watershed.

This breaks the byte-identical-to-single-pass contract in `_cli/CLAUDE.md` on the first
retry, and the resume classifier cannot tell a first Stage 3 from a second.

**Options:**

1. **Keep the raw Stage-2 array outside the store**, under
   `.phenotypic/progress/stage2_raw/<ds>/<stem>.npy`, paired with the token. Restores
   today's semantics exactly, adds no NGFF surface, and is a small change. Costs the spec's
   claim that "the `.npy` sidecar format disappears" — but see D6, which refutes the *other*
   stated benefit of the in-store write, so that claim is worth re-examining anyway.
2. Add a second label image `labels/objmap_raw`, written by Stage 2 and dropped by Stage 3's
   promote. Same semantics, but more inodes and more conformance surface.
3. Have Stage 2 apply `_write_object_output` itself so Stage 3 skips it. Fixes the
   colony-deletion half but leaves `post_pipeline.apply` running twice.
4. Make Stage 3's tail atomic so the marker cannot lag the promote. Narrows the window
   without closing it.

**Recommendation: (1).** It is the only cheap option that fully restores the current
guarantee. **This is a decision for you** — it partially reverses a stated design goal.

## D2 — The per-image completion marker breaks on a store directory

**Status:** OPEN — **needs a new task. Currently uncosted in both spec and plan.**

**Verified** by reading `_cli/_cli_completion.py:29-34` and `:117-130`, and by
`grep -rn 'publish_image_success|valid_image_success|_cli_completion|SUCCESS_MARKER_VERSION'`
over the spec and the whole plan directory → **no matches**.

`publish_image_success` records `{"size": ..., "sha256": _sha256(resolved)}` per declared
artifact, and `_sha256` does `path.open("rb")` — **`IsADirectoryError` on a store**,
uncaught, so the publishing worker dies. `valid_image_success` mirrors it with
`not artifact.is_file()` → **False for every store**, so branch 1 of `classify_staged_image`
returns `"stage3"` for every already-finished image on the work-id path, forever.

Five sites declare an `"hdf"` artifact, all confirmed present:
`phenotypicCLI.py:400`, `_cli_staged_slurm_worker.py:332` and `:382`,
`_cli_process_single.py:640`, `_cli_execution_strategies.py:167`.

**It is also invisible to the Phase 3 Task 3.4 gate.** That test parameterizes
`(image_state, stage2_signal, parquet, stage3_marker)` — there is no *image-completion
marker* axis, so `valid_image_success` returns `False` in both worlds and branch 1 is never
exercised. The parity test passes while production breaks.

**Plan now assumes:** a store gets its own descriptor kind
(`{"path": ..., "kind": "store", "fingerprint": paths_fingerprint([store / "zarr.json"])}`)

> **Superseded by Task 3.8 as implemented (C13).** Both the key and the function
> changed: the descriptor uses **`"sha256"`**, not `"fingerprint"`, and
> **`file_fingerprint`**, not `paths_fingerprint` (FLOW-3). The difference is not
> cosmetic -- `paths_fingerprint` is **path-sensitive**, so a relocated output tree
> would fail validation and reprocess every image. C13 pinned that with
> `test_a_relocated_output_tree_still_validates`, and mutating back to
> `paths_fingerprint` kills three tests. Read Task 3.8 for the live contract.
with `valid_image_success` dispatching on `kind`, and `SUCCESS_MARKER_VERSION`
(`_cli_completion.py:26`, currently `1`) bumped to `2`. Added as **Phase 3 Task 3.8**, and
the Task 3.4 differential test gains a fifth artifact axis.

## D3 — `_assert_canonical_metadata` rejects real production metadata

**Status:** RESOLVED — check removed. **Verified by execution** in this worktree:

```text
'Metadata_Strain'    | member: Metadata_Strain | is_metadata_header: True
'Metadata_PlateNum'  | member: None            | is_metadata_header: True
'MyColumn'           | member: None            | is_metadata_header: False
```

`metadata_member_for_header` is a **semantic-ownership resolver**, not a format check: it
returns `None` for `Metadata_PlateNum`, a real column in this project's canonical Results
matrix. And a legitimately loaded image really can carry a bare public key — verified by
HDF round-trip, `public after: {..., 'Metadata_PlateNum': 3, 'MyColumn': 'x'}`, because
`_remap_legacy_metadata_key` (`_image_io_handler.py:100-106`) deliberately preserves unknown
bare names verbatim.

So the check as written aborts `save2zarr` for most production runs, and the review's
suggested replacement (`is_metadata_header`) still rejects `MyColumn`.

**Resolved by deleting the assertion.** The HDF writer has no equivalent check, so adding
one is a regression, not a hardening. Phase 1 Task 1.3 loses `_assert_canonical_metadata`
and its `test_non_canonical_metadata_headers_are_rejected` test; the docstring records why.

## D7 — `Metadata_ImageType` does not survive the read path

**Status:** OPEN — **a spec §7 mandated test will fail as the plan is written.**

**Verified by execution:**

```text
before ImageType: GridSection
after  ImageType: Image
```

Spec §2.1 requires `image_class` and `Metadata_ImageType` to stay distinct, and §7 requires
a round-trip test asserting both "preserved independently". Phase 2 Task 2.2 instructs
`_load_from_store` to "mirror `_load_v2_grouped`" — **and `_load_v2_grouped` loses it**. The
cause is `_image_io_handler.py:1071-1073`:

```python
for mapped, value in decoded.items():
    if mapped in target and target[mapped] is not None:
        continue
    target[mapped] = value
```

The constructor has already set `Metadata_ImageType`, so the stored value never lands.
Mirroring this inherits the bug, and `test_image_class_and_image_type_are_independent`
(Phase 2 Task 2.2) fails.

**Plan now assumes** `_load_from_store` restores the three metadata sections **verbatim**
rather than by that skip-if-present merge, and Phase 2 Task 2.2 says so explicitly. Note
this makes the store read path **more** correct than the HDF one — a deliberate divergence,
flagged rather than silently introduced. Whether the HDF loader should be fixed too is out
of scope here (it is retired in Phase 6 anyway).

## D8 — The mandated `v2_enh_gray` fixture cannot be read by the loader the plan mandates

**Status:** OPEN — **needs a budgeted change to a legacy reader.**

**Verified** by reading `_image_io_handler.py:1035-1036` (v2 loader: bare
`layers["detect_mat"]`, no fallback) against `:1100-1108` (v1-flat loader: has the
`enh_gray` fallback).

Phase 5 Task 5.1 requires a **v2-grouped** fixture carrying `enh_gray` ("mandatory, not
optional") *and* requires reusing the existing loaders ("Do not write a third HDF reader").
Those are incompatible: `_load_v2_grouped` raises `KeyError` on that fixture. Meanwhile
`valid_staged_hdf` (`_cli_staged_resume.py:81-83`) accepts `enh_gray` at
`schema_version >= 2`, so the code believes such files exist in the wild.

**Options:** add the fallback to `_load_v2_grouped` (a change to a legacy reader that must
land **before** Phase 6 retires it, and that Phase 5 does not currently budget); or make the
fixture v1-flat only, leaving the schema-2 `enh_gray` case unmigratable.

**Plan now assumes** the fallback is added, as a new step in Phase 5 Task 5.1.

## D9 — `--mode migrate`'s `metadata.csv` rewrite collides with `metadata_sha256`

**Status:** OPEN.

`deliverables/metadata.csv` is not inert provenance — its SHA-256 is load-bearing state
(`phenotypicCLI.py:276`, `:1338-1341` write `state.config["metadata_sha256"]`;
`_cli_completion.py:541-547` folds it into `finalization_input_digest`; `:391-399`
recomputes `expected_finalization` from it). Task 5.2 rewrites the file and says nothing
about the digest. Leave it → the aggregate publication marker stops validating and the next
run re-finalizes everything. Update it → the recorded digest no longer matches
`metadata.original.csv`, which is the provenance the task exists to preserve.

Separately, and answering the review's own question: `metadata.csv` **is** read after
migration. `_snapshot_metadata_csv` (`phenotypicCLI.py:241-282`) runs at the start of
`full`, `recompile`, and incremental startup; if a user passes `--metadata <original.csv>`
again after migrating, `destination.read_bytes() != payload` and the canonicalized file is
**overwritten with the raw original**, silently reverting the migration.

**Plan currently assumes** neither problem exists. Both need a step in Task 5.2.

## D4, D5 — Two GUI fingerprints go content-blind, and I mis-scoped one of them

**Status:** OPEN — Phase 4 Task 4.1 understates both.

- `_image_source_token` (`_output_root.py:1138-1178`) hashes
  `st_dev/st_ino/st_size/st_mtime_ns/st_ctime_ns` per source path. **None of those five
  moves** when a chunk inside a store is rewritten. It drives `bound_image_source_token`
  (`:649`) and `_capture_image_source_tokens` (`:405`, `:1093`) — i.e. whether the viewer's
  binding to an image's pixel source is still valid. My Task 4.1 calls it "a label the
  report renders". That is wrong; it is a staleness fingerprint and must key on
  `store / "zarr.json"`.
- `_processing_snapshot_paths` (`:886-889`) feeds `_cancellable_paths_fingerprint`, whose
  directory branch (`:832-834`) emits a constant byte and does not recurse. If the port
  yields store **directories**, `snapshot.processing_fingerprint` — and therefore
  `OutputRoot.source_fingerprint` (`:512`) — stops changing when per-image results change. I
  framed line 888 purely as a cost problem (400k stat calls); the correctness problem is
  larger. The port must enumerate each store's `zarr.json`.

**Also a spec correction:** §4.2's table says "Use `paths_fingerprint()`, which handles
directories". It handles them **by ignoring their contents** (`_io_constants.py:215-217`
emits a single sentinel byte and does not recurse). `paths_fingerprint([store])` is a
constant function of the path and would freeze the tile cache permanently. The plan already
keys on `store / "zarr.json"`, but the spec sentence must be corrected before anyone
implements from it.

## D6 — The GUI cannot see the Stage-2 objmap at all, and Task 4.3 contradicted itself

**Status:** RESOLVED — **accepted as correct behaviour, and the contradiction removed.**

With D1 decided (the raw array is retained outside the store), the in-store Stage-2 write is
purely an interop convenience, not a correctness dependency — so the GUI not seeing it costs
nothing. Root-keying the cache means it invalidates on **promotes**, which is what should
gate consumers: the completion marker, not the store's shape, and a torn mid-Stage-2 objmap
is exactly what a viewer must not be shown.

Phase 4 Task 4.3's tests now pin **both** directions —
`test_served_tile_changes_after_a_promote` and
`test_served_tile_is_unchanged_by_an_in_place_write` — so a later "fix" cannot re-introduce
per-chunk invalidation. Spec §3.5 should drop its claim that the in-store write buys "the
GUI can render a real objmap mid-run"; it does not.

With Task 4.3's fix in place: Stage 2 rewrites `objmap/*` levels; the root `zarr.json` is
untouched; `_ensure_store_layer_source_png` returns early because the cached PNG's mtime was
`os.utime`'d to that same unchanged root. So the GUI serves the **Stage-1 zeros objmap** for
the entire Stage-2 → Stage-3 window.

Nothing is corrupted — Stage 3's promote fixes it — but it refutes spec §3.5's claim that
the in-store write buys "the GUI can render a real objmap mid-run". That was one of only two
stated justifications for writing in place; D1 attaches to the other. **Together, D1 and D6
mean the in-store Stage-2 write currently buys nothing and costs idempotency.** That is the
core of the D1 decision above.

Within Phase 4 Task 4.3, `test_served_tile_changes_after_an_in_place_rewrite` asserts the
tile *does* change after an in-place write, while Step 3 implements a check guaranteeing it
does not. One of the two has to give.

## D12 — P1's stated resolution is not in the plan text

**Status:** RESOLVED as a bookkeeping error; P1 itself re-graded below.

P1 says the plan resolves it "inside `write_objmap_in_place` (Phase 3, Task 3.3)". The
actual code block there writes array levels only — no `zarr.json` rewrite, no
`build_image_label` call, which appears only at write time. P1 is **unresolved in the plan
text**, not resolved-as-stated.

The review also traced what consumes `colors`: **nothing in PhenoTypic**. The GUI colourises
via `skimage.color.label2rgb` (`gui/builder/_image_renderer.py:155-166`); neither
`load_layer_zarr` nor `_load_from_store` reads it. The only consumers are the conformance
gate and external viewers. **P1 is therefore re-graded from data-integrity to
conformance-and-interop** — still real, because third-party readability is a headline goal
of this design, but no longer the most severe item. With no internal consumer, option (2)
from P1 (background-only `colors`) costs least.

## D11, D13, D14, D15, D16 — smaller corrections, all accepted

- **D11** (`--mode process --layer objmap` leaves raw detector output published forever).
  Today the residue is Stage 1's zeros in a non-user-facing HDF; under the plan it is a
  first-class NGFF label image that napari and Vizarr will render. Task 3.5 must re-promote
  after the export or restore the zeros objmap.
- **D13** (my `rmtree(store)` in `clear_downstream_artifacts_for_stage1` rests on a
  misreading). **Verified:** that function deletes only the `.npy` sidecar and the `.json`
  marker (`_cli_staged_resume.py:314-319`) — it never unlinks an image artifact, so no
  `IsADirectoryError` is possible. Adding an `rmtree` **introduces** behaviour: at its two
  call sites it would open a window where the image is absent, whereas today the previous
  HDF survives until Stage 1's atomic replace. Reverting to "delete nothing extra".
- **D14** (the run-start `.part`/`.trash` sweep can delete a live writer's directory). The
  uuid identifies the *attempt*, not whether its process is alive, and the staged SLURM
  engine explicitly assumes stale workers can still be running — that is what
  `assert_active_epoch` exists for. Gate the sweep on age or on a lifecycle epoch recorded
  inside the `.part`.
- **D15** (`tiles.py:518` is not a live staleness site). `crop_hdf_rgb` opens with
  `del mtime_ns` and its docstring says the parameter is accepted for API compatibility
  only. Calling it one of "four traps" spends attention on a non-issue while D4 and D5 —
  the two sites that genuinely go content-blind — were absent from the list. Task 4.3's
  framing is corrected: the site still needs the zarr port, but not a staleness fix.
- **D16** (Phase 5's two recompile tests contradict each other, and one references an
  undefined `legacy_run_v2`). The intended distinction — legacy *format* vs legacy *headers*
  — is real but the fixtures do not encode it. Both tests need distinct fixtures.

## Dead code the change strands

`clear_stage2_sidecars` (`_cli_staged_orchestration.py:661-674`, called from
`phenotypicCLI.py:1590` on `--restart`) globs `results/*/objmap/*.npy` and becomes a
permanent no-op. Not a correctness hole — `clear_machine_state` on the same path wipes
`.phenotypic/`, where the new token lives — but Phase 6 must remove it.

## Corrections to my own plan text, found while verifying

- Every test snippet imported `load_synth_yeast_plate` from `phenotypic.util`. It lives in
  `phenotypic.data`. Fixed across four phase documents.
- Every test snippet used `image.metadata.public[...]`. The accessor exposes
  `by_module/get/items/keys/table/...`, not the three sections; the established test idiom
  is `image._metadata.public[...]` (`tests/unit/sdk_/test_metadata_io.py:824`). Fixed
  across five phase documents.
- `_metadata` has a **fourth** section, `private`, which the HDF writer does not persist.
  The plan's three-section model is right for storage, but the plan should say `private` is
  deliberately not stored rather than leaving it unmentioned.

## Data-flow conclusions that came back clean

Recorded because a clean trace is a result:

- **Hard-link / promote / `rmtree(trash)` does not lose data.** Link-count walk:
  link → 2; `os.replace(store, trash)` → still 2 (a rename moves a dirent, not an inode);
  `os.replace(part, store)` → still 2; `rmtree(trash)` unlinks a *name* → 1, data survives.
  A crash mid-promote leaves an orphan `.trash` holding the second link, which the sweep
  decrements. The copy fallback on `os.link` failure keeps this sound; a **symlink** fallback
  would break it, and the plan correctly does not use one.
- **Metadata collision handling already exists and is correct.**
  `_normalize_stored_metadata_items` (`_image_io_handler.py:154-189`) raises `ValueError`
  when two source keys collide on one target with different values, and coalesces when
  equal. Migration surfaces that as `report.failed` — the image is named and skipped, never
  silently merged.
- **Token consumability is complete.** The sidecar is deleted at five sites
  (`_cli_staged_workers.py:258`, `_cli_staged_strategy.py:246` and `:382`,
  `_cli_staged_slurm_worker.py:409`, `_cli_staged_resume.py:364`) and all five are covered by
  the plan's file lists.
- **`migrate_legacy_stage3_markers` still fires**, because reaching `"complete"` for a legacy
  tree goes through the branch requiring the token's *absence*, and legacy trees have no
  token. This is exactly why the token must not live in `ome.labels` — P7's reasoning holds.
- **Pyramid dtype and value integrity, and the `rgb` moveaxis round-trip**, are sound.

**One correction to P7:** it implies the local path relies on the `work_id is None` guard.
It does not — `_cli_staged_strategy.py:243-246` writes the marker and deletes the sidecar
**unconditionally**. Preserving the guard verbatim is still right, but P7 should name that
third site so a future reader does not "simplify" the guard away.

---

# Round 2 — plan review findings (B1–B11, F1–F12, G1–G7, P13–P22, S1–S8)

From an independent plan review that verified every `file:line` against the worktree, ran
the logic-validation script (exit 0), and reached the zarr 3.x docs. **Every claim acted on
below was independently re-verified here first**; the method is stated per item.

## Verified and fixed

| | Finding | Verification | Disposition |
|---|---|---|---|
| **B1** | `shard_shape_for` returned `(3072, 2048)` for a 4000×3000 level — failing three of its own tests and making spec §1.4's file counts wrong | Arithmetic, then a script re-deriving shard-file counts at every level of three plate sizes against `ngff_store_geometry.py` | **Fixed.** Clamping removed: shard is `(C, 4096, 4096)` unless an extent is below one chunk, where `chunk == shard == extent`. Now agrees with the script at every level. Closes **P11/P13** |
| **B2** | Conformance harness validated `attributes["ome"]`; all three schemas are rooted at `attributes` | Downloaded and parsed all three: `required: ["ome"]`, `description: "The zarr.json attributes key"` | **Fixed.** Validates `payload["attributes"]` |
| **B3** | `_version.schema` not vendored; its `$ref` is remote, and `Unresolvable` is not a `ValidationError` | Parsed all three — each has exactly one remote `$ref` to it; fetched it (280 bytes) | **Fixed.** Vendored as a fourth file, resolved through a `referencing.Registry` |
| **B4** | `promote_store`'s `finally` deleted the previous store when the second rename failed — data loss the HDF path never had | Read the plan's own code | **Fixed.** Rolls back `trash → final` and re-raises; `rmtree` only on success. Closes **P14** |
| **B5** | Check-then-act made a concurrent promote raise, so "duplicate execution is benign" was not restored | Traced both interleavings | **Fixed.** The whole `exists → move-aside → replace` sequence is inside one retry loop that re-evaluates each attempt. Closes **P15** |
| **B6** | A uuid says nothing about liveness; a run-start sweep would `rmtree` a sibling SLURM task's in-flight `.part` | Read `assert_active_epoch`'s existence as proof the engine assumes live stale workers | **Fixed.** Controller-only, plus a `SWEEP_MIN_AGE_SECONDS` age guard, plus a bounded non-recursive scan. Closes **P16** |
| **B7** | A Stage-3 re-promote with unchanged metadata yields a byte-identical root, so the LRU key never moves and the "regenerated" PNG comes from the old array | Read `tiles.py:290-292`'s cache key | **Fixed.** Token is bytes **plus** `st_mtime_ns`. The in-place half of B7 dissolved when **D6** was decided. Closes **P17** |
| **B8** | Two Phase 7 gates were red on day one against code the plan keeps | `grep` → one `startswith("Metadata_")` at `_metadata_migration.py:210`; `h5py.File` at `_image_io_handler.py:1172,1211` | **Fixed.** Allow lists corrected; the HDF gate now matches write modes only |
| **B9** | `read_measurements()["ObjectLabel"]` — the column is `Object_Label` | `str(OBJECT.LABEL)` → `Object_Label` | **Fixed.** Resolved through `phenotypic.schema.OBJECT` so a rename cannot silently `KeyError` the most load-bearing test in the plan |
| **B10** | `write_objmap_in_place` was defined in Phase 3 but imported by Phase 4, contradicting the parallel DAG | Read both phase docs | **Fixed.** Moved to Phase 1 Task 1.6 with its own two tests |
| **B11** | Phase 6's `load_hdf5` rename named two call sites; Phase 5 has a third | Read `phase-5-migrate.md:113` | **Fixed.** All three named, plus a post-rename grep |
| **F1** | Spec §3.6's "none of zarr's error types are `ValueError` subclasses" is **inverted** — `BaseZarrError` inherits directly from `ValueError` | zarr readthedocs error hierarchy | **Fixed.** Tuple reduced to `(OSError, KeyError, TypeError, ValueError)`; the tautological test dropped; the spec's claim corrected in place. Closes **S2** |
| **F3** | `resolve_worker_count` reads only `SLURM_CPUS_PER_TASK`, never `SLURM_JOB_ID` | `_cli_utils.py:65-72` | **Fixed.** "exactly as" replaced with what the function actually does |
| **F5** | `save_image_hdf` has **three** callers, not two | `grep` → `_cli_process_single.py:183` | **Fixed**, plus the by-name monkeypatch at `test_staged_gpu_local.py:742` |
| **F6** | `strict_writer` and `swmr_reader` have **zero** call sites; the keeper justification was borrowed | `grep` over `src/` and `tests/` | **Fixed.** Kept for a stated reason (symmetric public surface), not a false one |
| **F7** | "977 lines total" counts a file the same task modifies | `wc -l` → 345 + 534 = 879 | **Fixed** |
| **F8** | `$defs/image-label` has **no `required` list** — `colors` is optional and nothing requires exhaustiveness | Parsed `label.schema` | **Confirmed.** This retroactively justifies the **P1** decision on schema grounds, not just cost grounds |
| **F9** | `$defs/omero` requires only `channels`; the channel item has no `required` list and `color` has no pattern | Parsed `image.schema` | **Recorded.** Emitting the full block stays PhenoTypic policy; `window`-if-present *is* schema-enforced, so `test_partial_omero_is_rejected` keeps its teeth |
| **G6** | Durability logging and the sweep were wired into the staged strategy only | Read Task 3.5 | **Fixed.** Both move to shared run-setup so a plain `--mode full` gets them. Closes **P21** |
| **G7** | **32** existing test files reference the removed HDF surface; the plan named 8, and `tests/gui` is in `testpaths` | `grep -rl` over `tests/`; `pyproject.toml:200` | **Fixed.** Full inventory in the README, assigned per phase; Phase 4's commands and exit criteria now run `tests/gui`. Closes **P20** |

## Still open

| | Question | Why it is still open |
|---|---|---|
| **G1/P19** | `build_multiscales(resolution=...)` is normative in spec §2.2 but **no caller passes it**, so the projection ships dead — and the hard-coded `"micrometer"` is a fabrication (TIFF carries `ResolutionUnit`, usually inch or cm) | Either wire it (read the tags from `metadata.imported`, handle the unit and the missing-tag case) or delete the parameter. Shipping an untested branch is the one option to reject |
| **G2/P18** | Spec §2.4's OME-XML failure fallback ("the consecutive-integer form") is not what Task 2.2 does — it keeps named groups and drops `series`, which is *less* conformant than either form | Recommend making an XML build failure **fatal**: it is pure string formatting over already-validated data, so the fallback exists for a path that cannot realistically happen, and removing it also removes the `_ome_xml_modules` seam that exists only to test it |
| **G3/P22** | `long_path` is applied at 3 of ~8 filesystem entry points — not in `_write_group_json`, `promote_store`, `fsync_tree`, `read_root_attributes`, or `sweep_orphan_parts`, i.e. most paths that actually approach `MAX_PATH` | Apply uniformly behind a helper so a new site cannot forget, or state that only array I/O is long-path-safe |
| **G4** | Spec §3.2 step 1 ("`rmtree` any pre-existing `.part` for this stem") is dropped; the plan relies on a fresh uuid | The plan is right and the spec is wrong — a stem-wide `rmtree` is exactly the sibling-eating behaviour **B6** guards against — but the divergence should be recorded as a deviation in the spec |
| **G5** | Nothing asserts the `"."` chunk-key separator is uniform **store-wide**; only `array_create_kwargs`'s return value is unit-tested and Task 7.2 checks `gray/0` | Needs one test walking every level of every series of a written store |
| **D9** | `--mode migrate`'s `metadata.csv` rewrite vs `metadata_sha256` | Carried from round 1 |
| **D10** | `sdk_/_metadata_migration.py`'s ~2,500 lines of HDF-target machinery are uncosted | Carried from round 1 |

## Adopted simplifications

**S1** (delete the clamping) — done via B1. **S2** (reduce the exception tuple) — done via
F1. **S3** (errno-discriminating retry) — done; `_is_retryable` gates on Windows
`ERROR_SHARING_VIOLATION`/`LOCK_VIOLATION` and POSIX `ENOTEMPTY`/`ENOENT`/`EEXIST`, so a
genuine `ENOSPC` fails fast instead of burning 3.1 s per image. **S4** (collapse the two
durability functions) — done via `_resolve_durability`. **S6** (drop four tautological
gates) — done, with the reasons recorded so they are not re-added.

**Not yet adopted:** **S5** (a `store_writer` context manager that makes the write-order
invariant unforgeable — genuinely attractive, but it restructures three callers and is worth
its own pass), **S7** (hoist `ngff_.py`'s function-local imports), **S8** (deduplicate the
two identical rootless-store assertions in Tasks 1.6 and 7.1).
