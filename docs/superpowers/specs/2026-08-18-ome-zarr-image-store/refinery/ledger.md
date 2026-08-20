# Concern ledger — OME-Zarr per-image store

Append-only entries; statuses updated in place. Every `resolved` entry names
what changed and why — that IS the provenance lock. A reviewer challenging a
locked entry must report `CONFLICT with <ID>`, never a fresh concern.

## Carried forward from pre-refinery rounds

Two review rounds ran before the refinery loop was started. Their full text is
in `docs/superpowers/plans/2026-08-18-ome-zarr-image-store/OPEN-QUESTIONS.md`.
Recorded here so panel reviewers can see what is locked and what the user has
already ruled on.

### PRE-D1 [Critical] [settled-by-user (round 0: retain the raw array outside the store)]
- Raised: pre-refinery round 1, data-flow review
- Concern: Stage 3 read its input from the store's own objmap, which Stage 3
  then re-promotes over — destroying the raw detector output. On a retry
  (the window between `save_image_store` at `_cli_staged_workers.py:225` and
  the completion marker at `:251`), `_write_object_output` runs a second time
  and `drop_frame_background` (`_objmap_accessor.py:498-509`) zeroes whichever
  **real colony** touches the frame most.
- Resolution: user chose option (1). Stage 2 retains the raw array at
  `.phenotypic/progress/stage2_raw/<ds>/<stem>.npy`, written before the token;
  Stage 3 replays from it and consumes both. Plan: Phase 3 Tasks 3.2, 3.3.

### PRE-P1 [Minor] [settled-by-user (round 0: background-only colors)]
- Raised: pre-refinery round 1
- Concern: `image-label.colors` would be stale between Stage 2 and Stage 3.
- Resolution: `build_image_label()` takes **no arguments** and emits only the
  transparent background entry. Independently justified afterwards: the
  published `label.schema`'s `$defs/image-label` has **no `required` list**, so
  `colors` is optional and exhaustiveness was never an NGFF rule.

### PRE-P2 [Major] [settled-by-user (round 0: omit omero from detect_mat)]
- Raised: pre-refinery round 1
- Concern: spec §2.2 fixes `omero.window` at `2**bit_depth - 1` for every
  series; `detect_mat` is float in `[0,1]` and renders solid black.
- Resolution: `build_omero` returns `{}` for `detect_mat`. Supersedes §2.2.

### PRE-P3 [Minor] [settled-by-user (round 0: descope the lever)]
- Raised: pre-refinery round 1
- Concern: changing `--pyramid-levels` between runs yields mixed geometry.
- Resolution: **`--pyramid-levels` is not implemented.** Depth is
  `pyramid_level_count(h, w)`, a pure function of shape, so mixed geometry is
  unreachable. Supersedes spec §1.3.

### PRE-D6 [Minor] [resolved (round 0: accepted as correct, both directions pinned)]
- Concern: the GUI cannot observe Stage 2's in-place objmap write.
- Resolution: accepted — the completion marker gates consumers, and a torn
  mid-Stage-2 objmap must not be served. Phase 4 Task 4.3 pins both directions.
  Spec §3.5's claim that the in-store write buys mid-run GUI rendering is false
  and should be dropped (**pending spec amendment**).

### PRE-D2 [Critical] [resolved (round 0: Phase 3 Task 3.8 added)]
- Concern: `_cli_completion.py`'s `_sha256` (`:29-34`) opens its argument as a
  file and `valid_image_success` (`:117-130`) requires `is_file()` — both fail
  on a store directory, killing the publishing worker and making every finished
  image reclassify `"stage3"` forever. Named in neither spec nor plan.
- Resolution: new Phase 3 Task 3.8 — `kind`-tagged artifact descriptors,
  store descriptors fingerprint the root `zarr.json`, `SUCCESS_MARKER_VERSION`
  bumped to 2. Task 3.4's differential matrix gains a fifth artifact axis.

### PRE-D3 [Critical] [resolved (round 0: check removed)]
- Concern: `_assert_canonical_metadata` rejected `Metadata_PlateNum` and bare
  public keys — verified by execution — aborting `save2zarr` on most runs.
- Resolution: assertion deleted. The HDF writer has no equivalent; adding one
  is a regression, not a hardening.

### PRE-B1 [Critical] [resolved (round 0: clamping deleted)]
- Concern: `shard_shape_for` returned `(3072, 2048)` for a 4000×3000 level,
  failing three of its own tests and contradicting the committed validation
  script's file counts.
- Resolution: shard is `(C, 4096, 4096)` unless an extent is below one chunk,
  where `chunk == shard == extent`. Re-verified by a script comparing
  shard-file counts against `ngff_store_geometry.py` at every level of three
  plate sizes — agrees everywhere.

### PRE-B2 [Critical] [resolved (round 0: validate `attributes`)]
- Concern: the conformance harness validated `attributes["ome"]`, but all three
  schemas are rooted at the `attributes` object.
- Resolution: harness passes `payload["attributes"]`.

### PRE-B3 [Critical] [resolved (round 0: vendor `_version.schema` + registry)]
- Concern: all three schemas carry a remote `$ref`; `jsonschema` raises
  `Unresolvable`, which is not a `ValidationError`, so the suite errors.
- Resolution: fourth vendored file + a `referencing.Registry`.

### PRE-B4 [Critical] [resolved (round 0: roll back on failure)]
- Concern: `promote_store`'s `finally` deleted the previous store when the
  second rename failed — a data-loss mode the HDF path never had.
- Resolution: roll `trash → final` back and re-raise; `rmtree` only on success.

### PRE-B5 [Critical] [resolved (round 0: retry the whole sequence)]
- Concern: check-then-act made a concurrent promote raise, so "duplicate
  execution is benign" was not restored.
- Resolution: `exists → move-aside → replace` inside one retry loop that
  re-evaluates each attempt.

### PRE-B6 [Critical] [resolved (round 0: controller-only + age guard)]
- Concern: a uuid gives no liveness signal; a run-start sweep would `rmtree` a
  sibling SLURM task's in-flight `.part`.
- Resolution: controller-only before submission, `SWEEP_MIN_AGE_SECONDS`
  backstop, bounded non-recursive scan.

### PRE-B7 [Major] [resolved (round 0: mtime folded into the token)]
- Concern: a Stage-3 re-promote with unchanged metadata is byte-identical, so
  the `@lru_cache` key never moves and the regenerated PNG uses the old array.
- Resolution: token = fingerprint **+** `st_mtime_ns`.

### PRE-B8..B11, F1..F12, G6, G7 [various] [resolved (round 0)]
- B8 gate allow-lists; B9 `Object_Label` not `ObjectLabel`; B10
  `write_objmap_in_place` moved to Phase 1; B11 third `load_hdf5` call site;
  F1 the inverted zarr-error claim (`BaseZarrError ⊂ ValueError`); F3
  `resolve_worker_count` reads only `SLURM_CPUS_PER_TASK`; F5 third
  `save_image_hdf` caller; F6 keeper justification; F7 879 not 977 lines; G6
  durability log + sweep on every path; G7 the 32-file test inventory.

## Open — carried into round 1 of the refinery

### PRE-G1 [Major] [open]
- Concern: `build_multiscales(resolution=...)` is normative in spec §2.2 but no
  caller passes it, so the projection ships dead; the hard-coded
  `"micrometer"` is a fabrication (TIFF carries `ResolutionUnit`).
- Resolution: — (wire it, or delete the parameter)

### PRE-G2 [Major] [open]
- Concern: spec §2.4's OME-XML failure fallback ("the consecutive-integer
  form") is not what the plan does — it keeps named groups and drops `series`,
  which is *less* conformant than either form.
- Resolution: — (recommendation on the table: make the failure fatal)

### PRE-G3 [Minor] [open]
- Concern: `long_path` applied at 3 of ~8 filesystem entry points.

### PRE-G5 [Minor] [open]
- Concern: nothing asserts the `"."` chunk-key separator is uniform store-wide.

### PRE-D9 [Major] [open]
- Concern: `--mode migrate` rewrites `deliverables/metadata.csv`, whose sha256
  is load-bearing state (`phenotypicCLI.py:276`, `_cli_completion.py:541-547`,
  `:391-399`). Leaving the digest stale re-finalizes everything; updating it
  orphans the provenance copy. Also `_snapshot_metadata_csv` (`:241-282`) can
  silently revert the migration.

### PRE-D10 [Major] [open]
- Concern: `sdk_/_metadata_migration.py`'s ~2,500 lines of HDF-target machinery
  appear in no phase's file list, and
  `refresh_success_markers_after_metadata_migration` (`_cli_completion.py:136-155`)
  needs a store equivalent.

### PRE-S5 [Advisory] [open]
- Concern: a `store_writer(final)` context manager would make the write-order
  invariant (arrays → `OME/zarr.json` → root last → promote) unforgeable
  instead of prose-enforced across three callers.

### PRE-S7 [Advisory] [open]
- Concern: `ngff_.py` uses function-local imports throughout.

### PRE-S8 [Advisory] [open]
- Concern: Tasks 1.6 and 7.1 duplicate the same rootless-store assertion.

## Round 1

### SIMP-1 [Major] [open]
- Raised: round 1, simplicity-reviewer
- Concern: Phase 5 Task 5.5 (`migrate_store_headers`) has **no production caller** — verified
  by grep over all nine plan documents: only its definition (`phase-5-migrate.md:593`) and its
  own five tests. Task 5.3's `--mode migrate` driver never invokes it. It is also unreachable
  by construction: a `.ome.zarr` store can only exist because this version wrote it, at
  `metadata_schema_version = 2`, and Task 5.1 states a converted store is "canonical by
  construction". For an unreachable path it buys the hard-link promote, the `os.link` copy
  fallback, a marker-refresh bridge, a conformance re-check, and the scoping of
  `_metadata_migration.py`.
- Resolution: — **HOLD: overlaps the migration specialist's charter; do not apply until MIG
  reports, to avoid an apply-then-revert.**
- Note: the reviewer's claim that cutting 5.5 "retires PRE-D10 entirely" is partly overstated.
  D10 also attaches to Task 5.1's "via the existing metadata-migration helpers". Cutting 5.5
  does remove the need for the *target* machinery (`TargetKind "hdf"`, rollback receipts);
  Task 5.1 still needs the pure header-mapping function.

### SIMP-2 [Major] [open]
- Raised: round 1, simplicity-reviewer
- Aliases: PRE-S8
- Concern: Phase 7 Task 7.1 duplicates four tests already in Phase 1 Tasks 1.5/1.6, one
  strictly weaker (2 uuid samples vs 64). Separately, Task 7.1 is the *only* place the commit
  protocol is exercised through a real `save2zarr` rather than `_fake_store`, so Phases 3-6
  build on it unverified until the final phase.
- Resolution: —

### SIMP-3 [Major] [open]
- Raised: round 1, simplicity-reviewer
- Concern: `test_interrupt_before_the_root_reads_as_absent` (Task 7.1) cannot fail. It
  monkeypatches `promote_store` to raise, but every byte goes to the `.part` sibling and only
  `promote_store` ever creates `final` — so `not final.exists()` holds under any write order,
  including the reversed one. **Confirmed by reading.** Same class as the four gates dropped
  last round.
- Resolution: —

### SIMP-4 [Minor] [open]
- Concern: Task 7.3's `test_file_fingerprint_is_never_called_on_a_store` greps
  `file_fingerprint\(\s*store` — fires only if the argument is literally named `store`. The
  real site is `file_fingerprint(h5)`. A name-coincidence check. **Confirmed by reading.**
- Resolution: —

### SIMP-5 [Minor] [open]
- Concern: two assertions pin nothing — Task 1.1's
  `test_level_count_is_a_pure_function_of_shape` (absence of a symbol nobody writes, plus a
  stateless function being deterministic) and Task 3.8's `test_marker_version_is_bumped` (a
  constant comparison, green the moment the constant is edited).
- Resolution: —

### SIMP-6 [Minor] [open]
- Concern: Task 3.1's `save_image_store` hand-rolls `.part` cleanup
  (`phase-3-cli-staged.md:188`), re-encoding the naming convention outside `ngff_.py`, and
  imports `new_part_path` at `:175` without using it. **Both confirmed by grep.**
- Resolution: —

### SIMP-7 [Advisory] [open]
- Aliases: PRE-S5
- Concern: verdict on PRE-S5 — **do not adopt.** Task 2.4 consolidates `save2zarr` and
  `save_intermediate_zarr` into one `_save_store`, and Task 3.1 delegates to `save2zarr`, so
  there is ONE writer of the ordering, not three. A `store_writer` CM would be an abstraction
  with a single call site.
- Resolution: —

### SIMP-8 [Advisory] [open]
- Aliases: PRE-S7
- Concern: verdict on PRE-S7 — **adopt**, with a `zarr` carve-out. `os` x5, `Path` x7,
  `shutil` x2, `time` x2 plus others are function-local in `ngff_.py`; none heavy or cyclic,
  and the deferral forces string-quoted annotations onto public signatures. Task 1.3's own
  Step 3 already says to hoist `json`/`Path`, so the plan contradicts itself within one phase.
- Resolution: —

### SIMP-9 [Advisory] [open]
- Aliases: PRE-S8
- Concern: verdict on PRE-S8 — adopt, subsumed by SIMP-2.
- Resolution: —

### SIMP-10 [Minor] [open] [spec-change] [needs-user-input]
- Raised: round 1, simplicity-reviewer
- Concern: the spec still carries four normative paragraphs the user's rulings overturned —
  §1.3 ("Tunable." `--pyramid-levels`), §2.2 (`omero.window` for every series), §2.3 (`colors`
  MUST be exhaustive, plus a 60 KB cost paragraph), §3.5 (the in-store write buys mid-run GUI
  rendering). Spec §4.2's "paths_fingerprint ... handles directories" row belongs in the same
  pass. Task 7.4 Step 5 marks the spec Implemented while these still read as requirements.
- Resolution: — **GATED TO USER.** Batch with any other `spec-change` items from this round.

### SIMP-11 [Minor] [open]
- Concern: Task 4.3's `test_paths_fingerprint_handles_a_store_directory` passes a *file*
  (`store / "zarr.json"`), so its name asserts the exact claim the task's own constraint block
  refutes, and the commit body repeats the falsehood verbatim.
- Resolution: —

### MIG-1 [Critical] [open]
- Raised: round 1, migration specialist
- Concern: `--mode migrate` never re-publishes per-image completion markers. **Verified in
  source:** `valid_image_success` rejects on strict version equality
  (`_cli_completion.py`, `marker.get("version") != SUCCESS_MARKER_VERSION`); the refresh
  bridge `continue`s on a version/work_id/dataset/stem mismatch (~:236) and **hard-`raise`s**
  `RuntimeError("Success marker artifact is missing")` on a non-file artifact (~:261). Task
  3.8 cites the docstring lines 136-155, not this failing logic. Consequence: after migration
  every finished image in every legacy tree is unknown-to-complete, and Task 5.2's own
  `test_migration_keeps_the_published_aggregate_valid` **cannot pass** — `source_set_digest`
  is computed from `valid_image_success`, False for every image.
- Resolution: —

### MIG-2 [Major] [open]
- Concern: the migration read path inherits `_load_v2_grouped`'s `Metadata_ImageType` drop.
  PRE-D7 fixed only the store loader on the reasoning "the HDF loader is retired in Phase 6" —
  but Phase 6 **keeps** `_load_v2_grouped` as the migration reader (`phase-6:176-178`), and
  Task 5.1 routes every legacy v2 file through it. Worse, Task 5.1's
  `test_converted_equals_a_freshly_written_store` compares `migrate_hdf_to_zarr(...)` against
  `Image.load_hdf5(...).save2zarr(...)` — **both sides through the same defective loader**, so
  the tautology hides the loss.
- Resolution: —

### MIG-3 [Major] [open]
- Concern: the three golden fixtures do not prove the migration. No `GridImage` fixture (grid
  state survival untested); none carries a non-default `Metadata_ImageType` (which is why
  MIG-2 goes undetected); none carries a `work_id` (MIG-1's path). And the generator must be
  hand-rolled on `HDF.save_array2hdf5`, so the "golden" fixture is an approximation of what
  `_save_image2hdfgroup` actually wrote, unverifiable after Phase 6.
- Resolution: —

### MIG-4 [Major] [open] [spec-change]
- Concern: `metadata_schema_version` is written unconditionally over metadata the same
  function documents as "verbatim and unvalidated", so the marker asserts what the writer does
  not enforce — inverting the HDF contract, where the attr is written only after a successful
  rewrite (`_metadata_migration.py:1401`) and its absence marks a target migratable
  (`:603-606`). Separately `store_schema_version` is checked for **presence only**, so
  `load_zarr` would read a future v4 store under v3 semantics silently.
- Resolution: —

### MIG-5 [Major] [open]
- Concern: Phase 5 declares "Runs in parallel with Phases 3 and 4", but Task 5.2's
  aggregate-validity test depends on Phase 3 Task 3.8's marker semantics. An agent executing
  Phase 5 per the README finds those symbols absent.
- Resolution: —

### MIG-6 [Major] [resolved-by-cut — see SIMP-1]
- Concern: header-only store migration drops the rollback/receipt machinery the HDF path has,
  with no `.h5` to fall back to for a natively written store.
- Resolution: dissolves entirely when Task 5.5 is cut.

### MIG-7 [Major] [open — part (a) dissolves with the 5.5 cut]
- Concern: (b) `_metadata_migration.py`'s targets are `csv|parquet|json|hdf|frame` and include
  per-dataset `measurements/*.parquet`, the root pipeline json, and deliverables (`:44`,
  `:780-830`). Task 5.4 stops recompile migrating anything; Phase 5 replaces it with per-image
  conversion + `metadata.csv` **only**. Every other legacy-header target loses its migration
  path — a regression against flat-metadata decision #1 that Task 6.4's supersession note does
  not acknowledge.
- Resolution: —

### MIG-8 [Major] [open] [spec-change]
- Concern: no defined behaviour for a **partially** migrated tree. The guard is only "output
  contains **only** `.h5` fails with a pointer", tested via `--mode recompile` alone. A
  half-migrated tree — the expected state after any interruption, since migration is
  explicitly resumable — passes the guard, so `--mode full` silently reprocesses every
  unconverted image. And after Phase 4 the GUI resolves `store_path → None` for unconverted
  images and renders silently empty, with no pointer anywhere.
- Resolution: —

### MIG-9 [Minor] [open]
- Concern: `keep_source=False` is unreachable from the product — both migrate functions take
  it and Task 5.1 tests both branches, but Task 5.3's CLI has no source-deletion flag. Every
  migration permanently doubles the tree's footprint with no supported reclaim path.
- Resolution: —

### MIG-10 [Minor] [open]
- Concern: Task 3.8's backward-compat provision is unreachable — `kind` defaults to `"file"`
  "so an older marker still parses", but the version gate rejects before any descriptor is
  read, and Task 3.8 bumps that constant.
- Resolution: —

### MIG-11 [Advisory] [open]
- Concern: Phase 7 never runs `--mode migrate` on a realistic tree and then `--mode full`
  against the result. That missing test is the one that would have caught MIG-1.
- Resolution: —

### MIG-TASK55 [—] [decision recorded]
- The migration specialist **agrees Task 5.5 is dead**, but corrects the reason: not
  "store_schema_version proves the headers", which is a hard-coded constant, but that legacy
  per-topic headers are canonicalized **on ingest, in memory** — `ensure_metadata_prefix`
  (`_metadata_helpers.py:292-305`) via `_remap_legacy_metadata_key`
  (`_image_io_handler.py:92-106`). An in-memory `Image` cannot hold a known legacy header, so
  `save2zarr` cannot write one and `migrate_store_headers` has no reachable input.
- Ripples to apply with the cut: re-base or delete Task 5.4's `legacy_headers_run` fixture and
  its three tests; MIG-6 dissolves; MIG-7a dissolves (7b stands); check nothing in Phases 1/3
  depends on the hard-link promote primitive.
- **Condition on the cut:** with 5.5 gone the product has *zero* header-migration path for
  stores, safe only while ingest normalization holds — so pin it with a test asserting a store
  written from an `Image` constructed with `MetadataPlate_*` headers comes out canonical.

### GEN-1 [Major] [open]
- Concern: Phase 0 says add `jsonschema` to the "**test** dependency group". **Verified: there
  is no `test` group** — only `dev`, `test-qt`, `docs` — and every CI lane runs
  `uv sync --group dev --group test-qt --all-extras` (`run-pytest.yml:147`). A new `test`
  group leaves jsonschema uninstalled in CI, so every conformance test fails there while
  passing locally, and `test_jsonschema_is_declared_not_transitive` passes either way.
- Resolution: —

### GEN-2 [Major] [open]
- Concern: **self-inflicted by the PRE-B6 fix.** `sweep_orphan_parts` now skips anything
  younger than 6 h, but its own test creates the dirs microseconds earlier and asserts
  `removed == 2`; Phase 7's equivalent asserts `== 1`. Both return 0. The Interfaces block also
  omits the new `min_age_seconds` keyword.
- Resolution: —

### GEN-3 [Major] [open]
- Concern: **self-inflicted by the S3 fix.** I replaced `_replace_with_retry` with
  `_is_retryable` + an inlined loop, but Phase 7 Task 7.2 still calls
  `ngff_._replace_with_retry` at `:267` and `:277`. **Verified: the function no longer exists
  in Phase 1.** Its tests also raise `OSError(32, …)`, which on POSIX is EPIPE with no
  `winerror`, so `_is_retryable` returns False and the "retries" test fails on the Linux lane.
- Resolution: —

### GEN-4 [Major] [open]
- Aliases: SIMP-3
- Concern: Task 7.1 Step 3's mutation proof cannot work — neither selected test is
  order-dependent, and the `-k "interrupt or rootless"` expression does not even select
  `test_a_part_without_a_root_never_validates`. Underneath: because a `.part` is never at the
  published path, "root last" is load-bearing only for §3.7 flush ordering, which nothing tests.
- Resolution: —

### GEN-5 [Major] [open]
- Concern: Phase 7's napari interop gate is unsatisfiable. **Verified: neither `ome-zarr` nor
  `napari-ome-zarr` is in `uv.lock`**, `napari-ome-zarr` depends on `ome-zarr`, and Global
  Constraints + `test_ome_zarr_packages_are_not_adopted_anywhere` ban it from every group. The
  step also calls `napari.run()`, which blocks on a display the HPCC context lacks.
- Resolution: —

### GEN-6 [Major] [open]
- Concern: `image.rgb.clear()` is used at three sites and **does not exist** — verified by
  execution (`hasattr(img.rgb, "clear")` is False). It is the plan's only construction of an
  rgb-less image, which is load-bearing for §1.1. Verified fix: `Image(<2-D ndarray>)` yields
  `rgb.isempty() is True`.
- Resolution: —

### GEN-7 [Major] [open] [spec-change]
- Aliases: SIMP-10
- Concern: same spec-amendment batch, with two additions the simplicity reviewer missed —
  locked **decision #8** and **OQ3** also restate the `--pyramid-levels` tunability, and §5.3
  becomes dead text if Task 5.5 is cut. Task 6.4 flips Status to Implemented without amending
  any of them.
- Resolution: — **GATED TO USER.**

### GEN-8 [Minor] [open]
- Concern: `live_viewer`, `builder_preview`, `published_store`, `legacy_file_marker` fixtures
  are used but no task creates them — unlike Task 3.3, which specifies its four in a table.
- Resolution: —

### GEN-9 [Minor] [open]
- Concern: Task 2.1's `xfail(strict=True)` instruction is wrong — the test never touches
  `load_zarr`, so it passes at Task 2.1 and `strict=True` turns the XPASS into a failure.
- Resolution: —

### GEN-10 [Minor] [open]
- Concern: Phase 6's `grep -rn "\.h5"` exit criterion is unsatisfiable — **verified**
  `hdf_.py:196 EXT` and `_metadata_migration.py:53 _HDF_SUFFIXES` both survive by design.
- Resolution: —

### GEN-11 [Minor] [open]
- Concern: three statements of the refuted zarr-error claim survive inside the task that fixed
  it (Task 1.6's Constraints bullet, its Step 7 commit message, and the tautological test
  Phase 7 lists as dropped).
- Resolution: —

### GEN-12 [Minor] [open]
- Concern: **self-inflicted by the B10 move.** `write_objmap_in_place` calls `Path(store_path)`
  but imports only `zarr`; every other function uses a local `_Path` alias. **Verified.**
  Raises `NameError` on first call, from both Phase 3 Stage 2 and Phase 4 Task 4.3.
- Resolution: —

### GEN-13 [Minor] [open] [spec-change]
- Concern: `fsync_tree` fsyncs files and the store root, but never the **nested** directories
  (`gray/0/`, `rgb/labels/objmap/0/`) and `promote_store` never fsyncs `final.parent` after the
  rename. On POSIX a durable file does not imply a durable dirent, so §3.7's stated protection
  against node loss has a gap of exactly the shape the section exists to close.
- Resolution: — **GATED TO USER.**

### GEN-14 [Minor] [open]
- Concern: two plan-authored tests cannot pass — Task 4.2's crop test compares a uint8 PIL
  channel against the **float** `gray` layer; Task 7.1's "distinct part directories" asserts
  only that two `uuid4()` calls differ, a test of the stdlib.
- Resolution: —

### GEN-15 [Advisory] [open]
- Concern: `promote_store` can wedge if a concurrent promoter recreates `final` after a
  successful move-aside — rollback is skipped, trash stays non-empty, every retry then fails
  ENOTEMPTY until the budget exhausts, stranding the previous store in trash.
- Resolution: —

### GEN-16 [Advisory] [open]
- Concern: Phase 0's `publish_to_pypi.yml` snippet silently bumps `setup-python@v4`→`@v5`;
  and the `tests/fixtures/ngff` ruff exclusion is inert (ruff visits only `.py`/`.pyi`/`.ipynb`;
  the sha256 guard is the actual protection).
- Resolution: —

### GEN-NOTE [—] [recorded]
- Stage-3 completion markers **self-heal** after migration: `migrate_legacy_stage3_markers`
  (`_cli_staged_resume.py:287-311`) regenerates them from **parquet presence** (`:295-303`),
  not from the image artifact. No work needed there.
- `classify_staged_image` hard-codes `dataset_hdf_dir(...)/f"{stem}.h5"` at
  `_cli_staged_resume.py:197` — inside the classifier, not only the scanner. Confirm it is on
  Task 3.4's line list.
- Uncovered by this reviewer: `.phenotypic/progress/` beyond the Stage-2 token/raw pair, and
  deliverables-side state.

## Round 1 — resolutions applied

Snapshots: `snapshots/round-1-{spec,plan}.md` (plan 9,726 → 10,058 lines).

**User rulings (permanent; no reviewer may re-raise absent new evidence):**

- **SIMP-10 / GEN-7** → `settled-by-user (round 1: amend the spec now, before Phase 0)`.
  Applied: 8 inline callouts in `design.md` — locked decision #8, §1.3, §2.2, §2.3, §3.5,
  §4.2, §5.3, OQ3 — each annotated **as superseded in place**, nothing deleted, matching how
  the flat-metadata spec's decisions #1/#7 are treated. Status line records the amendment.
- **MIG-4** → `settled-by-user (round 1: drop metadata_schema_version, gate store_schema_version
  by value)`. Applied to spec §2.1 and Phase 1 Task 1.6.
- **MIG-8** → `settled-by-user (round 1: one predicate, applied everywhere including the GUI)`.
  Applied: spec §5.1 amended; **new Phase 5 Task 5.7**. Cost was checked before committing to
  it — the GUI surface already exists (`OutputConsistencyReport.reasons` rendered by
  `gui/_snapshot_status.py:74-85`), so this is one added classifier case, not a new component.
- **GEN-13** → `settled-by-user (round 1: close the fsync gap)`. Applied: `fsync_tree` now
  flushes **every** directory deepest-first, and `promote_store` flushes `final.parent` after
  the rename; spec §3.7 amended.

**Plan fixes (provenance-locked):**

- **SIMP-1 + MIG-TASK55** → `resolved (round 1: Task 5.5 cut)`. Cut with all four ripples the
  migration specialist enumerated, and with the **condition** it attached: a test pinning that
  a store written from `MetadataPlate_*` headers comes out canonical, so the invariant making
  the cut safe is pinned rather than inferred. **MIG-6 dissolves with it.**
- **MIG-1** → `resolved (round 1: new Task 5.6, "Migration re-publishes run state")`. Markers
  re-published at v2 with `kind:"store"`, preserving `work_id`/`attempt_id`/`lifecycle_epoch`;
  the refresh bridge dispatches on `kind` instead of hard-raising. Task 3.8's inverted
  rationale corrected. Recorded that Stage-3 markers need no work (they regenerate from
  parquet presence).
- **MIG-5** → `resolved (round 1: narrow Phase 3 → Phase 5 edge declared)`.
- **SIMP-2 / SIMP-3 / SIMP-9 / GEN-4** → `resolved (round 1: Task 7.1 cut down)`. Four
  duplicated tests deleted, two moved to Phase 2, one left in Phase 3, and the unable-to-fail
  `test_interrupt_before_the_root_reads_as_absent` removed with its reasoning recorded. Task
  7.1 now states plainly that "root last" is load-bearing for **flush ordering**, not reader
  visibility.
- **GEN-1** → `resolved (round 1: name the dev group)`. Verified there is no `test` group and
  every CI lane syncs `dev` + `test-qt`.
- **GEN-2** → `resolved (round 1: min_age_seconds=0 in the sweep tests)`, plus a new
  `test_the_sweep_spares_a_young_leftover` — the behaviour the age guard was added for — and
  the Interfaces signature reconciled.
- **GEN-3** → `resolved (round 1: Phase 7 tests target promote_store and _is_retryable)`. The
  dangling `_replace_with_retry` calls are gone; the simulated error uses ENOTEMPTY so it
  actually reaches the retryable branch on Linux.
- **GEN-5** → `resolved (round 1: isolated headless ome-zarr read, manual release check)`.
  Verified neither `ome-zarr` nor `napari-ome-zarr` is in `uv.lock`; `uv run --isolated
  --no-project --with ome-zarr` keeps the ban intact.
- **GEN-6** → `resolved (round 1: Image(<2-D ndarray>))`. Verified `rgb.clear` does not exist
  and the replacement yields `rgb.isempty() is True`; numpy import added where needed.
- **GEN-9** → `resolved (round 1: xfail instruction dropped)`.
- **GEN-10** → `resolved (round 1: allow-list on the Phase 6 grep gate)`. Verified `hdf_.py:196`
  and `_metadata_migration.py:53` survive by design.
- **GEN-11** → `resolved (round 1: all three statements of the refuted zarr-error claim fixed)`,
  and the tautological test replaced with a reachable malformed-array-metadata case.
- **GEN-12** → `resolved (round 1: stdlib imports hoisted)`. Closes **SIMP-8 / PRE-S7** with the
  `zarr` carve-out stated once in the module docstring.
- **GEN-16** → `resolved (round 1)`. The `setup-python@v4→v5` bump reverted as unrequested; the
  ruff exclusion relabelled as documentation, since the sha256 test is the actual guard.
- **SIMP-4** → `resolved (round 1: name-coincidence gate dropped)`.
- **SIMP-5** → `resolved (round 1: tautological P3 test deleted)`.
- **SIMP-7 / PRE-S5** → `resolved (round 1: NOT adopted)`. Task 2.4 consolidates to a single
  `_save_store`, so a `store_writer` CM would be an abstraction with one call site.

**Deferred to round 2** (all Minor/Advisory, none blocking): SIMP-6 (hand-rolled `.part`
cleanup + unused import in Task 3.1), SIMP-11 (misnamed `paths_fingerprint` test), GEN-8
(undeclared GUI fixtures), GEN-14 (two unpassable tests), GEN-15 (promote wedge on a rare
interleaving), MIG-2 (the `_load_v2_grouped` `Metadata_ImageType` drop on the migration path
+ its tautological equivalence test), MIG-3 (fixture gaps), MIG-7b (non-image migration
targets), MIG-9, MIG-10, MIG-11, PRE-G1, PRE-G2, PRE-G3, PRE-G5, PRE-D9, PRE-D10, PRE-S8.

**Coverage gap — recorded, not scored as APPROVE.** `flow-r1` (data-flow) and `algo-r1`
(algorithm-fidelity) signalled idle but never returned a report, each after one re-dispatch.
Per the skill's reviewer-failure rule they are **not** counted as APPROVE. Their charters are
uncovered this round: the Stage-2 three-artifact crash windows and the `--restart`
interaction; the NGFF conformance diff (`chunk_key_encoding` dict form, `"."` separator
legality, 2-axis `(y, x)` legality). Round 2 must re-dispatch both.

**Protocol deviation, recorded:** something overwrote `check_shard.py` in the orchestrator's
scratchpad with an independent re-verification of the chunk/shard policy across 11 plate
sizes (result: `mismatches: NONE`, `level_count/level_shapes: OK`). No worktree file was
touched — `git status --porcelain` shows only the orchestrator's own edits. Treated as a
protocol deviation rather than a reviewer failure: the substance corroborates PRE-B1's fix
across a wider case set than the orchestrator's own check.

**Round 1 verdicts:** simplicity REVISE, migration REVISE, general REVISE, data-flow
NO REPORT, algorithm-fidelity NO REPORT. **Not converged — round 2 required.**

## Round 1 — late reports (both stragglers returned; coverage gap CLOSED)

`flow-r1` and `algo-r1` both delivered after the round-1 write-up. The recorded coverage gap
is **withdrawn** — all five charters were covered.

**Phase 1 unblocked.** ALGO answered both gating questions from the fetched reference: the
`chunk_key_encoding` dict form and the `"."` separator are **valid** (zarr-specs
default-chunk-key-encoding §Description; zarr-python `chunk_key_encodings.py:107-127`,
`parse_separator` `:22-23`), and **2-axis `(y, x)` series are legal** (`image.schema`
`$defs/axes` `minItems:2` with `minContains:2` on `type:"space"`; prose §2.4 makes time and
channel MAY, not MUST).

**ALGO independently re-verified PRE-B1** across 11 plate geometries at every level: zero
mismatches on divisibility, shard-file counts, or channel spanning. It also confirmed
shard-larger-than-array is legal (`sharding-indexed` §Definitions; `sharding.py:573-595`).

**Protocol deviation owned:** ALGO wrote `check_shard.py` into the orchestrator's scratchpad
after the worktree guard refused a heredoc. No repo or worktree file was touched — confirmed
by `git status --porcelain`. Recorded as a deviation, not a reviewer failure.

### FLOW-1 [Critical] [open]
- Concern: `--mode migrate` **drops `phenotypic_work_id`**. Verified:
  `staged_hdf_matches_work_id` reads the HDF **root attribute** (`_cli_staged_resume.py:99-110`),
  written only by the CLI's post-write patch (`_cli_output_manager.py:1665-1670`) and held in
  no image field; `load_image_from_hdf` reads only `phenotypic_class`; Task 5.1 calls
  `save2zarr` with no `work_id`. Every image then classifies `"stage1"` — full reprocessing
  from original inputs a migrated archive may no longer have. **Upstream of MIG-1.**
- Resolution: —

### FLOW-2 [Critical] [alias of MIG-1, three additions]
- (a) Task 3.8's version-bump rationale is **inverted**: with `keep_source=True` the `.h5`
  still exists with matching size and sha256, so without the bump a v1 marker validates
  against a *stale artifact* while the store goes unverified. (b) The only
  republish-without-reprocess path (`_cli_staged_slurm_worker.py:312-347`) is gated on the very
  work-id conjunct FLOW-1 breaks, and has no local/non-staged equivalent. (c) Blast radius
  includes the **dashboard** — `valid_image_success` gates image inclusion
  (`_dashboard/_manifest_builder.py:616`, `:661`).
- Resolution: —

### FLOW-3 [Major] [open]
- Concern: Task 3.8's `paths_fingerprint([store / "zarr.json"])` folds the **absolute resolved
  path** into the digest (`_io_constants.py:196-211`). File descriptors are relocatable today,
  so moving or copying an output tree silently invalidates every store marker — invisibly,
  since `valid_image_success` catches and returns `False`. **Interacts with the copy-mode
  ruling.**
- Resolution: — (use content-only `file_fingerprint(store / "zarr.json")`)

### FLOW-4 [Major] [settled-by-user (round 1: do NOT rewrite metadata.csv)]
- Concern: PRE-D9 answered, and **both offered options were wrong and backwards**.
  `metadata_sha256` is **recomputed from the file every run** (`phenotypicCLI.py:1338`,
  `:2135` — verified), not read from state. Leaving state stale is self-consistent;
  *updating* it is what breaks the aggregate. Either way the rewrite forces a
  re-finalization, and Task 5.2's two tests could not both hold.
- Resolution: **user chose the third option — do not rewrite it.** The read path already
  canonicalizes in memory (`_cli/_metadata_join.py:86-104`). Emit
  `deliverables/metadata.canonical.csv` if a canonical view is wanted. Keeps flat-metadata
  decision #7 **unnarrowed** (its supersession is withdrawn); deletes `metadata.original.csv`,
  `metadata_original_sha256`, the revert hazard, and most of Task 5.2.

### FLOW-5 [Major] [resolved by precedence — CONFLICT with PRE-D6 UPHELD]
- Concern: PRE-D6 covers only the **cached** tile route. The **crop route is uncached** —
  `crop_hdf_rgb` does `del mtime_ns` (`gui/_shared/tiles.py:386`), docstring `:375-376`, fresh
  open `:396`. So the colony-view crop serves Stage 2's **raw, pre-`drop_frame_background`,
  pre-relabel** objmap for the whole Stage-2 → Stage-3 window; on SLURM, hours. Same exposure
  for a cold LRU and for any third-party reader.
- **Ladder applied.** PRE-D6 was an orchestrator resolution, not a user ruling, and FLOW-5
  brings evidence it never considered (D6 reasoned about `@lru_cache` invalidation only).
  Precedence: **correctness (tier 3) over simplicity (tier 6)**; and spec §3.5's own rule is
  "the completion marker, not the store's shape, gates consumers", which the crop route does
  not honour. **FLOW-5 upheld; PRE-D6 narrowed to the cached route.**
- Resolution: — (gate the crop/tile source on the Stage-3 marker, falling back to the overlay
  as `crop_colony` already does at `tiles.py:521-530`)

### FLOW-8 [Major] [resolves PRE-D10]
- **`_metadata_migration.py` costs ZERO for Phase 5.** By the time `save2zarr` runs the
  metadata is already canonical (`_normalize_stored_metadata_items` inside both legacy
  loaders). Its `"hdf"` `TargetKind` builds targets from `dataset_root/"hdf"`; once stores
  replace HDFs that target set is **empty**, so those branches are unreachable, not incorrect.
  **Do not add a `"store"` TargetKind.**
- Resolution: — (record in Task 5.1; give Phase 6 an explicit retain-vs-delete decision)

### FLOW-6, FLOW-9..FLOW-13 [Major/Minor/Advisory] [open]
- FLOW-6: D11's remedy has no stated ordering; a re-promote *after* the marker publish
  invalidates the descriptor it just wrote.
- FLOW-9: `phenotypicCLI.py:196` imports the module Task 5.4 deletes, at **module scope** —
  missing it makes the CLI unimportable.
- FLOW-10: `_load_zarr_layer_rgb` gains a 4th LRU key element while the cache stays at 4 —
  thrashes on the path the pyramid exists to accelerate.
- FLOW-11: `_processing_inventory.py` is a **second** unnamed consumer of
  `_processing_snapshot_paths`; same D5 failure shape.
- FLOW-12: Stage 2's in-place objmap write is not atomic across levels.
- FLOW-13: `--restart` traced **clean** — `clear_machine_state` wipes both token and raw array
  while `results/` survives, so Stage 2 recomputes. No orphaned store. (This also confirms
  deleting `clear_stage2_sidecars` is correct.)

### ALGO-1 [Critical] [settled-by-user (round 1: emit valid OME-XML, vendor ome.xsd)]
- Concern: the emitted OME-XML is **invalid** against `ome.xsd` 2016-06 — `<Pixels />` missing
  all eight required attributes and the mandatory `<MetadataOnly/>` child; `<MapAnnotation>`
  missing its `<Value>` wrapper. NGFF §2.2.3 makes this a conditional MUST, so every store's
  `METADATA.ome.xml` is rejected by the tooling `bioformats2raw.layout: 3` exists for.
- Resolution: **emit valid OME-XML; vendor `ome.xsd`** beside the JSON schemas, read-only,
  with harness validation.

### ALGO-2 [Major] [needs-user-input — DISCUSSION OPEN, NOT implemented]
- Concern: the `omero` window defect PRE-P2 fixed for `detect_mat` is **also on `gray`** —
  verified by execution: float32, `[0.545, 0.955]`, `bit_depth` 8, so the window is `[0,255]`
  over `[0,1]` data. `gray` is the **primary series** in every rgb-less store. PRE-P2 keyed on
  the series *name*, so it missed this.
- Status: user asked whether converting the data to integers would "match the schema" instead.
  Tradeoffs presented; **explicitly not implemented pending discussion.** Key fact: NGFF
  requires integer pixels only for **label** images (§2.6) — image series are unconstrained,
  so there is nothing to match; the mismatch is window-vs-range.

### ALGO-3 [Major] [open — supplies the citation PRE-G2 lacked]
- §2.2.3: without `series`, images MUST sit in consecutively numbered groups. Keeping named
  groups and dropping `series` satisfies neither arm. `assert_store_conforms` skips `OME/` when
  absent, so the fallback has no gate.

### ALGO-4..ALGO-8 [Major/Minor/Advisory] [open]
- ALGO-4: nothing validates the OME-XML, the array-level `zarr.json`s, or the store through
  any reader; the "readable without a PhenoTypic install" motivation is never exercised.
- ALGO-5: `multiscales.type`/`metadata` (SHOULD, §2.4) omitted while the same values sit
  privately in `phenotypic.pyramid`. Also PRE-G1's `factors = {"x": 1.0/x_res}` treats a TIFF
  `XResolution` in px/inch as px/µm — a **25400×** error if ever wired up.
- ALGO-6: `build_image_label`'s docstring misreads the schema path (`required` lives under
  `properties.ome`); the inner `image-label.version` is a 0.4-ism; the Task 1.4 commit message
  still contradicts PRE-P1.
- ALGO-7: three structural MUSTs ungated — `dimension_names` matching `axes`, label pixels
  being an integer dtype, the `labels` group's `labels` array.
- ALGO-8: the validation script's **claim C6 still validates `--pyramid-levels`**, descoped by
  PRE-P3 — stale drift inside the normative reference Task 1.1's tests import.

### USER-MIGRATE-MODE [settled-by-user (round 1)]
- Ruling: `--mode migrate` operates **in place** when only `--input` is given, and **copies**
  when `--output` is given and differs. Equal paths (after `Path.resolve()`) are in-place, not
  an error.
- Applied: spec §5.1 rewritten with the two-mode table and the resumability note; Phase 5
  Task 5.3 constraints and five new tests. Also gives **MIG-9** its answer — copy mode is the
  safe path, revertible by `rm -rf <dst>`.

**Round 1 verdicts (final): all five REVISE.** Not converged; round 2 required.

## Round 1 — closing rulings

### ALGO-2 [Major] [settled-by-user (round 1: omit omero from gray too; defer rendering)]
- Ruling: **remove `omero` from `gray` for now**; revisit making these layers render once
  there is data on the effect on analysis quality.
- Applied **keyed on dtype rather than series name** — `build_omero(*, series, dtype,
  bit_depth, name)` returns `{}` for any floating dtype. Same outcome today (both `gray` and
  `detect_mat` are float32) and self-maintaining: a future float layer needs no list entry,
  and if the deferred integer conversion ever lands, the affected series get their block back
  automatically. The single-white-channel path is retained for exactly that case, so it is
  not dead code.
- Net effect: `rgb` is the only series carrying an `omero` block; an rgb-less store carries
  none. Conformant — §2.5 makes `omero` optional and `image.schema`'s
  `properties.ome.required` is `["multiscales", "version"]`.
- Recorded for the deferred discussion: NGFF mandates integer pixels only for **label**
  images (§2.6), so image series are unconstrained and nothing is being worked around.
  Conversion would break the bit-exact round-trip §7 requires, quantize an analysis input,
  and run into `detect_mat` values not bounded to `[0, 1]` — the `Image` data-model change
  spec §10 already defers to its own design.
- Sites updated: Phase 1 Task 1.4 (interface, constraint, three tests, implementation),
  Phase 2 Task 2.2 (caller passes `arrays[series].dtype`), README Global Constraints,
  spec §2.2 callout.

### FLOW-4 follow-through [applied]
- The spec's **Supersessions** section still narrowed flat-metadata decision #7 for a
  `metadata.csv` rewrite the user had just cancelled. **Withdrawn** — decision #7 now stands
  unnarrowed, with the three reasons recorded inline (the digest is recomputed per run, the
  read path already canonicalizes, and `_snapshot_metadata_csv` would revert it).
- **Phase 5 Task 5.2 rewritten**: from "rewrite with preserved original" to "emit
  `deliverables/metadata.canonical.csv` as a derived view; never touch the snapshot". Its
  three negative assertions are regression guards against the rewrite creeping back.
- **Phase 6 Task 6.4** no longer instructs annotating decision #7; only decision #1 gets a
  supersession note, with a grep to confirm no stale narrowing survived.

### USER-MIGRATE-MODE follow-through [applied]
- Copy mode interacts with **FLOW-3**: `paths_fingerprint` folds the absolute resolved path
  into the digest, so a copied tree's store markers would fail to validate at `<dst>` even
  though the bytes are identical. FLOW-3's fix (content-only `file_fingerprint`) is therefore
  **required** for copy mode to work, not merely tidy. Carried into round 2 with that
  dependency noted.

## Round 2 — REVIEWER FAILURE, coverage gap recorded

Snapshots: `snapshots/round-2-{spec,plan}.md` (plan 10,058 -> 10,704 lines; spec diff ~140
lines, plan diff ~2,288 lines).

**Four of five reviewers returned no report.** `gen-r2`, `flow-r2`, `simp-r2`, and `mig-r2`
were each dispatched with a full diff-scoped brief, then re-dispatched with a detailed
format reminder, then asked a third time for nothing but a single `VERDICT:` line. Each
responded only with bare idle notifications — four rounds of them, the last arriving seconds
after the minimal request. `algo-r2` never signalled at all.

**Per the skill's reviewer-failure rule these are NOT scored as APPROVE.** A silent reviewer
is not a clean one. Round 2 is **not converged** and the round-2 delta is **not
independently reviewed**.

### What round 2 IS verified by

An orchestrator-run mechanical consistency sweep over all nine plan documents
(`scratchpad/selfcheck.py`), which checks exactly the defect class that produced GEN-3 and
GEN-12 in round 1 — dangling symbols and signature/caller drift after a large edit pass:

| Check | Result |
|---|---|
| Removed symbols absent (`write_objmap_in_place`, `resolve_pyramid_levels`, `label_rgba`, `_assert_canonical_metadata`, `_replace_with_retry`) | clean (hits only in `OPEN-QUESTIONS.md`, a history document) |
| Every `build_ome_xml` call passes `series_shapes` / `series_dtypes` | clean |
| Every `build_omero` call passes `dtype` | clean |
| No call passes the deleted `build_multiscales(resolution=)` | clean |
| `build_image_label` called with no arguments everywhere | clean |
| Seven new helpers defined where used (`discard_parts_for`, `_resolve_durability`, `_is_retryable`, `_ome_pixel_type`, `assert_ome_xml_valid`, `DOWNSAMPLE_METHODS`, `SWEEP_MIN_AGE_SECONDS`) | clean |
| No `paths_fingerprint` on a bare store directory | clean |
| No `phenotypic.util` import; no `.metadata.<section>` accessor misuse | clean |
| **README task counts vs actual headings** | **1 REAL DEFECT — phases 5/6/7 stale after Tasks 5.6, 5.7, 6.3a, 7.3a were added and 5.5 cut. Fixed.** |

### What round 2 is NOT verified by

Nobody independently checked the round-2 delta for **judgment**-level defects. Specifically
uncovered, and each named as the top priority in its reviewer's dispatch:

1. **`build_ome_xml` against `ome.xsd` 2016-06.** The orchestrator wrote it from ALGO-1's
   *description* of the schema, not from the schema itself. Element ordering, `Pixels`
   attribute legality, `<MetadataOnly/>` position in the content model, the
   `MapAnnotation`/`Value`/`M` nesting, whether `StructuredAnnotations` may be empty, and the
   completeness of `_ome_pixel_type`'s dtype mapping are **all unverified**. This is the
   single largest untested surface in the plan.
2. **The Stage-2 -> Stage-3 handoff** with no in-store write. The crash-window shape changed
   and was not re-traced.
3. **Whether FLOW-2(b) is satisfied** — does marker republication now cover local and
   non-staged strategies, or only the SLURM stage-3 worker?
4. **Copy mode's partial states and rollback properties.**
5. **The Task 5.7 predicate vs copy mode** — in copy mode `<src>` legitimately keeps its
   `.h5` files forever and never gains stores, so "any dataset with `.h5` results and no
   corresponding store" may permanently flag the source as needing migration. This is an
   interaction between two separate user rulings and was flagged to `mig-r2` as its priority
   item. **Unanswered.**
6. **Whether the plan's growth (10,058 -> 10,704) is justified**, and whether the
   "what an earlier draft got wrong" blockquotes are now overdone.

### Status

**Round 1: converged on nothing — all five REVISE, all findings applied.**
**Round 2: reviewer failure. The plan is self-consistent but not independently reviewed
since round 1.** That is a materially weaker claim than convergence and must not be
reported as one.

Item 5 above is the concern the orchestrator would most want a second opinion on, because it
was predicted from first principles rather than found, and remains unconfirmed either way.

### COPY-MODE [spec-change] [settled-by-user (round 2: withdrawn — migration is in place only)]
- Raised: round 2, orchestrator (surfacing the reviewers' structural findings against it)
- Aliases: FLOW-25, FLOW-26, FLOW-27, FLOW-28, MIG-12, MIG-13, MIG-14, GEN-21, MIG-17,
  FLOW-29, GEN-25
- Concern: A user ruling earlier in round 2 added a copy mode to `--mode migrate`
  (`--input <src> --output <dst>`: convert into a new tree, never write to the source).
  Three independent reviewers then found it structurally incomplete along the same seam.
  In-place migration only has to convert **images**, because every other artifact is already
  in the right place. Copy mode has to reproduce the **entire output contract** at a new
  path — and the draft's copy set named only `deliverables/` and `.phenotypic/`, omitting
  `results/<ds>/measurements/*.parquet`. Those are marker-bound (each per-image completion
  marker carries that parquet's `size` and `sha256`), so the destination would have failed
  `valid_image_success` for every image, lost `migrate_legacy_stage3_markers`'
  parquet-presence signal, and silently reprocessed the whole run — the exact outcome
  migration exists to prevent. It also left a half-finished `<dst>` with no defined state,
  and collided with Task 5.7's predicate: in copy mode `<src>` legitimately keeps its `.h5`
  files forever and never gains stores, so "any dataset with `.h5` results and no
  corresponding store" would permanently flag the source as needing migration.
- Resolution: **Withdrawn on the user's ruling** ("remove copy mode then, I see what you
  mean"). The safety copy mode reached for is already supplied by the default
  `keep_source=True` — the `.h5` files survive the conversion, so if the stores are wrong the
  originals are still there. What copy mode added beyond that was "never write to the
  directory at all", which matters only for a tree the user cannot write to, at the cost of a
  full second copy of every artifact. If read-only-tree migration becomes a requirement it
  lands as its own change, specified against the full output contract rather than an artifact
  allow-list.

  Applied to spec §5.1 (rewritten to in-place-only, with the supersession callout) and to
  `phase-5-migrate.md`: Task 5.2's constraint and its `<dst>` test, Task 5.3's interface
  block, its five-item constraint list, and its five copy-mode tests. The interface reverts
  to **`--output` naming the tree to convert**, matching `recompile` — which also settles
  MIG-17 / FLOW-29 / GEN-25, since the surviving tests already assumed that form.

  `--delete-sources` **survives** the removal (ledger MIG-9: it is the only path to
  `keep_source=False`), and MIG-20's stronger precondition is folded into the same edit —
  unlink only on a positive re-read comparison **plus** a passing `valid_image_success`,
  because both Criticals this review found produce structurally valid stores.

  Deleted rather than fixed: FLOW-25, FLOW-26, FLOW-27, FLOW-28, MIG-12, MIG-13, MIG-14,
  GEN-21, and the half-finished-`<dst>` hole. Task 5.7's predicate needs no copy-mode carve-out.

> **Orchestrator note (process, not content).** The first attempt at this removal used an
> index-slice bounded by two `str.index` calls, and the closing anchor matched inside Task
> 5.7 rather than Task 5.3 — deleting Tasks 5.4 through 5.7 (1080 → 598 lines). Caught by a
> task-heading count, restored from `snapshots/round-2-plan.md`, and redone with exact-match
> replacements only, guarded by a before/after heading count that aborts the write on any
> change. **No slice-bounded edits on plan documents.**

---

# Round 2 (retry panel) — all five reviewers delivered, all five REVISE

**Supersedes the "REVIEWER FAILURE" section above.** That section recorded four of five
reviewers returning nothing. The cause was diagnosed — reviewers were writing their reports
into their own final message instead of sending them — and a fresh panel was dispatched with
an explicit `SendMessage({to: "team-lead"})` instruction at the top of every brief. **All five
delivered.** Round 2 is therefore independently reviewed after all; the coverage gap recorded
above is **closed**, including its item 1 (the OME-XML-vs-`ome.xsd` question), which `algo-r2`
answered by fetching the schema and checking element by element.

Reports: `gen-r2b` (9 Major, 7 Minor), `flow-r2b` (3 Critical, 18 findings), `mig-r2b`
(2 Critical, 13 findings), `simp-r2b` (1 Major, 7 Minor), `algo-r2` (1 Major, 4 Minor).
Recovered in full to `refinery/reports/round-2/` after a context compaction; every finding
below is transcribed from the delivered report, not reconstructed.

## The one clean verdict worth recording

**`build_ome_xml` IS valid against `ome.xsd` 2016-06.** `algo-r2` fetched
`http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd` and cleared all five
sub-questions: `Pixels`' eight required attributes are exactly what we emit and everything
else is `use="optional"`; the content model `Channel*, (BinData+ | TiffData+ | MetadataOnly),
Plane*` makes `<MetadataOnly/>` legal as the sole child even with `SizeC="3"`;
`MapAnnotation`→`Value type="Map"`→`M*` nesting is right; `OME`'s
`…Image*, StructuredAnnotations?, ROI*` order is what we emit and an **empty**
`StructuredAnnotations` is legal; and `_ome_pixel_type`'s map is complete over the reachable
dtypes (`bit`/`complex`/`double-complex` are unreachable for rgb/gray/detect_mat; int64/
uint64/float16 correctly raise, since OME has no equivalent). This was the largest untested
surface in the plan and it is now verified rather than assumed.

## Dispositions

### Dissolved by the copy-mode removal — see `COPY-MODE` above
MIG-12 (Critical), MIG-13, MIG-14, MIG-16, MIG-17, MIG-24, FLOW-25 (Critical),
FLOW-26 (Critical), FLOW-27, FLOW-28, FLOW-29, GEN-21, GEN-32(a), and the `<dst>`
half of MIG-19. Thirteen findings deleted rather than fixed. Two independent reviewers
had additionally predicted a Task 5.7 / copy-mode interaction; `simp-r2b` and `gen-r2b`
both judged it a non-defect and `flow-r2b` found the real hazard was the **inverse**
(a half-finished `<dst>` contains no `.h5` at all, so the predicate reads it as clean).
All three analyses are moot now, and are recorded because they were the evidence for the
removal.

### Applied — Critical
- **FLOW-14** — `test_stage3_publishes_the_post_refined_objmap` sourced `raw_labels` from the
  store, which under round 2 holds Stage 1's zeros, making `published < raw_labels` vacuously
  `False`. Now reads `load_stage2_raw`, with a non-empty guard. This test is named in two
  Phase 3 exit criteria.
- **FLOW-15** — Step 5a's mutation proof was degenerate for the same reason: the one-line
  substitution `result = image.objmap[:]` reads zeros, `drop_frame_background` early-returns,
  and **the mutated code passes**. Worse, the plan then told the executor that a pass means
  the fixture is wrong. Replaced with the two-line seed-then-read form and the reasoning
  recorded.
- **MIG-15** — two-pass ordering inverted **and** the pass entry point does not exist. Order
  is now non-image **first** (it rewrites the parquets the markers fingerprint), the entry
  point is `migrate_metadata_bundle` (`migrate_metadata_schema` is not a symbol), and
  `MigrationReport` gains fields so a pass failure has somewhere to appear. Also added: the
  MIG-21 skip, `--dry-run` semantics, between-pass resumability, and an exit-criterion grep
  for the non-existent symbol.

### Applied — Major
GEN-18/ALGO-9 (dead `if xml is not None` branch + its withdrawn "consecutive-integer
fallback" language, which also did not do what it claimed), GEN-19 (spec §2.4 and §2.2's
resolution row), GEN-20/ALGO-11 (`resolution=` in the Produces block), GEN-22 (eight
Stage-2-writes-the-store prose sites, incl. the `colors` rationale re-based on PRE-P1),
GEN-23 (undeclared Task 5.1 → Task 3.4 edge; Phase 5 now carries an edge table),
GEN-24 (`xmlschema` declared in Task 0.1, packaging test parametrized, exit criteria and
Tech Stack updated), GEN-25 (Phase 5/6/7 exit criteria regenerated; Phase 7's
self-contradicting mutation criterion replaced), FLOW-16 (`--mode process --layer objmap`
was specified to export an all-zeros PNG for every image), FLOW-22 (republication keyed on
marker state, so an interrupted migration's skipped images are still republished),
FLOW-31 (the `kind` dispatch must cover the whole comparison block — `_sha256` raises
`IsADirectoryError` two lines past the guard everyone fixes), FLOW-32/MIG-21
(`CONFLICT with FLOW-8`, **upheld**: `keep_source=True` means the `"hdf"` target set is not
empty; conclusion unchanged, justification corrected), MIG-18 (metadata.csv withdrawal
completed across two spec sites, the Phase 5 header, and the cross-reference rot),
MIG-19 (predicate made per-image and validity-based, `migrate` exempted, consumer treatment
split refuse/inform), MIG-20 (`--delete-sources` gated on a re-read comparison, not just
`valid_staged_store`), SIMP-12 (Task 7.3a folded into `assert_store_conforms`).

### Applied — Minor / Advisory
FLOW-17 (probe both token and raw), FLOW-18 (six `delete_sidecar` sites, not five),
FLOW-19 (four stale prose sites), FLOW-20 (`work_id` dropped from the token payload;
`objmap_shape` kept), FLOW-21 (recorded: idempotency is restored for the objmap only),
FLOW-23 (**third attempt** — the inverted marker-version rationale, in the constraint *and*
the test docstring), FLOW-24 (bridge justified by the cut Task 5.5), FLOW-30 (Task 3.5's
self-contradicting bullet), GEN-26 (`discard_parts_for` had its body only in a Produces
block and no test), GEN-27/SIMP-16 (`DOWNSAMPLE_METHODS`' anti-drift claim was false — the
constant had one reader; now genuinely single-source), GEN-28/SIMP-19 (README decided/
undecided lists and the DAG prose), GEN-29 (fixture table; `finished_legacy_run` is a
dataclass, not a `Path`), GEN-31 (Phase 7 Step 2's premise, false since Task 5.7 adds a
banner in Phase **5**), GEN-32(b) (unused `live_viewer` + the `"r+"` source grep),
MIG-22 (republication *replaces* the artifact set), MIG-23 (markerless legacy tree is a
no-op, not a `RuntimeError`), SIMP-13 (Task 6.3a folded into Task 6.4), SIMP-15,
SIMP-17 (plan archaeology out of six shipped docstrings, into `# NOTE:` comments),
SIMP-18 (redundant OME-XML failure test), ALGO-8 (the geometry script still documented the
descoped `--pyramid-levels` and asserted a whole claim block about it — removed; script
re-run, exits 0), ALGO-10 (`str(REMBI_MODULE.X)` is the Python repr, not `'Study'` — every
MapAnnotation Namespace would have shipped an internal name), ALGO-12 (`image-label.version`
is **not** a 0.4-ism; NGFF 0.5 §2.6 still specifies it), ALGO-13(a–d).

### Deferred with reasons
- **SIMP-14** (six golden fixtures → four via a merged `v2_rich`) — **not applied.** The
  merge is sound and its Step-1a argument is good, but MIG-3 set the count at six one round
  earlier and the two reviewers were reasoning from different snapshots. Not worth churning
  a provenance-locked decision for three files; revisit only if a third reviewer raises it.
- **ALGO-2** (integer conversion for float layers) — still `needs-user-input`, unchanged.
  The user asked to discuss tradeoffs before any implementation.

## Orchestrator-introduced defects in this pass, caught and fixed

Recorded because the round-1 pass produced the same class (GEN-2, GEN-3, GEN-12) and it is
the thing to watch for:

1. **A constraint block landed on Task 5.1 instead of Task 5.6** — the anchor
   `**Constraints specific to this task:**` is not unique, and a first-match replace put
   FLOW-22/MIG-22/MIG-23 under the wrong task. Caught by printing the owning heading;
   relocated.
2. **A slice-bounded excision overran into the C5 block** in `ngff_store_geometry.py` — the
   ALGO-8 removal was bounded by `def main()` with three intervening functions. Caught by an
   assertion on the number of `def`s inside the slice; re-bounded on the next rule banner.
3. **New exit criteria named three test files no task declares** — my own GEN-25 fix
   reintroduced GEN-25. Caught by the post-pass sweep; paths corrected and the two-pass
   end-to-end test (MIG-15c / MIG-24) given a real home in Task 5.6.
4. **A README replacement spliced two half-sentences together.** Caught on read-back.

The post-pass sweep (`scratchpad/selfcheck2.py`) now also checks helper-defined-where-used,
signature/caller drift for the three changed signatures, stale-prose phrases, exit-criteria
test paths against declared ones, and README task counts against actual headings — the last
of which caught Phases 6 and 7 after SIMP-12/SIMP-13 removed a task from each.

---

# Round 2, late delivery — a SECOND algorithm-fidelity report (`algo-r2b`)

`algo-r2b` delivered after the round-2 fix pass was already applied and round 3 dispatched.
It is an independent review of the same round-2 delta as `algo-r2`, and the two **agree on
the headline**: `build_ome_xml` is valid against `ome.xsd` 2016-06. `algo-r2b` reached that
by a stronger method — it fetched the 261,439-byte schema, read the content models, then
**executed the plan's `build_ome_xml` verbatim** against the real
`phenotypic.schema.header_to_module` and validated the output with `xmlschema` 4.3.2, in both
the populated and the empty-metadata case. It also independently confirmed the vendored
`ome.xsd`'s single remote `xsd:import` resolves offline via `xmlschema`'s packaged fallback,
so Phase 0 needs no fifth vendored file.

> **ID COLLISION — read this before citing any ALGO number.** `algo-r2b` numbered its
> findings from **ALGO-9**, colliding with `algo-r2`'s already-applied ALGO-9..ALGO-13, which
> mean different things. Its findings are recorded here as **ALGO-R2B-nn**, preserving its
> own numbers under that prefix. Concretely: r2b's ALGO-9 = r2's ALGO-10 (the `str(enum)`
> bug), r2b's ALGO-12 = r2's ALGO-9 (the dead fallback branch), r2b's ALGO-13 = r2's
> ALGO-13(a) (the `is_dir()` guard). Those three were **already fixed**; r2b confirms the
> diagnosis independently.

### ALGO-R2B-11 [Major] [applied] — control characters from real EXIF make the document not well-formed

The most valuable finding in either algorithm report, because it is about **this project's
actual inputs** rather than the synthetic fixture.

`_annotation` emitted `escape(str(value))`. `xml.sax.saxutils.escape` handles only `&`, `<`,
`>` — it does **not** remove the C0 control characters XML 1.0 forbids outright, not even as
character references. Verified by execution in this worktree: `escape("Canon\x00\x00 EOS")`
returns the NUL intact, and the resulting document raises `not well-formed (invalid token)`.

The input path is real and was verified in source: `_extract_raw_metadata`
(`_image_io_handler.py:495-524`) shells out to `exiftool -json -n` and stores every tag, and
`_normalize_metadata_value` (`:296-331`) decodes `bytes` via `value.decode("utf-8",
errors="replace")` — which repairs invalid UTF-8 but leaves `\x00` untouched. NUL-padded EXIF
strings and UTF-16LE `XP*` tags survive as Python strings containing control characters.

Two consequences, both silent:
1. **Production writes a broken file and nothing raises.** `build_ome_xml` is pure string
   formatting, `save2zarr` writes the string, and the conformance gate runs only over
   synthetic plates with clean metadata. The store then fails to open in exactly the
   Bio-Formats/OME tooling `bioformats2raw.layout: 3` exists to serve — the ALGO-1 failure
   mode one layer down, on the DSLR/raw captures this project ingests.
2. **The harness could not have reported it cleanly either.** `assert_ome_xml_valid` caught
   only `xmlschema.XMLSchemaValidationError`; a well-formedness failure raises
   `XMLResourceParseError`, whose MRO is `XMLResourceError` → `XMLSchemaException` →
   `ElementTree.ParseError` → `SyntaxError`. Verified: `isinstance(..., XMLSchemaValidationError)`
   is `False`. The documented `Raises: AssertionError` contract did not hold for the most
   likely real failure.

**Applied**, taking r2b's recommendation to **sanitize rather than raise**: a NUL in a camera
tag is legitimate input, and OME-XML failure is fatal by user ruling, so raising would abort
a real run over a `MakerNote`. Added `_XML_FORBIDDEN` (the complement of XML 1.0's `Char`
production) and `_xml_text()`, applied to **both** the `K` attribute and the text; widened the
harness `except` to `xmlschema.XMLSchemaException`; added a Task 1.4 test with a NUL-bearing
`imported` section.

> A first draft of that test also called `ElementTree.fromstring` as a well-formedness probe.
> Removed: with the widened `except` the single `assert_ome_xml_valid` call is the whole gate,
> and a bare stdlib parse over attacker-influenced EXIF text has no billion-laughs guard. A
> comment records why not to add one back.

### ALGO-R2B-10 [Major] [applied] — Phase 1's XSD test imported a Phase 2 module

`phase-1-ngff-core.md`'s `test_ome_xml_validates_against_the_vendored_xsd` did
`from tests._ngff_conformance import assert_ome_xml_valid`, but Task 2.5 **created** that
module — verified absent from the worktree today. Phase 1's exit criterion is
`uv run pytest tests/unit/sdk_/test_ngff_*.py -q`, so Phase 1 would have failed at
**collection**, not at assertion.

As r2b put it, this is worse than an ordinary ordering slip: it is precisely the ALGO-1
remediation, so the executing agent meets a red gate at the exact moment deleting the
assertion looks most reasonable. **Applied** via r2b's preferred branch — Task 1.4 now creates
the module with `_ome_xsd()` + `assert_ome_xml_valid` (both depend only on Phase 0's vendored
fixture and `xmlschema`, already in place), and Task 2.5 **extends** it with the JSON-schema
half, `assert_store_conforms`, and `_assert_reader_level_musts`.

### ALGO-R2B-14 [Minor] [applied] — the label group's `multiscales` was validated by nothing

`label.schema` declares only `image-label` and `version` under `properties.ome`, requires
those two, and sets no `additionalProperties: false` — so `_validate(..., "label.schema", ...)`
passes regardless of what the label's multiscales block contains. Meanwhile §2.6: *"The
zarr.json file for the label image MUST implement the multiscales specification"*, and §2.1's
`dimension_names` MUST applies to every multiscale level. The reader-level loops iterated
`("rgb", "gray", "detect_mat")` only.

**Applied**: `_assert_reader_level_musts` now iterates `[*series_names, label_member]`, so the
label group gets `datasets[].path` resolution and the `dimension_names`-vs-`axes` check like
any other. The level-count MUST (ALGO-13(c), already added) was rewritten to compare the two
loop results rather than re-read the group.

### ALGO-R2B-16 [Advisory] [applied] — the label multiscale omitted `name`

§2.4: *"Each 'multiscales' dictionary SHOULD contain the field 'name'."* All three image
series got `Metadata_ImageName`; the label call site passed none. Now passes
`f"{name}/objmap"`.

### ALGO-R2B-15 [Advisory] [applied as a docstring note] — `_ome_pixel_type` audited clean

r2b independently re-derived the same conclusion `algo-r2` reached: the eight mappings are
each a legal `PixelType` enumeration value, correctly assigned; `np.dtype(dtype).str[1:]`
strips byte order correctly for `|u1`, `<u2`, `<f4`, `<f8`; `bit`/`complex`/`double-complex`
are legal but unreachable; `int64`/`uint64` have no OME equivalent, so raising is right.
**No code change** — but r2b's suggestion was taken: the constant's comment now says it is a
deliberate subset, so a future reader does not "complete" it with a wrong entry.

### `CONFLICT with ALGO-6` — upheld, and already applied
r2b challenged ALGO-6's claim that the inner `image-label.version` is "a 0.4-ism", quoting
NGFF **0.5** §2.6: *"Second, a `version` key, whose value MUST be a string specifying the
version of the OME-Zarr image-label schema."* This is the same correction `algo-r2` raised as
ALGO-12 and it was already applied. r2b additionally caught a **residue neither I nor
`algo-r2` found**: Task 1.4's commit message still said *"with one deterministic-hash colour
per unique label value"*, contradicting the settled PRE-P1 background-only ruling. **Fixed.**

### Also applied from r2b's Task 7.3a assessment
- The chunk-key comment cited "§1.4", which is the **PhenoTypic design spec** §1.4, not NGFF —
  and the surrounding comments use bare section numbers for NGFF, so it read as a false
  citation. Now says so explicitly, with r2b's note that neither NGFF nor Zarr v3 imposes
  cross-array separator uniformity: it is a PhenoTypic rule.
- The `configuration` `KeyError` r2b flagged was already fixed by `.get` chains (ALGO-13(d)).
- r2b's sharding observation is recorded here rather than in the plan: with sharding, the
  top-level `chunk_key_encoding` governs shard filenames only, and the inner encoding lives in
  the `sharding_indexed` codec configuration, which the check does not reach. r2b judged this
  fine for what the test claims, and I agree — noted so a future reader does not mistake the
  check for broader than it is.
- ALGO-R2B-9's suggested assertions were added: `assert "REMBI_MODULE" not in xml` and
  `assert 'Namespace="ImageData"' in xml`. The underlying bug was already fixed, but nothing
  would have caught its return, since `@Namespace` is `xsd:anyURI` and accepts the Python repr.

### Process note
Two independent algorithm reviewers on the same delta produced **substantially different**
finding sets, overlapping on three items. The two Majors above — a phase-ordering break and a
production-data well-formedness defect — were found by r2b alone, and neither is visible to
the mechanical sweep. That is an argument for the redundancy, not against it; the cost was an
ID collision, which a per-reviewer ID prefix would prevent.

---

# Round 3 — five reviewers, five REVISE, 7 Critical

All five delivered. Reports in `refinery/reports/round-3/`. **Every Critical this round was
introduced by the round-2 fix pass**, and the dominant failure mode was named independently
by two reviewers: `gen-r3` — *"corrections written into the wrong task"* — and `simp-r3` —
*"the fix was written down more completely than it was applied."*

## Criticals

### GEN-33 [Critical] [fixed] — the folded reader-level check raised `KeyError` on every store
`_assert_reader_level_musts` read `_group_ome(store)["series"]`. Task 2.2 writes the root as
`{"version", "bioformats2raw.layout"}` and puts `series` on the **OME group** — which is also
what `ome.schema` requires and what `test_missing_series_is_rejected` deletes. Since
`assert_store_conforms` is called by every store-writing test in Phases 2, 3 and 5, **all of
them errored.** Mine, from the SIMP-12 fold. Fixed to `_group_ome(store / OME_GROUP)`.

Second defect in the same block: the fold turned a **tolerated** label-less store into
`FileNotFoundError`. The old harness iterated `block[LABELS].values()`, a no-op on an empty
mapping, and `save_intermediate_zarr(layers=("gray",))` writes exactly such a store. The label
arm is now guarded and a positive test pins it.

### FLOW-33 [Critical] [fixed] — the replacement mutation proof was a no-op
FLOW-15's fix replaced a mutation that silently passed with one that **also** passes.
`flow-r3` traced it: `_write_object_output` opens with
`image.objmap[:] = result.astype(np.uint16)` (`abc_/_gpu_detector.py:243`), discarding the
prior objmap before reading it, and `ObjectMap.__setitem__`'s full-slice fast path
(`_objmap_accessor.py:203-216`) round-trips a `uint16` array losslessly — so seed-then-read
inside `stage3_merge_measure_core` is byte-identical to the correct code, **and the seed
re-runs on the retry**, re-supplying clean labels to the pass that must see refined ones.
Worse than what it replaced, because FLOW-15 had correctly removed the "if it passes, the
fixture is wrong" escape hatch, so the executor now met a contradiction with no recovery text.

Fixed with `flow-r3`'s cheaper variant: seed the store **once, in the test**, between
`run_stage2()` and the first `run_stage3()`; production mutation stays the one-line store
read. Both failed drafts are now recorded in the blockquote with the reason each failed.

### MIG-25 / FLOW-35 [Critical] [fixed] — the pass inversion made pass 1 rewrite every `.h5`
**Raised independently by both reviewers, and it is the one place a fix I took on advice made
something materially worse.** Verified in source: `_discover_bundle_targets` appends every
`.h5` under `dataset_root/"hdf"` (`:797-812`); the apply path is `_migrate_hdf_copy`
(`:1365-1369`), which does **`shutil.copy2`** — a full byte copy — per file; and every
pre-flat-metadata `.h5` reads as `migratable` even when already canonical, because
`_inspect_hdf` sets `needs_metadata_marker` on a missing marker alone (`:604-606`) and
`_target_status` returns `"migratable"` on *either* signal (`:229-239`).

Under the pre-inversion order the MIG-21 store-existence skip fired for every converted image.
The inversion runs pass 1 **before any store exists**, so the skip is structurally dead on a
first migration — the only one that matters. Three consequences, all on the default path:
the retained-`.h5` rollback story that justified removing copy mode is destroyed (they are no
longer the originals); the pre-existing per-image markers are invalidated (MIG-1 in the `.h5`
direction); and Task 5.4's cost claim — the stated reason for deleting an 879-line SLURM
fan-out — is falsified, since the fan-out existed for exactly this rewrite.

The plan even stated it and drew the opposite conclusion: *"on a first migration nothing is
skipped and the cost claim holds."* Nothing skipped **is** the cost claim failing.

**Fixed by excluding `.h5` from pass 1 unconditionally**, justified by Task 6.4's own fact
that header canonicalization is a **read-path** property (`_normalize_stored_metadata_items`,
inside both legacy loaders) — so `save2zarr` writes canonical metadata whether or not the
source header was rewritten. Dead work in every case, not just on re-runs. Same fact that made
Task 5.5 unnecessary. No user ruling needed: both reviewers judged this strengthens the
copy-mode removal rather than reopening it.

### MIG-26 [Critical] [fixed] — the filter was not expressible and no task authorized the file
`migrate_metadata_bundle(source, *, expected_plan_fingerprint)` takes no filter,
`preflight_metadata_schema(source)` takes one argument, `_discover_bundle_targets(layout)` is
private, and `BundleLayout` is a frozen two-field dataclass. Neither
`_metadata_migration.py` nor `_io_constants.py` appeared in any task's `Files:`. The executor
would have met a constraint they were not authorized to satisfy. Task 5.3 now declares
`_metadata_migration.py` with an additive, default-off `kinds` parameter.

### MIG-27 [Critical] [fixed] — "lift the preflight" omitted the load-bearing line
The important line in `migrate_metadata_schema_for_recompile` is not the preflight but
`layout = _metadata_bundle_layout(resolved_output)` — which constructs `BundleLayout`
**directly**, and whose docstring says why. Passing a `Path` routes through
`BundleLayout.detect`, which **raises `FileNotFoundError` unless
`deliverables/master_measurements.parquet` exists** — so every Task 5.3 test would have failed
at pass 1, including against the plan's own `legacy_run` fixture, with an error message naming
nothing relevant. Task 5.3 now gives the three-line call contract explicitly.

### MIG-28 [Critical] [fixed] — `--delete-sources`' guard could not catch MIG-2
The comparison set ("layer names, shapes and dtypes, the metadata **key set**,
`phenotypic_work_id`") catches FLOW-1 and misses MIG-2 — because MIG-2 is not a dropped key.
`_load_v2_grouped`'s restore loop (`_image_io_handler.py:1073-1076`) **skips** a value when the
constructor already set a non-`None` default, so `Metadata_ImageType` is *present*, carrying
`"Image"` where the file said `"GridSection"`. Two identical key sets → `True` → the `.h5` is
unlinked and the correct value is gone permanently. Now specified as value-level: full mapping
comparison plus per-layer content digests plus grid state.

### FLOW-34 [Critical] [fixed] — `--dry-run` was rejected by the guard I said needed no exemption
I wrote *"`migrate` is added to the same tuples `recompile` already appears in. No guard needs
a new exemption."* Both halves wrong: `phenotypicCLI.py:1216-1242` has **booleans, not
tuples**, and the block rejects `--dry-run` for exactly those modes. `--dry-run` is required by
spec §5.1, Task 5.1's signature, a test, and an exit criterion. Now three explicit edits with
migrate exempted from the dry-run guard only.

## Majors and Minors — all applied
GEN-34/FLOW-38 (the `work_id` drop was recorded in Task 3.3 and never applied in Task 3.2,
which executes first and had a **passing test** asserting the field), GEN-35/FLOW-39 (FLOW-18's
raw-clear instruction landed in Task 3.5, which does not own `_cli_staged_resume.py`; Task 3.4
owns it and still said the opposite), FLOW-40 (`classify_staged_image` still gated on the token
alone, so a token-present/raw-missing image was `"stage3"` forever — and `flow-r3` correctly
warned that simply ANDing the raw in would flip such an image to `"complete"`), GEN-36
(`features-md-gate` is a **diff** gate, verified at `gui-checks.yml:92-106`: any PR touching
`src/phenotypic/gui/` without diffing `FEATURES.md` fails, so the change was going to be
blocked at its final step with the executor sent to hunt a chrome change that does not exist),
GEN-37 (the Stage-2 ruling had never reached the spec — locked decision #4, §3.4, and §3.5's
own correction block all still asserted the in-store write; **spec-change, applying an existing
ruling**), GEN-38 (a ninth GEN-22 site, instructing the withdrawn claim be written into two
project `CLAUDE.md` files), GEN-39, GEN-40, GEN-41, GEN-42, GEN-43, GEN-44, GEN-46, GEN-47
(nothing proved any folded MUST could fail — five negative tests added), GEN-48 (the Task
1.4/2.5 split was declared in a `Files:` bullet and never given a **step**; Step 5 staged two
paths, neither the harness), GEN-49, GEN-50 (the imports are function-local, so the failure is
at call time, not collection), GEN-51, GEN-52 (replaced "copy them so they cannot drift" with
a single definition — copying *causes* drift, and this repo's one deliberate duplicate carries
a CI byte-equality guard), MIG-29, MIG-30, MIG-31, MIG-32, FLOW-36, FLOW-37 (one word:
republication **rewrites, never creates** — the looser wording fired on every image of a
markerless archive, where `publish_image_success` has no `work_id`/`attempt_id`/
`lifecycle_epoch` to be given and, unlike its three siblings, does not short-circuit on
`success_markers_required`), SIMP-20, SIMP-21, SIMP-22, SIMP-23, SIMP-24.

## Deferred, with the reviewer's own agreement
**SIMP-14's fixture merge: withdrawn by its author.** `simp-r3` re-litigated as asked and
split its own finding: *"You were right not to merge. I was right about the gap. They are
separable, and I conflated them."* The merge is below the churn threshold; the **gap** —
Step 1a pins only one of the two production writer paths, and the window closes permanently at
Phase 6 — is real and cost two lines. Applied as a parametrize over both writer paths, plus
SIMP-22's two under-parametrized golden loops.

## Measured, on request
`simp-r3` counted the blockquote growth: 138 → 168 → **340** lines (1.37% → 1.57% → **3.06%**
of the plan), +174 added this round = 19.7% of all added lines. It sampled ten blocks and
judged **86% load-bearing**, verdict unchanged (*"do not thin them"*), but flagged the
trajectory and prescribed the house style for round 4: **full reasoning in one document, a
pointer plus one sentence everywhere else** — the form the copy-mode note already uses. Also
corrected my own claim about SIMP-12: the fold was **net −5 lines**, not the ~120 I asserted.
Sell it on coverage, not on lines.

## Orchestrator-introduced defects in THIS pass, caught before dispatch
5. **Two GEN-48 blocks landed under Task 1.1 and Task 1.2** instead of Task 1.4 — `Step 2: Run
   it to verify it fails` and `Step 3: Append to ngff_.py` each appear in five Phase 1 tasks,
   and a first-match replace put them under the earliest. Identical to defect #1 (the Task
   5.1/5.6 constraint block). Caught by printing the owning heading; repaired with a
   task-scoped inserter that asserts the anchor falls inside the named task's span.
6. **Two more `BundleLayout.resolve` call sites in Phase 2**, the same non-existent symbol
   MIG-32 found in Phase 5. Caught by the extended sweep, not by the reviewer.

`selfcheck3.py` now also checks: fixture identifiers named in exit criteria against fixtures
any task defines (would have caught SIMP-20); wrong-task placement of named blocks; `Files:`
appearing before a task's first step (would have caught GEN-39); and stale pre-inversion pass
numbering (would have caught FLOW-36/MIG-30 at all five sites).

**Standing rule, from two repeats:** inside a multi-task document, never anchor an edit on a
string that recurs across tasks. Anchor on the task heading and assert the insertion point
falls within that task's span.

---

# Round 3 — algorithm-fidelity report (`algo-r3`), delivered last

Fetched and read this session: the NGFF 0.5 spec HTML, the Zarr v3 core spec (mandatory vs
optional array metadata), the Zarr v3 `sharding_indexed` codec spec, W3C XML 1.0 5th ed. §2.2,
and upstream `xmlschema/exceptions.py`. `ome.xsd` was cleared in round 2 and not re-opened.

**ALGO-14 [Critical] is the same defect as GEN-33** — `_group_ome(store)["series"]` reading the
root instead of the OME group — found independently, and already fixed before this report
arrived. `algo-r3` added one detail neither GEN-33 nor I had: the negative lane would have
stayed **green** while the positive lane errored, because `_validate(..., "ome.schema", ...)`
fires first on a deleted `series` (`ome.schema` requires `["series","version"]`). Positive
tests erroring while negative tests pass is the worst possible failure signature for a
conformance harness, and worth recording as a pattern.

### ALGO-16 [Major] [applied] — the assertion's stated reason misquoted the spec
`"OME/ group is mandatory under layout 3"` is **not** what §2.2.3 says. Layout 3 makes the OME
metadata a **SHOULD** (*"SHOULD have OME metadata … in a file named `OME/METADATA.ome.xml`"*)
and `series` a **MAY**. This is the ALGO-12 failure mode verbatim: a future reader checks the
spec, finds SHOULD/MAY, concludes the assertion overreaches, and softens it back to
`if ome_group.is_dir():` — reinstating the silently-skipped surface ALGO-3/ALGO-13(a) closed.

The assertion is right; the reason is the **named-series layout**. §2.2.3: *"If the `series`
attribute does not exist and no `plate` is present: separate `multiscales` images MUST be
stored in consecutively numbered groups starting from 0."* This writer emits
`rgb`/`gray`/`detect_mat`, so `series` is load-bearing and `OME/` is the only place §2.2.3 puts
it. Message rewritten to say that.

### ALGO-17 [Minor] [applied] — the path-order check proved a writer convention, and could not count
Two gaps. **(a)** The comment elided §2.2.3's *"if provided"* clause. **(b)** §2.2.3 also
requires *"Every `multiscales` group MUST represent exactly one OME-XML `Image`"*, and `Name`
is `use="optional"` in `ome.xsd` — so a name-only scrape cannot see an unnamed `<Image>`, and a
document with three named Images matching `declared` **plus a fourth unnamed one** passed while
violating the 1:1 MUST. **(c)** `quoteattr` switches to single quotes when the value contains a
double quote, which `\bName="…"` cannot match.

Applied as **two** assertions — an independent `<Image\b` count, then the order comparison with
both quote styles matched — and the comment now states plainly that Name-equals-path is *this
writer's convention*, stronger than the positional MUST the spec requires. **Verified by
execution**: the count sees an unnamed Image (3, not 2) and the order regex returns
`['rgb', 'say "hi"']` across both quote styles.

`algo-r3` recommended `ElementTree.fromstring` instead. **Declined**, and the reason is
recorded so it is not re-proposed: a second stdlib XML parse over user-derived EXIF text has no
billion-laughs guard, and the same hazard already caused an `ET.fromstring` probe to be removed
from a Task 1.4 test this session. The two regexes close both gaps without it.

### ALGO-18 [Minor] [applied] — ALGO-13(d)'s `.get` treatment reached the separator and not `dimension_names`
`meta["dimension_names"]` raised `KeyError`, not `AssertionError`, for precisely the case the
assertion exists to catch: §2.1 makes the attribute a **MUST**, while Zarr v3 lists it as
**optional** array metadata — so an array missing it is a valid Zarr array and an NGFF
violation. The `Raises:` contract was false for that failure. Now a `.get` with an explicit
`is not None`. `algo-r3` also noted the ALGO-R2B-14 label-loop change makes this bite twice as
hard, since label levels now go through the same line.

Same finding killed `assert array.shape` as a **tautology** — every zarr array has a non-empty
shape tuple. Replaced with the §2.4 rank MUST (*"The number of dimensions and order MUST
correspond to number and order of `axes`"*), which nothing checked.

### ALGO-19 [Minor] [applied] — and it exposed a SIMP-23 fix that was recorded but not performed
The `rglob("*/0")` walk only ever inspected level `0`. More to the point, **SIMP-23's edit had
replaced the comment claiming the two chunk-key checks were merged while leaving the second
loop in place** — the exact "written down more completely than applied" pattern `simp-r3`
named, committed by me in the same pass that recorded it. The loop is now actually gone, and
the surviving check is gated on `node_type == "array"` so an array omitting
`chunk_key_encoding` (mandatory in Zarr v3) fails rather than being skipped alongside the group
documents.

### ALGO-20 [nit] [applied]
`label_member` was re-derived by hand. It now reads `block[PhenotypicAttr.LABELS][OBJMAP_LABEL]`
— the path **the store itself declares** — which turns the loop into a real check that the
declared label path resolves. A re-derived path cannot fail.

## Two of my own claims corrected by this report

1. **My sharding note was wrong.** I recorded `algo-r2b`'s observation that the uniformity check
   is an "acceptable narrowing" because sharded inner chunks carry their own encoding.
   `algo-r3` refuted it against the `sharding_indexed` spec: the codec's configuration has
   exactly four members (`chunk_shape`, `codecs`, `index_codecs`, `index_location`) and **inner
   chunks are addressed by byte offsets in the shard index, not by keys at all**. So the
   array's top-level `chunk_key_encoding` is the only thing in a Zarr v3 store that turns
   coordinates into a path segment, sharded or not — the check is **complete**, not narrowed.
   That matters because "acceptable narrowing" would send a future reader hunting for an inner
   encoding that does not exist. Corrected in the code comment.
2. **`_OME_PIXEL_TYPES`' comment gave the wrong reason** for `objmap` never reaching it: not
   that its dtype is unmappable (`uint16` maps fine) but that **labels get no `<Image>` element
   at all**. Corrected.

## Cleared explicitly, against fetched published text
`_XML_FORBIDDEN` was checked **boundary for boundary** against XML 1.0 5th ed. production [2]
and is exact — including that `#xFFFD` is *retained* (it is the top of `[#xE000-#xFFFD]` and is
what `decode(errors="replace")` emits, so stripping it would delete the very marks that record
a repair), and that lone surrogates are stripped, which matters because Python `str` can hold
them via `surrogateescape`. Two riders applied: a note that this is XML **1.0, not 1.1** — the
C1 block is *discouraged*, not forbidden, so "tightening" it would drop legitimate MakerNote
bytes — and `Namespace={quoteattr(module)}` now goes through `_xml_text` for symmetry.

The `XMLSchemaException` widening was verified against upstream's class hierarchy
(`XMLResourceParseError(XMLResourceError, ParseError)`, `XMLResourceError(XMLSchemaException)`)
and is **strict** — nothing previously caught is now missed. One rider applied: the message now
carries `type(exc).__name__`, so an `XMLSchemaOSError` is not misread as a conformance bug.

Also confirmed conforming with citations: the label dtype set, the label group's inclusion in
the reader-level loop, `axes` ordering for `rgb` as `(c,y,x)`, `multiscales[].type`/`metadata`
as SHOULDs surviving the `DOWNSAMPLE_METHODS` refactor, omitting `axes[].unit` after the
resolution withdrawal, and §2.6's level-count referent being the **primary** series (the
question I had flagged for scrutiny — the labels group is nested under primary, so "the
original unlabeled image" is the containing group).

**Recorded deviation, not an oversight:** `build_image_label`'s background-only `colors` is a
deliberate documented departure from §2.6's *"MUST be a JSON array describing color information
for the unique label values"* (under a SHOULD-level key). Pre-dates this round; on the record.

---

# IMPLEMENTATION — Phase 0 and Phase 1 complete

Four review rounds and a pre-dispatch gate hardened this plan on paper. Implementation
then found a further class of defect that **none of them could have caught**, because
reading a plan cannot tell you whether its tests are capable of failing.

## What the mutation surveys found

Phase 1's tests were surveyed twice: 13 + 8 mutants by the cluster that wrote Tasks 1.5/1.6,
then **90 mutants** by the phase gate over Tasks 1.1–1.4. **Twenty-six survivors in total** —
code that could be broken in consequential ways with the entire suite green.

The two worst were both "tests the leaf, not the branch":

- **`build_pyramid`'s `kind` dispatch was untested.** Hard-coding the image reducer
  mean-downsamples label maps, inventing label values present at no pixel of the original.
  The only `kind="label"` test asserted **shapes**, and both reducers produce identical
  shapes. This is the exact mutation the committed validation script names as "the one the
  pyramid test must catch" — and that script's own claims were never executed by any test or
  CI job, so it sat a directory away as documentation.
- **Task 1.5's fifteen tests for the crash-safety primitive caught none of the five
  crash-safety mutants.** Retry loop cut to one attempt, rollback deleted, `fsync=True`
  ignored, directories never flushed, a genuine `ENOSPC` retried as transient — all green.
  A task that exists *only* to be crash-safe had no test of its crash-safety.

Others worth recording: `array_create_kwargs` could drop sharding entirely (40 → 132 files
per plate by the plan's own numbers); **seven** `attributes.phenotypic` keys could be dropped
or crossed, including swapping `illuminant`/`gamma` (silently wrong colour on every read) and
hard-coding `image_class` (every `GridImage` reloads as a plain `Image`); OME-XML `SizeC`
could be pinned to 1, making an RGB store declare itself single-channel to every Bio-Formats
consumer while `multiscales` says three — schema-**valid**, so the XSD gate could not see it.

All twenty-six are now killed, each demonstrated red-then-green.

## Rulings the plan carried in prose but not in code

**`store_schema_version` by VALUE, and `metadata_schema_version` dropped** (user ruling
2026-08-19, spec §2.3). Task 1.3 wrote the dropped key anyway; Task 1.6 checked presence with
`not in block`. **Only Task 1.6's commit message recorded the ruling correctly** — which is how
the implementing agent noticed the contradiction, and it correctly declined to resolve it
alone since it changes what Phase 3's classifier and Phase 5's migrate accept.

Presence-only is not a weaker version of the same check: it **accepts** a v4 store and reads it
under v3 semantics, which is the ruling's stated reason for existing. The existing test covered
only the absent case, which passes under both implementations, so the bug was invisible.

## Defects found by executing, not reading

- **`valid_staged_store` raised `AttributeError`** on a store whose `series` is a list rather
  than a mapping. Its docstring insists it "must RETURN FALSE, never raise", and it is called
  by resume classification and migration to decide what to do next — a production crash, not a
  rejected store.
- **The plan contradicted itself on `long_path`**: the constraint names five call sites that
  must use it; the code block used bare paths at every one. Implemented verbatim, the MAX_PATH
  measure would have applied to array I/O only — the exact defect PRE-G3 was raised to fix.
- **`long_path`'s `resolve()` broke its own passthrough test on macOS**, a supported platform,
  because `$TMPDIR` is symlinked.
- **A test block used `np.dtype` nine times without importing numpy** — a `NameError` at
  *argument* position, firing before the `AttributeError` the plan's Step 2 predicts, which
  defeats the stop-and-investigate signal the TDD discipline depends on.
- **`test_no_metadata_literals` failed on the new code**: the gate bans the bare token
  `METADATA`, which collides with NGFF's mandated `METADATA.ome.xml` filename. Only a
  **full-suite** run finds this — the task's own tests do not scan the tree, and the task did
  not list the gate file, so no executor was authorised to fix it.

## Method notes worth keeping

**A green suite is not evidence.** Every claim in this phase that mattered was established by
mutation: apply the break, watch it go red, revert, watch it go green. Where an agent reported
"tests pass", that was treated as the beginning of the check rather than the end.

**Two of my own briefs were wrong and the executing agents corrected them** — a mutation I
described changed the output shape (so existing tests already caught it, meaning it could not
have been the survivor), and two tests I claimed overlapped are independent because `np.rint`
is banker's rounding. Both corrections came with the transcript that proved them.

**The full suite is 6,814 tests, not 946.** Both earlier "baseline" measurements used `-x`,
which stops at the first failure inside `tests/unit/cli/` — early in collection order. Every
comparison made against that number was measuring a seventh of the suite.

---

# IMPLEMENTATION — Phase 2 complete

Five tasks, three clusters. The gate ran 52 mutants over the two tasks no cluster had
surveyed and found **12 survivors plus 2 hard defects**; the harness's own author had
already found **10 of 16** assertions in `_assert_reader_level_musts` unguarded by the
plan's fourteen tests.

## The failure mode worth remembering: a fixture that cannot discriminate

`gray` and `detect_mat` are **lazily derived** from `rgb`, and the round-trip fixture was
never enhanced — so `np.array_equal(img.gray[:], img.detect_mat[:])` was literally `True`.
Consequences, all of which passed every test:

- delete the loader's read of `detect_mat` — passes, because reload re-derives it
- delete the loader's read of `gray` — same
- write **`detect_mat` pixels into the `gray/0` array on disk** — passes, store fully
  NGFF-conformant, any external viewer shows the wrong image

Nothing in the suite read a non-`rgb` series off disk and compared its **values** to
anything. The fix is an enhanced fixture (so the stored array and the lazy derivation
genuinely differ) plus a test that opens each array with `zarr.open_array` and compares to
the source, bypassing the loader entirely.

The general lesson: **a round-trip test over a fixture whose layers are derivable from one
another proves only that the derivation still works.**

## A defect from two namespaces that share a spelling

`load_image_from_store` dispatched on `class_name == IMAGE_TYPES.GRID.value`. `IMAGE_TYPES`
is the **`Metadata_ImageType` vocabulary**; `image_class` is `type(self).__name__`. They
match only because one enum member happens to spell it the same way — and **spec §2.1's own
example block writes `"Metadata_ImageType": "Grid"`**. Renaming that member would have
silently degraded every `GridImage` to a plain `Image`, with no error anywhere. Both
dispatchers (store and HDF) now compare against `GridImage.__name__`.

Both existing tests were non-discriminating: one used an `Image` where both fields said
"not a grid", the other a `GridImage` where both agreed. **Two fields that always agree
cannot pin which one is read.**

## Gates that were red, wrong, or unsatisfiable

- **`test_load_layer_hdf5.py` was red.** Task 2.4 removed `save_intermediate_layers`; this
  file was its last caller, so the exit criterion "green **and unmodified**" became
  unsatisfiable the moment the method went. Now writes the legacy-flat bytes directly —
  better, because this file was also the only remaining *producer* of that layout, and
  Phase 5 migration depends on the read path.
- **The suffix grep could not pass**, and shouldn't: written as a bare literal it caught
  seven builder-cache **node filenames**, a different namespace from the `results/<ds>/zarr/`
  store paths it exists to police. Routing them through `STORE_SUFFIX` was not available —
  every `phenotypic` import in `_preview_cache.py` is function-local (11, no exceptions),
  so a module-level constant cannot reference one. Tried it; `NameError` at import; reverted
  rather than restructuring a module's import strategy to satisfy a grep.
- **The metadata gate recurred.** Allowlisting file-by-file worked for Phase 1 and broke
  again on three Phase 2 files, and every remaining phase touches those attributes. Fixed
  structurally instead: the two colliding spellings (`METADATA.ome.xml`, the NGFF-mandated
  filename, and `PhenotypicAttr.METADATA`) are stripped by **context**. The allowlist
  **shrank**, and the gate's own staleness test is what flagged the now-redundant entries.
  Teeth re-proved by planting genuine legacy tokens in live source — all still caught.

## A claim of mine the gate refuted

I committed a note asserting `valid_staged_store` **requires** `detect_mat`, so a delta
store "is by definition not a staged store". **False**, disproved by execution: a
self-consistent delta store declaring only `gray` returns `True`.

Spec §3.6 settles which side is wrong — the predicate checks "every entry in
`phenotypic.series` **and** `phenotypic.labels`", i.e. whatever the store **declares**, and
"(objmap included — §3.3 guarantees it exists after Stage 1)" is a **writer** guarantee, not
a check. So the implementation is spec-correct and must **not** be tightened; only my
justification was wrong. **Phase 3's resume classification rests on this predicate**: it
catches a store that declares a member and lacks it, not one that never declared it.

## Method notes

**The first complete wide run happened here** — 8,646 passed, 6 failed, covering
`smoke`/`gui`/`integration` for the first time. Three failures were a missing `topology`
extra, invisible to every previous run because none included `tests/smoke`. **A gate that
has never run is not a gate.**

**Two duplications, both built in the same phase by different agents**:
`build_phenotypic_attributes(grid=)` had zero production callers while `ImageGridHandler`
injected the key post-hoc, and `_preview_cache._load_image_auto` was a hand-rolled copy of
`load_image_from_store`. Consolidating the second **survived its own mutation** at first —
the test compared `detect_mat` arrays, and losing the grid changes no pixels.

---

# IMPLEMENTATION — Phase 3 complete

Eight tasks, six clusters (C8–C13). The phase's defining failure was not in any single
task: it was **the tree being broken between them**, in a way every task's own tests
called green.

## The between-tasks break, hit three times by three clusters

Task 3.3 stopped Stage 1 writing `.h5`. Task 3.8, which owns per-image completion
markers, is **three clusters later**. In the window between them the marker still
declared `"hdf": results/<ds>/hdf/<stem>.h5`, and `publish_image_success` resolves every
artifact `strict=True` — so an image **failed after completing all of its work**:

```
…|img.tiff|started||||stage3
…|img.tiff|failed|FileNotFoundError: … '/out/results/ds/hdf/img.h5'|||stage3
```

C10 closed it for the staged engine, C11 for the standalone `phenotypic-process-single`
SLURM worker, C13 for the legacy promoter. **Three clusters, one defect class, each
discovered by executing rather than by reading.** The plan's task ordering was sound as a
dependency graph and wrong as an execution order; nothing in the review rounds caught it,
because each task is internally consistent.

The fix that generalizes is `image_data_artifact()` — one helper returning
`(kind, path)` — so a publisher **cannot** name the wrong artifact. Choosing
`<store>/zarr.json` initially was what kept it minimal: a regular file, so the existing
`{"size","sha256"}` descriptor kept working with no version bump, deferring the design
question to the task that owned it.

## Count publishers, not files

Every cluster found the site count wrong, always in the direction of **more**:

- C11: **8** `store_stem` sites where the plan said 5 — **and 3 of the plan's 5 were
  actively wrong.** `store_stem` *raises* on a non-store path by design, so following the
  plan literally would have turned every local forward run into a hard `ValueError`.
- C11 again: `_cli_process_single.py` has **two** publishers. C10 routed one. The plan
  named the file, so the file looked done.
- C12: **3** write sites, not the 4 my brief claimed — I had copied the plan's Files
  list, which includes a *transport* site. The plan's own corrected blockquote said 3.
- C13: **6** `publish_image_success` call sites, 5 declaring a data artifact, 1 left.

## Two tests that passed for the wrong reason

Both are the same shape — an assertion satisfied by something other than the behaviour
under test — and both were found only by running the *red* step honestly.

- **C12:** the plan's rejection test asserted `"--durable-writes" in result.output`. That
  passed **before the option existed**, satisfied by click's own
  `No such option: --durable-writes`. Five of nineteen "failures" in the red run were
  vacuous passes.
- **C13:** Task 3.4's parity fixture described only the parquet, so branch 1 never met a
  store descriptor. Measured: with `valid_image_success` forced to require `is_file()` on
  every descriptor, **385 passed** — the check was completely blind. C13 fixed the
  *fixture*, not the check; the defect now produces 23 failures.

A red run is evidence only if it fails for the stated reason. "It failed" is not the
observation; "it failed **with** X" is.

## Line numbers in a plan are stale by the time they are read

Task 3.8's target moved **400 → 405 → 426** across two clusters, as neighbouring tasks
edited the same file. Plans that address code by line number are self-invalidating
wherever tasks share a file; the plan now addresses it by symbol.

## Claims of mine the execution refuted

- I wrote that the legacy promoter's `resolve(strict=True)` **raises**. It does — and the
  loop's `except (OSError, RuntimeError, ValueError): continue` swallows it. The symptom
  is a **silent refusal to promote** and a full reprocess of the tree, which is worse than
  what I described and looks like nothing.
- I told the user the missing `store_schema_version` gate on `load_layer_zarr` was a
  performance question. It is not: that method **already** reads the root `zarr.json` to
  resolve the layer, so the gate is a dict lookup on a block in hand. The premise was
  wrong, not the concern — and it was the entry point that mattered most, being the GUI
  tile server's, called per tile.
- My brief's "four write sites", above.

## A gate that had never run, and had rotted

`tests/migration` is **not in `testpaths`**, so nothing runs it. 57 of its 341 tests fail
— **stale goldens, not drift**: they expect `LogGrowthModel_r` while the code emits
`LogGrowthModel_Area_r`, because analysis headers became metric-qualified in `67cfa259`
and the goldens were never re-captured. The pydantic-v2 regression detector has been
silently dead for some time.

Adding the bare directory to the phase gate would have made the gate **permanently red
for unrelated reasons**, which is worse than not running it — a gate that is always red
teaches you to ignore it. The two files Phase 3 actually needs
(`test_metadata_schema_migration.py`, `test_curation_imagefile_rename.py`) are named
explicitly; 52 passed.

## Method notes

**An equivalent mutant is a real answer, not a gap.** C13's M10 (dropping a
`root_json.is_file()` guard) survived because the broad `except OSError` converts the
resulting error to the same `False`. No test can distinguish it. The guard stayed anyway —
control flow should not lean on a broad except.

**Declining to write a test can be correct.** A store descriptor covers only the root
`zarr.json`, so a re-promote with identical root and different chunks would still
validate. C13 did not pin that, because a test asserting it would cement the weakness as
*intended* and block a future strengthening. Flagged instead.

---

# IMPLEMENTATION — Phase 4 complete

Four tasks, one agent, nine commits, run in a parallel worktree beside Phase 5. Gate:
**4 failed, 9,328 passed** — the identical four known failures as Phase 3, zero new.
24 mutants, 5 survivors, all closed, plus one reported rather than manufactured.

## The lesson this phase supplied three times: an assertion about results cannot see cost

This is now the project's most frequently re-learned failure, and Phase 4 hit it on three
unrelated code paths:

- **M5** — a bounded glob replaced by `rglob("*")`. The `(store/zarr.json).is_file()` filter
  prunes the recursive walk **back to the identical result set**, so every result-set
  assertion passes while the walk is pathological.
- **N7/N8** — the pyramid loader ignoring `target_px`. Its only caller asks for the level-0
  edge, so level 0 *is* the correct answer everywhere; the pyramid could have been doing
  nothing at all and no test would have noticed.
- **R2**, the sharpest — a recursive `rglob` that filters its results down to
  `<store>` + `<store>/zarr.json`, producing a **byte-identical** inventory while doing
  every bit of the work the bound exists to avoid. It passed every result-set assertion.
  There is no result-level observation that could distinguish it. **Only counting
  `os.scandir` caught it.**

When a change's purpose is cost, the test must measure cost. The plan's own criterion said
this for tiles ("assert on bytes read, not the level index"); it took three survivors to
learn it applies to discovery and caching too.

## And its converse, from the FLOW-5 guard: a behaviour-preserving refactor is a mutant too

Writing `test_nothing_writes_into_a_promoted_store`, I checked whether it was redundant with
the three existing promote tests by applying `os.replace` -> `os.rename` — a change that
alters **nothing** observable here. All three existing tests went red as false alarms,
because they mock `os.replace` and so pin the *mechanism*. The new test, which asserts inode
identity, correctly stayed green.

**Every mutant run in this project so far has changed behaviour**, so nothing in any suite
would have caught a mechanism-pinning test. Any test that fails a behaviour-preserving
refactor is testing the implementation, not the outcome.

## The plan pointed at dead code, and the real cost site was unowned

Task 4.1 named `_processing_snapshot_paths` as feeding `source_fingerprint`. It has **zero
production callers**. The live producer is `_scan_processing_inventory`'s
`results_root.rglob("*")`, which **no Phase 4 task owned** — so the task's headline goal
("discover stores without walking into them") was not achievable within its own scope.

Measured at realistic plate size: a 4000x3000 store is **58 entries** (36 files, 22 dirs —
the file count stays low because the shard policy works). At 10k images that is **580,000
stat calls versus 10,000**, every time the viewer opens.

**User ruling (2026-08-20): bound the results walk to `<store>/zarr.json`.** It detects every
PhenoTypic write, because the promote writes the root last and nothing writes into a
promoted store; it deliberately stops detecting out-of-contract modification (a hand-edited
chunk, a store rsynced mid-flight). Consistent with Task 3.8, which fingerprints a completion
marker's store by its root alone for the same reason — two subsystems now answer "did this
store change?" identically. The `ProcessingInventoryAssurance` enum was **not** extended;
`"exhaustive"` keeps its name and simply stops descending.

## A live bug the phase found, and a fixture that could not see it

`_load_zarr_level_rgb`'s LRU key did not move on a byte-identical republish, so the DZI
source PNG was regenerated **from the previous publish's decoded array** — new pixels on
disk, old pixels served. The token now carries bytes **and** `st_mtime_ns`.

Two of the tests written to catch it first failed for the wrong reason, both caught by
reading the failure text rather than the exit code:

1. It failed with **409**, not stale pixels — which is correct behaviour, since a republish
   stales the viewer binding. Restructured into three tests separating promote, rebind, and
   rebind-alone; without the third, the first would pass against an implementation that
   regenerates on every Refresh.
2. Then it failed on **byte-identical PNGs**, because `skimage.color.label2rgb` colours by
   the **rank** of the labels present — so republishing with label value 7 instead of 1
   renders identically. The republish was invisible to the *renderer*, not to the cache.
   This is Phase 2's non-discriminating-fixture lesson in a new costume.

## Built and deliberately uncalled

`select_pyramid_level` is correct and tested but has no in-app caller. Its only candidate
feeds the DZI source PNG, and capping that caps OpenSeadragon's maximum zoom — a
user-visible regression, not an optimisation. The pyramid is not speculative work
regardless: napari, QuPath, and Vizarr read those levels directly, which is a headline spec
goal rather than a side effect.

## Method note

Parallel worktrees held cleanly. Across nine commits the only file Phase 4 touched outside
`src/phenotypic/gui/` and `tests/**/gui/` was its own plan document. Boundaries stated as
paths, and reported rather than resolved when they chafed, are what made two phases
concurrent without a single conflict.

---

# IMPLEMENTATION — Phase 5 complete

Six live tasks (5.5 cut), seven commits, run in a parallel worktree beside Phase 4.
**57 mutants, 4 surviving** — all four reported rather than papered over. Six more
initially survived, and **every one of them exposed a real test defect** rather than a
code defect.

## The one that mattered most: a fidelity suite that could not see data loss

Dropping an **entire series** from the converted store **survived**. Every fidelity test
compared the converted store against a **freshly written** one — both through `save2zarr` —
so the writer dropped the series on both sides and they compared equal.

This is the single most consequential property in the phase (does migration lose data?) and
it was guarded by a test that structurally could not fail. Closed with an **h5py oracle**
that reads the legacy source file directly, so the comparison crosses the format boundary
instead of staying inside the new writer.

It is Phase 2's non-discriminating-fixture lesson again, and the generalization is now
unavoidable: **a round-trip test proves only that the round trip is self-consistent.** To
prove *fidelity* you need an oracle outside the code under test.

Adjacent, same phase: dropping the stored `grid_finder` also survived, because
`grid_finder is not None` can never fail — `GridImage.__init__` mints a default one. The
fixture now carries a different class with a non-default parameter.

## A fixture whose "legacy" state never existed, which certified the wrong conclusion

`finished_legacy_run` minted success markers at the **current** version over `.h5`
artifacts. No such tree has ever existed. Because `keep_source=True` leaves the `.h5` in
place, such a marker keeps validating straight through a migration — so Task 5.2's
aggregate test **XPASSed**, certifying that Task 5.6 (marker republication) was
unnecessary. Real legacy markers are version 1 with no `kind` tag.

A fixture that encodes an impossible state does not merely fail to catch bugs; it can
actively argue for deleting the code that fixes them.

## Four pass-1 mutations that survived for one shared reason

Unfiltered scope, receipt `kinds`, `_validate_receipt` re-derivation, and the dry-run guard
all survived. Root cause was singular: the `legacy_run` fixture's non-image targets were
**already canonical**, so pass 1 short-circuited to `_compatible_result` before writing a
receipt — **nothing about pass 1 was exercised at all.** Skipping pass 1 entirely also
survived, since no test asserted `headers_migrated > 0` or pass 1's effect on bytes.

Four "missing tests" that were really one dead fixture. When a cluster of mutants survives
together, look for the shared upstream cause before writing four tests.

## Two false greens, both the shape Phase 3 named

- `CliRunner().invoke(cli, ["--mode", "migrate", "--help"])` exits **0 before `--mode` is
  validated** — `--help` is eager. The test passed while `migrate` was still absent from the
  `Choice`. Now inspects `mode.type.choices` directly.
- `test_migration_never_rewrites_a_source_hdf` and `test_dry_run_writes_no_receipt` both
  passed against a CLI that **exited 2 and never ran**. Both now assert `exit_code == 0`
  first.

Also unsatisfiable as written: the plan asserted `exit_code != 2` to prove a rejection came
from the mode guard rather than click — but `click.UsageError` **is** exit 2, so the
assertion can never distinguish them. Discriminated on the guard's message instead.

## `.gitignore` silently excluded the golden fixtures

`.gitignore:85 *.h5` excluded all six legacy fixtures from Task 5.1's commit. `git add <dir>`
skips ignored files **with no error**, so the suite was green locally and would have failed
to collect on a fresh clone. Negated explicitly.

Worth generalizing: a test fixture that is also a build artifact by extension is invisible
to the one check (`git status`) that would otherwise catch it.

## Four surviving mutants, and why they are honest

`republish_aggregate` removed entirely, and called before the markers instead of after, both
survive — because **every input** to `current_aggregate_is_current` is unchanged by
migration. Pass 1's targets are the pipeline config and per-dataset parquets, none of which
the aggregate marker binds, and the snapshot is untouched. So the original aggregate is
still current and republication is a no-op on every reachable tree. It is defence-in-depth
and the plan mandates it, so it stays; a test could only pin `publication_id` changing,
which is mechanism rather than an observable property.

Likewise `publish_aggregate_snapshot`'s `try/except` removal survives because the
`success_markers_required` guard above makes the raising path unreachable — removing **both**
is caught. And ignoring the `.h5`'s `/grid/` `nrows`/`ncols` attrs survives because the
legacy format stores that geometry **twice**, in the attrs and inside `grid_finder_json`,
and `setdefault` lets the finder win. Format redundancy, not data loss.

## Documentation that instructed the withdrawn design

Root `CLAUDE.md` still told a future agent that `--mode migrate` rewrites
`deliverables/metadata.csv` after copying the original to `metadata.original.csv` — the D9 /
FLOW-4 carve-out that was **withdrawn and never implemented**, and the exact rewrite the
rule above it forbids. Phase 5 found it, correctly left it alone as outside its Files list,
and reported it. Now corrected: the rule has **no exception**, and migrate emits
`metadata.canonical.csv` alongside a byte-identical snapshot.

## Method note

Parallel worktrees held. Phase 5's only GUI file across the whole phase was
`results_viewer/_output_consistency.py`, exactly as scoped; Phase 4's only non-GUI file was
its own plan document. Two agents, two phases, ~2,400 lines of plan, **zero conflicts** —
boundaries stated as paths, and reported rather than resolved when they chafed.
