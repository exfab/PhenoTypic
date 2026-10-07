# Phase 1 review — ReferenceContext, RefMetadata, SubtractBlank

**Scope:** `git diff 47e55a683..c6eff4dac -- src tests` (d7476d9a, 4d16c200, c6eff4da),
against spec `docs/superpowers/specs/2026-10-05-ref-metadata-ops/design.md` §4/§6/§8/§9
and plan Tasks 1–3.
**Evidence:** static reading, plus two probe scripts the orchestrator ran
(`.scratch/refmeta_probe.py`, `.scratch/refmeta_probe2.py`; output quoted as
P*/Q*) and five mutation experiments run by the orchestrator (M1–M5). For each
mutation the orchestrator restored the file and verified its sha256; the full-tree
snapshot matched afterwards.

## Verdict: **fix first**

The core API is well built. Lookup grain, string reads, dataset narrowing, error
passthrough, and the corrector scan across applications and nested branches all
hold up under probing. Three paths still produce a **plausible array with no error**,
which success criterion 5 (§8, "no wrong-metadata case fails silently") forbids:

- a single-channel (integer) target, F1;
- a target whose colour configuration differs from the blank's, F2;
- a corrector recorded in a journal whose class this process has not imported, F3.

A fourth issue, F4, makes the resolver O(N²) in the input directory, and the
Phase 2 planner as sketched would hit that at ucr_033 scale. Five tests are false
greens: all five mutations M1–M5 survived.

---

## Findings (most severe first)

### F1 — major (silent wrong answer): integer `detect_mat` from single-channel inputs saturates or wraps

- **Where:** `src/phenotypic/enhance/_subtract_blank.py:281-289`. The root cause is
  outside this diff, in `_core/_image_parts/detection_modes/_gray_mode.py`
  (`compute` returns `image._data.gray.copy()`) and
  `_core/_image_parts/_image_data_manager.py:389,420,430`.
- **Scenario.** A 2-D integer image keeps its raw integer `gray`, so its
  `detect_mat` is **uint8 0–255** (or uint16 0–65535), not float32 in [0, 1].
  - Q3: `bit_depth=8 gray.max=100.0 detect_mat.max=100.0 dtype=uint8`.
  - Q5: uint16 gives `detect_mat.max=30000.0`.
  - Q4: target and blank both uint8 single-channel, blank = 100, faint growth = 110,
    strong growth = 200. SubtractBlank returns `faint=1.0000 strong=1.0000 agar=0.0000`.
    Every pixel brighter than agar becomes 1.0, so faint and strong growth are
    indistinguishable. In uint8 arithmetic `t - b` also **wraps**, so with
    `polarity="darker"` a pixel 10 levels brighter than the blank becomes 246 and
    clips to 1.0, which reads as a dark colony.
  - P14: a single-channel target with an RGB blank produces
    `max=207.000 mean=148.637`. The output is far outside [0, 1] because `"both"` is
    unclipped. The bit-depth check passes in every one of these cases.
- **Real input or test artefact?** **Real input.** `Image.imread` of any
  single-channel TIFF/PNG goes through `ski.io.imread` → `cls(arr=arr, …)`
  (`_image_io_handler.py:786,796`) → `_handle_array_input`. That function converts
  only **3-D float** arrays (`_image_data_manager.py:291-294`), so a 2-D integer
  scan arrives exactly as in Q3. The CLI reads inputs the same way.
- **Pre-existing core issue? Yes; report it separately.** `DetectionMode.compute`
  documents "a 2-D float32 array normalised to [0, 1]" (`_detection_mode.py`), and
  `GrayDetectionMode.compute` violates that for integer gray.
  `SubtractGaussian._operate` (`enhance/_subtract_gaussian.py:94-104`) has the same
  defect. `skimage.filters.gaussian` rescales a uint8 input to [0, 1] unless
  `preserve_range=True`, so it subtracts a [0, 1] background from a 0–255
  `detect_mat` and clips to [0, 1]. The output is near-binary there too. The proper
  fix normalises in the gray mode (by bit depth) and belongs to a separate change
  with its own regression surface.
- **Fix in SubtractBlank now:** refuse rather than compute until the core is fixed.
  - Raise `ReferenceImageError` when
    `not np.issubdtype(image.detect_mat[:].dtype, np.floating)` (or normalise both
    arrays by `2**bit_depth - 1` explicitly).
  - Refuse when `image.rgb.isempty() != blank.rgb.isempty()`. A single-channel
    scan's gray and an RGB frame's `rgb2gray` are not the same quantity (P14).
  - Clip `"both"` as well: `np.clip(np.abs(t - b), 0, 1)`.
  - Add a single-channel uint8 pair test and a mixed single-channel/RGB refusal test.

### F2 — major (latent silent wrong answer, currently a misleading refusal): the blank is projected with its own colour configuration, not the target's

- **Where:** `_subtract_blank.py:281`, `background = mode.compute(blank)`.
- **Scenario.** The Lab modes compute from `image.gamma`, `image.illuminant` and
  `image._observer` (`_lab_channel_modes.py`, `compute_from_rgb`). The blank is read
  through `Image.imread(path, **read_kwargs)` with defaults (D65, sRGB). A target
  built or loaded with `illuminant="D50"` or `gamma=None` is therefore compared
  against a blank projected differently.
  - Q2 (`inplace=True`, identical pixels, LabL, D50 target vs D65 blank):
    max = **0.0190**, where the correct answer is 0.0.
  - That bias is ~1.9 L\*, enough to shift a faint-mycelium threshold.
- **Why it currently shows up as a refusal:** a pre-existing core bug.
  `Image.copy()` (`_image_handler.py:704`, `self.__class__(self)`) **drops
  `illuminant`/`gamma`**. Q1: `orig=(D50,GammaEncoding_Linear) copy=(D65,GammaEncoding_sRGB)`.
  `op.apply(image)` (inplace=False) copies first, so the freshness check recomputes
  LabL under D65 against a `detect_mat` computed under D50 and fails. P1/P1b raise
  `StaleDetectMatError: detect_mat was already enhanced`. That message is false:
  nothing was enhanced. Once the copy bug is fixed, F2 turns silent.
- **Fix:**
  - Project the blank through the target's configuration:
    `mode.compute_from_rgb(blank.rgb.normed(), image=image) if mode.requires_rgb else mode.compute(blank)`.
    This is exactly what `compute_from_rgb` exists for. Alternatively, refuse when
    `(gamma, illuminant, _observer)` differ.
  - Add a D50 test with `inplace=True`.
  - **Report separately:** `Image.copy()` loses colour configuration, which affects
    every Lab/XYZ operation applied with `inplace=False` on a non-default image
    (`_image_color_handler.py:89-93` sets the defaults, and the copy constructor
    never carries them over).

### F3 — major (silent wrong answer): the corrector guard only sees correctors whose module happens to be imported

- **Where:** `_subtract_blank.py:152-162` (`_corrector_class_names` walks
  `ImageCorrector.__subclasses__()`) and `:305-314`.
- **Scenario:** P5 is the fresh-process run. After
  `from phenotypic.enhance import SubtractBlank` and a pipeline apply, `phenotypic.correction`
  **is not in `sys.modules`** (`False []`). Two facts combine here:
  - Reading a PhenoTypic store keeps its full journal (root `CLAUDE.md`, "Provenance
    is cumulative"), so a `--mode process --layer rgb` store whose history ran
    `ColorCorrector` / `CalibrateColorRpcc` / `ColorDenoise` / a wavelet corrector
    keeps that record.
  - A notebook then runs `ImagePipeline(ops={"sb": SubtractBlank(), …})` on it. That
    pipeline imports no correction class.

  The record's `operation_class` matches nothing, the shapes are equal, and the raw
  blank is subtracted from a colour-corrected frame with no error. The code comment
  ("every class the running pipeline uses is imported") is true but answers the
  wrong question: the history comes from an *earlier* pipeline.
- **Fix:** resolve each record's class from its `operation_class` string
  (`importlib.import_module(mod)`, then walk `qualname`) and test
  `issubclass(cls, ImageCorrector)`. If a recorded class cannot be resolved, refuse:
  the guard cannot vouch for a history it cannot read. A cheaper alternative is
  `import phenotypic.correction` inside `_corrector_class_names`, but that still
  misses third-party correctors. M3 survived as well (see False greens), so add a
  grandchild-subclass test.

### F4 — major (performance; blocks Phase 2 at scale): `resolve_image` rescans the whole root on every call, three times per apply

- **Where:** `_reference_context.py:398-400`. `root.iterdir()` runs with `is_image(p)`,
  which issues one or two `stat` calls per entry, *before* the cheap stem
  comparison. Per SubtractBlank apply the scan runs three times:
  - `_subtract_blank.py:257` (self-reference check);
  - `_reference_context.py:409`, via `load_image`;
  - again via `reference_image_digest`, called from `_ref_metadata.py:122`.
- **Scenario:** the Phase 2 planner sketch (plan Task 4, `plan_references`) calls
  `scoped.resolve_image(name)` once per input image. With 34,500 inputs in one
  dataset directory that is ~1.2×10⁹ `stat`s at startup, and the same again in the
  preflight. On GPFS that is hours. The workers avoid the scan only because the
  manifest supplies an `images=` map.
- **Fix:**
  - Build a per-root index once: `{source_image_stem: [paths]}` from one
    `os.scandir`, using `DirEntry.is_file()`/`is_dir()`, which on Linux usually need
    no extra stat. Store it on `_SharedTable` (shared by `narrow`), keyed by the root.
  - Have `_ref_image` call `ctx._load(name)` once and take both the image and the
    digest from it (see F10).
  - Optionally cache the blank's projected channel per (cache key, mode, colour
    configuration). Today `mode.compute(blank)`, a full Lab conversion, is redone
    for every frame of a plate. The freshness check adds one more full-frame
    projection per image.

### F5 — minor: the resolver counts non-image siblings and accepts paths outside `image_root`

- P12: `t00.tif` plus a `t00.json` sidecar is refused with "matches 2 files". The
  failure is loud but a nuisance on real acquisition directories (`.xmp`, `.json`,
  `.txt` sidecars). Fix: filter candidates to `IO.ACCEPTED_FILE_EXTENSIONS`
  (`sdk_/constants_.py:96`) or Zarr stores.
- P13: `root / name` with an absolute name resolves to a file **outside** `image_root`;
  `../x.tif` does the same. The spec (§4.1, §5.2) scopes resolution to the
  dataset's directory, and the Phase 2 manifest assumes it. Fix: refuse names that
  are absolute or contain a path separator, with `ReferenceImageError`.

### F6 — minor: duplicate or key columns crash `lookup` with an untyped polars error

- P2 asks for `["BlankImage", "Metadata_BlankImage"]` and gets
  `polars.exceptions.DuplicateError`. P2b asks for `["Metadata_ImageName"]` and gets
  the same error.
- Inside an op, that error is double-wrapped as a generic failure. Phase 2's
  `_union(pipeline.reference_columns())` will reach it whenever two ops spell one
  column differently.
- Fix: in `lookup`, dedupe the resolved names before `_index`
  (`tuple(dict.fromkeys(resolved))`), and serve key columns from the key itself
  rather than from `agg`.

### F7 — minor: empty and whitespace values pass as real values

- P7: a pandas frame containing `""` returns `{'Metadata_BlankImage': ''}`. P7b: a
  CSV cell containing `" "` returns `' '`.
- For a `RefImageColumn` the failure arrives late and is misnamed ("matches 0
  files"). For a plain `RefColumn`, `''` reaches the op silently.
- Fix: strip string values in `_read_table`, map `""` to null so that lookup reports
  `reason="null"`, and add a test for each.

### F8 — minor: gaps in self-reference detection

`_subtract_blank.py:257-267` misses two cases:

- an `images=` entry that is an in-memory `Image` of the target itself under another
  name (only the name is compared; the copy defeats identity);
- a case-insensitive filesystem (macOS) where `T04.tif` resolves to `t04.tif` while
  `source_image_stem` returns `T04`.

Fix: also compare `target_file.name == image.name` for in-memory entries. For paths,
compare `os.path.normcase` stems, or use `os.path.samefile` when the image records
its source path.

### F9 — minor: `_RESOLVED` entries leak after failed applies, and `id()` can be reused

- P11: `before=2 after=3`. Each apply that fails after `_ref_values` leaves its
  record behind.
- If the op is collected and a new `RefMetadata` op receives the same `id()`, and
  that op's `_operate` succeeds without calling `_ref_values` (allowed for a
  subclass that skips conditionally), it inherits the dead op's `_references`.
- Fix: in `RefMetadata.__init_subclass__`, wrap `cls._operate` so the `id(self)`
  entry is popped on exception. An override of `apply` would not work, because
  `RefMetadata` sits last in the MRO.

### F10 — minor: cache staleness for store directories, and a double `_load`

- The cache key uses the store directory's `st_mtime_ns` and `st_size`
  (`_reference_context.py:412-418`). These do not change when chunks deeper in the
  store are rewritten, so a long-lived GUI session can serve a stale blank.
- `_ref_image` loads twice: once through `load_image` and once through
  `reference_image_digest`. If the entry is evicted between the two calls (2-entry
  LRU, GUI threads), the image is re-read. If the file changed in between, the digest
  recorded is not the digest of the pixels that were used.
- Fix: one `_load` per `_ref_image`. For stores, add the root `zarr.json`'s
  mtime/size to the key.

### F11 — minor (Phase 3 hazard): one instance's token stack is not thread-safe

`self._tokens` is a plain list (`_reference_context.py:249-257`). The spec documents
"do not enter one instance from two threads". The risk is Task 8: if the builder
caches one context per session and enters it from Werkzeug request threads, one
thread pops another thread's token. The result is
`ValueError: Token was created in a different Context`, or the wrong context stays
active. Either enter a fresh `ctx.narrow()` per request in Task 8 (it shares the
parsed table and indexes), or keep the token stack in a `ContextVar` or
`threading.local`.

### F12 — minor (Phase 2 note): the measurement join and the reference lookup read the same table differently

`_cli/_metadata_join.py:150` reads with type inference (`infer_schema_length=None`);
`ReferenceContext` reads every value as a string. The two therefore **can** disagree,
despite D5's "cannot disagree":

- a digit-only blank stem `000100` is used correctly for subtraction, but joins
  onto `measurements.csv` as `100`, so the joined `Metadata_BlankImage` §6 promises
  does not name the file;
- digit-only image names do not join at all.

This is a pre-existing join behaviour; Phase 2 should at least document it.

### F13 — minor (accepted limitation; document it): a store's identity digest covers only its root `zarr.json`

`reference_file_digest` (`_reference_context.py:103-110`) hashes only the root
`zarr.json`. For a third-party OME-Zarr blank, a pixel edit leaves the digest
unchanged, so the Phase 2 work-id would reuse stale results. PhenoTypic stores
rewrite `zarr.json` with their provenance, so they are safe. State the limitation in
the how-to.

---

## False-green tests (mutation survived, or no assertion on the behaviour named)

| # | Test | Surviving mutation / gap | Fix |
|---|---|---|---|
| M1 | `tests/unit/tune/test_reference_refusal.py` (2 passed) | `_spec.py:305` → `pass`: the `TuningSpec` validator is never exercised; tests call the helper directly | Build a `TuningSpec` (or load a spec JSON) with a reference pipeline and expect `ValidationError` naming the op path |
| M2 | `test_ref_metadata.py` + `test_subtract_blank.py` (28 passed) | `_ref_metadata.py:122` → `pass`: `_references["images"]` (blank name and digest) is asserted **nowhere**, so success criterion 4's "which blank it used" is untested; `table_sha256` is never asserted on a path-backed table | With a file blank under `image_root` and a CSV table, assert `_references == {"table_sha256": sha(csv), "values": …, "images": {"t00": {"sha256": sha(tif)}}}` |
| M3 | `test_subtract_blank.py` (20 passed) | `_subtract_blank.py:30` (drop `stack.append(sub)`): recursion untested, so grandchild correctors (`GridCorrector` subclasses) are unguarded | Add a `class _G(_NoopCorrector)` refusal test |
| M4 | `test_reference_context.py` (22 passed) | `read_kwargs` removed from the cache key: untested | Load the same file under two `read_kwargs` and assert two reads |
| M5 | `test_pipeline_reports_reference_columns_tree_wide` (8 passed) | path → `path[0]`: `startswith("comp")` cannot fail | `assert found == {"comp/ops[0]": (...)}`; the CLI preflight messages depend on the exact spelling |
| — | `test_runs_inside_a_branch_pipeline_and_composites`, `test_runs_after_set_detect_mode_following_an_enhancer` | No output assertion: an implementation whose subtraction is lost on these paths still passes | Assert `detect_mat` equals the known clipped difference (normalisation by `CompositeEnhance` aside, assert at least `dm[1,1] > dm[0,0] == 0`) |
| — | `test_rgb_and_gray_are_untouched` | Uses a single-channel float image, so `rgb` is never checked | Use an RGB target and assert `rgb` and `gray` are unchanged |
| — | `test_load_image_reads_each_file_once_and_rereads_on_change` | Digest checked only for `is not None` | Compare it to `hashlib.sha256(blank_path.read_bytes())` |
| — | `tests/unit/enhance/test_detect_mat_invariant.py` gate | The blank is identical to the target, so the output is all zeros and the range gate is trivially satisfied | Acceptable as a gate; real range coverage comes from the F1 tests |

**Spec §9 items with no test:**

- parquet input, and the unsupported-suffix error (`_reference_context.py:154-159`);
- LRU eviction bound;
- a `RefMetadata` subclass overriding `_ref_columns`;
- single-channel integer targets (F1);
- non-default colour configuration (F2).

## Verified OK

- **Lookup grain:** a per-colony table collapses to one value; null, disagreeing and
  missing rows raise with the right `reason`. `infer_schema=False` keeps `000123`.
  Bare headers resolve. With two datasets and no `Metadata_Dataset` column the lookup
  is *ambiguous*, not first-row.
- **BOM:** an Excel CSV with a UTF-8 BOM works (P6).
- **Default name:** `Image.name` defaults to the UUID, never `None`, so an unnamed
  image cannot match null-key rows (`_image_handler.py:194-197`).
- **Provenance timing:** `provenance_parameters` is called after `_operate` succeeds
  (`_provenance.py:682-700, 973-977`). Recorded values are per image across a reused
  pipeline (P10: `t04→a0`, `t05→b0`).
- **Mixin guard:** the `RefMetadata.__init_subclass__` guard fires whichever side of
  the bases the mixin sits on (P3, P3b).
- **Corrector scan, positions covered:** it catches correctors inside nested branch
  pipelines and inside a `CompositeDetector` branch pipeline (P9a; Q6 chain
  `RuntimeError → Exception → RuntimeError → StaleDetectMatError`, with a passing
  control Q7). The scan covers every application, both journal schemas, via
  `_operations`.
- **Error wrapping:** `ReferenceContextError` passes through `ImageOperation.apply`
  as the commit message says, and the nesting chain matches the docstring. No
  caller catches `ValueError` around `apply` in a way that would swallow it.
  `GridApply` re-wraps loudly.
- **Freshness check:** it is exact for default colour configuration. `reset()`
  assigns `mode.compute(image)`, and copy preserves `detect_mat` bit for bit.
- **Self-reference by extension:** the extension form is refused, including the
  planner's mirror rule.
- **Activation:** `ContextVar` activation restores after an exception, and nesting
  replaces then restores.
- **Imports:** `ReferenceContext` is lazily exported.
- **Neutral side effect:** `_metadata_migration._is_column_reference_field` now
  treats `RefColumn` values as column names and may re-prefix them. This is harmless,
  because `lookup` resolves both spellings.
- **GUI:** `_schema_cache.columns_for("reference_metadata")` logs and returns `[]`
  until Task 8, so it does not crash.
