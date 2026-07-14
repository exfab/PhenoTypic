# A10 FilFinder G0 reference-gate report

## Status

**CORRECTED SUCCESSOR READY FOR INDEPENDENT REVIEW. The prior `b224fb59...` sign-off is invalid.
Production remains blocked until an independent reviewer returns an explicit G0 PASS on the exact
successor evidence commit.** No dependency file, export, typing alias, registry, GUI discovery, or
serialization entry is changed by this gate.

## Frozen authorities

- Official PyPI release `fil-finder==1.8`, published 2025-05-12 and still listed as latest stable
  when checked 2026-07-13. Artifact URLs and hashes are in `PROVENANCE.json`.
- Annotated Git tag `v1.8`, peeled commit
  `22539cf2176ad9b717658652e8da749158597f4d`.
- Fifty-nine locally retained source-distribution files with canonical aggregate SHA-256
  `73db4ddb96269a1602a66f1afdc9d6b036faf79f3d374861cf4544761a590174`.
- Koch and Rosolowsky 2015 paper for algorithm context, with exact executable behavior controlled
  by the maintained source. Paper workflow evidence is
  `paper/Koch_Rosolowsky_2015.txt:326-424`.
- MIT license and attribution at `upstream/LICENSE.rst:1-21` and
  `upstream/PKG-INFO:1-29`.

## Load-bearing source findings

1. A supplied mask with `create_mask(use_existing_mask=True)` skips flattening and adaptive
   segmentation (`upstream/fil_finder/filfinder2D.py:299-325`). The wrapper owns only the inclusive
   `detect_mat >= threshold` boundary.
2. `medskel(rng=seed)` forwards deterministic tie handling to scikit-image and exposes both
   skeleton and distance (`upstream/fil_finder/filfinder2D.py:524-573`).
3. The apparent skeleton-length parameter is not a useful public tuning dimension in 1.8. Source
   code computes `min(ceil(skel_thresh), 1 pix)`, capping every positive input at one pixel
   (`upstream/fil_finder/filfinder2D.py:658-669`). The corrected contract freezes exactly one pixel,
   removes the proposed field, and does not compensate or reimplement the upstream behavior.
4. The source separately assembles post-prune full skeleton and longest-path skeleton
   (`upstream/fil_finder/filfinder2D.py:740-753`). The wrapper selects the exact attribute named by
   its output field.
5. The source creates an implicit reusable/process executor when no pool is provided
   (`upstream/fil_finder/filfinder2D.py:167-175`). The wrapper instead owns one fresh one-process
   pool per apply and guarantees `shutdown(wait=True)` on both success and exception.
6. In the locked NetworkX 3.6 environment, every real worker task emits `Graph pruning reached max
   iterations.` on the nonempty fixture cases from
   `upstream/fil_finder/filament.py:331-361`. The loop/branch task also emits 352 deprecations from
   each `numpy.in1d` call site at `upstream/fil_finder/pixel_ident.py:831,861`. These are upstream
   behaviors and remain visible. Only the exact supplied-mask warning is adapter-suppressed.

## Golden evidence

### Precision-seam correction

G2 real-runtime tests proved that the schema-v2 oracle was not reachable through PhenoTypic:
`ImageData` coerces floating `detect_mat` arrays to `float32` at
`src/phenotypic/_core/_image_parts/_image_data_manager.py:24-57`, while that oracle passed native
float64 synthetic values directly to FilFinder. The nominal float64 value immediately below 0.5
rounded to 0.5 at the real image seam, and float32 intensity quantization changed two pixels in
the Y-spur/disconnected longest paths. This was evidence of a mismatched oracle, not an acceptable
tolerance.

Schema v3 first applies the exact float32 ImageData coercion, then copies those values into the
adapter's float64 source buffer. Threshold controls use float32 `nextafter` below/equal/above 0.5.
All eight cases and every source-visible intermediate were regenerated. A production red harness,
isolated from this evidence-only successor, then matched all 24 case/product combinations exactly
against the corrected fixture.

`tests/fixtures/reconnect/filfinder/oracle.json` was generated twice through a real
`ProcessPoolExecutor(max_workers=1)` and was byte-identical both times at SHA-256
`fabf4ddd818d51f7f376de85035b83f2c9393a55dbc5d1d91b8946f68e511106`.

The eight cases are straight, Y-spur, disconnected, loop/branch, deterministic noise, symmetric
tie, threshold boundary, and empty. For each applicable case the fixture captures:

- source image and inclusive threshold mask;
- FilFinder existing mask and its 8-connected label map;
- medial-axis distance and pre-prune skeleton;
- post-prune full skeleton and longest-path skeleton;
- selected-raster label maps, filament lengths, branch lengths, and effective thresholds;
- raw existing-mask warning evidence, the exact adapter suppression rule, and a visible
  nonmatching-warning control;
- parent warnings, warnings transported from each keyed process-worker task, and Astropy
  import-time stderr keyed separately by case and worker;
- threshold/pruning/unit parameters, RNG seed, Python/platform, and every transitive oracle
  dependency version.

The seven nonempty cases start one worker each and submit 1, 1, 2, 1, 3, 1, and 2 ordered tasks,
respectively. Every task retains the max-pruning warning. The empty case starts no worker. The
generator asserts these channel boundaries before it writes the fixture, so a parent-only warning
capture cannot silently produce an empty warning record.

The fixture directory pins JSON to LF with `.gitattributes`. The checksum verifier canonicalizes
line endings only for explicitly classified text and leaves binary evidence byte-exact. A fresh
checkout with `core.autocrlf=true` must reproduce the raw fixture SHA above and pass the complete
checksum gate before this successor can be reviewed.

The standalone script is intentionally narrower. It independently proves that a native `float64`
predecessor of 0.5 compares below the threshold before the ImageData seam but rounds to equality
and foreground through the required `float32` coercion. This control kills the superseded direct-
`float64` threshold helper. The script also re-derives threshold equality and monotonicity,
8-connected row-major labeling, `objmask == objmap > 0`, empty behavior, and source-layer
preservation without importing PhenoTypic, FilFinder, SciPy, or scikit-image. It does not disguise
a second FilFinder call as independent validation.

## Optional dependency and stage gates for implementation

The reviewer must require tests proving all of these boundaries before G3:

1. Importing the production module, constructing the operation, and generating its schema do not
   import FilFinder or Astropy.
2. Empty-mask apply succeeds in the base environment without optional imports.
3. Nonempty apply with the dependency absent raises an actionable call-time `ImportError` naming
   the `topology` extra; it never returns an empty map.
4. `mask` calls only `create_mask(use_existing_mask=True)`.
5. `skeleton` additionally calls only `medskel(rng=seed)`.
6. `longest_path` additionally calls `analyze_skeletons` with exact field names, pixel quantities,
   the frozen one-pixel skeleton threshold, and the selected branch-threshold mode.
7. Downstream-only fields remain inactive for earlier outputs.
8. A fresh process pool and FilFinder object are created per apply; shutdown occurs once with
   `wait=True` after success and after injected failure.

## Required mutation matrix

Each mutant must fail a named test: direct-float64 comparison that skips ImageData coercion, strict
threshold, wrong image layer, omitted existing-mask flag, bare float at either pixel-quantity seam,
changed one-pixel skeleton threshold, swapped prune fields, missing RNG seed, reused FilFinder
state, wrong output attribute, 4-connected labels, objmask/map mismatch, modified `detect_mat`,
eager optional import, swallowed dependency error, skipped executor shutdown, and execution of a
downstream stage for an earlier output.

## Local gate results before review

- Pinned source oracle fixture generation: PASS, twice with identical fixture hash.
- Float32 ImageData seam, native-float64 predecessor control, and float32-nextafter boundary: PASS.
- Corrected fixture versus all 24 real-runtime case/product outputs: PASS exact.
- Process-local warning transport, narrow-filter control, and keyed worker stderr: PASS.
- Source-independent adapter logic: PASS.
- Evidence/source aggregate verification: PASS.
- Ruff and byte compilation for all A10 evidence scripts: PASS.
- Fresh wheel/sdist reference exclusion: PASS.
- Fresh `core.autocrlf=true` checkout, raw fixture hash, checksum gate, and logic suite: PASS.
- Independent corrected-successor G0 review: FAIL at `9b217368716a19be5e9b98f24ca47756265cf92f`; the reviewer proved the
  superseded direct-float64 helper survived because all prior logic inputs were already quantized.
  This successor adds the missing native-float64 predecessor control and awaits rereview.

Numerical/source fidelity here is an adapter claim only. It is not evidence that FilFinder improves
fungal detection quality on PhenoTypic images; that requires a separate ground-truth benchmark.
[Based on general reasoning - no specific citation available.]
