# GPU detector resolution & tiling fixes — design

**Date:** 2026-07-08
**Status:** Design, awaiting review
**Branch:** `worktree-gpu-detect-fixes`

## Summary

Four defects across the GPU detectors, found while asking a narrower question ("at
what resolution does SAM2 accept images?"). Three are resolution losses that make
the DINO-backed detectors decide segmentation on a 14×14 or 16×16 patch grid and
upsample 32–73×. One is a correctness bug that turns a single colony straddling a
tile boundary into two objects with corrupted areas.

The headline: **`tile_px` currently has no effect on resolution, and larger values
make it worse.** Every tile, at every `tile_px`, reaches the ViT as 224×224.

## Evidence

All numbers measured in this worktree (Apple MPS, 3000×4000 plate, `overlap=0.15`),
not inferred from docstrings.

### Backbone geometry

| | `patch_size` | `config.image_size` | registers | processor default |
|---|---|---|---|---|
| `facebook/dinov2-base` | 14 | 518 | 0 | `shortest_edge=256` → center-crop `224` |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | 16 | **224** | 4 | resize `{224,224}`, **no center crop** |

### Cost and resolution

```
strategy                     tiles   proc  tok/tile  s/tile  plate s  px/patch
FssDino  today  512->224        70    224       257   0.016      1.1      32.0
FssDino  1:1    tile=518        63    518      1370   0.116      7.3      14.0
FssDino  1:1    tile=1022       20   1022      5330   1.188     23.8      14.0
Insid3   today 1024->224        20    224       201   0.012      0.2      73.1
Insid3   1:1    tile=512        70    512      1029   0.074      5.2      16.0
Insid3   1:1    tile=1024       20   1024      4101   0.671     13.4      16.0
```

Two laws fall out:

1. **Under 1:1, `px/patch == patch_size`, independent of `tile_px`.** Resolution is
   pinned by the architecture. `tile_px` is a compute/context knob, not a fidelity
   knob.
2. **Smaller tiles win.** `518` and `1022` deliver identical 14.0 px/patch; `1022`
   costs 3.3× more (attention is quadratic in tokens *per tile* while total tokens
   across the plate are ~fixed at `area / patch²`). Same for `512` vs `1024` on v3
   (2.6×).

Both invert the current `TuneSpec(512, 2048)` bounds and the "higher = better"
docstrings.

## Findings

### F1 — `tile_px` is inert (FssDino, Insid3, DinoSam2)

`_dino_support.py:175` (and the equivalents in `extract_reference_features` /
`extract_hidden_layer_features`) call:

```python
inputs = processor(images=rgb_uint8, return_tensors="pt").to(device)
```

No `size=`, no `do_center_crop=False`. The checkpoint's classification preset wins,
so every tile arrives at 224×224. `FssDinoDetector(tile_px=512)` → 32.0 px/patch;
`Insid3Detector(tile_px=1024)` → 73.1 px/patch. Raising `tile_px` costs 4× the tiles
to buy *half* the resolution.

### F2 — query-side mask misregistration (FssDino; DinoSam2)

The reference/support path is already correct: `_insid3_detector.py:333-341` and
`_fssdino_detector.py:460-468` thread `proc_hw` into `pool_prototype` /
`align_mask_to_grid`. The **query** path does not. `_fssdino_detector.py:565`:

```python
full = resize(grid_mask.astype(np.float32), (rgb.shape[0], rgb.shape[1]), order=0, ...)
```

This stretches a grid predicted for the processor's *cropped* view over the whole
tile. For DINOv2 that view is the central 224 of a 256-resize — an 87.5% crop, so
objects are displaced radially outward, up to ~6% of tile width at the seams.

**Scope correction:** DINOv3 has `do_center_crop=None` and resizes squarely to
`{224,224}`, so the squash and its inverse cancel. `Insid3Detector` (v3 default) is
**not misregistered — only coarse**. F2 affects `FssDinoDetector` and
`DinoSam2Detector` (both v2 default). The fix must still be version-agnostic since
`dino_version` is user-settable.

A second, smaller error survives even after removing the resize: a non-multiple size
truncates. `600 → 42 patches × 14 = 588`, leaving 12 rows uncovered. Mapping the grid
to `(h, w)` instead of `(hp*patch, wp*patch)` is a 2.0% vertical scale error. This
applies to **both** directions of the map, so `align_mask_to_grid` needs the same
treatment as the new upsample helper.

### F3 — `DinoSam2Detector`'s DINO scoring is inert

`_dinosam2_detector.py:348` runs `extract_patch_features(..., rgb)` on the **whole
plate**: a 16×16 grid over 4000×3000, i.e. 250×187 native px per patch.

```
colony  30px -> 0.160 x 0.120 patches
colony  60px -> 0.320 x 0.240 patches
colony 120px -> 0.640 x 0.480 patches
```

Colonies are sub-patch. `pool_prototype(dense, p)` is called without `proc_hw`, so the
mask is "resized straight to the grid" (`_dino_support.py:372`), rounds to empty, and
returns "a zero vector if the mask is empty". Every proposal scores identically;
`similarity_thresh` becomes all-or-nothing. The DINO half of DinoSam2 does no work on
real plates.

Separately, `_dinosam2_detector.py:311` passes only `min_mask_region_area` to
`build_sam2_generator`, so the SAM2 half silently runs a stock single pass.

### F4 — `Sam2Detector` leaves its own crop pyramid switched off

SAM2's encoder resizes to a fixed **1024×1024 square** (`sam2.1_hiera_*.yaml:
image_size: 1024`; `sam2/utils/transforms.py:32` `Resize((1024, 1024))`) — a
non-aspect-preserving squash. On a 4000×3000 plate that is 3.91 × 2.93 native px/px,
anisotropic 1.33×.

`crop_n_layers` defaults to `0`, so the crop pyramid never runs. That pyramid is four
stacked defenses, all currently unused:

1. **Edge rejection** — `automatic_mask_generator.py:373` drops masks whose bbox is
   within `atol=20` px (measured in *original image* coords) of a crop edge, unless
   that edge is also the image edge (`utils/amg.py:80-90`). `atol` is not a
   colony-size tolerance: it absorbs **mask-decoder quantization**. SAM2 decodes on a
   256×256 grid and upsamples, so at a 1024-px crop one decoded pixel is 4 native px
   and `atol=20` is ~5 decoded pixels of slack. It scales with `tile_px / decoder_grid`,
   not with `tile_px`.
2. **Overlap** — `crop_overlap_ratio` default `512/1500 ≈ 0.341` of the short side.
3. **Full-image fallback** — `generate_crop_boxes` always prepends
   `[0, 0, im_w, im_h]` (`utils/amg.py:214`).
4. **Resolution-preferring NMS** — `automatic_mask_generator.py:237-239`:
   `scores = 1 / box_area(data["crop_boxes"])`, "prefer masks from smaller crops".

`box_nms_thresh` (within-crop dedup, default `0.7`) is not exposed at all.

Docstring errors to fix: `_sam2_detector.py:163` says each layer contributes `2**i`
crops; the code is `n_crops_per_side = 2**(i+1)` crossed with itself, i.e.
`(2**(i+1))**2` (4, then 16). The error is inherited verbatim from upstream's own
docstring. The square-squash is undocumented, and "sliding-window crop" should read
"crop pyramid" to match upstream vocabulary.

### F5 — `Sam3Detector` fragment bug (correctness, not fidelity)

`_tiling.py` has **no edge rejection** — no `edge`, `border`, or `touch` logic
anywhere. Uniform tiles also mean SAM2's defenses 3 and 4 are unavailable: no tile is
smaller than another (so `1/box_area` is a constant), and there is no full-image pass.
**Edge rejection is therefore the only defense uniform tiling can have, and it is
absent.**

Consequence. A colony interior to tile A straddles into tile B by area fraction `f`.
A emits the whole mask (area `A`); B emits the fragment (area `f·A`). Intersection
`f·A`, union `A`, so **IoU = f**. `_merge_tiles_iou_nms` drops a candidate only when
`iou > iou_thresh`, and `tile_merge_iou` defaults to `0.5` (`_sam3_detector.py:219`).
So every fragment with `f ≤ 0.5` survives — and the fragment is by definition the
smaller side of the cleave, so `f ≤ 0.5` is the common case.

Worse, survivors are painted largest-first, so the **fragment paints over the whole
colony**. One colony becomes two objects and the real colony's `Size_Area` is
silently corrupted. `min_mask_region_area` filters only the smallest slivers; the
`f ∈ [0.2, 0.5]` window sails through.

Semantic tiling is unaffected: `stitch_semantic_tiles` ORs boolean masks, and
`union(whole, fragment) == whole`. Do **not** apply edge rejection there — it would
punch holes at seams.

#### Why porting `is_box_near_crop_edge` alone is **not** the fix

Edge rejection trades against overlap. On one axis, with tile size `t`, stride `s`,
and a colony of diameter `d`, the colony survives in the tile starting at `a` only if
it is fully inside and at least `atol` from both tile edges:

```
a + atol ≤ x₀   and   x₁ ≤ a + t - atol
⟹  x₀ + d + atol - t ≤ a ≤ x₀ - atol
```

The window of valid tile starts has length `t - d - 2·atol`. A stride-`s` grid is
guaranteed to land in it only when `t - d - 2·atol ≥ s`, i.e.

> **`overlap_px ≥ d + 2·atol`**

Violate it and the colony is within `atol` of an edge in *every* tile containing it,
so it is rejected from all of them. It does not become a fragment — **it disappears.**

SAM2's pyramid is immune because layer 0 always sees the whole image and the
`1/box_area` scoring outvotes that coarse copy when a crop found the colony properly.
Uniform tiling has **neither** defense: every tile has the same area (so the scoring
signal is constant) and there is no full-image pass. Porting edge rejection alone
would convert a visible fragment bug into an invisible missing-colony bug.

Measured against the synth plate (colony diameters 32 / 39 / 44 px, min / median / max):

```
tile_px=1008 overlap=0.15 -> overlap_px 151 | max safe colony d = 111 px
tile_px=1024 overlap=0.15 -> overlap_px 154 | max safe colony d = 114 px
tile_px= 512 overlap=0.15 -> overlap_px  77 | max safe colony d =  37 px  ← median is 39
tile_px=1008 overlap=0.30 -> overlap_px 302 | max safe colony d = 262 px
```

At `tile_px=512, overlap=0.15, atol=20` the safe diameter is 37 px while the synth
plate's median colony is 39 px: every median-or-larger colony on a seam would be
deleted. Sam3's shipped `1008 / 0.15` is safer (111 px) but still fails on sparse
plates with large colonies.

## Non-goals

- `MicroSamDetector`. `micro_sam` is conda-only and absent from this venv; its
  resolution behaviour is unverified. Its unique `input_layer="gray"` default
  (`_microsam_detector.py:164`) is also out of scope.
- Verifying `Sam3Detector`'s `tile_px=1008`. `facebook/sam3` returns 403 (gated,
  access not granted). The value is carried as an assumption.
- The semantic detectors' downstream issue where OR-then-connectivity-label merges
  touching colonies into one object.

## Design

### Module boundaries

**`_dino_support.py` — the single resize policy.**

```python
NATIVE_PROCESSOR_KWARGS = dict(do_resize=False, do_center_crop=False)
```

Applied in all three `extract_*` functions. (`do_center_crop=False` is a harmless
no-op for DINOv3, which has none; verified.)

New helpers, so the feature side and the mask side can never disagree:

- `patch_grid_hw(pixel_hw, patch) -> (hp, wp)` — extracted from the inline
  `in_h // patch` at `_dino_support.py:183`.
- `upsample_grid_to_image(grid, image_hw, patch)` — resize the grid to
  `(hp*patch, wp*patch)`, then edge-pad the ≤ `patch-1` truncated rows/columns.
  Replaces `_fssdino_detector.py:565` and Insid3's equivalent.
- `align_mask_to_grid` — amended to use the **covered extent** `(hp*patch, wp*patch)`,
  not the full `(h, w)`.
- `pool_prototype_tiled(dense_by_tile, tiles, mask, patch)` — pools a proposal's
  features from the tile(s) covering it (for DinoSam2).

`reshape_patch_tokens` is unchanged but now known to be load-bearing: DINOv3 carries
4 register tokens. Verified: `forward @512 1:1 -> tokens 1029 == 1 cls + 4 reg + 32²`.

**`_tiling.py` — the single tiling policy.**

- Move `_iou` and `_merge_tiles_iou_nms` in from `_sam3_detector.py`. (`_tiling.py`'s
  own module docstring already says they belong here.) It is retained for the
  single-tile relabel path and as the fallback merge.
- Add `assign_by_centroid_core(objmap, tile, tiles, image_hw)` — **the instance merge
  policy.** Each tile owns a *core*: its stride window, the region no neighbouring
  tile's core covers. An instance is kept by exactly the one tile whose core contains
  its centroid; every other copy is discarded. Border tiles' cores extend to the image
  edge so nothing falls in a gap.

  No NMS, no `atol`, no duplicates **by construction**. Fragments are dropped because
  nobody claims them, not because their distance to an edge was measured. The safety
  condition improves from `overlap_px ≥ d + 2·atol` to `overlap_px ≥ d`, and it is
  provable: the tile fully containing the colony sees the true centroid, which lies in
  its core; a tile that cleaves the colony sees a fragment whose centroid is within
  `d/2` of the tile edge, while that tile's core begins `overlap_px/2 ≥ d/2` inside it,
  so the fragment is never claimed.

- Add an **overlap guard**: after merging, warn when `overlap_px` is smaller than the
  largest retained instance's diameter. This is the condition under which a colony can
  still be lost, and it is measurable at runtime rather than guessed a priori. The
  current code has no guard at all.

- `stitch_semantic_tiles` unchanged.

`reject_edge_instances` / `atol` are **not** adopted. SAM2's `atol` exists to absorb
its own decoder quantization, and its edge rejection is safe only because the pyramid
carries a full-image fallback that uniform tiling lacks (see F5).

Note `_plan_tiles` already emits uniform `tile_px²` tiles whenever an axis exceeds
`tile_px` (the last start is clamped to `extent - tile_px`, overlapping more rather
than shrinking). Ragged tiles occur only on the un-tiled path, where an axis is
shorter than `tile_px` — which is every doctest, since `load_synth_yeast_plate()` is
600×800.

### Per-operation changes

| Operation | Change | Effect |
|---|---|---|
| `FssDinoDetector` | `tile_px: 512 → 518`, `TuneSpec(256, 1024)`; `_segment_crop` uses `upsample_grid_to_image` | 32.0 → 14.0 px/patch; 6.6× cost |
| `Insid3Detector` | `tile_px: 1024 → 512`, `TuneSpec(256, 1024)`; same upsample fix | 73.1 → 16.0 px/patch; 26× cost |
| `DinoSam2Detector` | new `tile_px = 518`, `tile_overlap = 0.15`; DINO per tile at 1:1; proposals pooled from covering tile(s); pass the four `crop_*` args through to `build_sam2_generator` and expose them as fields | prototypes stop collapsing to zero |
| `Sam2Detector` | `crop_n_layers: 0 → 1`; new `box_nms_thresh = 0.7`; docstring corrections (crop count, square squash, "crop pyramid") | 3.91 → ~1.9 px/px; ~5× cost |
| `Sam3Detector` | import merge from `_tiling`; replace `_merge_tiles_iou_nms` with `assign_by_centroid_core`; add the overlap guard | fragments eliminated without risking colony deletion |

### Choosing the `tile_px` defaults

`config.image_size` is **not** a usable signal: DINOv2 reports `518` (its high-res
adaptation), DINOv3 reports `224`. Defaulting to it would set Insid3 to exactly the
broken value.

The defensible basis: resolution is already pinned at `patch_size`, so `tile_px` is
chosen for compute, and 512 is the cheap end of the curve. Pick the **largest exact
patch-multiple near 512** for each detector's default `dino_version` — `518 = 14×37`
for v2, `512 = 16×32` for v3. `patch_size` is unknown until the model loads, so these
stay per-detector literals, with a **load-time warning** when the user flips
`dino_version` and leaves `tile_px` non-multiple.

These defaults are a compute choice validated by the accuracy gate below, not a claim
about model nativeness.

## Compatibility

`to_json()` pins **every** field, including defaults (verified). Existing serialized
pipelines carry `crop_n_layers: 0`, `tile_px: 512` explicitly and keep them, so **no
default change can alter a deserialized pipeline**. Adopting the fixes requires
re-serializing; the changelog must say so.

The `_dino_support` resize fix is code, not a field, so it **does** reach old
pipelines. A pipeline pinned at `tile_px=512` goes from a 16×16 grid to 36×36 without
asking. This is a behaviour change, not a bugfix, and belongs under that heading.

No field is removed, so `from_json()` on old payloads never fails. New fields
(`DinoSam2.tile_px`, `tile_overlap`, the four `crop_*`; `Sam2.box_nms_thresh`) are
additive and fill from defaults.

Every new numeric field needs a `TuneSpec` (or `TuneSpec(tunable=False)`) or the
coverage gate against `tests/fixtures/tune/annotation_allowlist.json` fails.

Docs: `docs/source/how_to/pages/gpu_detection_setup.md` is the only file hardcoding
these numbers — lines 167, 178 (`1008`), 261 (`1024`), 299 (`512`). All four need
updating, and the "higher = better" framing needs inverting.

## Testing

**Pure functions (no model, fast).**

- `assign_by_centroid_core`: cores partition the image (no gaps, no double-claims);
  border cores reach the image edge; an instance centred in tile T's core is kept by T
  and by no other tile.
- **Fragment regression:** two tiles, whole colony in A, fragment `f=0.3` in B →
  assert exactly **1** instance, area uncorrupted. Today: 2 instances, colony
  overpainted.
- **Colony-deletion regression** (the failure the rejected `atol` design would have
  introduced): a colony with `d > overlap_px` straddling a seam must still yield ≥ 1
  instance, and the overlap guard must warn.
- **Safety bound:** for `overlap_px ≥ d`, a colony swept across every seam offset is
  retained exactly once at every position.
- `upsample_grid_to_image`: synthetic disc centroid preserved within 1 px. Today:
  displaced ~2%.
- `align_mask_to_grid` round-trip symmetry.
- `patch_grid_hw` matches the real conv output: 518→37×37, 512→36×36 (v2),
  600×800→42×57, 512→32×32 (v3).

**Processor policy (fake capturing processor, no download).** Assert `do_resize=False`
and `do_center_crop=False` reach all three `extract_*` functions.

**Functional (behind the existing `_dinov2_backbone_loadable()` guard at
`tests/unit/detect/nn/test_fssdino_detector.py:201`, CPU, `dinov2-small`).**

- FssDino on `synth_plate`: dense grid is `(42, 57)`, not `(16, 16)`. Direct F1
  regression.
- DinoSam2 on `synth_plate`: pooled prototypes neither all-zero nor all-identical.
  Direct F3 regression.

## Accuracy budget

**Measured.** `scripts/accuracy_gate_gpu_detectors.py` replicates
`load_synth_yeast_plate()` 3×4 into an 1800×3200 plate with 1152 relabelled
ground-truth colonies. Colony diameters stay 32–44 px; only the plate grows, so it
exceeds SAM2's 1024 encoder and the detectors face real downsampling. The pre-fix
baseline restores the 224 classification preset by emptying `NATIVE_PROCESSOR_KWARGS`
— after the fix `tile_px` no longer controls resolution, so `512` vs `518` would
measure nothing. Both arms plan the same 32 tiles; only the per-tile resize differs.

Run: `PHENOTYPIC_ACCEPT_MODEL_LICENSE=dinov3 uv run python
scripts/accuracy_gate_gpu_detectors.py` (DINO `small`, `device="auto"`, Apple MPS).

| Arm | objmask IoU | objects / 1152 | wall-clock |
|---|---|---|---|
| FssDino — 224 preset (pre-fix) | 0.1961 | 31 | 4.4 s |
| **FssDino — 1:1 native (post-fix)** | **0.5877** | **942** | 2.3 s |
| Insid3 — 224 preset (pre-fix) | 0.0000 | 0 | 1.2 s |
| **Insid3 — 1:1 native (post-fix)** | **0.0043** | **103** | 2.0 s |
| Sam2 — `crop_n_layers=0` (old default) | 0.4076 | 481 | 18.2 s |
| **Sam2 — `crop_n_layers=1` (new default)** | **0.8722** | **1079** | 152.8 s |

**Branch taken: the new defaults ship.** Every new default scores ≥ its predecessor,
so no default is reverted. FssDino (0.1961 → 0.5877, 31 → 942 colonies) and Sam2
(0.4076 → 0.8722, 481 → 1079 colonies) both improve decisively, and their object
counts rise toward 1152 rather than overshooting — the gain is recovered colonies, not
over-segmentation.

**The gate does not validate Insid3's 26× cost.** Insid3 scores near-zero in *both*
arms; the 0.0000 → 0.0043 delta is noise, not evidence. A follow-up probe with the
`base` backbone returned **0 objects in both arms**, so the failure is neither a
backbone-size nor a processor-policy artifact. On the *un*-replicated 600×800 plate
Insid3 finds 8–9 of 96 colonies at its default `similarity_thresh=0.5`, and its own
functional test asserts only a *non-empty* mask at a permissive `similarity_thresh=0.0`
("a plumbing floor, not accuracy"). So Insid3 is weak at its default threshold
**independently of this change**. That needs its own investigation; it is out of scope
here.

**It is not a scale bug.** The 1800×3200 plate is literally 12 copies of the 600×800
one, so "works small, fails large" would have implied a tiling or coordinate defect.
It doesn't: Insid3's detection *rate* is the same at both scales — 8 of 96 colonies
(8.3%) on the small plate, 103 of 1152 (8.9%) on the large one. Per-tile probing
confirms the behaviour is positional, not size-dependent: some tiles fire (1357, 983
foreground px), most return 0–21 px. The prototype is unit-norm and the positional
basis is `(384, 4)` as expected, so the pipeline is wired correctly. What fails is the
`similarity_thresh = 0.5` cosine floor against a DINOv3-small prototype — the same
conclusion its own functional test reached when it lowered the floor to `0.0` to get a
non-empty mask.

**The decision rule in the plan was wrong for the DINO arms, and it matters.**
"If 1:1 loses, revert the default" cannot work: the 6.6× / 26× cost comes from
`NATIVE_PROCESSOR_KWARGS`, which is **code applied unconditionally**, not from
`tile_px`. Reverting `Insid3.tile_px` 512 → 1024 would *keep* 1:1 (still 16.0
px/patch) while quadrupling tile area — 13.4 s/plate instead of 5.2 s. Strictly worse
on both axes. The only lever that could undo the DINO cost is making the processor
policy opt-in, which is a code revert, not a default revert.

So Insid3's `tile_px = 512` is kept on **compute and patch-alignment grounds**
(`512 = 16×32`, exact; and it is the cheap end of a curve where resolution is already
pinned at `patch_size`), not on accuracy grounds. FssDino exercises the *identical*
processor-policy code path and improves 3× on IoU, which is the evidence that the
policy itself is sound.

Caveats on the numbers above:

- **Wall-clock is not cost evidence.** The first `evaluate()` per detector absorbs
  lazy model construction and MPS warmup, and each detector's pre-fix arm runs first.
  That is why FssDino's 224 arm (4.4 s) appears *slower* than its 1:1 arm (2.3 s),
  which contradicts its token counts. The 6.6× / 26× / ~5× cost figures elsewhere in
  this document come from separate measurements, not this table.
- **IoU is blind to the fragment bug.** A colony split into two labels preserves the
  mask union. Task 5's unit regressions cover the tiling merge; this gate does not.
- Object count is reported alongside IoU precisely because IoU cannot see
  over-segmentation.
- **Independently reproduced** by the orchestrator, outside the script, on the two
  decision-critical detectors. With the backbone warmed first (so wall-clock excludes
  model load):

  ```
  FssDino 224 preset (pre-fix)   IoU 0.1961  objects   31 / 1152    1.4s
  FssDino 1:1 native (post-fix)  IoU 0.5877  objects  942 / 1152    2.5s
  Sam2    crop_n_layers=0        IoU 0.4076  objects  481 / 1152   28.6s
  Sam2    crop_n_layers=1        IoU 0.8722  objects 1079 / 1152  149.3s
  ```

  IoU and object counts match the table exactly. Warmed, FssDino's 1:1 arm is the
  slower one (1.4 s → 2.5 s), as its token counts predict. Sam2's `crop_n_layers=1`
  costs **5.2×** here (28.6 s → 149.3 s), close to its 5-encoder-pass prediction.
- The script's `evaluate()` reads `objmask` from the value **returned** by `apply()`,
  not from the input image. `apply()` defaults to `inplace=False` and returns a copy;
  reading the input back yields 0 objects for every arm. The plan's original script had
  this bug.

The three defects (F2, F3, F5) assert correctness via unit tests, make **no** accuracy
claim, and ship regardless.

## Risks and unverified assumptions

- `Sam3Detector`'s `tile_px = 1008` is unverified; `facebook/sam3` returns 403.
- `MicroSamDetector` is entirely unverified (conda-only).
- Compute: FssDino 1.1 → 7.3 s/plate, Insid3 0.2 → 5.2 s/plate on an Apple GPU. A
  10,000-plate screen goes from ~3 to ~20 GPU-hours. This is an array-job sizing
  decision, consistent with the project's stated "accuracy over speed" philosophy.
- `crop_n_layers: 0 → 1` makes every newly-constructed `Sam2Detector` ~5× slower.
  Changelog: behaviour change.
- Higher resolution is not monotonically better in principle. DINOv2's 224 preset is
  where its *classification* head was evaluated; dense-prediction regimes exist where
  more global context beats more pixels. The accuracy gate exists to catch this.
- `assign_by_centroid_core` still loses a colony wider than `overlap_px` (no tiling
  policy can retain one without a full-image fallback). The overlap guard makes this
  observable; `Sam3Detector`'s default `1008 / 0.15` gives `overlap_px = 151`.
- `Insid3Detector`'s default `dino_version=3` is gated, so the stock constructor is
  unusable until Meta approves the user's DINOv3 access request.

## Resolved questions

**1. `Insid3Detector` keeps `dino_version = 3`.** The earlier framing — that its 26×
cost argued for switching to DINOv2 — rested on a misleading ratio. In absolute terms
DINOv3 is **cheaper**: 5.2 s/plate (70 tiles × 1029 tok) against FssDino's DINOv2 at
7.3 s/plate (63 tiles × 1370 tok). The 26× is large only because Insid3's baseline was
fast *because* it was broken (20 tiles at 224², 73 px/patch). The ratio is the
misleading statistic; the seconds are the real one.

What remains is a genuine trade, and cost is not part of it. INSID3 is DINOv3-native
(`_insid3_detector.py:257`) — its defining step removes the positional bias that
DINOv3's patch features specifically carry, and DINOv3's 4 register tokens are the
only reason `reshape_patch_tokens`' register-aware slice is exercised at all. The
price is patch-16's coarser 16.0 px/patch versus DINOv2's 14.0, a gap no `tile_px` can
close because under 1:1 the resolution *is* the patch size.

The actual problem with the v3 default is neither: **it is gated.** `Insid3Detector()`
with stock arguments cannot run until Meta approves the user's DINOv3 access request,
while `FssDinoDetector` defaults to ungated DINOv2 and works immediately. Undocumented.
Action: document the gate as a first-run requirement and raise a clear error at
construction rather than at first `apply()`.

**2. `atol` is not adopted.** See F5. It absorbs SAM2's decoder quantization
(`tile_px / decoder_grid`, ≈5 decoded px), not colony size, and edge rejection is only
safe alongside a full-image fallback that uniform tiling does not have. The merge
policy is `assign_by_centroid_core`, which needs no tolerance parameter and relaxes the
safety bound from `overlap_px ≥ d + 2·atol` to `overlap_px ≥ d`.

## Open questions

1. Should the overlap guard warn, or raise? A colony wider than `overlap_px` is
   silently at risk under any tiling policy; the guard makes it visible, but a hard
   failure may be preferable for batch runs.
