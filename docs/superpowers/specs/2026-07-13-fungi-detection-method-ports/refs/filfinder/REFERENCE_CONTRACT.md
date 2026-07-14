# A10 FilFinder 1.8 frozen wrapper contract

## Authority and scope

The executable authority is the official `fil-finder==1.8` source distribution and the matching
annotated `v1.8` tag commit `22539cf2176ad9b717658652e8da749158597f4d`. PyPI listed 1.8 as the
latest stable release when checked on 2026-07-13. Package metadata identifies version 1.8, Python
3.9 or newer, its dependency set, and MIT licensing at `upstream/PKG-INFO:1-29`.

The 2015 peer-reviewed paper establishes the high-level mask to medial-axis skeleton to pruned
graph workflow at `paper/Koch_Rosolowsky_2015.txt:326-424`. The maintained source controls exact
API calls, current output attributes, tie handling, thresholds, and dependency behavior. This is
an external wrapper, not a source transcription. Numerical fidelity means that the adapter passes
the frozen values into FilFinder 1.8 and selects the correct source-visible raster.

## Public operation contract

`FilFinderDetector(ObjectDetector)` has keyword-only pydantic fields:

```python
threshold: float = 0.5                    # finite, 0 <= value <= 1
output: Literal["mask", "skeleton", "longest_path"] = "mask"
beamwidth_px: float = 1.0                 # finite and > 0
prune_criteria: Literal["all", "intensity", "length"] = "all"
relative_intensity_threshold: float = 0.2 # finite, 0 < value <= 1
branch_threshold_px: float | None = None  # None or finite and > 0
max_prune_iterations: int = 10            # >= 1, booleans rejected
rng_seed: int = 0                         # >= 0, booleans rejected
```

There is no public skeleton-length threshold. FilFinder 1.8 accepts `skel_thresh`, but then sets
`self.skel_thresh = min(ceil(skel_thresh), 1 pix)` at `upstream/fil_finder/filfinder2D.py:658-669`.
For every positive pixel threshold the effective value is therefore one pixel. The wrapper passes
exactly `1 * astropy.units.pix`; it does not expose an ineffective tuning parameter and does not
compensate for the upstream cap.

## Input, threshold, and units

Read a copy of `image.detect_mat[:]`; do not modify RGB, gray, or `detect_mat`. The threshold mask
is exactly `detect_mat >= threshold`, so equality is foreground. NaN pixels compare false. Copy the
mask before giving it to FilFinder because its constructor mutates NaNs in a supplied mask at
`upstream/fil_finder/filfinder2D.py:154-161`.

Construct a fresh `FilFinder2D` for each application with the copied float image, copied boolean
mask, and `beamwidth=beamwidth_px * u.pix`. The source converts the supplied beam width to pixels at
`upstream/fil_finder/filfinder2D.py:136-150`. `branch_threshold_px`, when non-None, is likewise
passed as a pixel quantity. Bare floating-point values are forbidden at those quantity seams.

The wrapper owns a fresh `ProcessPoolExecutor(max_workers=1)` for each application and shuts it
down with `wait=True` on success or exception before returning. This replaces the source's
implicit reusable/process executor creation at `upstream/fil_finder/filfinder2D.py:167-175` with
an explicit lifetime; it does not change the source's process-execution model or ordered result
assembly. The executor and external object are never cached or reused.

## Frozen stage graph

For a nonempty threshold mask, all outputs first construct FilFinder with the supplied mask and
call `create_mask(use_existing_mask=True)`. That source branch skips flattening and adaptive
segmentation at `upstream/fil_finder/filfinder2D.py:299-325`. The wrapper suppresses only the
expected "Using inputted mask" warning from that call.

- `output="mask"`: select `filfinder.mask`; never call `medskel` or `analyze_skeletons`.
- `output="skeleton"`: call `medskel(rng=rng_seed)`, select the pre-analysis
  `filfinder.skeleton`, and never call `analyze_skeletons`. The source forwards the RNG to
  scikit-image medial axis and records its distance transform at
  `upstream/fil_finder/filfinder2D.py:524-573`.
- `output="longest_path"`: call `medskel(rng=rng_seed)`, then call
  `analyze_skeletons(prune_criteria=..., relintens_thresh=...,
  skel_thresh=1*u.pix, branch_thresh=..., max_prune_iter=...)`, and select
  `filfinder.skeleton_longpath`. Parameter forwarding and post-analysis raster assembly are at
  `upstream/fil_finder/filfinder2D.py:595-753`.

Fields used only by `analyze_skeletons` remain serializable but inactive for `mask` and `skeleton`.
Changing an inactive field must not alter an earlier output or cause a downstream call.

If the threshold mask is empty, return same-shape all-zero `objmap` and boolean `objmask` without
constructing FilFinder. This is an explicit adapter edge case because graph analysis assumes
filament objects; it also prevents an optional dependency call for a mathematically empty result.

## Optional-import timing

The production module must import without FilFinder or Astropy installed, and operation
construction and schema generation must also succeed in the base environment. After thresholding,
an empty mask completes without importing either optional package. A nonempty application imports
`FilFinder2D` and `astropy.units` immediately before constructing the external object. If either is
unavailable, raise an actionable call-time `ImportError` naming the `topology` extra; never swallow
the error or return an empty detection. Tests must block these imports, inspect `sys.modules`, and
separately prove module-import, construction, empty-apply, and nonempty-apply timing.

## Output translation

Treat the selected FilFinder raster as boolean. Label it with 8-connectivity using a full 3 by 3
structure. Components receive consecutive positive labels in the deterministic row-major order of
their first pixel; background is zero. Set `objmap` to that label array and set `objmask` to
`objmap > 0`, so both outputs describe exactly the selected raster. Mask output labels mask
components, skeleton output labels the pre-prune medial skeleton, and longest-path output labels
the source's `skeleton_longpath`. No graph or distance product is exposed because `ObjectDetector`
has no graph channel.

## External fixture and accuracy claim

`tests/fixtures/reconnect/filfinder/oracle.json` records straight, Y-spur, disconnected,
loop/branch, deterministic noise, symmetric tie, threshold-boundary, and empty cases. It includes
input, threshold mask, FilFinder mask, medial-axis distance, pre- and post-prune skeleton,
longest-path raster, labels, filament/branch lengths, warnings, parameters, seed, platform, and the
complete dependency vector.

The golden arrays require exact equality on the pinned oracle environment. This contract makes no
cross-version bitwise claim: medial-axis tie behavior and graph results are dependency-sensitive,
which is why every transitive oracle version is recorded. The standalone script independently
validates only PhenoTypic-owned thresholding, labeling, mask/map equivalence, empty behavior, and
layer preservation. It deliberately does not reimplement FilFinder skeletonization, pruning, or
longest-path logic.

Numerical fidelity to this package does not establish biological detection benefit. Such a claim
requires a separate ground-truth image benchmark. [Based on general reasoning - no specific
citation available.]
