# A10 FilFinder 1.8 source reconciliation

| Source behavior | Pinned source | Wrapper disposition |
|---|---|---|
| PhenoTypic stores floating `detect_mat` as `float32` | `src/phenotypic/_core/_image_parts/_image_data_manager.py:24-57` | Quantize at the existing ImageData seam, then copy those exact values to a `float64` FilFinder buffer without normalization. Fixture boundaries use `float32` neighbors. |
| Constructor accepts image, beam width, mask, pool | `upstream/fil_finder/filfinder2D.py:97-100` | Fresh object per apply; copied image/mask, pixel quantity, owned one-process executor. |
| Supplied mask must match image and is stored | `upstream/fil_finder/filfinder2D.py:154-161` | Inclusive threshold creates same-shape boolean mask; pass a copy. |
| Default constructor creates reusable/process pool | `upstream/fil_finder/filfinder2D.py:167-175` | Lifetime-only drift F06: wrapper supplies and shuts down a fresh one-process pool. |
| Existing-mask path skips segmentation | `upstream/fil_finder/filfinder2D.py:299-325` | Always call `create_mask(use_existing_mask=True)` for nonempty masks; suppress only its exact `UserWarning` text in a call-scoped filter. A nonmatching control warning proves other warnings remain visible. |
| Medial axis accepts RNG and returns skeleton/distance | `upstream/fil_finder/filfinder2D.py:524-566` | Forward exact nonnegative seed for skeleton and longest-path outputs. |
| Medial axis deletes points with distance below one pixel | `upstream/fil_finder/filfinder2D.py:567-573` | Preserve without compensation. |
| Analysis accepts prune, intensity, skeleton, branch, iteration fields | `upstream/fil_finder/filfinder2D.py:595-645` | Forward only for `longest_path`; pixel thresholds are quantities. |
| Relative intensity must be in `(0, 1]` | `upstream/fil_finder/filfinder2D.py:647-649` | Validate at operation construction and retain source check. |
| Source caps positive skeleton threshold to one pixel | `upstream/fil_finder/filfinder2D.py:658-669` | Freeze one pixel; remove ineffective public parameter. |
| None branch threshold is three beam widths, then ceil | `upstream/fil_finder/filfinder2D.py:671-677` | Preserve `None`; explicit values pass in pixels. |
| Source labels skeletons with 8-connectivity | `upstream/fil_finder/filfinder2D.py:679-695` | Selected raster is also labeled with 8-connectivity. |
| Analysis forwards prune parameters and preserves result order | `upstream/fil_finder/filfinder2D.py:709-720` | One-process execution preserves the same ordered future collection. |
| Source assembles post-prune and longest-path rasters separately | `upstream/fil_finder/filfinder2D.py:740-753` | Select only `skeleton_longpath` for the public longest-path output. |
| Filament analysis builds graph and longest path | `upstream/fil_finder/filament.py:288-387` | External behavior; exact fixture, not reimplemented. |
| Process workers emit pruning and dependency warnings | `upstream/fil_finder/filament.py:331-361`; `upstream/fil_finder/pixel_ident.py:831,861` | Capture warnings inside every real worker task and transport keyed records to the parent. Retain import-time Astropy stderr in a separate keyed channel. |
| Paper describes mask, medial skeleton, graph, and pruning | `paper/Koch_Rosolowsky_2015.txt:326-424` | Context only; maintained 1.8 source decides executable details. |

Optional imports are an adapter-only seam. FilFinder and Astropy are absent from module import,
construction, schema generation, and empty-mask application. A nonempty application imports them
at call time and reports the `topology` extra if unavailable; drift F08 records this boundary.

Every source-visible external output in the public path is present in the fixture. The pinned
source-oracle generator compares the threshold mask, FilFinder mask, distance, pre/post-prune
skeleton, longest path, lengths, and every warning/stderr channel. Production tests compare all 24
selected case/product label maps exactly, plus adapter-owned stage, warning-transport, dtype,
threshold, unit, and lifecycle behavior.

## G2 production implementation map

| Frozen wrapper behavior | Production implementation | Executable evidence |
|---|---|---|
| Float32 ImageData seam, copied float64 source, inclusive threshold, empty short circuit | `src/phenotypic/detect/_filfinder_detector.py:162-165,274-281` | `tests/unit/detect/test_filfinder_detector.py:241-269,290-370`; `docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/filfinder.py:26-52` |
| Lazy optional imports and actionable nonempty failure | `src/phenotypic/detect/_filfinder_detector.py:152-159,252-272,283` | `tests/unit/detect/test_filfinder_detector.py:177-269` |
| Fresh one-process executor with keyed child-warning forwarding and guaranteed shutdown | `src/phenotypic/detect/_filfinder_detector.py:46-149,284-320` | `tests/unit/detect/test_filfinder_detector.py:460-510` |
| Exact supplied-mask warning filter | `src/phenotypic/detect/_filfinder_detector.py:168-176` | `tests/unit/detect/test_filfinder_detector.py:444-458` |
| Constructor units/copies and frozen mask, medial-axis, analysis stage order | `src/phenotypic/detect/_filfinder_detector.py:285-318` | `tests/unit/detect/test_filfinder_detector.py:271-348` |
| Selected raster, 8-connected labeling, row-major labels, and mask/map consistency | `src/phenotypic/detect/_filfinder_detector.py:294-330` | `tests/unit/detect/test_filfinder_detector.py:372-396`; `docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/filfinder.py:55-86` |
| All 24 source-selected products across eight oracle cases | `src/phenotypic/detect/_filfinder_detector.py:274-331` | `tests/unit/detect/test_filfinder_detector.py:512-554`; `tests/fixtures/reconnect/filfinder/oracle.json` |
| Required transcription mutants | `docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/filfinder/run_mutations.py:378-603` | `docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/filfinder/MUTATION_MATRIX.md` |

The production module is private in G2. Public exports, dependency declarations, detector seams,
serialization registries, taxonomy, tuning, and GUI discovery remain integrator-owned G5 work.
