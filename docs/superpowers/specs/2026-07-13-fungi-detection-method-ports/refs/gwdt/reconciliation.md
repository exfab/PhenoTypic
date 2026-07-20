# APP2 GWDT source reconciliation

## Authority and frozen contract

APP2 defines GWDT as accumulated image intensity on a shortest path to background
(`paper/2013_BIOINFO_app2.txt:189-197`). Pixels with intensity no greater than the
threshold are background seeds (`paper/2013_BIOINFO_app2.txt:198-203`), edge weight is
destination intensity (`paper/2013_BIOINFO_app2.txt:205-212`), and background distance
initializes from input intensity (`paper/2013_BIOINFO_app2.txt:212-218`). The executable
authority is Vaa3D `fastmarching_dt` at
`vaa3d/app2/fastmarching_dt.h:33-199`. The 2-D PhenoTypic contract replaces threshold
selection with an explicit boolean background mask but otherwise preserves this source,
including its asymmetric frontier phase and float32 output.

The local cost helper preserves the fixed GI table and truncating index consumed by
APP2's later tree stage when bounds are well-defined. It is not a claim that
`fastmarching_dt` itself emits a cost map. The active macro
truncates normalized values into a fixed 256-value table
(`vaa3d/app2/fastmarching_macro.h:8-41`). The later tree recurrence averages source and
destination GI values and multiplies by hard-coded factors `1`, `1.414214`, or
`1.732051`
(`vaa3d/app2/fastmarching_tree.h:278-287,353-371`). Detector integration must therefore
use endpoint averaging; destination-only cost-surface composition is not APP2 fidelity.
The detector intentionally does not port the adjacent tree threshold gate: Vaa3D either
rejects every below-threshold destination or permits only isolated below-threshold breaks,
depending on `is_break_accept` (`fastmarching_tree.h:357-367`). PhenoTypic instead treats
every finite GI pixel as traversable and leaves gap rejection to its downstream path-quality
cascade. This explicit detector policy is drift D13.

## Line-by-line mapping

| Source behavior | Vaa3D evidence | Production evidence | Resolution |
|---|---|---|---|
| Three states `ALIVE`, `TRIAL`, `FAR` | `fastmarching_dt.h:35-58` | `_gwdt.py:133-137,149-150,171,178-179` | Boolean `alive` and `trial`; FAR is neither. |
| Background is `input <= threshold` | `fastmarching_dt.h:24,45-58` | `_gwdt.py:99-104,129-130` | Explicit boolean mask; drift D02. |
| Background distance equals input intensity | `fastmarching_dt.h:47-50` | `_gwdt.py:133-136` | Exact after float32 conversion. |
| Non-background starts at `1E20` | `fastmarching_macro.h:4`; `fastmarching_dt.h:54-57` | `_gwdt.py:62,134` | Exact float32 sentinel. |
| Background is scanned in flattened row-major order | `fastmarching_dt.h:67-71` | `_gwdt.py:143` | `np.argwhere` is C-order for a 2-D C-contiguous boolean result. |
| Neighbor offset is Manhattan count and accepted when `offset <= cnn_type` | `fastmarching_dt.h:73-87` | `_gwdt.py:15-30,140,144-148` | One-slice reduction: `cnn_type=1` becomes 4-connectivity; `cnn_type>=2` becomes 8-connectivity. Drift D01. |
| A frontier pixel is initialized only while FAR | `fastmarching_dt.h:88-89,115-120` | `_gwdt.py:149-150,170-172` | Existing TRIAL pixels are not refreshed during initialization. |
| Positive seed searches neighboring ALIVE values for the minimum | `fastmarching_dt.h:90-113` | `_gwdt.py:151-165` | Exact strict comparison and neighborhood. |
| Frontier recurrence omits geometric length | `fastmarching_dt.h:115` | `_gwdt.py:166-172` | Preserved exactly, including diagonal omission. |
| Minimum heap item becomes ALIVE | `fastmarching_dt.h:135-150` | `_gwdt.py:174-179` | Python heap with lazy stale-entry deletion; drift D05. |
| Ordinary recurrence uses destination intensity times `sqrt(offset)` | `fastmarching_dt.h:152-172` | `_gwdt.py:181-195` | Exact destination polarity and 1 or sqrt(2) one-slice length. |
| FAR insert, TRIAL strict-less decrease | `fastmarching_dt.h:173-188` | `_gwdt.py:196-202` | Same scalar update. Equal distances keep the existing value. Heap tie order is not claimed equivalent because no parent/path is public. |
| Distance storage is float32 | `fastmarching_dt.h:41,171` | `_gwdt.py:133-135,166,192` | Exact float32 cast at initialization and every ordinary update. |
| No-background result remains sentinel | `fastmarching_dt.h:45-59,135,198` | `_gwdt.py:134,143,174,204` | Exact `1e20` float32 map. Detector seam must guard this case. |
| All-background result is input intensity | `fastmarching_dt.h:45-59,135,198` | `_gwdt.py:134-136,143,174,204` | Exact float32 input values. Detector seam must guard this case. |
| Heap equality does not swap | `vaa3d/app2/heap.h:70-87` | `_gwdt.py:196-202` | Scalar equality remains unchanged; heap internal tie order is output-unobservable. |
| GI input range scans observed minimum and maximum | `fastmarching_tree.h:278-287` | `_gwdt.py:228-233` | Production uses independent min/max reductions; the source overload's `else if` can leave minimum unset for strictly increasing flattened input. Drift D09. |
| GI index truncates normalized value times 255 | `fastmarching_macro.h:8` | `_gwdt.py:233-234` | Exact NumPy integer truncation for nonnegative normalized input. |
| GI table contains 256 fixed values | `fastmarching_macro.h:10-41` | `_gwdt.py:32-62` | Literal transcription; fixture equality is exact. |
| Tree edge averages endpoint GI and multiplies a hard-coded factor | `fastmarching_tree.h:353-371` | `_filamentous_fungi_detector.py:59-68,71-163` | The opt-in S01 kernel preserves diagonal `1.414214`, endpoint averaging, and strict-less relaxation. It is separate from the destination-only legacy kernel. Multi-colony seeds and returned ownership maps are D11. |
| Tree traversal applies `bkg_thresh` according to `is_break_accept` | `fastmarching_tree.h:357-367` | `_filamentous_fungi_detector.py:_run_app2_gwdt_dijkstra` | The detector applies no tree-stage threshold gate; every finite GI pixel is traversable. Gap acceptance remains in the downstream quality cascade. Drift D13. |
| Tree has one root and scans 2-D neighbors northwest through southeast in nested row/column order | `fastmarching_tree.h:237-245,289-310,341-355` | `_filamentous_fungi_detector.py:_APP2_NEIGHBORS,_run_app2_gwdt_dijkstra` | The detector's multi-colony adaptation inserts boundary seeds row-major and scans clockwise from east. Strict-less updates retain the first owner and predecessor on exact ties. Drift D14 records both source-unclaimed seed ownership and the deliberate neighbor-order difference; S09 restores the exact source sequence and changes a path-visible predecessor. |
| Tree runs on one image domain | `fastmarching_tree.h:237-245` | `_filamentous_fungi_detector.py:_reconnect_fragments_tiled` | Overlap is the processing halo, edge tiles clip, the full-image GI map uses identical slice bounds, and first row-major tile wins overlap writes. Drift D15. |

## Fixture and independent oracle

`source_harness.cpp` invokes the unmodified template for standard, initialization
diagonal, threshold-equality/nonzero seed, all-background, no-background, and ordinary
post-frontier diagonal cases. `generate_fixture.py` stores every distance map and every
defined fixed-GI table/index map for both connectivities in
`tests/fixtures/reconnect/gwdt/app2_source.npz`. Harness COST output is a derived
execution of the pinned macro/table law with robust `std::minmax_element`; it is not a
direct output field of `fastmarching_dt` or the undefined increasing-input source scan.

The standalone logic script uses exact source frontier initialization followed by
whole-grid Bellman-Ford relaxation. Its float32 comparison bound is
`gamma_n * C`, where `gamma_n = n*eps32/(1-n*eps32)` and `C` bounds the largest simple
path magnitude. It also proves the runner-up path gap in the post-frontier diagonal case
exceeds the rounding envelope. Production-to-source fixture assertions are exact and do
not use this tolerance.
