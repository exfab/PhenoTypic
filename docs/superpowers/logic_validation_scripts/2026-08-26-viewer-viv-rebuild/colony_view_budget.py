"""Re-derive the per-frame cost of one OrthographicView per colony.

Claim under test (viewer-viv-rebuild spec section 6.2): deck.gl re-renders
every view each frame, so an uncapped colony grid degrades linearly in cell
count. This script derives the draw-call and texture budget for a plate's
worth of cells so the cap is chosen against a number.

Exits non-zero until the prototype measurement in task 4.1 step 3 has been
recorded, so no later task can proceed on an unmeasured cap.
"""

import math
import sys

#: Layers rendered into EACH view. NOT a constant to assert: Plate section
#: 6.1's Layers panel exposes rgb, gray, detect_mat and objmap, all
#: independently toggleable, so the worst case is 4. Take it from the stack
#: the caller actually builds.
VISIBLE_LAYERS = 4
#: A common plate: 32 x 48 = 1536 colonies (backend section 2.3's example).
PLATE_ROWS, PLATE_COLS = 32, 48
PLATE_CELLS = PLATE_ROWS * PLATE_COLS
#: The reference plate, and the chunk grid it implies.
PLATE_HW = (4000, 3000)
CHUNK_PX = 1024
#: Source-pixel side length one cell's viewport shows. The colony grid's own
#: default crop side (``_MIN_CROP_SIZE`` in ``colony_view/_grid.py``).
CELL_SRC_PX = 64
#: Measured cap, filled in from the prototype in step 3. None until measured.
#: 128 cells, chosen against the RATIO to the single-view frame time rather
#: than an absolute millisecond budget -- see the table in
#: ``plans/2026-08-26-viewer-viv-rebuild/spike/README.md``. The prototype ran
#: under SwiftShader (no GPU on the node), where even ONE view costs 91 ms,
#: so an absolute budget would encode this node's rasterizer, not the design.
#: A ratio survives a uniform hardware speedup; the rule is "the grid must
#: not cost more than the canvas it draws into", i.e. frame(N) <= 2 x frame(1).
RECORDED_CAP: int | None = 128
#: Frame time, in ms, observed at RECORDED_CAP. Recorded beside the number so
#: the cap can be re-judged later without re-running the prototype blind.
RECORDED_FRAME_MS: float | None = 163.6


def tiles_touched(centroids, level, cell_src_px, plate_hw=PLATE_HW,
                  chunk_px=CHUNK_PX) -> set:
    """UNION of chunk indices the cells touch.

    The load-bearing correction. An earlier draft modelled per-cell 64x64
    crops as if each cell owned private pixels -- that is D3, the
    overlay-PNG-slicing world. In D1 every cell is a VIEWPORT ONTO THE SAME
    STORE, so tiles are SHARED between cells and the resident set is the
    union, bounded above by the whole level. That makes the bound
    closed-form and small rather than linear in cell count.

    The window is CLAMPED to the level's extent. Without the clamp a colony
    nearer the plate edge than half a cell -- the first column of an arrayed
    plate is 31 px in on the reference geometry -- puts chunk index -1 in the
    union, and the union then "exceeds" a level it never left. Measured:
    16 chunks against a 12-chunk ceiling, which is a modelling artefact and
    not a resident set the browser could ever hold.
    """
    h, w = plate_hw
    level_h, level_w = h / 2**level, w / 2**level
    touched = set()
    half = cell_src_px / 2
    for rr, cc in centroids:
        top = min(max((rr - half) / 2**level, 0.0), level_h - 1)
        bottom = min(max((rr + half) / 2**level, 0.0), level_h - 1)
        left = min(max((cc - half) / 2**level, 0.0), level_w - 1)
        right = min(max((cc + half) / 2**level, 0.0), level_w - 1)
        for ty in range(math.floor(top / chunk_px),
                        math.floor(bottom / chunk_px) + 1):
            for tx in range(math.floor(left / chunk_px),
                            math.floor(right / chunk_px) + 1):
                touched.add((ty, tx))
    return touched


def resident_bytes(n_tiles, n_series, chunk_px=CHUNK_PX, channels=3, itemsize=1):
    """Bytes the tile cache holds for one store."""
    return n_tiles * chunk_px**2 * channels * itemsize * n_series


def level_tiles(h, w, level, chunk_px=CHUNK_PX) -> int:
    """Every chunk at a level -- the ceiling the union cannot exceed."""
    return (math.ceil(h / 2**level / chunk_px)
            * math.ceil(w / 2**level / chunk_px))


def plate_centroids(rows=PLATE_ROWS, cols=PLATE_COLS, plate_hw=PLATE_HW):
    """Colony centroids on an evenly spaced arrayed plate."""
    h, w = plate_hw
    return [
        ((r + 0.5) * h / rows, (c + 0.5) * w / cols)
        for r in range(rows)
        for c in range(cols)
    ]


def main() -> int:
    h, w = PLATE_HW
    ceiling = level_tiles(h, w, 0)
    print(
        f"level-0 ceiling: {ceiling} tiles = "
        f"{resident_bytes(ceiling, 1) / 1e6:.0f} MB for the WHOLE level"
    )
    # The union claim, exercised rather than asserted in prose: every cell of
    # a full plate at once still touches no more chunks than the level holds.
    union = tiles_touched(plate_centroids(), level=0, cell_src_px=CELL_SRC_PX)
    print(
        f"{PLATE_CELLS} cells over ONE store touch {len(union)} distinct "
        f"level-0 chunks (ceiling {ceiling})"
    )
    if len(union) > ceiling:
        print("UNION EXCEEDS THE LEVEL: the resident-set bound is wrong")
        return 1
    for cells in (64, 256, 1024, PLATE_CELLS):
        print(f"{cells:5d} cells: {cells * VISIBLE_LAYERS:6d} draw calls/frame")
    if RECORDED_CAP is None or RECORDED_FRAME_MS is None:
        print("NO MEASUREMENT: run the prototype in task 4.1 step 3")
        return 1
    print(
        f"cap {RECORDED_CAP} cells, {RECORDED_FRAME_MS:.1f} ms/frame measured; "
        f"set maxCacheByteSize >= {resident_bytes(ceiling, VISIBLE_LAYERS) / 1e6:.0f} MB "
        f"per distinct store in the grid"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
