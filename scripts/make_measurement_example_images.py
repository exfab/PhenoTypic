"""Generate illustrative figures for MeasurementInfo ``image`` examples.

These are visualizations to aid understanding — not biological claims. Run:

    uv run python scripts/make_measurement_example_images.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from phenotypic.data import load_synth_yeast_plate  # noqa: E402
from phenotypic.detect import OtsuDetector  # noqa: E402

_OUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "phenotypic"
    / "_assets"
    / "measurements"
)


def _shape_area() -> None:
    """A colony crop with its detected pixel area shaded — illustrates Shape_Area."""
    image = load_synth_yeast_plate()
    detected = OtsuDetector(ignore_zeros=True).apply(image)
    objmap = detected.objmap[:]
    labels = [v for v in set(objmap.ravel().tolist()) if v != 0]
    target = max(labels, key=lambda v: int((objmap == v).sum()))
    mask = objmap == target

    rr, cc = mask.nonzero()
    pad = 15
    r0, r1 = max(int(rr.min()) - pad, 0), min(int(rr.max()) + pad, mask.shape[0])
    c0, c1 = max(int(cc.min()) - pad, 0), min(int(cc.max()) + pad, mask.shape[1])

    crop_rgb = detected.rgb[:][r0:r1, c0:c1]
    crop_mask = mask[r0:r1, c0:c1].astype(float)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
    ax.imshow(crop_rgb)
    # light fill marks the counted pixels; contour traces the measured boundary,
    # leaving the underlying colony visible.
    ax.imshow(crop_mask, cmap="cool", alpha=0.30 * crop_mask)
    ax.contour(crop_mask, levels=[0.5], colors="#d81b60", linewidths=1.2)
    ax.set_title(f"Shape_Area = {int(mask.sum())} px (shaded)", fontsize=9)
    ax.axis("off")
    dest = _OUT / "shape" / "area.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    print("wrote", dest)


if __name__ == "__main__":
    _shape_area()
