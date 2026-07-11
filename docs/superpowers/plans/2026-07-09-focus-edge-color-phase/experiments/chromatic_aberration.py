#!/usr/bin/env python3
"""Spec §7.2 -- the acceptance criterion for the colour design.

Does cross-channel fusion buy anything under lateral chromatic aberration?

Prediction on record (`color-phase-congruency.md` §7.2): *"``joint`` merges the displaced
edges, so its error stays roughly flat in ``delta``, while ``l2`` degrades. **If this fails,
ship ``l2``.** Record the result either way; a null result must not be buried."*

This imports ``phenotypic``, so it is deliberately **not** a logic-validation script and does
not live under ``logic_validation_scripts/``.

Two methodological notes, both learned the hard way:

**``delta`` is the shift at the image corner, not at the object.** The aberration is radial,
so an edge at radius ``r`` moves by ``delta * r / r_max``. Verified against a synthetic
annulus: at ``delta = 10`` the R and B edges of a disc at ``r = 200`` (``r_max = 424``) split
by exactly ``10`` px at the corner scale, i.e. ``4.7`` px at the disc.

**The metric must be band-restricted.** A first version thresholded each response at its Otsu
level and averaged the distance-to-truth over the surviving pixels. That measures *detection
density*, not localization: a response spread uniformly over the plate scores ``26.3`` px on
the filamentous plate, and every method scored ``21``--``38``. Worse, ``coherent`` "won" at
``delta = 0`` (``7.3``) purely by detecting fewer pixels. The filamentous ``objmap`` marks only
the colony **cores** (median area 482 px, radius ~12 px) while hyphae radiate far beyond them,
so most of a correct response is legitimately far from any labelled boundary.

Restricting to a 6 px band around the true boundary and taking the response-weighted mean
distance measures where the response *concentrates*, independent of how much of the plate it
fires on. Its controls behave: the **G channel is the CA reference and must be flat in
``delta``** -- it is, at ``1.0992`` on every step -- while R and B degrade monotonically.

Run:
    timeout 3000 uv run python docs/superpowers/plans/2026-07-09-focus-edge-color-phase/\
experiments/chromatic_aberration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.segmentation import find_boundaries

import phenotypic
from phenotypic.data import load_synth_filamentous_plate, load_synth_yeast_plate
from phenotypic.enhance import (
    FocusEdgeColorPhase,
    FocusEdgeMonogenicPhase,
    FocusEdgePhase,
)
from phenotypic.enhance._monogenic_kernels import monogenic_phase_congruency

DELTAS = (0, 1, 2, 3)
BAND_RADIUS = 6.0
RESULTS_PATH = Path(__file__).with_name("chromatic_aberration_results.md")


def inject_radial_chromatic_aberration(rgb: np.ndarray, delta: float) -> np.ndarray:
    """Magnify R by ``1 + delta/r_max`` and shrink B by the same, about the image centre.

    Lateral CA: the same physical edge lands at a different radius in each channel. Green is
    the reference and is untouched. Bilinear resampling, edge-clamped.

    ``delta`` is the displacement **at the corner** (``r = r_max``). An edge at radius ``r``
    moves by ``delta * r / r_max``.
    """
    if delta == 0:
        return rgb.copy()

    rows, cols = rgb.shape[:2]
    centre = np.array([(rows - 1) / 2.0, (cols - 1) / 2.0])
    r_max = float(np.hypot(*centre))

    grid = np.indices((rows, cols), dtype=np.float64)
    offset = grid - centre[:, None, None]

    out = np.empty_like(rgb, dtype=np.float64)
    for channel, scale in ((0, 1.0 + delta / r_max), (1, 1.0), (2, 1.0 - delta / r_max)):
        if scale == 1.0:
            out[..., channel] = rgb[..., channel]
            continue
        # To magnify the OUTPUT by `scale`, sample the INPUT at offset/scale.
        source = centre[:, None, None] + offset / scale
        out[..., channel] = ndimage.map_coordinates(
            rgb[..., channel].astype(np.float64), source, order=1, mode="nearest"
        )

    return np.clip(np.rint(out), 0, 255).astype(rgb.dtype)


def band_localization_error(
        response: np.ndarray, distance: np.ndarray, band: np.ndarray
) -> float:
    """Response-weighted mean distance to the true boundary, within a band around it.

    Lower is better. Independent of how much of the plate the response fires on, which is
    what makes it a *localization* measure rather than a detection-rate one.
    """
    weights = response[band]
    weights = np.where(np.isfinite(weights), weights, 0.0)
    if weights.sum() <= 0.0:
        return float("nan")
    return float((weights * distance[band]).sum() / weights.sum())


def _methods(image) -> dict[str, np.ndarray]:
    """Response maps. Colour maps come from the **un-clipped** helper (drift ``C3``)."""
    rgb = image.rgb[:]
    maps: dict[str, np.ndarray] = {
        "FocusEdgePhase (baseline)": FocusEdgePhase()
        .apply(phenotypic.Image(rgb))
        .detect_mat[:]
        .astype(np.float64),
        "FocusEdgeMonogenicPhase (luminance)": FocusEdgeMonogenicPhase()
        .apply(phenotypic.Image(rgb))
        .detect_mat[:]
        .astype(np.float64),
    }
    for fusion in ("l2", "joint", "coherent"):
        maps[f"FocusEdgeColorPhase ({fusion})"] = FocusEdgeColorPhase(
            fusion=fusion
        )._color_phase_congruency(phenotypic.Image(rgb)).pc

    # Controls. G is the CA reference channel and MUST be flat in delta; R and B MUST
    # degrade. If these misbehave, the metric is broken and nothing below means anything.
    for index, channel in enumerate("RGB"):
        maps[f"control: monogenic PC on {channel} alone"] = monogenic_phase_congruency(
            rgb[..., index].astype(np.float64)
        ).pc
    return maps


def _run(name: str, loader) -> tuple[str, dict[str, list[float]], list[str]]:
    plate = loader()
    rgb, objmap = plate.rgb[:], plate.objmap[:]
    truth = find_boundaries(objmap, mode="inner")
    distance = ndimage.distance_transform_edt(~truth)
    band = distance <= BAND_RADIUS

    print(f"=== {name}: {rgb.shape[:2]}, {int(objmap.max())} objects, "
          f"{int(band.sum())} band px ===")

    order: list[str] = []
    table: dict[str, list[float]] = {}
    for delta in DELTAS:
        aberrated = phenotypic.Image(inject_radial_chromatic_aberration(rgb, delta))
        for method, response in _methods(aberrated).items():
            table.setdefault(method, []).append(
                band_localization_error(response, distance, band)
            )
            if method not in order:
                order.append(method)
        print(f"  delta = {delta} px done")
    return name, table, order


def _format(name: str, table: dict[str, list[float]], order: list[str]) -> str:
    lines = [
        f"### {name}",
        "",
        "| method | " + " | ".join(f"δ = {d}" for d in DELTAS) + " | slope |",
        "|---" * (len(DELTAS) + 2) + "|",
    ]
    for method in order:
        errors = table[method]
        slope = float(np.polyfit(DELTAS, errors, 1)[0])
        cells = " | ".join(f"{e:.4f}" for e in errors)
        lines.append(f"| `{method}` | {cells} | `{slope:+.4f}` |")
    return "\n".join(lines)


def main() -> int:
    sections, verdicts = [], []
    for name, loader in (
        ("filamentous plate", load_synth_filamentous_plate),
        ("yeast plate", load_synth_yeast_plate),
    ):
        name, table, order = _run(name, loader)

        green = np.array(table["control: monogenic PC on G alone"])
        if float(np.ptp(green)) > 1e-9:
            print(f"  CONTROL FAILED on {name}: the G channel is the CA reference and must "
                  f"be flat, but it moved by {np.ptp(green):.3e}. The metric is not "
                  f"measuring localization; nothing below is meaningful.")
            return 1

        red_slope = float(np.polyfit(DELTAS, table["control: monogenic PC on R alone"], 1)[0])
        if red_slope <= 0.0:
            print(f"  CONTROL FAILED on {name}: the R channel is displaced by construction "
                  f"and must degrade, but its slope is {red_slope:+.4f}.")
            return 1

        joint = float(np.polyfit(DELTAS, table["FocusEdgeColorPhase (joint)"], 1)[0])
        l2 = float(np.polyfit(DELTAS, table["FocusEdgeColorPhase (l2)"], 1)[0])
        verdicts.append((name, joint, l2))
        sections.append(_format(name, table, order))
        print()

    print("\n".join(sections))
    print()

    for name, joint, l2 in verdicts:
        print(f"{name}: joint slope {joint:+.4f}, l2 slope {l2:+.4f} -> "
              f"{'joint flatter' if joint < l2 else 'l2 flatter'}")

    # This script REPORTS. It does not decide. §7.2's rule turned out to test the wrong
    # statistic, and the decision that resolved it was a scoping decision, taken by a human
    # and recorded below. Rubber-stamping it here would hide that.
    decision = (
        "**The slope prediction FAILED on the yeast plate and HELD on the filamentous "
        "plate.** `joint` is nevertheless better localized than `l2` at *every* `δ` on "
        "*both* plates -- `l2`'s flat slope is largely an artifact of its already-poor "
        "localization (`1.60`--`1.78`), which leaves it little room to degrade.\n\n"
        "**Decided 2026-07-10 (user): keep `fusion=\"joint\"`, and scope the operation to "
        "filamentous plates.** This is a *scoping* decision, not a post-hoc swap of the "
        "test statistic: on the plate the operation is for, the prediction held on both the "
        "slope and the absolute error.\n\n"
        "**The null result, recorded rather than buried (§7.2, spec risk #2).** On the yeast "
        "plate at `δ = 3`, plain `FocusEdgeMonogenicPhase` on luminance scores `1.1433` and "
        "beats *every* fusion mode -- `joint` `1.3748`, `coherent` `1.7003`, `l2` `1.7759`. "
        "**On round-colony plates, colour buys nothing under chromatic aberration.** The "
        "measured benefit of `FocusEdgeColorPhase` is confined to the filamentous plate, "
        "where `joint` reaches `1.0083` against luminance's `1.1579`.\n\n"
        "The mechanism is not mysterious. Lateral CA *creates chromatic edges* -- that is "
        "what it is. `joint` asserts them coherently, so its detected edge follows the "
        "displaced chroma; `l2` combines three finished maps and its undisturbed luminance "
        "term survives the root-sum-of-squares. Spec §3.3's claim that joint's coherent "
        "summation 'merges the displaced edges into one response near the "
        "amplitude-weighted centroid' is **not** what the measurement shows on the yeast "
        "plate: joint's error grows five times faster than luminance-only's."
    )
    print()
    print("DECISION (recorded, not computed): keep fusion='joint'; scope to filamentous.")

    RESULTS_PATH.write_text(
        "# §7.2 — the chromatic-aberration experiment\n\n"
        "Response-weighted mean distance, in pixels, from the edge response to the true\n"
        f"object boundary, restricted to a **{BAND_RADIUS:.0f} px band** around that boundary.\n"
        "**Lower is better.** `slope` is a least-squares fit of the error against `δ`; a flat\n"
        "slope means the method is insensitive to lateral chromatic aberration.\n\n"
        "`δ` is the displacement **at the image corner**. An edge at radius `r` moves by\n"
        "`δ · r / r_max`, so the shift at a colony is smaller than `δ`.\n\n"
        "Colour responses come from the **un-clipped** `_color_phase_congruency` helper\n"
        "(drift `C3`). The three `control:` rows are the guard: **G is the CA reference\n"
        "channel and must be flat in `δ`**, while R and B are displaced by construction and\n"
        "must degrade. The script exits non-zero if either control misbehaves, because a\n"
        "metric that cannot see a displacement it created cannot see anything.\n\n"
        + "\n\n".join(sections)
        + "\n\n## Slopes\n\n"
        + "\n".join(
            f"- **{name}**: `joint` slope `{joint:+.4f}`, `l2` slope `{l2:+.4f}`"
            for name, joint, l2 in verdicts
        )
        + f"\n\n## Verdict\n\n{decision}\n",
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
