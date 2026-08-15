#!/usr/bin/env python
"""Re-derive the choice of contrast measure for the assay trait (§9.3.2).

The MCP-server spec proposes a probe-measured trait
``colony.contrast_vs_background``. An early draft specified Otsu's between-class
variance ratio η = σ²_B/σ²_T, on the reasoning that it is "precisely the
separability Otsu maximizes, so it is principled rather than invented".

That reasoning is correct about what η *is* and wrong about what the trait
*needs*. This script establishes four claims:

  C1  η is SCALE-INVARIANT. Shrinking image contrast toward the mean by a factor
      α leaves η numerically unchanged. So does Cohen's d. Both normalize by the
      very spread that reducing contrast shrinks — they measure histogram
      bimodality, not contrast magnitude.
  C2  Michelson contrast (μ_fg − μ_bg)/(μ_fg + μ_bg) at the Otsu split tracks α
      linearly, and is therefore a usable contrast measure.
  C3  On real plate images η has no usable dynamic range: whole-frame values sit
      at ~0.96–0.97 and per-cell values span ~2% of the nominal [0,1] scale.
      Three bands cannot be cut from that.
  C4  Whole-frame Otsu on a plate image splits PLATE vs SURROUND, not colony vs
      agar — which is why the one in-repo η implementation
      (``ReferenceFreeScorer._contrast``) needs an object mask. Per-grid-cell
      measurement is what removes that dependency.

Conclusion recorded in the spec: use per-cell Michelson, and keep the
categorical band human-sourced until a dataset spanning low contrast exists.

Depends only on the stdlib + numpy + scikit-image. Never imports ``phenotypic``
for the measurement itself (it optionally reads bundled plate TIFFs by path).

Exits non-zero on the first failed claim.

Run:  uv run python docs/superpowers/logic_validation_scripts/2026-08-12-phenotypic-mcp-server/contrast_trait_measure.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import NoReturn

import numpy as np
import skimage.filters
import skimage.io

PLATE_GLOB = "docs/source/_static/gui_images/_dataset/plates/*.tif"
GRID_ROWS, GRID_COLS = 8, 12
CROP_FRAC = 0.10  # trim 10% per side before per-cell measurement


def _fail(claim: str, detail: str) -> NoReturn:
    print(f"FAIL [{claim}] {detail}")
    sys.exit(1)


def _ok(claim: str, detail: str) -> None:
    print(f"ok   [{claim}] {detail}")


def _measures(a: np.ndarray) -> dict[str, float] | None:
    """η, Cohen's d, and Michelson contrast at the Otsu split."""
    a = np.asarray(a, dtype=np.float64)
    total_var = float(a.var())
    if total_var <= 0.0:
        return None
    t = float(skimage.filters.threshold_otsu(a))
    fg = a >= t
    w = float(fg.mean())
    if w <= 0.0 or w >= 1.0:
        return None
    mu_f, mu_b = float(a[fg].mean()), float(a[~fg].mean())
    sd_f, sd_b = float(a[fg].std()), float(a[~fg].std())
    return {
        "eta": w * (1.0 - w) * (mu_f - mu_b) ** 2 / total_var,
        "cohen_d": (mu_f - mu_b) / float(np.sqrt((sd_f**2 + sd_b**2) / 2.0 + 1e-12)),
        "michelson": (mu_f - mu_b) / (mu_f + mu_b + 1e-12),
        "fg_frac": w,
    }


def _gray(path: str) -> np.ndarray:
    im = skimage.io.imread(path)
    return im.mean(axis=2) if im.ndim == 3 else im.astype(np.float64)


def _plates() -> list[str]:
    return sorted(glob.glob(PLATE_GLOB))


def claim_1_and_2_scale_behaviour(g: np.ndarray) -> None:
    """η/d are invariant to contrast reduction; Michelson is proportional."""
    alphas = [1.0, 0.6, 0.3, 0.15, 0.05]
    rows = []
    mean = float(g.mean())
    for a in alphas:
        m = _measures(mean + (g - mean) * a)
        if m is None:
            _fail("C1", f"degenerate image at alpha={a}")
        rows.append((a, m))

    base = rows[0][1]
    for a, m in rows[1:]:
        if abs(m["eta"] - base["eta"]) > 1e-6:
            _fail("C1", f"eta moved with alpha={a} ({m['eta']:.6f} vs {base['eta']:.6f})")
        if abs(m["cohen_d"] - base["cohen_d"]) > 1e-6:
            _fail("C1", f"cohen_d moved with alpha={a}")
    _ok(
        "C1",
        f"eta={base['eta']:.3f} and cohen_d={base['cohen_d']:.3f} UNCHANGED across a "
        f"{alphas[0] / alphas[-1]:.0f}x contrast reduction — both are scale-invariant, "
        "so neither can measure contrast magnitude",
    )

    # Michelson must scale with alpha (ratio to alpha roughly constant).
    ratios = [m["michelson"] / a for a, m in rows]
    spread = (max(ratios) - min(ratios)) / max(ratios)
    if spread > 0.05:
        _fail("C2", f"michelson/alpha not constant (spread {spread:.3f}); not proportional")
    series = ", ".join(f"a={a}:{m['michelson']:.4f}" for a, m in rows)
    _ok("C2", f"michelson tracks contrast linearly ({series})")


def claim_3_eta_has_no_dynamic_range(plates: list[str]) -> None:
    whole = []
    for f in plates:
        m = _measures(_gray(f))
        if m is None:
            _fail("C3", f"degenerate plate {f}")
        whole.append(m["eta"])

    g = _gray(plates[0])
    h, w = g.shape
    dh, dw = int(h * CROP_FRAC), int(w * CROP_FRAC)
    c = g[dh : h - dh, dw : w - dw]
    ch, cw = c.shape
    cell_eta, cell_mich = [], []
    for r in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cell = c[
                r * ch // GRID_ROWS : (r + 1) * ch // GRID_ROWS,
                col * cw // GRID_COLS : (col + 1) * cw // GRID_COLS,
            ]
            m = _measures(cell)
            if m is not None:
                cell_eta.append(m["eta"])
                cell_mich.append(m["michelson"])

    if len(cell_eta) < GRID_ROWS * GRID_COLS // 2:
        _fail("C3", f"only {len(cell_eta)} usable cells")

    ce = np.array(cell_eta)
    p10, p90 = float(np.percentile(ce, 10)), float(np.percentile(ce, 90))
    band = p90 - p10
    if band > 0.10:
        _fail(
            "C3",
            f"per-cell eta p10-p90 span is {band:.3f} — wide enough to band after all; "
            "the spec's rejection of eta would need revisiting",
        )
    _ok(
        "C3",
        f"whole-frame eta {min(whole):.3f}-{max(whole):.3f} across {len(whole)} plates; "
        f"per-cell p10-p90 = {p10:.3f}-{p90:.3f} (span {band:.3f}, "
        f"{band * 100:.1f}% of the nominal [0,1] scale) — cannot support three bands",
    )
    print(
        f"       per-cell michelson: median={np.median(cell_mich):.4f} "
        f"p10={np.percentile(cell_mich, 10):.4f} p90={np.percentile(cell_mich, 90):.4f} "
        "  <- the anchor these plates provide (visually high-contrast)"
    )


def claim_4_whole_frame_splits_plate_not_colony(plates: list[str]) -> None:
    """A whole-frame Otsu split puts a huge fraction in 'foreground'.

    Colonies occupy a small share of a plate image. A foreground fraction near
    half means the split is separating the plate disc from the surround, not
    colonies from agar.
    """
    m = _measures(_gray(plates[0]))
    if m is None:
        _fail("C4", "degenerate plate")
    if m["fg_frac"] < 0.20:
        _fail(
            "C4",
            f"whole-frame fg fraction is {m['fg_frac']:.3f} — small enough to plausibly "
            "be colonies, so the plate-vs-surround argument does not hold here",
        )
    _ok(
        "C4",
        f"whole-frame Otsu puts {m['fg_frac']:.1%} of pixels in 'foreground' — far more "
        "than colonies occupy, confirming the split is plate vs surround. This is why "
        "ReferenceFreeScorer._contrast needs an object mask, and why the trait must be "
        "measured per grid cell instead.",
    )


def main() -> int:
    plates = _plates()
    if not plates:
        _fail("C0", f"no plate images found at {PLATE_GLOB} (run from the repo root)")
    print(f"# plates: {len(plates)} from {Path(PLATE_GLOB).parent}")

    g = _gray(plates[0])
    claim_1_and_2_scale_behaviour(g)
    claim_3_eta_has_no_dynamic_range(plates)
    claim_4_whole_frame_splits_plate_not_colony(plates)

    print(
        "\nAll claims re-derived.\n"
        "CONCLUSION: eta is the wrong measure for this trait — it is invariant to\n"
        "contrast. Use per-cell Michelson. Only high-contrast plates are available\n"
        "here, so the categorical band stays human-sourced (§9.3.2)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
