# Prototype reference

The working implementation `CalibrateColorRpcc` was ported from, copied here
unmodified from `SnP-ColorCorrection/chipdetect/` so the port can be read
against its source.

**Nothing here is imported, installed, or executed by PhenoTypic.** It is
reference material. It is rig-specific in ways the package deliberately is not
— hard-coded band coordinates, a stored prior keyed to one frame, and a
registration window per side.

| File | What it is |
|---|---|
| `chipdetect.py` | The detector family: seven lattice-refinement methods, the identity assignment, tile measurement and the QC gate. |
| `chiprig.py` | Rig constants — frame paths, band geometry, session dates. Not ported. |
| `rig_prior.json` | The stored lattice, keyed to frame `d000220_300_021` and the `InvertLinearLMMSE.pp3` develop profile. |
| `ColorChecker24_After_Nov2014.txt` | The chart reference values the prototype read. The package uses `colour.CCS_COLOURCHECKERS` instead. |
| `chip_detection_methods.md` | The method comparison the spec's capture-range table comes from. |
| `chip_extraction_summary.md` | The extraction write-up, including the crop-width failure that motivated native detection. |
| `prototype_README.md` | The prototype's own README (`chipdetect/README.md`). |

## What was carried across, and what was left

**Ported:** the lattice model, the rigid and ECC refinements, the
whole-placement identity scoring with its orientation hypotheses, the
impurity / robust-shift / clipping statistics, and the QC limits.

**Left behind, deliberately:**

- `M3_blob` and `M23_cascade` — 1.419 and 1.825 s per band against the rigid
  snap's 0.357, with no measured gain.
- Phase-whitened cross-correlation — recovers synthetic shifts exactly but
  collapses the horizontal estimate to zero on real cross-session pairs, a
  failure that hides behind a perfect synthetic test.
- The linear-RGB-into-`skimage.rgb2lab` measurement path. The prototype's own
  docstring flags that its Lab values are an internal contrast measure rather
  than calibrated CIE; the package converts once, correctly, via
  `Image.color.Lab`.
- Every rig-specific constant: `REG_WIN`, `ANCHOR_COL`, `BAND_Y`, `BAND_W`,
  `LAYOUT`. The ROI rectangle and the detected lattice carry that information
  in the package.

**Changed during the port**, with the reason in the code:

- The candidate-restricted ΔE2000 medoid replaces the seeded-subsample one.
- The anchor search window is clamped to the midpoint between columns, so a
  wide search on a densely-packed card cannot lock onto a neighbour.
- Column detection thresholds relative to the profile background rather than
  its midpoint, which is what finds the low-chroma neutral column.
- Identity margin gates at 0.20 rather than 0.1, because identity now selects
  each tile's reference colour instead of cross-checking a declared layout.
