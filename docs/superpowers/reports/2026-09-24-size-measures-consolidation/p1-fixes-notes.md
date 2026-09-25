# Phase-1 review fixes: notes

Fixes for `phase1-impl-test-review.md` (HIGH-1 option A, MEDIUM-1, LOW-1..LOW-7), plus
the orientation-zone golden that the phase-1 gate found red. Probe scripts and their
outputs are in the session scratchpad: `p1fix_probe.py`, `p1fix_probe2.py`,
`p1fix_probe3.py` and the matching `*_out.txt`.

## HIGH-1: what shipped, and two findings made on the way

`_trace_radial_signature` now pools every contour of the **hole-filled** label, traced
`fully_connected="high"`. The plateau is labelled **8-connected**.

1. **Pooling alone let hole vertices win bins.** The review said that "a hole's contour
   can never win a bin", but that holds per ray, not per bin. An outline of radius R has
   8R + 4 vertices. Below R ≈ 45 that is fewer than the 360 bins, so the outer outline
   leaves bins empty, and the hole's vertices filled some of them with the hole's radius.
   Measured, from the plateau centre without the fill: 18 bins won on a ring of radius 12,
   16 on radius 20, 1 on radius 30, 0 on radius 60. The synthetic plate has 96 Otsu
   colonies with holes, and 87 of them moved by more than 0.05 px. The fix is
   `binary_fill_holes` on the contour input only. The centre and InscribedRadius still use
   the unfilled `props.image`, so MEDIUM-1 is unaffected. With the fill, the ring's
   signature equals the old tip's exactly (probe 2, H1). The orchestrator accepted this
   as the way to deliver option A as presented.
2. **The plateau was labelled 4-connected.** A diagonal run of tied pixels split into
   single-pixel components, so the centre of a diagonal 1-px line was its end pixel. The
   user's choice ("treat diagonal contacts as connected") contradicts that, so it is now
   `structure=np.ones((3, 3))`. Effect on the synthetic plate: the centre moves by more
   than 0.05 px on 21 of 552 Otsu objects and on 16 of 452 after NearestNeighborMerger. Any
   radius moves by more than 0.05 px on 20 and 15 of them. The largest moves are specks of
   2 to 4 px (1.4 px). Two real colonies (areas 993 and 972) move by 0.70 px.

**`fully_connected="high"` is an equivalent mutant.** Once contours are pooled, the
connectivity choice only decides how a saddle cell's four edge crossings pair up into
contours (skimage `_find_contours.py:60-71`). It never changes which crossings exist, and
`np.maximum.at` does not depend on order. Probe 1 B confirms it on every synthetic-plate
object: the vertex sets are equal (552/552 and 452/452) and the signatures are equal. The
setting is kept (user decision, and it matches the labelling), and the docstring, spec
§4.1 and the test-section comment say that it cannot change the result. No test can kill
the "drop high" mutant, and none is claimed to.

## Existing test values that moved

Only one existing test's values moved. The **crescent**
(`test_crescent_keeps_the_radius_ordering`) went from [21.315, 28.199, 24.439, 62.310] to
[21.988, 28.125, 24.522, 59.195] (median/mean/robust/max). Its plateau centre moved from
(33.5, 16.0) to (40.0, 15.47) once 8-connected. The test asserts only the ordering, quotes
no values, and still passes.

Disk, runner, wide runner, rectangle, ellipse, the degenerate cases and runner@16 are
bit-identical (max |d| = 0).

## New tests

In `test_radial_profile.py`, through `MeasureSize().measure`:
- Separate line. The report's P3b geometry; this is row 1.
- Diagonal runner. Row 3.
- Diagonal line. Row 4. It now expects a centre at the middle pixel, with MaxRadius
  hypot(7.5, 7) = 10.26. The report's 20.16 was the 4-connected end-pixel centre.
- `test_a_hole_never_supplies_a_bin`, which drives `_trace_radial_signature` directly and
  pins the centre through the `edt` argument.

The probe values agree with the report's: 15.19 / 40.51 / 19.74 / 214.27 and 75.31. Every
asserted value comes from a stated mechanism, either exact geometry or a bound. None is
copied from a probe.

MEDIUM-1: `test_measure_shape.py::test_a_hole_counts_as_background_for_boundary_distances`
and `test_measure_size.py::test_a_hole_counts_as_background_for_the_inscribed_radius` use
main's whole-image EDT as the oracle.

LOW-5: `test_measure_size.py::test_rows_align_with_labels_that_are_non_contiguous_and_out_of_raster_order`.
LOW-7: `test_keep_section_largest.py::test_an_empty_plate_fails_with_runtime_error`.
LOW-1: `test_change_note.py::test_enum_docstring_dedents_to_one_margin`.

## Deviations from the report's proposals

- **LOW-1.** The fix is one shared helper, `append_change_note`, in
  `schema/_change_notes.py` (stdlib `inspect`), not two inline `cleandoc` calls.
  `append_rst_to_doc` is unchanged, as the report advised. The orchestrator's docs build
  of 3c91e880 confirmed the defect on the rendered page: one extra `<blockquote>` on each
  of the SIZE and SHAPE API pages.
- **LOW-6.** The docstring line in `_measure_intensity.py` is fixed now, not left to
  Task 7. The `_cli_output_manager.py` example gains a `...`, so that it reads as an
  excerpt. That is a hand edit, and no Shape token is involved.
- **LOW-7.** The report said "no fix needed". A test now pins the outer `RuntimeError`.
  The cause chain (`RuntimeError` ← `Exception` ← `NoObjectsError`) is recorded in the
  test's docstring and not asserted.
- **MEDIUM-1 ring.** The report suggested hard-coded 4.8374 / 5.0 / sqrt(85). The tests
  assert against the scipy whole-image EDT oracle instead, and guard the oracle with
  sqrt(85).
- **Logic-validation script.** It gains checks 07 (fragmented label), 08 (per ray, a hole
  crossing is always inner) and 09 (why the fill is needed: disk outlines have 8R + 4
  vertices, and at R = 12 some bins hold only hole vertices). Check 09 computes the
  vertices as edge midpoints with numpy, which marching squares at level 0.5 on a binary
  mask reduces to. So the script still depends only on the stdlib, numpy and scipy.
- **(c) orientation-zone golden.** The test drops each serialized pipeline's `version`
  stamp from both sides, compares the rest, then asserts that the running stamp is
  `phenotypic.__version__`. The golden JSON is untouched, and the wrong sentence in spec §7
  is corrected. The one skipped test in that file,
  `test_capture_orientation_zone_pre_simplification_golden`, was already skipped: it only
  runs with `PHENOTYPIC_CAPTURE_GOLDEN=1`.
