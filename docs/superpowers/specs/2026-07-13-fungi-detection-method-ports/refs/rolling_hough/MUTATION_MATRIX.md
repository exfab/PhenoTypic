# A09 Rolling Hough mutation evidence

The executable runner copies the reviewed production module to a temporary directory, applies one
unique textual mutation at a time, runs the named killing probe, and verifies the production file's
SHA-256 after every mutant has been evaluated. Run it from the repository root with:

```bash
uv run python \
  docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/rolling_hough/run_mutations.py
```

| Mutant | One semantic change | Named killing test | Result |
|---|---|---|---|
| RH-M01 | Treat positive infinity as a good pixel | `test_nonfinite_pixels_invalidate_both_source_halos` | KILLED |
| RH-M02 | Shrink the inclusive smoothing footprint by one radius step | `test_source_fixture_matches_every_core_output_and_intermediate` | KILLED |
| RH-M03 | Replace reflect correlation with constant-zero correlation | `test_source_fixture_matches_every_core_output_and_intermediate` | KILLED |
| RH-M04 | Change strict-positive unsharp threshold to nonnegative | `test_constant_image_returns_defined_empty_result` | KILLED |
| RH-M05 | Omit the second circular bad-pixel halo | `test_nonfinite_pixels_invalidate_both_source_halos` | KILLED |
| RH-M06 | Compute theta count from diameter instead of diameter minus one | `test_outputs_have_frozen_shapes_and_dtypes` | KILLED |
| RH-M07 | Include the pi endpoint in the theta grid | `test_source_fixture_matches_every_core_output_and_intermediate` | KILLED |
| RH-M08 | Swap row and column in the Hough normal equation | `test_diameter_eleven_geometry_counts_and_angles_match_source` | KILLED |
| RH-M09 | Replace round-to-nearest-even with floor | `test_geometry_contains_round_to_nearest_even_half_ties` | KILLED |
| RH-M10 | Replace angle-dependent support with circle area | `test_source_fixture_matches_every_core_output_and_intermediate` | KILLED |
| RH-M11 | Omit division by support before threshold subtraction | `test_source_fixture_matches_every_core_output_and_intermediate` | KILLED |
| RH-M12 | Replace the source multiplicative gate with `maximum`, losing negative zero | `test_threshold_equality_is_zero_and_rejected_values_keep_negative_zero` | KILLED |
| RH-M13 | Derive dense validity from raw counts instead of positive residuals | `test_constant_image_returns_defined_empty_result` | KILLED |
| RH-M14 | Globally normalize the raw response | `test_raw_response_is_not_globally_normalized` | KILLED |
| RH-M15 | Convert the Hough normal to a tangent angle | `test_orientation_is_axial_hough_normal_with_source_mapping` | KILLED |
| RH-M16 | Use pi instead of NaN for invalid orientation | `test_constant_image_returns_defined_empty_result` | KILLED |
| RH-M17 | Store raw counts as int32 instead of frozen int64 | `test_outputs_have_frozen_shapes_and_dtypes` | KILLED |
| RH-M18 | Silently convert float32 input at the arithmetic boundary | `test_invalid_images_raise` | KILLED |

The load-bearing transcription bug is RH-M09 because the exact discrete rho-zero support depends on
the source's NumPy round-to-nearest-even operation. The all-output fixture and direct geometry probe
both fail when it is replaced with floor.
