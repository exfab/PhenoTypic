# Tensor-voting mutation evidence

Each mutant was applied alone to `_tensor_voting.py` with `apply_patch`, the named focused
pytest command was executed, and the patch was reversed before the next mutant. `Max delta` is
the largest absolute difference from any corresponding MATLAB fixture array. A non-fixture
behavioral mutant reports the maximum fixture delta measured separately while the mutant was
active.

| Mutant | One semantic change | Command and killing test | Max delta | Result |
|---|---|---|---:|---|
| TV-M01 | replace squared arc length with zero | focused pytest `test_source_generated_fixture_matches_all_outputs` | 2.84924160668 | KILLED |
| TV-M02 | replace curvature penalty with zero | focused pytest `test_source_generated_fixture_matches_all_outputs` | 0.0779734846124 | KILLED |
| TV-M03 | use row offset for x | focused pytest `test_rotation_and_transpose_controls` | 2.89348375873 | KILLED |
| TV-M04 | rotate the decomposed normal by pi/2 before tangent conversion | focused pytest `test_source_generated_fixture_matches_all_outputs` | 1.5 | KILLED |
| TV-M05 | change fourfold source angle to unscaled angle | focused pytest `test_source_generated_fixture_matches_all_outputs` | 0.42323082139 | KILLED |
| TV-M06 | replace amplitude-times-decay with unit-amplitude decay | focused pytest `test_positive_linearity_for_fixed_active_mask` | 0.5 | KILLED |
| TV-M07 | overwrite component `d` rather than accumulate | focused pytest `test_source_generated_fixture_matches_all_outputs` | 1.5 | KILLED |
| TV-M08 | remove retained input from component `d` | focused pytest `test_isolated_token_has_retained_and_cast_self_vote` | 1.5 | KILLED |
| TV-M09 | skip the voting-field center | focused pytest `test_isolated_token_has_retained_and_cast_self_vote` | 1.5 | KILLED |
| TV-M10 | traverse full support and wrap target indices modulo image shape | focused pytest `test_boundary_vote_is_cropped_not_wrapped` | 0.253519973109 | KILLED after closing a test hole |
| TV-M11 | assign minor eigenvalue to stick output | focused pytest `test_closed_form_saliency_decomposition` | 3.0 | KILLED |
| TV-M12 | accumulate component `d` in float32 | focused pytest `test_source_generated_fixture_matches_all_outputs` | 5.64582376406e-08 | KILLED |
| TV-M13 | normalize stick by its observed maximum | focused pytest `test_positive_linearity_for_fixed_active_mask` | 2.0 | KILLED |
| TV-M14 | negate the signed local angle in the vote direction | focused pytest `test_source_generated_fixture_matches_all_outputs` | 0.0457775168014 | KILLED |
| TV-M15 | replace the source's nested support-window rounding with a direct ceiling | focused pytest `test_source_window_radius_matches_nested_rounding_boundary` | 0 on the fixture; targeted `sigma=0.52` radius changes from 1 to 2 | KILLED |

The load-bearing historical mutant is TV-M05 because the selected archive explicitly documents
line 70 as a fork. The source fixture must fail if that factor is removed.

TV-M10 initially survived because the boundary control checked only the opposite diagonal, which
the narrow stick aperture also suppresses. The control was strengthened to check the wrapped
same-row pixel; the same unchanged mutant then failed with a response of `0.82075480829826808`.
This is the only discovered test hole, and it was closed before the baseline gate.

After reversing TV-M15, the clean baseline was verified with 19 focused tests, the independent
logic-validation script, focused mypy, and ruff. All four commands passed. A source comparison on
that restored baseline observed maximum component/stick error `2.22044604925e-16` and maximum
ball error `5.55111512313e-17` against MATLAB R2023b.
