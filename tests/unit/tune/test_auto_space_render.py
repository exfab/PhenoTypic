"""Pure review-table rendering for ``--auto-space`` (P3-6).

``_render_review_table`` is a **pure** ``InferredSearchSpace -> str`` function
(no Dash, no I/O) so it is trivially unit-testable: trusted knobs render with a
``✓`` marker, ``needs_review`` knobs with ``⚠``, and excluded fields with their
reason. It drives the (non-blocking) terminal summary the CLI prints.
"""
from __future__ import annotations

from phenotypic.tune import (
    Categorical,
    Excluded,
    FloatRange,
    InferredSearchSpace,
    Knob,
)
from phenotypic.tune._tune_cli._auto_space import _render_review_table


def _proposal() -> InferredSearchSpace:
    return InferredSearchSpace(
        knobs=(
            Knob(
                key="0.sigma",
                domain=FloatRange(low=0.5, high=5.0),
                source="tune_spec",
                needs_review=False,
                description="Blur sigma.",
            ),
            Knob(
                key="1.mu",
                domain=FloatRange(low=0.1, high=1.6),
                source="unbounded_heuristic",
                needs_review=True,
                description="Chan-Vese mu.",
            ),
            Knob(
                key="1.ignore_zeros",
                domain=Categorical(choices=(True, False)),
                source="bool",
                needs_review=False,
            ),
        ),
        excluded=(
            Excluded(key="1.kernel", reason="ndarray", field_type="ndarray"),
            Excluded(key="2.label", reason="name_ref", field_type="ColumnRef"),
        ),
    )


def test_render_returns_a_string():
    out = _render_review_table(_proposal())
    assert isinstance(out, str)
    assert out  # non-empty


def test_trusted_knob_rendered_with_check_mark():
    out = _render_review_table(_proposal())
    # the trusted knob line carries the ✓ marker and its key
    trusted = [ln for ln in out.splitlines() if "0.sigma" in ln]
    assert trusted and "✓" in trusted[0]


def test_needs_review_knob_rendered_with_warning_mark():
    out = _render_review_table(_proposal())
    review = [ln for ln in out.splitlines() if "1.mu" in ln]
    assert review and "⚠" in review[0]


def test_excluded_field_renders_with_its_reason():
    out = _render_review_table(_proposal())
    kernel = [ln for ln in out.splitlines() if "1.kernel" in ln]
    assert kernel and "ndarray" in kernel[0]
    label = [ln for ln in out.splitlines() if "2.label" in ln]
    assert label and "name_ref" in label[0]


def test_render_lists_every_knob_and_excluded_key():
    out = _render_review_table(_proposal())
    for key in ("0.sigma", "1.mu", "1.ignore_zeros", "1.kernel", "2.label"):
        assert key in out


def test_render_is_pure_and_repeatable():
    proposal = _proposal()
    assert _render_review_table(proposal) == _render_review_table(proposal)
