"""``--auto-space`` — infer a reviewable search space from a pipeline.

``run_auto_space`` mines a configured ``ImagePipeline`` with
:func:`~phenotypic.tune.infer_search_space` and persists the resulting
:class:`~phenotypic.tune.InferredSearchSpace` proposal to
``deliverables/tuning_spec.json`` (the proposal round-trips through JSON). It
runs **no** engine — there is no ``trials.parquet``, no scoring, no winner. The
point is to hand the user a generous, flagged-for-review candidate space they
can edit before launching ``run``.

:func:`_render_review_table` is a **pure** ``InferredSearchSpace -> str``
function (no Dash, no I/O) that produces the non-blocking terminal summary:
trusted knobs (``✓``), knobs needing review (``⚠``), and excluded fields with
their reason.

Example:
    >>> from phenotypic import ImagePipeline
    >>> from phenotypic.enhance import GaussianBlur
    >>> from phenotypic.detect import OtsuDetector
    >>> pipe = ImagePipeline(ops=[GaussianBlur(sigma=2.0), OtsuDetector()])
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from phenotypic.tools_ import _io_constants as io
    >>> with TemporaryDirectory() as d:
    ...     proposal = run_auto_space(pipe, d)
    ...     io.tuning_spec_path(Path(d)).exists()
    True
    >>> "✓" in _render_review_table(proposal)
    True
"""
from __future__ import annotations

from pathlib import Path

from phenotypic.tools_ import _io_constants as io
from phenotypic.tools_ import atomic_write_text

from .._search_space import InferredSearchSpace, infer_search_space

#: Marker glyphs for the review table (a closed set; never bare strings inline).
_TRUSTED_MARK = "✓"
_REVIEW_MARK = "⚠"
_EXCLUDED_MARK = "✗"


def run_auto_space(pipeline: object, output_dir: str | Path) -> InferredSearchSpace:
    """Infer a search-space proposal from ``pipeline`` and persist it.

    Mines ``pipeline`` with :func:`~phenotypic.tune.infer_search_space` (one-level
    nested recursion on) and writes the proposal to
    ``deliverables/tuning_spec.json`` under ``output_dir``. No engine is run, so
    no ``trials.parquet`` / ``best_pipeline.json`` / ``param_importance.json`` is
    produced — this is the inspect-before-you-tune step.

    Args:
        pipeline: A live ``ImagePipeline`` whose ops are mined for tunable
            fields.
        output_dir: The run directory; the proposal lands at
            ``io.tuning_spec_path(output_dir)``.

    Returns:
        The :class:`~phenotypic.tune.InferredSearchSpace` proposal (also written
        to disk).
    """
    output_dir = Path(output_dir)
    io.deliverables_dir(output_dir).mkdir(parents=True, exist_ok=True)
    proposal = infer_search_space(pipeline)
    atomic_write_text(
        io.tuning_spec_path(output_dir), proposal.model_dump_json(indent=2)
    )
    return proposal


def _render_review_table(inferred: InferredSearchSpace) -> str:
    """Render a proposal as a plain-text review table (pure; no I/O).

    Each knob renders on one line marked ``✓`` (trusted) or ``⚠`` (needs review)
    with its key, domain kind, and provenance source. Each excluded field renders
    marked ``✗`` with its key and exclusion reason. A trailing summary line counts
    trusted / review / excluded entries.

    Args:
        inferred: The proposal to render.

    Returns:
        A newline-joined table string (always non-empty — at least the header
        and summary lines are present).
    """
    lines: list[str] = ["Inferred search space:"]
    for knob in inferred.knobs:
        mark = _REVIEW_MARK if knob.needs_review else _TRUSTED_MARK
        lines.append(
            f"  {mark} {knob.key}  [{knob.domain.kind}]  source={knob.source}"
        )
    for excl in inferred.excluded:
        lines.append(
            f"  {_EXCLUDED_MARK} {excl.key}  excluded: {excl.reason}"
        )
    n_trusted = inferred.n_knobs - inferred.n_needs_review
    lines.append(
        f"Summary: {n_trusted} trusted, {inferred.n_needs_review} need review, "
        f"{inferred.n_excluded} excluded."
    )
    return "\n".join(lines)
