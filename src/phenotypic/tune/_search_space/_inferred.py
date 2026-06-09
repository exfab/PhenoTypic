"""The ``InferredSearchSpace`` proposal + ``Excluded`` record.

``infer_search_space`` returns a reviewable *proposal*, **not** the
optimizer-facing object. These are frozen pydantic value-models (mirroring
``Knob``/``SearchSpace``) so the proposal round-trips through JSON. Calling
``.to_search_space()`` collapses it to the clean ``SearchSpace`` the strategies
consume — the ``source`` / ``needs_review`` / ``description`` / ``excluded``
data never leaks into the optimizer.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ._space import Knob, SearchSpace

#: Why a field was excluded from the inferred space — a closed set (never a bare
#: ``str``). ``"ndarray"``/``"path"``/``"name_ref"``/``"tune_spec_off"`` are
#: *deliberate* exclusions (the field is simply not scalar-tunable, or the author
#: opted it out). ``"non_numeric"`` (a non-numeric / ``None`` anchor for the
#: unbounded heuristic), ``"non_positive_default"`` (a numeric anchor ``<= 0``, so
#: the multiplicative ``[d/f, d·f]`` window collapses or flips sign), and
#: ``"unsupported_type"`` (a multi-type union / unrecognised annotation) are
#: *inference-blind*: inference could not produce a plausible domain, so the
#: proposal flags the whole space for review (see
#: ``InferredSearchSpace.needs_review``).
ExcludeReason = Literal[
    "ndarray",
    "path",
    "name_ref",
    "non_numeric",
    "non_positive_default",
    "tune_spec_off",
    "unsupported_type",
]

#: Exclusion reasons that mean "inference could not even guess" — these raise the
#: proposal-level review flag. The remaining reasons are deliberate exclusions.
_BLIND_REASONS: frozenset[ExcludeReason] = frozenset(
    {"non_numeric", "non_positive_default", "unsupported_type"}
)


class Excluded(BaseModel):
    """A field inference could not (or should not) turn into a knob.

    Args:
        key: The root-relative path of the excluded field, e.g. ``"0.kernel"``.
        reason: A closed ``ExcludeReason`` tag explaining the exclusion.
        field_type: The field's annotation rendered as a string, for display in
            the "couldn't infer — declare a ``TuneSpec``" review section.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    reason: ExcludeReason
    field_type: str


class InferredSearchSpace(BaseModel):
    """A generous, reviewable proposal of tunable domains.

    Args:
        knobs: The tuple of inferred ``Knob`` instances (Tier-1 ``TuneSpec``
            overrides + Tier-2 heuristics).
        excluded: The tuple of ``Excluded`` records for fields that did not
            become knobs — surfaced so nothing is silently dropped.

    Note:
        Frozen value-model: round-trips through JSON so the proposal can be
        persisted (``--auto-space``) and re-loaded for review.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    knobs: tuple[Knob, ...]
    excluded: tuple[Excluded, ...]

    @property
    def n_knobs(self) -> int:
        """Number of inferred knobs."""
        return len(self.knobs)

    @property
    def n_excluded(self) -> int:
        """Number of excluded fields."""
        return len(self.excluded)

    @property
    def n_needs_review(self) -> int:
        """Number of knobs flagged for human review."""
        return sum(1 for k in self.knobs if k.needs_review)

    @property
    def needs_review(self) -> bool:
        """Whether the proposal warrants a human check before tuning.

        ``True`` iff any knob is flagged ``needs_review`` **or** any field was
        excluded for an inference-blind reason (``non_numeric`` /
        ``non_positive_default`` / ``unsupported_type``). Deliberate exclusions
        (ndarray / path / name-ref / ``tune_spec_off``) do not raise the flag.
        """
        if any(k.needs_review for k in self.knobs):
            return True
        return any(e.reason in _BLIND_REASONS for e in self.excluded)

    def to_search_space(self) -> SearchSpace:
        """Collapse to the clean, optimizer-facing ``SearchSpace``.

        Drops the proposal-only metadata (``source`` / ``needs_review`` /
        ``description`` / ``excluded``) by keeping only the knobs.

        Returns:
            The ``SearchSpace`` the strategies consume.
        """
        return SearchSpace(knobs=self.knobs)
