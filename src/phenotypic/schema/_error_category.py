"""The closed taxonomy of colony-detection error categories.

``ErrorCategory`` is the vocabulary a user assigns when triaging a detected
object as bad (rather than merely removing it). Members are
``MeasurementInfo``-style ``(label, description)`` tuples so the description is
available to the GUI radial-menu tooltips and the generated docs. The
**bare label** (``member.label``, e.g. ``"oversegmented"``) is the canonical
token persisted in the ``Curation_Category`` column and used as the
per-category parquet filename, so every label is filename-safe.

``OTHER`` is the reserved catch-all: marking an object ``other`` is the
"remove without a specified reason" path. Categories beyond this enum are
user-defined custom categories, registered at runtime (not enum members).
"""

from __future__ import annotations

from ._measurement_info import Entry
from ._tiers import QualityInfo


class ErrorCategory(QualityInfo):
    """Closed taxonomy of detection-error categories for object triage.

    The enum *value* is the category-prefixed header (``ErrorCategory_<label>``)
    per the ``MeasurementInfo`` convention, but callers persist and compare on
    the bare :attr:`~MeasurementInfo.label` (e.g. ``"debris"``). Use
    :meth:`from_label` to resolve a stored token back to a member.
    """

    @classmethod
    def category(cls) -> str:
        return "ErrorCategory"

    @classmethod
    def labels(cls) -> list[str]:
        """Return the bare category tokens in declaration order.

        Returns:
            The ``.label`` of every member, e.g.
            ``["oversegmented", ..., "other"]``.
        """
        return cls.get_labels()

    @classmethod
    def from_label(cls, label: str) -> "ErrorCategory | None":
        """Resolve a bare category token to its member, or ``None``.

        Args:
            label: A bare category token (e.g. ``"merged"``).

        Returns:
            The matching member, or ``None`` if ``label`` is not a core
            category (e.g. a custom category or a typo).
        """
        for member in cls:
            if member.label == label:
                return member
        return None

    OVERSEGMENTED = Entry(
        "oversegmented",
        "One colony split into multiple detections.",
    )
    UNDERSEGMENTED = Entry(
        "undersegmented",
        "A single colony under-detected — its mask captured too small or "
        "only partially covering the colony.",
    )
    MERGED = Entry(
        "merged",
        "Multiple touching colonies detected as one object.",
    )
    BACKGROUND_NOISE = Entry(
        "background_noise",
        "Not a colony — agar texture, reflection, or vignette.",
    )
    DEBRIS = Entry(
        "debris",
        "Dust, scratch, bubble, or other plate artifact.",
    )
    OTHER = Entry(
        "other",
        "Removed without a specified reason (the catch-all bucket).",
    )
