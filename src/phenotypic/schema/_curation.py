"""Column names for GUI curation state written into derived frames."""

from __future__ import annotations

from ._measurement_info import Entry
from ._tiers import QualityInfo


class CURATION(QualityInfo):
    """Curation-state columns attached to derived measurement frames.

    ``Curation_Category`` carries the :class:`ErrorCategory` bare label (or a
    custom category token) for each removed object in the per-category error
    parquets.
    """

    @classmethod
    def metric_family(cls) -> str:
        return "Curation"

    # Member name avoids ``METRIC_FAMILY`` (a reserved ``MeasurementInfo``
    # property) and the legacy ``CATEGORY`` name (refused at class creation);
    # the label stays "Category" so the column is ``Curation_Category``.
    ERROR_CATEGORY = Entry(
        "Category",
        "Error-category token assigned to a removed/triaged object.",
    )
