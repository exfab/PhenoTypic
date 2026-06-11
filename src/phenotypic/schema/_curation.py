"""Column names for GUI curation state written into derived frames."""

from __future__ import annotations

from ._measurement_info import MeasurementInfo


class CURATION(MeasurementInfo):
    """Curation-state columns attached to derived measurement frames.

    ``Curation_Category`` carries the :class:`ErrorCategory` bare label (or a
    custom category token) for each removed object in the per-category error
    parquets.
    """

    @classmethod
    def category(cls) -> str:
        return "Curation"

    CATEGORY = (
        "Category",
        "Error-category token assigned to a removed/triaged object.",
    )
