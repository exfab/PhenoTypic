"""Column names describing how a row matched the external ``--metadata`` CSV."""

from __future__ import annotations

from .._base._measurement_info import Entry
from .._base._units import UNITS
from .._base._tiers import QualityInfo


class METADATA_MATCH(QualityInfo):
    """Metadata-join provenance flags attached to the post-applied mirror.

    Emitted only by the CLI's ``--metadata`` left join (into
    ``deliverables/measurements.{csv,parquet}`` and everything derived from it).
    The clean ``master_measurements.*`` archive never carries these columns.
    """

    @classmethod
    def metric_family(cls) -> str:
        return "QC"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.NOT_A_QUANTITY

    METADATA_ONLY = Entry(
        "MetadataOnly",
        "True on a row that comes only from the metadata file: a listed well "
        "(or other metadata row) where no colony was detected. Its "
        "measurement cells are empty, not zero, and both an empty well and a "
        "colony that detection missed produce such a row. It is False on "
        "every row that carries a measured colony.",
    )
