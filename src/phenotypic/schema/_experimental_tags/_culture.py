"""Culture and time-course metadata tags for the PhenoTypic module."""

from .._measurement_info import Entry
from .._rembi import REMBI_MODULE
from .._tiers import IdentityInfo


class CULTURE_METADATA(IdentityInfo):
    """Recommended ``MetadataCulture_*`` tags describing culture and time course.

    These capture the temporal and environmental culture parameters of an
    experiment (temperature, elapsed time, timepoint, generation, atmosphere).
    ``Time``/``Timepoint`` align with the QC time-series axis. Members render as
    ``MetadataCulture_<Label>`` (e.g. ``MetadataCulture_Time``) in the ``Metadata``
    column family shared with the other experimental-tag enums. Recommended
    vocabulary, not a validator.
    """

    @classmethod
    def category(cls) -> str:
        return "MetadataCulture"

    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:
        return REMBI_MODULE.SPECIMEN_PREP

    TIME = Entry("Time", "Elapsed growth time.",
                 rembi_module=REMBI_MODULE.BIOSAMPLE)
    TIME_UNIT = Entry("TimeUnit", "Unit for the Time value (e.g. hours, days).",
                      rembi_module=REMBI_MODULE.BIOSAMPLE)
    TIMEPOINT = Entry(
            "Timepoint",
            "Human-readable label for a discrete timepoint in a time series (e.g. "
            "'24h', 'stationary'); may be non-numeric. For the integer capture "
            "ordinal, use FrameIndex.",
            rembi_module=REMBI_MODULE.BIOSAMPLE,
    )
    FRAME_INDEX = Entry(
            "FrameIndex",
            "1-based ordinal index of the image within the time-course capture "
            "sequence; the monotonic-integer companion to the free-form Timepoint "
            "label.",
            rembi_module=REMBI_MODULE.BIOSAMPLE,
    )
    DAY = Entry("Day", "Day index of the experiment.")
    GENERATION = Entry("Generation", "Generation or passage number.")
    HUMIDITY = Entry("Humidity", "Relative humidity during culture.")
    ATMOSPHERE = Entry("Atmosphere", "Atmospheric condition (e.g. aerobic, anaerobic).")
