"""Incubation and time-course metadata tags for the PhenoTypic module."""

from .._measurement_info import Entry
from .._tiers import IdentityInfo


class INCUBATION_METADATA(IdentityInfo):
    """Recommended ``Metadata_*`` tags describing incubation and time course.

    These capture the temporal and environmental incubation parameters of an
    experiment (temperature, elapsed time, timepoint, generation, atmosphere).
    ``Time``/``Timepoint`` align with the QC time-series axis. Members render as
    ``Metadata_<Label>`` (e.g. ``Metadata_Time``) and share the ``Metadata_``
    namespace with the other experimental-tag enums. Recommended vocabulary, not a
    validator.
    """

    @classmethod
    def category(cls) -> str:
        return "Metadata"

    TEMPERATURE = Entry("Temperature", "Incubation temperature in degrees Celsius.")
    TIME = Entry("Time", "Elapsed growth time.")
    TIME_UNIT = Entry("TimeUnit", "Unit for the Time value (e.g. hours, days).")
    TIMEPOINT = Entry(
        "Timepoint",
        "Human-readable label for a discrete timepoint in a time series (e.g. "
        "'24h', 'stationary'); may be non-numeric. For the integer capture "
        "ordinal, use FrameIndex.",
    )
    FRAME_INDEX = Entry(
        "FrameIndex",
        "1-based ordinal index of the image within the time-course capture "
        "sequence; the monotonic-integer companion to the free-form Timepoint "
        "label.",
    )
    DAY = Entry("Day", "Day index of the experiment.")
    GENERATION = Entry("Generation", "Generation or passage number.")
    HUMIDITY = Entry("Humidity", "Relative humidity during incubation.")
    ATMOSPHERE = Entry("Atmosphere", "Atmospheric condition (e.g. aerobic, anaerobic).")
