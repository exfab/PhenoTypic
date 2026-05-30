"""Incubation and time-course metadata tags for the PhenoTypic module."""

from .._measurement_info import MeasurementInfo


class INCUBATION_METADATA(MeasurementInfo):
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

    TEMPERATURE = "Temperature", "Incubation temperature in degrees Celsius."
    TIME = "Time", "Elapsed growth time."
    TIME_UNIT = "TimeUnit", "Unit for the Time value (e.g. hours, days)."
    TIMEPOINT = "Timepoint", "Discrete timepoint label in a time series."
    DAY = "Day", "Day index of the experiment."
    GENERATION = "Generation", "Generation or passage number."
    HUMIDITY = "Humidity", "Relative humidity during incubation."
    ATMOSPHERE = "Atmosphere", "Atmospheric condition (e.g. aerobic, anaerobic)."
