"""Imaging and acquisition metadata tags for the PhenoTypic module."""

from .._measurement_info import Entry, MeasurementInfo


class ACQUISITION_METADATA(MeasurementInfo):
    """Recommended ``Metadata_*`` tags describing image acquisition.

    These record how and by whom an image was captured (acquisition date,
    instrument, operator, resolution, exposure). Members render as
    ``Metadata_<Label>`` (e.g. ``Metadata_ImagingDate``) and share the ``Metadata_``
    namespace with the other experimental-tag enums. Recommended vocabulary, not a
    validator.
    """

    @classmethod
    def category(cls) -> str:
        return "Metadata"

    IMAGING_DATE = Entry("ImagingDate", "Date the image was acquired.")
    INSTRUMENT = Entry("Instrument", "Imaging instrument or scanner model.")
    EXPERIMENTER = Entry(
        "Experimenter",
        "Person who acquired the image or ran the experiment.",
    )
    RESOLUTION = Entry("Resolution", "Image resolution (e.g. DPI or pixels per mm).")
    EXPOSURE_TIME = Entry("ExposureTime", "Camera exposure time.")
