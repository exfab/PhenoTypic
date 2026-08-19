"""Imaging and acquisition metadata tags for the PhenoTypic module."""

from .._measurement_info import Entry
from .._rembi import REMBI_MODULE
from .._tiers import MetadataInfo


class ACQUISITION(MetadataInfo):
    """Recommended metadata tags describing image acquisition.

    These record how and by whom an image was captured (acquisition date,
    instrument, operator, resolution, exposure). Members render in the shared
    ``Metadata_<Label>`` namespace with the other experimental-tag enums.
    Recommended vocabulary, not a validator.
    """

    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:
        return REMBI_MODULE.IMAGE_ACQUISITION

    IMAGING_DATE = Entry("ImagingDate", "Date the image was acquired.")
    INSTRUMENT = Entry("Instrument", "Imaging instrument or scanner model.")
    EXPERIMENTER = Entry(
        "Experimenter",
        "Person who acquired the image or ran the experiment.",
    )
    RESOLUTION = Entry("Resolution", "Image resolution (e.g. DPI or pixels per mm).")
    EXPOSURE_TIME = Entry("ExposureTime", "Camera exposure time.")
