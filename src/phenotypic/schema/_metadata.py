"""Framework metadata bookkeeping labels for the PhenoTypic module."""

from ._measurement_info import MeasurementInfo


class METADATA(MeasurementInfo):
    """Framework-populated image metadata keys.

    These labels are set automatically by the image pipeline (not by the user) and
    name the bookkeeping entries on the ``image.metadata`` accessor. Members render
    as ``Metadata_<Label>`` (e.g. ``Metadata_ImageName``) so they share the
    ``Metadata_`` namespace with the user-facing experimental tags in
    :mod:`phenotypic.schema` (see :class:`SAMPLE_METADATA` and siblings).

    For the standardized *biological/experimental* vocabulary users supply via the
    ``--metadata`` CSV, use the experimental-tag enums instead.
    """

    @classmethod
    def category(cls) -> str:
        return "Metadata"

    UUID = "UUID", "The unique identifier of the image."
    IMAGE_NAME = "ImageName", "The name of the image."
    PARENT_IMAGE_NAME = "ParentImageName", "The name of the parent image."
    PARENT_UUID = "ParentUUID", "The UUID of the parent image."
    IMFORMAT = "ImageFormat", "The format of the image."
    IMAGE_TYPE = "ImageType", "The type of the image."
    BIT_DEPTH = "BitDepth", "The bit depth of the image."
    SUFFIX = (
        "FileSuffix",
        "The file suffix of the original file the image was imported from",
    )
