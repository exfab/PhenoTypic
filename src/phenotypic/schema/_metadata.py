"""Framework metadata bookkeeping labels for the PhenoTypic module."""

from ._measurement_info import Entry
from ._rembi import REMBI_MODULE
from ._tiers import MetadataInfo


class IMAGE(MetadataInfo):
    """Framework-populated image metadata keys.

    These labels are set automatically by the image pipeline (not by the user) and
    name the bookkeeping entries on the ``image.metadata`` accessor. Members render
    in the shared ``Metadata_<Label>`` namespace alongside the user-facing tags in
    :mod:`phenotypic.schema` (see :class:`SAMPLE` and siblings).

    For the standardized *biological/experimental* vocabulary users supply via the
    ``--metadata`` CSV, use the experimental-tag enums instead.
    """

    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:
        return REMBI_MODULE.IMAGE_DATA

    UUID = Entry("UUID", "The unique identifier of the image.")
    IMAGE_NAME = Entry("ImageName", "The name of the image.")
    PARENT_IMAGE_NAME = Entry("ParentImageName", "The name of the parent image.")
    PARENT_UUID = Entry("ParentUUID", "The UUID of the parent image.")
    IMFORMAT = Entry("ImageFormat", "The format of the image.")
    IMAGE_TYPE = Entry("ImageType", "The type of the image.")
    BIT_DEPTH = Entry("BitDepth", "The bit depth of the image.")
    SUFFIX = Entry(
        "FileSuffix",
        "The file suffix of the original file the image was imported from",
    )
