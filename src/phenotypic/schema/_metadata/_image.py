"""Framework metadata bookkeeping labels for the PhenoTypic module."""

from .._base._measurement_info import Entry
from .._base._rembi import REMBI_MODULE
from .._base._tiers import MetadataInfo


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

    UUID = Entry(
        "UUID",
        "Random identifier (a version-4 UUID) that PhenoTypic generates for "
        "each image object.",
    )
    IMAGE_NAME = Entry(
        "ImageName",
        "Name of the image, which is the file name without its extension when "
        "the image is read from a file.",
    )
    PARENT_IMAGE_NAME = Entry("ParentImageName", "The name of the parent image.")
    PARENT_UUID = Entry("ParentUUID", "The UUID of the parent image.")
    IMFORMAT = Entry("ImageFormat", "The format of the image.")
    IMAGE_TYPE = Entry(
        "ImageType",
        "Kind of image the row was measured from: Image, GridImage, Crop, "
        "Object or GridSection.",
    )
    BIT_DEPTH = Entry(
        "BitDepth",
        "Bit depth of the image's source pixel data, 8 or 16, either given "
        "when the image is loaded or inferred from the array's data type.",
    )
    SUFFIX = Entry(
        "FileSuffix",
        "The file suffix of the original file the image was imported from",
    )
