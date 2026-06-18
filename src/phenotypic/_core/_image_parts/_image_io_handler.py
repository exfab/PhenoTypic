from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import warnings
from datetime import datetime
from fractions import Fraction
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
    from phenotypic._core._image import Image

# Check for optional dependencies and import if available
if importlib.util.find_spec("exifread") is not None:
    import exifread
else:
    exifread = None

import h5py
import numpy as np
import pickle  # noqa: S403 - existing load_pickle/save2pickle support
from os import PathLike
from pathlib import Path
from PIL import Image as PIL_Image
from PIL.TiffTags import TAGS as TIFF_TAGS

if importlib.util.find_spec("rawpy") is not None:
    import rawpy
else:
    rawpy = None

import skimage as ski

import phenotypic
from phenotypic.sdk_.exceptions_ import UnsupportedFileTypeError
from phenotypic.schema import METADATA
from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS, IO
from phenotypic.sdk_.hdf_ import HDF
from ._image_color_handler import ImageColorSpace

# -----------------------------------------------------------------------------
# HDF5 schema helpers
# -----------------------------------------------------------------------------

_SCHEMA_VERSION = 2


def _encode_meta(val: Any) -> str:
    """Encode a metadata value to a JSON string for HDF5 attribute storage.

    Uses ``default=str`` as a fallback so non-JSON-native types (e.g.
    ``np.datetime64``) are still serialisable.
    """
    return json.dumps(val, default=str)


def _decode_meta(s: Any) -> Any:
    """Decode a metadata value from its JSON-string representation.

    Falls back to returning the raw value if it is not valid JSON (legacy
    files stored values as bare strings). h5py may return attribute values
    as ``bytes`` (fixed-length byte strings, older HDF5 files, cross-platform);
    decode them to ``str`` before attempting JSON parsing.
    """
    if isinstance(s, bytes):
        s = s.decode("utf-8", errors="replace")
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError, TypeError):
        return s


# Legacy metadata-key shim.
# Files written before the ``METADATA`` enum gained its ``Metadata_`` category
# prefix stored framework metadata keys *bare* (e.g. ``ImageName``, ``BitDepth``,
# ``UUID``). Map those legacy bare keys to their current prefixed form on load so
# old HDF5 outputs stay readable. Built from the enum so it tracks any future
# member additions: ``{"ImageName": "Metadata_ImageName", ...}``.
_LEGACY_METADATA_KEY_MAP: dict[str, str] = {
    member.label: member.value for member in METADATA
}


def _remap_legacy_metadata_key(key: str) -> str:
    """Return the current ``Metadata_*`` key for a legacy bare metadata key.

    Idempotent and targeted: already-prefixed keys (new files) and arbitrary
    user-supplied keys pass through unchanged — only the known framework labels
    are remapped.
    """
    return _LEGACY_METADATA_KEY_MAP.get(key, key)


class _BackCompatUnpickler(pickle.Unpickler):
    """Unpickler that resolves classes which moved since older pickles were written.

    ``METADATA`` moved from ``phenotypic.tools_.constants_`` to the public
    ``phenotypic.schema`` package. Pickled ``image.metadata`` dicts key on
    ``METADATA`` enum *members*, which unpickle by their class's import path — so
    pre-move pickles would otherwise fail with ``AttributeError`` at load time.
    Remapping the old path here lets them load; because enum members resolve by
    name, an old bare-valued ``METADATA.IMAGE_NAME`` resolves to the current
    ``Metadata_``-prefixed member, auto-upgrading both keys and values.

    Scoped to the moved symbol only and used solely for unpickling — it does not
    reintroduce ``METADATA`` into ``constants_``'s import namespace.
    """

    _MOVED_CLASSES: dict[tuple[str, str], tuple[str, str]] = {
        ("phenotypic.tools_.constants_", "METADATA"): (
            "phenotypic.schema",
            "METADATA",
        ),
    }

    def find_class(self, module: str, name: str):
        module, name = self._MOVED_CLASSES.get((module, name), (module, name))
        return super().find_class(module, name)


class ImageIOHandler(ImageColorSpace):
    """Handles input/output operations and metadata for images.

    This class extends ImageColorSpace to provide comprehensive file I/O capabilities,
    including:
    - Reading images from various file formats (JPEG, PNG, TIFF, RAW)
    - Writing images to HDF5 and pickle formats
    - Extracting and parsing metadata from image files
    - Managing EXIF, TIFF tags, and custom PhenoTypic metadata
    - Support for raw sensor data processing via rawpy

    The class abstracts file format details, providing a unified interface for loading
    and saving images while preserving metadata and calibration information. Metadata
    extraction supports round-trip storage and recovery of PhenoTypic-specific data.

    Examples:
        Basic usage:

        >>> img = ImageIOHandler.imread('photo.jpg')
        >>> img.save2hdf5('output.h5')
        >>> loaded = ImageIOHandler.load_hdf5('output.h5')
    """

    def __init__(
            self, arr: np.ndarray | Image | None = None, name: str | None = None,
            **kwargs
    ):
        """Initialize ImageIOHandler with I/O capabilities.

        Args:
            arr (np.ndarray | Image | None): Optional initial image data. Defaults to None.
            name (str | None): Optional image name. Defaults to None.
            **kwargs: Additional keyword arguments passed to parent class (ImageColorSpace).
        """
        super().__init__(arr=arr, name=name, **kwargs)

    # -------------------------------------------------------------------------
    # Metadata Extraction Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_metadata_value(value) -> int | float | bool | str | None:
        """Convert metadata value to a normalized scalar type.

        Normalizes various metadata value types to scalar values that can be safely
        stored and retrieved from metadata dictionaries. Handles special cases like
        exifread IfdTag objects, NumPy types, fractions, and datetime objects.

        Args:
            value: Any metadata value to normalize. Supports: int, float, bool, str,
                bytes, Fraction, datetime, list, tuple, np.ndarray, exifread IfdTag.

        Returns:
            int | float | bool | str | None: Normalized scalar value. Converts complex
                types to appropriate scalar representations:
                - bytes: decoded to str (UTF-8)
                - Fraction: converted to float
                - datetime: converted to ISO format string
                - list/tuple: single items unwrapped, multiple items converted to string
                - np.ndarray: single items unwrapped, multiple items converted to list string
                - exifread IfdTag: unwrapped or converted to printable string

        Note:
            Unrecognized types are converted to string representation as fallback.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return float(value)
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8", errors="replace")
            except Exception:
                return str(value)
        if isinstance(value, Fraction):
            return float(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return ImageIOHandler._normalize_metadata_value(value[0])
            # Convert list/tuple to string representation
            return str(value)
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return ImageIOHandler._normalize_metadata_value(value.item())
            return str(value.tolist())
        # For exifread IfdTag objects
        if hasattr(value, "values"):
            vals = value.values
            if isinstance(vals, (list, tuple)) and len(vals) == 1:
                return ImageIOHandler._normalize_metadata_value(vals[0])
            if hasattr(value, "printable"):
                return str(value.printable)
            return str(vals)
        # Fallback to string
        return str(value)

    @staticmethod
    def _extract_jpeg_png_metadata(filepath: Path) -> dict:
        """Extract EXIF metadata from JPEG/PNG files and parse PhenoTypic JSON if present.

        Args:
            filepath: Path to the image file.

        Returns:
            Dictionary containing extracted metadata with normalized values.
        """
        metadata = {}

        # Use exifread for comprehensive EXIF parsing (if available)
        if exifread is not None:
            try:
                with open(filepath, "rb") as f:
                    tags = exifread.process_file(f, details=True)
                    for tag, value in tags.items():
                        if tag.startswith("Thumbnail"):
                            continue  # Skip thumbnail data
                        normalized = ImageIOHandler._normalize_metadata_value(value)
                        if normalized is not None:
                            metadata[tag] = normalized

                            # Check for PhenoTypic data in EXIF UserComment
                            if tag == "EXIF UserComment" and isinstance(normalized,
                                                                        str):
                                try:
                                    phenotypic_data = json.loads(normalized)
                                    if (
                                            isinstance(phenotypic_data, dict)
                                            and "phenotypic_version" in phenotypic_data
                                    ):
                                        metadata["_phenotypic_data"] = phenotypic_data
                                except json.JSONDecodeError:
                                    pass
            except Exception:
                pass  # File may not have EXIF data or exifread failed

        # Also try PIL for additional info
        try:
            with PIL_Image.open(filepath) as img:
                # Get basic image info
                if hasattr(img, "info") and img.info:
                    for key, value in img.info.items():
                        if key == "exif":
                            continue  # Already handled by exifread
                        # Check for PhenoTypic PNG tEXt chunk
                        if key == IO.PHENOTYPIC_METADATA_KEY:
                            try:
                                phenotypic_data = json.loads(value)
                                if (
                                        isinstance(phenotypic_data, dict)
                                        and "phenotypic_version" in phenotypic_data
                                ):
                                    metadata["_phenotypic_data"] = phenotypic_data
                            except json.JSONDecodeError:
                                pass
                            continue
                        normalized = ImageIOHandler._normalize_metadata_value(value)
                        if normalized is not None:
                            metadata[f"PIL:{key}"] = normalized

                # Try to get EXIF UserComment for PhenoTypic data (JPEG)
                exif_data = img.getexif()
                if exif_data:
                    # UserComment tag is 37510
                    user_comment = exif_data.get(37510)
                    if user_comment:
                        try:
                            # UserComment may have encoding prefix
                            if isinstance(user_comment, bytes):
                                # Skip encoding prefix if present (first 8 bytes)
                                if user_comment.startswith(b"ASCII\x00\x00\x00"):
                                    user_comment = user_comment[8:]
                                elif user_comment.startswith(b"UNICODE\x00"):
                                    user_comment = user_comment[8:].decode("utf-16")
                                else:
                                    user_comment = user_comment.decode(
                                            "utf-8", errors="replace"
                                    )
                            phenotypic_data = json.loads(user_comment)
                            if (
                                    isinstance(phenotypic_data, dict)
                                    and "phenotypic_version" in phenotypic_data
                            ):
                                metadata["_phenotypic_data"] = phenotypic_data
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass
        except Exception:
            pass

        return metadata

    @staticmethod
    def _extract_tiff_metadata(filepath: Path) -> dict:
        """Extract TIFF tag metadata and parse PhenoTypic JSON from ImageDescription.

        Args:
            filepath: Path to the TIFF file.

        Returns:
            Dictionary containing extracted metadata with normalized values.
        """
        metadata = {}

        try:
            with PIL_Image.open(filepath) as img:
                # Get TIFF tags
                if hasattr(img, "tag_v2") and img.tag_v2:
                    for tag_id, value in img.tag_v2.items():
                        tag_name = TIFF_TAGS.get(tag_id, f"Tag_{tag_id}")
                        normalized = ImageIOHandler._normalize_metadata_value(value)
                        if normalized is not None:
                            metadata[f"TIFF:{tag_name}"] = normalized

                # Check ImageDescription (tag 270) for PhenoTypic JSON
                if hasattr(img, "tag_v2") and 270 in img.tag_v2:
                    desc = img.tag_v2[270]
                    if isinstance(desc, bytes):
                        desc = desc.decode("utf-8", errors="replace")
                    try:
                        phenotypic_data = json.loads(desc)
                        if (
                                isinstance(phenotypic_data, dict)
                                and "phenotypic_version" in phenotypic_data
                        ):
                            metadata["_phenotypic_data"] = phenotypic_data
                    except json.JSONDecodeError:
                        pass  # Not JSON, keep as regular metadata
        except Exception:
            pass

        return metadata

    @staticmethod
    def _extract_raw_metadata(filepath: Path) -> dict:
        """Extract metadata from RAW files using exiftool (if available) or rawpy fallback.

        Args:
            filepath: Path to the RAW file.

        Returns:
            Dictionary containing extracted metadata with normalized values.
        """
        metadata = {}

        # Try exiftool first (more comprehensive)
        if shutil.which("exiftool"):
            try:
                result = subprocess.run(
                        ["exiftool", "-json", "-n", str(filepath)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                )
                if result.returncode == 0:
                    exif_data = json.loads(result.stdout)
                    if exif_data and isinstance(exif_data, list):
                        for key, value in exif_data[0].items():
                            if key == "SourceFile":
                                continue
                            normalized = ImageIOHandler._normalize_metadata_value(value)
                            if normalized is not None:
                                metadata[f"EXIF:{key}"] = normalized
                    return metadata
            except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
                pass  # Fall through to rawpy

        # Fallback to rawpy if exiftool not available
        if rawpy is not None:
            try:
                with rawpy.imread(str(filepath)) as raw:
                    # Extract available rawpy metadata attributes
                    metadata["rawpy:camera_whitebalance"] = str(
                            list(raw.camera_whitebalance)
                    )
                    metadata["rawpy:daylight_whitebalance"] = str(
                            list(raw.daylight_whitebalance)
                    )
                    metadata["rawpy:num_colors"] = int(raw.num_colors)
                    metadata["rawpy:color_desc"] = (
                        raw.color_desc.decode("utf-8") if raw.color_desc else None
                    )
                    metadata["rawpy:raw_type"] = str(raw.raw_type)

                    if raw.sizes:
                        metadata["rawpy:raw_height"] = int(raw.sizes.raw_height)
                        metadata["rawpy:raw_width"] = int(raw.sizes.raw_width)
                        metadata["rawpy:height"] = int(raw.sizes.height)
                        metadata["rawpy:width"] = int(raw.sizes.width)

                    # Black and white levels
                    if (
                            hasattr(raw, "black_level_per_channel")
                            and raw.black_level_per_channel is not None
                    ):
                        metadata["rawpy:black_level_per_channel"] = str(
                                list(raw.black_level_per_channel)
                        )
                    if hasattr(raw, "white_level") and raw.white_level is not None:
                        metadata["rawpy:white_level"] = int(raw.white_level)
            except Exception:
                pass

        return metadata

    @classmethod
    def imread(
            cls, filepath: PathLike, rawpy_params: dict | None = None, **kwargs
    ) -> Image:
        """
        imread is a class method responsible for reading an image file from the specified
        path and performing necessary preprocessing based on the file format and additional
        parameters. The method supports a variety of image file types including common
        formats (e.g., JPEG, PNG) as well as raw sensor data. It uses the scikit-image library
        for loading standard images and rawpy for processing raw image files. This method also
        handles additional configurations for raw image preprocessing via rawpy parameters, such
        as white balance, gamma correction, and demosaic algorithm.

        Args:
            filepath (PathLike): Path to the image file to be read. It can be any valid file
                path-like object (e.g., str, pathlib.Path).
            rawpy_params (dict | None): Optional dictionary of parameters for processing raw image
                files when using rawpy. Supports options like white balance settings, demosaic
                algorithm, gamma correction, and others. Defaults to None.
            **kwargs: Arbitrary keyword arguments to be passed for additional configurations
                specific to the Image instantiation.

        Returns:
            Image: An instance of the Image class containing the processed image array and any
                additional metadata.

        Raises:
            UnsupportedFileTypeError: If the file type of the provided filepath is not supported
                by the method, either due to its extension not being recognized or due to the
                absence of required libraries like rawpy.
        """
        # Convert to a Path object
        filepath: Path = Path(filepath)
        rawpy_params = rawpy_params or {}

        suffix = filepath.suffix.lower()
        if suffix in IO.ACCEPTED_FILE_EXTENSIONS:  # normal images
            arr = ski.io.imread(fname=filepath)

        elif (
                suffix in IO.RAW_FILE_EXTENSIONS and rawpy is not None
        ):  # raw sensor data handling
            use_auto_wb = rawpy_params.pop("use_auto_wb", False)
            use_camera_wb = rawpy_params.pop("use_camera_wb", False)

            no_auto_scale = rawpy_params.pop(
                    "no_auto_scale", False
            )  # TODO: implement calibration schema
            no_auto_bright = rawpy_params.pop(
                    "no_auto_bright", False
            )  # TODO: implement calibration schema

            # noinspection PyUnresolvedReferences
            if rawpy.DemosaicAlgorithm.AMAZE.isSupported:
                # noinspection PyUnresolvedReferences
                default_demosaic = rawpy.DemosaicAlgorithm.AMAZE
            else:
                # noinspection PyUnresolvedReferences
                default_demosaic = rawpy.DemosaicAlgorithm.AHD

            demosaic_algorithm = rawpy_params.pop(
                    "demosaic_algorithm", default_demosaic
            )
            gamma = rawpy_params.pop("gamma", (1, 1))
            with rawpy.imread(str(filepath)) as raw:
                # noinspection PyUnresolvedReferences
                arr = raw.postprocess(
                        demosaic_algorithm=demosaic_algorithm,
                        use_camera_wb=use_camera_wb,
                        use_auto_wb=use_auto_wb,
                        no_auto_scale=no_auto_scale,
                        no_auto_bright=no_auto_bright,
                        gamma=gamma,
                        median_filter_passes=0,
                        output_bps=16,  # Preserve as much detail as possible
                        output_color=rawpy.ColorSpace.sRGB,
                        **rawpy_params,
                )

        else:
            raise UnsupportedFileTypeError(filepath.suffix)

        bit_depth = kwargs.pop("bit_depth", None)
        if suffix in IO.JPEG_FILE_EXTENSIONS:
            bit_depth = 8

        image = cls(arr=arr, name=filepath.stem, bit_depth=bit_depth, **kwargs)
        image.name = filepath.stem
        image.metadata[METADATA.SUFFIX] = suffix

        # Extract and store metadata based on file type
        if suffix in IO.JPEG_FILE_EXTENSIONS or suffix in IO.PNG_FILE_EXTENSIONS:
            imported_metadata = cls._extract_jpeg_png_metadata(filepath)
        elif suffix in IO.TIFF_EXTENSIONS:
            imported_metadata = cls._extract_tiff_metadata(filepath)
        elif suffix in IO.RAW_FILE_EXTENSIONS:
            imported_metadata = cls._extract_raw_metadata(filepath)
        else:
            imported_metadata = {}

        # Handle PhenoTypic round-trip data if present
        if "_phenotypic_data" in imported_metadata:
            phenotypic_data = imported_metadata.pop("_phenotypic_data")
            # Only restore protected/public metadata if saved from rgb or gray property
            # (not from color space accessors or detect_mat which are derived views)
            source_property = phenotypic_data.get("phenotypic_image_property", "")
            if source_property in ("Image.rgb", "Image.gray"):
                if "protected" in phenotypic_data:
                    for key, value in phenotypic_data["protected"].items():
                        # Remap legacy bare keys (pre-Metadata_ prefix) first, so
                        # the critical-field skip below matches old files too.
                        key = _remap_legacy_metadata_key(key)
                        # Don't overwrite critical protected fields
                        if key not in (METADATA.UUID, METADATA.IMAGE_NAME):
                            image._metadata.protected[key] = value
                if "public" in phenotypic_data:
                    image._metadata.public.update(
                            {
                                _remap_legacy_metadata_key(k): v
                                for k, v in phenotypic_data["public"].items()
                            }
                    )

        # Store remaining imported metadata
        image._metadata.imported.update(imported_metadata)

        return image

    @staticmethod
    def _get_hdf5_group(handler: h5py.File | h5py.Group, name: str):
        """
        Retrieves an HDF5 group from the given handler by name. If the group does not
        exist, it creates a new group with the specified name.

        Args:
            handler: HDF5 file or group handler used to manage HDF5 groups.
            name: The name of the group to retrieve or create.

        Returns:
            h5py.Group: The requested or newly created HDF5 group.
        """
        file_handler = handler if isinstance(handler, h5py.File) else handler.file
        name = str(name)
        if name in handler:
            return handler[name]
        elif file_handler.swmr_mode is True:
            raise ValueError("hdf5 handler in SWMR mode cannot create group")
        else:
            return handler.create_group(name)

    @staticmethod
    def _save_array2hdf5(group: h5py.Group, array: np.ndarray, name: str, **kwargs):
        """
        Saves a given numpy array to an HDF5 group. If a dataset with the specified
        name already exists in the group, it checks if the shapes match. If the
        shapes match, it updates the existing dataset; otherwise, it removes the
        existing dataset and creates a new one with the specified name. If a dataset
        with the given name doesn't exist, it creates a new dataset.

        Args:
            group: h5py.Group
                The HDF5 group in which the dataset will be saved.
            array: numpy.ndarray
                The data array to be stored in the dataset.
            name: str
                The name of the dataset within the group.
            **kwargs: dict
                Additional keyword arguments to pass when creating a new dataset.
        """
        assert isinstance(array, np.ndarray), "array must be a numpy array."

        file_handler = group.file if isinstance(group, h5py.Group) else group
        if name in group:
            dataset = group[name]
            assert isinstance(dataset, h5py.Dataset), f"{name} is not a dataset."
            if dataset.shape == array.shape:
                dataset[:] = array
            elif file_handler.swmr_mode is True:
                raise ValueError(
                        "Shape does not match existing dataset shape and cannot be changed because file handler is in SWMR mode"
                )
            else:
                del group[name]
                group.create_dataset(name, data=array, dtype=array.dtype, **kwargs)
        else:
            group.create_dataset(name, data=array, dtype=array.dtype, **kwargs)

    def _save_image2hdfgroup(
            self,
            grp,
            compression="gzip",
            compression_opts=4,
    ):
        """Save image datasets and metadata into the given HDF5 group.

        Uses schema_version=2 layout:
          - root attrs: version, schema_version, phenotypic_class,
            bit_depth, illuminant, gamma
          - /layers/: rgb (optional), gray, detect_mat, objmap
          - /metadata/{protected,public,imported}/: JSON-encoded attrs
        """
        # ------------------------------------------------------------------
        # Root attributes
        # ------------------------------------------------------------------
        grp.attrs["version"] = phenotypic.__version__
        grp.attrs["schema_version"] = _SCHEMA_VERSION
        grp.attrs["phenotypic_class"] = type(self).__name__

        if self.bit_depth is not None:
            grp.attrs["bit_depth"] = int(self.bit_depth)
        if self.illuminant is not None:
            grp.attrs["illuminant"] = str(self.illuminant)
        if self.gamma is not None:
            grp.attrs["gamma"] = (
                self.gamma.name if hasattr(self.gamma, "name") else str(self.gamma)
            )

        # ------------------------------------------------------------------
        # Layers subgroup
        # ------------------------------------------------------------------
        layers = grp.require_group("layers")

        if not self.rgb.isempty():
            array = self.rgb[:]
            HDF.save_array2hdf5(
                    group=layers,
                    array=array,
                    name="rgb",
                    dtype=array.dtype,
                    compression=compression,
                    compression_opts=compression_opts,
            )

        matrix = self.gray[:]
        HDF.save_array2hdf5(
                group=layers,
                array=matrix,
                name="gray",
                dtype=matrix.dtype,
                compression=compression,
                compression_opts=compression_opts,
        )

        detect_matrix = self.detect_mat[:]
        HDF.save_array2hdf5(
                group=layers,
                array=detect_matrix,
                name="detect_mat",
                dtype=detect_matrix.dtype,
                compression=compression,
                compression_opts=compression_opts,
        )
        layers["detect_mat"].attrs["detect_mode"] = self._data.detect_mode

        objmap = self.objmap[:]
        HDF.save_array2hdf5(
                group=layers,
                array=objmap,
                name="objmap",
                dtype=objmap.dtype,
                compression=compression,
                compression_opts=compression_opts,
        )

        # ------------------------------------------------------------------
        # Metadata subgroups (protected / public / imported)
        # ------------------------------------------------------------------
        meta = grp.require_group("metadata")
        sections = {
            "protected": self._metadata.protected,
            "public"   : self._metadata.public,
            "imported" : self._metadata.imported,
        }
        for name, section in sections.items():
            sub = meta.require_group(name)
            for key, val in section.items():
                sub.attrs[str(key)] = _encode_meta(val)

    def _save_hdf5_metadata(self, grp) -> None:
        """Write version info and metadata subgroups into an HDF5 group.

        Deprecated legacy-flat layout helper retained so
        :meth:`save_intermediate_layers` (which still writes the v1 flat
        layout) continues to round-trip. The v2 writer is inlined in
        :meth:`_save_image2hdfgroup`.
        """
        grp.attrs["version"] = phenotypic.__version__

        prot = grp.require_group("protected_metadata")
        for key, val in self._metadata.protected.items():
            prot.attrs.modify(key, str(val))

        pub = grp.require_group("public_metadata")
        for key, val in self._metadata.public.items():
            pub.attrs.modify(key, str(val))

    def save2hdf5(
            self, filename, compression="gzip", compression_opts=4,
    ):
        """Save the image to an HDF5 file with all data and metadata.

        Stores the complete image data (RGB, gray, detection matrix,
        object map) and metadata (protected and public) directly at
        the HDF5 file's root group. The file is always overwritten
        if it already exists.

        Args:
            filename: Path to the HDF5 file (.h5 extension
                recommended). Created or overwritten on each call.
            compression: Compression filter to apply to datasets.
                Options: 'gzip' (recommended), 'szip', or None.
                Defaults to 'gzip'.
            compression_opts: Compression level for 'gzip' (1-9,
                where 1=fastest, 9=best). Ignored for 'szip' and
                None. Defaults to 4 (balanced).

        Examples:
            Save to HDF5:

            >>> img = Image.imread('photo.jpg')
            >>> img.save2hdf5('output.h5')
            >>> img.save2hdf5('output.h5', compression='szip')
        """
        with h5py.File(filename, mode="w") as filehandler:
            self._save_image2hdfgroup(
                    grp=filehandler,
                    compression=compression,
                    compression_opts=compression_opts,
            )

    def save_intermediate_layers(
            self,
            filename,
            layers: tuple[str, ...],
            compression="gzip",
            compression_opts=4,
    ):
        """Save only the specified image layers to an HDF5 file.

        Args:
            filename: Path to the HDF5 file to create.
            layers: Tuple of layer names to save. Valid names are
                ``"rgb"``, ``"gray"``, ``"detect_mat"``, ``"objmap"``.
            compression: Compression filter. Defaults to ``"gzip"``.
            compression_opts: Compression level (1-9). Defaults to 4.

        Raises:
            ValueError: If *layers* contains unknown layer names.
        """
        _valid = {"rgb", "gray", "detect_mat", "objmap"}
        unknown = set(layers) - _valid
        if unknown:
            raise ValueError(f"Unknown layer names: {unknown}")

        with h5py.File(filename, mode="w") as f:
            for layer in layers:
                if layer == "rgb":
                    if not self.rgb.isempty():
                        array = self.rgb[:]
                        HDF.save_array2hdf5(
                                group=f, array=array, name="rgb",
                                dtype=array.dtype,
                                compression=compression,
                                compression_opts=compression_opts,
                        )
                elif layer == "gray":
                    array = self.gray[:]
                    HDF.save_array2hdf5(
                            group=f, array=array, name="gray",
                            dtype=array.dtype,
                            compression=compression,
                            compression_opts=compression_opts,
                    )
                elif layer == "detect_mat":
                    array = self.detect_mat[:]
                    HDF.save_array2hdf5(
                            group=f, array=array, name="detect_mat",
                            dtype=array.dtype,
                            compression=compression,
                            compression_opts=compression_opts,
                    )
                    f["detect_mat"].attrs["detect_mode"] = (
                        self._data.detect_mode
                    )
                elif layer == "objmap":
                    array = self.objmap[:]
                    HDF.save_array2hdf5(
                            group=f, array=array, name="objmap",
                            dtype=array.dtype,
                            compression=compression,
                            compression_opts=compression_opts,
                    )

            self._save_hdf5_metadata(f)

    @classmethod
    def _load_from_hdf5_group(cls, group, **kwargs) -> Image:
        """Load an Image from an HDF5 group.

        Dispatches to the v2 grouped loader when the root attributes mark the
        group as schema_version >= 2 and a ``layers`` subgroup exists, and
        falls back to the legacy flat-root loader otherwise.
        """
        schema_version = int(group.attrs.get("schema_version", 1))
        if schema_version >= _SCHEMA_VERSION and "layers" in group:
            return cls._load_v2_grouped(group, **kwargs)
        return cls._load_legacy_flat_group(group, **kwargs)

    @classmethod
    def _load_v2_grouped(cls, group, **kwargs) -> Image:
        """Load from the schema_version=2 grouped layout.

        Reads ``/layers/`` for image arrays, ``/metadata/{protected,public,
        imported}/`` for JSON-decoded metadata, and merges root attributes
        (``bit_depth``, ``illuminant``, ``gamma``) into ``kwargs`` via
        ``setdefault`` so explicit caller kwargs take priority.
        """
        # Merge root attrs into kwargs (caller-provided kwargs win).
        if "bit_depth" in group.attrs:
            try:
                kwargs.setdefault("bit_depth", int(group.attrs["bit_depth"]))
            except (TypeError, ValueError):
                pass
        if "illuminant" in group.attrs:
            illum = group.attrs["illuminant"]
            if isinstance(illum, bytes):
                illum = illum.decode("utf-8", errors="replace")
            kwargs.setdefault("illuminant", str(illum))
        if "gamma" in group.attrs:
            gamma_name = group.attrs["gamma"]
            if isinstance(gamma_name, bytes):
                gamma_name = gamma_name.decode("utf-8", errors="replace")
            gamma_name = str(gamma_name)
            # Map name back via GAMMA_ENCODINGS; on unknown names fall back to
            # GAMMA_ENCODINGS.SRGB (the constructor-validated default) so we
            # never hand ImageColorSpace.__init__ a string it will reject.
            try:
                gamma_val: GAMMA_ENCODINGS = GAMMA_ENCODINGS[gamma_name]
            except KeyError:
                warnings.warn(
                        f"Unknown gamma encoding {gamma_name!r}; falling back to "
                        f"GAMMA_ENCODINGS.SRGB",
                        UserWarning,
                        stacklevel=2,
                )
                kwargs.setdefault("gamma", GAMMA_ENCODINGS.SRGB)
            else:
                kwargs.setdefault("gamma", gamma_val)

        layers = group["layers"]

        # Read image/gray arrays.
        matrix_data = layers["gray"][()]
        if "rgb" in layers:
            array_data = layers["rgb"][()]
            img = cls(arr=array_data, **kwargs)
            img.gray[:] = matrix_data
        else:
            img = cls(arr=matrix_data, **kwargs)

        # Detection matrix + mode.
        detect_mat_ds = layers["detect_mat"]
        detect_matrix_data = detect_mat_ds[()]
        detect_mode = detect_mat_ds.attrs.get("detect_mode", "gray")
        if isinstance(detect_mode, bytes):
            detect_mode = detect_mode.decode("utf-8", errors="replace")
        img.detect_mat[:] = detect_matrix_data
        img._data.detect_mode = detect_mode

        # Object map preserves integer-label dtype.
        img.objmap[:] = layers["objmap"][()]

        # Metadata: protected / public / imported — each section JSON decoded.
        if "metadata" in group:
            meta = group["metadata"]
            targets = {
                "protected": img._metadata.protected,
                "public"   : img._metadata.public,
                "imported" : img._metadata.imported,
            }
            for name, target in targets.items():
                if name not in meta:
                    continue
                attrs = meta[name].attrs
                # setdefault-style merge: caller kwargs (e.g. bit_depth) have
                # already populated matching protected-metadata keys via the
                # Image constructor, so we must not clobber them here.
                # public/imported are effectively fresh dicts at construction,
                # so the same logic is a no-op for them but keeps behaviour
                # uniform across sections.
                for key in attrs:
                    # Remap legacy bare keys (pre-Metadata_ prefix) before the
                    # membership check, so an old file's "ImageName" matches the
                    # constructor-populated "Metadata_ImageName" and is skipped
                    # rather than added as a stale duplicate.
                    mapped = _remap_legacy_metadata_key(key)
                    if mapped in target:
                        continue
                    target[mapped] = _decode_meta(attrs[key])

        return img

    @classmethod
    def _load_legacy_flat_group(cls, group, **kwargs) -> Image:
        """Load from the legacy (schema_version=1 / unversioned) flat layout.

        Preserves the current behaviour for files that store rgb / gray /
        detect_mat / objmap at the group root, with ``protected_metadata`` /
        ``public_metadata`` attribute subgroups and the legacy
        ``int(v) if v.isdigit() else v`` coercion. The ``imported`` section is
        left empty because legacy files never persisted it.
        """
        # Read datasets back into numpy arrays with proper dtype handling.
        matrix_data = group["gray"][()]

        # Determine format from available datasets.
        if "rgb" in group:
            array_data = group["rgb"][()]
            img = cls(arr=array_data, **kwargs)
            img.gray[:] = matrix_data
        else:
            img = cls(arr=matrix_data, **kwargs)

        # Load detection matrix and object map with proper dtype casting.
        # Backward compat: try 'detect_mat' first, fall back to 'enh_gray'.
        if "detect_mat" in group:
            detect_mat_ds = group["detect_mat"]
            detect_matrix_data = detect_mat_ds[()]
            detect_mode = detect_mat_ds.attrs.get("detect_mode", "gray")
            if isinstance(detect_mode, bytes):
                detect_mode = detect_mode.decode("utf-8", errors="replace")
        else:
            detect_matrix_data = group["enh_gray"][()]
            detect_mode = "gray"
        img.detect_mat[:] = detect_matrix_data
        img._data.detect_mode = detect_mode

        # Object map should preserve its original dtype (usually integer labels).
        img.objmap[:] = group["objmap"][()]

        # Restore metadata — legacy coercion: digit-looking values become ints.
        if "protected_metadata" in group:
            prot = group["protected_metadata"].attrs
            img._metadata.protected.clear()
            img._metadata.protected.update(
                    {
                        _remap_legacy_metadata_key(k): int(prot[k])
                        if isinstance(prot[k], str) and prot[k].isdigit()
                        else prot[k]
                        for k in prot
                    }
            )

        if "public_metadata" in group:
            pub = group["public_metadata"].attrs
            img._metadata.public.clear()
            img._metadata.public.update(
                    {
                        _remap_legacy_metadata_key(k): int(pub[k])
                        if isinstance(pub[k], str) and pub[k].isdigit()
                        else pub[k]
                        for k in pub
                    }
            )

        # Legacy files never stored imported metadata — leave dict empty.
        return img

    @classmethod
    def load_hdf5(cls, filename, **kwargs) -> Image:
        """
        Loads an image object from an HDF5 file.

        Args:
            filename (str): The path to the HDF5 file containing the image data. Providing an incorrect or
                improperly structured file may result in incomplete or corrupted image data being loaded.
            **kwargs: Additional keyword arguments that may alter the image import and processing behavior.
                These settings may include options for subsampling, resizing, or preprocessing specific
                to the characteristics of the microbe images. Proper configurations can optimize the
                processing performance while preserving essential colony features critical to analysis.

        Returns:
            Image: An image object constructed from the data stored in the HDF5 file. The fidelity
            of this object depends on both the input file's quality and the parameters provided
            through kwargs, as they affect the image's suitability for detailed microbe colony studies.
        """
        with h5py.File(filename, "r") as filehandler:
            # Auto-dispatch warning: if a user calls Image.load_hdf5 on a file
            # that was saved as a GridImage, warn but do NOT upcast — still
            # return a plain Image so behaviour stays explicit.
            saved_class = filehandler.attrs.get("phenotypic_class")
            if isinstance(saved_class, bytes):
                saved_class = saved_class.decode("utf-8", errors="replace")
            if (
                    saved_class == "GridImage"
                    and cls.__name__ != "GridImage"
            ):
                warnings.warn(
                        "File was saved as GridImage; "
                        "use GridImage.load_hdf5 to preserve grid state",
                        UserWarning,
                        stacklevel=2,
                )
            img = cls._load_from_hdf5_group(filehandler, **kwargs)

        return img

    @classmethod
    def load_layer_hdf5(cls, filename, layer: str) -> np.ndarray:
        """Read a single image layer from an intermediate HDF5 file.

        Reads only the requested dataset without reconstructing a full
        :class:`Image`. Handles both the schema-v2 grouped layout
        (``/layers/<layer>``) and the legacy flat layout (``/<layer>``).

        Args:
            filename: Path to the HDF5 file.
            layer: One of ``"rgb"``, ``"gray"``, ``"detect_mat"``, ``"objmap"``.

        Returns:
            The layer array.

        Raises:
            KeyError: If *layer* is not present in the file.
        """
        with h5py.File(filename, "r") as f:
            schema_version = int(f.attrs.get("schema_version", 1))
            if schema_version >= _SCHEMA_VERSION and "layers" in f:
                group = f["layers"]
            else:
                group = f
            if layer not in group:
                raise KeyError(
                    f"Layer {layer!r} not found in {filename}"
                )
            return group[layer][()]

    def save2pickle(self, filename: str) -> None:
        """Save the image to a pickle file for fast serialization and deserialization.

        Stores all image data components and metadata in Python's pickle format, which
        preserves data types and structure exactly. This is the fastest serialization
        method but produces larger files than HDF5 and is not suitable for inter-language
        data exchange.

        Args:
            filename (str | PathLike): Path to the pickle file to write (.pkl or .pickle
                extension recommended).

        Notes:
            - Pickle format is Python-specific and cannot be read by other languages.
            - File size is typically larger than HDF5 compressed files.
            - Load/save is faster than HDF5 for small to medium images.
            - Pickle files may not be compatible across Python versions.

        Examples:
            Save to pickle:

            >>> img = Image.imread('photo.jpg')
            >>> img.save2pickle('image.pkl')
            >>> loaded = Image.load_pickle('image.pkl')
        """
        with open(filename, "wb") as filehandler:
            data2save = {
                "_data.rgb"         : self._data.rgb,
                "_data.gray"        : self._data.gray,
                "_data.detect_mat"  : self._data.detect_mat,
                "_data.detect_mode" : self._data.detect_mode,
                "objmap"            : self.objmap[:],
                "protected_metadata": self._metadata.protected,
                "public_metadata"   : self._metadata.public,
            }

            if hasattr(self, "grid_finder"):
                data2save["grid_finder"] = self.grid_finder

            pickle.dump(data2save, filehandler)

    @classmethod
    def load_pickle(cls, filename: str | Path | PathLike) -> Image | GridImage:
        """Load an Image or GridImage from a pickle file with automatic type detection.

        Deserializes image data and metadata that were previously saved with save2pickle().
        Automatically detects if the pickle contains a GridFinder and instantiates the
        appropriate class (GridImage if grid_finder is present, Image otherwise).
        Restores all image components including RGB, grayscale, detection matrix, object map,
        metadata, and GridFinder (if present).

        Args:
            filename (str | PathLike): Path to the pickle file to read.

        Returns:
            Image | GridImage: A new Image or GridImage instance with all data and metadata
                restored from the pickle file. Returns GridImage if the pickle contains a
                grid_finder, otherwise returns Image.

        Raises:
            FileNotFoundError: If the specified pickle file does not exist.
            pickle.UnpicklingError: If the file is not a valid pickle file or is corrupted.

        Notes:
            - Pickle files must be created with save2pickle() to ensure compatibility.
            - Detection matrix and object map are reset and reconstructed from saved data.
            - Metadata (protected and public) is fully restored.
            - GridFinder is restored if present in the pickle (GridImage only).
            - Backward compatible: Old pickles without grid_finder load as regular Image.

        Examples:
            Load GridImage from pickle:

            >>> from phenotypic import Image, GridImage
            >>> from phenotypic.grid import AutoGridFinder
            >>> # Save a GridImage
            >>> finder = AutoGridFinder(nrows=16, ncols=24)
            >>> grid_img = GridImage('plate.jpg', grid_finder=finder)
            >>> grid_img.save2pickle('plate.pkl')
            >>> # Load automatically returns GridImage
            >>> loaded = Image.load_pickle('plate.pkl')
            >>> isinstance(loaded, GridImage)  # True
            >>> loaded.nrows  # 16
            >>> loaded.ncols  # 24

            Load regular Image from pickle:

            >>> img = Image.imread('photo.jpg')
            >>> img.save2pickle('photo.pkl')
            >>> loaded = Image.load_pickle('photo.pkl')
            >>> isinstance(loaded, Image)  # True
        """
        # Pickle trust model is unchanged from the prior ``pickle.load`` call:
        # callers load their own previously-saved image pickles. The custom
        # unpickler only remaps the moved ``METADATA`` class so pre-move pickles
        # still load; it does not relax any trust assumptions.
        with open(filename, "rb") as f:
            loaded = _BackCompatUnpickler(f).load()  # noqa: S301 - user's own pickle

        # Check if pickle contains grid_finder -> use GridImage
        has_grid_finder = "grid_finder" in loaded

        if has_grid_finder:
            # Import here to avoid circular dependency
            from phenotypic._core._grid_image import GridImage

            target_class = GridImage
            grid_finder = loaded["grid_finder"]
        else:
            target_class = cls  # Use Image or whatever class this was called on
            grid_finder = None

        # Create instance with appropriate type
        # GridImage constructor accepts grid_finder parameter
        if loaded["_data.rgb"].size > 0:
            if has_grid_finder:
                instance = target_class(arr=loaded["_data.rgb"], name=None,
                                        grid_finder=grid_finder)
            else:
                instance = target_class(arr=loaded["_data.rgb"], name=None)
        else:
            if has_grid_finder:
                instance = target_class(arr=loaded["_data.gray"], name=None,
                                        grid_finder=grid_finder)
            else:
                instance = target_class(arr=loaded["_data.gray"], name=None)

        # Restore detection matrix, object map, and metadata
        instance.detect_mat.reset()
        instance.objmap.reset()

        # Backward compat: old pickles use '_data.enh_gray', new use '_data.detect_mat'
        instance._data.detect_mat = loaded.get(
                "_data.detect_mat", loaded.get("_data.enh_gray")
        )
        instance._data.detect_mode = loaded.get("_data.detect_mode", "gray")
        instance.objmap[:] = loaded["objmap"]
        # Remap legacy bare metadata keys (pre-Metadata_ prefix) to their current
        # prefixed form. No-op for new pickles (keys are already prefixed) and for
        # arbitrary user keys.
        instance._metadata.protected = {
                _remap_legacy_metadata_key(k): v
                for k, v in loaded["protected_metadata"].items()
        }
        instance._metadata.public = {
                _remap_legacy_metadata_key(k): v
                for k, v in loaded["public_metadata"].items()
        }

        # If mode is not 'gray', reinitialize to ensure correct source channel
        if instance._data.detect_mode != "gray":
            instance.set_detect_mode(instance._data.detect_mode)

        return instance
