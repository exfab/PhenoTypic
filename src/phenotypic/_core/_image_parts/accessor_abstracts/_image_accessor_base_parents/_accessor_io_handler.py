from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import skimage as ski
from PIL import Image as PIL_Image

import phenotypic
from phenotypic.sdk_.constants_ import IO

from ._accessor_data_interface import AccessorDataInterface

if TYPE_CHECKING:
    pass


class AccessorIOHandler(AccessorDataInterface):
    """File I/O layer — loading, saving, metadata extraction and embedding.

    Provides ``load`` / ``imsave`` / ``save_overlay`` and all format-specific
    helper methods (JPEG, PNG, TIFF).
    """

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, filepath: str | Path) -> np.ndarray:
        """Load an image array from file and verify it was saved from this accessor type.

        Checks if the image contains PhenoTypic metadata indicating it was saved
        from the same accessor type (e.g., Image.gray, Image.rgb). If metadata
        doesn't match or is missing, a warning is raised but the array is still loaded.

        Args:
            filepath: Path to the image file to load.

        Returns:
            np.ndarray: The loaded image array.

        Warns:
            UserWarning: If metadata is missing or indicates the image was saved
                from a different accessor type.

        Examples:
            Load a grayscale image from file:

            >>> from phenotypic import Image
            >>> image = Image(arr)
            >>> # load an object map you saved or hand-graded
            >>> image.objmap.load("path/to/map.png")
        """
        filepath = Path(filepath)
        expected_property = f"Image.{cls._accessor_property_name_value()}"

        # Load the array using cv2 for reliable uint16 round-trip
        import cv2

        arr = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(
                f"Could not read image file: {filepath}. "
                "File may not exist, be corrupt, or be in an "
                "unsupported format."
            )
        # cv2 loads colour images as BGR/BGRA; convert to RGB/RGBA
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
            elif arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

        # Try to extract and verify PhenoTypic metadata
        phenotypic_data = cls._extract_phenotypic_metadata(filepath)

        if phenotypic_data is None:
            warnings.warn(
                f"No PhenoTypic metadata found in '{filepath.name}'. "
                f"Cannot verify this image was saved from {expected_property}. "
                "Loading anyway, but this may lead to undefined behavior.",
                UserWarning,
            )
        else:
            saved_property = phenotypic_data.get("phenotypic_image_property", "unknown")
            if saved_property != expected_property:
                warnings.warn(
                    f"Metadata mismatch: Image was saved from '{saved_property}' "
                    f"but being loaded as '{expected_property}'. "
                    "This may lead to undefined behavior.",
                    UserWarning,
                )

        return arr

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    @classmethod
    def _extract_phenotypic_metadata(cls, filepath: Path) -> dict | None:
        """Extract PhenoTypic metadata from an image file.

        Args:
            filepath: Path to the image file.

        Returns:
            dict or None: The PhenoTypic metadata dict if found, None otherwise.
        """
        suffix = filepath.suffix.lower()

        try:
            if suffix in IO.PNG_FILE_EXTENSIONS:
                with PIL_Image.open(filepath) as img:
                    phenotypic_json = img.info.get(IO.PHENOTYPIC_METADATA_KEY)
                    if phenotypic_json:
                        return json.loads(phenotypic_json)

            elif suffix in IO.JPEG_FILE_EXTENSIONS:
                # Try exiftool for JPEG UserComment
                if shutil.which("exiftool"):
                    result = subprocess.run(
                        ["exiftool", "-json", "-UserComment", str(filepath)],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        exif_data = json.loads(result.stdout)
                        user_comment = (
                            exif_data[0].get("UserComment") if exif_data else None
                        )
                        if user_comment:
                            data = json.loads(user_comment)
                            if "phenotypic_version" in data:
                                return data

            elif suffix in IO.TIFF_EXTENSIONS:
                with PIL_Image.open(filepath) as img:
                    desc = img.tag_v2.get(270) if hasattr(img, "tag_v2") else None
                    if desc:
                        try:
                            data = json.loads(desc)
                            if "phenotypic_version" in data:
                                return data
                        except json.JSONDecodeError:
                            pass

        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Metadata building
    # ------------------------------------------------------------------

    def _build_phenotypic_metadata(self) -> dict:
        """Build PhenoTypic metadata dictionary for embedding in saved images.

        Returns:
            Dictionary containing phenotypic version, source property, and metadata.
        """
        # Filter out None values and convert to JSON-serializable types
        protected = {}
        for key, value in self._root_image._metadata.protected.items():
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                protected[str(key)] = value

        public = {}
        for key, value in self._root_image._metadata.public.items():
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                public[str(key)] = value

        return {
            "phenotypic_version": phenotypic.__version__,
            "phenotypic_image_property": f"Image.{self._accessor_property_name}",
            "protected": protected,
            "public": public,
        }

    def _check_bit_depth(self, bit_depth: int | None) -> Literal[8, 16]:
        if bit_depth is None:
            bit_depth = self._root_image.bit_depth
        elif bit_depth not in [8, 16]:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")

        return bit_depth

    # ------------------------------------------------------------------
    # Format-specific writers (static)
    # ------------------------------------------------------------------

    @staticmethod
    def _write_jpeg_metadata(filepath: Path, pil_image, metadata_json: str) -> None:
        """Write metadata to JPEG file using EXIF UserComment tag via exiftool.

        Args:
            filepath: Path to save the JPEG file.
            pil_image: PIL Image object to save.
            metadata_json: JSON string of PhenoTypic metadata.
        """
        # First save the image
        pil_image.save(filepath, quality=100)

        # Then add metadata using exiftool if available
        if shutil.which("exiftool"):
            try:
                subprocess.run(
                    [
                        "exiftool",
                        "-overwrite_original",
                        f"-UserComment={metadata_json}",
                        str(filepath),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                warnings.warn(f"Failed to write EXIF metadata to JPEG: {e}")
        else:
            warnings.warn(
                "exiftool not found. JPEG metadata will not be saved. "
                "Install exiftool for full metadata support."
            )

    @staticmethod
    def _inject_png_text_chunk(
        filepath: Path, key: str, value: str
    ) -> None:
        """Inject a tEXt metadata chunk into an existing PNG file.

        Inserts the chunk immediately after IHDR without re-encoding
        pixel data.

        Args:
            filepath: Path to the PNG file.
            key: Metadata key (latin-1 encodable, max 79 chars).
            value: Metadata value (latin-1 encodable).
        """
        import struct
        import zlib

        with open(filepath, "rb") as f:
            data = f.read()

        # PNG: 8-byte signature + IHDR chunk
        # (4 len + 4 type + 13 data + 4 CRC = 25 bytes)
        ihdr_end = 33

        chunk_data = (
            key.encode("latin-1") + b"\x00" + value.encode("latin-1")
        )
        chunk_type = b"tEXt"
        chunk = (
            struct.pack(">I", len(chunk_data))
            + chunk_type
            + chunk_data
            + struct.pack(
                ">I", zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
            )
        )

        with open(filepath, "wb") as f:
            f.write(data[:ihdr_end] + chunk + data[ihdr_end:])

    @staticmethod
    def _write_png_cv2(
        filepath: Path,
        arr: np.ndarray,
        metadata_json: str | None,
    ) -> None:
        """Save a uint16 array as a 16-bit PNG using OpenCV.

        Args:
            filepath: Destination path.
            arr: uint16 array (2-D grayscale or 3-D RGB).
            metadata_json: Optional JSON metadata to embed as a
                tEXt chunk.
        """
        import cv2

        # cv2 expects BGR for colour images
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, ::-1]

        cv2.imwrite(str(filepath), arr)

        if metadata_json:
            AccessorIOHandler._inject_png_text_chunk(
                filepath, IO.PHENOTYPIC_METADATA_KEY, metadata_json
            )

    @staticmethod
    def _write_png_metadata(filepath: Path, pil_image, metadata_json: str) -> None:
        """Write metadata to PNG file using tEXt chunk.

        Args:
            filepath: Path to save the PNG file.
            pil_image: PIL Image object to save.
            metadata_json: JSON string of PhenoTypic metadata.
        """
        from PIL import PngImagePlugin

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text(IO.PHENOTYPIC_METADATA_KEY, metadata_json)
        pil_image.save(filepath, optimize=True, pnginfo=pnginfo)

    @staticmethod
    def _write_tiff_tifffile(
        filepath: Path,
        arr: np.ndarray,
        metadata_json: str | None,
    ) -> None:
        """Save a uint16 array as a 16-bit TIFF using tifffile.

        Uses tifffile for lossless uint16 TIFF writing with metadata
        support. This avoids PIL's limitation with multi-channel uint16
        arrays.

        Args:
            filepath: Destination path.
            arr: uint16 array (2-D grayscale or 3-D RGB).
            metadata_json: Optional JSON metadata to embed as TIFF
                ImageDescription tag.
        """
        import tifffile

        photometric = "rgb" if arr.ndim == 3 and arr.shape[2] >= 3 else "minisblack"
        tifffile.imwrite(
            filepath,
            arr,
            description=metadata_json if metadata_json else None,
            photometric=photometric,
        )

    @staticmethod
    def _write_tiff_metadata(filepath: Path, pil_image, metadata_json: str) -> None:
        """Write metadata to TIFF file using ImageDescription tag.

        Args:
            filepath: Path to save the TIFF file.
            pil_image: PIL Image object to save.
            metadata_json: JSON string of PhenoTypic metadata.
        """
        # TIFF ImageDescription tag is 270
        pil_image.save(filepath, tiffinfo={270: metadata_json})

    # ------------------------------------------------------------------
    # Core save logic
    # ------------------------------------------------------------------

    def _save_image(
        self,
        filepath: Path,
        arr: np.ndarray,
        bit_depth: Literal[8, 16],
        metadata_json: str | None,
    ) -> None:
        """Save an image array to disk with embedded PhenoTypic metadata.

        Args:
            filepath: Destination file path including extension.
            arr: Image data to save.
            bit_depth: Target bit depth used when coercing float arrays for PNG.
            metadata_json: JSON string containing PhenoTypic metadata to embed.

        Raises:
            ValueError: If the file extension is not supported.

        Warns:
            UserWarning: When saving arrays that require downcasting and may lose
                information (e.g., float or 16-bit arrays to JPEG, float arrays to PNG).
        """
        filepath = Path(filepath)
        arr2save = arr
        suffix = filepath.suffix.lower()

        match suffix:
            case x if x in IO.JPEG_FILE_EXTENSIONS:
                match arr2save.dtype:
                    case np.uint8:
                        pass
                    case np.uint16:
                        warnings.warn(
                            "Saving a 16 bit array as a jpeg will potentially "
                            "result in information loss during conversion"
                        )
                        arr2save = ski.util.img_as_ubyte(arr2save)
                    case dt if np.issubdtype(dt, np.floating):
                        warnings.warn(
                            "Saving a float array as a jpeg will potentially"
                            "result in information loss during conversion"
                        )
                        arr2save = ski.util.img_as_ubyte(arr2save)
                pil_img = PIL_Image.fromarray(arr2save)
                if metadata_json:
                    self._write_jpeg_metadata(filepath, pil_img, metadata_json)
                else:
                    pil_img.save(filepath)

            case x if x in IO.PNG_FILE_EXTENSIONS:
                match arr2save.dtype:
                    case np.uint8:
                        pass
                    case np.uint16:
                        pass  # preserve uint16 for 16-bit PNG
                    case dt if np.issubdtype(dt, np.floating):
                        warnings.warn(
                            ".png images only accept 8 bit and 16 bit "
                            "integer arrays. Converting this array may cause "
                            "information loss"
                        )
                        arr2save = (
                            ski.util.img_as_ubyte(arr2save)
                            if bit_depth == 8
                            else ski.util.img_as_uint(arr2save)
                        )

                if arr2save.dtype == np.uint16:
                    self._write_png_cv2(filepath, arr2save, metadata_json)
                else:
                    pil_img = PIL_Image.fromarray(arr2save)
                    if metadata_json:
                        self._write_png_metadata(
                            filepath, pil_img, metadata_json
                        )
                    else:
                        pil_img.save(filepath)

            case x if x in IO.TIFF_EXTENSIONS:
                if arr2save.dtype == np.uint16:
                    self._write_tiff_tifffile(filepath, arr2save, metadata_json)
                else:
                    pil_img = PIL_Image.fromarray(arr2save)
                    if metadata_json:
                        self._write_tiff_metadata(filepath, pil_img, metadata_json)
                    else:
                        pil_img.save(filepath)

            case _:
                raise ValueError(f"unknown file extension for saving:{filepath.suffix}")

    # ------------------------------------------------------------------
    # Public save API
    # ------------------------------------------------------------------

    def imsave(
        self, filepath: str | Path | None = None, bit_depth: Literal[8, 16] | None = None
    ) -> None:
        """
        Saves an array representing a microbe colony image to a specified file format while preserving or adjusting
        metadata and pixel depth as needed. Supports JPEG, PNG, and TIFF formats.

        The behavior of the function is context-sensitive based on the
        file format's restrictions and array properties. Proper file format selection
        and bit depth adjustment can have an impact on the accuracy of image analysis
        and preservation of data integrity.

        Args:
            filepath (str | Path | None): The destination file path where the image will be saved. The extension of the
                file path determines the image format (e.g., .jpeg, .png, .tiff). Changing the file format influences how
                the image data is handled during saving:

                    1. `.jpeg`: Compression or loss of data may occur. Maximal value limit (255) for uint8 pixel
                       depth affects the fidelity of rich intensity details in microbe colonies.
                    2. `.png`: Retains high-quality output but supports only 8-bit or 16-bit images. Conversions may
                       occur if the array has a different data type, which could result in data loss.
                    3. `.tiff`: Ideal for high-bit-depth precision and analysis preservation; best for maintaining
                       intricate morphological details of microbial colonies.

            bit_depth (Literal[8, 16] | None, optional): Specifies the bit depth of the saved image (either 8-bit or
                16-bit). The provided bit depth must align with the file format's capabilities. Misalignment could
                trigger conversion with possible data truncation or rounding. For example:

                    - 8-bit: Useful for efficiently representing intensity when detail is moderate, suitable for JPEG
                      or simple PNG outputs.
                    - 16-bit: Allows for higher intensity ranges, especially valuable for preserving subtle
                      morphological gradient differentiation when analyzing colonies.

        Raises:
            ValueError: An error occurs when an unsupported file extension is provided in `filepath`.

        Warns:
            UserWarnings: Warnings are issued under the following conditions:

                - Saving a 16-bit or floating-point array as JPEG, as these conversions may cause information loss due
                  to format restrictions.
                - Saving a floating-point array as PNG when conversions to 8-bit or 16-bit integers might lead to truncated
                  or altered pixel intensity values.
        """
        bit_depth = self._check_bit_depth(bit_depth)

        filepath = Path(filepath)

        arr2save = self._subject_arr

        # Build metadata JSON
        phenotypic_metadata = self._build_phenotypic_metadata()
        metadata_json = json.dumps(phenotypic_metadata, ensure_ascii=True)

        self._save_image(
            filepath=filepath,
            arr=arr2save,
            bit_depth=bit_depth,
            metadata_json=metadata_json,
        )

    # ------------------------------------------------------------------
    # Overlay generation & saving
    # ------------------------------------------------------------------

    def _generate_overlay_array(
        self,
        overlay_alpha: float = 0.3,
        bg_label: int = 0,
        colors: list | None = None,
        **label2rgb_kwargs,
    ) -> np.ndarray:
        """Generate a full-resolution overlay array blending objmap with the subject image.

        Creates an RGB overlay by blending the object map labels with the underlying
        image data using skimage.color.label2rgb. Unlike show(overlay=True) which
        returns a matplotlib figure, this returns the raw array suitable for pixel-level
        inspection and high-resolution saving.

        Args:
            overlay_alpha: Alpha value for label overlay (0.0 = transparent,
                1.0 = opaque). Higher values make the colored labels more prominent.
                Defaults to 0.3.
            bg_label: Label value to treat as background (will be transparent).
                Defaults to 0.
            colors: List of RGB colors to use for labels. If None, uses default
                label2rgb colormap.
            **label2rgb_kwargs: Additional keyword arguments passed to
                skimage.color.label2rgb.

        Returns:
            np.ndarray: 8-bit RGB array (dtype uint8, shape H x W x 3) containing
                the blended overlay image.
        """
        arr = self._subject_arr
        objmap = self._root_image.objmap[:]

        # Handle grayscale images: normalize and convert to 3-channel for label2rgb
        if arr.ndim == 2:
            if np.issubdtype(arr.dtype, np.floating):
                arr_norm = arr
            else:
                arr_norm = arr.astype(np.float64) / np.iinfo(arr.dtype).max
            # Stack to create 3-channel grayscale image
            arr_rgb = np.stack([arr_norm] * 3, axis=-1)
        else:
            # RGB image: normalize to [0, 1] for label2rgb
            if np.issubdtype(arr.dtype, np.floating):
                arr_rgb = arr
            else:
                arr_rgb = arr.astype(np.float64) / np.iinfo(arr.dtype).max

        # Build label2rgb kwargs
        kwargs = {
            "label": objmap,
            "image": arr_rgb,
            "bg_label": bg_label,
            "alpha": overlay_alpha,
        }
        if colors is not None:
            kwargs["colors"] = colors
        kwargs.update(label2rgb_kwargs)

        # Generate overlay using label2rgb
        import skimage as ski
        overlay_arr = ski.color.label2rgb(**kwargs)

        # Convert to 8-bit uint8 for saving
        overlay_uint8 = (overlay_arr * 255).astype(np.uint8)

        return overlay_uint8

    def save_overlay(
        self,
        filepath: str | Path,
        overlay_alpha: float = 0.3,
        bg_label: int = 0,
        colors: list | None = None,
        show_grid: bool = True,
        gridline_color: tuple[int, int, int] = (0, 255, 255),
        section_box_colors: list[tuple[int, int, int]] | None = None,
        **label2rgb_kwargs,
    ) -> None:
        """Save a full-resolution overlay image blending objmap with the subject array.

        Creates an RGB overlay by blending the object map labels with the underlying
        image data and saves it to disk. Unlike show(overlay=True) which produces a
        matplotlib figure, this method saves the raw pixel data at full resolution,
        suitable for pixel-level quality validation of detection results.

        For GridImage objects, gridlines and section boxes are automatically drawn
        when ``show_grid`` is True. The line widths scale dynamically with image
        size.

        Args:
            filepath: Destination file path. Should have .png or .jpeg extension.
            overlay_alpha: Alpha value for label overlay (0.0 = transparent,
                1.0 = opaque). Defaults to 0.3.
            bg_label: Label value to treat as background. Defaults to 0.
            colors: List of RGB colors to use for labels. If None, uses default
                colormap.
            show_grid: Whether to draw gridlines and section boxes on overlay
                for GridImage objects. Ignored for regular Image objects.
                Defaults to True.
            gridline_color: RGB color tuple for gridlines. Defaults to cyan
                (0, 255, 255).
            section_box_colors: List of RGB tuples for cycling through section
                box colors. Defaults to tab20 colormap colors.
            **label2rgb_kwargs: Additional keyword arguments for label2rgb.

        Raises:
            ValueError: If the file extension is not supported.

        Examples:
            Save full-resolution overlay:

            >>> from phenotypic.data import load_synth_yeast_plate
            >>> image = load_synth_yeast_plate()
            >>> image.rgb.save_overlay("overlay_rgb.png", overlay_alpha=0.4)
        """
        filepath = Path(filepath)

        # Generate full-resolution overlay array
        overlay_arr = self._generate_overlay_array(
            overlay_alpha=overlay_alpha,
            bg_label=bg_label,
            colors=colors,
            **label2rgb_kwargs,
        )

        # For GridImage, draw gridlines and section boxes (duck typing check)
        if show_grid:
            if hasattr(self._root_image, "_draw_gridlines_on_overlay"):
                overlay_arr = self._root_image._draw_gridlines_on_overlay(
                    overlay_arr, gridline_color
                )
            if hasattr(self._root_image, "_draw_section_boxes_on_overlay"):
                overlay_arr = self._root_image._draw_section_boxes_on_overlay(
                    overlay_arr, section_box_colors
                )

        # Save using existing _save_image infrastructure (no metadata for overlays)
        self._save_image(
            filepath=filepath,
            arr=overlay_arr,
            bit_depth=8,  # Overlays are always 8-bit
            metadata_json=None,  # No phenotypic metadata for overlay images
        )
