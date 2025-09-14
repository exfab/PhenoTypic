import json
import posixpath
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
from packaging.version import Version

import phenotypic


class HDF:
    """
    Represents an interface to manage HDF5 files with support for single or set image modes,
    and ensures safe and compatible file access with retry and error-handling mechanisms.

    The class facilitates operations on HDF5 files commonly used for storing phenotypic
    data in both single image and image set modes. This class includes utilities to
    handle locking errors and ensure compatibility by initializing proper HDF5 modes
    while providing safe access methods for writing.

    Attributes:
        filepath (Path): Path to the HDF5 file on the filesystem.
        name (str): Name associated with the HDF5 resource, often used as an identifier.
        mode (Literal['single', 'set']): Specifies the mode for the HDF5 file, either
            single image or image set.
        root_posix (str): The root path for the HDF5 resource, determined by the mode.
        home_posix (str): The specific root directory of the HDF5 resource in the file,
            derived based on its mode.
        set_data_posix (str, optional): The subgroup path for the data entity in image
            set mode, if applicable.
        SINGLE_IMAGE_ROOT_POSIX (str): Base path for single image mode.
        IMAGE_SET_ROOT_POSIX (str): Base path for image set mode.
        IMAGE_SET_DATA_POSIX (str): Subgroup marker for image set data.
        EXT (set): Set of valid file extensions used to recognize HDF5 files.
        IMAGE_MEASUREMENT_SUBGROUP_KEY (str): Key for accessing measurements in an
            image's group.
        IMAGE_STATUS_SUBGROUP_KEY (str): Key for accessing statuses in an image's group.
    """

    if Version(phenotypic.__version__) < Version("0.7.1"):
        SINGLE_IMAGE_ROOT_POSIX = f'/phenotypic/'
    else:
        SINGLE_IMAGE_ROOT_POSIX = f'/phenotypic/images/'

    IMAGE_SET_ROOT_POSIX = f'/phenotypic/image_sets/'
    IMAGE_SET_DATA_POSIX = 'data'  # The image and individual measurement group

    # measurements and status are stored within in each image's group
    IMAGE_MEASUREMENT_SUBGROUP_KEY = 'measurements'
    IMAGE_STATUS_SUBGROUP_KEY = "status"

    EXT = {'.h5', '.hdf5', '.hdf', '.he5'}

    def __init__(self, filepath, name: str, mode: Literal['single', 'set']):
        self.filepath = Path(filepath)
        self.name = name
        self.mode = mode
        if mode == 'single':
            self.root_posix = self.SINGLE_IMAGE_ROOT_POSIX
            self.home_posix = posixpath.join(self.SINGLE_IMAGE_ROOT_POSIX, self.name)
        elif mode == 'set':
            self.root_posix = self.IMAGE_SET_ROOT_POSIX
            self.home_posix = posixpath.join(self.IMAGE_SET_ROOT_POSIX, self.name)
            self.set_data_posix = posixpath.join(self.home_posix, self.IMAGE_SET_DATA_POSIX)
        else:
            raise ValueError(f"Invalid mode {mode}")

    def safe_writer(self) -> h5py.File:
        """
        Returns a writer object that provides safe and controlled write access to an
        HDF5 file at the specified filepath or creates it if it doesn't exist. Ensures that the file uses the 'latest'
        version of the HDF5 library for compatibility and performance.
        
        Handles HDF5 file locking conflicts by attempting to clear consistency flags
        and retrying file opening with exponential backoff.

        Returns:
            h5py.File: A file writer object with append mode and 'latest' library
            version enabled.
            
        Raises:
            OSError: If file cannot be opened after all retry attempts.
        """
        import time
        import subprocess
        import os
        import logging

        logger = logging.getLogger(__name__)
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                return h5py.File(self.filepath, 'a', libver='latest')
            except OSError as e:
                error_msg = str(e).lower()
                # Handle various HDF5 locking scenarios
                is_lock_error = any([
                    "file is already open for write/swmr write" in error_msg,
                    "file is already open" in error_msg,
                    "unable to lock file" in error_msg,
                    "resource temporarily unavailable" in error_msg,
                    "file locking disabled" in error_msg
                ])

                if is_lock_error:
                    logger.warning(f"HDF5 file access conflict (attempt {attempt + 1}/{max_retries}): {e}")

                    # Try to clear HDF5 consistency flags if h5clear is available
                    if attempt < max_retries - 1:  # Don't try h5clear on last attempt
                        try:
                            if os.path.exists(self.filepath):
                                logger.info(f"Attempting to clear HDF5 consistency flags for {self.filepath}")
                                result = subprocess.run(['h5clear', '-s', str(self.filepath)],
                                                        capture_output=True, text=True, timeout=10)
                                if result.returncode == 0:
                                    logger.info("Successfully cleared HDF5 consistency flags")
                                else:
                                    logger.warning(f"h5clear failed: {result.stderr}")
                        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as clear_error:
                            logger.warning(f"Could not run h5clear: {clear_error}")

                        # Wait before retrying
                        logger.info(f"Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        # Last attempt failed - provide helpful error message
                        logger.error(f"Failed to open HDF5 file after {max_retries} attempts")
                        raise RuntimeError(f"Failed to open HDF5 file after {max_retries} attempts. "
                                           f"The file {self.filepath} may be locked by another process. "
                                           f"Try manually running: h5clear -s {self.filepath} && h5clear -f {self.filepath}") from e
                else:
                    # Different OSError, re-raise immediately
                    raise

        # This should not be reached due to the raise in the loop
        raise OSError(f"Unexpected error opening HDF5 file {self.filepath}")

    def swmr_writer(self) -> h5py.File:
        """
        Returns a writer object that provides safe SWMR-compatible write access to an
        HDF5 file. Creates the file if it doesn't exist and enables SWMR mode properly.
        
        This method ensures proper SWMR mode initialization by creating the file
        with the correct settings from the start, avoiding cache conflicts that
        occur when trying to enable SWMR mode after opening.
        
        Returns:
            h5py.File: A file writer object with SWMR mode enabled.
            
        Raises:
            OSError: If file cannot be opened after all retry attempts.
        """
        import time
        import subprocess
        import os
        import logging

        logger = logging.getLogger(__name__)
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                # Create/open file with proper SWMR settings
                file_handle = h5py.File(self.filepath, 'a', libver='latest')
                # Enable SWMR mode immediately after opening
                try:
                    file_handle.swmr_mode = True
                    logger.debug(f"SWMR mode enabled successfully for {self.filepath}")
                    return file_handle
                except Exception as swmr_error:
                    logger.warning(f"Could not enable SWMR mode: {swmr_error}")
                    # Return file handle without SWMR mode as fallback
                    return file_handle

            except OSError as e:
                error_msg = str(e).lower()
                # Handle various HDF5 locking scenarios
                is_lock_error = any([
                    "file is already open for write/swmr write" in error_msg,
                    "file is already open" in error_msg,
                    "unable to lock file" in error_msg,
                    "resource temporarily unavailable" in error_msg,
                    "file locking disabled" in error_msg,
                    "ring type mismatch" in error_msg,
                    "pinned entry count" in error_msg
                ])

                if is_lock_error:
                    logger.warning(f"HDF5 SWMR file access conflict (attempt {attempt + 1}/{max_retries}): {e}")

                    # Try to clear HDF5 consistency flags if h5clear is available
                    if attempt < max_retries - 1:  # Don't try h5clear on last attempt
                        try:
                            if os.path.exists(self.filepath):
                                logger.info(f"Attempting to clear HDF5 consistency flags for {self.filepath}")
                                # Clear both status and force flags for SWMR issues
                                subprocess.run(['h5clear', '-s', str(self.filepath)],
                                               capture_output=True, text=True, timeout=10)
                                subprocess.run(['h5clear', '-f', str(self.filepath)],
                                               capture_output=True, text=True, timeout=10)
                                logger.info("Cleared HDF5 consistency flags")
                        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as clear_error:
                            logger.warning(f"Could not run h5clear: {clear_error}")

                        # Wait before retrying
                        logger.info(f"Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        # Last attempt failed - provide helpful error message
                        logger.error(f"Failed to open HDF5 file in SWMR mode after {max_retries} attempts")
                        raise RuntimeError(f"Failed to open HDF5 file in SWMR mode after {max_retries} attempts. "
                                           f"The file {self.filepath} may have cache conflicts. "
                                           f"Try manually running: h5clear -s {self.filepath} && h5clear -f {self.filepath}") from e
                else:
                    # Different OSError, re-raise immediately
                    raise

        # This should not be reached due to the raise in the loop
        raise OSError(f"Unexpected error opening HDF5 file in SWMR mode {self.filepath}")

    def writer(self) -> h5py.File:
        """
        Provides access to an HDF5 file in read/write mode using the `h5py` library. This
        property is used to obtain an `h5py.File` object configured with the latest library version.

        Note:
            If using SWMR mode, don't forget to enable SWMR mode with:
                .. code-block:: python
                    hdf = HDF(filepath)
                    with hdf.writer as writer:
                        writer.swmr_mode = True
                        # rest of your code

        Returns:
            h5py.File: An HDF5 file object opened in 'r+' mode, enabling reading and writing.

        Raises:
            OSError: If the file cannot be opened or accessed.
        """
        return h5py.File(self.filepath, 'r+', libver='latest')

    def swmr_reader(self) -> h5py.File:
        return h5py.File(self.filepath, 'r', libver='latest', swmr=True)

    def reader(self) -> h5py.File:
        return h5py.File(self.filepath, 'r', libver='latest', swmr=True)

    @staticmethod
    def get_group(handle: h5py.File, posix) -> h5py.Group:
        """
        Retrieves or creates a group in an HDF5 file.

        This method checks the validity of the provided HDF5 file handle and tries to
        retrieve the specified group based on the given posix path. If the group does not
        exist and the file is not opened in read-only mode, the group gets created. If the
        file is in read-only mode and the group does not exist, an error is raised.

        Args:
            handle (h5py.File): The HDF5 file handle to operate on.
            posix (str): The posix path of the group to retrieve or create in the HDF5 file.

        Returns:
            h5py.Group: The corresponding h5py group within the HDF5 file.

        Raises:
            ValueError: If the HDF5 file handle is invalid or no longer valid.
            ValueError: If the file handle mode cannot be determined.
            KeyError: If the specified group does not exist in read-only mode.
        """
        posix = str(posix)

        # Check if the handle is valid before accessing it
        try:
            # Test if handle is still valid by checking if it's open
            if not handle.id.valid:
                raise ValueError("HDF5 file handle is no longer valid (file may have been closed)")
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid HDF5 file handle: {e}")

        if posix in handle:
            return handle[posix]
        else:
            # Check if file is opened in read-only mode - with error handling
            try:
                file_mode = handle.mode
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Cannot determine file mode - HDF5 handle may be invalid: {e}")

            if file_mode == 'r':
                raise KeyError(f"Group '{posix}' not found in HDF5 file opened in read-only mode")
            if handle.swmr_mode is True:
                raise KeyError(f"Group '{posix}' not found in HDF5 file opened in SWMR mode")
            else:
                # File has write permissions, safe to create group
                return handle.create_group(posix)

    def get_home(self, handle):
        """
        Retrieves a specific group from an HDF file corresponding to single image data.

        This method is used to fetch a predefined group from an HDF container, where the group
        is identified by a constant key related to single image data. The function provides
        a static interface allowing invocation without requiring an instance of the class.

        Args:
            handle: The HDF file handle from which the group should be retrieved.

        Returns:
            The group corresponding to single image data, retrieved based on the defined
            SINGLE_IMAGE_ROOT_POSIX.

        Raises:
            Appropriate exceptions may be raised by the underlying HDF.get_group() method,
            based on the implementation and provided handle or key.
        """
        return self.get_group(handle=handle, posix=self.home_posix)

    def get_root_group(self, handle) -> h5py.Group:
        return self.get_group(handle=handle, posix=self.root_posix)

    def get_data_group(self, handle):
        if self.mode != 'set': raise AttributeError('This method is only available for image sets')
        return self.get_group(handle, self.set_data_posix)

    def get_image_group(self, handle, image_name):
        if self.mode == 'single':
            return self.get_home(handle)
        elif self.mode == 'set':
            return self.get_group(handle, posixpath.join(self.set_data_posix, image_name))
        else:
            raise ValueError(f"Invalid mode {self.mode}")

    def get_image_measurement_subgroup(self, handle, image_name):
        return self.get_group(handle, posixpath.join(self.set_data_posix, image_name, self.IMAGE_MEASUREMENT_SUBGROUP_KEY))

    def get_image_status_subgroup(self, handle, image_name):
        return self.get_group(handle, posixpath.join(self.set_data_posix, image_name, self.IMAGE_STATUS_SUBGROUP_KEY))

    @staticmethod
    def save_array2hdf5(group, array, name, **kwargs):
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
        if name in group:
            dset = group[name]

            if dset.shape == array.shape:
                dset[...] = array
            else:
                del group[name]
                group.create_dataset(name, data=array, **kwargs)
        else:
            group.create_dataset(name, data=array, **kwargs)

    # =================== PANDAS2HDF FUNCTIONALITY ===================

    @staticmethod
    def assert_swmr_on(g: h5py.Group) -> None:
        """Assert that SWMR mode is enabled on the group's file.

        Args:
            g: HDF5 group to check.

        Raises:
            RuntimeError: If SWMR mode is not enabled.
        """
        if not g.file.swmr_mode:
            raise RuntimeError(
                f"SWMR mode is required but not enabled on file {g.file.filename}"
            )

    @staticmethod
    def _get_string_dtype(length: int | None = None) -> h5py.special_dtype:
        """Get UTF-8 string dtype for HDF5.

        Args:
            length: If None, returns variable-length string dtype.
                    If an integer, returns fixed-length string dtype with that character length.

        Returns:
            HDF5 string dtype (variable-length or fixed-length).
        """
        if length is None:
            return h5py.string_dtype("utf-8")
        else:
            # Convert to native Python int to avoid h5py type issues
            return h5py.string_dtype("utf-8", int(length))

    @staticmethod
    def _pad_or_truncate_string(s: str, fixed_length: int) -> str:
        """Pad or truncate a string to a fixed character length.

        Args:
            s: Input string.
            fixed_length: Target character length.

        Returns:
            String padded with spaces or truncated to exactly fixed_length characters.
        """
        if len(s) < fixed_length:
            # Pad with spaces on the right
            return s + " " * (fixed_length - len(s))
        elif len(s) > fixed_length:
            # Truncate to exactly fixed_length characters
            return s[:fixed_length]
        else:
            # Already the right length
            return s

    @staticmethod
    def _apply_fixed_length_to_strings(
        str_array: np.ndarray[Any, np.dtype[Any]],
        mask: np.ndarray[Any, np.dtype[Any]],
        fixed_length: int,
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Apply fixed-length padding/truncation to string array.

        Args:
            str_array: Array of strings.
            mask: Mask array (1=valid, 0=missing).
            fixed_length: Target character length.

        Returns:
            Array with strings padded/truncated to fixed_length.
        """
        result = str_array.copy()
        valid_indices = mask == 1
        for i in np.where(valid_indices)[0]:
            # Ensure we're working with proper Unicode strings
            original_str = str(result[i])
            processed_str = HDF._pad_or_truncate_string(original_str, fixed_length)
            # Encode to UTF-8 bytes to ensure it fits in fixed-length storage
            result[i] = processed_str
        return result

    @staticmethod
    def _trim_trailing_whitespace(s: str) -> str:
        """Trim trailing whitespace from a string.

        Args:
            s: Input string.

        Returns:
            String with trailing whitespace removed.
        """
        return s.rstrip()

    @staticmethod
    def _decode_fixed_length_strings(
        str_array: np.ndarray[Any, np.dtype[Any]], mask: np.ndarray[Any, np.dtype[Any]]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Decode fixed-length strings and trim trailing whitespace from valid entries.

        Args:
            str_array: Array of fixed-length strings.
            mask: Mask array (1=valid, 0=missing).

        Returns:
            Object array with trimmed strings and None for missing values.
        """
        result = np.empty(len(str_array), dtype=object)
        valid_indices = mask == 1

        # Set valid entries with trimmed strings
        for i in np.where(valid_indices)[0]:
            raw_str = str_array[i]
            if isinstance(raw_str, bytes):
                raw_str = raw_str.decode("utf-8")
            result[i] = HDF._trim_trailing_whitespace(str(raw_str))

        # Set missing entries to None
        result[mask == 0] = None

        return result

    @staticmethod
    def _encode_values_for_hdf5(
        values: pd.Series,
        *,
        string_fixed_length: int | None = None,
    ) -> tuple[
        np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]] | None, str, str
    ]:
        """Encode pandas Series values for HDF5 storage.

        Args:
            values: pandas Series to encode.
            string_fixed_length: If provided, use fixed-length strings with this character length.

        Returns:
            Tuple of (encoded_values, mask, values_kind, orig_dtype):
            - encoded_values: numpy array ready for HDF5 storage
            - mask: optional mask array (uint8, 1=valid, 0=missing) for strings
            - values_kind: "numeric_float64", "string_utf8_fixed", or "string_utf8_vlen"
            - orig_dtype: string representation of original dtype
        """
        orig_dtype = str(values.dtype)

        if pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(
            values.dtype
        ):
            # Convert to float64, with NaN for missing values
            encoded = values.astype(np.float64).values
            mask = None
            values_kind = "numeric_float64"
        elif pd.api.types.is_object_dtype(values.dtype):
            # Check if object dtype contains boolean-like values
            non_null_values = values.dropna()
            if len(non_null_values) > 0 and all(
                isinstance(v, bool | np.bool_) or v in (True, False)
                for v in non_null_values
            ):
                # Treat as boolean/numeric data
                encoded = values.astype(np.float64).values
                mask = None
                values_kind = "numeric_float64"
            else:
                # Treat as string data
                str_values = values.astype(str)
                mask = (~values.isna()).astype(np.uint8)
                # Replace NaN string representations with empty strings
                str_array = str_values.values
                str_array[np.asarray(values.isna())] = ""

                if string_fixed_length is not None:
                    # Apply fixed-length padding/truncation
                    str_array = HDF._apply_fixed_length_to_strings(
                        np.asarray(str_array), np.asarray(mask), string_fixed_length
                    )
                    # Create array with explicit UTF-8 encoding for fixed-length
                    try:
                        encoded = str_array.astype(HDF._get_string_dtype(string_fixed_length))
                    except UnicodeEncodeError:
                        # Fallback: truncate to ASCII-safe characters if Unicode fails
                        ascii_safe_array = str_array.copy()
                        for i, s in enumerate(str_array):
                            if mask[i] == 1:
                                # Keep only ASCII characters for problematic Unicode
                                ascii_safe_array[i] = s.encode("ascii", "ignore").decode(
                                    "ascii"
                                )[:string_fixed_length]
                        encoded = ascii_safe_array.astype(
                            HDF._get_string_dtype(string_fixed_length)
                        )
                    values_kind = "string_utf8_fixed"
                else:
                    encoded = str_array.astype(HDF._get_string_dtype())
                    values_kind = "string_utf8_vlen"
        elif pd.api.types.is_string_dtype(values.dtype):
            # Convert to UTF-8 strings with mask for missing values
            str_values = values.astype(str)
            mask = (~values.isna()).astype(np.uint8)
            # Replace NaN string representations with empty strings
            str_array = str_values.values
            str_array[np.asarray(values.isna())] = ""

            if string_fixed_length is not None:
                # Apply fixed-length padding/truncation
                str_array = HDF._apply_fixed_length_to_strings(
                    np.asarray(str_array), np.asarray(mask), string_fixed_length
                )
                # Create array with explicit UTF-8 encoding for fixed-length
                try:
                    encoded = str_array.astype(HDF._get_string_dtype(string_fixed_length))
                except UnicodeEncodeError:
                    # Fallback: truncate to ASCII-safe characters if Unicode fails
                    ascii_safe_array = str_array.copy()
                    for i, s in enumerate(str_array):
                        if mask[i] == 1:
                            # Keep only ASCII characters for problematic Unicode
                            ascii_safe_array[i] = s.encode("ascii", "ignore").decode(
                                "ascii"
                            )[:string_fixed_length]
                    encoded = ascii_safe_array.astype(
                        HDF._get_string_dtype(string_fixed_length)
                    )
                values_kind = "string_utf8_fixed"
            else:
                encoded = str_array.astype(HDF._get_string_dtype())
                values_kind = "string_utf8_vlen"
        else:
            raise ValueError(f"Unsupported dtype: {values.dtype}")

        # Ensure we return numpy arrays, not ExtensionArrays
        encoded_array: np.ndarray[Any, np.dtype[Any]] = np.asarray(encoded)
        mask_array: np.ndarray[Any, np.dtype[Any]] | None = (
            np.asarray(mask) if mask is not None else None
        )
        return encoded_array, mask_array, values_kind, orig_dtype

    @staticmethod
    def _encode_index_for_hdf5(
        index: pd.Index,
        *,
        string_fixed_length: int | None = None,
    ) -> tuple[
        np.ndarray[Any, np.dtype[Any]] | list[np.ndarray[Any, np.dtype[Any]]],
        np.ndarray[Any, np.dtype[Any]] | list[np.ndarray[Any, np.dtype[Any]]],
        dict[str, Any],
        str,
    ]:
        """Encode pandas Index/MultiIndex for HDF5 storage.

        Args:
            index: pandas Index or MultiIndex to encode.
            string_fixed_length: If provided, use fixed-length strings with this character length.

        Returns:
            Tuple of (encoded_arrays, mask_arrays, metadata, orig_dtype):
            - encoded_arrays: array or list of arrays for MultiIndex levels
            - mask_arrays: mask array or list of mask arrays
            - metadata: dict with index metadata
            - orig_dtype: string representation of original dtype
        """
        orig_dtype = (
            str(index.dtype) if not isinstance(index, pd.MultiIndex) else "MultiIndex"
        )

        if isinstance(index, pd.MultiIndex):
            # Handle MultiIndex
            encoded_arrays = []
            mask_arrays = []

            for level_values in index.to_frame().values.T:
                level_series = pd.Series(level_values)
                str_values = level_series.astype(str)
                mask = (~level_series.isna()).astype(np.uint8)
                # Replace NaN representations
                str_array = str_values.values
                str_array[np.asarray(level_series.isna())] = ""

                if string_fixed_length is not None:
                    # Apply fixed-length padding/truncation
                    str_array = HDF._apply_fixed_length_to_strings(
                        np.asarray(str_array), np.asarray(mask), string_fixed_length
                    )
                    encoded_arrays.append(
                        str_array.astype(HDF._get_string_dtype(string_fixed_length))
                    )
                else:
                    encoded_arrays.append(str_array.astype(HDF._get_string_dtype()))
                mask_arrays.append(np.asarray(mask))

            metadata = {
                "index_is_multiindex": 1,
                "index_levels": index.nlevels,
                "index_names": json.dumps(
                    [str(name) if name is not None else None for name in index.names]
                ),
            }
            if string_fixed_length is not None:
                metadata["index_kind"] = "string_utf8_fixed"
            else:
                metadata["index_kind"] = "string_utf8_vlen"
        else:
            # Handle regular Index
            index_series = pd.Series(index)
            str_values = index_series.astype(str)
            mask = (~index_series.isna()).astype(np.uint8)
            # Replace NaN representations
            str_array = str_values.values
            str_array[np.asarray(index_series.isna())] = ""

            if string_fixed_length is not None:
                # Apply fixed-length padding/truncation
                str_array = HDF._apply_fixed_length_to_strings(
                    np.asarray(str_array), np.asarray(mask), string_fixed_length
                )
                encoded_arrays = str_array.astype(HDF._get_string_dtype(string_fixed_length))  # type: ignore[assignment]
            else:
                encoded_arrays = str_array.astype(HDF._get_string_dtype())  # type: ignore[assignment]
            mask_arrays = mask  # type: ignore[assignment]

            metadata = {
                "index_is_multiindex": 0,
                "index_levels": 1,
                "index_names": json.dumps(
                    [str(index.name) if index.name is not None else None]
                ),
            }
            if string_fixed_length is not None:
                metadata["index_kind"] = "string_utf8_fixed"
            else:
                metadata["index_kind"] = "string_utf8_vlen"

        return encoded_arrays, mask_arrays, metadata, orig_dtype

    @staticmethod
    def _decode_values_from_hdf5(
        group: h5py.Group,
        dataset_name: str = "values",
        length: int | None = None,
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], str]:
        """Decode values from HDF5 storage back to numpy array.

        Args:
            group: HDF5 group containing the datasets.
            dataset_name: Name of the values dataset.
            length: Logical length to read (respects preallocated space).

        Returns:
            Tuple of (decoded_values, values_kind).
        """
        values_kind = group.attrs["values_kind"]
        if isinstance(values_kind, bytes):
            values_kind = values_kind.decode("utf-8")
        logical_length = length if length is not None else group.attrs["len"]

        if values_kind == "numeric_float64":
            values = group[dataset_name][:logical_length]
        elif values_kind == "string_utf8_vlen":
            values = group[dataset_name][:logical_length]
            mask = group[f"{dataset_name}_mask"][:logical_length]
            # Convert back to object array with proper NaN handling
            result = np.empty(logical_length, dtype=object)
            result[mask == 1] = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in values[mask == 1]
            ]
            result[mask == 0] = None
            values = result
        elif values_kind == "string_utf8_fixed":
            values = group[dataset_name][:logical_length]
            mask = group[f"{dataset_name}_mask"][:logical_length]
            # Decode fixed-length strings and trim trailing whitespace
            values = HDF._decode_fixed_length_strings(values, mask)
        else:
            raise ValueError(f"Unknown values_kind: {values_kind}")

        return values, values_kind

    @staticmethod
    def _decode_index_from_hdf5(
        group: h5py.Group,
        index_dataset_name: str = "index",
        length: int | None = None,
    ) -> pd.Index:
        """Decode index from HDF5 storage back to pandas Index/MultiIndex.

        Args:
            group: HDF5 group containing the index datasets.
            index_dataset_name: Base name for index datasets.
            length: Logical length to read.

        Returns:
            Reconstructed pandas Index or MultiIndex.
        """
        logical_length = length if length is not None else group.attrs["len"]
        is_multiindex = bool(group.attrs["index_is_multiindex"])
        index_names_attr = group.attrs["index_names"]
        if isinstance(index_names_attr, bytes):
            index_names_attr = index_names_attr.decode("utf-8")
        index_names = json.loads(index_names_attr)

        # Check if index uses fixed-length strings
        index_kind = group.attrs.get("index_kind", "string_utf8_vlen")
        if isinstance(index_kind, bytes):
            index_kind = index_kind.decode("utf-8")

        if is_multiindex:
            levels = []
            for i in range(group.attrs["index_levels"]):
                level_data = group[f"{index_dataset_name}/levels/L{i}"][:logical_length]
                level_mask = group[f"{index_dataset_name}/levels/L{i}_mask"][
                    :logical_length
                ]

                if index_kind == "string_utf8_fixed":
                    # Decode fixed-length strings and trim trailing whitespace
                    level_values = HDF._decode_fixed_length_strings(level_data, level_mask)
                else:
                    # Original variable-length string handling
                    level_values = np.empty(logical_length, dtype=object)
                    level_values[level_mask == 1] = [
                        s.decode("utf-8") if isinstance(s, bytes) else str(s)
                        for s in level_data[level_mask == 1]
                    ]
                    level_values[level_mask == 0] = None
                levels.append(level_values)

            return pd.MultiIndex.from_arrays(levels, names=index_names)
        else:
            index_data = group[f"{index_dataset_name}/values"][:logical_length]
            index_mask = group[f"{index_dataset_name}/index_mask"][:logical_length]

            if index_kind == "string_utf8_fixed":
                # Decode fixed-length strings and trim trailing whitespace
                index_values = HDF._decode_fixed_length_strings(index_data, index_mask)
            else:
                # Original variable-length string handling
                index_values = np.empty(logical_length, dtype=object)
                index_values[index_mask == 1] = [
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in index_data[index_mask == 1]
                ]
                index_values[index_mask == 0] = None

            return pd.Index(index_values, name=index_names[0])

    @staticmethod
    def _create_resizable_dataset(
        group: h5py.Group,
        name: str,
        dtype: Any,
        shape: tuple[int, ...],
        maxshape: tuple[int | None, ...],
        chunks: tuple[int, ...],
        compression: str,
    ) -> h5py.Dataset:
        """Create a resizable, chunked, compressed dataset."""
        return group.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=chunks,
            compression=compression,
        )

    # =================== MAIN PANDAS2HDF FUNCTIONS ===================

    @staticmethod
    def preallocate_series_layout(
        group: h5py.Group,
        series: pd.Series,
        *,
        dataset: str = "values",
        index_dataset: str = "index",
        chunks: tuple[int, ...] = (25,),
        compression: str = "gzip",
        preallocate: int = 100,
        string_fixed_length: int = 100,
    ) -> None:
        """Preallocate HDF5 layout for a pandas Series without writing data.

        Creates resizable, chunked, compressed datasets with initial shape (preallocate,)
        and maxshape (None,). Initializes masks to zeros and sets len=0.

        Args:
            group: HDF5 group to write to.
            series: pandas Series to create layout for (used for schema).
            dataset: Name for the values dataset.
            index_dataset: Name for the index dataset.
            chunks: Chunk shape for datasets.
            compression: Compression algorithm.
            preallocate: Initial allocation size.
            string_fixed_length: Character length for fixed-length strings.

        Raises:
            RuntimeError: If require_swmr=True and SWMR mode not enabled.
                         If group.file.swmr_mode is True and datasets don't exist.
            ValueError: If series validation fails.
        """

        # Prevent object creation under SWMR (SWMR programming model compliance)
        if group.file.swmr_mode and dataset not in group:
            raise RuntimeError(
                "Cannot create new datasets while SWMR mode is enabled. "
                "Create all objects before starting SWMR mode."
            )

        # Encode series for schema information using fixed-length strings
        encoded_values, values_mask, values_kind, orig_values_dtype = (
            HDF._encode_values_for_hdf5(series, string_fixed_length=string_fixed_length)
        )
        encoded_index, index_masks, index_metadata, orig_index_dtype = (
            HDF._encode_index_for_hdf5(series.index, string_fixed_length=string_fixed_length)
        )

        # Create values dataset
        if values_kind == "numeric_float64":
            HDF._create_resizable_dataset(
                group, dataset, np.float64, (preallocate,), (None,), chunks, compression
            )
        else:  # string_utf8_fixed or string_utf8_vlen
            if values_kind == "string_utf8_fixed":
                dtype = HDF._get_string_dtype(string_fixed_length)
            else:
                dtype = HDF._get_string_dtype()

            HDF._create_resizable_dataset(
                group,
                dataset,
                dtype,
                (preallocate,),
                (None,),
                chunks,
                compression,
            )
            # Create values mask
            mask_dataset = HDF._create_resizable_dataset(
                group,
                f"{dataset}_mask",
                np.uint8,
                (preallocate,),
                (None,),
                chunks,
                compression,
            )
            mask_dataset[:] = 0  # Initialize to all missing

        # Create index datasets
        index_kind = index_metadata["index_kind"]
        if index_kind == "string_utf8_fixed":
            index_dtype = HDF._get_string_dtype(string_fixed_length)
        else:
            index_dtype = HDF._get_string_dtype()

        if index_metadata["index_is_multiindex"]:
            # Create index group and level datasets
            index_group = group.create_group(index_dataset)
            levels_group = index_group.create_group("levels")

            for i in range(index_metadata["index_levels"]):
                HDF._create_resizable_dataset(
                    levels_group,
                    f"L{i}",
                    index_dtype,
                    (preallocate,),
                    (None,),
                    chunks,
                    compression,
                )
                mask_dataset = HDF._create_resizable_dataset(
                    levels_group,
                    f"L{i}_mask",
                    np.uint8,
                    (preallocate,),
                    (None,),
                    chunks,
                    compression,
                )
                mask_dataset[:] = 0  # Initialize to all missing
        else:
            # Create index group
            index_group = group.create_group(index_dataset)
            HDF._create_resizable_dataset(
                index_group,
                "values",
                index_dtype,
                (preallocate,),
                (None,),
                chunks,
                compression,
            )
            mask_dataset = HDF._create_resizable_dataset(
                index_group,
                "index_mask",
                np.uint8,
                (preallocate,),
                (None,),
                chunks,
                compression,
            )
            mask_dataset[:] = 0  # Initialize to all missing

        # Set attributes
        group.attrs["series_name"] = str(series.name) if series.name is not None else ""
        group.attrs["len"] = 0  # Logical length is 0
        group.attrs["values_kind"] = values_kind
        group.attrs["orig_values_dtype"] = orig_values_dtype
        group.attrs["orig_index_dtype"] = orig_index_dtype
        group.attrs["created_at_iso"] = datetime.now().isoformat()
        group.attrs["version"] = "1.0"

        # Set string fixed length attributes when applicable
        if values_kind == "string_utf8_fixed":
            group.attrs["string_fixed_length"] = string_fixed_length
        if index_kind == "string_utf8_fixed":
            group.attrs["index_string_fixed_length"] = string_fixed_length

        # Add index metadata
        for key, value in index_metadata.items():
            if isinstance(value, str):
                group.attrs[key] = value
            else:
                group.attrs[key] = value


    @staticmethod
    def save_series_new(
        group: h5py.Group,
        series: pd.Series,
        *,
        dataset: str = "values",
        index_dataset: str = "index",
        chunks: tuple[int, ...] = (25,),
        compression: str = "gzip",
        preallocate: int = 100,
        string_fixed_length: int = 100,
        require_swmr: bool = False,
    ) -> None:
        """Create datasets and write a pandas Series to HDF5.

        Creates new datasets or reuses existing preallocated layout.
        Writes the first len(series) elements and sets logical length.

        Args:
            group: HDF5 group to write to.
            series: pandas Series to persist.
            dataset: Name for the values dataset.
            index_dataset: Name for the index dataset.
            chunks: Chunk shape for new datasets.
            compression: Compression algorithm for new datasets.
            preallocate: Initial allocation size for new datasets.
            string_fixed_length: Character length for fixed-length strings.
            require_swmr: If True, assert SWMR mode is enabled.

        Raises:
            RuntimeError: If require_swmr=True and SWMR mode not enabled.
            ValueError: If series validation fails.
        """
        if len(series) == 0:
            raise ValueError("Cannot save empty series")

        # Check if layout already exists (preallocated)
        if dataset in group and group.attrs.get("len", -1) == 0:
            # Use existing preallocated layout
            HDF.save_series_update(
                group,
                series,
                start=0,
                dataset=dataset,
                index_dataset=index_dataset,
                require_swmr=require_swmr,
            )
            return

        # Creation path: if require_swmr=True, only permit write (no object creation)
        if require_swmr:
            HDF.assert_swmr_on(group)
            # For SWMR writes, datasets must already exist
            if dataset not in group:
                raise RuntimeError(
                    "Datasets must be created before starting SWMR mode. "
                    "Use preallocate_series_layout() first, then start SWMR."
                )
            # Use update path
            HDF.save_series_update(
                group,
                series,
                start=0,
                dataset=dataset,
                index_dataset=index_dataset,
                require_swmr=require_swmr,
            )
            return

        # Create new layout (require_swmr=False for creation phase)
        HDF.preallocate_series_layout(
            group,
            series,
            dataset=dataset,
            index_dataset=index_dataset,
            chunks=chunks,
            compression=compression,
            preallocate=max(preallocate, len(series)),
            string_fixed_length=string_fixed_length,
            require_swmr=False,
        )

        # Write the data
        HDF.save_series_update(
            group,
            series,
            start=0,
            dataset=dataset,
            index_dataset=index_dataset,
            require_swmr=require_swmr,
        )

    @staticmethod
    def save_series_update(
        group: h5py.Group,
        series: pd.Series,
        *,
        start: int = 0,
        dataset: str = "values",
        index_dataset: str = "index",
        require_swmr: bool = True,
    ) -> None:
        """Update a pandas Series in HDF5 at specified position.

        Overwrites [start:start+len(series)] and updates logical length
        to the largest contiguous written extent.

        Args:
            group: HDF5 group containing existing datasets.
            series: pandas Series to write.
            start: Starting position for the update.
            dataset: Name of the values dataset.
            index_dataset: Name of the index dataset.
            require_swmr: If True, assert SWMR mode is enabled.

        Raises:
            RuntimeError: If require_swmr=True and SWMR mode not enabled.
            ValueError: If validation fails or schema mismatch.
        """
        if require_swmr:
            HDF.assert_swmr_on(group)

        if len(series) == 0:
            raise ValueError("Cannot update with empty series")

        current_len = group.attrs["len"]
        end_pos = start + len(series)

        # Validate that this is a contiguous update
        if start > current_len:
            raise ValueError(
                f"Non-contiguous update: start={start}, current_len={current_len}"
            )

        # Get stored schema to determine if fixed-length strings are used
        stored_values_kind = group.attrs["values_kind"]
        if isinstance(stored_values_kind, bytes):
            stored_values_kind = stored_values_kind.decode("utf-8")

        stored_index_kind = group.attrs.get("index_kind", "string_utf8_vlen")
        if isinstance(stored_index_kind, bytes):
            stored_index_kind = stored_index_kind.decode("utf-8")

        # Check for SWMR restrictions on variable-length strings
        if require_swmr and group.file.swmr_mode:
            if stored_values_kind == "string_utf8_vlen":
                raise RuntimeError(
                    "Cannot write to variable-length string datasets under SWMR mode. "
                    "Variable-length string writes are not allowed in SWMR mode."
                )
            if stored_index_kind == "string_utf8_vlen":
                raise RuntimeError(
                    "Cannot write to variable-length string index datasets under SWMR mode. "
                    "Variable-length string writes are not allowed in SWMR mode."
                )

        # Get fixed-length parameters from stored attributes
        string_fixed_length = None
        if stored_values_kind == "string_utf8_fixed":
            string_fixed_length = group.attrs["string_fixed_length"]

        index_string_fixed_length = None
        if stored_index_kind == "string_utf8_fixed":
            index_string_fixed_length = group.attrs["index_string_fixed_length"]

        # Encode data using stored schema
        encoded_values, values_mask, values_kind, _ = HDF._encode_values_for_hdf5(
            series, string_fixed_length=string_fixed_length
        )
        encoded_index, index_masks, index_metadata, _ = HDF._encode_index_for_hdf5(
            series.index, string_fixed_length=index_string_fixed_length
        )

        # Validate schema compatibility
        if stored_values_kind != values_kind:
            raise ValueError(
                f"Values kind mismatch: expected {stored_values_kind}, got {values_kind}"
            )

        expected_multiindex = bool(group.attrs["index_is_multiindex"])
        actual_multiindex = bool(index_metadata["index_is_multiindex"])
        if expected_multiindex != actual_multiindex:
            raise ValueError(
                f"Index type mismatch: expected multiindex={expected_multiindex}, got {actual_multiindex}"
            )

        # Resize datasets if needed
        values_dataset = group[dataset]
        if end_pos > values_dataset.shape[0]:
            values_dataset.resize((end_pos,))
            if values_mask is not None:
                group[f"{dataset}_mask"].resize((end_pos,))

        # Write values
        values_dataset[start:end_pos] = encoded_values
        if values_mask is not None:
            group[f"{dataset}_mask"][start:end_pos] = values_mask

        # Write index
        if expected_multiindex:
            levels_group = group[f"{index_dataset}/levels"]
            for i, (level_data, level_mask) in enumerate(
                zip(encoded_index, index_masks, strict=False)
            ):
                level_dataset = levels_group[f"L{i}"]
                if end_pos > level_dataset.shape[0]:
                    level_dataset.resize((end_pos,))
                    levels_group[f"L{i}_mask"].resize((end_pos,))
                level_dataset[start:end_pos] = level_data
                levels_group[f"L{i}_mask"][start:end_pos] = level_mask
        else:
            index_group = group[index_dataset]
            index_values_dataset = index_group["values"]
            if end_pos > index_values_dataset.shape[0]:
                index_values_dataset.resize((end_pos,))
                index_group["index_mask"].resize((end_pos,))
            index_values_dataset[start:end_pos] = encoded_index
            index_group["index_mask"][start:end_pos] = index_masks

        # Update logical length
        group.attrs["len"] = max(current_len, end_pos)

        if require_swmr:
            group.file.flush()

    @staticmethod
    def save_series_append(
        group: h5py.Group,
        series: pd.Series,
        *,
        dataset: str = "values",
        index_dataset: str = "index",
        require_swmr: bool = True,
    ) -> None:
        """Append a pandas Series to existing HDF5 datasets.

        Appends at the end using current logical length.
        Resizes datasets if needed and updates logical length.

        Args:
            group: HDF5 group containing existing datasets.
            series: pandas Series to append.
            dataset: Name of the values dataset.
            index_dataset: Name of the index dataset.
            require_swmr: If True, assert SWMR mode is enabled.

        Raises:
            RuntimeError: If require_swmr=True and SWMR mode not enabled.
            ValueError: If validation fails or schema mismatch.
        """
        if require_swmr:
            HDF.assert_swmr_on(group)

        current_len = group.attrs["len"]
        HDF.save_series_update(
            group,
            series,
            start=current_len,
            dataset=dataset,
            index_dataset=index_dataset,
            require_swmr=require_swmr,
        )

    @staticmethod
    def load_series(
        group: h5py.Group,
        *,
        dataset: str = "values",
        index_dataset: str = "index",
        require_swmr: bool = False,
    ) -> pd.Series:
        """Load a pandas Series from HDF5 storage.

        Reconstructs the Series with original name, index names, order,
        and missingness. Respects logical length from attributes.

        Args:
            group: HDF5 group containing the Series data.
            dataset: Name of the values dataset.
            index_dataset: Name of the index dataset.
            require_swmr: If True, assert SWMR mode is enabled.

        Returns:
            Reconstructed pandas Series.

        Raises:
            RuntimeError: If require_swmr=True and SWMR mode not enabled.
            ValueError: If data validation fails.
        """
        if require_swmr:
            HDF.assert_swmr_on(group)

        logical_length = group.attrs["len"]
        if logical_length == 0:
            # Return empty series with proper schema
            series_name = group.attrs["series_name"]
            if isinstance(series_name, bytes):
                series_name = series_name.decode("utf-8")
            series_name = series_name if series_name else None
            return pd.Series([], name=series_name, dtype=object)

        # Decode values and index
        values, _ = HDF._decode_values_from_hdf5(group, dataset, logical_length)
        index = HDF._decode_index_from_hdf5(group, index_dataset, logical_length)

        # Get series name
        series_name = group.attrs["series_name"]
        if isinstance(series_name, bytes):
            series_name = series_name.decode("utf-8")
        series_name = series_name if series_name else None

        return pd.Series(values, index=index, name=series_name)

    # =================== DATAFRAME FUNCTIONS ===================

    @staticmethod
    def preallocate_frame_layout(
        group: h5py.Group,
        dataframe: pd.DataFrame,
        *,
        chunks: tuple[int, ...] = (25,),
        compression: str = "gzip",
        preallocate: int = 100,
        string_fixed_length: int = 100,
        require_swmr: bool = False,
    ) -> None:
        """Preallocate HDF5 layout for a pandas DataFrame without writing data.

        Creates layout for shared index and column series using Series preallocation.

        Args:
            group: HDF5 group to write to.
            dataframe: pandas DataFrame to create layout for.
            chunks: Chunk shape for datasets.
            compression: Compression algorithm.
            preallocate: Initial allocation size.
            string_fixed_length: Character length for fixed-length strings.
            require_swmr: If True, assert SWMR mode is enabled.

        Raises:
            RuntimeError: If require_swmr=True and SWMR mode not enabled.
                         If group.file.swmr_mode is True and datasets don't exist.
            ValueError: If DataFrame validation fails.
        """
        if require_swmr:
            HDF.assert_swmr_on(group)

        # Prevent object creation under SWMR (SWMR programming model compliance)
        if group.file.swmr_mode and "index" not in group:
            raise RuntimeError(
                "Cannot create new groups/datasets while SWMR mode is enabled. "
                "Create all objects before starting SWMR mode."
            )

        if len(dataframe.columns) == 0:
            raise ValueError("Cannot preallocate layout for DataFrame with no columns")

        # Set frame attributes
        group.attrs["column_order"] = json.dumps(list(dataframe.columns))
        group.attrs["len"] = 0

        # Preallocate index layout
        index_group = group.create_group("index")
        # Create a dummy series with string values to match the schema
        dummy_series = pd.Series([], dtype=str, index=dataframe.index[:0], name="__index__")
        HDF.preallocate_series_layout(
            index_group,
            dummy_series,
            dataset="values",
            index_dataset="index",
            chunks=chunks,
            compression=compression,
            preallocate=preallocate,
            string_fixed_length=string_fixed_length,
        )

        # Preallocate column layouts
        columns_group = group.create_group("columns")
        for col_name in dataframe.columns:
            col_group = columns_group.create_group(str(col_name))
            # Create dummy series - use actual data to determine schema if available
            if len(dataframe) > 0:
                # Use first few values to determine the proper schema
                col_data = dataframe[col_name]
                dummy_col_series = pd.Series(
                    [col_data.iloc[0]] if not col_data.isna().iloc[0] else [None],
                    dtype=col_data.dtype,
                    index=dataframe.index[:1],
                    name=col_name,
                )
            else:
                # Fallback to dtype for empty dataframe
                col_dtype = dataframe[col_name].dtype
                dummy_col_series = pd.Series(
                    [], dtype=col_dtype, index=dataframe.index[:0], name=col_name
                )

            HDF.preallocate_series_layout(
                col_group,
                dummy_col_series,
                dataset="values",
                index_dataset="index",
                chunks=chunks,
                compression=compression,
                preallocate=preallocate,
                string_fixed_length=string_fixed_length,
            )

    @staticmethod
    def save_frame_new(
        group: h5py.Group,
        dataframe: pd.DataFrame,
        *,
        chunks: tuple[int, ...] = (25,),
        compression: str = "gzip",
        preallocate: int = 100,
        string_fixed_length: int = 100,
        require_swmr: bool = False,
    ) -> None:
        """Create datasets and write a pandas DataFrame to HDF5.

        Args:
            group: HDF5 group to write to.
            dataframe: pandas DataFrame to persist.
            chunks: Chunk shape for new datasets.
            compression: Compression algorithm for new datasets.
            preallocate: Initial allocation size for new datasets.
            string_fixed_length: Character length for fixed-length strings.
            require_swmr: If True, assert SWMR mode is enabled.

        Raises:
            RuntimeError: If require_swmr=True and SWMR mode not enabled.
            ValueError: If DataFrame validation fails.
        """
        if len(dataframe) == 0:
            raise ValueError("Cannot save empty DataFrame")

        # Check if layout already exists (preallocated)
        if "columns" in group and group.attrs.get("len", -1) == 0:
            HDF.save_frame_update(group, dataframe, start=0, require_swmr=require_swmr)
            return

        # Creation path: if require_swmr=True, only permit write (no object creation)
        if require_swmr:
            HDF.assert_swmr_on(group)
            # For SWMR writes, groups/datasets must already exist
            if "columns" not in group:
                raise RuntimeError(
                    "Groups/datasets must be created before starting SWMR mode. "
                    "Use preallocate_frame_layout() first, then start SWMR."
                )
            # Use update path
            HDF.save_frame_update(group, dataframe, start=0, require_swmr=require_swmr)
            return

        # Create new layout (require_swmr=False for creation phase)
        HDF.preallocate_frame_layout(
            group,
            dataframe,
            chunks=chunks,
            compression=compression,
            preallocate=max(preallocate, len(dataframe)),
            string_fixed_length=string_fixed_length,
            require_swmr=False,
        )

        # Write the data
        HDF.save_frame_update(group, dataframe, start=0, require_swmr=require_swmr)

    @staticmethod
    def save_frame_update(
        group: h5py.Group,
        dataframe: pd.DataFrame,
        *,
        start: int = 0,
        require_swmr: bool = True,
    ) -> None:
        """Update a pandas DataFrame in HDF5 at specified position.

        Args:
            group: HDF5 group containing existing datasets.
            dataframe: pandas DataFrame to write.
            start: Starting position for the update.
            require_swmr: If True, assert SWMR mode is enabled.

        Raises:
            RuntimeError: If require_swmr=True and SWMR mode not enabled.
            ValueError: If validation fails or schema mismatch.
        """
        if require_swmr:
            HDF.assert_swmr_on(group)

        if len(dataframe) == 0:
            raise ValueError("Cannot update with empty DataFrame")

        current_len = group.attrs["len"]
        end_pos = start + len(dataframe)

        # Validate contiguous update
        if start > current_len:
            raise ValueError(
                f"Non-contiguous update: start={start}, current_len={current_len}"
            )

        # Validate column order matches
        column_order_attr = group.attrs["column_order"]
        if isinstance(column_order_attr, bytes):
            column_order_attr = column_order_attr.decode("utf-8")
        stored_columns = json.loads(column_order_attr)
        if list(dataframe.columns) != stored_columns:
            raise ValueError(
                f"Column order mismatch: expected {stored_columns}, got {list(dataframe.columns)}"
            )

        # Update index - create a dummy series to represent the actual index
        # We need to store the index structure, so we create a dummy series where the
        # index is the actual DataFrame index and values are just placeholders
        index_series = pd.Series(
            ["dummy"] * len(dataframe), index=dataframe.index, name="__index__"
        )
        HDF.save_series_update(
            group["index"],
            index_series,
            start=start,
            dataset="values",
            index_dataset="index",
            require_swmr=require_swmr,
        )

        # Update each column
        columns_group = group["columns"]
        for col_name in dataframe.columns:
            col_series = dataframe[col_name]
            col_series.name = col_name
            HDF.save_series_update(
                columns_group[str(col_name)],
                col_series,
                start=start,
                dataset="values",
                index_dataset="index",
                require_swmr=require_swmr,
            )

        # Update frame length
        group.attrs["len"] = max(current_len, end_pos)

        if require_swmr:
            group.file.flush()

    @staticmethod
    def save_frame_append(
        group: h5py.Group,
        dataframe: pd.DataFrame,
        *,
        require_swmr: bool = True,
    ) -> None:
        """Append a pandas DataFrame to existing HDF5 datasets.

        Args:
            group: HDF5 group containing existing datasets.
            dataframe: pandas DataFrame to append.
            require_swmr: If True, assert SWMR mode is enabled.

        Raises:
            RuntimeError: If require_swmr=True and SWMR mode not enabled.
            ValueError: If validation fails or schema mismatch.
        """
        if require_swmr:
            HDF.assert_swmr_on(group)

        current_len = group.attrs["len"]
        HDF.save_frame_update(group, dataframe, start=current_len, require_swmr=require_swmr)

    @staticmethod
    def load_frame(
        group: h5py.Group,
        *,
        require_swmr: bool = False,
    ) -> pd.DataFrame:
        """Load a pandas DataFrame from HDF5 storage.

        Args:
            group: HDF5 group containing the DataFrame data.
            require_swmr: If True, assert SWMR mode is enabled.

        Returns:
            Reconstructed pandas DataFrame.

        Raises:
            RuntimeError: If require_swmr=True and SWMR mode not enabled.
            ValueError: If data validation fails.
        """
        if require_swmr:
            HDF.assert_swmr_on(group)

        logical_length = group.attrs["len"]
        if logical_length == 0:
            # Return empty DataFrame with proper schema
            column_order_attr = group.attrs["column_order"]
            if isinstance(column_order_attr, bytes):
                column_order_attr = column_order_attr.decode("utf-8")
            column_order = json.loads(column_order_attr)
            return pd.DataFrame(columns=column_order)

        # Load index
        index = HDF._decode_index_from_hdf5(group["index"], "index", logical_length)

        # Load columns in order
        column_order_attr = group.attrs["column_order"]
        if isinstance(column_order_attr, bytes):
            column_order_attr = column_order_attr.decode("utf-8")
        column_order = json.loads(column_order_attr)
        columns_group = group["columns"]

        columns_data = {}
        for col_name in column_order:
            col_series = HDF.load_series(
                columns_group[str(col_name)],
                dataset="values",
                index_dataset="index",
                require_swmr=require_swmr,
            )
            columns_data[col_name] = col_series.values

        # Reconstruct DataFrame
        dataframe = pd.DataFrame(columns_data, index=index)
        result: pd.DataFrame = dataframe[column_order]  # Ensure column order
        return result
