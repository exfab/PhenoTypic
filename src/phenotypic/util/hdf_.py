import posixpath
from typing import Literal

from packaging.version import Version

import h5py
import phenotypic


class HDF:
    """
    A utility class to help with managing hdf5 files
    """

    if Version(phenotypic.__version__) < Version("0.7.1"):
        SINGLE_IMAGE_ROOT_POSIX = f'/phenotypic/'
    else:
        SINGLE_IMAGE_ROOT_POSIX = f'/phenotypic/images/'

    IMAGE_SET_ROOT_POSIX = f'/phenotypic/image_sets/'
    IMAGE_SET_DATA_POSIX = 'data'   # The image and individual measurement group

    # measurements and status are stored within in each image's group
    IMAGE_MEASUREMENT_SUBGROUP_KEY = 'measurements'
    IMAGE_STATUS_SUBGROUP_KEY = "status"

    def __init__(self, filepath, name: str, mode: Literal['single', 'set']):
        self.filepath = filepath
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

    def reader(self) -> h5py.File:
        try:
            return h5py.File(self.filepath, 'r', libver='latest', swmr=True)
        except (RuntimeError, ValueError):
            return h5py.File(self.filepath, 'r', libver='latest')

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
