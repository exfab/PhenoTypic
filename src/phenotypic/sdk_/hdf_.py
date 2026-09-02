import logging
import posixpath
import subprocess
import time
from pathlib import Path
from typing import Callable, Literal

import h5py
from packaging.version import Version

import phenotypic

logger = logging.getLogger(__name__)

_SAFE_WRITER_LOCK_MARKERS = (
    "file is already open for write/swmr write",
    "file is already open",
    "unable to lock file",
    "resource temporarily unavailable",
    "file locking disabled",
)

_SWMR_WRITER_LOCK_MARKERS = (
    *_SAFE_WRITER_LOCK_MARKERS,
    "ring type mismatch",
    "pinned entry count",
)


def _open_hdf_with_recovery(
    filepath: Path,
    opener: Callable[[], h5py.File],
    *,
    context: str,
    lock_markers: tuple[str, ...],
    clear_status: bool,
    clear_force: bool,
    max_retries: int = 3,
    retry_delay: float = 0.5,
) -> h5py.File:
    """Open an HDF5 file with lock-conflict recovery.

    Args:
        filepath: Path to the HDF5 file being opened.
        opener: Callable that opens and returns the file handle.
        context: Human-readable context used in log/error messages.
        lock_markers: Lowercase substrings that identify recoverable lock or
            cache-conflict errors.
        clear_status: Whether to run ``h5clear -s`` between attempts.
        clear_force: Whether to run ``h5clear -f`` between attempts.
        max_retries: Maximum open attempts.
        retry_delay: Initial retry delay in seconds. Doubles after each retry.

    Returns:
        The opened HDF5 file handle.

    Raises:
        RuntimeError: If a recoverable lock conflict persists through the final
            attempt.
        OSError: Non-lock open errors are re-raised immediately.
    """
    filepath = Path(filepath)
    delay = retry_delay
    for attempt in range(max_retries):
        try:
            return opener()
        except OSError as exc:
            error_msg = str(exc).lower()
            is_lock_error = any(marker in error_msg for marker in lock_markers)
            if not is_lock_error:
                raise

            logger.warning(
                "%s access conflict (attempt %d/%d): %s",
                context,
                attempt + 1,
                max_retries,
                exc,
            )

            if attempt >= max_retries - 1:
                logger.error(
                    "Failed to open %s after %d attempts",
                    context,
                    max_retries,
                )
                failure_reason = (
                    "may have cache conflicts"
                    if "SWMR" in context
                    else "may be locked by another process"
                )
                raise RuntimeError(
                    f"Failed to open {context} after {max_retries} attempts. "
                    f"The file {filepath} {failure_reason}. "
                    "Try manually running: "
                    f"h5clear -s {filepath} && h5clear -f {filepath}"
                ) from exc

            _clear_hdf_consistency_flags(
                filepath, clear_status=clear_status, clear_force=clear_force
            )
            logger.info("Waiting %s seconds before retry...", delay)
            time.sleep(delay)
            delay *= 2

    raise OSError(f"Unexpected error opening HDF5 file {filepath}")


def _clear_hdf_consistency_flags(
    filepath: Path, *, clear_status: bool, clear_force: bool
) -> None:
    """Best-effort ``h5clear`` recovery for an existing HDF5 file."""
    if not filepath.exists():
        return
    flags = []
    if clear_status:
        flags.append("-s")
    if clear_force:
        flags.append("-f")

    for flag in flags:
        try:
            logger.info(
                "Attempting to clear HDF5 consistency flag %s for %s",
                flag,
                filepath,
            )
            result = subprocess.run(
                ["h5clear", flag, str(filepath)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info(
                    "Successfully cleared HDF5 consistency flag %s", flag
                )
            else:
                logger.warning("h5clear %s failed: %s", flag, result.stderr)
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ) as clear_error:
            logger.warning("Could not run h5clear %s: %s", flag, clear_error)


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
        SINGLE_IMAGE_ROOT_POSIX = "/phenotypic/"
    else:
        SINGLE_IMAGE_ROOT_POSIX = "/phenotypic/images/"

    IMAGE_SET_ROOT_POSIX = "/phenotypic/image_sets/"
    IMAGE_SET_DATA_POSIX = "data"  # The image and individual measurement group

    # measurements and status are stored within in each image's group
    IMAGE_MEASUREMENT_SUBGROUP_KEY = "measurements"
    IMAGE_STATUS_SUBGROUP_KEY = "status"

    PROTECTED_METADATA_SUBGROUP_KEY = "protected_metadata"
    PUBLIC_METADATA_SUBGROUP_KEY = "public_metadata"

    EXT = {".h5", ".hdf5", ".hdf", ".he5"}

    def __init__(self, filepath, name: str, mode: Literal["single", "set"]):
        """
        Initializes a class instance to manage HDF5 file structures for single or set image
        data based on the given filepath, name of the resource, and operational mode.

        Attributes:
            filepath (Path): Path to the HDF5 file.
            name (str): Identifier for the resource within the HDF5 file.
            mode (Literal['single', 'set']): Operational mode determining the structure
                and organization within the HDF5 file. Must be either 'single' or 'set'.
            root_posix (str): Posix path representing the root directory within the
                HDF5 file based on the mode.
            home_posix (str): Posix path representing the home directory for the resource
                within the HDF5 file based on the mode.
            set_data_posix (Optional[str]): Posix path for the data subdirectory within
                the resource home directory. Only initialized in 'set' mode.

        Args:
            filepath: Path to the target HDF5 file. Must have an HDF5-compatible extension,
                or a ValueError is raised.
            name: Name of the resource to be managed in the file. Used to construct the
                home directory for the resource within the HDF5 file.
            mode: Operational mode. Specifies whether the resource represents a 'single'
                or 'set' image data. If the mode is invalid, a ValueError is raised.

        Raises:
            ValueError: If the filepath does not have an HDF5-compatible extension.
            ValueError: If the mode is neither 'single' nor 'set'.
        """
        self.filepath = Path(filepath)
        if self.filepath.suffix not in self.EXT:
            raise ValueError("filepath is not an hdf5 file")
        if not self.filepath.exists():
            with h5py.File(name=self.filepath, mode="a", libver="latest"):
                pass

        self.name = name
        self.mode = mode
        if mode == "single":
            self.root_posix = self.SINGLE_IMAGE_ROOT_POSIX
            self.home_posix = posixpath.join(
                self.SINGLE_IMAGE_ROOT_POSIX, self.name
            )
        elif mode == "set":
            self.root_posix = self.IMAGE_SET_ROOT_POSIX
            self.home_posix = posixpath.join(
                self.IMAGE_SET_ROOT_POSIX, self.name
            )
            self.set_data_posix = posixpath.join(
                self.home_posix, self.IMAGE_SET_DATA_POSIX
            )
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
        return _open_hdf_with_recovery(
            self.filepath,
            lambda: h5py.File(self.filepath, "a", libver="latest"),
            context="HDF5 file",
            lock_markers=_SAFE_WRITER_LOCK_MARKERS,
            clear_status=True,
            clear_force=False,
        )

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

        def _open_swmr_writer() -> h5py.File:
            file_handle = h5py.File(self.filepath, "a", libver="latest")
            try:
                file_handle.swmr_mode = True
                logger.debug(
                    "SWMR mode enabled successfully for %s", self.filepath
                )
                return file_handle
            except Exception as swmr_error:
                logger.warning("Could not enable SWMR mode: %s", swmr_error)
                return file_handle

        return _open_hdf_with_recovery(
            self.filepath,
            _open_swmr_writer,
            context="HDF5 file in SWMR mode",
            lock_markers=_SWMR_WRITER_LOCK_MARKERS,
            clear_status=True,
            clear_force=True,
        )

    def strict_writer(self) -> h5py.File:
        """
        Provides access to an HDF5 file in read/write mode using the `h5py` library. This
        property is used to obtain an `h5py.File` object configured with the latest library version.

        Note:
            If using SWMR mode, don't forget to enable SWMR mode:

            >>> hdf = HDF(filepath)  # doctest: +SKIP
            >>> with hdf.writer as writer:  # doctest: +SKIP
            ...     writer.swmr_mode = True
            ...     # rest of your code

        Returns:
            h5py.File: An HDF5 file object opened in 'r+' mode, enabling reading and writing.

        Raises:
            OSError: If the file cannot be opened or accessed.
        """
        return h5py.File(self.filepath, "r+", libver="latest")

    def swmr_reader(self) -> h5py.File:
        return h5py.File(self.filepath, "r", libver="latest", swmr=True)

    def reader(self) -> h5py.File:
        return h5py.File(self.filepath, "r", libver="latest", swmr=False)

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
                raise ValueError(
                    "HDF5 file handle is no longer valid (file may have been closed)"
                )
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid HDF5 file handle: {e}")

        if posix in handle:
            return handle[posix]
        else:
            # Check if file is opened in read-only mode - with error handling
            # Handle both File and Group objects (Groups have a .file attribute)
            try:
                if isinstance(handle, h5py.Group) and not isinstance(
                    handle, h5py.File
                ):
                    # For Group objects, access the parent file
                    file_obj = handle.file
                    file_mode = file_obj.mode
                    swmr_mode = file_obj.swmr_mode
                else:
                    # For File objects, access directly
                    file_mode = handle.mode
                    swmr_mode = handle.swmr_mode
            except (ValueError, AttributeError) as e:
                raise ValueError(
                    f"Cannot determine file mode - HDF5 handle may be invalid: {e}"
                )

            if file_mode == "r":
                raise KeyError(
                    f"Group '{posix}' not found in HDF5 file opened in read-only mode"
                )
            if swmr_mode is True:
                raise KeyError(
                    f"Group '{posix}' not found in HDF5 file opened in SWMR mode"
                )
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
        if self.mode != "set":
            raise AttributeError(
                "This method is only available for image sets"
            )
        return self.get_group(handle, self.set_data_posix)

    def get_image_group(self, handle, image_name):
        if self.mode == "single":
            return self.get_home(handle)
        elif self.mode == "set":
            return self.get_group(
                handle, posixpath.join(self.set_data_posix, image_name)
            )
        else:
            raise ValueError(f"Invalid mode {self.mode}")

    def get_image_measurement_subgroup(self, handle, image_name):
        return self.get_group(
            handle,
            posixpath.join(
                self.set_data_posix,
                image_name,
                self.IMAGE_MEASUREMENT_SUBGROUP_KEY,
            ),
        )

    def get_status_subgroup(self, handle, image_name):
        return self.get_group(
            handle,
            posixpath.join(
                self.set_data_posix, image_name, self.IMAGE_STATUS_SUBGROUP_KEY
            ),
        )

    def get_protected_metadata_subgroup(
        self, handle: h5py.File, image_name: str
    ) -> h5py.Group:
        return self.get_group(
            handle=handle,
            posix=posixpath.join(
                self.set_data_posix,
                image_name,
                self.PROTECTED_METADATA_SUBGROUP_KEY,
            ),
        )

    def get_public_metadata_subgroup(
        self, handle: h5py.File, image_name: str
    ) -> h5py.Group:
        return self.get_group(
            handle=handle,
            posix=posixpath.join(
                self.set_data_posix,
                image_name,
                self.PUBLIC_METADATA_SUBGROUP_KEY,
            ),
        )

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
