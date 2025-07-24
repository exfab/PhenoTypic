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
        SINGLE_IMAGE_GROUP_POSIX = f'/phenotypic/'
    else:
        SINGLE_IMAGE_GROUP_POSIX = f'/phenotypic/images/'

    IMAGE_SET_HDF5_PARENT_GROUP = f'/phenotypic/image_sets/'
    IMAGE_SET_IMAGES_SUBGROUP_KEY = 'images'

    # measurements and status are stored within in each image's group
    IMAGE_MEASUREMENT_SUBGROUP_KEY = 'measurements'
    IMAGE_STATUS_SUBGROUP_KEY = "status"

    def __init__(self, filepath, name: str, mode: Literal['single', 'set']):
        self.filepath = filepath
        self.name = name
        self.mode = mode
        if mode == 'single':
            self.root_posix = posixpath.join(self.SINGLE_IMAGE_GROUP_POSIX, self.name)
        elif mode == 'set':
            self.root_posix = posixpath.join(self.IMAGE_SET_HDF5_PARENT_GROUP, self.name)
            self.set_images_posix = posixpath.join(self.root_posix, self.IMAGE_SET_IMAGES_SUBGROUP_KEY)
        else:
            raise ValueError(f"Invalid mode {mode}")


    @property
    def safe_writer(self) -> h5py.File:
        """
        Returns a writer object that provides safe and controlled write access to an
        HDF5 file at the specified filepath or creates it if it doesn't exist. Ensures that the file uses the 'latest'
        version of the HDF5 library for compatibility and performance.

        Returns:
            h5py.File: A file writer object with append mode and 'latest' library
            version enabled.
        """
        return h5py.File(self.filepath, 'a', libver='latest')

    @property
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

    @property
    def reader(self) -> h5py.File:
        try:
            return h5py.File(self.filepath, 'r', libver='latest', swmr=True)
        except (RuntimeError, ValueError):
            return h5py.File(self.filepath, 'r', libver='latest')

    @staticmethod
    def get_group(handle: h5py.File, group_name) -> h5py.Group:
        group_name = str(group_name)
        if group_name in handle:
            return handle[group_name]
        else:
            return handle.create_group(group_name)

    def get_root(self, handle):
        """
        Retrieves a specific group from an HDF file corresponding to single image data.

        This method is used to fetch a predefined group from an HDF container, where the group
        is identified by a constant key related to single image data. The function provides
        a static interface allowing invocation without requiring an instance of the class.

        Args:
            handle: The HDF file handle from which the group should be retrieved.

        Returns:
            The group corresponding to single image data, retrieved based on the defined
            SINGLE_IMAGE_GROUP_POSIX.

        Raises:
            Appropriate exceptions may be raised by the underlying HDF.get_group() method,
            based on the implementation and provided handle or key.
        """
        return self.get_group(handle=handle, group_name=self.root_posix)


    def get_parent_group(self, handle)->h5py.Group:
        return self.get_root(handle).parent

    def get_images_subgroup(self, handle):
        if self.mode != 'set': raise AttributeError('This method is only available for image sets')
        return self.get_group(handle, self.set_images_posix)

    def get_image_group(self, handle, image_name):
        if self.mode == 'single':
            return self.get_root(handle)
        elif self.mode == 'set':
            return self.get_group(handle, posixpath.join(self.set_images_posix, image_name))
        else:
            raise ValueError(f"Invalid mode {self.mode}")

    def get_image_measurement_subgroup(self, handle, image_name):
        return self.get_group(handle, posixpath.join(self.set_images_posix, image_name, self.IMAGE_MEASUREMENT_SUBGROUP_KEY))

    def get_image_status_subgroup(self, handle, image_name):
        return self.get_group(handle, posixpath.join(self.set_images_posix, image_name, self.IMAGE_STATUS_SUBGROUP_KEY))

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
