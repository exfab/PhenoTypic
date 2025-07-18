import os
from pathlib import Path

from typing import List
import h5py
from os import PathLike

from phenotypic import Image, GridImage
from phenotypic.util.constants_ import IO


class ImageSetCore:
    """
    Handles the management and bulk processing of an image set, including importing from
    various sources, storing into an HDF5 file, and managing images efficiently.

    The `ImageSetCore` class facilitates large-scale image operations by importing images
    from either an in-memory list or a specified source directory/HDF5 file, storing the
    images into an output HDF5 file, and providing methods to manage and query the image set.
    It supports overwriting of existing datasets and ensures proper handling of HDF5 file
    groups for structured storage.

    Notes:
        - for developers: open a new writer in each function in order to prevent and data corruption with the hdf5 file

    Attributes:
        name (str): Name of the image set used for identification and structured storage.
        _src_path (Path | None): Path to the source directory or HDF5 file containing images.
            Initialized as a `Path` object or None if `image_list` is used.
        _out_path (Path): Path to the output HDF5 file storing the image set. Initialized
            as a `Path` object and defaults to the current working directory if not specified.
        _overwrite (bool): Indicates whether to overwrite existing data in the output HDF5 file.
        _hdf5_set_group_key (str): The group path in the HDF5 file where the image set is stored.
    """

    def __init__(self, name: str, image_template: Image | None = None, image_list: List[Image] | None = None,
                 src_path: PathLike | None = None,
                 out_path: PathLike | None = None,
                 overwrite: bool = False, ):
        """
        Initializes an image set for bulk processing of images.

        This constructor is responsible for setting up an image set by either importing
        images from a provided list, an HDF5 file, or a directory. It also handles the
        storing of the images into an output HDF5 file, with options for overwriting
        existing data. Only one of the following should be specified: `image_list`
        or `src_path`.

        Args:
            name (str): The name of the image set to initialize.
            image_template: (Image | None): The Image object with settings to be used when constructing the Image.
                Can be a GridImage with ncols and nrows specified. Default is a 96-Plate GridImage
            image_list (List[Image] | None, optional): A list of Image objects if importing
                from in-memory images.
            src_path (PathLike | None, optional): The source directory or HDF5 file
                containing images.
            out_path (PathLike | None, optional): The output HDF5 file where
                the image set will be stored. Defaults to the current working directory.
            overwrite (bool): Determines whether to overwrite existing data in the output
                HDF5 file. Defaults to False.

        Raises:
            AssertionError: If both `image_list` and `src_path` are provided or none
                is provided.
            ValueError: If no images or image sections are found in the provided `src_path`.
            ValueError: If `image_list` is not a list of `Image` objects or `src_path`
                is not a valid HDF5 file.
        """
        self.name = name

        assert isinstance(image_template, (Image, type(None))), "image_template must be an Image object or None."
        self.image_template = image_template if image_template else GridImage(ncols=12, nrows=8)

        # Only an image_list xor src_path should be given
        assert (image_list and src_path is None) or (image_list is None and src_path), 'Only one of image_list or src_path can be provided.'

        src_path, out_path = Path(src_path) if src_path else None, Path(out_path) if out_path else Path.cwd() / f'{self.name}.hdf5'
        self.name, self._src_path, self._out_path = str(name), src_path, out_path
        self._overwrite = overwrite

        # Define hdf5 group paths
        self._hdf5_parent_group_key = IO.IMAGE_SET_HDF5_PARENT_GROUP
        self._hdf5_set_group_key = self._hdf5_parent_group_key / self.name
        self._hdf5_image_group_key = self._hdf5_set_group_key / 'images'
        self._hdf5_measurement_group_key = self._hdf5_set_group_key / 'measurements'

        # If input source path handling
        if src_path:
            # If input path is an hdf5 file

            # If src and out are the same
            if (src_path.is_file()) and (src_path == out_path):
                pass

            # If src and out are different
            elif src_path.is_file() and src_path.suffix == '.h5':
                with h5py.File(src_path, mode='a') as src_filehandler, h5py.File(out_path, mode='a') as out_filehandler:
                    src_parent_group = self._get_hdf5_group(src_filehandler, self)
                    out_parent_group = self._get_hdf5_group(out_filehandler, IO.IMAGE_SET_HDF5_PARENT_GROUP)

                    #   if the image set name is in the ImageSet group, copy the images over
                    if self.name in src_parent_group:
                        src_group = self._get_hdf5_group(src_filehandler, self._hdf5_set_group_key)

                        # overwrite if overwrite is true
                        if self.name in out_parent_group and overwrite is True: del out_parent_group[self.name]

                        # Should place a copy of the src_group into the parent group
                        src_filehandler.copy(src_group, out_parent_group, name=self.name, shallow=False)

                    #   else import all the images from Image section
                    elif IO.SINGLE_IMAGE_HDF5_PARENT_GROUP in src_filehandler:
                        src_image_group = self._get_hdf5_group(src_filehandler, IO.SINGLE_IMAGE_HDF5_PARENT_GROUP)

                        # overwrite if overwrite is true
                        if self.name in out_parent_group and overwrite is True: del out_parent_group[self.name]
                        src_filehandler.copy(src_image_group, out_parent_group, name=self.name, shallow=False)

                    else:
                        raise ValueError(f'No ImageSet named {self.name} or Image section found in {src_path}')

            # src_path image is a directory handling
            # only need out handler
            elif src_path.is_dir():
                image_filenames = [x for x in os.listdir(src_path) if x.endswith(IO.ACCEPTED_FILE_EXTENSIONS)]
                with h5py.File(out_path, mode='a') as out_handler:
                    out_group = self._get_hdf5_group(out_handler, self._hdf5_set_group_key)

                    # Overwrite handling
                    if self.name in out_group and overwrite is True: del out_group[self.name]
                    out_set_group = self._get_hdf5_group(out_handler, self._hdf5_set_group_key)

                    for fname in image_filenames:
                        image = Image.imread(src_path / fname)
                        image._save_image2hdf5(grp=out_set_group, compression="gzip", compression_opts=4)

        # Image list handling
        # Only need out handler for this
        elif isinstance(image_list, list):
            assert all(isinstance(x, Image) for x in image_list), 'image_list must be a list of Image objects.'
            with h5py.File(out_path, mode='a') as out_handler:
                out_group = self._get_hdf5_group(out_handler, IO.IMAGE_SET_HDF5_PARENT_GROUP)

                # Overwrite the data in the output folder
                if self.name in out_group and overwrite is True: del out_group[self.name]
                out_set_group = self._get_hdf5_group(out_handler, self._hdf5_set_group_key)

                for image in image_list:
                    image._save_image2hdf5(grp=out_set_group, compression="gzip", compression_opts=4)
        else:
            raise ValueError('image_list must be a list of Image objects or src_path must be a valid hdf5 file.')

    def _add_image2group(self, group, image: Image, overwrite: bool):
        """Helper function to add an image to a group that allows for reusing file handlers"""
        if image.name in group and overwrite is False:
            raise ValueError(f'Image named {image.name} already exists in ImageSet {self.name}.')
        else:
            image._save_image2hdf5(grp=group, compression="gzip", compression_opts=4)

    def add_image(self, image: Image, overwrite: bool | None = None):
        """
        Adds an image to an HDF5 file within a specified group.

        This method writes the provided image to an HDF5 file under a specified group.
        If the `overwrite` flag is set to True, the image will replace an existing
        dataset with the same name in the group. If set to False and a dataset with the
        same name already exists, the method will raise a ValueError.

        Args:
            image (Image): The image object to be added to the HDF5 group.
            overwrite (bool, optional): Indicates whether to overwrite an existing
                dataset if one with the same name exists. Defaults to None. If None, the method uses the
                initial overwrite value used when the class was created

        Raises:
            ValueError: If the `overwrite` flag is set to False and the image name is already in the ImageSet
        """
        with h5py.File(self._out_path, mode='r+') as out_handler:
            set_group = self._get_hdf5_group(out_handler, self._hdf5_set_group_key)
            self._add_image2group(group=set_group, image=image, overwrite=overwrite if overwrite else self._overwrite)

    @staticmethod
    def _get_hdf5_group(handler, name):
        name = str(name)
        if name in handler:
            return handler[name]
        else:
            return handler.create_group(name)

    def get_image_names(self) -> List[str]:
        """
        Retrieves the names of all images stored within the specified HDF5 group.

        This method opens an HDF5 file in read mode, accesses the specific group defined
        by the class's `_hdf5_set_group_key`, and retrieves the keys within that group,
        which represent the names of stored images.

        Returns:
            List[str]: A list of image names present in the specified HDF5 group.
        """
        with h5py.File(self._out_path, mode='r') as out_handler:
            set_group = self._get_hdf5_group(out_handler, self._hdf5_set_group_key)
            names = set_group.keys()
        return list(names)

    def iter_images(self) -> iter:
        for image_name in self.get_image_names():
            with h5py.File(self._out_path, mode='r') as out_handler:
                pass

                # TODO: Make Image hdf5 loader
