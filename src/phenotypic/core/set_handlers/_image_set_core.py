import os
from pathlib import Path

from typing import List
import h5py
from os import PathLike

from phenotypic import Image
from phenotypic.util.constants_ import IO


class ImageSetCore:

    def __init__(self, name: str, image_list: List[Image] | None = None,
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
        # Only an image_list xor src_path should be given
        assert (image_list and src_path is None) or (image_list is None and src_path), 'Only one of image_list or src_path can be provided.'

        self.name = name
        src_path, out_path = Path(src_path) if src_path else None, Path(out_path) if out_path else Path.cwd()
        self.name, self._src_path, self._out_path = str(name), src_path, out_path
        self._overwrite = overwrite
        self._hdf5_set_group_path = IO.IMAGE_SET_HDF5_PARENT_GROUP / self.name

        # If input source path handling
        if src_path:
            # If input path is an hdf5 file
            if src_path.is_file() and src_path.suffix == '.h5':
                with h5py.File(src_path, mode='a') as src_filehandler, h5py.File(out_path, mode='a') as out_filehandler:
                    src_parent_group = self._get_hdf5_group(src_filehandler, IO.IMAGE_SET_HDF5_PARENT_GROUP)
                    out_parent_group = self._get_hdf5_group(out_filehandler, IO.IMAGE_SET_HDF5_PARENT_GROUP)

                    #   if the image set name is in the ImageSet group, copy the images over
                    if self.name in src_parent_group:
                        src_group = self._get_hdf5_group(src_filehandler, self._hdf5_set_group_path)

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
                    out_group = self._get_hdf5_group(out_handler, self._hdf5_set_group_path)

                    # Overwrite handling
                    if self.name in out_group and overwrite is True: del out_group[self.name]
                    out_set_group = self._get_hdf5_group(out_handler, self._hdf5_set_group_path)

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
                out_set_group = self._get_hdf5_group(out_handler, self._hdf5_set_group_path)

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

    def add_image(self, image: Image, overwrite: bool = False):
        """
        Adds an image to an HDF5 file within a specified group.

        This method writes the provided image to an HDF5 file under a specified group.
        If the `overwrite` flag is set to True, the image will replace an existing
        dataset with the same name in the group. If set to False and a dataset with the
        same name already exists, the method will raise a ValueError.

        Args:
            image (Image): The image object to be added to the HDF5 group.
            overwrite (bool, optional): Indicates whether to overwrite an existing
                dataset if one with the same name exists. Defaults to False.

        Raises:
            ValueError: If the `overwrite` flag is set to False and the image name is already in the ImageSet
        """
        with h5py.File(self._out_path, mode='a') as out_handler:
            set_group = self._get_hdf5_group(out_handler, self._hdf5_set_group_path)
            self._add_image2group(group=set_group, image=image, overwrite=overwrite)

    @staticmethod
    def _get_hdf5_group(handler, name):
        name = str(name)
        if name in handler:
            return handler[name]
        else:
            return handler.create_group(name)
