from __future__ import annotations

import tempfile
import weakref
from typing import TYPE_CHECKING, Literal, Any, Dict

if TYPE_CHECKING: from phenotypic import Image

import os
import posixpath
from pathlib import Path

from typing import List
import h5py
from os import PathLike

from phenotypic.abstract import GridFinder
from phenotypic.util.constants_ import IO
from phenotypic.util import HDF
import phenotypic as pht


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

    def __init__(self,
                 name: str,
                 imparams: Dict[str, Any] | None = None,
                 imtype: Literal["Image", "GridImage"] = "Image",
                 src: List[Image] | PathLike | str | None = None,
                 outpath: PathLike | str | None = None,
                 default_mode: Literal['temp', 'cwd'] = 'temp',
                 overwrite: bool = False, ):
        """
        Initializes an image set for bulk processing of images.

        This constructor is responsible for setting up an image set by either importing
        images from a provided list, an HDF5 file, or a directory. It also handles the
        storing of the images into an output HDF5 file, with options for overwriting
        existing data. The `src` parameter automatically detects the input type.

        Args:
            name (str): The name of the image set to initialize.
            grid_finder: (Image | None): a grid finder object used for defining grids in an Image. If None ImageSet will use regular images
            src (List[Image] | PathLike | None, optional): The source for images. Can be:
                - A list of Image objects for importing from in-memory images
                - A PathLike object pointing to a source directory or HDF5 file containing images
                - None to connect to the output HDF5 file only
            outpath (PathLike | None, optional): The output HDF5 file where
                the image set will be stored. Defaults to the current working directory.
            overwrite (bool): Determines whether to overwrite existing data in the output
                HDF5 file. Defaults to False.

        Raises:
            ValueError: If no images or image sections are found in the provided `src` path.
            ValueError: If `src` is not a list of `Image` objects or a valid path.
        """
        self.default_mode = default_mode
        import phenotypic

        self.name = name

        self.imparams = imparams or {}
        self.imtype = imtype

        # Automatically detect the type of src parameter
        image_list = None
        src_path = None

        if src is not None:
            if isinstance(src, list):
                # src is a list of images
                image_list = src
            else:
                # src is a path-like object
                src_path = src

        src_path = Path(src_path) if src_path else None
        # Track ownership of outpath for cleanup
        owns_outpath = False
        if outpath:
            outpath = Path(outpath)
        else:  # if outpath is None
            if self.default_mode == 'cwd':
                outpath = Path.cwd()/f'{self.name}.hdf5'
            elif self.default_mode == 'temp':
                # Create a temporary file path we own and can clean up later.
                fd, tmp = tempfile.mkstemp(suffix='.h5', prefix=f'{self.name}_')
                os.close(fd)  # Close OS-level fd; HDF will reopen as needed
                outpath = Path(tmp)
                owns_outpath = True

        if outpath.is_dir(): outpath = outpath/f'{self.name}.hdf5'

        # Track whether this instance owns the outpath and should delete it on GC
        self._owns_outpath = owns_outpath
        self._out_finalizer = weakref.finalize(self, self._cleanup_outpath, outpath) if self._owns_outpath else None

        self.name, self._src_path, self._out_path = str(name), src_path, outpath
        self.hdf_ = HDF(filepath=outpath, name=self.name, mode='set')

        self._overwrite = overwrite

        # Define hdf5 group paths
        self._hdf5_parent_group_key: str = HDF.IMAGE_SET_ROOT_POSIX
        self._hdf5_set_group_key: str = posixpath.join(self._hdf5_parent_group_key, self.name)

        # Reminder: SetMeasurementAccessor are stored with each image
        #   self._hdf5_images_group_key/images/<image_name>/measurements <- that image's measurements
        self._hdf5_images_group_key: str = posixpath.join(self._hdf5_set_group_key, 'images')

        if src_path:  # If src is pathlike object

            # If input path is an hdf5 file
            if src_path.is_file() and src_path.suffix == '.h5':
                if src_path == outpath:  # If src and outpath are the same
                    pass

                elif src_path.suffix in HDF.EXT:  # If src and outpath are different, but both hdf files
                    with h5py.File(src_path, mode='r',
                                   libver='latest') as src_filehandler, self.hdf_.safe_writer() as writer:
                        src_parent_group = self._get_hdf5_group(src_filehandler, self)
                        out_parent_group = self.hdf_.get_root_group(writer)

                        if self.name in src_parent_group:  # If matching set name in src -> cpy images to outpath
                            src_group = self._get_hdf5_group(src_filehandler, self._hdf5_set_group_key)

                            # overwrite if overwrite is true
                            if self.name in out_parent_group and overwrite is True: del out_parent_group[self.name]

                            # Should place a copy of the src_group into the parent group
                            src_filehandler.copy(src_group, out_parent_group, name=self.name, shallow=False)

                        elif HDF.SINGLE_IMAGE_ROOT_POSIX in src_filehandler:  # If no matching set name -> import all single images
                            src_image_group = self._get_hdf5_group(src_filehandler, self._hdf5_parent_group_key)

                            # overwrite if overwrite is true
                            if self.name in out_parent_group and overwrite is True: del out_parent_group[self.name]
                            src_filehandler.copy(src_image_group, out_parent_group, name=self.name, shallow=False)

                        else:  # No matching name and no images in the src hdf file
                            raise ValueError(f'No ImageSet named {self.name} or Image section found in {src_path}')

                elif src_path.is_dir():  # If src_path is a directory -> Assume directory of images
                    image_filenames = [x for x in os.listdir(src_path) if
                                       x.endswith(IO.ACCEPTED_FILE_EXTENSIONS + IO.RAW_FILE_EXTENSIONS)]
                    image_filenames.sort()
                    with self.hdf_.safe_writer() as writer:
                        out_parent_group = self.hdf_.get_root_group(writer)

                        # Overwrite handling
                        if self.name in out_parent_group and overwrite is True: del out_parent_group[self.name]
                        images_group = self.hdf_.get_data_group(writer)

                        for fname in image_filenames:
                            if self.grid_finder is not None:
                                template = pht.GridImage
                            else:
                                template = pht.Image
                            image = template.imread(src_path/fname)
                            image._save_image2hdfgroup(grp=images_group, compression="gzip", compression_opts=4)

        # Image list handling
        # Only need out handler for this
        elif isinstance(image_list, list):
            assert all(
                    isinstance(x, phenotypic.Image) for x in image_list), 'image_list must be a list of Image objects.'
            with self.hdf_.safe_writer() as writer:
                out_group = self._get_hdf5_group(writer, self._hdf5_parent_group_key)

                # Overwrite the data in the output folder
                if self.name in out_group and overwrite is True: del out_group[self.name]
                images_group = self.hdf_.get_data_group(writer)

                for image in image_list:
                    image._save_image2hdfgroup(grp=images_group, compression="gzip", compression_opts=4)
        elif not src_path and not image_list:  # connect to outpath hdf5 file only
            pass
        else:
            raise ValueError('image_list must be a list of Image objects or src_path must be a valid hdf5 file.')

    def close(self) -> None:
        """Close resources and delete the temporary outpath if this instance owns it."""
        fin = getattr(self, "_out_finalizer", None)
        if fin and fin.alive:
            fin()

    def _get_template(self):
        if self.imtype == 'GridImage':
            return pht.GridImage
        elif self.imtype == 'Image':
            return pht.Image
        else:
            raise ValueError(f'Image type {self.imtype} is not supported.')

    # TODO: -[] import from other hdf5 file

    # TODO: -[] load from list of images

    # TODO: -[x] import directory case
    def import_dir(self, dirpath: Path) -> None:
        dirpath = Path(dirpath)
        if not dirpath.is_dir(): raise ValueError(f'{dirpath} is not a directory.')
        filepaths = [dirpath/x for x in os.listdir(dirpath)
                     if x.endswith(IO.ACCEPTED_FILE_EXTENSIONS + IO.RAW_FILE_EXTENSIONS)]
        filepaths.sort()
        with self.hdf_.safe_writer() as writer:
            data_group = self.hdf_.get_data_group(writer)
            template = self._get_template()
            for fpath in filepaths:
                image = template.imread(fpath, **self.imparams)
                image._save_image2hdfgroup(grp=data_group, compression="gzip", compression_opts=4)

    @staticmethod
    def _cleanup_outpath(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _add_image2group(self, group, image: Image, overwrite: bool):
        """Helper function to add an image to a group that allows for reusing file handlers"""
        if image.name in group and overwrite is False:
            raise ValueError(f'Image named {image.name} already exists in ImageSet {self.name}.')
        else:
            image._save_image2hdfgroup(grp=group, compression="gzip", compression_opts=4)

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
        with self.hdf_.writer() as writer:
            set_group = self.hdf_.get_group(writer, self._hdf5_images_group_key)
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
        with self.hdf_.reader() as reader:
            set_group = self.hdf_.get_data_group(reader)
            names = list(set_group.keys())
        return names

    def get_image(self, image_name: str) -> Image:
        import phenotypic as pt

        with self.hdf_.swmr_reader() as reader:
            image_group = self.hdf_.get_data_group(reader)
            if image_name in image_group:
                if self.grid_finder:
                    image = pt.GridImage(grid_finder=self.grid_finder)._load_from_hdf5_group(image_group[image_name])
                else:
                    image = pt.Image()._load_from_hdf5_group(image_group[image_name])
            else:
                raise ValueError(f'Image named {image_name} not found in ImageSet {self.name}.')
        return image

    def iter_images(self) -> iter:
        for image_name in self.get_image_names():
            with h5py.File(self._out_path, mode='r', libver='latest', swmr=True) as out_handler:
                image_group = self._get_hdf5_group(out_handler, posixpath.join(self._hdf5_images_group_key, image_name))

                template = self._get_template()

                image = template(**self.imparams)._load_from_hdf5_group(image_group)
            yield image
