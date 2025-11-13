from __future__ import annotations

import tempfile
import weakref
from typing import Any, Dict, Literal, TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import os
import posixpath
from pathlib import Path

from typing import List
import h5py
from os import PathLike

from phenotypic.tools.constants_ import IO
from phenotypic.tools import HDF
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
                 outpath: PathLike | str | None = None,
                 imtype: Literal["Image", "GridImage"] = "Image",
                 imparams: Dict[str, Any] | None = None,
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

        self.name = name

        self.imparams = imparams or {}
        self.imtype = imtype

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

        if outpath.is_dir():
            outpath = outpath/f'{self.name}.hdf5'
        else:
            if not outpath.suffix in HDF.EXT:
                raise ValueError(f'Invalid output file extension: {outpath.suffix}')

        # Track whether this instance owns the outpath and should delete it on GC
        self._owns_outpath = owns_outpath
        self._out_finalizer = weakref.finalize(self, self._cleanup_outpath, outpath) if self._owns_outpath else None

        self.name, self._out_path = str(name), outpath
        self.hdf_ = HDF(filepath=outpath, name=self.name, mode='set')

        self._overwrite = overwrite

        return

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

    def import_images(self, images: List[Image]) -> None:
        assert all(isinstance(x, pht.Image) for x in images), 'images must be a list of Image objects.'
        with self.hdf_.safe_writer() as writer:
            data_grp = self.hdf_.get_data_group(writer)
            for image in images:
                image._save_image2hdfgroup(grp=data_grp, compression="gzip", compression_opts=4)

        return

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

        return

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
        with self.hdf_.strict_writer() as writer:
            set_group = self.hdf_.get_data_group(writer)
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

    def _get_image(self, image_name: str, handle: h5py.File | h5py.Group, **kwargs) -> Image:
        """
        Fetches an image object from an HDF5 group given the image name and handle.

        This method retrieves an image from an HDF5 file or group using its name and
        a given handle. The handle can be an HDF5 file or group and should contain
        the desired image group.

        Args:
            image_name: str
                The name of the image to retrieve from the HDF5 group.
            handle: h5py.File | h5py.Group
                The HDF5 file or group handle where the image is stored.
            **kwargs:
                Additional keyword arguments passed to the Image template constructor.

        Returns:
            Image
                The corresponding Image object retrieved based on the provided
                arguments.

        Raises:
            ValueError:
                Raised if the specified image group cannot be located or loaded.
            TypeError:
                Raised if the handle provided is not of suitable types
                (h5py.File or h5py.Group).
        """
        return self._get_template()._load_from_hdf5_group(
                group=self.hdf_.get_image_group(handle=handle, image_name=image_name),
                **kwargs
        )

    def get_image(self, image_name: str) -> Image:

        with self.hdf_.swmr_reader() as reader:
            image_group = self.hdf_.get_data_group(reader)
            if image_name in image_group:
                image = self._get_template()._load_from_hdf5_group(image_group[image_name], **self.imparams)
            else:
                raise ValueError(f'Image named {image_name} not found in ImageSet {self.name}.')
        return image

    def iter_images(self) -> iter:
        for image_name in self.get_image_names():
            with h5py.File(self._out_path, mode='r', libver='latest', swmr=True) as out_handler:
                image_group = self._get_hdf5_group(out_handler, posixpath.join(self.hdf_.set_data_posix, image_name))

                template = self._get_template()

                image = template(**self.imparams)._load_from_hdf5_group(image_group)
            yield image
