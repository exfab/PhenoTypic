from __future__ import annotations

import warnings
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import h5py
import numpy as np
import pickle
from os import PathLike
from pathlib import Path

try:
    import rawpy
except ImportError:
    rawpy = None

import skimage as ski

import phenotypic
from phenotypic.tools.exceptions_ import UnsupportedFileTypeError
from phenotypic.tools.constants_ import IMAGE_FORMATS, IO
from phenotypic.tools.hdf_ import HDF
from ._image_color_handler import ImageColorSpace


class ImageIOHandler(ImageColorSpace):

    def __init__(self,
                 arr: np.ndarray | Image | None = None,
                 imformat: str | None = None,
                 name: str | None = None, **kwargs):
        super().__init__(arr=arr, imformat=imformat, name=name)

    @classmethod
    def imread(cls,
               filepath: PathLike,
               rawpy_params: dict | None = None,
               **kwargs) -> Image:
        """
        Reads an image file from the specified file path and returns an Image instance.
        Supports common image formats and raw sensor data formats with optional processing
        configurations.

        Args:
            filepath (PathLike): The path to the image file to be read.
            gamma (Tuple[int, int] | None): Gamma correction as a tuple (numerator, denominator).
                If None, defaults to (1, 1) for no gamma correction.
            demosaic_algorithm (rawpy.DemosaicAlgorithm | None): The demosaicing algorithm to
                apply if reading raw sensor data. Defaults to None, using the
                `rawpy.DemosaicAlgorithm.LINEAR`.
            use_camera_wb (bool): Flag to indicate whether to use the camera's white balance
                configuration if reading raw sensor data. Defaults to False.
            median_filter_passes (int): The number of median filtering passes to apply to raw
                sensor data. Defaults to 0.
            rawpy_params (dict | None): Additional keyword arguments for rawpy's `postprocess`
                function when reading raw sensor data. Defaults to None.
            **kwargs: Additional keyword arguments for further configuration during the
                initialization of the Image instance.

        Returns:
            Image: An Image instance containing the image data and metadata.

        Raises:
            UnsupportedFileTypeError: If the file extension is not supported or the necessary
                libraries are not available for processing the specified file type.
        """
        # Convert to a Path object
        filepath = Path(filepath)
        rawpy_params = rawpy_params or {}
        imformat = None

        if filepath.suffix.lower() in IO.ACCEPTED_FILE_EXTENSIONS:  # normal images
            arr = ski.io.imread(fname=filepath)

        elif filepath.suffix.lower() in IO.RAW_FILE_EXTENSIONS and rawpy is not None:  # raw sensor data handling
            use_auto_wb = rawpy_params.pop('use_auto_wb', False)
            use_camera_wb = rawpy_params.pop('use_camera_wb', False)

            no_auto_scale = rawpy_params.pop('no_auto_scale', False)  # TODO: implement calibration schema
            no_auto_bright = rawpy_params.pop('no_auto_bright', False)  # TODO: implement calibration schema

            if rawpy.DemosaicAlgorithm.AMAZE.isSupported():
                default_demosaic = rawpy.DemosaicAlgorithm.AMAZE
            else:
                default_demosaic = rawpy.DemosaicAlgorithm.AHD

            demosaic_algorithm = rawpy_params.pop('demosaic_algorithm', default_demosaic)
            gamma = rawpy_params.pop('gamma', (1, 1))
            with rawpy.imread(str(filepath)) as raw:
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
            imformat = IMAGE_FORMATS.LINEAR_RGB

        else:
            raise UnsupportedFileTypeError(filepath.suffix)

        imformat = kwargs.pop('imformat', imformat)
        image = cls(arr=arr, imformat=imformat, **kwargs)
        image.name = filepath.stem
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
            raise ValueError('hdf5 handler in SWMR mode cannot create group')
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
                        'Shape does not match existing dataset shape and cannot be changed because file handler is in SWMR mode')
            else:
                del group[name]
                group.create_dataset(name, data=array, dtype=array.dtype, **kwargs)
        else:
            group.create_dataset(name, data=array, dtype=array.dtype, **kwargs)

    def _save_image2hdfgroup(self, grp, compression='gzip', compression_opts=4, overwrite=False, ):
        """Saves the image as a new group into the input hdf5 group."""
        if overwrite and self.name in grp:
            del grp[self.name]

        # create the group container for the images information
        image_group = self._get_hdf5_group(grp, self.name)

        if self._image_format.is_array():
            array = self.array[:]
            HDF.save_array2hdf5(
                    group=image_group, array=array, name="array",
                    dtype=array.dtype,
                    compression=compression, compression_opts=compression_opts,
            )

        matrix = self.matrix[:]
        HDF.save_array2hdf5(
                group=image_group, array=matrix, name="matrix",
                dtype=matrix.dtype,
                compression=compression, compression_opts=compression_opts,
        )

        enh_matrix = self.enh_matrix[:]
        HDF.save_array2hdf5(
                group=image_group, array=enh_matrix, name="enh_matrix",
                dtype=enh_matrix.dtype,
                compression=compression, compression_opts=compression_opts,
        )

        objmap = self.objmap[:]
        HDF.save_array2hdf5(
                group=image_group, array=objmap, name="objmap",
                dtype=objmap.dtype,
                compression=compression, compression_opts=compression_opts,
        )

        # 3) Store string/enum as a group attribute
        #    h5py supports variable-length UTF-8 strings automatically
        image_group.attrs["imformat"] = self.imformat.value

        image_group.attrs["version"] = phenotypic.__version__

        # 4) Store protected metadata in its own subgroup
        prot = image_group.require_group("protected_metadata")
        for key, val in self._metadata.protected.items():
            prot.attrs.modify(key, str(val))

        # 5) Store public metadata in its own subgroup
        pub = image_group.require_group("public_metadata")
        for key, val in self._metadata.public.items():
            pub.attrs.modify(key, str(val))

    def save2hdf5(self, filename, compression="gzip", compression_opts=4, overwrite=False, ):
        """
        Save an ImageHandler instance to an HDF5 file under /phenotypic/images<self.name>/.

        Parameters:
          self: your ImageHandler instance
          filename: path to .h5 file (will be created or appended)
          compression: compression filter (e.g., "gzip", "szip", or None)
          compression_opts: level for gzip (1–9)

        Raises:
            Warnings: If the phenotypic version does not match the version used when saving to the HDF5 file.
        """
        with h5py.File(filename, mode="a") as filehandler:
            # 1) Create image group if it doesnt already exist & sets grp obj
            parent_grp = self._get_hdf5_group(filehandler, IO.SINGLE_IMAGE_HDF5_PARENT_GROUP)
            if 'version' in parent_grp.attrs:
                if parent_grp.attrs['version'] != phenotypic.__version__:
                    raise warnings.warn(f"Version mismatch: {parent_grp.attrs['version']} != {phenotypic.__version__}")
            else:
                parent_grp.attrs['version'] = phenotypic.__version__

            grp = self._get_hdf5_group(filehandler, IO.SINGLE_IMAGE_HDF5_PARENT_GROUP)

            # 2) Save large arrays as datasets with chunking & compression
            self._save_image2hdfgroup(grp=grp, compression=compression, compression_opts=compression_opts,
                                      overwrite=overwrite, )

    @classmethod
    def _load_from_hdf5_group(cls, group, **kwargs) -> Image:
        # Instantiate a blank handler and populate internals
        # Load Image Format
        imformat = IMAGE_FORMATS[group.attrs["imformat"]]
        # Read datasets back into numpy arrays with proper dtype handling
        matrix_data = group["matrix"][()]
        if imformat.is_array():
            # For arrays, preserve the original dtype from HDF5
            array_data = group["array"][()]
            img = cls(arr=array_data, imformat=imformat.value, **kwargs)
            img.matrix[:] = matrix_data
        else:
            img = cls(arr=matrix_data, imformat=imformat.value, **kwargs)

        # Load enhanced matrix and object map with proper dtype casting
        enh_matrix_data = group["enh_matrix"][()]
        img.enh_matrix[:] = enh_matrix_data

        # Object map should preserve its original dtype (usually integer labels)
        img.objmap[:] = group["objmap"][()]

        # 3) Restore metadata
        prot = group["protected_metadata"].attrs
        img._metadata.protected.clear()
        img._metadata.protected.update({k: prot[k] for k in prot})

        pub = group["public_metadata"].attrs
        img._metadata.public.clear()
        img._metadata.public.update({k: pub[k] for k in pub})
        return img

    @classmethod
    def load_hdf5(cls, filename, image_name) -> Image:
        """
        Load an ImageHandler instance from an HDF5 file at the default hdf5 location
        """
        with h5py.File(filename, "r") as filehandler:
            grp = filehandler[str(IO.SINGLE_IMAGE_HDF5_PARENT_GROUP/image_name)]
            img = cls._load_from_hdf5_group(grp)

        return img

    def save2pickle(self, filename: str) -> None:
        """
        Saves the current ImageIOHandler instance's data and metadata to a pickle file.

        Args:
            filename: Path to the pickle file to write.
        """
        with open(filename, 'wb') as filehandler:
            pickle.dump({
                '_image_format'     : self._image_format,
                "_data.array"       : self._data.array,
                '_data.matrix'      : self._data.matrix,
                '_data.enh_matrix'  : self._data.enh_matrix,
                'objmap'            : self.objmap[:],
                "protected_metadata": self._metadata.protected,
                "public_metadata"   : self._metadata.public,
            }, filehandler,
            )

    @classmethod
    def load_pickle(cls, filename: str) -> Image:
        """
        Loads ImageIOHandler data and metadata from a pickle file and returns a new instance.

        Args:
            filename: Path to the pickle file to read.

        Returns:
            A new Image instance with data and metadata restored.
        """
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)

        instance = cls(arr=None, imformat=None, name=None)

        instance._image_format = loaded["_image_format"]
        instance._data.array = loaded["_data.array"]
        instance._data.matrix = loaded["_data.matrix"]

        instance.enh_matrix.reset()
        instance.objmap.reset()

        instance._data.enh_matrix = loaded["_data.enh_matrix"]
        instance.objmap[:] = loaded["objmap"]
        instance._metadata.protected = loaded["protected_metadata"]
        instance._metadata.public = loaded["public_metadata"]
        return instance
