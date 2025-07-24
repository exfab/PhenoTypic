from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING: from phenotypic import Image

import h5py
import numpy as np
import pickle
from os import PathLike
from pathlib import Path
import posixpath
from packaging.version import Version

import skimage as ski

import phenotypic
from phenotypic.util.exceptions_ import UnsupportedFileTypeError
from phenotypic.util.constants_ import IMAGE_FORMATS, IO
from phenotypic.util import HDF
from ._image_color_handler import ImageColorSpace

class ImageIOHandler(ImageColorSpace):

    def __init__(self,
                 input_image: np.ndarray | Image | PathLike | Path | str | None = None,
                 imformat: str | None = None,
                 name: str | None = None, bit_depth: Literal[8, 16, 32] | None = 16):
        if isinstance(input_image, (PathLike, Path, str)):
            input_image = Path(input_image)
            super().__init__(input_image=self.imread(input_image), imformat=imformat, name=name, bit_depth=bit_depth)
        else:
            super().__init__(input_image=input_image, imformat=imformat, name=name, bit_depth=bit_depth)

    @classmethod
    def imread(cls, filepath: PathLike) -> Image:
        """
        Reads an image file from a given file path, processes it as per its format, and sets the image
        along with its schema in the current instance. Supports RGB formats (png, jpg, jpeg) and
        grayscale formats (tif, tiff). The name of the image processing instance is updated to match
        the file name without the extension. If the file format is unsupported, an exception is raised.

        Args:
            filepath (PathLike): Path to the image file to be read.

        Returns:
            Type[Image]: The current instance with the newly loaded image and schema.

        Raises:
            UnsupportedFileType: If the file format is not supported.
        """
        # Convert to a Path object
        filepath = Path(filepath)
        if filepath.suffix in IO.ACCEPTED_FILE_EXTENSIONS:
            image = cls(input_image=None)
            image.set_image(
                input_image=ski.io.imread(filepath),
            )
            image.name = filepath.stem
            return image
        else:
            raise UnsupportedFileTypeError(filepath.suffix)

    @staticmethod
    def _get_hdf5_group(handler, name):
        """
        Retrieves an HDF5 group from the given handler by name. If the group does not
        exist, it creates a new group with the specified name.

        Args:
            handler: HDF5 file or group handler used to manage HDF5 groups.
            name: The name of the group to retrieve or create.

        Returns:
            h5py.Group: The requested or newly created HDF5 group.
        """
        name = str(name)
        if name in handler:
            return handler[name]
        else:
            return handler.create_group(name)

    @staticmethod
    def _save_array2hdf5(group, array, name, **kwargs):
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

    def _save_image2hdfgroup(self, grp, compression, compression_opts, overwrite=False,):
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

        match self._bit_depth:
            case np.uint8:
                bit_depth = 8
            case np.uint16:
                bit_depth = 16
            case np.float32:
                bit_depth = 32
            case np.float64:
                bit_depth = 64
            case _:
                raise RuntimeError(f'Unsupported bit depth {self._bit_depth}')
        image_group.attrs["bit_depth"] = bit_depth

        image_group.attrs["version"] = phenotypic.__version__
        image_group.attrs["image_type"] = self._image_format.value

        # 4) Store protected metadata in its own subgroup
        prot = image_group.require_group("protected_metadata")
        for key, val in self._metadata.protected.items():
            prot.attrs[key] = str(val)

        # 5) Store public metadata in its own subgroup
        pub = image_group.require_group("public_metadata")
        for key, val in self._metadata.public.items():
            pub.attrs[key] = str(val)

    def save2hdf5(self, filename, compression="gzip", compression_opts=4, overwrite=False,):
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
            self._save_image2hdfgroup(grp=grp, compression=compression, compression_opts=compression_opts, overwrite=overwrite,)

    @classmethod
    def _load_from_hdf5_group(cls, group)->Image:
        # Instantiate a blank handler and populate internals
        img = cls(bit_depth=group.attrs["bit_depth"])

        # Load Image Format
        try:
            img._image_format = IMAGE_FORMATS(group.attrs["imformat"])
        except ValueError:
            raise ValueError(f"Unsupported imformat {group.attrs['imformat']} for Image")

        # Read datasets back into numpy arrays
        if img._image_format.is_array():
            img._data.array = group["array"][()]
        img._data.matrix = group["matrix"][()]
        img = img.reset()

        img.enh_matrix[:] = group["enh_matrix"][()]
        # If your objmap backend expects a sparse matrix, convert accordingly;
        # here we load as dense:
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
            grp = filehandler[str(IO.SINGLE_IMAGE_HDF5_PARENT_GROUP / image_name)]
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
                '_image_format': self._image_format,
                '_bit_depth': self._bit_depth,
                "_data.array": self._data.array,
                '_data.matrix': self._data.matrix,
                '_data.enh_matrix': self._data.enh_matrix,
                'objmap': self.objmap[:],
                "protected_metadata": self._metadata.protected,
                "public_metadata": self._metadata.public,
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

        if Version(phenotypic.__version__) < Version("0.7.1"):
            instance = cls(input_image=None, imformat=None, name=None)
        else:
            instance = cls(input_image=None, imformat=None, name=None, bit_depth=loaded["_bit_depth"])

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
