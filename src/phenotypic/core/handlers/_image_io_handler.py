from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import h5py
import numpy as np
import pickle
from os import PathLike
from pathlib import Path

import skimage as ski

from phenotypic.util.exceptions_ import UnsupportedFileTypeError
from phenotypic.util.constants_ import IMAGE_FORMATS
from ._image_hsv_handler import ImageHsvHandler


class ImageIOHandler(ImageHsvHandler):
    def __init__(self,
                 input_image: np.ndarray | Image | PathLike | Path | str | None = None,
                 imformat: str | None = None,
                 name: str | None = None):
        if isinstance(input_image, (PathLike, Path, str)):
            input_image = Path(input_image)
            super().__init__(input_image=self.imread(input_image), imformat=imformat, name=name)
        else:
            super().__init__(input_image=input_image, imformat=imformat, name=name)

    def save2hdf5(self, filename, compression="gzip", compression_opts=4):
        """
        Save an ImageHandler instance to an HDF5 file under /phenotypic/<self.name>/.

        Parameters:
          self: your ImageHandler instance
          filename: path to .h5 file (will be created or appended)
          compression: compression filter (e.g., "gzip", "szip", or None)
          compression_opts: level for gzip (1–9)
        """
        with h5py.File(filename, "a") as filehandler:
            # 1) Create (or open) /phenotypic/<image_name> group
            grp = filehandler.require_group(f"phenotypic/{self.name}")

            # 2) Save large arrays as datasets with chunking & compression

            array = self.array[:]
            grp.create_dataset(
                "array",
                data=array,
                dtype=array.dtype,
                chunks=True,
                compression=compression,
                compression_opts=compression_opts
            )

            matrix = self.matrix[:]
            grp.create_dataset(
                "matrix",
                data=matrix,
                dtype=matrix.dtype,
                chunks=True,
                compression=compression,
                compression_opts=compression_opts
            )

            enh_matrix = self.enh_matrix[:]
            grp.create_dataset(
                "enh_matrix",
                data=enh_matrix,
                dtype=enh_matrix.dtype,
                chunks=True,
                compression=compression,
                compression_opts=compression_opts
            )

            objmap = self.objmap[:]
            grp.create_dataset(
                "objmap",
                data=objmap,
                dtype=objmap.dtype,
                chunks=True,
                compression=compression,
                compression_opts=compression_opts
            )

            # 3) Store string/enum as a group attribute
            #    h5py supports variable-length UTF-8 strings automatically
            grp.attrs["imformat"] = self.imformat.value

            # 4) Store protected metadata in its own subgroup
            prot = grp.require_group("protected_metadata")
            for key, val in self._metadata.protected.items():
                prot.attrs[key] = str(val)

            # 5) Store public metadata in its own subgroup
            pub = grp.require_group("public_metadata")
            for key, val in self._metadata.public.items():
                pub.attrs[key] = str(val)

    @classmethod
    def load_hdf5(cls, filename, image_name) -> Image:
        """
        Load an ImageHandler instance from an HDF5 file at /phenotypic/<image_name>/.
        """
        with h5py.File(filename, "r") as f:
            grp = f[f"phenotypic/{image_name}"]

            # Instantiate a blank handler and populate internals
            img = cls(input_image=None)

            # 1) Read datasets back into numpy arrays
            img._array = grp["array"][()]
            img._matrix = grp["matrix"][()]
            img._enh_matrix = grp["enh_matrix"][()]
            # If your objmap backend expects a sparse matrix, convert accordingly;
            # here we load as dense:
            img._data.sparse_object_map = grp["objmap"][()]

            # 2) Restore format
            try:
                img._image_format = IMAGE_FORMATS(grp.attrs["imformat"])
            except ValueError:
                raise ValueError(f"Unsupported imformat {grp.attrs['imformat']} for Image")

            # 3) Restore metadata
            prot = grp["protected_metadata"].attrs
            img._metadata.protected.clear()
            img._metadata.protected.update({k: prot[k] for k in prot})

            pub = grp["public_metadata"].attrs
            img._metadata.public.clear()
            img._metadata.public.update({k: pub[k] for k in pub})

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
                "_data.array": self._data.array,
                '_data.matrix': self._data.matrix,
                '_data.enh_matrix': self._data.enh_matrix,
                'objmap': self.objmap[:],
                "protected_metadata": self._metadata.protected,
                "public_metadata": self._metadata.public,
            }, filehandler
            )

    @classmethod
    def load_pickle(cls, filename: str) -> Image:
        """
        Loads ImageIOHandler data and metadata from a pickle file and returns a new instance.

        Args:
            filename: Path to the pickle file to read.

        Returns:
            A new ImageIOHandler instance with _data and _metadata restored.
        """
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)
        instance = cls(input_image=None)
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

    @classmethod
    def imread(cls, filepath: PathLike) -> Image:
        """
        Reads an _root_image file from a given file path, processes it as per its format, and sets the _root_image
        along with its schema in the current instance. Supports RGB formats (png, jpg, jpeg) and
        grayscale formats (tif, tiff). The name of the _root_image processing instance is updated to match
        the file name without the extension. If the file format is unsupported, an exception is raised.

        Args:
            filepath (PathLike): Path to the _root_image file to be read.

        Returns:
            Type[Image]: The current instance with the newly loaded _root_image and schema.

        Raises:
            UnsupportedFileType: If the file format is not supported.
        """
        # Convert to a Path object
        filepath = Path(filepath)
        if filepath.suffix in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            image = cls(input_image=None)
            image.set_image(
                input_image=ski.io.imread(filepath)
            )
            image.name = filepath.stem
            return image
        else:
            raise UnsupportedFileTypeError(filepath.suffix)
