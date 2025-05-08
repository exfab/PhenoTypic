from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import h5py
import numpy as np

from ._image_hsv_handler import ImageHsvHandler

class ImageIOHandler(ImageHsvHandler):

    def save_image_to_hdf5(self, filename, compression="gzip", compression_opts=4):
        """
        Save an ImageHandler instance to an HDF5 file under /phenotypic/<self.name>/.

        Parameters:
          self: your ImageHandler instance
          filename: path to .h5 file (will be created or appended)
          compression: compression filter (e.g., "gzip", "szip", or None)
          compression_opts: level for gzip (1–9)
        """
        with h5py.File(filename, "a") as f:
            # 1) Create (or open) /phenotypic/<image_name> group
            grp = f.require_group(f"phenotypic/{self.name}")

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

            # TODO: enh_matrix
            grp.create_dataset(
                "enh_matrix",
                data=self.enh_matrix,
                dtype=self.enh_matrix.dtype,
                chunks=True,
                compression=compression,
                compression_opts=compression_opts
            )
            grp.create_dataset(
                "objmap",
                data=self.objmap,
                dtype=self.objmap.dtype,
                chunks=True,
                compression=compression,
                compression_opts=compression_opts
            )

            # 3) Store string/enum as a group attribute
            #    h5py supports variable-length UTF-8 strings automatically
            grp.attrs["imformat"] = self.imformat

            # 4) Store protected metadata in its own subgroup
            prot = grp.require_group("protected_metadata")
            for key, val in self._metadata.protected.items():
                prot.attrs[key] = val

            # 5) Store public metadata in its own subgroup
            pub = grp.require_group("public_metadata")
            for key, val in self._metadata.public.items():
                pub.attrs[key] = val

    def load_image_from_hdf5(self, filename, image_name):
        """
        Load an ImageHandler instance from an HDF5 file at /phenotypic/<image_name>/.
        """
        with h5py.File(filename, "r") as f:
            grp = f[f"phenotypic/{image_name}"]

            # Instantiate a blank handler and populate internals
            img = self.__class__(input_image=None)

            # 1) Read datasets back into numpy arrays
            img._array = grp["array"][()]
            img._matrix = grp["matrix"][()]
            img._enh_matrix = grp["enh_matrix"][()]
            # If your objmap backend expects a sparse matrix, convert accordingly;
            # here we load as dense:
            img._data.sparse_object_map = grp["objmap"][()]

            # 2) Restore format
            img._image_format = grp.attrs["imformat"]

            # 3) Restore metadata
            prot = grp["protected_metadata"].attrs
            img._metadata.protected.clear()
            img._metadata.protected.update({k: prot[k] for k in prot})

            pub = grp["public_metadata"].attrs
            img._metadata.public.clear()
            img._metadata.public.update({k: pub[k] for k in pub})

        return img