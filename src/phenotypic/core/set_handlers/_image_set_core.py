import os
from pathlib import Path

from typing import List
import h5py
from os import PathLike
from packaging.version import Version

from phenotypic import Image
from phenotypic.util.constants_ import IMAGE_SET_PARENT_GROUP, SINGLE_IMAGE_HDF5_PARENT_GROUP

class ImageSetCore:
    IMAGE_SET_PARENT_GROUP = Path(f'phenotypic/ImageSet/')

    def __init__(self, name: str, image_list: List[Image] | None = None,
                 src_path: PathLike | None = None,
                 outpath: PathLike | None = None,
                 overwrite: bool = True, ):
        assert (image_list and src_path is None) or (image_list is None and src_path), 'Only one of image_list or src_path can be provided.'

        self.name = name
        src_path, outpath = Path(src_path) if src_path else None, Path(outpath) if src_path else Path.cwd()
        if outpath is None:

        # If input path is an hdf5 file,
        if src_path is not None and src_path.is_file() and src_path.suffix == '.h5':

            with h5py.File(src_path, mode='a') as src_filehandler, h5py.File(outpath, mode='a') as out_filehandler:
                src_parent_group = self._get_hdf5_group(src_filehandler, IMAGE_SET_PARENT_GROUP)
                out_parent_group = self._get_hdf5_group(out_filehandler, IMAGE_SET_PARENT_GROUP)

                #   if the image set name is in the ImageSet group copy the images over
                if self.name in src_parent_group:
                    src_group = self._get_hdf5_group(src_filehandler, IMAGE_SET_PARENT_GROUP / self.name)

                    # Overwrite the data in the output file
                    if self.name in out_parent_group and overwrite is True:
                        del out_parent_group[self.name]

                    src_filehandler.copy(src_group, out_parent_group, name=self.name, shallow=False)

                #   else import all the images from Image section
                elif SINGLE_IMAGE_HDF5_PARENT_GROUP in src_filehandler:
                    src_image_group = self._get_hdf5_group(src_filehandler, SINGLE_IMAGE_HDF5_PARENT_GROUP)
                    src_filehandler.copy(src_image_group, out_parent_group, name=self.name, shallow=False)

                else:
                    raise ValueError(f'No ImageSet named {self.name} or Image section found in {src_path}')

        # TODO: add case for image_list



    @staticmethod
    def _get_hdf5_group(handler, name):
        name = str(name)
        if name in handler:
            return handler[name]
        else:
            return handler.create_group(name)
