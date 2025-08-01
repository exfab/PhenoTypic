from __future__ import annotations
from typing import TYPE_CHECKING, Literal, List

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
from os import PathLike
from phenotypic.util.constants_ import SET_STATUS
from ._image_set_core import ImageSetCore


class ImageSetStatus(ImageSetCore):
    """
    This class adds processing status tracking to the ImageSetCore class.

    The status will be attributes stored within each image's group similar to measurements.

    Location:
        /phenotypic/image_sets/images/<image_name>/status

    """
    def __init__(self,
                 name: str,
                 image_template: Image | None = None,
                 src: List[Image] | PathLike | None = None,
                 outpath: PathLike | None = None,
                 overwrite: bool = False, ):
        super().__init__(name=name, image_template=image_template,
                         src=src, outpath=outpath, overwrite=overwrite)
        if overwrite:
            self.reset_status()

    def reset_status(self, image_names: List[str] | str | None = None):
        """
        Resets the status attributes of specified image names or all available image
        names within the HDF5 file. This method updates the status attributes to an
        initially false state, as defined by the SET_STATUS constants, for each
        specified image.

        This method interacts with the HDF5 storage layer to ensure that the relevant
        attributes for each image (or all images if none are explicitly specified) are
        properly reset.

        Args:
            image_names (List[str] | str | None): A list of image names, a single
                image name string, or None. If None, all images' statuses are reset.
                If a string is provided, it is converted to a single-element list.
                The type must be a string, list of strings, or None.
        """
        if image_names is None:
            image_names = self.get_image_names()
        else:
            assert isinstance(image_names, (str, list)), 'image_names must be a list of image names or a str.'
            if isinstance(image_names, str):
                image_names = [image_names]

        with self.hdf_.writer() as handle:
            for name in image_names:
                status_group = self.hdf_.get_image_status_subgroup(handle=handle, image_name=name)
                for stat in SET_STATUS:
                    # Statuses are worded in a way that they should be initially false
                    status_group.attrs[stat.label] = False

    def get_status(self, image_names: List[str] | str | None = None):
        if image_names is None:
            image_names = self.get_image_names()
        else:
            assert isinstance(image_names, (str, list)), 'image_names must be a list of image names or a str.'
            if isinstance(image_names, str):
                image_names = [image_names]

        with self.hdf_.reader() as handle:
            status = []
            for name in image_names:
                status_group = self.hdf_.get_data_group(handle=handle, image_name=name)
                status.append(
                    status_group.attrs[x.label] for x in SET_STATUS
                )
        return pd.DataFrame(
            data=status,
            index=image_names,
            columns=SET_STATUS.get_headers()
        )

    def _add_image2group(self, group, image: Image, overwrite: bool):
        super()._add_image2group(group=group, image=image, overwrite=overwrite)

        # should work since it uses absolute pathing underneath
        stat_group = self.hdf_.get_image_status_subgroup(handle=group, image_name=image.name)
        for stat in SET_STATUS:
            stat_group.attrs[stat.label] = False