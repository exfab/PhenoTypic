from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table, regionprops
from typing import List

from phenotypic.util.constants_ import OBJECT, METADATA_LABELS, IMAGE_TYPES, BBOX
from phenotypic.core._image_parts.accessor_abstracts import ImageAccessor


class ObjectsAccessor(ImageAccessor):
    """An accessor for an image objects and provides various utilities for interacting with labeled objects in an image.

    This class provides methods to retrieve information about labeled objects, interact with object properties,
    and manipulate the image analysis target. It supports operations such as accessing object labels, obtaining
    region properties, slicing specific objects, and calculating object details. It interacts with image
    labeling and analysis tools like `skimage.regionprops`.

    Notes:
        - Can only be called if an :class:`PhenoTypic.abstract.ObjectDetector` has been applied to the :class:`PhenoTypic.Image` object.

    """

    def __len__(self):
        return self._root_image.num_objects

    def __iter__(self) -> Generator[Image, Any, None]:
        for i in range(self._root_image.num_objects):
            yield self[i]

    def __getitem__(self, index: int) -> Image:
        """Returns a slice of the object image based on the object's index."""
        current_object = self.props[index]
        label = current_object.label
        object_image = self._root_image[current_object.slice]
        object_image.metadata[METADATA_LABELS.IMAGE_TYPE] = IMAGE_TYPES.OBJECT.value
        object_image.objmap[object_image.objmap[:] != label] = 0
        return object_image

    @property
    def props(self):
        """Returns a list of skimage.regionprops object for each of image's objects. Useful for simple calculations.

        Returns:
            list[skimage.measure.RegionProperties]: A list of properties for all
                regions in the provided image.
        """
        return regionprops(label_image=self._root_image.objmap[:], intensity_image=self._root_image.matrix[:], cache=False)

    @property
    def labels(self) -> List[str]:
        """Returns the labels in the image. If no objects are in the image, returns a list with a single element of 1.

        Note:
            - The returned list of 1 represents that the entire image is the analysis target. This is to ensure consistency with the skimage.regionprops output and other measurements.
        """
        # considered using a simple numpy.unique() call on the object map, but wanted to guarantee that the labels will always be consistent
        # with any skimage outputs.
        return [x.label for x in self.props] if self.num_objects > 0 else []

    @property
    def slices(self):
        """Returns a list of image slices for each object in the image"""
        return [x.slice for x in self.props]

    def get_label_idx(self, object_label):
        """Returns the index of the object with the given label from a sorted array of object labels."""
        return np.where(self.labels == object_label)[0][0]

    @property
    def num_objects(self) -> int:
        """Returns the number of objects in the map."""
        return self._root_image.num_objects

    def reset(self):
        """Resets the image object map such that the analysis target is the entire image."""
        self._root_image.objmap.reset()

    def iloc(self, index: int) -> Image:
        """Returns a slice of the object image based on the object's index."""
        return self._root_image[self.props[index].slice]

    def loc(self, label_number) -> Image:
        """Returns a crop of an object from the image based on its label number.

        Args:
            label_number: (int) The label number of the object
        Returns:
            (Image) The cropped bounding box of an object as an Image
        """
        idx = self.get_label_idx(label_number)
        return self._root_image[self.props[idx].slice]

    def info(self, include_metadata=True) -> pd.DataFrame:
        """Returns a panda.DataFrame containing basic information about each object's label, bounds, and centroid in the image.

        This is useful for joining measurements across different tables.
        """
        info= pd.DataFrame(
            data=regionprops_table(
                label_image=self._root_image.objmap[:],
                properties=['label', 'centroid', 'bbox'],
            ),
        ).rename(columns={
            'label': OBJECT.LABEL,
            'centroid-0':str(BBOX.CENTER_RR),
            'centroid-1': str(BBOX.CENTER_CC),
            'bbox-0': str(BBOX.MIN_RR),
            'bbox-1': str(BBOX.MIN_CC),
            'bbox-2': str(BBOX.MAX_RR),
            'bbox-3': str(BBOX.MAX_CC),
        },
        ).set_index(OBJECT.LABEL)
        if include_metadata:
            return self._root_image.metadata.insert_metadata(info)
        else:
            return info

    def labels2series(self) -> pd.Series:
        """Returns a consistently named pandas.Series containing the label number for each object in the image. Useful as an index for joining different measurements"""
        labels = self.labels
        return pd.Series(
            data=labels,
            index=range(len(labels)),
            name=OBJECT.LABEL,
        )

    def relabel(self):
        """Relabels all the objects based on their connectivity"""
        self._root_image.objmap.relabel()
