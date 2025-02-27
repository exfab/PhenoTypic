import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import regionprops_table, regionprops
from ...util.constants import C_ObjectInfo, C_ImageObjects
from typing import Optional


class ImageObjectsSubhandler:
    def __init__(self, handler):
        self._handler = handler

    @property
    def labels(self) -> list:
        """Returns the labels in the image.

        We considered using a simple numpy.unique() call on the object map, but wanted to guarantee that the labels will always be consistent
        with any skimage version outputs.

        """
        return [x.label for x in self.props]

    @property
    def props(self):
        """Returns a skimage.regionprops object for the image. Useful for simple calculations"""
        return regionprops(label_image=self._handler.obj_map[:], intensity_image=self._handler.matrix[:], cache=False)

    def get_object_idx(self, object_label):
        """Returns the index of the object with the given label from a sorted array of object labels."""
        return np.where(self.labels == object_label)[0]

    @property
    def num_objects(self) -> int:
        """Returns the number of objects in the map."""
        return len(self.labels)

    def reset(self):
        self._handler.obj_map.reset()

    # def __getitem__(self, object_label: int):
    #     """Returns a slice of the object image based on the object's label."""
    #     if object_label not in self.labels: raise C_ImageObjects.InvalidObjectLabel(object_label)
    #     return self._handler[self.props[self.get_object_idx(object_label)].slice]

    def __getitem__(self, index: int):
        """Returns a slice of the object image based on the object's index."""
        return self._handler[self.props[index].slice]


    def at(self, idx):
        """Returns a crop of object from the image based on it's idx in a sorted list of labels

        Args:
            idx:
        Returns:
            (Image) The cropped bounding box of an object as an Image
        """
        return self._handler[self.props[idx].slice]


    def get_labels(self, label):
        """Returns a crop of an object from the image based on its label

        Args:
            label: (int) The label number of the object
        :return: (Image)
        """
        raise NotImplementedError  # TODO

    def info(self):
        return pd.DataFrame(
            data=regionprops_table(
                label_image=self._handler.obj_map[:],
                properties=['label', 'centroid', 'bbox']
            )
        ).rename(columns={
            'label': C_ObjectInfo.OBJECT_LABELS,
            'centroid-0': C_ObjectInfo.CENTER_RR,
            'centroid-1': C_ObjectInfo.CENTER_CC,
            'bbox-0': C_ObjectInfo.MIN_RR,
            'bbox-1': C_ObjectInfo.MIN_CC,
            'bbox-2': C_ObjectInfo.MAX_RR,
            'bbox-3': C_ObjectInfo.MAX_CC,
        }
        ).set_index(C_ObjectInfo.OBJECT_LABELS)

    # TODO
    def show_overlay(self):
        raise NotImplementedError
