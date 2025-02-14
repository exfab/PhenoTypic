import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import regionprops_table
from ...util.constants import C_ObjectInfo
from typing import Optional
class ImageObjects:
    def __init__(self, handler):
        self._handler = handler

    def get(self, idx):
        """Returns a crop of object from the image based on it's idx in a sorted list of labels

        Args:
            idx:
        Returns:
            (Image) The cropped bounding box of an object as an Image
        """
        raise NotImplementedError    # TODO

    def get_label(self, label):
        """Returns a crop of an object from the image based on its label

        Args:
            label: (int) The label number of the object
        :return: (Image)
        """
        raise NotImplementedError # TODO

    def info(self):
        return pd.DataFrame(
            data=regionprops_table(
                label_image=self._handler.object_map[:],
                properties=['label','centroid','bbox']
            )
        ).rename(columns={
            'label':C_ObjectInfo.OBJECT_MAP_ID,
            'centroid-0':C_ObjectInfo.CENTER_RR,
            'centroid-1':C_ObjectInfo.CENTER_CC,
            'bbox-0':C_ObjectInfo.MIN_RR,
            'bbox-1':C_ObjectInfo.MIN_CC,
            'bbox-2':C_ObjectInfo.MAX_RR,
            'bbox-3':C_ObjectInfo.MAX_CC,
        }).set_index(C_ObjectInfo.OBJECT_MAP_ID)

    #TODO
    def show_overlay(self):
        raise NotImplementedError




