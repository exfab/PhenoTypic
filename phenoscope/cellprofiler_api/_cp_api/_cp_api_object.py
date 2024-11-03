import logging

formatter = logging.Formatter(fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S')
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

# ----- Module Import -----
import pandas as pd
import numpy as np


from cellprofiler_core.object import Objects
from cellprofiler_core.module.image_segmentation import ImageSegmentation

# ----- Pkg Relative Import -----
from ._cp_api_base import CellProfilerApiBase


# ----- Main Class Definition -----
class CellProfilerApiObject(CellProfilerApiBase):
    def add_object(self, obj_mask:np.ndarray, obj_name:str=None, parent_img_name:str=None):
        if obj_name is None:
            obj_name = self.colony_name
        if parent_img_name is None:
            parent_img_name = self.image_name
        obj = Objects()
        obj.segmented = obj_mask
        
        img = self.img_set.get_image(parent_img_name)
        obj.parent_image = img.parent_image
        self.obj_set.add_objects(obj, f"{obj_name}")
        img_segmentation = ImageSegmentation()
        img_segmentation.add_measurements(workspace=self.workspace, object_name=obj_name)
        