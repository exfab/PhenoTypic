from typing import Optional, Tuple, Dict

import pandas as pd
from skimage.measure import regionprops_table
from skimage.transform import resize

from phenoscope import Image
from phenoscope.grid import GriddedImage
from phenoscope.interface import FeatureExtractor

LABEL_METADATA_GRIDNUM = 'Grid_SectionNum'
LABEL_METADATA_OBJECT_LABEL = 'Object_Label'
LABEL_PARENT_IMAGE = 'Metadata_ParentImageName'
class ObjectImageExtractor(FeatureExtractor):
    def __init__(
            self, target_dimensions: Optional[Tuple] = None, use_color=False, mode: str = 'constant', cval: float = 0,
            anti_aliasing: bool = True, order: Optional[int] = None, anti_aliasing_sigma: Optional[float] = None,
    ):
        """

        :param target_dimensions: (Optional[Tuple]) If None, the object image arrays are extracted as they appear. If a tuple with length 2 is defined the array height and width is resized
        """
        self.target_dimensions = target_dimensions
        self.use_color = use_color
        self.mode = mode
        self.cval = cval
        self.anti_aliasing = anti_aliasing
        self.order = order
        self.anti_aliasing_sigma = anti_aliasing_sigma

    def _operate(self, image: Image) -> Dict[int, Image]:
        """
        Extract the object bounding boxes into a flattened array that can be reshaped. The first two columns of the dataframe are the height and width of the original image allowing it to be reshaped.
        :param image:
        :return: A table containing the details of the object's bounding box and the necessary info to reconstruct the image from a flattened array.
        """
        # Get the object image slices
        obj_slices = pd.DataFrame(regionprops_table(image.object_map, properties=['label', 'slice'])).set_index('label')

        # Create dict for flattened image arrays
        object_images = {}
        if isinstance(image, GriddedImage):
            grid_info = image.grid_info

        for label in obj_slices.index:

            # Extract object image as new Image
            curr_obj_img = image[obj_slices.loc[label, 'slice']]

            # TODO: Add hybrid resizing
            new_img = curr_obj_img.copy()
            if self.target_dimensions is not None:
                if self.use_color:
                    new_img.matrix = self._resize_color_array(curr_obj_img)

                else:
                    new_img.matrix = self._resize_array(curr_obj_img)

                if curr_obj_img.object_mask is not None and curr_obj_img.object_map is not None:
                    new_img.object_mask = self._resize_object_mask(curr_obj_img)

                    new_img.object_map = new_img.object_mask
                    new_img.object_map[new_img.object_mask] = label

            new_img.name = f'{image.name}_Obj{label}'
            new_img.set_metadata(key=LABEL_METADATA_OBJECT_LABEL, value=label)
            new_img.set_metadata(key=LABEL_PARENT_IMAGE, value=image.name)
            if isinstance(image, GriddedImage):
                new_img.set_metadata(key=LABEL_METADATA_GRIDNUM, value=int(grid_info.loc[label, "Grid_SectionNum"]))

            object_images[label] = new_img

        return object_images

    def _resize_color_array(self, image):
        return resize(image=image.matrix,
                      output_shape=self.target_dimensions,
                      anti_aliasing=self.anti_aliasing,
                      order=self.order,
                      anti_aliasing_sigma=self.anti_aliasing_sigma)

    def _resize_array(self, image):
        return resize(image=image.matrix, output_shape=self.target_dimensions,
                      anti_aliasing=self.anti_aliasing,
                      order=self.order,
                      anti_aliasing_sigma=self.anti_aliasing_sigma)

    def _resize_object_mask(self, image):
        return resize(image=image.object_mask, output_shape=self.target_dimensions,
                      anti_aliasing=False,
                      order=self.order,
                      anti_aliasing_sigma=self.anti_aliasing_sigma)
