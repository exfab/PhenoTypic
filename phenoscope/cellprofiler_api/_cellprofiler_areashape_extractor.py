import numpy as np

from cellprofiler_core.preferences import set_headless
from cellprofiler_core.image import ImageSetList, Image as CpImage
from cellprofiler_core.object import ObjectSet, Objects
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace
from cellprofiler_core.module import Module

from cellprofiler_core.module.image_segmentation import ImageSegmentation
from cellprofiler.modules.measureobjectsizeshape import MeasureObjectSizeShape

import pandas as pd
from typing import Optional
import numpy as np

from phenoscope import Image
from phenoscope.interface import FeatureExtractor
from phenoscope.util.exceptions import ValueWarning


class CellProfilerAreaShape(FeatureExtractor):
    def __init__(
            self, calculate_adv: bool = False, calculate_zernikes: bool = False, max_object_num: Optional[int] = None,
            exc_type: Optional[str] = 'error'
    ):
        """

        :param calculate_adv:
        :param calculate_zernikes:
        :param max_object_num: (Optional[int]) This sets a max number of objects to be analyzed in the image. This can be useful for plate colony analysis, where you want to be sure which object in the image you are measuring
        :param exc_type: (Optional[str]) This sets which type of exception is raised in the event that there are more objects in the image than the max. This should be either None, 'error', or 'warning
        """
        self.calculate_adv: bool = calculate_adv
        self.calculate_zernikes: bool = calculate_zernikes
        self.max_object_num: int = max_object_num
        self.exc_type: Optional[str] = exc_type

    def _operate(self, image: Image) -> pd.DataFrame:
        # Initialize CellProfiler in Non-GUI mode
        set_headless()

        # Create CellProfiler Workspace Objects
        img_set_list = ImageSetList()
        img_set_list_idx = img_set_list.count()
        img_set = img_set_list.get_image_set(img_set_list_idx)

        # Create a cellprofiler object set
        cpc_obj_set = ObjectSet(can_overwrite=False)

        # Create a CellProfiler Measurement object
        cpc_measurements = Measurements(
                mode='memory',
                multithread=True
        )

        # Create a CellProfiler Pipeline object
        pipeline = Pipeline()
        pipeline.set_needs_headless_extraction(True)
        pipeline.turn_off_batch_mode()

        # Integrate into workspace
        workspace = Workspace(
                pipeline,
                Module(),
                img_set,
                cpc_obj_set,
                cpc_measurements,
                img_set_list

        )

        # Convert the phenoscope Image to a CellProfiler Image
        cp_img = CpImage(image.array)
        img_set.add(image.name, cp_img)

        map_labels = np.unique(image.object_map)
        map_labels = map_labels[np.nonzero(map_labels)]

        # Check to make sure the number of items in the image is expected
        if self.max_object_num is not None and len(map_labels) > self.max_object_num:
            if self.exc_type == 'error':
                raise ValueError(
                        'There are more objects in the image being inputted into CellProfiler than the user-defined max amount')
            elif self.exc_type == 'warning':
                raise ValueWarning(
                        'There are more objects in the image being inputted into CellProfiler than the user-defined max amount')
            elif self.exc_type is None:
                pass
            else:
                raise ValueError('Unknown exception type')

        map_results = []
        for label in map_labels:

            # Create an object map that only contains the specfied label
            obj_map = image.object_map
            obj_map[obj_map != label] = 0

            # DEPRECATED in favor of naming consistency:
            # obj_name = f'ImgName({image.name})_ObjId({label:02})_CpObjects'
            obj_name = f'CpObj_{label}'

            # Create a CellProfiler Objects obj that describes the objects in a CellProfiler Image
            cp_obj = Objects()
            cp_obj.segmented = obj_map
            cp_obj.parent_image = cp_img.parent_image
            cpc_obj_set.add_objects(cp_obj, obj_name)

            # Add segmentation as a measurement to the workspace
            img_segmentation = ImageSegmentation()
            img_segmentation.add_measurements(workspace=workspace, object_name=obj_name)

            # Generate target module and set the necessary values
            # ----- Module Implementation -----
            mod = MeasureObjectSizeShape()
            mod.calculate_advanced.value = self.calculate_adv
            mod.calculate_zernikes.value = self.calculate_zernikes
            mod.objects_list.value = obj_name

            # Execute measurements
            pipeline.run_module(mod, workspace)

            # Get all the metrics labels generated as feature keys
            cpc_metric_info = pd.DataFrame(mod.get_measurement_columns(pipeline), columns=['source', 'keys', 'dtype'])
            keys = cpc_metric_info.loc[cpc_metric_info.loc[:, 'source'] == obj_name, 'keys']

            # Get the measurement results
            obj_results = {}
            for key in keys:
                curr_result = np.array(cpc_measurements.get_measurement(obj_name, key))
                curr_result = curr_result[np.nonzero(curr_result)]

                obj_results[key] = curr_result
            obj_results = pd.Series(data=obj_results, index=keys, name=obj_name)
            obj_results.index.name = 'Metric'
            map_results.append(obj_results)

        # Close CellProfiler API (IMPORTANT!!!)
        workspace.close()

        # Compile results
        map_results = pd.concat(map_results, axis=1).T

        # Check Integrity (Only needed if images_list was a module parameter)
        if hasattr(mod, 'images_list') and not all(image.name in col for col in map_results.columns):
            raise RuntimeError("The measurement's data integrity could not be guaranteed due to an unknown issue.")

        # Remove Image Name since we only needed it for the check
        map_results.columns = map_results.columns.str.replace(f'_{image.name}', '', regex=False)

        # Remove CpObj from labels
        map_results.index.name = 'label'
        map_results.index = map_results.index.str.replace('CpObj_', '', regex=False)

        # Results are currently within a numpy. This function will extract the values, while checking that there was only one value in the array
        def extract_value_from_embed_arr(element):
            if isinstance(element, np.ndarray):
                # Check for multiple values
                if len(element) > 1: raise RuntimeWarning(
                    f'CellProfiler result returned more than one object, but only the first one will be kept. Review data for {image.name}.')

                if len(element) == 0:
                    return np.nan
                else:
                    return element[0]

            else:
                return element  # Catch all else conditions. Defer handling to user.

        map_results = map_results.map(lambda x: extract_value_from_embed_arr(x))
        map_results.columns.name=''

        # Drop nan columns (usually stem from using grayscale vs rgb image)
        map_results = map_results.dropna(axis=1, how='all')

        return map_results
