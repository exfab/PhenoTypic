# ----- Imports -----
from skimage.color import rgb2gray
import pandas as pd

from cellprofiler_core.preferences import set_headless
from cellprofiler_core.image import ImageSetList, Image
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace
from cellprofiler_core.module import Module

import os
import logging

logger_name = "phenomics-cellprofiler_api"
log = logging.getLogger(logger_name)
logging.basicConfig(format=f'[%(asctime)s|%(levelname)s|{os.path.basename(__file__)}] %(message)s')

# ----- Pkg Relative Import -----

# ----- Main Class Definition -----
set_headless()


class CellProfilerApiBase:
    def __init__(self):
        self.input_img = None
        self.image_name = None

        self.cp_img = None
        self.obj_set = None
        self.cpc_measurements = None
        self.pipeline = None
        self.workspace = None
        self.results_table = None

        self.status_validity = True
        self.keys = {}
        self.results = {}
        self.img_set_list = ImageSetList()
        self.img_set_list_idx = self.img_set_list.count()
        self.img_set = self.img_set_list.get_image_set(
            self.img_set_list_idx
        )
        self._init_workspace()

    @property
    def gray_img(self):
        if len(self.input_img.shape)==3:
            return rgb2gray(self.input_img)
        elif len(self.input_img.shape)==2:
            return self.input_img
        else:
            raise ValueError('Input image is not grayscale or RGB')

    @property
    def colony_name(self):
        return f"{self.image_name}_Colony"

    def run(self, img, name):
        self._set_name(name)
        self._set_img(img)

    def refresh(self):
        self._init_workspace()

    def add_img(self, gray_img, name):
        if len(gray_img.shape)==3:
            gray_img = rgb2gray(gray_img)
        cp_img = Image(gray_img)
        self.img_set.add(name, cp_img)

    def _set_name(self, image_name):
        self.image_name = image_name
        self.status_validity = True

    def _set_img(self, well_img):
        self.input_img = well_img

        self.cp_img = Image(self.gray_img)
        self.img_set.add(self.image_name, self.cp_img)

    def _init_workspace(self):
        '''
        ----- Objective -----
        Initialize CellProfiler Workspace and Identifies Primary Object in Images. This has to be done first before running other modules
        '''

        self.obj_set = ObjectSet(can_overwrite=True)  # Results work when true, mask segmentation does not

        self.cpc_measurements = Measurements(
            mode="memory",
            multithread=True
        )

        self.pipeline = Pipeline()
        self.pipeline.set_needs_headless_extraction(True)
        self.pipeline.turn_off_batch_mode()

        self.workspace = Workspace(
            self.pipeline,
            Module(),
            self.img_set,
            self.obj_set,
            self.cpc_measurements,
            self.img_set_list
        )

    def _get_feature_keys(self, obj_name, module):
        cpc_metric_cols = pd.DataFrame(
            module.get_measurement_columns(self.pipeline),
            columns=["source", "keys", "dtype"]
        )
        keys = cpc_metric_cols[cpc_metric_cols["source"]==obj_name]
        return keys.loc[:, "keys"]

    def _get_results(self, obj_name, feature_keys):
        results = []
        for key in feature_keys:
            curr_result = self.cpc_measurements.get_measurement(
                obj_name,
                key
            )
            assert (len(curr_result)==1), "Too many objects"
            results.append(curr_result[0])
        # results = pd.DataFrame(
        #     list(zip(feature_keys, results)),
        #     columns=["Metric", obj_name]
        # ).set_index("Metric").copy(deep=True)
        results = pd.Series(
                data=results,
                index=feature_keys,
                name=obj_name
        )
        results.index.name="Metric"
        log.debug(f"Got results for {obj_name} resulting shape: {results.shape}")
        return results

    def _update_results(self, metric_name: str, data):
        self.results[metric_name] = data.copy()
