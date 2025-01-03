import pandas as pd
from typing import Optional
import numpy as np

from .cellprofiler_sou.preferences import set_headless, set_awt_headless
from cellprofiler_core.image import ImageSetList, Image as CpImage, ImageSet as CpImageSet
from cellprofiler_core.object import ObjectSet, Objects
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.workspace import Workspace
from cellprofiler_core.module import Module

from cellprofiler_core.module.image_segmentation import ImageSegmentation
from cellprofiler_api.modules.measureobjectsizeshape import MeasureObjectSizeShape

import phenoscope as ps
from phenoscope.interface import FeatureExtractor
from phenoscope.util.exceptions import ValueWarning


class CellProfilerAPI(FeatureExtractor):
    """
    Provides an API to interact with CellProfiler measurement modules.
    """

    def __init__(self):
        self._cellprofiler_image_set_list: Optional[ImageSetList] = None
        self._cellprofiler_image_set: Optional[CpImageSet] = None
        self._cellprofiler_measurements: Optional[Measurements] = None
        self._cellprofiler_object_set: Optional[ObjectSet] = None
        self._cellprofiler_pipeline: Optional[Pipeline] = None
        self._cellprofiler_workspace: Optional[Workspace] = None
        self._cellprofiler_module: Optional[Module] = None

    def extract(self, image: ps.Image, inplace: bool = False) -> pd.DataFrame:
        self._init_cellprofiler()

        measurements = super().extract(image, inplace)
        self._rename_cellprofiler_measurements(measurements)

        self._cleanup_cellprofiler()
        return measurements

    def _init_cellprofiler(self):
        """
        Initialize the cellprofiler_api backend
        :return:
        """
        self._cellprofiler_image_set_list = ImageSetList()

        self._cellprofiler_image_set = (self._cellprofiler_image_set_list
                                        .get_image_set(self._cellprofiler_image_set_list.count()))

        self._cellprofiler_object_set = ObjectSet(can_overwrite=False)

        self._cellprofiler_measurements = Measurements(
            mode='memory',
            multithread=True
        )

        self._cellprofiler_pipeline = Pipeline()
        self._cellprofiler_pipeline.set_needs_headless_extraction(True)
        self._cellprofiler_pipeline.turn_off_batch_mode()

        self._cellprofiler_module = self._init_cellprofiler_module()

        self._cellprofiler_workspace = Workspace(
            pipeline=self._cellprofiler_pipeline,
            module=self._cellprofiler_module,
            image_set=self._cellprofiler_image_set_list,
            object_set=self._cellprofiler_object_set,
            measurements=self._cellprofiler_measurements,
            image_set_list=self._cellprofiler_image_set_list,
        )

    def _cleanup_cellprofiler(self):
        """
        Deletes and closes cellprofiler_api backend.
        :return:
        """
        self._cellprofiler_workspace.close()

        del self._cellprofiler_image_set_list
        self._cellprofiler_image_set_list = None

        del self._cellprofiler_image_set
        self._cellprofiler_image_set = None

        del self._cellprofiler_object_set
        self._cellprofiler_object_set = None

        del self._cellprofiler_measurements
        self._cellprofiler_measurements = None

        del self._cellprofiler_pipeline
        self._cellprofiler_pipeline=None

        del self._cellprofiler_module
        self._cellprofiler_module=None

        del self._cellprofiler_workspace
        self._cellprofiler_workspace=None


    def _init_cellprofiler_module(self) -> Module:
        """
        The interface method for integrating cellprofiler_api modules. Define the module settings from cellprofiler_api here.
        Add necessary parameters to the init constructor.
        :return: (cellprofile_core.module.Module)
        """
        raise NotImplementedError()

    @staticmethod
    def _rename_cellprofiler_measurements(cellprofiler_table:pd.DataFrame)->pd.DataFrame:
        """
        This is run after collecting cellprofiler_api measurements. This allows the user to rename the measurement results.
        :return: (pd.DataFrame) A cellprofiler_api table with columns renamed
        """

    def _PsImage2CpImage(self, image: ps.Image) -> CpImage:
        """
        Cellprofiler only knows how to handle 2D-Matrix
        :param image:
        :return:
        """
        return CpImage(image.matrix)
