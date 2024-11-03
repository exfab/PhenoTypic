import logging

formatter = logging.Formatter(fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
                              datefmt='%m/%d/%Y %I:%M:%S')
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

import pandas as pd
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

from typing import List

# ----- Pkg Relative Import -----
from ._cp_api_measure_texture import CellProfilerApiMeasureTexture

# ----- Global Constants -----
METRIC_LABEL = "Metric"


# ----- Main Class Definition -----


class CellProfilerApiAnalysis(CellProfilerApiMeasureTexture):
    def __init__(self):
        super().__init__()
        self.results_table = None

    def run(self, colony_profile, **kwargs):
        Results = namedtuple("Results", ["results", "status_validity"])
        self._set_name(colony_profile.plate_name)
        self._set_img(colony_profile.input_img)
        self.add_object(np.copy(colony_profile.colony_mask), colony_profile.colony_name, colony_profile.plate_name)
        self.status_validity = colony_profile.status_validity

        if self.status_validity:
            self.measure_areashape(colony_profile.colony_name,
                                   calculate_adv=kwargs.get("calculate_adv", False),
                                   calculate_zernike=kwargs.get("calculate_zernike", False)
                                   )

        if self.status_validity:
            self.measure_intensity(colony_profile.colony_name)

        if self.status_validity:
            self.measure_texture(colony_profile.colony_name,
                                 texture_scale=kwargs.get("texture_scale", None),
                                 gray_levels=kwargs.get("gray_levels", 256)
                                 )

        try:
            self.pipeline.end_run()
            self._compile_results()
            results = Results(self.results_table.copy(),
                              self.status_validity)
            return results
        except:
            log.warning(f"Could not compile results for {colony_profile.plate_name}")

    def get_results(self):
        # Deprecated when module was converted to an API instead of integration
        return self.results_table.copy()

    def _compile_results(self):
        validity = pd.DataFrame({
            f"{self.colony_name}": self.status_validity,
        }, index=["status_valid_analysis"]
        ).astype(int)
        self.results_table = pd.concat(
            self.results.values(), axis=0
        )
        self.results_table = pd.concat([self.results_table, validity], axis=0)
