from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt

from phenotypic import ImageSet
from phenotypic.analysis.abstract import ModelFitter

# TODO
class LogGrowthModel(ModelFitter):

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def show(self)->Tuple[plt.Figure, plt.Axes]:
        pass

    def results(self)->pd.DataFrame:
        return self._latest_measurements

    @staticmethod
    def _model_func():
        pass

    @staticmethod
    def _loss_func():
        pass

    @staticmethod
    def _apply2group_func(group: pd.DataFrame):
        pass
