from __future__ import annotations

import abc
from abc import ABC
from typing import TYPE_CHECKING, Callable, List

import pandas as pd

if TYPE_CHECKING: pass

from ._set_analyzer import SetAnalyzer


class ModelFitter(SetAnalyzer, ABC):

    def __init__(self, on: str, groupby: List[str],
                 agg_func: Callable | str | list | dict | None = 'mean',
                 *,
                 num_workers: int = 1, ):
        super().__init__(on=on, groupby=groupby, agg_func=agg_func, num_workers=num_workers)
        self._latest_model_scores: pd.DataFrame = pd.DataFrame()

    @staticmethod
    @abc.abstractmethod
    def model_func():
        """The mathematical model that should be implemented. The first parameter should be the independent variable
        such as time"""
        pass

    @staticmethod
    @abc.abstractmethod
    def _loss_func():
        """The loss function that should be implemented for linear least squares fitting"""
        pass
