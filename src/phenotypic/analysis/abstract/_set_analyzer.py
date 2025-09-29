from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Callable, List

import pandas as pd

if TYPE_CHECKING: from phenotypic import ImageSet


class SetAnalyzer(abc.ABC):

    def __init__(self, on: str, groupby: List[str],
                 agg_func: Callable | str | list | dict|None = 'mean', *, num_workers=1):
        self.groupby = groupby
        self.agg_func = agg_func
        self.on = on
        self.num_workers = num_workers
        self._latest_measurements: pd.DataFrame = pd.DataFrame()

    @abc.abstractmethod
    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def show(self):
        pass

    @abc.abstractmethod
    def results(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def _apply2group_func(group: pd.DataFrame):
        pass
