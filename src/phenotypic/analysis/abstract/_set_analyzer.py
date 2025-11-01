from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Callable, List

import pandas as pd
from collections.abc import Iterable
from typing import Any, Mapping

if TYPE_CHECKING: from phenotypic import ImageSet


class SetAnalyzer(abc.ABC):

    def __init__(self, on: str, groupby: List[str],
                 agg_func: Callable | str | list | dict | None = 'mean', *, num_workers=1):
        self.groupby = groupby
        self.agg_func = agg_func
        self.on = on
        self.n_jobs = num_workers
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

    @staticmethod
    def _filter_by(df: pd.DataFrame,
                   criteria: Mapping[str, Any],
                   *,
                   copy: bool = True,
                   match_na: bool = False) -> pd.DataFrame:
        """Filter a DataFrame by column->values mapping.

        For each column in `criteria`:
          • If the value is scalar, keep nrows where column == value.
          • If the value is an iterable (e.g., list/tuple/set/ndarray), keep nrows where column ∈ values.
          • If the value is NA and `match_na=True`, match NA in that column.

        Args:
            df: Input DataFrame.
            criteria: Dict of {column_name: value_or_iterable_of_values}.
            copy: Return a copy to avoid view warnings.
            match_na: If True, NA in `criteria` matches NA in the column.

        Returns:
            Filtered DataFrame.
        """

        def _is_list_like(x: Any) -> bool:
            return isinstance(x, Iterable) and not isinstance(x, (str, bytes))

        mask = pd.Series(True, index=df.index)
        for col, val in criteria.items():
            if col not in df.columns:
                raise KeyError(f"Column not found: {col}")

            s = df[col]
            if _is_list_like(val):
                vals = list(val)
                part = s.isin(vals)
                if match_na and any(pd.isna(v) for v in vals):
                    part = part | s.isna()
            else:
                if pd.isna(val):
                    part = s.isna() if match_na else pd.Series(False, index=s.index)
                else:
                    part = s.eq(val)

            mask &= part

            # Short-circuit if empty
            if not mask.any():
                return df.iloc[0:0].copy() if copy else df.iloc[0:0]

        out = df[mask]
        return out.copy() if copy else out
