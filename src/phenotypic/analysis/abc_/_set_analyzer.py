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
        """Row-wise filter by column-value criteria.

        This helper builds a boolean mask across rows using an "AND across columns"
        logic based on a mapping from column names to desired values. It is
        intentionally lightweight and side-effect free (unless ``copy=False``),
        making it convenient to pre-filter measurement tables before grouping or
        aggregation in concrete ``SetAnalyzer`` implementations.

        Matching rules per criterion (for each ``col -> val``):
          - If ``val`` is a scalar (not list-like): keep rows where ``df[col] == val``.
          - If ``val`` is list-like (e.g., list/tuple/set/ndarray): keep rows where
            ``df[col]`` is contained in that collection (``isin`` semantics).
          - If ``val`` is NA and ``match_na=True``: treat NA as a match for NA values in ``df[col]``.
            If ``match_na=False``, NA does not match anything.

        The final mask is the conjunction (logical AND) of every per-column mask.
        If any referenced column is missing, a ``KeyError`` is raised. The function
        may short-circuit and return an empty frame early if intermediate masks
        eliminate all rows.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame to filter.
        criteria : Mapping[str, Any]
            Mapping from column name to either a scalar value or an iterable of
            acceptable values for that column.
        copy : bool, default True
            If True, return a copy of the filtered frame to avoid pandas' view
            warnings. If False, return a view when possible.
        match_na : bool, default False
            Whether NA values provided in ``criteria`` should match NA values in
            the corresponding DataFrame column.

        Returns
        -------
        pandas.DataFrame
            The filtered DataFrame (empty if no rows satisfy all criteria).

        Raises
        ------
        KeyError
            If a column specified in ``criteria`` is not present in ``df``.

        Notes
        -----
        - String values are treated as scalars, not list-like.
        - For list-like criteria, presence of NA in the list only matters when
          ``match_na=True``; in that case, NA in the column is also considered a match.

        Examples
        --------
        Filter by a single scalar value:
        >>> import pandas as pd
        >>> from phenotypic.analysis.abc_._set_analyzer import SetAnalyzer
        >>> data = pd.DataFrame({
        ...     'plate': ['P1', 'P1', 'P2', 'P2'],
        ...     'strain': ['WT', 'KO', 'WT', 'KO'],
        ...     'rep': [1, 1, 2, 2],
        ...     'value': [10.0, 12.5, 9.7, 11.2],
        ... })
        >>> SetAnalyzer._filter_by(data, {'plate': 'P1'})
          plate strain  rep  value
        0    P1     WT    1   10.0
        1    P1     KO    1   12.5

        Filter where a column is in a list of acceptable values:
        >>> SetAnalyzer._filter_by(data, {'strain': ['WT', 'KO'], 'rep': [2]})
          plate strain  rep  value
        2    P2     WT    2    9.7
        3    P2     KO    2   11.2

        Match NA values explicitly:
        >>> data2 = data.copy()
        >>> data2.loc[1, 'strain'] = pd.NA
        >>> SetAnalyzer._filter_by(data2, {'strain': [pd.NA, 'WT']}, match_na=True)
          plate strain  rep  value
        0    P1     WT    1   10.0
        1    P1   <NA>    1   12.5
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
