from __future__ import annotations

import abc
from typing import Callable

import pandas as pd
import numpy as np
from collections.abc import Iterable
from typing import Any, Mapping

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
)

from phenotypic.sdk_ import (
    ColumnRef,
    ColumnRefList,
    ensure_metadata_prefix,
    is_metadata_header,
    metadata_member_for_header,
    metadata_member_for_label,
    normalize_metadata_columns,
)
from phenotypic.sdk_._docstring_params import apply_docstring_descriptions


def normalize_measurement_metadata_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with only metadata-family columns normalized.

    ``normalize_metadata_columns`` is intentionally for external metadata
    tables, where a bare column is metadata by definition. Measurement tables
    additionally carry columns such as ``Size_Area`` and ``Object_Label``;
    this adapter isolates metadata columns before delegating, so those columns
    are never reclassified or renamed. The caller's frame is not mutated.
    """
    source_names = [str(column) for column in frame.columns]
    metadata_positions = [
        position
        for position, name in enumerate(source_names)
        if (
            is_metadata_header(name)
            or metadata_member_for_header(name) is not None
            or metadata_member_for_label(name) is not None
        )
    ]
    if not metadata_positions:
        return frame.copy(deep=True)

    metadata = normalize_metadata_columns(frame.iloc[:, metadata_positions])
    targets = [ensure_metadata_prefix(source_names[position]) for position in metadata_positions]
    groups: dict[str, list[int]] = {}
    for position, target in zip(metadata_positions, targets, strict=True):
        groups.setdefault(target, []).append(position)
    anchors = {
        target: next(
            (position for position in positions if source_names[position] == target),
            positions[0],
        )
        for target, positions in groups.items()
    }
    normalized_by_anchor = {
        anchor: metadata.iloc[:, index]
        for index, anchor in enumerate(sorted(anchors.values()))
    }
    consumed = {position for positions in groups.values() for position in positions}
    columns = [
        normalized_by_anchor[position]
        if position in normalized_by_anchor
        else frame.iloc[:, position].copy(deep=True)
        for position in range(len(source_names))
        if position not in consumed or position in normalized_by_anchor
    ]
    return pd.concat(columns, axis=1)


def normalize_metadata_column_reference(column: str) -> str:
    """Resolve a known metadata reference without reclassifying measurements."""
    name = str(column)
    if (
        is_metadata_header(name)
        or metadata_member_for_header(name) is not None
        or metadata_member_for_label(name) is not None
    ):
        return ensure_metadata_prefix(name)
    return name


def normalize_metadata_column_references(value: Any) -> list[str]:
    """Normalize a user-supplied column-reference list for Pydantic fields.

    A lone string is accepted as a one-element list for historical ``groupby``
    ergonomics. ``None`` is intentionally not accepted here: optional callers
    must handle it before calling this function, while required Pydantic fields
    receive a clear validation error instead of iterating ``None`` or splitting
    a string into characters.
    """
    if isinstance(value, str):
        return [normalize_metadata_column_reference(value)]
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("column-reference arrays must be one-dimensional")
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("column references must be a string, list, tuple, or 1-D array")
    if not all(isinstance(column, str) for column in value):
        raise ValueError("every column reference must be a string")
    return [normalize_metadata_column_reference(column) for column in value]


def normalize_metadata_criteria(criteria: Mapping[str, Any]) -> dict[str, Any]:
    """Return criteria with known metadata keys normalized and collision-checked."""
    normalized: dict[str, Any] = {}
    source_for_target: dict[str, str] = {}
    for source, value in criteria.items():
        if not isinstance(source, str):
            raise ValueError("criteria keys must be strings")
        target = normalize_metadata_column_reference(source)
        previous = source_for_target.get(target)
        if previous is not None and previous != source:
            raise ValueError(
                "criteria keys normalize to the same metadata column: "
                f"{previous!r} and {source!r} -> {target!r}"
            )
        normalized[target] = value
        source_for_target[target] = source
    return normalized


class SetAnalyzer(BaseModel, abc.ABC):
    """Abstract base for grouped analyses over a measurement DataFrame.

    ``SetAnalyzer`` is the root of PhenoTypic's analyzer hierarchy. It is a
    separate pydantic ``BaseModel`` root — it is **not** a
    :class:`~phenotypic.abc_.BaseOperation`. Subclasses operate on an
    already-assembled measurement DataFrame via :meth:`analyze` (analyzers
    use ``.analyze()``, not ``.apply()``).

    Analyzer parameters are declared as annotated class-level fields;
    pydantic generates the constructor and validates inputs.

    Args:
        on: Measurement column the analysis operates on.
        groupby: Columns that define the per-group iteration unit.
        agg_func: Aggregation applied within each group. A ``Callable`` is
            accepted at runtime but cannot round-trip through JSON; only
            the ``str``/``list``/``dict`` forms serialize losslessly.
            Defaults to ``"mean"``.
        n_jobs: Worker count for parallel group processing. The legacy
            ``num_workers`` keyword is accepted as an alias. Defaults to 1.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    on: ColumnRef
    groupby: ColumnRefList
    agg_func: Callable | str | list | dict | None = "mean"
    n_jobs: int = Field(
        default=1,
        validation_alias=AliasChoices("n_jobs", "num_workers"),
    )

    @field_validator("on", mode="before")
    @classmethod
    def _normalize_on_reference(cls, value: str) -> str:
        """Accept legacy, flat, and bare known metadata column references."""
        if not isinstance(value, str):
            raise ValueError("on must be a string column reference")
        return normalize_metadata_column_reference(value)

    @field_validator("groupby", mode="before")
    @classmethod
    def _normalize_groupby_references(cls, value: Any) -> Any:
        """Accept legacy, flat, and bare known metadata grouping references."""
        return normalize_metadata_column_references(value)

    @field_validator(
        "time_label",
        "subject_label",
        "rater_label",
        "stderr_label",
        "Kmax_label",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def _normalize_optional_column_reference(cls, value: Any) -> Any:
        """Normalize metadata-capable subclass column-reference fields."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("column reference must be a string or None")
        return normalize_metadata_column_reference(value)

    @field_validator("s0_prior_groupby", mode="before", check_fields=False)
    @classmethod
    def _normalize_optional_groupby_references(cls, value: Any) -> Any:
        """Normalize the optional inoculum-prior grouping list."""
        if value is None:
            return None
        return normalize_metadata_column_references(value)

    _latest_measurements: pd.DataFrame = PrivateAttr(
        default_factory=pd.DataFrame
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Populate field descriptions from the subclass docstring.

        Runs once per concrete subclass after pydantic has built its
        model, copying parameter descriptions parsed from the Google-style
        ``Args:`` docstring block onto each field's ``description`` slot.

        Args:
            **kwargs: Class-keyword arguments forwarded by pydantic.
        """
        super().__pydantic_init_subclass__(**kwargs)
        apply_docstring_descriptions(cls)

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
    def _apply2group_func(group: pd.DataFrame, **kwargs):
        pass

    @staticmethod
    def _filter_by(
        df: pd.DataFrame,
        criteria: Mapping[str, Any],
        *,
        copy: bool = True,
        match_na: bool = False,
    ) -> pd.DataFrame:
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

        Match NA values explicitly. With ``match_na=True`` the criterion
        ``[pd.NA, 'WT']`` keeps every ``'WT'`` row *and* every NA row, so
        all three matching rows are returned:
        >>> data2 = data.copy()
        >>> data2.loc[1, 'strain'] = pd.NA
        >>> SetAnalyzer._filter_by(data2, {'strain': [pd.NA, 'WT']}, match_na=True)
          plate strain  rep  value
        0    P1     WT    1   10.0
        1    P1   <NA>    1   12.5
        2    P2     WT    2    9.7
        """

        criteria = normalize_metadata_criteria(criteria)

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

    @staticmethod
    def _ensure_float_array(arr):
        """
        Detects dtype and converts string-numeric or mixed arrays to float.
        Leaves numeric arrays unchanged.
        """
        k = arr.dtype.kind

        # Already numeric
        if k in {"i", "u", "f", "c"}:
            return arr.astype(float)

        # String or object with strings
        if k in {"U", "S", "O"}:
            return SetAnalyzer.__smart_float_convert(arr)

        raise TypeError(f"Unsupported array dtype: {arr.dtype}")

    @staticmethod
    def __smart_float_convert(arr):
        out = []
        for x in arr:
            if x is None:
                out.append(np.nan)
                continue
            try:
                out.append(float(str(x).replace(",", "").strip()))
            except ValueError:
                raise ValueError(f"Value '{x}' cannot be converted to float")
        return np.array(out, dtype=float)
