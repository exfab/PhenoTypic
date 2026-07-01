"""Utilities for documenting and splitting measurement output tables."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, TypeAlias, TypeGuard

import pandas as pd
import polars as pl

from phenotypic.schema import MeasurementInfo


MeasurementFrame: TypeAlias = pd.DataFrame | pl.DataFrame


@dataclass(frozen=True)
class _MeasurementProducer:
    """A public producer class and the ``MeasurementInfo`` enums it owns."""

    output_key: str
    primary_infos: tuple[type[MeasurementInfo], ...]
    shared_infos: tuple[type[MeasurementInfo], ...] = ()


def split_measurements(df: MeasurementFrame) -> dict[str, MeasurementFrame]:
    """Split a measurements table into producer-specific data frames.

    Columns backed by public ``MeasureFeatures`` and ``SetAnalyzer``
    ``MeasurementInfo`` enums define each split. Columns that do not belong to
    any producer-owned enum are preserved in every split as context columns.

    Args:
        df: A pandas or polars measurements DataFrame.

    Returns:
        Mapping of producer class name to a same-type DataFrame containing all
        context columns plus that producer's recognized measurement columns.
        Returns an empty mapping when no producer-owned measurement columns are
        present.

    Raises:
        TypeError: If *df* is not a pandas or polars DataFrame.
    """
    columns = _columns(df)
    groups = _producer_column_groups(columns)
    if not groups:
        return {}

    producer_columns = {
        column for group_columns in groups.values() for column in group_columns
    }
    context_columns = [column for column in columns if column not in producer_columns]

    return {
        output_key: _select_columns(df, context_columns + group_columns)
        for output_key, group_columns in groups.items()
    }


def generate_output_key(df: MeasurementFrame) -> pd.DataFrame:
    """Generate a column-description key for recognized output columns.

    Args:
        df: A pandas or polars measurements DataFrame.

    Returns:
        A pandas DataFrame with ``column_header`` and ``description`` columns,
        preserving input column order and omitting columns not backed by a
        public ``MeasurementInfo`` member (in any header scheme).

    Raises:
        TypeError: If *df* is not a pandas or polars DataFrame.
    """
    records = [
        {"column_header": column, "description": desc}
        for column in _columns(df)
        if (desc := _describe_column(column)) is not None
    ]
    return pd.DataFrame(records, columns=["column_header", "description"])


def _columns(df: MeasurementFrame) -> list[str]:
    """Return DataFrame columns as strings after validating the frame type."""
    if isinstance(df, pd.DataFrame):
        return [str(column) for column in df.columns]
    if isinstance(df, pl.DataFrame):
        return [str(column) for column in df.columns]
    raise TypeError(
        "split_measurements() and generate_output_key() require a pandas "
        f"or polars DataFrame, got {type(df).__name__}."
    )


def _select_columns(df: MeasurementFrame, columns: list[str]) -> MeasurementFrame:
    """Select *columns* while preserving the input DataFrame implementation."""
    if isinstance(df, pd.DataFrame):
        return df.loc[:, columns].copy()
    if isinstance(df, pl.DataFrame):
        return df.select(columns)
    raise TypeError(
        "split_measurements() requires a pandas or polars DataFrame, "
        f"got {type(df).__name__}."
    )


def _producer_column_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    """Map producer class names to their present measurement columns."""
    ordered_columns = list(columns)
    groups: dict[str, list[str]] = {}

    for producer in _discover_measurement_producers():
        present_primary = [
            column
            for column in ordered_columns
            if any(info.owns_header(column) for info in producer.primary_infos)
        ]
        if not present_primary:
            continue

        producer_headers = set(present_primary)
        producer_headers.update(
            column
            for column in ordered_columns
            if any(info.owns_header(column) for info in producer.shared_infos)
        )
        groups[producer.output_key] = [
            column for column in ordered_columns if column in producer_headers
        ]

    return groups


@lru_cache(maxsize=1)
def _discover_measurement_producers() -> tuple[_MeasurementProducer, ...]:
    """Discover public measurement producers from public modules."""
    import phenotypic.analysis as analysis_module
    import phenotypic.measure as measure_module
    from phenotypic.abc_ import MeasureFeatures
    from phenotypic.analysis.abc_ import ModelFitter, QualityCheck, SetAnalyzer
    from phenotypic.schema import MODEL_METRICS

    producers: list[_MeasurementProducer] = []

    for name, cls in inspect.getmembers(measure_module, inspect.isclass):
        if name.startswith("_"):
            continue
        if not issubclass(cls, MeasureFeatures) or cls is MeasureFeatures:
            continue
        infos = _declared_info_classes(cls)
        if infos:
            producers.append(_MeasurementProducer(name, infos))

    for name, cls in inspect.getmembers(analysis_module, inspect.isclass):
        if name.startswith("_"):
            continue
        if not issubclass(cls, SetAnalyzer) or cls in (
            SetAnalyzer,
            ModelFitter,
            QualityCheck,
        ):
            continue
        infos = _declared_info_classes(cls)
        if not infos:
            continue
        if issubclass(cls, ModelFitter):
            primary_infos = tuple(info for info in infos if info is not MODEL_METRICS)
            if primary_infos:
                producers.append(
                    _MeasurementProducer(
                        name,
                        primary_infos,
                        shared_infos=(MODEL_METRICS,),
                    )
                )
        else:
            producers.append(_MeasurementProducer(name, infos))

    return tuple(producers)


def _declared_info_classes(cls: type) -> tuple[type[MeasurementInfo], ...]:
    """Return ``MeasurementInfo`` classes declared on a producer class."""
    infos: list[type[MeasurementInfo]] = []

    single = _as_info_class(
        cls.__dict__.get(
            "_measurement_infoclass",
            getattr(cls, "_measurement_infoclass", None),
        )
    )
    if single is not None:
        infos.append(single)

    plural = cls.__dict__.get(
        "_measurement_infoclasses",
        getattr(cls, "_measurement_infoclasses", ()),
    )
    if isinstance(plural, (list, tuple)):
        for item in plural:
            info = _as_info_class(item)
            if info is not None:
                infos.append(info)

    return tuple(dict.fromkeys(infos))


def _as_info_class(value: object) -> type[MeasurementInfo] | None:
    """Coerce a class/private-attr value into a ``MeasurementInfo`` class."""
    if _is_info_class(value):
        return value
    default = getattr(value, "default", None)
    if _is_info_class(default):
        return default
    return None


def _is_info_class(value: object) -> TypeGuard[type[MeasurementInfo]]:
    """Return whether *value* is a concrete ``MeasurementInfo`` subclass."""
    return (
        isinstance(value, type)
        and issubclass(value, MeasurementInfo)
        and value is not MeasurementInfo
    )


_SANITIZE_TOKEN_RE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _known_categories() -> tuple[str, ...]:
    """All public schema categories, sorted longest-first for prefix matching."""
    import phenotypic.schema as schema

    cats: set[str] = set()
    for name in getattr(schema, "__all__", ()):
        obj = getattr(schema, name, None)
        if not _is_info_class(obj):
            continue
        try:
            cats.add(obj.category())
        except NotImplementedError:  # member-less classification bases
            continue
    return tuple(sorted(cats, key=len, reverse=True))


def _sanitize_token(token: str) -> str:
    return _SANITIZE_TOKEN_RE.sub("", token.strip())


def metric_token(on: str) -> str:
    """Derive the ``<metric>`` header segment from a fitter's ``on`` column.

    Strips the longest known schema **category** prefix if present
    (``Shape_Area`` → ``Area``), else returns the value verbatim
    (``x`` → ``x``); then removes whitespace.
    """
    value = str(on).strip()
    for category in _known_categories():
        if value.startswith(category + "_"):
            return _sanitize_token(value[len(category) + 1:])
    return _sanitize_token(value)


@lru_cache(maxsize=1)
def _public_info_classes() -> tuple[type[MeasurementInfo], ...]:
    """Public, member-ful ``MeasurementInfo`` subclasses exported by schema."""
    import phenotypic.schema as schema

    classes: list[type[MeasurementInfo]] = []
    for name in getattr(schema, "__all__", ()):
        obj = getattr(schema, name, None)
        if _is_info_class(obj) and list(obj):
            classes.append(obj)
    return tuple(classes)


def _describe_column(column: str) -> str | None:
    """Resolve *column* to its member's ``desc`` across all schemes, or None."""
    for info in _public_info_classes():
        member = info.member_for_header(column)
        if member is not None:
            return member.desc
    return None


__all__ = ["generate_output_key", "split_measurements"]
