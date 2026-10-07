"""Utilities for documenting and splitting measurement output tables."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Iterator, Protocol, TypeAlias, TypeGuard

import pandas as pd
import polars as pl

from phenotypic.schema import CATEGORIES, MeasurementInfo


MeasurementFrame: TypeAlias = pd.DataFrame | pl.DataFrame


class _TableProducer(Protocol):
    """What the registry needs from a producer class: its header emitter."""

    @classmethod
    def output_header(cls, member: MeasurementInfo, on: str | None = None) -> str: ...

    @classmethod
    def output_header_placeholders(cls) -> dict[str, str]: ...


@dataclass(frozen=True)
class MeasurementProducer:
    """A public operation that writes a table, and the schemas of its columns.

    Attributes:
        output_key: The producer's class name; also its file stem under
            ``deliverables/measurements_by_feature/``.
        producer: The operation class (a ``MeasureFeatures`` or
            ``SetAnalyzer`` subclass).
        primary_infos: Schemas this producer owns, in declaration order,
            including any a parameter switches on (``MeasureColor``'s XYZ).
        shared_infos: Schemas every producer of its kind writes alongside its
            own (``MODEL_METRICS`` for growth models, ``QUALITY_CHECK`` for
            quality checks).
    """

    output_key: str
    producer: type[_TableProducer]
    primary_infos: tuple[type[MeasurementInfo], ...]
    shared_infos: tuple[type[MeasurementInfo], ...] = ()

    def output_header(self, member: MeasurementInfo, on: str | None = None) -> str:
        """Return the column header the producer writes for *member*.

        Delegates to the producer class's ``output_header``. Analyzers whose
        headers differ from the enum value (quality checks, growth models,
        edge correction) call that same method when they write their columns.
        Measurement operations write ``member.value`` or an enum helper
        (``TEXTURE.header``) directly, so for them agreement is not structural:
        ``tests/unit/util/test_output_headers.py`` runs every producer and
        checks the documented names against the columns written.

        Args:
            member: A member of one of this producer's schemas.
            on: The analyzed column, for producers whose headers embed it;
                ``None`` renders a placeholder such as ``<metric>``.

        Returns:
            The emitted header.
        """
        return self.producer.output_header(member, on)


def measurement_producers() -> tuple[MeasurementProducer, ...]:
    """Return every public operation that writes a table of schema columns.

    Covers the public ``phenotypic.measure`` operations and the public
    ``phenotypic.analysis`` analyzers that declare a ``MeasurementInfo``
    schema. Analyzers that only remove rows (the outlier removers) declare
    none and are not listed. The Measurements reference, the
    ``measurements_by_feature/`` split and the deliverables README all read
    this registry.

    Returns:
        Producers in discovery order: measurement operations, then analyzers,
        each alphabetical.

    Examples:
        >>> from phenotypic.util import measurement_producers
        >>> size = next(p for p in measurement_producers() if p.output_key == "MeasureSize")
        >>> [info.__name__ for info in size.primary_infos]
        ['SIZE']
    """
    return _discover_measurement_producers()


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
    return _split_by_groups(df, _context_columns(columns, groups), groups)


def split_measurements_by_category(df: MeasurementFrame) -> dict[str, MeasurementFrame]:
    """Split a measurements table into one data frame per measurement category.

    Context columns are the same as :func:`split_measurements`: every column
    not owned by a producer's ``MeasurementInfo`` (metadata, object label,
    grid, joined external metadata). Each category frame holds those context
    columns followed by the present columns whose member carries the category
    (see :class:`phenotypic.schema.CATEGORIES`), in input order. Measurement
    columns outside a category are dropped from its frame. A column in several
    categories appears in each.

    Args:
        df: A pandas or polars measurements DataFrame.

    Returns:
        Mapping of category label (e.g. ``"StartingMetrics"``) to a same-type
        DataFrame. Categories with no present columns are omitted.

    Raises:
        TypeError: If *df* is not a pandas or polars DataFrame.
    """
    columns = _columns(df)
    producer_groups = _producer_column_groups(columns)
    if not producer_groups:
        return {}
    context = _context_columns(columns, producer_groups)
    context_set = set(context)
    measured = [column for column in columns if column not in context_set]
    return _split_by_groups(df, context, _category_column_groups(measured))


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
        "split_measurements(), split_measurements_by_category() and "
        "generate_output_key() require a pandas or polars DataFrame, "
        f"got {type(df).__name__}."
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


def _context_columns(columns: list[str], groups: dict[str, list[str]]) -> list[str]:
    """Columns not claimed by any producer group, in input order."""
    owned = {column for group_columns in groups.values() for column in group_columns}
    return [column for column in columns if column not in owned]


def _split_by_groups(
    df: MeasurementFrame,
    context: list[str],
    groups: dict[str, list[str]],
) -> dict[str, MeasurementFrame]:
    """Select ``context + group`` columns for every group, preserving frame type."""
    return {key: _select_columns(df, context + cols) for key, cols in groups.items()}


def _category_column_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    """Map category labels to the *columns* whose member carries that category.

    Keys follow ``CATEGORIES`` declaration order; values follow *columns* order.
    Reads the module-level ``CATEGORIES`` at call time.
    """
    by_category: dict[CATEGORIES, list[str]] = {category: [] for category in CATEGORIES}
    for column in columns:
        member = _member_for_column(column)
        if member is None:
            continue
        for category in CATEGORIES.in_order(member.categories):
            by_category[category].append(column)
    return {category.label: cols for category, cols in by_category.items() if cols}


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
def _discover_measurement_producers() -> tuple[MeasurementProducer, ...]:
    """Discover public measurement producers from public modules."""
    import phenotypic.analysis as analysis_module
    import phenotypic.measure as measure_module
    from phenotypic.abc_ import MeasureFeatures
    from phenotypic.analysis.abc_ import ModelFitter, QualityCheck, SetAnalyzer
    from phenotypic.schema import MODEL_METRICS, QUALITY_CHECK

    producers: list[MeasurementProducer] = []

    for name, cls in inspect.getmembers(measure_module, inspect.isclass):
        if name.startswith("_"):
            continue
        if not issubclass(cls, MeasureFeatures) or cls is MeasureFeatures:
            continue
        infos = _declared_info_classes(cls)
        if infos:
            producers.append(MeasurementProducer(name, cls, infos))

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
                    MeasurementProducer(
                        name,
                        cls,
                        primary_infos,
                        shared_infos=(MODEL_METRICS,),
                    )
                )
        elif issubclass(cls, QualityCheck):
            # The shared QC trio is written as QC_<name>_Metric/Flag/Status,
            # which QUALITY_CHECK does not own, so it never claims a column
            # in split_measurements; it is listed for documentation.
            producers.append(
                MeasurementProducer(name, cls, infos, shared_infos=(QUALITY_CHECK,))
            )
        else:
            producers.append(MeasurementProducer(name, cls, infos))

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


def _iter_public_info_classes() -> Iterator[type[MeasurementInfo]]:
    """Yield every concrete ``MeasurementInfo`` subclass exported by schema."""
    import phenotypic.schema as schema

    for name in getattr(schema, "__all__", ()):
        obj = getattr(schema, name, None)
        if _is_info_class(obj):
            yield obj


@lru_cache(maxsize=1)
def _known_families() -> tuple[str, ...]:
    """All public schema metric families, sorted longest-first for prefix matching."""
    families: set[str] = set()
    for obj in _iter_public_info_classes():
        try:
            families.add(obj.metric_family())
        except NotImplementedError:  # member-less classification bases
            continue
    return tuple(sorted(families, key=len, reverse=True))


def _sanitize_token(token: str) -> str:
    return _SANITIZE_TOKEN_RE.sub("", token.strip())


def metric_token(on: str) -> str:
    """Derive the ``<metric>`` header segment from a fitter's ``on`` column.

    Strips the longest known schema **metric family** prefix if present
    (``Size_Area`` → ``Area``), else returns the value verbatim
    (``x`` → ``x``); then removes whitespace.
    """
    value = str(on).strip()
    for family in _known_families():
        if value.startswith(family + "_"):
            return _sanitize_token(value[len(family) + 1:])
    return _sanitize_token(value)


@lru_cache(maxsize=1)
def _public_info_classes() -> tuple[type[MeasurementInfo], ...]:
    """Public, member-ful ``MeasurementInfo`` subclasses exported by schema."""
    return tuple(obj for obj in _iter_public_info_classes() if list(obj))


def _member_for_column(column: str) -> MeasurementInfo | None:
    """Resolve *column* to its public schema member across all header schemes."""
    for info in _public_info_classes():
        member = info.member_for_header(column)
        if member is not None:
            return member
    return None


def _describe_column(column: str) -> str | None:
    """Resolve *column* to its member's ``desc`` across all schemes, or None."""
    member = _member_for_column(column)
    return member.desc if member is not None else None


__all__ = ["generate_output_key", "split_measurements", "split_measurements_by_category"]
