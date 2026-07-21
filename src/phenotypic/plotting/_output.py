"""Backend-neutral single and multi-page plot output values."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Mapping, TypeAlias

FigureLike: TypeAlias = Any


@dataclass(frozen=True)
class PlotPage:
    """One independently saveable figure page.

    Args:
        key: Stable logical page key.
        figure: Plotly or Matplotlib figure.
        label: Optional human-readable page label.
        metadata: Immutable-by-convention selector metadata.
    """

    key: str
    figure: FigureLike
    label: str | None = None
    metadata: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("plot page key must be a non-empty string")


@dataclass(frozen=True)
class PlotOutput:
    """Ordered pages returned by a plotting invocation."""

    pages: tuple[PlotPage, ...]

    def __post_init__(self) -> None:
        keys = [page.key for page in self.pages]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(f"plot output contains duplicate page keys: {duplicates}")


def normalize_plot_output(value: FigureLike | PlotOutput) -> PlotOutput:
    """Normalize a raw figure to the one-page runtime output contract."""
    if isinstance(value, PlotOutput):
        return value
    if value is None:
        return PlotOutput(pages=())
    return PlotOutput(pages=(PlotPage(key="default", figure=value),))


def canonical_group_key(
    pairs: list[tuple[str, Any]],
) -> str:
    """Encode typed grouping pairs without ambiguous string concatenation.

    Integer, float, temporal, and string values retain distinct tags. Numeric
    and temporal payloads use canonical strings so JSON encoder settings and
    platform formatting cannot change a page identity.
    """
    encoded: list[list[str | None]] = []
    for column, value in pairs:
        if hasattr(value, "item"):
            value = value.item()
        if value is None:
            kind = "null"
            canonical = None
        elif isinstance(value, bool):
            kind = "bool"
            canonical = "true" if value else "false"
        elif isinstance(value, datetime):
            kind = "datetime"
            canonical = value.isoformat()
        elif isinstance(value, date):
            kind = "date"
            canonical = value.isoformat()
        elif isinstance(value, timedelta):
            kind = "timedelta_ns"
            canonical = str(
                (
                    value.days * 86_400
                    + value.seconds
                )
                * 1_000_000_000
                + value.microseconds * 1_000
            )
        elif isinstance(value, int):
            kind = "int"
            canonical = str(value)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("grouping floats must be finite")
            kind = "float"
            canonical = value.hex()
        elif isinstance(value, str):
            kind = "str"
            canonical = value
        else:
            raise TypeError(
                f"unsupported grouping value {value!r} "
                f"({type(value).__name__})"
            )
        encoded.append([column, kind, canonical])
    return json.dumps(encoded, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "FigureLike",
    "PlotOutput",
    "PlotPage",
    "canonical_group_key",
    "normalize_plot_output",
]
