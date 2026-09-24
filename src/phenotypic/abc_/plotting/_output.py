"""Backend-neutral single and multi-page plot output values."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal, Mapping, TypeAlias

FigureLike: TypeAlias = Any


class FigureInputUnavailable(RuntimeError):
    """``inspect()`` cannot draw this figure here, for this image.

    Raised by a provider whose figure is drawn from state only its own
    ``apply()`` produces -- such as the as-shot pixels an image corrector
    overwrites -- when that ``apply()`` did not run in this process on the
    image it was given. It says where the figure can be drawn, not that the
    figure failed: the CLI keeps a stored copy of it when one exists.
    """


def figure_backend_of(figure: Any) -> Literal["plotly", "mpl"] | None:
    """Return the rendering backend of ``figure``, or ``None`` if unknown.

    Identification is by module string so this module stays standard-library
    only at runtime -- importing plotly or matplotlib to answer the question
    would defeat the lazy-import contract this package is held to.

    Args:
        figure: Any object that might be a supported figure.

    Returns:
        ``"plotly"``, ``"mpl"``, or ``None`` for anything unrecognised. Never
        raises: callers that need an error raise their own, with the context
        only they have.
    """
    module = type(figure).__module__
    if type(figure).__name__ != "Figure":
        return None
    if module.startswith("plotly."):
        return "plotly"
    if module.startswith("matplotlib."):
        return "mpl"
    return None


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
    "FigureInputUnavailable",
    "FigureLike",
    "PlotOutput",
    "PlotPage",
    "canonical_group_key",
    "figure_backend_of",
]
