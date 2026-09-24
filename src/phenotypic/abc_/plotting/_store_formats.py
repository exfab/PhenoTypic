"""The closed set of formats a per-image figure can be stored in (spec §2).

Standard library only, like the rest of :mod:`phenotypic.abc_.plotting`:
validation runs at class-definition time, and importing a plotting library to
answer "is this name allowed?" would break the lazy-import contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

StoreFormat = Literal["plotly-json", "png"]


@dataclass(frozen=True)
class StoreFormatInfo:
    """How one store format is named on disk and labelled for a consumer.

    Attributes:
        extension: File suffix, including the leading dot.
        media_type: The media type a consumer dispatches on.
        backends: The figure backends that can produce this format.
    """

    extension: str
    media_type: str
    backends: frozenset[str]


#: The folder layout is storage only: a consumer reads ``media_type`` from the
#: descriptor, never the extension.
STORE_FORMATS: Mapping[str, StoreFormatInfo] = MappingProxyType({
    "plotly-json": StoreFormatInfo(
        ".plotly.json", "application/vnd.plotly.v1+json", frozenset({"plotly"})
    ),
    "png": StoreFormatInfo(".png", "image/png", frozenset({"plotly", "mpl"})),
})

_DEFAULTS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "plotly": ("plotly-json",),
    "mpl": ("png",),
})


def default_store_formats(backend: str) -> tuple[str, ...]:
    """Return the formats stored when a figure declares none.

    Args:
        backend: ``"plotly"`` or ``"mpl"``.

    Returns:
        The backend's default format tuple.

    Raises:
        ValueError: If *backend* is not a known figure backend.
    """
    try:
        return _DEFAULTS[backend]
    except KeyError:
        raise ValueError(f"unknown figure backend {backend!r}") from None


def resolve_store_formats(
    store: tuple[str, ...] | None, *, backend: str, owner: str
) -> tuple[str, ...]:
    """Validate a ``store=`` declaration, or supply the backend default.

    Args:
        store: The declared formats, or ``None`` for the default.
        backend: The figure's declared backend.
        owner: Method name, for the error message.

    Returns:
        The formats to store, in declared order.

    Raises:
        TypeError: On a bare string, an empty tuple, an unknown or duplicate
            name, or a format the backend cannot produce.
    """
    if store is None:
        return default_store_formats(backend)
    if isinstance(store, str):
        raise TypeError(
            f"@figure({owner!r}): store must be a tuple of format names, "
            f"got the string {store!r}; write store=({store!r},)"
        )
    formats = tuple(store)
    if not formats:
        raise TypeError(
            f"@figure({owner!r}): store=() is refused -- an image figure must "
            "store at least one format, because deliverables/plots is copied "
            "out of the store and an unstored figure would appear nowhere"
        )
    unknown = [name for name in formats if name not in STORE_FORMATS]
    if unknown:
        raise TypeError(
            f"@figure({owner!r}): unknown store format(s) {unknown}; "
            f"expected any of {list(STORE_FORMATS)}"
        )
    duplicates = sorted({name for name in formats if formats.count(name) > 1})
    if duplicates:
        raise TypeError(f"@figure({owner!r}): duplicate store format(s) {duplicates}")
    unsupported = [n for n in formats if backend not in STORE_FORMATS[n].backends]
    if unsupported:
        raise TypeError(
            f"@figure({owner!r}): backend={backend!r} cannot produce "
            f"store format(s) {unsupported}"
        )
    return formats


__all__ = [
    "STORE_FORMATS",
    "StoreFormat",
    "StoreFormatInfo",
    "default_store_formats",
    "resolve_store_formats",
]
