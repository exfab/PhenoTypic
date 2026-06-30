"""Single source of truth for the metadata column namespace.

Replaces every hardcoded ``"Metadata_"`` prefix literal across the codebase.
Prefixes and labels are derived from the ``phenotypic.schema`` metadata enums,
so they track the per-enum ``category()`` strings automatically: while every
metadata enum still returns ``"Metadata"`` the namespace collapses to a single
``"Metadata_"`` prefix, and once the categories flip to per-topic
``Metadata<Topic>`` strings these helpers pick up the new prefixes with no
caller changes.
"""

from __future__ import annotations

from functools import lru_cache

import phenotypic.schema as _schema
from phenotypic.schema import MeasurementInfo, REMBI_MODULE


@lru_cache(maxsize=1)
def _metadata_enums() -> tuple[type, ...]:
    """Every exported ``MeasurementInfo`` enum in the metadata namespace.

    A metadata-namespace enum is one whose ``category()`` starts with
    ``"Metadata"`` (the framework ``METADATA`` enum plus the experimental-tag
    vocabulary). Cached: the schema export surface is fixed at import time.
    """
    out = []
    for name in _schema.__all__:
        obj = getattr(_schema, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, MeasurementInfo)
            and obj is not MeasurementInfo
            and list(obj)
            and obj.category().startswith("Metadata")
        ):
            out.append(obj)
    return tuple(out)


@lru_cache(maxsize=1)
def metadata_category_prefixes() -> tuple[str, ...]:
    """All metadata category prefixes (e.g. ``'MetadataGenetic_'``) in REMBI order.

    Ordered by each enum's REMBI module then its category string, then
    deduplicated, so callers building bucket-priority lists get a stable,
    canonical ordering.
    """
    order = {m: i for i, m in enumerate(REMBI_MODULE)}
    enums = sorted(
        _metadata_enums(),
        key=lambda e: (order.get(next(iter(e)).resolved_rembi_module, 99), e.category()),
    )
    seen: set[str] = set()
    prefixes: list[str] = []
    for e in enums:
        p = f"{e.category()}_"
        if p not in seen:
            seen.add(p)
            prefixes.append(p)
    return tuple(prefixes)


def is_metadata_header(col: str) -> bool:
    """True if ``col`` is a metadata-family column (any ``MetadataXxx_`` prefix)."""
    return any(str(col).startswith(p) for p in metadata_category_prefixes())


@lru_cache(maxsize=1)
def _label_to_category() -> dict[str, str]:
    """Map each bare metadata label to the category of the enum that owns it.

    The first enum (in schema export order) to declare a label wins, mirroring
    the recommended-vocabulary contract where labels are unique across enums.
    """
    out: dict[str, str] = {}
    for e in _metadata_enums():
        for m in e:
            out.setdefault(m.label, e.category())
    return out


def metadata_category_for_label(label: str) -> str | None:
    """Category that owns a bare label (``'Strain' -> 'MetadataGenetic'``), or None."""
    return _label_to_category().get(label)
