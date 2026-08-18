"""Permanent compatibility data for historical metadata header spellings.

The canonical schema never branches on these prefixes. Keeping historical
storage knowledge here lets readers accept exact headers emitted by older
PhenoTypic releases without treating arbitrary ``MetadataFoo_*`` strings as
metadata.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import cast

from phenotypic.schema import (
    ACQUISITION,
    CONDITION,
    CULTURE,
    EXPERIMENT,
    GENETIC,
    IMAGE,
    PLATE,
    SAMPLE,
    STUDY,
    MetadataInfo,
)


_LEGACY_PREFIX_BY_OWNER: Mapping[type[MetadataInfo], str] = MappingProxyType(
    {
        IMAGE: "MetadataImage",
        STUDY: "MetadataStudy",
        EXPERIMENT: "MetadataExperiment",
        GENETIC: "MetadataGenetic",
        SAMPLE: "MetadataSample",
        CONDITION: "MetadataCondition",
        CULTURE: "MetadataCulture",
        PLATE: "MetadataPlate",
        ACQUISITION: "MetadataAcquisition",
    }
)


def _build_legacy_header_maps() -> tuple[
    Mapping[str, MetadataInfo], Mapping[str, str]
]:
    """Generate exact historical-header indexes from owner declarations."""
    member_by_header: dict[str, MetadataInfo] = {}
    canonical_by_header: dict[str, str] = {}
    for owner, prefix in _LEGACY_PREFIX_BY_OWNER.items():
        for member in cast(Iterable[MetadataInfo], owner):
            legacy_header = f"{prefix}_{member.label}"
            if legacy_header in member_by_header:
                raise ValueError(f"Duplicate legacy metadata header {legacy_header!r}")
            member_by_header[legacy_header] = member
            canonical_by_header[legacy_header] = member.value
    return MappingProxyType(member_by_header), MappingProxyType(canonical_by_header)


LEGACY_HEADER_TO_MEMBER, LEGACY_HEADER_TO_CANONICAL = _build_legacy_header_maps()

