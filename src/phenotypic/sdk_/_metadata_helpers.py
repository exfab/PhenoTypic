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
from phenotypic.schema import MeasurementInfo


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


#: Bio-semantic cluster order for the metadata front-block (category granularity).
#: The human-facing column axis. REMBI (header_to_module / by_module / manifest)
#: remains a SEPARATE provenance axis and is not affected by this ordering.
_METADATA_CLUSTER_ORDER: tuple[str, ...] = (
    # (1) Identity — who / where is this colony
    "MetadataSample",
    "MetadataPlate",
    # (2) Strain — genetic identity
    "MetadataGenetic",
    # (3) Condition — chemical then temporal/physical environment
    "MetadataCondition",
    "MetadataCulture",
    # (4) Design & provenance
    "MetadataExperiment",
    "MetadataStudy",
    "MetadataAcquisition",
    # Framework per-image bookkeeping — last (relocated to the trailing region
    # of the measurement frame by order_measurement_columns()).
    "MetadataImage",
)


@lru_cache(maxsize=1)
def _cluster_ordered_enums() -> tuple[type, ...]:
    """The metadata enums sorted by ``_METADATA_CLUSTER_ORDER`` (then stable).

    An enum whose category is absent from the cluster order sorts last; the
    coverage-gate test forbids that state, so it is a defensive fallback only.
    """
    rank = {cat: i for i, cat in enumerate(_METADATA_CLUSTER_ORDER)}
    return tuple(
        sorted(_metadata_enums(), key=lambda e: rank.get(e.category(), len(rank)))
    )


@lru_cache(maxsize=1)
def canonical_metadata_order() -> dict[str, int]:
    """Global rank for every known metadata header (cluster then definition order).

    Cluster-order major, enum definition-order minor. A header absent from this
    map is an unknown/uncategorized user tag; callers rank those last. The map is
    derived entirely from the import-time schema enums, so it is cached.
    """
    cat_rank = {e.category(): i for i, e in enumerate(_cluster_ordered_enums())}
    out: dict[str, int] = {}
    for enum in _cluster_ordered_enums():
        # Category stride (1000) must exceed the largest enum's member count
        # (~12 today) so per-category definition ranks never bleed into the
        # next category. Guard the invariant rather than let a silent collision
        # corrupt ordering.
        assert len(enum) < 1000, f"{enum.__name__} exceeds the category stride"
        base = cat_rank[enum.category()] * 1000
        for i, member in enumerate(enum):
            out[member.value] = base + i
    return out


@lru_cache(maxsize=1)
def metadata_category_prefixes() -> tuple[str, ...]:
    """All metadata category prefixes (e.g. ``'MetadataGenetic_'``) in cluster order.

    Ordered by the bio-semantic cluster order (``_METADATA_CLUSTER_ORDER``), then
    deduplicated, so callers building bucket-priority lists get a stable, canonical
    ordering. REMBI is a separate axis (see ``by_module`` / ``header_to_module``).
    """
    seen: set[str] = set()
    prefixes: list[str] = []
    for e in _cluster_ordered_enums():
        p = f"{e.category()}_"
        if p not in seen:
            seen.add(p)
            prefixes.append(p)
    return tuple(prefixes)


#: Fallback prefix for metadata columns whose bare label is not in the
#: recommended vocabulary (routed to REMBI ``Uncategorized``). Every metadata
#: column — per-topic or generic — belongs to the ``Metadata`` family.
_GENERIC_PREFIX = "Metadata_"


def is_metadata_header(col: str) -> bool:
    """True if ``col`` is a metadata-family column.

    Matches any per-topic ``Metadata<Topic>_`` prefix (e.g.
    ``MetadataGenetic_``) *and* the generic ``Metadata_`` fallback used for
    uncategorized user metadata, per the namespace design (spec §10.3): the
    whole ``Metadata`` family is recognized, measurement/identity columns
    (``Shape_``, ``Object_``, ...) are not.
    """
    s = str(col)
    return s.startswith(_GENERIC_PREFIX) or any(
        s.startswith(p) for p in metadata_category_prefixes()
    )


def ensure_metadata_prefix(name: str) -> str:
    """Prefix a bare metadata label with its schema category, else generic.

    ``Strain -> MetadataGenetic_Strain`` (the owning enum's category); an
    unknown ``Foo -> Metadata_Foo`` (kept, uncategorized). Names that already
    carry a metadata prefix -- any ``Metadata<Topic>_`` category prefix or the
    generic ``Metadata_`` -- pass through unchanged.
    """
    if is_metadata_header(name) or name.startswith(_GENERIC_PREFIX):
        return name
    category = metadata_category_for_label(name)
    return f"{category}_{name}" if category else f"{_GENERIC_PREFIX}{name}"


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
