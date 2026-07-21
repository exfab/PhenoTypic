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

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import phenotypic.schema as _schema
from phenotypic.schema import MeasurementInfo

if TYPE_CHECKING:
    import pandas as pd


@lru_cache(maxsize=1)
def _metadata_enums() -> tuple[type[MeasurementInfo], ...]:
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
def _cluster_ordered_enums() -> tuple[type[MeasurementInfo], ...]:
    """The metadata enums sorted by ``_METADATA_CLUSTER_ORDER`` (then stable).

    An enum whose category is absent from the cluster order sorts last; the
    coverage-gate test forbids that state, so it is a defensive fallback only.
    """
    rank = {cat: i for i, cat in enumerate(_METADATA_CLUSTER_ORDER)}
    return tuple(
        sorted(_metadata_enums(), key=lambda e: rank.get(e.category(), len(rank)))
    )


#: Stride between metadata categories in ``canonical_metadata_order``. Must exceed
#: the largest metadata enum's member count so per-category definition ranks never
#: bleed into the next category.
_CATEGORY_STRIDE = 1000


@lru_cache(maxsize=1)
def canonical_metadata_order() -> dict[str, int]:
    """Global rank for every known metadata header (cluster then definition order).

    Cluster-order major, enum definition-order minor. A header absent from this
    map is an unknown/uncategorized user tag; callers rank those last. The map is
    derived entirely from the import-time schema enums, so it is cached. The
    returned dict is read-only by contract; callers must not mutate it (mirrors
    :func:`~phenotypic.schema.header_to_module`).
    """
    enums = _cluster_ordered_enums()
    cat_rank = {e.category(): i for i, e in enumerate(enums)}
    out: dict[str, int] = {}
    for enum in enums:
        # A hard raise (not an ``assert``, which ``python -O`` strips) — a stride
        # overflow would silently corrupt ordering, so it must fail loudly.
        if len(enum) >= _CATEGORY_STRIDE:
            raise ValueError(
                f"{enum.__name__} has {len(enum)} members, exceeding the "
                f"canonical-order category stride ({_CATEGORY_STRIDE})"
            )
        base = cat_rank[enum.category()] * _CATEGORY_STRIDE
        for i, member in enumerate(enum):
            out[member.value] = base + i
    return out


def order_measurement_columns(columns: Sequence[str]) -> list[str]:
    """Canonical measurement-frame column order.

    ``[front metadata] -> [measurements] -> [MetadataImage_*] -> [info block]``.

    Front (user/experimental) metadata is cluster/definition ordered via
    :func:`canonical_metadata_order`; unknown/uncategorized ``Metadata_*`` tags fall
    to the end of the front block alphabetically. The framework ``MetadataImage_*``
    block is per-image provenance and trails the measurements. The per-object info
    block (``Object_Label`` + ``Bbox_*`` / ``Grid_*``) is detected by name and moves
    last. Measurements keep their incoming relative order.

    Pure over column-name strings, so both the pandas (``df[...]``) and polars
    (``df.select(...)``) paths reuse it.
    """
    from phenotypic.schema import METADATA, OBJECT

    rank = canonical_metadata_order()
    image_prefix = f"{METADATA.category()}_"
    label = str(OBJECT.LABEL)

    front: list[str] = []
    image_meta: list[str] = []
    info: list[str] = []
    meas: list[str] = []
    for c in columns:
        if c.startswith(image_prefix):
            image_meta.append(c)
        elif is_metadata_header(c):
            front.append(c)
        elif c == label or c.startswith("Bbox_") or c.startswith("Grid_"):
            info.append(c)
        else:
            meas.append(c)
    # Unknown/uncategorized Metadata_* tags sort AFTER every known header. Ranks
    # use a 1000-stride, so len(rank) (~72) is not a valid "after everything"
    # sentinel — derive it from the actual max rank.
    unknown_rank = max(rank.values(), default=0) + 1
    front.sort(key=lambda c: (rank.get(c, unknown_rank), str(c)))
    # Object_Label leads the info block; Bbox_*/Grid_* keep their incoming order
    # (stable sort) so the trailing block matches #180's info-frame geometry order.
    info.sort(key=lambda c: 0 if c == label else 1)
    return front + meas + image_meta + info


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


def metadata_only_mask(df: "pd.DataFrame") -> "pd.Series":
    """Mask of ``--metadata`` phantom rows; all-``False`` when unknowable.

    A *phantom* row is one the CLI's ``--metadata`` left join carried through
    from the metadata CSV even though no measured object matched its key — every
    measurement/info column on it is null. Those rows are marked with the
    :attr:`~phenotypic.schema.METADATA_MATCH.METADATA_ONLY` (``QC_MetadataOnly``)
    boolean column.

    The flag is CLI-only, so public analysis/post entry points that a user calls
    on a hand-built or :meth:`~phenotypic._core._image.Image.measure` frame see
    no flag at all. This helper degrades to an all-``False`` mask in that case,
    which reproduces exactly the pre-left-join behavior for every caller.

    The dtype check is deliberately **strict**: only a real boolean column is
    trusted. An object/string column is rejected rather than coerced, because
    ``pd.Series(["False", "True"]).astype(bool)`` is ``[True, True]`` — the
    string ``"False"`` is truthy — which would silently mark every row a
    phantom. Rejecting costs nothing (it falls back to today's behavior); a
    lenient coercion would corrupt every result. Both real CLI round-trips
    preserve the dtype: parquet stores a native ``bool``, and polars'
    ``write_csv`` emits ``true``/``false``, which ``pd.read_csv`` parses to
    ``bool``.

    Args:
        df: Any measurement-shaped DataFrame.

    Returns:
        Boolean Series aligned to ``df.index``: ``True`` where the row is a
        metadata-only phantom, ``False`` everywhere else (and everywhere when
        the flag column is absent or not a boolean column).

    Examples:
        >>> import pandas as pd
        >>> from phenotypic.sdk_ import metadata_only_mask
        >>> # A frame from a notebook ``image.measure()`` carries no flag.
        >>> metadata_only_mask(pd.DataFrame({"Shape_Area": [10.0, 12.0]})).tolist()
        [False, False]
        >>> # A CLI mirror does: the undetected strain is flagged.
        >>> mirror = pd.DataFrame({"QC_MetadataOnly": [False, True]})
        >>> metadata_only_mask(mirror).tolist()
        [False, True]
    """
    import pandas as pd
    from pandas.api.types import is_bool_dtype

    from phenotypic.schema import METADATA_MATCH

    col = df.get(str(METADATA_MATCH.METADATA_ONLY))
    if col is not None and is_bool_dtype(col):
        return col.fillna(False).astype(bool)
    return pd.Series(False, index=df.index)
