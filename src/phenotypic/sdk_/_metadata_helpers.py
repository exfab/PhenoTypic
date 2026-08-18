"""Metadata ownership, normalization, and presentation ordering."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, TypeVar, cast, overload
import warnings

import phenotypic.schema as _schema
from phenotypic.schema import MetadataInfo
from ._metadata_compatibility import LEGACY_HEADER_TO_MEMBER

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

_FrameT = TypeVar("_FrameT", "pd.DataFrame", "pl.DataFrame")


def _metadata_members(owner: type[MetadataInfo]) -> tuple[MetadataInfo, ...]:
    """Return one metadata owner's members with an explicit static type."""
    return tuple(cast(Iterable[MetadataInfo], owner))


@lru_cache(maxsize=1)
def _metadata_enums() -> tuple[type[MetadataInfo], ...]:
    """Return exported concrete metadata owners, deduplicated by class identity."""
    out: list[type[MetadataInfo]] = []
    seen: set[type[MetadataInfo]] = set()
    for name in _schema.__all__:
        obj = getattr(_schema, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, MetadataInfo)
            and obj is not MetadataInfo
            and _metadata_members(obj)
            and obj not in seen
        ):
            out.append(obj)
            seen.add(obj)
    return tuple(out)


#: Bio-semantic owner order for the metadata front block. REMBI provenance is a
#: separate axis and does not affect presentation ordering.
_METADATA_OWNER_ORDER: tuple[type[MetadataInfo], ...] = (
    # (1) Identity — who / where is this colony
    _schema.SAMPLE,
    _schema.PLATE,
    # (2) Strain — genetic identity
    _schema.GENETIC,
    # (3) Condition — chemical then temporal/physical environment
    _schema.CONDITION,
    _schema.CULTURE,
    # (4) Design & provenance
    _schema.EXPERIMENT,
    _schema.STUDY,
    _schema.ACQUISITION,
    # Framework per-image bookkeeping — last (relocated to the trailing region
    # of the measurement frame by order_measurement_columns()).
    _schema.IMAGE,
)


@lru_cache(maxsize=1)
def _cluster_ordered_enums() -> tuple[type[MetadataInfo], ...]:
    """Return metadata owners in the explicit bio-semantic order."""
    discovered = _metadata_enums()
    rank = {owner: i for i, owner in enumerate(_METADATA_OWNER_ORDER)}
    return tuple(sorted(discovered, key=lambda owner: rank.get(owner, len(rank))))


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
    owner_rank = {owner: i for i, owner in enumerate(enums)}
    out: dict[str, int] = {}
    for enum in enums:
        # A hard raise (not an ``assert``, which ``python -O`` strips) — a stride
        # overflow would silently corrupt ordering, so it must fail loudly.
        members = _metadata_members(enum)
        if len(members) >= _CATEGORY_STRIDE:
            raise ValueError(
                f"{enum.__name__} has {len(members)} members, exceeding the "
                f"canonical-order category stride ({_CATEGORY_STRIDE})"
            )
        base = owner_rank[enum] * _CATEGORY_STRIDE
        for i, member in enumerate(members):
            member_rank = base + i
            out[member.value] = member_rank
    return out


def order_measurement_columns(columns: Sequence[str]) -> list[str]:
    """Canonical measurement-frame column order.

    ``[front metadata] -> [measurements] -> [IMAGE metadata] -> [info block]``.

    Front (user/experimental) metadata is cluster/definition ordered via
    :func:`canonical_metadata_order`; unknown/uncategorized ``Metadata_*`` tags fall
    to the end of the front block alphabetically. The framework ``IMAGE``-owned
    block is per-image provenance and trails the measurements. The per-object info
    block (``Object_Label`` + ``Bbox_*`` / ``Grid_*``) is detected by name and moves
    last. Measurements keep their incoming relative order.

    Pure over column-name strings, so both the pandas (``df[...]``) and polars
    (``df.select(...)``) paths reuse it.
    """
    from phenotypic.schema import IMAGE, OBJECT

    rank = canonical_metadata_order()
    label = str(OBJECT.LABEL)

    front: list[str] = []
    image_meta: list[str] = []
    info: list[str] = []
    meas: list[str] = []
    for c in columns:
        if metadata_owner_for_header(c) is IMAGE:
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

    def _known_member_rank(column: str) -> int:
        member = metadata_member_for_header(column)
        return rank.get(member.value, unknown_rank) if member is not None else unknown_rank

    front.sort(key=lambda c: (_known_member_rank(c), str(c)))
    # Object_Label leads the info block; Bbox_*/Grid_* keep their incoming order
    # (stable sort) so the trailing block matches #180's info-frame geometry order.
    info.sort(key=lambda c: 0 if c == label else 1)
    return front + meas + image_meta + info


def metadata_category_prefixes() -> tuple[str, ...]:
    """Return the canonical namespace prefix.

    Deprecated:
        Use :func:`is_metadata_header` for namespace detection or owner lookup
        helpers for semantic routing.
    """
    warnings.warn(
        "metadata_category_prefixes() is deprecated; use is_metadata_header() "
        "or metadata owner lookup helpers instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return (_GENERIC_PREFIX,)


#: Fallback prefix for metadata columns whose bare label is not in the
#: recommended vocabulary (routed to REMBI ``Uncategorized``). Every metadata
#: column — per-topic or generic — belongs to the ``Metadata`` family.
_GENERIC_PREFIX = "Metadata_"


def _flat_header(label: str) -> str:
    """Return the canonical flat-namespace spelling for a bare metadata label."""
    return f"{_GENERIC_PREFIX}{label}"


@dataclass(frozen=True, slots=True)
class _MetadataRegistry:
    """Immutable reverse indexes for all known metadata owners."""

    header_to_member: Mapping[str, MetadataInfo]
    label_to_member: Mapping[str, MetadataInfo]


def _register_unique(
    registry: dict[str, MetadataInfo],
    key: str,
    member: MetadataInfo,
    *,
    key_kind: str,
) -> None:
    """Register one key or fail if another metadata owner already claims it."""
    existing = registry.get(key)
    if existing is not None and existing is not member:
        raise ValueError(
            f"Duplicate metadata {key_kind} {key!r}: "
            f"{type(existing).__name__}.{existing.name} and "
            f"{type(member).__name__}.{member.name}"
        )
    registry[key] = member


def _build_metadata_registry(
    owners: Sequence[type[MetadataInfo]],
) -> _MetadataRegistry:
    """Build fail-fast label/header indexes for concrete metadata owners."""
    by_header: dict[str, MetadataInfo] = {}
    by_label: dict[str, MetadataInfo] = {}
    for owner in owners:
        members = _metadata_members(owner)
        declarations = cast(Mapping[str, MetadataInfo], owner.__members__)
        if len(declarations) != len(members):
            aliases = [
                f"{owner.__name__}.{name} -> {member.label!r}"
                for name, member in declarations.items()
                if member.name != name
            ]
            raise ValueError(
                "Duplicate metadata declarations are not allowed; Enum aliases "
                "would hide label/header collisions: "
                + ", ".join(aliases)
            )
        for member in members:
            _register_unique(by_label, member.label, member, key_kind="label")
            _register_unique(by_header, member.value, member, key_kind="header")
    return _MetadataRegistry(
        header_to_member=MappingProxyType(by_header),
        label_to_member=MappingProxyType(by_label),
    )


@lru_cache(maxsize=1)
def _metadata_registry() -> _MetadataRegistry:
    """Return the validated immutable registry for the live schema."""
    return _build_metadata_registry(_metadata_enums())


def _metadata_member_for_name(name: str) -> MetadataInfo | None:
    """Resolve a bare, canonical, or exact historical metadata name."""
    normalized_name = str(name)
    registry = _metadata_registry()
    return (
        registry.header_to_member.get(normalized_name)
        or registry.label_to_member.get(normalized_name)
        or LEGACY_HEADER_TO_MEMBER.get(normalized_name)
    )


def metadata_member_for_header(header: str) -> MetadataInfo | None:
    """Return the member for a bare, canonical, or exact historical header."""
    return _metadata_member_for_name(header)


def metadata_owner_for_header(header: str) -> type[MetadataInfo] | None:
    """Return the owner for a bare, canonical, or exact historical header."""
    member = metadata_member_for_header(header)
    return type(member) if member is not None else None


def metadata_member_for_label(label: str) -> MetadataInfo | None:
    """Return the member for a bare, canonical, or exact historical label."""
    return _metadata_member_for_name(label)


def metadata_owner_for_label(label: str) -> type[MetadataInfo] | None:
    """Return the owner for a bare, canonical, or exact historical label."""
    member = metadata_member_for_label(label)
    return type(member) if member is not None else None


def is_metadata_header(col: str) -> bool:
    """True if ``col`` is a metadata-family column.

    Matches canonical ``Metadata_*`` columns and the exact finite set of
    historical per-topic headers. Arbitrary lookalikes such as
    ``MetadataFoo_Bar`` are rejected.
    """
    s = str(col)
    return s.startswith(_GENERIC_PREFIX) or s in LEGACY_HEADER_TO_MEMBER


def ensure_metadata_prefix(name: str) -> str:
    """Normalize a metadata name to the live schema's emitted spelling.

    Bare labels, canonical flat headers, and exact historical per-topic headers
    for known members all resolve centrally. Unknown bare labels receive the
    generic prefix. Unknown canonical metadata headers remain unchanged.
    """
    normalized_name = str(name)
    member = _metadata_member_for_name(normalized_name)
    if member is not None:
        return member.value
    if is_metadata_header(normalized_name):
        return normalized_name
    return _flat_header(normalized_name)


def metadata_category_for_label(label: str) -> str | None:
    """Return the shared category for a known label, or ``None``.

    Deprecated:
        Use :func:`metadata_owner_for_label` or
        :func:`metadata_member_for_label` for semantic routing.
    """
    warnings.warn(
        "metadata_category_for_label() is deprecated; use "
        "metadata_owner_for_label() or metadata_member_for_label() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    owner = metadata_owner_for_label(label)
    return owner.category() if owner is not None else None


def _pandas_values_are_strings(series: "pd.Series") -> bool:
    """Return whether every populated pandas value is a string."""
    return all(isinstance(value, str) for value in series[series.notna()].tolist())


def _pandas_numeric_supertype(left_dtype: Any, right_dtype: Any) -> Any | None:
    """Return pandas' nullable dtype for NumPy's numeric promotion result."""
    import numpy as np
    import pandas as pd
    from pandas.api.types import is_bool_dtype, is_numeric_dtype

    if (
        not is_numeric_dtype(left_dtype)
        or not is_numeric_dtype(right_dtype)
        or is_bool_dtype(left_dtype)
        or is_bool_dtype(right_dtype)
    ):
        return None
    try:
        left_numpy = np.dtype(getattr(left_dtype, "numpy_dtype", left_dtype))
        right_numpy = np.dtype(getattr(right_dtype, "numpy_dtype", right_dtype))
        promoted = np.promote_types(left_numpy, right_numpy)
    except (TypeError, ValueError):
        return None
    nullable_types: dict[tuple[str, int], Any] = {
        ("i", 1): pd.Int8Dtype(),
        ("i", 2): pd.Int16Dtype(),
        ("i", 4): pd.Int32Dtype(),
        ("i", 8): pd.Int64Dtype(),
        ("u", 1): pd.UInt8Dtype(),
        ("u", 2): pd.UInt16Dtype(),
        ("u", 4): pd.UInt32Dtype(),
        ("u", 8): pd.UInt64Dtype(),
        ("f", 4): pd.Float32Dtype(),
        ("f", 8): pd.Float64Dtype(),
    }
    return nullable_types.get((promoted.kind, promoted.itemsize))


def _pandas_common_dtype(left: "pd.Series", right: "pd.Series") -> Any | None:
    """Choose a lossless common dtype candidate for two pandas columns."""
    import pandas as pd
    from pandas.api.types import is_dtype_equal

    left_has_values = bool(left.notna().any())
    right_has_values = bool(right.notna().any())
    if not left_has_values and not right_has_values:
        return left.dtype
    if not left_has_values:
        return right.dtype
    if not right_has_values:
        return left.dtype
    if is_dtype_equal(left.dtype, right.dtype):
        return left.dtype
    if _pandas_values_are_strings(left) and _pandas_values_are_strings(right):
        return pd.StringDtype(storage="python")
    return _pandas_numeric_supertype(left.dtype, right.dtype)


def _pandas_values_agree(left: "pd.Series", right: "pd.Series") -> bool:
    """Return whether overlapping non-null pandas values compare equal."""
    overlap = left.notna() & right.notna()
    if not overlap.any():
        return True
    try:
        return bool(left[overlap].eq(right[overlap]).fillna(False).all())
    except (TypeError, ValueError):
        return False


def _scalar_values_equal(left: Any, right: Any) -> bool:
    """Compare scalar values without array/backend dtype coercion."""
    import math

    if (
        isinstance(left, float)
        and isinstance(right, float)
        and math.isnan(left)
        and math.isnan(right)
    ):
        return True
    try:
        result = left == right
        return isinstance(result, bool) and result
    except (TypeError, ValueError):
        return False


def _pandas_values_equal_exact(left: "pd.Series", right: "pd.Series") -> bool:
    """Compare pandas values and nulls scalar-by-scalar without coercion."""
    left_null = left.isna().tolist()
    right_null = right.isna().tolist()
    if left_null != right_null:
        return False
    for is_null, left_value, right_value in zip(
        left_null, left.tolist(), right.tolist(), strict=True
    ):
        if not is_null and not _scalar_values_equal(left_value, right_value):
            return False
    return True


def _pandas_cast_losslessly(series: "pd.Series", dtype: Any) -> "pd.Series | None":
    """Cast through ``dtype`` and prove a roundtrip preserves values and nulls."""
    try:
        casted = series.astype(dtype)
        roundtripped = casted.astype(series.dtype)
    except (TypeError, ValueError, OverflowError):
        return None
    if not _pandas_values_equal_exact(series, casted):
        return None
    if not _pandas_values_equal_exact(series, roundtripped):
        return None
    return casted


def _normalize_pandas_metadata_columns(frame: "pd.DataFrame") -> "pd.DataFrame":
    """Return a normalized deep copy of a pandas DataFrame."""
    import pandas as pd

    source_names = [str(column) for column in frame.columns]
    targets = [ensure_metadata_prefix(column) for column in source_names]
    groups: dict[str, list[int]] = {}
    for position, target in enumerate(targets):
        groups.setdefault(target, []).append(position)

    combined: dict[int, "pd.Series"] = {}
    consumed: set[int] = set()
    for target, positions in groups.items():
        anchor = next(
            (position for position in positions if source_names[position] == target),
            positions[0],
        )
        merged = frame.iloc[:, positions[0]].copy(deep=True)
        for position in positions[1:]:
            candidate = frame.iloc[:, position]
            if not _pandas_values_agree(merged, candidate):
                raise ValueError(
                    f"Metadata columns normalizing to {target!r} contain "
                    "conflicting non-null values"
                )
            common_dtype = _pandas_common_dtype(merged, candidate)
            left_cast = (
                _pandas_cast_losslessly(merged, common_dtype)
                if common_dtype is not None
                else None
            )
            right_cast = (
                _pandas_cast_losslessly(candidate, common_dtype)
                if common_dtype is not None
                else None
            )
            if left_cast is None or right_cast is None:
                raise ValueError(
                    f"Metadata columns normalizing to {target!r} have incompatible "
                    f"dtypes that cannot share a lossless representation: "
                    f"{merged.dtype} and {candidate.dtype}"
                )
            try:
                merged = left_cast.combine_first(right_cast)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Metadata columns normalizing to {target!r} could not be "
                    "coalesced losslessly"
                ) from exc
        merged.name = target
        combined[anchor] = merged
        consumed.update(position for position in positions if position != anchor)

    output_columns = [
        combined.get(position, frame.iloc[:, position].copy(deep=True)).rename(
            targets[position]
        )
        for position in range(len(source_names))
        if position not in consumed
    ]
    if not output_columns:
        return pd.DataFrame(index=frame.index.copy())
    return pd.concat(output_columns, axis=1)


def _polars_common_dtype(left: "pl.Series", right: "pl.Series") -> Any | None:
    """Choose a lossless common dtype candidate for two Polars columns."""
    import polars as pl

    left_has_values = left.null_count() != len(left)
    right_has_values = right.null_count() != len(right)
    if not left_has_values and not right_has_values:
        return left.dtype
    if not left_has_values:
        return right.dtype
    if not right_has_values:
        return left.dtype
    if left.dtype == right.dtype:
        return left.dtype
    textual_types = {pl.String, pl.Categorical, pl.Enum}
    if left.dtype.base_type() in textual_types and right.dtype.base_type() in textual_types:
        return pl.String
    return _polars_numeric_supertype(left.dtype, right.dtype)


def _polars_numeric_supertype(left_dtype: Any, right_dtype: Any) -> Any | None:
    """Return the Polars dtype corresponding to NumPy numeric promotion."""
    import numpy as np
    import polars as pl

    numpy_types: dict[Any, Any] = {
        pl.Int8: np.dtype("int8"),
        pl.Int16: np.dtype("int16"),
        pl.Int32: np.dtype("int32"),
        pl.Int64: np.dtype("int64"),
        pl.UInt8: np.dtype("uint8"),
        pl.UInt16: np.dtype("uint16"),
        pl.UInt32: np.dtype("uint32"),
        pl.UInt64: np.dtype("uint64"),
        pl.Float32: np.dtype("float32"),
        pl.Float64: np.dtype("float64"),
    }
    left_numpy = numpy_types.get(left_dtype.base_type())
    right_numpy = numpy_types.get(right_dtype.base_type())
    if left_numpy is None or right_numpy is None:
        return None
    promoted = np.promote_types(left_numpy, right_numpy)
    polars_types: dict[Any, Any] = {
        np.dtype("int8"): pl.Int8,
        np.dtype("int16"): pl.Int16,
        np.dtype("int32"): pl.Int32,
        np.dtype("int64"): pl.Int64,
        np.dtype("uint8"): pl.UInt8,
        np.dtype("uint16"): pl.UInt16,
        np.dtype("uint32"): pl.UInt32,
        np.dtype("uint64"): pl.UInt64,
        np.dtype("float32"): pl.Float32,
        np.dtype("float64"): pl.Float64,
    }
    return polars_types.get(promoted)


def _polars_values_agree(left: "pl.Series", right: "pl.Series") -> bool:
    """Return whether overlapping non-null Polars values compare equal."""
    import math

    for left_value, right_value in zip(left.to_list(), right.to_list(), strict=True):
        if left_value is None or right_value is None:
            continue
        if (
            isinstance(left_value, float)
            and isinstance(right_value, float)
            and math.isnan(left_value)
            and math.isnan(right_value)
        ):
            continue
        try:
            if left_value != right_value:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _polars_values_equal_exact(left: "pl.Series", right: "pl.Series") -> bool:
    """Compare Polars values and nulls scalar-by-scalar without coercion."""
    left_null = left.is_null().to_list()
    right_null = right.is_null().to_list()
    if left_null != right_null:
        return False
    for is_null, left_value, right_value in zip(
        left_null, left.to_list(), right.to_list(), strict=True
    ):
        if not is_null and not _scalar_values_equal(left_value, right_value):
            return False
    return True


def _polars_cast_losslessly(series: "pl.Series", dtype: Any) -> "pl.Series | None":
    """Cast through ``dtype`` and prove a roundtrip preserves values and nulls."""
    import polars as pl

    try:
        casted = series.cast(dtype, strict=True)
        roundtripped = casted.cast(series.dtype, strict=True)
    except (pl.exceptions.PolarsError, TypeError, ValueError, OverflowError):
        return None
    if not _polars_values_equal_exact(series, casted):
        return None
    if not _polars_values_equal_exact(series, roundtripped):
        return None
    return casted


def _normalize_polars_metadata_columns(frame: "pl.DataFrame") -> "pl.DataFrame":
    """Return a normalized clone of a Polars DataFrame."""
    import polars as pl

    source_names = list(frame.columns)
    targets = [ensure_metadata_prefix(column) for column in source_names]
    groups: dict[str, list[int]] = {}
    for position, target in enumerate(targets):
        groups.setdefault(target, []).append(position)

    combined: dict[int, "pl.Series"] = {}
    consumed: set[int] = set()
    for target, positions in groups.items():
        anchor = next(
            (position for position in positions if source_names[position] == target),
            positions[0],
        )
        merged = frame.to_series(positions[0]).clone()
        for position in positions[1:]:
            candidate = frame.to_series(position)
            if not _polars_values_agree(merged, candidate):
                raise ValueError(
                    f"Metadata columns normalizing to {target!r} contain "
                    "conflicting non-null values"
                )
            common_dtype = _polars_common_dtype(merged, candidate)
            left_cast = (
                _polars_cast_losslessly(merged, common_dtype)
                if common_dtype is not None
                else None
            )
            right_cast = (
                _polars_cast_losslessly(candidate, common_dtype)
                if common_dtype is not None
                else None
            )
            if left_cast is None or right_cast is None:
                raise ValueError(
                    f"Metadata columns normalizing to {target!r} have incompatible "
                    f"dtypes that cannot share a lossless representation: "
                    f"{merged.dtype} and {candidate.dtype}"
                )
            pair = pl.DataFrame(
                {
                    "__metadata_left": left_cast,
                    "__metadata_right": right_cast,
                }
            )
            try:
                merged = pair.select(
                    pl.coalesce("__metadata_left", "__metadata_right").alias(target)
                ).to_series()
            except (pl.exceptions.PolarsError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Metadata columns normalizing to {target!r} could not be "
                    "coalesced losslessly"
                ) from exc
        combined[anchor] = merged.rename(target)
        consumed.update(position for position in positions if position != anchor)

    output_columns = [
        combined.get(position, frame.to_series(position).clone()).rename(targets[position])
        for position in range(len(source_names))
        if position not in consumed
    ]
    return pl.DataFrame(output_columns)


@overload
def normalize_metadata_columns(frame: "pd.DataFrame") -> "pd.DataFrame": ...


@overload
def normalize_metadata_columns(frame: "pl.DataFrame") -> "pl.DataFrame": ...


def normalize_metadata_columns(frame: _FrameT) -> _FrameT:
    """Normalize external metadata columns without mutating the input frame.

    The input and output frame implementations match. Columns that normalize to
    one target are coalesced only when their dtypes are compatible and all
    overlapping non-null values agree. A conflict raises before any caller-owned
    state is changed.
    """
    import pandas as pd
    import polars as pl

    if isinstance(frame, pd.DataFrame):
        return _normalize_pandas_metadata_columns(frame)  # type: ignore[return-value]
    if isinstance(frame, pl.DataFrame):
        return _normalize_polars_metadata_columns(frame)  # type: ignore[return-value]
    raise TypeError(
        "normalize_metadata_columns requires a pandas or Polars DataFrame; "
        f"got {type(frame).__name__}"
    )


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
