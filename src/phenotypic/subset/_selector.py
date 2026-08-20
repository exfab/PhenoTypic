"""The ``SubsetSelector`` ABC, its result type, and its metadata errors.

Everything from triage through campaign execution runs on a **subset**; the
full dataset is touched exactly once, after an explicit human promotion
(§10.1). This module owns the boundary that produces one: a pluggable,
serializable strategy for choosing which of a parent's images the development
loop will spend its compute on.

The hierarchy follows the same pattern as every other extensible class in this
codebase — a pydantic ABC, concrete subclasses, ``{class, params}``
serialization, resolution by bare class name — so adding a fourth selector is a
subclass plus one ``__init__.py`` export, with no tool signature change and no
schema bump.
"""

from __future__ import annotations

import csv
import random
from abc import ABC, abstractmethod
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, Mapping, Sequence

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    model_validator,
)

from phenotypic.sdk_._metadata_helpers import ensure_metadata_prefix

#: Header spellings accepted as a grouping CSV's **image identity** column,
#: compared case-insensitively. This is an identity key, not a metadata value,
#: so it is matched literally rather than through
#: :func:`~phenotypic.sdk_._metadata_helpers.ensure_metadata_prefix` — a column
#: named ``image`` canonicalizes to ``Metadata_image``, which is not the
#: schema's ``Metadata_ImageFile`` and would silently miss.
IMAGE_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "image",
    "images",
    "image_file",
    "imagefile",
    "image_path",
    "file",
    "filename",
    "path",
    "relative_path",
    "metadata_imagefile",
)

def _deep_freeze(value: Any) -> Any:
    """Recursively replace mutable containers with immutable equivalents.

    ``ConfigDict(frozen=True)`` blocks attribute **assignment** and nothing
    else, so a ``dict`` field on a frozen model is wide open:
    ``selection.group_filter[key] = other`` succeeds. Nesting matters as much
    as the top level, because a selector's ``params`` is its
    ``model_dump(mode="json")`` and therefore carries its own ``group_filter``
    one level down.

    Mappings become :class:`~types.MappingProxyType` over an already-frozen
    copy — a *copy*, so no caller keeps a live handle to the underlying dict —
    and sequences become tuples. Both still compare equal to the plain
    containers they replace, so callers and artifacts read unchanged.
    """
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    """Undo :func:`_deep_freeze` for serialization.

    A ``MappingProxyType`` is not JSON-serializable and pydantic will not
    descend into an ``Any``, so the artifact writer would otherwise receive
    proxies where it expects objects.
    """
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(_deep_thaw(item) for item in value)
    return value


#: A ``{str: str}`` field that is immutable *through* the value, not merely
#: unassignable. Serializes back to a plain ``dict``.
FrozenStrMap = Annotated[
    Mapping[str, str],
    AfterValidator(_deep_freeze),
    PlainSerializer(_deep_thaw, return_type=dict),
]

#: A ``{str: int}`` field, frozen the same way.
FrozenIntMap = Annotated[
    Mapping[str, int],
    AfterValidator(_deep_freeze),
    PlainSerializer(_deep_thaw, return_type=dict),
]

#: A ``{str: Any}`` field, frozen **recursively** — a selector's serialized
#: params nest, and a shallow freeze leaves the nested levels editable.
FrozenParams = Annotated[
    Mapping[str, Any],
    AfterValidator(_deep_freeze),
    PlainSerializer(_deep_thaw, return_type=dict),
]


#: Cost tiers a selector can report, matching the server's work classes.
#: ``W0`` needs only files already on disk; ``W2`` is scheduled compute.
SelectorCostClass = Literal["W0", "W1", "W2"]

#: How a stratified selector spreads ``n`` across its groups.
GroupAllocation = Literal["proportional", "equal"]


class SubsetMetadataError(ValueError):
    """Base for a grouping-CSV problem a selector refuses to guess past.

    A ``ValueError`` subclass so ordinary callers catch it without importing
    this module, and a distinct type so the tool layer can map each subclass to
    its own error code (§6.2) instead of matching on a message string.
    """


class GroupFilterColumnNotFound(SubsetMetadataError):
    """``group_filter`` named a column the grouping CSV does not have.

    Maps to ``group_filter_column_not_found``. Carries the CSV's actual column
    list so the tool layer can offer a did-you-mean, exactly as
    ``group_key_not_in_metadata`` does.

    Attributes:
        column: The requested column, canonicalized.
        available_columns: The CSV's metadata columns, canonicalized and sorted.
    """

    def __init__(self, column: str, available_columns: Sequence[str]) -> None:
        self.column = column
        self.available_columns = tuple(available_columns)
        super().__init__(
            f"group_filter column {column!r} is not in the grouping metadata; "
            f"available: {', '.join(self.available_columns) or '(none)'}"
        )


class GroupFilterMatchesNothing(SubsetMetadataError):
    """``group_filter`` matched no candidate image.

    Maps to ``group_filter_matches_nothing``. It is an error and not an empty
    selection: an empty subset passes every downstream shape check and produces
    a study of nothing.

    Attributes:
        group_filter: The filter that matched nothing.
    """

    def __init__(self, group_filter: dict[str, str]) -> None:
        self.group_filter = dict(group_filter)
        super().__init__(
            f"group_filter {self.group_filter!r} matched none of the candidate "
            "images; refusing to select an empty subset"
        )


class GroupKeyNotInMetadata(SubsetMetadataError):
    """``group_key`` named a column no candidate image has a value for.

    Maps to ``group_key_not_in_metadata``. Raised both when the column is
    absent from the CSV and when it is present but the CSV's identity column
    joins to none of the candidates — the second case is the one that matters,
    because falling through it would stratify on a single implicit group and
    report a stratified selection that never stratified.

    Attributes:
        group_key: The requested column, canonicalized.
        available_columns: The CSV's metadata columns, canonicalized and sorted.
    """

    def __init__(self, group_key: str, available_columns: Sequence[str]) -> None:
        self.group_key = group_key
        self.available_columns = tuple(available_columns)
        super().__init__(
            f"group_key {group_key!r} resolved no group for any candidate image; "
            f"available columns: {', '.join(self.available_columns) or '(none)'}"
        )


class GroupFloorsExceedTarget(ValueError):
    """``min_per_group`` cannot be honoured inside ``n``.

    ``n`` is the **compute budget** and it wins: bounding compute is the whole
    reason the subset boundary exists, and a selector that quietly returned
    three times the budget defeated it with every downstream check green. The
    ABC's "``n`` is a target, not a contract" licenses taking *fewer* images
    than asked, never more.

    ``min_per_group`` is a **statistical floor**, so silently dropping below
    it is equally wrong: it would report a stratified selection whose strata
    are too thin to mean anything. When the two genuinely conflict there is no
    correct silent answer, so the selector refuses and names both numbers. The
    caller raises ``n``, lowers ``min_per_group``, or narrows the groups.

    Not a :class:`SubsetMetadataError`: nothing is wrong with the grouping CSV.
    It is a ``ValueError`` so ordinary callers catch it without importing this
    module.

    Attributes:
        n: The requested target.
        min_per_group: The requested per-group floor.
        required: Images the floors demand, after clamping each to its group's
            size.
        group_sizes: Group name → candidates available in it.
    """

    def __init__(
        self,
        *,
        n: int,
        min_per_group: int,
        required: int,
        group_sizes: Mapping[str, int],
    ) -> None:
        self.n = n
        self.min_per_group = min_per_group
        self.required = required
        self.group_sizes = dict(group_sizes)
        super().__init__(
            f"min_per_group={min_per_group} over {len(self.group_sizes)} "
            f"groups needs at least {required} images, but n={n}. "
            "n is the compute budget and is never exceeded; raise n, lower "
            "min_per_group, or narrow the groups with group_filter. "
            f"Group sizes: {dict(sorted(self.group_sizes.items()))}"
        )


class ImageRef(BaseModel):
    """One candidate image: where it is, and what it is called under its parent.

    Both halves are needed and neither derives from the other. ``path`` is what
    a symlink points at; ``relative_path`` is the identity recorded on the
    subset artifact and joined against a grouping CSV.
    ``scan_directory_structure`` treats one level of subdirectories as separate
    datasets, so a bare filename cannot disambiguate two datasets that both
    contain ``plate_001.tif`` (§10.2) — which is why the relative path, not the
    name, is the key everywhere.

    Args:
        path: Location of the image on disk.
        relative_path: POSIX path relative to the subset's parent directory,
            e.g. ``plateA/plateA_01.tif``.

    Example:
        >>> ref = ImageRef(path="/data/plates/plateA/plateA_01.tif",
        ...                relative_path="plateA/plateA_01.tif")
        >>> ref.dataset
        'plateA'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    relative_path: str

    @property
    def dataset(self) -> str:
        """The dataset name ``scan_directory_structure`` would derive.

        The first path segment for a nested parent, and ``""`` for a flat one
        (where the dataset name comes from the directory itself, which an
        ``ImageRef`` does not know).
        """
        head, separator, _ = self.relative_path.partition("/")
        return head if separator else ""


class SubsetSelection(BaseModel):
    """A chosen subset, with everything needed to explain and reproduce it.

    Becomes ``selection`` on the subset artifact (§10.2). Frozen because the
    artifact is written from it: a selection that could be edited afterwards is
    a recorded provenance that need not match what actually ran.

    **Frozen through the values, not just against assignment.**
    ``ConfigDict(frozen=True)`` stops ``selection.group_filter = other`` and
    nothing more, so plain ``dict`` fields left the mapping that matters most
    editable in place. USER-21 binds a human's approval to a specific
    ``group_filter``; an ack spendable on another group by one item assignment
    is not an ack. The mapping fields are therefore immutable proxies over
    deep-frozen copies — including :attr:`params`, which carries a nested copy
    of the selector's own ``group_filter``. They compare equal to the plain
    dicts they replace and serialize back to them.

    Constructible **directly**, not only as ``select()``'s return value — a
    hand-picked ``user_named`` subset is a first-class selection method with no
    selector behind it, and it must be able to carry a ``group_filter`` too
    (GEN-37). Without that, a human who hand-picks one group's plates and
    promotes to full scope deploys that group's pipeline over every group, with
    every digest check passing.

    Args:
        images: Parent-relative POSIX paths, deduplicated and sorted.
        method: The selector's class name, or ``"user_named"``.
        params: The selector's serialized parameters, for the artifact.
        seed: The RNG seed that reproduces this selection.
        group_filter: The ``{column: value}`` map the candidate set was
            restricted to before selection, or ``{}`` for unfiltered.
        allocation: Group name → images taken, for a stratified selection;
            ``{}`` when the method does not stratify.
        rationale: Human-readable explanation, so the artifact explains itself.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    images: tuple[str, ...]
    method: str
    params: FrozenParams = Field(default_factory=dict)
    seed: int = 0
    group_filter: FrozenStrMap = Field(default_factory=dict)
    allocation: FrozenIntMap = Field(default_factory=dict)
    rationale: str = ""


class SubsetSelector(BaseModel, ABC):
    """Choose a development subset from a parent image set.

    Subclasses implement :meth:`_select` only. The base
    :meth:`select` is a template method that applies :attr:`group_filter`,
    refuses if the selector reports itself unavailable, delegates, and then
    deduplicates, orders, and records the rationale — so no subclass can skip
    the filter or reimplement it differently.

    Args:
        n: Target subset size, and a **ceiling**. A candidate set smaller than
            ``n`` is taken whole rather than erroring — ``n`` is a target, not
            a contract — but no selector ever returns *more* than ``n``, since
            bounding compute is what the subset boundary is for. A subclass
            constraint that cannot be satisfied within ``n`` raises
            :class:`GroupFloorsExceedTarget` rather than overrunning it.
        seed: RNG seed; recorded on the artifact so a selection is reproducible.
        grouping_metadata: CSV supplying the metadata columns
            :attr:`group_filter` (and any stratification) reads, joined to
            images by **parent-relative path**. Named distinctly on purpose:
            three different CSVs appear in this design, and passing the wrong
            one at ``QCScorer.check.metadata`` produces a meaningless objective
            rather than an error. That field ships today with no alias and
            **keeps its name** — this naming choice does not extend to it.
        group_filter: ``{metadata column: value}`` restricting the candidate
            set *before* any selector runs. Conjunctive; values compare as
            strings against the CSV cell after the central ``Metadata_``
            canonicalization. Empty means the whole parent. A non-empty filter
            with no ``grouping_metadata`` is a construction error, not an empty
            result.

    Note:
        ``model_config`` forbids extra keys, so a field added later is a schema
        change to every selector's serialized ``params``. That is why
        ``group_filter`` ships on the ABC now rather than being deferred: it
        cannot arrive as an extra key on an artifact that already exists on
        disk.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    n: int = Field(..., ge=1)
    seed: int = 0
    grouping_metadata: str | None = None
    group_filter: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _group_filter_needs_a_csv(self) -> "SubsetSelector":
        """A filter no selector can evaluate must not be constructible.

        The alternative — accepting it and selecting from everything — is the
        silent-widening failure ``group_filter`` exists to prevent.
        """
        if self.group_filter and self.grouping_metadata is None:
            raise ValueError(
                "group_filter needs grouping_metadata: there is no source for "
                f"the columns {sorted(self.group_filter)}"
            )
        return self

    # -- subclass hook ---------------------------------------------------

    @abstractmethod
    def _select(self, candidates: list[ImageRef]) -> list[str]:
        """Choose from ``candidates``, returning parent-relative paths.

        Called by :meth:`select` with the candidate set **already filtered** by
        :attr:`group_filter`, so an implementation never applies it itself.

        Args:
            candidates: The filtered candidate images.

        Returns:
            Parent-relative POSIX paths. Order and duplicates do not matter;
            :meth:`select` deduplicates and sorts.
        """

    # -- reportable properties -------------------------------------------

    def availability(self) -> tuple[bool, str]:
        """Whether this selector can select at all, and why not (default: yes).

        Mirrors ``Scorer.availability()`` so ``subset_generate`` can report
        which selectors are usable *before* the agent commits — the same
        affordance that stops the most common tuning failure. It returns a
        reason string where the scorer returns a bare bool, because the agent
        has to be told what is missing.

        ``False`` means the selector cannot select **at all**, which in v1 is
        only ever an unimplemented backend; :meth:`select` therefore turns it
        into ``NotImplementedError``. A selector that is merely *misconfigured*
        must raise at construction (as a filter with no CSV does) or from
        :meth:`_select` with a specific :class:`SubsetMetadataError` — not
        report itself unavailable.

        Returns:
            ``(usable, reason)``; ``reason`` is ``""`` when usable.
        """
        return True, ""

    def cost_class(self) -> SelectorCostClass:
        """What running this selector costs (default: ``"W0"``).

        Selection cost is not uniform and the difference is structural: reading
        a file list or a metadata CSV is a planning step, while embedding every
        parent image is a scheduled job. Reporting it is what keeps an
        expensive selector from being smuggled into triage.
        """
        return "W0"

    # -- template --------------------------------------------------------

    def select(self, candidates: Sequence[ImageRef]) -> SubsetSelection:
        """Choose a subset and record how it was chosen.

        Args:
            candidates: Every image in the parent, as ``ImageRef``s.

        Returns:
            A :class:`SubsetSelection` carrying the chosen parent-relative
            paths, this selector's params and seed, the applied
            ``group_filter``, and a rationale.

        Raises:
            NotImplementedError: If :meth:`availability` reports ``False``.
            GroupFilterColumnNotFound: If ``group_filter`` names a column the
                grouping CSV does not have.
            GroupFilterMatchesNothing: If ``group_filter`` matches no candidate.
        """
        pool = list(candidates)
        filtered = self._apply_group_filter(pool)

        available, reason = self.availability()
        if not available:
            raise NotImplementedError(reason)

        chosen = self._select(filtered)
        eligible = {ref.relative_path for ref in filtered}
        images = tuple(sorted(set(chosen) & eligible))

        return SubsetSelection(
            images=images,
            method=type(self).__name__,
            params=self.model_dump(mode="json"),
            seed=self.seed,
            group_filter=dict(self.group_filter),
            allocation=self._allocation_of(filtered, images),
            rationale=self._rationale(pool, filtered, images),
        )

    # -- shared machinery ------------------------------------------------

    def _apply_group_filter(self, candidates: list[ImageRef]) -> list[ImageRef]:
        """Restrict ``candidates`` to the rows matching every filter pair."""
        if not self.group_filter:
            return list(candidates)

        rows = self._read_grouping_metadata()
        wanted = {ensure_metadata_prefix(key): value
                  for key, value in self.group_filter.items()}
        columns = self._metadata_columns(rows)
        for column in wanted:
            if column not in columns:
                raise GroupFilterColumnNotFound(column, sorted(columns))

        kept = [
            ref for ref in candidates
            if all(rows.get(ref.relative_path, {}).get(column) == value
                   for column, value in wanted.items())
        ]
        if not kept:
            raise GroupFilterMatchesNothing(dict(self.group_filter))
        return kept

    def _read_grouping_metadata(self) -> dict[str, dict[str, str]]:
        """Read the grouping CSV as ``{relative path: {column: cell}}``.

        Read with the stdlib reader rather than a DataFrame library on purpose:
        every cell must stay the **string** it is on disk, and a dtype-inferring
        reader turns a numeric batch label into an int that no string filter
        value can ever equal.

        Column names are canonicalized through the central metadata helper, so
        ``Batch`` and ``Metadata_Batch`` are one column. The identity column is
        matched literally against :data:`IMAGE_IDENTITY_COLUMNS` instead —
        it names a file, not a measurement.
        """
        if self.grouping_metadata is None:
            return {}
        path = Path(self.grouping_metadata)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or ())
            identity = next(
                (name for name in fieldnames
                 if name.strip().lower() in IMAGE_IDENTITY_COLUMNS),
                None,
            )
            if identity is None:
                raise SubsetMetadataError(
                    f"{path} has no image identity column; expected one of "
                    f"{', '.join(IMAGE_IDENTITY_COLUMNS)} but found "
                    f"{', '.join(fieldnames) or '(no header)'}"
                )
            rows: dict[str, dict[str, str]] = {}
            for row in reader:
                key = (row.get(identity) or "").strip().replace("\\", "/")
                if not key:
                    continue
                rows[key] = {
                    ensure_metadata_prefix(name): (value or "").strip()
                    for name, value in row.items()
                    if name is not None and name != identity
                }
        return rows

    @staticmethod
    def _metadata_columns(rows: dict[str, dict[str, str]]) -> set[str]:
        """Every canonicalized column name appearing in ``rows``."""
        columns: set[str] = set()
        for row in rows.values():
            columns.update(row)
        return columns

    def _rng(self) -> random.Random:
        """A generator seeded only by :attr:`seed`, so a selection reproduces."""
        return random.Random(self.seed)

    def _allocation_of(
        self, candidates: list[ImageRef], images: tuple[str, ...]
    ) -> dict[str, int]:
        """Per-group counts for a stratified selector; ``{}`` by default."""
        return {}

    def _rationale(
        self,
        candidates: list[ImageRef],
        filtered: list[ImageRef],
        images: tuple[str, ...],
    ) -> str:
        """One sentence explaining the selection, for the artifact."""
        parts = [
            f"{type(self).__name__} chose {len(images)} of {len(filtered)} "
            f"candidate images (target n={self.n}, seed={self.seed})"
        ]
        if self.group_filter:
            pairs = ", ".join(
                f"{key}={value}" for key, value in sorted(self.group_filter.items())
            )
            parts.append(
                f"group_filter {pairs} narrowed the parent's {len(candidates)} "
                f"images to {len(filtered)}"
            )
        return "; ".join(parts)
