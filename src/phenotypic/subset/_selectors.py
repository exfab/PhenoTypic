"""The three shipped ``SubsetSelector`` implementations.

``RandomSubsetSelector`` is the honest default and the right *baseline* even
when metadata exists; ``MetadataGroupSubsetSelector`` stratifies across a
grouping already recorded on disk; ``EmbeddingSubsetSelector`` is a declared
placeholder that fails loudly rather than degrading.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field, model_validator

from phenotypic.sdk_._metadata_helpers import ensure_metadata_prefix

from ._selector import (
    GroupAllocation,
    GroupFloorsExceedTarget,
    GroupKeyNotInMetadata,
    ImageRef,
    SelectorCostClass,
    SubsetSelector,
)

_EMBEDDING_UNAVAILABLE: Final[str] = (
    "EmbeddingSubsetSelector is not implemented; no embedding backend is "
    "configured"
)


class RandomSubsetSelector(SubsetSelector):
    """Uniform sampling without replacement, seeded.

    Honest and unstratified: the right default when no metadata exists, and the
    right baseline even when it does — a stratified subset that does not beat a
    random one of the same size bought nothing.

    Composes with :attr:`~SubsetSelector.group_filter` without knowing anything
    about metadata, because the filter is applied by the ABC's template before
    this class is asked for anything. "Random within one species" is therefore
    this selector plus a filter, not a second selector.

    Args:
        n: Target subset size.
        seed: RNG seed; the same seed over the same candidate set reproduces
            the selection exactly.
        grouping_metadata: Only needed when ``group_filter`` is set.
        group_filter: See :class:`~SubsetSelector`.

    Example:
        >>> from phenotypic.subset import ImageRef, RandomSubsetSelector
        >>> refs = [ImageRef(path=f"/plates/plateA/p{i:02d}.tif",
        ...                  relative_path=f"plateA/p{i:02d}.tif")
        ...         for i in range(10)]
        >>> selection = RandomSubsetSelector(n=3, seed=0).select(refs)
        >>> len(selection.images)
        3
        >>> selection.images == RandomSubsetSelector(n=3, seed=0).select(refs).images
        True
    """

    def _select(self, candidates: list[ImageRef]) -> list[str]:
        """Sample ``n`` of ``candidates``, or all of them if there are fewer."""
        paths = sorted(ref.relative_path for ref in candidates)
        if len(paths) <= self.n:
            return paths
        return self._rng().sample(paths, self.n)


class MetadataGroupSubsetSelector(SubsetSelector):
    """Sample across the groups named by a column of a metadata CSV.

    Stratification that needs no measurement: the grouping already exists on
    disk, so this is a ``W0`` planning step rather than the whole-dataset
    characterization trait or embedding stratification would require.

    **It performs its own CSV→filename join and does not reuse
    ``_resolve_groups``.** That helper (``tune/_evaluation/_split.py``) is a
    pure in-memory ``image.metadata.get(group_key)`` lookup with no CSV and no
    join, and a freshly read image carries only its
    ``MetadataImage_*`` fields — so an externally-sourced key resolves to
    ``None`` and the tune split falls through to a weaker tier **silently**.
    External CSV columns reach data only via ``join_metadata``, which runs on
    the measurement DataFrame after a full run: the opposite end of the
    pipeline from Phase-0 triage. Stratifying a subset therefore does not
    change how the held-out split is derived; that would need an engine change
    populating ``image.metadata[group_key]`` at tune load time.

    Args:
        n: Target subset size.
        seed: RNG seed; recorded so the selection reproduces.
        grouping_metadata: CSV supplying :attr:`group_key`, joined to images by
            parent-relative path. Required for this selector.
        group_key: The column in ``grouping_metadata`` naming each image's
            group. Resolved through the central metadata canonicalization, so
            ``Batch`` and ``Metadata_Batch`` are the same column.
        allocation: ``"proportional"`` mirrors group sizes; ``"equal"`` gives
            every group the same count, so a rare condition is not swamped.
        min_per_group: Floor per group. Groups smaller than it are taken whole.
            Bounded by ``n``: the two are separate constraints and can
            conflict, and when they do the selector raises rather than
            silently overrunning the budget. See
            :class:`~GroupFloorsExceedTarget` for which one wins and why.
        group_filter: See :class:`~SubsetSelector`. Applied first, so
            ``allocation`` stratifies *inside* the filtered set.

    Raises:
        GroupKeyNotInMetadata: From :meth:`select`, if ``group_key`` resolves no
            group for any candidate — including the case where the CSV has the
            column but is keyed by bare filename and so joins to nothing.
        GroupFloorsExceedTarget: From :meth:`select`, if ``min_per_group``
            cannot be honoured across the resolved groups within ``n``.
    """

    group_key: str
    allocation: GroupAllocation = "proportional"
    min_per_group: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _requires_grouping_metadata(self) -> "MetadataGroupSubsetSelector":
        """Without a CSV there is no grouping, and no way to say so later."""
        if self.grouping_metadata is None:
            raise ValueError(
                "MetadataGroupSubsetSelector requires grouping_metadata: "
                f"there is no source for the column {self.group_key!r}"
            )
        return self

    def _select(self, candidates: list[ImageRef]) -> list[str]:
        """Allocate ``n`` across the groups, then sample inside each."""
        groups = self._groups_of(candidates)
        allocation = self._allocate({name: len(refs) for name, refs in groups.items()})
        rng = self._rng()

        chosen: list[str] = []
        for name in sorted(groups):
            paths = sorted(ref.relative_path for ref in groups[name])
            take = allocation.get(name, 0)
            chosen.extend(paths if take >= len(paths) else rng.sample(paths, take))
        return chosen

    def _groups_of(self, candidates: list[ImageRef]) -> dict[str, list[ImageRef]]:
        """Group ``candidates`` by their ``group_key`` cell.

        Images the CSV does not cover are left out of the grouping rather than
        pooled into an implicit group — a stratification over a group that only
        means "unknown" is not the stratification the caller asked for. If that
        leaves nothing, the selector raises instead of stratifying over one
        implicit group and reporting success.
        """
        rows = self._read_grouping_metadata()
        column = ensure_metadata_prefix(self.group_key)

        groups: dict[str, list[ImageRef]] = {}
        for ref in candidates:
            value = rows.get(ref.relative_path, {}).get(column, "")
            if not value:
                continue
            groups.setdefault(value, []).append(ref)

        if not groups:
            raise GroupKeyNotInMetadata(
                column, sorted(self._metadata_columns(rows))
            )
        return groups

    def _allocate(self, group_sizes: dict[str, int]) -> dict[str, int]:
        """Spread ``n`` across the groups, honouring caps and the floor.

        Three constraints have to hold at once and they can conflict: the
        requested :attr:`allocation` shape, :attr:`min_per_group`, and the fact
        that a group cannot yield more images than it has. So the shape is
        computed first, then clamped between the floor and the group's size,
        and the residual is rebalanced deterministically until the total is the
        target again. Ties break on the group name so the result does not
        depend on dict ordering.

        Args:
            group_sizes: Group name → number of candidates in it.

        Returns:
            Group name → images to take, summing to ``min(n, total)`` whenever
            the caps allow it, and **never** more than ``n``.

        Raises:
            GroupFloorsExceedTarget: If the floors alone need more than ``n``.
        """
        names = sorted(group_sizes)
        total = sum(group_sizes.values())
        if not names or total == 0:
            return {}
        target = min(self.n, total)

        if self.allocation == "equal":
            base, remainder = divmod(target, len(names))
            raw = {
                name: base + (1 if index < remainder else 0)
                for index, name in enumerate(names)
            }
        else:
            exact = {name: target * group_sizes[name] / total for name in names}
            raw = {name: int(exact[name]) for name in names}
            shortfall = target - sum(raw.values())
            by_fraction = sorted(
                names, key=lambda name: (-(exact[name] - raw[name]), name)
            )
            for name in by_fraction[:shortfall]:
                raw[name] += 1

        # Clamped to each group's size first: a floor of 3 over a group of 2
        # asks for an image that does not exist, and the docstring already
        # promises such a group is taken whole.
        floors = {name: min(self.min_per_group, group_sizes[name]) for name in names}
        required = sum(floors.values())
        if required > target:
            raise GroupFloorsExceedTarget(
                n=self.n,
                min_per_group=self.min_per_group,
                required=required,
                group_sizes=group_sizes,
            )
        allocation = {
            name: min(group_sizes[name], max(raw[name], floors[name]))
            for name in names
        }

        while sum(allocation.values()) > target:
            reducible = [name for name in names if allocation[name] > floors[name]]
            if not reducible:
                break
            victim = max(reducible, key=lambda name: (allocation[name], name))
            allocation[victim] -= 1
        while sum(allocation.values()) < target:
            growable = [name for name in names if allocation[name] < group_sizes[name]]
            if not growable:
                break
            winner = min(growable, key=lambda name: (allocation[name], name))
            allocation[winner] += 1
        return allocation

    def _allocation_of(
        self, candidates: list[ImageRef], images: tuple[str, ...]
    ) -> dict[str, int]:
        """Per-group counts of what was **actually** selected.

        Recounted from the selection rather than returned from
        :meth:`_allocate`, so the artifact records what happened instead of
        what was planned.
        """
        selected = set(images)
        counts: dict[str, int] = {}
        for name, refs in self._groups_of(candidates).items():
            taken = sum(1 for ref in refs if ref.relative_path in selected)
            if taken:
                counts[name] = taken
        return counts


class EmbeddingSubsetSelector(SubsetSelector):
    """Placeholder for visual-coverage selection — **it fails loudly**.

    Intended shape: embed every parent image with a vision model, cluster, and
    take medoids, giving visual coverage with no metadata and no hand-labelling.

    Until that exists, :meth:`availability` reports ``False`` and both
    :meth:`select` and :meth:`_select` raise. It does **not** fall back to
    random. A placeholder that quietly degraded to a different strategy is the
    worst failure available here: the artifact would record
    ``method: EmbeddingSubsetSelector``, the agent and the human would both
    believe the subset had visual coverage, and nothing downstream could
    contradict them. A check that cannot run must fail rather than skip.

    :meth:`cost_class` already returns ``"W2"`` while unimplemented, so the
    routing story is correct when it lands: embedding a 480-image parent is a
    scheduled job, not a planning step.

    Args:
        n: Target subset size.
        seed: RNG seed.
        model: Embedding backend identifier. Unset until a backend exists.
        strategy: Cluster-and-take-medoids is the intended shape.
    """

    model: str | None = None
    strategy: str = "kmeans_medoids"

    def availability(self) -> tuple[bool, str]:
        """Always ``(False, …)`` — there is no embedding backend."""
        return False, _EMBEDDING_UNAVAILABLE

    def cost_class(self) -> SelectorCostClass:
        """``"W2"``: it would have to encode **every parent image**."""
        return "W2"

    def _select(self, candidates: list[ImageRef]) -> list[str]:
        """Raise. The second barrier, in case ``select`` is bypassed."""
        raise NotImplementedError(_EMBEDDING_UNAVAILABLE)
