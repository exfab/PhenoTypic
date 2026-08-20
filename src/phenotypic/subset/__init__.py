"""Development subsets — choosing which images the development loop spends on.

Everything from triage through campaign execution runs on a **subset**; the
full dataset is touched exactly once, after an explicit human promotion
(§10.1). This subpackage owns the strategy that produces one.

Selectors are pydantic models resolvable by bare class name, exactly like
operations, analyzers and scorers, so ``{class, params}`` round-trips and
adding a fourth selector is a subclass plus one export here.

Example:
    >>> from phenotypic.subset import ImageRef, RandomSubsetSelector
    >>> refs = [ImageRef(path=f"/plates/plateA/p{i:02d}.tif",
    ...                  relative_path=f"plateA/p{i:02d}.tif")
    ...         for i in range(12)]
    >>> selection = RandomSubsetSelector(n=4, seed=0).select(refs)
    >>> selection.method
    'RandomSubsetSelector'
    >>> len(selection.images)
    4
"""

from ._selector import (
    GroupAllocation,
    GroupFilterColumnNotFound,
    GroupFilterMatchesNothing,
    GroupKeyNotInMetadata,
    ImageRef,
    IMAGE_IDENTITY_COLUMNS,
    SelectorCostClass,
    SubsetMetadataError,
    SubsetSelection,
    SubsetSelector,
)
from ._selectors import (
    EmbeddingSubsetSelector,
    MetadataGroupSubsetSelector,
    RandomSubsetSelector,
)

__all__ = [
    "EmbeddingSubsetSelector",
    "GroupAllocation",
    "GroupFilterColumnNotFound",
    "GroupFilterMatchesNothing",
    "GroupKeyNotInMetadata",
    "IMAGE_IDENTITY_COLUMNS",
    "ImageRef",
    "MetadataGroupSubsetSelector",
    "RandomSubsetSelector",
    "SelectorCostClass",
    "SubsetMetadataError",
    "SubsetSelection",
    "SubsetSelector",
]
