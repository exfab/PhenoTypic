"""Post-measurement DataFrame transforms for measurement pipelines.

Provides operations that reshape or enrich measurement DataFrames after
feature extraction. These run as the final stage of ImagePipeline.measure().
"""

from ._expand_metadata import ExpandMetadata
from ._merge_metadata import MergeMetadata

__all__ = [
    "ExpandMetadata",
    "MergeMetadata",
]
