"""Post-measurement DataFrame transforms for measurement pipelines.

Provides operations that reshape or enrich measurement DataFrames after
feature extraction. These run as the final stage of ImagePipeline.measure().
"""

from ._append_string import AppendString
from ._expand_metadata import ExpandMetadata
from ._merge_metadata import MergeMetadata
from ._prepend_string import PrependString

__all__ = [
    "AppendString",
    "ExpandMetadata",
    "MergeMetadata",
    "PrependString",
]
