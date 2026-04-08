from __future__ import annotations

import re
from typing import List

import pandas as pd

from phenotypic.abc_._post_measurement import PostMeasurement
from ._utils import _ensure_prefix


class ExpandMetadata(PostMeasurement):
    """Split a metadata column into multiple new metadata columns.

    Takes a single delimited metadata column (e.g., a filename encoding
    experimental conditions) and expands it into separate labeled columns.
    Every row must produce exactly len(labels) parts or a ValueError is
    raised.

    Args:
        column: Name of the metadata column to split. ``Metadata_`` prefix
            is added automatically if missing.
        labels: Names for the resulting columns, one per split part.
            ``Metadata_`` prefix is added automatically if missing.
        delimiter: String or regex pattern to split on. Defaults to ``"_"``.
        regex: If True, treat delimiter as a regex pattern. Defaults to
            False.

    Returns:
        pd.DataFrame: The input DataFrame with new columns inserted
            adjacent to the source column. The source column is always
            kept.

    Raises:
        ValueError: If labels is empty, or if any row produces a different
            number of parts than len(labels).
        KeyError: If the source column does not exist in the DataFrame.

    Examples:
        Split a filename into experimental conditions:

        >>> import pandas as pd
        >>> from phenotypic.post import ExpandMetadata
        >>> df = pd.DataFrame({
        ...     "Metadata_ImageName": ["WT_30C_24h", "mut_37C_48h"],
        ...     "ObjectLabel": [1, 2],
        ... })
        >>> expand = ExpandMetadata(
        ...     column="ImageName",
        ...     labels=["Strain", "Condition", "Time"],
        ...     delimiter="_",
        ... )
        >>> result = expand.apply(df)
        >>> list(result["Metadata_Strain"])
        ['WT', 'mut']
    """

    def __init__(
        self,
        column: str = "",
        labels: List[str] | None = None,
        delimiter: str = "_",
        regex: bool = False,
    ):
        super().__init__()
        if labels is not None and not labels:
            raise ValueError("labels must be a non-empty list")
        self.column = _ensure_prefix(column) if column else ""
        self.labels = [_ensure_prefix(lbl) for lbl in labels] if labels else []
        self.delimiter = delimiter
        self.regex = regex

    def _operate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split the metadata column and insert new columns.

        Args:
            df: Measurement DataFrame containing the source column.

        Returns:
            DataFrame with new columns inserted after the source column.
        """
        if self.column not in df.columns:
            raise KeyError(
                f"Column '{self.column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        # Split the column values
        if self.regex:
            parts = df[self.column].apply(lambda x: re.split(self.delimiter, str(x)))
        else:
            parts = df[self.column].str.split(self.delimiter)

        # Validate that every row has the expected number of parts
        n_expected = len(self.labels)
        counts = parts.apply(len)
        bad_mask = counts != n_expected
        if bad_mask.any():
            first_bad_idx = bad_mask.idxmax()
            first_bad_val = df[self.column].iloc[first_bad_idx]
            first_bad_count = counts.iloc[first_bad_idx]
            raise ValueError(
                f"Column '{self.column}' split produced {first_bad_count} parts "
                f"for value '{first_bad_val}', but {n_expected} labels were "
                f"provided: {self.labels}"
            )

        # Build new columns DataFrame
        split_df = pd.DataFrame(parts.tolist(), columns=self.labels, index=df.index)

        # Insert new columns after the source column
        src_pos = df.columns.get_loc(self.column)
        result = df.copy()
        for i, label in enumerate(self.labels):
            result.insert(src_pos + 1 + i, label, split_df[label])

        return result
