from __future__ import annotations

from typing import List

import pandas as pd
from pydantic import field_validator

from phenotypic.abc_._post_measurement import PostMeasurement
from ._utils import ensure_metadata_prefix


class MergeMetadata(PostMeasurement):
    """Merge multiple metadata columns into a single new metadata column.

    Concatenates the string values of two or more metadata columns using
    a delimiter to produce a combined identifier. Useful for creating
    composite keys (e.g., combining Strain and Condition into SampleID).

    Args:
        columns: Names of the metadata columns to merge. ``Metadata_``
            prefix is added automatically if missing. Must contain at
            least 2 names.
        label: Name for the new merged column. ``Metadata_`` prefix is
            added automatically if missing.
        delimiter: String used to join the column values. Defaults to
            ``"_"``.

    Returns:
        pd.DataFrame: The input DataFrame with the new merged column
            inserted after the last source column. All source columns
            are kept.

    Raises:
        ValueError: If columns contains fewer than 2 names.
        KeyError: If any source column does not exist in the DataFrame.

    Examples:
        Merge strain and condition into a sample ID:

        >>> import pandas as pd
        >>> from phenotypic.post import MergeMetadata
        >>> df = pd.DataFrame({
        ...     "Metadata_Strain": ["WT", "mut"],
        ...     "Metadata_Condition": ["30C", "37C"],
        ...     "Object_Label": [1, 2],
        ... })
        >>> merge = MergeMetadata(
        ...     columns=["Strain", "Condition"],
        ...     label="SampleID",
        ...     delimiter="_",
        ... )
        >>> result = merge.apply(df)
        >>> list(result["Metadata_SampleID"])
        ['WT_30C', 'mut_37C']
    """

    columns: List[str] = []
    label: str = ""
    delimiter: str = "_"

    @field_validator("columns", mode="before")
    @classmethod
    def _prefix_columns(cls, columns: List[str] | None) -> List[str]:
        """Add the ``Metadata_`` prefix and reject a single-column merge.

        Accepts ``None``/``[]`` (the "unset" state) and normalizes to an
        empty list. A genuinely-invalid *single*-column list raises; the
        empty default validates cleanly so ``model_validate`` / assignment
        round-trips work.
        """
        if columns and len(columns) < 2:
            raise ValueError("columns must contain at least 2 column names to merge")
        return [ensure_metadata_prefix(c) for c in columns] if columns else []

    @field_validator("label")
    @classmethod
    def _prefix_label(cls, label: str) -> str:
        """Prepend the ``Metadata_`` prefix to a non-empty label."""
        return ensure_metadata_prefix(label) if label else ""

    def _operate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge the specified columns into a new column.

        Args:
            df: Measurement DataFrame containing the source columns.

        Returns:
            DataFrame with the new merged column inserted after the last
            source column.
        """
        # Validate all source columns exist
        for col in self.columns:
            if col not in df.columns:
                raise KeyError(
                    f"Column '{col}' not found in DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )

        # Join column values with delimiter
        merged = df[self.columns[0]].astype(str)
        for col in self.columns[1:]:
            merged = merged + self.delimiter + df[col].astype(str)

        # Insert after the last source column
        result = df.copy()
        last_pos = max(result.columns.get_loc(c) for c in self.columns)
        result.insert(last_pos + 1, self.label, merged)

        return result
