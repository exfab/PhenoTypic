from __future__ import annotations

import pandas as pd
from pydantic import field_validator

from phenotypic.abc_._post_measurement import PostMeasurement
from ._utils import affix_preserving_na, ensure_metadata_prefix


class PrependString(PostMeasurement):
    """Prepend a string to every value in a metadata column.

    Converts each cell to a string and concatenates the given value to the
    beginning. Useful for adding prefixes like identifiers or labels.

    Args:
        column: Name of the metadata column to modify. The schema category
            prefix is added automatically if missing (e.g. ``SampleID`` ->
            ``MetadataSample_SampleID``; unknown labels get a generic
            ``Metadata_`` prefix).
        value: The string to prepend to each cell value.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with the modified column.

    Notes:
        Missing values (``NA``/``NaN``) stay missing — they are not stringified
        into ``"WT-nan"``. Only present values get the prefix.

    Raises:
        KeyError: If the column does not exist in the DataFrame.

    Examples:
        Prepend a strain prefix to an ID column:

        >>> import pandas as pd
        >>> from phenotypic.post import PrependString
        >>> df = pd.DataFrame({
        ...     "MetadataSample_SampleID": ["001", "002"],
        ...     "Object_Label": [1, 2],
        ... })
        >>> op = PrependString(column="SampleID", value="WT-")
        >>> result = op.apply(df)
        >>> list(result["MetadataSample_SampleID"])
        ['WT-001', 'WT-002']

        An undetected colony's missing value stays missing:

        >>> df.loc[1, "MetadataSample_SampleID"] = None
        >>> list(op.apply(df)["MetadataSample_SampleID"])
        ['WT-001', nan]
    """

    column: str = ""
    value: str = ""

    @field_validator("column")
    @classmethod
    def _prefix_column(cls, column: str) -> str:
        """Apply the schema category prefix (generic ``Metadata_`` fallback) to a non-empty column name."""
        return ensure_metadata_prefix(column) if column else ""

    def _operate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepend the string value to each cell in the target column.

        Args:
            df: Measurement DataFrame containing the target column.

        Returns:
            DataFrame with the modified column. NA cells are left as NA.
        """
        if self.column not in df.columns:
            raise KeyError(
                f"Column '{self.column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        result = df.copy()
        result[self.column] = affix_preserving_na(
            result[self.column], prefix=self.value
        )
        return result
