from __future__ import annotations

import pandas as pd
from pydantic import field_validator

from phenotypic.abc_._post_measurement import PostMeasurement
from ._utils import ensure_metadata_prefix


class AppendString(PostMeasurement):
    """Append a string to every value in a metadata column.

    Converts each cell to a string and concatenates the given value to the
    end. Useful for adding suffixes like units or experimental labels.

    Args:
        column: Name of the metadata column to modify. ``Metadata_`` prefix
            is added automatically if missing.
        value: The string to append to each cell value.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with the modified column.

    Raises:
        KeyError: If the column does not exist in the DataFrame.

    Examples:
        Append a unit suffix to a temperature column:

        >>> import pandas as pd
        >>> from phenotypic.post import AppendString
        >>> df = pd.DataFrame({
        ...     "Metadata_Temp": ["30", "37"],
        ...     "Object_Label": [1, 2],
        ... })
        >>> op = AppendString(column="Temp", value="C")
        >>> result = op.apply(df)
        >>> list(result["Metadata_Temp"])
        ['30C', '37C']
    """

    column: str = ""
    value: str = ""

    @field_validator("column")
    @classmethod
    def _prefix_column(cls, column: str) -> str:
        """Prepend the ``Metadata_`` prefix to a non-empty column name."""
        return ensure_metadata_prefix(column) if column else ""

    def _operate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append the string value to each cell in the target column.

        Args:
            df: Measurement DataFrame containing the target column.

        Returns:
            DataFrame with the modified column.
        """
        if self.column not in df.columns:
            raise KeyError(
                f"Column '{self.column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        result = df.copy()
        result[self.column] = result[self.column].astype(str) + self.value
        return result
