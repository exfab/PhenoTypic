from __future__ import annotations

import pandas as pd

from phenotypic.abc_._post_measurement import PostMeasurement
from ._utils import _ensure_prefix


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
        ...     "ObjectLabel": [1, 2],
        ... })
        >>> op = AppendString(column="Temp", value="C")
        >>> result = op.apply(df)
        >>> list(result["Metadata_Temp"])
        ['30C', '37C']
    """

    def __init__(self, column: str = "", value: str = ""):
        super().__init__()
        self.column = _ensure_prefix(column) if column else ""
        self.value = value

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
