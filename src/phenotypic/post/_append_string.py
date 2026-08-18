from __future__ import annotations

import pandas as pd
from pydantic import field_validator

from phenotypic.abc_._post_measurement import PostMeasurement
from ._utils import (
    affix_preserving_na,
    coalesce_metadata_aliases,
    ensure_metadata_prefix,
    resolve_metadata_column,
)


class AppendString(PostMeasurement):
    """Append a string to every value in a metadata column.

    Converts each cell to a string and concatenates the given value to the
    end. Useful for adding suffixes like units or experimental labels.

    Args:
        column: Name of the metadata column to modify. The schema category
            prefix is added automatically if missing (e.g. ``Temperature`` ->
            ``Metadata_Temperature``; unknown labels get a generic
            ``Metadata_`` prefix).
        value: The string to append to each cell value.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with the modified column.

    Notes:
        Missing values (``NA``/``NaN``) stay missing — they are not stringified
        into ``"nanC"``. Only present values get the suffix.

    Raises:
        KeyError: If the column does not exist in the DataFrame.

    Examples:
        Append a unit suffix to a temperature column:

        >>> import pandas as pd
        >>> from phenotypic.post import AppendString
        >>> from phenotypic.schema import CULTURE
        >>> temperature = str(CULTURE.TEMPERATURE)
        >>> df = pd.DataFrame({
        ...     temperature: ["30", "37"],
        ...     "Object_Label": [1, 2],
        ... })
        >>> op = AppendString(column="Temperature", value="C")
        >>> result = op.apply(df)
        >>> list(result[temperature])
        ['30C', '37C']

        An undetected colony's missing value stays missing:

        >>> df.loc[1, temperature] = None
        >>> list(op.apply(df)[temperature])
        ['30C', nan]
    """

    column: str = ""
    value: str = ""

    @field_validator("column")
    @classmethod
    def _prefix_column(cls, column: str) -> str:
        """Apply the schema category prefix (generic ``Metadata_`` fallback) to a non-empty column name."""
        return ensure_metadata_prefix(column) if column else ""

    def _operate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append the string value to each cell in the target column.

        Args:
            df: Measurement DataFrame containing the target column.

        Returns:
            DataFrame with the modified column. NA cells are left as NA.
        """
        try:
            result = coalesce_metadata_aliases(df, [self.column])
            source_column = resolve_metadata_column(result.columns, self.column)
        except KeyError:
            raise KeyError(
                f"Column '{self.column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            ) from None
        result[source_column] = affix_preserving_na(
            result[source_column], suffix=self.value
        )
        return result
