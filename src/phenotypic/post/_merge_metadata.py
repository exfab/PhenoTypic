from __future__ import annotations

from typing import List, cast

import pandas as pd
from pydantic import field_validator

from phenotypic.abc_._post_measurement import PostMeasurement
from ._utils import (
    coalesce_metadata_aliases,
    ensure_metadata_prefix,
    resolve_metadata_column,
)


class MergeMetadata(PostMeasurement):
    """Merge multiple metadata columns into a single new metadata column.

    Concatenates the string values of two or more metadata columns using
    a delimiter to produce a combined identifier. Useful for creating
    composite keys (e.g., combining Strain and Condition into SampleID).

    Args:
        columns: Names of the metadata columns to merge. The schema
            category prefix is added automatically if missing (e.g.
            ``Strain`` -> ``Metadata_Strain``; unknown labels get a
            generic ``Metadata_`` prefix). Must contain at least 2 names.
        label: Name for the new merged column. The schema category prefix
            is added automatically if missing.
        delimiter: String used to join the column values. Defaults to
            ``"_"``.

    Returns:
        pd.DataFrame: The input DataFrame with the new merged column
            inserted after the last source column. All source columns
            are kept.

    Notes:
        Rows where *any* source column is missing (``NA``/``NaN``) merge to
        ``NaN`` rather than to a string containing the literal ``"nan"``.
        This matters for undetected colonies, whose measurement columns are
        null: a merged key like ``"nan_A1"`` would look valid and silently
        group unrelated rows together. Missing stays missing.

    Raises:
        ValueError: If columns contains fewer than 2 names.
        KeyError: If any source column does not exist in the DataFrame.

    Examples:
        Merge strain and condition into a sample ID:

        >>> import pandas as pd
        >>> from phenotypic.post import MergeMetadata
        >>> from phenotypic.schema import GENETIC, SAMPLE
        >>> strain = str(GENETIC.STRAIN)
        >>> sample_id = str(SAMPLE.SAMPLE_ID)
        >>> df = pd.DataFrame({
        ...     strain: ["WT", "mut"],
        ...     "Metadata_Condition": ["30C", "37C"],
        ...     "Object_Label": [1, 2],
        ... })
        >>> merge = MergeMetadata(
        ...     columns=["Strain", "Condition"],
        ...     label="SampleID",
        ...     delimiter="_",
        ... )
        >>> result = merge.apply(df)
        >>> list(result[sample_id])
        ['WT_30C', 'mut_37C']

        A missing source value yields a missing merged key, never
        ``'nan_37C'``:

        >>> df.loc[1, strain] = None
        >>> list(merge.apply(df)[sample_id])
        ['WT_30C', nan]
    """

    columns: List[str] = []
    label: str = ""
    delimiter: str = "_"

    @field_validator("columns", mode="before")
    @classmethod
    def _prefix_columns(cls, columns: List[str] | None) -> List[str]:
        """Apply the schema category prefix (generic ``Metadata_`` fallback) and reject a single-column merge.

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
        """Apply the schema category prefix (generic ``Metadata_`` fallback) to a non-empty label."""
        return ensure_metadata_prefix(label) if label else ""

    def _operate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge the specified columns into a new column.

        Args:
            df: Measurement DataFrame containing the source columns.

        Returns:
            DataFrame with the new merged column inserted after the last
            source column. Rows with an NA in any source column get NaN.
        """
        try:
            result = coalesce_metadata_aliases(df, [*self.columns, self.label])
            source_columns = [
                resolve_metadata_column(result.columns, column) for column in self.columns
            ]
        except KeyError as exc:
            raise KeyError(
                f"Column '{exc.args[0]}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            ) from None

        # Join column values with delimiter
        merged = result[source_columns[0]].astype(str)
        for col in source_columns[1:]:
            merged = merged + self.delimiter + result[col].astype(str)

        # `.astype(str)` turns NA into the literal "nan", which would produce a
        # valid-looking key (e.g. "nan_A1"). Blank the merged value instead so a
        # missing source stays missing. No-op when nothing is NA.
        merged = merged.mask(result[source_columns].isna().any(axis=1))

        # Insert after the last source column
        last_pos = max(
            cast(int, result.columns.get_loc(column)) for column in source_columns
        )
        try:
            target_column = resolve_metadata_column(result.columns, self.label)
        except KeyError:
            result.insert(last_pos + 1, self.label, merged)
        else:
            result[target_column] = merged

        return result
