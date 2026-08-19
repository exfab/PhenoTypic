from __future__ import annotations

import re
from typing import List, cast

import pandas as pd
from pydantic import field_validator

from phenotypic.abc_._post_measurement import PostMeasurement
from ._utils import (
    coalesce_metadata_aliases,
    ensure_metadata_prefix,
    resolve_metadata_column,
)


class ExpandMetadata(PostMeasurement):
    """Split a metadata column into multiple new metadata columns.

    Takes a single delimited metadata column (e.g., a filename encoding
    experimental conditions) and expands it into separate labeled columns.
    Every row must produce exactly len(labels) parts or a ValueError is
    raised.

    Args:
        column: Name of the metadata column to split. The schema category
            prefix is added automatically if missing (e.g. ``ImageName`` ->
            ``Metadata_ImageName``; unknown labels get a generic
            ``Metadata_`` prefix).
        labels: Names for the resulting columns, one per split part. The
            schema category prefix is added automatically if missing (e.g.
            ``Strain`` -> ``Metadata_Strain``).
        delimiter: String or regex pattern to split on. Defaults to ``"_"``.
        regex: If True, treat delimiter as a regex pattern. Defaults to
            False.

    Returns:
        pd.DataFrame: The input DataFrame with new columns inserted
            adjacent to the source column. The source column is always
            kept.

    Notes:
        Rows whose source value is missing (``NA``/``NaN``) are excluded from
        both the split and the arity check, and come back as ``NaN`` in every
        new column. An undetected colony carries no filename to expand, so it
        neither crashes the split nor trips the "wrong number of parts" guard.

    Raises:
        ValueError: If labels is empty, or if any row with a present source
            value produces a different number of parts than len(labels).
        KeyError: If the source column does not exist in the DataFrame.

    Examples:
        Split a filename into experimental conditions:

        >>> import pandas as pd
        >>> from phenotypic.post import ExpandMetadata
        >>> from phenotypic.schema import GENETIC, IMAGE
        >>> image_name = str(IMAGE.IMAGE_NAME)
        >>> strain = str(GENETIC.STRAIN)
        >>> df = pd.DataFrame({
        ...     image_name: ["WT_30C_24h", "mut_37C_48h"],
        ...     "Object_Label": [1, 2],
        ... })
        >>> expand = ExpandMetadata(
        ...     column="ImageName",
        ...     labels=["Strain", "Condition", "Time"],
        ...     delimiter="_",
        ... )
        >>> result = expand.apply(df)
        >>> list(result[strain])
        ['WT', 'mut']

        A row with no source value expands to NaN instead of raising:

        >>> df.loc[1, image_name] = None
        >>> list(expand.apply(df)[strain])
        ['WT', nan]
    """

    column: str = ""
    labels: List[str] = []
    delimiter: str = "_"
    regex: bool = False

    @field_validator("column")
    @classmethod
    def _prefix_column(cls, column: str) -> str:
        """Apply the schema category prefix (generic ``Metadata_`` fallback) to a non-empty column name."""
        return ensure_metadata_prefix(column) if column else ""

    @field_validator("labels", mode="before")
    @classmethod
    def _prefix_labels(cls, labels: List[str] | None) -> List[str]:
        """Apply the schema category prefix (generic ``Metadata_`` fallback) to each label.

        Accepts ``None``/``[]`` (the "unset" state) unchanged so the empty
        default validates cleanly (``model_validate`` / assignment
        round-trips). A genuinely-empty ``labels`` is caught at
        ``apply()`` time by ``_operate``'s split-count check.
        """
        return [ensure_metadata_prefix(lbl) for lbl in labels] if labels else []

    def _split_to_labels(self, source: pd.Series) -> pd.DataFrame:
        """Split *source* into one column per label, aligned to *source*'s index.

        Rows with no source value have nothing to split and no arity to check;
        they are reindexed back in as NaN. Splitting them would either yield the
        literal ``"nan"`` (regex branch) or a non-list NaN that breaks the
        ``len`` count (non-regex branch). No-op when nothing is NA.

        Args:
            source: The source metadata column.

        Returns:
            A DataFrame of ``self.labels`` columns on ``source``'s index; rows
            whose source value is NA are NaN throughout.

        Raises:
            ValueError: If a row with a present source value splits into a
                number of parts other than ``len(self.labels)``.
        """
        present = source[source.notna()]
        if present.empty:
            # Nothing to split, and no dtype to trust: an all-NA column arrives as
            # float64, whose `.str` accessor would raise. Every output is NaN.
            return pd.DataFrame(index=source.index, columns=self.labels, dtype=object)

        if self.regex:
            parts = present.apply(lambda x: re.split(self.delimiter, str(x)))
        else:
            parts = present.str.split(self.delimiter)

        # Validate that every present row has the expected number of parts
        n_expected = len(self.labels)
        counts = parts.apply(len)
        bad_mask = counts != n_expected
        if bad_mask.any():
            # Positional lookup: `bad_mask`/`counts`/`present` share one index,
            # so locating by position is correct even for a non-integer or
            # duplicated index (the label-vs-position mix-up this replaces).
            first_bad_pos = int(bad_mask.to_numpy().argmax())
            raise ValueError(
                f"Column '{self.column}' split produced "
                f"{counts.iloc[first_bad_pos]} parts for value "
                f"'{present.iloc[first_bad_pos]}', but {n_expected} labels were "
                f"provided: {self.labels}"
            )

        # Restore the NA rows as all-NaN
        return pd.DataFrame(
            parts.tolist(), columns=self.labels, index=present.index
        ).reindex(source.index)

    def _operate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split the metadata column and insert new columns.

        Args:
            df: Measurement DataFrame containing the source column.

        Returns:
            DataFrame with new columns inserted after the source column.
            Rows whose source value is NA get NaN in every new column.
        """
        try:
            result = coalesce_metadata_aliases(df, [self.column, *self.labels])
            source_column = resolve_metadata_column(result.columns, self.column)
        except KeyError:
            raise KeyError(
                f"Column '{self.column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            ) from None

        split_df = self._split_to_labels(result[source_column])

        # Insert new columns after the source column
        src_pos = cast(int, result.columns.get_loc(source_column))
        for i, label in enumerate(self.labels):
            try:
                target_column = resolve_metadata_column(result.columns, label)
            except KeyError:
                result.insert(src_pos + 1 + i, label, split_df[label])
            else:
                result[target_column] = split_df[label]

        return result
