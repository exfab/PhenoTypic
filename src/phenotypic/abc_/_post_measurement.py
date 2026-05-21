from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ._base_operation import BaseOperation


class PostMeasurement(BaseOperation, ABC):
    """Transform a measurement DataFrame after feature extraction.

    PostMeasurement is the abstract base class for operations that reshape,
    enrich, or clean measurement DataFrames produced by the pipeline's
    measurement step. Unlike MeasureFeatures (which extracts data from images),
    PostMeasurement operates on the already-assembled DataFrame.

    Post-measurement transforms run after all MeasureFeatures have executed
    and their results have been merged. They receive the complete DataFrame
    and return a modified copy.

    Args:
        None

    Returns:
        pd.DataFrame: The transformed measurement DataFrame.

    Examples:
        Subclass to create a custom post-measurement transform:

        >>> from phenotypic.abc_ import PostMeasurement
        >>> import pandas as pd
        >>> class AddConstant(PostMeasurement):
        ...     '''Add a constant-valued column.
        ...
        ...     Args:
        ...         column: Name of the column to add.
        ...         value: Constant value to assign to every row.
        ...     '''
        ...     column: str
        ...     value: str
        ...     def _operate(self, df):
        ...         df[self.column] = self.value
        ...         return df
        >>> post = AddConstant(column="Metadata_Flag", value="OK")
        >>> df = pd.DataFrame({"ObjectLabel": [1, 2]})
        >>> result = post.apply(df)
        >>> list(result.columns)
        ['ObjectLabel', 'Metadata_Flag']
    """

    @abstractmethod
    def _operate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the measurement DataFrame.

        Args:
            df: The merged measurement DataFrame with all measurement and
                metadata columns.

        Returns:
            The transformed DataFrame.
        """
        ...

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the post-measurement transform.

        Args:
            df: The merged measurement DataFrame.

        Returns:
            The transformed DataFrame.
        """
        return self._operate(df)
