import pandas as pd
import numpy as np

from phenotypic.core._pipeline_parts._image_pipeline_core import ImagePipelineCore
from phenotypic.tools.constants_ import OBJECT


def _build_base_dataframes(index_name: bool = True):
    """Helper to build three DataFrames with shared object labels.

    When ``index_name`` is True, OBJECT.LABEL is used as the index name.
    Otherwise, OBJECT.LABEL is kept as a regular column.
    """

    index = pd.Index(["a", "b", "c"], name=OBJECT.LABEL if index_name else None)

    df1 = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "col3": [7, 8, 9],
            },
            index=index,
    )

    df2 = pd.DataFrame(
            {
                "col2": [4, 5, 6],  # Identical to df1.col2
                "col3": [10, 11, 12],  # Different values from df1.col3
                "col4": [13, 14, 15],
            },
            index=index,
    )

    df3 = pd.DataFrame(
            {
                "col5": [16, 17, 18],
                "col6": [19, 20, 21],
            },
            index=index,
    )

    # If OBJECT.LABEL should be a column instead of the index, reset here.
    if not index_name:
        df1 = (
            df1.reset_index().rename(columns={"index": OBJECT.LABEL})
            if df1.index.name is None
            else df1.reset_index()
        )
        df2 = (
            df2.reset_index().rename(columns={"index": OBJECT.LABEL})
            if df2.index.name is None
            else df2.reset_index()
        )
        df3 = (
            df3.reset_index().rename(columns={"index": OBJECT.LABEL})
            if df3.index.name is None
            else df3.reset_index()
        )

    return df1, df2, df3
