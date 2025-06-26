from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
from typing import Dict, Optional, List
import inspect

from phenotypic.abstract import MeasureFeatures, ImageOperation


class ImagePipeline(ImageOperation):
    """
    Represents a handler for processing and measurement queues used in _root_image operations
    and feature extraction tasks.

    This class manages two queues: a processing queue and a measurement queue. The processing
    queue contains _root_image operations that are applied sequentially to an _root_image. The measurement
    queue contains feature extractors that are used to analyze an _root_image and produce results
    as a pandas DataFrame. Both queues are optional and can be specified as dictionaries. If not
    provided, empty queues are initialized by default to enable flexibility in pipeline
    construction and usage.

    Attributes:
        _op_queue (Dict[str, ImageOperation]): A dictionary where keys are string
            identifiers and values are `ImageOperation` objects representing operations to apply
            to an _root_image.
        _measurement_queue (Dict[str, MeasureFeatures]): A dictionary where keys are string
            identifiers and values are `FeatureExtractor` objects for extracting features
            from images.
    """

    def __init__(self,
                 ops: Dict[str, ImageOperation] | None = None,
                 measurements: Dict[str, MeasureFeatures] | None = None):
        """
        This class represents a processing and measurement abstract for _root_image operations
        and feature extraction. It initializes operational and measurement queues based
        on the provided dictionaries.

        Args:
            ops: A dictionary where the keys are operation names (strings)
                and the values are ImageOperation objects responsible for performing
                specific _root_image processing tasks.
            measurements: An optional dictionary where the keys are feature names
                (strings) and the values are FeatureExtractor objects responsible for
                extracting specific features.
        """
        self._op_queue: Dict[str, ImageOperation] = ops if ops is not None else {}
        self._measurement_queue: Dict[str, MeasureFeatures] = measurements if measurements is not None else {}

    def apply(self, image: Image, inplace: bool = False) -> Image:
        """
        The class provides an abstract to process and apply a series of operations on
        an _root_image. The operations are maintained in a queue and executed sequentially
        when applied to the given _root_image.

        Args:
            image (Image): The input_image _root_image to be processed. The type `Image` refers to
                an instance of the _root_image object to which transformations are applied.
            inplace (bool, optional): A flag indicating whether to apply the
                transformations directly on the provided _root_image (`True`) or create a
                copy of the _root_image before performing transformations (`False`). Defaults
                to `False`.
        """
        img = image if inplace else image.copy()
        for key, operation in self._op_queue.items():
            try:
                sig = inspect.signature(operation.apply)
                if 'inplace' in sig.parameters:
                    operation.apply(img, inplace=True)
                else:
                    img = operation.apply(img)
            except Exception as e:
                raise Exception(f'Failed to apply {operation} during step {key} to _root_image {img.name}: {e}') from e

        return img

    def measure(self, image: Image, inplace: bool = False) -> pd.DataFrame:
        """
        Measures various properties of an _root_image using queued measurement strategies.

        The `measure` function applies the queued measurement strategies to the given
        _root_image and returns a DataFrame containing consolidated object measurement results.

        Args:
            image (Image): The input_image _root_image on which the measurements will be applied.
            inplace (bool): A flag indicating whether the modifications should be applied
                directly to the input_image _root_image. Default is False.

        Returns:
            pd.DataFrame: A DataFrame containing measurement results from all the
            queued measurement strategies, merged on the same index.
        """
        measurements = [image.grid.info() if hasattr(image, 'grid') else image.objects.info()]
        for key in self._measurement_queue.keys():
            measurements.append(self._measurement_queue[key].measure(image))
        return self._merge_on_same_index(measurements)

    @staticmethod
    def _merge_on_same_index(dataframes_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple DataFrames only if they share the same index name.

        Args:
            dataframes_list: List of pandas DataFrames to merge

        Returns:
            Merged DataFrame containing only the data from DataFrames with matching index names

        Raises:
            ValueError: If no DataFrames are provided or if no matching index names are found
        """
        if not dataframes_list:
            raise ValueError("No DataFrames provided")

        # Get index names for all dataframes_list
        index_names = [df.index.name for df in dataframes_list]

        # Group DataFrames by their index names
        index_groups = {}
        for df, idx_name in zip(dataframes_list, index_names):
            if idx_name is not None:  # Skip unnamed indices
                index_groups.setdefault(idx_name, []).append(df)

        if not index_groups:
            raise ValueError("No named indices found in the provided DataFrames")

        # Merge DataFrames for each index name
        merged_results = []
        for idx_name, df_group in index_groups.items():
            if len(df_group) > 1:  # Only merge if we have multiple DataFrames with same index
                # Merge all DataFrames in the group
                merged_df = df_group[0]
                for df in df_group[1:]:
                    merged_df = merged_df.join(df, how='outer')
                merged_results.append(merged_df)

        if not merged_results:
            raise ValueError("No DataFrames with matching index names found")

        return merged_results[0] if len(merged_results) == 1 else merged_results
