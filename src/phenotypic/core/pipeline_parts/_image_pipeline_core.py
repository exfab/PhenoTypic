from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
from typing import Dict, Optional, List
import inspect

from phenotypic.abstract import MeasureFeatures, ImageOperation


class ImagePipelineCore(ImageOperation):
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
        _ops (Dict[str, ImageOperation]): A dictionary where keys are string
            identifiers and values are `ImageOperation` objects representing operations to apply
            to an _root_image.
        _measurements (Dict[str, MeasureFeatures]): A dictionary where keys are string
            identifiers and values are `FeatureExtractor` objects for extracting features
            from images.
    """

    def __init__(self,
                 ops: List[ImageOperation] | Dict[str, ImageOperation] | None = None,
                 measurements: List[MeasureFeatures] | Dict[str, MeasureFeatures] | None = None
                 ):
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
        # If ops is a list of operations convert to a dictionary
        self._ops: Dict[str, ImageOperation] = {}
        if ops is not None: self.set_ops(ops)

        self._measurements: Dict[str, MeasureFeatures] = {}
        if measurements is not None: self.set_measurements(measurements)

    def set_ops(self, ops: List[ImageOperation] | Dict[str, ImageOperation]):
        """
        Sets the operations to be performed. The operations can be passed as either a list of
        ImageOperation instances or a dictionary mapping operation names to ImageOperation instances.
        This method ensures that each operation in the list has a unique name. Raises a TypeError
        if the input is neither a list nor a dictionary.

        Args:
            ops (List[ImageOperation] | Dict[str, ImageOperation]): A list of ImageOperation objects
                or a dictionary where keys are operation names and values are ImageOperation objects.

        Raises:
            TypeError: If the input is not a list or a dictionary.
        """
        # If ops is a list of ImageOperation
        if isinstance(ops, list):
            op_names = [x.__class__.__name__ for x in ops if isinstance(x, ImageOperation)]
            op_names = self.__make_unique(op_names)
            self._ops = {op_names[i]: ops[i] for i in range(len(ops))}
        # If ops is a dictionary
        elif isinstance(ops, dict):
            self._ops = ops
        else:
            raise TypeError(f'ops must be a list or a dictionary, got {type(ops)}')

    def set_measurements(self, measurements: List[MeasureFeatures] | Dict[str, MeasureFeatures]):
        """
        Sets the measurements to be used for further computation. The input can be either
        a list of `MeasureFeatures` objects or a dictionary with string keys and `MeasureFeatures`
        objects as values.

        The method processes the given input to construct a dictionary mapping measurement names
        to `MeasureFeatures` instances. If a list is passed, unique class names of the
        `MeasureFeatures` instances in the list are used as keys.

        Args:
            measurements (List[MeasureFeatures] | Dict[str, MeasureFeatures]): A collection
                of measurement features either as a list of `MeasureFeatures` objects, where
                class names are used as keys for dictionary creation, or as a dictionary where
                keys are predefined strings and values are `MeasureFeatures` objects.

        Raises:
            TypeError: If the `measurements` argument is neither a list nor a dictionary.
        """
        if isinstance(measurements, list):
            measurement_names = [x.__class__.__name__ for x in measurements if isinstance(x, MeasureFeatures)]
            measurement_names = self.__make_unique(measurement_names)
            self._measurements = {measurement_names[i]: measurements[i] for i in range(len(measurements))}
        elif isinstance(measurements, dict):
            self._measurements = measurements
        else:
            raise TypeError(f'measurements must be a list or a dictionary, got {type(measurements)}')

    @staticmethod
    def __make_unique(class_names):
        """
        Ensures uniqueness of strings in the given list by appending numeric suffixes when duplicates are
        found. If duplicates exist, subsequent occurrences of the duplicate string are modified by adding a
        numeric suffix to make them unique.

        Args:
            class_names (List[str]): A list of strings where duplicates may exist.

        Returns:
            List[str]: A new list of strings where each string is guaranteed to be unique.

        Raises:
            None
        """
        seen = {}
        result = []

        for s in class_names:
            if s not in seen:
                seen[s] = 0
                result.append(s)
            else:
                seen[s] += 1
                new_s = f"{s}_{seen[s]}"
                while new_s in seen:
                    seen[s] += 1
                    new_s = f"{s}_{seen[s]}"
                seen[new_s] = 0
                result.append(new_s)

        return result

    def apply(self, image: Image, inplace: bool = False, reset: bool = True) -> Image:
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
            reset (bool): Whether to reset the image before applying the pipeline
        """
        img = image if inplace else image.copy()
        if reset: image.reset()
        for key, operation in self._ops.items():
            try:
                sig = inspect.signature(operation.apply)
                if 'inplace' in sig.parameters:
                    operation.apply(img, inplace=True)
                else:
                    img = operation.apply(img)
            except Exception as e:
                raise Exception(f'Failed to apply {operation} during step {key} to _root_image {img.name}: {e}') from e

        return img

    def measure(self, image: Image) -> pd.DataFrame:
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
        for key in self._measurements.keys():
            measurements.append(self._measurements[key].measure(image))
        return self._merge_on_same_index(measurements)

    def apply_and_measure(self, image: Image, inplace: bool = False, reset: bool = True) -> (Image, pd.DataFrame):
        img = self.apply(image, inplace=inplace, reset=reset)
        return self.measure(img)

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
