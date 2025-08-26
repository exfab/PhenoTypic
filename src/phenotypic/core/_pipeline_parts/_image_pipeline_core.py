from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
from typing import Dict, Optional, List, Tuple
import inspect
import time
import sys

from phenotypic.abstract import MeasureFeatures, ImageOperation


class ImagePipelineCore(ImageOperation):
    """
    Represents a handler for processing and measurement queues used in Image operations
    and feature extraction tasks.

    This class manages two queues: a processing queue and a measurement queue. The processing
    queue contains Image operations that are applied sequentially to an Image. The measurement
    queue contains feature extractors that are used to analyze an Image and produce results
    as a pandas DataFrame. Both queues are optional and can be specified as dictionaries. If not
    provided, empty queues are initialized by default to enable flexibility in pipeline
    construction and usage.

    Attributes:
        _ops (Dict[str, ImageOperation]): A dictionary where keys are string
            identifiers and values are `ImageOperation` objects representing operations to apply
            to an Image.
        _measurements (Dict[str, MeasureFeatures]): A dictionary where keys are string
            identifiers and values are `FeatureExtractor` objects for extracting features
            from images.
    """

    def __init__(self,
                 ops: List[ImageOperation] | Dict[str, ImageOperation] | None = None,
                 measurements: List[MeasureFeatures] | Dict[str, MeasureFeatures] | None = None,
                 benchmark: bool = False,
                 verbose: bool = False
                 ):
        """
        This class represents a processing and measurement abstract for Image operations
        and feature extraction. It initializes operational and measurement queues based
        on the provided dictionaries.

        Args:
            ops: A dictionary where the keys are operation names (strings)
                and the values are ImageOperation objects responsible for performing
                specific Image processing tasks.
            measurements: An optional dictionary where the keys are feature names
                (strings) and the values are FeatureExtractor objects responsible for
                extracting specific features.
            benchmark: A flag indicating whether to track execution times for operations
                and measurements. Defaults to False.
            verbose: A flag indicating whether to print progress information when
                benchmark mode is on. Defaults to False.
        """
        # If ops is a list of operations convert to a dictionary
        self._ops: Dict[str, ImageOperation] = {}
        if ops is not None: self.set_ops(ops)

        self._measurements: Dict[str, MeasureFeatures] = {}
        if measurements is not None: self.set_measurements(measurements)

        # Store benchmark and verbose flags
        self._benchmark = benchmark
        self._verbose = verbose

        # Initialize dictionaries to store execution times
        self._operation_times: Dict[str, float] = {}
        self._measurement_times: Dict[str, float] = {}

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
        an Image. The operations are maintained in a queue and executed sequentially
        when applied to the given Image.

        Args:
            image (Image): The input_image Image to be processed. The type `Image` refers to
                an instance of the Image object to which transformations are applied.
            inplace (bool, optional): A flag indicating whether to apply the
                transformations directly on the provided Image (`True`) or create a
                copy of the Image before performing transformations (`False`). Defaults
                to `False`.
            reset (bool): Whether to reset the image before applying the pipeline
        """
        img = image if inplace else image.copy()
        if reset: image.reset()

        # Reset operation times for new apply run if benchmarking is enabled
        if self._benchmark:
            self._operation_times = {}

        # Create progress bar if verbose and benchmark are enabled
        if self._benchmark and self._verbose:
            try:
                from tqdm import tqdm
                # Create a tqdm instance without items to manually update it
                total_ops = len(self._ops)
                pbar = tqdm(total=total_ops, desc="Applying operations", file=sys.stdout)
                has_tqdm = True
            except ImportError:
                # If tqdm is not available, fall back to simple printing
                print("Applying operations...")
                has_tqdm = False
        else:
            has_tqdm = False

        for i, (key, operation) in enumerate(self._ops.items()):
            try:
                # Update progress bar description with current operation
                if self._benchmark and self._verbose:
                    if has_tqdm:
                        pbar.set_description(f"Operation: {key}")
                    else:
                        print(f"  Applying operation: {key}")

                # Measure execution time if benchmarking is enabled
                if self._benchmark:
                    start_time = time.time()

                sig = inspect.signature(operation.apply)
                if 'inplace' in sig.parameters:
                    operation.apply(img, inplace=True)
                else:
                    img = operation.apply(img)

                # Store execution time if benchmarking is enabled
                if self._benchmark:
                    self._operation_times[key] = time.time() - start_time

                    # Print execution time if verbose and benchmark are enabled
                    if self._verbose:
                        if has_tqdm:
                            pbar.set_postfix(time=f"{self._operation_times[key]:.4f}s")
                            pbar.update(1)
                        else:
                            print(f"    Completed in {self._operation_times[key]:.4f} seconds")
            except Exception as e:
                if self._benchmark and self._verbose and has_tqdm:
                    pbar.close()
                raise Exception(f'Failed to apply {operation} during step {key} to Image {img.name}: {e}') from e

        # Close the progress bar if it exists
        if self._benchmark and self._verbose and has_tqdm:
            pbar.close()

        return img

    def measure(self, image: Image, include_metadata=True) -> pd.DataFrame:
        """
        Measures various properties of an Image using queued measurement strategies.

        The `measure` function applies the queued measurement strategies to the given
        Image and returns a DataFrame containing consolidated object measurement results.

        Args:
            image (Image): The input_image Image on which the measurements will be applied.
            inplace (bool): A flag indicating whether the modifications should be applied
                directly to the input_image Image. Default is False.

        Returns:
            pd.DataFrame: A DataFrame containing measurement results from all the
            queued measurement strategies, merged on the same index.
        """
        # Reset measurement times for new measure run if benchmarking is enabled
        if self._benchmark:
            self._measurement_times = {}

        # Print message if verbose and benchmark are enabled
        if self._benchmark and self._verbose:
            print("Measuring image properties...")

        # Get image info and measure time if benchmarking is enabled
        if self._benchmark:
            start_time = time.time()
            measurements = [image.info(include_metadata=include_metadata)]
            self._measurement_times['image_info'] = time.time() - start_time

            # Print execution time if verbose and benchmark are enabled
            if self._verbose:
                print(f"  Image info: {self._measurement_times['image_info']:.4f} seconds")
        else:
            measurements = [image.grid.info() if hasattr(image, 'grid') else image.objects.info()]

        # Create progress bar if verbose and benchmark are enabled
        if self._benchmark and self._verbose:
            try:
                from tqdm import tqdm
                # Create a tqdm instance without items to manually update it
                total_measurements = len(self._measurements)
                pbar = tqdm(total=total_measurements, desc="Applying measurements", file=sys.stdout)
                has_tqdm = True
            except ImportError:
                # If tqdm is not available, fall back to simple printing
                print("Applying measurements...")
                has_tqdm = False
        else:
            has_tqdm = False

        for i, (key, measurement) in enumerate(self._measurements.items()):
            try:
                # Update progress bar description with current measurement
                if self._benchmark and self._verbose:
                    if has_tqdm:
                        pbar.set_description(f"Measurement: {key}")
                    else:
                        print(f"  Applying measurement: {key}")

                # Measure execution time for each measurement if benchmarking is enabled
                if self._benchmark:
                    start_time = time.time()
                    measurements.append(measurement.measure(image))
                    self._measurement_times[key] = time.time() - start_time

                    # Print execution time if verbose and benchmark are enabled
                    if self._verbose:
                        if has_tqdm:
                            pbar.set_postfix(time=f"{self._measurement_times[key]:.4f}s")
                            pbar.update(1)
                        else:
                            print(f"    Completed in {self._measurement_times[key]:.4f} seconds")
                else:
                    measurements.append(measurement.measure(image))
            except Exception as e:
                if self._benchmark and self._verbose and has_tqdm:
                    pbar.close()
                raise Exception(f'Failed to apply measurement {key} to Image: {e}') from e

        # Close the progress bar if it exists
        if self._benchmark and self._verbose and has_tqdm:
            pbar.close()

        return self._merge_on_same_index(measurements)

    def apply_and_measure(self, image: Image, inplace: bool = False, reset: bool = True) -> pd.DataFrame:
        print("🚨🚨🚨 PARENT METHOD CALLED: ImagePipelineCore.apply_and_measure 🚨🚨🚨")
        print(f"🚨 Parent method called with image type: {type(image)}")
        img = self.apply(image, inplace=inplace, reset=reset)
        return self.measure(img)

    def benchmark(self) -> pd.DataFrame:
        """
        Returns a table of execution times for operations and measurements.

        This method should be called after applying the pipeline on an image to get
        the execution times of the different processes.

        Returns:
            pd.DataFrame: A DataFrame containing execution times for each operation and measurement.
        """
        # Create a list to store the data
        data = []

        # Add operation times
        for op_name, op_time in self._operation_times.items():
            data.append({
                'Process Type': 'Operation',
                'Process Name': op_name,
                'Execution Time (s)': op_time
            })

        # Add measurement times
        for measure_name, measure_time in self._measurement_times.items():
            data.append({
                'Process Type': 'Measurement',
                'Process Name': measure_name,
                'Execution Time (s)': measure_time
            })

        # Create DataFrame
        if not data:
            return pd.DataFrame(columns=['Process Type', 'Process Name', 'Execution Time (s)'])

        df = pd.DataFrame(data)

        # Calculate total time
        total_time = df['Execution Time (s)'].sum()
        total_row = pd.DataFrame([{
            'Process Type': 'Total',
            'Process Name': 'All Processes',
            'Execution Time (s)': total_time
        }])
        df = pd.concat([df, total_row], ignore_index=True)

        return df

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
                    # Check for identical columns (same name and values)
                    duplicate_columns = []
                    for col in df.columns:
                        if col in merged_df.columns:
                            # Check if the column values are identical
                            if df[col].equals(merged_df[col]):
                                duplicate_columns.append(col)

                    # Remove duplicate columns from the incoming dataframe before merging
                    if duplicate_columns:
                        df = df.drop(columns=duplicate_columns)

                    # Only join if there are columns left to merge
                    if not df.empty and len(df.columns) > 0:
                        merged_df = merged_df.join(df, how='outer', lsuffix='', rsuffix='_duplicate')
                merged_results.append(merged_df)

        if not merged_results:
            raise ValueError("No DataFrames with matching index names found")

        return merged_results[0] if len(merged_results) == 1 else merged_results
