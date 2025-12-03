import numpy as np
import pandas as pd

from typing import Dict, Union, List, Optional

# TODO: Implement


class MeasurementAccessor:
    """Container for storing and managing measurement data as pandas Series or DataFrames.

    This class provides a dictionary-like interface for storing, accessing, and
    manipulating measurement data associated with image analysis. It enforces type
    constraints on both keys and values, ensuring keys are non-empty strings without
    spaces and values are pandas Series or DataFrames.

    The accessor supports common operations like indexing, iteration, merging
    measurements by index names, and exporting to various formats including
    dictionaries and numpy structured arrays.

    Attributes:
        __measurements (dict[str, Union[pd.Series, pd.DataFrame]]): Internal storage
            for measurement data, mapping measurement names to pandas objects.

    Examples:
        .. dropdown:: Basic usage with Series

            .. code-block:: python

                accessor = MeasurementAccessor()
                accessor['color_intensity'] = pd.Series([1.2, 3.4, 5.6])
                value = accessor['color_intensity']  # Returns a copy

        .. dropdown:: Working with DataFrames

            .. code-block:: python

                accessor = MeasurementAccessor()
                df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
                accessor['morphology'] = df

        .. dropdown:: Accessing keys and values

            .. code-block:: python

                keys = accessor.keys()  # ['color_intensity', 'morphology']
                values = accessor.values()  # [Series(...), DataFrame(...)]
    """

    def __init__(self):
        """Initialize an empty MeasurementAccessor.

        Creates a new MeasurementAccessor instance with an empty measurements
        dictionary. Measurements can be added using dictionary-like assignment
        via __setitem__.
        """
        self.__measurements: Dict[str, Union[pd.Series, pd.DataFrame]] = {}

    def keys(self) -> List[str]:
        """Return a list of all measurement names.

        Returns:
            list[str]: A list containing all measurement keys currently stored
                in the accessor.

        Examples:
            .. dropdown:: Retrieve all measurement keys

                .. code-block:: python

                    accessor = MeasurementAccessor()
                    accessor['metric1'] = pd.Series([1, 2, 3])
                    accessor['metric2'] = pd.Series([4, 5, 6])
                    keys = accessor.keys()
                    # keys == ['metric1', 'metric2']
        """
        return list(self.__measurements.keys())

    def values(self) -> List[Union[pd.Series, pd.DataFrame]]:
        """Return a list of all measurements.

        Returns independent copies of all stored measurements to prevent external
        modifications from affecting the internal state.

        Returns:
            list[Union[pd.Series, pd.DataFrame]]: A list containing independent
                copies of all measurement data.

        Examples:
            .. dropdown:: Get independent copies of all measurements

                .. code-block:: python

                    accessor = MeasurementAccessor()
                    accessor['data1'] = pd.Series([1, 2, 3])
                    accessor['data2'] = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
                    values = accessor.values()
                    # values == [Series([1, 2, 3]), DataFrame({'A': [1, 2], 'B': [3, 4]})]
                    # Modifications to values do not affect stored data
                    values[0].iloc[0] = 999  # Does not modify accessor['data1']
        """
        return list(x.copy() for x in self.__measurements.values())

    def __getitem__(self, key: str) -> Union[pd.Series, pd.DataFrame]:
        """Retrieve a copy of a measurement by key.

        Args:
            key (str): The name of the measurement to retrieve.

        Returns:
            Union[pd.Series, pd.DataFrame]: An independent copy of the measurement
                data. Modifications to the returned object do not affect the
                stored measurement.

        Raises:
            KeyError: If the specified key does not exist in the measurements.

        Examples:
            .. dropdown:: Retrieve a measurement by key

                .. code-block:: python

                    accessor = MeasurementAccessor()
                    accessor['colors'] = pd.Series([255, 128, 64])
                    colors = accessor['colors']
                    # colors is a copy, independent from stored data
        """
        return self.__measurements[key].copy()

    def __setitem__(self, key: str, value: Union[pd.Series, pd.DataFrame]) -> None:
        """Store a measurement with validation.

        Stores a measurement (Series or DataFrame) with the given key. Enforces
        constraints on the key to ensure consistency and avoid conflicts: keys
        must be strings without spaces.

        Args:
            key (str): The name for the measurement. Must be a non-empty string
                without spaces.
            value (Union[pd.Series, pd.DataFrame]): The measurement data to store.
                Must be either a pandas Series or DataFrame.

        Raises:
            TypeError: If key is not a string or value is not a pandas Series
                or DataFrame.
            ValueError: If key contains spaces.

        Examples:
            .. dropdown:: Store Series and DataFrame measurements

                .. code-block:: python

                    accessor = MeasurementAccessor()
                    # Store a Series
                    accessor['intensity'] = pd.Series([1.0, 2.5, 3.8])
                    # Store a DataFrame
                    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
                    accessor['coordinates'] = df
                    # TypeError: key must be a string
                    accessor[123] = pd.Series([1, 2, 3])
                    # ValueError: key must not contain spaces
                    accessor['my measurement'] = pd.Series([1, 2, 3])
        """
        if type(key) != str:
            raise TypeError("key must be a string")

        if " " in key:
            raise ValueError("key must not contain spaces")

        if type(value) not in [pd.Series, pd.DataFrame]:
            raise TypeError(
                "Measurement container only supports pd.Series or pd.DataFrame"
            )
        self.__measurements[key] = value

    def __len__(self) -> int:
        """Return the number of measurements stored.

        Returns:
            int: The count of measurements currently stored in the accessor.

        Examples:
            .. dropdown:: Count stored measurements

                .. code-block:: python

                    accessor = MeasurementAccessor()
                    len(accessor)  # 0
                    accessor['m1'] = pd.Series([1, 2, 3])
                    len(accessor)  # 1
                    accessor['m2'] = pd.Series([4, 5, 6])
                    len(accessor)  # 2
        """
        return len(self.keys())

    def pop(
        self, key: str, exc_type: Optional[str] = "raise"
    ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """Remove and return a measurement.

        Removes a measurement from the accessor and returns it. Allows control
        over exception handling when the specified key is not found.

        Args:
            key (str): The name of the measurement to remove.
            exc_type (str, optional): Controls behavior when key is not found.
                Must be either 'raise' or 'ignore'. Defaults to 'raise'.
                - 'raise': Raises KeyError if key does not exist.
                - 'ignore': Returns None if key does not exist.

        Returns:
            Optional[Union[pd.Series, pd.DataFrame]]: The measurement that was
                removed. Returns None if exc_type is 'ignore' and the key does
                not exist.

        Raises:
            KeyError: If exc_type is 'raise' and the specified key does not
                exist in the measurements.

        Examples:
            .. dropdown:: Remove and return measurements with error handling

                .. code-block:: python

                    accessor = MeasurementAccessor()
                    accessor['metric'] = pd.Series([1, 2, 3])
                    # Remove with exception on missing key
                    result = accessor.pop('metric', exc_type='raise')
                    # result is a Series, 'metric' is now removed
                    # KeyError raised if 'metric' doesn't exist
                    missing = accessor.pop('nonexistent', exc_type='raise')
                    # Safely remove with None return on missing key
                    result = accessor.pop('metric', exc_type='ignore')
                    # Returns None if 'metric' doesn't exist
        """
        if exc_type == "raise":
            return self.__measurements.pop(key)
        if exc_type == "ignore":
            return self.__measurements.pop(key, None)

    def clear(self) -> None:
        """Remove all measurements from the accessor.

        Removes all stored measurements and releases their memory. After calling
        this method, the accessor will be empty and contain no measurements.

        Examples:
            .. dropdown:: Clear all measurements from the accessor

                .. code-block:: python

                    accessor = MeasurementAccessor()
                    accessor['m1'] = pd.Series([1, 2, 3])
                    accessor['m2'] = pd.DataFrame({'A': [1, 2]})
                    len(accessor)  # 2
                    accessor.clear()
                    len(accessor)  # 0
        """
        for key in self.__measurements.keys():
            tmp = self.__measurements.pop(key)
            del tmp

    def merge_on_index_names(
        self,
        idx_name_subset: Optional[List[str]] = None,
        join_type: str = "outer",
        verify_integrity: bool = False,
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """Merge measurements by their index names.

        Groups measurements by their index name and merges each group by
        concatenating along the columns (axis=1). This is useful for combining
        related measurements that share the same index structure.

        If a measurement key conflicts with a generated index name, it will be
        renamed with a counter suffix (e.g., 'index_name (2)').

        Args:
            idx_name_subset (Optional[List[str]], optional): Specific index names
                to merge. If None, all unique index names in measurements are
                merged. Can be a single string or list of strings. Defaults to None.
            join_type (str, optional): Type of join operation to use when merging.
                Can be 'inner', 'outer', 'left', or 'right'. Defaults to 'outer'.
                See pandas.concat for details.
            verify_integrity (bool, optional): If True, raises an exception if
                the concatenated result has duplicate indices. Defaults to False.

        Returns:
            Dict[str, Union[pd.Series, pd.DataFrame]]: Dictionary containing the
                newly created merged measurements. These are also added to the
                accessor's internal storage.

        Raises:
            ValueError: If idx_name_subset contains index names not found in the
                stored measurements.

        Examples:
            .. dropdown:: Merge measurements by index name

                .. code-block:: python

                    accessor = MeasurementAccessor()
                    # Create measurements with the same index name
                    s1 = pd.Series([1, 2, 3], index=pd.Index([0, 1, 2], name='image_id'))
                    s2 = pd.Series([4, 5, 6], index=pd.Index([0, 1, 2], name='image_id'))
                    s3 = pd.Series([7, 8, 9], index=pd.Index([0, 1, 2], name='object_id'))
                    accessor['metric1'] = s1
                    accessor['metric2'] = s2
                    accessor['metric3'] = s3
                    # Merge all measurements by index name
                    merged = accessor.merge_on_index_names()
                    # merged contains 'image_id' (concatenation of s1 and s2)
                    # and 'object_id' (s3 alone)
                    # Merge only specific index name
                    merged = accessor.merge_on_index_names(idx_name_subset=['image_id'])
        """
        if type(idx_name_subset) is str:
            idx_name_subset = [idx_name_subset]

        idx_name_list = [df.index.name for df in self.__measurements.values()]
        idx_names = set(idx_name_list)
        if idx_name_subset is None:
            target_index_names = idx_names
        elif set(idx_name_subset).issubset(set(idx_name_list)) is False:
            raise ValueError(
                "the index names in idx_name_subset must be a found in the index names of the measurements."
            )
        else:
            target_index_names = idx_name_subset

        merged_measurements = {}
        for idx_name in target_index_names:
            current_tables = list(
                measurement
                for measurement in (self.__measurements.values())
                if measurement.index.name == idx_name
            )

            # In the event the index name appears in the measurements key
            if (measurement_key := idx_name) in self.keys():
                idx_name_appearances = len(
                    list(
                        idx_name_iter
                        for idx_name_iter in self.keys()
                        if idx_name in idx_name_iter
                    )
                )

                measurement_key = f"{idx_name} ({idx_name_appearances})"

            merged_measurements[measurement_key] = pd.concat(
                objs=current_tables,
                axis=1,
                join=join_type,
                ignore_index=False,
                verify_integrity=verify_integrity,
                copy=True,
            )
        self.__measurements = {**self.__measurements, **merged_measurements}
        return merged_measurements

    def to_dict(self) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """Export all measurements as a dictionary with copies.

        Returns a dictionary containing independent copies of all measurements.
        Modifications to the returned dictionary or its contents do not affect
        the stored measurements.

        Returns:
            Dict[str, Union[pd.Series, pd.DataFrame]]: A dictionary mapping
                measurement names to independent copies of the data.

        Examples:
            .. dropdown:: Export measurements as a dictionary with copies

                .. code-block:: python

                    accessor = MeasurementAccessor()
                    accessor['intensity'] = pd.Series([1, 2, 3])
                    accessor['texture'] = pd.DataFrame({'smooth': [0.5, 0.6]})
                    d = accessor.to_dict()
                    # d == {'intensity': Series(...), 'texture': DataFrame(...)}
                    # Modifications to d do not affect stored data
                    d['intensity'].iloc[0] = 999
                    accessor['intensity'].iloc[0]  # Still 1
        """
        return {key: table.copy() for key, table in self.__measurements.items()}

    def to_recarrays_dict(self) -> Dict[str, np.recarray]:
        """Export all measurements as a dictionary of numpy structured arrays.

        Converts each measurement to a numpy structured array (recarray) with
        the index included. This is useful for compatibility with numpy-based
        processing or low-level data access.

        Returns:
            Dict[str, np.recarray]: A dictionary mapping measurement names to
                numpy structured arrays created from the measurements, with the
                original index included as a field.

        Examples:
            .. dropdown:: Convert measurements to numpy structured arrays

                .. code-block:: python

                    accessor = MeasurementAccessor()
                    s = pd.Series([1, 2, 3], index=[10, 20, 30], name='values')
                    accessor['data'] = s
                    recarrays = accessor.to_recarrays_dict()
                    # recarrays['data'] is a numpy recarray with index and values
                    # Access via: recarrays['data']['index'], recarrays['data']['values']
        """
        return {
            key: table.copy().to_records(index=True)
            for key, table in self.__measurements.items()
        }

    def copy(self) -> "MeasurementAccessor":
        """Create an independent copy of this MeasurementAccessor.

        Creates a new MeasurementAccessor instance with a shallow copy of the
        internal measurements dictionary. The measurements themselves are not
        copied, so modifications to the data in the returned accessor will
        affect the original accessor's data (and vice versa). To avoid shared
        data, use to_dict() to get copies of the individual measurements.

        Returns:
            MeasurementAccessor: A new MeasurementAccessor instance sharing the
                same measurement objects.

        Examples:
            .. dropdown:: Create a copy with shared measurement references

                .. code-block:: python

                    accessor1 = MeasurementAccessor()
                    accessor1['metric'] = pd.Series([1, 2, 3])
                    accessor2 = accessor1.copy()
                    # accessor2['metric'] references the same Series as accessor1['metric']
                    # Modifications to the data affect both
                    accessor2['metric'].iloc[0] = 999
                    accessor1['metric'].iloc[0]  # Also 999
                    # To avoid shared data, use copies
                    accessor2['metric'] = accessor1['metric'].copy()
        """
        new_container = self.__class__()
        new_container.__measurements = {**self.__measurements}
        return new_container
