from __future__ import annotations
from typing import TYPE_CHECKING, Dict

import h5py
import numpy as np

if TYPE_CHECKING: from phenotypic import ImageSet

import pandas as pd
from pandas.api.types import pandas_dtype, is_extension_array_dtype, is_integer_dtype, is_bool_dtype
from phenotypic.util.constants_ import IO


# TODO: Not fully integrated yet
class SetMeasurementAccessor:

    def __init__(self, image_set: ImageSet):
        self._image_set = image_set

    def table(self) -> pd.DataFrame:
        measurements = []
        with self._image_set.hdf_.reader() as reader:
            images = self._image_set.hdf_.get_data_group(reader)
            for image_name in images.keys():
                image_group = images[image_name]
                if self._image_set.hdf_.IMAGE_MEASUREMENT_SUBGROUP_KEY in image_group:
                    measurements.append(self._load_dataframe_from_hdf5_group(group=image_group,
                                                                             measurement_key=self._image_set.hdf_.IMAGE_MEASUREMENT_SUBGROUP_KEY))
        return pd.concat(measurements) if measurements else pd.DataFrame()

    @staticmethod
    def _save_dataframe_to_hdf5_group_swmr(df: pd.DataFrame, group, measurement_key: str | None = None):
        """Save a DataFrame to HDF5 group using SWMR-compatible index/values structure.
        
        This method creates the new structure:
        measurements/
        ├── index/          # DataFrame index datasets
        └── values/         # DataFrame column datasets
        
        Args:
            df: pandas DataFrame to save
            group: HDF5 group object where to save the DataFrame
            measurement_key: name of the subgroup to create for the DataFrame data
        """
        import logging
        import os
        import time
        
        # Create parallel-safe logger with process/thread identification
        logger = logging.getLogger(f"{__name__}.swmr_save")
        process_id = os.getpid()
        thread_name = getattr(__import__('threading').current_thread(), 'name', 'MainThread')
        log_prefix = f"[PID:{process_id}|{thread_name}]"
        
        start_time = time.time()
        
        if measurement_key is None: 
            measurement_key = IO.IMAGE_MEASUREMENT_IMAGE_SUBGROUP_KEY
            
        logger.info(f"{log_prefix} Starting SWMR DataFrame save - Shape: {df.shape}, Key: '{measurement_key}'")
        logger.debug(f"{log_prefix} DataFrame info - Index: {df.index.name or 'unnamed'} ({df.index.dtype}), Columns: {list(df.columns)}")
            
        # Remove existing measurements if any
        if measurement_key in group:
            logger.debug(f"{log_prefix} Removing existing measurement group '{measurement_key}'")
            del group[measurement_key]
            
        # Create measurements subgroup
        logger.debug(f"{log_prefix} Creating measurement group '{measurement_key}'")
        meas_group = group.create_group(measurement_key)
        
        # Create index and values subgroups
        logger.debug(f"{log_prefix} Creating index and values subgroups")
        index_group = meas_group.create_group('index')
        values_group = meas_group.create_group('values')
        
        # Store metadata about the DataFrame structure
        logger.debug(f"{log_prefix} Storing DataFrame metadata - Rows: {len(df)}, Cols: {len(df.columns)}")
        meas_group.attrs['num_rows'] = len(df)
        meas_group.attrs['num_cols'] = len(df.columns)
        
        # Store index data
        index_name = df.index.name if df.index.name is not None else 'level_0'
        if hasattr(df.index, 'values'):
            index_values = df.index.values
            if index_values.dtype.kind in ['U', 'S']:  # String types
                index_group.create_dataset(index_name, data=index_values.astype('S'), compression="gzip", compression_opts=4)
            else:
                index_group.create_dataset(index_name, data=index_values, compression="gzip", compression_opts=4)
        else:
            # Handle non-numpy index types
            try:
                import numpy as np
                index_array = np.array(df.index)
                index_group.create_dataset(index_name, data=index_array, compression="gzip", compression_opts=4)
            except (ValueError, TypeError):
                # Fallback to string conversion
                index_data = [str(i).encode() for i in df.index]
                index_group.create_dataset(index_name, data=index_data, compression="gzip", compression_opts=4)
        
        # Store index metadata
        index_group.attrs['native_dtype'] = str(df.index.dtype) if hasattr(df.index, 'dtype') else 'object'
        index_group.attrs['pandas_dtype'] = str(df.index.dtype) if hasattr(df.index, 'dtype') else 'object'
        index_group.attrs['numpy_dtype'] = str(df.index.values.dtype) if hasattr(df.index, 'values') else 'object'
        index_group.attrs['position'] = 0
        
        # Store each column in values group
        for i, col in enumerate(df.columns):
            col_data = df[col].values
            dataset_name = col  # Use actual column name as dataset name
            
            # Handle different data types appropriately
            if col_data.dtype.kind in ['U', 'S']:  # String types
                values_group.create_dataset(dataset_name, data=col_data.astype('S'), compression="gzip", compression_opts=4)
            elif col_data.dtype.kind in ['f', 'i', 'u']:  # Numeric types
                values_group.create_dataset(dataset_name, data=col_data, compression="gzip", compression_opts=4)
            elif col_data.dtype.kind == 'b':  # Boolean types
                values_group.create_dataset(dataset_name, data=col_data, compression="gzip", compression_opts=4)
            elif col_data.dtype.kind == 'O':  # Object dtype - check if contains strings
                sample_val = col_data[0] if len(col_data) > 0 else None
                if isinstance(sample_val, str) or sample_val is None:
                    # Object column contains strings - convert to UTF-8 strings for HDF5 storage
                    string_data = [str(val) if val is not None else '' for val in col_data]
                    values_group.create_dataset(dataset_name, data=string_data, dtype=h5py.string_dtype(encoding='utf-8'), compression="gzip", compression_opts=4)
                else:
                    # Object column contains other types - convert to string
                    string_data = [str(val) for val in col_data]
                    values_group.create_dataset(dataset_name, data=string_data, dtype=h5py.string_dtype(encoding='utf-8'), compression="gzip", compression_opts=4)
            else:  # Other types - try to preserve original type, fallback to string
                try:
                    values_group.create_dataset(dataset_name, data=col_data, compression="gzip", compression_opts=4)
                except (TypeError, ValueError):
                    # Fallback to string conversion
                    string_data = [str(val) for val in col_data]
                    values_group.create_dataset(dataset_name, data=string_data, dtype=h5py.string_dtype(encoding='utf-8'), compression="gzip", compression_opts=4)
            
            # Store column metadata as dataset attributes
            dataset = values_group[dataset_name]
            dataset.attrs['native_dtype'] = str(df[col].dtype)
            dataset.attrs['pandas_dtype'] = str(df[col].dtype)
            dataset.attrs['numpy_dtype'] = str(col_data.dtype)
            dataset.attrs['position'] = i
            dataset.attrs['column_name'] = str(col).encode('utf-8')
            
            logger.debug(f"{log_prefix} Saved column '{col}' as '{dataset_name}' - Shape: {col_data.shape}, Dtype: {col_data.dtype}")
        
        # Log completion with timing
        elapsed_time = time.time() - start_time
        logger.info(f"{log_prefix} SWMR DataFrame save completed - {len(df)} rows, {len(df.columns)} columns in {elapsed_time:.3f}s")
        logger.debug(f"{log_prefix} Final structure: {measurement_key}/{{index: {list(index_group.keys())}, values: {list(values_group.keys())}}}")

    @staticmethod
    def _save_dataframe_to_hdf5_group(df: pd.DataFrame, group, measurement_key: str | None = None):
        """Save a DataFrame to an HDF5 group while preserving column data types.

        Args:
            df: pandas DataFrame to save
            group: HDF5 group object where to save the DataFrame
            measurement_key: name of the subgroup to create for the DataFrame data
        """
        # Remove existing measurements if any
        if measurement_key is None: measurement_key = IO.IMAGE_MEASUREMENT_IMAGE_SUBGROUP_KEY
        if measurement_key in group:
            del group[measurement_key]

        # Create measurements subgroup
        meas_data = group.create_group(measurement_key)

        # Store index with proper type handling
        if hasattr(df.index, 'values'):
            index_values = df.index.values
            if index_values.dtype.kind in ['U', 'S']:  # String types
                meas_data.create_dataset("index", data=index_values.astype('S'))
                meas_data.attrs["index_dtype"] = str(df.index.dtype)
            else:
                meas_data.create_dataset("index", data=index_values)
                meas_data.attrs["index_dtype"] = str(df.index.dtype)
        else:
            # Handle non-numpy index types - preserve original type info
            try:
                # Try to preserve numeric types
                import numpy as np
                index_array = np.array(df.index)
                meas_data.create_dataset("index", data=index_array)
                meas_data.attrs["index_dtype"] = str(index_array.dtype)
            except (ValueError, TypeError):
                # Fallback to string conversion
                index_data = [str(i).encode() for i in df.index]
                meas_data.create_dataset("index", data=index_data)
                meas_data.attrs["index_dtype"] = "object"

        # Store column names
        column_data = [str(c).encode() for c in df.columns]
        meas_data.create_dataset("columns", data=column_data)

        # Store each column separately to preserve data types
        for i, col in enumerate(df.columns):
            col_data = df[col].values
            dataset_name = f"col_{i:04d}"  # Use zero-padded index for consistent ordering

            # Handle different data types appropriately
            if col_data.dtype.kind in ['U', 'S']:  # String types (Unicode/bytes)
                meas_data.create_dataset(dataset_name, data=col_data.astype('S'), compression="gzip", compression_opts=4)
            elif col_data.dtype.kind in ['f', 'i', 'u']:  # Numeric types
                meas_data.create_dataset(dataset_name, data=col_data, compression="gzip", compression_opts=4)
            elif col_data.dtype.kind == 'O':  # Object dtype - check if contains strings
                # Check if the object column contains strings
                sample_val = col_data[0] if len(col_data) > 0 else None
                if isinstance(sample_val, str) or sample_val is None or (
                        hasattr(sample_val, '__class__') and sample_val.__class__.__name__ == 'str'):
                    # Object column contains strings - convert to bytes for HDF5 storage
                    string_data = [str(val).encode('utf-8') if val is not None else b'' for val in col_data]
                    meas_data.create_dataset(dataset_name, data=string_data, compression="gzip", compression_opts=4)
                else:
                    # Object column contains other types - convert to string then bytes
                    string_data = [str(val).encode('utf-8') for val in col_data]
                    meas_data.create_dataset(dataset_name, data=string_data, compression="gzip", compression_opts=4)
            else:  # Other types - convert to string
                string_data = [str(val).encode('utf-8') for val in col_data]
                meas_data.create_dataset(dataset_name, data=string_data, compression="gzip", compression_opts=4)

        # Store original dtypes as metadata
        dtype_info = {str(i): str(df[col].dtype) for i, col in enumerate(df.columns)}
        for key, dtype_str in dtype_info.items():
            meas_data.attrs[f"dtype_{key}"] = dtype_str

    @staticmethod
    def _load_dataframe_from_hdf5_group_swmr(group, measurement_key: str | None = None):
        """Load a DataFrame from HDF5 group using SWMR-compatible index/values structure.
        
        This method loads from the new structure:
        measurements/
        ├── index/          # DataFrame index datasets
        └── values/         # DataFrame column datasets
        
        Args:
            group: HDF5 group object containing the DataFrame data
            measurement_key: name of the subgroup containing the DataFrame data
            
        Returns:
            pd.DataFrame: Reconstructed DataFrame with original dtypes
        """
        import logging
        import os
        import time
        
        # Create parallel-safe logger with process/thread identification
        logger = logging.getLogger(f"{__name__}.swmr_load")
        process_id = os.getpid()
        thread_name = getattr(__import__('threading').current_thread(), 'name', 'MainThread')
        log_prefix = f"[PID:{process_id}|{thread_name}]"
        
        start_time = time.time()
        
        if measurement_key is None:
            measurement_key = IO.IMAGE_MEASUREMENT_IMAGE_SUBGROUP_KEY
            
        logger.info(f"{log_prefix} Starting SWMR DataFrame load - Key: '{measurement_key}'")
        
        # Check if measurement exists
        if measurement_key not in group:
            logger.error(f"{log_prefix} Measurement key '{measurement_key}' not found in group")
            raise KeyError(f"Measurement key '{measurement_key}' not found in group")
            
        meas_group = group[measurement_key]
        logger.debug(f"{log_prefix} Found measurement group '{measurement_key}'")
        
        # Check if this uses the new structure
        if 'index' not in meas_group or 'values' not in meas_group:
            logger.info(f"{log_prefix} Old structure detected - falling back to legacy loader")
            return SetMeasurementAccessor._load_dataframe_from_hdf5_group(group, measurement_key)
            
        logger.debug(f"{log_prefix} SWMR structure confirmed - accessing index and values groups")
        index_group = meas_group['index']
        values_group = meas_group['values']
        
        # Get DataFrame metadata
        num_rows = meas_group.attrs.get('num_rows', 0)
        num_cols = meas_group.attrs.get('num_cols', 0)
        logger.debug(f"{log_prefix} DataFrame metadata - Rows: {num_rows}, Cols: {num_cols}")
        
        if num_rows == 0:
            logger.warning(f"{log_prefix} No data rows found - returning empty DataFrame")
            return pd.DataFrame()
            
        # Load index data - use the first available index dataset
        index_data = None
        index_name = None
        for dataset_name in index_group.keys():
            index_dataset = index_group[dataset_name]
            index_values = index_dataset[:num_rows]  # Only read valid rows
            index_name = dataset_name
            
            # Handle string decoding if needed
            if index_values.dtype.kind == 'S':
                try:
                    index_data = [val.decode('utf-8') if isinstance(val, bytes) else str(val) for val in index_values]
                except (UnicodeDecodeError, AttributeError):
                    index_data = [str(val) for val in index_values]
            elif index_values.dtype.kind == 'O':
                # Handle object arrays that might contain bytes
                decoded_values = []
                for val in index_values:
                    if isinstance(val, bytes):
                        try:
                            decoded_values.append(val.decode('utf-8'))
                        except UnicodeDecodeError:
                            decoded_values.append(str(val))
                    else:
                        decoded_values.append(str(val))
                index_data = decoded_values
            else:
                index_data = index_values
            break  # Use first index dataset found
                
        # Load column data
        columns = []
        column_data = []
        
        # Get all column datasets and sort by position
        col_datasets = [(name, values_group[name]) for name in values_group.keys()]
        col_datasets.sort(key=lambda x: x[1].attrs.get('position', 0))
        
        for dataset_name, dataset in col_datasets:
            # Use dataset name directly as column name (it should be the actual column name)
            col_name = dataset.attrs.get('column_name', dataset_name)
            if isinstance(col_name, bytes):
                col_name = col_name.decode('utf-8')
            else:
                col_name = dataset_name  # Use dataset name directly
            columns.append(col_name)
            
            # Load column values
            col_values = dataset[:num_rows]  # Only read valid rows
            
            # Handle data type restoration
            original_dtype = dataset.attrs.get('pandas_dtype', 'object')
            
            # Handle string decoding for object/string columns
            if col_values.dtype.kind == 'S':
                # Handle byte strings from HDF5
                try:
                    decoded_values = [val.decode('utf-8') if isinstance(val, bytes) else str(val) for val in col_values]
                    col_values = np.array(decoded_values, dtype=object)
                except (UnicodeDecodeError, AttributeError):
                    col_values = np.array([str(val) for val in col_values], dtype=object)
            elif col_values.dtype.kind == 'O':
                # Handle object arrays that might contain bytes or need string conversion
                decoded_values = []
                for val in col_values:
                    if isinstance(val, bytes):
                        try:
                            decoded_values.append(val.decode('utf-8'))
                        except UnicodeDecodeError:
                            decoded_values.append(str(val))
                    else:
                        # Keep original value for non-bytes objects
                        decoded_values.append(val)
                col_values = np.array(decoded_values, dtype=object)
            
            column_data.append(col_values)
        
        # Create DataFrame
        logger.debug(f"{log_prefix} Creating DataFrame from loaded data - {len(columns)} columns, {len(index_data) if index_data is not None else 0} rows")
        
        if column_data:
            df = pd.DataFrame(dict(zip(columns, column_data)), index=index_data)
            logger.debug(f"{log_prefix} DataFrame created with columns: {list(df.columns)}")
        else:
            df = pd.DataFrame(index=index_data)
            logger.debug(f"{log_prefix} Empty DataFrame created with index only")
        
        # Log completion with timing and validation
        elapsed_time = time.time() - start_time
        logger.info(f"{log_prefix} SWMR DataFrame load completed - Shape: {df.shape} in {elapsed_time:.3f}s")
        
        # Validate data integrity
        if not df.empty:
            logger.debug(f"{log_prefix} Data validation - Index dtype: {df.index.dtype}, Column dtypes: {dict(df.dtypes)}")
            
        return df

    @staticmethod
    def _load_dataframe_from_hdf5_group(group, measurement_key: str | None = None) -> pd.DataFrame:
        """Load a DataFrame from an HDF5 group, preserving column data types.

        Args:
            group: HDF5 group object containing the DataFrame data
            measurement_key: name of the subgroup containing the DataFrame data

        Returns:
            pandas DataFrame with original data types preserved
        """
        import pandas as pd
        if measurement_key is None: measurement_key = IO.IMAGE_MEASUREMENT_IMAGE_SUBGROUP_KEY
        if measurement_key not in group:
            return pd.DataFrame()

        meas_group = group[measurement_key]

        # Check if required components exist
        if "columns" not in meas_group or "index" not in meas_group:
            return pd.DataFrame()

        # Load column names
        columns = [col.decode() for col in meas_group["columns"][:]]

        # Load index with original type restoration
        index_data = meas_group["index"][:]

        # Restore original index dtype if available
        if "index_dtype" in meas_group.attrs:
            original_index_dtype = meas_group.attrs["index_dtype"]
            if isinstance(original_index_dtype, bytes):
                original_index_dtype = original_index_dtype.decode()

            try:
                if index_data.dtype.kind in ['S', 'U']:  # String types
                    decoded_index = [idx.decode() if hasattr(idx, 'decode') else str(idx) for idx in index_data]
                    if original_index_dtype == "object":
                        index = decoded_index
                    else:
                        # Try to convert back to original numeric type
                        import pandas as pd
                        index = pd.Index(decoded_index).astype(original_index_dtype).tolist()
                else:
                    index = index_data.astype(original_index_dtype).tolist()
            except (ValueError, TypeError):
                # Fallback to basic handling
                if index_data.dtype.kind in ['S', 'U']:
                    index = [idx.decode() if hasattr(idx, 'decode') else str(idx) for idx in index_data]
                else:
                    index = index_data.tolist()
        else:
            # No dtype metadata - handle based on current type
            if index_data.dtype.kind in ['S', 'U']:
                index = [idx.decode() if hasattr(idx, 'decode') else str(idx) for idx in index_data]
            else:
                index = index_data.tolist()

        # Load column data with validation
        data_dict = {}
        missing_columns = []

        for i, col in enumerate(columns):
            dataset_name = f"col_{i:04d}"
            if dataset_name not in meas_group:
                missing_columns.append(f"Column '{col}' (dataset '{dataset_name}')")
                continue

            col_data = meas_group[dataset_name][:]

            # Restore original data type if metadata is available
            dtype_key = f"dtype_{i}"
            if dtype_key in meas_group.attrs:
                original_dtype = meas_group.attrs[dtype_key]
                if isinstance(original_dtype, bytes):
                    original_dtype = original_dtype.decode()

                try:
                    # Handle string types or object types that contain bytes
                    needs_decoding = False
                    if col_data.dtype.kind in ['S', 'U']:  # String/Unicode dtypes
                        needs_decoding = True
                    elif col_data.dtype.kind == 'O':  # Object dtype - check if contains bytes
                        # Check if the object column contains bytes that need decoding
                        sample_val = col_data[0] if len(col_data) > 0 else None
                        if isinstance(sample_val, bytes):
                            needs_decoding = True

                    if needs_decoding:
                        # Properly decode bytes to strings
                        decoded_data = []
                        for val in col_data:
                            if isinstance(val, bytes):
                                decoded_data.append(val.decode('utf-8'))
                            elif hasattr(val, 'decode'):
                                decoded_data.append(val.decode('utf-8'))
                            else:
                                decoded_data.append(str(val))

                        if 'object' in original_dtype or 'str' in original_dtype:
                            data_dict[col] = decoded_data
                        else:
                            # Try to convert to original numeric type
                            data_dict[col] = pd.Series(decoded_data).astype(original_dtype)
                    else:
                        # Numeric data - restore original dtype
                        data_dict[col] = col_data.astype(original_dtype)
                except (ValueError, TypeError):
                    # If conversion fails, use data as-is but ensure strings are decoded
                    needs_decoding = False
                    if col_data.dtype.kind in ['S', 'U']:  # String/Unicode dtypes
                        needs_decoding = True
                    elif col_data.dtype.kind == 'O':  # Object dtype - check if contains bytes
                        sample_val = col_data[0] if len(col_data) > 0 else None
                        if isinstance(sample_val, bytes):
                            needs_decoding = True

                    if needs_decoding:
                        decoded_data = []
                        for val in col_data:
                            if isinstance(val, bytes):
                                decoded_data.append(val.decode('utf-8'))
                            elif hasattr(val, 'decode'):
                                decoded_data.append(val.decode('utf-8'))
                            else:
                                decoded_data.append(str(val))
                        data_dict[col] = decoded_data
                    else:
                        data_dict[col] = col_data
            else:
                # No dtype metadata - handle based on current type
                needs_decoding = False
                if col_data.dtype.kind in ['S', 'U']:  # String/Unicode dtypes
                    needs_decoding = True
                elif col_data.dtype.kind == 'O':  # Object dtype - check if contains bytes
                    sample_val = col_data[0] if len(col_data) > 0 else None
                    if isinstance(sample_val, bytes):
                        needs_decoding = True

                if needs_decoding:
                    # Ensure proper decoding when no dtype metadata is available
                    decoded_data = []
                    for val in col_data:
                        if isinstance(val, bytes):
                            decoded_data.append(val.decode('utf-8'))
                        elif hasattr(val, 'decode'):
                            decoded_data.append(val.decode('utf-8'))
                        else:
                            decoded_data.append(str(val))
                    data_dict[col] = decoded_data
                else:
                    data_dict[col] = col_data

        # Warn about missing columns if any
        if missing_columns:
            import warnings
            warnings.warn(f"Missing column datasets in HDF5 group: {', '.join(missing_columns)}")

        # Create DataFrame
        if data_dict:
            return pd.DataFrame(data_dict, index=index)
        else:
            return pd.DataFrame(index=index)

    @staticmethod
    def get_dtype_null(dtype):
        """Return a valid null value for the given dtype that can be imported and interpreted in an HDF5 group dataset.

        Args:
            dtype: The data type for which to return a null value. Can be a numpy dtype,
                  a native Python type, or a string representation of a dtype.

        Returns:
            A null value appropriate for the given dtype that is compatible with HDF5 storage.
        """
        # Handle native Python types
        if dtype is int or dtype == 'int':
            return -9223372036854775808  # Minimum value for int64
        if dtype is float or dtype == 'float':
            return np.nan
        if dtype is str or dtype == 'str':
            return ''
        if dtype is bool or dtype == 'bool' or (isinstance(dtype, str) and dtype == 'b') or dtype is np.bool_:
            return -1  # booleans encoded as i8 with -1 for missing

        # Handle string representation of numpy dtypes
        if isinstance(dtype, str):
            if dtype.startswith('datetime64'):
                return np.datetime64('NaT')
            if dtype.startswith('timedelta64'):
                return np.timedelta64('NaT')
            if dtype == 'object':
                return b''

            # Try to convert string to numpy dtype
            try:
                dtype = np.dtype(dtype)
            except TypeError:
                # If conversion fails, return empty bytes as default
                return b''

        # Handle numpy scalar types directly
        if isinstance(dtype, type):
            # NumPy scalar types
            if dtype is np.int8 or dtype is np.int16 or dtype is np.int32 or dtype is np.int64:
                return np.iinfo(dtype).min
            if dtype is np.uint8 or dtype is np.uint16 or dtype is np.uint32 or dtype is np.uint64:
                return np.iinfo(dtype).min
            if dtype is np.float16 or dtype is np.float32 or dtype is np.float64:
                return np.nan
            if dtype is np.str_ or dtype is np.bytes_:
                return b''

        # Handle numpy dtype objects
        if hasattr(dtype, 'kind'):
            # For integer types, we need to use the specific dtype
            if dtype.kind == 'i':
                return np.iinfo(dtype).min  # signed integer
            elif dtype.kind == 'u':
                return np.iinfo(dtype).min  # unsigned integer
            elif dtype.kind == 'f':
                return np.nan  # float
            elif dtype.kind == 'c':
                return complex(np.nan, np.nan)  # complex
            elif dtype.kind == 'b':
                return -1  # boolean (encoded as integer)
            elif dtype.kind == 'S':
                return b''  # byte string
            elif dtype.kind == 'U':
                return ''  # unicode string
            elif dtype.kind == 'V':
                return None  # void
            elif dtype.kind == 'O':
                return b''  # object
            elif dtype.kind == 'M':
                return np.datetime64('NaT')  # datetime
            elif dtype.kind == 'm':
                return np.timedelta64('NaT')  # timedelta
            else:
                return b''  # default

        # Default fallback
        return b''

    def _create_empty_column_dataset(self, meas_name: str, dtype: type, measurement_home_group: h5py.Group)->h5py.Dataset:
        """Create a column group in the HDF5 file group.
        preallocates 100 rows for each measurement

        Note:
            - index should be prepended with index_
            - columns should be prepended with col_

        """
        if meas_name in measurement_home_group:
            del measurement_home_group[meas_name]
        return measurement_home_group.require_dataset(
            name='values',
            shape=(100,),
            maxshape=(None, 1),
            chunks=(50, 1),
            dtype=dtype,
            fillvalue=self.get_dtype_null(dtype),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

    @staticmethod
    def _dtype_metadata(dtype) -> dict[str, bytes]:
        """Produce a metadata dict for a dtype, encoding each name as UTF-8 bytes."""
        meta = {
            "native_dtype": "",
            "numpy_dtype": "",
            "pandas_dtype": ""
        }
        # 1) native Python built-ins
        if isinstance(dtype, type) and dtype.__module__ == "builtins":
            meta["native_dtype"] = dtype.__name__
        # 2) NumPy dtype
        try:
            np_dt = np.dtype(dtype)
            meta["numpy_dtype"] = np_dt.name
        except Exception:
            pass
        # 3) pandas Extension dtype
        try:
            pd_dt = pandas_dtype(dtype)
            meta["pandas_dtype"] = pd_dt.name
        except Exception:
            pass
        # encode for HDF5 attr storage
        return {k: (v.encode("utf-8") if v else b"") for k, v in meta.items()}


    # TODO: create equivalent casting to save to exisitng datasets
    @staticmethod
    def _preallocate_swmr_measurement_datasets(group, measurement_key: str, index_dtypes: list, column_dtypes: list, initial_size: int = 100):
        """Pre-allocate empty SWMR-compatible measurement datasets.
        
        Args:
            group: HDF5 group where to create the measurement structure
            measurement_key: name of the measurement subgroup
            index_dtypes: list of (name, dtype) tuples for index datasets
            column_dtypes: list of (name, dtype, position) tuples for column datasets
            initial_size: initial size for chunked datasets
        """
        import logging
        import os
        import time
        
        # Create parallel-safe logger with process/thread identification
        logger = logging.getLogger(f"{__name__}.swmr_preallocate")
        process_id = os.getpid()
        thread_name = getattr(__import__('threading').current_thread(), 'name', 'MainThread')
        log_prefix = f"[PID:{process_id}|{thread_name}]"
        
        start_time = time.time()
        
        logger.info(f"{log_prefix} Starting SWMR dataset pre-allocation - Key: '{measurement_key}', Initial size: {initial_size}")
        logger.debug(f"{log_prefix} Index dtypes: {index_dtypes}")
        logger.debug(f"{log_prefix} Column dtypes: {[(name, dtype, pos) for name, dtype, pos in column_dtypes]}")
        
        # Remove existing measurements if any
        if measurement_key in group:
            logger.debug(f"{log_prefix} Removing existing measurement group '{measurement_key}'")
            del group[measurement_key]
            
        # Create measurements subgroup
        meas_group = group.create_group(measurement_key)
        
        # Create index and values subgroups
        index_group = meas_group.create_group('index')
        values_group = meas_group.create_group('values')
        
        # Initialize metadata
        meas_group.attrs['num_rows'] = 0  # Will be updated as data is written
        meas_group.attrs['num_cols'] = len(column_dtypes)
        
        # Pre-allocate index datasets
        for i, (name, dtype) in enumerate(index_dtypes):
            # Use the actual index name as dataset name
            dataset_name = name
            
            # Convert dtype to HDF5-compatible type
            if isinstance(dtype, str):
                if dtype == 'object' or 'str' in dtype.lower():
                    np_dtype = h5py.string_dtype(encoding='utf-8')
                    fillvalue = ''
                else:
                    try:
                        np_dtype = np.dtype(dtype)
                        fillvalue = SetMeasurementAccessor.get_dtype_null(np_dtype)
                    except TypeError:
                        np_dtype = h5py.string_dtype(encoding='utf-8')
                        fillvalue = ''
            else:
                # Handle numpy dtype objects
                if hasattr(dtype, 'kind') and dtype.kind == 'O':  # Object dtype
                    np_dtype = h5py.string_dtype(encoding='utf-8')
                    fillvalue = ''
                elif hasattr(dtype, 'kind') and dtype.kind in ['U', 'S']:  # String dtypes
                    np_dtype = h5py.string_dtype(encoding='utf-8')
                    fillvalue = ''
                else:
                    np_dtype = dtype
                    fillvalue = SetMeasurementAccessor.get_dtype_null(np_dtype)
            
            # Create chunked dataset with unlimited dimension
            index_group.create_dataset(
                dataset_name,
                shape=(initial_size,),
                maxshape=(None,),
                chunks=(50,),
                dtype=np_dtype,
                fillvalue=fillvalue,
                compression="gzip",
                compression_opts=4
            )
            
            # Store metadata
        
        # Pre-allocate column datasets
        for name, dtype, position in column_dtypes:
            # Use the actual column name as dataset name
            dataset_name = name
            
            # Convert dtype to HDF5-compatible type
            if isinstance(dtype, str):
                if dtype == 'object' or 'str' in dtype.lower():
                    np_dtype = h5py.string_dtype(encoding='utf-8')
                    fillvalue = ''
                else:
                    try:
                        np_dtype = np.dtype(dtype)
                        fillvalue = SetMeasurementAccessor.get_dtype_null(np_dtype)
                    except TypeError:
                        np_dtype = h5py.string_dtype(encoding='utf-8')
                        fillvalue = ''
            else:
                # Handle numpy dtype objects
                if hasattr(dtype, 'kind') and dtype.kind == 'O':  # Object dtype
                    np_dtype = h5py.string_dtype(encoding='utf-8')
                    fillvalue = ''
                elif hasattr(dtype, 'kind') and dtype.kind in ['U', 'S']:  # String dtypes
                    np_dtype = h5py.string_dtype(encoding='utf-8')
                    fillvalue = ''
                else:
                    np_dtype = dtype
                    fillvalue = SetMeasurementAccessor.get_dtype_null(np_dtype)
            
            # Create chunked dataset with unlimited dimension
            dataset = values_group.create_dataset(
                dataset_name,
                shape=(initial_size,),
                maxshape=(None,),
                chunks=(50,),
                dtype=np_dtype,
                fillvalue=fillvalue,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            
            # Store metadata as dataset attributes
            dataset.attrs['native_dtype'] = str(dtype)
            dataset.attrs['pandas_dtype'] = str(dtype)
            dataset.attrs['numpy_dtype'] = str(np_dtype)
            dataset.attrs['position'] = position
            dataset.attrs['column_name'] = str(name).encode('utf-8')
            
            logger.debug(f"{log_prefix} Pre-allocated column dataset '{dataset_name}' - Shape: {dataset.shape}, Dtype: {np_dtype}, Chunks: {dataset.chunks}")
        
        # Log completion with timing
        elapsed_time = time.time() - start_time
        logger.info(f"{log_prefix} SWMR pre-allocation completed - {len(index_dtypes)} index, {len(column_dtypes)} column datasets in {elapsed_time:.3f}s")
        logger.debug(f"{log_prefix} Pre-allocation summary - Index datasets: {list(index_group.keys())}, Column datasets: {list(values_group.keys())}")
    
    @staticmethod
    def _write_dataframe_to_preallocated_datasets(df: pd.DataFrame, group, measurement_key: str, row_offset: int = 0):
        """Write DataFrame data to pre-allocated SWMR datasets.
        
        Args:
            df: DataFrame to write
            group: HDF5 group containing the measurement structure
            measurement_key: name of the measurement subgroup
            row_offset: starting row index for writing data
        """
        import logging
        import os
        import time
        
        # Create parallel-safe logger with process/thread identification
        logger = logging.getLogger(f"{__name__}.swmr_write")
        process_id = os.getpid()
        thread_name = getattr(__import__('threading').current_thread(), 'name', 'MainThread')
        log_prefix = f"[PID:{process_id}|{thread_name}]"
        
        start_time = time.time()
        
        logger.info(f"{log_prefix} Starting SWMR dataset write - Shape: {df.shape}, Key: '{measurement_key}', Offset: {row_offset}")
        
        if measurement_key not in group:
            logger.error(f"{log_prefix} Measurement group '{measurement_key}' not found - pre-allocation required")
            raise ValueError(f"Measurement group '{measurement_key}' not found. Must pre-allocate first.")
            
        meas_group = group[measurement_key]
        index_group = meas_group['index']
        values_group = meas_group['values']
        
        num_rows = len(df)
        end_row = row_offset + num_rows
        
        # Resize datasets if needed
        for dataset_name in index_group.keys():
            dataset = index_group[dataset_name]
            if dataset.shape[0] < end_row:
                dataset.resize((end_row,))
                
        for dataset_name in values_group.keys():
            dataset = values_group[dataset_name]
            if dataset.shape[0] < end_row:
                dataset.resize((end_row,))
        
        # Write index data
        # Use the actual index name from the DataFrame
        index_name = df.index.name if df.index.name is not None else 'level_0'
        if index_name in index_group:
            index_dataset = index_group[index_name]
            index_values = df.index.values
            
            # Handle string data with proper encoding
            if index_values.dtype.kind in ['U', 'S', 'O']:
                # Convert to UTF-8 strings for HDF5 storage
                string_values = [str(val) for val in index_values]
                index_dataset[row_offset:end_row] = string_values
            else:
                index_dataset[row_offset:end_row] = index_values
        
        # Write column data using actual column names
        for col in df.columns:
            if col in values_group:
                dataset = values_group[col]
                col_data = df[col].values
                
                # Handle different data types with proper encoding
                if col_data.dtype.kind in ['U', 'S']:
                    # Convert Unicode/byte strings to UTF-8
                    string_values = [str(val) for val in col_data]
                    dataset[row_offset:end_row] = string_values
                elif col_data.dtype.kind == 'O':
                    # Convert object data to strings with UTF-8 encoding
                    string_values = [str(val) if val is not None else '' for val in col_data]
                    dataset[row_offset:end_row] = string_values
                else:
                    dataset[row_offset:end_row] = col_data
        
        # Update row count
        old_num_rows = meas_group.attrs.get('num_rows', 0)
        new_num_rows = max(old_num_rows, end_row)
        meas_group.attrs['num_rows'] = new_num_rows
        
        # Log completion with timing and validation
        elapsed_time = time.time() - start_time
        logger.info(f"{log_prefix} SWMR dataset write completed - {len(df)} rows written to offset {row_offset} in {elapsed_time:.3f}s")
        logger.debug(f"{log_prefix} Updated row count: {old_num_rows} → {new_num_rows}")
        
        # Validate write operation
        logger.debug(f"{log_prefix} Write validation - Index datasets: {list(index_group.keys())}, Value datasets: {list(values_group.keys())}")

    def _create_empty_series_on_hdf(self, group: h5py.Group, name: str, series: pd.Series):
        """Store one pandas Series as an HDF5 dataset with dtype metadata."""
        # 1) pick a storage array: numpy or float64 if we need NaN
        dt = series.dtype
        has_ext_int = is_extension_array_dtype(dt) and is_integer_dtype(dt)
        has_ext_bool = is_extension_array_dtype(dt) and is_bool_dtype(dt)
        if has_ext_int or has_ext_bool:
            # extension types must convert to NumPy first
            if series.isna().any():
                arr = series.to_numpy(dtype="float64", na_value=np.nan)
            else:
                # No missing: pick a NumPy dtype from the pandas dtype
                np_dtype = getattr(dt, "numpy_dtype", dt.type)
                arr = series.to_numpy(dtype=np_dtype)
        else:
            # for pure NumPy-backed columns, just extract the array
            arr = series.to_numpy()
        # 2) create the dataset
        ds = self._create_empty_column_dataset(name, arr.dtype, group)
        # 3) attach dtype metadata
        for k, v in self._dtype_metadata(dt).items():
            ds.attrs[k] = v

    @staticmethod
    def _dataset2series(ds: h5py.Dataset) -> pd.Series:
        """Load one column from its HDF5 dataset back into the original pandas Series."""
        raw = ds[()]  # numpy array
        # decode metadata attrs
        meta = {k: ds.attrs.get(k, b"").decode("utf-8") or None
                for k in ("native_dtype", "numpy_dtype", "pandas_dtype")}
        pd_name = meta["pandas_dtype"]
        np_name = meta["numpy_dtype"]

        if pd_name:
            # reconstruct ExtensionArray
            return pd.Series(raw, dtype=pd_name)
        elif np_name:
            # pure NumPy dtype
            return pd.Series(raw.astype(np_name), dtype=np_name)
        else:
            # fallback to Python-native (object) series
            # e.g. if native_dtype == "str" or "int", Python will auto-box
            return pd.Series(raw.tolist(), dtype="object")
