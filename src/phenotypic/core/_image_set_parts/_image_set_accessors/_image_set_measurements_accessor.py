from __future__ import annotations
from typing import TYPE_CHECKING, Dict


if TYPE_CHECKING: from phenotypic import ImageSet

import pandas as pd
from phenotypic.util.constants_ import IO

# TODO: Not fully integrated yet
class SetMeasurementAccessor:

    def __init__(self, image_set:ImageSet):
        self._image_set = image_set


    def table(self)->pd.DataFrame:
        measurements = []
        with self._image_set._hdf.reader() as reader:
            images = self._image_set._hdf.get_image_data_group(reader)
            for image_name in images.keys():
                image_group = images[image_name]
                if self._image_set._hdf.IMAGE_MEASUREMENT_SUBGROUP_KEY in image_group:
                    measurements.append(self._load_dataframe_from_hdf5_group(group=image_group, measurement_key=self._image_set._hdf.IMAGE_MEASUREMENT_SUBGROUP_KEY))
        return pd.concat(measurements) if measurements else pd.DataFrame()



    @staticmethod
    def _save_dataframe_to_hdf5_group(df: pd.DataFrame, group, measurement_key: str | None= None):
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
                if isinstance(sample_val, str) or sample_val is None or (hasattr(sample_val, '__class__') and sample_val.__class__.__name__ == 'str'):
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
    def _load_dataframe_from_hdf5_group(group, measurement_key: str|None = None) -> pd.DataFrame:
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

