import pandas as pd
from phenotypic.util.constants_ import IO

# TODO: Not fully integrated yet
class Measurements:
    """A measurement container to hold image measurements for an image that is returned from MeasureFeature classes after measuring an image."""

    def __init__(self, name: str, image_name: str, measurement: pd.DataFrame):
        """
        Represents an object with a name, associated image name, and a dataframe
        for measurements. This class initializes the core attributes required
        for handling and processing data related to these entities.

        Args:
            name: The name of the object.
            image_name: The name of the associated image
            measurement: A pandas DataFrame containing measurement data.
        """
        self.name = name
        self.image_name = image_name
        self.table = measurement

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
            else:
                meas_data.create_dataset("index", data=index_values)
        else:
            # Handle non-numpy index types
            index_data = [str(i).encode() for i in df.index]
            meas_data.create_dataset("index", data=index_data)

        # Store column names
        column_data = [str(c).encode() for c in df.columns]
        meas_data.create_dataset("columns", data=column_data)

        # Store each column separately to preserve data types
        for i, col in enumerate(df.columns):
            col_data = df[col].values
            dataset_name = f"col_{i:04d}"  # Use zero-padded index for consistent ordering

            # Handle different data types appropriately
            if col_data.dtype.kind in ['U', 'S']:  # String types
                meas_data.create_dataset(dataset_name, data=col_data.astype('S'), compression="gzip", compression_opts=4)
            elif col_data.dtype.kind in ['f', 'i', 'u']:  # Numeric types
                meas_data.create_dataset(dataset_name, data=col_data, compression="gzip", compression_opts=4)
            else:  # Other types - convert to string
                string_data = [str(val).encode() for val in col_data]
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
        if measurement_key is None: measurement_key = IO.IMAGE_MEASUREMENT_IMAGE_SUBGROUP_KEY
        if measurement_key not in group:
            return pd.DataFrame()

        meas_group = group[measurement_key]

        # Check if required components exist
        if "columns" not in meas_group or "index" not in meas_group:
            return pd.DataFrame()

        # Load column names
        columns = [col.decode() for col in meas_group["columns"][:]]

        # Load index
        index_data = meas_group["index"][:]
        if index_data.dtype.kind in ['S', 'U']:  # String types
            index = [idx.decode() if hasattr(idx, 'decode') else str(idx) for idx in index_data]
        else:
            index = index_data.tolist()

        # Load column data
        data_dict = {}
        for i, col in enumerate(columns):
            dataset_name = f"col_{i:04d}"
            if dataset_name in meas_group:
                col_data = meas_group[dataset_name][:]

                # Restore original data type if metadata is available
                dtype_key = f"dtype_{i}"
                if dtype_key in meas_group.attrs:
                    original_dtype = meas_group.attrs[dtype_key]
                    if isinstance(original_dtype, bytes):
                        original_dtype = original_dtype.decode()

                    try:
                        # Handle string types
                        if col_data.dtype.kind in ['S', 'U']:
                            col_data = [val.decode() if hasattr(val, 'decode') else str(val) for val in col_data]
                            if 'object' in original_dtype or 'str' in original_dtype:
                                data_dict[col] = col_data
                            else:
                                # Try to convert to original numeric type
                                data_dict[col] = pd.Series(col_data).astype(original_dtype)
                        else:
                            # Numeric data - restore original dtype
                            data_dict[col] = col_data.astype(original_dtype)
                    except (ValueError, TypeError):
                        # If conversion fails, use data as-is
                        if col_data.dtype.kind in ['S', 'U']:
                            data_dict[col] = [val.decode() if hasattr(val, 'decode') else str(val) for val in col_data]
                        else:
                            data_dict[col] = col_data
                else:
                    # No dtype metadata - handle based on current type
                    if col_data.dtype.kind in ['S', 'U']:
                        data_dict[col] = [val.decode() if hasattr(val, 'decode') else str(val) for val in col_data]
                    else:
                        data_dict[col] = col_data

        # Create DataFrame
        if data_dict:
            return pd.DataFrame(data_dict, index=index)
        else:
            return pd.DataFrame(index=index)