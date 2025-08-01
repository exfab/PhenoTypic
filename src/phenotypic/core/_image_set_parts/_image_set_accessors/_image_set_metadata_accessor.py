from typing import Dict, Tuple, Set, Any, Callable, Union, List, Optional

import pandas as pd
import numpy as np
import h5py
import posixpath
import inspect
from collections.abc import Mapping


class ImageSetMetadataAccessor:

    def __init__(self, image_set):
        self._image_set = image_set

    def _get_image_metadata(self, image_group) -> Dict[str, Any]:
        """Get image metadata with proper type conversion."""
        metadata = {}
        
        # Get protected metadata
        try:
            if 'protected_metadata' in image_group:
                prot_attrs = image_group['protected_metadata'].attrs
                for key in prot_attrs.keys():
                    metadata[key] = self._convert_hdf5_attribute(prot_attrs[key])
        except (KeyError, AttributeError):
            pass
        
        # Get public metadata (may overwrite protected if same key exists)
        try:
            if 'public_metadata' in image_group:
                pub_attrs = image_group['public_metadata'].attrs
                for key in pub_attrs.keys():
                    metadata[key] = self._convert_hdf5_attribute(pub_attrs[key])
        except (KeyError, AttributeError):
            pass
        
        return metadata

    def table(self) -> pd.DataFrame:
        """
        Aggregates metadata from all images in the image set into a pandas DataFrame.
        
        Each row represents an image, with columns for all metadata keys found across
        all images. Missing values are filled with np.nan.
        
        Returns:
            pd.DataFrame: DataFrame with image names as index and metadata as columns.
                         Columns include both protected and public metadata from all images.
        """
        image_names = self._image_set.get_image_names()
        
        if not image_names:
            return pd.DataFrame()
        
        # First pass: collect all unique metadata keys across all images
        all_keys = self._collect_all_metadata_keys(image_names)
        
        # Second pass: build the DataFrame
        metadata_records = []
        
        with h5py.File(self._image_set._out_path, mode='r') as file_handler:
            images_group = file_handler[self._image_set._hdf5_images_group_key]
            
            for image_name in image_names:
                if image_name in images_group:
                    image_group = images_group[image_name]
                    metadata_dict = self._extract_image_metadata_safe(image_group, all_keys)
                    metadata_dict['image_name'] = image_name
                    metadata_records.append(metadata_dict)
                else:
                    # Handle missing image gracefully
                    metadata_dict = {key: np.nan for key in all_keys}
                    metadata_dict['image_name'] = image_name
                    metadata_records.append(metadata_dict)
        
        if not metadata_records:
            return pd.DataFrame()
        
        # Create DataFrame and set image_name as index
        df = pd.DataFrame(metadata_records)
        df.set_index('image_name', inplace=True)
        
        return df
    
    def _collect_all_metadata_keys(self, image_names: list) -> Set[str]:
        """
        Collects all unique metadata keys from all images in the image set.
        
        Args:
            image_names (list): List of image names to process.
            
        Returns:
            Set[str]: Set of all unique metadata keys found across all images.
        """
        all_keys = set()
        
        with h5py.File(self._image_set._out_path, mode='r') as file_handler:
            images_group = file_handler[self._image_set._hdf5_images_group_key]
            
            for image_name in image_names:
                if image_name in images_group:
                    image_group = images_group[image_name]
                    keys = self._get_image_metadata_keys_safe(image_group)
                    all_keys.update(keys)
        
        return all_keys
    
    def _get_image_metadata_keys_safe(self, image_group) -> Set[str]:
        """
        Safely extracts all metadata keys from an image group.
        
        Args:
            image_group: HDF5 group for a single image.
            
        Returns:
            Set[str]: Set of metadata keys for this image.
        """
        keys = set()
        
        try:
            if 'protected_metadata' in image_group:
                prot_group = image_group['protected_metadata']
                keys.update(prot_group.attrs.keys())
        except (KeyError, AttributeError):
            pass
        
        try:
            if 'public_metadata' in image_group:
                pub_group = image_group['public_metadata']
                keys.update(pub_group.attrs.keys())
        except (KeyError, AttributeError):
            pass
        
        return keys
    
    def _extract_image_metadata_safe(self, image_group, all_keys: Set[str]) -> Dict[str, Any]:
        """
        Safely extracts metadata from an image group, filling missing keys with np.nan.
        
        Args:
            image_group: HDF5 group for a single image.
            all_keys (Set[str]): Set of all possible metadata keys.
            
        Returns:
            Dict[str, Any]: Dictionary with metadata values, missing keys filled with np.nan.
        """
        metadata_dict = {}
        
        # Initialize all keys with np.nan
        for key in all_keys:
            metadata_dict[key] = np.nan
        
        # Extract protected metadata
        try:
            if 'protected_metadata' in image_group:
                prot_attrs = image_group['protected_metadata'].attrs
                for key in prot_attrs.keys():
                    metadata_dict[key] = self._convert_hdf5_attribute(prot_attrs[key])
        except (KeyError, AttributeError):
            pass
        
        # Extract public metadata (may overwrite protected if same key exists)
        try:
            if 'public_metadata' in image_group:
                pub_attrs = image_group['public_metadata'].attrs
                for key in pub_attrs.keys():
                    metadata_dict[key] = self._convert_hdf5_attribute(pub_attrs[key])
        except (KeyError, AttributeError):
            pass
        
        return metadata_dict
    
    def _convert_hdf5_attribute(self, value: Any) -> Any:
        """
        Convert HDF5 attribute value to appropriate Python type.
        
        Args:
            value: Raw value from HDF5 attribute
            
        Returns:
            Any: Converted value with appropriate Python type
        """
        try:
            # Handle bytes (common in HDF5)
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            
            # Handle numpy scalar types
            if hasattr(value, 'item'):
                return value.item()
            
            # Handle native boolean values (HDF5 stores these as numpy.bool_)
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            
            # Handle native numeric types
            if isinstance(value, (int, float, np.integer, np.floating)):
                return value.item() if hasattr(value, 'item') else value
            
            # Handle string representations that might be numeric or boolean
            if isinstance(value, str):
                # Handle boolean strings (for backward compatibility)
                if value.lower() in ('true', 'false'):
                    return value.lower() == 'true'
                
                # Handle empty strings as None
                if value == "":
                    return None
                
                # Try to convert to numeric types
                try:
                    # Try integer first
                    if '.' not in value and 'e' not in value.lower():
                        return int(value)
                    else:
                        return float(value)
                except ValueError:
                    # Keep as string if conversion fails
                    return value
            
            return value
        except Exception:
            # If any conversion fails, return the original value
            return value

    def update_metadata(self, 
                        data: Callable | Dict[str, Any] | pd.Series,
                        image_names: List[str] | None = None,
                        inplace: bool = True,
                        **kwargs) -> Dict[str, Dict[str, Any]] | None:
        """
        Update metadata for images in the ImageSet with automatic input type detection.
        
        This method provides a flexible, pandas-like API for updating image metadata.
        It automatically detects the input type and handles it appropriately:
        
        - **Function**: Apply a custom function to each image's metadata
        - **Dictionary**: Map metadata keys to constant values or callable functions  
        - **pandas.Series**: Update from external data with image names as index
        
        For functions, the method automatically detects the signature and calls them
        with either `func(metadata_dict)` or `func(metadata_dict, name)` depending
        on the function's parameter requirements.
        
        Args:
            data (Callable | Dict[str, Any] | pd.Series): The input data for updating metadata.
                - **Callable**: Function to apply to each image's metadata. Can have signature
                  `func(metadata_dict, **kwargs)` or `func(metadata_dict, name, **kwargs)`
                  where `name` is the image name.
                - **Dict**: Dictionary mapping metadata keys to values. Values can be 
                  constants or callable functions that take the image name as input.
                - **pd.Series**: Pandas Series with image names as index and new metadata 
                  values. Series name becomes the metadata key.
            image_names (List[str], optional): List of specific image names to update.
                If None, updates all images in the ImageSet.
            inplace (bool): If True, apply changes immediately. If False, return 
                proposed changes without modifying the actual metadata.
            **kwargs: Additional keyword arguments passed to functions (when data is Callable).
                
        Returns:
            Dict[str, Dict[str, Any]] | None: If inplace=False, returns a dictionary 
                mapping image names to their proposed metadata updates. If inplace=True, 
                returns None.
                
        Raises:
            TypeError: If data type is not supported, function signature is invalid, 
                or function returns non-dict.
            ValueError: If image names don't exist in ImageSet.
            RuntimeError: If there are issues with HDF5 file operations.
            
        Examples:
            >>> # Example 1: Dictionary with constant values
            >>> image_set.metadata.update_metadata({
            ...     'experiment': 'growth_assay_1',
            ...     'temperature': 37.0,
            ...     'media_type': 'LB'
            ... })
            
            >>> # Example 2: Dictionary with lambda functions
            >>> image_set.metadata.update_metadata({
            ...     'name_length': lambda name: len(name),
            ...     'plate_row': lambda name: name.split('_')[0] if '_' in name else 'unknown'
            ... })
            
            >>> # Example 3: Custom function for parsing complex metadata
            >>> def parse_colony_metadata(metadata_dict, name):
            ...     # Parse image name like "1_S_3" -> time=1, media=S, replicate=3
            ...     parts = name.split('_')
            ...     if len(parts) >= 3:
            ...         return {
            ...             'timepoint': int(parts[0]),
            ...             'media_condition': parts[1], 
            ...             'biological_replicate': int(parts[2]),
            ...             'colony_count': len(metadata_dict.get('detected_objects', []))
            ...         }
            ...     return {}
            >>> image_set.metadata.update_metadata(parse_colony_metadata)
            
            >>> # Example 4: Pandas Series for external data integration
            >>> growth_rates = pd.Series([0.15, 0.23, 0.18], 
            ...                          index=['img_1', 'img_2', 'img_3'],
            ...                          name='growth_rate')
            >>> image_set.metadata.update_metadata(growth_rates)
            
            >>> # Example 5: Preview mode to review changes before applying
            >>> proposed = image_set.metadata.update_metadata(
            ...     {'quality_score': lambda name: random.uniform(0.7, 1.0)},
            ...     inplace=False
            ... )
            >>> print(proposed)  # Review proposed changes
            >>> # Apply if satisfied
            >>> image_set.metadata.update_metadata(
            ...     {'quality_score': lambda name: random.uniform(0.7, 1.0)}
            ... )
            
            >>> # Example 6: Selective updates on specific images
            >>> image_set.metadata.update_metadata(
            ...     {'processed': True},
            ...     image_names=['img_1', 'img_3']
            ... )
            
            >>> # Example 7: Function with additional parameters
            >>> def calculate_growth_rate(metadata_dict, name, baseline=0.1):
            ...     current_area = metadata_dict.get('colony_area', 0)
            ...     return {'growth_rate': max(0, current_area - baseline)}
            >>> image_set.metadata.update_metadata(
            ...     calculate_growth_rate, 
            ...     baseline=0.05
            ... )
        """
        # Get target image names
        if image_names is None:
            image_names = self._image_set.get_image_names()
        else:
            # Validate that specified image names exist
            available_names = self._image_set.get_image_names()
            invalid_names = [name for name in image_names if name not in available_names]
            if invalid_names:
                raise ValueError(f"Image names not found in ImageSet: {invalid_names}")
        
        # Automatically detect input type and apply appropriate method
        if callable(data):
            proposed_updates = self._apply_function_mapping(data, image_names, **kwargs)
        elif isinstance(data, dict):
            proposed_updates = self._apply_dict_mapping(data, image_names)
        elif isinstance(data, pd.Series):
            proposed_updates = self._apply_series_mapping(data, image_names)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}. Expected Callable, Dict, or pd.Series.")
        
        # Validate all proposed updates
        validated_updates = self._validate_metadata_updates(proposed_updates)
        
        if inplace:
            # Apply updates to HDF5 file
            self._batch_update_hdf5(validated_updates)
            return None
        else:
            # Return proposed updates for preview
            return validated_updates
    
    def _apply_function_mapping(self, func: Callable, image_names: List[str], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Apply a custom function to update metadata for specified images.
        
        Automatically detects function signature and calls appropriately:
        - func(metadata_dict, **kwargs) -> dict
        - func(metadata_dict, name, **kwargs) -> dict
        
        Args:
            func (Callable): Function with flexible signature
            image_names (List[str]): List of image names to process
            **kwargs: Additional arguments passed to the function
            
        Returns:
            Dict[str, Dict[str, Any]]: Mapping of image names to updated metadata dictionaries
            
        Raises:
            TypeError: If function signature is incompatible
        """
        updates = {}
        
        # Detect function signature
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        
        # Determine if function expects name parameter
        expects_name = len(param_names) >= 2 and 'name' in param_names
        
        with h5py.File(self._image_set._out_path, mode='r') as file_handler:
            images_group = file_handler[self._image_set._hdf5_images_group_key]
            
            for image_name in image_names:
                try:
                    if image_name in images_group:
                        image_group = images_group[image_name]
                        current_metadata = self._get_image_metadata(image_group)
                    else:
                        current_metadata = {}
                    
                    # Apply function with appropriate signature
                    try:
                        if expects_name:
                            updated_metadata = func(current_metadata.copy(), image_name, **kwargs)
                        else:
                            updated_metadata = func(current_metadata.copy(), **kwargs)
                            
                        if not isinstance(updated_metadata, dict):
                            raise TypeError(f"Function must return a dictionary, got {type(updated_metadata)}")
                        updates[image_name] = updated_metadata
                    except Exception as e:
                        raise TypeError(f"Error applying function to image '{image_name}': {e}")
                        
                except Exception as e:
                    raise RuntimeError(f"Error processing image '{image_name}': {e}")
        
        return updates
    
    def _apply_dict_mapping(self, mapping: Dict[str, Any], image_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Apply dictionary-based updates to metadata for specified images.
        
        Args:
            mapping (Dict[str, Any]): Dictionary of metadata key-value pairs to update.
                Values can be constants or callable functions that take image_name as input.
            image_names (List[str]): List of image names to process
            
        Returns:
            Dict[str, Dict[str, Any]]: Mapping of image names to updated metadata dictionaries
        """
        updates = {}
        
        with h5py.File(self._image_set._out_path, mode='r') as file_handler:
            images_group = file_handler[self._image_set._hdf5_images_group_key]
            
            for image_name in image_names:
                try:
                    if image_name in images_group:
                        image_group = images_group[image_name]
                        current_metadata = self._get_image_metadata(image_group)
                    else:
                        current_metadata = {}
                    
                    # Start with current metadata
                    updated_metadata = current_metadata.copy()
                    
                    # Apply mapping updates
                    for key, value in mapping.items():
                        if callable(value):
                            try:
                                updated_metadata[key] = value(image_name)
                            except Exception as e:
                                raise ValueError(f"Error applying function for key '{key}' to image '{image_name}': {e}")
                        else:
                            updated_metadata[key] = value
                    
                    updates[image_name] = updated_metadata
                    
                except Exception as e:
                    raise RuntimeError(f"Error processing image '{image_name}': {e}")
        
        return updates
    
    def _apply_series_mapping(self, series: pd.Series, image_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Apply pandas Series-based updates to metadata for specified images.
        
        Args:
            series (pd.Series): Series with image names as index. Values can be:
                - Simple values: Series name becomes the metadata key
                - Dictionaries: Multiple metadata updates per image
            image_names (List[str]): List of image names to process
            
        Returns:
            Dict[str, Dict[str, Any]]: Mapping of image names to updated metadata dictionaries
            
        Raises:
            ValueError: If Series name is missing for simple values or if images are missing
        """
        updates = {}
        
        # Only process images that exist in both image_names and series index
        available_images = [name for name in image_names if name in series.index]
        
        with h5py.File(self._image_set._out_path, mode='r') as file_handler:
            images_group = file_handler[self._image_set._hdf5_images_group_key]
            
            for image_name in available_images:
                try:
                    if image_name in images_group:
                        image_group = images_group[image_name]
                        current_metadata = self._get_image_metadata(image_group)
                    else:
                        current_metadata = {}
                    
                    # Get updates from series
                    series_value = series.loc[image_name]
                    
                    if isinstance(series_value, dict):
                        # Dictionary values: use as-is
                        series_updates = series_value
                    else:
                        # Simple values: use series name as key
                        if series.name is None:
                            raise ValueError("Series must have a name when using simple values")
                        series_updates = {series.name: series_value}
                    
                    # Merge with current metadata
                    updated_metadata = current_metadata.copy()
                    updated_metadata.update(series_updates)
                    
                    updates[image_name] = updated_metadata
                    
                except Exception as e:
                    raise RuntimeError(f"Error processing image '{image_name}': {e}")
        
        return updates
    
    def _validate_metadata_updates(self, updates: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Validate metadata updates to ensure they are compatible with HDF5 storage.
        
        Args:
            updates (Dict[str, Dict[str, Any]]): Proposed metadata updates
            
        Returns:
            Dict[str, Dict[str, Any]]: Validated and cleaned metadata updates
            
        Raises:
            ValueError: If metadata values are not HDF5-compatible
        """
        validated_updates = {}
        
        for image_name, metadata_dict in updates.items():
            validated_metadata = {}
            
            for key, value in metadata_dict.items():
                # Convert key to string
                str_key = str(key)
                
                # Validate and convert value for HDF5 compatibility
                try:
                    validated_value = self._prepare_value_for_hdf5(value)
                    validated_metadata[str_key] = validated_value
                except Exception as e:
                    raise ValueError(f"Invalid metadata value for key '{key}' in image '{image_name}': {e}")
            
            validated_updates[image_name] = validated_metadata
        
        return validated_updates
    
    def _prepare_value_for_hdf5(self, value: Any) -> Any:
        """
        Prepare a metadata value for HDF5 storage with proper type preservation.
        
        Args:
            value: Raw metadata value
            
        Returns:
            Any: Value suitable for HDF5 attribute storage with preserved types
            
        Raises:
            ValueError: If value cannot be converted to a valid HDF5 type
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""  # Empty string for null values
        
        # Handle booleans natively
        if isinstance(value, bool):
            return value  # HDF5 supports native boolean storage
        
        # Handle numeric types natively
        if isinstance(value, (int, float, np.integer, np.floating)):
            return value  # HDF5 supports native numeric storage
        
        # Convert everything else to string
        try:
            return str(value)
        except Exception as e:
            raise ValueError(f"Cannot convert value to HDF5-compatible type: {e}")
    
    def _batch_update_hdf5(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """
        Efficiently update HDF5 file with metadata changes in batch mode.
        
        Args:
            updates (Dict[str, Dict[str, Any]]): Validated metadata updates to apply
            
        Raises:
            RuntimeError: If HDF5 file cannot be opened or updated
        """
        if not updates:
            return
        
        try:
            with h5py.File(self._image_set._out_path, mode='r+', libver='latest') as file_handler:
                images_group = file_handler[self._image_set._hdf5_images_group_key]
                
                for image_name, metadata_dict in updates.items():
                    if image_name in images_group:
                        image_group = images_group[image_name]
                        
                        # Update only public metadata (preserve protected metadata)
                        if 'public_metadata' in image_group:
                            pub_group = image_group['public_metadata']
                            
                            # Get current protected metadata to avoid overwriting
                            protected_keys = set()
                            if 'protected_metadata' in image_group:
                                protected_keys = set(image_group['protected_metadata'].attrs.keys())
                            
                            # Update public metadata attributes
                            for key, value in metadata_dict.items():
                                if key not in protected_keys:  # Only update non-protected keys
                                    pub_group.attrs[key] = self._prepare_value_for_hdf5(value)
                        else:
                            # Create public_metadata group if it doesn't exist
                            pub_group = image_group.create_group('public_metadata')
                            for key, value in metadata_dict.items():
                                pub_group.attrs[key] = self._prepare_value_for_hdf5(value)
                                
        except Exception as e:
            raise RuntimeError(f"Error updating HDF5 file: {e}")

