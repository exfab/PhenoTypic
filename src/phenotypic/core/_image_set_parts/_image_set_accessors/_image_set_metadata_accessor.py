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

    @staticmethod
    def _get_image_metadata(image_group) -> Dict[str, (str | int | float | bool)]:
        prot_metadata = image_group['protected_metadata'].attrs
        pub_metadata = image_group['public_metadata'].attrs
        return {**prot_metadata, **pub_metadata}

    @staticmethod
    def _set_image_metadata(image_group, new_metadata: dict):
        # only keep changes to public metadata. Changes to protected metadata should only occur during pipeline processing
        protected_group = image_group['protected_metadata'].attrs
        filtered_metadata = {
            key: new_metadata[key]
            for key in new_metadata
            if key not in protected_group
        }

        pub_metadata = image_group['public_metadata'].attrs
        for key, value in filtered_metadata.items():
            pub_metadata[key] = value

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
        Converts HDF5 attribute values to appropriate Python types.
        
        Args:
            value: Raw HDF5 attribute value.
            
        Returns:
            Any: Converted value suitable for pandas DataFrame.
        """
        try:
            # Handle bytes (common in HDF5)
            if isinstance(value, bytes):
                return value.decode('utf-8')
            
            # Handle numpy scalars
            if hasattr(value, 'item'):
                return value.item()
            
            # Handle string representations that might be numeric
            if isinstance(value, str):
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
                       func: Optional[Callable] = None,
                       mapping: Optional[Dict[str, Any]] = None, 
                       series: Optional[pd.Series] = None,
                       image_names: Optional[List[str]] = None,
                       inplace: bool = True,
                       **kwargs) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Update metadata for images in the image set using flexible input methods.
        
        This method provides a pandas-like interface for updating image metadata,
        supporting dictionary mappings, custom functions, and pandas Series inputs.
        Only public metadata is updated; protected metadata remains unchanged.
        
        Args:
            func (Optional[Callable]): Function to apply to each image's metadata.
                Function signature can be either:
                - ``func(metadata_dict) -> dict`` for functions that only need metadata
                - ``func(metadata_dict, name) -> dict`` for functions that need the image name
                The system automatically detects which signature to use.
            mapping (Optional[Dict[str, Any]]): Dictionary of key-value pairs to update.
                Values can be constants or callable functions that take image_name as input.
            series (Optional[pd.Series]): Pandas Series with image names as index and
                dictionaries of metadata updates as values.
            image_names (Optional[List[str]]): List of specific image names to update.
                If None, updates all images in the image set.
            inplace (bool): If True, updates are written to HDF5 file immediately.
                If False, returns dictionary of proposed updates without writing.
            **kwargs: Additional keyword arguments passed to the update function.
                
        Returns:
            Optional[Dict[str, Dict[str, Any]]]: If inplace=False, returns dictionary
                mapping image names to their updated metadata. If inplace=True, returns None.
                
        Raises:
            ValueError: If multiple input methods are provided simultaneously or if
                no input method is specified.
            KeyError: If specified image names don't exist in the image set.
            TypeError: If function signature is incompatible or inputs are invalid.
            
        Examples:
            Update metadata using a dictionary mapping:
            
            .. code-block:: python
            
                # Simple key-value updates
                image_set.metadata.update_metadata(mapping={
                    'experiment_date': '2024-01-15',
                    'researcher': 'Dr. Smith'
                })
                
                # Function-based mapping for specific keys
                image_set.metadata.update_metadata(mapping={
                    'timepoint': lambda name: int(name.split('_')[0]),
                    'condition': lambda name: name.split('_')[1]
                })
            
            Update metadata using a custom function:
            
            Custom functions can have either of these signatures:
            - ``func(metadata_dict, **kwargs) -> dict`` for functions that only need metadata
            - ``func(metadata_dict, name, **kwargs) -> dict`` for functions that need the image name
            The system automatically detects which signature your function uses.
            
            .. code-block:: python
            
                # Example 1: Parse experimental info from image names like "1_control_3"
                def parse_experimental_metadata(metadata_dict, name):
                    '''Extract experimental info from image name.
                    
                    Args:
                        metadata_dict (dict): Current metadata for this image
                        name (str): Name of the image (e.g., "1_control_3")
                        
                    Returns:
                        dict: Updated metadata dictionary
                    '''
                    parts = name.split('_')
                    
                    # Start with existing metadata and add new fields
                    updated_metadata = metadata_dict.copy()
                    updated_metadata.update({
                        'timepoint': int(parts[0]),        # "1" -> 1
                        'treatment': parts[1],             # "control"
                        'replicate': int(parts[2]),        # "3" -> 3
                        'parsed_from_name': True,
                        'original_name': name
                    })
                    return updated_metadata
                
                # Apply to all images in the image set
                image_set.metadata.update_metadata(func=parse_experimental_metadata)
                
                # Apply to specific images only
                image_set.metadata.update_metadata(
                    func=parse_experimental_metadata,
                    image_names=['1_control_1', '2_treatment_1']
                )
                
                # Example 2: Function that only uses metadata (no image_name needed)
                def parse_from_existing_metadata(metadata_dict):
                    '''Generate new metadata from existing metadata fields only.
                    
                    This function demonstrates the simpler signature that only takes
                    metadata_dict as input. No image_name parameter needed!
                    
                    Args:
                        metadata_dict (dict): Current metadata
                        
                    Returns:
                        dict: Updated metadata with derived fields
                    '''
                    updated = metadata_dict.copy()
                    
                    # Parse from existing 'sample_id' field if it exists
                    if 'sample_id' in metadata_dict:
                        sample_id = metadata_dict['sample_id']
                        if '_' in sample_id:
                            parts = sample_id.split('_')
                            updated['plate_number'] = parts[0]
                            updated['well_position'] = parts[1] if len(parts) > 1 else 'unknown'
                    
                    # Generate metadata from 'treatment_code' field
                    if 'treatment_code' in metadata_dict:
                        code = metadata_dict['treatment_code']
                        # Map treatment codes to readable names
                        treatment_map = {
                            'CTL': 'control',
                            'ABX': 'antibiotic',
                            'GRW': 'growth_medium'
                        }
                        updated['treatment_name'] = treatment_map.get(code, 'unknown')
                        updated['is_control'] = (code == 'CTL')
                    
                    # Calculate derived values from numeric metadata
                    if 'concentration' in metadata_dict and 'volume' in metadata_dict:
                        try:
                            conc = float(metadata_dict['concentration'])
                            vol = float(metadata_dict['volume'])
                            updated['total_amount'] = conc * vol
                            updated['concentration_category'] = 'high' if conc > 10 else 'low'
                        except (ValueError, TypeError):
                            pass  # Skip if conversion fails
                    
                    return updated
                
                # Example 3: Complex function using multiple metadata sources
                def calculate_growth_metrics(metadata_dict, name, 
                                           reference_timepoint=0):
                    '''Calculate growth metrics from existing timepoint and measurement data.
                    
                    Args:
                        metadata_dict (dict): Current metadata
                        name (str): Image identifier  
                        reference_timepoint (int): Reference time for growth calculations
                        
                    Returns:
                        dict: Updated metadata with calculated growth metrics
                    '''
                    updated = metadata_dict.copy()
                    
                    # Calculate growth rate if timepoint and colony_count exist
                    if all(key in metadata_dict for key in ['timepoint', 'colony_count']):
                        try:
                            timepoint = float(metadata_dict['timepoint'])
                            colony_count = float(metadata_dict['colony_count'])
                            
                            # Calculate growth metrics
                            time_elapsed = timepoint - reference_timepoint
                            if time_elapsed > 0:
                                updated['growth_rate'] = colony_count / time_elapsed
                                updated['doubling_time'] = time_elapsed / colony_count if colony_count > 0 else float('inf')
                            
                            # Categorize growth phase
                            if colony_count < 10:
                                updated['growth_phase'] = 'lag'
                            elif colony_count < 100:
                                updated['growth_phase'] = 'exponential'
                            else:
                                updated['growth_phase'] = 'stationary'
                                
                        except (ValueError, TypeError):
                            updated['growth_rate'] = 'calculation_failed'
                    
                    # Add quality control flags based on existing metadata
                    qc_flags = []
                    if 'temperature' in metadata_dict:
                        temp = float(metadata_dict.get('temperature', 0))
                        if temp < 20 or temp > 40:
                            qc_flags.append('temperature_out_of_range')
                    
                    if 'ph' in metadata_dict:
                        ph = float(metadata_dict.get('ph', 7))
                        if ph < 6 or ph > 8:
                            qc_flags.append('ph_out_of_range')
                    
                    updated['qc_flags'] = ','.join(qc_flags) if qc_flags else 'passed'
                    
                    return updated
                
                # Example 4: Function with only metadata parameter (cleaner signature)
                def normalize_treatment_data(metadata_dict, default_dose=1.0):
                    '''Normalize and clean treatment data using only metadata.
                    
                    Args:
                        metadata_dict (dict): Current metadata
                        default_dose (float): Default dose if missing
                        
                    Returns:
                        dict: Updated metadata with normalized treatment data
                    '''
                    updated = metadata_dict.copy()
                    
                    # Normalize treatment names to lowercase
                    if 'treatment' in metadata_dict:
                        updated['treatment_normalized'] = metadata_dict['treatment'].lower().strip()
                    
                    # Ensure dose is present and numeric
                    if 'dose' not in metadata_dict or not metadata_dict['dose']:
                        updated['dose'] = default_dose
                    else:
                        try:
                            updated['dose'] = float(metadata_dict['dose'])
                        except (ValueError, TypeError):
                            updated['dose'] = default_dose
                    
                    # Calculate dose category
                    dose_val = float(updated['dose'])
                    if dose_val == 0:
                        updated['dose_category'] = 'control'
                    elif dose_val < 1.0:
                        updated['dose_category'] = 'low'
                    elif dose_val < 10.0:
                        updated['dose_category'] = 'medium'
                    else:
                        updated['dose_category'] = 'high'
                    
                    return updated
                
                # Apply functions - system automatically detects signatures
                image_set.metadata.update_metadata(func=parse_from_existing_metadata)
                
                image_set.metadata.update_metadata(
                    func=calculate_growth_metrics,
                    reference_timepoint=0  # passed as kwarg
                )
                
                # Function with only metadata parameter
                image_set.metadata.update_metadata(
                    func=normalize_treatment_data,
                    default_dose=0.5  # passed as kwarg
                )
            
            Update metadata using a pandas Series:
            
            .. code-block:: python
            
                # Create Series with image names as index
                external_data = pd.Series([
                    {'drug_concentration': 10.0, 'batch': 'A'},
                    {'drug_concentration': 20.0, 'batch': 'B'}
                ], index=['img_001', 'img_002'])
                
                image_set.metadata.update_metadata(series=external_data)
            
            Preview updates without applying them:
            
            .. code-block:: python
            
                # Get proposed updates without writing to file
                proposed_updates = image_set.metadata.update_metadata(
                    mapping={'new_field': 'test_value'},
                    inplace=False
                )
                
                # Review updates before applying
                for image_name, updates in proposed_updates.items():
                    print(f"{image_name}: {updates}")
                    
                # Apply updates after review
                image_set.metadata.update_metadata(
                    mapping={'new_field': 'test_value'},
                    inplace=True
                )
        """
        # Validate input parameters
        input_methods = [func is not None, mapping is not None, series is not None]
        if sum(input_methods) != 1:
            raise ValueError("Exactly one of 'func', 'mapping', or 'series' must be provided")
        
        # Get target image names
        all_image_names = self._image_set.get_image_names()
        if image_names is None:
            target_images = all_image_names
        else:
            # Validate that specified images exist
            missing_images = set(image_names) - set(all_image_names)
            if missing_images:
                raise KeyError(f"Images not found in image set: {missing_images}")
            target_images = image_names
        
        if not target_images:
            return {} if not inplace else None
        
        # Apply the appropriate update method
        if func is not None:
            updates = self._apply_function_mapping(func, target_images, **kwargs)
        elif mapping is not None:
            updates = self._apply_dict_mapping(mapping, target_images)
        else:  # series is not None
            updates = self._apply_series_mapping(series, target_images)
        
        # Validate updates before applying
        validated_updates = self._validate_metadata_updates(updates)
        
        if inplace:
            self._batch_update_hdf5(validated_updates)
            return None
        else:
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
            series (pd.Series): Series with image names as index and metadata dictionaries as values
            image_names (List[str]): List of image names to process
            
        Returns:
            Dict[str, Dict[str, Any]]: Mapping of image names to updated metadata dictionaries
            
        Raises:
            ValueError: If series values are not dictionaries or if required images are missing
        """
        updates = {}
        
        # Check that all target images have corresponding series entries
        missing_in_series = set(image_names) - set(series.index)
        if missing_in_series:
            raise ValueError(f"Images not found in series index: {missing_in_series}")
        
        with h5py.File(self._image_set._out_path, mode='r') as file_handler:
            images_group = file_handler[self._image_set._hdf5_images_group_key]
            
            for image_name in image_names:
                try:
                    if image_name in images_group:
                        image_group = images_group[image_name]
                        current_metadata = self._get_image_metadata(image_group)
                    else:
                        current_metadata = {}
                    
                    # Get updates from series
                    series_updates = series.loc[image_name]
                    if not isinstance(series_updates, dict):
                        raise ValueError(f"Series value for '{image_name}' must be a dictionary, got {type(series_updates)}")
                    
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
    
    def _prepare_value_for_hdf5(self, value: Any) -> str:
        """
        Prepare a metadata value for HDF5 storage by converting to string.
        
        Args:
            value: Raw metadata value
            
        Returns:
            str: String representation suitable for HDF5 attribute storage
            
        Raises:
            ValueError: If value cannot be converted to a valid string
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""  # Empty string for null values
        
        try:
            return str(value)
        except Exception as e:
            raise ValueError(f"Cannot convert value to string: {e}")
    
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
            with h5py.File(self._image_set._out_path, mode='r+') as file_handler:
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
                                    pub_group.attrs[key] = value
                        else:
                            # Create public_metadata group if it doesn't exist
                            pub_group = image_group.create_group('public_metadata')
                            for key, value in metadata_dict.items():
                                pub_group.attrs[key] = value
                                
        except Exception as e:
            raise RuntimeError(f"Error updating HDF5 file: {e}")

