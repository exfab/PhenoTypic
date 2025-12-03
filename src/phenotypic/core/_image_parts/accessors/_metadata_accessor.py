from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from phenotypic import Image
from phenotypic.tools.constants_ import METADATA
from collections import ChainMap


class MetadataAccessor:
    """Accessor for managing image metadata with hierarchical read/write permissions.

    This class provides dictionary-like access to image metadata with three permission levels:
    private (read-only), protected (read/write), and public (read/write/delete). All metadata
    is combined using ChainMap for unified access while preserving permission constraints.

    Private metadata is typically reserved for internal use (e.g., UUID), protected metadata
    contains system properties (e.g., image name, type), and public metadata contains user-defined
    or imported metadata that can be freely modified.

    Attributes:
        _parent_image (Image): The parent Image instance containing the metadata storage.

    Examples:
        .. dropdown:: Access metadata like a dictionary

            .. code-block:: python

                img = Image(arr, name='sample')
                # Get metadata value
                image_name = img.metadata['ImageName']
                # Set public metadata
                img.metadata['user_notes'] = 'A sample image'
                # Check if key exists
                if 'user_notes' in img.metadata:
                    print(img.metadata['user_notes'])

        .. dropdown:: Iterate through metadata

            .. code-block:: python

                for key, value in img.metadata.items():
                    print(f'{key}: {value}')
    """

    def __init__(self, image: Image) -> None:
        """Initialize the metadata accessor.

        Args:
            image (Image): The parent Image instance containing the metadata storage.
        """
        self._parent_image = image

    @property
    def _combined_metadata(self):
        """ChainMap combining all metadata levels (private, protected, public).

        Returns:
            ChainMap: A ChainMap with private metadata at highest priority, then protected,
                then public. Enables unified read access while maintaining search order.
        """
        return ChainMap(
            self._private_metadata, self._protected_metadata, self._public_metadata
        )

    @property
    def _private_metadata(self):
        """Access the private metadata dictionary from the parent image.

        Private metadata is read-only and cannot be modified or deleted.
        Typically contains internal system information like UUID.

        Returns:
            dict[str, Any]: The private metadata dictionary.
        """
        return self._parent_image._metadata.private

    @property
    def _protected_metadata(self):
        """Access the protected metadata dictionary from the parent image.

        Protected metadata can be read and modified but cannot be deleted.
        Typically contains system properties like image name, type, and bit depth.

        Returns:
            dict[str, Union[int, str, float, bool, np.nan]]: The protected metadata dictionary.
        """
        return self._parent_image._metadata.protected

    @property
    def _public_metadata(self):
        """Access the public metadata dictionary from the parent image.

        Public metadata can be read, modified, and deleted without restrictions.
        Typically contains user-defined metadata or metadata imported from files.

        Returns:
            dict[str, Union[int, str, float, bool, np.nan]]: The public metadata dictionary.
        """
        return self._parent_image._metadata.public

    @property
    def _public_protected_metadata(self):
        """ChainMap combining public and protected metadata.

        Returns:
            ChainMap: A ChainMap with public metadata at highest priority, then protected.
                Used for operations that should include both modifiable levels.
        """
        return ChainMap(self._public_metadata, self._protected_metadata)

    def keys(self):
        """Get all metadata keys across all permission levels.

        Returns:
            KeysView: A view of all keys from combined metadata (private, protected, public).
                Keys from private metadata take precedence in the view.
        """
        return self._combined_metadata.keys()

    def values(self):
        """Get all metadata values across all permission levels.

        Returns:
            ValuesView: A view of all values from combined metadata (private, protected, public).
                Values from private metadata take precedence.
        """
        return self._combined_metadata.values()

    def items(self):
        """Get all metadata key-value pairs across all permission levels.

        Returns:
            ItemsView: A view of all key-value pairs from combined metadata (private,
                protected, public). Items from private metadata take precedence.
        """
        return self._combined_metadata.items()

    def __contains__(self, key):
        """Check if a metadata key exists at any permission level.

        Args:
            key: The metadata key to check.

        Returns:
            bool: True if the key exists in private, protected, or public metadata.
        """
        return key in self.keys()

    def __getitem__(self, key):
        """Retrieve a metadata value by key with hierarchical search.

        Searches in order: private -> protected -> public. Returns the first match found.

        Args:
            key: The metadata key to retrieve.

        Returns:
            Any: The metadata value associated with the key.

        Raises:
            KeyError: If the key does not exist in any metadata level.
        """
        if key in self._private_metadata:
            return self._private_metadata[key]
        elif key in self._protected_metadata:
            return self._protected_metadata[key]
        elif key in self._public_metadata:
            return self._public_metadata[key]
        else:
            raise KeyError

    def __setitem__(self, key, value):
        """Set a metadata value with validation and permission checking.

        Only scalar types (str, int, float, bool) or None are allowed as values.
        If the key exists in protected metadata, updates the protected value.
        Otherwise, creates or updates a public metadata entry.
        Private metadata cannot be modified.

        Args:
            key: The metadata key to set.
            value: The metadata value (must be str, int, float, bool, or None).

        Raises:
            ValueError: If value is not a scalar type or None.
            PermissionError: If attempting to modify private metadata.

        Examples:
            .. dropdown:: Set metadata values with permission checking

                .. code-block:: python

                    img.metadata['resolution'] = 300  # Creates public metadata
                    img.metadata['ImageName'] = 'updated_name'  # Updates protected metadata
        """
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError("Metadata values must be of scalar types or None.")
        if key in self._private_metadata:
            raise PermissionError("Private metadata cannot be modified.")
        elif key in self._protected_metadata:
            self._protected_metadata[key] = value
        else:
            self._public_metadata[key] = value

    def __delitem__(self, key):
        """Delete a metadata entry with permission checking.

        Only public metadata can be deleted. Private and protected metadata
        cannot be removed.

        Args:
            key: The metadata key to delete.

        Raises:
            PermissionError: If attempting to delete private or protected metadata.
            KeyError: If the key does not exist in public metadata.

        Examples:
            .. dropdown:: Delete public metadata entries

                .. code-block:: python

                    del img.metadata['user_notes']  # Deletes public metadata
        """
        if key in self._private_metadata or key in self._protected_metadata:
            raise PermissionError("Private and protected metadata cannot be removed.")
        elif key in self._public_metadata:
            del self._public_metadata[key]
        else:
            raise KeyError

    def pop(self, key):
        """Remove and return a metadata value.

        Only public metadata can be popped. Private and protected metadata
        cannot be removed.

        Args:
            key: The metadata key to remove.

        Returns:
            Any: The value associated with the key before removal.

        Raises:
            PermissionError: If attempting to pop private or protected metadata.
            KeyError: If the key does not exist in public metadata.

        Examples:
            .. dropdown:: Remove and return a public metadata value

                .. code-block:: python

                    old_value = img.metadata.pop('user_notes')
        """
        if key in self._private_metadata or key in self._protected_metadata:
            raise PermissionError("Private and protected metadata cannot be removed.")
        elif key in self._public_metadata:
            return self._public_metadata.pop(key)
        else:
            raise KeyError

    def get(self, key, default=None):
        """Retrieve a metadata value with a default fallback.

        Searches across all permission levels (private -> protected -> public)
        and returns the first match found.

        Args:
            key: The metadata key to retrieve.
            default: The value to return if the key is not found. Defaults to None.

        Returns:
            Any: The metadata value if found, otherwise the default value.

        Examples:
            .. dropdown:: Retrieve metadata with default fallback

                .. code-block:: python

                    resolution = img.metadata.get('resolution', 100)
                    name = img.metadata.get('ImageName')  # Returns None if not found
        """
        if key in self._combined_metadata:
            return self._combined_metadata[key]
        else:
            return default

    def insert_metadata(
        self, df: pd.DataFrame, inplace=False, allow_duplicates=False
    ) -> pd.DataFrame:
        """Insert metadata as columns into a DataFrame.

        Adds public and protected metadata as new columns at the beginning of the DataFrame.
        Column names are prefixed with 'Metadata_' if not already present. Image name is
        retrieved from the parent image instance rather than metadata storage.

        Args:
            df (pd.DataFrame): The DataFrame to insert metadata columns into.
            inplace (bool, optional): If True, modifies the input DataFrame in place.
                If False, creates a copy before modification. Defaults to False.
            allow_duplicates (bool, optional): If True, allows duplicate column names
                to be inserted. If False, skips insertion for columns that already exist.
                Defaults to False.

        Returns:
            pd.DataFrame: The DataFrame with metadata columns inserted at the beginning
                (position 0). If inplace=True, returns the same object as input.

        Notes:
            - Only public and protected metadata are included (private metadata is excluded)
            - IMAGE_NAME metadata is populated from parent_image.name instead of the metadata dict
            - Columns are inserted from right to left at position 0, so iteration order determines final order
            - Metadata columns without 'Metadata_' prefix are automatically prefixed

        Examples:
            .. dropdown:: Insert metadata as DataFrame columns

                .. code-block:: python

                    import pandas as pd
                    df = pd.DataFrame({'data': [1, 2, 3]})
                    img = Image(arr, name='sample')
                    img.metadata['resolution'] = 300
                    result_df = img.metadata.insert_metadata(df)
                    # result_df now has Metadata_ImageName and Metadata_resolution columns at position 0
        """
        working_df = df if inplace else df.copy()
        for key, value in self._public_protected_metadata.items():
            if key == METADATA.IMAGE_NAME:
                value = (
                    self._parent_image.name
                )  # offload handling to image handler class
            if not key.startswith(f"Metadata_"):
                header = f"Metadata_{key}"
            else:
                header = key
            if header not in working_df.columns:
                working_df.insert(
                    loc=0, column=header, value=value, allow_duplicates=allow_duplicates
                )
        return working_df

    def table(self) -> pd.Series:
        """Convert metadata to a pandas Series.

        Creates a Series containing all metadata (private, protected, and public)
        with the parent image name as the Series name.

        Returns:
            pd.Series: A Series where the index is metadata keys and values are
                metadata values. The Series name is the parent image name.

        Examples:
            .. dropdown:: Convert metadata to pandas Series

                .. code-block:: python

                    img = Image(arr, name='sample_image')
                    img.metadata['resolution'] = 300
                    series = img.metadata.table()
                    print(series.name)  # 'sample_image'
                    print(series['ImageName'])  # 'sample_image'
        """
        return pd.Series(
            self._combined_metadata,
            name=self._parent_image.name,
        )
