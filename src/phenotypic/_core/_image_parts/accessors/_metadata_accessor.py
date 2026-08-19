from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from phenotypic._core._image import Image
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import metadata_member_for_header, metadata_member_for_label
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
        Access metadata like a dictionary:

        >>> img = Image(arr, name='sample')
        >>> # Get metadata value
        >>> image_name = img.metadata['ImageName']
        >>> # Set public metadata
        >>> img.metadata['user_notes'] = 'A sample image'
        >>> # Check if key exists
        >>> if 'user_notes' in img.metadata:
        ...     print(img.metadata['user_notes'])

        Iterate through metadata:

        >>> for key, value in img.metadata.items():
        ...     print(f'{key}: {value}')
    """

    def __init__(self, image: Image) -> None:
        """Initialize the metadata accessor.

        Args:
            image (Image): The parent Image instance containing the metadata storage.
        """
        self._parent_image = image

    @staticmethod
    def _resolve_key(key):
        """Resolve known metadata aliases to their owning schema member.

        Storage predates the flat namespace, so public metadata may arrive as a
        bare label, the current per-topic header, or a future flat header. The
        enum member is the common identity across those spellings and prevents a
        public alias from shadowing protected or private image bookkeeping.
        Unknown user keys remain literal keys.
        """
        member = metadata_member_for_header(str(key))
        if member is None:
            member = metadata_member_for_label(str(key))
        return member if member is not None else key

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
        return self._resolve_key(key) in self.keys()

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
        key = self._resolve_key(key)
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
            Set metadata values with permission checking:

            >>> img.metadata['resolution'] = 300  # Creates public metadata
            >>> img.metadata['ImageName'] = 'updated_name'  # Updates protected metadata
        """
        key = self._resolve_key(key)
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
            Delete public metadata entries:

            >>> del img.metadata['user_notes']  # Deletes public metadata
        """
        key = self._resolve_key(key)
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
            Remove and return a public metadata value:

            >>> old_value = img.metadata.pop('user_notes')
        """
        key = self._resolve_key(key)
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
            Retrieve metadata with default fallback:

            >>> resolution = img.metadata.get('resolution', 100)
            >>> name = img.metadata.get('ImageName')  # Returns None if not found
        """
        key = self._resolve_key(key)
        if key in self._combined_metadata:
            return self._combined_metadata[key]
        else:
            return default

    def insert_metadata(
            self, df: pd.DataFrame, inplace=False, allow_duplicates=False
    ) -> pd.DataFrame:
        """Insert metadata as columns into a DataFrame.

        Adds public and protected metadata as new columns at the beginning of the DataFrame.
        Column names are prefixed with ``Metadata_`` if not already present. Image name is
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
            - Columns are inserted from right to left at position 0, so iteration order
              determines final order; iteration follows the bio-semantic cluster order
              (Identity -> Strain -> Condition -> Design, via ``canonical_metadata_order``),
              not REMBI module order
            - Metadata columns without a ``Metadata<Topic>_`` prefix are automatically prefixed
              via the schema (e.g. ``Strain`` -> ``Metadata_Strain``; unknown labels
              fall back to a generic ``Metadata_`` prefix)

        Examples:
            Insert metadata as DataFrame columns:

            >>> import pandas as pd
            >>> df = pd.DataFrame({'data': [1, 2, 3]})
            >>> img = Image(arr, name='sample')
            >>> img.metadata['resolution'] = 300
            >>> result_df = img.metadata.insert_metadata(df)
            >>> # result_df now has Metadata_ImageName and Metadata_resolution columns at position 0
        """
        working_df = df if inplace else df.copy()
        # Insert metadata columns in canonical bio-semantic cluster order (then
        # definition order, then alpha for unknown tags). insert() places each
        # column at loc=0, so iterate in reverse rank to land the lowest-rank
        # category (Identity) at the leftmost position.
        from phenotypic.sdk_ import (
            canonical_metadata_order,
            ensure_metadata_prefix,
        )

        rank = canonical_metadata_order()
        # Unknown/uncategorized tags sort after every known header (1000-stride
        # ranks, so len(rank) is not a valid sentinel — mirrors
        # order_measurement_columns). reverse=True + insert(loc=0) lands the
        # lowest-rank category (Identity) leftmost and unknown tags at the tail
        # of the front block.
        unknown_rank = max(rank.values(), default=0) + 1

        def _rank(item):
            header = ensure_metadata_prefix(item[0])
            return (rank.get(header, unknown_rank), str(item[0]))

        # Resolve aliases before combining permission tiers. Protected framework
        # fields win over a stale public spelling of the same schema member.
        resolved_items = {
            self._resolve_key(key): value
            for key, value in self._public_metadata.items()
        }
        resolved_items.update(
            {
                self._resolve_key(key): value
                for key, value in self._protected_metadata.items()
            }
        )
        items = sorted(resolved_items.items(), key=_rank, reverse=True)
        existing_members = {
            metadata_member_for_header(str(column))
            or metadata_member_for_label(str(column))
            for column in working_df.columns
        }
        for key, value in items:
            if key == IMAGE.IMAGE_NAME:
                value = (
                    self._parent_image.name
                )  # offload handling to image handler class
            header = ensure_metadata_prefix(key)
            member = metadata_member_for_header(str(key))
            if header not in working_df.columns and (
                member is None or member not in existing_members
            ):
                working_df.insert(
                        loc=0, column=header, value=value,
                        allow_duplicates=allow_duplicates
                )
                existing_members.add(member)
        return working_df

    def by_module(self, module) -> dict:
        """Group metadata keys/values by REMBI module (read-only view).

        Framework private/protected keys (e.g. ``ImageName``, ``UUID``) map to
        :attr:`~phenotypic.schema.REMBI_MODULE.IMAGE_DATA`; public tags resolve
        via the schema reverse index; unrecognized keys fall to
        :attr:`~phenotypic.schema.REMBI_MODULE.UNCATEGORIZED`.

        Args:
            module: A :class:`~phenotypic.schema.REMBI_MODULE` or its string
                value (e.g. ``"ImageData"``).

        Returns:
            dict: ``{key: value}`` for every metadata entry resolving to
            *module*, in combined-metadata iteration order.
        """
        from phenotypic.schema import REMBI_MODULE, header_to_module
        from phenotypic.sdk_ import ensure_metadata_prefix

        target = module if isinstance(module, REMBI_MODULE) else REMBI_MODULE(module)
        idx = header_to_module()
        out: dict = {}
        for key, value in self._combined_metadata.items():
            # Resolve to the full schema header (bare "Strain" ->
            # ``str(GENETIC.STRAIN)``) so the reverse index, keyed on the
            # per-topic Scheme-B headers, resolves the REMBI module.
            header = ensure_metadata_prefix(key)
            mod = idx.get(header)
            if mod is None:
                mod = (
                    REMBI_MODULE.IMAGE_DATA
                    if key in self._private_metadata or key in self._protected_metadata
                    else REMBI_MODULE.UNCATEGORIZED
                )
            if mod is target:
                out[key] = value
        return out

    def table(self) -> pd.Series:
        """Convert metadata to a pandas Series.

        Creates a Series containing all metadata (private, protected, and public)
        with the parent image name as the Series name.

        Returns:
            pd.Series: A Series where the index is metadata keys and values are
                metadata values. The Series name is the parent image name.

        Examples:
            Convert metadata to pandas Series:

            >>> img = Image(arr, name='sample_image')
            >>> img.metadata['resolution'] = 300
            >>> series = img.metadata.table()
            >>> print(series.name)  # 'sample_image'
            >>> print(series['ImageName'])  # 'sample_image'
        """
        return pd.Series(
                self._combined_metadata,
                name=self._parent_image.name,
        )
