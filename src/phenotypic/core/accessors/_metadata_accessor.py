from __future__ import annotations
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING: from phenotypic import Image

from collections import ChainMap


class MetadataAccessor:
    """An accessor for image metadata that manages read/write permissions related to the metadata information."""

    def __init__(self, image: Image) -> None:
        self._parent_image = image
        self._combined_metadata = ChainMap(self._private_metadata, self._protected_metadata, self._public_metadata)

    @property
    def _private_metadata(self):
        return self._parent_image._metadata.private

    @property
    def _protected_metadata(self):
        return self._parent_image._metadata.protected

    @property
    def _public_metadata(self):
        return self._parent_image._metadata.public

    @property
    def _public_protected_metadata(self):
        return ChainMap(self._public_metadata, self._protected_metadata)

    def keys(self):
        return self._combined_metadata.keys()

    def values(self):
        return self._combined_metadata.values()

    def items(self):
        return self._combined_metadata.items()

    def __contains__(self, key):
        return key in self.keys()

    def __getitem__(self, key):
        if key in self._private_metadata:
            return self._private_metadata[key]
        elif key in self._protected_metadata:
            return self._protected_metadata[key]
        elif key in self._public_metadata:
            return self._public_metadata[key]
        else:
            raise KeyError

    def __setitem__(self, key, value):
        if not isinstance(value, (str, int, float, bool, None)):
            raise ValueError('Metadata values must be of scalar types or None.')
        if key in self._private_metadata:
            raise PermissionError('Private metadata cannot be modified.')
        elif key in self._protected_metadata:
            self._protected_metadata[key] = value
        elif key in self._public_metadata:
            self._public_metadata[key] = value
        else:
            raise KeyError

    def __delitem__(self, key):
        if key in self._private_metadata or key in self._protected_metadata:
            raise PermissionError('Private and protected metadata cannot be removed.')
        elif key in self._public_metadata:
            del self._public_metadata[key]
        else:
            raise KeyError

    def pop(self, key):
        if key in self._private_metadata or key in self._protected_metadata:
            raise PermissionError('Private and protected metadata cannot be removed.')
        elif key in self._public_metadata:
            return self._public_metadata.pop(key)
        else:
            raise KeyError

    def get(self, key, default=None):
        if key in self._combined_metadata:
            return self._combined_metadata[key]
        else:
            return default

    def insert_metadata(self, df: pd.DataFrame, inplace=False, allow_duplicates=False) -> pd.DataFrame:
        working_df = df if inplace else df.copy()
        for key, value in self._public_protected_metadata.items():
            working_df.insert(loc=0, column=f'Metadata_{key}', value=value, allow_duplicates=allow_duplicates)
        return working_df
