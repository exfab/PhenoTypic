from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from collections import ChainMap

class MetadataAccessor:
    """An accessor for image metadata that manages read/write permissions related to the metadata information."""

    def __init__(self, image: Image) -> None:
        self._parent_image = image
        self._combined_metadata = ChainMap(self._private_metadata, self._protected_metadata, self._public_metadata)

    @property
    def _private_metadata(self):
        return self._parent_image._private_metadata

    @property
    def _protected_metadata(self):
        return self._parent_image._protected_metadata

    @property
    def _public_metadata(self):
        return self._parent_image._public_metadata

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
        if key in self._private_metadata:
            raise PermissionError('Private metadata cannot be modified.')
        elif key in self._protected_metadata:
            self._protected_metadata[key] = value
        elif key in self._public_metadata:
            self._public_metadata[key] = value
        else:
            raise KeyError

    def pop(self, key):
        if key in self._private_metadata or key in self._protected_metadata:
            raise PermissionError('Private and protected metadata cannot be removed.')
        elif key in self._public_metadata:
            return self._public_metadata.pop(key)
        else:
            raise KeyError
