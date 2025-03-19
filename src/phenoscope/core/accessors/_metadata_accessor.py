import numpy as np
import pandas as pd

from typing import Dict, Union, List, Optional

from phenoscope.util.exceptions_ import MetadataKeySpacesError, MetadataValueNonScalarError

# TODO: Implemenet

class MetadataAccessor:
    SCALAR_TYPES = [int, float, str, bool, np.integer, np.floating, np.bool_, np.complexfloating]

    def __init__(self, handler):
        self._handler = handler
        self.__metadata: Dict[str, Union[int, float, str, bool, np.integer, np.floating, np.bool_, np.complexfloating]] = {}

    def __getitem__(self, key):
        return self.__metadata[key]

    def __setitem__(self, key, value):
        if type(key) != str:
            raise MetadataValueNonScalarError(f'{type(key)}')

        if " " in key:
            raise MetadataKeySpacesError()

        if type(value) not in self.SCALAR_TYPES:
            raise TypeError(f"Value {value} of type {type(value)} not supported for metadata scalar values")

        self.__metadata[key] = value

    def __len__(self):
        return len(self.keys())

    def keys(self) -> List[str]:
        return list(self.__metadata.keys())

    def values(self) -> List[Union[int, float, str, bool, np.integer, np.floating, np.bool_, np.complexfloating]]:
        return list(self.__metadata.values())

    def pop(self, key, exc_type: Optional[str] = 'raise') -> Optional[
        Union[int, float, str, bool, np.integer, np.floating, np.bool_, np.complexfloating]]:
        """Removes the key and returns the corresponding value.

        Args:
            key: The name of the value to remove
            exc_type: (optional[str]) Can be either 'raise' or 'ignore'. Default 'raise'. Dictates handling when key is not in dict.
        Returns:
            (optional[Union[pd.Series,pd.DataFrame]]) Returns the corresponding value or None if there is no value and exc_type is 'ignore'.
        """
        if exc_type == 'raise':
            return self.__metadata.pop(key)
        if exc_type == 'ignore':
            return self.__metadata.pop(key, None)

    def clear(self) -> None:
        """
        Removes all associated metadata from memory
        :return:
        """
        for key in self.__metadata.keys():
            tmp = self.__metadata.pop(key)
            del tmp

    def to_dict(self) -> Dict[str, Union[int, float, str, bool, np.integer, np.floating, np.bool_, np.complexfloating]]:
        return {key: value for key, value in self.__metadata.items()}

    def to_recarray(self) -> np.recarray:
        dtypes = [type(value) for value in self.__metadata.values()]
        names = list(self.__metadata.keys())
        return np.rec.fromarrays(
            arrayList=np.array([[val] for val in list(self.__metadata.values())]),
            dtype=np.dtype(list(zip(names, dtypes)))
        )

    def copy(self):
        """
        return a copy of the metadata
        :return: (MetadataContainer) Returns a copy of the metadata
        """
        new_container = self.__class__()
        new_container.__metadata = {**self.__metadata}
        return new_container
