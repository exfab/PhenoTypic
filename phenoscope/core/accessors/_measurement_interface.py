import numpy as np
import pandas as pd

from typing import Dict, Union, List, Optional

# TODO: Implement

class MeasurementAccessor:
    def __init__(self):
        self.__measurements: Dict[str, Union[pd.Series, pd.DataFrame]] = {}

    def keys(self) -> List[str]:
        return list(self.__measurements.keys())

    def values(self)->List[Union[pd.Series, pd.DataFrame]]:
        return list(x.copy() for x in self.__measurements.values())

    def __getitem__(self, key: str) -> Union[pd.Series, pd.DataFrame]:
        return self.__measurements[key].copy()

    def __setitem__(self, key: str, value: Union[pd.Series, pd.DataFrame]) -> None:
        if type(key) != str:
            raise TypeError("key must be a string")

        if " " in key:
            raise ValueError("key must not contain spaces")

        if type(value) not in [pd.Series, pd.DataFrame]:
            raise TypeError("Measurement container only supports pd.Series or pd.DataFrame")
        self.__measurements[key] = value

    def __len__(self) -> int:
        return len(self.keys())

    def pop(self, key, exc_type: Optional[str] = 'raise') -> Optional[Union[pd.Series, pd.DataFrame]]:
        """
        Removes the key and returns the corresponding value.
        :param key: The name of the value to remove
        :param exc_type: (optional[str]) Can be either 'raise' or 'ignore'. Default 'raise'. Dictates handling when key is not in dict.
        :return: (optional[Union[pd.Series,pd.DataFrame]]) Returns the corresponding value or None if there is no value and exc_type is 'ignore'.
        """
        if exc_type == 'raise':
            return self.__measurements.pop(key)
        if exc_type == 'ignore':
            return self.__measurements.pop(key, None)

    def clear(self)->None:
        """
        Removes all associated measurements from memory
        :return:
        """
        for key in self.__measurements.keys():
            tmp = self.__measurements.pop(key)
            del tmp

    def merge_on_index_names(self, idx_name_subset: Optional[List[str]] = None,
                             join_type: str = 'outer',
                             verify_integrity=False) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        if type(idx_name_subset) is str:
            idx_name_subset = [idx_name_subset]

        idx_name_list = [df.index.name for df in self.__measurements.values()]
        idx_names = set(idx_name_list)
        if idx_name_subset is None:
            target_index_names = idx_names
        elif set(idx_name_subset).issubset(set(idx_name_list)) is False:
            raise ValueError(
                "the index names in idx_name_subset must be a found in the index names of the measurements.")
        else:
            target_index_names = idx_name_subset

        merged_measurements = {}
        for idx_name in target_index_names:
            current_tables = list(
                measurement for measurement in (self.__measurements.values())
                if measurement.index.name == idx_name
            )

            # In the event the index name appears in the measurements key
            if (measurement_key := idx_name) in self.keys():
                idx_name_appearances = len(list(
                    idx_name_iter for idx_name_iter in self.keys()
                    if idx_name in idx_name_iter
                ))

                measurement_key = f'{idx_name} ({idx_name_appearances})'

            merged_measurements[measurement_key] = pd.concat(objs=current_tables,
                                                             axis=1,
                                                             join=join_type,
                                                             ignore_index=False,
                                                             verify_integrity=verify_integrity,
                                                             copy=True)
        self.__measurements = {**self.__measurements, **merged_measurements}
        return merged_measurements

    def to_dict(self)->Dict[str, Union[pd.Series, pd.DataFrame]]:
        return {key: table.copy() for key, table in self.__measurements.items()}

    def to_recarrays_dict(self)->Dict[str, np.recarray]:
        return {key: table.copy().to_records(index=True) for key, table in self.__measurements.items()}

    def copy(self):
        new_container = self.__class__()
        new_container.__measurements = {**self.__measurements}
        return new_container