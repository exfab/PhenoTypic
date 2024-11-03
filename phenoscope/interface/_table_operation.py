import pandas as pd

from ..util.error_message import INTERFACE_ERROR_MSG

class TableOperation:
    def __init__(self):
        raise NotImplementedError(INTERFACE_ERROR_MSG)

    def _operate(self, table: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(INTERFACE_ERROR_MSG)
