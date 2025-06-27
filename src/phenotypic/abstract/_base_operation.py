from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable
import functools, types

if TYPE_CHECKING: from phenotypic import Image


class BaseOperation:
    """BaseOperation is an abstract object intended to be the parent of all other operations.
    It provides the basic functionality for all operations, including measurements."""

    def _get_matched_operation_args(self) -> dict:
        """Returns a dictionary of matched attributes with the arguments for the _operate method. This aids in parallel execution

        Returns:
            dict: A dictionary of matched attributes with the arguments for the _operate method or blank dict if _operate is
            not a staticmethod. This is used for parallel execution of operations.
        """
        raw_operate_method = inspect.getattr_static(self.__class__, '_operate')
        if isinstance(raw_operate_method, staticmethod):
            return self._matched_args(raw_operate_method.__func__)
        else:
            return {}

    def _matched_args(self, func):
        """Return a dict of attributes that satisfy *func*'s signature."""
        sig = inspect.signature(func)
        matched = {}

        for name, param in sig.parameters.items():
            if name == "image":  # The image provided by the user is always passed as the first argument.
                continue
            if hasattr(self, name):
                value = getattr(self, name)
                if isinstance(value, types.MethodType):  # transform a bounded method into a pickleable object
                    value = functools.partial(value.__func__, self)
                matched[name] = value
            elif hasattr(self.__class__, name):
                matched[name] = getattr(self.__class__, name)
            elif param.default is not param.empty:
                continue  # default will be used
            else:
                raise AttributeError(
                    f"{self.__class__.__name__} lacks attribute '{name}' "
                    f"required by {func.__qualname__}",
                )
        return matched
