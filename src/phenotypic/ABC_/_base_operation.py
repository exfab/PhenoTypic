from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable
import functools, types

if TYPE_CHECKING: from phenotypic import Image

import logging
import tracemalloc

try:
    from pympler import muppy, summary

    PYMPLER_AVAILABLE = True
except ImportError:
    PYMPLER_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from abc import ABC


class BaseOperation(ABC):
    """BaseOperation is an ABC_ object intended to be the parent of all other operations.
    It provides the basic functionality for all operations, including measurements."""

    def __init__(self):
        self._logger = logging.getLogger(f'{self.__class__.__module__}.{self.__class__.__name__}')
        self._tracemalloc_started = False

        # Start tracemalloc automatically if logger is enabled for INFO level
        if self._logger.isEnabledFor(logging.INFO):
            tracemalloc.start()
            self._tracemalloc_started = True
            self._logger.debug("Tracemalloc started for memory logging")

    def _log_memory_usage(self, step: str, include_process: bool = False, include_tracemalloc: bool = False) -> None:
        """Log memory usage if logger is in INFO mode."""
        if self._logger.isEnabledFor(logging.INFO):
            log_msg_parts = [f"Memory usage after {step}:"]

            # Object memory using pympler
            if PYMPLER_AVAILABLE:
                try:
                    all_objects = muppy.get_objects()
                    mem_summary = summary.summarize(all_objects)
                    object_memory = sum(mem[2] for mem in mem_summary)  # mem[2] is total size
                    log_msg_parts.append(f"{object_memory/1024/1024:.2f} MB (objects)")
                except Exception as e:
                    self._logger.debug(f"Failed to get object memory: {e}")
            else:
                log_msg_parts.append("pympler not available")

            # Process memory using psutil
            if include_process and PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    process_memory = process.memory_info().rss
                    log_msg_parts.append(f"{process_memory/1024/1024:.2f} MB (process)")
                except Exception as e:
                    self._logger.debug(f"Failed to get process memory: {e}")

            # Tracemalloc snapshot
            if include_tracemalloc:
                try:
                    current, peak = tracemalloc.get_traced_memory()
                    log_msg_parts.append(f"{current/1024/1024:.2f} MB current, {peak/1024/1024:.2f} MB peak (tracemalloc)")
                except Exception as e:
                    self._logger.debug(f"Failed to get tracemalloc memory: {e}")

            log_msg = ", ".join(log_msg_parts)
            self._logger.info(log_msg)

    def __del__(self):
        """Automatically stop tracemalloc when the object is deleted."""
        if hasattr(self, '_tracemalloc_started') and self._tracemalloc_started:
            try:
                tracemalloc.stop()
                # Only log if we can determine logging is still available
                if hasattr(self, '_logger') and hasattr(self._logger, 'isEnabledFor'):
                    self._logger.debug("Tracemalloc stopped automatically")
            except Exception:
                # Ignore errors during cleanup
                pass

    def _get_matched_operation_args(self) -> dict:
        """Returns a dictionary of matched attributes with the arguments for the _operate method. This aids in parallel execution

        Returns:
            dict: A dictionary of matched attributes with the arguments for the _operate method or blank dict if
            _operate is a staticmethod. This is used for parallel execution of operations.
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
