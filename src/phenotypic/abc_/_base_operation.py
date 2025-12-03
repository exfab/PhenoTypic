from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable
import functools, types

if TYPE_CHECKING:
    from phenotypic import Image

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
    """Root abstract base class for all operations in PhenoTypic.

    BaseOperation is the foundation of PhenoTypic's operation system. It provides
    automatic memory tracking, logging integration, and utilities for parallel
    execution. All operations in PhenoTypic inherit from BaseOperation (either
    directly or through intermediate ABCs like ImageOperation and MeasureFeatures).

    This class is a blueprint for extending the framework: when you create a new
    operation, BaseOperation automatically handles memory profiling and logging so
    you can focus on the algorithm implementation.

    What it provides automatically:

    - **Memory Tracking:** BaseOperation automatically initiates tracemalloc when
      the logger is enabled for INFO level or higher. This enables per-operation
      memory usage monitoring without explicit instrumentation. Three levels of
      memory tracking are available:

      1. Object memory (via pympler if available): Detailed breakdown of memory
         used by Python objects in your operation.
      2. Process memory (via psutil if available): System-level memory usage
         (RSS - resident set size).
      3. Tracemalloc snapshots: Python's built-in memory tracking showing current
         and peak allocations.

    - **Logging Integration:** A logger is created automatically for each operation
      class with the name format: `module.ClassName`. Subclasses can log messages
      and memory usage without additional setup.

    - **Parallel Execution Support:** The `_get_matched_operation_args()` method
      enables serialization of operation state for parallel execution by extracting
      operation attributes that match the `_operate()` method's parameters.

    Inheritance hierarchy:

        BaseOperation (this class)
        ├── ImageOperation
        │   ├── ImageEnhancer (preprocessing filters, noise reduction)
        │   ├── ImageCorrector (rotation, alignment, quality fixes)
        │   └── ObjectDetector (colony detection algorithms)
        │
        ├── MeasureFeatures (feature extraction from detected objects)
        │
        └── GridOperation (grid detection and refinement)

    How to subclass BaseOperation:

    When extending BaseOperation, you typically implement one of its subclasses
    (ImageOperation, MeasureFeatures, etc.) which provides the specific interface
    for your operation type. All the memory tracking and logging happens
    automatically in the parent class.

    Example: Creating a custom operation (without image details):

        from phenotypic.abc_ import BaseOperation
        import logging

        class MyCustomOperation(BaseOperation):
            def __init__(self, param1, param2=5):
                # Always call parent __init__ first
                super().__init__()

                # Store your parameters as attributes
                self.param1 = param1
                self.param2 = param2

            def _operate(self, data):
                # Your algorithm here
                # Logger available as self._logger
                self._logger.info(f"Processing with param1={self.param1}")

                # Log memory usage after expensive operations
                self._log_memory_usage("after processing")

                return result

    Attributes:
        _logger (logging.Logger): Logger instance created automatically with
            the format `module.ClassName`. Use `_logger.info()`, `_logger.debug()`
            to log messages during operation execution.
        _tracemalloc_started (bool): Internal flag indicating whether tracemalloc
            was started. Set to True automatically if logger is enabled for INFO
            level or higher.

    Notes:
        - Memory tracking is only enabled if the logger is configured to handle
          INFO level messages or higher. If you want to disable memory tracking,
          set the logger level to WARNING or higher.
        - Tracemalloc is automatically stopped when the operation object is
          deleted (in `__del__`), even if an exception occurs.
        - The `_get_matched_operation_args()` method is used internally by the
          pipeline system for parallel execution. It extracts operation attributes
          that match the `_operate()` method signature, enabling operations to be
          serialized and executed in worker processes.
        - On Windows, pympler may not be available, so object memory tracking
          will fall back gracefully. psutil is available on all platforms.

    Examples:
        .. dropdown:: Enabling memory tracking for an operation

            .. code-block:: python

                import logging
                from phenotypic.detect import OtsuDetector

                # Set up logging to see memory usage
                logging.basicConfig(level=logging.INFO)

                # Create detector instance
                detector = OtsuDetector()

                # Apply operation - memory usage is logged automatically
                result = detector.apply(image)

                # Console output shows:
                # INFO: Memory usage after <step>: XX.XX MB (objects), YY.YY MB (process)

        .. dropdown:: Accessing memory information programmatically

            .. code-block:: python

                import logging
                from phenotypic.enhance import GaussianBlur

                # Create custom logger to capture memory messages
                logger = logging.getLogger('phenotypic.enhance.GaussianBlur')
                logger.setLevel(logging.INFO)

                handler = logging.StreamHandler()
                handler.setLevel(logging.INFO)
                logger.addHandler(handler)

                # Use operation
                blur = GaussianBlur(sigma=2)
                enhanced = blur.apply(image)

                # Memory tracking happens automatically during operation

        .. dropdown:: Custom operation with parameter matching for parallel execution

            .. code-block:: python

                from phenotypic.abc_ import ImageOperation
                from phenotypic import Image

                class CustomThreshold(ImageOperation):
                    def __init__(self, threshold_value: int):
                        super().__init__()
                        self.threshold_value = threshold_value

                    @staticmethod
                    def _operate(image: Image, threshold_value: int = 128) -> Image:
                        # Apply threshold algorithm
                        image.enh_gray[:] = image.enh_gray[:] > threshold_value
                        return image

                # When operation is applied via pipeline:
                operation = CustomThreshold(threshold_value=100)

                # _get_matched_operation_args() automatically extracts:
                # {'threshold_value': 100}
                # This enables parallel execution in pipelines
    """

    def __init__(self):
        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._tracemalloc_started = False

        # Start tracemalloc automatically if logger is enabled for INFO level
        if self._logger.isEnabledFor(logging.INFO):
            tracemalloc.start()
            self._tracemalloc_started = True
            self._logger.debug("Tracemalloc started for memory logging")

    def _log_memory_usage(
        self,
        step: str,
        include_process: bool = False,
        include_tracemalloc: bool = False,
    ) -> None:
        """Log memory usage if logger is in INFO mode."""
        if self._logger.isEnabledFor(logging.INFO):
            log_msg_parts = [f"Memory usage after {step}:"]

            # Object memory using pympler
            if PYMPLER_AVAILABLE:
                try:
                    all_objects = muppy.get_objects()
                    mem_summary = summary.summarize(all_objects)
                    object_memory = sum(
                        mem[2] for mem in mem_summary
                    )  # mem[2] is total size
                    log_msg_parts.append(
                        f"{object_memory / 1024 / 1024:.2f} MB (objects)"
                    )
                except Exception as e:
                    self._logger.debug(f"Failed to get object memory: {e}")
            else:
                log_msg_parts.append("pympler not available")

            # Process memory using psutil
            if include_process and PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    process_memory = process.memory_info().rss
                    log_msg_parts.append(
                        f"{process_memory / 1024 / 1024:.2f} MB (process)"
                    )
                except Exception as e:
                    self._logger.debug(f"Failed to get process memory: {e}")

            # Tracemalloc snapshot
            if include_tracemalloc:
                try:
                    current, peak = tracemalloc.get_traced_memory()
                    log_msg_parts.append(
                        f"{current / 1024 / 1024:.2f} MB current, {peak / 1024 / 1024:.2f} MB peak (tracemalloc)"
                    )
                except Exception as e:
                    self._logger.debug(f"Failed to get tracemalloc memory: {e}")

            log_msg = ", ".join(log_msg_parts)
            self._logger.info(log_msg)

    def __del__(self):
        """Automatically stop tracemalloc when the object is deleted."""
        if hasattr(self, "_tracemalloc_started") and self._tracemalloc_started:
            try:
                tracemalloc.stop()
                # Only log if we can determine logging is still available
                if hasattr(self, "_logger") and hasattr(self._logger, "isEnabledFor"):
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
        raw_operate_method = inspect.getattr_static(self.__class__, "_operate")
        if isinstance(raw_operate_method, staticmethod):
            return self._matched_args(raw_operate_method.__func__)
        else:
            return {}

    def _matched_args(self, func):
        """Return a dict of attributes that satisfy *func*'s signature."""
        sig = inspect.signature(func)
        matched = {}

        for name, param in sig.parameters.items():
            if (
                name == "image"
            ):  # The image provided by the user is always passed as the first argument.
                continue
            if hasattr(self, name):
                value = getattr(self, name)
                if isinstance(
                    value, types.MethodType
                ):  # transform a bounded method into a pickleable object
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
