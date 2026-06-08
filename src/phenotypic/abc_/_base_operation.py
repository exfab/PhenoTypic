from __future__ import annotations

from typing import Any

import importlib.util
import json
import logging
import tracemalloc
from abc import ABC
from pathlib import Path

from pydantic import BaseModel, ConfigDict, PrivateAttr

from phenotypic.tools_._docstring_params import apply_docstring_descriptions
from phenotypic.tools_._io_constants import (
    CONFIG_SUFFIX_OPERATION,
    ensure_typed_json_suffix,
)
from phenotypic.tools_._json_io import read_json_source

# Check for optional dependencies
PYMPLER_AVAILABLE = importlib.util.find_spec("pympler") is not None
PSUTIL_AVAILABLE = importlib.util.find_spec("psutil") is not None

# Import if available
if PYMPLER_AVAILABLE:
    from pympler import muppy, summary
if PSUTIL_AVAILABLE:
    import psutil


class BaseOperation(BaseModel, ABC):
    """Root abstract base class for all operations in PhenoTypic.

    BaseOperation is the foundation of PhenoTypic's operation system. It provides
    automatic memory tracking, logging integration, and utilities for parallel
    execution. All operations in PhenoTypic inherit from BaseOperation (either
    directly or through intermediate ABCs like ImageOperation and MeasureFeatures).

    This class is a blueprint for extending the framework: when you create a new
    operation, BaseOperation automatically handles memory profiling and logging so
    you can focus on the algorithm implementation.

    BaseOperation is a pydantic v2 ``BaseModel``. Operation parameters are declared
    as **annotated class-level fields** — pydantic generates the constructor,
    validates inputs, and exposes a machine-readable contract via
    ``model_json_schema()``. There is no hand-written ``__init__``.

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

    - **Parallel Execution Support:** Operations are serialized via pydantic
      (``model_dump()``/``model_validate()``) for parallel execution. Worker
      processes reconstruct the complete operation object and execute it.

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

    Example: Creating a custom operation (without image details)::

        from phenotypic.abc_ import BaseOperation
        import logging

        class MyCustomOperation(BaseOperation):
            # Parameters are declared as annotated class-level fields.
            param1: int
            param2: int = 5

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
        - On Windows, pympler may not be available, so object memory tracking
          will fall back gracefully. psutil is available on all platforms.

    Examples:
        Enabling memory tracking for an operation:

        >>> import logging
        >>> from phenotypic.detect import OtsuDetector
        >>> # Set up logging to see memory usage
        >>> logging.basicConfig(level=logging.INFO)
        >>> # Create detector instance
        >>> detector = OtsuDetector()
        >>> # Apply operation - memory usage is logged automatically
        >>> result = detector.apply(image)
        # Console output shows:
        # INFO: Memory usage after <step>: XX.XX MB (objects), YY.YY MB (process)

        Accessing memory information programmatically:

        >>> import logging
        >>> from phenotypic.enhance import GaussianBlur
        >>> # Create custom logger to capture memory messages
        >>> logger = logging.getLogger('phenotypic.enhance.GaussianBlur')
        >>> logger.setLevel(logging.INFO)
        >>> handler = logging.StreamHandler()
        >>> handler.setLevel(logging.INFO)
        >>> logger.addHandler(handler)
        >>> # Use operation
        >>> blur = GaussianBlur(sigma=2)
        >>> enhanced = blur.apply(image)
        # Memory tracking happens automatically during operation

        Custom operation with declared fields for parallel execution:

        >>> from phenotypic.abc_ import ImageOperation
        >>> from phenotypic import Image
        >>> class CustomThreshold(ImageOperation):
        ...     threshold_value: int
        ...
        ...     def _operate(self, image: Image) -> Image:
        ...         # Apply threshold algorithm
        ...         image.detect_mat[:] = (
        ...             image.detect_mat[:] > self.threshold_value
        ...         )
        ...         return image
        >>> # When operation is applied via pipeline:
        >>> operation = CustomThreshold(threshold_value=100)
        # The operation object is serialized with all fields
        # for parallel execution in worker processes
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    _logger: logging.Logger = PrivateAttr()
    _tracemalloc_started: bool = PrivateAttr(default=False)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Populate field descriptions from the subclass docstring.

        Runs once per concrete subclass after pydantic has built its
        model. Copies parameter descriptions parsed from the Google-style
        ``Args:`` docstring block onto each field's ``description`` slot
        so they surface in ``model_json_schema()`` — the machine-readable
        contract used by downstream tooling (e.g. an MCP server).

        Args:
            **kwargs: Class-keyword arguments forwarded by pydantic.
        """
        super().__pydantic_init_subclass__(**kwargs)
        apply_docstring_descriptions(cls)

    def model_post_init(self, __context: Any) -> None:
        """Initialize logging and memory tracking after model construction.

        Replaces the legacy ``__init__`` body: creates the per-class
        logger and, when that logger is enabled for INFO level or higher,
        starts ``tracemalloc`` so per-operation memory usage can be
        logged.

        Args:
            __context: Pydantic post-init context (unused).
        """
        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self._tracemalloc_started = False

        # Start tracemalloc automatically if logger is enabled for INFO level
        if self._logger.isEnabledFor(logging.INFO):
            tracemalloc.start()
            self._tracemalloc_started = True
            self._logger.debug("Tracemalloc started for memory logging")

    def to_json(self, filepath: str | Path | None = None) -> str | None:
        """Serialize this operation to JSON.

        Captures the operation as a ``{"class", "params"}`` envelope: ``params``
        is ``model_dump(mode="json")`` (every declared field, including nested
        operations and raw arrays; ``PrivateAttr`` state such as loggers and
        timing is excluded automatically), and ``class`` records the concrete
        class name so :meth:`from_json` can rebuild the right subclass. This
        mirrors :meth:`ImagePipeline.to_json`.

        Args:
            filepath: Optional path to write the JSON to. When None, the JSON
                string is returned instead. Accepts a ``str`` or ``Path``.

        Returns:
            The JSON string when ``filepath`` is None, otherwise None.

        Example:
            >>> import tempfile
            >>> from pathlib import Path
            >>> from phenotypic.detect import OtsuDetector
            >>> from phenotypic.tools_ import CONFIG_SUFFIX_OPERATION, ensure_typed_json_suffix
            >>> with tempfile.TemporaryDirectory() as d:
            ...     p = Path(d) / "op.json"
            ...     saved = ensure_typed_json_suffix(p, CONFIG_SUFFIX_OPERATION)
            ...     OtsuDetector(ignore_zeros=True).to_json(p)
            ...     loaded = OtsuDetector.from_json(saved)
            >>> loaded.ignore_zeros
            True
        """
        envelope = {
            "class": type(self).__name__,
            "params": self.model_dump(mode="json"),
        }
        json_str = json.dumps(envelope, indent=2)
        if filepath is not None:
            ensure_typed_json_suffix(filepath, CONFIG_SUFFIX_OPERATION).write_text(
                json_str
            )
            return None
        return json_str

    @classmethod
    def from_json(cls, json_data: str | Path | dict) -> "BaseOperation":
        """Reconstruct an operation from JSON written by :meth:`to_json`.

        Accepts a JSON string, a path to a JSON file, or a pre-parsed envelope
        dict (same input handling as :meth:`ImagePipeline.from_json`).
        Polymorphic: ``ImageOperation.from_json(path)`` returns whatever concrete
        operation the file holds. When called on a narrower subclass, the
        resolved class must be a subclass of it, else a :class:`TypeError` is
        raised.

        Args:
            json_data: A JSON string, path to a JSON file, or envelope dict.

        Returns:
            The reconstructed operation instance.

        Raises:
            AttributeError: If the recorded class cannot be resolved in the
                ``phenotypic`` namespace.
            TypeError: If called on a concrete subclass and the file holds a
                class that is not a subclass of it.

        Example:
            >>> import tempfile
            >>> from pathlib import Path
            >>> from phenotypic.abc_ import ImageOperation
            >>> from phenotypic.detect import OtsuDetector
            >>> with tempfile.TemporaryDirectory() as d:
            ...     p = Path(d) / "op.json"
            ...     OtsuDetector().to_json(p)
            ...     loaded = ImageOperation.from_json(p)  # polymorphic
            >>> type(loaded).__name__
            'OtsuDetector'
        """
        # Lazy import: ``_serializable_pipeline`` imports ``phenotypic.abc_`` at
        # module top (see _serializable_pipeline.py), so a top-level import here
        # would create a cycle.
        from phenotypic._core._pipeline_parts._serializable_pipeline import (
            SerializablePipeline,
        )

        envelope = read_json_source(json_data)
        class_name = envelope["class"]
        op_class = SerializablePipeline._find_class_in_phenotypic(class_name)
        if op_class is None:
            raise AttributeError(
                f"Class '{class_name}' not found in the phenotypic namespace. "
                f"Ensure it is exported from its submodule's __init__.py."
            )
        if cls is not BaseOperation and not issubclass(op_class, cls):
            raise TypeError(
                f"{cls.__name__}.from_json() got a '{class_name}', which is not "
                f"a subclass of {cls.__name__}. Use BaseOperation.from_json() "
                f"or a compatible class."
            )
        return op_class.model_validate(envelope.get("params", {}) or {})

    def _log_memory_usage(
            self,
            step: str,
            include_process: bool = False,
            include_tracemalloc: bool = False,
    ) -> None:
        """Log memory usage if logger is in INFO mode."""
        if self._logger.isEnabledFor(logging.INFO):
            log_msg_parts = [f"Memory usage: {step}:"]

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
