from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Dict, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import ImagePipeline

import pandas as pd
import warnings

from phenotypic.abc_ import ImageOperation, MeasureFeatures
from ._image_pipeline_core import ImagePipelineCore

# Import version for serialization - must be after other phenotypic imports to avoid circular import
import phenotypic

__version__ = phenotypic.__version__


class SerializablePipeline(ImagePipelineCore):
    """
    An extension of ImagePipelineCore that adds JSON serialization capabilities.

    This class allows pipelines to be saved to and loaded from JSON files, enabling
    pipeline configurations to be stored, shared, and reused across sessions.

    The serialization captures:
    - PhenoTypic version for compatibility checking
    - Pipeline name and description
    - Operation instances with their parameters
    - Measurement instances with their parameters

    Internal state (attributes starting with '_') and pandas DataFrames are
    automatically excluded from serialization.
    """

    def to_json(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """
        Serialize the pipeline configuration to JSON format.

        This method captures the pipeline's operations and measurements.
        It excludes internal state (attributes starting with '_') and pandas
        DataFrames to keep the serialization clean and focused on reproducible configuration.

        Args:
            filepath: Optional path to save the JSON. If None, returns JSON string.
                Can be a string or Path object.

        Returns:
            str: JSON string representation of the pipeline configuration.

        Example:
            Serialize a pipeline to JSON format:

            >>> from phenotypic import ImagePipeline
            >>> from phenotypic.detect import OtsuDetector
            >>> from phenotypic.measure import MeasureShape
            >>> pipe = ImagePipeline(pipe_cfgs=[OtsuDetector()], meas=[MeasureShape()])
            >>> json_str = pipe.to_json()
            >>> pipe.to_json('my_pipeline.json')  # Save to file
        """
        json_str = str(self)

        if filepath is not None:
            Path(filepath).write_text(json_str)

        return json_str

    def __str__(self) -> str:
        """
        Return a JSON-formatted string representation of the pipeline.

        The generated JSON string provides a structured representation of the object's
        current state, including its operations and measurements. This output can be
        used for logging, debugging, or to recreate the object's configuration in
        another context.

        Returns:
            str: A JSON-formatted string that encodes the object's current configuration
            in a human-readable manner. This includes the phenotypic version, pipeline
            name, description, and the lists of operations and measurements.
        """
        config = {
            "version"  : __version__,
            "name"     : self.name,
            "desc"     : self._desc,
            "reset"    : self._reset,
            "pipe_cfgs": self._serialize_operations(self._ops),
            "meas"     : self._serialize_operations(self._meas),
        }

        return json.dumps(config, indent=2)

    def to_json_str(self) -> str:
        """
        Alias for __str__. Returns JSON string representation of the pipeline.

        Returns:
            str: A JSON-formatted string encoding the pipeline configuration.
        """
        return str(self)

    @classmethod
    def from_json(
            cls,
            json_data: Union[str, Path, dict],
            benchmark: bool = False,
            verbose: bool = False,
    ) -> ImagePipeline:
        """
        Deserialize a pipeline from JSON format.

        This method reconstructs a pipeline from a JSON string or file, restoring
        all operations and measurements. Classes are imported from the phenotypic
        namespace and instantiated with their saved parameters.

        Args:
            json_data: A JSON string, path to a JSON file, or a pre-parsed dict.
            benchmark: Whether to enable benchmarking for the pipeline. Defaults to False.
            verbose: Whether to enable verbose output. Defaults to False.

        Returns:
            SerializablePipeline: A new pipeline instance with the loaded configuration.

        Raises:
            ValueError: If the JSON is invalid or cannot be parsed.
            ImportError: If a required operation or measurement class cannot be imported.
            AttributeError: If a class cannot be found in the phenotypic namespace.

        Example:
            Deserialize a pipeline from JSON format:

            >>> from phenotypic import ImagePipeline
            >>> # Load from file
            >>> pipe = ImagePipeline.from_json('my_pipeline.json')
            >>> # Load from string
            >>> json_str = '{"pipe_cfgs": {...}, "meas": {...}}'
            >>> pipe = ImagePipeline.from_json(json_str)
            >>> # Load with benchmarking enabled
            >>> pipe = ImagePipeline.from_json('pipeline.json', benchmark=True)
        """
        # If already a parsed dict, use directly
        if isinstance(json_data, dict):
            config = json_data
        else:
            # Check if json_data is a file path
            if isinstance(json_data, (str, Path)):
                try:
                    path = Path(json_data)
                    # Only try to read as file if it looks like a path and exists
                    # This prevents trying to stat very long JSON strings
                    if len(str(json_data)) < 256 and path.exists() and path.is_file():
                        json_data = path.read_text()
                except (OSError, ValueError):
                    # If Path operations fail, treat as JSON string
                    pass

            # Parse JSON
            try:
                config = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON data: {e}")

        # Deserialize operations and measurements
        ops = cls._deserialize_operations(config.get("pipe_cfgs", {}))
        meas = cls._deserialize_operations(config.get("meas", {}))
        name = config.get("name", None)
        desc = config.get("desc", None)
        reset = config.get("reset", False)  # Default False for backwards compatibility
        saved_version = config.get("version", None)

        # Check version compatibility
        if saved_version is not None and saved_version != __version__:
            warnings.warn(
                    f"Pipeline was saved with phenotypic version {saved_version} "
                    f"but current version is {__version__}. "
                    f"This may cause compatibility issues.",
                    UserWarning,
                    stacklevel=2
            )

        # Create and return new pipeline instance
        return cls(ops=ops, meas=meas, benchmark=benchmark, verbose=verbose, name=name,
                   desc=desc, reset=reset)

    @staticmethod
    def _serialize_operations(
            operations: Dict[str, Union[ImageOperation, MeasureFeatures]],
    ) -> Dict:
        """
        Serialize a dictionary of operations or measurements.

        Args:
            operations: Dictionary mapping names to operation/measurement instances.

        Returns:
            Dict: Serialized representation with class names and parameters.
        """
        serialized = {}

        for name, op in operations.items():
            # Get class name
            class_name = op.__class__.__name__

            # Get instance parameters, excluding internal state and DataFrames
            params = {}
            for key, value in op.__dict__.items():
                # Skip internal attributes (starting with _) except name-mangled private attributes
                # Name-mangled attributes look like _ClassName__attribute
                if key.startswith("_"):
                    # Allow name-mangled private attributes (e.g., _ClassName__attr)
                    class_name = op.__class__.__name__
                    if not key.startswith(f"_{class_name}__"):
                        continue

                # Skip pandas DataFrames
                if isinstance(value, pd.DataFrame):
                    continue

                # Handle nested ImageOperation or MeasureFeatures instances
                if isinstance(value, (ImageOperation, MeasureFeatures)):
                    params[key] = {
                        "__type__": "operation",
                        "class": value.__class__.__name__,
                        "params": SerializablePipeline._serialize_single_operation(value)
                    }
                    continue

                # Handle nested ImagePipeline instances (check before operations)
                if isinstance(value, SerializablePipeline):
                    pipeline_config = {
                        "version": __version__,
                        "name": value.name,
                        "desc": value._desc,
                        "pipe_cfgs": SerializablePipeline._serialize_operations(value._ops),
                        "meas": SerializablePipeline._serialize_operations(value._meas),
                    }
                    params[key] = {
                        "__type__": "pipeline",
                        "config": pipeline_config
                    }
                    continue

                # Handle lists - check for mixed types (operations and/or pipelines)
                if isinstance(value, list) and value:
                    # Check if list contains any pipelines
                    has_pipeline = any(isinstance(item, SerializablePipeline) for item in value)
                    has_operation = any(isinstance(item, (ImageOperation, MeasureFeatures)) for item in value)

                    if has_pipeline and not has_operation:
                        # Pure pipeline list
                        params[key] = {
                            "__type__": "pipeline_list",
                            "items": [
                                {
                                    "config": {
                                        "version": __version__,
                                        "name": item.name,
                                        "desc": item._desc,
                                        "pipe_cfgs": SerializablePipeline._serialize_operations(item._ops),
                                        "meas": SerializablePipeline._serialize_operations(item._meas),
                                    }
                                }
                                for item in value
                            ]
                        }
                        continue
                    elif has_operation and not has_pipeline:
                        # Pure operation list
                        params[key] = {
                            "__type__": "operation_list",
                            "items": [
                                {
                                    "class": item.__class__.__name__,
                                    "params": SerializablePipeline._serialize_single_operation(item)
                                }
                                for item in value
                            ]
                        }
                        continue
                    elif has_operation and has_pipeline:
                        # Mixed list - serialize as operation_list with special handling for pipelines
                        items = []
                        for item in value:
                            if isinstance(item, SerializablePipeline):
                                items.append({
                                    "__type__": "pipeline",
                                    "config": {
                                        "version": __version__,
                                        "name": item.name,
                                        "desc": item._desc,
                                        "pipe_cfgs": SerializablePipeline._serialize_operations(item._ops),
                                        "meas": SerializablePipeline._serialize_operations(item._meas),
                                    }
                                })
                            else:
                                items.append({
                                    "class": item.__class__.__name__,
                                    "params": SerializablePipeline._serialize_single_operation(item)
                                })
                        params[key] = {
                            "__type__": "operation_list",
                            "items": items
                        }
                        continue

                # Check if value is JSON serializable
                try:
                    json.dumps(value)
                    params[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable objects
                    continue

            serialized[name] = {"class": class_name, "params": params}

        return serialized

    @staticmethod
    def _serialize_single_operation(op: Union[ImageOperation, MeasureFeatures]) -> Dict:
        """
        Serialize a single operation instance (helper for nested operations).

        This enables recursive serialization of operations containing other operations.

        Args:
            op: An ImageOperation or MeasureFeatures instance.

        Returns:
            Dict: Serialized parameters of the operation.
        """
        params = {}
        for key, value in op.__dict__.items():
            # Skip internal attributes (same logic as _serialize_operations)
            if key.startswith("_"):
                class_name = op.__class__.__name__
                if not key.startswith(f"_{class_name}__"):
                    continue

            # Skip pandas DataFrames
            if isinstance(value, pd.DataFrame):
                continue

            # Handle nested ImagePipeline instances (check before operations)
            if isinstance(value, SerializablePipeline):
                pipeline_config = {
                    "version": __version__,
                    "name": value.name,
                    "desc": value._desc,
                    "pipe_cfgs": SerializablePipeline._serialize_operations(value._ops),
                    "meas": SerializablePipeline._serialize_operations(value._meas),
                }
                params[key] = {
                    "__type__": "pipeline",
                    "config": pipeline_config
                }
                continue

            # Recursively handle nested operations
            if isinstance(value, (ImageOperation, MeasureFeatures)):
                params[key] = {
                    "__type__": "operation",
                    "class": value.__class__.__name__,
                    "params": SerializablePipeline._serialize_single_operation(value)
                }
                continue

            # Handle lists - check for mixed types (operations and/or pipelines)
            if isinstance(value, list) and value:
                # Check if list contains any pipelines
                has_pipeline = any(isinstance(item, SerializablePipeline) for item in value)
                has_operation = any(isinstance(item, (ImageOperation, MeasureFeatures)) for item in value)

                if has_pipeline and not has_operation:
                    # Pure pipeline list
                    params[key] = {
                        "__type__": "pipeline_list",
                        "items": [
                            {
                                "config": {
                                    "version": __version__,
                                    "name": item.name,
                                    "desc": item._desc,
                                    "pipe_cfgs": SerializablePipeline._serialize_operations(item._ops),
                                    "meas": SerializablePipeline._serialize_operations(item._meas),
                                }
                            }
                            for item in value
                        ]
                    }
                    continue
                elif has_operation and not has_pipeline:
                    # Pure operation list
                    params[key] = {
                        "__type__": "operation_list",
                        "items": [
                            {
                                "class": item.__class__.__name__,
                                "params": SerializablePipeline._serialize_single_operation(item)
                            }
                            for item in value
                        ]
                    }
                    continue
                elif has_operation and has_pipeline:
                    # Mixed list - serialize as operation_list with special handling for pipelines
                    items = []
                    for item in value:
                        if isinstance(item, SerializablePipeline):
                            items.append({
                                "__type__": "pipeline",
                                "config": {
                                    "version": __version__,
                                    "name": item.name,
                                    "desc": item._desc,
                                    "pipe_cfgs": SerializablePipeline._serialize_operations(item._ops),
                                    "meas": SerializablePipeline._serialize_operations(item._meas),
                                }
                            })
                        else:
                            items.append({
                                "class": item.__class__.__name__,
                                "params": SerializablePipeline._serialize_single_operation(item)
                            })
                    params[key] = {
                        "__type__": "operation_list",
                        "items": items
                    }
                    continue

            # Try JSON serialization for primitives
            try:
                json.dumps(value)
                params[key] = value
            except (TypeError, ValueError):
                continue

        return params

    @staticmethod
    def _deserialize_operations(
            serialized: Dict,
    ) -> Dict[str, Union[ImageOperation, MeasureFeatures]]:
        """
        Deserialize a dictionary of operations or measurements.

        Args:
            serialized: Dictionary with serialized operation/measurement data.

        Returns:
            Dict: Dictionary mapping names to reconstructed instances.

        Raises:
            ImportError: If a required class cannot be imported.
            AttributeError: If a class cannot be found in phenotypic namespace.
        """

        operations = {}

        for name, op_data in serialized.items():
            class_name = op_data["class"]
            params = op_data["params"]

            # Try to find the class in phenotypic namespace
            op_class = SerializablePipeline._find_class_in_phenotypic(class_name)

            if op_class is None:
                raise AttributeError(
                        f"Class '{class_name}' not found in phenotypic namespace. "
                        f"Make sure it's properly imported in phenotypic.__init__.py"
                )

            # Instantiate the class with empty constructor
            try:
                instance = op_class()
            except TypeError:
                # If empty constructor fails, try with default parameters
                raise TypeError(
                        f"Cannot instantiate {class_name} with empty constructor. "
                        f"The class may require mandatory parameters."
                )

            # Set the parameters from saved state, handling nested operations
            for key, value in params.items():
                deserialized_value = SerializablePipeline._deserialize_value(value)
                setattr(instance, key, deserialized_value)

            operations[name] = instance

        return operations

    @classmethod
    def _deserialize_value(cls, value):
        """
        Recursively deserialize a parameter value, handling nested operations.

        Args:
            value: The value to deserialize (may be primitive, nested operation, or operation list).

        Returns:
            Deserialized value (primitive, ImageOperation, MeasureFeatures, or list thereof).
        """
        # Handle nested operation
        if isinstance(value, dict) and value.get("__type__") == "operation":
            op_class = cls._find_class_in_phenotypic(value["class"])
            if op_class is None:
                raise AttributeError(
                    f"Class '{value['class']}' not found in phenotypic namespace. "
                    f"Make sure it's properly imported in phenotypic.__init__.py"
                )
            instance = op_class()
            for nested_key, nested_value in value["params"].items():
                setattr(instance, nested_key, cls._deserialize_value(nested_value))
            return instance

        # Handle list of operations (may contain mixed operations and pipelines)
        elif isinstance(value, dict) and value.get("__type__") == "operation_list":
            operation_list = []
            for item_data in value["items"]:
                # Check if item is a pipeline
                if item_data.get("__type__") == "pipeline":
                    from phenotypic import ImagePipeline
                    json_str = json.dumps(item_data["config"])
                    pipeline = ImagePipeline.from_json(json_str)
                    operation_list.append(pipeline)
                else:
                    # Handle as operation
                    item_class = cls._find_class_in_phenotypic(item_data["class"])
                    if item_class is None:
                        raise AttributeError(
                            f"Class '{item_data['class']}' not found in phenotypic namespace. "
                            f"Make sure it's properly imported in phenotypic.__init__.py"
                        )
                    item_instance = item_class()
                    for item_key, item_value in item_data["params"].items():
                        setattr(item_instance, item_key, cls._deserialize_value(item_value))
                    operation_list.append(item_instance)
            return operation_list

        # Handle nested ImagePipeline
        elif isinstance(value, dict) and value.get("__type__") == "pipeline":
            from phenotypic import ImagePipeline
            json_str = json.dumps(value["config"])
            pipeline = ImagePipeline.from_json(json_str)
            return pipeline

        # Handle list of ImagePipeline instances
        elif isinstance(value, dict) and value.get("__type__") == "pipeline_list":
            from phenotypic import ImagePipeline
            pipeline_list = []
            for item_data in value["items"]:
                json_str = json.dumps(item_data["config"])
                pipeline = ImagePipeline.from_json(json_str)
                pipeline_list.append(pipeline)
            return pipeline_list

        # Primitive value
        else:
            return value

    @staticmethod
    def _find_class_in_phenotypic(class_name: str):
        """
        Find a class by name in the phenotypic namespace.

        This method searches through all submodules of phenotypic to find the
        requested class. It checks the main phenotypic module as well as common
        submodules like detect, measure, enhance, refine, etc.

        Args:
            class_name: Name of the class to find.

        Returns:
            The class object if found, None otherwise.
        """
        import phenotypic

        # First try the main phenotypic namespace
        if hasattr(phenotypic, class_name):
            return getattr(phenotypic, class_name)

        # Try common submodules
        submodules = [
            "phenotypic.detect",
            "phenotypic.measure",
            "phenotypic.enhance",
            "phenotypic.refine",
            "phenotypic.grid",
            "phenotypic.correction",
            "phenotypic.analysis",
        ]

        for module_name in submodules:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    return getattr(module, class_name)
            except ImportError:
                continue

        return None
