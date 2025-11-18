from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Dict, List, Union, Optional

import pandas as pd

from phenotypic.abc_ import ImageOperation, MeasureFeatures
from ._image_pipeline_core import ImagePipelineCore


class SerializablePipeline(ImagePipelineCore):
    """
    An extension of ImagePipelineCore that adds JSON serialization capabilities.
    
    This class allows pipelines to be saved to and loaded from JSON files, enabling
    pipeline configurations to be stored, shared, and reused across sessions.
    
    The serialization captures:
    - Operation instances with their parameters
    - Measurement instances with their parameters
    - Benchmark and verbose flags
    
    Internal state (attributes starting with '_') and pandas DataFrames are
    automatically excluded from serialization.
    """

    def to_json(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """
        Serialize the pipeline configuration to JSON format.
        
        This method captures the pipeline's operations, measurements, and configuration
        flags. It excludes internal state (attributes starting with '_') and pandas
        DataFrames to keep the serialization clean and focused on reproducible configuration.
        
        Args:
            filepath: Optional path to save the JSON. If None, returns JSON string.
                Can be a string or Path object.
        
        Returns:
            str: JSON string representation of the pipeline configuration.
        
        Example:
            >>> from phenotypic import ImagePipeline
            >>> from phenotypic.detect import OtsuDetector
            >>> from phenotypic.measure import MeasureShape
            >>> 
            >>> pipe = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape()])
            >>> json_str = pipe.to_json()
            >>> pipe.to_json('my_pipeline.json')  # Save to file
        """
        config = {
            'ops': self._serialize_operations(self._ops),
            'meas': self._serialize_operations(self._meas),
            'benchmark': self._benchmark,
            'verbose': self._verbose
        }
        
        json_str = json.dumps(config, indent=2)
        
        if filepath is not None:
            filepath = Path(filepath)
            filepath.write_text(json_str)
        
        return json_str

    @classmethod
    def from_json(cls, json_data: Union[str, Path]) -> SerializablePipeline:
        """
        Deserialize a pipeline from JSON format.
        
        This method reconstructs a pipeline from a JSON string or file, restoring
        all operations, measurements, and configuration flags. Classes are imported
        from the phenotypic namespace and instantiated with their saved parameters.
        
        Args:
            json_data: Either a JSON string or a path to a JSON file.
        
        Returns:
            SerializablePipeline: A new pipeline instance with the loaded configuration.
        
        Raises:
            ValueError: If the JSON is invalid or cannot be parsed.
            ImportError: If a required operation or measurement class cannot be imported.
            AttributeError: If a class cannot be found in the phenotypic namespace.
        
        Example:
            >>> from phenotypic import ImagePipeline
            >>> 
            >>> # Load from file
            >>> pipe = ImagePipeline.from_json('my_pipeline.json')
            >>> 
            >>> # Load from string
            >>> json_str = '{"ops": {...}, "meas": {...}}'
            >>> pipe = ImagePipeline.from_json(json_str)
        """
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
        ops = cls._deserialize_operations(config.get('ops', {}))
        meas = cls._deserialize_operations(config.get('meas', {}))
        benchmark = config.get('benchmark', False)
        verbose = config.get('verbose', False)
        
        # Create and return new pipeline instance
        return cls(ops=ops, meas=meas, benchmark=benchmark, verbose=verbose)

    @staticmethod
    def _serialize_operations(operations: Dict[str, Union[ImageOperation, MeasureFeatures]]) -> Dict:
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
                if key.startswith('_'):
                    # Allow name-mangled private attributes (e.g., _ClassName__attr)
                    class_name = op.__class__.__name__
                    if not key.startswith(f'_{class_name}__'):
                        continue
                
                # Skip pandas DataFrames
                if isinstance(value, pd.DataFrame):
                    continue
                
                # Check if value is JSON serializable
                try:
                    json.dumps(value)
                    params[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable objects
                    continue
            
            serialized[name] = {
                'class': class_name,
                'params': params
            }
        
        return serialized

    @staticmethod
    def _deserialize_operations(serialized: Dict) -> Dict[str, Union[ImageOperation, MeasureFeatures]]:
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
        import phenotypic
        
        operations = {}
        
        for name, op_data in serialized.items():
            class_name = op_data['class']
            params = op_data['params']
            
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
            
            # Set the parameters from saved state
            for key, value in params.items():
                setattr(instance, key, value)
            
            operations[name] = instance
        
        return operations

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
            'phenotypic.detect',
            'phenotypic.measure',
            'phenotypic.enhance',
            'phenotypic.refine',
            'phenotypic.grid',
            'phenotypic.correction',
            'phenotypic.analysis',
        ]
        
        for module_name in submodules:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    return getattr(module, class_name)
            except ImportError:
                continue
        
        return None

