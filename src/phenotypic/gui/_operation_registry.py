"""Operation discovery and metadata registry for PhenoTypic GUI.

This module provides OperationRegistry for discovering available operations
and extracting parameter metadata including type hints for nested operations/pipelines.

No Panel/GUI dependencies - uses only stdlib and existing phenotypic dependencies.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from phenotypic import ImagePipeline
from phenotypic.abc_ import ImageOperation
from phenotypic.tools_.mixin import PointPickerMixin


@dataclass
class ParamInfo:
    """Metadata about an operation parameter.

    Attributes:
        name: Parameter name
        type_hint: Type annotation from function signature
        default: Default value (None if no default)
        has_default: Whether parameter has a default value
        is_operation: True if parameter accepts ImageOperation subclass
        is_pipeline: True if parameter accepts ImagePipeline
        is_optional: True if Union[..., None] (parameter can be None)
        description: Parameter description from docstring (None if not available)
    """

    name: str
    type_hint: Any
    default: Any
    has_default: bool
    is_operation: bool
    is_pipeline: bool
    is_optional: bool
    description: Optional[str] = None


@dataclass
class OperationInfo:
    """Metadata about an operation class.

    Attributes:
        cls: Operation class
        name: Class name
        category: Operation category (Enhancer, Detector, etc.)
        module: Module path
        docstring: Class docstring
        parameters: Dict mapping parameter names to ParamInfo
        is_point_pickable: True if the class inherits ``PointPickerMixin`` —
            its centres parameter can be filled by the GUI's interactive
            point picker.
        point_picker_param: Name of the parameter that holds the picked
            ``(y, x)`` coordinates (lifted from
            ``cls._point_picker_param_name``). ``None`` when
            ``is_point_pickable`` is False.
    """

    cls: Type[ImageOperation]
    name: str
    category: str
    module: str
    docstring: Optional[str] = None
    parameters: Dict[str, ParamInfo] = field(default_factory=dict)
    is_point_pickable: bool = False
    point_picker_param: Optional[str] = None


class OperationRegistry:
    """Registry of available ImageOperation classes.

    Discovers operations from phenotypic modules and provides metadata
    for dynamic parameter UI generation.

    Examples:
        >>> from phenotypic.gui import OperationRegistry
        >>>
        >>> registry = OperationRegistry()
        >>> registry.discover()
        >>>
        >>> # Get all categories
        >>> categories = registry.get_categories()
        >>> print(categories)
        ['Enhancer', 'Detector', 'Refiner', ...]
        >>>
        >>> # Get operations by category
        >>> enhancers = registry.get_by_category('Enhancer')
        >>> print([op.name for op in enhancers])
        ['GaussianBlur', 'CLAHE', 'MedianFilter', ...]
        >>>
        >>> # Get specific operation info
        >>> info = registry.get('GaussianBlur')
        >>> print(info.parameters.keys())
        dict_keys(['sigma'])
        >>>
        >>> # Create instance with defaults
        >>> blur = registry.create_instance('GaussianBlur')
        >>> print(blur.sigma)
        1.0
    """

    def __init__(self):
        """Initialize empty operation registry."""
        self._operations: Dict[str, OperationInfo] = {}
        self._categories: Dict[str, List[OperationInfo]] = {}

    def discover(self) -> None:
        """Discover all available operations from phenotypic modules.

        Scans phenotypic.enhance, phenotypic.detect, phenotypic.refine, etc.
        for ImageOperation subclasses and extracts their metadata.
        """
        # Import operation modules
        import phenotypic.enhance as enhance_module
        import phenotypic.detect as detect_module
        import phenotypic.refine as refine_module
        import phenotypic.correction as correction_module
        import phenotypic.measure as measure_module
        import phenotypic.grid as grid_module
        import phenotypic.post as post_module

        from phenotypic.abc_ import (
            ImageEnhancer,
            ObjectDetector,
            ObjectRefiner,
            ImageCorrector,
            MeasureFeatures,
            GridOperation,
            PostMeasurement,
        )

        # Map modules to categories
        module_category_map = [
            (enhance_module, "Enhancer", ImageEnhancer),
            (detect_module, "Detector", ObjectDetector),
            (refine_module, "Refiner", ObjectRefiner),
            (correction_module, "Corrector", ImageCorrector),
            (measure_module, "Measure", MeasureFeatures),
            (grid_module, "Grid", GridOperation),
            (post_module, "Post", PostMeasurement),
        ]

        for module, category, base_class in module_category_map:
            self._discover_from_module(module, category, base_class)

    def _discover_from_module(
        self,
        module: Any,
        category: str,
        base_class: Type[ImageOperation],
    ) -> None:
        """Discover operations from a specific module.

        Args:
            module: Python module to scan
            category: Category name for these operations
            base_class: Base class to filter for
        """
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if it's a subclass of base_class (but not the base itself)
            if (
                issubclass(obj, base_class)
                and obj is not base_class
                and not name.startswith("_")
            ):
                try:
                    is_point_pickable = issubclass(obj, PointPickerMixin)
                    point_picker_param = (
                        obj._point_picker_param_name if is_point_pickable else None
                    )

                    # Extract operation info
                    op_info = OperationInfo(
                        cls=obj,
                        name=name,
                        category=category,
                        module=obj.__module__,
                        docstring=obj.__doc__,
                        parameters=self._extract_parameters(obj),
                        is_point_pickable=is_point_pickable,
                        point_picker_param=point_picker_param,
                    )

                    # Register operation
                    self._operations[name] = op_info

                    # Add to category
                    if category not in self._categories:
                        self._categories[category] = []
                    self._categories[category].append(op_info)

                except Exception as e:
                    # Skip operations that fail to introspect
                    print(f"Warning: Could not register {name}: {e}")

    def _parse_docstring_params(self, cls: Type) -> Dict[str, str]:
        """Parse parameter descriptions from class docstring.

        Extracts parameter descriptions from Google-style docstrings.
        Supports Args, Attributes, and Parameters sections.

        Args:
            cls: Class to extract docstring from

        Returns:
            Dict mapping parameter names to description strings
        """
        import re

        doc = cls.__doc__
        if not doc:
            return {}

        params = {}
        lines = doc.split("\n")

        # State machine variables
        in_param_section = False
        current_param = None
        current_desc = []

        # Regex patterns for Google-style docstrings
        section_header_re = re.compile(
            r"^\s*(Args|Attributes|Parameters)\s*:\s*$", re.IGNORECASE
        )
        google_param_re = re.compile(r"^\s*(\w+)\s*(\(.*?\))?\s*:\s*(.+)$")

        for i, line in enumerate(lines):
            stripped_line = line.strip()

            # Check for section headers
            if section_header_re.match(stripped_line):
                in_param_section = True
                # Save previous param if exists
                if current_param:
                    params[current_param] = " ".join(current_desc).strip()
                    current_param = None
                    current_desc = []
                continue

            if in_param_section:
                # End of section detection
                if (
                    line
                    and not line[0].isspace()
                    and not google_param_re.match(line)
                ):
                    in_param_section = False
                    if current_param:
                        params[current_param] = " ".join(current_desc).strip()
                        current_param = None
                        current_desc = []
                    continue

                # Check for new parameter definition
                g_match = google_param_re.match(line)
                if g_match:
                    if current_param:
                        params[current_param] = " ".join(current_desc).strip()

                    current_param = g_match.group(1)
                    current_desc = [g_match.group(3)]
                    continue

                # Continuation of description
                if current_param and stripped_line:
                    current_desc.append(stripped_line)

        # Save last param
        if current_param:
            params[current_param] = " ".join(current_desc).strip()

        return params

    def _extract_parameters(self, cls: Type) -> Dict[str, ParamInfo]:
        """Extract parameter info with operation/pipeline type detection.

        Args:
            cls: Operation class to introspect

        Returns:
            Dict mapping parameter names to ParamInfo
        """
        try:
            sig = inspect.signature(cls.__init__)
        except Exception:
            # If signature extraction fails, return empty dict
            return {}

        # Try to get resolved type hints, fall back to signature annotations
        try:
            hints = typing.get_type_hints(cls.__init__)
        except Exception:
            # Fall back to raw annotations from signature (handles forward refs)
            hints = {
                name: p.annotation
                for name, p in sig.parameters.items()
                if p.annotation is not inspect.Parameter.empty
            }

        # Parse parameter descriptions from docstring
        param_descriptions = self._parse_docstring_params(cls)

        params = {}

        for name, p in sig.parameters.items():
            if name in ("self", "args", "kwargs"):
                continue

            hint = hints.get(name, Any)
            default = p.default if p.default is not inspect.Parameter.empty else None
            has_default = p.default is not inspect.Parameter.empty

            # Detect operation/pipeline types (enhanced for forward refs)
            is_operation, is_pipeline, is_optional = self._detect_operation_types(hint)

            params[name] = ParamInfo(
                name=name,
                type_hint=hint,
                default=default,
                has_default=has_default,
                is_operation=is_operation,
                is_pipeline=is_pipeline,
                is_optional=is_optional,
                description=param_descriptions.get(name),
            )

        return params

    def _detect_operation_types(
        self, hint: Any
    ) -> tuple[bool, bool, bool]:
        """Detect if type hint is operation/pipeline, handling forward refs.

        Args:
            hint: Type hint to analyze

        Returns:
            Tuple of (is_operation, is_pipeline, is_optional)
        """
        is_operation = False
        is_pipeline = False
        is_optional = False

        origin = get_origin(hint)
        if origin is Union:
            args = get_args(hint)
            is_optional = type(None) in args
            for arg in args:
                if arg is type(None):
                    continue
                # Check resolved types
                if isinstance(arg, type):
                    try:
                        if issubclass(arg, ImageOperation):
                            is_operation = True
                        if arg is ImagePipeline or issubclass(arg, ImagePipeline):
                            is_pipeline = True
                    except TypeError:
                        pass  # Not a class, skip
                # Check string forward references
                elif isinstance(arg, str):
                    if "ImagePipeline" in arg or "Pipeline" in arg:
                        is_pipeline = True
                    if any(
                        kw in arg
                        for kw in ("Enhancer", "Detector", "Operation", "Refiner")
                    ):
                        is_operation = True
                # Check ForwardRef objects
                elif hasattr(arg, "__forward_arg__"):
                    ref_name = arg.__forward_arg__
                    if "ImagePipeline" in ref_name or "Pipeline" in ref_name:
                        is_pipeline = True
                    if any(
                        kw in ref_name
                        for kw in ("Enhancer", "Detector", "Operation", "Refiner")
                    ):
                        is_operation = True
        elif isinstance(hint, type):
            try:
                if issubclass(hint, ImageOperation):
                    is_operation = True
                elif issubclass(hint, ImagePipeline):
                    is_pipeline = True
            except TypeError:
                pass  # Not a class
        # Handle string annotations (from __annotations__ without resolution)
        elif isinstance(hint, str):
            if "ImagePipeline" in hint or "Pipeline" in hint:
                is_pipeline = True
            if any(
                kw in hint
                for kw in ("Enhancer", "Detector", "Operation", "Refiner")
            ):
                is_operation = True

        return is_operation, is_pipeline, is_optional

    def get_categories(self) -> List[str]:
        """Get list of all operation categories.

        Returns:
            Sorted list of category names
        """
        return sorted(self._categories.keys())

    def get_by_category(self, category: str) -> List[OperationInfo]:
        """Get all operations in a category.

        Args:
            category: Category name

        Returns:
            List of OperationInfo for this category
        """
        return self._categories.get(category, [])

    def get(self, name: str) -> Optional[OperationInfo]:
        """Get operation info by name.

        Args:
            name: Operation class name

        Returns:
            OperationInfo if found, None otherwise
        """
        return self._operations.get(name)

    def get_all(self) -> Dict[str, OperationInfo]:
        """Get all registered operations.

        Returns:
            Dict mapping operation names to OperationInfo
        """
        return self._operations.copy()

    def create_instance(self, name: str, **kwargs) -> ImageOperation:
        """Create operation instance with given parameters.

        Args:
            name: Operation class name
            **kwargs: Parameters to pass to __init__

        Returns:
            Operation instance

        Raises:
            KeyError: If operation not found
            TypeError: If invalid parameters
        """
        info = self.get(name)
        if info is None:
            raise KeyError(f"Operation '{name}' not found in registry")

        return info.cls(**kwargs)


# Global registry instance (lazy initialization)
_REGISTRY: Optional[OperationRegistry] = None


def get_registry() -> OperationRegistry:
    """Get global OperationRegistry instance.

    Returns:
        Singleton OperationRegistry with operations discovered
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = OperationRegistry()
        _REGISTRY.discover()
    return _REGISTRY
