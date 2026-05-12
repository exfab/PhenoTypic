"""Operation discovery and metadata registry for PhenoTypic GUI.

This module provides OperationRegistry for discovering available operations
and extracting parameter metadata including type hints for nested operations/pipelines.

No Panel/GUI dependencies - uses only stdlib and existing phenotypic dependencies.
"""

from __future__ import annotations

import inspect
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from phenotypic import ImagePipeline
from phenotypic.abc_ import ImageOperation
from phenotypic.tools_._column_ref import _ColumnRefMarker
from phenotypic.tools_.mixin import PointPickerMixin


def _is_union_origin(origin: Any) -> bool:
    """Return ``True`` for both ``typing.Union`` and PEP 604 unions."""

    return origin is Union or origin is types.UnionType


@dataclass
class ColumnRefSpec:
    """GUI-side resolution of a :class:`_ColumnRefMarker` annotation.

    Attributes:
        source: ``"measurements"`` or ``"master_measurements"`` —
            tells the GUI which file's schema to draw column names from.
        multi: ``True`` when the carrier type is a list (renders as
            multi-select), ``False`` when scalar (single dropdown).
        with_alt: ``True`` when the column-ref appears inside a Union
            with at least one non-``None``-only alternate branch
            (e.g. ``ColumnRef | None``). Drives the two-button dtype
            toggle in ``_param_forms``.
    """

    source: str
    multi: bool
    with_alt: bool = False


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
        is_list: True if the (possibly-Optional-wrapped) annotation is a
            list type — i.e. ``list[T]`` / ``List[T]`` /
            ``Optional[List[T]]`` / bare ``list``. Drives the multi-port
            ``+``/``×`` controls on aux input ports in the GUI builder
            so list-typed slots can hold an ordered set of values while
            scalar slots still render a single connection.
        description: Parameter description from docstring (None if not available)
        column_ref: Populated when the annotation carries a
            :class:`~phenotypic.tools_._column_ref._ColumnRefMarker`.
            Drives the column-aware dropdown widgets in the analysis
            sub-app's section forms; ``None`` for ordinary params.
    """

    name: str
    type_hint: Any
    default: Any
    has_default: bool
    is_operation: bool
    is_pipeline: bool
    is_optional: bool
    is_list: bool = False
    description: Optional[str] = None
    column_ref: Optional[ColumnRefSpec] = None


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

    cls: Type[Any]  # ``ImageOperation`` for ops/measure/post; ``SetAnalyzer`` /
    # ``ModelFitter`` for analysis-category records.
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
        for ImageOperation subclasses and extracts their metadata. Also
        scans :mod:`phenotypic.analysis` for ``SetAnalyzer`` / ``ModelFitter``
        subclasses so the analysis sub-app's section forms can read param
        metadata from the same registry the builder uses.
        """
        # Import operation modules
        import phenotypic.enhance as enhance_module
        import phenotypic.detect as detect_module
        import phenotypic.refine as refine_module
        import phenotypic.correction as correction_module
        import phenotypic.measure as measure_module
        import phenotypic.grid as grid_module
        import phenotypic.post as post_module
        import phenotypic.analysis as analysis_module

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

        # Analysis classes use the SetAnalyzer / ModelFitter hierarchy and
        # are not ``ImageOperation`` subclasses; walk them separately so the
        # analysis sub-app's section dropdowns can populate from the same
        # registry the builder uses.
        self._discover_analyzers(analysis_module)

    def _discover_analyzers(self, module: Any) -> None:
        """Walk an analysis module and register filters + models.

        ``ModelFitter`` extends ``SetAnalyzer``; subclasses of the former
        become category ``"Model"`` and subclasses of the latter (excluding
        ``ModelFitter`` lineage) become category ``"Filter"``. Analyzers do
        NOT extend ``ImageOperation`` so they bypass
        :meth:`_discover_from_module`'s ``base_class`` constraint.
        """
        from phenotypic.analysis.abc_ import ModelFitter, SetAnalyzer

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name.startswith("_"):
                continue
            if not issubclass(obj, SetAnalyzer) or obj in (SetAnalyzer, ModelFitter):
                continue
            category = "Model" if issubclass(obj, ModelFitter) else "Filter"
            try:
                op_info = OperationInfo(
                    cls=obj,
                    name=name,
                    category=category,
                    module=obj.__module__,
                    docstring=obj.__doc__,
                    parameters=self._extract_parameters(obj),
                    is_point_pickable=False,
                    point_picker_param=None,
                )
                self._operations[name] = op_info
                self._categories.setdefault(category, []).append(op_info)
            except Exception as e:  # noqa: BLE001
                print(f"Warning: Could not register analyzer {name}: {e}")

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

        # Try to get resolved type hints, fall back to signature annotations.
        # ``include_extras=True`` preserves ``Annotated[T, ...]`` metadata so
        # the column-ref marker scan below can pick it up.
        try:
            hints = typing.get_type_hints(cls.__init__, include_extras=True)
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
            is_operation, is_pipeline, is_optional, is_list = (
                self._detect_operation_types(hint)
            )

            column_ref = _detect_column_ref(hint)

            params[name] = ParamInfo(
                name=name,
                type_hint=hint,
                default=default,
                has_default=has_default,
                is_operation=is_operation,
                is_pipeline=is_pipeline,
                is_optional=is_optional,
                is_list=is_list,
                description=param_descriptions.get(name),
                column_ref=column_ref,
            )

        return params

    def _detect_operation_types(
        self, hint: Any
    ) -> tuple[bool, bool, bool, bool]:
        """Detect if type hint is operation/pipeline/list, handling forward refs.

        Unwraps ``Optional[T]`` and ``list[T]`` / ``List[T]`` so that
        list-typed parameters (including ``Optional[List[T]]``) carry an
        ``is_list=True`` flag and the inner ``T`` is still inspected for
        operation/pipeline membership. A bare ``list`` annotation with no
        type argument is reported as ``is_list=True`` with op/pipeline
        flags both ``False``. Unresolved string annotations (e.g. when a
        class lives behind ``TYPE_CHECKING``) are also matched: a
        substring scan recognises ``List[`` / ``list[`` carriers and the
        ``Optional[...]`` / ``... | None`` wrappers around them.

        Args:
            hint: Type hint to analyze

        Returns:
            Tuple of (is_operation, is_pipeline, is_optional, is_list)
        """
        is_operation = False
        is_pipeline = False
        is_optional = False
        is_list = False

        # Step 1: peel an Optional/Union wrapper. If the union contains
        # exactly one non-None branch we recurse into it so callers see the
        # inner annotation's flags (e.g. Optional[List[T]] -> List[T]).
        origin = get_origin(hint)
        inner: Any = hint
        if _is_union_origin(origin):
            args = get_args(hint)
            is_optional = type(None) in args
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                inner = non_none[0]
                origin = get_origin(inner)
            else:
                # Multi-branch union (e.g. Union[Op, Pipeline, None]) — fall
                # through to the union-scanning logic below; ``inner``
                # stays as the original union for that branch.
                inner = hint

        # Step 2: if the (possibly-unwrapped) hint is a list, mark it and
        # recurse into the element type for op/pipeline detection.
        if origin is list or inner is list:
            is_list = True
            element_args = get_args(inner) if inner is not list else ()
            if element_args:
                inner_op, inner_pipe, _, _ = self._detect_operation_types(
                    element_args[0]
                )
                is_operation = inner_op
                is_pipeline = inner_pipe
            return is_operation, is_pipeline, is_optional, is_list

        # Step 2b: ``get_type_hints`` falls back to raw string annotations
        # when forward refs cannot be resolved (e.g. ``ImagePipeline`` lives
        # behind ``TYPE_CHECKING``). Walk the string to recover the same
        # is_list / is_optional flags. Operation/pipeline keyword detection
        # still happens in step 3 against the same string.
        if isinstance(inner, str):
            string_optional, string_is_list = _scan_string_list_optional(inner)
            if string_optional:
                is_optional = True
            if string_is_list:
                is_list = True

        # Step 3: original union-scan path — when the wrapper still resolves
        # to a Union (multi-branch unions like ``Union[Op, Pipeline, None]``
        # never get peeled in step 1).
        if _is_union_origin(get_origin(inner)):
            args = get_args(inner)
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
        elif isinstance(inner, type):
            try:
                if issubclass(inner, ImageOperation):
                    is_operation = True
                elif issubclass(inner, ImagePipeline):
                    is_pipeline = True
            except TypeError:
                pass  # Not a class
        # Handle string annotations (from __annotations__ without resolution)
        elif isinstance(inner, str):
            if "ImagePipeline" in inner or "Pipeline" in inner:
                is_pipeline = True
            if any(
                kw in inner
                for kw in ("Enhancer", "Detector", "Operation", "Refiner")
            ):
                is_operation = True

        return is_operation, is_pipeline, is_optional, is_list

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


def _scan_string_list_optional(annotation: str) -> tuple[bool, bool]:
    """Recover ``(is_optional, is_list)`` from an unresolved string annotation.

    Used when ``typing.get_type_hints`` falls back to raw strings because a
    forward-referenced class (e.g. one guarded by ``TYPE_CHECKING``) cannot
    be resolved at runtime. Returns a flag tuple via simple substring tests
    rather than parsing — sufficient for the project's annotations and far
    cheaper than dragging in ``ast`` for what is purely a UI hint.

    Recognises the common shapes:

    * ``"Optional[T]"`` and ``"T | None"`` -> ``is_optional=True``
    * ``"List[T]"`` / ``"list[T]"`` carriers (anywhere in the string) ->
      ``is_list=True``
    """
    is_optional = False
    is_list = False
    stripped = annotation.strip()

    if stripped.startswith("Optional[") or stripped.endswith(
        "| None"
    ) or stripped.endswith("|None"):
        is_optional = True
    if " | None" in stripped or "|None" in stripped:
        is_optional = True

    if (
        "List[" in stripped
        or "list[" in stripped
        or stripped == "list"
        or stripped == "List"
    ):
        is_list = True

    return is_optional, is_list


def _detect_column_ref(hint: Any) -> Optional[ColumnRefSpec]:
    """Extract a :class:`ColumnRefSpec` from a (possibly nested) annotation.

    Walks ``Annotated[T, ...]`` directly and recurses into ``Union`` /
    ``T | None`` branches so ``ColumnRef | None`` is recognised. The
    ``with_alt`` flag is set whenever the marker is found inside a Union
    (i.e. there is at least one alternate branch) — that drives the
    two-button dtype toggle in the GUI param-form.

    First-wins for multi-marker unions: if a Union carries the marker on
    more than one branch (e.g. a hypothetical
    ``ColumnRef | ColumnRefList``), the spec returned reflects the first
    matching branch. Update this rule when (and only when) the GUI
    grows a renderer for unions of column refs.
    """
    spec = _column_ref_from_annotated(hint)
    if spec is not None:
        return spec

    if not _is_union_origin(get_origin(hint)):
        return None

    for branch in get_args(hint):
        if branch is type(None):
            continue
        inner = _column_ref_from_annotated(branch)
        if inner is not None:
            return ColumnRefSpec(
                source=inner.source, multi=inner.multi, with_alt=True
            )
    return None


def _column_ref_from_annotated(hint: Any) -> Optional[ColumnRefSpec]:
    """Return a :class:`ColumnRefSpec` if ``hint`` is a marker-bearing Annotated."""
    marker = next(
        (
            m
            for m in getattr(hint, "__metadata__", ())
            if isinstance(m, _ColumnRefMarker)
        ),
        None,
    )
    if marker is None:
        return None
    carrier = getattr(hint, "__origin__", None)
    multi = get_origin(carrier) is list or carrier is list
    return ColumnRefSpec(source=marker.source, multi=multi)


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
