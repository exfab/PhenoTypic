"""Operation discovery and metadata registry.

This module provides OperationRegistry for discovering available operations
and extracting parameter metadata including type hints for nested operations/pipelines.

No GUI dependencies — uses only stdlib and existing phenotypic dependencies, so
the Dash GUI and the MCP server can share one registry. The GUI reaches it
through the :mod:`phenotypic.gui._operation_registry` back-compat shim.
"""

from __future__ import annotations

import importlib
import inspect
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from pydantic_core import PydanticUndefined

from phenotypic import ImagePipeline
from phenotypic.abc_ import ImageOperation
from phenotypic.sdk_._column_ref import _ColumnRefMarker
from phenotypic.sdk_._docstring_params import parse_param_descriptions
from phenotypic.sdk_.mixin import PointPickerMixin
from phenotypic.sdk_.typing_ import _OperationFieldMarker


#: Sentinel category routing a module to :meth:`OperationRegistry._discover_analyzers`
#: instead of the ``ImageOperation``-rooted walk. Analyzers split into four
#: categories by their own hierarchy, so no single category name fits.
_ANALYZER_CATEGORIES = object()


def _is_union_origin(origin: Any) -> bool:
    """Return ``True`` for both ``typing.Union`` and PEP 604 unions."""

    return origin is Union or origin is types.UnionType


def _has_operation_field_marker(hint: Any) -> bool:
    """Return ``True`` if ``hint`` carries an :class:`_OperationFieldMarker`.

    :data:`~phenotypic.sdk_.typing_.OperationField` erases its core
    type to ``Any``, so the usual ``issubclass(..., ImageOperation)``
    detection in :meth:`OperationRegistry._detect_operation_types` never
    fires. The marker is the distinguishing token. Because a field is
    typically declared as ``list[OperationField]`` or
    ``OperationField | None``, the ``Annotated`` carrying the marker is
    nested *inside* a ``list`` / ``Optional`` wrapper rather than sitting
    in ``FieldInfo.metadata``; this walks the annotation tree —
    ``Annotated`` extras and every ``get_args`` branch — to find it at
    any depth.

    Args:
        hint: A type annotation (possibly wrapped / nested).

    Returns:
        ``True`` if an :class:`_OperationFieldMarker` is present anywhere
        in the annotation tree.
    """
    # Direct ``Annotated[...]`` extras.
    for meta in getattr(hint, "__metadata__", ()):
        if isinstance(meta, _OperationFieldMarker):
            return True
    # Recurse into list / Optional / Union element types.
    return any(_has_operation_field_marker(arg) for arg in get_args(hint))


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
        type_hint: Type annotation from the pydantic field
            (``FieldInfo.annotation``)
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
            :class:`~phenotypic.sdk_._column_ref._ColumnRefMarker`.
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
        ['BlurGauss', 'EnhanceLocalContrast', 'MedianFilter', ...]
        >>>
        >>> # Get specific operation info
        >>> info = registry.get('BlurGauss')
        >>> print(info.parameters.keys())
        dict_keys(['sigma'])
        >>>
        >>> # Create instance with defaults
        >>> blur = registry.create_instance('BlurGauss')
        >>> print(blur.sigma)
        1.0
    """

    def __init__(self):
        """Initialize empty operation registry."""
        self._operations: Dict[str, OperationInfo] = {}
        self._categories: Dict[str, List[OperationInfo]] = {}
        self._skipped_imports: Dict[str, str] = {}

    def discover(self) -> None:
        """Discover all available operations from phenotypic modules.

        Walks :data:`~phenotypic._core._pipeline_parts._serializable_pipeline.PHENOTYPIC_CLASS_MODULES`
        — the same ordered module list the pipeline loader resolves class
        names against — so a class the loader can deserialize is also a
        class the catalog can list. Each module is paired with the category
        and base class its members register under by
        :meth:`_discovery_targets`.

        :mod:`phenotypic.analysis` is walked separately: analyzers use the
        ``SetAnalyzer`` / ``ModelFitter`` hierarchy, are not
        ``ImageOperation`` subclasses, and split into four categories by
        their own rules. :mod:`phenotypic.tune` carries the ``TuningSpec``
        config rather than operations and has no target, so it is skipped.

        Modules whose import fails are skipped and recorded in
        :attr:`skipped_imports`: ``detect.nn`` pulls optional heavy
        dependencies, and the catalog must degrade to "that family is
        unavailable here" rather than failing to build at all.
        """
        from phenotypic._core._pipeline_parts._serializable_pipeline import (
            PHENOTYPIC_CLASS_MODULES,
        )

        targets = self._discovery_targets()

        for module_name in PHENOTYPIC_CLASS_MODULES:
            target = targets.get(module_name)
            if target is None:
                continue
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:  # optional dependency family absent
                self._skipped_imports[module_name] = str(exc)
                continue
            category, base_class = target
            if category is _ANALYZER_CATEGORIES:
                self._discover_analyzers(module)
            else:
                self._discover_from_module(module, category, base_class)

    def _discovery_targets(self) -> Dict[str, tuple[Any, Any]]:
        """Map each discoverable module to its ``(category, base_class)``.

        Keyed by the dotted module names in
        :data:`~phenotypic._core._pipeline_parts._serializable_pipeline.PHENOTYPIC_CLASS_MODULES`;
        a module absent from this map is not walked. The base classes are
        imported here rather than at module scope so importing the registry
        stays cheap.

        Three families are not ``ImageOperation`` subclasses and so cannot
        be found by the ``ImageOperation``-rooted walks: prefab pipelines
        extend :class:`~phenotypic.abc_.PrefabPipeline`, tuning objectives
        extend ``Scorer``, and search strategies extend ``StrategyConfig``.
        Each gets its own base class so the catalog can reach it.

        Returns:
            Dict mapping module name to ``(category, base_class)``. The
            sentinel category :data:`_ANALYZER_CATEGORIES` routes a module
            to :meth:`_discover_analyzers` instead. The tuning entries are
            absent when ``phenotypic.tune`` cannot be imported.
        """
        from phenotypic.abc_ import (
            GridOperation,
            ImageCorrector,
            ImageEnhancer,
            MeasureFeatures,
            ObjectDetector,
            ObjectRefiner,
            PostMeasurement,
            PrefabPipeline,
        )

        targets: Dict[str, tuple[Any, Any]] = {
            "phenotypic.detect": ("Detector", ObjectDetector),
            "phenotypic.detect.nn": ("Detector", ObjectDetector),
            "phenotypic.measure": ("Measure", MeasureFeatures),
            "phenotypic.enhance": ("Enhancer", ImageEnhancer),
            "phenotypic.refine": ("Refiner", ObjectRefiner),
            "phenotypic.grid": ("Grid", GridOperation),
            "phenotypic.correction": ("Corrector", ImageCorrector),
            "phenotypic.analysis": (_ANALYZER_CATEGORIES, None),
            "phenotypic.prefab": ("Prefab", PrefabPipeline),
            "phenotypic.post": ("Post", PostMeasurement),
        }

        # The two tuning bases are the only ones outside ``abc_``, so naming
        # them here newly couples every registry consumer — the Dash GUI
        # included — to ``phenotypic.tune`` importing cleanly. Guard it: a
        # broken tune install should cost the catalog its scorers, not its
        # entire contents. The per-module guard in :meth:`discover` cannot
        # cover this, because the failure is in resolving the base class.
        try:
            from phenotypic.tune.score import Scorer
            from phenotypic.tune.strategy import StrategyConfig
        except ImportError as exc:
            self._skipped_imports["phenotypic.tune"] = str(exc)
            return targets

        targets["phenotypic.tune.score"] = ("Scorer", Scorer)
        targets["phenotypic.tune.strategy"] = ("Strategy", StrategyConfig)
        return targets

    @property
    def skipped_imports(self) -> Dict[str, str]:
        """Modules that failed to import during :meth:`discover`.

        Maps module name to the ``ImportError`` message, so a caller can
        tell an agent "this operation family is unavailable in this
        environment" instead of silently listing nothing.
        """
        return dict(self._skipped_imports)

    def _iter_public_classes(self, module: Any) -> List[tuple[str, Any]]:
        """Public classes of *module*, including lazily-exported ones.

        ``inspect.getmembers`` reads ``dir(module)``, which for a module
        that exports through a module-level ``__getattr__`` (and declares
        no ``__dir__``) lists nothing — :mod:`phenotypic.detect.nn` is
        exactly that shape, which is why its detectors were deserializable
        by the pipeline loader but absent from the catalog. So the eager
        walk is followed by an ``__all__``-driven ``getattr`` walk that
        touches each declared export.

        The guard sits at ``getattr`` time, not around the module import:
        for a lazy loader the heavy optional dependency is pulled in when
        the attribute is *touched*, so an absent ``torch`` raises here and
        nowhere else. A failing export is recorded in
        :attr:`skipped_imports` and the remaining exports still register.

        Args:
            module: An imported module to walk.

        Returns:
            ``(name, class)`` pairs, eager members first, then the
            lazily-resolved ``__all__`` entries that are classes. Typed
            ``Any`` rather than ``type`` to match what
            ``inspect.getmembers`` hands callers: the walk cannot narrow
            each class to the base its caller filters on, and the callers
            read subclass-only attributes off it.
        """
        found: List[tuple[str, Any]] = []
        seen: set[str] = set()

        for name, obj in inspect.getmembers(module, inspect.isclass):
            seen.add(name)
            found.append((name, obj))

        for name in getattr(module, "__all__", ()):
            if name in seen or name.startswith("_"):
                continue
            seen.add(name)
            try:
                obj = getattr(module, name)
            except Exception as exc:  # noqa: BLE001 — heavy optional import
                self._skipped_imports[f"{module.__name__}.{name}"] = str(exc)
                continue
            if inspect.isclass(obj):
                found.append((name, obj))

        return found

    def _discover_analyzers(self, module: Any) -> None:
        """Walk an analysis module and register filters + models.

        ``ModelFitter`` and ``EdgeCorrection`` both extend ``SetAnalyzer``;
        ``ModelFitter`` subclasses become category ``"Model"``,
        ``EdgeCorrection`` subclasses become ``"Edge Correction"``, and the
        remaining ``SetAnalyzer`` subclasses become ``"Filter"``. Analyzers do
        NOT extend ``ImageOperation`` so they bypass
        :meth:`_discover_from_module`'s ``base_class`` constraint.
        """
        from phenotypic.analysis.abc_ import EdgeCorrection, ModelFitter, QualityCheck, SetAnalyzer

        for name, obj in self._iter_public_classes(module):
            if name.startswith("_"):
                continue
            if not issubclass(obj, SetAnalyzer) or obj in (
                SetAnalyzer,
                ModelFitter,
                QualityCheck,
            ):
                continue
            if issubclass(obj, QualityCheck):
                category = "quality_check"
            elif issubclass(obj, ModelFitter):
                category = "Model"
            elif issubclass(obj, EdgeCorrection):
                category = "Edge Correction"
            else:
                category = "Filter"
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
        base_class: Type[Any],
    ) -> None:
        """Discover operations from a specific module.

        Args:
            module: Python module to scan
            category: Category name for these operations
            base_class: Base class to filter for. Usually an
                ``ImageOperation`` subclass, but prefab pipelines, tuning
                scorers, and search-strategy configs sit on their own
                hierarchies and are registered through the same walk.
        """
        for name, obj in self._iter_public_classes(module):
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

    def _extract_parameters(self, cls: Type) -> Dict[str, ParamInfo]:
        """Extract parameter info with operation/pipeline type detection.

        Operations and analyzers are pydantic v2 ``BaseModel`` subclasses,
        so the parameter contract is read from ``cls.model_fields`` — a
        ``dict[str, FieldInfo]`` — rather than from
        ``inspect.signature(cls.__init__)`` (which collapses to a generic
        ``(self, data)`` signature on a pydantic model).

        For non-pydantic classes (e.g. a plain class passed directly to
        this method) the legacy ``inspect.signature`` path is used as a
        fallback so callers outside the migrated operation tree still
        work.

        The pydantic test is ``is not None``, not truthiness: a pydantic
        model that declares **no** fields has ``model_fields == {}``, which
        is falsy. Falling through on it sends a *pydantic* class down the
        signature path, where ``inspect.signature(cls.__init__)`` reports
        pydantic's own ``(self, /, **data)`` and the ``**data`` catch-all
        is registered as a required parameter named ``data``. Fourteen
        registered classes take no parameters at all — six of the eleven
        measurers among them — and every one of them advertised that
        phantom. ``MeasureSize(data=...)`` raises ``ValidationError:
        Extra inputs are not permitted``.

        Args:
            cls: Operation class to introspect

        Returns:
            Dict mapping parameter names to ParamInfo. Empty for a
            pydantic class that declares no fields — which is the true
            contract, not a reason to look elsewhere.
        """
        model_fields = getattr(cls, "model_fields", None)
        if model_fields is not None:
            return self._extract_parameters_from_model_fields(cls, model_fields)
        return self._extract_parameters_from_signature(cls)

    def _extract_parameters_from_model_fields(
        self, cls: Type, model_fields: Dict[str, Any]
    ) -> Dict[str, ParamInfo]:
        """Extract ``ParamInfo`` from a pydantic model's ``model_fields``.

        Each ``FieldInfo`` carries:

        * ``.annotation`` — the carrier type (used for op/pipeline/list
          detection). ``Annotated`` extras are *not* on the annotation;
          pydantic peels them off into ``.metadata``.
        * ``.default`` — the default value (``PydanticUndefined`` marks a
          required field).
        * ``.description`` — populated by ``BaseOperation``'s docstring
          hook from the class ``Args:`` block.
        * ``.metadata`` — a flat list of the ``Annotated[...]`` extras,
          including the :class:`_ColumnRefMarker` column-ref markers.
        """
        params: Dict[str, ParamInfo] = {}

        # Per spec §1419–1428, ``QualityCheck`` subclasses inherit
        # ``agg_func`` from ``SetAnalyzer`` but no v1 check actually
        # aggregates values. Subclasses opt in to exposing the parameter
        # via ``_exposes_agg_func: ClassVar[bool] = True``; the default
        # ``False`` filters the param out of the GUI form. Classes
        # without the attribute (e.g. ``EdgeCorrector``, ``LogGrowthModel``)
        # are treated as opted in for backward-compat.
        exposes_agg_func = getattr(cls, "_exposes_agg_func", True)

        for name, fi in model_fields.items():
            if name == "agg_func" and not exposes_agg_func:
                continue

            hint = fi.annotation if fi.annotation is not None else Any
            has_default = fi.default is not PydanticUndefined
            default = fi.default if has_default else None

            # Detect operation/pipeline types (enhanced for forward refs)
            is_operation, is_pipeline, is_optional, is_list = (
                self._detect_operation_types(hint)
            )

            column_ref = _detect_column_ref(hint, fi.metadata)

            params[name] = ParamInfo(
                name=name,
                type_hint=hint,
                default=default,
                has_default=has_default,
                is_operation=is_operation,
                is_pipeline=is_pipeline,
                is_optional=is_optional,
                is_list=is_list,
                description=fi.description,
                column_ref=column_ref,
            )

        return params

    def _extract_parameters_from_signature(
        self, cls: Type
    ) -> Dict[str, ParamInfo]:
        """Extract ``ParamInfo`` from a non-pydantic class's ``__init__``.

        Fallback path for plain classes that are not pydantic
        ``BaseModel`` subclasses. ``Annotated[...]`` extras are recovered
        from ``typing.get_type_hints(..., include_extras=True)`` and the
        column-ref marker scan walks the annotation's ``__metadata__``.
        """
        import inspect
        import typing

        try:
            sig = inspect.signature(cls.__init__)
        except (ValueError, TypeError):
            return {}

        # ``include_extras=True`` preserves ``Annotated[T, ...]`` metadata
        # so the column-ref marker scan can pick it up.
        try:
            hints = typing.get_type_hints(cls.__init__, include_extras=True)
        except Exception:  # noqa: BLE001 — forward refs / partial annotations
            hints = {
                name: p.annotation
                for name, p in sig.parameters.items()
                if p.annotation is not inspect.Parameter.empty
            }

        param_descriptions = parse_param_descriptions(cls.__doc__)
        exposes_agg_func = getattr(cls, "_exposes_agg_func", True)

        params: Dict[str, ParamInfo] = {}
        for name, p in sig.parameters.items():
            if name in ("self", "args", "kwargs"):
                continue
            if name == "agg_func" and not exposes_agg_func:
                continue

            hint = hints.get(name, Any)
            has_default = p.default is not inspect.Parameter.empty
            default = p.default if has_default else None

            is_operation, is_pipeline, is_optional, is_list = (
                self._detect_operation_types(hint)
            )

            # An ``Annotated[...]`` hint stores its extras on
            # ``__metadata__``; pass the inner type as the annotation.
            metadata = getattr(hint, "__metadata__", ())
            annotation = getattr(hint, "__origin__", hint) if metadata else hint
            column_ref = _detect_column_ref(annotation, metadata)

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

        A field typed :data:`~phenotypic.sdk_.typing_.OperationField`
        (whose core type is erased to ``Any``) is recognised via an
        :class:`_OperationFieldMarker` scan of the annotation tree —
        ``OperationField`` accepts an operation *or* a nested pipeline,
        so both ``is_operation`` and ``is_pipeline`` are set when the
        marker is found.

        Args:
            hint: Type hint to analyze

        Returns:
            Tuple of (is_operation, is_pipeline, is_optional, is_list)
        """
        is_operation = False
        is_pipeline = False
        is_optional = False
        is_list = False

        # ``OperationField`` erases its core type to ``Any``; the marker
        # scan is the only reliable signal that the field accepts an
        # operation / nested pipeline. Detected up front so it survives
        # every wrapper-peeling branch (including the early ``list``
        # return below).
        operation_field = _has_operation_field_marker(hint)
        if operation_field:
            is_operation = True
            is_pipeline = True

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
                # OR rather than assign so a top-level ``OperationField``
                # marker (already detected above) is never clobbered.
                is_operation = is_operation or inner_op
                is_pipeline = is_pipeline or inner_pipe
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


def _detect_column_ref(
    annotation: Any, metadata: Any = ()
) -> Optional[ColumnRefSpec]:
    """Extract a :class:`ColumnRefSpec` from a parameter annotation.

    The column-ref markers ride on the ``Annotated[...]`` aliases
    (:data:`~phenotypic.sdk_.ColumnRef` /
    :data:`~phenotypic.sdk_.ColumnRefList`).

    Two introspection paths feed this:

    * **pydantic** — pydantic peels every ``Annotated`` extra off the
      field annotation into the flat ``FieldInfo.metadata`` list, so
      ``metadata`` carries the marker and ``annotation`` is the bare
      carrier (e.g. ``str`` / ``list[str]`` / ``str | None``).
    * **signature fallback** — ``metadata`` is empty; the marker is
      walked off the ``annotation`` itself (``Annotated.__metadata__``,
      recursing into ``Union`` branches).

    * ``multi`` — ``True`` when the carrier resolves to a ``list`` (i.e.
      ``ColumnRefList`` / ``list[str]``), driving a multi-select widget.
    * ``with_alt`` — ``True`` when the annotation is a ``Union`` with at
      least one non-``None`` alternate branch (e.g. ``ColumnRef | None``
      → ``str | None``), driving the two-button dtype toggle in the GUI
      param-form.

    Returns ``None`` for ordinary (non-column-ref) parameters.
    """
    marker = next(
        (m for m in metadata if isinstance(m, _ColumnRefMarker)),
        None,
    )

    # pydantic path: the marker is in the flat metadata list and the
    # annotation is already the bare carrier type.
    if marker is not None:
        with_alt = False
        carrier: Any = annotation
        if _is_union_origin(get_origin(annotation)):
            branches = get_args(annotation)
            non_none = [b for b in branches if b is not type(None)]
            # Any non-None alternate branch makes this a dtype-toggle field.
            with_alt = len(non_none) >= 1 and (
                len(non_none) > 1 or type(None) in branches
            )
            if non_none:
                carrier = non_none[0]
        multi = get_origin(carrier) is list or carrier is list
        return ColumnRefSpec(source=marker.source, multi=multi, with_alt=with_alt)

    # Signature-fallback path: walk the annotation for an ``Annotated``
    # column-ref marker, recursing into ``Union`` / ``T | None`` branches.
    spec = _column_ref_from_annotated(annotation)
    if spec is not None:
        return spec

    if not _is_union_origin(get_origin(annotation)):
        return None

    for branch in get_args(annotation):
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
