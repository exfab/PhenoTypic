from __future__ import annotations

import importlib.util
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Union, Optional, NamedTuple

import numpy as np
from pydantic import (
    AliasChoices,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
)

from phenotypic.schema import OBJECT

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
    from phenotypic._core._image import Image
    from phenotypic._core._image_pipeline import ImagePipeline

import pandas as pd
from typing import Dict, List, Tuple
import inspect
import time
import sys

from phenotypic.abc_ import MeasureFeatures, BaseOperation, ImageOperation
from phenotypic.abc_._post_measurement import PostMeasurement
from phenotypic.analysis.abc_._model_fitter import ModelFitter
from phenotypic.analysis.abc_._set_analyzer import SetAnalyzer
# Import the entry type from the qc._recipe *submodule* (not the package
# __init__) so the edge stays ``_core -> qc._recipe -> analysis.abc_`` and
# never pulls in qc._runner / _cli / gui at module load.
from phenotypic.sdk_._qc_recipe import QcRecipeEntry
from phenotypic.sdk_.mixin import LazyWidgetMixin

logger = logging.getLogger("ImagePipeline")


def _normalize_operation_collection(
        value: Any,
        kind: str,
        expected_type: type | tuple[type, ...] | None = None,
) -> Any:
    """Normalize a ``list | dict`` of operations into a name-keyed dict.

    Shared list-to-dict normalization used by both the ``set_*`` mutators
    and the ``field_validator``s on :class:`ImagePipelineCore` so the two
    code paths always agree. A list is keyed by class name (deduplicated
    with numeric suffixes). A dict is passed through unchanged. ``None``
    yields an empty dict. Any other type raises :class:`TypeError`.

    Args:
        value: The raw ``list``/``dict``/``None`` collection to normalize.
        kind: Human-readable collection name used in the ``TypeError``
            message (e.g. ``"ops"``, ``"meas"``).
        expected_type: Optional class (or tuple of classes); when given,
            list entries are filtered to instances of this type before
            keys are derived — matching the historical ``set_meas`` /
            ``set_post`` / ``set_filters`` behaviour.

    Returns:
        The normalized name-keyed ``dict`` (or the input unchanged when it
        is already a dict / not a recognised collection — pydantic then
        reports the type error).

    Raises:
        TypeError: If ``value`` is neither a list, a dict, nor ``None``.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        if expected_type is not None:
            items = [x for x in value if isinstance(x, expected_type)]
            if len(items) != len(value):
                raise TypeError(
                    f"every {kind} list entry must be an instance of "
                    f"{expected_type}; got "
                    f"{[type(x).__name__ for x in value]}"
                )
        else:
            items = list(value)
        names = ImagePipelineCore._make_unique(
            [x.__class__.__name__ for x in items]
        )
        return {names[i]: items[i] for i in range(len(items))}
    raise TypeError(
        f"{kind} must be a list or a dictionary, got {type(value)}"
    )


def _layers_modified_by(operation: BaseOperation) -> tuple[str, ...] | None:
    """Return layer names modified by this operation, or None for read-only ops.

    Args:
        operation: An operation instance from the pipeline.

    Returns:
        Tuple of layer name strings that this operation modifies,
        or ``None`` if the operation is read-only (e.g. measurements).
    """
    from phenotypic.abc_ import (
        ImageCorrector,
        ImageEnhancer,
        MeasureFeatures,
        ObjectDetector,
        ObjectRefiner,
    )

    if isinstance(operation, MeasureFeatures):
        return None
    if isinstance(operation, ImageCorrector):
        return ("rgb", "gray", "detect_mat", "objmap")
    if isinstance(operation, ImageEnhancer):
        return ("detect_mat",)
    if isinstance(operation, (ObjectDetector, ObjectRefiner)):
        return ("objmap",)
    return ("rgb", "gray", "detect_mat", "objmap")


class IntermediateResult(NamedTuple):
    """Result of ``apply_with_intermediates``.

    Attributes:
        image: The final processed image (same type as the input).
        intermediates: Dictionary mapping operation names to image snapshots
            taken after each operation. Values are ``Image`` copies when
            results are kept in memory, or ``None`` when saved to disk.
    """

    image: Union[GridImage, Image]
    intermediates: Dict[str, Optional[Image]]


class ImagePipelineCore(BaseOperation, LazyWidgetMixin):
    """
    Represents a handler for processing and measurement queues used in Image operations
    and feature extraction tasks.

    This class manages two queues: a processing queue and a measurement queue. The processing
    queue contains Image operations that are applied sequentially to an Image. The measurement
    queue contains feature extractors that are used to analyze an Image and produce results
    as a pandas DataFrame. Both queues are optional and can be specified as dictionaries. If not
    provided, empty queues are initialized by default to enable flexibility in pipeline
    construction and usage.

    Attributes:
        name (str): A unique identifier for this pipeline. Defaults to a randomly
            generated UUID4 string if not provided during initialization.
        desc (str): Pipeline description accessed via property. Returns the class docstring
            if no custom description was set. Can be modified via the property setter.
        _ops (Dict[str, ImageOperation]): A dictionary where keys are string
            identifiers and values are `ImageOperation` objects representing operations to apply
            to an Image.
        _meas (Dict[str, MeasureFeatures]): A dictionary where keys are string
            identifiers and values are `FeatureExtractor` objects for extracting features
            from images.
    """

    # ------------------------------------------------------------------ #
    # Pydantic fields — the constructor is generated by pydantic.
    # ------------------------------------------------------------------ #
    #
    # The pipeline takes a ``list | dict`` of operations and stores a
    # normalized ``dict``. Each collection field declares a
    # ``field_validator(mode="before")`` that runs the same list-to-dict
    # normalization as the matching ``set_*`` method (both delegate to
    # :func:`_normalize_operation_collection` so they cannot drift).
    #
    # ``desc`` is the one aliased field: it is stored under ``desc_value``
    # but accepted/serialized as ``desc`` so the docstring-fallback
    # :attr:`desc` *property* (and the legacy ``_desc`` attribute the
    # serializer reads) can keep their names.

    # validate_by_name (merged onto BaseOperation's ConfigDict) lets
    # model_validate accept the field name "desc_value" as well as the
    # "desc" alias, so a model_dump() -> model_validate() round-trip
    # works without by_alias=True (relied on by the Phase 4 serializer).
    model_config = ConfigDict(validate_by_name=True)

    name: str = Field(default_factory=lambda: str(uuid.uuid4()))
    desc_value: Optional[str] = Field(default=None, alias="desc")
    benchmark: bool = False
    verbose: bool = False
    reset: bool = False
    nrows: Optional[int] = None
    ncols: Optional[int] = None

    # ``pipe_cfgs`` is the historical constructor keyword and the JSON
    # envelope key for the operations collection; ``ops`` is the canonical
    # field name. ``AliasChoices`` accepts either at construction /
    # ``model_validate`` while ``validate_by_name=True`` keeps ``ops``
    # working too.
    ops: Dict[str, Union[ImageOperation, "ImagePipelineCore"]] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("ops", "pipe_cfgs"),
    )
    meas: Dict[str, MeasureFeatures] = {}
    post: Dict[str, PostMeasurement] = {}
    filters: Dict[str, SetAnalyzer] = {}
    model: Optional[ModelFitter] = None
    # QC config: an ordered LIST of ``{instance_id, class, enabled, params}``
    # entries (not a name-keyed dict like ``filters``, and not bare
    # ``QualityCheck`` instances — the entries carry the stable
    # ``instance_id``/``enabled`` metadata the GUI per-card IDs and
    # ``review_state.json`` key off). Excluded from ``model_dump``: the
    # pipeline (de)serializer emits/reads the ``qc`` array explicitly via the
    # dedicated entry shape, so it must not also leak through the generic
    # pydantic dump.
    qc: List[QcRecipeEntry] = Field(default_factory=list, exclude=True)

    # Internal (non-config) benchmarking state — never serialized.
    _operation_times: Dict[str, float] = PrivateAttr(default_factory=dict)
    _measurement_times: Dict[str, float] = PrivateAttr(default_factory=dict)
    _operation_memory: Dict[str, float] = PrivateAttr(default_factory=dict)
    _measurement_memory: Dict[str, float] = PrivateAttr(default_factory=dict)
    _operation_rss: Dict[str, float] = PrivateAttr(default_factory=dict)
    _measurement_rss: Dict[str, float] = PrivateAttr(default_factory=dict)

    # Lazily-populated ipywidget UI handles used by ``LazyWidgetMixin``.
    # Mirrors the declaration on ``ImageOperation``: both classes mix in
    # ``LazyWidgetMixin`` (a plain, non-model class), so every pydantic
    # model that uses it must declare these private attributes itself.
    _ui: Any = PrivateAttr(default=None)
    _param_widgets: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _view_dropdown: Any = PrivateAttr(default=None)
    _update_button: Any = PrivateAttr(default=None)
    _output_widget: Any = PrivateAttr(default=None)
    _image_ref: Any = PrivateAttr(default=None)

    @field_validator("name", mode="before")
    @classmethod
    def _default_name(cls, value: Any) -> Any:
        """Generate a UUID4 name when ``None`` is passed explicitly.

        The ``default_factory`` covers ``name`` being omitted entirely,
        but callers such as the JSON loader pass ``name=None`` explicitly
        (legacy JSONs may have no ``name`` key). Reproduce the original
        ``__init__`` effect — ``name if name is not None else uuid4()`` —
        so an explicit ``None`` still yields a fresh UUID4 string.
        """
        if value is None:
            return str(uuid.uuid4())
        return value

    @field_validator("ops", mode="before")
    @classmethod
    def _normalize_ops(cls, value: Any) -> Any:
        """Coerce a ``list``/``dict``/``None`` of operations into a dict."""
        return _normalize_operation_collection(value, "ops")

    @field_validator("meas", mode="before")
    @classmethod
    def _normalize_meas(cls, value: Any) -> Any:
        """Coerce a ``list``/``dict``/``None`` of measurements into a dict."""
        return _normalize_operation_collection(
            value, "measurements", MeasureFeatures
        )

    @field_validator("post", mode="before")
    @classmethod
    def _normalize_post(cls, value: Any) -> Any:
        """Coerce a ``list``/``dict``/``None`` of post transforms into a dict."""
        return _normalize_operation_collection(
            value, "post", PostMeasurement
        )

    @field_validator("filters", mode="before")
    @classmethod
    def _normalize_filters(cls, value: Any) -> Any:
        """Coerce a ``list``/``dict``/``None`` of filters into a dict."""
        return _normalize_operation_collection(
            value, "filters", SetAnalyzer
        )

    @field_validator("qc", mode="before")
    @classmethod
    def _normalize_qc(cls, value: Any) -> Any:
        """Coerce a ``None``/``list`` of QC entries into a list.

        ``qc`` is always an ordered list of :class:`QcRecipeEntry`. ``None``
        (e.g. an explicit ``qc=None`` from a legacy loader) yields an empty
        list; a list passes through for pydantic to validate each entry.
        Any other type raises so a misuse fails loudly at construction.

        Args:
            value: The raw ``qc`` argument.

        Returns:
            A list of entries (empty for ``None``).

        Raises:
            TypeError: If ``value`` is neither ``None`` nor a list.
        """
        if value is None:
            return []
        if isinstance(value, list):
            return value
        raise TypeError(
            f"qc must be a list of QcRecipeEntry or None, got {type(value)}"
        )

    # ------------------------------------------------------------------ #
    # Legacy attribute shims.
    #
    # Much of the codebase (the serializer, CLI, GUI, sweep, mixins)
    # reads the pre-pydantic protected names ``_ops``/``_meas``/...
    # These properties keep those read **and write** sites working while
    # the canonical storage is the pydantic field.
    # ------------------------------------------------------------------ #

    @property
    def desc(self) -> str:
        """Get pipeline description. Returns class docstring if no description set."""
        if self.desc_value is not None:
            return self.desc_value
        # Return the actual class's docstring (e.g., ImagePipeline's docstring)
        # This uses self.__class__.__doc__ to get the docstring of the instantiated class
        return self.__class__.__doc__ or ""

    @desc.setter
    def desc(self, value: Optional[str]) -> None:
        """Set pipeline description."""
        self.desc_value = value

    @property
    def _desc(self) -> Optional[str]:
        """Legacy alias for the raw description value (``None`` when unset)."""
        return self.desc_value

    @_desc.setter
    def _desc(self, value: Optional[str]) -> None:
        self.desc_value = value

    @property
    def _ops(self) -> Dict[str, Union[ImageOperation, "ImagePipelineCore"]]:
        """Legacy alias for the operations dict."""
        return self.ops

    @_ops.setter
    def _ops(self, value: Dict[str, Union[ImageOperation, "ImagePipelineCore"]]) -> None:
        self.ops = value

    @property
    def _meas(self) -> Dict[str, MeasureFeatures]:
        """Legacy alias for the measurements dict."""
        return self.meas

    @_meas.setter
    def _meas(self, value: Dict[str, MeasureFeatures]) -> None:
        self.meas = value

    @property
    def _post(self) -> Dict[str, PostMeasurement]:
        """Legacy alias for the post-measurement transforms dict."""
        return self.post

    @_post.setter
    def _post(self, value: Dict[str, PostMeasurement]) -> None:
        self.post = value

    @property
    def _filters(self) -> Dict[str, SetAnalyzer]:
        """Legacy alias for the analysis filter chain dict."""
        return self.filters

    @_filters.setter
    def _filters(self, value: Dict[str, SetAnalyzer]) -> None:
        self.filters = value

    @property
    def _model(self) -> Optional[ModelFitter]:
        """Legacy alias for the analysis endpoint model."""
        return self.model

    @_model.setter
    def _model(self, value: Optional[ModelFitter]) -> None:
        self.model = value

    @property
    def _qc(self) -> List[QcRecipeEntry]:
        """Legacy/protected alias for the QC config entry list."""
        return self.qc

    @_qc.setter
    def _qc(self, value: List[QcRecipeEntry]) -> None:
        self.qc = value

    @property
    def _benchmark(self) -> bool:
        """Legacy alias for the benchmark flag."""
        return self.benchmark

    @_benchmark.setter
    def _benchmark(self, value: bool) -> None:
        self.benchmark = value

    @property
    def _verbose(self) -> bool:
        """Legacy alias for the verbose flag."""
        return self.verbose

    @_verbose.setter
    def _verbose(self, value: bool) -> None:
        self.verbose = value

    @property
    def _reset(self) -> bool:
        """Legacy alias for the reset flag."""
        return self.reset

    @_reset.setter
    def _reset(self, value: bool) -> None:
        self.reset = value

    @property
    def _nrows(self) -> Optional[int]:
        """Legacy alias for the soft grid-row preset."""
        return self.nrows

    @_nrows.setter
    def _nrows(self, value: Optional[int]) -> None:
        self.nrows = value

    @property
    def _ncols(self) -> Optional[int]:
        """Legacy alias for the soft grid-column preset."""
        return self.ncols

    @_ncols.setter
    def _ncols(self, value: Optional[int]) -> None:
        self.ncols = value

    def set_ops(self, ops: List[ImageOperation | ImagePipeline] | Dict[str, ImageOperation | ImagePipeline]):
        """
        Sets the operations to be performed. The operations can be passed as either a list of
        ImageOperation or ImagePipeline instances or a dictionary mapping operation names to
        ImageOperation or ImagePipeline instances. This method ensures that each operation in
        the list has a unique name. Raises a TypeError if the input is neither a list nor a dictionary.

        Args:
            ops (List[ImageOperation | ImagePipeline] | Dict[str, ImageOperation | ImagePipeline]):
                A list of ImageOperation or ImagePipeline objects, or a dictionary where keys are
                operation names and values are ImageOperation or ImagePipeline objects.

        Raises:
            TypeError: If the input is not a list or a dictionary.
        """
        # Assigning the field re-runs the ``_normalize_ops`` validator
        # (validate_assignment=True), so list/dict normalization here is
        # identical to construction-time normalization.
        self.ops = _normalize_operation_collection(ops, "ops")

    def set_meas(
            self, measurements: List[MeasureFeatures] | Dict[str, MeasureFeatures]
    ):
        """
        Sets the measurements to be used for further computation. The input can be either
        a list of `MeasureFeatures` objects or a dictionary with string keys and `MeasureFeatures`
        objects as values.

        The method processes the given input to construct a dictionary mapping measurement names
        to `MeasureFeatures` instances. If a list is passed, unique class names of the
        `MeasureFeatures` instances in the list are used as keys.

        Args:
            measurements (List[MeasureFeatures] | Dict[str, MeasureFeatures]): A collection
                of measurement features either as a list of `MeasureFeatures` objects, where
                class names are used as keys for dictionary creation, or as a dictionary where
                keys are predefined strings and values are `MeasureFeatures` objects.

        Raises:
            TypeError: If the `measurements` argument is neither a list nor a dictionary.
        """
        self.meas = _normalize_operation_collection(
            measurements, "measurements", MeasureFeatures
        )

    def set_post(
            self, post: List[PostMeasurement] | Dict[str, PostMeasurement]
    ):
        """Set the post-measurement transforms.

        Args:
            post: A list or dictionary of PostMeasurement objects.
                If a list, class names are used as keys.

        Raises:
            TypeError: If post is neither a list nor a dictionary.
        """
        self.post = _normalize_operation_collection(
            post, "post", PostMeasurement
        )

    def get_ops(self) -> Dict[str, Union[ImageOperation, "ImagePipelineCore"]]:
        """Get a copy of the operations dictionary.

        Returns a shallow copy to prevent accidental mutation of internal state.

        Returns:
            Dict[str, ImageOperation | ImagePipelineCore]: Dictionary mapping
                operation names to ``ImageOperation`` instances (or nested
                pipelines).
        """
        return dict(self._ops)

    def get_meas(self) -> Dict[str, MeasureFeatures]:
        """Get a copy of the measurements dictionary.

        Returns a shallow copy to prevent accidental mutation of internal state.

        Returns:
            Dict[str, MeasureFeatures]: Dictionary mapping measurement names to
                MeasureFeatures instances.
        """
        return dict(self._meas)

    def get_post(self) -> Dict[str, PostMeasurement]:
        """Get a copy of the post-measurement transforms dictionary.

        Returns a shallow copy to prevent accidental mutation of internal state.

        Returns:
            Dict[str, PostMeasurement]: Dictionary mapping post-measurement
                names to ``PostMeasurement`` instances.
        """
        return dict(self._post)

    def set_filters(
            self, filters: List["SetAnalyzer"] | Dict[str, "SetAnalyzer"]
    ) -> None:
        """Set the analysis filter chain.

        Args:
            filters: A list or dictionary of :class:`SetAnalyzer` instances. If a
                list, class names (deduplicated by suffix) are used as keys.

        Raises:
            TypeError: If ``filters`` is neither a list nor a dictionary.
        """
        self.filters = _normalize_operation_collection(
            filters, "filters", SetAnalyzer
        )

    def get_filters(self) -> Dict[str, "SetAnalyzer"]:
        """Get a copy of the analysis filter chain.

        Returns a shallow copy to prevent accidental mutation of internal state.

        Returns:
            Dict[str, SetAnalyzer]: Ordered dict mapping filter names to
                :class:`SetAnalyzer` instances. Empty when no filters configured.
        """
        return dict(self._filters)

    def set_model(self, model: Optional["ModelFitter"]) -> None:
        """Set the analysis endpoint model.

        Only one model can be configured per pipeline; assigning a new value
        replaces any prior model. Pass ``None`` to clear.

        Args:
            model: A :class:`ModelFitter` instance, or ``None`` to clear.

        Raises:
            TypeError: If ``model`` is not a :class:`ModelFitter` instance and
                not ``None``.
        """
        if model is None:
            self.model = None
            return

        if not isinstance(model, ModelFitter):
            raise TypeError(
                f"model must be a ModelFitter instance or None, got "
                f"{type(model).__name__}"
            )
        self.model = model

    def get_model(self) -> Optional["ModelFitter"]:
        """Get the analysis endpoint model, if configured.

        Returns:
            Optional[ModelFitter]: The configured :class:`ModelFitter` instance,
                or ``None`` when no model is set.
        """
        return self._model

    def set_qc(self, qc: Optional[List[QcRecipeEntry]]) -> None:
        """Set the QC config entry list.

        Mirrors :meth:`set_post` / :meth:`set_filters` but the QC section is
        an ordered **list** of :class:`QcRecipeEntry` (carrying stable
        ``instance_id``/``enabled`` metadata), not a name-keyed dict.

        Args:
            qc: A list of :class:`QcRecipeEntry`, or ``None`` to clear.

        Raises:
            TypeError: If ``qc`` is neither a list nor ``None`` (raised by
                the ``_normalize_qc`` validator on assignment).
        """
        self.qc = qc if qc is not None else []

    def get_qc(self) -> List[QcRecipeEntry]:
        """Get a copy of the QC config entry list.

        Returns a shallow copy so callers cannot mutate the pipeline's
        internal list by appending/removing. The entries themselves are
        shared (they are lightweight config dataclasses); ``run_qc`` and the
        GUI only read them and instantiate fresh checks from their params.

        Returns:
            List[QcRecipeEntry]: Ordered QC config entries. Empty when no
                checks are configured.
        """
        return list(self._qc)

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the analysis chain (filters then model) against an aggregate frame.

        Applies each filter in order — each transforms a DataFrame into another
        DataFrame — then runs the configured terminal model to produce a fit
        summary (typically one row per group).

        The expected input is the post-applied, aggregated DataFrame the CLI
        seeds into ``measurements.parquet``. ``analyze`` does **not** apply
        post itself: by default :meth:`measure` runs ``self._post`` before
        returning, and the CLI applies post explicitly to a copy of the
        aggregated master before invoking ``analyze``. Callers passing
        ``apply_post=False`` to :meth:`measure` are responsible for applying
        post (or accepting that filters/model see pre-post data) before
        calling ``analyze``.

        Args:
            df: The aggregate measurements DataFrame.

        Returns:
            pd.DataFrame: The model-fit output (one row per group, plus shared
                :class:`MODEL_METRICS` columns for fit quality).

        Raises:
            ValueError: If no model is configured. Configure via
                :meth:`set_model` (or pass ``model=`` at construction).
        """
        if self._model is None:
            raise ValueError(
                "pipeline has no analysis model configured; assign a "
                "ModelFitter via set_model() or pass model= at construction"
            )

        current = df
        for key, flt in self._filters.items():
            logger.debug("Running analysis filter: %s", key)
            current = flt.analyze(current)
        return self._model.analyze(current)

    def _analyze_steps(
            self, df: pd.DataFrame
    ) -> List[Tuple[str, pd.DataFrame]]:
        """Run the analysis chain step by step, returning per-step outputs.

        Internal helper for the analysis GUI's live preview — exposes the
        intermediate frame after each filter and the model fit so each section
        in the stepper can render its own preview without re-running upstream
        steps from scratch. Not part of the public API; use :meth:`analyze`
        for end-to-end runs.

        Args:
            df: The aggregate measurements DataFrame.

        Returns:
            List[Tuple[str, pd.DataFrame]]: One ``(label, df)`` per step. The
                label matches the key in :attr:`_filters` for filter steps,
                and the model class name for the terminal entry. When no
                model is configured, only filter entries are returned.
        """
        steps: List[Tuple[str, pd.DataFrame]] = []
        current = df
        for key, flt in self._filters.items():
            current = flt.analyze(current)
            steps.append((key, current))
        if self._model is not None:
            fit = self._model.analyze(current)
            steps.append((type(self._model).__name__, fit))
        return steps

    @staticmethod
    def _make_unique(class_names):
        """
        Ensures uniqueness of strings in the given list by appending numeric suffixes when duplicates are
        found. If duplicates exist, subsequent occurrences of the duplicate string are modified by adding a
        numeric suffix to make them unique.

        Args:
            class_names (List[str]): A list of strings where duplicates may exist.

        Returns:
            List[str]: A new list of strings where each string is guaranteed to be unique.

        Raises:
            None
        """
        seen = {}
        result = []

        for s in class_names:
            if s not in seen:
                seen[s] = 0
                result.append(s)
            else:
                seen[s] += 1
                new_s = f"{s}_{seen[s]}"
                while new_s in seen:
                    seen[s] += 1
                    new_s = f"{s}_{seen[s]}"
                seen[new_s] = 0
                result.append(new_s)

        return result

    @staticmethod
    def _get_process_rss_mb() -> float:
        """Return current process RSS in megabytes.

        Returns:
            float: Resident set size in MB, or ``nan`` if unavailable.
        """
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return float("nan")

    def _run_operations(
        self,
        img: Image,
        on_op_complete: Optional[
            Callable[
                [int, str, Image, Union[ImageOperation, "ImagePipelineCore"]],
                None,
            ]
        ] = None,
    ) -> None:
        """Execute all queued operations on *img* in order.

        Args:
            img: The image to process (modified in place).
            on_op_complete: Optional callback invoked after each successful
                operation with ``(index, op_name, img, operation)``.
        """
        # Reset operation times and memory for new apply run if benchmarking
        if self._benchmark:
            self._operation_times = {}
            self._operation_memory = {}
            self._operation_rss = {}

        # Create progress bar if verbose and benchmark are enabled
        if self._benchmark and self._verbose:
            has_tqdm = importlib.util.find_spec("tqdm") is not None
            if has_tqdm:
                from tqdm import tqdm

                total_ops = len(self._ops)
                pbar = tqdm(
                    total=total_ops, desc="Applying operations", file=sys.stdout
                )
            else:
                print("Applying operations...")
        else:
            has_tqdm = False

        for i, (key, operation) in enumerate(self._ops.items()):
            logger.debug("Applying operation [%d/%d]: %s", i + 1, len(self._ops), key)
            try:
                # Update progress bar description with current operation
                if self._benchmark and self._verbose:
                    if has_tqdm:
                        pbar.set_description(f"Operation: {key}")
                    else:
                        print(f"  Applying operation: {key}")

                # Measure execution time and memory if benchmarking is enabled
                if self._benchmark:
                    rss_before = self._get_process_rss_mb()
                    start_time = time.time()

                sig = inspect.signature(operation.apply)

                apply_params = {}
                if "inplace" in sig.parameters:
                    apply_params["inplace"] = True

                if "reset" in sig.parameters:
                    apply_params["reset"] = (
                        False  # Prevents intermediate pipelines from resetting progress
                    )

                # Propagate benchmark flag to nested pipelines
                nested_was_benchmarking = None
                if self._benchmark and isinstance(operation, ImagePipelineCore):
                    nested_was_benchmarking = operation._benchmark
                    operation._benchmark = True

                # Apply actual operation
                operation.apply(img, **apply_params)

                # Restore nested pipeline benchmark flag
                if nested_was_benchmarking is not None and isinstance(
                    operation, ImagePipelineCore
                ):
                    operation._benchmark = nested_was_benchmarking

                # Store execution time and memory if benchmarking is enabled
                if self._benchmark:
                    self._operation_times[key] = time.time() - start_time
                    rss_after = self._get_process_rss_mb()
                    self._operation_memory[key] = rss_after - rss_before
                    self._operation_rss[key] = rss_after

                    if self._verbose:
                        delta = self._operation_memory[key]
                        delta_str = f"{delta:+.1f} MB"
                        if has_tqdm:
                            pbar.set_postfix(
                                time=f"{self._operation_times[key]:.4f}s",
                                mem=delta_str,
                            )
                            pbar.update(1)
                        else:
                            print(
                                f"    Completed in {self._operation_times[key]:.4f} seconds"
                                f" ({delta_str})"
                            )

                if on_op_complete is not None:
                    on_op_complete(i, key, img, operation)

            except Exception as exc:
                if self._benchmark and self._verbose and has_tqdm:
                    pbar.close()
                op_class = type(operation).__name__
                raise type(exc)(
                    f"[{op_class}] (step {i + 1}/{len(self._ops)}, "
                    f"key='{key}'): {exc}"
                ) from exc

        # Close the progress bar if it exists
        if self._benchmark and self._verbose and has_tqdm:
            pbar.close()

    def apply(
            self, image: Image, inplace: bool = False, reset: Optional[bool] = None
    ) -> Union[GridImage, Image]:
        """
        The class provides an interface to process and apply a series of operations on
        an Image. The operations are maintained in a queue and executed sequentially
        when applied to the given Image.

        Args:
            image (Image): The arr Image to be processed. The type `Image` refers to
                an instance of the Image object to which transformations are applied.
            inplace (bool, optional): A flag indicating whether to apply the
                transformations directly on the provided Image (`True`) or create a
                copy of the Image before performing transformations (`False`). Defaults
                to `False`.
            reset (bool, optional): Whether to reset the image before applying the pipeline.
                If None (default), uses the pipeline's reset setting from __init__.
                If explicitly set to True or False, overrides the pipeline setting.
        """
        effective_reset = reset if reset is not None else self._reset
        img = image if inplace else image.copy()
        if effective_reset:
            img.reset()
        self._run_operations(img)
        return img

    def apply_with_intermediates(
        self,
        image: Image,
        inplace: bool = False,
        reset: Optional[bool] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> IntermediateResult:
        """Apply the pipeline and capture a snapshot of the image after each operation.

        Behaves identically to :meth:`apply` (respecting *inplace*, *reset*,
        benchmark timing, and verbose/tqdm progress) but additionally records
        the image state after every operation completes.

        Args:
            image: The input image to process.
            inplace: If ``True`` the image is modified in place; otherwise a
                copy is made first.  Defaults to ``False``.
            reset: Whether to reset the image before applying operations.
                ``None`` (default) uses the pipeline-level setting.
            output_dir: Optional directory path.  When provided, each
                intermediate image is persisted to an HDF5 file inside this
                directory (created automatically) and the corresponding dict
                value is set to ``None`` to conserve memory.  When ``None``,
                intermediates are kept in memory as ``Image`` copies.

        Returns:
            IntermediateResult: A named tuple containing the final image and a
            dictionary mapping operation names to intermediate snapshots (or
            ``None`` when *output_dir* is used).
        """
        effective_reset = reset if reset is not None else self._reset
        img = image if inplace else image.copy()
        if effective_reset:
            img.reset()

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        intermediates: Dict[str, Optional[Image]] = {}

        if output_dir is not None:
            # Save initial base with all layers (pre-pipeline state)
            _all_layers = ("rgb", "gray", "detect_mat", "objmap")
            img.copy().save_intermediate_layers(
                output_dir / "base_00.h5", layers=_all_layers,
            )

        def _capture(
            i: int,
            key: str,
            current: Image,
            operation: Union[ImageOperation, "ImagePipelineCore"],
        ) -> None:
            if output_dir is not None:
                layers = _layers_modified_by(operation)

                if layers is None:
                    # Read-only op (MeasureFeatures, GridFinder): no file
                    intermediates[key] = None
                elif len(layers) == 4:
                    # Corrector: emit a new base with all layers
                    current.copy().save_intermediate_layers(
                        output_dir / f"base_{i:02d}.h5", layers=layers,
                    )
                    intermediates[key] = None
                else:
                    # Delta: save only modified layers
                    current.copy().save_intermediate_layers(
                        output_dir / f"{i:02d}_{key}.h5", layers=layers,
                    )
                    intermediates[key] = None
            else:
                intermediates[key] = current.copy()

        self._run_operations(img, on_op_complete=_capture)
        return IntermediateResult(image=img, intermediates=intermediates)

    def measure(
            self,
            image: Image,
            include_metadata: bool = True,
            apply_post: bool = True,
    ) -> pd.DataFrame:
        """
        Measures properties of a given image and optionally includes metadata. The method performs
        measurements using a set of predefined measurement operations. If benchmarking is enabled,
        the execution time of each measurement is recorded. When verbose mode is active, detailed
        logging of the measurement process is displayed. A progress bar is used to track progress
        if the tqdm library is available.

        Args:
            image (Image): The image object for which measurements are performed. It must support
                the `info` method and optionally a `grid` or `objects` attribute.
            include_metadata (bool, optional): Indicates whether metadata should be included in
                the measurements. Defaults to True.
            apply_post (bool, optional): Whether to apply the configured
                :class:`PostMeasurement` operations to the merged frame before
                returning. Defaults to True. Pass ``False`` to obtain the
                pre-post merged DataFrame — useful when the caller (e.g. the
                CLI) wants to persist a clean copy and apply post separately.

        Returns:
            pd.DataFrame: A DataFrame containing the results of all performed measurements combined
                on the same index.

        Raises:
            Exception: An exception is raised if a measurement operation fails while being
                applied to the image.
        """
        # Reset measurement times and memory for new measure run if benchmarking
        if self._benchmark:
            self._measurement_times = {}
            self._measurement_memory = {}
            self._measurement_rss = {}

        meas_to_run: Dict[str, MeasureFeatures] = self._build_measurement_run_order()

        # Print message if verbose and benchmark are enabled
        if self._benchmark and self._verbose:
            print("Measuring image properties...")

        # Get image info and measure time/memory if benchmarking is enabled
        if self._benchmark:
            rss_before = self._get_process_rss_mb()
            start_time = time.time()
            measurements = [image.info(include_metadata=include_metadata)]
            self._measurement_times["image_info"] = time.time() - start_time
            rss_after = self._get_process_rss_mb()
            self._measurement_memory["image_info"] = rss_after - rss_before
            self._measurement_rss["image_info"] = rss_after

            # Print execution time if verbose and benchmark are enabled
            if self._verbose:
                delta = self._measurement_memory["image_info"]
                print(
                        f"  Image info: {self._measurement_times['image_info']:.4f} seconds"
                        f" ({delta:+.1f} MB)"
                )
        else:
            measurements = [
                image.grid.info(include_metadata=include_metadata)
                if hasattr(image, "grid")
                else image.objects.info(include_metadata=include_metadata)
            ]

        # Create progress bar if verbose and benchmark are enabled
        if self._benchmark and self._verbose:
            has_tqdm = importlib.util.find_spec("tqdm") is not None
            if has_tqdm:
                from tqdm import tqdm

                # Create a tqdm instance without items to manually update it
                total_measurements = len(meas_to_run)
                pbar = tqdm(
                        total=total_measurements,
                        desc="Applying measurements",
                        file=sys.stdout,
                )
            else:
                # If tqdm is not available, fall back to simple printing
                print("Applying measurements...")
        else:
            has_tqdm = False

        # perform measurements
        for i, (key, measurement) in enumerate(meas_to_run.items()):
            logger.debug("Running measurement [%d/%d]: %s", i + 1, len(meas_to_run), key)
            try:
                # Update progress bar description with current measurement
                if self._benchmark and self._verbose:
                    if has_tqdm:
                        pbar.set_description(f"Measurement: {key}")
                    else:
                        print(f"  Applying measurement: {key}")

                # Measure execution time and memory if benchmarking is enabled
                if self._benchmark:
                    rss_before = self._get_process_rss_mb()
                    start_time = time.time()

                    # Measurement is taken here
                    measurements.append(measurement.measure(image))
                    self._measurement_times[key] = time.time() - start_time
                    rss_after = self._get_process_rss_mb()
                    self._measurement_memory[key] = rss_after - rss_before
                    self._measurement_rss[key] = rss_after

                    # Print execution time if verbose and benchmark are enabled
                    if self._verbose:
                        delta = self._measurement_memory[key]
                        delta_str = f"{delta:+.1f} MB"
                        if has_tqdm:
                            pbar.set_postfix(
                                    time=f"{self._measurement_times[key]:.4f}s",
                                    mem=delta_str,
                            )
                            pbar.update(1)
                        else:
                            print(
                                    f"    Completed in {self._measurement_times[key]:.4f} seconds"
                                    f" ({delta_str})"
                            )
                else:
                    measurements.append(measurement.measure(image))
            except Exception as exc:
                if self._benchmark and self._verbose and has_tqdm:
                    pbar.close()
                meas_class = type(measurement).__name__
                raise type(exc)(
                    f"[{meas_class}] (step {i + 1}/{len(self._meas)}, "
                    f"key='{key}'): {exc}"
                ) from exc

        # Close the progress bar if it exists
        if self._benchmark and self._verbose and has_tqdm:
            pbar.close()

        df = self._merge_on_object_labels(measurements)

        # Apply post-measurement transforms
        if apply_post:
            for key, post_op in self._post.items():
                logger.debug("Running post-measurement transform: %s", key)
                df = post_op.apply(df)

        return df

    def _build_measurement_run_order(self) -> Dict[str, MeasureFeatures]:
        """Return the measurements to execute for this ``measure()`` call.

        Returns a copy of ``self._meas`` with one optional addition: when both
        ``self._nrows`` and ``self._ncols`` are set and no existing measurement
        is an instance of :class:`GridFinder`, an ``AutoGridFinder`` configured
        with the preset is prepended so it runs before downstream grid-aware
        measurements. The persistent ``self._meas`` mapping is never mutated,
        which keeps repeat ``measure()`` calls idempotent and serialization
        unaffected.

        Returns:
            Dict[str, MeasureFeatures]: Ordered measurement dict for this run.
        """
        if self._nrows is None or self._ncols is None:
            return self._meas

        # Lazy imports to avoid circular dependency with phenotypic.grid.
        from phenotypic.abc_ import GridFinder
        from phenotypic.grid import CenteredAutoGridFinder

        if any(isinstance(m, GridFinder) for m in self._meas.values()):
            return self._meas

        injected_key = (
            "CenteredAutoGridFinder"
            if "CenteredAutoGridFinder" not in self._meas
            else "_CenteredAutoGridFinder_preset"
        )
        run_order: Dict[str, MeasureFeatures] = {
            injected_key: CenteredAutoGridFinder(nrows=self._nrows, ncols=self._ncols),
        }
        run_order.update(self._meas)
        return run_order

    def apply_and_measure(
            self,
            image: Image,
            inplace: bool = False,
            reset: Optional[bool] = None,
            include_metadata: bool = True,
            apply_post: bool = True,
    ) -> pd.DataFrame:
        """
        Applies processing to the given image and measures the results.

        This function first applies a processing method to the supplied image,
        adjusting it based on the given parameters. After processing, the
        resulting image is measured, and a DataFrame containing the measurement
        data is returned.

        Args:
            image (Image): The image to process and measure.
            inplace (bool): Whether to modify the original image directly or
                work on a copy. Default is False.
            reset (bool, optional): Whether to reset any previous processing on the image
                before applying the current method. If None (default), uses the pipeline's
                reset setting. If explicitly set, overrides the pipeline setting.
            include_metadata (bool): Whether to include metadata in the
                measurement results. Default is True.
            apply_post (bool): Forwarded to :meth:`measure`. When ``False``,
                the returned DataFrame skips :class:`PostMeasurement` ops.
                Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing measurement data for the
            processed image.
        """
        img = self.apply(image=image, inplace=inplace, reset=reset)
        return self.measure(
            image=img,
            include_metadata=include_metadata,
            apply_post=apply_post,
        )

    def benchmark_results(self) -> pd.DataFrame:
        """Return execution times and memory usage for operations and measurements.

        This method should be called after applying the pipeline on an image to get
        the execution times and memory consumption of the different processes.

        When an operation is itself an ``ImagePipelineCore`` (nested pipeline),
        its sub-operations are expanded as indented sub-rows beneath the parent
        entry with names like ``"ParentOp > ChildOp"``.

        Returns:
            pd.DataFrame: A DataFrame with columns ``Process Type``,
                ``Process Name``, ``Execution Time (s)``, ``Memory Delta (MB)``,
                and ``RSS After (MB)``.
        """
        columns = [
            "Process Type",
            "Process Name",
            "Execution Time (s)",
            "Memory Delta (MB)",
            "RSS After (MB)",
        ]

        data: List[Dict[str, object]] = []

        # Add operation rows (with nested expansion)
        for op_name, op_time in self._operation_times.items():
            data.append({
                "Process Type"      : "Operation",
                "Process Name"      : op_name,
                "Execution Time (s)": op_time,
                "Memory Delta (MB)" : self._operation_memory.get(op_name, float("nan")),
                "RSS After (MB)"    : self._operation_rss.get(op_name, float("nan")),
            })

            # Expand nested pipeline sub-operations
            operation = self._ops.get(op_name)
            if (
                isinstance(operation, ImagePipelineCore)
                and operation._operation_times
            ):
                for sub_name, sub_time in operation._operation_times.items():
                    data.append({
                        "Process Type"      : "Operation",
                        "Process Name"      : f"  {op_name} > {sub_name}",
                        "Execution Time (s)": sub_time,
                        "Memory Delta (MB)" : operation._operation_memory.get(
                            sub_name, float("nan")
                        ),
                        "RSS After (MB)"    : operation._operation_rss.get(
                            sub_name, float("nan")
                        ),
                    })

        # Add measurement rows
        for meas_name, meas_time in self._measurement_times.items():
            data.append({
                "Process Type"      : "Measurement",
                "Process Name"      : meas_name,
                "Execution Time (s)": meas_time,
                "Memory Delta (MB)" : self._measurement_memory.get(
                    meas_name, float("nan")
                ),
                "RSS After (MB)"    : self._measurement_rss.get(
                    meas_name, float("nan")
                ),
            })

        if not data:
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(data)

        # Total row: sum top-level times and deltas only (exclude sub-rows)
        top_level_mask = ~df["Process Name"].str.startswith("  ")
        total_time = df.loc[top_level_mask, "Execution Time (s)"].sum()
        total_delta = df.loc[top_level_mask, "Memory Delta (MB)"].sum()
        # RSS After for total: last top-level RSS value
        top_level_rss = df.loc[top_level_mask, "RSS After (MB)"]
        final_rss = top_level_rss.iloc[-1] if len(top_level_rss) > 0 else float("nan")

        total_row = pd.DataFrame([{
            "Process Type"      : "Total",
            "Process Name"      : "All Processes",
            "Execution Time (s)": total_time,
            "Memory Delta (MB)" : total_delta,
            "RSS After (MB)"    : final_rss,
        }])
        df = pd.concat([df, total_row], ignore_index=True)

        return df

    @staticmethod
    def _merge_on_object_labels(dataframes_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple DataFrames only if share object labels

        Args:
            dataframes_list: List of pandas DataFrames to merge

        Returns:
            Merged DataFrame containing only the data from DataFrames with matching index names

        Raises:
            ValueError: If no DataFrames are provided or if no matching index names are found
        """
        if not dataframes_list or not all(
                [isinstance(x, pd.DataFrame) for x in dataframes_list]
        ):
            raise ValueError("No DataFrames provided")
        new_df = dataframes_list[0]
        if new_df.index.name == OBJECT.LABEL:
            new_df = new_df.reset_index(drop=False)

        if len(dataframes_list) > 1:
            for df in dataframes_list[1:]:
                if df.index.name == OBJECT.LABEL:
                    df = df.reset_index(drop=False)

                cols_to_merge_on = [OBJECT.LABEL]  # Resets each new other df

                for col_new_df in new_df.columns:
                    if col_new_df != OBJECT.LABEL:  # skip the object label
                        for col_other_df in df.columns:
                            if col_new_df == col_other_df and np.all(
                                    df[col_new_df] == df[col_other_df]
                            ):
                                cols_to_merge_on.append(col_other_df)

                new_df = new_df.merge(df, on=cols_to_merge_on, suffixes=("", "_merged"))

        return new_df
