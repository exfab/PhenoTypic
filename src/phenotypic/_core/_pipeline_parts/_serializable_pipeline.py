from __future__ import annotations

import json
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image_pipeline import ImagePipeline
    from phenotypic.analysis.abc_ import ModelFitter, SetAnalyzer
    from phenotypic.tools_._qc_recipe import QcRecipeEntry

import warnings

from phenotypic.abc_ import ImageOperation, MeasureFeatures
from ._napari_pipeline_viewer import NapariPipelineViewer

# Import version for serialization - must be after other phenotypic imports to avoid circular import
import phenotypic

__version__ = phenotypic.__version__


@dataclass(frozen=True)
class PipelineLoadWarning:
    """A single skipped analyzer entry from :meth:`SerializablePipeline.from_json`.

    Emitted when ``from_json`` is called with ``skip_unknown_analyzers=True``
    and a filter/model class referenced by the JSON cannot be resolved in
    the live ``phenotypic`` namespace (e.g. the class was renamed or
    removed since the pipeline was saved). The caller decides whether to
    surface the warning, prune the JSON, or rebuild the entry under its
    new name.

    Attributes:
        slot: Which pipeline section the entry came from — ``"filter"``
            for an entry in the ``filters`` dict, ``"model"`` for the
            single ``model`` field.
        name: The dict key the entry was stored under in the JSON.
            For the model slot this is always ``"model"``.
        class_name: The unresolved class name from the JSON's ``class``
            field.
    """

    slot: str
    name: str
    class_name: str


class SerializablePipeline(NapariPipelineViewer):
    """
    An extension of ImagePipelineCore that adds JSON serialization capabilities.

    This class allows pipelines to be saved to and loaded from JSON files, enabling
    pipeline configurations to be stored, shared, and reused across sessions.

    The serialization captures:
    - PhenoTypic version for compatibility checking
    - Pipeline name and description
    - Operation instances with their parameters
    - Measurement instances with their parameters

    Serialization delegates to pydantic v2: every operation, measurement,
    post transform, and analyzer is a ``BaseModel``, so each entry is
    captured as ``{"class": <name>, "params": model.model_dump(mode="json")}``
    and rebuilt with ``cls.model_validate(params)``. ``model_dump`` skips
    ``PrivateAttr`` state (loggers, timing dicts, fitted DataFrames)
    automatically, so internal state is excluded without ad-hoc filtering.
    """

    def to_json(self, filepath: Optional[Union[str, Path]] = None) -> str | None:
        """
        Serialize the pipeline configuration to JSON format.

        This method captures the pipeline's operations and measurements.
        It excludes internal state (pydantic ``PrivateAttr`` fields) and
        pandas DataFrames to keep the serialization clean and focused on
        reproducible configuration.

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
            return None
        else:
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
        return json.dumps(
            SerializablePipeline._serialize_pipeline_config(self), indent=2
        )

    @staticmethod
    def _serialize_pipeline_config(
            pipeline: "SerializablePipeline",
    ) -> Dict[str, object]:
        """Build the JSON-native config envelope for a pipeline.

        Shared by :meth:`__str__` (top-level pipeline) and by the
        pipeline-as-operation / nested-pipeline serialization paths so
        every pipeline — wherever it sits in the tree — is encoded the
        same way.

        Args:
            pipeline: The pipeline whose configuration to capture.

        Returns:
            Dict: The ``{version, name, desc, reset, pipe_cfgs, meas,
            post, filters, model[, nrows, ncols]}`` envelope.
        """
        config: Dict[str, object] = {
            "version"  : __version__,
            "name"     : pipeline.name,
            "desc"     : pipeline._desc,
            "reset"    : pipeline._reset,
            "pipe_cfgs": SerializablePipeline._serialize_operations(
                pipeline._ops
            ),
            "meas"     : SerializablePipeline._serialize_operations(
                pipeline._meas
            ),
            "post"     : SerializablePipeline._serialize_operations(
                pipeline._post
            ),
            "filters"  : SerializablePipeline._serialize_analyzers(
                pipeline._filters
            ),
            "model"    : (
                SerializablePipeline._serialize_analyzer(pipeline._model)
                if pipeline._model is not None
                else None
            ),
        }

        # QC entries serialize with their own dedicated shape — a LIST of
        # ``{instance_id, class, enabled, params}`` (NOT the bare
        # ``{class, params}`` analyzer shape) — so the stable instance_id +
        # enabled flag round-trip. Omitted when empty so QC-free / legacy
        # pipelines round-trip byte-identically.
        qc_entries = SerializablePipeline._serialize_qc(pipeline.get_qc())
        if qc_entries:
            config["qc"] = qc_entries

        # Omit when unset so legacy JSONs round-trip unchanged.
        if pipeline._nrows is not None:
            config["nrows"] = pipeline._nrows
        if pipeline._ncols is not None:
            config["ncols"] = pipeline._ncols

        return config

    @classmethod
    def from_json(
            cls,
            json_data: Union[str, Path, dict],
            benchmark: bool = False,
            verbose: bool = False,
            *,
            skip_unknown_analyzers: bool = False,
            load_warnings: Optional[List[PipelineLoadWarning]] = None,
    ) -> ImagePipeline:
        """
        Deserialize a pipeline from JSON format.

        This method reconstructs a pipeline from a JSON string or file, restoring
        all operations and measurements. Classes are imported from the phenotypic
        namespace and instantiated with their saved parameters via
        ``model_validate``.

        Args:
            json_data: A JSON string, path to a JSON file, or a pre-parsed dict.
            benchmark: Whether to enable benchmarking for the pipeline. Defaults to False.
            verbose: Whether to enable verbose output. Defaults to False.
            skip_unknown_analyzers: When True, filter and model entries whose
                class cannot be resolved in the live ``phenotypic`` namespace
                are dropped silently instead of raising
                :class:`AttributeError`. Useful for forward-compatible loaders
                (e.g. the analysis GUI) that prefer to render a partial
                pipeline plus a warning over a hard 500. Operations,
                measurements, and post entries are *not* covered — they
                still raise on unknown class. Defaults to False (raise on
                unknown analyzer class, preserving the historical contract).
            load_warnings: Optional list to be populated with one
                :class:`PipelineLoadWarning` per skipped analyzer entry.
                Only consulted when ``skip_unknown_analyzers`` is True.
                The on-disk JSON is never modified — the caller decides
                whether to re-save the pruned pipeline.

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

        return cls._deserialize_pipeline_config(
            config,
            benchmark=benchmark,
            verbose=verbose,
            skip_unknown_analyzers=skip_unknown_analyzers,
            load_warnings=load_warnings,
        )

    @classmethod
    def _deserialize_pipeline_config(
            cls,
            config: Dict,
            *,
            benchmark: bool = False,
            verbose: bool = False,
            skip_unknown_analyzers: bool = False,
            load_warnings: Optional[List[PipelineLoadWarning]] = None,
    ) -> ImagePipeline:
        """Reconstruct a pipeline from a parsed config dict.

        Shared by :meth:`from_json` (top-level pipeline) and by the
        nested-pipeline / pipeline-as-operation deserialization paths so
        every pipeline is rebuilt the same way.

        Args:
            config: A parsed pipeline config envelope (see
                :meth:`_serialize_pipeline_config`).
            benchmark: Whether to enable benchmarking for the pipeline.
            verbose: Whether to enable verbose output.
            skip_unknown_analyzers: Forwarded to the analyzer
                deserialization — see :meth:`from_json`.
            load_warnings: Optional list populated with one
                :class:`PipelineLoadWarning` per skipped analyzer entry.

        Returns:
            ImagePipeline: A new pipeline instance with the loaded
            configuration.
        """
        # Deserialize operations, measurements, and post-measurement transforms
        ops = cls._deserialize_operations(config.get("pipe_cfgs", {}))
        meas = cls._deserialize_operations(config.get("meas", {}))
        post = cls._deserialize_operations(config.get("post", {}))
        # Analysis chain — filters/model default to empty/None for backward compat
        from phenotypic.analysis.abc_._model_fitter import ModelFitter

        skipped: Optional[List[PipelineLoadWarning]] = (
            load_warnings if skip_unknown_analyzers else None
        )
        filters = cls._deserialize_analyzers(
            config.get("filters", {}) or {},
            skipped=skipped,
        )
        model_data = config.get("model")
        model: Optional[ModelFitter] = None
        if model_data:
            candidate = cls._deserialize_analyzer(
                model_data, skipped=skipped, slot_name="model"
            )
            if candidate is None:
                # Class wasn't found and skip mode swallowed it; the warning
                # has already been recorded in ``skipped``.
                model = None
            elif not isinstance(candidate, ModelFitter):
                raise TypeError(
                    f"pipeline 'model' must deserialize to a ModelFitter, "
                    f"got {type(candidate).__name__}"
                )
            else:
                model = candidate
        # QC config — a list of ``{instance_id, class, enabled, params}``
        # entries. Unknown classes follow the analyzer path: dropped +
        # recorded when ``skip_unknown_analyzers`` is set, else a hard
        # error. Defaults to ``[]`` for QC-free / legacy pipelines.
        qc = cls._deserialize_qc(
            config.get("qc", []) or [], skipped=skipped
        )
        name = config.get("name", None)
        desc = config.get("desc", None)
        reset = config.get("reset", False)  # Default False for backwards compatibility
        nrows = config.get("nrows", None)
        ncols = config.get("ncols", None)
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

        # Create and return new pipeline instance. ``from_json`` is
        # documented to yield an ``ImagePipeline``; nested-pipeline /
        # pipeline-as-operation callers reach this method through the
        # ``SerializablePipeline`` static context, so the construction
        # class is pinned to ``ImagePipeline`` rather than ``cls`` (a
        # prefab pipeline-as-operation is re-tagged by the caller after
        # reconstruction).
        from phenotypic._core._image_pipeline import ImagePipeline

        return ImagePipeline(
            ops=ops, meas=meas, post=post, filters=filters, model=model,
            qc=qc,
            benchmark=benchmark, verbose=verbose,
            name=name, desc=desc, reset=reset, nrows=nrows, ncols=ncols,
        )

    # ------------------------------------------------------------------ #
    # Operation / measurement / post serialization
    # ------------------------------------------------------------------ #
    #
    # Every operation is a pydantic ``BaseModel``, so each entry is
    # encoded as ``{"class": <name>, "params": model.model_dump(mode="json")}``
    # and rebuilt with ``cls.model_validate(params)``. ``model_dump``
    # already drops ``PrivateAttr`` state and serializes nested
    # ``OperationField`` / ``NdArrayField`` parameters, so there is no
    # ``__dict__`` walking and no ``try/except json.dumps`` block that
    # could silently drop a non-JSON-native parameter.
    #
    # A pipeline used *as* an operation keeps the legacy
    # ``{"class", "__type__": "pipeline_operation", "config": ...}`` shape
    # so previously-written JSON round-trips unchanged.

    @staticmethod
    def _serialize_operations(
            operations: Dict[str, Union[ImageOperation, MeasureFeatures]],
    ) -> Dict:
        """
        Serialize a dictionary of operations, measurements, or post transforms.

        Args:
            operations: Dictionary mapping names to operation instances.

        Returns:
            Dict: Serialized representation with class names and parameters.
        """
        serialized: Dict[str, object] = {}

        for name, op in operations.items():
            class_name = op.__class__.__name__

            # A pipeline used as an operation keeps the legacy
            # ``pipeline_operation`` envelope so old JSON round-trips.
            if isinstance(op, SerializablePipeline):
                serialized[name] = {
                    "class"   : class_name,
                    "__type__": "pipeline_operation",
                    "config"  : SerializablePipeline._serialize_pipeline_config(
                        op
                    ),
                }
                continue

            serialized[name] = {
                "class" : class_name,
                "params": SerializablePipeline._serialize_single_operation(op),
            }

        return serialized

    @staticmethod
    def _serialize_single_operation(
            op: Union[ImageOperation, MeasureFeatures],
    ) -> Dict:
        """
        Serialize a single operation instance to its JSON-native parameters.

        Delegates to pydantic's ``model_dump(mode="json")``: the operation
        is a ``BaseModel``, so this captures every declared field —
        including nested operations carried by an ``OperationField`` and
        raw arrays carried by an ``NdArrayField`` — and skips
        ``PrivateAttr`` internal state.

        Args:
            op: An operation/measurement/post instance.

        Returns:
            Dict: The JSON-native parameter dict produced by ``model_dump``.
        """
        return op.model_dump(mode="json")

    @staticmethod
    def _deserialize_operations(
            serialized: Dict,
    ) -> Dict[str, Union[ImageOperation, MeasureFeatures]]:
        """
        Deserialize a dictionary of operations, measurements, or post transforms.

        Args:
            serialized: Dictionary with serialized operation data.

        Returns:
            Dict: Dictionary mapping names to reconstructed instances.

        Raises:
            ImportError: If a required class cannot be imported.
            AttributeError: If a class cannot be found in phenotypic namespace.
        """

        operations: Dict[str, Union[ImageOperation, MeasureFeatures]] = {}

        for name, op_data in serialized.items():
            class_name = op_data["class"]

            # Pipeline-as-operation entries carry a nested config envelope.
            if op_data.get("__type__") == "pipeline_operation":
                from phenotypic._core._image_pipeline import ImagePipeline

                pipeline = SerializablePipeline._deserialize_pipeline_config(
                    op_data["config"]
                )
                # Re-tag to the specific pipeline subclass when resolvable.
                op_class = SerializablePipeline._find_class_in_phenotypic(
                    class_name
                )
                if op_class is not None and op_class is not ImagePipeline:
                    pipeline.__class__ = op_class
                operations[name] = pipeline
                continue

            op_class = SerializablePipeline._find_class_in_phenotypic(class_name)
            if op_class is None:
                raise AttributeError(
                        f"Class '{class_name}' not found in phenotypic namespace. "
                        f"Make sure it's properly imported in phenotypic.__init__.py"
                )

            params = SerializablePipeline._deserialize_value(
                op_data.get("params", {}) or {}
            )
            operations[name] = op_class.model_validate(params)

        return operations

    @classmethod
    def _deserialize_value(cls, value):
        """
        Normalize a serialized ``params`` payload for ``model_validate``.

        New JSON stores ``params`` as a plain pydantic ``model_dump`` — a
        dict of JSON-native values that ``model_validate`` consumes
        directly, with nested ``OperationField`` parameters reconstructed
        by their own field validators. This method therefore only has to
        translate the *legacy* pre-pydantic nesting markers
        (``__type__`` of ``operation`` / ``operation_list`` / ``pipeline``
        / ``pipeline_list``) into the live operation instances those
        markers described, so pipelines saved before the pydantic
        migration still load.

        Args:
            value: A serialized value — a primitive, a legacy-tagged dict,
                a list, or a plain params dict.

        Returns:
            The value with any legacy operation/pipeline markers replaced
            by reconstructed instances; primitives pass through unchanged.
        """
        # Legacy nested operation marker.
        if isinstance(value, dict) and value.get("__type__") == "operation":
            op_class = cls._find_class_in_phenotypic(value["class"])
            if op_class is None:
                raise AttributeError(
                        f"Class '{value['class']}' not found in phenotypic namespace. "
                        f"Make sure it's properly imported in phenotypic.__init__.py"
                )
            nested_params = {
                k: cls._deserialize_value(v)
                for k, v in (value.get("params", {}) or {}).items()
            }
            return op_class.model_validate(nested_params)

        # Legacy operation-list marker (may mix operations and pipelines).
        if isinstance(value, dict) and value.get("__type__") == "operation_list":
            return [
                cls._deserialize_value(item)
                if item.get("__type__") in ("operation", "pipeline")
                else cls._deserialize_value(
                    {"__type__": "operation", **item}
                )
                for item in value["items"]
            ]

        # Legacy nested-pipeline marker.
        if isinstance(value, dict) and value.get("__type__") == "pipeline":
            return SerializablePipeline._deserialize_pipeline_config(
                value["config"]
            )

        # Legacy pipeline-list marker.
        if isinstance(value, dict) and value.get("__type__") == "pipeline_list":
            return [
                SerializablePipeline._deserialize_pipeline_config(
                    item["config"]
                )
                for item in value["items"]
            ]

        # Recurse into plain dict params / lists so nested legacy markers
        # are still reached; pure JSON-native values pass straight through.
        if isinstance(value, dict):
            return {k: cls._deserialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._deserialize_value(v) for v in value]
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
            "phenotypic.prefab",
            "phenotypic.post",
            "phenotypic.detect.nn",
            "phenotypic.tune",
        ]

        for module_name in submodules:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, class_name):
                    return getattr(module, class_name)
            except ImportError:
                continue

        return None

    # ------------------------------------------------------------------ #
    # Analysis chain serialization (filters / model)
    # ------------------------------------------------------------------ #
    #
    # Analyzers (``SetAnalyzer`` / ``ModelFitter`` subclasses) are pydantic
    # ``BaseModel``s too, so they serialize and reconstruct exactly like
    # operations — ``{"class", "params": model_dump(mode="json")}`` out,
    # ``cls.model_validate(params)`` back. The legacy ``num_workers`` →
    # ``n_jobs`` rename is handled by an ``AliasChoices`` on
    # ``SetAnalyzer.n_jobs``: a JSON file written before the rename spells
    # the key ``num_workers`` and ``model_validate`` accepts it directly,
    # so no alias map is needed here.

    @staticmethod
    def _serialize_analyzer(
            analyzer: Union["SetAnalyzer", "ModelFitter"],
    ) -> Dict[str, object]:
        """Serialize one analyzer instance to a ``{class, params}`` dict.

        Delegates to ``model_dump(mode="json")``; ``PrivateAttr`` state
        (e.g. the cached ``_latest_measurements`` DataFrame) is excluded
        automatically.

        Args:
            analyzer: A :class:`SetAnalyzer` or :class:`ModelFitter`
                instance.

        Returns:
            Dict: ``{"class": <name>, "params": {...}}``.
        """
        return {
            "class" : analyzer.__class__.__name__,
            "params": analyzer.model_dump(mode="json"),
        }

    @staticmethod
    def _serialize_analyzers(
            analyzers: Dict[str, Union["SetAnalyzer", "ModelFitter"]],
    ) -> Dict[str, Dict[str, object]]:
        """Serialize a name-keyed dict of analyzers (the filter chain)."""
        return {
            name: SerializablePipeline._serialize_analyzer(analyzer)
            for name, analyzer in analyzers.items()
        }

    @classmethod
    def _deserialize_analyzer(
            cls, serialized: Dict[str, object],
            *,
            skipped: Optional[List[PipelineLoadWarning]] = None,
            slot_name: Optional[str] = None,
    ) -> Union["SetAnalyzer", "ModelFitter", None]:
        """Reconstruct one analyzer instance from a ``{class, params}`` dict.

        Resolves the class through the ``phenotypic`` registry and rebuilds
        it with ``cls.model_validate(params)``. The legacy
        ``num_workers`` key is accepted via the ``AliasChoices`` on
        ``SetAnalyzer.n_jobs`` — ``model_validate`` translates it to
        ``n_jobs`` with no help needed here.

        Args:
            serialized: ``{"class": ..., "params": {...}}`` dict.
            skipped: When provided, unresolved-class errors are appended to
                this list as :class:`PipelineLoadWarning` and the method
                returns ``None`` instead of raising. When ``None`` (the
                historical default), unresolved classes raise
                :class:`AttributeError`.
            slot_name: The dict-key the entry was stored under (``"model"``
                for the single model field, or the filter chain key).
                Only used to populate the warning's ``name`` field; ignored
                when ``skipped`` is ``None``.
        """
        class_name = serialized["class"]
        if not isinstance(class_name, str):
            raise TypeError(
                f"analyzer 'class' must be a str, got {type(class_name).__name__}"
            )

        op_class = cls._find_class_in_phenotypic(class_name)
        if op_class is None:
            if skipped is not None:
                skipped.append(
                    PipelineLoadWarning(
                        slot="model" if slot_name == "model" else "filter",
                        name=slot_name or class_name,
                        class_name=class_name,
                    )
                )
                return None
            raise AttributeError(
                f"Class '{class_name}' not found in phenotypic namespace. "
                f"Make sure it's properly exported from phenotypic.analysis."
            )

        params = serialized.get("params", {}) or {}
        if not isinstance(params, dict):
            raise TypeError(
                f"analyzer 'params' must be a dict, got {type(params).__name__}"
            )

        return op_class.model_validate(params)

    @classmethod
    def _deserialize_analyzers(
            cls, serialized: Dict[str, Dict[str, object]],
            *,
            skipped: Optional[List[PipelineLoadWarning]] = None,
    ) -> Dict[str, Union["SetAnalyzer", "ModelFitter"]]:
        """Reconstruct a name-keyed dict of analyzers (the filter chain).

        When ``skipped`` is provided, entries whose class can't be
        resolved are dropped from the returned dict and recorded as a
        :class:`PipelineLoadWarning`. Otherwise (the default), an
        unresolved class raises :class:`AttributeError`.
        """
        result: Dict[str, Union["SetAnalyzer", "ModelFitter"]] = {}
        for name, entry in serialized.items():
            instance = cls._deserialize_analyzer(
                entry, skipped=skipped, slot_name=name
            )
            if instance is not None:
                result[name] = instance
        return result

    # ------------------------------------------------------------------ #
    # QC config serialization (the ``qc`` array)
    # ------------------------------------------------------------------ #
    #
    # The QC section is a LIST (not a name-keyed dict) of entries shaped
    # ``{instance_id, class, enabled, params}`` — distinct from the bare
    # ``{class, params}`` analyzer shape because QC entries carry stable
    # ``instance_id`` + ``enabled`` metadata the GUI per-card IDs and
    # ``review_state.json`` key off. The on-disk shape is owned by
    # :class:`phenotypic.qc._recipe.QcRecipeEntry` (``to_dict`` /
    # ``from_dict``) so the (de)serializer and the GUI recipe adapter can
    # never drift.

    @staticmethod
    def _serialize_qc(
            qc_entries: List["QcRecipeEntry"],
    ) -> List[Dict[str, object]]:
        """Serialize the ``qc`` list to a list of entry dicts.

        Args:
            qc_entries: The pipeline's QC config entries (typically from
                :meth:`ImagePipelineCore.get_qc`).

        Returns:
            One ``{instance_id, class, enabled, params}`` dict per entry,
            in order. Empty list when no QC is configured.
        """
        return [entry.to_dict() for entry in qc_entries]

    @classmethod
    def _deserialize_qc(
            cls, serialized: object,
            *,
            skipped: Optional[List[PipelineLoadWarning]] = None,
    ) -> List["QcRecipeEntry"]:
        """Reconstruct the ``qc`` list from its serialized entry dicts.

        Each entry is rebuilt via
        :meth:`phenotypic.qc._recipe.QcRecipeEntry.from_dict`, which resolves
        the check class within :mod:`phenotypic.analysis`. Unknown classes
        follow the **analyzer** contract (not the operations one): when
        ``skipped`` is provided (``skip_unknown_analyzers=True``) the entry
        is dropped and recorded as a :class:`PipelineLoadWarning` so one
        stale entry never bricks pipeline load; otherwise an unresolved
        class raises :class:`AttributeError`.

        Note that resolution failure is the *only* tolerance applied here —
        an entry whose class resolves but whose params are unbuildable
        (e.g. a missing metadata file) still deserializes fine; that
        failure surfaces later, lazily, at instantiate time (``run_qc`` /
        the GUI).

        Args:
            serialized: The raw ``qc`` value from the config (expected to
                be a list of entry dicts; a non-list yields an empty list).
            skipped: When provided, unresolved-class entries are dropped and
                appended here as :class:`PipelineLoadWarning`. When ``None``
                (the strict default), an unresolved class raises.

        Returns:
            The reconstructed :class:`QcRecipeEntry` list, in order.

        Raises:
            AttributeError: If an entry's class cannot be resolved and
                ``skipped`` is ``None``.
        """
        from phenotypic.tools_._qc_recipe import QcRecipeEntry, QcRecipeLoadWarning

        if not isinstance(serialized, list):
            return []

        entries: List["QcRecipeEntry"] = []
        for item in serialized:
            if not isinstance(item, dict):
                continue
            parsed = QcRecipeEntry.from_dict(item)
            if isinstance(parsed, QcRecipeLoadWarning):
                if skipped is not None:
                    skipped.append(
                        PipelineLoadWarning(
                            slot="qc",
                            name=parsed.instance_id,
                            class_name=parsed.class_name,
                        )
                    )
                    continue
                raise AttributeError(
                    f"QC check class '{parsed.class_name}' not found in "
                    f"phenotypic.analysis. Make sure it's properly exported "
                    f"from phenotypic.analysis."
                )
            entries.append(parsed)
        return entries
