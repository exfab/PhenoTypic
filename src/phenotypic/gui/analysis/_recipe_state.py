"""Thin wrapper around ``<output>/pipeline.json`` for the analysis sub-app.

The canonical ``pipeline.json`` written by the CLI's
:func:`~phenotypic._cli._cli_output_manager._persist_pipeline_to_output_dir`
captures the entire reproducibility surface (operations, measurements,
post, filters, model). The analysis GUI reads from and writes back to
this file every time the user adds, removes, or re-parameterises a
section.

This module provides:

- :class:`RecipeState`, a dataclass wrapping the in-memory
  :class:`~phenotypic._core._image_pipeline.ImagePipeline` instance plus
  the on-disk ``pipeline.json`` path.
- :meth:`RecipeState.load`, the boot-time loader.
- :meth:`RecipeState.save`, atomic JSON write + mtime refresh.
- mtime-staleness detection mirroring the pattern in
  :mod:`phenotypic.gui.results_viewer._filtered_state`. When the on-disk
  mtime no longer matches what we observed at load time (typical cause:
  a CLI recompile-mode run happened while the viewer session was open), we
  refuse to clobber the fresh seed and surface a "reload required"
  banner instead.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

from pydantic import AliasChoices, AliasPath, BaseModel

from phenotypic._core._pipeline_parts._serializable_pipeline import (
    PipelineLoadWarning,
    SerializablePipeline,
)
from phenotypic.sdk_ import (
    DIR_DELIVERABLES,
    atomic_write_text,
    pipeline_json_path,
    pipeline_publication_lock,
    resolve_pipeline_config_path,
)

if TYPE_CHECKING:
    from phenotypic._core._image_pipeline import ImagePipeline
    from phenotypic.sdk_ import BundleLayout

logger = logging.getLogger(__name__)

_SERIALIZED_PIPELINE_KEYS = frozenset({
    "version",
    "name",
    "desc",
    "reset",
    "pipe_cfgs",
    "meas",
    "post",
    "filters",
    "model",
    "qc",
    "plots",
    "nrows",
    "ncols",
})
_ANALYZER_ENVELOPE_KEYS = frozenset({"class", "params"})
_QC_ENVELOPE_KEYS = frozenset({"instance_id", "class", "enabled", "params"})
_PLOT_ENVELOPE_KEYS = frozenset({"id", "ref", "inline", "input"})
_INLINE_PLOT_KEYS = frozenset({"module", "qualname", "params"})


def _validation_alias_roots(
    alias: str | AliasChoices | AliasPath | None,
) -> set[str]:
    """Return top-level mapping keys accepted by one Pydantic alias."""
    if isinstance(alias, str):
        return {alias}
    if isinstance(alias, AliasChoices):
        roots: set[str] = set()
        for choice in alias.choices:
            roots.update(_validation_alias_roots(choice))
        return roots
    if isinstance(alias, AliasPath) and alias.path:
        root = alias.path[0]
        return {root} if isinstance(root, str) else set()
    return set()


def _strip_unowned_model_fields(
    raw: object,
    model_class: type[BaseModel] | None,
) -> object:
    """Copy a model payload with only fields its validation schema owns."""
    if not isinstance(raw, dict) or model_class is None:
        return copy.deepcopy(raw)
    if model_class.model_config.get("extra") != "forbid":
        return copy.deepcopy(raw)

    accepted: set[str] = set()
    for name, model_field in model_class.model_fields.items():
        accepted.add(name)
        accepted.update(_validation_alias_roots(model_field.validation_alias))
        if isinstance(model_field.alias, str):
            accepted.add(model_field.alias)
    return {
        key: copy.deepcopy(value)
        for key, value in raw.items()
        if key in accepted
    }


def _resolved_analyzer_class(node: object) -> type[BaseModel] | None:
    """Resolve a known analyzer envelope through the production registry."""
    if not isinstance(node, dict):
        return None
    class_name = node.get("class")
    if not isinstance(class_name, str):
        return None
    resolved = SerializablePipeline._find_class_in_phenotypic(class_name)
    if (
        isinstance(resolved, type)
        and issubclass(resolved, BaseModel)
    ):
        return resolved
    return None


def _analyzer_validation_node(node: object) -> object:
    """Build the strict-loader copy of one filter or model envelope."""
    if not isinstance(node, dict):
        return copy.deepcopy(node)
    validation_node = {
        key: copy.deepcopy(value)
        for key, value in node.items()
        if key in _ANALYZER_ENVELOPE_KEYS
    }
    if "params" in validation_node:
        validation_node["params"] = _strip_unowned_model_fields(
            validation_node["params"],
            _resolved_analyzer_class(node),
        )
    return validation_node


def _qc_validation_node(node: object) -> object:
    """Build the strict-loader copy of one QC envelope."""
    if not isinstance(node, dict):
        return copy.deepcopy(node)
    validation_node = {
        key: copy.deepcopy(value)
        for key, value in node.items()
        if key in _QC_ENVELOPE_KEYS
    }
    from phenotypic.sdk_._qc_recipe import QcRecipeEntry

    parsed = QcRecipeEntry.from_dict(node)
    model_class = parsed.cls if isinstance(parsed, QcRecipeEntry) else None
    if "params" in validation_node:
        validation_node["params"] = _strip_unowned_model_fields(
            validation_node["params"],
            model_class,
        )
    return validation_node


def _plot_validation_node(node: object) -> object:
    """Build the strict-loader copy of one plot binding envelope."""
    if not isinstance(node, dict):
        return copy.deepcopy(node)
    validation_node = {
        key: copy.deepcopy(value)
        for key, value in node.items()
        if key in _PLOT_ENVELOPE_KEYS
    }

    from phenotypic.plotting._bindings import (
        AnalysisInput,
        MeasurementInput,
        PipelineObjectRef,
        _load_qualified_class,
    )

    if "ref" in validation_node:
        validation_node["ref"] = _strip_unowned_model_fields(
            validation_node["ref"],
            PipelineObjectRef,
        )

    input_payload = validation_node.get("input")
    if isinstance(input_payload, dict):
        input_models = {
            "measurements": MeasurementInput,
            "analysis": AnalysisInput,
        }
        input_kind = input_payload.get("kind")
        validation_node["input"] = _strip_unowned_model_fields(
            input_payload,
            input_models.get(input_kind)
            if isinstance(input_kind, str)
            else None,
        )

    inline = validation_node.get("inline")
    if isinstance(inline, dict):
        validation_inline = {
            key: copy.deepcopy(value)
            for key, value in inline.items()
            if key in _INLINE_PLOT_KEYS
        }
        module = inline.get("module")
        qualname = inline.get("qualname")
        plot_class: type[BaseModel] | None = None
        if isinstance(module, str) and isinstance(qualname, str):
            try:
                resolved = _load_qualified_class(
                    module_name=module,
                    qualname=qualname,
                )
            except (AttributeError, ImportError):
                pass
            else:
                if isinstance(resolved, type) and issubclass(
                    resolved,
                    BaseModel,
                ):
                    plot_class = resolved
        if "params" in validation_inline:
            validation_inline["params"] = _strip_unowned_model_fields(
                validation_inline["params"],
                plot_class,
            )
        validation_node["inline"] = validation_inline
    return validation_node


def _pipeline_validation_payload(
    source_payload: dict[str, Any],
) -> dict[str, Any]:
    """Return a deep-copied payload safe for this version's strict schemas.

    The exact parsed payload remains the round-trip source of truth. This
    temporary copy removes only fields unowned by current known-node schemas
    so forward-compatible extensions cannot block tolerant GUI loading.
    """
    validation_payload = copy.deepcopy(source_payload)

    filters = validation_payload.get("filters")
    if isinstance(filters, dict):
        validation_payload["filters"] = {
            name: _analyzer_validation_node(node)
            for name, node in filters.items()
        }

    model = validation_payload.get("model")
    if model is not None:
        validation_payload["model"] = _analyzer_validation_node(model)

    qc = validation_payload.get("qc")
    if isinstance(qc, list):
        validation_payload["qc"] = [
            _qc_validation_node(node) for node in qc
        ]

    plots = validation_payload.get("plots")
    if isinstance(plots, list):
        validation_payload["plots"] = [
            _plot_validation_node(node) for node in plots
        ]
    return validation_payload


def _merge_missing_mapping_fields(
    current: object,
    original: object,
) -> object:
    """Recursively retain forward-compatible mapping fields absent today."""
    if not isinstance(current, dict) or not isinstance(original, dict):
        return copy.deepcopy(current)
    merged = copy.deepcopy(current)
    for key, raw_value in original.items():
        if key not in merged:
            merged[key] = copy.deepcopy(raw_value)
        elif isinstance(merged[key], dict) and isinstance(raw_value, dict):
            merged[key] = _merge_missing_mapping_fields(
                merged[key],
                raw_value,
            )
    return merged


def _merge_envelope_extensions(
    current: object,
    original: object,
    *,
    owned_keys: frozenset[str],
    same_identity: bool,
    nested_mapping_keys: frozenset[str],
) -> object:
    """Retain extension fields only while the serialized node is the same."""
    if (
        not same_identity
        or not isinstance(current, dict)
        or not isinstance(original, dict)
    ):
        return copy.deepcopy(current)
    merged = copy.deepcopy(current)
    for key, raw_value in original.items():
        if key not in owned_keys:
            if key not in merged:
                merged[key] = copy.deepcopy(raw_value)
            elif isinstance(merged[key], dict) and isinstance(raw_value, dict):
                merged[key] = _merge_missing_mapping_fields(
                    merged[key],
                    raw_value,
                )
        elif key in nested_mapping_keys and key in merged:
            merged[key] = _merge_missing_mapping_fields(
                merged[key],
                raw_value,
            )
    return merged


def _referenced_class(
    pipeline_payload: dict[str, Any],
    ref: dict[str, Any],
) -> object:
    """Return the serialized class targeted by one plot reference."""
    slot = ref.get("slot")
    key = ref.get("key")
    if slot == "model":
        model = pipeline_payload.get("model")
        return model.get("class") if isinstance(model, dict) else None
    if not isinstance(slot, str):
        return None
    section = pipeline_payload.get(slot)
    if isinstance(section, dict):
        node = section.get(key)
        return node.get("class") if isinstance(node, dict) else None
    if isinstance(section, list):
        for node in section:
            if (
                isinstance(node, dict)
                and node.get("instance_id") == key
            ):
                return node.get("class")
    return None


def _plot_identity(
    node: object,
    pipeline_payload: dict[str, Any],
) -> tuple[object, ...] | None:
    """Return the stable serialized identity and class of one plot."""
    if not isinstance(node, dict):
        return None
    ref = node.get("ref")
    if isinstance(ref, dict):
        return (
            "ref",
            ref.get("slot"),
            ref.get("key"),
            _referenced_class(pipeline_payload, ref),
        )
    inline = node.get("inline")
    if isinstance(inline, dict):
        return ("inline", inline.get("module"), inline.get("qualname"))
    return None


def _plot_input_identity(node: object) -> tuple[object, ...] | None:
    """Return one known input variant's durable serialized identity."""
    if not isinstance(node, dict):
        return None
    kind = node.get("kind")
    if kind == "measurements":
        return ("measurements",)
    if kind == "analysis":
        analysis_id = node.get("analysis_id")
        if isinstance(analysis_id, str):
            return ("analysis", analysis_id)
    return None


def _merge_known_node_extensions(
    current: dict[str, Any],
    original: dict[str, Any],
) -> dict[str, Any]:
    """Merge extensions while durable serialized node identities are stable."""
    merged = copy.deepcopy(current)

    current_filters = merged.get("filters")
    original_filters = original.get("filters")
    if isinstance(current_filters, dict) and isinstance(original_filters, dict):
        for name, current_node in list(current_filters.items()):
            original_node = original_filters.get(name)
            same_class = (
                isinstance(current_node, dict)
                and isinstance(original_node, dict)
                and current_node.get("class") == original_node.get("class")
            )
            current_filters[name] = _merge_envelope_extensions(
                current_node,
                original_node,
                owned_keys=_ANALYZER_ENVELOPE_KEYS,
                same_identity=same_class,
                nested_mapping_keys=frozenset({"params"}),
            )

    current_model = merged.get("model")
    original_model = original.get("model")
    same_model_class = (
        isinstance(current_model, dict)
        and isinstance(original_model, dict)
        and current_model.get("class") == original_model.get("class")
    )
    if current_model is not None:
        merged["model"] = _merge_envelope_extensions(
            current_model,
            original_model,
            owned_keys=_ANALYZER_ENVELOPE_KEYS,
            same_identity=same_model_class,
            nested_mapping_keys=frozenset({"params"}),
        )

    current_qc = merged.get("qc")
    original_qc = original.get("qc")
    if isinstance(current_qc, list) and isinstance(original_qc, list):
        original_by_id = {
            str(node.get("instance_id")): node
            for node in original_qc
            if isinstance(node, dict) and "instance_id" in node
        }
        for index, current_node in enumerate(current_qc):
            if not isinstance(current_node, dict):
                continue
            original_node = original_by_id.get(
                str(current_node.get("instance_id"))
            )
            same_qc_class = (
                isinstance(original_node, dict)
                and current_node.get("class") == original_node.get("class")
            )
            current_qc[index] = _merge_envelope_extensions(
                current_node,
                original_node,
                owned_keys=_QC_ENVELOPE_KEYS,
                same_identity=same_qc_class,
                nested_mapping_keys=frozenset({"params"}),
            )

    current_plots = merged.get("plots")
    original_plots = original.get("plots")
    if isinstance(current_plots, list) and isinstance(original_plots, list):
        original_by_id = {
            str(node.get("id")): node
            for node in original_plots
            if isinstance(node, dict) and "id" in node
        }
        for index, current_node in enumerate(current_plots):
            if not isinstance(current_node, dict):
                continue
            original_node = original_by_id.get(str(current_node.get("id")))
            current_input = current_node.get("input")
            original_input = (
                original_node.get("input")
                if isinstance(original_node, dict)
                else None
            )
            current_input_identity = _plot_input_identity(current_input)
            original_input_identity = _plot_input_identity(original_input)
            same_input_identity = (
                current_input is None
                and original_input is None
            ) or (
                current_input_identity is not None
                and current_input_identity == original_input_identity
            )
            same_plot_identity = (
                _plot_identity(current_node, current)
                == _plot_identity(original_node, original)
                and same_input_identity
            )
            merged_node = _merge_envelope_extensions(
                current_node,
                original_node,
                owned_keys=_PLOT_ENVELOPE_KEYS,
                same_identity=same_plot_identity,
                nested_mapping_keys=frozenset({"ref", "inline"}),
            )
            if (
                same_plot_identity
                and isinstance(merged_node, dict)
                and isinstance(original_node, dict)
            ):
                if (
                    current_input_identity is not None
                    and current_input_identity == original_input_identity
                ):
                    merged_node["input"] = _merge_missing_mapping_fields(
                        merged_node.get("input"),
                        original_input,
                    )
            current_plots[index] = merged_node
    return merged


def _merge_named_opaque_nodes(
    current: object,
    original: object,
    opaque_names: set[str],
    *,
    identity_key: str | None = None,
) -> object:
    """Merge opaque nodes into their original positions.

    Args:
        current: Newly serialized dict or list.
        original: Raw dict or list retained from tolerant load.
        opaque_names: Original keys or identities that must survive.
        identity_key: List-entry key used as identity, or ``None`` for dicts.

    Returns:
        A deep-copied collection with current editable nodes and original
        opaque nodes in their original relative positions.
    """
    if identity_key is None:
        if not isinstance(current, dict) or not isinstance(original, dict):
            return current
        merged: dict[str, Any] = {}
        for name, raw_node in original.items():
            if name in current:
                merged[name] = copy.deepcopy(current[name])
            elif name in opaque_names:
                merged[name] = copy.deepcopy(raw_node)
        for name, node in current.items():
            if name not in merged:
                merged[name] = copy.deepcopy(node)
        return merged

    if not isinstance(current, list) or not isinstance(original, list):
        return current
    current_by_name = {
        str(node.get(identity_key)): node
        for node in current
        if isinstance(node, dict) and identity_key in node
    }
    merged_list: list[Any] = []
    emitted: set[str] = set()
    for raw_node in original:
        if not isinstance(raw_node, dict) or identity_key not in raw_node:
            continue
        name = str(raw_node[identity_key])
        if name in current_by_name:
            merged_list.append(copy.deepcopy(current_by_name[name]))
            emitted.add(name)
        elif name in opaque_names:
            merged_list.append(copy.deepcopy(raw_node))
            emitted.add(name)
    for node in current:
        if not isinstance(node, dict) or identity_key not in node:
            merged_list.append(copy.deepcopy(node))
            continue
        name = str(node[identity_key])
        if name not in emitted:
            merged_list.append(copy.deepcopy(node))
            emitted.add(name)
    return merged_list


def _merge_opaque_pipeline_payload(
    current: dict[str, Any],
    original: dict[str, Any] | None,
    warnings: List[PipelineLoadWarning],
) -> dict[str, Any]:
    """Preserve opaque and forward-compatible nodes during a scoped save."""
    if original is None:
        return current

    merged = _merge_known_node_extensions(current, original)
    opaque_filters = {w.name for w in warnings if w.slot == "filter"}
    if opaque_filters:
        merged["filters"] = _merge_named_opaque_nodes(
            merged.get("filters", {}),
            original.get("filters", {}),
            opaque_filters,
        )

    if any(w.slot == "model" for w in warnings) and merged.get("model") is None:
        merged["model"] = copy.deepcopy(original.get("model"))

    opaque_qc = {w.name for w in warnings if w.slot == "qc"}
    if opaque_qc:
        merged["qc"] = _merge_named_opaque_nodes(
            merged.get("qc", []),
            original.get("qc", []),
            opaque_qc,
            identity_key="instance_id",
        )

    opaque_plot_ids = {w.name for w in warnings if w.slot == "plot"}
    opaque_refs: set[tuple[str, str | None]] = {
        ("filters", name) for name in opaque_filters
    }
    if any(w.slot == "model" for w in warnings):
        opaque_refs.add(("model", None))
    opaque_refs.update(("qc", name) for name in opaque_qc)
    original_plots = original.get("plots", [])
    if isinstance(original_plots, list):
        for node in original_plots:
            if not isinstance(node, dict):
                continue
            ref = node.get("ref")
            if not isinstance(ref, dict):
                continue
            if (ref.get("slot"), ref.get("key")) in opaque_refs:
                opaque_plot_ids.add(str(node.get("id")))
    if opaque_plot_ids:
        merged["plots"] = _merge_named_opaque_nodes(
            merged.get("plots", []),
            original_plots,
            opaque_plot_ids,
            identity_key="id",
        )

    # Preserve extension metadata that this version does not own. Known
    # pipeline sections keep current serialization semantics, including
    # intentional removal of known nodes.
    ordered: dict[str, Any] = {}
    for key, raw_value in original.items():
        if key in merged:
            ordered[key] = merged[key]
        elif key not in _SERIALIZED_PIPELINE_KEYS:
            ordered[key] = copy.deepcopy(raw_value)
    for key, value in merged.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _remaining_opaque_warnings(
    current: dict[str, Any],
    warnings: List[PipelineLoadWarning],
) -> List[PipelineLoadWarning]:
    """Drop warnings whose slot was explicitly replaced by a live node."""
    current_filters = current.get("filters", {})
    filter_names = (
        set(current_filters) if isinstance(current_filters, dict) else set()
    )
    current_qc = current.get("qc", [])
    qc_names = {
        str(node.get("instance_id"))
        for node in current_qc
        if isinstance(node, dict) and "instance_id" in node
    } if isinstance(current_qc, list) else set()
    current_plots = current.get("plots", [])
    plot_names = {
        str(node.get("id"))
        for node in current_plots
        if isinstance(node, dict) and "id" in node
    } if isinstance(current_plots, list) else set()

    remaining: List[PipelineLoadWarning] = []
    for warning in warnings:
        replaced = (
            (warning.slot == "filter" and warning.name in filter_names)
            or (warning.slot == "model" and current.get("model") is not None)
            or (warning.slot == "qc" and warning.name in qc_names)
            or (warning.slot == "plot" and warning.name in plot_names)
        )
        if not replaced:
            remaining.append(warning)
    return remaining


@dataclass
class RecipeState:
    """In-memory + on-disk view of an output dir's ``pipeline.json``.

    Attributes:
        path: Path to the ``pipeline.json`` file under management.
        pipeline: The currently-loaded :class:`ImagePipeline` instance.
            Mutate this reference directly (e.g. ``state.pipeline.set_model(...)``)
            then call :meth:`save` to persist.
        seed_mtime_ns: Nanosecond mtime of :attr:`path` as last observed
            by this instance. ``None`` means the file did not exist when
            the instance was built. Refreshed by :meth:`load` and
            :meth:`save`.
    """

    path: Path
    pipeline: "ImagePipeline"
    seed_mtime_ns: Optional[int] = None
    source_path: Optional[Path] = None
    #: JSON string from the most recent successful :meth:`save`. Callbacks
    #: read this for the ``ANALYSIS_PIPELINE_STORE`` payload instead of
    #: re-serializing the pipeline a second time.
    last_json: str = ""
    #: Filter / model entries that the on-disk JSON referenced but whose
    #: class could not be resolved in the live ``phenotypic`` namespace
    #: (typical cause: an analyzer was renamed or removed since the
    #: pipeline was saved). The analysis page renders a banner listing
    #: these so the user can manually select a replacement. Their exact raw
    #: nodes are retained in :attr:`source_payload` and merged through
    #: unrelated saves until a live node explicitly replaces the same slot.
    load_warnings: List[PipelineLoadWarning] = field(default_factory=list)
    #: Exact parsed payload retained across tolerant loading so a scoped
    #: Analysis edit cannot silently discard unknown analyzer nodes.
    source_payload: dict[str, Any] | None = field(default=None, repr=False)
    #: Resolved bundle topology this recipe was built from, when constructed
    #: via :meth:`from_layout`. ``None`` for the legacy ``output_dir``-rooted
    #: :meth:`load` path. When set, :meth:`reload` re-resolves through it so a
    #: standalone bundle (``root`` IS the deliverables folder) never
    #: double-joins ``deliverables/`` on a staleness refresh.
    layout: Optional["BundleLayout"] = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def load(cls, output_dir: Path) -> "RecipeState":
        """Load (or seed) the recipe state for *output_dir*.

        When ``<output>/pipeline.json`` is present it is parsed via
        :meth:`ImagePipeline.from_json`. When absent, an empty
        :class:`ImagePipeline` is used and the file is *not* created
        until the first :meth:`save` — the empty file would just be
        chrome with no information, and creating it eagerly would break
        the "freshly-curated output dir without analysis configured"
        affordance the GUI relies on.

        Args:
            output_dir: Path to a CLI output root (must contain
                ``master_measurements.parquet`` for the broader sub-app
                to function, but :class:`RecipeState` itself only cares
                about the pipeline JSON).

        Returns:
            A :class:`RecipeState` ready for in-place mutation +
            :meth:`save`.
        """
        pipeline_path = pipeline_json_path(output_dir)
        read_path = resolve_pipeline_config_path(output_dir)
        return cls._load_from_paths(
            read_path, pipeline_path, name_hint=output_dir.name
        )

    @classmethod
    def from_layout(cls, layout: "BundleLayout") -> "RecipeState":
        """Load (or seed) the recipe state from a resolved :class:`BundleLayout`.

        The :class:`BundleLayout`-aware sibling of :meth:`load`, mirroring
        :meth:`phenotypic.sdk_._qc_recipe._recipe.QcRecipe.from_layout`. Anchors
        the pipeline config on ``layout.deliverables_base`` directly (with the
        same legacy plain-``pipeline.json`` fallback ``resolve_pipeline_config_path``
        provides), so a standalone deliverables bundle — whose
        ``layout.output_root is None`` and whose viewer ``root`` is already the
        deliverables folder — resolves ``pipeline.json`` *inside the bundle*
        rather than via ``pipeline_json_path(output_root)``, which would
        double-join ``deliverables/``.

        The resolved ``layout`` is retained on :attr:`layout` so
        :meth:`reload` re-resolves through it instead of the ``output_dir``
        heuristic.

        Args:
            layout: Resolved bundle topology.

        Returns:
            A :class:`RecipeState` ready for in-place mutation + :meth:`save`.
        """
        name_root = (
            layout.output_root
            if layout.output_root is not None
            else layout.deliverables_base
        )
        return cls._load_from_paths(
            layout.resolved_pipeline_config_path,
            layout.pipeline_config_path,
            name_hint=name_root.name,
            layout=layout,
        )

    @classmethod
    def _load_from_paths(
        cls,
        read_path: Path,
        pipeline_path: Path,
        *,
        name_hint: str,
        layout: "BundleLayout | None" = None,
    ) -> "RecipeState":
        """Build a recipe from explicit read + canonical-write paths.

        Shared core of :meth:`load` and :meth:`from_layout`. ``read_path`` is the
        existing config to parse (canonical typed or legacy ``.json``);
        ``pipeline_path`` is the canonical typed path future writes target.
        ``name_hint`` seeds the empty-pipeline name when no config exists.
        """
        from phenotypic._core._image_pipeline import ImagePipeline

        load_warnings: List[PipelineLoadWarning] = []
        source_payload: dict[str, Any] | None = None
        source_text = ""

        if read_path.exists():
            source_text = read_path.read_text(encoding="utf-8")
            parsed_payload = json.loads(source_text)
            if not isinstance(parsed_payload, dict):
                raise TypeError("pipeline configuration must be a JSON object")
            source_payload = parsed_payload
            pipeline = ImagePipeline.from_json(
                _pipeline_validation_payload(parsed_payload),
                skip_unknown_analyzers=True,
                load_warnings=load_warnings,
            )
            try:
                mtime = read_path.stat().st_mtime_ns
            except OSError:
                mtime = None
            if load_warnings:
                logger.warning(
                    "%s referenced %d unknown analyzer class(es); the "
                    "analysis page will render a banner. Skipped: %s",
                    read_path,
                    len(load_warnings),
                    ", ".join(w.class_name for w in load_warnings),
                )
        else:
            pipeline = ImagePipeline(name=f"analysis-{name_hint}")
            mtime = None

        return cls(
            path=pipeline_path,
            pipeline=pipeline,
            seed_mtime_ns=mtime,
            source_path=read_path if read_path != pipeline_path else None,
            last_json=source_text,
            load_warnings=load_warnings,
            source_payload=source_payload,
            layout=layout,
        )

    def is_stale(self) -> bool:
        """Return ``True`` when the on-disk file changed since load.

        The CLI seeds ``pipeline.json`` on every aggregate run; if the
        user re-runs the CLI while a viewer session is open, this method
        will start returning ``True`` until they reload via :meth:`load`.
        Callers should refuse to :meth:`save` until the staleness is
        cleared so we don't overwrite a fresh seed with a stale recipe.
        """
        if self.seed_mtime_ns is None:
            # No on-disk file yet — nothing to be stale against.
            return False
        tracked_path = self._tracked_path()
        try:
            current = tracked_path.stat().st_mtime_ns
        except FileNotFoundError:
            # File deleted out from under us; treat as stale.
            return True
        return current != self.seed_mtime_ns

    def _tracked_path(self) -> Path:
        """Return the path whose mtime should be compared against the seed."""
        if self.path.exists() or self.source_path is None:
            return self.path
        return self.source_path

    def _output_root(self) -> Path:
        """Return the output root that owns :attr:`path`."""
        if self.path.parent.name == DIR_DELIVERABLES:
            return self.path.parent.parent
        return self.path.parent

    def save(self) -> bool:
        """Atomically write :attr:`pipeline` to :attr:`path`.

        Caches the serialized JSON on :attr:`last_json` on success so
        callers (e.g. the analysis-page store) can read it without
        re-serializing the pipeline a second time.

        Returns:
            ``True`` when the write succeeded, ``False`` when the file
            was stale (caller must reload before retrying) or the
            atomic rename failed. Failures other than staleness are
            logged at WARNING.
        """
        with self._lock:
            with pipeline_publication_lock(self.path):
                if self.is_stale():
                    logger.warning(
                        "Refusing to overwrite %s — mtime changed since "
                        "load (likely a CLI recompile-mode run). Reload "
                        "before saving again.",
                        self.path,
                    )
                    return False

                serialized = json.loads(self.pipeline.to_json() or "{}")
                if not isinstance(serialized, dict):
                    logger.warning(
                        "Refusing to save non-object pipeline payload to %s",
                        self.path,
                    )
                    return False
                remaining_warnings = _remaining_opaque_warnings(
                    serialized,
                    self.load_warnings,
                )
                merged = _merge_opaque_pipeline_payload(
                    serialized,
                    self.source_payload,
                    remaining_warnings,
                )
                payload = json.dumps(merged, indent=2)

                try:
                    atomic_write_text(self.path, payload)
                except Exception:
                    logger.warning(
                        "Atomic write failed for %s",
                        self.path,
                        exc_info=True,
                    )
                    return False

                self.seed_mtime_ns = self.path.stat().st_mtime_ns
                self.source_path = None
                self.last_json = payload
                self.source_payload = merged
                self.load_warnings = remaining_warnings
                return True

    def reload(self) -> None:
        """Re-read the on-disk pipeline, replacing :attr:`pipeline`.

        Used after :meth:`is_stale` returns ``True`` to pick up a fresh
        CLI seed before resuming edits. When this state was built via
        :meth:`from_layout`, re-resolve through the retained layout so a
        standalone bundle never double-joins ``deliverables/`` here.
        """
        with self._lock:
            if self.layout is not None:
                fresh = type(self).from_layout(self.layout)
            else:
                fresh = type(self).load(self._output_root())
            self.pipeline = fresh.pipeline
            self.seed_mtime_ns = fresh.seed_mtime_ns
            self.source_path = fresh.source_path
            self.load_warnings = fresh.load_warnings
            self.last_json = fresh.last_json
            self.source_payload = fresh.source_payload
