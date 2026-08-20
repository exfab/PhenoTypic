"""Disk-backed preview cache for the builder node-preview modal.

One directory per (session, scope). Each scope dir holds full-resolution
per-node OME-Zarr snapshots (written by ``apply_with_intermediates(...,
full_layers=True)``), a ``manifest.json`` mapping block_id -> store/layers,
and (lazily) staged PNGs + DZI tile pyramids. The cache lives under the
system temp dir and is wiped on launch + ``atexit``.
"""
from __future__ import annotations

import atexit
import copy as _copy
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Final, Optional

__all__ = [
    "MANIFEST_VERSION",
    "preview_cache_root",
    "init_cache",
    "wipe_cache",
    "scope_hash",
    "scope_dir",
    "wipe_scope",
    "read_manifest",
    "write_manifest",
    "compute_scope",
]

_CACHE_SUBPATH = ("phenotypic", "pipeline-preview")
_atexit_registered = False

#: Builder DAG manifest schema version. Introduced when the per-node artifact
#: moved from a single ``.h5`` file to an ``.ome.zarr`` store, so a manifest
#: written by an older session must be rebuilt rather than misread through a
#: key that no longer exists. Manifests predating this constant carry no
#: ``"version"`` at all, which is why ``compute_scope``'s cache-hit guard
#: treats a missing value as stale.
MANIFEST_VERSION: Final[int] = 2

#: Per-node artifact names, in the order the manifest reports them.
_LAYER_ORDER: Final[tuple[str, ...]] = ("rgb", "gray", "detect_mat", "objmap")

#: The pre-pipeline snapshot every scope writes first.
BASE_STORE_NAME: Final[str] = "base_00.ome.zarr"


def preview_cache_root() -> Path:
    """Cache root (recomputed each call so ``$TMPDIR`` changes are honoured)."""
    return Path(tempfile.gettempdir()).joinpath(*_CACHE_SUBPATH)


def wipe_cache() -> None:
    """Best-effort recursive delete of the cache root. Never raises."""
    shutil.rmtree(preview_cache_root(), ignore_errors=True)


def init_cache() -> None:
    """Wipe stale previews on launch and register an atexit cleanup (idempotent)."""
    global _atexit_registered
    wipe_cache()
    preview_cache_root().mkdir(parents=True, exist_ok=True)
    if not _atexit_registered:
        atexit.register(wipe_cache)
        _atexit_registered = True


def scope_hash(scope_path: list[str]) -> str:
    """Stable hash of a scope_path (list of container block_ids)."""
    return hashlib.sha1(
        "/".join(scope_path).encode("utf-8"), usedforsecurity=False
    ).hexdigest()


def _scope_path(session_id: str, scope_path: list[str]) -> Path:
    """Per-(session, scope) directory path WITHOUT creating it."""
    return preview_cache_root() / session_id / scope_hash(scope_path)


def scope_dir(session_id: str, scope_path: list[str]) -> Path:
    """Per-(session, scope) directory, created if missing."""
    d = _scope_path(session_id, scope_path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def wipe_scope(session_id: str, scope_path: list[str]) -> None:
    """Remove a single scope's cache dir (best-effort)."""
    shutil.rmtree(_scope_path(session_id, scope_path), ignore_errors=True)


def read_manifest(session_id: str, scope_path: list[str]) -> Optional[dict]:
    """Return the scope manifest dict, or None if absent/unreadable.

    Pure read: never creates the scope dir as a side effect.
    """
    path = _scope_path(session_id, scope_path) / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_manifest(session_id: str, scope_path: list[str], manifest: dict) -> None:
    """Write the scope manifest atomically (creates the scope dir)."""
    path = scope_dir(session_id, scope_path) / "manifest.json"
    tmp = path.with_name("manifest.json.tmp")
    tmp.write_text(json.dumps(manifest))
    tmp.replace(path)


def _scope_signature_payload(scope) -> dict:
    """Canonical compute-signature payload for a scope."""
    blocks = sorted(
        (
            {
                "id": b.block_id,
                "cls": b.class_name,
                "params": b.params,
                "nested": (
                    _scope_signature_payload(b.nested)
                    if b.nested is not None
                    else None
                ),
            }
            for b in scope.blocks
        ),
        key=lambda d: d["id"],
    )
    edges = sorted(
        (
            {"s": e.source_block_id, "t": e.target_block_id,
             "p": e.target_port, "k": e.kind}
            for e in scope.edges
        ),
        key=lambda d: (d["s"], d["t"], d["p"]),
    )
    return {"blocks": blocks, "edges": edges}


def _scope_signature(scope) -> str:
    """Canonical compute-signature of a scope (ignores UI-only state)."""
    return json.dumps(_scope_signature_payload(scope), sort_keys=True)


def _source_identity(image_path, nrows, ncols) -> str:
    from phenotypic.gui.builder._directory_browser import SYNTHETIC_SENTINEL

    key = image_path or SYNTHETIC_SENTINEL
    return f"{key}|{nrows}|{ncols}"


def _promote_scope_state(scope):
    """Build a temp DAG state whose root IS *scope* (whole scope, no prefix)."""
    from phenotypic.gui.builder._state import _DagBuilderState

    return _DagBuilderState(root=_copy.deepcopy(scope))


def _predecessor_block_id(scope, container_id: str):
    """block_id feeding the container's image input (None if it's the source)."""
    for edge in scope.edges:
        if edge.kind == "image" and edge.target_block_id == container_id:
            return edge.source_block_id
    return None


def _build_manifest(fingerprint, fingerprint_inputs, scope, pipeline,
                    sdir) -> dict:
    """Map the scope's input + op blocks to their stores + layer metadata."""
    from phenotypic import Image
    from phenotypic.gui.builder._conversion_dag import (
        _find_input_block, _topological_image_order,
    )
    from phenotypic.gui.builder._state import stage_of
    from phenotypic.sdk_ import ngff_

    input_block = _find_input_block(scope)
    order = _topological_image_order(scope, input_block)
    non_input = [b for b in order if b.block_id != input_block.block_id]
    # ``stage_of`` returns "ops" for both real ops and the ImagePipeline
    # container sentinel, so this single check covers nested pipelines too.
    ops_blocks = [b for b in non_input if stage_of(b.class_name) == "ops"]

    nodes: dict = {}

    def _describe(block_id, name):
        store = sdir / name
        # An interrupted write leaves no root ``zarr.json``, so the store reads
        # as ABSENT rather than as partial -- the same disposition the missing
        # ``.h5`` file used to get.
        if not (store / ngff_.STORE_ROOT_JSON).is_file():
            return
        block = ngff_.read_phenotypic_attributes(store)
        series = block.get(ngff_.PhenotypicAttr.SERIES, {})
        # ``.get``: a label-less store omits the key entirely (ledger C3).
        labels = block.get(ngff_.PhenotypicAttr.LABELS, {})
        layers = [
            layer for layer in _LAYER_ORDER if layer in series or layer in labels
        ]
        shape, num_objects = [0, 0], 0
        if "gray" in series:
            level0 = ngff_.store_level0_shape(store, series["gray"])
            if level0 is not None:
                shape = list(level0[-2:])
        if ngff_.OBJMAP_LABEL in labels:
            objmap = Image.load_layer_zarr(store, ngff_.OBJMAP_LABEL)
            num_objects = int(objmap.max()) if objmap.size else 0
        nodes[block_id] = {
            "store": name, "layers": layers, "shape": shape,
            "num_objects": num_objects,
        }

    _describe(input_block.block_id, BASE_STORE_NAME)
    # Invariant: pipeline.get_ops() insertion order == _topological_image_order
    # over the same scope == _run_operations' {i:02d}_{key}.ome.zarr naming. The
    # three must stay in lockstep; do not reorder one without the others.
    for i, (op_key, block) in enumerate(zip(pipeline.get_ops().keys(), ops_blocks)):
        _describe(block.block_id, f"{i:02d}_{op_key}.ome.zarr")

    return {
        "version": MANIFEST_VERSION,
        "fingerprint": fingerprint,
        "fingerprint_inputs": fingerprint_inputs,
        "scope_key": "",  # overwritten by the caller with the real scope key
        "nodes": nodes,
        "error": None,
    }


def compute_scope(session_id, state, scope_path, image_path, nrows, ncols) -> dict:
    """Ensure a scope's full-res preview cache is fresh; return its manifest.

    Recursive: a nested scope's input is threaded from its parent's cache
    (the container's main-flow predecessor store). Fingerprints chain so any
    upstream edit invalidates this scope and its descendants.
    """
    from phenotypic.abc_ import GridOperation
    from phenotypic.gui.builder._conversion_dag import to_pipeline_dag
    from phenotypic.gui.builder._linear_model import scope_at_path

    scope = scope_at_path(state.root, list(scope_path))
    if scope is None:
        raise ValueError("compute_scope: stale scope_path")

    promoted = _promote_scope_state(scope)
    pipeline = to_pipeline_dag(promoted)
    sig = _scope_signature(scope)

    if not scope_path:
        input_identity = _source_identity(image_path, nrows, ncols)
        parent_fp = ""
    else:
        parent_manifest = compute_scope(
            session_id, state, list(scope_path[:-1]), image_path, nrows, ncols,
        )
        parent_fp = parent_manifest["fingerprint"]
        input_identity = parent_fp

    fingerprint_inputs = [sig, input_identity]
    fingerprint = hashlib.sha1(
        "\x00".join(fingerprint_inputs).encode(),
        usedforsecurity=False,
    ).hexdigest()

    cached = read_manifest(session_id, list(scope_path))
    # ``.get("version")`` returning None for a pre-version manifest is the
    # whole point: every cache written before the ``.h5`` -> ``.ome.zarr`` move
    # lacks the field, and must MISS and rebuild rather than be read back
    # through a ``"hdf"`` key that no longer exists.
    if cached is not None and cached.get("version") == MANIFEST_VERSION \
            and cached.get("fingerprint") == fingerprint \
            and cached.get("error") is None:
        return cached

    wipe_scope(session_id, list(scope_path))
    sdir = scope_dir(session_id, list(scope_path))

    try:
        if not scope_path:
            from phenotypic.gui.builder._callbacks import (
                _load_preview_image, _pipeline_uses_grid,
            )
            uses_grid = _pipeline_uses_grid(pipeline, GridOperation)
            image = _load_preview_image(image_path, uses_grid, nrows, ncols)
        else:
            parent_scope = scope_at_path(state.root, list(scope_path[:-1]))
            container_id = scope_path[-1]
            pred_id = _predecessor_block_id(parent_scope, container_id)
            parent_dir = scope_dir(session_id, list(scope_path[:-1]))
            if pred_id is None or pred_id not in parent_manifest["nodes"]:
                pred_store = BASE_STORE_NAME
            else:
                pred_store = parent_manifest["nodes"][pred_id]["store"]
            # Dispatches on the stored ``image_class``. Shared with the CLI
            # rather than re-implemented here -- an earlier local copy compared
            # against a hard-coded ``"GridImage"`` literal and had no fallback.
            # Function-local, like every other phenotypic import in this module.
            from phenotypic.sdk_ import load_image_from_store

            image = load_image_from_store(parent_dir / pred_store)

        # Side effect: writes one full-layer store per node into ``sdir``;
        # ``_build_manifest`` rebuilds the manifest from those on-disk stores.
        pipeline.apply_with_intermediates(image, output_dir=sdir, full_layers=True)
        manifest = _build_manifest(
            fingerprint, fingerprint_inputs, scope, pipeline, sdir,
        )
        manifest["scope_key"] = "/".join(scope_path)
    except Exception as exc:  # noqa: BLE001
        manifest = {
            # Versioned too -- without it a failed scope is permanently
            # re-read as stale and recomputed on every callback.
            "version": MANIFEST_VERSION,
            "fingerprint": fingerprint, "fingerprint_inputs": fingerprint_inputs,
            "scope_key": "/".join(scope_path), "nodes": {},
            "error": f"{type(exc).__name__}: {exc}",
        }

    write_manifest(session_id, list(scope_path), manifest)
    return manifest
