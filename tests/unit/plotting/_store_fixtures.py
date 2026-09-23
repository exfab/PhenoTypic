"""Minimal stores for copy-out tests, and the build -> store -> copy-out path.

`figure_store` writes only what copy-out reads (figures/ + a root carrying the
descriptor), so tests need no pixels. `emit_image_via_store` puts its store
OUTSIDE the test's tmp_path, in a fresh directory per call, so assertions over
tmp_path see only deliverables and a second emit for the same stem never
collides (plan-review B1). The store is removed once copy-out has read it.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from phenotypic.sdk_._image_figures import (
    StoredFigures,
    apply_image_figures_attributes,
    write_image_figures,
)


def figure_store(root: Path, stored: StoredFigures) -> Path:
    """Write a promoted-looking store under *root* and return its path."""
    store = Path(root) / "p.ome.zarr"
    store.mkdir(parents=True)
    phenotypic: dict = {"store_schema_version": 3}
    apply_image_figures_attributes(phenotypic, write_image_figures(store, stored))
    (store / "zarr.json").write_text(
        json.dumps(
            {"zarr_format": 3, "node_type": "group", "attributes": {"phenotypic": phenotypic}}
        ),
        encoding="utf-8",
    )
    return store


def emit_image_via_store(coordinator, image=None, *, dataset="ds", image_stem="plate-1"):
    """build -> minimal store -> copy-out: the path every CLI mode now takes."""
    from phenotypic.plotting._pipeline._store_figures import build_image_figures

    stored = build_image_figures(coordinator._pipeline, object() if image is None else image)
    if stored is None:
        return None
    output_root = coordinator._plots_base.parent.parent
    scratch = Path(tempfile.mkdtemp(prefix=f"{output_root.name}-store-", dir=output_root.parent))
    try:
        store = figure_store(scratch, stored)
        coordinator.publish_store_figures(store, dataset=dataset, image_stem=image_stem)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return stored
