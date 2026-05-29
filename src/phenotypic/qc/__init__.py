"""GUI-free QC config + compute, shared by the CLI and the results viewer.

This package is the neutral home for the smart-QC machinery so neither the
``_cli`` nor the ``gui`` layer has to depend on the other:

* :mod:`phenotypic.qc._recipe` — the serializable QC config types
  (:class:`QcRecipeEntry`, :class:`QcRecipeLoadWarning`) that back an
  :class:`~phenotypic._core._image_pipeline.ImagePipeline`'s ``qc``
  section, plus :class:`QcRecipe`, a thin adapter that performs scoped
  atomic read-modify-write of *only* the ``qc`` array in ``pipeline.json``
  (mtime-guarded) and folds a legacy ``.viewer_cache/qc_recipe.json``
  sidecar into the pipeline once.
* :mod:`phenotypic.qc._runner` — :func:`run_qc`, which instantiates the
  enabled checks, runs them, and writes the compact ``qc/`` artifact.

**Import hygiene:** this ``__init__`` deliberately re-exports only the
lightweight recipe types. It does **not** import :mod:`._runner` eagerly —
``run_qc`` lazy-imports :class:`ImagePipeline` and the CLI's
``_atomic_write`` *inside* the function body — so importing
``phenotypic.qc`` never pulls in ``_cli`` or ``gui`` and the
``_core -> qc._recipe -> analysis`` dependency edge stays acyclic.
"""

from __future__ import annotations

from ._recipe import (
    QC_RECIPE_FILENAME,
    QC_RECIPE_VERSION,
    QcRecipe,
    QcRecipeEntry,
    QcRecipeLoadWarning,
)

__all__ = [
    "QC_RECIPE_FILENAME",
    "QC_RECIPE_VERSION",
    "QcRecipe",
    "QcRecipeEntry",
    "QcRecipeLoadWarning",
]
