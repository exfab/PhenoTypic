"""Analysis sub-app — composes ``phenotypic.analysis`` chain over a CLI output.

Mounts at :data:`~phenotypic.gui._config.MOUNT_ANALYSIS` (``/analysis/``)
under the unified GUI hub. Exposes a :func:`create_app` factory that
returns a Dash instance configured to read/write
``<output>/pipeline.json`` and emit ``<output>/analysis.{csv,parquet}``
when the pipeline has a model configured.
"""

from __future__ import annotations

from phenotypic.gui.analysis._app import create_app

__all__ = ["create_app"]
