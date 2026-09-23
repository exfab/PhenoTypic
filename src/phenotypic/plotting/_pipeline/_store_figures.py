"""Build one image's figures in memory for the store (spec §3 step 1).

Writes nothing, so it is safe to call anywhere before a store transaction.
Everything per binding sits inside that binding's failure boundary; only
``PlotPublicationBlocked`` propagates, as in every handler.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from phenotypic.abc_.plotting import PlotImage, figure_backend_of
from phenotypic.abc_.plotting._store_formats import STORE_FORMATS, default_store_formats
from phenotypic.sdk_._image_figures import (
    StoredFigureBinding,
    StoredFigureFailure,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
)

from ._adapter import FigureAdapter
from ._backends import declared_figure_spec
from ._failures import _format_error
from ._output import normalize_plot_output
from ._store_formats import serialize_store_format
from ._writer import PlotPublicationBlocked, safe_path_component, unique_page_stems

logger = logging.getLogger(__name__)

_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def normalize_figure_error(error: BaseException) -> str:
    """Spell *error* for the store: ``"Type: message"``, addresses masked.

    A CPython object address differs every run, so a deterministic failure
    would otherwise produce different store bytes each time (spec §1).
    """
    return _ADDRESS.sub("0x…", _format_error(error))


def build_image_figures(pipeline: Any, image: Any) -> StoredFigures | None:
    """Render every ``PlotImage`` binding of *pipeline* for *image*.

    Args:
        pipeline: An ``ImagePipeline`` with normalized plot bindings.
        image: The image each binding's ``inspect`` receives.

    Returns:
        ``None`` when the pipeline has no ``PlotImage`` binding -- the store
        then carries no ``figures`` key at all. Otherwise the built value,
        which holds no bindings if every one failed.

    Raises:
        PlotPublicationBlocked: Never swallowed.
    """
    image_bindings = [b for b in pipeline.get_plots() if isinstance(b.plot, PlotImage)]
    if not image_bindings:
        return None
    built: list[StoredFigureBinding] = []
    failed: list[StoredFigureFailure] = []
    for binding in image_bindings:
        try:
            value = binding.plot.inspect(image, for_save=True)
            if value is None:
                raise TypeError("inspect() returned None; expected a figure or PlotOutput")
            pages = _build_pages(binding, value, failed)
        except PlotPublicationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - one figure never kills an image
            logger.warning("Plot %s failed while building its figure", binding.id, exc_info=exc)
            failed.append(StoredFigureFailure(binding.id, None, None, normalize_figure_error(exc)))
            continue
        if pages:
            built.append(StoredFigureBinding(
                binding_id=binding.id,
                plot_class=type(binding.plot).__name__,
                directory=safe_path_component(binding.id),
                pages=tuple(pages),
            ))
    return StoredFigures(bindings=tuple(built), failed=tuple(failed))


def _build_pages(
    binding: Any, value: Any, failed: list[StoredFigureFailure]
) -> list[StoredFigurePage]:
    """Serialize every page of one binding, recording page/format failures."""
    output = normalize_plot_output(value)
    if not output.pages:
        # Same as `inspect() -> None`: absence would read as "not configured".
        raise ValueError("inspect() returned a PlotOutput with no pages")
    try:
        spec = declared_figure_spec(binding.plot)
        stems = unique_page_stems([(page.key, page.key) for page in output.pages])
    except BaseException:
        for page in output.pages:
            FigureAdapter.close(page.figure)
        raise
    pages: list[StoredFigurePage] = []
    for page, stem in zip(output.pages, stems):
        try:
            built = _build_page(binding, page, stem, spec, failed)
        except PlotPublicationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - per-page best effort
            failed.append(StoredFigureFailure(
                binding.id, page.key, None, normalize_figure_error(exc)
            ))
            built = None
        finally:
            FigureAdapter.close(page.figure)
        if built is not None:
            pages.append(built)
    return pages


def _build_page(
    binding: Any,
    page: Any,
    stem: str,
    spec: Any,
    failed: list[StoredFigureFailure],
) -> StoredFigurePage | None:
    """One page: backend check, metadata check, then each declared format."""
    backend = figure_backend_of(page.figure)
    if backend is None:
        raise TypeError(
            "unsupported figure type "
            f"{type(page.figure).__module__}.{type(page.figure).__qualname__}"
        )
    # Normalised to exactly what a reader gets back, under the strictest
    # writer's rules: the measure-mode root has no `default=` and sorts keys,
    # and a browser rejects a root holding NaN. So a numpy scalar, an
    # unsortable key mix or a non-finite float refuses this page rather than
    # failing the store or making its root unreadable (spec §1).
    metadata = json.loads(
        json.dumps(dict(page.metadata), allow_nan=False, sort_keys=True)
    )
    formats = spec.store if spec is not None else default_store_formats(backend)
    files: list[StoredFigureFile] = []
    for fmt in formats:
        info = STORE_FORMATS[fmt]
        try:
            if backend not in info.backends:
                raise TypeError(f"a {backend} figure cannot be stored as {fmt}")
            data = serialize_store_format(
                fmt, page.figure, binding_id=binding.id, page_key=page.key
            )
        except PlotPublicationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - per-format best effort
            failed.append(StoredFigureFailure(
                binding.id, page.key, fmt, normalize_figure_error(exc)
            ))
            continue
        files.append(StoredFigureFile(fmt, info.media_type, f"{stem}{info.extension}", data))
    if not files:
        return None
    return StoredFigurePage(
        key=page.key, label=page.label, backend=backend,
        metadata=metadata, files=tuple(files),
    )


__all__ = ["build_image_figures", "normalize_figure_error"]
