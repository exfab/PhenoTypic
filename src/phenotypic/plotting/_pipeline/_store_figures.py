"""Build one image's figures in memory for the store (spec §3 step 1).

Writes nothing, so it is safe to call anywhere before a store transaction.
Everything per binding sits inside that binding's failure boundary; only
``PlotPublicationBlocked`` propagates, as in every handler.

What is built is one run folder's worth (spec §1a): the caller names the run.
A figure that cannot be drawn in this process (``FigureInputUnavailable``) is
kept from the same run's folder, or listed as ``unavailable`` (spec §3a).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from phenotypic.abc_.plotting import FigureInputUnavailable, PlotImage, figure_backend_of
from phenotypic.abc_.plotting._store_formats import STORE_FORMATS, default_store_formats
from phenotypic.sdk_._image_figures import (
    FigureRun,
    RunInitiation,
    StoredFigureBinding,
    StoredFigureFailure,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
    read_figure_run,
    split_figure_file_path,
    utc_run_date,
)
from phenotypic.sdk_.ngff_ import FIGURES_GROUP

from ._adapter import FigureAdapter
from ._backends import declared_figure_spec
from ._failures import _format_error
from ._output import normalize_plot_output
from ._store_copyout import _read_stored_file
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


def figure_run_for(
    image: Any,
    *,
    initiation: RunInitiation | None = None,
    date: str | None = None,
    pipeline_sha256: str | None = None,
) -> FigureRun | None:
    """The run folder *image*'s figures belong to (spec §1a).

    Args:
        image: The image being written. When *pipeline_sha256* is not given,
            its provenance journal's current application supplies it -- the
            digest the journal records for this run.
        initiation: The run's initial CLI call. Its date is the folder's
            ``{date}``, and its timestamp and pid go into the run entry.
            ``None`` writes neither -- how a process-mode store omits them.
        date: A date that wins over *initiation*'s (measure mode reusing a
            folder). With neither, today in UTC.
        pipeline_sha256: The pipeline's sha256, when the caller has it.

    Returns:
        The run, or ``None`` when no pipeline digest is known.
    """
    if pipeline_sha256 is None:
        journal = getattr(getattr(image, "_metadata", None), "provenance_journal", None)
        applications = journal.get("applications") if isinstance(journal, dict) else None
        pipeline = applications[-1].get("pipeline") if applications else None
        pipeline_sha256 = pipeline.get("sha256") if isinstance(pipeline, dict) else None
    if pipeline_sha256 is None:
        return None
    return FigureRun(
        date=date or (initiation.date if initiation is not None else None) or utc_run_date(),
        pipeline_sha256=pipeline_sha256,
        initiated_at_utc=initiation.at_utc if initiation is not None else None,
        initiated_pid=initiation.pid if initiation is not None else None,
    )


def build_image_figures(
    pipeline: Any,
    image: Any,
    *,
    run: FigureRun | None,
    keep_from: Path | None = None,
) -> StoredFigures | None:
    """Render every ``PlotImage`` binding of *pipeline* for *image*.

    A binding whose ``inspect()`` raises ``FigureInputUnavailable`` cannot be
    drawn here (spec §3a). It keeps its entry from *keep_from*'s folder for
    **this same run**, if that folder holds it, and is otherwise listed as
    ``unavailable`` -- never copied from another run's folder.

    Args:
        pipeline: An ``ImagePipeline`` with normalized plot bindings.
        image: The image each binding's ``inspect`` receives.
        run: The run folder the figures belong to (spec §1a). Needed only
            when the pipeline has an image binding.
        keep_from: The store being replaced, whose ``run`` folder may already
            hold a §3a binding (staged Stage 1, or a same-run rerun).

    Returns:
        ``None`` when the pipeline has no ``PlotImage`` binding -- the store
        then gains no run folder. Otherwise the built value, which holds no
        bindings if every one failed.

    Raises:
        PlotPublicationBlocked: Never swallowed.
        ValueError: If there is an image binding but no *run*.
    """
    image_bindings = [b for b in pipeline.get_plots() if isinstance(b.plot, PlotImage)]
    if not image_bindings:
        return None
    if run is None:
        raise ValueError(
            "image figures need the run folder they belong to (spec §1a); "
            "no pipeline digest was known for this image"
        )
    built: list[StoredFigureBinding] = []
    failed: list[StoredFigureFailure] = []
    unavailable: list[str] = []
    keeper = _SameRunKeeper(keep_from, run)
    for binding in image_bindings:
        try:
            try:
                value = binding.plot.inspect(image, for_save=True)
            except FigureInputUnavailable:
                if not keeper.keep(binding, built, failed):
                    unavailable.append(binding.id)
                continue
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
    return StoredFigures(
        run=run, bindings=tuple(built), failed=tuple(failed),
        unavailable=tuple(unavailable),
    )


class _SameRunKeeper:
    """Keep a §3a binding from the same run's folder, reading it at most once."""

    def __init__(self, store: Path | None, run: FigureRun) -> None:
        self._store = None if store is None else Path(store)
        self._run = run
        self._entry: dict[str, Any] | None = None
        self._read_done = False

    def _run_entry(self) -> dict[str, Any] | None:
        if not self._read_done:
            try:
                self._entry = read_figure_run(self._store, self._run.run_id)
            except FileNotFoundError:
                self._entry = None
            self._read_done = True
        return self._entry

    def keep(
        self,
        binding: Any,
        built: list[StoredFigureBinding],
        failed: list[StoredFigureFailure],
    ) -> bool:
        """Append *binding*'s entry and failures from this run's folder.

        All or nothing: nothing is appended unless every file verified. Its
        stored failures come too, a binding-level one included -- that is the
        truth about a figure the earlier process could not draw.

        Returns:
            ``False`` when the folder names the binding nowhere.

        Raises:
            ValueError: If the stored entry was drawn by another class, is not
                laid out as this writer lays it out, or a file does not match
                its sha256 -- a binding-level failure, never a silent keep.
        """
        if self._store is None:
            return False
        entry = self._run_entry()
        if entry is None:
            return False
        stored = entry.get("bindings", {}).get(binding.id)
        failures = [
            StoredFigureFailure(f["binding"], f["page"], f["format"], f["error"])
            for f in entry.get("failed", [])
            if f.get("binding") == binding.id
        ]
        if stored is None and not failures:
            return False
        if stored is not None:
            built.append(self._kept_binding(binding, stored))
        failed.extend(failures)
        return True

    def _kept_binding(self, binding: Any, stored: dict[str, Any]) -> StoredFigureBinding:
        """Rebuild one entry so ``write_image_figures`` rewrites it unchanged."""
        assert self._store is not None
        plot_class = type(binding.plot).__name__
        if stored.get("class") != plot_class:
            # Same id, different producer: its figure is not this plot's.
            raise ValueError(
                f"stored figure {binding.id!r} was drawn by {stored.get('class')!r}, "
                f"not {plot_class}"
            )
        figures_root = (self._store / FIGURES_GROUP).resolve()
        directories: set[str] = set()
        pages: list[StoredFigurePage] = []
        for page in stored["pages"]:
            files: list[StoredFigureFile] = []
            for entry in page["files"]:
                run_id, directory, filename = split_figure_file_path(entry["path"])
                if run_id != self._run.run_id:
                    raise ValueError(
                        f"{entry['path']!r} is not in run folder {self._run.run_id!r}"
                    )
                directories.add(directory)
                data = _read_stored_file(self._store, figures_root, entry)
                files.append(StoredFigureFile(
                    entry["format"], entry["media_type"], filename, data
                ))
            pages.append(StoredFigurePage(
                key=page["key"], label=page["label"], backend=page["backend"],
                metadata=page["metadata"], files=tuple(files),
            ))
        if len(directories) != 1:
            raise ValueError(
                f"stored figure {binding.id!r} does not live in one directory: "
                f"{sorted(directories)}"
            )
        return StoredFigureBinding(
            binding_id=binding.id, plot_class=plot_class,
            directory=directories.pop(), pages=tuple(pages),
        )


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


__all__ = ["build_image_figures", "figure_run_for", "normalize_figure_error"]
