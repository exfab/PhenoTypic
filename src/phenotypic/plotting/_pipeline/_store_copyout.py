"""Copy a promoted store's figures out to deliverables/plots (spec §3 step 3).

The store is the single source: this never renders a PNG. The one thing it
produces rather than copies is an HTML page for each stored ``plotly-json``,
so a Plotly figure stays browsable in deliverables. Best-effort: every failure
is recorded to ``.failures.jsonl``; only a refused guard propagates.
"""
from __future__ import annotations

import functools
import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Mapping

from phenotypic.abc_.plotting._store_formats import STORE_FORMATS
from phenotypic.sdk_ import CommitGuard
from phenotypic.sdk_._file_locking import exclusive_path_lock
from phenotypic.sdk_._image_figures import read_figure_run, split_figure_file_path
from phenotypic.sdk_.ngff_ import FIGURES_GROUP

from ._adapter import FigureAdapter
from ._failures import _format_error, record_plot_failure
from ._writer import (
    PlotPublicationBlocked,
    _atomic_write,
    _commit_manifest,
    _guarded_commit,
    _require_plot_publication,
    safe_path_component,
)

logger = logging.getLogger(__name__)

#: `<unresolved>` is the coordinator's spelling for "class not knowable here".
_UNRESOLVED = "<unresolved>"

#: Every suffix a page can have in deliverables, for leftover removal only.
_DELIVERABLE_SUFFIXES = (".plotly.json", ".html", ".png")


def publish_store_figures(
    store_path: Path,
    plots_base: Path,
    *,
    run_id: str,
    dataset: str,
    image_stem: str,
    plot_classes: Mapping[str, str] | None = None,
    publication_guard: Callable[[], bool] | None = None,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Republish one run folder of a promoted store at today's deliverables paths.

    Only *run_id*'s folder is published: the deliverables tree belongs to the
    run that wrote it (spec §1a). A store with no folder for it publishes
    nothing.

    Args:
        store_path: A promoted ``*.ome.zarr`` store.
        plots_base: Resolved ``deliverables/plots`` directory.
        run_id: This run's folder name, ``{date}-{pipeline hash}``.
        dataset: Dataset name (unsanitized; hashed into the output stem).
        image_stem: Image stem (unsanitized).
        plot_classes: ``binding_id -> class name`` from the pipeline. The
            descriptor records no class for a binding that failed outright.
        publication_guard: Optional GUI compare-and-set predicate.
        commit_guard: Optional commit guard.

    Raises:
        PlotPublicationBlocked: If a guard refuses. Never swallowed.
    """
    classes = dict(plot_classes or {})

    def _record(
        binding_id: str,
        error: BaseException | str,
        *,
        page: str | None = None,
        fmt: str | None = None,
    ) -> None:
        record_plot_failure(
            plots_base, binding_id=binding_id,
            plot_class=classes.get(binding_id, _UNRESOLVED), lifecycle="image",
            error=error, dataset=dataset, image_stem=image_stem, page=page, fmt=fmt,
        )

    try:
        # A newer schema_version raises here: reading a newer layout as this
        # one would publish a guess.
        run = read_figure_run(store_path, run_id)
        if run is None:
            return
        # Before the first record, too: a refused guard means "do not touch
        # this tree", and the failure log lives in it (_coordinator F1).
        _require_plot_publication(publication_guard)
        from ._coordinator import _image_output_stem

        output_stem = _image_output_stem(dataset, image_stem)
        failures = list(run.get("failed", []))
        bindings = dict(run.get("bindings", {}))
        # The pipeline's class wins; the descriptor's is the fallback for a
        # binding the pipeline no longer carries. Filled before the failure
        # records below, so a partial failure of a published binding is not
        # recorded as `<unresolved>` when the store knows its class.
        for binding_id, binding in bindings.items():
            classes.setdefault(binding_id, binding.get("class", _UNRESOLVED))
    except PlotPublicationBlocked:
        raise
    except Exception as exc:  # noqa: BLE001 - copy-out is best-effort
        logger.warning("Copy-out could not read %s", store_path, exc_info=exc)
        # The read failed before the guard was asked; ask it before writing
        # the record into the tree it protects.
        _require_plot_publication(publication_guard)
        _record("<store>", exc)
        return
    for failure in failures:
        try:
            _record(failure["binding"], failure["error"],
                    page=failure.get("page"), fmt=failure.get("format"))
        except Exception as exc:  # noqa: BLE001 - a malformed entry is one record
            _record("<store>", exc)
    # A page that failed outright is absent from `pages` but still part of
    # its binding's output, so a binding with only such pages is published
    # too. A binding-level failure (`page: null`) stays a record only.
    page_failures = [
        f for f in failures
        if isinstance(f, dict)
        and isinstance(f.get("binding"), str)
        and isinstance(f.get("page"), str)
    ]
    targets = dict.fromkeys([*bindings, *(f["binding"] for f in page_failures)])
    for binding_id in targets:
        try:
            _publish_binding(
                Path(store_path), plots_base, binding_id,
                bindings.get(binding_id, {}),
                [f for f in page_failures if f["binding"] == binding_id],
                plot_class=classes.get(binding_id, _UNRESOLVED),
                dataset=dataset, output_stem=output_stem,
                record=functools.partial(_record, binding_id),
                publication_guard=publication_guard, commit_guard=commit_guard,
            )
        except PlotPublicationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - copy-out is best-effort
            logger.warning("Copy-out of plot %s failed", binding_id, exc_info=exc)
            _record(binding_id, exc)


def _publish_binding(
    store: Path,
    plots_base: Path,
    binding_id: str,
    binding: dict[str, Any],
    page_failures: list[dict[str, Any]],
    *,
    plot_class: str,
    dataset: str,
    output_stem: str,
    record: Callable[..., None],
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> None:
    """Publish one binding as a manifest directory mirroring the store.

    Spec 2026-09-30 §3. Every page lands at its store-relative
    ``<plot folder>/<file>`` inside the image's directory; a flat version 1
    page lands in that directory itself.
    A page is identified by ``(key, plot)``: one key may appear in two plots.

    *record* is already bound to this binding's id.
    """
    pages = binding.get("pages", [])
    published_ids = {(p["key"], p.get("plot")) for p in pages}
    failed_only: dict[tuple[str, str | None], str] = {}
    for failure in page_failures:
        page_id = (failure["page"], failure.get("plot"))
        if page_id not in published_ids:
            failed_only.setdefault(page_id, str(failure.get("error")))
    directory = (
        plots_base / safe_path_component(binding_id)
        / safe_path_component(dataset) / output_stem
    )
    _require_plot_publication(publication_guard)
    directory.mkdir(parents=True, exist_ok=True)
    # Same lock `publish_plot_output` takes for a manifest directory, so two
    # writers of one image's directory cannot interleave pages and manifest.
    with exclusive_path_lock(directory / ".publication.lock"):
        _require_plot_publication(publication_guard)
        published, failed = _publish_pages(
            store, plots_base, directory, pages, page_failures,
            record=record,
            publication_guard=publication_guard, commit_guard=commit_guard,
        )
        # The store records no label for a page that stored nothing.
        failed += [
            {"key": key, "plot": plot, "label": None, "error": error}
            for (key, plot), error in failed_only.items()
        ]
        # A capability, not an outcome; see the note in `_writer`.
        renderers: dict[str, str] = {}
        if any(p["backend"] == "plotly" for p in published):
            renderers["html"] = "available"
        if any(p["backend"] == "matplotlib" for p in published):
            renderers["png"] = "available"
        _commit_manifest(
            directory,
            {
                "schema_version": 3, "plot_id": binding_id,
                "class": plot_class,
                "renderers": renderers, "pages": published, "failed": failed,
            },
            publication_guard=publication_guard, commit_guard=commit_guard,
        )


def _publish_pages(
    store: Path,
    plots_base: Path,
    directory: Path,
    pages: list[dict[str, Any]],
    page_failures: list[dict[str, Any]],
    *,
    record: Callable[..., None],
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Copy every page's stored files; return manifest pages and failures.

    Names come from the stored paths, never from ``page["plot"]``: the store
    already chose them (decision P1), and ``plot`` is the unsanitized name.
    Manifest ``files`` values are relative to *directory*.
    """
    published: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    figures_root = (store / FIGURES_GROUP).resolve()
    for page in pages:
        files: dict[str, str] = {}
        errors: list[BaseException] = []
        created: Path | None = None
        for entry in page["files"]:
            try:
                data = _read_stored_file(store, figures_root, entry)
                _run, _binding, plot_directory, name = split_figure_file_path(entry["path"])
                stem = name[: -len(STORE_FORMATS[entry["format"]].extension)]
                page_directory = (
                    directory if plot_directory is None else directory / plot_directory
                )
                if not page_directory.exists():
                    # The writer's contract: the guard is asked immediately
                    # before any directory is created (plan review I5).
                    _require_plot_publication(publication_guard)
                    page_directory.mkdir()
                    created = page_directory

                def _copy(dest: Path) -> None:
                    dest.write_bytes(data)

                _atomic_write(
                    page_directory / name, _copy,
                    publication_guard=publication_guard, commit_guard=commit_guard,
                )
                files[entry["format"]] = _relative(plot_directory, name)
            except PlotPublicationBlocked:
                _discard_page(directory, files)
                raise
            except Exception as exc:  # noqa: BLE001 - per-file best effort
                errors.append(exc)
                record(exc, page=page["key"], fmt=entry.get("format"))
                continue
            if entry["format"] != "plotly-json":
                continue
            # Its own step: the stored JSON was verified and copied, so a
            # failure here is the deliverable rendering, not the store.
            try:
                html = _write_html_from_json(
                    data, page_directory, stem, plots_base,
                    publication_guard=publication_guard, commit_guard=commit_guard,
                )
                files["html"] = _relative(plot_directory, html)
            except PlotPublicationBlocked:
                _discard_page(directory, files)
                raise
            except Exception as exc:  # noqa: BLE001 - per-rendering best effort
                errors.append(exc)
                record(exc, page=page["key"], fmt="html")
        if not files:
            if created is not None and not any(created.iterdir()):
                # As in the writer: no empty plot folder for a page that
                # copied nothing. Only one this page created, so a folder a
                # sibling published into stays.
                created.rmdir()
            failed.append({"key": page["key"], "plot": page.get("plot"),
                           "label": page["label"],
                           "error": "no stored file could be copied out"})
            continue
        # Same rule as the writer: only a page this pass published has its
        # leftover renderings removed, inside its own folder. A copied file
        # set `page_directory` and `stem`; one page's formats share both.
        _remove_leftovers(page_directory, stem, {Path(v).name for v in files.values()},
                          publication_guard=publication_guard, commit_guard=commit_guard)
        entry_out: dict[str, Any] = {
            "key": page["key"], "plot": page.get("plot"), "label": page["label"],
            "files": files,
            "backend": "matplotlib" if page["backend"] == "mpl" else "plotly",
            "metadata": page.get("metadata", {}),
        }
        # Today's meaning: this page published, and these failed too -- the
        # store's own failures, then this pass's, spelled as `.failures.jsonl`.
        page_id = (page["key"], page.get("plot"))
        partial = [f["error"] for f in page_failures
                   if (f.get("page"), f.get("plot")) == page_id]
        partial += [_format_error(exc) for exc in errors]
        if partial:
            entry_out["partial"] = partial
        published.append(entry_out)
    return published, failed


def _relative(plot_directory: str | None, name: str) -> str:
    """A manifest ``files`` value: *name* inside its plot folder, if any."""
    return name if plot_directory is None else f"{plot_directory}/{name}"


def _read_stored_file(
    store: Path, figures_root: Path, entry: Mapping[str, Any]
) -> bytes:
    """Return one stored file's bytes, as its descriptor entry vouches for them.

    Refused: a path that resolves outside ``figures/``, or bytes whose sha256
    differs from the descriptor's.
    """
    source = (store / entry["path"]).resolve()
    if not source.is_relative_to(figures_root):
        raise ValueError(f"stored path {entry['path']!r} is outside {FIGURES_GROUP}/")
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(
            f"stored {entry['path']} does not match its sha256 "
            f"(descriptor {entry['sha256'][:12]}…, file {digest[:12]}…)"
        )
    return data


def _discard_page(directory: Path, files: Mapping[str, str]) -> None:
    """S11, as in `_render_page`: a refused guard must not leave half a page
    behind asserting what this pass no longer owns."""
    for written in files.values():
        (directory / written).unlink(missing_ok=True)


def _write_html_from_json(
    data: bytes,
    directory: Path,
    stem: str,
    plots_base: Path,
    *,
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> str:
    """Render the deliverables HTML for one stored Plotly JSON; return its name."""
    import plotly.io as pio

    from ._backends import ensure_plotlyjs_bundle, plotlyjs_src_for

    figure = pio.from_json(data.decode("utf-8"))
    src = plotlyjs_src_for(directory, ensure_plotlyjs_bundle(plots_base))
    name = f"{stem}.html"
    _atomic_write(
        directory / name,
        lambda dest: FigureAdapter.save_html(figure, dest, plotlyjs_src=src),
        publication_guard=publication_guard, commit_guard=commit_guard,
    )
    return name


def _remove_leftovers(
    directory: Path,
    stem: str,
    written: set[str],
    *,
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> None:
    """Remove *stem*'s renderings this pass did not write (today's stale rule).

    Only for a page this pass published: a page that published nothing keeps
    its previous files whole rather than half of them.
    """
    for suffix in _DELIVERABLE_SUFFIXES:
        path = directory / f"{stem}{suffix}"
        if path.name in written or not path.exists():
            continue
        with _guarded_commit(publication_guard, commit_guard):
            path.unlink(missing_ok=True)


__all__ = ["publish_store_figures"]
