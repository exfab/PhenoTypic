"""Copy a promoted store's figures out to deliverables/plots (spec §3 step 3).

The store is the single source: this never renders a PNG. The one thing it
produces rather than copies is an HTML page for each stored ``plotly-json``,
so a Plotly figure stays browsable in deliverables. Best-effort: every failure
is recorded to ``.failures.jsonl``; only a refused guard propagates.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Mapping

from phenotypic.abc_.plotting._store_formats import STORE_FORMATS
from phenotypic.sdk_ import CommitGuard
from phenotypic.sdk_._file_locking import exclusive_path_lock
from phenotypic.sdk_._image_figures import read_image_figures_descriptor
from phenotypic.sdk_.ngff_ import FIGURES_GROUP, FIGURES_SCHEMA_VERSION

from ._adapter import FigureAdapter
from ._failures import _format_error, record_plot_failure
from ._writer import (
    PlotPublicationBlocked,
    _atomic_write,
    _commit_manifest,
    _guarded_commit,
    _require_plot_publication,
    safe_path_component,
    unique_page_stems,
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
    dataset: str,
    image_stem: str,
    plot_classes: Mapping[str, str] | None = None,
    publication_guard: Callable[[], bool] | None = None,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Republish one promoted store's figures at today's deliverables paths.

    Args:
        store_path: A promoted ``*.ome.zarr`` store.
        plots_base: Resolved ``deliverables/plots`` directory.
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
        descriptor = read_image_figures_descriptor(store_path)
        if descriptor is None:
            return
        version = descriptor.get("schema_version")
        if version != FIGURES_SCHEMA_VERSION:
            # Reading a newer layout as this one would publish a guess.
            raise ValueError(
                f"figures schema_version {version!r} is not supported "
                f"(this reader knows {FIGURES_SCHEMA_VERSION})"
            )
        # Before the first record, too: a refused guard means "do not touch
        # this tree", and the failure log lives in it (_coordinator F1).
        _require_plot_publication(publication_guard)
        from ._coordinator import _image_output_stem

        output_stem = _image_output_stem(dataset, image_stem)
        failures = list(descriptor.get("failed", []))
        bindings = dict(descriptor.get("bindings", {}))
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
    targets = list(bindings)
    for failure in page_failures:
        if failure["binding"] not in targets:
            targets.append(failure["binding"])
    for binding_id in targets:
        try:
            _publish_binding(
                Path(store_path), plots_base, binding_id,
                bindings.get(binding_id, {}),
                [f for f in page_failures if f["binding"] == binding_id],
                plot_class=classes.get(binding_id, _UNRESOLVED),
                dataset=dataset, output_stem=output_stem, record=_record,
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
    """Publish one binding flat (a lone ``default`` page) or as a manifest dir.

    The layout is decided over every page the binding produced, failed ones
    included, as the retired writer decided it over the whole ``PlotOutput``:
    a failure must not flip a multi-page binding to the flat layout.
    """
    pages = binding.get("pages", [])
    published_keys = [p["key"] for p in pages]
    failed_only: dict[str, str] = {}
    for failure in page_failures:
        if failure["page"] not in published_keys:
            failed_only.setdefault(failure["page"], str(failure.get("error")))
    universe = published_keys + list(failed_only)
    flat = universe == ["default"]
    if flat and not pages:
        return  # a page that published nothing keeps its previous files
    base = plots_base / safe_path_component(binding_id) / safe_path_component(dataset)
    directory = base if flat else base / output_stem
    stems = (
        [output_stem]
        if flat
        else unique_page_stems(
            [(p["key"], p["label"] or p["key"]) for p in pages]
            + [(key, key) for key in failed_only]
        )[: len(pages)]
    )
    _require_plot_publication(publication_guard)
    directory.mkdir(parents=True, exist_ok=True)
    if flat:
        _publish_pages(store, plots_base, directory, pages, stems, page_failures,
                       record=record, binding_id=binding_id,
                       publication_guard=publication_guard, commit_guard=commit_guard)
        return
    # Same lock `publish_plot_output` takes for a manifest directory, so two
    # writers of one image's directory cannot interleave pages and manifest.
    with exclusive_path_lock(directory / ".publication.lock"):
        _require_plot_publication(publication_guard)
        published, failed = _publish_pages(
            store, plots_base, directory, pages, stems, page_failures,
            record=record, binding_id=binding_id,
            publication_guard=publication_guard, commit_guard=commit_guard,
        )
        # The store records no label for a page that stored nothing.
        failed += [
            {"key": key, "label": None, "error": error}
            for key, error in failed_only.items()
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
                "schema_version": 2, "plot_id": binding_id,
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
    stems: list[str],
    page_failures: list[dict[str, Any]],
    *,
    record: Callable[..., None],
    binding_id: str,
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Copy every page's stored files; return manifest pages and failures."""
    published: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    figures_root = (store / FIGURES_GROUP).resolve()
    for page, stem in zip(pages, stems):
        files: dict[str, str] = {}
        errors: list[BaseException] = []
        for entry in page["files"]:
            try:
                source = (store / entry["path"]).resolve()
                if not source.is_relative_to(figures_root):
                    raise ValueError(
                        f"stored path {entry['path']!r} is outside {FIGURES_GROUP}/"
                    )
                data = source.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest != entry["sha256"]:
                    raise ValueError(
                        f"stored {entry['path']} does not match its sha256 "
                        f"(descriptor {entry['sha256'][:12]}…, file {digest[:12]}…)"
                    )
                name = f"{stem}{STORE_FORMATS[entry['format']].extension}"

                def _copy(dest: Path) -> None:
                    dest.write_bytes(data)

                _atomic_write(
                    directory / name, _copy,
                    publication_guard=publication_guard, commit_guard=commit_guard,
                )
                files[entry["format"]] = name
            except PlotPublicationBlocked:
                _discard_page(directory, files)
                raise
            except Exception as exc:  # noqa: BLE001 - per-file best effort
                errors.append(exc)
                record(binding_id, exc, page=page["key"], fmt=entry.get("format"))
                continue
            if entry["format"] != "plotly-json":
                continue
            # Its own step: the stored JSON was verified and copied, so a
            # failure here is the deliverable rendering, not the store.
            try:
                files["html"] = _write_html_from_json(
                    data, directory, stem, plots_base,
                    publication_guard=publication_guard, commit_guard=commit_guard,
                )
            except PlotPublicationBlocked:
                _discard_page(directory, files)
                raise
            except Exception as exc:  # noqa: BLE001 - per-rendering best effort
                errors.append(exc)
                record(binding_id, exc, page=page["key"], fmt="html")
        if not files:
            failed.append({"key": page["key"], "label": page["label"],
                           "error": "no stored file could be copied out"})
            continue
        # Same rule as the writer: only a page this pass published has its
        # leftover renderings removed.
        _remove_leftovers(directory, stem, set(files.values()),
                          publication_guard=publication_guard, commit_guard=commit_guard)
        entry_out: dict[str, Any] = {
            "key": page["key"], "label": page["label"], "files": files,
            "backend": "matplotlib" if page["backend"] == "mpl" else "plotly",
            "metadata": page.get("metadata", {}),
        }
        # Today's meaning: this page published, and these failed too -- the
        # store's own failures, then this pass's, spelled as `.failures.jsonl`.
        partial = [f["error"] for f in page_failures if f.get("page") == page["key"]]
        partial += [_format_error(exc) for exc in errors]
        if partial:
            entry_out["partial"] = partial
        published.append(entry_out)
    return published, failed


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
