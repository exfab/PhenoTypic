"""Per-image figures inside an OME-Zarr store (spec 2026-09-22 §1, §1a).

Storage-neutral: the caller decides directory names, filenames, formats and
media types; this module writes bytes where it is told, hashes them, and
describes them. It never imports a plotting library.

Figures live in one folder per run, ``figures/<date>-<pipeline hash>/``, and
the root descriptor is keyed by run. No write path here deletes or rewrites
another run's folder (§1a "never wiped").
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, TypeGuard, cast

from ._atomic_io import atomic_write_json

logger = logging.getLogger(__name__)

#: Same document every `tables/*` group uses.
_GROUP_DOCUMENT: dict[str, object] = {
    "zarr_format": 3,
    "node_type": "group",
    "attributes": {},
}

#: Characters of the pipeline sha256 in a run folder name (spec §1a).
RUN_HASH_LENGTH = 12

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _utc_now() -> datetime:
    """The one clock read behind every run date and initiation time."""
    return datetime.now(timezone.utc)


def utc_run_date() -> str:
    """Today's UTC calendar date as ``YYYY-MM-DD`` -- a run's ``{date}``."""
    return _utc_now().date().isoformat()


def is_run_date(value: object) -> TypeGuard[str]:
    """Whether *value* is a real calendar date spelled ``YYYY-MM-DD``."""
    if not isinstance(value, str):
        return False
    try:
        return _date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def is_initiation_timestamp(value: object) -> TypeGuard[str]:
    """Whether *value* is a UTC ISO-8601 timestamp with a trailing ``Z``,
    as :func:`mint_run_initiation` spells one."""
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class RunInitiation:
    """The initial CLI call of a run (spec §1a).

    Minted once per run and recorded -- in the run's processing state, or in
    ``job_metadata.json`` for a measure-mode SLURM invocation -- so every
    worker, stage and resume sees the same values.

    Args:
        date: The call's UTC date, ``YYYY-MM-DD``: the run folder's ``{date}``.
        at_utc: The call's UTC timestamp, ISO-8601 with a trailing ``Z``, or
            ``None`` for a run recorded before it was captured.
        pid: The CLI process's id, or ``None`` likewise.
    """

    date: str
    at_utc: str | None = None
    pid: int | None = None


def mint_run_initiation() -> RunInitiation:
    """Record this CLI call: date and timestamp from ONE instant, and its pid."""
    import os

    now = _utc_now()
    return RunInitiation(
        date=now.date().isoformat(),
        at_utc=now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        pid=os.getpid(),
    )


@dataclass(frozen=True)
class FigureRun:
    """Which run folder a store write belongs to (spec §1a).

    Args:
        date: The UTC date the run started, ``YYYY-MM-DD``.
        pipeline_sha256: The pipeline's full sha256 hex digest, as the
            provenance journal records it.
        initiated_at_utc: The initial CLI call's UTC timestamp, written into
            the run entry. ``None`` omits it -- as process-mode stores do, so
            same-day byte identity holds. Not part of :attr:`run_id`.
        initiated_pid: The initial CLI call's process id, likewise.

    Raises:
        ValueError: If the date or digest is malformed. Both become part of a
            path.
    """

    date: str
    pipeline_sha256: str
    initiated_at_utc: str | None = None
    initiated_pid: int | None = None

    def __post_init__(self) -> None:
        if not is_run_date(self.date):
            raise ValueError(f"run date must be YYYY-MM-DD; got {self.date!r}")
        if not isinstance(self.pipeline_sha256, str) or not _SHA256.fullmatch(
            self.pipeline_sha256
        ):
            raise ValueError(
                "pipeline_sha256 must be a lowercase 64-hex sha256; got "
                f"{self.pipeline_sha256!r}"
            )

    @property
    def run_id(self) -> str:
        """The run folder name, ``{date}-{pipeline hash}``."""
        return f"{self.date}-{self.pipeline_sha256[:RUN_HASH_LENGTH]}"


@dataclass(frozen=True)
class StoredFigureFile:
    """One rendering of one page. ``filename`` is final; no separators."""

    format: str
    media_type: str
    filename: str
    data: bytes


@dataclass(frozen=True)
class StoredFigurePage:
    """One page and the renderings that succeeded for it.

    ``metadata`` is strict, key-sorted JSON by the time it gets here: the
    builder round-trips it and refuses a page it cannot (spec §1).
    """

    key: str
    label: str | None
    backend: str
    metadata: Mapping[str, Any]
    files: tuple[StoredFigureFile, ...]


@dataclass(frozen=True)
class StoredFigureBinding:
    """One binding's pages. ``directory`` is the sanitized group name."""

    binding_id: str
    plot_class: str
    directory: str
    pages: tuple[StoredFigurePage, ...]


@dataclass(frozen=True)
class StoredFigureFailure:
    """A failure at the finest level available (spec §1)."""

    binding: str
    page: str | None
    format: str | None
    error: str


@dataclass(frozen=True)
class StoredFigures:
    """Everything one run folder of one image's store will say, in order.

    ``unavailable`` names the bindings that could not be drawn in this run and
    had nothing to keep (spec §3a); it is not a failure.
    """

    run: FigureRun
    bindings: tuple[StoredFigureBinding, ...]
    failed: tuple[StoredFigureFailure, ...]
    unavailable: tuple[str, ...] = ()


def figure_file_path(run_id: str, directory: str, filename: str) -> str:
    """Store-relative path of one figure file (spec §1a)."""
    from . import ngff_

    return f"{ngff_.FIGURES_GROUP}/{run_id}/{directory}/{filename}"


def split_figure_file_path(path: str) -> tuple[str, str, str]:
    """Invert :func:`figure_file_path`: ``(run_id, directory, filename)``.

    Raises:
        ValueError: If *path* is not laid out as this writer lays it out --
            which also refuses ``..`` and absolute paths.
    """
    from . import ngff_

    parts = PurePosixPath(path).parts
    if (
        len(parts) != 4
        or parts[0] != ngff_.FIGURES_GROUP
        or any(part in {".", ".."} for part in parts)
    ):
        raise ValueError(
            f"figure path {path!r} is not {ngff_.FIGURES_GROUP}/<run>/<binding>/<file>"
        )
    return parts[1], parts[2], parts[3]


def _ensure_group(directory: Path) -> None:
    """Create *directory* as a Zarr group, never rewriting an existing document.

    An existing ``zarr.json`` in a part may be a hard link into the live
    store; leaving it alone is how nothing is written through one.
    """
    from . import ngff_

    Path(ngff_.long_path(directory)).mkdir(parents=True, exist_ok=True)
    document = Path(ngff_.long_path(directory / ngff_.STORE_ROOT_JSON))
    if not document.exists():
        atomic_write_json(document, _GROUP_DOCUMENT)


def write_image_figures(
    store_part: Path, figures: StoredFigures
) -> dict[str, object]:
    """Write one run folder into an unpromoted part and return its descriptor.

    Args:
        store_part: An unpromoted ``*.ome.zarr.part`` directory in which this
            run's folder does not yet exist. Callers rewriting a promoted store
            remove the part's copy of that one folder first: its files are
            hard links into the live store, and writing through one would
            change the published bytes (``replace_image_tables``). A file
            that survived that removal is unlinked, never opened for writing.
        figures: The built figures of one run.

    Returns:
        ``{"figures": {"schema_version": 1, "runs": {run_id: entry}}}``, to
        merge with :func:`apply_image_figures_attributes`.
    """
    import os

    from . import ngff_

    run_id = figures.run.run_id
    group = Path(store_part) / ngff_.FIGURES_GROUP
    _ensure_group(group)
    _ensure_group(group / run_id)
    bindings: dict[str, object] = {}
    for binding in figures.bindings:
        directory = group / run_id / binding.directory
        _ensure_group(directory)
        pages = []
        for page in binding.pages:
            entries = []
            for stored in page.files:
                target = ngff_.long_path(directory / stored.filename)
                # A new inode, always: an existing path may be a hard link
                # into the live store, and exclusive create refuses to write
                # through one the unlink did not remove.
                try:
                    os.unlink(target)
                except FileNotFoundError:
                    pass
                with open(target, "xb") as handle:
                    handle.write(stored.data)
                entries.append({
                    "format": stored.format,
                    "media_type": stored.media_type,
                    "path": figure_file_path(run_id, binding.directory, stored.filename),
                    "sha256": hashlib.sha256(stored.data).hexdigest(),
                })
            pages.append({
                "key": page.key,
                "label": page.label,
                "backend": page.backend,
                "metadata": dict(page.metadata),
                "files": entries,
            })
        bindings[binding.binding_id] = {"class": binding.plot_class, "pages": pages}
    entry: dict[str, object] = {
        "date": figures.run.date,
        "pipeline_sha256": figures.run.pipeline_sha256,
    }
    # The CLI call whose run last wrote this folder; absent when not given
    # (process-mode stores, for same-day byte identity).
    if figures.run.initiated_at_utc is not None:
        entry["initiated_at_utc"] = figures.run.initiated_at_utc
    if figures.run.initiated_pid is not None:
        entry["initiated_pid"] = figures.run.initiated_pid
    entry |= {
        "bindings": bindings,
        "failed": [
            {"binding": f.binding, "page": f.page, "format": f.format, "error": f.error}
            for f in figures.failed
        ],
        "unavailable": list(figures.unavailable),
    }
    return _fragment({run_id: entry})


def carry_figure_runs(
    source_store: Path, store_part: Path, *, exclude: str | None = None
) -> dict[str, object] | None:
    """Carry every run folder of *source_store* but *exclude* into *store_part*.

    Used when a store is rewritten from scratch over an existing one (a
    ``save2zarr`` over an existing store: ``--restart``, a re-derived process
    store, Stage 3 over Stage 1): the other runs' files are hard-linked (or
    copied) across byte for byte, with their descriptor entries unchanged
    (spec §1a "never wiped").

    Each file is checked against its recorded sha256. A mismatch or a missing
    file is logged and carried as it is -- dropping it would wipe history, and
    the descriptor's sha256 still exposes it to any consumer that verifies.

    A descriptor whose ``schema_version`` this writer does not know is
    carried whole: every file under ``figures/`` and the descriptor as it is,
    *exclude* included. Relabelling a newer layout as this one would be a
    guess, and dropping it would wipe history, so the caller then writes no
    run into it (:func:`known_figures_schema`).

    Args:
        source_store: The store being replaced. Absent or unreadable carries
            nothing.
        store_part: The unpromoted part being written.
        exclude: The run folder the caller is about to write itself.

    Returns:
        A fragment of the carried runs, or ``None`` when there are none.
    """
    from . import ngff_

    try:
        descriptor = read_image_figures_descriptor(Path(source_store))
    except (FileNotFoundError, NotADirectoryError):
        return None
    except ValueError as exc:  # json.JSONDecodeError included
        logger.warning("Cannot carry figures from %s: %s", source_store, exc)
        return None
    if descriptor is None:
        return None
    if not known_figures_schema(descriptor):
        logger.warning(
            "Carrying the figures of %s untouched and adding no run: "
            "schema_version %r is not %r",
            source_store, descriptor.get("schema_version"), ngff_.FIGURES_SCHEMA_VERSION,
        )
        _carry_tree(
            Path(source_store) / ngff_.FIGURES_GROUP,
            Path(store_part) / ngff_.FIGURES_GROUP,
        )
        return {ngff_.PhenotypicAttr.FIGURES: descriptor}
    runs = descriptor.get("runs")
    if not isinstance(runs, dict):
        return None
    carried = {run_id: entry for run_id, entry in runs.items() if run_id != exclude}
    if not carried:
        return None
    source_root = Path(source_store)
    # The parent group first: a run with no files (every binding unavailable
    # or failed) never reaches _carry_file, and Zarr v3 has no implicit
    # groups, so figures/<run>/ alone would not open as a hierarchy.
    _ensure_group(Path(store_part) / ngff_.FIGURES_GROUP)
    for run_id, entry in carried.items():
        for binding in entry.get("bindings", {}).values():
            for page in binding.get("pages", []):
                for stored in page.get("files", []):
                    _carry_file(source_root, Path(store_part), run_id, stored)
        _ensure_group(Path(store_part) / ngff_.FIGURES_GROUP / run_id)
    return _fragment(carried)


def _carry_file(
    source_root: Path, store_part: Path, run_id: str, stored: Mapping[str, Any]
) -> None:
    """Link one descriptor-listed file into the part; warn, never raise."""
    from . import ngff_

    try:
        file_run, directory, _filename = split_figure_file_path(stored["path"])
        if file_run != run_id:
            raise ValueError(f"{stored['path']!r} is not in run folder {run_id!r}")
        source = source_root / stored["path"]
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != stored.get("sha256"):
            logger.warning(
                "Carrying %s although it does not match its recorded sha256",
                source,
            )
        _ensure_group(store_part / ngff_.FIGURES_GROUP)
        _ensure_group(store_part / ngff_.FIGURES_GROUP / run_id)
        _ensure_group(store_part / ngff_.FIGURES_GROUP / run_id / directory)
        _link_or_copy(source, store_part / stored["path"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not carry figure file %r: %s", stored.get("path"), exc)


def _carry_tree(source: Path, target: Path) -> None:
    """Link every file under *source* to the same place under *target*.

    For a layout this module cannot read: nothing is selected, so nothing is
    lost. Warns per file, never raises.
    """
    import os

    from . import ngff_

    root = ngff_.long_path(source)
    for directory, _subdirectories, filenames in os.walk(root):
        destination = Path(ngff_.long_path(target)) / os.path.relpath(directory, root)
        for filename in filenames:
            try:
                destination.mkdir(parents=True, exist_ok=True)
                _link_or_copy(Path(directory) / filename, destination / filename)
            except OSError as exc:
                logger.warning(
                    "Could not carry figure file %s: %s", Path(directory) / filename, exc
                )


def _link_or_copy(source: Path, target: Path) -> None:
    """Hard-link *source* at *target*, copying where links are refused."""
    import os
    import shutil

    from . import ngff_

    try:
        os.link(ngff_.long_path(source), ngff_.long_path(target))
    except OSError:
        shutil.copy2(ngff_.long_path(source), ngff_.long_path(target))


def _fragment(runs: dict[str, object]) -> dict[str, object]:
    from . import ngff_

    return {
        ngff_.PhenotypicAttr.FIGURES: {
            "schema_version": ngff_.FIGURES_SCHEMA_VERSION,
            "runs": runs,
        }
    }


def apply_image_figures_attributes(
    phenotypic: dict[str, object], fragment: dict[str, object] | None
) -> None:
    """Merge *fragment*'s runs into the root's ``figures`` key.

    A run in *fragment* replaces the entry of the same run id; every other run
    already present is kept. ``None`` changes nothing: a pipeline with no
    ``PlotImage`` binding adds no run and removes none (spec §1a).

    A descriptor whose ``schema_version`` this writer does not know is never
    merged with: already in the root, it stays untouched and nothing is
    added; arriving in *fragment* (carried whole) into a root with none, it
    is set as it is (:func:`known_figures_schema`).
    """
    from . import ngff_

    if fragment is None:
        return
    # Always a descriptor dict: _fragment's, or one carried whole.
    incoming = cast("dict[str, Any]", fragment[ngff_.PhenotypicAttr.FIGURES])
    current = phenotypic.get(ngff_.PhenotypicAttr.FIGURES)
    if not (known_figures_schema(current) and known_figures_schema(incoming)):
        if isinstance(current, dict):
            logger.warning(
                "Adding no figure run to a figures descriptor of schema_version "
                "%r; this writer knows %r",
                current.get("schema_version"), ngff_.FIGURES_SCHEMA_VERSION,
            )
            return
        phenotypic[ngff_.PhenotypicAttr.FIGURES] = incoming
        return
    runs = dict(current.get("runs", {})) if isinstance(current, dict) else {}
    runs.update(incoming["runs"])
    phenotypic[ngff_.PhenotypicAttr.FIGURES] = {
        "schema_version": incoming["schema_version"],
        "runs": runs,
    }


def known_figures_schema(descriptor: object) -> bool:
    """Whether a run may be added to *descriptor* -- ``True`` when there is none.

    ``False`` for a descriptor whose ``schema_version`` this writer does not
    know. Every write path then leaves that descriptor and its files as they
    are and adds no run: never relabelled, never dropped (spec §1a).
    """
    from . import ngff_

    return (
        not isinstance(descriptor, Mapping)
        or descriptor.get("schema_version") == ngff_.FIGURES_SCHEMA_VERSION
    )


def read_image_figures_descriptor(store_path: Path) -> dict[str, Any] | None:
    """Return a store's figures descriptor, or ``None`` when it has none.

    A root with no ``phenotypic`` block -- a third-party OME-Zarr store --
    has none, like a PhenoTypic store written without figures.

    Raises:
        FileNotFoundError: If the store has no root ``zarr.json``.
        json.JSONDecodeError: If the root is present but unparseable.
    """
    from . import ngff_

    phenotypic = ngff_.read_root_attributes(Path(store_path)).get(
        ngff_.PhenotypicAttr.ROOT
    )
    if not isinstance(phenotypic, dict):
        return None
    descriptor = phenotypic.get(ngff_.PhenotypicAttr.FIGURES)
    return descriptor if isinstance(descriptor, dict) else None


def latest_run_date(
    descriptor: Mapping[str, Any] | None, pipeline_sha256: str
) -> str | None:
    """The most recent ``date`` among the runs made by this pipeline, if any.

    ``--mode measure`` with the same pipeline as an earlier run reuses that
    run's folder, so its figures overwrite that run's (spec §1a, revision 14).

    Args:
        descriptor: A store's figures descriptor, or ``None``.
        pipeline_sha256: The pipeline's full sha256.

    Returns:
        ``YYYY-MM-DD``, or ``None`` when no run has this pipeline.
    """
    runs = descriptor.get("runs") if isinstance(descriptor, Mapping) else None
    if not isinstance(runs, Mapping):
        return None
    dates = [
        run["date"]
        for run in runs.values()
        if isinstance(run, Mapping)
        and run.get("pipeline_sha256") == pipeline_sha256
        # A malformed date would win the lexical max and name no folder.
        and is_run_date(run.get("date"))
    ]
    # ISO dates order lexically.
    return max(dates) if dates else None


def read_figure_run(store_path: Path, run_id: str) -> dict[str, Any] | None:
    """Return one run's descriptor entry, or ``None`` when the store has none.

    Raises:
        ValueError: If the descriptor's ``schema_version`` is not one this
            reader knows -- reading a newer layout as this one would publish
            a guess.
        FileNotFoundError: If the store has no root ``zarr.json``.
    """
    from . import ngff_

    descriptor = read_image_figures_descriptor(store_path)
    if descriptor is None:
        return None
    version = descriptor.get("schema_version")
    if version != ngff_.FIGURES_SCHEMA_VERSION:
        raise ValueError(
            f"figures schema_version {version!r} is not supported "
            f"(this reader knows {ngff_.FIGURES_SCHEMA_VERSION})"
        )
    runs = descriptor.get("runs")
    entry = runs.get(run_id) if isinstance(runs, dict) else None
    return entry if isinstance(entry, dict) else None


__all__ = [
    "RUN_HASH_LENGTH",
    "FigureRun",
    "RunInitiation",
    "StoredFigureBinding",
    "StoredFigureFailure",
    "StoredFigureFile",
    "StoredFigurePage",
    "StoredFigures",
    "apply_image_figures_attributes",
    "carry_figure_runs",
    "figure_file_path",
    "is_initiation_timestamp",
    "is_run_date",
    "known_figures_schema",
    "latest_run_date",
    "mint_run_initiation",
    "read_figure_run",
    "read_image_figures_descriptor",
    "split_figure_file_path",
    "utc_run_date",
    "write_image_figures",
]
