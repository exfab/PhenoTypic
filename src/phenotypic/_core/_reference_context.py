"""Per-image reference data for operations that read experiment metadata.

A :class:`ReferenceContext` holds one metadata table and the means to resolve
the reference images it names. Activating it (``with ctx:``) makes it visible
to every :class:`~phenotypic.abc_.RefMetadata` operation that runs inside the
block, at any nesting depth, without threading an argument through
``_operate``. The mechanism is a :class:`contextvars.ContextVar`, as in
``phenotypic._core._provenance``.

This module imports only the standard library at module level; polars and
pandas are imported inside the functions that need them.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from contextvars import ContextVar, Token
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import polars as pl

    from phenotypic._core._image import Image

__all__ = [
    "RefMetadataUnavailableError",
    "ReferenceContext",
    "ReferenceContextError",
    "ReferenceImageChangedError",
    "ReferenceImageError",
    "ReferenceLookupError",
    "ReferenceTableError",
]

LookupReason = Literal["unmatched", "null", "ambiguous", "self"]


class ReferenceContextError(ValueError):
    """Base class for every reference-metadata failure."""


class RefMetadataUnavailableError(ReferenceContextError):
    """A reference-metadata operation ran with no active ReferenceContext."""


class ReferenceTableError(ReferenceContextError):
    """The reference table is missing, unreadable, or lacks a needed column."""


class ReferenceLookupError(ReferenceContextError):
    """An image's rows cannot supply exactly one value for a column.

    Attributes:
        reason: ``"unmatched"`` (no row), ``"null"`` (only empty values),
            ``"ambiguous"`` (rows disagree), or ``"self"`` (an image names
            itself as its own reference).
        image_name: The image whose lookup failed.
        column: The column that failed, when one did.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: LookupReason,
        image_name: str,
        column: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason: LookupReason = reason
        self.image_name = image_name
        self.column = column


class ReferenceImageError(ReferenceContextError):
    """A reference image cannot be resolved, read, or matched to its target."""


class ReferenceImageChangedError(RuntimeError):
    """A reference file's bytes differ from the digest its context was planned with.

    Raised only by a context narrowed with ``planned_digests`` (the CLI's run
    plan). Deliberately not a :class:`ReferenceContextError`: the file changed
    under the run, so the image is not at fault, and the CLI turns this into
    its non-terminal ``ReferencePlanStaleError``.
    """


_ACTIVE: ContextVar["ReferenceContext | None"] = ContextVar(
    "phenotypic_reference_context", default=None
)
#: The activation tokens of the enclosing ``with`` blocks, innermost last. A
#: ContextVar rather than a list on the instance: each thread (and each asyncio
#: task) then unwinds only its own activations, even when one instance is
#: entered from several threads at once (GUI request threads).
_TOKENS: "ContextVar[tuple[Token, ...]]" = ContextVar(
    "phenotypic_reference_context_tokens", default=()
)

#: Loaded reference images, most recently used last. A worker processes many
#: frames of one plate against one blank, so a small cache removes almost all
#: re-reads while bounding memory (each entry is one full image; at --njobs 32
#: that is 64 images machine-wide). The lock is for the GUI, whose Werkzeug
#: server runs previews on threads.
_IMAGE_CACHE: "OrderedDict[tuple, tuple[Image, str | None]]" = OrderedDict()
_IMAGE_CACHE_SIZE = 2
_IMAGE_CACHE_LOCK = threading.Lock()


def _clear_image_cache() -> None:
    """Drop every cached reference image (tests and long-lived sessions)."""
    with _IMAGE_CACHE_LOCK:
        _IMAGE_CACHE.clear()


def reference_file_digest(path: Path) -> str:
    """SHA-256 identifying a reference image on disk.

    A file's bytes; for an OME-Zarr store directory, its root ``zarr.json``
    (hashing every chunk would cost more than the image read it identifies).
    """
    target = Path(path) / "zarr.json" if Path(path).is_dir() else Path(path)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _read_image(path: Path, read_kwargs: Mapping[str, Any]) -> "Image":
    """Read one reference image from disk (the cache's single read path)."""
    from phenotypic._core._image import Image

    return Image.imread(path, **dict(read_kwargs))


def _image_name_header() -> str:
    from phenotypic.schema import IMAGE

    return str(IMAGE.IMAGE_NAME)


def _dataset_header() -> str:
    from phenotypic.schema import EXPERIMENT

    return str(EXPERIMENT.DATASET)


def _read_table(source: Any) -> "tuple[pl.DataFrame, str | None]":
    """Read *source* as an all-string polars frame with canonical headers."""
    import pandas as pd
    import polars as pl

    from phenotypic.sdk_._metadata_helpers import normalize_metadata_columns

    digest: str | None = None
    if isinstance(source, pl.DataFrame):
        frame = source
    elif isinstance(source, pd.DataFrame):
        frame = pl.from_pandas(source)
    else:
        path = Path(source)
        if not path.is_file():
            raise ReferenceTableError(f"Reference metadata table not found: {path}")
        suffix = path.suffix.lower()
        if suffix not in (".csv", ".parquet"):
            raise ReferenceTableError(
                f"Reference metadata must be .csv or .parquet, got {path.name!r}"
            )
        try:
            if suffix == ".csv":
                # infer_schema=False: every value is a name, and inference would
                # turn the stem "000123" into the integer 123 (Review Focus 1).
                frame = pl.read_csv(path, infer_schema=False)
            else:
                frame = pl.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 -- any parse failure is the error
            raise ReferenceTableError(f"Cannot read reference metadata {path}: {exc}") from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        frame = normalize_metadata_columns(frame)
    except ValueError as exc:
        raise ReferenceTableError(f"Reference metadata headers conflict: {exc}") from exc
    frame = frame.with_columns(pl.all().cast(pl.String))
    # Strip every value, and make a blank one null: an Excel cell holding " "
    # names no image, and must fail as "null" rather than as "matches 0 files".
    frame = frame.with_columns([
        pl.when(pl.col(c).str.strip_chars().str.len_chars() > 0)
        .then(pl.col(c).str.strip_chars())
        .alias(c)
        for c in frame.columns
    ])
    if _image_name_header() not in frame.columns:
        raise ReferenceTableError(
            f"Reference metadata needs a {_image_name_header()} column "
            f"(the image file name without its extension); columns are {frame.columns}"
        )
    return frame, digest


class _RootIndex:
    """One directory listing of an image root, keyed for name and stem lookup.

    Keys pass through ``os.path.normcase``, so a case-insensitive Windows
    filesystem matches case-insensitively.
    """

    __slots__ = ("mtime_ns", "by_name", "by_stem")

    def __init__(self, mtime_ns: int) -> None:
        self.mtime_ns = mtime_ns
        self.by_name: dict[str, Path] = {}
        self.by_stem: dict[str, list[Path]] = {}


def _scan_image_root(root: Path, mtime_ns: int) -> _RootIndex:
    """List *root* once: image files by accepted suffix, and Zarr stores.

    ``DirEntry.is_file``/``is_dir`` reuse the directory read on Linux, so this
    costs one listing rather than a ``stat`` per entry. Sidecars (``.json``,
    ``.xmp``, ``.txt``) are not candidates.
    """
    from phenotypic.sdk_._io_constants import is_zarr_store_name, source_image_stem
    from phenotypic.sdk_.constants_ import IO

    suffixes = {suffix.lower() for suffix in IO.ACCEPTED_FILE_EXTENSIONS}
    index = _RootIndex(mtime_ns)
    with os.scandir(root) as entries:
        for entry in entries:
            if is_zarr_store_name(entry.name):
                accepted = entry.is_dir()
            else:
                accepted = Path(entry.name).suffix.lower() in suffixes and entry.is_file()
            if not accepted:
                continue
            path = root / entry.name
            index.by_name[os.path.normcase(entry.name)] = path
            index.by_stem.setdefault(os.path.normcase(source_image_stem(path)), []).append(path)
    for paths in index.by_stem.values():
        paths.sort()
    return index


class _SharedTable:
    """The parsed table and lookup indexes shared by a context and its narrowings."""

    __slots__ = ("table", "sha256", "indexes", "roots")

    def __init__(self, table: "pl.DataFrame", sha256: str | None) -> None:
        self.table = table
        self.sha256 = sha256
        self.indexes: dict[tuple, dict[tuple, dict[str, list[str]]]] = {}
        #: Image-root listings by root path, rebuilt when the root's mtime moves.
        self.roots: dict[str, _RootIndex] = {}


class ReferenceContext:
    """Per-image reference data that RefMetadata operations read during apply().

    Activate it around a pipeline call. Every operation inside the block that
    mixes in :class:`~phenotypic.abc_.RefMetadata` looks up its own columns
    for the image being processed, and may load the reference images those
    columns name (a media-blank frame, say).

    A context is per-process: worker processes build their own. Activation is
    per-thread, so one instance may be entered from several threads at once
    (each ``with`` block sees and restores only its own thread's state).

    Args:
        metadata: A ``.csv``/``.parquet`` path, or a pandas/polars frame. It must
            carry ``Metadata_ImageName`` (or the bare ``ImageName``). Values are
            read as strings.
        image_root: Directory that reference-image names resolve against. A bare
            stem must match exactly one file there; a full file name matches
            that file.
        images: Mapping of name to an in-memory ``Image`` or a path. Checked
            before ``image_root``; lets a notebook or doctest prototype without
            files on disk.
        dataset: Narrows lookups to ``Metadata_Dataset == dataset`` when the
            table has that column.
        read_kwargs: Keyword arguments for ``Image.imread`` when loading
            reference images (the CLI passes its input reader settings).

    Raises:
        ReferenceTableError: The table is missing, unreadable, has conflicting
            header spellings, or lacks ``Metadata_ImageName``. Raised here, at
            construction, so a bad table fails before the first image.

    Examples:
        Ask what an operation would see, without running it:

        >>> import pandas as pd
        >>> from phenotypic import ReferenceContext
        >>> layout = pd.DataFrame({
        ...     "Metadata_ImageName": ["plate1_t04", "plate1_t04"],
        ...     "Grid_RowNum": [1, 2],
        ...     "Metadata_BlankImage": ["plate1_t00", "plate1_t00"],
        ... })
        >>> ReferenceContext(layout).lookup("plate1_t04", ["Metadata_BlankImage"])
        {'Metadata_BlankImage': 'plate1_t00'}
    """

    def __init__(
        self,
        metadata: "str | Path | pd.DataFrame | pl.DataFrame",
        *,
        image_root: str | Path | None = None,
        images: Mapping[str, "Image | str | Path"] | None = None,
        dataset: str | None = None,
        read_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        table, digest = _read_table(metadata)
        self._shared = _SharedTable(table, digest)
        self.image_root: Path | None = Path(image_root) if image_root is not None else None
        self.images: dict[str, Any] = dict(images) if images is not None else {}
        self.dataset = dataset
        self.read_kwargs: dict[str, Any] = dict(read_kwargs or {})
        #: ``{resolved path: sha256}`` a reference file must still match when
        #: loaded; ``None`` checks nothing. Set only through :meth:`narrow`.
        self.planned_digests: dict[str, str] | None = None

    # ------------------------------------------------------------- activation
    def __enter__(self) -> "ReferenceContext":
        _TOKENS.set(_TOKENS.get() + (_ACTIVE.set(self),))
        return self

    def __exit__(self, *exc: object) -> None:
        tokens = _TOKENS.get()
        _TOKENS.set(tokens[:-1])
        _ACTIVE.reset(tokens[-1])

    @classmethod
    def current(cls) -> "ReferenceContext | None":
        """Return the innermost active context, or ``None`` outside any."""
        return _ACTIVE.get()

    # -------------------------------------------------------------- the table
    @property
    def table(self) -> "pl.DataFrame":
        """The parsed, header-normalized, all-string table (shared; do not mutate)."""
        return self._shared.table

    @property
    def columns(self) -> tuple[str, ...]:
        """The table's normalized column names."""
        return tuple(self._shared.table.columns)

    @property
    def table_sha256(self) -> str | None:
        """SHA-256 of the table file's bytes, or ``None`` for an in-memory frame."""
        return self._shared.sha256

    def _resolve_column(self, name: str) -> str | None:
        from phenotypic.sdk_._metadata_helpers import ensure_metadata_prefix

        available = self._shared.table.columns
        if name in available:
            return name
        prefixed = ensure_metadata_prefix(name)
        return prefixed if prefixed in available else None

    def has_column(self, name: str) -> bool:
        """Whether *name* (literal first, then its ``Metadata_`` spelling) is a column."""
        return self._resolve_column(name) is not None

    def _key_columns(self) -> tuple[str, ...]:
        if self.dataset is not None and _dataset_header() in self._shared.table.columns:
            return (_dataset_header(), _image_name_header())
        return (_image_name_header(),)

    def _index(self, keys: tuple[str, ...], columns: tuple[str, ...]) -> dict:
        cache_key = (keys, columns)
        index = self._shared.indexes.get(cache_key)
        if index is None:
            import polars as pl

            grouped = self._shared.table.group_by(list(keys)).agg(
                [pl.col(c).drop_nulls().unique(maintain_order=True) for c in columns]
            )
            index = {
                tuple(row[k] for k in keys): {c: list(row[c]) for c in columns}
                for row in grouped.iter_rows(named=True)
            }
            self._shared.indexes[cache_key] = index
        return index

    def lookup(self, image: "Image | str", columns: Sequence[str]) -> dict[str, str]:
        """Return one value per requested column for *image*.

        Args:
            image: The image (its ``name`` is used) or an image name.
            columns: Column names; each resolves literally first, then by its
                ``Metadata_`` spelling.

        Returns:
            ``{requested_name: value}``.

        Raises:
            ReferenceTableError: A requested column is not in the table.
            ReferenceLookupError: No row matches (``reason="unmatched"``), a
                column holds only empty values (``"null"``), or its rows
                disagree (``"ambiguous"``).
        """
        name = image if isinstance(image, str) else image.name
        resolved: list[str] = []
        for requested in columns:
            column = self._resolve_column(requested)
            if column is None:
                raise ReferenceTableError(
                    f"Reference metadata has no column {requested!r}; "
                    f"columns are {list(self.columns)}"
                )
            resolved.append(column)
        keys = self._key_columns()
        key = (self.dataset, name) if len(keys) == 2 else (name,)
        # Two spellings of one column resolve to one name, and a key column is
        # answered by the key itself; neither may reach the aggregation, where
        # a repeated name is a polars DuplicateError.
        aggregated = tuple(dict.fromkeys(c for c in resolved if c not in keys))
        entry = self._index(keys, aggregated).get(key)
        where = f" in dataset {self.dataset!r}" if len(keys) == 2 else ""
        if entry is None:
            raise ReferenceLookupError(
                f"No row in the reference metadata for image {name!r}{where}",
                reason="unmatched",
                image_name=name,
            )
        from_key = dict(zip(keys, key))
        values: dict[str, str] = {}
        for requested, column in zip(columns, resolved):
            if column in from_key:
                values[requested] = str(from_key[column])
                continue
            found = entry[column]
            if not found:
                raise ReferenceLookupError(
                    f"{column} is empty for image {name!r}{where}",
                    reason="null",
                    image_name=name,
                    column=column,
                )
            if len(found) > 1:
                raise ReferenceLookupError(
                    f"{column} disagrees across the rows for image {name!r}{where}: "
                    f"{sorted(found)!r}",
                    reason="ambiguous",
                    image_name=name,
                    column=column,
                )
            values[requested] = found[0]
        return values

    # ---------------------------------------------------- reference images
    def resolve_image(self, name: str) -> "Path | Image":
        """Return the in-memory image or the file that *name* refers to.

        Only image files (by accepted suffix) and Zarr stores directly inside
        ``image_root`` are candidates. The directory is listed once and the
        listing reused until its modification time changes.

        Raises:
            ReferenceImageError: No ``images`` entry and no ``image_root``; the
                name is a path rather than a file name or stem; or it matches
                zero or several images in ``image_root``.
        """
        if name in self.images:
            target = self.images[name]
            return Path(target) if isinstance(target, (str, Path)) else target
        if self.image_root is None:
            raise ReferenceImageError(
                f"Cannot resolve reference image {name!r}: the ReferenceContext has "
                f"no image_root and no images entry for it"
            )
        root = self.image_root
        if not name or name in (".", "..") or os.path.isabs(name) or "/" in name or "\\" in name:
            raise ReferenceImageError(
                f"Reference image {name!r} must be a file name or stem inside "
                f"image_root {root}, not a path"
            )
        index = self._root_index(root)
        wanted = os.path.normcase(name)
        exact = index.by_name.get(wanted)
        if exact is not None:
            return exact
        # Stems are what Image.imread names an image: "x.ome.zarr" -> "x".
        candidates = index.by_stem.get(wanted, [])
        if len(candidates) != 1:
            raise ReferenceImageError(
                f"Reference image {name!r} matches {len(candidates)} files in {root}: "
                f"{[c.name for c in candidates]}"
            )
        return candidates[0]

    def _root_index(self, root: Path) -> _RootIndex:
        try:
            mtime_ns = os.stat(root).st_mtime_ns
        except OSError as exc:
            raise ReferenceImageError(f"Cannot read image_root {root}: {exc}") from exc
        index = self._shared.roots.get(str(root))
        if index is None or index.mtime_ns != mtime_ns:
            index = _scan_image_root(root, mtime_ns)
            self._shared.roots[str(root)] = index
        return index

    def _load(self, name: str) -> "tuple[Image, str | None]":
        """Return the reference image and its digest, from one resolution.

        Raises:
            ReferenceImageChangedError: The context has ``planned_digests``
                and the file's digest is not the one planned for its path.
        """
        target = self.resolve_image(name)
        if not isinstance(target, Path):
            return target, None
        cached = self._read_cached(target)
        if self.planned_digests is not None:
            path = str(target.resolve())
            if self.planned_digests.get(path) != cached[1]:
                raise ReferenceImageChangedError(
                    f"Reference image {name!r} ({path}) changed since the run "
                    f"planned its references"
                )
        return cached

    def _read_cached(self, target: Path) -> "tuple[Image, str]":
        """Read *target* through the process cache, with its digest.

        The digest is cached with the pixels of the same load, under a key that
        changes when the file does (its stat, and a store's root ``zarr.json``
        stat). A file rewritten on disk is therefore read and hashed again, and
        the digest :meth:`_load` checks always describes the pixels returned.
        """
        stat = target.stat()
        # A store's directory entry does not change when its content is
        # rewritten; its root zarr.json (which the digest hashes) does.
        manifest = target / "zarr.json"
        manifest_stat = manifest.stat() if target.is_dir() and manifest.is_file() else None
        key = (
            str(target.resolve()),
            stat.st_mtime_ns,
            stat.st_size,
            (manifest_stat.st_mtime_ns, manifest_stat.st_size) if manifest_stat else None,
            json.dumps(self.read_kwargs, sort_keys=True, default=str),
        )
        with _IMAGE_CACHE_LOCK:
            cached = _IMAGE_CACHE.get(key)
            if cached is not None:
                _IMAGE_CACHE.move_to_end(key)
                return cached
        try:
            image = _read_image(target, self.read_kwargs)
        except Exception as exc:  # noqa: BLE001 -- surface as the reference failure
            raise ReferenceImageError(f"Cannot read reference image {target}: {exc}") from exc
        cached = (image, reference_file_digest(target))
        with _IMAGE_CACHE_LOCK:
            _IMAGE_CACHE[key] = cached
            while len(_IMAGE_CACHE) > _IMAGE_CACHE_SIZE:
                _IMAGE_CACHE.popitem(last=False)
        return cached

    def load_image(self, name: str) -> "Image":
        """Load the reference image *name* (cached per process; do not modify it)."""
        return self._load(name)[0]

    def reference_image_digest(self, name: str) -> str | None:
        """SHA-256 of the reference image file, or ``None`` for an in-memory image."""
        return self._load(name)[1]

    # ------------------------------------------------------------ derivation
    def narrow(
        self,
        *,
        dataset: str | None = None,
        image_root: str | Path | None = None,
        images: Mapping[str, Any] | None = None,
        planned_digests: Mapping[str, str] | None = None,
    ) -> "ReferenceContext":
        """Return a context sharing this table, its indexes and its root listings.

        Each argument left ``None`` keeps this context's value.

        Args:
            dataset: Narrow lookups to this dataset.
            image_root: Resolve reference-image names against this directory.
            images: Name-to-image (or path) entries checked before
                ``image_root``.
            planned_digests: ``{resolved path: sha256}`` every reference file
                loaded through the clone must still match (the CLI's run plan);
                a file absent from it never matches. A load that does not
                raises :class:`ReferenceImageChangedError`.
        """
        clone = object.__new__(ReferenceContext)
        clone._shared = self._shared
        clone.dataset = dataset if dataset is not None else self.dataset
        clone.image_root = Path(image_root) if image_root is not None else self.image_root
        clone.images = dict(images) if images is not None else dict(self.images)
        clone.read_kwargs = dict(self.read_kwargs)
        clone.planned_digests = (
            dict(planned_digests)
            if planned_digests is not None
            else None if self.planned_digests is None else dict(self.planned_digests)
        )
        return clone

    def __repr__(self) -> str:
        return (
            f"ReferenceContext(rows={self._shared.table.height}, dataset={self.dataset!r}, "
            f"image_root={str(self.image_root) if self.image_root else None!r})"
        )
