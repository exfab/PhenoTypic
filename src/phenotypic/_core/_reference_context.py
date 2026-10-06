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


_ACTIVE: ContextVar["ReferenceContext | None"] = ContextVar(
    "phenotypic_reference_context", default=None
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
        try:
            if suffix == ".csv":
                # infer_schema=False: every value is a name, and inference would
                # turn the stem "000123" into the integer 123 (Review Focus 1).
                frame = pl.read_csv(path, infer_schema=False)
            elif suffix == ".parquet":
                frame = pl.read_parquet(path)
            else:
                raise ReferenceTableError(
                    f"Reference metadata must be .csv or .parquet, got {path.name!r}"
                )
        except ReferenceTableError:
            raise
        except Exception as exc:  # noqa: BLE001 -- any parse failure is the error
            raise ReferenceTableError(f"Cannot read reference metadata {path}: {exc}") from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        frame = normalize_metadata_columns(frame)
    except ValueError as exc:
        raise ReferenceTableError(f"Reference metadata headers conflict: {exc}") from exc
    frame = frame.with_columns(pl.all().cast(pl.String))
    if _image_name_header() not in frame.columns:
        raise ReferenceTableError(
            f"Reference metadata needs a {_image_name_header()} column "
            f"(the image file name without its extension); columns are {frame.columns}"
        )
    return frame, digest


class _SharedTable:
    """The parsed table and lookup indexes shared by a context and its narrowings."""

    __slots__ = ("table", "sha256", "indexes")

    def __init__(self, table: "pl.DataFrame", sha256: str | None) -> None:
        self.table = table
        self.sha256 = sha256
        self.indexes: dict[tuple, dict[tuple, dict[str, list[str]]]] = {}


class ReferenceContext:
    """Per-image reference data that RefMetadata operations read during apply().

    Activate it around a pipeline call. Every operation inside the block that
    mixes in :class:`~phenotypic.abc_.RefMetadata` looks up its own columns
    for the image being processed, and may load the reference images those
    columns name (a media-blank frame, say).

    A context is per-process and per-thread: worker processes build their own,
    and one instance must not be entered concurrently from two threads.

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
        self._tokens: list[Token] = []

    # ------------------------------------------------------------- activation
    def __enter__(self) -> "ReferenceContext":
        self._tokens.append(_ACTIVE.set(self))
        return self

    def __exit__(self, *exc: object) -> None:
        _ACTIVE.reset(self._tokens.pop())

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
        entry = self._index(keys, tuple(resolved)).get(key)
        where = f" in dataset {self.dataset!r}" if len(keys) == 2 else ""
        if entry is None:
            raise ReferenceLookupError(
                f"No row in the reference metadata for image {name!r}{where}",
                reason="unmatched",
                image_name=name,
            )
        values: dict[str, str] = {}
        for requested, column in zip(columns, resolved):
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

        Raises:
            ReferenceImageError: No ``images`` entry and no ``image_root``, or
                the name matches zero or several files in ``image_root``.
        """
        if name in self.images:
            target = self.images[name]
            return Path(target) if isinstance(target, (str, Path)) else target
        if self.image_root is None:
            raise ReferenceImageError(
                f"Cannot resolve reference image {name!r}: the ReferenceContext has "
                f"no image_root and no images entry for it"
            )
        from phenotypic.sdk_._io_constants import is_zarr_store_name, source_image_stem

        def is_image(p: Path) -> bool:
            return p.is_file() or (p.is_dir() and is_zarr_store_name(p))

        root = self.image_root
        exact = root / name
        if is_image(exact):
            return exact
        # source_image_stem is what Image.imread names an image: "x.ome.zarr" -> "x".
        candidates = sorted(
            p for p in root.iterdir() if is_image(p) and source_image_stem(p) == name
        )
        if len(candidates) != 1:
            raise ReferenceImageError(
                f"Reference image {name!r} matches {len(candidates)} files in {root}: "
                f"{[c.name for c in candidates]}"
            )
        return candidates[0]

    def _load(self, name: str) -> "tuple[Image, str | None]":
        target = self.resolve_image(name)
        if not isinstance(target, Path):
            return target, None
        stat = target.stat()
        key = (
            str(target.resolve()),
            stat.st_mtime_ns,
            stat.st_size,
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
    ) -> "ReferenceContext":
        """Return a context sharing this table and its indexes.

        Each argument left ``None`` keeps this context's value.
        """
        clone = object.__new__(ReferenceContext)
        clone._shared = self._shared
        clone.dataset = dataset if dataset is not None else self.dataset
        clone.image_root = Path(image_root) if image_root is not None else self.image_root
        clone.images = dict(images) if images is not None else dict(self.images)
        clone.read_kwargs = dict(self.read_kwargs)
        clone._tokens = []
        return clone

    def __repr__(self) -> str:
        return (
            f"ReferenceContext(rows={self._shared.table.height}, dataset={self.dataset!r}, "
            f"image_root={str(self.image_root) if self.image_root else None!r})"
        )
