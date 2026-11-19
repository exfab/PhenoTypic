"""The ``MeasureEmbedding`` ABC: measurers that embed image regions as vectors."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Final, Union

import numpy as np
import pandas as pd
from pydantic import field_validator, model_validator

from phenotypic.schema import GRID, OBJECT
from phenotypic.sdk_.exceptions_ import OperationFailedError
from phenotypic.sdk_._table_runs import CROPS_REGION, check_grid_name, grid_region
from phenotypic.sdk_.funcs_ import validate_measure_integrity
from phenotypic.sdk_.typing_ import EmbeddingRegions, OperationField, TuneSpec

from ._measure_features import MeasureFeatures

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic.sdk_._embedding_tables import EmbeddingPayload

#: The key columns of a grid table, in table order (spec D1). The first is the
#: entry's ``key``.
_GRID_KEY_COLUMNS: Final[tuple[GRID, ...]] = (
    GRID.ROW_MAJOR_IDX,
    GRID.ROW_NUM,
    GRID.COL_NUM,
    GRID.ROW_INTERVAL_START,
    GRID.ROW_INTERVAL_END,
    GRID.COL_INTERVAL_START,
    GRID.COL_INTERVAL_END,
)

#: Descriptor fields :meth:`MeasureEmbedding.entry_descriptor` must return.
_ENTRY_DESCRIPTOR_FIELDS: Final[frozenset[str]] = frozenset(
    {"model", "dim", "dtype", "pooling"}
)


def is_out_of_memory(exc: BaseException) -> bool:
    """Whether *exc* is a host or GPU out-of-memory error (review M7).

    ``MemoryError``, or torch's ``OutOfMemoryError`` (``torch.OutOfMemoryError``,
    and ``torch.cuda.OutOfMemoryError`` on a torch that has only that).
    torch is looked up in ``sys.modules``, never imported: an error raised
    by torch means torch is loaded already. An embedder is the step most
    likely to run out of memory, and running out is the node's state, not
    the image's, so these are re-raised unwrapped: the CLI's failure tracker
    then records no terminal failure, and the next run retries the image.
    """
    import sys

    if isinstance(exc, MemoryError):
        return True
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    for owner in (torch, getattr(torch, "cuda", None)):
        oom = getattr(owner, "OutOfMemoryError", None)
        if isinstance(oom, type) and isinstance(exc, oom):
            return True
    return False


class GridCellOutsideImageError(ValueError):
    """A grid cell is empty or not inside the image (#281 spec E3a).

    The cell is never clipped: a rig's grid that does not fit an image means
    the image is not the plate the grid was drawn for, so that image fails.
    ``MeasureEmbedding.measure`` wraps it in an ``OperationFailedError``
    whose ``err_type`` is this class.
    """

    def __init__(
        self, row: int, col: int, rows: tuple[int, int], cols: tuple[int, int],
        shape: tuple[int, int],
    ) -> None:
        self.row, self.col, self.rows, self.cols, self.shape = row, col, rows, cols, shape
        super().__init__(
            f"grid cell ({row}, {col}) spans rows {rows[0]}:{rows[1]} and columns "
            f"{cols[0]}:{cols[1]}, which is not inside the image; the image is "
            f"{shape[0]} x {shape[1]}"
        )

    def __reduce__(self):
        """Rebuild from the constructor's arguments (pickling across workers)."""
        return (type(self), (self.row, self.col, self.rows, self.cols, self.shape))


class MeasureEmbedding(MeasureFeatures, ABC):
    """Embed detected colonies, or the cells of a fixed grid, as feature vectors.

    An embedder crops image regions and turns each crop into one vector with
    a model (a DINOv2 backbone, say). Unlike every other measurer its output
    is not a set of scalar columns in the measurement table: each embedder
    writes its own table, ``embeddings/<model>/<region>/table.parquet``,
    inside the run folder of the pipeline that produced it. The region is
    ``crops`` for colonies, ``grid`` for a grid, and ``grid-<grid_name>``
    for a named grid.

    **Which runs which.** :meth:`ImagePipeline.measure` does **not** run
    embedders -- its frame never carries an ``embedding`` column.
    :meth:`ImagePipeline.embed` runs them and returns one
    :class:`~phenotypic.sdk_._embedding_tables.EmbeddingPayload` per
    embedder. Calling :meth:`measure` on an embedder directly returns its
    table's frame, for notebook use.

    **Regions.** ``regions="objects"`` embeds each detected colony's bounding
    box and needs a detection to have run (:attr:`requires_objects`);
    ``regions="grid"`` embeds each cell of ``grid``, a
    :class:`~phenotypic.grid.ManualGridFinder` whose pixel edges are set per
    rig, and needs no objects. A whole-image embedding is a 1 x 1 grid.
    Naming a grid (``grid_name="wells"``) lets one pipeline embed several
    grids with one model, each in its own table.

    To implement one, subclass this and define three methods:

    - :meth:`_embed_boxes` turns ``(N, 4)`` boxes into ``(N, D)`` vectors;
    - :meth:`model_token` names the model folder (``"dinov2-vitb14"``);
    - :meth:`entry_descriptor` describes the model and the vectors.

    Override :meth:`_embed_object_boxes` as well when a colony's crop needs
    its own pixels (a background mask, a masked pool), and list the fields
    that name the model in ``_leading_fields`` to put them before
    ``regions``, ``grid`` and ``grid_name`` in the field order.

    Args:
        regions: ``"objects"`` to embed detected colonies, ``"grid"`` to
            embed the cells of ``grid``.
        grid: The grid whose cells are embedded, in pixels of the image the
            measurer sees. Required with ``regions="grid"`` and refused
            otherwise. A cell outside the image fails that image.
        grid_name: A name for ``grid``, so that one pipeline can embed
            several grids with one model: the table's region becomes
            ``grid-<grid_name>`` instead of ``grid``. A lowercase slug of 1
            to 32 characters (``a-z``, ``0-9`` and ``_``, starting with a
            letter or digit). ``None`` keeps the region ``grid``. Refused
            with ``regions="objects"``.

    Examples:
        A toy embedder whose vector is each box's height and width:

        >>> import numpy as np
        >>> from phenotypic.abc_ import MeasureEmbedding
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> class BoxShape(MeasureEmbedding):
        ...     def model_token(self):
        ...         return "boxshape-v1"
        ...     def entry_descriptor(self):
        ...         return {"model": {"family": "boxshape"}, "dim": 2,
        ...                 "dtype": "float16", "pooling": "none"}
        ...     def _embed_boxes(self, image, boxes):
        ...         return np.stack([boxes[:, 2] - boxes[:, 0],
        ...                          boxes[:, 3] - boxes[:, 1]], axis=1)
        >>> plate = load_synth_yeast_plate()
        >>> frame = BoxShape().measure(plate)
        >>> list(frame.columns)
        ['Object_Label', 'embedding']
        >>> len(frame) == plate.num_objects, frame["embedding"][0].dtype
        (True, dtype('float16'))
    """

    regions: Annotated[EmbeddingRegions, TuneSpec(tunable=False)] = "objects"
    grid: Annotated[  # type: ignore[valid-type]
        Union[OperationField, None], TuneSpec(tunable=False)
    ] = None
    grid_name: Annotated[Union[str, None], TuneSpec(tunable=False)] = None

    #: Subclass fields placed before ``regions``, ``grid`` and ``grid_name``, in this order:
    #: the ones that name the model (``size``, ``version``, ...), which a
    #: reader of a pipeline file looks for first. Empty keeps pydantic's
    #: order, the ABC's fields first.
    _leading_fields: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Move ``_leading_fields`` to the front of the subclass's field order.

        Pydantic puts inherited fields first, so without this every
        embedder's own model fields would follow ``regions``, ``grid`` and
        ``grid_name``.
        The same forced-rebuild pattern as the field-append mixins
        (``sdk_/mixin/_input_layer_mixin.py``); ``TuneSpec`` metadata and the
        docstring descriptions survive it.
        """
        super().__pydantic_init_subclass__(**kwargs)
        leading = cls._leading_fields
        fields = cls.__pydantic_fields__
        unknown = [name for name in leading if name not in fields]
        if unknown:
            raise TypeError(f"{cls.__name__}._leading_fields names no field {unknown}")
        order = [*leading, *(name for name in fields if name not in leading)]
        if list(fields) != order:
            reordered = {name: fields[name] for name in order}
            fields.clear()
            fields.update(reordered)
            cls.model_rebuild(force=True)

    @field_validator("grid_name")
    @classmethod
    def _grid_name_is_a_slug(cls, value: str | None) -> str | None:
        """A grid name is one lowercase path segment (:func:`check_grid_name`)."""
        return value if value is None else check_grid_name(value)

    @model_validator(mode="after")
    def _check_grid_matches_regions(self) -> "MeasureEmbedding":
        """``grid`` is required with ``regions="grid"``; it and ``grid_name`` are refused otherwise."""
        if self.regions == "grid":
            from phenotypic.grid import ManualGridFinder

            if self.grid is None:
                raise ValueError(
                    "regions='grid' needs a grid: pass grid=ManualGridFinder("
                    "row_edges=[...], col_edges=[...]) with the rig's cell edges"
                )
            if not isinstance(self.grid, ManualGridFinder):
                raise ValueError(
                    "grid must be a ManualGridFinder (fixed pixel edges), got "
                    f"{type(self.grid).__name__}"
                )
        elif self.grid is not None:
            raise ValueError(
                "grid is only used with regions='grid'; drop it, or set "
                "regions='grid'"
            )
        elif self.grid_name is not None:
            raise ValueError(
                "grid_name is only used with regions='grid'; drop it, or set "
                "regions='grid' with a grid"
            )
        return self

    # ------------------------------------------------------------------ #
    # The subclass contract
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _embed_boxes(self, image: Image, boxes: np.ndarray) -> np.ndarray:
        """Embed each box of *image*.

        Args:
            image: The image being measured.
            boxes: ``(N, 4)`` int64 ``(r0, c0, r1, c1)`` boxes, end-exclusive,
                inside the image. ``N >= 1``.

        Returns:
            An ``(N, D)`` float array, ``D == entry_descriptor()["dim"]``.
        """

    def _embed_object_boxes(
        self, image: Image, boxes: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Embed each detected object's box, knowing which object it is.

        ``regions="objects"`` calls this; ``regions="grid"`` calls
        :meth:`_embed_boxes`. The default ignores *labels*. Override it when
        a crop needs the colony's own pixels, ``image.objmap[...] == label``.

        Args:
            image: The image being measured.
            boxes: As :meth:`_embed_boxes`.
            labels: ``(N,)`` int64, the ``Object_Label`` of each box.

        Returns:
            As :meth:`_embed_boxes`.
        """
        return self._embed_boxes(image, boxes)

    @abstractmethod
    def model_token(self) -> str:
        """The model folder name, e.g. ``"dinov2-vitb14"``: one path segment."""

    @abstractmethod
    def entry_descriptor(self) -> dict:
        """The model's part of the table's entry (spec D3).

        Returns:
            A dict with at least ``model`` (``{family, hf_id, revision,
            patch_size}``, plus ``device``, the device that ran, when the
            embedder can resolve one), ``dim`` (the vector length), ``dtype``
            (the stored type) and ``pooling`` (the pooling mode).
        """

    # ------------------------------------------------------------------ #
    # Derived properties
    # ------------------------------------------------------------------ #

    @property
    def requires_objects(self) -> bool:
        """Whether this embedder needs detected objects: ``regions == "objects"``."""
        return self.regions == "objects"

    @property
    def region(self) -> str:
        """The table's region folder: ``crops``, ``grid`` or ``grid-<grid_name>``."""
        if self.regions == "objects":
            return CROPS_REGION
        return grid_region(self.grid_name)

    def payload_key(self) -> str:
        """``"<model_token()>/<region>"``: the entry's key in its run folder.

        Two embedders of one pipeline must not share it
        (``ImagePipeline`` refuses them), so two grids of one model need
        different ``grid_name``.
        """
        return f"{self.model_token()}/{self.region}"

    def _dim(self) -> int:
        descriptor = self.entry_descriptor()
        missing = _ENTRY_DESCRIPTOR_FIELDS - set(descriptor)
        if missing:
            raise ValueError(
                f"{type(self).__name__}.entry_descriptor() lacks {sorted(missing)}"
            )
        return int(descriptor["dim"])

    # ------------------------------------------------------------------ #
    # Measuring
    # ------------------------------------------------------------------ #

    @validate_measure_integrity()
    def measure(self, image: Image, include_meta: bool = False) -> pd.DataFrame:
        """Embed *image*'s regions and return the embedding table's frame.

        This is the notebook entry point. In a pipeline,
        :meth:`ImagePipeline.measure` skips embedders and
        :meth:`ImagePipeline.embed` runs them, through :meth:`payload`.

        Args:
            image: The image to embed. With ``regions="objects"`` it must
                carry the objmap of a detection; an image with no objects
                gives an empty table.
            include_meta: Not supported: an embedding table is never joined
                to metadata (spec E5). Must be ``False``.

        Returns:
            For ``regions="objects"``, ``Object_Label`` and ``embedding``, one
            row per object in label order. For ``regions="grid"``, the
            ``Grid_*`` key columns (``RowMajorIdx``, ``RowNum``, ``ColNum``
            and the row and column intervals) and ``embedding``, one row per
            cell in row-major order. Each ``embedding`` cell is a 1-D
            ``numpy.float16`` array of length ``dim``.

        Raises:
            ValueError: If *include_meta* is true.
            OperationFailedError: If embedding fails, a grid cell falls
                outside the image, or the vectors have the wrong shape.
            MemoryError: Re-raised unwrapped, as is torch's
                ``OutOfMemoryError`` (:func:`is_out_of_memory`): running out
                of memory is the node's state, not the image's.
        """
        if include_meta:
            raise ValueError("an embedding table carries no metadata (include_meta)")
        try:
            return self._operate(image)
        except Exception as exc:
            if is_out_of_memory(exc):
                raise
            raise OperationFailedError(
                operation=type(self).__name__,
                image_name=image.name,
                err_type=type(exc),
                message=str(exc),
            ) from exc

    def _operate(self, image: Image) -> pd.DataFrame:  # type: ignore[override]
        labels: np.ndarray | None = None
        if self.regions == "objects":
            keys, boxes = self._object_boxes(image)
            labels = keys[str(OBJECT.LABEL)].to_numpy(dtype=np.int64)
        else:
            keys, boxes = self._grid_cells(image)
        vectors = self._embed_vectors(image, boxes, labels)
        frame = keys.reset_index(drop=True)
        frame["embedding"] = list(vectors)
        return frame

    @staticmethod
    def _object_boxes(image: Image) -> tuple[pd.DataFrame, np.ndarray]:
        """``Object_Label`` per object, and its end-exclusive bounding box."""
        from skimage.measure import regionprops

        props = regionprops(image.objmap[:])
        labels = np.array([prop.label for prop in props], dtype=np.int64)
        boxes = np.array([prop.bbox for prop in props], dtype=np.int64).reshape(-1, 4)
        return pd.DataFrame({str(OBJECT.LABEL): labels}), boxes

    def _grid_edges(self, image: Image) -> tuple[np.ndarray, np.ndarray]:
        assert self.grid is not None
        return (
            np.asarray(self.grid.get_row_edges(image), dtype=np.int64),
            np.asarray(self.grid.get_col_edges(image), dtype=np.int64),
        )

    def _grid_cells(self, image: Image) -> tuple[pd.DataFrame, np.ndarray]:
        """The ``Grid_*`` keys of every cell, row-major, and its box.

        Raises:
            GridCellOutsideImageError: Naming the first cell that is empty or
                not inside the image, and the image size. It is never clipped
                (spec E3a).
        """
        row_edges, col_edges = self._grid_edges(image)
        height, width = (int(n) for n in image.shape[:2])
        ncols = len(col_edges) - 1
        rows: list[tuple[int, ...]] = []
        for row in range(len(row_edges) - 1):
            r0, r1 = int(row_edges[row]), int(row_edges[row + 1])
            for col in range(ncols):
                c0, c1 = int(col_edges[col]), int(col_edges[col + 1])
                if not (0 <= r0 < r1 <= height and 0 <= c0 < c1 <= width):
                    raise GridCellOutsideImageError(
                        row, col, (r0, r1), (c0, c1), (height, width)
                    )
                rows.append((row * ncols + col, row, col, r0, r1, c0, c1))
        keys = pd.DataFrame(
            np.array(rows, dtype=np.int64).reshape(-1, len(_GRID_KEY_COLUMNS)),
            columns=[str(column) for column in _GRID_KEY_COLUMNS],
        )
        boxes = keys[
            [
                str(GRID.ROW_INTERVAL_START),
                str(GRID.COL_INTERVAL_START),
                str(GRID.ROW_INTERVAL_END),
                str(GRID.COL_INTERVAL_END),
            ]
        ].to_numpy(dtype=np.int64)
        return keys, boxes

    def _embed_vectors(
        self, image: Image, boxes: np.ndarray, labels: np.ndarray | None = None
    ) -> np.ndarray:
        """Embed *boxes*, checked for shape and cast to ``float16``.

        :meth:`_embed_object_boxes` when *labels* are given (objects), else
        :meth:`_embed_boxes` (grid cells).
        """
        dim = self._dim()
        if len(boxes) == 0:
            return np.zeros((0, dim), dtype=np.float16)
        if labels is None:
            vectors = np.asarray(self._embed_boxes(image, boxes))
        else:
            vectors = np.asarray(self._embed_object_boxes(image, boxes, labels))
        if vectors.ndim != 2 or vectors.shape != (len(boxes), dim):
            raise ValueError(
                f"_embed_boxes returned shape {vectors.shape} for {len(boxes)} "
                f"boxes; expected ({len(boxes)}, dim={dim})"
            )
        stored = vectors.astype(np.float16)
        if not np.isfinite(stored).all():
            raise ValueError(
                "embedding vectors are not finite in float16 (NaN, inf, or "
                "beyond +/-65504)"
            )
        return stored

    # ------------------------------------------------------------------ #
    # The pipeline seam
    # ------------------------------------------------------------------ #

    def _params_sha256(self) -> str:
        """sha256 of this embedder's parameters, canonical JSON (``producer.params_sha256``).

        ``grid_name`` is left out while it is ``None``, so an embedder that
        names no grid has the digest it had before the field existed, and a
        run continued across the upgrade keeps one descriptor per key
        (#281 named grid regions). A name is part of the digest.
        """
        params = self.model_dump(mode="json")
        if params.get("grid_name") is None:
            params.pop("grid_name", None)
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def payload(self, image: Image) -> EmbeddingPayload:
        """Embed *image* and describe the table, for a store writer (spec D3).

        :meth:`ImagePipeline.embed` calls this for each embedder. The
        descriptor's ``binds`` are digests of the image the vectors were
        computed from, so a writer can tell whether the store it writes still
        holds that image (spec D7): for crops the objmap, for a grid the
        level-0 array of the series read and the grid edges.

        Args:
            image: The image to embed.

        Returns:
            The payload, keyed :meth:`payload_key`.
        """
        from phenotypic.sdk_ import ngff_
        from phenotypic.sdk_._embedding_tables import EmbeddingPayload
        from phenotypic.sdk_._measurement_tables import array_digest, objmap_digest

        frame = self.measure(image)
        series = ngff_.primary_series(image._series_names())
        binds: dict[str, Any]
        if self.regions == "objects":
            key = str(OBJECT.LABEL)
            binds = {
                "objmap_path": ngff_.objmap_path(series),
                "objmap_sha256": objmap_digest(image.objmap[:]),
            }
        else:
            key = str(GRID.ROW_MAJOR_IDX)
            row_edges, col_edges = self._grid_edges(image)
            edges = json.dumps(
                {"row_edges": row_edges.tolist(), "col_edges": col_edges.tolist()},
                separators=(",", ":"),
            )
            binds = {
                "array_sha256": array_digest(image._store_series_array(series)),
                "grid_sha256": hashlib.sha256(edges.encode("utf-8")).hexdigest(),
            }
        descriptor = {
            **self.entry_descriptor(),
            "key": key,
            "input": {"series": series, "level": 0},
            "producer": {
                "class": type(self).__name__,
                "params_sha256": self._params_sha256(),
            },
            "binds": binds,
        }
        return EmbeddingPayload(key=self.payload_key(), frame=frame, descriptor=descriptor)
