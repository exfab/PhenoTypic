"""The ``MeasureEmbedding`` ABC: measurers that embed image regions as vectors."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Final, Literal, Protocol, Union

import numpy as np
import pandas as pd
from pydantic import field_validator, model_validator

from phenotypic.schema import GRID, OBJECT
from phenotypic.sdk_.exceptions_ import OperationFailedError
from phenotypic.sdk_._table_runs import (
    OBJECTS_REGION,
    GRID_REGION,
    check_region_name,
    embedding_key,
    embedding_region,
)
from phenotypic.sdk_.funcs_ import validate_measure_integrity
from phenotypic.sdk_.typing_ import (
    EmbeddingLayer,
    EmbeddingRegions,
    OperationField,
    TuneSpec,
)

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

#: Fields added to ``MeasureEmbedding`` after embedders were in use (#281
#: named embedders / layer), and the default that leaves
#: :meth:`MeasureEmbedding._params_sha256` where it was before each existed.
_DIGEST_NEUTRAL_DEFAULTS: Final[dict[str, object]] = {"name": None, "layer": "rgb"}

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


@dataclass(frozen=True, eq=False)
class BackboneInputs:
    """What an encoder runs on: one image's regions, prepared on the CPU (#282).

    ``kind`` says what ``arrays`` hold:

    - ``"crops"``: one ``(N, px, px, 3)`` uint8 array, each region's crop
      already resized to the backbone's input size;
    - ``"tiles"``: one ``(3, h, w)`` float32 array per tile of the plate, the
      HF processor's own ``pixel_values`` (each side a multiple of the patch),
      so every encoder normalises a tile exactly as the inline path does;
    - ``"boxes"``: the ``(N, 4)`` int64 ``(r0, c0, r1, c1)`` boxes, followed,
      for ``regions="objects"``, by the ``(N,)`` int64 ``Object_Label`` of
      each box. For an embedder with no backbone; inline only.

    ``pooling`` is what an encoder does with the backbone's tokens:
    ``"crop_cls"`` / ``"crop_mean"`` pool each crop to one vector,
    ``"patches"`` returns each tile's patch grid for
    :meth:`MeasureEmbedding.vectors_from` to pool under the regions' masks,
    and ``"none"`` is the ``"boxes"`` kind's, which has no tokens.

    Equality is identity (``eq=False``): an array has no single truth value.
    """

    kind: Literal["crops", "tiles", "boxes"]
    arrays: tuple[np.ndarray, ...]
    pooling: Literal["crop_cls", "crop_mean", "patches", "none"]


@dataclass(frozen=True, eq=False)
class PreparedRegions:
    """One image's regions, ready to encode: everything but the vectors (#282).

    :meth:`MeasureEmbedding.prepare_regions` builds it on the CPU, with no
    model loaded, and :meth:`MeasureEmbedding.payload_from` turns it and the
    encoder's vectors into the table's payload. It pickles, so a DataLoader
    worker process can build it for a GPU encoder in another process.

    Attributes:
        key: The payload key, ``<model>/<layer>/<region>``
            (:meth:`MeasureEmbedding.payload_key`).
        keys: The table's key columns, one row per region, in table order
            (``Object_Label``, or the ``Grid_*`` columns), index ``0..N-1``.
        key_column: The entry's ``key``: ``Object_Label`` or
            ``Grid_RowMajorIdx``.
        inputs: What the encoder runs on.
        binds: The digests binding the vectors to this image (spec D7).
        input: The series, level and layer the regions were read from.
        pool_plan: What :meth:`MeasureEmbedding.vectors_from` needs after the
            backbone, for an embedder that pools patch grids (tile offsets,
            the plate's feature-grid shape, the regions' masks); ``None``
            otherwise.
    """

    key: str
    keys: pd.DataFrame
    key_column: str
    inputs: BackboneInputs
    binds: dict[str, Any]
    input: dict[str, Any]
    pool_plan: Mapping[str, Any] | None = None

    @property
    def n_regions(self) -> int:
        """The number of regions, ``N``: the rows the table will have."""
        return len(self.keys)


class RegionEncoder(Protocol):
    """Runs a backbone over :class:`BackboneInputs` (#282).

    :meth:`MeasureEmbedding.inline_encoder` gives the one the inline path
    uses; #282's Stage 4 brings a batched GPU one. Either is called only for
    an image with at least one region.
    """

    def run(self, inputs: BackboneInputs) -> list[np.ndarray]:
        """The backbone's outputs for *inputs*.

        Returns:
            For ``"crops"``, one ``(N, D)`` float32 array, already pooled.
            For ``"tiles"``, one ``(h // p, w // p, D)`` float32 patch grid
            per tile. For ``"boxes"``, one ``(N, D)`` array.
        """
        ...


class _InlineBoxEncoder:
    """The ABC's inline :class:`RegionEncoder`: the subclass's ``_embed_*`` hooks."""

    def __init__(self, embedder: MeasureEmbedding, image: Image) -> None:
        self._embedder = embedder
        self._image = image

    def run(self, inputs: BackboneInputs) -> list[np.ndarray]:
        if inputs.kind != "boxes":
            raise ValueError(
                f"{type(self._embedder).__name__} prepared {inputs.kind!r} inputs "
                "but has no encoder for them; override inline_encoder"
            )
        boxes = inputs.arrays[0]
        if len(inputs.arrays) == 1:
            vectors = self._embedder._embed_boxes(self._image, boxes)
        else:
            vectors = self._embedder._embed_object_boxes(self._image, boxes, inputs.arrays[1])
        return [np.asarray(vectors)]


class MeasureEmbedding(MeasureFeatures, ABC):
    """Embed detected colonies, or the cells of a fixed grid, as feature vectors.

    An embedder crops image regions and turns each crop into one vector with
    a model (a DINOv2 backbone, say). Unlike every other measurer its output
    is not a set of scalar columns in the measurement table: each embedder
    writes its own table, ``embeddings/<model>/<layer>/<region>/table.parquet``,
    inside the run folder of the pipeline that produced it, keyed
    ``<model>/<layer>/<region>``. The layer is ``rgb`` or ``gray``, the
    pixels it read. The region is ``objects`` for colonies and ``grid`` for a
    grid, or ``objects-<name>`` / ``grid-<name>`` for an embedder given a
    ``name``.

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
    Naming an embedder (``name="wells"``) lets one pipeline hold several
    embedders of one model, layer and region -- two grids, or two poolings
    of the colonies -- each in its own table. An ``rgb`` and a ``gray`` copy
    need no name: the layer is a folder of its own.

    **Pixels.** :meth:`_layer_pixels` is what an embedder reads:
    ``layer="rgb"`` the colour layer as stored, ``layer="gray"`` the gray
    layer scaled to uint8 and stacked to three identical channels, so a
    model that takes RGB embeds either.

    To implement one, subclass this and define three methods:

    - :meth:`_embed_boxes` turns ``(N, 4)`` boxes into ``(N, D)`` vectors;
    - :meth:`model_token` names the model folder (``"dinov2-vitb14"``);
    - :meth:`entry_descriptor` describes the model and the vectors.

    Override :meth:`_embed_object_boxes` as well when a colony's crop needs
    its own pixels (a background mask, a masked pool), and list the fields
    that name the model in ``_leading_fields`` to put them before
    ``regions``, ``grid``, ``name`` and ``layer`` in the field order.

    **Three steps.** :meth:`payload` and :meth:`measure` run
    :meth:`prepare_regions` (CPU: keys, binds and the backbone's inputs),
    :meth:`encode` (the backbone, through a :class:`RegionEncoder`) and
    :meth:`payload_from` (checks, ``float16``, the descriptor). An embedder
    with a backbone overrides :meth:`prepare_regions` to cut its crops,
    :meth:`inline_encoder` to run them, and :meth:`vectors_from` if it pools
    patch grids, so a GPU stage can swap only the encoder.

    Args:
        regions: ``"objects"`` to embed detected colonies, ``"grid"`` to
            embed the cells of ``grid``.
        grid: The grid whose cells are embedded, in pixels of the image the
            measurer sees. Required with ``regions="grid"`` and refused
            otherwise. A cell outside the image fails that image.
        name: Names this embedder's table, so that one pipeline can hold
            several embedders of one model and region: the region becomes
            ``objects-<name>`` or ``grid-<name>`` instead of ``objects`` or
            ``grid``. A lowercase slug of 1 to 32 characters (``a-z``,
            ``0-9`` and ``_``, starting with a letter or digit). ``None``
            keeps the plain region.
        layer: The pixels embedded: ``"rgb"``, the colour layer as stored,
            or ``"gray"``, the gray layer as three identical uint8 channels.
            It is the ``<layer>`` folder of the table and its key, and
            the entry's ``input.layer``.

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
    name: Annotated[Union[str, None], TuneSpec(tunable=False)] = None
    layer: Annotated[EmbeddingLayer, TuneSpec(tunable=False)] = "rgb"

    #: Subclass fields placed before ``regions``, ``grid``, ``name`` and ``layer``, in this order:
    #: the ones that name the model (``size``, ``version``, ...), which a
    #: reader of a pipeline file looks for first. Empty keeps pydantic's
    #: order, the ABC's fields first.
    _leading_fields: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Move ``_leading_fields`` to the front of the subclass's field order.

        Pydantic puts inherited fields first, so without this every
        embedder's own model fields would follow ``regions``, ``grid``,
        ``name`` and ``layer``.
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

    @field_validator("name")
    @classmethod
    def _name_is_a_slug(cls, value: str | None) -> str | None:
        """A name is one lowercase path segment (:func:`check_region_name`)."""
        return value if value is None else check_region_name(value)

    @model_validator(mode="after")
    def _check_grid_matches_regions(self) -> "MeasureEmbedding":
        """``grid`` is required with ``regions="grid"`` and refused otherwise."""
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

    def _layer_pixels(self, image: Image) -> np.ndarray:
        """The ``(H, W, 3)`` pixels this embedder reads: its ``layer`` of *image*.

        ``"rgb"`` is ``image.rgb[:]`` as stored (uint8 or uint16, never
        copied). ``"gray"`` is ``image.gray[:]`` scaled to uint8 by its dtype
        range, the scaling an RGB crop gets (``[0, 1]`` to ``0..255``), and
        stacked to three identical channels, so a model that takes RGB gets
        a gray plate unchanged by any colour weighting.
        """
        if self.layer == "rgb":
            return np.asarray(image.rgb[:])
        from phenotypic.abc_._gpu_detector import _scale_to_uint8

        gray = np.asarray(image.gray[:])
        scaled = gray if gray.dtype == np.uint8 else _scale_to_uint8(gray, "dtype_range")
        return np.repeat(scaled[..., None], 3, axis=2)

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
        """The table's region folder: ``objects`` or ``grid``, suffixed ``-<name>`` when named."""
        base = OBJECTS_REGION if self.regions == "objects" else GRID_REGION
        return embedding_region(base, self.name)

    def payload_key(self) -> str:
        """``"<model_token()>/<layer>/<region>"``: the entry's key in its run folder.

        Two embedders of one pipeline must not share it
        (``ImagePipeline`` refuses them), so two embedders of one model,
        layer and region need different ``name``.
        """
        return embedding_key(self.model_token(), self.layer, self.region)

    def _dim(self) -> int:
        return self._descriptor_dim(self.entry_descriptor())

    def _descriptor_dim(self, descriptor: Mapping[str, Any]) -> int:
        """*descriptor*'s ``dim``, after checking it has every required field."""
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
            raise self._image_failure(image, exc) from exc

    def _image_failure(self, image: Image, exc: Exception) -> OperationFailedError:
        """*exc* as the image's failure: what :meth:`measure` and :meth:`payload` raise."""
        return OperationFailedError(
            operation=type(self).__name__,
            image_name=image.name,
            err_type=type(exc),
            message=str(exc),
        )

    def _operate(self, image: Image) -> pd.DataFrame:  # type: ignore[override]
        prepared = self.prepare_regions(image)
        vectors = self.encode(prepared, self.inline_encoder(image))
        return self._frame(
            prepared, self._stored_vectors(vectors, prepared.n_regions, self._dim())
        )

    @staticmethod
    def _frame(prepared: PreparedRegions, stored: np.ndarray) -> pd.DataFrame:
        """The table's frame: *prepared*'s key columns, then ``embedding``."""
        frame = prepared.keys.reset_index(drop=True)
        frame["embedding"] = list(stored)
        return frame

    def _regions(self, image: Image) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
        """The key columns, the ``(N, 4)`` boxes, and the labels (objects) or ``None``.

        Raises:
            GridCellOutsideImageError: As :meth:`_grid_cells`.
        """
        labels: np.ndarray | None = None
        if self.regions == "objects":
            keys, boxes = self._object_boxes(image)
            labels = keys[str(OBJECT.LABEL)].to_numpy(dtype=np.int64)
        else:
            keys, boxes = self._grid_cells(image)
        return keys.reset_index(drop=True), boxes, labels

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

    @staticmethod
    def _stored_vectors(vectors: np.ndarray, n: int, dim: int) -> np.ndarray:
        """*vectors*, checked to be ``(n, dim)`` and finite, as ``float16``.

        A zero-row *vectors* (``encode``'s ``(0, 0)``) becomes ``(0, dim)``.

        Args:
            vectors: The encoder's vectors, region order.
            n: The number of regions.
            dim: The descriptor's ``dim``.

        Raises:
            ValueError: If the shape is not ``(n, dim)``, or a value is not
                finite once stored as ``float16``.
        """
        vectors = np.asarray(vectors)
        if n == 0 and vectors.ndim == 2 and vectors.shape[0] == 0:
            vectors = vectors.reshape(0, dim)
        if vectors.ndim != 2 or vectors.shape != (n, dim):
            raise ValueError(
                f"the encoder returned shape {vectors.shape} for {n} regions; "
                f"expected ({n}, dim={dim})"
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

        ``name=None`` and ``layer="rgb"`` are left out
        (:data:`_DIGEST_NEUTRAL_DEFAULTS`), so an unnamed rgb embedder has the
        digest it had before either field existed; a name or ``layer="gray"``
        is part of it.
        """
        params = self.model_dump(mode="json")
        for field, neutral in _DIGEST_NEUTRAL_DEFAULTS.items():
            if field in params and params[field] == neutral:
                del params[field]
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def prepare_regions(self, image: Image) -> PreparedRegions:
        """Everything about *image*'s regions an encoder does not compute (#282).

        The key columns, the inputs, and the descriptor's ``binds`` and
        ``input``, on the CPU and with no model loaded. The ABC's inputs are
        ``"boxes"``: the regions' boxes (and, for objects, their labels), for
        :meth:`inline_encoder` to hand to :meth:`_embed_boxes` /
        :meth:`_embed_object_boxes`. An embedder with a backbone overrides
        this to cut its crops or tiles here, so a GPU stage can run the same
        preparation as the inline path.

        Args:
            image: The image to embed.

        Returns:
            The prepared regions, in table order.

        Raises:
            GridCellOutsideImageError: Unwrapped, as :meth:`_grid_cells`
                raises it; :meth:`payload` and :meth:`measure` wrap it.
        """
        keys, boxes, labels = self._regions(image)
        arrays = (boxes,) if labels is None else (boxes, labels)
        return self._prepared(
            image, keys, BackboneInputs(kind="boxes", arrays=arrays, pooling="none")
        )

    def _prepared(
        self,
        image: Image,
        keys: pd.DataFrame,
        inputs: BackboneInputs,
        pool_plan: Mapping[str, Any] | None = None,
    ) -> PreparedRegions:
        """A :class:`PreparedRegions` of *keys* and *inputs*, with *image*'s binds.

        The descriptor's ``binds`` are digests of the image the vectors are
        computed from, so a writer can tell whether the store it writes still
        holds that image (spec D7): for objects the objmap, for a grid the
        level-0 array of the series read and the grid edges. ``input`` names
        that series and the ``layer`` read; a ``gray`` grid still binds the
        primary series' array, the stored pixels gray is computed from, so
        staleness tracks the store's pixels.
        """
        from phenotypic.sdk_ import ngff_
        from phenotypic.sdk_._measurement_tables import array_digest, objmap_digest

        series = ngff_.primary_series(image._series_names())
        binds: dict[str, Any]
        if self.regions == "objects":
            key_column = str(OBJECT.LABEL)
            binds = {
                "objmap_path": ngff_.objmap_path(series),
                "objmap_sha256": objmap_digest(image.objmap[:]),
            }
        else:
            key_column = str(GRID.ROW_MAJOR_IDX)
            row_edges, col_edges = self._grid_edges(image)
            edges = json.dumps(
                {"row_edges": row_edges.tolist(), "col_edges": col_edges.tolist()},
                separators=(",", ":"),
            )
            binds = {
                "array_sha256": array_digest(image._store_series_array(series)),
                "grid_sha256": hashlib.sha256(edges.encode("utf-8")).hexdigest(),
            }
        return PreparedRegions(
            key=self.payload_key(),
            keys=keys,
            key_column=key_column,
            inputs=inputs,
            binds=binds,
            input={"series": series, "level": 0, "layer": self.layer},
            pool_plan=pool_plan,
        )

    def inline_encoder(self, image: Image) -> RegionEncoder:
        """The encoder the inline path uses for *image* (#282).

        The ABC's calls :meth:`_embed_object_boxes` (objects) or
        :meth:`_embed_boxes` (a grid) on *image*. An embedder whose
        :meth:`prepare_regions` cuts crops or tiles overrides this.
        """
        return _InlineBoxEncoder(self, image)

    def encode(self, prepared: PreparedRegions, encoder: RegionEncoder) -> np.ndarray:
        """*prepared*'s vectors from *encoder*: ``(N, D)``, region order (#282).

        An image with no regions returns a ``(0, 0)`` float32 array without
        calling ``encoder.run`` or :meth:`_dim` (whose descriptor would load a
        model); :meth:`payload_from` gives it the descriptor's ``dim``.
        """
        if prepared.n_regions == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return self.vectors_from(prepared, encoder.run(prepared.inputs))

    def vectors_from(
        self, prepared: PreparedRegions, outputs: list[np.ndarray]
    ) -> np.ndarray:
        """The ``(N, D)`` vectors, region order, from an encoder's *outputs* (#282).

        The ABC's inputs are pooled by the encoder already, so its one output
        is the vectors, returned as they are (no cast: :meth:`payload_from`
        casts once, to ``float16``). An embedder whose encoder returns patch
        grids (``pooling="patches"``) overrides this to pool them under the
        regions' masks (``prepared.pool_plan``).

        Raises:
            NotImplementedError: For ``"patches"`` outputs, which need an
                override.
            ValueError: If there is not exactly one output.
        """
        if prepared.inputs.pooling == "patches":
            raise NotImplementedError(
                f"{type(self).__name__} prepares patch grids; it must override vectors_from"
            )
        if len(outputs) != 1:
            raise ValueError(
                f"a {prepared.inputs.kind!r} encoder returns one array, got {len(outputs)}"
            )
        return np.asarray(outputs[0])

    def payload_from(
        self,
        prepared: PreparedRegions,
        vectors: np.ndarray,
        descriptor: Mapping[str, Any],
    ) -> EmbeddingPayload:
        """The payload of *prepared*'s regions and their *vectors* (spec D3, #282).

        Args:
            prepared: From :meth:`prepare_regions`.
            vectors: From :meth:`encode`, ``(N, descriptor["dim"])``, or
                ``(0, 0)`` for an image with no regions.
            descriptor: The model's part of the entry
                (:meth:`entry_descriptor`).

        Returns:
            The payload, keyed :meth:`payload_key`, with ``float16`` vectors
            and the descriptor ``{**descriptor, key, input, producer, binds}``.

        Raises:
            ValueError: If *descriptor* lacks a required field, the vectors
                are not ``(N, dim)``, or one is not finite in ``float16``.
        """
        from phenotypic.sdk_._embedding_tables import EmbeddingPayload

        dim = self._descriptor_dim(descriptor)
        stored = self._stored_vectors(vectors, prepared.n_regions, dim)
        entry = {
            **descriptor,
            "key": prepared.key_column,
            "input": dict(prepared.input),
            "producer": {
                "class": type(self).__name__,
                "params_sha256": self._params_sha256(),
            },
            "binds": dict(prepared.binds),
        }
        return EmbeddingPayload(
            key=prepared.key, frame=self._frame(prepared, stored), descriptor=entry
        )

    @validate_measure_integrity()
    def payload(self, image: Image) -> EmbeddingPayload:
        """Embed *image* and describe the table, for a store writer (spec D3).

        :meth:`ImagePipeline.embed` calls this for each embedder. It is
        :meth:`prepare_regions`, then :meth:`encode` with
        :meth:`inline_encoder`, then :meth:`payload_from` with
        :meth:`entry_descriptor`; see :meth:`_prepared` for the descriptor's
        ``binds`` and ``input``.

        Args:
            image: The image to embed.

        Returns:
            The payload, keyed :meth:`payload_key`.

        Raises:
            OperationFailedError: As :meth:`measure`.
            MemoryError: Re-raised unwrapped, as :meth:`measure` does.
        """
        try:
            prepared = self.prepare_regions(image)
            vectors = self.encode(prepared, self.inline_encoder(image))
            return self.payload_from(prepared, vectors, self.entry_descriptor())
        except Exception as exc:
            if is_out_of_memory(exc):
                raise
            raise self._image_failure(image, exc) from exc
