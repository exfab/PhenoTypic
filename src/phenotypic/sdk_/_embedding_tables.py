"""Embedding tables of a table run folder (#281 spec D1, D3, D7, D9; E1, E2).

An embedding belongs to the pipeline that produced it, so its table lives in
that pipeline's run folder::

    tables/<n>-<sha12>/embeddings/<model>/crops/table.parquet   Object_Label | embedding
    tables/<n>-<sha12>/embeddings/<model>/grid/table.parquet    Grid_* keys  | embedding

:class:`EmbeddingPayload` is what :meth:`ImagePipeline.embed` returns and a
store writer consumes: one table's frame plus the descriptor fields the
producer knows (model, vector length, what it read, what it is bound to).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import pandas as pd

#: The vector column of every embedding table (spec D1, E5).
EMBEDDING_COLUMN: Final[str] = "embedding"

#: The descriptor fields a producer must supply (spec D3); the writer adds
#: ``stale`` and ``files``.
PAYLOAD_DESCRIPTOR_FIELDS: Final[frozenset[str]] = frozenset(
    {"model", "dim", "dtype", "pooling", "key", "input", "producer", "binds"}
)


@dataclass(frozen=True)
class EmbeddingPayload:
    """One embedding table a pipeline produced for one image, not yet written.

    Attributes:
        key: ``"<model>/<crops|grid>"``, the entry's key in the run's
            ``embeddings`` block and its folder under ``embeddings/``.
        frame: The table: its key columns (``Object_Label`` for crops, the
            ``Grid_*`` keys for a grid) and ``embedding``, one 1-D
            ``float16`` vector per row.
        descriptor: The entry fields the producer knows (spec D3): ``model``,
            ``dim``, ``dtype``, ``pooling``, ``key`` (the index column),
            ``input`` (``{series, level}``), ``producer``
            (``{class, params_sha256}``) and ``binds`` -- for crops
            ``{objmap_path, objmap_sha256}``, for a grid
            ``{array_sha256, grid_sha256}``, digests of the image the vectors
            were computed from. The writer adds ``stale`` and ``files``.

    Raises:
        ValueError: If *key* is not ``"<model>/<crops|grid>"`` with *model*
            one path segment, or *descriptor* lacks a required field.
    """

    key: str
    frame: pd.DataFrame
    descriptor: Mapping[str, object]

    def __post_init__(self) -> None:
        from ._table_runs import embedding_table_relative_path

        model, _, region = self.key.partition("/")
        # Validates both halves: the region is crops or grid, the model one segment.
        embedding_table_relative_path("1-unknown", model, region)  # type: ignore[arg-type]
        missing = PAYLOAD_DESCRIPTOR_FIELDS - set(self.descriptor)
        if missing:
            raise ValueError(
                f"embedding payload {self.key!r} lacks descriptor fields {sorted(missing)}"
            )

    @property
    def model(self) -> str:
        """The model folder, ``<model>`` of :attr:`key`."""
        return self.key.partition("/")[0]

    @property
    def region(self) -> str:
        """``"crops"`` or ``"grid"``, the region folder of :attr:`key`."""
        return self.key.partition("/")[2]
