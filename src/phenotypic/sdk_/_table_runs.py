"""Per-pipeline table runs in a store: naming, numbering and the resolver (spec D1-D6a).

Since #281 P3 a store's tables live in one folder per measuring pipeline::

    tables/<n>-<sha12>/{measurements,metadata,embeddings/<model>/{crops,grid}}/

described by ``attributes.phenotypic.tables``::

    {"schema_version": 2,
     "runs": {"<n>-<sha12>": {"order": n, "pipeline_sha256": ..., ...}},
     "upstream": {"<n>-<sha12>": {"source": ..., "root_sha256": ...,
                                  "pipeline_sha256": ...}}}

Every module that reaches a run folder does so through this one resolver, so
the layout lives here and nowhere else. This module is pure parsing of the
root ``zarr.json`` block: it imports no pandas and opens no Parquet.

**Folder numbers** are a store-local call-order counter (spec D2). A new folder
gets ``1 +`` the highest number among the store's run folders and its
``upstream`` entries, so numbers are never reused and two folders never share
one. The provenance journal's ``sequence`` is deliberately **not** used: it is
local to one image's journal, and a forward run over a raw input is always
``1``.

**Which folder a writer owns** (spec D6): the folder whose pipeline digest is
the writer's (it is replaced, keeping its number); a writer with no known
digest replaces the highest-numbered ``-unknown`` folder; otherwise a new
folder takes the next number. :func:`choose_table_run_id` is that rule.

**Which folder a reader uses** (spec D6a): a reader holding a per-image record
uses the folder the record certifies; a reader without one defaults to
:func:`latest_measurement_run`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

#: A run folder's name: ``<order>-<first 12 hex of the pipeline sha256>``, or
#: ``<order>-unknown`` when no digest is known. ``order`` starts at 1.
TABLE_RUN_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<order>[1-9][0-9]*)-(?P<sha>[0-9a-f]{12}|unknown)$"
)

#: The ``-unknown`` suffix of a run whose pipeline digest is not known (D2).
UNKNOWN_PIPELINE: Final[str] = "unknown"

#: Keys of the v2 ``tables`` descriptor and of each run entry.
RUNS_KEY: Final[str] = "runs"
UPSTREAM_KEY: Final[str] = "upstream"
ORDER_KEY: Final[str] = "order"
PIPELINE_SHA256_KEY: Final[str] = "pipeline_sha256"
SCHEMA_VERSION_KEY: Final[str] = "schema_version"
EMBEDDINGS_KEY: Final[str] = "embeddings"

#: Embedding regions a run folder may hold (spec D1).
EmbeddingRegion = Literal["crops", "grid"]
_EMBEDDING_REGIONS: Final[frozenset[str]] = frozenset({"crops", "grid"})

_SHA_PREFIX_LENGTH: Final[int] = 12


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def table_run_id(order: int, pipeline_sha256: str | None) -> str:
    """Return the folder name of run *order* for *pipeline_sha256*.

    Args:
        order: The store-local call-order number, ``>= 1``.
        pipeline_sha256: The pipeline file's 64-hex sha256, or ``None`` when
            no digest is known, which names the folder ``<order>-unknown``.
            A digest is never fabricated (spec D2).

    Returns:
        ``"<order>-<sha12>"`` or ``"<order>-unknown"``.

    Raises:
        ValueError: If *order* is below 1 or *pipeline_sha256* is not 64 hex.

    Examples:
        >>> table_run_id(2, "a71e09c2b4f5" + "0" * 52)
        '2-a71e09c2b4f5'
        >>> table_run_id(3, None)
        '3-unknown'
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError(f"table run order must be an integer >= 1, got {order!r}")
    if pipeline_sha256 is None:
        return f"{order}-{UNKNOWN_PIPELINE}"
    if not _is_sha256(pipeline_sha256):
        raise ValueError(
            f"pipeline_sha256 must be 64 lowercase hex, got {pipeline_sha256!r}"
        )
    return f"{order}-{pipeline_sha256[:_SHA_PREFIX_LENGTH]}"


def table_run_relative_path(run_id: str) -> PurePosixPath:
    """Store-relative path of run folder *run_id*: ``tables/<run_id>``."""
    from . import ngff_

    return PurePosixPath(ngff_.TABLES_GROUP, run_id)


def measurement_table_relative_path(run_id: str) -> PurePosixPath:
    """``tables/<run_id>/measurements/table.parquet``, store-relative."""
    from . import ngff_

    return table_run_relative_path(run_id) / ngff_.MEASUREMENT_TABLE_GROUP / (
        ngff_.MEASUREMENT_TABLE_FILENAME
    )


def metadata_table_relative_path(run_id: str) -> PurePosixPath:
    """``tables/<run_id>/metadata/pht-metadata.parquet``, store-relative."""
    from . import ngff_

    return table_run_relative_path(run_id) / ngff_.METADATA_TABLE_GROUP / (
        ngff_.METADATA_TABLE_FILENAME
    )


def embedding_table_relative_path(
    run_id: str, model: str, region: EmbeddingRegion
) -> PurePosixPath:
    """``tables/<run_id>/embeddings/<model>/<region>/table.parquet``, store-relative.

    Raises:
        ValueError: If *region* is not ``"crops"`` or ``"grid"``, or *model*
            is not a single path segment.
    """
    from . import ngff_

    if region not in _EMBEDDING_REGIONS:
        raise ValueError(f"embedding region must be crops or grid, got {region!r}")
    if not model or "/" in model or "\\" in model or model in {".", ".."}:
        raise ValueError(f"embedding model must be one path segment, got {model!r}")
    return (
        table_run_relative_path(run_id)
        / ngff_.EMBEDDINGS_TABLE_GROUP
        / model
        / region
        / ngff_.MEASUREMENT_TABLE_FILENAME
    )


@dataclass(frozen=True)
class TableRun:
    """One run folder of one store, as its root descriptor declares it.

    Attributes:
        store: The ``*.ome.zarr`` directory.
        run_id: The folder name, ``<order>-<sha12>`` or ``<order>-unknown``.
        order: The store-local call-order number (spec D2).
        pipeline_sha256: The pipeline file's sha256, or ``None`` when unknown.
        descriptor: The run's entry in ``attributes.phenotypic.tables.runs``.
    """

    store: Path
    run_id: str
    order: int
    pipeline_sha256: str | None
    descriptor: Mapping[str, object]

    @property
    def has_measurements(self) -> bool:
        """Whether the run declares a ``measurements`` table (D6a's reader default)."""
        from . import ngff_

        return isinstance(self.descriptor.get(ngff_.MEASUREMENT_TABLE_GROUP), Mapping)

    @property
    def folder(self) -> Path:
        """The run folder, ``<store>/tables/<run_id>``."""
        return Path(self.store) / table_run_relative_path(self.run_id)

    def measurements_path(self) -> Path:
        """The run's ``measurements/table.parquet`` (whether or not it exists)."""
        return Path(self.store) / measurement_table_relative_path(self.run_id)

    def metadata_path(self) -> Path:
        """The run's ``metadata/pht-metadata.parquet`` (whether or not it exists)."""
        return Path(self.store) / metadata_table_relative_path(self.run_id)

    def embedding_path(self, model: str, region: EmbeddingRegion) -> Path:
        """The run's ``embeddings/<model>/<region>/table.parquet``."""
        return Path(self.store) / embedding_table_relative_path(
            self.run_id, model, region
        )


def _tables_block(block: Mapping[str, object]) -> Mapping[str, object] | None:
    """The ``tables`` mapping of an ``attributes.phenotypic`` *block*, or ``None``."""
    from . import ngff_

    tables = block.get(ngff_.PhenotypicAttr.TABLES)
    return tables if isinstance(tables, Mapping) else None


def tables_schema_of_block(block: Mapping[str, object]) -> Literal[0, 1, 2]:
    """The tables layout an ``attributes.phenotypic`` *block* declares.

    Returns:
        ``0`` for no ``tables`` descriptor, ``1`` for the flat v1 layout
        (a ``tables`` mapping with no top-level ``schema_version``), ``2`` for
        per-run folders.

    Raises:
        ValueError: If the descriptor's ``schema_version`` is one this build
            does not know -- a newer writer's layout, which must be refused
            rather than read under today's semantics.
    """
    from . import ngff_

    tables = _tables_block(block)
    if tables is None:
        return 0
    if SCHEMA_VERSION_KEY not in tables:
        return 1
    version = tables[SCHEMA_VERSION_KEY]
    if version == ngff_.TABLES_SCHEMA_VERSION:
        return 2
    raise ValueError(
        f"tables schema_version {version!r} is not one this build reads "
        f"({ngff_.TABLES_SCHEMA_VERSION}); upgrade PhenoTypic to read it"
    )


def _parse_runs(
    block: Mapping[str, object],
) -> list[tuple[str, int, str | None, Mapping[str, object]]]:
    """Validate a v2 block's runs; return ``(run_id, order, sha, entry)`` by ascending order.

    A v0 or v1 block has no runs. Every run id must match
    :data:`TABLE_RUN_ID_RE`, its entry's ``order`` must equal the id's
    number, a declared ``pipeline_sha256`` must agree with the id, and no two
    runs may share an order -- the counter makes that unreachable, so it means
    a corrupt descriptor.

    Raises:
        ValueError: On any of the above.
    """
    if tables_schema_of_block(block) != 2:
        return []
    tables = _tables_block(block)
    assert tables is not None
    runs = tables.get(RUNS_KEY, {})
    if not isinstance(runs, Mapping):
        raise ValueError("tables descriptor 'runs' is not a mapping")
    parsed: list[tuple[str, int, str | None, Mapping[str, object]]] = []
    seen: dict[int, str] = {}
    for run_id, entry in runs.items():
        match = TABLE_RUN_ID_RE.match(str(run_id))
        if match is None:
            raise ValueError(f"table run id {run_id!r} is not <order>-<sha12>")
        if not isinstance(entry, Mapping):
            raise ValueError(f"table run {run_id!r} entry is not a mapping")
        order = entry.get(ORDER_KEY)
        if isinstance(order, bool) or not isinstance(order, int):
            raise ValueError(f"table run {run_id!r} has no integer order")
        if order != int(match["order"]):
            raise ValueError(
                f"table run {run_id!r} declares order {order}, not its folder's "
                f"{match['order']}"
            )
        if order in seen:
            raise ValueError(
                f"table runs {seen[order]!r} and {run_id!r} share order {order}: "
                "the store's tables descriptor is corrupt"
            )
        seen[order] = str(run_id)
        sha = entry.get(PIPELINE_SHA256_KEY)
        if sha is not None:
            if not _is_sha256(sha):
                raise ValueError(f"table run {run_id!r} pipeline_sha256 is malformed")
            if match["sha"] != str(sha)[:_SHA_PREFIX_LENGTH]:
                raise ValueError(
                    f"table run {run_id!r} declares pipeline_sha256 {sha!r}, "
                    "which its folder name does not carry"
                )
        parsed.append((str(run_id), order, sha, entry))
    parsed.sort(key=lambda item: item[1])
    return parsed


def _read_block(store: Path) -> Mapping[str, object]:
    from . import ngff_

    return ngff_.require_readable_store(Path(store))


def store_tables_schema(store: Path) -> Literal[0, 1, 2]:
    """The tables layout of *store*: ``0`` none, ``1`` flat v1, ``2`` per-run folders.

    Raises:
        FileNotFoundError: If the store has no root ``zarr.json``.
        ValueError: If the store, or its tables descriptor, is a version this
            build does not read.
    """
    return tables_schema_of_block(_read_block(store))


def table_runs_of_block(
    store: Path, block: Mapping[str, object]
) -> tuple[TableRun, ...]:
    """:func:`list_table_runs` over an already-read *block* of *store*."""
    return tuple(
        TableRun(
            store=Path(store),
            run_id=run_id,
            order=order,
            pipeline_sha256=sha,
            descriptor=entry,
        )
        for run_id, order, sha, entry in _parse_runs(block)
    )


def list_table_runs(store: Path) -> tuple[TableRun, ...]:
    """Every run folder *store* declares, by ascending ``order``.

    Returns:
        ``()`` for a store with no tables or with the flat v1 layout.

    Raises:
        FileNotFoundError: If the store has no root ``zarr.json``.
        ValueError: If the descriptor is corrupt (two runs with one order, a
            malformed id) or a version this build does not read.
    """
    return table_runs_of_block(store, _read_block(store))


def latest_table_run(store: Path) -> TableRun | None:
    """The run with the highest ``order``, or ``None`` when there is none."""
    runs = list_table_runs(store)
    return runs[-1] if runs else None


def latest_measurement_run(store: Path) -> TableRun | None:
    """The highest-``order`` run that holds ``measurements`` (spec D6a).

    The default for a reader without a per-image record. A grid-only process
    folder holds no ``measurements`` and is skipped.
    """
    return next(
        (run for run in reversed(list_table_runs(store)) if run.has_measurements),
        None,
    )


def table_run_for_pipeline(store: Path, pipeline_sha256: str) -> TableRun | None:
    """The run folder *pipeline_sha256* owns in *store*, or ``None``."""
    matches = [
        run for run in list_table_runs(store) if run.pipeline_sha256 == pipeline_sha256
    ]
    return matches[-1] if matches else None


def table_run_by_id(store: Path, run_id: str) -> TableRun | None:
    """The run folder named *run_id* in *store*, or ``None``."""
    return next(
        (run for run in list_table_runs(store) if run.run_id == run_id), None
    )


def _upstream_orders(block: Mapping[str, object]) -> list[int]:
    tables = _tables_block(block)
    if tables is None or tables_schema_of_block(block) != 2:
        return []
    upstream = tables.get(UPSTREAM_KEY, {})
    if not isinstance(upstream, Mapping):
        raise ValueError("tables descriptor 'upstream' is not a mapping")
    orders = []
    for key in upstream:
        match = TABLE_RUN_ID_RE.match(str(key))
        if match is None:
            raise ValueError(f"upstream run id {key!r} is not <order>-<sha12>")
        orders.append(int(match["order"]))
    return orders


def next_table_run_order(block: Mapping[str, object]) -> int:
    """``1 +`` the highest order among *block*'s runs and upstream entries (D2).

    Args:
        block: A store's ``attributes.phenotypic`` block.

    Returns:
        The number a new run folder takes; ``1`` when there are none.
    """
    orders = [order for _run_id, order, _sha, _entry in _parse_runs(block)]
    orders += _upstream_orders(block)
    return max(orders, default=0) + 1


def choose_table_run_id(
    block: Mapping[str, object], *, pipeline_sha256: str | None
) -> str:
    """The run folder a writer with *pipeline_sha256* owns in *block* (spec D6).

    1. A run with this digest exists: its id (replaced, keeping its number).
    2. *pipeline_sha256* is ``None`` and an ``-unknown`` run exists: the
       highest-numbered one, so notebook re-saves do not pile up folders.
    3. Otherwise a new folder with :func:`next_table_run_order`.

    Args:
        block: The live store's ``attributes.phenotypic`` block (an empty
            mapping for a new store), with any ``upstream`` the write records.
        pipeline_sha256: The writer's pipeline digest, or ``None``.

    Returns:
        The run id.
    """
    runs = _parse_runs(block)
    if pipeline_sha256 is not None:
        owned = [run_id for run_id, _o, sha, _e in runs if sha == pipeline_sha256]
    else:
        owned = [
            run_id
            for run_id, _o, sha, _e in runs
            if sha is None and run_id.endswith(f"-{UNKNOWN_PIPELINE}")
        ]
    if owned:
        return owned[-1]
    return table_run_id(next_table_run_order(block), pipeline_sha256)


def store_of_table_path(path: Path) -> Path:
    """The store a table path lies in: its nearest ancestor ending in ``.ome.zarr``.

    Works at any depth, so it serves both the v1 flat path and a v2 run
    folder's, which the old ``parents[len(MEASUREMENT_TABLE_RELATIVE_PATH.parts)
    - 1]`` arithmetic could not.

    Raises:
        ValueError: If no ancestor of *path* is a store.

    Examples:
        >>> from pathlib import Path
        >>> s = Path("out/x.ome.zarr")
        >>> store_of_table_path(s / "tables/2-a71e09c2b4f5/measurements/table.parquet") == s
        True
    """
    from . import ngff_

    for ancestor in Path(path).parents:
        if ancestor.name.endswith(ngff_.STORE_SUFFIX):
            return ancestor
    raise ValueError(f"{path} does not lie inside an OME-Zarr store")
