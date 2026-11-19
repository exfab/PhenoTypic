"""Per-pipeline table runs in a store: naming, numbering and the resolver (spec D1-D6a).

Since #281 P3 a store's tables live in one folder per measuring pipeline::

    tables/<n>-<sha12>/{measurements,metadata,embeddings/<model>/{crops,grid,grid-<name>}}/

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
:func:`record_less_measurement_table`, whose v2 run is
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

#: An embedding region folder of a run (spec D1, E3): ``crops``, ``grid``,
#: or ``grid-<name>`` for a named grid. :func:`split_embedding_key` is the
#: one rule for which strings are regions.
EmbeddingRegion = str

#: The region of a crop embedder (``regions="objects"``).
CROPS_REGION: Final[str] = "crops"
#: The region of an unnamed grid embedder; a named one is ``grid-<name>``.
GRID_REGION: Final[str] = "grid"

#: A grid name (#281 named grid regions, user decision 2026-11-19): a
#: lowercase slug, safe as one path segment and as a file name.
_GRID_NAME: Final[str] = r"[a-z0-9][a-z0-9_]{0,31}"
_GRID_NAME_RE: Final[re.Pattern[str]] = re.compile(_GRID_NAME)
_EMBEDDING_REGION_RE: Final[re.Pattern[str]] = re.compile(
    rf"{CROPS_REGION}|{GRID_REGION}(?:-{_GRID_NAME})?"
)

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


def check_grid_name(name: str) -> str:
    """Return *name* if it is a grid name: ``^[a-z0-9][a-z0-9_]{0,31}$``.

    The name a ``MeasureEmbedding``'s ``grid_name`` must be, so its region
    ``grid-<name>`` passes :func:`split_embedding_key`.

    Raises:
        ValueError: If *name* is not a lowercase slug of 1 to 32 characters
            (letters, digits and ``_``, starting with a letter or digit).
    """
    if not isinstance(name, str) or _GRID_NAME_RE.fullmatch(name) is None:
        raise ValueError(
            "grid_name must be a lowercase slug of 1 to 32 characters "
            f"(a-z, 0-9 and _, starting with a-z or 0-9), got {name!r}"
        )
    return name


def grid_region(grid_name: str | None) -> str:
    """The region folder of a grid embedder: ``grid``, or ``grid-<grid_name>``.

    Raises:
        ValueError: If *grid_name* is not ``None`` and not a grid name
            (:func:`check_grid_name`).

    Examples:
        >>> grid_region(None), grid_region("wells")
        ('grid', 'grid-wells')
    """
    if grid_name is None:
        return GRID_REGION
    return f"{GRID_REGION}-{check_grid_name(grid_name)}"


def _check_embedding_folders(model: str, region: str) -> None:
    """Refuse a region that is not crops, grid or grid-<name>, or a model that is not one path segment."""
    if not isinstance(region, str) or _EMBEDDING_REGION_RE.fullmatch(region) is None:
        raise ValueError(
            "embedding region must be crops or grid, or grid-<name> with <name> "
            f"a lowercase slug (^[a-z0-9][a-z0-9_]{{0,31}}$), got {region!r}"
        )
    if not model or "/" in model or "\\" in model or model in {".", ".."}:
        raise ValueError(f"embedding model must be one path segment, got {model!r}")


def split_embedding_key(key: str) -> tuple[str, EmbeddingRegion]:
    """``(model, region)`` of an embedding entry key, ``"<model>/<region>"``.

    The region is ``crops``, ``grid`` or ``grid-<name>``, where ``<name>``
    is a grid name (:func:`check_grid_name`): ``dinov2-vitb14/grid-wells``.

    The one rule for a key: an
    :class:`~phenotypic.sdk_._embedding_tables.EmbeddingPayload` is refused
    on it when built, and the run's embedding deliverables refuse a store
    entry on it when finalization reads one (a store root edited by hand,
    say), because the key names the deliverable's file.

    Raises:
        ValueError: If the part after the first ``/`` is not a region (so
            ``"a/b/c"``, ``"m/grid-"`` and ``"m/grid-Wells"`` are refused),
            or the part before it is not one path segment.

    Examples:
        >>> split_embedding_key("dinov2-vitb14/grid-wells")
        ('dinov2-vitb14', 'grid-wells')
    """
    model, _, region = key.partition("/")
    _check_embedding_folders(model, region)
    return model, region  # type: ignore[return-value]


def embedding_table_relative_path(
    run_id: str, model: str, region: EmbeddingRegion
) -> PurePosixPath:
    """``tables/<run_id>/embeddings/<model>/<region>/table.parquet``, store-relative.

    Raises:
        ValueError: If *region* is not ``"crops"``, ``"grid"`` or
            ``"grid-<name>"`` (:func:`split_embedding_key`), or *model* is
            not a single path segment.
    """
    from . import ngff_

    _check_embedding_folders(model, region)
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


def _run_of_block(
    store: Path, block: Mapping[str, object], run_id: str
) -> TableRun | None:
    """The run named *run_id* among *block*'s runs, or ``None``."""
    return next(
        (run for run in table_runs_of_block(store, block) if run.run_id == run_id),
        None,
    )


def latest_measurement_run(store: Path) -> TableRun | None:
    """The highest-``order`` run that holds ``measurements`` (spec D6a).

    The default for a reader without a per-image record
    (:func:`record_less_measurement_table`). A grid-only process folder holds
    no ``measurements`` and is skipped; a v1 flat store has no runs.
    """
    table = record_less_measurement_table(store, _read_block(store))
    return None if table is None else table.run


@dataclass(frozen=True)
class DeclaredMeasurementTable:
    """One ``measurements`` table a store root declares, for a reader.

    The common shape of a v2 run's table and of a **v1 flat** store's single
    table, so a reader that must also read an unconverted store does not
    grow a second code path. Read-only: nothing here, and nothing a reader
    builds from it, writes the v1 layout.

    Attributes:
        relative_path: The payload, relative to the store: the run's
            ``tables/<run>/measurements/table.parquet``, or the v1
            ``tables/measurements/table.parquet``. Derived from the layout,
            never from the descriptor's own ``path`` string.
        entry: A run-entry-shaped mapping: ``measurements`` (the table's
            descriptor), and ``metadata_table`` when recorded. For a v2 run
            it is the run's own entry; for a v1 store it is assembled from
            ``tables.measurements`` and the root-level ``metadata_table``
            block, where the v1 layout keeps them.
        run: The v2 run, or ``None`` for a v1 store.
    """

    relative_path: PurePosixPath
    entry: Mapping[str, object]
    run: TableRun | None

    @property
    def descriptor(self) -> Mapping[str, object]:
        """The table's ``measurements`` descriptor."""
        from . import ngff_

        descriptor = self.entry[ngff_.MEASUREMENT_TABLE_GROUP]
        assert isinstance(descriptor, Mapping)
        return descriptor


def declared_measurement_tables(
    store: Path, block: Mapping[str, object]
) -> tuple[DeclaredMeasurementTable, ...]:
    """Every ``measurements`` table *block* declares, by ascending run order.

    The one place a reader learns where a store's measurement tables are,
    for both layouts. Keyed **strictly on the tables schema**
    (:func:`tables_schema_of_block`), never on which paths exist:

    - schema 2: each run that holds ``measurements``, in ``order``;
    - schema 1 (the v1 flat layout ``--mode migrate`` converts): the single
      ``tables.measurements`` table, when the descriptor declares one. Spec
      D5 refuses v1 only in the *forward* modes; ``resolve_run_state``, the
      GUI and migrate's own planning read unconverted trees;
    - schema 0: none.

    The last element is spec D6a's record-less default
    (:func:`record_less_measurement_table`); a reader holding a per-image
    record picks the element whose
    :attr:`~DeclaredMeasurementTable.relative_path` the record certifies
    instead.

    Args:
        store: The store the block belongs to.
        block: Its ``attributes.phenotypic`` block.

    Raises:
        ValueError: If the tables descriptor is corrupt or of a version this
            build does not read.
    """
    from . import ngff_

    schema = tables_schema_of_block(block)
    if schema == 2:
        return tuple(
            DeclaredMeasurementTable(
                relative_path=measurement_table_relative_path(run.run_id),
                entry=run.descriptor,
                run=run,
            )
            for run in table_runs_of_block(store, block)
            if run.has_measurements
        )
    if schema == 1:
        tables = _tables_block(block)
        assert tables is not None
        descriptor = tables.get(ngff_.MEASUREMENT_TABLE_GROUP)
        if not isinstance(descriptor, Mapping):
            return ()
        entry: dict[str, object] = {ngff_.MEASUREMENT_TABLE_GROUP: descriptor}
        metadata_table = block.get(ngff_.PhenotypicAttr.METADATA_TABLE)
        if isinstance(metadata_table, Mapping):
            entry[ngff_.PhenotypicAttr.METADATA_TABLE] = metadata_table
        return (
            DeclaredMeasurementTable(
                relative_path=PurePosixPath(
                    ngff_.MEASUREMENT_TABLE_RELATIVE_PATH.as_posix()
                ),
                entry=entry,
                run=None,
            ),
        )
    return ()


def record_less_measurement_table(
    store: Path, block: Mapping[str, object]
) -> DeclaredMeasurementTable | None:
    """The table a reader with no per-image record uses (spec D6a), in either layout.

    The one owner of that rule: the last element of
    :func:`declared_measurement_tables` -- a v2 store's highest-numbered run
    that holds ``measurements``, or a v1 flat store's one table. Every
    record-less reader (:func:`latest_measurement_run`,
    :func:`declared_measurement_table_path`, the embedded-descriptor reader,
    the run-state fallback) delegates here.

    Args:
        store: The store the block belongs to.
        block: Its ``attributes.phenotypic`` block.

    Returns:
        The table, or ``None`` when *block* declares none.

    Raises:
        ValueError: If the tables descriptor is corrupt or of a version this
            build does not read.
    """
    declared = declared_measurement_tables(store, block)
    return declared[-1] if declared else None


def declared_measurement_table_path(store: Path) -> Path | None:
    """The record-less default measurements table of *store*, in either layout.

    The payload of :func:`record_less_measurement_table`: a v2 store's
    latest run that holds ``measurements``, or a v1 flat store's one table
    (``--mode migrate`` reads both). The path is where the layout puts the
    table; it may not exist on disk.

    Returns:
        The payload path, or ``None`` when the store has no root, no
        ``phenotypic`` block, an unreadable tables descriptor, or declares no
        measurement table.
    """
    from . import ngff_

    try:
        block = ngff_.read_phenotypic_attributes(Path(store))
        if not isinstance(block, Mapping):
            return None
        table = record_less_measurement_table(Path(store), block)
    except (OSError, KeyError, ValueError):
        return None
    return None if table is None else Path(store) / table.relative_path


def _figure_run_recency(run_id: str, entry: Mapping[str, object]) -> tuple[str, str, str]:
    """Sort key for a figure run: its date, then its initiating call's time, then its id."""
    date = entry.get("date")
    if not isinstance(date, str):
        # The run folder is named `<YYYY-MM-DD>-<sha12>`.
        date = run_id[:10]
    initiated = entry.get("initiated_at_utc")
    return (date, initiated if isinstance(initiated, str) else "", run_id)


def v1_table_pipeline_sha(block: Mapping[str, object]) -> str | None:
    """The pipeline digest a v1 flat store's tables belong to (spec D10).

    ``--mode migrate`` names the converted folder ``1-<sha12>`` after it:

    1. the most recent figure run's ``pipeline_sha256``, when the figures
       descriptor has one. A v1 table rewritten by ``--mode measure`` was
       never journaled (measure mode opens no application), so only its
       figure run names its pipeline. Recency is the run's ``date``, then its
       ``initiated_at_utc``, then its folder name;
    2. otherwise the latest journal application of kind ``full`` or
       ``legacy`` that carries a pipeline digest (a schema-1 journal's own
       ``pipeline`` counts as its one application);
    3. otherwise ``None``: the folder is ``1-unknown``. A digest is never
       fabricated (spec D2).

    Pure parsing of an ``attributes.phenotypic`` *block*.

    Examples:
        >>> sha = "b" * 64
        >>> v1_table_pipeline_sha(
        ...     {"figures": {"runs": {"2026-11-01-" + sha[:12]: {"pipeline_sha256": sha}}}}
        ... ) == sha
        True
        >>> v1_table_pipeline_sha({}) is None
        True
    """
    from . import ngff_

    figures = block.get(ngff_.PhenotypicAttr.FIGURES)
    runs = figures.get("runs") if isinstance(figures, Mapping) else None
    if isinstance(runs, Mapping):
        candidates = sorted(
            (
                _figure_run_recency(str(run_id), entry),
                entry[PIPELINE_SHA256_KEY],
            )
            for run_id, entry in runs.items()
            if isinstance(entry, Mapping)
            and _is_sha256(entry.get(PIPELINE_SHA256_KEY))
        )
        if candidates:
            return str(candidates[-1][1])

    journal = block.get(ngff_.PhenotypicAttr.PROVENANCE)
    if not isinstance(journal, Mapping):
        return None
    applications = journal.get("applications")
    if not isinstance(applications, list):
        # A schema-1 journal: one implicit application, its own `pipeline`.
        applications = [{"kind": "legacy", "pipeline": journal.get("pipeline")}]
    for application in reversed(applications):
        if not isinstance(application, Mapping) or application.get("kind") not in {
            "full",
            "legacy",
        }:
            continue
        pipeline = application.get("pipeline")
        digest = pipeline.get("sha256") if isinstance(pipeline, Mapping) else None
        if _is_sha256(digest):
            return str(digest)
    return None


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


def run_upstream_collisions(
    runs: Mapping[str, object], upstream: Mapping[str, object]
) -> list[str]:
    """Where *runs* and *upstream* share a run id or an order (spec D2), as sentences.

    D2 numbers a new folder past every run and upstream entry, so a store
    never makes a collision itself. It can still inherit one: a run keeps its
    number when its pipeline re-saves (D6) while the input it references has
    since grown a run of the same number, or one id lands in both blocks.
    Readers key runs and upstream entries separately, so nothing breaks; the
    writer logs it (review M1) rather than renumbering a folder a record may
    certify.

    Args:
        runs: The descriptor's ``runs`` (run id -> entry).
        upstream: The descriptor's ``upstream`` (run id -> entry).

    Returns:
        One sentence per collision; ``[]`` when there is none.
    """
    def _orders(ids: Mapping[str, object]) -> dict[int, str]:
        found: dict[int, str] = {}
        for run_id in ids:
            match = TABLE_RUN_ID_RE.match(str(run_id))
            if match is not None:
                found.setdefault(int(match["order"]), str(run_id))
        return found

    collisions = [
        f"run {run_id!r} is also an upstream entry"
        for run_id in sorted(set(map(str, runs)) & set(map(str, upstream)))
    ]
    upstream_orders = _orders(upstream)
    for order, run_id in sorted(_orders(runs).items()):
        other = upstream_orders.get(order)
        if other is not None and other != run_id:
            collisions.append(
                f"run {run_id!r} shares order {order} with upstream entry {other!r}"
            )
    return collisions


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
    owned = owned_table_run_id(block, pipeline_sha256=pipeline_sha256)
    if owned is not None:
        return owned
    return table_run_id(next_table_run_order(block), pipeline_sha256)


def import_table_run_id(block: Mapping[str, object]) -> str:
    """The run folder ``--mode migrate``'s legacy-Parquet import owns in *block* (D10).

    :func:`choose_table_run_id` for :func:`v1_table_pipeline_sha`, the rule
    that names a converted v1 folder, over a v2 *block* (any other layout
    numbers from scratch, so a v1 store's import lands in ``1-<sha12>``,
    the folder its conversion would have made). One function, because the
    import's writer and migrate's checks of what it wrote must name the same
    run: two rules here are what kept a multi-run store from converging
    (review M2).

    Raises:
        ValueError: If the tables descriptor is corrupt or of a version this
            build does not read.
    """
    return choose_table_run_id(
        block if tables_schema_of_block(block) == 2 else {},
        pipeline_sha256=v1_table_pipeline_sha(block),
    )


def owned_table_run_id(
    block: Mapping[str, object], *, pipeline_sha256: str | None
) -> str | None:
    """The existing run folder *pipeline_sha256* owns in *block*, or ``None``.

    Rules 1 and 2 of :func:`choose_table_run_id`, without rule 3's new
    folder: the highest-numbered run with this digest, or, for ``None``, the
    highest-numbered ``-unknown`` run.
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
    return owned[-1] if owned else None


def _last_application_is_legacy(journal: object) -> bool:
    """Whether *journal*'s last application is a converted ``legacy`` one."""
    applications = journal.get("applications") if isinstance(journal, Mapping) else None
    return (
        isinstance(applications, list)
        and bool(applications)
        and isinstance(applications[-1], Mapping)
        and applications[-1].get("kind") == "legacy"
    )


def journal_table_run(store: Path) -> TableRun | None:
    """The run folder a forward save of *store*'s own journal wrote (spec D6, D6a).

    The writer picks its folder with :func:`choose_table_run_id` for the
    digest of the image's last journal application
    (:func:`~phenotypic._core._provenance.current_pipeline_sha256`), and
    persists that same journal in the root it writes last. So reading the
    digest back from the promoted root and applying the same ownership rule
    names exactly the folder the save wrote -- which is what a forward
    per-image record (full mode, staged Stage 3) certifies, and what its
    resume probes ask about. A reader holding a record uses the record
    instead; a reader with neither uses :func:`latest_measurement_run`.

    It is **not** the folder a ``--mode measure`` rewrite wrote: measure mode
    opens no journal application, so its pipeline is not the journal's. That
    rewrite returns the run it wrote, and its record declares that.

    **A ``legacy`` last application reads the folder spec D10 labelled.**
    ``--mode migrate`` converts a v1 store's tables into ``1-<sha12>`` for
    :func:`v1_table_pipeline_sha` (the latest figure run, else the latest
    ``full``/``legacy`` application's digest), while the writer's rule
    (:func:`~phenotypic._core._provenance.journal_pipeline_sha256`, spec D2)
    treats a ``legacy`` application as unknown. Applying the writer's rule
    here would name ``-unknown`` for a folder migrate named after a digest,
    so the staged and full resume probes would read a migrated image as
    having no table of its own and re-derive it. The writer's rule is
    unchanged: a save over a ``legacy`` journal still owns ``-unknown``
    (review OQ3).

    Raises:
        FileNotFoundError: If the store has no root ``zarr.json``.
        KeyError: If the root carries no ``phenotypic`` block.
        ValueError: If the store or its tables descriptor is unreadable.

    Returns:
        The run, or ``None`` when the store holds no folder for that digest
        (no tables yet, or the flat v1 layout).
    """
    from phenotypic._core._provenance import journal_pipeline_sha256

    from . import ngff_

    block = _read_block(store)
    journal = block.get(ngff_.PhenotypicAttr.PROVENANCE)
    pipeline_sha256 = (
        v1_table_pipeline_sha(block)
        if _last_application_is_legacy(journal)
        else journal_pipeline_sha256(journal)
    )
    run_id = owned_table_run_id(block, pipeline_sha256=pipeline_sha256)
    return None if run_id is None else _run_of_block(store, block, run_id)


def journal_measurement_table(store: Path) -> Path | None:
    """The measurements table of :func:`journal_table_run`, if it is on disk.

    The probe form for resume classification, which must degrade rather than
    raise: an absent or unreadable store, a run without ``measurements`` and
    a missing file all answer ``None`` -- "this image has no table of its
    own yet", which routes it back through processing.
    """
    try:
        run = journal_table_run(store)
    except (FileNotFoundError, KeyError, ValueError):
        return None
    if run is None or not run.has_measurements:
        return None
    path = run.measurements_path()
    return path if path.is_file() else None


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


# ---------------------------------------------------------------------------
# The upstream reference (spec D2, D8)
# ---------------------------------------------------------------------------

#: Keys of one ``upstream`` entry (spec D3).
SOURCE_KEY: Final[str] = "source"
ROOT_SHA256_KEY: Final[str] = "root_sha256"


class UpstreamChangedError(RuntimeError):
    """An upstream reference no longer matches the input it was recorded from (D8).

    The input's root ``zarr.json`` has a different sha256 from the one the
    result store recorded when it read that input -- for example after
    AutoConvertRaw-GC reprocessed it in place. The run the reference names
    may still exist, but it is no longer the run the result was derived from.

    Attributes:
        run_id: The upstream run id that was followed.
        expected: The ``root_sha256`` the result store recorded.
        actual: The sha256 of the input's root ``zarr.json`` now.
    """

    def __init__(self, run_id: str, expected: str, actual: str) -> None:
        self.run_id = run_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"upstream changed: table run {run_id!r} was recorded from an input "
            f"whose root zarr.json had sha256 {expected}, and it now has {actual}"
        )


def _upstream_of_block(block: Mapping[str, object]) -> dict[str, dict]:
    """The ``upstream`` entries of a v2 *block*; ``{}`` for v0/v1 or none."""
    if tables_schema_of_block(block) != 2:
        return {}
    _upstream_orders(block)  # validates the keys
    tables = _tables_block(block)
    assert tables is not None
    upstream = tables.get(UPSTREAM_KEY, {})
    assert isinstance(upstream, Mapping)
    return {str(key): dict(value) for key, value in upstream.items()}


def _input_relative_source(input_path: Path, input_root: Path | None) -> str:
    """*input_path* relative to *input_root*, as the CLI's work identity spells it.

    The same rule as the work id's relative path: the file name when there is
    no root, when the root is a file, when the paths are unrelated, or when
    the input is its own root (a direct store input).
    """
    source = Path(input_path)
    if input_root is None or Path(input_root).is_file():
        return source.name
    try:
        relative = source.relative_to(Path(input_root))
    except ValueError:
        return source.name
    return source.name if relative == Path(".") else relative.as_posix()


def _parse_root_bytes(raw: bytes) -> Mapping[str, object] | None:
    """The ``attributes.phenotypic`` block of root bytes *raw*, or ``None`` when absent."""
    import json

    from . import ngff_

    payload = json.loads(raw.decode("utf-8"))
    attributes = payload.get("attributes") if isinstance(payload, Mapping) else None
    block = (
        attributes.get(ngff_.PhenotypicAttr.ROOT)
        if isinstance(attributes, Mapping)
        else None
    )
    return block if isinstance(block, Mapping) else None


def upstream_entries_for_input(
    input_path: Path, input_root: Path | None
) -> dict[str, dict]:
    """The ``upstream`` entries a store derived from *input_path* records (spec D8).

    When the input is a PhenoTypic store with a v2 ``tables`` descriptor, each
    of its run folders becomes one entry::

        {run_id: {"source": <input path relative to input_root>,
                  "root_sha256": <sha256 of the input's root zarr.json>,
                  "pipeline_sha256": <the run's pipeline digest, or None>}}

    merged with the input's own ``upstream`` entries, so the chain is complete.
    An inherited entry is kept **verbatim**: its ``source`` and
    ``root_sha256`` name the store that holds that run, relative to the
    ``--input`` of the run that first recorded it, not to *input_root*.
    Nothing is copied; the caller passes the result to the store write as
    ``upstream=``, which also counts it when numbering a new folder (D2).

    The digest and the descriptor come from **one** read of the root, so
    ``root_sha256`` certifies exactly the runs recorded beside it.

    Args:
        input_path: The image the result store is derived from: a store
            directory, or any other image file.
        input_root: The run's ``--input``. ``None`` (a worker given no input
            root) records the input's file name as ``source``.

    Returns:
        The entries, or ``{}`` for a non-store input, a store with no
        PhenoTypic block (a third-party OME-Zarr), and a store whose tables
        are absent or in the v1 flat layout.

    Raises:
        ValueError: If the input's tables descriptor is corrupt or a version
            this build does not read.
    """
    import hashlib

    from . import ngff_

    store = Path(input_path)
    root = store / ngff_.STORE_ROOT_JSON
    if not store.name.endswith(ngff_.STORE_SUFFIX) or not root.is_file():
        return {}
    raw = root.read_bytes()
    block = _parse_root_bytes(raw)
    if block is None or tables_schema_of_block(block) != 2:
        return {}
    entries = _upstream_of_block(block)
    runs = table_runs_of_block(store, block)
    if not runs:
        return entries
    source = _input_relative_source(store, input_root)
    root_sha256 = hashlib.sha256(raw).hexdigest()
    for run in runs:
        entries[run.run_id] = {
            SOURCE_KEY: source,
            ROOT_SHA256_KEY: root_sha256,
            PIPELINE_SHA256_KEY: run.pipeline_sha256,
        }
    return entries


def _resolve_upstream_source(input_root: Path, source: str) -> Path:
    """The store an upstream ``source`` names under *input_root*.

    ``input_root / source``; or *input_root* itself when it is the store
    (a direct store input records its own name as ``source``).

    Raises:
        FileNotFoundError: If neither is a store with a root ``zarr.json``.
    """
    from . import ngff_

    root = Path(input_root)
    candidate = root / PurePosixPath(source)
    if (candidate / ngff_.STORE_ROOT_JSON).is_file():
        return candidate
    if root.name == source and (root / ngff_.STORE_ROOT_JSON).is_file():
        return root
    raise FileNotFoundError(
        f"upstream input {source!r} is not reachable under {root}: "
        f"no store root at {candidate / ngff_.STORE_ROOT_JSON}"
    )


def follow_upstream(store: Path, run_id: str, *, input_root: Path) -> TableRun:
    """Follow *store*'s upstream reference *run_id* to the input run it names (D8).

    Args:
        store: The result store holding the reference.
        run_id: A key of its ``tables.upstream``.
        input_root: The ``--input`` the reference's ``source`` is relative
            to: the result run's own for an entry it recorded, or the earlier
            run's for an entry inherited through the chain.

    Returns:
        The input's run folder.

    Raises:
        KeyError: If *store* records no upstream entry *run_id*.
        FileNotFoundError: If the input is unreachable, or *store* has no
            root ``zarr.json``.
        UpstreamChangedError: If the input's root ``zarr.json`` no longer has
            the recorded ``root_sha256``.
        ValueError: If the entry is malformed, or the unchanged input does not
            declare *run_id* (a corrupt reference).
    """
    import hashlib

    from . import ngff_

    entry = _upstream_of_block(_read_block(store)).get(run_id)
    if entry is None:
        raise KeyError(f"{store} records no upstream table run {run_id!r}")
    source = entry.get(SOURCE_KEY)
    expected = entry.get(ROOT_SHA256_KEY)
    if not isinstance(source, str) or not source or not _is_sha256(expected):
        raise ValueError(f"upstream entry {run_id!r} of {store} is malformed: {entry!r}")
    assert isinstance(expected, str)
    input_store = _resolve_upstream_source(Path(input_root), source)
    raw = (input_store / ngff_.STORE_ROOT_JSON).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise UpstreamChangedError(run_id, expected, actual)
    run = _run_of_block(input_store, _parse_root_bytes(raw) or {}, run_id)
    if run is None:
        raise ValueError(
            f"upstream input {input_store} is unchanged but declares no table "
            f"run {run_id!r}: the reference in {store} is corrupt"
        )
    return run
