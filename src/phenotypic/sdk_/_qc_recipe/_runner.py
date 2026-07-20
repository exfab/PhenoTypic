"""Build the ``deliverables/qc/qc.duckdb`` artifact from a pipeline's QC config.

:func:`run_qc` is the single, GUI-free seam that turns a measurement frame
plus a pipeline's ``qc`` entries into the on-disk QC analysis database the
results-viewer Review + Error tabs read. It is called in two places with
identical semantics:

* the CLI's ``finalize_post_master_outputs`` (once per recompile/remeasure,
  on the post-applied + metadata-joined frame), and
* the GUI's after-each-group recompute (in-process, on the curated frame).

It is **pure with respect to review progress**: it writes ``qc.duckdb`` and
**never** touches ``review_state.json`` (that file is GUI-owned; the CLI
finalize path resets it separately). It also never writes
``measurements.parquet`` — only the QC database.

Atomic full rebuild
-------------------
``run_qc`` is always an **atomic full rebuild**: it builds a fresh DuckDB
into a unique sibling ``*.tmp`` file and then ``os.replace``-s it over the
canonical ``qc.duckdb`` so a reader never observes a partial database. The
replace is wrapped in a bounded retry on :class:`PermissionError` for the
Windows open-handle race. When the pipeline has no *enabled* QC entries (or
entries exist but none analyze successfully), ``run_qc`` removes any existing
canonical ``qc.duckdb`` so readers do not see stale modules from a prior run.

Database schema
---------------
``qc_modules`` — the catalog, one row per enabled+built module: the 15
:class:`~phenotypic.analysis.abc_.QcTableSpec` role fields plus
``table_name``, ``summary_table``, ``ordinal``, and a JSON ``params``
snapshot. Any consumer can render a module generically from this row.

``<table_name>`` — the module's self-describing data table
(``QualityCheck.to_table()``); columns vary per check.

``<table_name>__summary`` — the per-module worklist (one row per group):
``[*groupby, n_members, n_flagged, metric, status, flag, rank]`` where
``rank`` is worst-first within the module (0 = worst).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pandas as pd

if TYPE_CHECKING:
    from phenotypic._core._image_pipeline import ImagePipeline
    from phenotypic.analysis.abc_ import QcTableSpec, QualityCheck

from phenotypic.sdk_ import QC_DUCKDB, qc_duckdb_path

from ._recipe import QcRecipeEntry

logger = logging.getLogger(__name__)

#: Status severity ranking used to order the worklist worst-first.
_STATUS_RANK: dict[str, int] = {"pass": 0, "warn": 1, "fail": 2}

#: Matches any run of characters that are illegal in a DuckDB identifier.
_IDENT_UNSAFE = re.compile(r"[^a-z0-9]+")

#: ``qc_modules`` catalog DDL. Exactly 19 columns; the positional INSERT in
#: :func:`_insert_catalog_row` supplies 19 values in this same order
#: (the 15 :class:`QcTableSpec` fields + ``table_name``/``summary_table``/
#: ``ordinal``/``params``).
_CATALOG_DDL = """
CREATE TABLE qc_modules (
    instance_id TEXT, class TEXT, name TEXT,
    table_name TEXT, summary_table TEXT, ordinal INTEGER,
    groupby_cols TEXT, metric_col TEXT, status_col TEXT, flag_col TEXT,
    on_col TEXT, member_key_cols TEXT, supports_object_curation BOOLEAN,
    time_col TEXT, higher_is_bad BOOLEAN, extra_cols TEXT, params TEXT,
    warn_threshold DOUBLE, fail_threshold DOUBLE
)
"""

#: Number of columns in :data:`_CATALOG_DDL` (kept in lock-step with the
#: positional INSERT placeholder count).
_CATALOG_NCOLS: int = 19

_WRITER_LOCKS_GUARD = threading.Lock()
_WRITER_LOCKS: dict[Path, threading.Lock] = {}


@dataclass(frozen=True)
class SuccessfulQcModule:
    """One analyzed check whose tables were published successfully."""

    instance_id: str
    check: "QualityCheck"
    table_spec: "QcTableSpec"


def run_qc(
    measurements_df: pd.DataFrame,
    pipeline: "ImagePipeline",
    output_dir: Path,
    *,
    qc_output_dir: Path | None = None,
) -> list[SuccessfulQcModule]:
    """Run enabled QC checks and atomically (re)build ``qc.duckdb``.

    Always a FULL rebuild: a temp DB is built then ``os.replace``-d over the
    canonical path so readers never see a partial DB. Removes any stale
    canonical database when the pipeline has no enabled QC entries (or when
    entries exist but none analyze). Each enabled+built check becomes one
    ``qc_modules`` catalog row plus a
    ``<table_name>`` data table and a ``<table_name>__summary`` worklist.

    Tolerant: a check that fails to instantiate, analyze, *or* ingest into
    DuckDB (its ``CREATE``/``INSERT``) is skipped with a WARNING, never
    aborting the rest of the rebuild — the resulting DB carries the good
    modules. When *no* module ingests cleanly, the staging ``.tmp`` and any
    stale canonical DB are removed; any failure before the atomic swap also
    removes the ``.tmp`` so a failed rebuild never leaves a stale staging file
    behind.

    Args:
        measurements_df: The frame to evaluate. The CLI passes the
            post-applied + metadata-joined frame (``measurements.parquet``,
            converted to pandas), since QC ``groupby`` columns often come
            from joined metadata. A pandas DataFrame — not polars.
        pipeline: The pipeline whose ``qc`` entries (via
            :meth:`ImagePipeline.get_qc`) define the checks to run.
        output_dir: Run output root; ``qc.duckdb`` is resolved under it via
            :func:`~phenotypic.sdk_.qc_duckdb_path` when ``qc_output_dir`` is
            not supplied (the CLI path).
        qc_output_dir: Explicit, already-resolved QC directory to write into
            (e.g. ``output_root.layout.qc_dir`` from the GUI). When provided,
            ``qc.duckdb`` is written directly under it instead of via
            ``qc_duckdb_path(output_dir)`` — this is what keeps a standalone
            deliverables bundle (where ``output_dir`` is the deliverables
            folder) from double-joining ``deliverables/``.

    Side effects:
        Writes or removes ``deliverables/qc/qc.duckdb`` (via a unique
        ``.tmp`` + atomic replace for successful rebuilds). Never touches
        ``review_state.json`` or ``measurements.parquet``.

    Returns:
        Successful modules with the exact analyzed check instances reused for
        plotting. Modules that failed table publication are not returned.
    """
    output_dir = Path(output_dir)
    if qc_output_dir is not None:
        target = Path(qc_output_dir) / QC_DUCKDB
    else:
        target = qc_duckdb_path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    with _writer_lock(target):
        entries = [e for e in pipeline.get_qc() if e.enabled]
        if not entries:
            logger.debug(
                "No enabled QC entries; removing stale qc.duckdb if present"
            )
            _remove_qc_db_if_present(target)
            return []

        tmp = _qc_temp_db_path(target)

        # Thread `entry` alongside the built frames so the catalog row carries the
        # real params snapshot (QcTableSpec does not hold params).
        built: list[
            tuple[
                QcRecipeEntry,
                "QualityCheck",
                "QcTableSpec",
                pd.DataFrame,
                pd.DataFrame,
            ]
        ] = []
        for entry in entries:
            result = _run_one_check(entry, measurements_df)
            if result is not None:
                check, spec, table_df, summary_df = result
                built.append((entry, check, spec, table_df, summary_df))

        if not built:
            logger.info(
                "No QC check produced a table; removing stale qc.duckdb"
            )
            _remove_qc_db_if_present(target)
            return []

        # Build the temp DB; on *any* failure (catalog DDL, the per-module
        # ingest, or the atomic swap) never leave a stale ``.tmp`` behind.
        try:
            written = 0
            successful: list[SuccessfulQcModule] = []
            con = duckdb.connect(str(tmp))
            try:
                _create_catalog(con)
                for ordinal, (
                    entry,
                    check,
                    spec,
                    table_df,
                    summary_df,
                ) in enumerate(
                    built
                ):
                    if _create_module_tables(
                        con, entry, spec, table_df, summary_df, ordinal
                    ):
                        written += 1
                        successful.append(
                            SuccessfulQcModule(
                                instance_id=entry.instance_id,
                                check=check,
                                table_spec=spec,
                            )
                        )
            finally:
                con.close()

            if written == 0:
                logger.info(
                    "No QC module ingested cleanly; removing stale qc.duckdb"
                )
                _remove_tmp_if_present(tmp)
                _remove_qc_db_if_present(target)
                return []

            _atomic_replace_with_retry(tmp, target)
        except Exception:
            _remove_tmp_if_present(tmp)
            raise

    logger.info("Wrote QC DuckDB for %d module(s) -> %s", written, target)
    return successful


def _writer_lock(target: Path) -> threading.Lock:
    resolved = target.resolve()
    with _WRITER_LOCKS_GUARD:
        lock = _WRITER_LOCKS.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _WRITER_LOCKS[resolved] = lock
        return lock


def _qc_temp_db_path(target: Path) -> Path:
    return target.with_name(
        f"{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )


def _remove_tmp_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _remove_qc_db_if_present(path: Path) -> None:
    for attempt in range(5):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _create_module_tables(
    con: duckdb.DuckDBPyConnection,
    entry: QcRecipeEntry,
    spec: "QcTableSpec",
    table_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    ordinal: int,
) -> bool:
    """Create one module's data + summary tables and catalog row.

    Tolerant: any failure to ingest this module's frames (a ``CREATE``/
    ``INSERT`` DuckDB rejects) is logged at WARNING and yields ``False`` so
    the offending module is skipped without aborting the rest of the
    rebuild — mirroring the tolerant-skip contract of
    :func:`_run_one_check`. A partial-but-valid DB carrying only the good
    modules is the accepted outcome (any orphaned data table left by a
    half-built module is unreferenced by ``qc_modules`` and invisible to
    the catalog-driven readers).

    Args:
        con: An open DuckDB connection to the temp database.
        entry: The recipe entry (its params snapshot lands in the catalog).
        spec: The module's catalog descriptor.
        table_df: The self-describing data table frame.
        summary_df: The per-module summary worklist frame.
        ordinal: Zero-based display/run order.

    Returns:
        ``True`` when the module's tables + catalog row were written;
        ``False`` when the module was skipped.
    """
    tname = _safe_table_name(spec.instance_id)
    stname = f"{tname}__summary"
    try:
        con.register("_qc_data", table_df)
        con.execute(f'CREATE TABLE "{tname}" AS SELECT * FROM _qc_data')
        con.unregister("_qc_data")
        con.register("_qc_summary", summary_df)
        con.execute(f'CREATE TABLE "{stname}" AS SELECT * FROM _qc_summary')
        con.unregister("_qc_summary")
        _insert_catalog_row(
            con,
            spec,
            tname,
            stname,
            ordinal,
            params=entry.to_dict()["params"],
        )
    except Exception as exc:  # noqa: BLE001 - tolerant; surfaced as warning
        logger.warning(
            "Skipping QC module %s (%s): table build failed: %s",
            spec.instance_id,
            spec.cls_name,
            exc,
        )
        return False
    return True


def _run_one_check(
    entry: QcRecipeEntry,
    measurements_df: pd.DataFrame,
) -> tuple[
    "QualityCheck", "QcTableSpec", pd.DataFrame, pd.DataFrame
] | None:
    """Instantiate + analyze one entry, returning its catalog spec + frames.

    Tolerant: any construction or analysis/build failure is logged at
    WARNING and yields ``None`` so the offending check is skipped without
    aborting the whole rebuild.

    Args:
        entry: The QC config entry to run.
        measurements_df: The frame to analyze.

    Returns:
        ``(spec, table_df, summary_df)`` for this instance — the
        :class:`QcTableSpec` catalog descriptor, the self-describing data
        table (:meth:`QualityCheck.to_table`), and the per-module summary
        worklist — or ``None`` when the check could not be built or analyzed.
    """
    try:
        check = entry.instantiate()
    except Exception as exc:  # noqa: BLE001 - tolerant; surfaced as warning
        logger.warning(
            "Skipping QC check %s (%s): instantiation failed: %s",
            entry.instance_id,
            entry.cls.__name__,
            exc,
        )
        return None

    try:
        check.analyze(measurements_df)
        table_df = check.to_table()
        spec = check.table_spec(entry.instance_id)
        summary_df = _build_summary_frame(check)
    except Exception as exc:  # noqa: BLE001 - tolerant; surfaced as warning
        logger.warning(
            "Skipping QC check %s (%s): analyze/build failed: %s",
            entry.instance_id,
            entry.cls.__name__,
            exc,
        )
        return None

    return check, spec, table_df, summary_df


def _build_summary_frame(check: "QualityCheck") -> pd.DataFrame:
    """Map ``check.summary()`` into the per-module worklist frame.

    ``QualityCheck.summary()`` returns ``[*groupby, qc_n_members,
    qc_n_flagged, qc_worst_metric, qc_status]``. This renames those into the
    catalog-agnostic ``metric``/``status``/``n_members``/``n_flagged``
    columns (the instance/class columns now live in ``qc_modules``, not in
    every summary row), derives a group-level ``flag`` (any member flagged),
    and assigns a worst-first ``rank`` within the module via
    :func:`_rank_worst_first`.

    Args:
        check: The analyzed check instance.

    Returns:
        A frame with columns ``[*groupby, n_members, n_flagged, metric,
        status, flag, rank]``.
    """
    raw = check.summary()
    out = raw.rename(
        columns={
            "qc_worst_metric": "metric",
            "qc_status": "status",
            "qc_n_members": "n_members",
            "qc_n_flagged": "n_flagged",
        }
    )
    out["flag"] = out["n_flagged"].astype(int) > 0
    out["rank"] = _rank_worst_first(
        out["metric"], out["status"], higher_is_bad=check._HIGHER_IS_BAD
    )
    return out


def _rank_worst_first(
    metric: pd.Series,
    status: pd.Series,
    *,
    higher_is_bad: bool,
) -> pd.Series:
    """Assign a dense 0-based worst-first rank within one instance.

    Ordering keys, worst first:

    1. status severity (``fail`` > ``warn`` > ``pass``);
    2. the bad-direction extremity of ``metric`` — descending when
       higher-is-bad, ascending when lower-is-bad;

    ``NaN`` metrics always sort last regardless of direction (an
    under-powered / degenerate bin never leads the worklist). ``inf``
    metrics (e.g. an unmatched-group ``Count``) are finite-ordered above any
    real value in the higher-is-bad direction, which is the intended
    "worst" placement.

    Args:
        metric: Per-group worst-direction metric values.
        status: Per-group worst status labels.
        higher_is_bad: The check's ``_HIGHER_IS_BAD`` direction.

    Returns:
        An integer ``Series`` (aligned to the inputs) of ranks ``0..n-1``,
        worst first.
    """
    metric = pd.to_numeric(metric, errors="coerce")
    severity = status.map(_STATUS_RANK).fillna(0).astype(int)

    # Build a sort frame so we can express "NaN last" independently of the
    # metric direction (``sort_values(na_position="last")`` only applies to
    # the last sort key, so isolate NaN-ness into its own leading key).
    order = pd.DataFrame(
        {
            "_nan_last": metric.isna().astype(int),  # 0 = finite first
            "_severity": -severity,  # higher severity first
            "_metric": (-metric if higher_is_bad else metric),
        }
    )
    sorted_index = order.sort_values(
        ["_nan_last", "_severity", "_metric"],
        kind="mergesort",  # stable: preserves group order on exact ties
    ).index

    ranks = pd.Series(range(len(sorted_index)), index=sorted_index)
    return ranks.reindex(metric.index).astype(int)


def _safe_table_name(instance_id: str) -> str:
    """Return a deterministic, DuckDB-identifier-safe table name.

    Lowercases, replaces any run of non-alphanumerics with ``_``, strips
    leading/trailing ``_``, and prefixes ``qc_`` so the result always starts
    with a letter. Instance ids are already unique (``qc-<name>-<8hex>``), so
    the lowercased, underscore-joined form stays unique.

    Args:
        instance_id: The recipe entry id.

    Returns:
        A valid table identifier (e.g. ``qc_se_1a2b3c4d``).
    """
    core = _IDENT_UNSAFE.sub("_", instance_id.lower()).strip("_")
    return core if core.startswith("qc_") else f"qc_{core}"


def _create_catalog(con: duckdb.DuckDBPyConnection) -> None:
    """Create the empty ``qc_modules`` catalog table.

    Args:
        con: An open DuckDB connection to the temp database.
    """
    con.execute(_CATALOG_DDL)


def _insert_catalog_row(
    con: duckdb.DuckDBPyConnection,
    spec: "QcTableSpec",
    tname: str,
    stname: str,
    ordinal: int,
    params: dict[str, Any],
) -> None:
    """Insert one module's catalog row into ``qc_modules``.

    The positional value list MUST match :data:`_CATALOG_DDL`'s column order
    exactly (the 15 :class:`QcTableSpec` fields then ``table_name``,
    ``summary_table``, ``ordinal``, ``params`` interleaved per the DDL).
    List-valued spec fields and the params dict are JSON-encoded into their
    ``TEXT`` columns.

    Args:
        con: An open DuckDB connection to the temp database.
        spec: The module's catalog descriptor.
        tname: The module's data table name.
        stname: The module's summary table name.
        ordinal: Zero-based display/run order.
        params: The entry's JSON-native params snapshot.
    """
    placeholders = ",".join(["?"] * _CATALOG_NCOLS)
    con.execute(
        f"INSERT INTO qc_modules VALUES ({placeholders})",
        [
            spec.instance_id,
            spec.cls_name,
            spec.name,
            tname,
            stname,
            ordinal,
            json.dumps(spec.groupby_cols),
            spec.metric_col,
            spec.status_col,
            spec.flag_col,
            spec.on_col,
            json.dumps(spec.member_key_cols),
            spec.supports_object_curation,
            spec.time_col,
            spec.higher_is_bad,
            json.dumps(spec.extra_cols),
            json.dumps(params),
            spec.warn_threshold,
            spec.fail_threshold,
        ],
    )


def _atomic_replace_with_retry(
    tmp: Path, target: Path, attempts: int = 5
) -> None:
    """``os.replace`` *tmp* onto *target* with a bounded Windows retry.

    ``os.replace`` is atomic on POSIX. On Windows it can raise
    :class:`PermissionError` when a reader's handle transiently overlaps the
    swap; this retries with a brief linear backoff before re-raising.

    Args:
        tmp: The freshly-built temp database path.
        target: The canonical ``qc.duckdb`` path to swap into place.
        attempts: Maximum number of replace attempts.
    """
    for i in range(attempts):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            # A reader handle overlaps the swap (Windows). Brief backoff.
            time.sleep(0.1 * (i + 1))
