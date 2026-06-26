"""Catalog-driven DuckDB read API for the QC Review + Error tabs.

Connections are short-lived and ``read_only`` — opened per query and closed
immediately, never held across Dash callbacks (Windows ``os.replace`` +
single DuckDB writer). Returns polars frames / plain dataclasses; Dash-free.

The ``qc.duckdb`` database (written by
:func:`phenotypic.sdk_._qc_recipe._runner.run_qc`) holds a ``qc_modules``
catalog (one row per enabled+built module) plus, per module, a
self-describing ``<table_name>`` data table and a ``<table_name>__summary``
worklist. Every read here is catalog-driven: :func:`list_modules` reads the
catalog, then :func:`module_summary` / :func:`module_members` resolve a
module's tables from its :class:`QcModule` descriptor. A missing/corrupt DB
degrades to an empty module list (and empty frames), so the Review + Error
tabs render an empty worklist rather than raising.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import duckdb
import polars as pl

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QcModule:
    """One ``qc_modules`` catalog row: a module's self-describing descriptor.

    Mirrors the catalog columns so any consumer can render a module's
    worklist + detail gallery generically, without hard-coding its schema.

    Attributes:
        instance_id: The recipe entry id (``qc-<name>-<hex>``).
        cls_name: The ``QualityCheck`` subclass name.
        name: The check's short ``name`` (e.g. ``"ZMax"``).
        table_name: The module's data table identifier in ``qc.duckdb``.
        summary_table: The module's ``<table>__summary`` worklist table.
        groupby_cols: Ordered group-key column names.
        metric_col / status_col / flag_col: The generic QC column names.
        member_key_cols: Per-object curation-key columns (``[]`` when the
            module has no per-object key).
        supports_object_curation: Whether the module's rows map to curatable
            detected objects (``False`` for diagnostic-only modules — the
            Review tab hides the curation radial + tile gallery, and the
            Error tab's verified-good skips them).
        time_col: Time-course facet column, or ``None``.
        higher_is_bad: The check's ``_HIGHER_IS_BAD`` direction.
        extra_cols: Check-specific columns beyond the generic trio.
    """

    instance_id: str
    cls_name: str
    name: str
    table_name: str
    summary_table: str
    groupby_cols: list[str]
    metric_col: str
    status_col: str
    flag_col: str
    member_key_cols: list[str]
    supports_object_curation: bool
    time_col: str | None
    higher_is_bad: bool
    extra_cols: list[str]


def open_qc_db(output_root: "OutputRoot") -> duckdb.DuckDBPyConnection | None:
    """Open a short-lived ``read_only`` connection, or ``None`` when absent.

    Args:
        output_root: The active results-viewer output root (provides the
            resolved ``layout.qc_duckdb`` path).

    Returns:
        An open ``read_only`` DuckDB connection the caller MUST close, or
        ``None`` when ``qc.duckdb`` is missing or cannot be opened (a
        corrupt/locked DB is non-fatal — the worklist degrades to empty).
    """
    path = output_root.layout.qc_duckdb
    if not path.is_file():
        return None
    try:
        return duckdb.connect(str(path), read_only=True)
    except Exception:  # noqa: BLE001 - a corrupt/locked DB is non-fatal
        logger.warning("Failed to open QC DuckDB %s", path, exc_info=True)
        return None


def list_modules(output_root: "OutputRoot") -> list[QcModule]:
    """Return the catalog's modules, ordered by ``ordinal`` (recipe order).

    Args:
        output_root: The active output root.

    Returns:
        One :class:`QcModule` per ``qc_modules`` row, in run order. Empty
        when ``qc.duckdb`` is absent / unreadable.
    """
    con = open_qc_db(output_root)
    if con is None:
        return []
    try:
        rows = con.execute(
            "SELECT instance_id, class, name, table_name, summary_table, "
            "groupby_cols, metric_col, status_col, flag_col, member_key_cols, "
            "supports_object_curation, time_col, higher_is_bad, extra_cols "
            "FROM qc_modules ORDER BY ordinal"
        ).fetchall()
    except Exception:  # noqa: BLE001 - a corrupt catalog is non-fatal
        logger.warning("Failed to read qc_modules catalog", exc_info=True)
        return []
    finally:
        con.close()
    return [
        QcModule(
            instance_id=r[0],
            cls_name=r[1],
            name=r[2],
            table_name=r[3],
            summary_table=r[4],
            groupby_cols=json.loads(r[5]),
            metric_col=r[6],
            status_col=r[7],
            flag_col=r[8],
            member_key_cols=json.loads(r[9]),
            supports_object_curation=bool(r[10]),
            time_col=r[11],
            higher_is_bad=bool(r[12]),
            extra_cols=json.loads(r[13]),
        )
        for r in rows
    ]


def _module(output_root: "OutputRoot", instance_id: str) -> QcModule | None:
    """Return the catalog descriptor for ``instance_id``, or ``None``."""
    return next(
        (m for m in list_modules(output_root) if m.instance_id == instance_id),
        None,
    )


def module_summary(output_root: "OutputRoot", instance_id: str) -> pl.DataFrame:
    """Return a module's worklist (``<table>__summary``), worst-first.

    Args:
        output_root: The active output root.
        instance_id: The module to slice.

    Returns:
        The module's summary rows ordered ascending by ``rank`` (0 = worst,
        NaN ranks last). Empty when the module / DB is absent.
    """
    mod = _module(output_root, instance_id)
    con = open_qc_db(output_root)
    if mod is None or con is None:
        if con is not None:
            con.close()
        return pl.DataFrame()
    try:
        return con.execute(
            f'SELECT * FROM "{mod.summary_table}" ORDER BY rank NULLS LAST'
        ).pl()
    finally:
        con.close()


def module_members(
    output_root: "OutputRoot", instance_id: str, group_key: tuple
) -> pl.DataFrame:
    """Return a module's data rows for one group (or all rows for ``()``).

    The data table (``QualityCheck.to_table()``) is read whole then filtered
    in polars on the module's ``groupby_cols`` zipped with ``group_key`` —
    null/NaN-safe. Passing an empty ``group_key`` applies no filter and
    returns the full data table.

    Args:
        output_root: The active output root.
        instance_id: The module to slice.
        group_key: The group-key value tuple aligned to the module's
            ``groupby_cols`` (``()`` → no filter, the whole table).

    Returns:
        The (optionally filtered) data frame. Empty when the module / DB is
        absent.
    """
    mod = _module(output_root, instance_id)
    con = open_qc_db(output_root)
    if mod is None or con is None:
        if con is not None:
            con.close()
        return pl.DataFrame()
    try:
        frame = con.execute(f'SELECT * FROM "{mod.table_name}"').pl()
    finally:
        con.close()
    # Filter in polars (avoids dynamic SQL on column names); null/NaN-safe.
    for col, val in zip(mod.groupby_cols, group_key):
        if col not in frame.columns:
            continue
        if val is None or (isinstance(val, float) and math.isnan(val)):
            frame = frame.filter(pl.col(col).is_null())
        else:
            frame = frame.filter(pl.col(col).cast(pl.String) == str(val))
    return frame


def summary_stats(module_summary: pl.DataFrame) -> dict[str, Any]:
    """Compute the summary-header stat tiles for one module's worklist.

    Distinguishes **insufficient/NaN** groups (a metric that could not be
    computed — e.g. an ICC on a sparse grid) from genuine ``pass`` so the
    header never paints a no-signal group green (spec §D risk refinement).
    The median metric is **robust**: computed over finite values only, so
    a single ``inf`` (an unmatched-group ``Count``) or a swarm of ``NaN``
    cannot poison it.

    Args:
        module_summary: One module's summary rows (any order).

    Returns:
        A dict with integer counts ``total``/``fail``/``warn``/``pass``/
        ``insufficient`` plus ``colonies_removed`` placeholder ``0`` (the
        caller fills it from the live removal set) and ``median_metric``
        (``float`` or ``None`` when no finite metric exists).
    """
    total = int(module_summary.height)
    if total == 0:
        return {
            "total": 0,
            "fail": 0,
            "warn": 0,
            "pass": 0,
            "insufficient": 0,
            "colonies_removed": 0,
            "median_metric": None,
        }

    metrics = module_summary.get_column("metric").to_list()
    statuses = module_summary.get_column("status").to_list()

    fail = warn = passed = insufficient = 0
    finite_metrics: list[float] = []
    for metric, status in zip(metrics, statuses):
        is_nan = metric is None or (
            isinstance(metric, float) and math.isnan(metric)
        )
        if is_nan:
            insufficient += 1
        elif status == "fail":
            fail += 1
        elif status == "warn":
            warn += 1
        else:
            passed += 1
        if (
            metric is not None
            and isinstance(metric, (int, float))
            and math.isfinite(metric)
        ):
            finite_metrics.append(float(metric))

    median_metric = _robust_median(finite_metrics)
    return {
        "total": total,
        "fail": fail,
        "warn": warn,
        "pass": passed,
        "insufficient": insufficient,
        "colonies_removed": 0,
        "median_metric": median_metric,
    }


def _robust_median(values: list[float]) -> float | None:
    """Median over finite values only; ``None`` when the list is empty."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


__all__ = [
    "QcModule",
    "open_qc_db",
    "list_modules",
    "module_summary",
    "module_members",
    "summary_stats",
]
