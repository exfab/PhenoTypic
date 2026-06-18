"""Compute the compact ``qc/`` artifact from a pipeline's QC config.

:func:`run_qc` is the single, GUI-free seam that turns a measurement frame
plus a pipeline's ``qc`` entries into the on-disk ``qc/`` artifact the
results-viewer Review tab reads. It is called in two places with identical
semantics:

* the CLI's ``finalize_post_master_outputs`` (once per recompile/remeasure,
  on the post-applied + metadata-joined frame), and
* the GUI's after-each-group recompute (in-process, on the curated frame).

It is **pure with respect to review progress**: it writes
``qc_summary.parquet`` / ``qc_members.parquet`` / ``qc_config.json`` and
**never** touches ``review_state.json`` (that file is GUI-owned; the CLI
finalize path resets it separately). It also never writes
``measurements.parquet`` — only the three ``qc/`` files.

Artifact schema
---------------
``qc_summary.parquet`` — one row per ``(instance_id, groupby key)``:

    instance_id, class, <groupby cols...>, metric, status, flag,
    n_members, n_flagged, rank

``rank`` is worst-first *within each instance* (0 = worst): rows are ordered
by status severity (fail > warn > pass) then by the bad-direction extremity
of ``metric``; ``NaN`` metrics sort last.

``qc_members.parquet`` — one row per ``(instance_id, group member)``:

    instance_id, <groupby cols...>, Metadata_ImageFile, Object_Label,
    member_value

``qc_config.json`` — a snapshot of the enabled ``qc`` entries that produced
this artifact (``{instance_id, class, enabled, params}`` each).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from phenotypic._core._image_pipeline import ImagePipeline
    from phenotypic.analysis.abc_ import QualityCheck

from phenotypic.schema import OBJECT
from phenotypic.tools_ import (
    qc_config_json_path,
    qc_members_parquet_path,
    qc_summary_parquet_path,
)

from ._recipe import QcRecipeEntry

logger = logging.getLogger(__name__)

#: Per-object primary-key column shared by every measurement table.
_OBJECT_LABEL: str = str(OBJECT.LABEL)

#: Status severity ranking used to order the worklist worst-first.
_STATUS_RANK: dict[str, int] = {"pass": 0, "warn": 1, "fail": 2}

#: Column order of ``qc_summary.parquet`` (group key columns are spliced in
#: between ``class`` and ``metric`` at write time).
_SUMMARY_LEAD_COLS: tuple[str, ...] = ("instance_id", "class")
_SUMMARY_TAIL_COLS: tuple[str, ...] = (
    "metric",
    "status",
    "flag",
    "n_members",
    "n_flagged",
    "rank",
)

#: Column order of ``qc_members.parquet`` (group key columns spliced after
#: ``instance_id``).
_MEMBERS_LEAD_COLS: tuple[str, ...] = ("instance_id",)
_MEMBERS_TAIL_COLS: tuple[str, ...] = (
    "Metadata_ImageFile",
    _OBJECT_LABEL,
    "member_value",
)


def run_qc(
    measurements_df: pd.DataFrame,
    pipeline: "ImagePipeline",
    output_dir: Path,
) -> None:
    """Run the pipeline's enabled QC checks and write the ``qc/`` artifact.

    Instantiates each enabled :class:`~phenotypic.analysis.abc_.QualityCheck`
    from ``pipeline.get_qc()`` (tolerantly — a check that fails to build or
    fails to analyze is skipped with a warning, never aborting the run),
    analyzes ``measurements_df``, and assembles + atomically writes
    ``qc_summary.parquet``, ``qc_members.parquet``, and ``qc_config.json``
    under ``<output_dir>/qc/``.

    No-op (writes nothing) when the pipeline has no QC entries. When entries
    exist but *none* analyze successfully, empty-but-schema-correct summary /
    members parquets are still written alongside the config snapshot, so a
    stale prior artifact never lingers.

    Args:
        measurements_df: The frame to evaluate. The CLI passes the
            post-applied + metadata-joined frame (``measurements.parquet``,
            converted to pandas), since QC ``groupby`` columns often come
            from joined metadata. A pandas DataFrame — not polars.
        pipeline: The pipeline whose ``qc`` entries (via
            :meth:`ImagePipeline.get_qc`) define the checks to run.
        output_dir: Run output root; the ``qc/`` subdirectory is created
            under it.

    Side effects:
        Writes ``qc/qc_summary.parquet``, ``qc/qc_members.parquet``, and
        ``qc/qc_config.json``. Never touches ``review_state.json`` or
        ``measurements.parquet``.
    """
    entries = list(pipeline.get_qc())
    if not entries:
        logger.debug("Pipeline has no QC entries; skipping run_qc")
        return

    output_dir = Path(output_dir)

    summary_frames: list[pd.DataFrame] = []
    member_frames: list[pd.DataFrame] = []
    used_entries: list[QcRecipeEntry] = []

    for entry in entries:
        if not entry.enabled:
            continue
        result = _run_one_check(entry, measurements_df)
        if result is None:
            continue
        summary_df, members_df = result
        summary_frames.append(summary_df)
        member_frames.append(members_df)
        used_entries.append(entry)

    summary = _concat_or_empty(summary_frames, _summary_empty())
    members = _concat_or_empty(member_frames, _members_empty())

    _write_parquet(qc_summary_parquet_path(output_dir), summary)
    _write_parquet(qc_members_parquet_path(output_dir), members)
    _write_config(qc_config_json_path(output_dir), used_entries)

    logger.info(
        "Wrote QC artifact for %d check(s): %d summary rows, %d member rows -> %s",
        len(used_entries),
        len(summary),
        len(members),
        output_dir / "qc",
    )


def _run_one_check(
    entry: QcRecipeEntry,
    measurements_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Instantiate + analyze one entry, returning its summary/member frames.

    Tolerant: any construction or analysis failure is logged at WARNING and
    yields ``None`` so the offending check is skipped without aborting the
    whole ``qc/`` write.

    Args:
        entry: The QC config entry to run.
        measurements_df: The frame to analyze.

    Returns:
        ``(summary_df, members_df)`` for this instance, or ``None`` when the
        check could not be built or analyzed.
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
        summary_df = _build_summary_rows(entry, check)
        members_df = _build_member_rows(entry, check)
    except Exception as exc:  # noqa: BLE001 - tolerant; surfaced as warning
        logger.warning(
            "Skipping QC check %s (%s): analyze failed: %s",
            entry.instance_id,
            entry.cls.__name__,
            exc,
        )
        return None

    return summary_df, members_df


def _build_summary_rows(
    entry: QcRecipeEntry,
    check: "QualityCheck",
) -> pd.DataFrame:
    """Map ``check.summary()`` into the ``qc_summary.parquet`` row schema.

    ``QualityCheck.summary()`` returns ``[*groupby, qc_n_members,
    qc_n_flagged, qc_worst_metric, qc_status]``. This renames those into the
    artifact's ``metric``/``status``/``n_members``/``n_flagged`` columns,
    derives a group-level ``flag`` (any member flagged), tags every row with
    ``instance_id`` + ``class``, and assigns a worst-first ``rank`` within
    the instance.

    Args:
        entry: The originating config entry (for ``instance_id``/``class``).
        check: The analyzed check instance.

    Returns:
        A summary frame with columns ``instance_id, class, <groupby...>,
        metric, status, flag, n_members, n_flagged, rank``.
    """
    raw = check.summary()
    groupby_cols = list(check.groupby)

    out = raw.rename(
        columns={
            "qc_worst_metric": "metric",
            "qc_status": "status",
            "qc_n_members": "n_members",
            "qc_n_flagged": "n_flagged",
        }
    )
    out["instance_id"] = entry.instance_id
    out["class"] = entry.cls.__name__
    out["flag"] = out["n_flagged"].astype(int) > 0
    out["rank"] = _rank_worst_first(
        out["metric"], out["status"], higher_is_bad=check._HIGHER_IS_BAD
    )

    ordered = [
        *_SUMMARY_LEAD_COLS,
        *groupby_cols,
        *_SUMMARY_TAIL_COLS,
    ]
    return out[ordered]


def _build_member_rows(
    entry: QcRecipeEntry,
    check: "QualityCheck",
) -> pd.DataFrame:
    """Map ``check.group_members()`` into the ``qc_members.parquet`` schema.

    ``QualityCheck.group_members()`` returns ``{group_key_tuple:
    [(image_file, object_label, member_value), ...]}``. This flattens it to
    one row per member, splicing the group-key tuple back into its
    ``groupby`` columns and tagging each row with ``instance_id``.

    Args:
        entry: The originating config entry (for ``instance_id``).
        check: The analyzed check instance.

    Returns:
        A member frame with columns ``instance_id, <groupby...>,
        Metadata_ImageFile, Object_Label, member_value``. Empty (schema
        only) when the analyzed frame lacked the curation-key columns.
        ``Metadata_ImageFile`` / ``Object_Label`` are emitted once even when
        they also appear in ``groupby`` — the per-member curation key is
        authoritative and is never duplicated as a group-key column.
    """
    # Curation-key columns are emitted explicitly from the member tuple, so
    # drop them from the group-key splice to avoid duplicate columns when
    # ``groupby`` already contains e.g. ``Metadata_ImageFile``.
    splice_cols = [
        col
        for col in check.groupby
        if col not in ("Metadata_ImageFile", _OBJECT_LABEL)
    ]
    groupby_cols = list(check.groupby)
    members = check.group_members()

    rows: list[dict[str, Any]] = []
    for key_tuple, member_list in members.items():
        key_map = {
            col: val
            for col, val in zip(groupby_cols, key_tuple)
            if col in splice_cols
        }
        for image_file, object_label, member_value in member_list:
            rows.append({
                "instance_id": entry.instance_id,
                **key_map,
                "Metadata_ImageFile": image_file,
                _OBJECT_LABEL: object_label,
                "member_value": member_value,
            })

    ordered = [*_MEMBERS_LEAD_COLS, *splice_cols, *_MEMBERS_TAIL_COLS]
    if not rows:
        return pd.DataFrame(columns=ordered)

    return pd.DataFrame(rows)[ordered]


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
    order = pd.DataFrame({
        "_nan_last": metric.isna().astype(int),  # 0 = finite first
        "_severity": -severity,                  # higher severity first
        "_metric": (-metric if higher_is_bad else metric),
    })
    sorted_index = order.sort_values(
        ["_nan_last", "_severity", "_metric"],
        kind="mergesort",  # stable: preserves group order on exact ties
    ).index

    ranks = pd.Series(range(len(sorted_index)), index=sorted_index)
    return ranks.reindex(metric.index).astype(int)


def _concat_or_empty(
    frames: list[pd.DataFrame], empty: pd.DataFrame
) -> pd.DataFrame:
    """Concatenate per-check frames, or return a schema-only empty frame.

    Per-check frames have heterogeneous ``groupby`` columns; an outer concat
    aligns them (missing keys become ``NaN`` for checks that don't use that
    column), which is the desired union schema for the artifact.

    Args:
        frames: Per-check summary/member frames (may be empty).
        empty: The minimal schema-only frame to return when ``frames`` is
            empty.

    Returns:
        The concatenated frame, or ``empty``.
    """
    if not frames:
        return empty
    return pd.concat(frames, axis=0, ignore_index=True)


def _summary_empty() -> pd.DataFrame:
    """Return a schema-only empty ``qc_summary`` frame (no groupby cols)."""
    return pd.DataFrame(
        columns=[*_SUMMARY_LEAD_COLS, *_SUMMARY_TAIL_COLS]
    )


def _members_empty() -> pd.DataFrame:
    """Return a schema-only empty ``qc_members`` frame (no groupby cols)."""
    return pd.DataFrame(
        columns=[*_MEMBERS_LEAD_COLS, *_MEMBERS_TAIL_COLS]
    )


def _write_parquet(target: Path, df: pd.DataFrame) -> None:
    """Atomically write *df* to *target* as zstd Parquet.

    Reuses the CLI's :func:`_atomic_write` (lazy-imported to keep
    ``phenotypic.tools_._qc_recipe`` free of an eager ``_cli`` import). Failure is logged at
    WARNING and swallowed so one bad write never aborts the others — the
    caller (CLI finalize) already runs ``run_qc`` under its own try/except.

    Args:
        target: Destination parquet path.
        df: Frame to write.
    """
    from phenotypic._cli._cli_output_manager import _atomic_write

    try:
        _atomic_write(
            target,
            lambda p: df.to_parquet(p, compression="zstd", index=False),
        )
    except Exception:
        logger.warning("Failed to write QC parquet %s", target, exc_info=True)


def _write_config(target: Path, entries: list[QcRecipeEntry]) -> None:
    """Atomically write the ``qc_config.json`` snapshot.

    Args:
        target: Destination ``qc_config.json`` path.
        entries: The entries that produced this artifact.
    """
    from phenotypic._cli._cli_output_manager import _atomic_write

    payload = json.dumps(
        {"qc": [entry.to_dict() for entry in entries]}, indent=2
    )

    def _write(p: str) -> None:
        Path(p).write_text(payload, encoding="utf-8")

    try:
        _atomic_write(target, _write)
    except Exception:
        logger.warning("Failed to write QC config %s", target, exc_info=True)
