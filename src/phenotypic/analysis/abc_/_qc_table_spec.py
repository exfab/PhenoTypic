"""Self-describing catalog descriptor for a QC module's persisted table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QcTableSpec:
    """Column-role descriptor for one QC module's DuckDB table.

    Produced by :meth:`QualityCheck.table_spec` and written as one row of the
    ``qc_modules`` catalog, so any consumer can render a module generically
    without hard-coding its schema.

    Attributes:
        instance_id: The recipe entry id (``qc-<name>-<hex>``).
        cls_name: The ``QualityCheck`` subclass name.
        name: The check's short ``name`` (e.g. ``"ICC"``).
        groupby_cols: Ordered group-key column names.
        metric_col / status_col / flag_col: The generic QC column names.
        on_col: The measurement column the check operates on.
        member_key_cols: Per-object curation-key columns (``[]`` when the
            module has no per-object key).
        supports_object_curation: Whether the table's rows map to curatable
            detected objects (``False`` for diagnostic-only modules).
        time_col: Time-course facet column, or ``None``.
        higher_is_bad: The check's ``_HIGHER_IS_BAD`` direction.
        extra_cols: Check-specific columns beyond the generic trio.
        warn_threshold / fail_threshold: For status legends.
    """

    instance_id: str
    cls_name: str
    name: str
    groupby_cols: list[str]
    metric_col: str
    status_col: str
    flag_col: str
    on_col: str
    member_key_cols: list[str]
    supports_object_curation: bool
    time_col: str | None
    higher_is_bad: bool
    extra_cols: list[str]
    warn_threshold: float
    fail_threshold: float
