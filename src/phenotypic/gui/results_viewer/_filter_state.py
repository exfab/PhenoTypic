"""Filter specification for the results viewer sidebar.

This module exposes the pure data layer that backs the viewer's filter
sidebar. It mediates between the Dash ``dcc.Store`` payload (a list of
``{"column": str, "values": list[str]}`` dicts) and a polars
``DataFrame.filter`` expression chain. Keeping this logic separate from the
Dash callbacks lets it be unit-tested without spinning up a browser.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class FilterRow:
    """A single filter clause: one column, many accepted values.

    The values list is interpreted as an OR — a row matches if the cell's
    string representation is in ``values``. Stored as strings because the
    Dash JSON store round-trips numerics as numbers, and we always compare
    against ``pl.col(...).cast(pl.String)``.

    Attributes:
        column: Name of the column to filter on. Empty string means the row
            is unset (skipped at apply time).
        values: Accepted string values for the column. Empty list means the
            row is unset (skipped — NOT a "match nothing" sentinel).
    """

    column: str
    values: list[str]


@dataclass
class FilterSpec:
    """A composite filter: AND across rows, OR within each row's values.

    The spec is the source of truth for what the user has typed into the
    sidebar. ``from_store`` and ``to_store`` round-trip it through Dash's
    JSON-backed ``dcc.Store``; ``apply_to`` projects it onto a polars frame.

    Attributes:
        rows: Ordered list of filter clauses. Empty list means no filtering
            (``apply_to`` returns the input frame unchanged).
    """

    rows: list[FilterRow] = field(default_factory=list)

    @classmethod
    def from_store(cls, payload: list[dict] | None) -> "FilterSpec":
        """Build a ``FilterSpec`` from a Dash store payload.

        Malformed entries (missing ``column``) are silently skipped because
        Dash stores can transiently hold partial entries while the user is
        building a row in the sidebar. Values are coerced to ``str`` since
        the JSON store preserves numeric types.

        Args:
            payload: List of dicts shaped ``{"column": str, "values": list}``,
                or ``None`` (treated as an empty payload).

        Returns:
            A new ``FilterSpec`` whose rows mirror the payload.
        """
        if not payload:
            return cls(rows=[])
        rows: list[FilterRow] = []
        for entry in payload:
            if not isinstance(entry, dict) or "column" not in entry:
                continue
            column = entry["column"]
            raw_values: Any = entry.get("values", [])
            if raw_values is None:
                raw_values = []
            values = [str(v) for v in raw_values]
            rows.append(FilterRow(column=str(column), values=values))
        return cls(rows=rows)

    def to_store(self) -> list[dict]:
        """Serialise this spec back into a Dash-store-friendly payload.

        Returns:
            A list of dicts matching the ``from_store`` input shape. The
            ``values`` list is shallow-copied so downstream mutations on the
            store don't leak back into this spec.
        """
        return [{"column": row.column, "values": list(row.values)} for row in self.rows]

    def apply_to(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply every active row to ``df`` as ``AND`` across rows.

        Each row is applied as ``pl.col(column).cast(pl.String).is_in(values)``.
        Rows with an empty/falsy ``column`` or empty ``values`` are skipped
        (an unset filter is a no-op, never a "match nothing"). Rows whose
        ``column`` is not present in the frame log a warning and are
        skipped — column drift between pipeline runs should not crash the
        viewer.

        Args:
            df: The polars frame to filter (typically the master
                measurements table).

        Returns:
            The filtered frame. Returned as-is if ``self.rows`` is empty or
            every row is unset/skipped.
        """
        result = df
        for row in self.rows:
            if not row.column or not row.values:
                continue
            if row.column not in result.columns:
                logger.warning(
                    "Filter column %r is not in the DataFrame; skipping row.",
                    row.column,
                )
                continue
            result = result.filter(pl.col(row.column).cast(pl.String).is_in(row.values))
        return result
