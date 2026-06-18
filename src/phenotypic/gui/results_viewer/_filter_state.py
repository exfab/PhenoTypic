"""Filter specification for the results viewer sidebar.

This module exposes the pure data layer that backs the viewer's filter
sidebar. It mediates between the Dash ``dcc.Store`` payload (a list of
per-row dicts, each carrying a ``column``, a ``method`` — one of
``is_any_of`` / ``is_none_of`` / ``range`` / ``compare`` / ``contains`` —
and that method's payload fields) and a polars ``DataFrame.filter``
expression chain. Each row compiles to a single predicate via
:meth:`FilterRow.to_expr`; rows AND together. Keeping this logic separate
from the Dash callbacks lets it be unit-tested without spinning up a browser.
"""

from __future__ import annotations

import functools
import logging
import operator
from dataclasses import dataclass, field
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

METHOD_IS_ANY_OF = "is_any_of"
METHOD_IS_NONE_OF = "is_none_of"
METHOD_RANGE = "range"
METHOD_COMPARE = "compare"
METHOD_CONTAINS = "contains"

VALID_METHODS: frozenset[str] = frozenset(
    {METHOD_IS_ANY_OF, METHOD_IS_NONE_OF, METHOD_RANGE, METHOD_COMPARE, METHOD_CONTAINS}
)

#: Ordering-only comparison operators. Equality is intentionally excluded —
#: exact float equality is fragile; use list mode for exact match.
COMPARE_OPS: frozenset[str] = frozenset({">", ">=", "<", "<="})


def _coerce_float(value: Any) -> float | None:
    """Best-effort float coercion; blanks / unparseable values become None."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class FilterRow:
    """A single filter clause: one column matched by one ``method``.

    Only the fields relevant to ``method`` are read by :meth:`to_expr`:

    - ``is_any_of`` / ``is_none_of`` → ``values``
    - ``range`` → ``range_min`` / ``range_max`` (either bound optional)
    - ``compare`` → ``compare_op`` (in :data:`COMPARE_OPS`) / ``compare_value``
    - ``contains`` → ``text_pattern`` / ``text_regex`` / ``text_case_sensitive``

    An unset clause (no column, or no usable payload for ``method``) is a
    no-op at apply time — never a "match nothing" sentinel.
    """

    column: str
    method: str = METHOD_IS_ANY_OF
    values: list[str] = field(default_factory=list)
    range_min: float | None = None
    range_max: float | None = None
    compare_op: str | None = None
    compare_value: float | None = None
    text_pattern: str = ""
    text_regex: bool = False
    text_case_sensitive: bool = False

    @classmethod
    def from_dict(cls, entry: dict[str, Any]) -> "FilterRow":
        """Build a row from a (possibly legacy / partial) store dict."""
        method = entry.get("method") or METHOD_IS_ANY_OF
        if method not in VALID_METHODS:
            method = METHOD_IS_ANY_OF
        raw_values = entry.get("values") or []
        if not isinstance(raw_values, list):
            raw_values = []
        compare_op = entry.get("compare_op")
        if compare_op not in COMPARE_OPS:
            compare_op = None
        return cls(
            column=str(entry.get("column", "") or ""),
            method=method,
            values=[str(v) for v in raw_values],
            range_min=_coerce_float(entry.get("range_min")),
            range_max=_coerce_float(entry.get("range_max")),
            compare_op=compare_op,
            compare_value=_coerce_float(entry.get("compare_value")),
            text_pattern=str(entry.get("text_pattern", "") or ""),
            text_regex=bool(entry.get("text_regex", False)),
            text_case_sensitive=bool(entry.get("text_case_sensitive", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat, JSON-store-friendly dict."""
        return {
            "column": self.column,
            "method": self.method,
            "values": list(self.values),
            "range_min": self.range_min,
            "range_max": self.range_max,
            "compare_op": self.compare_op,
            "compare_value": self.compare_value,
            "text_pattern": self.text_pattern,
            "text_regex": self.text_regex,
            "text_case_sensitive": self.text_case_sensitive,
        }

    def to_expr(self) -> "pl.Expr | None":
        """Return the polars predicate for this row, or None if unset.

        None means "skip this row" (no column, or no usable payload for the
        active method). Numeric methods cast the column with
        ``strict=False`` so non-numeric cells become null and fall out of
        the match without raising.
        """
        if not self.column:
            return None
        method = self.method or METHOD_IS_ANY_OF

        if method in (METHOD_IS_ANY_OF, METHOD_IS_NONE_OF):
            if not self.values:
                return None
            expr = pl.col(self.column).cast(pl.String).is_in(self.values)
            return ~expr if method == METHOD_IS_NONE_OF else expr

        if method == METHOD_RANGE:
            if self.range_min is None and self.range_max is None:
                return None
            numeric = pl.col(self.column).cast(pl.Float64, strict=False)
            bounds: list[pl.Expr] = []
            if self.range_min is not None:
                bounds.append(numeric >= self.range_min)
            if self.range_max is not None:
                bounds.append(numeric <= self.range_max)
            return functools.reduce(operator.and_, bounds)

        if method == METHOD_COMPARE:
            if self.compare_op not in COMPARE_OPS or self.compare_value is None:
                return None
            numeric = pl.col(self.column).cast(pl.Float64, strict=False)
            ops = {
                ">": operator.gt,
                ">=": operator.ge,
                "<": operator.lt,
                "<=": operator.le,
            }
            return ops[self.compare_op](numeric, self.compare_value)

        if method == METHOD_CONTAINS:
            pattern = self.text_pattern or ""
            if not pattern.strip():
                return None
            as_str = pl.col(self.column).cast(pl.String)
            if self.text_regex:
                rx = pattern if self.text_case_sensitive else f"(?i){pattern}"
                return as_str.str.contains(rx, literal=False)
            if self.text_case_sensitive:
                return as_str.str.contains(pattern, literal=True)
            return as_str.str.to_lowercase().str.contains(
                pattern.lower(), literal=True
            )

        return None


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
            rows.append(FilterRow.from_dict(entry))
        return cls(rows=rows)

    def to_store(self) -> list[dict]:
        """Serialise this spec back into a Dash-store-friendly payload.

        Returns:
            A list of dicts matching the ``from_store`` input shape. The
            ``values`` list is shallow-copied so downstream mutations on the
            store don't leak back into this spec.
        """
        return [row.to_dict() for row in self.rows]

    def apply_to(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply every active row as ``AND`` across rows.

        Each row contributes ``FilterRow.to_expr()``; ``None`` (unset) rows
        are skipped. Rows naming a column absent from the frame log a
        warning and skip. A row whose expression fails to evaluate (e.g. an
        invalid user-supplied regex raising ``ComputeError``) is logged and
        skipped rather than crashing the viewer.
        """
        result = df
        for row in self.rows:
            if not row.column:
                continue
            if row.column not in result.columns:
                logger.warning(
                    "Filter column %r is not in the DataFrame; skipping row.",
                    row.column,
                )
                continue
            expr = row.to_expr()
            if expr is None:
                continue
            try:
                result = result.filter(expr)
            except Exception:
                logger.exception(
                    "Filter row failed to evaluate; skipping. column=%r method=%r",
                    row.column,
                    row.method,
                )
                continue
        return result
