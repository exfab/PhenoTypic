"""Pure data layer for the Error-analysis tab.

Dash-free, side-effect-free except the optional disk *reads* of the QC
review artifacts. Builds the good/error frames
:class:`phenotypic.analysis.ErrorCutoffFinder` consumes (spec §7) in both
good-baseline modes, plus category counts and the at-cutoff
classification metrics the draggable readout needs.

What lives here:

* :func:`category_counts` / :func:`default_category` — per-category tallies
  and the highest-count-non-other default the chip switcher seeds with.
* :func:`verified_good_keys` — the verified-good derivation (spec §7,
  resolved any-module, good-only): an object is *verified-good* iff it is
  **unlabeled** and belongs to ≥1 QC group marked reviewed in any module.
* :func:`build_good_error_frames` — the ``(good_pdf, error_pdf)`` pair for
  one category in the chosen good mode (polars filter → pandas boundary).
* :func:`classify_at_cutoff` — recall / specificity / good-flagged for an
  arbitrary dragged cutoff, NaN-safe.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from phenotypic.gui.results_viewer._qc_tab.review import _db
from phenotypic.gui.results_viewer._qc_tab.review._review_state import (
    ReviewState,
    decode_group_key,
)
from phenotypic.schema import IMAGE

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)

KEY_IMAGE_FILE = str(IMAGE.IMAGE_NAME)
KEY_OBJECT_LABEL = "Object_Label"

#: A ``(image_file, object_label)`` curation key (mirrors the curation store).
LabelKey = tuple[str, int]

#: Good-baseline mode: every unlabeled object, or the verified-only subset.
GoodMode = Literal["all_unlabeled", "verified"]


# ---------------------------------------------------------------------------
# Category counts + default
# ---------------------------------------------------------------------------


def category_counts(labels: dict[LabelKey, str]) -> dict[str, int]:
    """Count labeled objects per category token.

    Args:
        labels: The curation store's ``(image, label) -> category`` map.

    Returns:
        ``{category_token: count}`` over ``labels.values()``.
    """
    return dict(Counter(labels.values()))


def default_category(counts: dict[str, int], other_token: str) -> str | None:
    """Return the category the switcher should focus by default.

    Highest-count NON-``other_token`` category wins; falls back to
    ``other_token`` when it is the only label class (the common
    legacy-migration case — focusing *something* beats nothing); ``None``
    when there are no labels at all (R6).

    Args:
        counts: Per-category tallies from :func:`category_counts`.
        other_token: The reserved catch-all token (``ErrorCategory.OTHER``).

    Returns:
        The token to focus, or ``None`` when ``counts`` is empty.
    """
    if not counts:
        return None
    non_other = {tok: n for tok, n in counts.items() if tok != other_token}
    if non_other:
        # Highest count wins; tie-break on token name for determinism.
        return max(non_other, key=lambda tok: (non_other[tok], tok))
    return other_token if other_token in counts else None


# ---------------------------------------------------------------------------
# Verified-good derivation
# ---------------------------------------------------------------------------


def verified_good_keys(
    output_root: "OutputRoot", labeled_keys: set[LabelKey]
) -> set[LabelKey]:
    """Return the unlabeled objects in ≥1 reviewed QC group (any module).

    Reads the QC catalog (``qc.duckdb``) + ``review_state.json``. For each
    module with a non-empty reviewed set, decode each reviewed group key and
    resolve its ``(image_file, label)`` members via
    :func:`._db.module_members`. Diagnostic-only modules
    (``supports_object_curation == False``) carry no curatable objects, so
    they are skipped. Union across all curation-supporting modules yields the
    reviewed-member keys; subtracting ``labeled_keys`` yields the
    verified-good set (spec §7, resolved any-module, good-only).

    Empty set when the QC database is absent / has no catalog.

    Args:
        output_root: The active results-viewer output root.
        labeled_keys: Every currently-labeled curation key (any category).

    Returns:
        The verified-good ``(image_file, object_label)`` key set.
    """
    state = ReviewState.load(output_root.layout)
    modules = {m.instance_id: m for m in _db.list_modules(output_root)}
    reviewed_members: set[LabelKey] = set()
    for instance_id, progress in state.modules.items():
        mod = modules.get(instance_id)
        # Skip modules that vanished from the catalog or are diagnostic-only
        # (no curatable per-object rows), and groups never reviewed.
        if (
            mod is None
            or not mod.supports_object_curation
            or not progress.reviewed
        ):
            continue
        for encoded in progress.reviewed:
            key = decode_group_key(encoded)
            members = _db.module_members(output_root, instance_id, key)
            if (
                members.is_empty()
                or KEY_IMAGE_FILE not in members.columns
                or KEY_OBJECT_LABEL not in members.columns
            ):
                continue
            for img, lbl in zip(
                members.get_column(KEY_IMAGE_FILE).to_list(),
                members.get_column(KEY_OBJECT_LABEL).to_list(),
            ):
                if img is None or lbl is None:
                    continue
                reviewed_members.add((str(img), int(lbl)))
    return reviewed_members - labeled_keys


def legacy_qc_cutover_message(output_root: "OutputRoot") -> str | None:
    """Return the hard-cutover message for legacy parquet-only QC outputs."""
    return _db.legacy_qc_cutover_message(output_root)


# ---------------------------------------------------------------------------
# Good / error frames
# ---------------------------------------------------------------------------


def build_good_error_frames(
    output_root: "OutputRoot",
    labels: dict[LabelKey, str],
    category: str,
    good_mode: GoodMode,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Return ``(good_pdf, error_pdf)`` for one category in the chosen mode.

    * ``error`` = master rows whose key is labeled ``category``.
    * ``good`` (``all_unlabeled``) = master rows whose key is NOT in
      ``labels``.
    * ``good`` (``verified``) = master rows whose key is in
      :func:`verified_good_keys` (always the unlabeled reviewed-member set).

    The polars master is filtered, then converted to pandas at the engine
    boundary (:class:`ErrorCutoffFinder` is pandas-typed).

    Args:
        output_root: The active output root. The frames are built from
            ``clean_master_df`` (the full pre-post object set), NOT the curated
            ``master_df`` mirror — the mirror has the labeled (error) rows
            removed, so the error class would be empty after a reload.
        labels: The curation store's ``(image, label) -> category`` map.
        category: The focused error category token.
        good_mode: ``"all_unlabeled"`` or ``"verified"``.

    Returns:
        A ``(good_pdf, error_pdf)`` pair of pandas frames.
    """
    master = output_root.clean_master_df
    labeled_keys = set(labels.keys())
    error_keys = {key for key, cat in labels.items() if cat == category}

    if good_mode == "verified":
        good_keys = verified_good_keys(output_root, labeled_keys)
        good_pdf = _select_keys(master, good_keys)
    else:
        good_pdf = _exclude_keys(master, labeled_keys)

    error_pdf = _select_keys(master, error_keys)
    return good_pdf, error_pdf


def _keyed(master: pl.DataFrame) -> pl.DataFrame:
    """Cast the curation-key columns to ``(String, Int64)`` for joins."""
    return master.with_columns(
        pl.col(KEY_IMAGE_FILE).cast(pl.String),
        pl.col(KEY_OBJECT_LABEL).cast(pl.Int64),
    )


def _key_frame(keys: set[LabelKey]) -> pl.DataFrame:
    """Build the 2-column ``(String, Int64)`` join frame from a key set."""
    key_list = list(keys)
    return pl.DataFrame(
        {
            KEY_IMAGE_FILE: [k[0] for k in key_list],
            KEY_OBJECT_LABEL: [k[1] for k in key_list],
        },
        schema={KEY_IMAGE_FILE: pl.String, KEY_OBJECT_LABEL: pl.Int64},
    )


def _select_keys(master: pl.DataFrame, keys: set[LabelKey]) -> "pd.DataFrame":
    """Return the master rows whose key is in ``keys`` (semi-join → pandas)."""
    if not keys:
        return _keyed(master).head(0).to_pandas()
    selected = _keyed(master).join(
        _key_frame(keys), on=[KEY_IMAGE_FILE, KEY_OBJECT_LABEL], how="semi"
    )
    return selected.to_pandas()


def _exclude_keys(master: pl.DataFrame, keys: set[LabelKey]) -> "pd.DataFrame":
    """Return the master rows whose key is NOT in ``keys`` (anti-join → pandas)."""
    if not keys:
        return _keyed(master).to_pandas()
    kept = _keyed(master).join(
        _key_frame(keys), on=[KEY_IMAGE_FILE, KEY_OBJECT_LABEL], how="anti"
    )
    return kept.to_pandas()


# ---------------------------------------------------------------------------
# At-cutoff classification metrics (drag readout)
# ---------------------------------------------------------------------------


def classify_at_cutoff(
    good_values: np.ndarray,
    error_values: np.ndarray,
    cutoff: float,
    direction: str,
) -> dict[str, float]:
    """Recall / specificity / good-flagged at an arbitrary cutoff.

    ``direction`` ``">"`` flags values strictly above ``cutoff`` as error;
    ``"<"`` flags values strictly below. NaN-safe (NaN dropped first).

    Args:
        good_values: Good-class measurement values (may contain NaN).
        error_values: Error-class measurement values (may contain NaN).
        cutoff: The (possibly dragged) decision threshold.
        direction: ``">"`` or ``"<"`` — the error-flagging side.

    Returns:
        ``{"recall", "specificity", "good_flagged"}`` where
        ``recall = flagged_error / n_error``,
        ``specificity = kept_good / n_good``, and ``good_flagged`` is the
        count of good values on the flagged side. Empty classes report
        ``0.0`` for their derived ratio.
    """
    good = good_values[~np.isnan(good_values)]
    error = error_values[~np.isnan(error_values)]

    if direction == "<":
        error_flagged = int(np.sum(error < cutoff))
        good_flagged = int(np.sum(good < cutoff))
    else:
        error_flagged = int(np.sum(error > cutoff))
        good_flagged = int(np.sum(good > cutoff))

    n_error = int(error.size)
    n_good = int(good.size)
    recall = error_flagged / n_error if n_error else 0.0
    specificity = (n_good - good_flagged) / n_good if n_good else 0.0
    return {
        "recall": float(recall),
        "specificity": float(specificity),
        "good_flagged": float(good_flagged),
    }


__all__ = [
    "GoodMode",
    "LabelKey",
    "build_good_error_frames",
    "category_counts",
    "classify_at_cutoff",
    "default_category",
    "legacy_qc_cutover_message",
    "verified_good_keys",
]
