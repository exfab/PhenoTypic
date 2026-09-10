"""Attribute measurement columns to the measurer that emitted them."""

from __future__ import annotations

import logging

from phenotypic.gui.results_viewer._scatter_tab._spec import (
    CURATION_PHANTOM_COL,
)
from phenotypic.sdk_ import is_metadata_header

logger = logging.getLogger(__name__)

#: The curation column gets its own heading rather than falling into
#: Unattributed: no measurer claims it and ``is_metadata_header`` rejects
#: it, but it is the column the phantom predicate depends on, so burying
#: it in a bucket named "Unattributed" is the worst of the options.
CURATION_COLUMNS: tuple[str, ...] = (CURATION_PHANTOM_COL,)


def group_columns(
    columns: list[str], meas_cfg: dict[str, dict]
) -> dict[str, list[str]]:
    """Group columns under the ``MeasureFeatures`` class that emits them.

    Measurers are instantiated **from their recorded params**, not used as
    classes: ``get_measurement_infoclasses`` is an instance method and its
    result depends on parameters -- ``MeasureColor()`` yields ColorLab and
    ColorHSV, while ``MeasureColor(include_XYZ=True, include_xy=True)``
    yields four schemas.

    ``get_headers()`` is not uniformly zero-argument. ``TEXTURE`` takes a
    ``scale`` because its column names carry the offset, so a bare call
    raises ``TypeError``. Rather than special-casing each schema, such a
    class falls back to matching the frame's columns against its
    ``category()`` -- which generalizes to schemas that do not exist yet.

    Args:
        columns: Column names to group.
        meas_cfg: The ``"meas"`` block of the run's pipeline config,
            mapping a key to ``{"class": str, "params": dict}``.

    Returns:
        Group name to column names. Always includes ``"Metadata"``,
        ``"Curation"`` and ``"Unattributed"`` keys when non-empty.
    """
    import phenotypic.measure as measure_mod

    owner: dict[str, str] = {}
    for cfg in meas_cfg.values():
        name = cfg.get("class")
        if not isinstance(name, str):
            logger.debug("scatter grouping: no class name in %r", cfg)
            continue
        cls = getattr(measure_mod, name, None)
        if cls is None:
            logger.debug("scatter grouping: unknown measurer %r", name)
            continue
        try:
            op = cls(**cfg.get("params", {}))
        except Exception:
            logger.debug(
                "scatter grouping: could not construct %r", name,
                exc_info=True,
            )
            continue
        for info in op.get_measurement_infoclasses():
            try:
                headers = list(info.get_headers())
            except TypeError:
                prefix = f"{info.category()}_"
                headers = [c for c in columns if c.startswith(prefix)]
            for header in headers:
                owner.setdefault(header, name)

    groups: dict[str, list[str]] = {}
    for col in columns:
        if col in CURATION_COLUMNS:
            key = "Curation"
        elif is_metadata_header(col):
            key = "Metadata"
        else:
            key = owner.get(col, "Unattributed")
        groups.setdefault(key, []).append(col)
    return groups
