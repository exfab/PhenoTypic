"""The forward modes' check for stores still in the v1 flat tables layout (#281 spec D5).

Before #281 P3 a store kept one measurement table at
``tables/measurements/table.parquet``. Every forward mode now writes and reads
per-pipeline run folders, ``tables/<n>-<sha12>/``, so a tree still holding a
v1 store is refused and pointed at ``--mode migrate``, which converts it
(spec D10).

**CLI-only, on purpose.** The check is not part of
:func:`~phenotypic.sdk_.datasets_needing_migration`: that predicate runs in
``_run_state._advisories`` on every ``resolve_run_state`` call, the GUI's and
``phenotypic-wait``'s polls included, and a v1 tree stays readable there. Only
a mode about to write refuses, and it stops at the first v1 store it finds,
reading one root per store.
"""

from __future__ import annotations

from pathlib import Path

from phenotypic.sdk_ import ngff_
from phenotypic.sdk_._table_runs import tables_schema_of_block


def first_v1_store(output_dir: Path) -> Path | None:
    """The first store under ``results/*/zarr/`` still in the v1 flat tables layout.

    Reads one root ``zarr.json`` per store, in sorted order, and stops at the
    first store whose tables schema is 1. A store whose root is missing,
    has no ``phenotypic`` block, or declares a tables version this build
    does not read is not a v1 store; other checks own those.

    Args:
        output_dir: A run's output root.

    Returns:
        The v1 store, or ``None`` when there is none.
    """
    results = Path(output_dir) / "results"
    if not results.is_dir():
        return None
    for store in sorted(results.glob(f"*/zarr/*{ngff_.STORE_SUFFIX}")):
        if store.name.startswith("."):
            continue
        try:
            block = ngff_.read_phenotypic_attributes(store)
            if isinstance(block, dict) and tables_schema_of_block(block) == 1:
                return store
        except (OSError, KeyError, ValueError):
            continue
    return None
