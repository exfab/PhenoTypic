"""Build the client-facing source spec for one per-image store.

Every fact here is READ from the store, never inferred -- the series list,
the primary series, the resolved label path, the pyramid ladder. None of it
is recomputed, because backend section 1.1 forbids hard-coding the label path
and section 1.3 records that re-deriving the level count has already been got
wrong once (``floor`` where ``ceil`` was needed).

Built on the LANDED resolvers rather than a second implementation of them.
:func:`~phenotypic.gui._shared.tiles._readable_block` raises
:class:`~phenotypic.gui._shared.tiles.StoreUnreadable` on a schema-version
mismatch, which is what keeps Plate and Colony agreeing about a store this
build cannot decode -- ``crop_colony`` deliberately does not catch it
(``tiles.py:685-688``). A raw ``json.loads`` here would let Plate open a
store that Colony 422s on, with the two surfaces disagreeing about one store.

A store observed between Stage 1 and Stage 3 holds an ALL-ZERO objmap: the
landed staged engine keeps Stage 2 read-only and publishes the segmentation
only when Stage 3 re-promotes. This is a valid store, and the Layers panel
renders the label layer normally. Do not treat zeros as a fault -- see the
plan's ``DRIFT.md``, row D-4.

An earlier revision exported a ``measured`` key -- ``tables in block`` -- and
the Layers panel captioned the objmap "measurement pending" when it was
false. **That inference was wrong and both are gone.** An absent ``tables``
descriptor does not mean Stage 3 is still coming: a ``--mode process`` run
never measures, a store written before embedded tables has none, and a
migrated store may carry none either. Captioning those "pending" tells the
user to wait for something that will never arrive.

The mistake was asking a **run-state** question of a **store** attribute. A
store cannot know whether a run is in flight; that lives in
``.phenotypic/progress/``. There is no sound store-only discriminator between
"Stage 3 pending" and "detector found nothing", so this module does not
pretend to one.
"""

from __future__ import annotations

from pathlib import Path

from phenotypic.gui._shared.tiles import _readable_block
from phenotypic.gui.results_viewer._zarr_routes import store_generation_token
from phenotypic.sdk_ import ngff_


def build_source_spec(store: Path, base_url: str) -> dict:
    """Describe one store generation to the Viv facade.

    Takes a STORE PATH and a base URL rather than an ``OutputRoot``: the
    builder preview has stores but no output root, and it is the second
    caller. Written at its final signature here so that phase adds a caller
    instead of refactoring this function's own work.

    The returned mapping IS the facade's source spec, extended. Its
    ``storeUrl`` / ``seriesPath`` / ``labelPath`` keys are the three
    ``window.phenotypicViv.setSource`` validates and consumes
    (``_assets/viv_viewer.js``), so a caller hands the dict over unmodified;
    ``series``, ``pyramid`` and ``token`` are read by the surface's own
    chrome -- the Layers panel and the pyramid readout. ``labelColorDomain``
    bounds the categorical hue cycle used by the facade for the objmap.

    Args:
        store: Path to a promoted ``*.ome.zarr`` directory.
        base_url: Browser-visible base URL of this store generation, as
            built by
            :func:`~phenotypic.gui.results_viewer._zarr_routes.zarr_store_url`.
            Every key the client resolves is relative to it.

    Returns:
        A JSON-serialisable mapping with keys ``storeUrl``, ``token``,
        ``series`` (ordered, primary first), ``seriesPath`` (the primary),
        ``labelPath`` (**may be** ``None``), ``labelColorDomain`` and
        ``pyramid``.

    Raises:
        OSError: If the store's root ``zarr.json`` does not exist -- the
            routine signal that a promote is in flight.
        KeyError: If the root exists but carries no ``phenotypic`` block.
        StoreUnreadable: If the store's schema version is not this build's.
        ValueError: If the store declares neither ``rgb`` nor ``gray``.

    Examples:
        Describe a freshly written store to the facade:

        >>> import tempfile
        >>> from pathlib import Path
        >>> from phenotypic import Image
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> img = Image(load_synth_yeast_plate())
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     store = img.save2zarr(Path(tmp) / 'plate.ome.zarr')
        ...     spec = build_source_spec(store, '/zarr/d1/plate.ome.zarr/t')
        ...     (spec['seriesPath'], spec['labelPath'])
        ('rgb', 'rgb/labels/objmap')
    """
    block = _readable_block(store)

    series_map = block.get(ngff_.PhenotypicAttr.SERIES, {})
    primary = ngff_.primary_series(list(series_map))
    ordered = [primary, *(name for name in series_map if name != primary)]

    # ``labels`` is OMITTED ENTIRELY when the store carries no label image
    # (``ngff_.py:576-581``) -- and ``save_intermediate_zarr`` sets
    # ``write_objmap = "objmap" in layers``, so MOST builder-preview stores
    # have no ``labels`` key at all. ``block["labels"]`` would ``KeyError``
    # on every one of them.
    label_path = block.get(ngff_.PhenotypicAttr.LABELS, {}).get(
        ngff_.OBJMAP_LABEL
    )
    grid = block.get(ngff_.PhenotypicAttr.GRID, {})
    nrows = grid.get("nrows") if isinstance(grid, dict) else None
    ncols = grid.get("ncols") if isinstance(grid, dict) else None
    grid_capacity = (
        nrows * ncols
        if isinstance(nrows, int)
        and not isinstance(nrows, bool)
        and isinstance(ncols, int)
        and not isinstance(ncols, bool)
        and nrows > 0
        and ncols > 0
        else None
    )

    return {
        "storeUrl": base_url,
        "token": store_generation_token(store),
        "series": ordered,
        "seriesPath": primary,
        "labelPath": label_path,  # may be None -- the facade copes
        # Grid labels are bounded by the declared plate capacity. Generic
        # images lack a cheap store-level object count, so one byte remains
        # the conservative categorical display domain until the schema grows
        # an explicit label maximum.
        "labelColorDomain": [0, grid_capacity or 255],
        "pyramid": block[ngff_.PhenotypicAttr.PYRAMID],
    }
