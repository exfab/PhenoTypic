"""Build the on-disk shape of a ``--mode process`` store, for tests.

``write_process_only_layer`` is the production writer, but it always
consolidates and always strips the journal's wall-clock fields. Several suites
need the *unconsolidated* store -- ``test_imread_reads_a_consolidated_store``
and ``test_consolidation_adds_no_files`` both draw their distinction against
one -- so they call ``Image._save_store`` directly. That call carries six
keyword arguments of which only ``series`` varies, repeated across
``tests/unit/cli`` and ``tests/unit/sdk_``; this is that call, named once.

It is deliberately not a fixture: the callers build stores at paths and stems
they choose, several of them more than one per test.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from phenotypic.sdk_ import ngff_

if TYPE_CHECKING:
    from phenotypic import Image


def write_process_store(
    path: Path,
    image: "Image",
    *,
    series: str = "gray",
    levels: int | None = None,
    consolidate: bool = False,
    write_image_class: bool = False,
) -> Path:
    """Write *image*'s *series* to *path* as a single-series store.

    Args:
        path: Store path, ``*.ome.zarr``. Parents are not created.
        image: Source image.
        series: The one series written -- ``"rgb"`` or ``"gray"``.
        levels: Pyramid level count. ``None`` resolves it the way
            ``write_process_only_layer`` does, from ``image.shape[:2]`` --
            the same two numbers every layer's array carries.
        consolidate: Consolidate the part's metadata before the promote.
            Default ``False``, which is *not* the production default: the
            suites that use this helper mostly want the plain store, and the
            two that test consolidation pass it explicitly either way.
        write_image_class: Write ``image_class``. Default ``False``, matching
            ``--mode process``, which is what makes ``load_zarr`` refuse the
            result.

    Returns:
        The promoted store path.
    """
    if levels is None:
        levels = ngff_.pyramid_level_count(*image.shape[:2])
    return image._save_store(
        path,
        series=(series,),
        write_objmap=False,
        levels=levels,
        work_id=None,
        durable=False,
        write_image_class=write_image_class,
        consolidate=consolidate,
    )
