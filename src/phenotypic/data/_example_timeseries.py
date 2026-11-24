"""The example time series: real yeast plates, fetched on demand.

Three replicate plates of one library plate (the same strains in the same
wells), imaged seven times from 0 to 107 h, about 18 h apart. Each image is a
colour-corrected, denoised OME-Zarr store named ``plate<N>_t<HHH>`` (replicate
plate ``N``, ``HHH`` hours). Strain IDs are synthetic (``SYN-…``, ``BLANK-…``).

Each kind of data downloads only when its loader is called: the two tables are
small, and the images are about 1.1 GB per plate. Everything is verified by
SHA256 and cached under ``PHENOTYPIC_DATA_DIR`` (default: the user cache).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from phenotypic.sdk_ import STORE_SUFFIX

from ._remote_registry import fetch_example_file, fetch_example_plate

#: The replicate plates in the example time series.
EXAMPLE_PLATES: tuple[int, ...] = (1, 2, 3)


def load_example_metadata() -> pd.DataFrame:
    """Load the example time series' well metadata.

    One row per well per image: ``Metadata_ImageName`` (``plate<N>_t<HHH>``),
    ``Grid_RowNum``, ``Grid_ColNum`` (0-based, 8 × 12) and ``Metadata_Strain``.
    Join it to :func:`load_example_measurements` on image name and grid
    position; a well with no measurement row had no colony detected.

    Returns:
        The metadata table, 2,016 rows (21 images × 96 wells).

    Examples:
        >>> from phenotypic.data import load_example_metadata
        >>> from phenotypic.schema import GENETIC, GRID, IMAGE
        >>> meta = load_example_metadata()  # doctest: +SKIP
        >>> meta.columns.tolist() == [  # doctest: +SKIP
        ...     str(IMAGE.IMAGE_NAME), str(GRID.ROW_NUM),
        ...     str(GRID.COL_NUM), str(GENETIC.STRAIN),
        ... ]
        True
    """
    return pd.read_csv(fetch_example_file("example-metadata.tsv"), sep="\t")


def load_example_measurements() -> pd.DataFrame:
    """Load the example time series' colony measurements.

    One row per detected colony: ``Metadata_ImageName``, ``Grid_RowNum``,
    ``Grid_ColNum``, ``Object_Label`` and PhenoTypic's measurement columns
    (``Size_*``, ``Shape_*``, ``Intensity_*``, ``Color_*`` and more). Strains are
    not in this table; join :func:`load_example_metadata` for them.

    Returns:
        The measurement table, one row per measured colony.

    Examples:
        >>> from phenotypic.data import load_example_measurements, load_example_metadata
        >>> from phenotypic.schema import GRID, IMAGE
        >>> m = load_example_measurements()  # doctest: +SKIP
        >>> with_strain = m.merge(  # doctest: +SKIP
        ...     load_example_metadata(),
        ...     on=[str(IMAGE.IMAGE_NAME), str(GRID.ROW_NUM), str(GRID.COL_NUM)],
        ... )
    """
    return pd.read_csv(fetch_example_file("example-measurements.tsv"), sep="\t")


def load_example_images(plates: Iterable[int] | None = None) -> list[Path]:
    """Fetch the example time series' images and return their store paths.

    Only the plates asked for are downloaded (about 1.1 GB each). Read a store's
    pixels with ``Image.imread(path)`` or ``GridImage.imread(path, nrows=8,
    ncols=12)``.

    Args:
        plates: Replicate plates to fetch, any of ``1``, ``2``, ``3``. Default:
            all three.

    Returns:
        The ``plate<N>_t<HHH>.ome.zarr`` store paths, sorted by name.

    Raises:
        ValueError: If a plate is not 1, 2 or 3.

    Examples:
        >>> from phenotypic import GridImage
        >>> from phenotypic.data import load_example_images
        >>> stores = load_example_images(plates=[1])  # doctest: +SKIP
        >>> plate = GridImage.imread(stores[-1], nrows=8, ncols=12)  # doctest: +SKIP
    """
    wanted = EXAMPLE_PLATES if plates is None else tuple(plates)
    unknown = sorted(set(wanted) - set(EXAMPLE_PLATES))
    if unknown:
        raise ValueError(f"unknown example plate(s) {unknown}; choose from {list(EXAMPLE_PLATES)}")
    stores: list[Path] = []
    for plate in sorted(set(wanted)):
        stores.extend(Path(fetch_example_plate(plate)).glob(f"*{STORE_SUFFIX}"))
    return sorted(stores, key=lambda p: p.name)
