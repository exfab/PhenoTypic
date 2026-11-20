"""Remote sample-data fetch via pooch.

Large sample PNGs (formerly ``src/phenotypic/data/SnP_images/``) are too big
to bundle in the wheel without tripping PyPI's 100 MiB file-size limit. They
are instead attached as assets to the ``data-v1`` GitHub Release on
``exfab/PhenoTypic`` and fetched on demand, verified against SHA256 hashes
pinned here.

Override the cache location via the ``PHENOTYPIC_DATA_DIR`` environment
variable (useful for CI caching and air-gapped installs).
"""
from __future__ import annotations

import pooch

_DATA_TAG = "data-v1"
_BASE_URL = (
    "https://github.com/exfab/PhenoTypic/"
    f"releases/download/{_DATA_TAG}"
)

_REGISTRY = pooch.create(
    path=pooch.os_cache("phenotypic"),
    base_url=_BASE_URL,
    env="PHENOTYPIC_DATA_DIR",
    registry={
        "RhodotorulaYeastCropped.png":
            "sha256:77b3b2730553649724993204faa16a334b891d5ce5994ae85caaae10480c96c4",
        "NeurosporaFilamentousFungiCropped.png":
            "sha256:ef266dafe7ed98bb671229f35136b362fd7682ad6a943433096825dc75734b40",
        "RhodotorulaYeastFullPlate.png":
            "sha256:3939f6099fff6ab721558ea64ee482317839da15a3ec0722461272ad3e7fbf9d",
        "NeurosporaFilamentousFungiFullPlate.png":
            "sha256:e1831d22b884c55cf6b1f76223d1446879685e4b36e95aa064765dd99c58d211",
    },
)


_EXAMPLE_TAG = "data-v2"

#: The example time series (``data-v2``): three replicate plates of one
#: library plate, seven frames each, as one zip of OME-Zarr stores per plate,
#: plus the wells' metadata and the colonies' measurements. Strain IDs are
#: synthetic. Each file is fetched only when a loader asks for it.
_EXAMPLES = pooch.create(
    path=pooch.os_cache("phenotypic"),
    base_url=(
        "https://github.com/exfab/PhenoTypic/"
        f"releases/download/{_EXAMPLE_TAG}/"
    ),
    env="PHENOTYPIC_DATA_DIR",
    registry={
        "plate1.zip":
            "sha256:3162d7555b7b42727a11494f135767de6a79d14b5ec3ec9c2a85e97fb6900cd0",
        "plate2.zip":
            "sha256:629b311f11c61d6a7d6a37983266f7dbb07a674c0f51c9a14f99fe46a25b6480",
        "plate3.zip":
            "sha256:8f0eab4c0cc1dbd7fb65135f312566df10eb3c4b97b1ed50e0cb715168be3af3",
        "example-metadata.tsv":
            "sha256:ef57e7cb885bcf72f318ba28f600051aa7a16b5be566f8da398701478f79827f",
        "example-measurements.tsv":
            "sha256:726748957098833db619b82f0fef95e8353bf5628da9d903f9d5dc303adeea63",
    },
)


def fetch_example_file(filename: str) -> str:
    """Fetch one table of the example time series, returning its local path.

    Args:
        filename: ``"example-metadata.tsv"`` or ``"example-measurements.tsv"``.

    Returns:
        Absolute path (as a string) to the cached, SHA256-verified file.
    """
    return _EXAMPLES.fetch(filename, progressbar=True)


def fetch_example_plate(plate: int) -> str:
    """Fetch and unpack one plate of the example time series.

    The zip is downloaded once, verified, and extracted beside itself in the
    cache; later calls reuse both without network access.

    Args:
        plate: The replicate plate, 1, 2 or 3.

    Returns:
        Absolute path (as a string) of the folder holding that plate's
        ``plate<N>_t<HHH>.ome.zarr`` stores.
    """
    name = f"plate{plate}.zip"
    _EXAMPLES.fetch(name, processor=pooch.Unzip(), progressbar=True)
    return str(_EXAMPLES.abspath / f"{name}.unzip")


def fetch_snp(filename: str) -> str:
    """Fetch a sample-data PNG, returning a local filesystem path.

    First call downloads from the pinned ``data-v1`` GitHub Release and
    verifies the SHA256. Subsequent calls return the cached path without
    network access.

    Args:
        filename: One of the keys registered in ``_REGISTRY``.

    Returns:
        Absolute path (as a string) to the cached file.
    """
    return _REGISTRY.fetch(filename, progressbar=True)
