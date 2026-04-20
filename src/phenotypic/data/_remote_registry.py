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
