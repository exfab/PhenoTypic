"""Shared runner for the legacy-HDF migration goldens.

Imported by BOTH ``scripts/capture_import_laziness_goldens.py`` and
``tests/unit/test_import_surface.py``, so capture and verification measure a
migrated store identically.

Why these goldens exist for an *import* refactor: deferring ``h5py`` moves the
import that the HDF -> OME-Zarr converter depends on. The existing migration
tests are differential -- they compare a migrated store against a freshly
written one, or against the source ``.h5`` -- which means a defect introduced on
*both* sides of the comparison passes. A committed digest of the migrated store
is an independent anchor.

Not a raw byte digest of the tree. ``_tree_digest`` in
``tests/integration/cli/test_migrate_end_to_end.py`` is correct for the
within-run comparisons it serves, but a store's journal carries wall-clock
fields, so a byte digest is not reproducible across runs and could not be
committed. This records structure (relative paths), pixels (per-series content
hash), and metadata (the phenotypic attribute block with volatile keys
removed) instead -- all three deterministic by construction, and each one
diagnostic on its own when a comparison fails.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "legacy_hdf"
GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "import_laziness"
MIGRATION_GOLDEN = GOLDEN_DIR / "legacy_migration.json"

SCHEMA_VERSION = 1

#: The committed legacy layouts, matching the parametrization in
#: ``tests/unit/sdk_/test_hdf_to_zarr.py``.
LAYOUTS: tuple[str, ...] = (
    "v1_flat",
    "v2_grouped",
    "v2_enh_gray",
    "v2_grid",
    "v2_image_type",
    "v2_work_id",
)

#: Attribute keys dropped before digesting. ``phenotypic_version`` moves with
#: every release; the rest are wall-clock or duration fields written into the
#: provenance journal. Excluding them is deliberate -- tolerating a mismatch
#: instead would let a real difference hide behind the tolerance.
VOLATILE_ATTR_KEYS: frozenset[str] = frozenset(
    {
        "phenotypic_version",
        "applied_at_utc",
        "duration_seconds",
        "started_at_utc",
        "finished_at_utc",
    }
)


def _strip_volatile(value: Any) -> Any:
    """Recursively drop :data:`VOLATILE_ATTR_KEYS` from a decoded JSON value."""
    if isinstance(value, dict):
        return {
            key: _strip_volatile(item)
            for key, item in value.items()
            if key not in VOLATILE_ATTR_KEYS
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


def _array_digest(array: Any) -> dict[str, Any]:
    """Shape, dtype and a content hash for one decoded layer."""
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def digest_store(store: Path) -> dict[str, Any]:
    """Structure, pixels and metadata for one migrated OME-Zarr store.

    Args:
        store: Path to a written ``*.ome.zarr`` store directory.

    Returns:
        A JSON-serializable digest: ``paths`` (store-relative file names),
        ``series`` (per-layer shape/dtype/content hash) and ``attributes``
        (the ``phenotypic`` block with volatile keys removed).
    """
    from phenotypic import Image
    from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes

    attributes = read_phenotypic_attributes(store)
    series: dict[str, Any] = {}
    for name in sorted(attributes.get(PhenotypicAttr.SERIES, ())):
        series[name] = _array_digest(Image.load_layer_zarr(store, name))

    return {
        "paths": sorted(
            path.relative_to(store).as_posix()
            for path in store.rglob("*")
            if path.is_file()
        ),
        "series": series,
        "attributes": _strip_volatile(attributes),
    }


def migrate_and_digest(layout: str, destination: Path) -> dict[str, Any]:
    """Convert one committed legacy layout and digest the result.

    Args:
        layout: One of :data:`LAYOUTS`.
        destination: Directory to write the ``.ome.zarr`` store into. Must not
            already exist.

    Returns:
        The digest produced by :func:`digest_store`.
    """
    from phenotypic.sdk_._hdf_to_zarr import migrate_hdf_to_zarr

    store = migrate_hdf_to_zarr(FIXTURES / layout / "img.h5", destination)
    return digest_store(Path(store))


def load_golden() -> dict[str, Any]:
    """Read the committed migration golden."""
    return json.loads(MIGRATION_GOLDEN.read_text(encoding="utf-8"))


def write_golden(payload: dict[str, Any]) -> None:
    """Write *payload* as the migration golden, pretty-printed for diffing."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    MIGRATION_GOLDEN.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
