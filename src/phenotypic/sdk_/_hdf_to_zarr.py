"""Convert legacy per-image HDF5 files into OME-Zarr stores.

This is the engine behind ``--mode migrate``. It reuses the **existing**
v1-flat and v2-grouped HDF readers rather than adding a third one, so the only
thing here that knows the legacy layout is the metadata restore below --
everything else goes through :func:`load_image_from_hdf` and
:meth:`Image.save2zarr`.

Three properties the callers depend on:

* **Retention.** Sources are kept (``keep_source=True``); deletion is opt-in
  and, in a run-level migration, gated on a value-level re-read comparison.
* **Resumability.** A stem whose store already passes ``valid_staged_store``
  is skipped, so re-running after an interruption *is* the recovery procedure.
  There is no ``--resume`` flag.
* **Atomicity.** Conversion writes through the §3.2 promote, so an interrupted
  conversion leaves no valid root and is simply redone.

Header canonicalization happens in the same pass -- ``_normalize_stored_metadata_items``
runs inside both legacy readers -- so a converted store is canonical by
construction and needs no second header migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from ._io_constants import (
    load_image_from_hdf,
    results_dir,
    zarr_store_path,
)
from .ngff_ import STORE_SUFFIX, valid_staged_store

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phenotypic._core._image import Image


@dataclass(frozen=True)
class MigrationReport:
    """What one ``--mode migrate`` invocation did.

    ``converted``/``skipped``/``failed`` describe **pass 2** (per-image
    conversion); ``headers_migrated``/``header_failures`` describe **pass 1**
    (the metadata-schema migration over non-image targets), which without
    somewhere to land would have failed silently.

    Attributes:
        converted: Images converted this run. Under ``dry_run`` this is what
            *would* be converted.
        skipped: Images whose store already passed ``valid_staged_store``.
        failed: ``(source, reason)`` per image that could not be converted,
            or whose source could not be safely deleted.
        headers_migrated: Non-image targets whose headers were rewritten.
        header_failures: ``(target, reason)`` per pass-1 failure.
    """

    converted: int = 0
    skipped: int = 0
    failed: tuple[tuple[Path, str], ...] = ()
    headers_migrated: int = 0
    header_failures: tuple[tuple[Path, str], ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the run had no per-image and no header failures."""
        return not self.failed and not self.header_failures


# ---------------------------------------------------------------------------
# Metadata restore
# ---------------------------------------------------------------------------


def _stored_v2_metadata_sections(src: Path) -> dict[str, dict] | None:
    """Return a v2-grouped file's metadata sections, decoded and normalized.

    ``None`` for any other layout -- the v1-flat reader already assigns its
    sections verbatim (``clear()`` then ``update()``), so there is nothing to
    repair there.

    This exists because ``_load_v2_grouped``'s restore loop **skips** any key
    the constructor already populated:

    .. code-block:: python

        for mapped, value in decoded.items():
            if mapped in target and target[mapped] is not None:
                continue

    ``Metadata_ImageType`` is set to ``"Image"`` by the constructor, so a file
    saying ``"GridSection"`` loads as ``"Image"`` -- the key is present and
    carries the wrong value, which no key-set comparison can see (ledger
    MIG-2). The loader itself is deliberately not changed: Phase 2 recorded
    that decision as OPEN-QUESTIONS D7, and the loader is retired in Phase 6.

    Args:
        src: Path to a per-image ``.h5``.

    Returns:
        ``{"protected": …, "public": …, "imported": …}`` or ``None``.
    """
    import h5py

    from phenotypic._core._image_parts._image_io_handler import (
        _decode_meta,
        _normalize_stored_metadata_items,
    )

    with h5py.File(src, "r") as handle:
        if int(handle.attrs.get("schema_version", 1)) < 2:
            return None
        if "metadata" not in handle:
            return None
        meta = handle["metadata"]
        sections: dict[str, dict] = {}
        for name in ("protected", "public", "imported"):
            if name not in meta:
                sections[name] = {}
                continue
            attrs = meta[name].attrs
            sections[name] = _normalize_stored_metadata_items(
                ((key, _decode_meta(attrs[key])) for key in attrs),
                section=name,
            )
    return sections


def _stored_work_id(src: Path) -> str | None:
    """Return the source's root ``phenotypic_work_id``, or ``None``.

    The CLI wrote this as a **post-write patch** on the HDF root; it lives in
    no image field, so the loader does not carry it and ``save2zarr`` would
    never see it. Dropping it makes every migrated image reclassify
    ``"stage1"`` -- a full reprocess from original inputs a migrated archive
    may no longer have (ledger FLOW-1).

    Args:
        src: Path to a per-image ``.h5``.

    Returns:
        The work id, or ``None`` when the file carries none.
    """
    import h5py

    with h5py.File(src, "r") as handle:
        value = handle.attrs.get("phenotypic_work_id")
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value) if value is not None else None


def _load_for_migration(src: Path) -> "Image":
    """Load a legacy ``.h5`` with its stored metadata sections intact.

    Args:
        src: Path to a per-image ``.h5``.

    Returns:
        The loaded ``Image`` or ``GridImage``.
    """
    image = load_image_from_hdf(Path(src))
    sections = _stored_v2_metadata_sections(Path(src))
    if sections is not None:
        for name, stored in sections.items():
            target = getattr(image._metadata, name)
            target.clear()
            target.update(stored)
    return image


# ---------------------------------------------------------------------------
# One image
# ---------------------------------------------------------------------------


def default_store_path_for(src: Path) -> Path:
    """Return the sibling ``*.ome.zarr`` path for a bare ``.h5``.

    Only used when a caller passes no explicit destination. Run-level
    migration always names its target through :func:`zarr_store_path`, so the
    ``.ome.zarr`` double suffix is never hand-joined there.

    Args:
        src: Path to a per-image ``.h5``.

    Returns:
        ``<src.parent>/<stem>.ome.zarr``.
    """
    src = Path(src)
    return src.parent / f"{src.stem}{STORE_SUFFIX}"


def migrate_hdf_to_zarr(
    src: Path,
    dst: Path | None = None,
    *,
    keep_source: bool = True,
) -> Path:
    """Convert one legacy per-image HDF5 into an OME-Zarr store.

    Reuses the existing v1-flat and v2-grouped readers, restores the stored
    metadata sections verbatim over the loader's lossy merge, carries the root
    ``phenotypic_work_id`` onto the store, and writes through the promote --
    so an interrupted conversion leaves no valid root.

    The legacy ``enh_gray`` layer is handled by the reader's fallback and
    lands as ``detect_mat``; there is no rename step here.

    Args:
        src: Path to the per-image ``.h5``.
        dst: Target ``*.ome.zarr`` directory. Defaults to a sibling of *src*.
        keep_source: Retain the ``.h5``. Deletion is opt-in.

    Returns:
        The promoted store path.
    """
    src = Path(src)
    target = default_store_path_for(src) if dst is None else Path(dst)
    image = _load_for_migration(src)
    store = image.save2zarr(target, work_id=_stored_work_id(src))
    if not keep_source:
        src.unlink()
    return store


# ---------------------------------------------------------------------------
# A whole run
# ---------------------------------------------------------------------------


def iter_legacy_hdfs(output_dir: Path) -> Iterator[tuple[str, Path]]:
    """Yield ``(dataset, hdf_path)`` for every ``results/*/hdf/*.h5``.

    Args:
        output_dir: Run output root.

    Yields:
        The dataset name and the per-image ``.h5`` path, in sorted order.
    """
    root = results_dir(Path(output_dir))
    if not root.is_dir():
        return
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        hdf_dir = dataset_dir / "hdf"
        if not hdf_dir.is_dir():
            continue
        for hdf_path in sorted(hdf_dir.glob("*.h5")):
            yield dataset_dir.name, hdf_path


def _convert_one(
    output_dir: Path, dataset: str, hdf_path: Path
) -> tuple[Path, str | None]:
    """Convert one image, returning ``(source, error-or-None)``."""
    try:
        migrate_hdf_to_zarr(
            hdf_path,
            zarr_store_path(output_dir, dataset, hdf_path.stem),
            keep_source=True,
        )
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return hdf_path, f"{type(exc).__name__}: {exc}"
    return hdf_path, None


def migrate_run_hdf_to_zarr(
    output_dir: Path,
    *,
    keep_source: bool = True,
    njobs: int = 1,
    dry_run: bool = False,
) -> MigrationReport:
    """Convert every legacy per-image HDF5 in a run tree, in place.

    ``results/<ds>/zarr/`` appears beside ``results/<ds>/hdf/``; measurements,
    overlays, deliverables and machine state stay exactly where they are.

    A stem whose store already passes ``valid_staged_store`` is skipped, so
    re-running after an interruption is the recovery procedure. A conversion
    that raises is **reported**, not raised: one unreadable file does not
    abandon the other ninety-nine.

    Args:
        output_dir: Run output root, converted in place.
        keep_source: Retain the ``.h5`` sources.
        njobs: Worker processes for the conversion pass.
        dry_run: Report what would be converted and write nothing.

    Returns:
        A :class:`MigrationReport`.
    """
    output_dir = Path(output_dir)
    pending: list[tuple[str, Path]] = []
    skipped = 0
    for dataset, hdf_path in iter_legacy_hdfs(output_dir):
        store = zarr_store_path(output_dir, dataset, hdf_path.stem)
        if valid_staged_store(store):
            skipped += 1
            continue
        pending.append((dataset, hdf_path))

    if dry_run:
        return MigrationReport(converted=len(pending), skipped=skipped)

    results: list[tuple[Path, str | None]]
    if njobs > 1 and len(pending) > 1:
        from joblib import Parallel, delayed

        results = list(
            Parallel(n_jobs=njobs)(
                delayed(_convert_one)(output_dir, dataset, hdf_path)
                for dataset, hdf_path in pending
            )
        )
    else:
        results = [
            _convert_one(output_dir, dataset, hdf_path)
            for dataset, hdf_path in pending
        ]

    converted = 0
    failed: list[tuple[Path, str]] = []
    for hdf_path, error in results:
        if error is None:
            converted += 1
        else:
            failed.append((hdf_path, error))

    return MigrationReport(
        converted=converted, skipped=skipped, failed=tuple(failed)
    )


__all__ = [
    "MigrationReport",
    "iter_legacy_hdfs",
    "migrate_hdf_to_zarr",
    "migrate_run_hdf_to_zarr",
]
