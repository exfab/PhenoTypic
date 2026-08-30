"""Convert legacy per-image HDF5 files into OME-Zarr stores.

This is the engine behind ``--mode migrate``. It reuses the **existing**
v1-flat and v2-grouped HDF readers rather than adding a third one, so the only
thing here that knows the legacy layout is the metadata restore below --
everything else goes through :func:`_load_image_from_hdf` and
:meth:`Image.save2zarr`.

The legacy tree layout (``results/<ds>/hdf/<stem>.h5``) and the legacy
root attribute name are known **only here** since Phase 6: the shared
layout module ``_io_constants`` no longer publishes ``DIR_HDF``,
``dataset_hdf_dir``, ``HdfAttr``, or ``load_image_from_hdf``.

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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from ._io_constants import (
    dataset_results_dir,
    deliverables_dir,
    metadata_csv_deliverable_path,
    results_dir,
    zarr_store_path,
)
from .ngff_ import STORE_SUFFIX, valid_staged_store

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phenotypic._core._grid_image import GridImage

    from ._io_constants import ImageTypeName

#: HDF5 image-state subdirectory of a legacy run: ``<output>/results/<ds>/hdf/``.
#: Private, and defined here rather than in ``_io_constants``, so the legacy
#: layout is known only to the module allowed to know it.
_DIR_HDF: str = "hdf"

#: Top-level attribute key naming the writing class on a per-image HDF5.
#: Was ``HdfAttr.PHENOTYPIC_CLASS``, deleted with the rest of the HDF path
#: constants in Phase 6.
_PHENOTYPIC_CLASS: str = "phenotypic_class"


def _dataset_hdf_dir(output_dir: Path, dataset: str) -> Path:
    """Return ``<output>/results/<dataset>/hdf/``.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.

    Returns:
        The legacy per-image HDF directory for *dataset*. Not required to
        exist.
    """
    return dataset_results_dir(Path(output_dir), dataset) / _DIR_HDF


def _load_image_from_hdf(
    hdf_path: Path,
    *,
    fallback: "ImageTypeName" = "Image",
) -> "Image | GridImage":
    """Open an HDF5, read its class attr, dispatch to the right Image class.

    Moved out of ``_io_constants`` in Phase 6 together with the rest of the
    legacy path constants; ``--mode migrate`` is its only caller.

    Args:
        hdf_path: Path to a per-image HDF5 file.
        fallback: Image class name to use when the HDF lacks the
            ``phenotypic_class`` attribute (legacy files). Type-checked
            (statically) against :data:`ImageTypeName` -- there is no
            runtime validation; the only effect of an unrecognized
            string is that the dispatch falls through to :class:`Image`.

    Returns:
        An :class:`Image` or :class:`GridImage` instance loaded from the HDF.
    """
    import h5py  # type: ignore[import-untyped]

    from phenotypic import (
        GridImage,
        Image,
    )  # lazy: avoids circular import at module load

    with h5py.File(hdf_path, "r") as fh:
        cls_attr = fh.attrs.get(_PHENOTYPIC_CLASS, fallback)
    if isinstance(cls_attr, bytes):
        cls_attr = cls_attr.decode("utf-8", errors="replace")
    # Compared against the CLASS NAME, never against ``IMAGE_TYPES.GRID``: the
    # writer stores ``type(self).__name__``, while ``IMAGE_TYPES`` is the
    # ``Metadata_ImageType`` vocabulary. The two agree only by the coincidence
    # that ``IMAGE_TYPES.GRID.value`` is spelled ``"GridImage"`` -- rename that
    # member and every GridImage silently degrades to ``Image`` with no error.
    image_cls = GridImage if cls_attr == GridImage.__name__ else Image
    return image_cls._load_hdf5_for_migration(hdf_path)


if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

    from phenotypic._core._image import Image


@dataclass(frozen=True)
class MigrationReport:
    """What one ``--mode migrate`` invocation did.

    ``converted``/``skipped``/``failed`` describe **pass 2** (per-image
    conversion); ``headers_migrated``/``header_failures`` describe **pass 1**
    (the metadata-schema migration over non-image targets). The table fields
    describe pass 3, which installs legacy external Parquets in image stores.

    Attributes:
        converted: Images converted this run. Under ``dry_run`` this is what
            *would* be converted.
        skipped: Images whose store already passed ``valid_staged_store``.
        failed: ``(source, reason)`` per image that could not be converted,
            or whose source could not be safely deleted.
        headers_migrated: Non-image targets whose headers were rewritten.
        header_failures: ``(target, reason)`` per pass-1 failure.
        tables_migrated: External Parquets embedded this invocation.
        tables_skipped: Stores whose embedded table was already valid.
        table_failures: ``(source, reason)`` per pass-3 failure.
    """

    converted: int = 0
    skipped: int = 0
    failed: tuple[tuple[Path, str], ...] = ()
    headers_migrated: int = 0
    header_failures: tuple[tuple[Path, str], ...] = ()
    tables_migrated: int = 0
    tables_skipped: int = 0
    table_failures: tuple[tuple[Path, str], ...] = ()

    @property
    def ok(self) -> bool:
        """Whether all conversion, header, and table passes were clean."""
        return (
            not self.failed
            and not self.header_failures
            and not self.table_failures
        )


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
    image = _load_image_from_hdf(Path(src))
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
# The precondition for the one irreversible step
# ---------------------------------------------------------------------------

#: Layers compared byte-for-byte before a source may be unlinked.
_FAITHFULNESS_LAYERS: tuple[str, ...] = ("rgb", "gray", "detect_mat", "objmap")


def _layer_digest(array: "np.ndarray") -> str:
    """Return a content digest of one layer.

    Shapes and dtypes are blind to **content**: a conversion that wrote a
    correctly-shaped zero ``detect_mat`` has both right. Only the bytes settle
    it.
    """
    import hashlib

    import numpy as np

    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(
        f"{contiguous.shape}|{contiguous.dtype}|".encode()
        + contiguous.tobytes()
    ).hexdigest()


def _conversion_is_faithful(src: Path, store: Path) -> bool:
    """Return whether *store* carries everything *src* held.

    The precondition for ``--delete-sources``, and deliberately **stronger
    than ``valid_staged_store``** (ledger MIG-20). Structural validity proves a
    store is well-formed, not that the conversion preserved content -- and both
    Criticals this design round found (a dropped ``Metadata_ImageType``, a
    dropped ``phenotypic_work_id``) produce structurally valid stores. Unlinking
    the ``.h5`` on that evidence loses the original permanently, with no receipt
    and no rollback.

    The comparison is **value-level, not key-level** (ledger MIG-28). MIG-2 is
    not a dropped key: the loader's restore loop *skips* a key the constructor
    already set, so ``Metadata_ImageType`` is **present**, carrying ``"Image"``
    where the file said ``"GridSection"``. Two identical key sets, one wrong
    value, and a key-set comparison returns ``True``.

    What is compared: the three metadata sections as full mappings; a content
    digest per layer; ``phenotypic_work_id``; and ``nrows``/``ncols`` when the
    source is a ``GridImage``.

    Args:
        src: The legacy ``.h5`` about to be deleted.
        store: The store that replaced it.

    Returns:
        ``True`` only when every compared field agrees.
    """
    from ._io_constants import load_image_from_store

    try:
        if not valid_staged_store(store):
            return False
        source_image = _load_for_migration(src)
        stored_image = load_image_from_store(store)

        if type(source_image).__name__ != type(stored_image).__name__:
            return False

        for section in ("protected", "public", "imported"):
            expected = dict(getattr(source_image._metadata, section))
            actual = dict(getattr(stored_image._metadata, section))
            if {str(k): v for k, v in expected.items()} != {
                str(k): v for k, v in actual.items()
            }:
                return False

        for layer in _FAITHFULNESS_LAYERS:
            expected_array = getattr(source_image, layer)[:]
            actual_array = getattr(stored_image, layer)[:]
            if _layer_digest(expected_array) != _layer_digest(actual_array):
                return False

        from .ngff_ import PhenotypicAttr, read_phenotypic_attributes

        block = read_phenotypic_attributes(store)
        if _stored_work_id(src) != block.get(PhenotypicAttr.WORK_ID):
            return False

        # Narrowed by class, not by `hasattr`: mypy cannot see through the
        # latter, and `Image` genuinely has no `nrows`.
        from phenotypic import GridImage

        if isinstance(source_image, GridImage):
            if not isinstance(stored_image, GridImage):
                return False
            if (source_image.nrows, source_image.ncols) != (
                stored_image.nrows,
                stored_image.ncols,
            ):
                return False
    except Exception:  # noqa: BLE001 - a comparison that cannot run is a refusal
        return False
    return True


# ---------------------------------------------------------------------------
# The canonical metadata view
# ---------------------------------------------------------------------------

#: Name of the derived view emitted beside the snapshot. It is **not** the
#: snapshot and carries no provenance role, so it needs no digest in state.
CANONICAL_METADATA_CSV_NAME: str = "metadata.canonical.csv"


def canonical_metadata_view_path(output_dir: Path) -> Path:
    """Return where the derived canonical metadata view lives.

    Beside the snapshot, in the same ``deliverables/`` directory. Migration is
    in place, so there is no second location to reconcile.

    Args:
        output_dir: Run output root.

    Returns:
        ``<output>/deliverables/metadata.canonical.csv``.
    """
    return deliverables_dir(Path(output_dir)) / CANONICAL_METADATA_CSV_NAME


def emit_canonical_metadata_view(output_dir: Path) -> Path | None:
    """Derive a canonical-header view of the metadata snapshot.

    **``deliverables/metadata.csv`` is never touched.** It is immutable input
    provenance (user ruling, ledger FLOW-4): the CLI recomputes
    ``metadata_sha256`` from that file on *every* run rather than reading it
    back from state, so rewriting it makes the next run's
    ``expected_finalization`` diverge from the published
    ``finalization_input_digest`` and re-finalize the whole tree -- whatever
    migration wrote into ``state.config``. No ``metadata.original.csv`` is
    created either; that file only existed to make the withdrawn rewrite
    reversible.

    The view is additive and optional. Canonicalization goes through the SDK's
    own :func:`normalize_metadata_columns` -- the primitive the read path
    already delegates to -- rather than a second mapping.

    Args:
        output_dir: Run output root.

    Returns:
        The view's path, or ``None`` when there is no snapshot to derive from.
    """
    import polars as pl

    from ._metadata_helpers import normalize_metadata_columns

    source = metadata_csv_deliverable_path(Path(output_dir))
    if not source.is_file():
        return None
    # `infer_schema_length=0` reads every column as a string: this is a view of
    # the user's own bytes, and inferring dtypes would rewrite values (a
    # zero-padded well id losing its padding, say) that the snapshot holds
    # verbatim.
    frame = pl.read_csv(source, infer_schema_length=0)
    target = canonical_metadata_view_path(output_dir)
    normalize_metadata_columns(frame).write_csv(target)
    return target


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
        hdf_dir = dataset_dir / _DIR_HDF
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


# ---------------------------------------------------------------------------
# Run state: markers and the aggregate
# ---------------------------------------------------------------------------


def _republish_image_marker(
    output_dir: Path, dataset: str, stem: str, store: Path
) -> bool:
    """Rewrite one per-image marker so it describes the store. Never create one.

    **Rewrites; never creates** (ledger FLOW-37). A *missing* marker also
    "does not describe the store", and the looser wording fired on every
    image of a pre-markers archive -- where ``publish_image_success`` has no
    ``work_id``, ``attempt_id`` or ``lifecycle_epoch`` to be given, and,
    unlike its three siblings, does not short-circuit on
    ``success_markers_required``.

    **Replaces the artifact set; does not add to it** (ledger MIG-22). The
    stale ``"hdf"`` descriptor still validates under the default
    ``keep_source=True``, so merely adding a store descriptor beside it hides
    the defect entirely. Every surviving key keeps its **literal name** --
    ``_current_success_work_ids`` indexes ``artifacts["measurements"]`` by
    name -- but its descriptor is re-fingerprinted, because those are the
    bytes the migration's metadata pass just rewrote.

    ``work_id``, ``attempt_id`` and ``lifecycle_epoch`` are preserved: they
    identify the run that produced the result, and rewriting them would
    falsely re-attribute it to the migration.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        stem: Image stem.
        store: The promoted store.

    Returns:
        Whether a marker was rewritten.
    """
    from phenotypic._cli._cli_completion import (
        ARTIFACT_KIND_STORE,
        SUCCESS_MARKER_VERSION,
        _artifact_descriptor,
    )

    from ._atomic_io import atomic_write_json
    from ._io_constants import image_completion_marker_path

    marker_path = image_completion_marker_path(output_dir, dataset, stem)
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(marker, dict) or not isinstance(
        marker.get("artifacts"), dict
    ):
        return False

    output_root = Path(output_dir).resolve()
    artifacts: dict[str, dict] = {}
    for name, descriptor in marker["artifacts"].items():
        if name == "hdf" or not isinstance(descriptor, dict):
            continue
        relative = descriptor.get("path")
        if not isinstance(relative, str):
            continue
        resolved = (output_root / relative).resolve()
        if not resolved.exists():
            continue
        artifacts[name] = _artifact_descriptor(
            resolved, resolved.relative_to(output_root)
        )
    resolved_store = store.resolve()
    artifacts[ARTIFACT_KIND_STORE] = _artifact_descriptor(
        resolved_store, resolved_store.relative_to(output_root)
    )

    marker["version"] = SUCCESS_MARKER_VERSION
    marker["artifacts"] = artifacts
    atomic_write_json(marker_path, marker)
    return True


def republish_image_markers(output_dir: Path) -> int:
    """Rewrite every existing marker whose image now has a valid store.

    **Keyed on marker state, not on conversion state** (ledger FLOW-22).
    Trace an interruption: migration promotes image X's store, then dies
    before rewriting X's marker. On resume X is *skipped*, because its store
    already passes ``valid_staged_store`` -- so republication riding on "was
    converted this run" would leave X at v1 forever, and the next local run
    would reprocess it from source inputs a migrated archive may no longer
    have.

    The operation is idempotent, so running it over skipped images costs a
    marker read.

    Args:
        output_dir: Run output root.

    Returns:
        How many markers were rewritten.
    """
    from ._io_constants import dataset_results_dir, results_dir

    republished = 0
    root = results_dir(Path(output_dir))
    if not root.is_dir():
        return 0
    for dataset_dir in sorted(
        path for path in root.iterdir() if path.is_dir()
    ):
        zarr_dir = dataset_results_dir(output_dir, dataset_dir.name) / "zarr"
        if not zarr_dir.is_dir():
            continue
        for store in sorted(zarr_dir.glob(f"*{STORE_SUFFIX}")):
            if store.name.startswith(".") or not valid_staged_store(store):
                continue
            stem = store.name[: -len(STORE_SUFFIX)]
            if _republish_image_marker(
                Path(output_dir), dataset_dir.name, stem, store
            ):
                republished += 1
    return republished


def republish_aggregate(output_dir: Path) -> bool:
    """Re-publish the aggregate marker over the migrated tree.

    Guarded, because a legacy tree with no markers is a **documented no-op,
    not an exception** (ledger MIG-23). ``publish_aggregate_snapshot`` raises
    when state is missing or no marker is authorized, and resolves the four
    deliverables paths with ``strict=True``. A pre-markers archive is a likely
    migration subject; aborting there would leave the stores written and the
    run reported as failed.

    Args:
        output_dir: Run output root.

    Returns:
        Whether an aggregate marker was published.
    """
    from phenotypic._cli._cli_completion import (
        current_success_counts,
        publish_aggregate_snapshot,
    )
    from phenotypic._cli._cli_state_management import load_processing_state

    try:
        state = load_processing_state(output_dir)
    except (KeyError, TypeError, ValueError):
        return False
    if state is None or not state.config.get(
        "success_markers_required", False
    ):
        return False
    counts = current_success_counts(output_dir)
    if counts is None or counts[0] == 0:
        return False
    try:
        publish_aggregate_snapshot(output_dir)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _marker_authority_permits_unlink(
    output_dir: Path, dataset: str, stem: str
) -> bool:
    """Return whether the per-image marker allows this source to be deleted.

    The second half of ``--delete-sources``' precondition: the image must
    still validate as a success **after** republication, not merely have
    converted. A tree that never required success markers -- a pre-markers
    archive -- has no authority to satisfy, and refusing there would make
    ``--delete-sources`` unreachable for exactly the oldest archives (ledger
    MIG-23).

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        stem: Image stem.

    Returns:
        ``True`` when there is no marker authority to satisfy, or when the
        image's marker validates.
    """
    # Lazy, and across the layer on purpose: marker authority lives in the
    # CLI, and this module is migration glue rather than general SDK surface.
    from phenotypic._cli._cli_completion import valid_image_success
    from phenotypic._cli._cli_state_management import load_processing_state

    from ._io_constants import image_completion_marker_path

    try:
        state = load_processing_state(output_dir)
    except (KeyError, TypeError, ValueError):
        return False
    if state is None or not state.config.get(
        "success_markers_required", False
    ):
        return True
    if not image_completion_marker_path(output_dir, dataset, stem).is_file():
        # Nothing was ever certified for this image, so nothing is invalidated
        # by deleting a source the forward path no longer reads.
        return True
    work_ids = state.config.get("work_ids", {})
    images = work_ids.get(dataset, {}) if isinstance(work_ids, dict) else {}
    work_id = next(
        (
            value
            for name, value in images.items()
            if isinstance(name, str) and Path(name).stem == stem
        ),
        None,
    )
    if not isinstance(work_id, str):
        return False
    return valid_image_success(
        output_dir, dataset=dataset, image_stem=stem, work_id=work_id
    )


def _reclaim_sources(output_dir: Path) -> list[tuple[Path, str]]:
    """Delete every source whose conversion is provably faithful.

    Per image, immediately before that image's unlink -- so a single
    unfaithful image does not block the other ninety-nine from reclaiming
    space, and the run still reports non-clean, naming every source left in
    place.

    Args:
        output_dir: Run output root.

    Returns:
        ``(source, reason)`` for every source deliberately left behind.
    """
    refusals: list[tuple[Path, str]] = []
    for dataset, hdf_path in list(iter_legacy_hdfs(output_dir)):
        store = zarr_store_path(output_dir, dataset, hdf_path.stem)
        # Read through the module namespace so a monkeypatch of the gate --
        # which is how the refusal path is exercised -- actually takes effect.
        import sys

        gate = sys.modules[__name__]._conversion_is_faithful
        if not gate(hdf_path, store):
            refusals.append(
                (
                    hdf_path,
                    "re-read of the converted store does not match the source; "
                    "source retained",
                )
            )
            continue
        if not _marker_authority_permits_unlink(
            output_dir, dataset, hdf_path.stem
        ):
            refusals.append(
                (
                    hdf_path,
                    "the image does not validate as a success after "
                    "migration; source retained",
                )
            )
            continue
        hdf_path.unlink()
    return refusals


def migrate_run_hdf_to_zarr(
    output_dir: Path,
    *,
    keep_source: bool = True,
    njobs: int = 1,
    dry_run: bool = False,
    finalize_publication: bool = True,
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
        finalize_publication: Publish legacy markers/aggregates immediately.
            The combined CLI migration sets this false until tables are embedded.

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

    if finalize_publication:
        # Markers first, then the aggregate: source_set_digest is computed from
        # valid_image_success, so aggregate publication must remain last.
        republish_image_markers(output_dir)
        republish_aggregate(output_dir)
        emit_canonical_metadata_view(output_dir)

        if not keep_source:
            # After conversion, and after marker republication, so the
            # precondition sees the final state of the tree.
            failed.extend(_reclaim_sources(output_dir))

    return MigrationReport(
        converted=converted, skipped=skipped, failed=tuple(failed)
    )


__all__ = [
    "CANONICAL_METADATA_CSV_NAME",
    "republish_aggregate",
    "republish_image_markers",
    "_conversion_is_faithful",
    "MigrationReport",
    "canonical_metadata_view_path",
    "emit_canonical_metadata_view",
    "iter_legacy_hdfs",
    "migrate_hdf_to_zarr",
    "migrate_run_hdf_to_zarr",
]
