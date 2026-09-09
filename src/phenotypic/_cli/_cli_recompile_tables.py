"""Per-store embedded measurement-table rewrites for recompile."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from phenotypic.sdk_ import (
    CommitGuard,
    DIR_IMAGE_COMPLETE,
    DIR_IMAGE_RECORDS,
    MEASUREMENT_TABLE_RELATIVE_PATH,
    PhenotypicAttr,
    STORE_SUFFIX,
    progress_dir,
    read_phenotypic_attributes,
    store_stem,
)

from ._cli_completion import (
    authorized_measurement_sources,
    publish_image_success,
)
from ._embedded_measurement_tables import prepare_embedded_measurement_table
from ._cli_recompile_recovery import (
    _fsync_recompile_directory,
    assert_no_unrecoverable_measurement_authority,
    begin_recompile_table_transition,
    clear_recompile_table_transition,
    recoverable_recompile_measurement_sources,
    promote_recompile_table_transition,
    recompile_store_lock_path,
)
from phenotypic.sdk_._file_locking import exclusive_path_lock
from phenotypic.sdk_._image_record import record_rejection
from phenotypic.sdk_._run_state import marker_rejection


def _marker_artifacts(output_dir: Path, marker: dict) -> dict[str, Path]:
    """Resolve the existing marker's artifacts below its output root."""
    raw = marker.get("artifacts")
    if not isinstance(raw, dict):
        raise ValueError("Image completion marker has no artifact mapping")
    artifacts: dict[str, Path] = {}
    output_root = Path(output_dir).resolve()
    for name, descriptor in raw.items():
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise ValueError("Image completion marker has invalid artifacts")
        relative = descriptor.get("path")
        if not isinstance(relative, str):
            raise ValueError("Image completion marker artifact has no path")
        resolved = (output_root / relative).resolve()
        resolved.relative_to(output_root)
        artifacts[name] = resolved
    return artifacts


def _republish_table_marker(
    output_dir: Path,
    marker_path: Path,
    *,
    commit_guard: CommitGuard | None,
    lifecycle_epoch: str | None = None,
) -> None:
    """Rehash all existing artifacts and publish the marker last."""
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    publish_image_success(
        output_dir,
        work_id=str(marker["work_id"]),
        dataset=str(marker["dataset"]),
        relative_image_path=str(marker["relative_image_path"]),
        image_stem=str(marker["image_stem"]),
        mode=str(marker["mode"]),
        attempt_id=str(marker["attempt_id"]),
        lifecycle_epoch=(
            lifecycle_epoch
            if lifecycle_epoch is not None
            else str(marker["lifecycle_epoch"])
        ),
        artifacts=_marker_artifacts(output_dir, marker),
        commit_guard=commit_guard,
    )


def _declares_metadata_table(store_path: Path) -> bool:
    """Return whether *store_path* declares an embedded metadata table.

    The single predicate behind both the whole-tree pre-flight scan and the
    per-store guard. One producer, deliberately: a scan that answered this
    question differently from the guard it fronts would either refuse a tree
    the guard would have accepted, or -- far worse -- pass a tree the guard
    then aborts halfway through, which is the exact failure the scan exists to
    prevent.

    Unreadable attributes read as "not inverted" here. That is not a judgement
    that the store is fine; it defers to the specific diagnosis
    :func:`recompile_embedded_measurement_table` raises a few lines later,
    rather than replacing it with a vaguer one.

    Args:
        store_path: A promoted store's root directory.

    Returns:
        ``True`` when the store root declares ``tables.metadata``.
    """
    from phenotypic.sdk_.ngff_ import METADATA_TABLE_GROUP

    try:
        attrs = read_phenotypic_attributes(store_path)
    except (OSError, KeyError, ValueError):
        return False
    tables = attrs.get(PhenotypicAttr.TABLES)
    return isinstance(tables, dict) and METADATA_TABLE_GROUP in tables


def _refuse_inverted_stores_before_any_write(
    authorized: dict[Path, str],
) -> None:
    """Refuse the whole run if **any** authorized store is inverted.

    **Ordered before the first store's lock, not merely before each store's
    own write.** The per-store :func:`_refuse_inverted_store` is correctly
    ordered *within* a store, so on a uniform tree it fails having destroyed
    nothing. A **mixed** tree is the case it cannot cover: recompile rewrites
    the un-inverted stores it reaches first, then aborts on an inverted one,
    leaving exactly the mixed Parquet generations
    :func:`recompile_embedded_measurement_tables` warns about -- and leaving
    them by this guard's own action rather than by an interruption.

    Mixed trees are reachable, not hypothetical. ``prepare_image_tables``
    emits a metadata table only when ``deliverables/metadata.csv`` exists at
    the moment the image is written, and ``metadata_csv`` is in **neither**
    ``processing_configuration_digest`` nor ``compute_work_id``
    (``_cli_failure_tracker.py``). So a run begun without ``--metadata`` and
    resumed with it keeps every already-complete image's un-inverted store --
    continuation does not reprocess them, because their work ids did not
    change -- while every newly processed image gets an inverted one.

    Args:
        authorized: The full authorized ``table path -> dataset`` mapping,
            already merged with recovery sources.

    Raises:
        RuntimeError: If any authorized store declares a metadata table.
    """
    inverted = sorted(
        {
            table_path.parents[2]
            for table_path in authorized
            if tuple(table_path.parts[-3:])
            == MEASUREMENT_TABLE_RELATIVE_PATH.parts
            and _declares_metadata_table(table_path.parents[2])
        }
    )
    if not inverted:
        return
    listed = "\n  ".join(str(path) for path in inverted)
    raise RuntimeError(
        f"Cannot recompile: {len(inverted)} of {len(authorized)} authorized "
        "store(s) are inverted -- they declare an embedded metadata table, "
        "and recompile still rejoins metadata into the measurement table. "
        "Refusing the whole run before rewriting any store, because "
        "rewriting the rest first would leave mixed Parquet generations.\n  "
        f"{listed}\n"
        "Recompile is unsupported on runs built with --metadata. To re-join "
        "an updated metadata CSV, re-run the original forward command with "
        "the new --metadata and --force-local: finalization performs the "
        "join, so it refreshes the deliverables without rewriting any store. "
        "Note that --mode measure is NOT a substitute -- it never snapshots "
        "the CSV, so it would silently re-join the old one."
    )


def _refuse_inverted_store(store_path: Path) -> None:
    """Refuse a store whose tables this producer would silently un-invert.

    The per-store half of the refusal, kept as defence in depth behind
    :func:`_refuse_inverted_stores_before_any_write`. The scan is what makes
    a mixed tree safe; this is what makes a *single* store safe even if a
    future caller reaches :func:`recompile_embedded_measurement_table`
    directly, bypassing the scan.

    INVERTED-STORE ARM -- DELETE WHEN: **P7 Task 5 Step 1e** repoints
    :func:`recompile_embedded_measurement_tables` onto ``prepare_image_tables``
    and deletes this guard (``phase-7-migrate-mode.md``). Retire
    :func:`_refuse_inverted_stores_before_any_write` and
    :func:`_declares_metadata_table` in the same edit; all three exist for
    this one window and are meaningless after it.

    **Not Step 1d, and not the schema gate.** The other legacy arms this phase
    added (``_cli_recompile_recovery.py``, ``_cli_finalize_run.py``) retire
    when ``SCHEMA_GATE_ARMED`` starts refusing *legacy* trees. That does
    nothing here: an inverted store is a **forward** tree, the shape this
    build writes, so the gate would wave it straight through to the producer
    that un-inverts it. Folding this trigger into 1d's would retire the guard
    while the hazard it covers is still live.

    Recompile still uses the **pre-inversion** producer: it re-joins the
    metadata snapshot into ``tables/measurements/table.parquet`` and writes no
    ``tables/metadata/pht-metadata.parquet``. Run against an inverted store,
    that un-inverts it -- a joined measurements table, and a dropped metadata
    table, on a tree whose other stores are split. Nothing raises and nothing
    reads differently until someone aggregates.

    P4 Task 1 expected the ``isinstance`` check above to catch this by
    failing closed. **It does not**: the producer still returns the legacy
    type, so the check passes and the un-inversion is silent. Loud-and-broken
    beats silent-and-wrong, so this restores the refusal the phase intended
    until the producer is repointed.

    Args:
        store_path: The promoted store recompile is about to rewrite.

    Raises:
        RuntimeError: If the store declares an embedded metadata table.
    """
    from phenotypic.sdk_.ngff_ import METADATA_TABLE_GROUP

    if _declares_metadata_table(store_path):
        raise RuntimeError(
            "Cannot recompile an inverted store: "
            f"{store_path} declares tables.{METADATA_TABLE_GROUP}, and "
            "recompile still rejoins metadata into the measurement table. "
            "Rewriting it here would drop the store's metadata table and "
            "un-invert its measurements."
        )


def _replace_and_republish_table(
    output_dir: Path,
    dataset: str,
    store_path: Path,
    prepared: object,
    *,
    commit_guard: CommitGuard | None,
    lifecycle_epoch: str | None,
) -> None:
    """Journal, replace, marker-publish, and clear under one store lock."""
    from phenotypic.sdk_ import image_record_path
    from phenotypic.sdk_._measurement_tables import (
        PreparedEmbeddedMeasurementTable,
    )

    if not isinstance(prepared, PreparedEmbeddedMeasurementTable):
        raise TypeError(
            "Recompile table preparation returned an invalid payload"
        )
    _refuse_inverted_store(store_path)
    stem = store_stem(store_path)
    with exclusive_path_lock(
        recompile_store_lock_path(output_dir, dataset, stem),
        timeout=60.0,
    ):
        staged = begin_recompile_table_transition(
            output_dir,
            dataset,
            stem,
            store_path,
            prepared,
        )
        promote_recompile_table_transition(
            output_dir,
            dataset,
            stem,
            store_path,
            staged,
            commit_guard=commit_guard,
        )
        # The record, for the same reason as the measure path -- see
        # `_cli_process_single`. This is the recompile half of the same
        # defect: after D1 the legacy marker is absent on a forward tree, so
        # `_republish_table_marker` would read a file that is not there.
        #
        # **That was only half of what this function did with the legacy
        # path.** The other half was the lock taken above: it derived from
        # `image_completion_marker_path`, and `exclusive_path_lock` mkdirs the
        # parent, so entering this block *created* `progress/image_complete/`
        # on a forward tree -- which schema signal 1 probes by directory
        # existence, making `requires_conversion` answer CONVERT for a tree
        # this build wrote. Both halves are fixed; `recompile_store_lock_path`
        # now derives from the record path too, and carries the reasoning.
        record_path = image_record_path(output_dir, dataset, stem)
        _republish_table_marker(
            output_dir,
            record_path,
            commit_guard=commit_guard,
            lifecycle_epoch=lifecycle_epoch,
        )
        _fsync_recompile_directory(record_path.parent)
        clear_recompile_table_transition(output_dir, dataset, stem)


def _standalone_marker_sources(output_dir: Path) -> dict[Path, str]:
    """Discover valid embedded authority when no processing state is present.

    **Both shapes, each on its own predicate** -- the seventh site of the
    defect fixed in ``authorized_measurement_sources``. This globbed the
    legacy ``image_complete/`` tree and then asked ``valid_image_success``, a
    *record* predicate, whether each legacy marker was valid, so after D1's
    clean break every image on a legacy tree was skipped and this returned
    ``{}``.

    Its consequence is narrower than arm 1's and worth stating exactly,
    because "returns empty" is not by itself a bug here. At the call site,
    ``{}`` with no recovery sources still falls through to the ``--mode
    migrate`` error -- correct, but by accident. **With recovery sources
    present, ``authorized`` becomes recovery-only and every image whose
    authority came from a marker is silently dropped from the authorized
    set.** Not a loud error going quiet, but authority missing for a subset.

    ``_payload_authorizes`` is imported rather than reimplemented: the
    identity check and the ``fenced_artifact_path`` walk are the same work
    whichever shape is in hand, and a second copy is how this arm and arm 1
    drifted apart in the first place. The embedded-table filter below stays
    local, because it is this function's own question -- arm 1 authorizes any
    described measurement source, while recompile only rewrites tables that
    live *inside* a store.
    """
    from ._cli_completion import _payload_authorizes

    sources: dict[Path, str] = {}
    output_root = Path(output_dir).resolve()
    progress = progress_dir(output_dir)
    shapes = (
        (progress / DIR_IMAGE_RECORDS, record_rejection),
        (progress / DIR_IMAGE_COMPLETE, marker_rejection),
    )
    for root, rejection in shapes:
        for payload_path in sorted(root.glob("*/*.json")):
            try:
                payload = json.loads(
                    payload_path.read_text(encoding="utf-8")
                )
                if not _payload_authorizes(output_root, payload, rejection):
                    continue
                dataset = str(payload["dataset"])
                relative = payload["artifacts"]["measurements"]["path"]
                table_path = (output_dir / str(relative)).resolve()
                if tuple(table_path.parts[-3:]) != (
                    MEASUREMENT_TABLE_RELATIVE_PATH.parts
                ):
                    continue
            except (
                KeyError,
                TypeError,
                ValueError,
                OSError,
                json.JSONDecodeError,
            ):
                continue
            sources[table_path] = dataset
    return sources


def recompile_embedded_measurement_table(
    output_dir: Path,
    table_path: Path,
    dataset: str,
    metadata_csv: Path | None,
    *,
    commit_guard: CommitGuard | None = None,
    lifecycle_epoch: str | None = None,
) -> None:
    """Rewrite one authorized embedded table and republish its marker last."""

    table_path = Path(table_path)
    if tuple(table_path.parts[-3:]) != MEASUREMENT_TABLE_RELATIVE_PATH.parts:
        raise RuntimeError(
            "Current-schema recompile requires embedded measurement tables; "
            "run --mode migrate"
        )
    store_path = table_path.parents[2]
    attrs = read_phenotypic_attributes(store_path)
    descriptor = attrs.get(PhenotypicAttr.TABLES, {}).get("measurements")
    if not isinstance(descriptor, dict):
        raise ValueError(f"Store lacks measurement descriptor: {store_path}")
    raw_baseline = descriptor.get("measurement_columns")
    if not isinstance(raw_baseline, list) or not all(
        isinstance(column, str) for column in raw_baseline
    ):
        raise ValueError(
            f"Store has invalid measurement baseline: {store_path}"
        )
    payload = pq.read_table(table_path).to_pandas()
    missing = [
        column for column in raw_baseline if column not in payload.columns
    ]
    if missing:
        raise ValueError(
            f"Embedded table cannot project its baseline {missing}: {table_path}"
        )
    prepared = prepare_embedded_measurement_table(
        payload.loc[:, raw_baseline], metadata_csv
    )
    _replace_and_republish_table(
        output_dir,
        dataset,
        store_path,
        prepared,
        commit_guard=commit_guard,
        lifecycle_epoch=lifecycle_epoch,
    )


def recompile_embedded_measurement_tables(
    output_dir: Path,
    metadata_csv: Path | None,
    *,
    commit_guard: CommitGuard | None = None,
    lifecycle_epoch: str | None = None,
) -> int:
    """Project, rejoin, replace, and marker-publish every authorized table.

    A legacy-only run is refused with a migration remedy. Once the first store
    is rewritten, any interruption leaves mixed Parquet generations; aggregate
    publication independently rejects that state until a retry converges.

    An **inverted** store is refused before the first rewrite rather than on
    reaching it -- see :func:`_refuse_inverted_stores_before_any_write`, which
    exists precisely so this function cannot *itself* create the mixed state
    the paragraph above describes as an interruption hazard.
    """

    output_dir = Path(output_dir)
    authorized = authorized_measurement_sources(output_dir)
    dataset_names = (
        sorted(
            path.name
            for path in (output_dir / "results").iterdir()
            if path.is_dir()
        )
        if (output_dir / "results").is_dir()
        else []
    )
    recovery_sources = recoverable_recompile_measurement_sources(
        output_dir, dataset_names
    )
    if authorized is None:
        marker_sources = _standalone_marker_sources(output_dir)
        if marker_sources or recovery_sources:
            authorized = {**marker_sources, **recovery_sources}
        else:
            legacy = sorted(
                (output_dir / "results").glob("*/measurements/*.parquet")
            )
            image_sources = list(
                (output_dir / "results").glob("*/hdf/*.h5")
            ) + list((output_dir / "results").glob(f"*/zarr/*{STORE_SUFFIX}"))
            if legacy and image_sources:
                raise RuntimeError(
                    "Legacy external measurement Parquets require --mode migrate "
                    "before recompile"
                )
            return 0

    authorized = {**authorized, **recovery_sources}
    assert_no_unrecoverable_measurement_authority(
        output_dir,
        dataset_names,
        set(authorized),
    )
    # Before the first store's lock, never inside the loop -- see the
    # function's docstring for why per-store ordering is not enough on a
    # mixed tree.
    _refuse_inverted_stores_before_any_write(authorized)
    changed = 0
    for table_path, dataset in sorted(
        authorized.items(), key=lambda item: str(item[0])
    ):
        recompile_embedded_measurement_table(
            output_dir,
            table_path,
            dataset,
            metadata_csv,
            commit_guard=commit_guard,
            lifecycle_epoch=lifecycle_epoch,
        )
        changed += 1
    return changed


__all__ = [
    "recompile_embedded_measurement_table",
    "recompile_embedded_measurement_tables",
]
