"""Recompile rewrites embedded tables rather than rejoining an aggregate."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import shutil
from pathlib import Path

import polars as pl

from phenotypic._cli._cli_completion import valid_image_success
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    image_record_path,
    zarr_store_path,
)


def _pixel_digest(store: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((store / "rgb").rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(store).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_recompile_replaces_each_embedded_table_and_refreshes_marker(
    tmp_path: Path,
) -> None:
    """New metadata reaches stores first and marker authority is republished last."""
    module_name = "phenotypic._cli._cli_recompile_tables"
    assert importlib.util.find_spec(module_name) is not None, (
        "embedded-table recompile phase is missing"
    )
    recompile_tables = importlib.import_module(module_name)

    from click.testing import CliRunner
    from skimage.io import imsave

    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_process_single import main
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize

    input_root = tmp_path / "input"
    input_root.mkdir()
    image_path = input_root / "plate.tiff"
    imsave(
        str(image_path), load_synth_yeast_plate().rgb[:], check_contrast=False
    )
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        ImagePipeline(ops=[OtsuDetector()], meas=[MeasureSize()]).to_json(),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    result = CliRunner().invoke(
        main,
        [
            "--pipeline",
            str(pipeline_path),
            "--image",
            str(image_path),
            "--output-dir",
            str(output),
            "--dataset-name",
            "input",
            "--input-root",
            str(input_root),
            "--no-save-overlays",
        ],
    )
    assert result.exit_code == 0, result.output
    store = zarr_store_path(output, "input", "plate")
    pixels_before = _pixel_digest(store)
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): ["plate"],
            "Strain": ["mutant"],
        }
    ).write_csv(metadata)

    changed = recompile_tables.recompile_embedded_measurement_tables(
        output, metadata
    )

    assert changed == 1
    table = pl.read_parquet(store / MEASUREMENT_TABLE_RELATIVE_PATH)
    assert table["Metadata_Strain"].to_list() == ["mutant"] * table.height
    assert _pixel_digest(store) == pixels_before
    # The RECORD. `_cli_process_single.main` above is the forward publisher,
    # and D1's clean break moved what it writes out of `image_complete/`, so
    # the legacy path is simply absent on the tree this test just built.
    record = __import__("json").loads(
        image_record_path(output, "input", "plate").read_text(
            encoding="utf-8"
        )
    )
    assert valid_image_success(
        output,
        dataset="input",
        image_stem="plate",
        work_id=record["work_id"],
    )


def _build_store(
    *,
    pipeline_path: Path,
    input_root: Path,
    image_path: Path,
    output: Path,
) -> None:
    """Drive the real forward worker for one image, asserting it succeeded."""
    from click.testing import CliRunner

    from phenotypic._cli._cli_process_single import main

    result = CliRunner().invoke(
        main,
        [
            "--pipeline",
            str(pipeline_path),
            "--image",
            str(image_path),
            "--output-dir",
            str(output),
            "--dataset-name",
            "input",
            "--input-root",
            str(input_root),
            "--no-save-overlays",
        ],
    )
    assert result.exit_code == 0, result.output


def _table_sha256(output: Path, stem: str) -> str:
    path = (
        zarr_store_path(output, "input", stem) / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_mixed_tree_is_refused_before_any_store_is_rewritten(
    tmp_path: Path,
) -> None:
    """The pre-flight scan, and the case the per-store guard cannot cover.

    ``_refuse_inverted_store`` is correctly ordered *within* a store, so a
    uniform ``--metadata`` tree fails on the first store having destroyed
    nothing. A **mixed** tree is different: absent a whole-tree scan,
    recompile rewrites the un-inverted stores it reaches first and only then
    hits an inverted one, leaving exactly the mixed Parquet generations
    ``recompile_embedded_measurement_tables`` documents as an interruption
    hazard -- except caused by the guard rather than by an interruption.

    **The tree is built by the real vector, not by hand-editing a root.**
    ``prepare_image_tables`` emits a metadata table only when
    ``deliverables/metadata.csv`` exists as the image is written, and
    ``metadata_csv`` is in neither ``processing_configuration_digest`` nor
    ``compute_work_id`` -- so a run begun without ``--metadata`` and resumed
    with it keeps every finished image's un-inverted store (their work ids did
    not change, so continuation does not reprocess them) and inverts only the
    new ones. That is what the two invocations below reproduce.

    **The control is what stops a do-nothing scan from passing.** Phase 3
    re-runs the identical call on the identical tree with only the inverted
    store removed, and requires ``aaa``'s bytes to change. So "unchanged" in
    phase 2 is a fact about the scan, not about a store recompile would have
    left alone anyway:

    * a scan that does nothing fails phase 2's ``raises``;
    * a scan ordered after the first write fails phase 2's byte equality;
    * a scan that refuses every tree fails phase 3.
    """
    import json

    import pytest
    from skimage.io import imsave

    from phenotypic import ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize
    from phenotypic.sdk_ import metadata_csv_deliverable_path
    from phenotypic.sdk_.ngff_ import METADATA_TABLE_GROUP, PhenotypicAttr
    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_tables,
    )

    input_root = tmp_path / "input"
    input_root.mkdir()
    pixels = load_synth_yeast_plate().rgb[:]
    # "aaa" sorts before "zzz", and the rewrite loop iterates sorted by table
    # path -- so absent the scan, aaa is rewritten BEFORE zzz is refused.
    for stem in ("aaa", "zzz"):
        imsave(
            str(input_root / f"{stem}.tiff"), pixels, check_contrast=False
        )
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        ImagePipeline(ops=[OtsuDetector()], meas=[MeasureSize()]).to_json(),
        encoding="utf-8",
    )
    output = tmp_path / "out"

    # Phase 1a: the run BEFORE --metadata -- aaa's store is un-inverted.
    _build_store(
        pipeline_path=pipeline_path,
        input_root=input_root,
        image_path=input_root / "aaa.tiff",
        output=output,
    )

    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): ["aaa", "zzz"],
            "Strain": ["mutant", "wildtype"],
        }
    ).write_csv(metadata)

    # Phase 1b: the resume WITH --metadata -- zzz's store is inverted.
    snapshot = metadata_csv_deliverable_path(output)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(metadata.read_bytes())
    _build_store(
        pipeline_path=pipeline_path,
        input_root=input_root,
        image_path=input_root / "zzz.tiff",
        output=output,
    )

    def _tables(stem: str) -> dict:
        root = json.loads(
            (zarr_store_path(output, "input", stem) / "zarr.json").read_text(
                encoding="utf-8"
            )
        )
        return root["attributes"][PhenotypicAttr.ROOT][PhenotypicAttr.TABLES]

    # The tree really is MIXED. Without this, everything below could hold on a
    # tree that was uniform in either direction.
    assert METADATA_TABLE_GROUP not in _tables("aaa")
    assert METADATA_TABLE_GROUP in _tables("zzz")

    before = _table_sha256(output, "aaa")

    # Phase 2: the whole run is refused, and aaa is untouched.
    with pytest.raises(RuntimeError, match="inverted"):
        recompile_embedded_measurement_tables(output, metadata)

    assert _table_sha256(output, "aaa") == before, (
        "the scan let recompile rewrite a store before refusing the tree"
    )

    # Phase 3, the control: same call, same tree, inverted store removed.
    # If this does not rewrite aaa, phase 2 proved nothing.
    shutil.rmtree(zarr_store_path(output, "input", "zzz"))
    image_record_path(output, "input", "zzz").unlink()

    assert recompile_embedded_measurement_tables(output, metadata) == 1
    assert _table_sha256(output, "aaa") != before, (
        "recompile would not have rewritten aaa anyway -- phase 2 is vacuous"
    )
    table = pl.read_parquet(
        zarr_store_path(output, "input", "aaa")
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    assert table["Metadata_Strain"].to_list() == ["mutant"] * table.height


# ---------------------------------------------------------------------------
# The per-store inverted-store guard: reached, and reached BEFORE any write
# ---------------------------------------------------------------------------


def _store_digests(store: Path) -> dict[str, str]:
    """Digest every byte of a store that the rewrite would touch.

    The measurements table, the metadata table, and the root -- the three
    artifacts ``_replace_and_republish_table`` rewrites. A guard that raises
    *after* the promote leaves the raise looking identical to one that raised
    first, and only these bytes tell them apart.
    """
    from phenotypic.sdk_ import METADATA_TABLE_RELATIVE_PATH

    digests: dict[str, str] = {}
    for relative in (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        METADATA_TABLE_RELATIVE_PATH,
        Path("zarr.json"),
    ):
        path = store / relative
        digests[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def test_the_inverted_store_guard_runs_before_the_rewrite_transaction(
    tmp_path: Path,
) -> None:
    """HIGH 4. The call site, pinned by behaviour rather than by a substring.

    This replaces ``test_the_recompile_guard_is_wired_into_the_rewrite_path``
    (``test_embedded_table_inversion.py``), which asserted
    ``"_refuse_inverted_store(store_path)" in inspect.getsource(...)``. A
    substring says the text occurs; it says nothing about *where*. Moving the
    call to the last line of the ``with exclusive_path_lock(...)`` block left
    all three guard tests green while the store was rewritten, its metadata
    table stranded and its measurements re-joined, before anything raised.

    So the guard is driven for real: an inverted store, a valid record, and
    the **single-store** entry point -- which bypasses
    ``_refuse_inverted_stores_before_any_write``, so the only thing that can
    raise ``inverted store`` here is the per-store guard on the path this
    test is about.

    Two independent claims, and the second is the one a substring cannot make:

    * it raises, and
    * every byte the rewrite would have touched is unchanged afterwards.
    """
    import pytest

    from phenotypic._cli._cli_recompile_tables import (
        recompile_embedded_measurement_table,
    )
    from phenotypic.sdk_ import metadata_csv_deliverable_path
    from phenotypic.sdk_.ngff_ import METADATA_TABLE_GROUP, PhenotypicAttr

    metadata, output = _inverted_store_run(tmp_path)
    store = zarr_store_path(output, "input", "plate")

    # STANDING RULE. The store must really BE inverted and really carry a
    # record, or the raise below could come from the guard refusing
    # everything, or the transaction failing for want of authority.
    root = __import__("json").loads(
        (store / "zarr.json").read_text(encoding="utf-8")
    )
    tables = root["attributes"][PhenotypicAttr.ROOT][PhenotypicAttr.TABLES]
    assert METADATA_TABLE_GROUP in tables, "the fixture store is not inverted"
    record = __import__("json").loads(
        image_record_path(output, "input", "plate").read_text(encoding="utf-8")
    )
    assert valid_image_success(
        output,
        dataset="input",
        image_stem="plate",
        work_id=record["work_id"],
    ), (
        "the store carries no valid record, so the rewrite transaction would "
        "have refused it anyway and the guard proves nothing"
    )
    assert metadata_csv_deliverable_path(output).is_file()

    before = _store_digests(store)

    with pytest.raises(RuntimeError, match="inverted store"):
        recompile_embedded_measurement_table(
            output,
            store / MEASUREMENT_TABLE_RELATIVE_PATH,
            "input",
            metadata,
        )

    assert _store_digests(store) == before, (
        "the guard raised, but only after the rewrite transaction had already "
        "replaced the store's bytes -- the un-inversion it exists to prevent "
        "happened anyway"
    )


def _inverted_store_run(tmp_path: Path) -> tuple[Path, Path]:
    """Build a one-image run whose store is inverted, via the real worker.

    ``prepare_image_tables`` splits the tables only when
    ``deliverables/metadata.csv`` exists as the image is written, so the
    snapshot is installed first and the forward worker does the rest. Nothing
    here hand-writes a store root.

    Returns:
        ``(metadata csv, output root)``.
    """
    from skimage.io import imsave

    from phenotypic import ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize
    from phenotypic.sdk_ import metadata_csv_deliverable_path

    input_root = tmp_path / "input"
    input_root.mkdir()
    imsave(
        str(input_root / "plate.tiff"),
        load_synth_yeast_plate().rgb[:],
        check_contrast=False,
    )
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text(
        ImagePipeline(ops=[OtsuDetector()], meas=[MeasureSize()]).to_json(),
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.csv"
    pl.DataFrame(
        {str(IMAGE.IMAGE_NAME): ["plate"], "Strain": ["mutant"]}
    ).write_csv(metadata)

    output = tmp_path / "out"
    snapshot = metadata_csv_deliverable_path(output)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(metadata.read_bytes())
    _build_store(
        pipeline_path=pipeline_path,
        input_root=input_root,
        image_path=input_root / "plate.tiff",
        output=output,
    )
    return metadata, output


# ---------------------------------------------------------------------------
# HIGH 3 -- the legacy leg of `_image_authority_shapes`, and its version
# ---------------------------------------------------------------------------


def _authority_payload(*, version: int, work_id: str = "w") -> dict:
    """A minimal authority payload stamped with *version*."""
    return {
        "version": version,
        "work_id": work_id,
        "dataset": "input",
        "image_stem": "plate",
    }


def test_each_authority_shape_is_paired_with_its_own_version(
    tmp_path: Path,
) -> None:
    """HIGH 3. ``_image_authority_shapes`` pairs a PATH with a VERSION.

    ``RECORD_VERSION`` is 1 and ``SUCCESS_MARKER_VERSION`` is 2, and
    ``_marker_allows_table_transition``'s docstring names what the pairing
    prevents in terms: *"checking one shape against the other's number
    returns False silently -- which is exactly how a path-only repoint would
    have disabled table-authority repair with nothing failing."* Every
    recompile fixture in the tree is a forward record tree, so before this
    test only leg 1 was ever returned: reversing the tuple, renumbering the
    legacy leg, or deleting it outright changed nothing anywhere.

    Three tree shapes, because each catches a different one of those:

    * record only -- the forward shape, and what deleting leg 1 breaks;
    * marker only -- the legacy shape, and what deleting leg 2 or
      renumbering it breaks;
    * both -- *"best first"*, and what reversing the tuple breaks.
    """
    import json

    import pytest

    from phenotypic._cli._cli_completion import SUCCESS_MARKER_VERSION
    from phenotypic._cli._cli_recompile_recovery import (
        image_authority_path,
        image_authority_payload,
    )
    from phenotypic.sdk_ import image_completion_marker_path
    from phenotypic.sdk_._image_record import RECORD_VERSION

    # STANDING RULE, and it is the whole premise: if the two shapes agreed on
    # their number there would be nothing for the pairing to get wrong, and
    # every assertion below would hold against a shape-blind constant.
    assert RECORD_VERSION != SUCCESS_MARKER_VERSION, (
        "the two authority shapes now carry the same version, so this test "
        "no longer distinguishes a paired check from a shape-blind one"
    )

    def _write(path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    record_only = tmp_path / "record-only"
    record = _write(
        image_record_path(record_only, "input", "plate"),
        _authority_payload(version=RECORD_VERSION, work_id="from-record"),
    )
    path, payload, version = image_authority_payload(
        record_only, "input", "plate"
    )
    assert (path, payload["work_id"], version) == (
        record,
        "from-record",
        RECORD_VERSION,
    )

    marker_only = tmp_path / "marker-only"
    marker = _write(
        image_completion_marker_path(marker_only, "input", "plate"),
        _authority_payload(
            version=SUCCESS_MARKER_VERSION, work_id="from-marker"
        ),
    )
    path, payload, version = image_authority_payload(
        marker_only, "input", "plate"
    )
    assert (path, payload["work_id"], version) == (
        marker,
        "from-marker",
        SUCCESS_MARKER_VERSION,
    ), (
        "the legacy leg was not returned with SUCCESS_MARKER_VERSION -- a "
        "legacy tree is judged against the record's number and every "
        "version check on it silently returns False"
    )

    both = tmp_path / "both"
    record = _write(
        image_record_path(both, "input", "plate"),
        _authority_payload(version=RECORD_VERSION, work_id="from-record"),
    )
    _write(
        image_completion_marker_path(both, "input", "plate"),
        _authority_payload(
            version=SUCCESS_MARKER_VERSION, work_id="from-marker"
        ),
    )
    path, payload, version = image_authority_payload(both, "input", "plate")
    assert (path, payload["work_id"], version) == (
        record,
        "from-record",
        RECORD_VERSION,
    ), "the record no longer outranks a legacy marker (\"best first\")"
    assert image_authority_path(both, "input", "plate") == record

    # The path-only sibling must agree with the payload reader on which shape
    # wins, or `assert_no_unrecoverable_measurement_authority` and the
    # transition validators would read two different files for one image.
    assert image_authority_path(marker_only, "input", "plate") == marker
    with pytest.raises(FileNotFoundError):
        image_authority_payload(tmp_path / "empty", "input", "plate")


def test_a_legacy_tree_repairs_overlay_authority_on_its_own_version(
    tmp_path: Path,
) -> None:
    """HIGH 3, the reachability half: the pairing consumed at a real site.

    The test above pins ``image_authority_payload`` by direct call. This one
    drives ``_marker_binds_overlay_and_table``, whose ``marker.get("version")
    != authority_version`` comparison is one of the four sites that consume
    the paired number -- and the one a legacy tree genuinely reaches, because
    it validates through ``_marker_work_id_matches_state`` rather than
    through the record-only ``valid_image_success``.

    A legacy tree is built from a real forward run by converting its record
    into an ``image_complete/`` marker, so every artifact digest in the
    payload describes bytes that are actually on disk.

    Both directions, because only the pair is the claim: stamped
    ``SUCCESS_MARKER_VERSION`` the marker must be accepted; stamped
    ``RECORD_VERSION`` -- the shape-blind number -- it must be refused.
    """
    import json

    from phenotypic._cli._cli_completion import (
        SUCCESS_MARKER_VERSION,
        publish_image_success,
    )
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        _marker_binds_overlay_and_table,
    )
    from phenotypic.sdk_ import (
        dataset_overlays_dir,
        image_completion_marker_path,
    )
    from phenotypic.sdk_._image_record import RECORD_VERSION

    _metadata, output = _inverted_store_run(tmp_path)
    store = zarr_store_path(output, "input", "plate")
    table = store / MEASUREMENT_TABLE_RELATIVE_PATH

    # The overlay has to be a declared artifact for this path to consider the
    # image at all, and absent from disk for `overlay_present=False`. Publish
    # it, then delete it -- which is the state the repair path is FOR.
    overlay = dataset_overlays_dir(output, "input") / "plate.png"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_bytes(b"overlay-bytes")
    record_path = image_record_path(output, "input", "plate")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    publish_image_success(
        output,
        work_id=str(record["work_id"]),
        dataset="input",
        relative_image_path=str(record["relative_image_path"]),
        image_stem="plate",
        mode=str(record["mode"]),
        attempt_id=str(record["attempt_id"]),
        lifecycle_epoch=str(record["lifecycle_epoch"]),
        artifacts={"store": store, "measurements": table, "overlay": overlay},
    )
    overlay.unlink()

    # Convert the forward tree into a LEGACY one: the same payload, stamped
    # with the legacy version, at the legacy path, and no record left behind.
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    marker_path = image_completion_marker_path(output, "input", "plate")
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.unlink()

    def _stamp(version: int) -> None:
        marker_path.write_text(
            json.dumps({**payload, "version": version}), encoding="utf-8"
        )

    # STANDING RULE: the tree must really be legacy, or the record would be
    # answering and the legacy leg would never be reached.
    assert not record_path.exists()

    _stamp(SUCCESS_MARKER_VERSION)
    assert _marker_binds_overlay_and_table(
        output, "input", "plate", overlay, table
    ), (
        "a legacy marker carrying its own version was refused -- the legacy "
        "leg is being judged against the record's number, which is how "
        "overlay-authority repair silently stops working on legacy trees"
    )

    _stamp(RECORD_VERSION)
    assert not _marker_binds_overlay_and_table(
        output, "input", "plate", overlay, table
    ), (
        "a marker stamped with the RECORD version was accepted at the legacy "
        "path -- the version is not paired with the shape at all"
    )
