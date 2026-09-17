"""Per-image measurement authority as ``--mode recompile`` reads it.

Recompile no longer rewrites embedded tables (user ruling, 2026-09-11), so the
rewrite tests that lived here -- the whole-tree pre-flight scan, the per-store
inverted-store guard and its call-site proof -- went with the producer they
drove. What remains is the authority *reader*: the two-shape
record/legacy-marker pairing that decides which payload speaks for an image,
and the version travelling with each shape.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from phenotypic.schema import IMAGE
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    image_record_path,
    zarr_store_path,
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
