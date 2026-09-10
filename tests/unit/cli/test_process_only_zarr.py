"""--mode process writes a single-series store carrying its own provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import numpy as np

from phenotypic import Image, ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.enhance import BlurGauss
from phenotypic.sdk_ import ngff_
from phenotypic.sdk_.ngff_ import PhenotypicAttr
from phenotypic._cli._cli_process_only import (
    process_only_output_path,
    process_single_apply_only_core,
    write_process_only_layer,
)


@pytest.fixture
def source_image(tmp_path: Path) -> Path:
    root = tmp_path / "in"
    root.mkdir()
    path = root / "IMG_4471.tiff"
    Image(load_synth_yeast_plate()).rgb.imsave(filepath=path)
    return path


@pytest.fixture
def pipeline_file(tmp_path: Path) -> Path:
    """Named `.pht-pipe` because `to_json` RENAMES anything else.

    `ImagePipeline().to_json(tmp / "preprocess.json")` writes
    `preprocess.json.pht-pipe` and returns None -- verified by execution -- so
    a fixture that returns the path it passed in returns a path that does not
    exist, and every test using it dies in `from_json`.

    It carries ONE real operation, and that is load-bearing rather than
    decorative: an empty pipeline records `operations == []` whichever side of
    `pipeline.apply()` the provenance init runs on, so every ordering assertion
    made against it holds vacuously. The field is `ops` --
    `ImagePipeline([op])` raises `TypeError: BaseModel.__init__() takes 1
    positional argument` and `operations=` raises `extra_forbidden`.
    """
    nested = tmp_path / "config" / "deep"
    nested.mkdir(parents=True)
    path = nested / "preprocess.json.pht-pipe"
    ImagePipeline(ops=[BlurGauss()]).to_json(path)
    assert path.is_file()
    return path


def _block(store: Path) -> dict:
    payload = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    return payload["attributes"][PhenotypicAttr.ROOT]


def test_output_path_is_a_store_for_zarr(tmp_path: Path) -> None:
    out = process_only_output_path(
        tmp_path / "out", tmp_path / "in" / "a" / "p01.tiff",
        tmp_path / "in", "rgb", fmt="zarr",
    )
    assert out == tmp_path / "out" / "a" / f"p01{ngff_.STORE_SUFFIX}"


def test_output_path_is_unchanged_for_tiff(tmp_path: Path) -> None:
    assert process_only_output_path(
        tmp_path / "out", tmp_path / "in" / "p01.tiff", tmp_path / "in", "rgb",
    ).name == "p01.tiff"
    assert process_only_output_path(
        tmp_path / "out", tmp_path / "in" / "p01.tiff", tmp_path / "in", "objmap",
    ).name == "p01.png"


def test_a_store_input_does_not_double_its_suffix(tmp_path: Path) -> None:
    """`Path("p01.ome.zarr").stem` is `"p01.ome"` -> `p01.ome.ome.zarr`.

    Spec 7.3. A tree of stores is valid input, so this is the ordinary case
    for the second run of the loop, not an exotic one -- and the wrong name is
    a plausible-looking one that nothing raises on.
    """
    store_in = tmp_path / "in" / f"p01{ngff_.STORE_SUFFIX}"
    assert process_only_output_path(
        tmp_path / "out", store_in, tmp_path / "in", "rgb", fmt="zarr",
    ).name == f"p01{ngff_.STORE_SUFFIX}"
    assert process_only_output_path(
        tmp_path / "out", store_in, tmp_path / "in", "detect_mat", fmt="tiff",
    ).name == "p01.tiff"


#: The zarr-capable layers, and the series each one must NOT emit. Both are
#: exercised: `--process-format zarr` is the default for `gray` too (spec
#: 5.2), and a gray store is structurally different -- two axes, no `omero`.
_OTHER_SERIES: dict[str, tuple[str, ...]] = {
    "rgb": ("gray", "detect_mat"),
    "gray": ("rgb", "detect_mat"),
}


@pytest.mark.parametrize("layer", ["rgb", "gray"])
def test_writer_emits_only_the_requested_series(
    tmp_path: Path, layer: str
) -> None:
    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, layer, out, fmt="zarr")
    assert (out / layer).is_dir()
    for other in _OTHER_SERIES[layer]:
        assert not (out / other).exists()
    assert not (out / layer / "labels").exists()
    assert _block(out)[PhenotypicAttr.SERIES] == {layer: layer}


@pytest.mark.parametrize("layer", ["rgb", "gray"])
def test_writer_omits_image_class(tmp_path: Path, layer: str) -> None:
    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, layer, out, fmt="zarr")
    assert PhenotypicAttr.IMAGE_CLASS not in _block(out)


def test_a_gray_store_is_two_axes_and_carries_no_omero(tmp_path: Path) -> None:
    """Spec 2.6 / 2.1. The two zarr layers are not the same artifact.

    `rgb` is `('c','y','x')` with an `omero` rendering block; `gray` is
    `('y','x')` with none, because NGFF's `omero` window is defined for
    integer display data and the 2026-08-18 ruling declines to invent one for
    a float series. A suite that only ever writes `rgb` pins neither.
    """
    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, "gray", out, fmt="zarr")
    ome = json.loads((out / "gray" / "zarr.json").read_text(encoding="utf-8"))
    ome = ome["attributes"]["ome"]
    assert [a["name"] for a in ome["multiscales"][0]["axes"]] == ["y", "x"]
    assert "omero" not in ome

    rgb_out = tmp_path / "p02.ome.zarr"
    write_process_only_layer(img, "rgb", rgb_out, fmt="zarr")
    rgb_ome = json.loads(
        (rgb_out / "rgb" / "zarr.json").read_text(encoding="utf-8")
    )["attributes"]["ome"]
    assert [a["name"] for a in rgb_ome["multiscales"][0]["axes"]] == [
        "c", "y", "x",
    ]
    assert "omero" in rgb_ome


@pytest.mark.parametrize(
    "layer, expected",
    [
        ("objmap", "labels group is not itself an image"),
        ("detect_mat", r"store writer requires a primary series"),
    ],
)
def test_the_writer_refuses_a_layer_with_no_store_form(
    tmp_path: Path, layer: str, expected: str
) -> None:
    """Spec 5.3: two refusals for two different reasons, said differently.

    Matching on the layer name alone would NOT pin the guard. Delete it and
    `_save_store` raises `no primary series among ['objmap']` -- which names
    the layer, so a `match=layer` test stays green with the guard gone, and
    for `detect_mat` even a `match="primary series"` test does. The
    distinguishing text is the *reason*: `objmap` is refused by an NGFF
    structural rule (a labels group is not an image), `detect_mat` by
    PhenoTypic's own primary-series requirement. Neither string is
    producible by `_save_store`.
    """
    img = Image(load_synth_yeast_plate())
    with pytest.raises(ValueError, match=expected):
        write_process_only_layer(img, layer, tmp_path / "x.ome.zarr", fmt="zarr")


@pytest.mark.parametrize("layer", ["rgb", "gray"])
def test_a_single_series_store_is_four_files_plus_two_per_level(
    tmp_path: Path, layer: str
) -> None:
    """Spec 1.1. Guards against an accidental extra series or level.

    The name says `4 + 2 * levels` rather than a bare number because that is
    what the body checks: the 600x800 synthetic plate yields 2 levels and so 8
    files, and hard-coding 12 would assert a size this fixture never has.

    The shorthand holds only while every pyramid level fits inside ONE shard
    -- true up to a 4096-pixel level-0 edge, and true for this plate. Above
    that a level contributes more than one shard file; the committed
    validation script (Task 11) carries the general form.
    """
    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, layer, out, fmt="zarr")
    levels = ngff_.pyramid_level_count(*img.shape[:2])
    files = [p for p in out.rglob("*") if p.is_file()]
    # root + OME/zarr.json + OME xml + series zarr.json + 2 per level
    assert len(files) == 4 + 2 * levels


def test_core_records_the_pipeline_basename_not_its_path(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """A published store must not carry cluster filesystem layout."""
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=source_image,
        input_root=source_image.parent,
        output_dir=out,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="zarr",
    )
    store = out / f"IMG_4471{ngff_.STORE_SUFFIX}"
    journal = _block(store)[PhenotypicAttr.PROVENANCE]
    application = journal["applications"][-1]
    assert application["pipeline"]["source_path"] == "preprocess.json.pht-pipe"
    assert "/" not in application["pipeline"]["source_path"]
    assert len(application["pipeline"]["sha256"]) == 64


def test_provenance_init_runs_before_apply_not_after(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """`initialize_cli_provenance` resets the journal (_provenance.py:294).

    Called after `pipeline.apply()` it would discard `operations[]` -- and the
    store would still have a `pipeline` key, so the store looks fine and the
    operations are simply gone. The fixture pipeline therefore carries a real
    operation: with an empty one, `operations == []` in BOTH orderings and
    swapping the two lines leaves this green.

    This is also the only coverage of spec 2.3's actual claim -- that the
    published store carries the operations that ran, with their RESOLVED
    parameters (`operation.model_dump(mode="json")`), not the configuration
    that was requested.
    """
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=source_image,
        input_root=source_image.parent,
        output_dir=out,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="zarr",
    )
    journal = _block(out / f"IMG_4471{ngff_.STORE_SUFFIX}")[
        PhenotypicAttr.PROVENANCE
    ]
    application = journal["applications"][-1]
    assert application["pipeline"] is not None
    assert [o["operation_name"] for o in application["operations"]] == ["BlurGauss"]
    only = application["operations"][0]
    assert only["operation_class"].endswith(".BlurGauss")
    assert only["parameters"] == BlurGauss().model_dump(mode="json")
    assert only["parameters"]["sigma"] == 2.0


def test_the_store_round_trips_through_imread(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """The loop closes: what process mode writes, imread reads."""
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=source_image,
        input_root=source_image.parent,
        output_dir=out,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="zarr",
    )
    store = out / f"IMG_4471{ngff_.STORE_SUFFIX}"
    assert Image.imread(store).name == "IMG_4471"


def test_tiff_output_is_unchanged(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """The default path, and the AutoConvertRaw contract, must not move."""
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=source_image,
        input_root=source_image.parent,
        output_dir=out,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="tiff",
    )
    assert (out / "IMG_4471.tiff").is_file()


def test_a_store_round_trips_store_in_to_store_out(
    tmp_path: Path, pipeline_file: Path
) -> None:
    """Spec 7. The loop closes on itself: store in, store out, same pixels.

    The second hop must retain the first process application and append a
    distinct second one. A process store is both a pixel artifact and a lineage
    carrier; treating it as plain pixels would silently overwrite the first
    pipeline application and break process-to-CLI provenance continuity.
    """
    first = tmp_path / "first"
    src_root = tmp_path / "in"
    src_root.mkdir()
    Image(load_synth_yeast_plate()).rgb.imsave(filepath=src_root / "p01.tiff")
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=src_root / "p01.tiff",
        input_root=src_root,
        output_dir=first,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="zarr",
    )
    store_in = first / f"p01{ngff_.STORE_SUFFIX}"
    assert store_in.is_dir()

    second = tmp_path / "second"
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=store_in,
        input_root=first,
        output_dir=second,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="zarr",
    )
    store_out = second / f"p01{ngff_.STORE_SUFFIX}"
    assert Image.imread(store_out).name == "p01"
    # Same pipeline applied to already-processed pixels, so hop 2 is not a
    # no-op; what must match is the store and a direct re-apply of the second
    # hop's pipeline to the first hop's output.
    expected = Image.imread(store_in)
    ImagePipeline.from_json(pipeline_file).apply(expected, inplace=True)
    assert np.array_equal(Image.imread(store_out).rgb[:], expected.rgb[:])

    journal = _block(store_out)[PhenotypicAttr.PROVENANCE]
    applications = journal["applications"]
    assert journal["original_filename"] == "p01.tiff"
    assert [application["kind"] for application in applications] == [
        "process",
        "process",
    ]
    assert [application["input_filename"] for application in applications] == [
        "p01.tiff",
        "p01.ome.zarr",
    ]
    assert [
        operation["sequence"]
        for application in applications
        for operation in application["operations"]
    ] == [1, 2]
    assert [
        operation["operation_name"]
        for application in applications
        for operation in application["operations"]
    ] == ["BlurGauss", "BlurGauss"]


# ---------------------------------------------------------------------------
# Spec 2.3.3 -- a published store is bit-reproducible
# ---------------------------------------------------------------------------

#: The two journal fields a published store omits. Both are wall-clock
#: readings taken at apply time (`_provenance.py:377,380`), and measurement
#: across two runs of one image through one pipeline found them to be the
#: ENTIRE source of non-reproducibility in a store -- everything else in the
#: block is a pure function of the inputs.
_NON_REPRODUCIBLE_FIELDS = ("applied_at_utc", "duration_seconds")


def _tree_bytes(store: Path) -> dict[str, bytes]:
    """Every member of *store*, keyed by store-relative path.

    Comparing the root `zarr.json` alone is not enough and the suite has
    already been burned by it once (spec 7.3): two stores holding entirely
    different pixels shared one root. Byte-identity is a claim about the whole
    published artifact, so the check reads the whole published artifact.
    """
    return {
        p.relative_to(store).as_posix(): p.read_bytes()
        for p in sorted(store.rglob("*"))
        if p.is_file()
    }


def _run_to_store(
    pipeline_file: Path, source_image: Path, out: Path
) -> Path:
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=source_image,
        input_root=source_image.parent,
        output_dir=out,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="zarr",
    )
    return out / f"{source_image.stem}{ngff_.STORE_SUFFIX}"


def test_two_runs_of_one_image_produce_byte_identical_stores(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """Spec 2.3.3. The headline property, and it is a property of the bytes.

    Content-addressed storage, server-side dedup, and "did these two runs
    agree?" all reduce to this one comparison. It is asserted over every
    member rather than the root because the root is not the artifact.
    """
    first = _run_to_store(pipeline_file, source_image, tmp_path / "a")
    second = _run_to_store(pipeline_file, source_image, tmp_path / "b")

    left, right = _tree_bytes(first), _tree_bytes(second)
    assert sorted(left) == sorted(right)
    assert [name for name in left if left[name] != right[name]] == []


def test_file_sha256_of_two_identical_runs_matches(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """Spec 7.3's whole-tree digest is stable across an identical regeneration.

    This is the practical reason 2.3.3 exists. `file_sha256` feeds
    `work_id_for_image` and the SLURM identity ledger, so a digest that moved
    on every regeneration would give one image a new work ID per run and
    defeat continuation. Keeping the digest a dumb complete hash -- no
    exclusion list, no JSON round-trip, no coupling to the metadata schema --
    is bought by the writer dropping the two wall-clock fields, not by the
    walk knowing about them.
    """
    from phenotypic._cli._cli_failure_tracker import file_sha256

    first = _run_to_store(pipeline_file, source_image, tmp_path / "a")
    second = _run_to_store(pipeline_file, source_image, tmp_path / "b")
    assert file_sha256(first) == file_sha256(second)


def test_a_published_store_omits_the_wall_clock_fields(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """Absent from every entry, not merely from the first."""
    store = _run_to_store(pipeline_file, source_image, tmp_path / "out")
    journal = _block(store)[PhenotypicAttr.PROVENANCE]
    operations = [
        operation
        for application in journal["applications"]
        for operation in application["operations"]
    ]
    assert operations, "fixture pipeline must record at least one operation"
    for entry in operations:
        for field in _NON_REPRODUCIBLE_FIELDS:
            assert field not in entry


def test_save2zarr_keeps_the_wall_clock_fields(
    tmp_path: Path, pipeline_file: Path
) -> None:
    """Only the PUBLISHED artifact drops them (spec 2.3.3).

    The bundle store never leaves the run directory, so it keeps the
    human-facing telemetry -- when the image was processed, and how long each
    operation took. Without this half the paired assertion, a strip applied
    unconditionally in `_build_store_attributes` would look correct: the
    process store would be reproducible and nothing would say the bundle had
    silently lost its timestamps.
    """
    from phenotypic._core._provenance import (
        continuing_provenance_application,
        initialize_cli_provenance,
    )

    image = Image(load_synth_yeast_plate())
    initialize_cli_provenance(image, pipeline_file)
    with continuing_provenance_application(image):
        ImagePipeline.from_json(pipeline_file).apply(image, inplace=True)

    store = image.save2zarr(tmp_path / "bundle.ome.zarr")
    journal = _block(Path(store))[PhenotypicAttr.PROVENANCE]
    operations = journal["applications"][-1]["operations"]
    assert [o["operation_name"] for o in operations] == ["BlurGauss"]
    for field in _NON_REPRODUCIBLE_FIELDS:
        assert field in operations[0]


def test_stripping_leaves_the_rest_of_the_journal_intact(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """Two keys, from `operations[]` only -- not a general journal scrub.

    The journal is the store's provenance. A strip that took the surrounding
    keys with it, or that reached into `parameters`, would make the artifact
    reproducible by making it say nothing.
    """
    store = _run_to_store(pipeline_file, source_image, tmp_path / "out")
    journal = _block(store)[PhenotypicAttr.PROVENANCE]
    for key in ("schema_version", "status", "original_filename", "applications"):
        assert key in journal
    application = journal["applications"][-1]
    for key in ("pipeline", "retry_base_length", "operations"):
        assert key in application

    only = application["operations"][0]
    assert only["operation_name"] == "BlurGauss"
    assert only["parameters"] == BlurGauss().model_dump(mode="json")
    for key in (
        "sequence",
        "operation_class",
        "phenotypic_version",
        "pipeline_step_path",
    ):
        assert key in only


# ---------------------------------------------------------------------------
# Spec 2.3.4 -- a published store reports a terminal status
# ---------------------------------------------------------------------------


def test_a_published_store_reports_a_terminal_status(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """Spec 2.3.4. `in_progress` in a published artifact is a lie about it.

    `_cli_staged_resume.py` gates on `status in {"staged", "complete"}`, so a
    consumer following that same convention would reject every store this mode
    publishes.
    """
    store = _run_to_store(pipeline_file, source_image, tmp_path / "out")
    assert _block(store)[PhenotypicAttr.PROVENANCE]["status"] == "complete"


def test_a_failed_apply_leaves_the_journal_marked_failed(
    tmp_path: Path,
    source_image: Path,
    pipeline_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error path is the other half of the status contract.

    No store is written when apply raises, so the mark is observed on the
    image the core decoded -- captured through `imread` -- rather than on
    disk.
    """
    from phenotypic._cli._cli_failure_tracker import PerImageScientificError

    decoded: list[Image] = []
    real_imread = Image.imread

    def _spy_imread(path, **kwargs):
        image = real_imread(path, **kwargs)
        decoded.append(image)
        return image

    def _explode(self, image, inplace=False):
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(Image, "imread", staticmethod(_spy_imread))
    monkeypatch.setattr(ImagePipeline, "apply", _explode)

    with pytest.raises(PerImageScientificError):
        process_single_apply_only_core(
            pipeline_path=pipeline_file,
            image_path=source_image,
            input_root=source_image.parent,
            output_dir=tmp_path / "out",
            image_type="Image",
            layer="rgb",
            read_kwargs={},
            process_format="zarr",
        )

    assert decoded, "the core must have decoded an image before applying"
    assert decoded[-1]._metadata.provenance_journal["status"] == "failed"


def test_provenance_initialization_failure_is_not_masked_by_cleanup(
    tmp_path: Path,
    source_image: Path,
    pipeline_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A decoded image does not imply that a CLI application was opened."""
    from phenotypic._cli import _cli_process_only
    from phenotypic._cli._cli_failure_tracker import PerImageScientificError

    original = RuntimeError("provenance initialization failed")

    def _fail_initialization(*args, **kwargs):
        del args, kwargs
        raise original

    monkeypatch.setattr(
        _cli_process_only, "initialize_cli_provenance", _fail_initialization
    )

    with pytest.raises(PerImageScientificError) as caught:
        process_single_apply_only_core(
            pipeline_path=pipeline_file,
            image_path=source_image,
            input_root=source_image.parent,
            output_dir=tmp_path / "out",
            image_type="Image",
            layer="rgb",
            read_kwargs={},
            process_format="zarr",
        )

    assert caught.value.stage == "process"
    assert caught.value.cause is original
    assert caught.value.__cause__ is original


def test_failed_status_cleanup_does_not_mask_apply_failure(
    tmp_path: Path,
    source_image: Path,
    pipeline_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure-status publication is secondary to the scientific exception."""
    from phenotypic._cli import _cli_process_only
    from phenotypic._cli._cli_failure_tracker import PerImageScientificError

    original = RuntimeError("detector exploded")

    def _fail_apply(self, image, inplace=False):
        del self, image, inplace
        raise original

    def _fail_cleanup(image, status):
        del image
        assert status == "failed"
        raise ValueError("failure-status cleanup failed")

    monkeypatch.setattr(ImagePipeline, "apply", _fail_apply)
    monkeypatch.setattr(_cli_process_only, "set_provenance_status", _fail_cleanup)

    with pytest.raises(PerImageScientificError) as caught:
        process_single_apply_only_core(
            pipeline_path=pipeline_file,
            image_path=source_image,
            input_root=source_image.parent,
            output_dir=tmp_path / "out",
            image_type="Image",
            layer="rgb",
            read_kwargs={},
            process_format="zarr",
        )

    assert caught.value.stage == "process"
    assert caught.value.cause is original
    assert caught.value.__cause__ is original
