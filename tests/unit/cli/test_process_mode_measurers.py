"""Process mode runs what fits in a store and skips the rest (#281 P5 T1).

Spec E4 (user decision 2026-11-18): ``MODE_SLOTS["process"]`` is ``ops`` +
``meas``. A ``meas`` entry whose ``requires_objects`` is false (a grid
embedder) runs and stores its table; an entry that needs objects is skipped,
as process mode always skipped ``meas``, with one warning naming it, and the
preflight walk leaves it out. Only a runnable entry with a flat output is
refused, because its table would be lost: the CLI refuses in its read-only
half (usage error, exit 2, nothing written); the per-image core refuses on
its own, with a ``ValueError``, because AutoConvertRaw-GC calls it directly.

Torch: every refusal and skip here happens before a backbone loads, so it
runs without torch. ``test_process_accepts_grid_embedding`` and
``test_a_size_measurer_beside_a_dino_grid_embedder_stores_only_the_grid``
run the DINO stub and skip without torch; their toy twins pin the same
behaviour in every environment.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest

from phenotypic._cli._cli_preflight import MODE_SLOTS, operations_run_in_mode
from phenotypic._cli._cli_process_meas import (
    PROCESS_STORES_NO_OBJECTS,
    ProcessMeasRefusedError,
    object_measurers_in_process_mode,
    process_meas_refusal,
    skipped_measurers_warning,
)
from phenotypic._cli._cli_process_only import process_single_apply_only_core
from phenotypic.measure import MeasureDinoEmbedding, MeasureShape, MeasureSize
from tests._tiny_run import (
    TOY_GRID_COL_EDGES,
    TOY_GRID_ROW_EDGES,
    grid_embedder,
    run_cli,
    toy_embedder_class,
    toy_grid_process_pipeline,
    with_meas,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="Requires torch (stub backbone)"
)

ZARR_RGB = ["--layer", "rgb", "--process-format", "zarr"]
TIFF_RGB = ["--layer", "rgb", "--process-format", "tiff"]


def _error_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("Error:")]


def _warning_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if PROCESS_STORES_NO_OBJECTS in line]


def _toy_grid():
    from phenotypic.grid import ManualGridFinder

    return toy_embedder_class()(
        regions="grid",
        grid=ManualGridFinder(
            row_edges=list(TOY_GRID_ROW_EDGES), col_edges=list(TOY_GRID_COL_EDGES)
        ),
    )


# ---------------------------------------------------------------------------
# The shared check
# ---------------------------------------------------------------------------


def test_every_plain_measurer_requires_objects():
    assert MeasureSize().requires_objects is True
    assert MeasureDinoEmbedding().requires_objects is True
    assert grid_embedder().requires_objects is False


def test_object_measurers_are_named_by_slot_path():
    from phenotypic import ImagePipeline

    pipeline = ImagePipeline(
        meas={"size": MeasureSize(), "grid": grid_embedder(), "shape": MeasureShape()}
    )
    assert object_measurers_in_process_mode(pipeline) == ["meas:size", "meas:shape"]


def test_a_grid_embedder_into_a_store_is_admitted():
    from phenotypic import ImagePipeline

    pipeline = ImagePipeline(meas={"grid": grid_embedder()})
    assert process_meas_refusal(pipeline, layer="rgb", process_format="zarr") is None
    assert process_meas_refusal(pipeline, layer="gray", process_format="zarr") is None


def test_an_object_measurer_is_never_refused():
    """Skipped, not refused, into a store or a flat file alike."""
    from phenotypic import ImagePipeline

    pipeline = ImagePipeline(meas={"size": MeasureSize(), "emb": MeasureDinoEmbedding()})
    for layer, fmt in [("rgb", "zarr"), ("rgb", "tiff"), ("objmap", "tiff"), ("detect_mat", "tiff")]:
        assert process_meas_refusal(pipeline, layer=layer, process_format=fmt) is None


def test_an_empty_meas_is_admitted_for_every_output():
    from phenotypic import ImagePipeline

    for layer, fmt in [("rgb", "tiff"), ("objmap", "tiff"), ("detect_mat", "tiff")]:
        assert process_meas_refusal(ImagePipeline(), layer=layer, process_format=fmt) is None


def test_the_flat_output_refusal_names_only_the_runnable_entries():
    from phenotypic import ImagePipeline

    pipeline = ImagePipeline(meas={"size": MeasureSize(), "grid": grid_embedder()})
    refusal = process_meas_refusal(pipeline, layer="rgb", process_format="tiff")
    assert refusal is not None
    assert "meas:grid" in refusal and "meas:size" not in refusal and "lost" in refusal


def test_the_skip_warning_names_every_skipped_slot_in_one_line():
    from phenotypic import ImagePipeline

    pipeline = ImagePipeline(
        meas={"size": MeasureSize(), "grid": grid_embedder(), "shape": MeasureShape()}
    )
    assert skipped_measurers_warning(pipeline) == (
        f"meas:size, meas:shape skipped: {PROCESS_STORES_NO_OBJECTS}"
    )
    assert skipped_measurers_warning(ImagePipeline(meas={"grid": grid_embedder()})) is None


def test_preflight_now_walks_meas_in_process_mode():
    assert MODE_SLOTS["process"] == frozenset({"ops", "meas"})


def test_the_process_walk_leaves_out_the_skipped_measurers():
    """The preflight and reference planning walk exactly what process mode runs."""
    from phenotypic import ImagePipeline
    from phenotypic.enhance import BlurGauss

    pipeline = ImagePipeline(
        ops=[BlurGauss(sigma=1.0)],
        meas={"size": MeasureSize(), "grid": grid_embedder()},
    )
    roots = {path[0] for path, _ in operations_run_in_mode(pipeline, "process")}
    assert "meas:grid" in roots and "meas:size" not in roots
    # The grid embedder's own ManualGridFinder runs with it.
    assert ("meas:grid", "grid") in {path for path, _ in operations_run_in_mode(pipeline, "process")}
    full_roots = {path[0] for path, _ in operations_run_in_mode(pipeline, "full")}
    assert {"meas:size", "meas:grid"} <= full_roots


# ---------------------------------------------------------------------------
# The CLI: skip with a warning; refuse a lost table in the read-only half
# ---------------------------------------------------------------------------


def test_process_skips_a_size_measurer_with_a_warning(tmp_path, tiny_inputs):
    out = tmp_path / "out"
    result = run_cli(
        tiny_inputs, out, mode="process", pipeline=with_meas(tmp_path, size=MeasureSize()),
        extra=ZARR_RGB,
    )
    assert result.exit_code == 0, result.output
    # The CLI's one line; the core may also log per image.
    assert [line for line in _warning_lines(result.output) if line.startswith("Warning:")] == [
        f"Warning: meas:size skipped: {PROCESS_STORES_NO_OBJECTS}."
    ]
    stores = sorted(out.rglob("*.ome.zarr"))
    assert len(stores) == 2
    for store in stores:
        assert not (store / "tables").exists(), "a skipped measurer stores no table"


def test_process_skips_a_crop_embedder(tmp_path, tiny_inputs, stub_backbone):
    """A crop embedder needs objects: skipped, never loaded, no torch needed."""
    out = tmp_path / "out"
    result = run_cli(
        tiny_inputs, out, mode="process",
        pipeline=with_meas(tmp_path, emb=MeasureDinoEmbedding(device="cpu")),
        extra=ZARR_RGB,
    )
    assert result.exit_code == 0, result.output
    assert "meas:emb skipped" in result.output
    assert stub_backbone.load_calls == []
    assert not any((store / "tables").exists() for store in out.rglob("*.ome.zarr"))


def test_a_size_measurer_with_a_flat_output_still_runs(tmp_path, tiny_inputs):
    """No table is lost, so no refusal: the export is what it always was."""
    out = tmp_path / "out"
    result = run_cli(
        tiny_inputs, out, mode="process", pipeline=with_meas(tmp_path, size=MeasureSize()),
        extra=TIFF_RGB,
    )
    assert result.exit_code == 0, result.output
    assert len(list(out.rglob("*.tiff"))) == 2


@pytest.mark.parametrize(
    "extra",
    [
        TIFF_RGB,
        ["--layer", "gray", "--process-format", "tiff"],
        ["--layer", "detect_mat"],
        ["--layer", "objmap"],
    ],
)
def test_process_refuses_meas_without_a_store_output(
    tmp_path, tiny_inputs, stub_backbone, extra
):
    result = run_cli(
        tiny_inputs,
        tmp_path / "out",
        mode="process",
        pipeline=with_meas(tmp_path, emb=grid_embedder()),
        extra=extra,
    )
    assert result.exit_code == 2, result.output
    errors = _error_lines(result.output)
    assert len(errors) == 1 and "store" in errors[0] and "meas:emb" in errors[0]
    assert stub_backbone.load_calls == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("flag", ["--overwrite", "--restart", "--dry-run"])
def test_the_refusal_sits_above_every_mutation(tmp_path, tiny_inputs, flag):
    """A previous run survives the refused command, whatever flag it carries."""
    out = tmp_path / "out"
    (out / ".phenotypic").mkdir(parents=True)
    (out / "previous-run.txt").write_text("keep me", encoding="utf-8")
    (out / ".phenotypic" / "state.txt").write_text("keep me", encoding="utf-8")

    result = run_cli(
        tiny_inputs,
        out,
        mode="process",
        pipeline=with_meas(tmp_path, emb=grid_embedder()),
        extra=[*TIFF_RGB, flag],
    )
    assert result.exit_code == 2, result.output
    assert len(_error_lines(result.output)) == 1 and "lost" in result.output
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == [
        ".phenotypic",
        ".phenotypic/state.txt",
        "previous-run.txt",
    ]


def test_the_refusal_survives_skip_validation(tmp_path, tiny_inputs):
    result = run_cli(
        tiny_inputs,
        tmp_path / "out",
        mode="process",
        pipeline=with_meas(tmp_path, emb=grid_embedder()),
        extra=[*TIFF_RGB, "--skip-validation"],
    )
    assert result.exit_code == 2, result.output
    assert "lost" in result.output


# ---------------------------------------------------------------------------
# The core refuses on its own
# ---------------------------------------------------------------------------


def _first_tif(root: Path) -> Path:
    return next(root.rglob("*.tif"))


def test_the_core_refuses_what_the_cli_refuses(tmp_path, tiny_inputs, stub_backbone):
    """A runnable entry into a flat file, called directly (AutoConvertRaw-GC)."""
    with pytest.raises(ValueError, match="lost") as caught:
        process_single_apply_only_core(
            with_meas(tmp_path, emb=grid_embedder()),
            _first_tif(tiny_inputs),
            tiny_inputs,
            tmp_path / "out",
            "Image",
            "rgb",
            {},
            process_format="tiff",
        )
    assert isinstance(caught.value, ProcessMeasRefusedError)
    assert stub_backbone.load_calls == []
    assert not (tmp_path / "out").exists()


def test_the_core_refuses_before_reading_the_image(tmp_path):
    """An image that does not exist is never opened: the pipeline is refused first."""
    with pytest.raises(ProcessMeasRefusedError):
        process_single_apply_only_core(
            with_meas(tmp_path, emb=grid_embedder()),
            tmp_path / "missing.tif",
            tmp_path,
            tmp_path / "out",
            "Image",
            "objmap",
            {},
            process_format="tiff",
        )


def test_the_core_skips_a_size_measurer_with_a_logged_warning(tmp_path, tiny_inputs, caplog):
    out = tmp_path / "out"
    with caplog.at_level(logging.WARNING, logger="phenotypic._cli._cli_process_only"):
        assert process_single_apply_only_core(
            with_meas(tmp_path, size=MeasureSize()),
            _first_tif(tiny_inputs),
            tiny_inputs,
            out,
            "Image",
            "rgb",
            {},
            process_format="zarr",
        )
    assert f"meas:size skipped: {PROCESS_STORES_NO_OBJECTS}" in caplog.text
    (store,) = out.rglob("*.ome.zarr")
    assert not (store / "tables").exists()


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def test_process_accepts_a_toy_grid_embedding(tmp_path, tiny_inputs, registered_toy_embedder):
    result = run_cli(
        tiny_inputs,
        tmp_path / "out",
        mode="process",
        pipeline=toy_grid_process_pipeline(tmp_path),
        extra=ZARR_RGB,
    )
    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "out").rglob("*.ome.zarr"))) == 2


def test_a_size_measurer_beside_a_toy_grid_embedder_stores_only_the_grid(
    tmp_path, tiny_inputs, registered_toy_embedder
):
    from phenotypic.sdk_ import list_table_runs

    out = tmp_path / "out"
    result = run_cli(
        tiny_inputs, out, mode="process",
        pipeline=with_meas(tmp_path, size=MeasureSize(), grid=_toy_grid()),
        extra=ZARR_RGB,
    )
    assert result.exit_code == 0, result.output
    assert "meas:size skipped" in result.output and "meas:grid" not in "".join(
        _warning_lines(result.output)
    )
    for store in out.rglob("*.ome.zarr"):
        (run,) = list_table_runs(store)
        assert not run.has_measurements
        assert set(run.descriptor["embeddings"]) == {"toy-v1/grid"}


@requires_torch
def test_a_size_measurer_beside_a_dino_grid_embedder_stores_only_the_grid(
    tmp_path, tiny_inputs, stub_backbone
):
    from phenotypic.sdk_ import list_table_runs

    out = tmp_path / "out"
    result = run_cli(
        tiny_inputs, out, mode="process",
        pipeline=with_meas(
            tmp_path, size=MeasureSize(), crops=MeasureDinoEmbedding(device="cpu"),
            grid=grid_embedder(),
        ),
        extra=ZARR_RGB,
    )
    assert result.exit_code == 0, result.output
    assert "meas:size, meas:crops skipped" in result.output
    for store in out.rglob("*.ome.zarr"):
        (run,) = list_table_runs(store)
        assert not run.has_measurements
        assert set(run.descriptor["embeddings"]) == {"dinov2-vitb14/grid"}


@requires_torch
def test_process_accepts_grid_embedding(tmp_path, tiny_inputs, stub_backbone):
    result = run_cli(
        tiny_inputs,
        tmp_path / "out",
        mode="process",
        pipeline=with_meas(tmp_path, emb=grid_embedder()),
        extra=ZARR_RGB,
    )
    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "out").rglob("*.ome.zarr"))) == 2


# ---------------------------------------------------------------------------
# v1 runs embedders only: a no-objects measurer that is not one is named
# ---------------------------------------------------------------------------

_NO_OBJECTS_MEASURER: type | None = None


def _no_objects_measurer_class() -> type:
    """A bare ``MeasureFeatures`` that needs no objects: not an embedder.

    Built lazily and once, so the subclass exists only in a process that runs
    these tests.
    """
    global _NO_OBJECTS_MEASURER
    if _NO_OBJECTS_MEASURER is None:
        import pandas as pd

        from phenotypic.abc_ import MeasureFeatures

        class NoObjectsMeasurer(MeasureFeatures):
            """A test double: claims to need no objects, emits one column."""

            @property
            def requires_objects(self) -> bool:
                return False

            def _operate(self, image):
                return pd.DataFrame({"NoObjects_Value": [1.0]})

        _NO_OBJECTS_MEASURER = NoObjectsMeasurer
    return _NO_OBJECTS_MEASURER


@pytest.fixture
def registered_no_objects_measurer(monkeypatch):
    """Make ``NoObjectsMeasurer`` resolvable by ``ImagePipeline.from_json``."""
    import phenotypic

    cls = _no_objects_measurer_class()
    monkeypatch.setattr(phenotypic, cls.__name__, cls, raising=False)
    return cls


def test_a_no_objects_measurer_that_is_no_embedder_is_skipped_by_name():
    from phenotypic import ImagePipeline

    from phenotypic._cli._cli_process_meas import (
        PROCESS_STORES_ONLY_EMBEDDINGS,
        runs_in_process_mode,
    )

    foo = _no_objects_measurer_class()()
    assert foo.requires_objects is False and not runs_in_process_mode(foo)
    pipeline = ImagePipeline(
        meas={"size": MeasureSize(), "foo": foo, "grid": grid_embedder()}
    )
    assert skipped_measurers_warning(pipeline) == (
        f"meas:size skipped: {PROCESS_STORES_NO_OBJECTS}; "
        f"meas:foo skipped: {PROCESS_STORES_ONLY_EMBEDDINGS}"
    )
    # It does not run, so a flat output loses nothing and is not refused.
    only_foo = ImagePipeline(meas={"foo": foo})
    assert process_meas_refusal(only_foo, layer="rgb", process_format="tiff") is None
    # The walk is what runs: it is left out of the process scope.
    roots = {path[0] for path, _ in operations_run_in_mode(pipeline, "process")}
    assert "meas:foo" not in roots and "meas:grid" in roots


def test_process_skips_a_no_objects_measurer_with_the_embeddings_wording(
    tmp_path, tiny_inputs, registered_no_objects_measurer
):
    from phenotypic._cli._cli_process_meas import PROCESS_STORES_ONLY_EMBEDDINGS

    out = tmp_path / "out"
    result = run_cli(
        tiny_inputs, out, mode="process",
        pipeline=with_meas(tmp_path, foo=registered_no_objects_measurer()),
        extra=ZARR_RGB,
    )
    assert result.exit_code == 0, result.output
    warnings = [
        line for line in result.output.splitlines()
        if line.startswith("Warning:") and "skipped" in line
    ]
    assert warnings == [f"Warning: meas:foo skipped: {PROCESS_STORES_ONLY_EMBEDDINGS}."]
    stores = sorted(out.rglob("*.ome.zarr"))
    assert len(stores) == 2
    for store in stores:
        assert not (store / "tables").exists(), "a skipped measurer stores no table"


# ---------------------------------------------------------------------------
# #281 named grid regions
# ---------------------------------------------------------------------------


def test_named_grid_embedders_run_in_process_mode():
    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_process_meas import process_mode_measurers, runs_in_process_mode
    from tests._tiny_run import named_grid_embedders, toy_grid_embedder_of

    for make in (toy_grid_embedder_of, grid_embedder):
        grids = named_grid_embedders(make)
        assert all(runs_in_process_mode(embedder) for embedder in grids.values())
        pipeline = ImagePipeline(meas={"size": MeasureSize(), **grids})
        assert list(process_mode_measurers(pipeline)) == ["wells", "agar"]
        assert process_meas_refusal(pipeline, layer="rgb", process_format="zarr") is None
