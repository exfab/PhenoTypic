"""Nothing under ``--output`` changes before validation and the dry-run exit.

Spec: ``docs/superpowers/specs/2026-09-24-cli-preflight/design.md`` §1
(findings F1, F2, F3, F27). Before this change ``phenotypic_cli`` cleared
``.phenotypic/`` for ``--restart`` and ran ``shutil.rmtree`` for
``--overwrite`` *above* pipeline validation and the ``--dry-run`` exit, so:

- a corrupt pipeline with ``--overwrite`` deleted the previous run and only
  then reported "Pipeline loading failed";
- ``--overwrite --dry-run`` deleted the output and exited 0;
- ``--restart --dry-run`` cleared machine state and bumped the restart epoch;
- ``--overwrite`` deleted any run input stored under ``--output``.

Every case asserts the exit status, the message and the on-disk state,
because each assertion alone passes on broken code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize
from phenotypic.phenotypicCLI import phenotypic_cli

PREVIOUS = "previous-run.txt"


def _plate_image() -> np.ndarray:
    """A small RGB plate: dark agar with two bright colonies."""
    image = np.full((32, 32, 3), 20, dtype=np.uint8)
    image[6:12, 6:12] = 220
    image[20:27, 18:25] = 200
    return image


def _write_tree(root: Path) -> Path:
    image = root / "plate1" / "img001.tiff"
    image.parent.mkdir(parents=True)
    tifffile.imwrite(image, _plate_image())
    return root


@pytest.fixture
def image_tree(tmp_path: Path) -> Path:
    """One dataset holding one decodable TIFF (review R12: never placeholder bytes)."""
    return _write_tree(tmp_path / "images")


@pytest.fixture
def valid_pipeline(tmp_path: Path) -> Path:
    path = tmp_path / "pipeline.json"
    pipeline = ImagePipeline(
        ops={"det": OtsuDetector()}, meas={"size": MeasureSize()}
    )
    # write_text, not to_json(path): the latter appends ``.pht-pipe``.
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return path


@pytest.fixture
def previous_run(tmp_path: Path) -> Path:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / PREVIOUS).write_text("keep me", encoding="utf-8")
    return output_dir


def _invoke(*args: str):
    return CliRunner().invoke(phenotypic_cli, [*args])


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_overwrite_with_a_corrupt_pipeline_keeps_the_previous_run(
    image_tree: Path, previous_run: Path, tmp_path: Path
) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{ not json", encoding="utf-8")

    result = _invoke(
        "--pipeline", str(corrupt),
        "--input", str(image_tree),
        "--output", str(previous_run),
        "--overwrite",
    )

    assert result.exit_code != 0, result.output
    assert "Pipeline loading failed" in result.output, result.output
    assert (previous_run / PREVIOUS).read_text(encoding="utf-8") == "keep me"


def test_overwrite_dry_run_previews_and_deletes_nothing(
    valid_pipeline: Path, image_tree: Path, previous_run: Path
) -> None:
    result = _invoke(
        "--pipeline", str(valid_pipeline),
        "--input", str(image_tree),
        "--output", str(previous_run),
        "--overwrite",
        "--dry-run",
    )

    assert result.exit_code == 0, result.output
    assert "would delete" in result.output, result.output
    assert sorted(p.name for p in previous_run.iterdir()) == [PREVIOUS]
    assert (previous_run / PREVIOUS).read_text(encoding="utf-8") == "keep me"


def test_restart_dry_run_leaves_machine_state_untouched(
    valid_pipeline: Path, image_tree: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "run"
    common = (
        "--pipeline", str(valid_pipeline),
        "--input", str(image_tree),
        "--output", str(output_dir),
        "--image-type", "Image",
        "--njobs", "1",
        "--no-qc",
    )
    first = _invoke(*common)
    assert first.exit_code == 0, first.output
    machine_state = output_dir / ".phenotypic"
    assert machine_state.is_dir(), first.output
    before = _tree_digest(machine_state)
    assert before, "the real run left no machine state to protect"

    result = _invoke(*common, "--restart", "--dry-run")

    assert result.exit_code == 0, result.output
    assert "would clear" in result.output, result.output
    assert _tree_digest(machine_state) == before


@pytest.mark.parametrize(
    "option", ["--input", "--pipeline", "--metadata", "--image-manifest"]
)
def test_overwrite_refuses_each_run_input_inside_the_output(
    option: str,
    valid_pipeline: Path,
    image_tree: Path,
    previous_run: Path,
) -> None:
    """F3 and F27: the delete would remove a file the run is about to read."""
    paths = {
        "--input": image_tree,
        "--pipeline": valid_pipeline,
        "--metadata": None,
        "--image-manifest": None,
    }
    inside = previous_run / "deliverables"
    inside.mkdir()
    if option == "--input":
        target = _write_tree(inside / "images")
    elif option == "--pipeline":
        target = inside / "pipeline.json"
        target.write_bytes(valid_pipeline.read_bytes())
    elif option == "--metadata":
        target = inside / "metadata.csv"
        target.write_text("ImageName,Strain\nimg001,WT\n", encoding="utf-8")
    else:
        target = inside / "approved.images"
        target.write_text("plate1/img001.tiff\n", encoding="utf-8")
    paths[option] = target

    args = [
        "--pipeline", str(paths["--pipeline"]),
        "--input", str(paths["--input"]),
        "--output", str(previous_run),
        "--overwrite",
    ]
    if paths["--metadata"] is not None:
        args += ["--metadata", str(paths["--metadata"])]
    if paths["--image-manifest"] is not None:
        args += ["--image-manifest", str(paths["--image-manifest"])]

    result = _invoke(*args)

    assert result.exit_code != 0, result.output
    assert option in result.output, result.output
    assert target.exists()
    assert (previous_run / PREVIOUS).read_text(encoding="utf-8") == "keep me"


def test_restart_refuses_a_run_input_inside_machine_state(
    valid_pipeline: Path, image_tree: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "run"
    hidden = output_dir / ".phenotypic" / "pipeline.json"
    hidden.parent.mkdir(parents=True)
    hidden.write_bytes(valid_pipeline.read_bytes())

    result = _invoke(
        "--pipeline", str(hidden),
        "--input", str(image_tree),
        "--output", str(output_dir),
        "--restart",
    )

    assert result.exit_code != 0, result.output
    assert "--pipeline" in result.output, result.output
    assert hidden.exists()


def test_overlap_refusal_survives_skip_validation(
    valid_pipeline: Path, previous_run: Path
) -> None:
    """The overlap refusal protects the run's own inputs, so it is not skippable."""
    inputs = _write_tree(previous_run / "images")

    result = _invoke(
        "--pipeline", str(valid_pipeline),
        "--input", str(inputs),
        "--output", str(previous_run),
        "--overwrite",
        "--skip-validation",
    )

    assert result.exit_code != 0, result.output
    assert "--input" in result.output, result.output
    assert (inputs / "plate1" / "img001.tiff").exists()


# --- phase-A review A1: symlinks stored under the deleted location --------------


@pytest.mark.parametrize("option", ["--pipeline", "--metadata"])
def test_overwrite_refuses_a_symlinked_input_stored_under_the_output(
    option: str, valid_pipeline: Path, image_tree: Path, previous_run: Path, tmp_path: Path
) -> None:
    """``rmtree`` unlinks a link without following it, so the lexical path decides."""
    outside = tmp_path / "outside"
    outside.mkdir()
    if option == "--pipeline":
        real = outside / "pipeline.json"
        real.write_bytes(valid_pipeline.read_bytes())
    else:
        real = outside / "metadata.csv"
        real.write_text("ImageName,Strain\nimg001,WT\n", encoding="utf-8")
    link = previous_run / f"linked{real.suffix}"
    link.symlink_to(real)
    args = ["--pipeline", str(valid_pipeline), "--input", str(image_tree),
            "--output", str(previous_run), "--overwrite"]
    if option == "--pipeline":
        args[1] = str(link)
    else:
        args += ["--metadata", str(link)]

    result = _invoke(*args)

    assert result.exit_code != 0, result.output
    assert option in result.output, result.output
    assert link.is_symlink()
    assert (previous_run / PREVIOUS).read_text(encoding="utf-8") == "keep me"


def test_restart_refuses_a_symlink_nested_in_a_cleared_state_directory(
    valid_pipeline: Path, image_tree: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "run"
    nested = output_dir / ".phenotypic" / "progress" / "pipeline.json"
    nested.parent.mkdir(parents=True)
    nested.symlink_to(valid_pipeline)

    result = _invoke(
        "--pipeline", str(nested), "--input", str(image_tree),
        "--output", str(output_dir), "--restart",
    )

    assert result.exit_code != 0, result.output
    assert "--pipeline" in result.output
    assert nested.is_symlink()


# --- phase-A review A3: negative controls ------------------------------------------


def test_restart_admits_an_input_in_a_preserved_state_entry(
    valid_pipeline: Path, image_tree: Path, tmp_path: Path
) -> None:
    """``legacy-v2/`` survives ``--restart``, so a pipeline there is safe."""
    output_dir = tmp_path / "run"
    kept = output_dir / ".phenotypic" / "legacy-v2" / "pipeline.json"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(valid_pipeline.read_bytes())

    result = _invoke(
        "--pipeline", str(kept), "--input", str(image_tree),
        "--output", str(output_dir), "--restart", "--dry-run",
    )

    assert result.exit_code == 0, result.output
    assert kept.exists()


def test_restart_without_machine_state_refuses_nothing(
    valid_pipeline: Path, image_tree: Path, tmp_path: Path
) -> None:
    result = _invoke(
        "--pipeline", str(valid_pipeline), "--input", str(image_tree),
        "--output", str(tmp_path / "fresh"), "--restart", "--dry-run",
    )

    assert result.exit_code == 0, result.output


def test_a_relative_spelling_inside_the_output_is_still_refused(
    valid_pipeline: Path, image_tree: Path, previous_run: Path, monkeypatch
) -> None:
    inside = previous_run / "pipeline.json"
    inside.write_bytes(valid_pipeline.read_bytes())
    monkeypatch.chdir(previous_run.parent)

    result = _invoke(
        "--pipeline", "out/pipeline.json", "--input", str(image_tree),
        "--output", "out", "--overwrite",
    )

    assert result.exit_code != 0, result.output
    assert "--pipeline" in result.output
    assert inside.exists()


def test_recompile_is_out_of_scope_for_the_refusal(
    valid_pipeline: Path, previous_run: Path, monkeypatch
) -> None:
    """``--mode recompile`` exits before any delete, so it is not refused (A2)."""
    from phenotypic import phenotypicCLI

    reached: list[bool] = []
    monkeypatch.setattr(
        phenotypicCLI, "_handle_recompile", lambda *a, **k: reached.append(True)
    )
    metadata = previous_run / "deliverables" / "my_meta.csv"
    metadata.parent.mkdir()
    metadata.write_text("ImageName,Strain\nimg001,WT\n", encoding="utf-8")

    result = _invoke(
        "--mode", "recompile", "--output", str(previous_run),
        "--metadata", str(metadata), "--overwrite",
    )

    assert "lies inside --output" not in result.output, result.output
    assert metadata.exists()


def test_process_mode_ignores_metadata_in_the_refusal(
    valid_pipeline: Path, image_tree: Path, previous_run: Path
) -> None:
    """``--mode process`` ignores ``--metadata``, so it is not a run input (A6)."""
    metadata = previous_run / "ignored.csv"
    metadata.write_text("ImageName\nimg001\n", encoding="utf-8")

    result = _invoke(
        "--mode", "process", "--layer", "gray",
        "--pipeline", str(valid_pipeline), "--input", str(image_tree),
        "--output", str(previous_run), "--metadata", str(metadata),
        "--overwrite", "--dry-run",
    )

    assert "lies inside --output" not in result.output, result.output
    assert metadata.exists()


@pytest.mark.parametrize("flag", ["--overwrite", "--restart"])
def test_a_process_table_it_would_lose_is_refused_before_the_mutating_half(
    image_tree: Path, previous_run: Path, tmp_path: Path, flag: str
) -> None:
    """#281 P5 T1: a grid embedder into a flat export is refused above every mutation.

    ``--restart`` clears ``.phenotypic/`` and ``--overwrite`` deletes the run;
    the refusal must leave both untouched. Torch-free: nothing loads a model.
    """
    from tests._tiny_run import grid_embedder, with_meas

    state = previous_run / ".phenotypic" / "processing_state.json"
    state.parent.mkdir()
    state.write_text("{}", encoding="utf-8")

    result = _invoke(
        "--mode", "process", "--layer", "rgb", "--process-format", "tiff",
        "--pipeline", str(with_meas(tmp_path, grid=grid_embedder())),
        "--input", str(image_tree), "--output", str(previous_run), flag,
    )

    assert result.exit_code == 2, result.output
    assert "meas:grid" in result.output and "would be lost" in result.output
    assert (previous_run / PREVIOUS).read_text(encoding="utf-8") == "keep me"
    assert state.read_text(encoding="utf-8") == "{}"


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        (("--overwrite",), "--overwrite is refused for an in-place"),
        (("--overwrite", "--dry-run"), "--overwrite is refused for an in-place"),
        (("--overwrite", "--skip-validation"), "--overwrite is refused for an in-place"),
        # --restart is admitted in place (user ruling 2026-11-18), so the
        # next refusal is what stops it: a raw tree has no store to write into.
        (("--restart",), "is not a PhenoTypic store"),
        (("--restart", "--dry-run"), "is not a PhenoTypic store"),
    ],
)
def test_an_in_place_refusal_comes_before_the_mutating_half(
    image_tree: Path, tmp_path: Path, extra: tuple[str, ...], needle: str
) -> None:
    """#281 P5 T3: in place (--output equal to --input), refusals precede every mutation.

    The refusals are un-skippable and sit above the --restart clear, the
    --overwrite rmtree and the --dry-run exit, so the input tree, its
    machine state included, is byte-identical afterwards. The --overwrite
    refusal also fires before the deletion-path refusal, which would name
    --input otherwise. Torch-free: nothing loads a model.
    """
    from tests._tiny_run import grid_embedder, with_meas

    state = image_tree / ".phenotypic" / "processing_state.json"
    state.parent.mkdir()
    state.write_text("{}", encoding="utf-8")
    before = _tree_digest(image_tree)

    result = _invoke(
        "--mode", "process", "--layer", "rgb", "--process-format", "zarr",
        "--pipeline", str(with_meas(tmp_path, grid=grid_embedder())),
        "--input", str(image_tree), "--output", str(image_tree), *extra,
    )

    assert result.exit_code == 2, result.output
    errors = [line for line in result.output.splitlines() if line.startswith("Error:")]
    assert len(errors) == 1 and needle in errors[0], result.output
    assert _tree_digest(image_tree) == before


def _dino_pipeline(tmp_path: Path, *, device: str) -> Path:
    """A pipeline whose ``meas`` holds a DINO embedder on *device*; nothing loads it."""
    from phenotypic.measure import MeasureDinoEmbedding

    path = tmp_path / f"dino-{device}.json"
    pipeline = ImagePipeline(
        ops={"det": OtsuDetector()},
        meas={"size": MeasureSize(), "emb": MeasureDinoEmbedding(device=device)},
    )
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return path


_REFUSAL_FLAGS = [
    ("--overwrite",), ("--restart",), ("--dry-run",), ("--skip-validation",),
    ("--overwrite", "--dry-run"),
]


@pytest.mark.parametrize("extra", _REFUSAL_FLAGS, ids=" ".join)
def test_r1_is_refused_before_the_mutating_half(
    image_tree: Path, previous_run: Path, tmp_path: Path, extra: tuple[str, ...]
) -> None:
    """#282 R1: ``--embed-stage gpu`` with ``--mode process``, above every mutation."""
    from tests._tiny_run import grid_embedder, with_meas

    state = previous_run / ".phenotypic" / "processing_state.json"
    state.parent.mkdir()
    state.write_text("{}", encoding="utf-8")
    before = _tree_digest(previous_run)

    result = _invoke(
        "--mode", "process", "--layer", "rgb", "--process-format", "zarr",
        "--pipeline", str(with_meas(tmp_path, grid=grid_embedder())),
        "--input", str(image_tree), "--output", str(previous_run),
        "--embed-stage", "gpu", *extra,
    )

    assert result.exit_code == 2, result.output
    errors = [line for line in result.output.splitlines() if line.startswith("Error:")]
    assert len(errors) == 1 and "does not apply to --mode process" in errors[0], result.output
    assert _tree_digest(previous_run) == before


@pytest.mark.parametrize("extra", _REFUSAL_FLAGS, ids=" ".join)
def test_r2_is_refused_before_the_mutating_half(
    image_tree: Path, previous_run: Path, tmp_path: Path, extra: tuple[str, ...]
) -> None:
    """#282 R2 (U9): ``--embed-stage gpu`` over a ``device="cpu"`` pin, above every mutation.

    Torch-free: the refusal comes before the CUDA check, which imports torch.
    """
    state = previous_run / ".phenotypic" / "processing_state.json"
    state.parent.mkdir()
    state.write_text("{}", encoding="utf-8")
    before = _tree_digest(previous_run)

    result = _invoke(
        "--pipeline", str(_dino_pipeline(tmp_path, device="cpu")),
        "--input", str(image_tree), "--output", str(previous_run),
        "--njobs", "1", "--embed-stage", "gpu", *extra,
    )

    assert result.exit_code == 2, result.output
    errors = [line for line in result.output.splitlines() if line.startswith("Error:")]
    assert len(errors) == 1 and "is pinned to device='cpu'" in errors[0], result.output
    assert _tree_digest(previous_run) == before


@pytest.mark.parametrize("recorded", ["inline", None], ids=["recorded", "pre-282-state"])
@pytest.mark.parametrize("extra", [(), ("--dry-run",), ("--skip-validation",)], ids=lambda e: " ".join(e) or "plain")
def test_r3_is_refused_before_the_mutating_half(
    valid_pipeline: Path, image_tree: Path, tmp_path: Path,
    recorded: str | None, extra: tuple[str, ...],
) -> None:
    """#282 R3 (U3): a continuation switched to ``gpu`` is refused, the run byte-unchanged.

    A real finished run records ``embed_stage: "inline"``; a state written before
    #282 has no key, which reads as ``"inline"``. Torch-free: the refusal comes
    before the CUDA check.
    """
    import json

    from phenotypic.sdk_ import processing_state_path

    output_dir = tmp_path / "run"
    first = _invoke(
        "--pipeline", str(valid_pipeline), "--input", str(image_tree),
        "--output", str(output_dir), "--njobs", "1",
    )
    assert first.exit_code == 0, first.output
    state_path = processing_state_path(output_dir)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["config"]["embed_stage"] == "inline"
    if recorded is None:
        del state["config"]["embed_stage"]
        state_path.write_text(json.dumps(state), encoding="utf-8")
    before = _tree_digest(output_dir)

    result = _invoke(
        "--pipeline", str(_dino_pipeline(tmp_path, device="auto")),
        "--input", str(image_tree), "--output", str(output_dir),
        "--njobs", "1", "--embed-stage", "gpu", *extra,
    )

    assert result.exit_code == 2, result.output
    errors = [line for line in result.output.splitlines() if line.startswith("Error:")]
    assert len(errors) == 1 and "started with --embed-stage inline" in errors[0], result.output
    assert _tree_digest(output_dir) == before
