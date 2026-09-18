"""A pipeline the staged engine refuses ends the CLI with one clean error.

``pipeline_requires_gpu`` raises ``UnstageableGpuDetectorError`` for a
``GpuDetector`` it cannot stage (here: nested inside
``TwoKFilamentousDetector``). Before this file the CLI let that escape into the
generic ``except Exception`` branch, which prints ``Unexpected error`` plus a
full traceback -- and ``--dry-run`` (the GUI's Validate button) never reached
the check at all, so it reported a pipeline as fine that Run then refused.

Every case asserts three things, because each one alone passes on broken code:
the exit is non-zero, the message names the offending class, and no traceback
is printed. The output tree is also checked: the refusal must fire before the
CLI mutates anything under ``--output``, not after it has minted a run
identity or cleared the directory for ``--overwrite``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

import phenotypic
from phenotypic import ImagePipeline
from phenotypic.detect import TwoKFilamentousDetector
from phenotypic.phenotypicCLI import phenotypic_cli
from tests._fakes.fake_gpu_detector import FakeGpuDetector

#: The class the refusal must name -- the container the user has to lift the
#: detector out of.
OFFENDING_CLASS = "TwoKFilamentousDetector"


@pytest.fixture(autouse=True)
def _register_fake_gpu_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """from_json resolves classes by bare name in the phenotypic namespace."""
    monkeypatch.setattr(
        phenotypic, "FakeGpuDetector", FakeGpuDetector, raising=False
    )


@pytest.fixture
def refused_pipeline(tmp_path: Path) -> Path:
    path = tmp_path / "pipeline.json"
    pipeline = ImagePipeline(
        ops={
            OFFENDING_CLASS: TwoKFilamentousDetector(
                branch_base=FakeGpuDetector()
            )
        }
    )
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return path


@pytest.fixture
def image_tree(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    image = root / "plate1" / "img001.tiff"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"never decoded: the refusal fires first")
    return root


@pytest.fixture
def no_scheduler(tmp_path: Path) -> dict[str, str]:
    """An environment with no ``sbatch`` on PATH.

    The SLURM cases must not be able to submit anything if the refusal
    regresses to firing late, so the scheduler is made unreachable rather
    than trusted not to be called.
    """
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    return {"PATH": str(empty)}


def _assert_clean_refusal(result, output_dir: Path) -> None:
    assert result.exit_code != 0, result.output
    assert OFFENDING_CLASS in result.output, result.output
    assert "only composition primitives" in result.output, result.output
    assert "Traceback" not in result.output, result.output
    assert "Unexpected error" not in result.output, result.output
    assert not output_dir.exists(), sorted(output_dir.rglob("*"))


@pytest.mark.parametrize(
    "mode_args",
    [
        pytest.param([], id="full-local"),
        pytest.param(["--dry-run"], id="full-dry-run"),
        pytest.param(["--mode", "process", "--layer", "objmap"], id="process-objmap"),
        pytest.param(["--mode", "process", "--layer", "rgb"], id="process-rgb"),
    ],
)
def test_a_refused_pipeline_exits_cleanly_on_every_local_mode(
    mode_args: list[str],
    refused_pipeline: Path,
    image_tree: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            *mode_args,
            "--pipeline",
            str(refused_pipeline),
            "--input",
            str(image_tree),
            "--output",
            str(output_dir),
        ],
    )

    _assert_clean_refusal(result, output_dir)


@pytest.mark.parametrize(
    "mode_args",
    [
        pytest.param([], id="full-slurm"),
        pytest.param(["--mode", "process", "--layer", "gray"], id="process-slurm"),
    ],
)
def test_a_refused_pipeline_exits_cleanly_before_any_slurm_submission(
    mode_args: list[str],
    refused_pipeline: Path,
    image_tree: Path,
    tmp_path: Path,
    no_scheduler: dict[str, str],
) -> None:
    output_dir = tmp_path / "out"

    result = CliRunner(env=no_scheduler).invoke(
        phenotypic_cli,
        [
            *mode_args,
            "--pipeline",
            str(refused_pipeline),
            "--input",
            str(image_tree),
            "--output",
            str(output_dir),
            "--slurm",
            "slurm_partition=exfab",
        ],
    )

    _assert_clean_refusal(result, output_dir)


def test_the_refusal_fires_before_overwrite_clears_the_output(
    refused_pipeline: Path, image_tree: Path, tmp_path: Path
) -> None:
    """A late catch would already have deleted the user's previous run."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    previous = output_dir / "previous-run.txt"
    previous.write_text("keep me", encoding="utf-8")

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(refused_pipeline),
            "--input",
            str(image_tree),
            "--output",
            str(output_dir),
            "--overwrite",
        ],
    )

    assert result.exit_code != 0, result.output
    assert OFFENDING_CLASS in result.output, result.output
    assert "Traceback" not in result.output, result.output
    assert sorted(p.name for p in output_dir.iterdir()) == [previous.name]
    assert previous.read_text(encoding="utf-8") == "keep me"


def test_an_unreadable_pipeline_is_still_reported_by_validation(
    image_tree: Path, tmp_path: Path
) -> None:
    """The preflight only converts the refusal; it swallows nothing else.

    A corrupt pipeline must keep reaching the existing "Pipeline loading
    failed" validation report. If the preflight converted every ValueError
    into a usage error, this message would change; if it re-raised them, the
    user would get a traceback where they used to get a clean validation line.
    """
    corrupt = tmp_path / "pipeline.json"
    corrupt.write_text("{ this is not json", encoding="utf-8")

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(corrupt),
            "--input",
            str(image_tree),
            "--output",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code != 0, result.output
    assert "Pipeline loading failed" in result.output, result.output
    assert "Traceback" not in result.output, result.output
    assert OFFENDING_CLASS not in result.output
