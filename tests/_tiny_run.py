"""Shared test helpers for the #281 plan (DVC outputs, TSV, tables v2, embeddings).

Fixtures here are re-exported by ``tests/conftest.py`` with an explicit import
list, so every test under ``tests/`` can request them by name. ``phenotypic``
is imported inside each fixture rather than at module scope: the top-level
conftest must stay importable in dependency-light pytest runs (the
packaging-integrity CI job), where ``phenotypic`` may be absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from click.testing import Result

    from phenotypic import ImagePipeline

#: Modes whose invocation takes ``--pipeline`` (see :func:`run_cli`).
_PIPELINE_MODES = frozenset({"full", "measure", "process"})


def _build_tiny_pipeline() -> "ImagePipeline":
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize

    return ImagePipeline(ops=[OtsuDetector()], meas={"size": MeasureSize()})


@pytest.fixture
def tiny_pipeline() -> "ImagePipeline":
    """One detector and one measurer: the smallest pipeline a run can measure with."""
    return _build_tiny_pipeline()


@pytest.fixture
def tiny_pipeline_file(tmp_path: Path, tiny_pipeline: "ImagePipeline") -> Path:
    """:func:`tiny_pipeline` written to ``tmp_path / "tiny.pht-pipe.json"``."""
    path = tmp_path / "tiny.pht-pipe.json"
    path.write_text(tiny_pipeline.to_json() or "", encoding="utf-8")
    return path


@pytest.fixture
def tiny_inputs(tmp_path: Path) -> Path:
    """Two synthetic yeast plates: ``<tmp>/in/ds/plate_{1,2}.tif``.

    Returns the input root (``<tmp>/in``), ready for ``--input``.
    """
    from PIL import Image as PILImage

    from phenotypic.data import load_synth_yeast_plate

    root = tmp_path / "in"
    dataset = root / "ds"
    dataset.mkdir(parents=True)
    pixels = load_synth_yeast_plate().rgb[:].astype("uint8")
    for index in (1, 2):
        PILImage.fromarray(pixels).save(dataset / f"plate_{index}.tif")
    return root



@pytest.fixture
def tiny_inputs_one_bad(tiny_inputs: Path) -> Path:
    """:func:`tiny_inputs` plus one truncated plate, ``ds/plate_bad.tif``.

    The file keeps a real TIFF header and loses the rest of its bytes, so the
    input scan admits it by suffix and its pixel read fails inside the
    per-image worker in every mode. A run over this tree records exactly one
    terminal failure and succeeds on the other two plates.

    Returns the input root, ready for ``--input``.
    """
    good = (tiny_inputs / "ds" / "plate_1.tif").read_bytes()
    (tiny_inputs / "ds" / "plate_bad.tif").write_bytes(good[:64])
    return tiny_inputs


@pytest.fixture
def all_bad_inputs(tiny_inputs: Path) -> Path:
    """:func:`tiny_inputs` with **both** plates truncated to half their bytes.

    Unlike :func:`tiny_inputs_one_bad`, each file keeps its whole TIFF header
    and IFD, so the run preflight's header check passes (a ``[:64]`` cut on
    every input would be a ``PF-HEADER-UNREADABLE`` refusal, which writes
    nothing). The pixel read then fails inside the per-image worker, so a run
    over this tree starts, records two terminal failures and measures nothing.

    Returns the input root, ready for ``--input``.
    """
    for plate in sorted((tiny_inputs / "ds").glob("*.tif")):
        data = plate.read_bytes()
        plate.write_bytes(data[: len(data) // 2])
    return tiny_inputs


def make_dvc_repo(
    root: Path, *, cache_type: str | None, local_type: str | None = None
) -> Path:
    """Make *root* a DVC project root as far as PhenoTypic's preflight reads one.

    Writes ``.dvc/config`` with ``[cache] type = <cache_type>`` (an empty
    ``[core]`` section when ``cache_type`` is ``None``) and, when
    ``local_type`` is given, ``.dvc/config.local`` with its own
    ``[cache] type``. DVC itself is never invoked.

    Returns:
        The ``.dvc`` directory.
    """
    dvc = root / ".dvc"
    dvc.mkdir(parents=True, exist_ok=True)
    body = "[core]\n    autostage = true\n"
    if cache_type is not None:
        body += f"[cache]\n    type = {cache_type}\n"
    (dvc / "config").write_text(body, encoding="utf-8")
    if local_type is not None:
        (dvc / "config.local").write_text(
            f"[cache]\n    type = {local_type}\n", encoding="utf-8"
        )
    return dvc


def run_preflight(*, output_dir: Path, mode: str = "full") -> list:
    """The run preflight over :func:`tiny_pipeline` with ``--output output_dir``.

    A local run with no scanned inputs, so only the configuration-level checks
    have anything to say.

    Returns:
        Every :class:`~phenotypic._cli._cli_preflight.PreflightFinding`.
    """
    from phenotypic._cli._cli_preflight import PreflightContext
    from phenotypic._cli._cli_preflight import run_preflight as _run_preflight
    from phenotypic._cli._cli_types import ExecutionConfig

    config = ExecutionConfig(
        pipeline_json=output_dir.parent / "tiny_run.pht-pipe.json",
        input_path=output_dir.parent / "in",
        output_dir=output_dir,
        image_type="GridImage",
        nrows=None,
        ncols=None,
        bit_depth=None,
        n_jobs=1,
        slurm_args={},
        force_local=True,
        ext=".tiff",
        overlay_alpha=0.3,
        include_dataset_column=True,
        dry_run=False,
        sample=None,
        resume=False,
        retry_failures=False,
        skip_validation=False,
        measure_only=mode == "measure",
        process_only_layer="gray" if mode == "process" else None,
    )
    context = PreflightContext(
        config=config, pipeline=_build_tiny_pipeline(), datasets=(), mode=mode  # type: ignore[arg-type]
    )
    return list(_run_preflight(context).findings)


def run_cli(
    inputs: Path | None,
    out: Path,
    *,
    mode: str = "full",
    pipeline: Path | None = None,
    slurm: bool = False,
    extra: Sequence[str] = (),
) -> "Result":
    """Invoke ``python -m phenotypic`` in-process with :class:`click.testing.CliRunner`.

    Args:
        inputs: The ``--input`` root, or ``None`` to omit ``--input`` (as
            ``recompile`` over an existing output does).
        out: The ``--output`` root.
        mode: The ``--mode`` value.
        pipeline: The ``--pipeline`` file for ``full``, ``measure`` and
            ``process``. Defaults to :func:`tiny_pipeline`, written beside
            ``out`` as ``tiny_run.pht-pipe.json``.
        slurm: Add ``--slurm partition=intel`` (pair it with a stubbed
            submitter); otherwise ``full``, ``measure`` and ``process`` get
            ``--njobs 1``.
        extra: Further arguments, appended last.

    Returns:
        The runner's result; callers assert on ``exit_code`` and ``output``.
    """
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    args: list[str] = ["--output", str(out), "--mode", mode]
    if inputs is not None:
        args += ["--input", str(inputs)]
    if mode in _PIPELINE_MODES:
        if pipeline is None:
            pipeline = out.parent / "tiny_run.pht-pipe.json"
            if not pipeline.is_file():
                pipeline.parent.mkdir(parents=True, exist_ok=True)
                pipeline.write_text(
                    _build_tiny_pipeline().to_json() or "", encoding="utf-8"
                )
        args += ["--pipeline", str(pipeline)]
    if slurm:
        args += ["--slurm", "partition=intel"]
    elif mode in _PIPELINE_MODES:
        args += ["--njobs", "1"]
    args += list(extra)
    return CliRunner().invoke(phenotypic_cli, args)


#: The level-0 objmap array metadata :func:`break_store_for_measure` corrupts.
_MEASURE_BREAK_PATH = ("rgb", "labels", "objmap", "0", "zarr.json")


def break_store_for_measure(
    out: Path, stem: str = "plate_1"
) -> tuple[Path, bytes]:
    """Make one store of :func:`finished_tiny_run` fail a ``--mode measure`` re-measure.

    Corrupts the store's level-0 objmap array metadata. The root
    ``zarr.json`` is untouched, so the store is still scanned and its
    per-image record still verifies (it fingerprints the root only), but the
    store cannot be loaded, so its re-measure fails inside the per-image
    worker.

    Returns:
        ``(store, original_bytes)``; write the bytes back to
        ``store.joinpath(*_MEASURE_BREAK_PATH)`` to repair it, or call
        :func:`repair_store_for_measure`.
    """
    store = out / "results" / "ds" / "zarr" / f"{stem}.ome.zarr"
    target = store.joinpath(*_MEASURE_BREAK_PATH)
    original = target.read_bytes()
    target.write_text("{not json", encoding="utf-8")
    return store, original


def repair_store_for_measure(store: Path, original: bytes) -> None:
    """Undo :func:`break_store_for_measure`."""
    store.joinpath(*_MEASURE_BREAK_PATH).write_bytes(original)


def run_slurm_measure_worker(
    out: Path,
    store: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    generation: str | None = None,
) -> "Result":
    """Run the real ordinary SLURM array worker in ``--mode measure`` over *store*.

    ``python -m phenotypic._cli._cli_process_single`` is what each array task
    of a SLURM measure submission runs. ``SLURM_JOB_ID`` is set, so it takes
    its SLURM branch, and ``PHENOTYPIC_SLURM_GENERATION`` is *generation*
    (unset when ``None``), the generation it journals its terminal records
    under. With a generation, the output must carry an active lifecycle fence
    for it, or the worker's publication guard refuses.
    """
    from click.testing import CliRunner

    from phenotypic._cli import _cli_process_single

    with monkeypatch.context() as scoped:
        scoped.setenv("SLURM_JOB_ID", "123")
        if generation is None:
            scoped.delenv("PHENOTYPIC_SLURM_GENERATION", raising=False)
        else:
            scoped.setenv("PHENOTYPIC_SLURM_GENERATION", generation)
        return CliRunner().invoke(
            _cli_process_single.main,
            [
                "--pipeline",
                str(out.parent / "tiny_run.pht-pipe.json"),
                "--image",
                str(store),
                "--output-dir",
                str(out),
                "--dataset-name",
                "ds",
                "--image-type",
                "GridImage",
                "--mode",
                "measure",
            ],
        )


@pytest.fixture
def finished_tiny_run(tmp_path: Path, tiny_inputs: Path) -> Path:
    """A finished local full run of :func:`tiny_pipeline` over :func:`tiny_inputs`.

    Returns the output root (``<tmp>/out``).
    """
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out)
    assert result.exit_code == 0, result.output
    return out


@pytest.fixture
def tiny_mirror(finished_tiny_run: Path):
    """The ``deliverables/measurements.parquet`` frame of :func:`finished_tiny_run`.

    Returns:
        A polars ``DataFrame``: the post-applied, metadata-joined mirror.
    """
    import polars as pl

    return pl.read_parquet(finished_tiny_run / "deliverables" / "measurements.parquet")


@pytest.fixture
def old_tree_with_measurements_csv(
    tmp_path: Path, tiny_inputs: Path, metadata_bytes: dict[str, bytes]
) -> Path:
    """A finished run as a pre-P1 finalize left it: generated tables are ``.csv``.

    Built from a real run with a ``--metadata`` CSV. Every generated ``.tsv``
    under ``deliverables/`` is deleted and its rows are written back under the
    pre-cutover ``.csv`` name (comma-separated, ``measurements.csv`` included),
    so the tree holds no ``.tsv`` at all. The metadata snapshot is untouched.

    Returns:
        The output root.
    """
    import polars as pl

    metadata = tmp_path / "layout.csv"
    metadata.write_bytes(metadata_bytes[".csv"])
    out = tmp_path / "old_out"
    result = run_cli(tiny_inputs, out, extra=["--metadata", str(metadata)])
    assert result.exit_code == 0, result.output
    for tsv in sorted((out / "deliverables").rglob("*.tsv")):
        pl.read_csv(tsv, separator="\t", infer_schema_length=0).write_csv(
            tsv.with_suffix(".csv")
        )
        tsv.unlink()
    assert not list(out.rglob("*.tsv"))
    return out


@pytest.fixture
def metadata_bytes() -> dict[str, bytes]:
    """The same ``--metadata`` rows as CSV and as TSV, keyed by suffix.

    One row per :func:`tiny_inputs` plate (``plate_1``, ``plate_2``), joined on
    ``Metadata_ImageName``. The strain ``"S288C, MATa"`` holds a comma, so the
    CSV must quote it and the TSV must not: a reader that picks the wrong
    delimiter splits the field, and the join comparison fails rather than
    passing by accident.
    """
    rows = [("plate_1", "S288C, MATa"), ("plate_2", "BY4741")]
    header = ("Metadata_ImageName", "Metadata_Strain")
    csv_lines = [",".join(header)] + [f'{name},"{strain}"' for name, strain in rows]
    tsv_lines = ["\t".join(header)] + [f"{name}\t{strain}" for name, strain in rows]
    return {
        ".csv": ("\n".join(csv_lines) + "\n").encode("utf-8"),
        ".tsv": ("\n".join(tsv_lines) + "\n").encode("utf-8"),
    }
