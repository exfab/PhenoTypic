"""Shared test helpers for the #281 plan (DVC outputs, TSV, tables v2, embeddings).

Fixtures here are re-exported by ``tests/conftest.py`` with an explicit import
list, so every test under ``tests/`` can request them by name. ``phenotypic``
is imported inside each fixture rather than at module scope: the top-level
conftest must stay importable in dependency-light pytest runs (the
packaging-integrity CI job), where ``phenotypic`` may be absent.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from click.testing import Result

    from phenotypic import ImagePipeline
    from phenotypic.sdk_ import TableRun

#: Modes whose invocation takes ``--pipeline`` (see :func:`run_cli`).
_PIPELINE_MODES = frozenset({"full", "measure", "process"})


def _build_tiny_pipeline() -> "ImagePipeline":
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize

    # A fixed ``name``: the default is a fresh uuid, which gives every
    # serialization different bytes (and sha), so ``pipeline_file_a`` would not
    # be byte-identical to ``run_cli``'s default and ``tables_for`` could never
    # recognise a journal's digest as A's.
    return ImagePipeline(
        name="tiny-pipeline-A", ops=[OtsuDetector()], meas={"size": MeasureSize()}
    )


def _build_pipeline_b() -> "ImagePipeline":
    """:func:`_build_tiny_pipeline` plus ``MeasureShape()``: a second, different pipeline."""
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureShape, MeasureSize

    return ImagePipeline(
        name="tiny-pipeline-B",
        ops=[OtsuDetector()],
        meas={"size": MeasureSize(), "shape": MeasureShape()},
    )


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
            ``out`` as ``tiny_run.pht-pipe.json``. Those are the bytes of
            :func:`pipeline_file_a`, so a default run's table folder is
            ``<n>-<SHA_A[:12]>``.
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


# ---------------------------------------------------------------------------
# P3 · tables v2 (spec D1-D10)
# ---------------------------------------------------------------------------

#: The committed v1 fixtures (``tests/fixtures/tables_v1/_generate.py``).
_TABLES_V1_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "tables_v1"


@pytest.fixture
def v1_tree(tmp_path: Path) -> Path:
    """A ``tmp_path`` copy of the committed v1 full-run tree ``tables_v1/v1_run``.

    A complete one-image run written before #281 P3: its one store holds the
    flat ``tables/measurements/table.parquet``. Returns the output root.
    """
    import shutil

    target = tmp_path / "v1_run"
    shutil.copytree(_TABLES_V1_FIXTURES / "v1_run", target, symlinks=True)
    return target


@pytest.fixture
def v1_store(v1_tree: Path) -> Path:
    """The one store of :func:`v1_tree` (``results/ds/zarr/plate.ome.zarr``)."""
    return v1_tree / "results" / "ds" / "zarr" / "plate.ome.zarr"


@pytest.fixture
def v1_store_legacy_null_pipeline(tmp_path: Path) -> Path:
    """A ``tmp_path`` copy of ``tables_v1/v1_store_null_pipeline.ome.zarr``.

    :func:`v1_store` with a journal of one ``legacy`` application whose
    ``pipeline`` is ``null`` and no figure run, so its table digest is unknown.
    """
    import shutil

    target = tmp_path / "v1_store_null_pipeline.ome.zarr"
    shutil.copytree(_TABLES_V1_FIXTURES / "v1_store_null_pipeline.ome.zarr", target)
    return target


def v2_block(
    *,
    runs: Mapping[str, Mapping] = MappingProxyType({}),
    upstream: Mapping[str, Mapping] = MappingProxyType({}),
) -> dict:
    """A v2 ``attributes.phenotypic`` block whose ``tables`` holds *runs* and *upstream*.

    Only ``tables`` (and ``store_schema_version``, so the block is readable as
    a store root) is filled in; entries are copied as given.
    """
    from phenotypic.sdk_ import ngff_

    return {
        ngff_.PhenotypicAttr.STORE_SCHEMA_VERSION: ngff_.STORE_SCHEMA_VERSION,
        ngff_.PhenotypicAttr.TABLES: {
            "schema_version": ngff_.TABLES_SCHEMA_VERSION,
            "runs": {key: dict(value) for key, value in runs.items()},
            "upstream": {key: dict(value) for key, value in upstream.items()},
        },
    }


def _pipeline_bytes(builder) -> bytes:
    return (builder().to_json() or "").encode("utf-8")


@pytest.fixture
def pipeline_file_a(tmp_path: Path) -> Path:
    """The :func:`tiny_pipeline` as ``<tmp>/A.pht-pipe.json``; ``SHA_A`` is its ``file_sha256``.

    Byte-identical to :func:`run_cli`'s default pipeline file.
    """
    path = tmp_path / "A.pht-pipe.json"
    path.write_bytes(_pipeline_bytes(_build_tiny_pipeline))
    return path


@pytest.fixture
def pipeline_file_b(tmp_path: Path) -> Path:
    """The :func:`tiny_pipeline` plus ``MeasureShape()``, as ``<tmp>/B.pht-pipe.json``."""
    path = tmp_path / "B.pht-pipe.json"
    path.write_bytes(_pipeline_bytes(_build_pipeline_b))
    return path


def _pipeline_for_sha(sha: str | None) -> "tuple[str, ImagePipeline]":
    """``(marker, pipeline)`` for a journal digest: ``A``, ``B`` or ``unknown``."""
    import hashlib

    for marker, builder in (("A", _build_tiny_pipeline), ("B", _build_pipeline_b)):
        if sha == hashlib.sha256(_pipeline_bytes(builder)).hexdigest():
            return marker, builder()
    return "unknown", _build_tiny_pipeline()


@pytest.fixture
def measured_image(pipeline_file_a: Path):
    """``load_synth_yeast_plate()`` detected by ``OtsuDetector``, then opened as a ``full`` application of A.

    The application is left ``in_progress``, as a CLI worker holds it until
    the store is written.
    """
    from phenotypic._core._provenance import initialize_cli_provenance
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector

    image = OtsuDetector().apply(load_synth_yeast_plate())
    initialize_cli_provenance(
        image, pipeline_file_a, kind="full", input_filename="plate.tif"
    )
    return image


def tables_for(image):
    """The tables the forward worker builds for *image*, plus a ``Pipeline_Marker`` column.

    The pipeline is the one the image's journal names (``A`` or ``B``, by
    digest; the tiny pipeline when the journal names neither), and the marker
    is that letter, so a reader can tell which run folder a row came from.

    Returns:
        A :class:`~phenotypic.sdk_.PreparedImageTables`, with no metadata.
    """
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables
    from phenotypic._core._provenance import current_pipeline_sha256

    marker, pipeline = _pipeline_for_sha(current_pipeline_sha256(image))
    frame = pipeline.measure(image)
    frame["Pipeline_Marker"] = marker
    return prepare_image_tables(frame, None)


def tree_digest(path: Path) -> str:
    """sha256 over the sorted relative paths and bytes of every file under *path*."""
    import hashlib

    digest = hashlib.sha256()
    root = Path(path)
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(file.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(file.read_bytes() + b"\0")
    return digest.hexdigest()


def one_store(out: Path, *, stem: str | None = None) -> Path:
    """The single ``*.ome.zarr`` store under ``<out>/results``.

    Args:
        out: A run's output root.
        stem: Select the store of this image stem, for a run with several.

    Raises:
        AssertionError: Unless exactly one store matches.
    """
    from phenotypic.sdk_ import ngff_

    pattern = f"{stem}{ngff_.STORE_SUFFIX}" if stem is not None else f"*{ngff_.STORE_SUFFIX}"
    stores = sorted((Path(out) / "results").glob(f"*/zarr/{pattern}"))
    assert len(stores) == 1, f"expected one store matching {pattern}, found {stores}"
    return stores[0]


def owned_table_run(store: Path, pipeline_sha256: str | None) -> TableRun | None:
    """The run folder *pipeline_sha256* owns in *store*, or ``None``.

    The writer's ownership rule (:func:`~phenotypic.sdk_.owned_table_run_id`)
    over the store's root, resolved to its :class:`~phenotypic.sdk_.TableRun`.
    """
    from phenotypic.sdk_ import list_table_runs, ngff_, owned_table_run_id

    run_id = owned_table_run_id(
        ngff_.require_readable_store(Path(store)), pipeline_sha256=pipeline_sha256
    )
    return next((run for run in list_table_runs(store) if run.run_id == run_id), None)


def _journal_run_id(store: Path) -> str:
    import json

    from phenotypic._core._provenance import journal_pipeline_sha256
    from phenotypic.sdk_ import choose_table_run_id

    root = Path(store) / "zarr.json"
    block = (
        json.loads(root.read_text(encoding="utf-8"))["attributes"]["phenotypic"]
        if root.is_file()
        else {}
    )
    return choose_table_run_id(
        block, pipeline_sha256=journal_pipeline_sha256(block.get("provenance"))
    )


def journal_table_path(store: Path) -> Path:
    """Where *store*'s journal-owned run folder keeps ``measurements/table.parquet``.

    The folder a forward save of the store's journal would own (spec D6):
    the existing folder for the journal's pipeline digest, else the next
    number. A store with no root yet is numbered as an empty one, so this
    names ``tables/1-<sha12|unknown>/...`` for a store about to be created.
    """
    from phenotypic.sdk_._table_runs import measurement_table_relative_path

    return Path(store) / measurement_table_relative_path(_journal_run_id(store))


def plant_measurement_table(store: Path, payload: bytes = b"measurements") -> Path:
    """Write *payload* as *store*'s own measurements table, in the v2 layout.

    The stand-in for "the forward save embedded a table" in resume and
    classification tests, which care that the table exists, not what it
    holds. The bytes go to :func:`journal_table_path`, and when the store has
    a root its ``tables`` descriptor gains (or keeps) that run with a
    ``measurements`` block, so the resolver finds it exactly as it finds a
    real save's. A store with no root gets the file only: nothing can declare
    it, which is the state a reader must treat as "no table".

    Returns:
        The table path.
    """
    import json

    from phenotypic._core._provenance import journal_pipeline_sha256
    from phenotypic.sdk_ import ngff_
    from phenotypic.sdk_._table_runs import TABLE_RUN_ID_RE

    from phenotypic.sdk_._table_runs import measurement_table_relative_path

    run_id = _journal_run_id(store)
    table = Path(store) / measurement_table_relative_path(run_id)
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_bytes(payload)
    root_path = Path(store) / "zarr.json"
    if root_path.is_file():
        match = TABLE_RUN_ID_RE.match(run_id)
        assert match is not None, run_id
        root = json.loads(root_path.read_text(encoding="utf-8"))
        block = root["attributes"]["phenotypic"]
        tables = block.get(ngff_.PhenotypicAttr.TABLES)
        if not isinstance(tables, dict) or "schema_version" not in tables:
            tables = {"schema_version": ngff_.TABLES_SCHEMA_VERSION, "runs": {}, "upstream": {}}
        entry = tables["runs"].setdefault(
            run_id,
            {
                "order": int(match["order"]),
                "pipeline_sha256": journal_pipeline_sha256(block.get("provenance")),
            },
        )
        entry["measurements"] = {
            "path": table.relative_to(store).as_posix(),
        }
        block[ngff_.PhenotypicAttr.TABLES] = tables
        root_path.write_text(json.dumps(root), encoding="utf-8")
    return table


def _load_store(store: Path):
    """Load *store* with the class its root names (``GridImage`` or ``Image``)."""
    import phenotypic
    from phenotypic.sdk_ import ngff_

    image_class = ngff_.read_phenotypic_attributes(store).get(
        ngff_.PhenotypicAttr.IMAGE_CLASS, "Image"
    )
    return getattr(phenotypic, image_class).load_zarr(store)


def _close_open_application(image) -> None:
    from phenotypic._core._provenance import set_provenance_status

    applications = image._metadata.provenance_journal["applications"]
    if applications and applications[-1]["status"] not in {"complete", "failed"}:
        set_provenance_status(image, "complete")


def resave_with_pipeline(store: Path, pipeline_file: Path, *, tables: bool = True) -> Path:
    """Load *store*, open a ``full`` application of *pipeline_file*, and save it back.

    With ``tables=False`` the save carries no tables of its own, as the
    initial-checkpoint and Stage-1 saves do.
    """
    from phenotypic._core._provenance import initialize_cli_provenance

    image = _load_store(store)
    _close_open_application(image)
    initialize_cli_provenance(image, pipeline_file, kind="full", input_filename="plate.tif")
    return image.save2zarr(store, measurement_table=tables_for(image) if tables else None)


@pytest.fixture
def store_with_two_runs(tmp_path: Path, measured_image, pipeline_file_b: Path) -> Path:
    """:func:`measured_image` saved with A's tables, then re-saved by B: folders ``1-<A>`` and ``2-<B>``."""
    store = measured_image.save2zarr(
        tmp_path / "two_runs.ome.zarr", measurement_table=tables_for(measured_image)
    )
    resave_with_pipeline(store, pipeline_file_b)
    return store


#: The embedding key and model folder :func:`store_with_crop_entry` writes.
OBJECTS_ENTRY_KEY = "dinov2-vitb14/rgb/objects"


def _write_crop_payload(part: Path, run_id: str, labels) -> dict:
    """Write a 4-D float16 crop embedding table into *part* and return its raw entry."""
    import hashlib

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from phenotypic.sdk_ import ngff_
    from phenotypic.sdk_._table_runs import embedding_table_relative_path

    relative = embedding_table_relative_path(run_id, OBJECTS_ENTRY_KEY)
    payload = part / relative
    payload.parent.mkdir(parents=True, exist_ok=True)
    for group in payload.parents[:4]:  # <region>/, <layer>/, <model>/, embeddings/
        (group / ngff_.STORE_ROOT_JSON).write_text(
            '{"zarr_format": 3, "node_type": "group", "attributes": {}}',
            encoding="utf-8",
        )
    vectors = pa.FixedSizeListArray.from_arrays(
        pa.array(np.tile(np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float16), len(labels))),
        4,
    )
    pq.write_table(
        pa.table({"Object_Label": pa.array(labels, type=pa.int64()), "embedding": vectors}),
        payload,
    )
    return {
        "model": {"family": "dinov2", "hf_id": "facebook/dinov2-base",
                  "revision": "0" * 40, "patch_size": 14},
        "dim": 4,
        "dtype": "float16",
        "pooling": "crop_cls",
        "key": "Object_Label",
        "input": {"series": "rgb", "level": 0, "layer": "rgb"},
        "producer": {"class": "MeasureDinoEmbedding", "params_sha256": "0" * 64},
        "binds": {"objmap_path": "rgb/labels/objmap", "objmap_sha256": None},
        "stale": False,
        "files": [{
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            "media_type": "application/vnd.apache.parquet",
        }],
    }


@pytest.fixture
def store_with_crop_entry(tmp_path: Path, measured_image) -> Path:
    """A measured store whose one run entry holds a hand-built crop embedding entry.

    The entry is bound to the store's objmap (``binds.objmap_sha256`` is
    :func:`~phenotypic.sdk_._measurement_tables.objmap_digest` of it) and is
    written through ``write_run_entry``'s ``embeddings=`` raw-entry seam,
    inside a root-last rewrite of the store.
    """
    import numpy as np

    from phenotypic.sdk_ import list_table_runs
    from phenotypic.sdk_._measurement_tables import (
        _rewrite_store_tables,
        objmap_digest,
        write_run_entry,
    )

    store = measured_image.save2zarr(
        tmp_path / "crops.ome.zarr", measurement_table=tables_for(measured_image)
    )
    run = list_table_runs(store)[-1]
    objmap = measured_image.objmap[:]
    tables = tables_for(measured_image)

    def _plan(phenotypic: dict):
        def _populate(part: Path) -> None:
            raw = _write_crop_payload(
                part, run.run_id, [int(v) for v in np.unique(objmap) if v]
            )
            raw["binds"]["objmap_sha256"] = objmap_digest(objmap)
            entry = write_run_entry(
                part,
                run.run_id,
                order=run.order,
                pipeline_sha256=run.pipeline_sha256,
                measurements=tables,
                objmap_target="rgb/labels/objmap",
                embeddings={OBJECTS_ENTRY_KEY: raw},
                objmap_sha256=objmap_digest(objmap),
            )
            phenotypic["tables"]["runs"][run.run_id] = entry

        return _populate

    _rewrite_store_tables(
        store, plan=_plan, durable=False, commit_guard=None, clear_table_run=run.run_id
    )
    return store


def resave_with_objmap(store: Path, *, zeros: bool) -> Path:
    """Re-save *store* table-less, with an all-zeros objmap (Stage 1) or the detected one (Stage 3).

    ``zeros=False`` re-runs ``OtsuDetector`` on the loaded image, which gives
    back the labels the store was measured with.
    """
    from phenotypic.detect import OtsuDetector

    image = _load_store(store)
    _close_open_application(image)
    if zeros:
        image.objmap[:] = 0
    else:
        image = OtsuDetector().apply(image)
    return image.save2zarr(store)


def objects_entry(store: Path) -> dict:
    """The :data:`OBJECTS_ENTRY_KEY` embedding entry of *store*'s first run that has one."""
    from phenotypic.sdk_ import list_table_runs

    for run in list_table_runs(store):
        embeddings = run.descriptor.get("embeddings") or {}
        if OBJECTS_ENTRY_KEY in embeddings:
            return dict(embeddings[OBJECTS_ENTRY_KEY])
    raise KeyError(f"{store} has no {OBJECTS_ENTRY_KEY} entry")


# ---------------------------------------------------------------------------
# P3 T3 (C13): the ``--mode measure`` rewrite (spec D5-D7)
# ---------------------------------------------------------------------------


@pytest.fixture
def full_run_store(tmp_path: Path, measured_image) -> Path:
    """:func:`measured_image` saved with A's tables: one run folder, ``1-<A>``."""
    return measured_image.save2zarr(
        tmp_path / "full_run.ome.zarr", measurement_table=tables_for(measured_image)
    )


def _stored_tables(store: Path, *, marker: str | None):
    """The tables :func:`tables_for` builds for *store*'s own image, optionally re-marked."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables
    from phenotypic._core._provenance import current_pipeline_sha256

    image = _load_store(store)
    journal_marker, pipeline = _pipeline_for_sha(current_pipeline_sha256(image))
    frame = pipeline.measure(image)
    frame["Pipeline_Marker"] = journal_marker if marker is None else marker
    return prepare_image_tables(frame, None)


def tables_for_store(store: Path):
    """The tables *store*'s forward save wrote, rebuilt from the stored image.

    Same rows as the store's own run folder, and the ``Pipeline_Marker`` of
    the pipeline its journal names.
    """
    return _stored_tables(store, marker=None)


def other_tables(store: Path):
    """:func:`tables_for_store` with ``Pipeline_Marker`` ``"other"``.

    The stand-in for another pipeline's measurements, so a reader can tell
    a row of the new run folder from one of the folder it sits beside.
    """
    return _stored_tables(store, marker="other")


# ---------------------------------------------------------------------------
# P3 T6 (C16): the upstream reference (spec D2, D8)
# ---------------------------------------------------------------------------


@pytest.fixture
def input_store_v2(tmp_path: Path, measured_image) -> Path:
    """:func:`measured_image` saved with A's tables as ``<tmp>/in/ds/plate.ome.zarr``.

    It holds one run folder, ``1-<A>``, and is a complete store a forward
    run can take as ``--input <tmp>/in`` (dataset ``ds``). The application is
    closed first, as a finished CLI run leaves it.
    """
    _close_open_application(measured_image)
    return measured_image.save2zarr(
        tmp_path / "in" / "ds" / "plate.ome.zarr",
        measurement_table=tables_for(measured_image),
    )


def root_tables(store: Path) -> dict:
    """The store's ``attributes.phenotypic.tables`` block, read from its root ``zarr.json``."""
    import json

    root = json.loads((Path(store) / "zarr.json").read_text(encoding="utf-8"))
    return root["attributes"]["phenotypic"]["tables"]


def touch_root(store: Path) -> None:
    """Rewrite *store*'s root ``zarr.json`` with one more byte of whitespace.

    The parsed document is unchanged; only its bytes, and so its sha256,
    move -- the stand-in for an input reprocessed in place.
    """
    root = Path(store) / "zarr.json"
    root.write_bytes(root.read_bytes() + b" ")


@pytest.fixture
def certified_older_than_latest(finished_tiny_run: Path, pipeline_file_b: Path) -> Path:
    """A finished run whose records certify folder 1 while folder 2 is the latest.

    Forward A writes folder 1; ``--mode measure`` with B adds folder 2 and
    moves the records to it; ``--mode measure`` with A again replaces folder
    1 and moves the records back. So the store's latest measurement folder
    (B's, the only one with shape columns) is not the one any record
    certifies, which is the case a "latest folder" reader gets wrong.

    Returns the output root.
    """
    from phenotypic._cli._cli_completion import valid_image_success
    from phenotypic._cli._cli_failure_tracker import file_sha256
    from phenotypic.sdk_ import list_table_runs
    from phenotypic.sdk_._image_record import read_image_record

    out = finished_tiny_run
    pipeline_a = out.parent / "tiny_run.pht-pipe.json"
    for pipeline in (pipeline_file_b, pipeline_a):
        result = run_cli(None, out, mode="measure", pipeline=pipeline)
        assert result.exit_code == 0, result.output
    for stem in ("plate_1", "plate_2"):
        store = one_store(out, stem=stem)
        first, latest = list_table_runs(store)
        assert latest.pipeline_sha256 == file_sha256(pipeline_file_b)
        record = read_image_record(out, "ds", stem)
        assert record is not None
        assert valid_image_success(
            out, dataset="ds", image_stem=stem, work_id=str(record["work_id"])
        )
        certified = out / record["artifacts"]["measurements"]["path"]
        assert certified == first.measurements_path()
    return out


# ---------------------------------------------------------------------------
# P4 T1 (C17): ``MeasureEmbedding`` and pipeline routing (spec E1, E2)
# ---------------------------------------------------------------------------


#: Vector length of the toy embedders.
TOY_DIM = 4

_TOY_EMBEDDER: type | None = None


def toy_box_layer_mean(pixels, boxes):
    """The toy vector's last value: the mean of uint8 *pixels* in each box, over 255.

    *pixels* is what the toy embedder reads, ``embedder._layer_pixels(image)``:
    ``image.rgb[:]`` for ``layer="rgb"``, the gray layer as three uint8
    channels for ``layer="gray"``. Each ``(r0, c0, r1, c1)`` box's mean is
    taken over every channel.
    """
    import numpy as np

    return np.array(
        [float(pixels[r0:r1, c0:c1].mean()) / 255.0 for r0, c0, r1, c1 in boxes],
        dtype=np.float32,
    )


def toy_embedder_class() -> type:
    """The toy :class:`~phenotypic.abc_.MeasureEmbedding`, built once per process.

    Defined lazily rather than at module scope, because this module must
    import without ``phenotypic``, and cached so every fixture shares one
    class. Its first ``TOY_DIM - 1`` values are ``boxes.sum(1)``, so a test can
    tell which box a row came from without a model, and its last is the box's
    mean pixel of the embedder's ``layer`` (:func:`toy_box_layer_mean`, read
    through ``_layer_pixels`` as every embedder reads it), so a vector also
    depends on the pixels it was computed from: an embedding of the wrong
    pixels or layer differs (#281 P5 review, test quality). ``token`` is the model token, so
    two "models" can sit side by side in one run.
    """
    global _TOY_EMBEDDER
    if _TOY_EMBEDDER is not None:
        return _TOY_EMBEDDER

    import numpy as np

    from phenotypic.abc_ import MeasureEmbedding

    class ToyEmbedder(MeasureEmbedding):
        """A model-free embedder for tests.

        Args:
            token: The model token this embedder reports.
        """

        token: str = "toy-v1"

        def model_token(self) -> str:
            return self.token

        def entry_descriptor(self) -> dict:
            return {
                "model": {
                    "family": "toy",
                    "hf_id": None,
                    "revision": None,
                    "patch_size": None,
                },
                "dim": TOY_DIM,
                "dtype": "float16",
                "pooling": "box_sum",
            }

        def _embed_boxes(self, image, boxes):
            sums = boxes.sum(axis=1).astype(np.float32)
            means = toy_box_layer_mean(self._layer_pixels(image), boxes)
            return np.concatenate(
                [np.repeat(sums[:, None], TOY_DIM - 1, axis=1), means[:, None]],
                axis=1,
            )

    _TOY_EMBEDDER = ToyEmbedder
    return ToyEmbedder


@pytest.fixture
def plate():
    """``load_synth_yeast_plate()``: a 96-colony ``GridImage`` with its objmap set."""
    from phenotypic.data import load_synth_yeast_plate

    return load_synth_yeast_plate()


@pytest.fixture
def detected_plate(plate):
    """:func:`plate` with ``OtsuDetector`` applied."""
    from phenotypic.detect import OtsuDetector

    return OtsuDetector().apply(plate)


@pytest.fixture
def toy_embedder():
    """A crop embedder: ``model_token() == "toy-v1"``, ``regions="objects"``."""
    return toy_embedder_class()()


#: The 2 x 2 grid :func:`toy_grid_embedder` embeds, in pixels of the synth plate.
TOY_GRID_ROW_EDGES = (0, 100, 200)
TOY_GRID_COL_EDGES = (0, 150, 300)


@pytest.fixture
def toy_grid_embedder():
    """A grid embedder: ``regions="grid"`` over a 2 x 2 ``ManualGridFinder``."""
    from phenotypic.grid import ManualGridFinder

    return toy_embedder_class()(
        regions="grid",
        grid=ManualGridFinder(
            row_edges=list(TOY_GRID_ROW_EDGES), col_edges=list(TOY_GRID_COL_EDGES)
        ),
    )


# ---------------------------------------------------------------------------
# P4 T2 (C17): embedding tables in the store (spec D1, D3, D7, D9)
# ---------------------------------------------------------------------------

#: Payloads computed for an image by the fixtures below, keyed by ``id(image)``.
#: An ``Image`` defines ``__eq__`` and so is unhashable; the image is held in
#: the value, so its id cannot be reused while the entry lives, and each
#: fixture removes its entry on teardown.
_IMAGE_PAYLOADS: dict[int, tuple[object, dict]] = {}


def payloads_of(image) -> dict:
    """The ``{"<model>/<layer>/<region>": EmbeddingPayload}`` a fixture computed for *image*."""
    return _IMAGE_PAYLOADS[id(image)][1]


def _with_cli_payloads(image, pipeline_file: Path, embedders: Mapping[str, object]):
    """Open a ``full`` CLI application of *pipeline_file* on *image* and embed it.

    The payloads are what ``ImagePipeline(meas=embedders).embed(image)``
    returns, recorded for :func:`payloads_of` / :func:`save_with_payload`.
    """
    from phenotypic import ImagePipeline
    from phenotypic._core._provenance import initialize_cli_provenance

    initialize_cli_provenance(
        image, pipeline_file, kind="full", input_filename="plate.tif"
    )
    payloads = ImagePipeline(meas=dict(embedders)).embed(image)
    _IMAGE_PAYLOADS[id(image)] = (image, payloads)
    return image


@pytest.fixture
def image_with_objects_payload(detected_plate, pipeline_file_a: Path, toy_embedder):
    """:func:`detected_plate` as a ``full`` application of A, with one ``toy-v1/rgb/objects`` payload."""
    image = _with_cli_payloads(detected_plate, pipeline_file_a, {"emb": toy_embedder})
    yield image
    _IMAGE_PAYLOADS.pop(id(image), None)


@pytest.fixture
def image_with_two_models(detected_plate, pipeline_file_a: Path, toy_embedder):
    """:func:`detected_plate` with payloads ``toy-v1/rgb/objects`` and ``toy-v2/rgb/objects``."""
    second = toy_embedder_class()(token="toy-v2")
    image = _with_cli_payloads(
        detected_plate, pipeline_file_a, {"v1": toy_embedder, "v2": second}
    )
    yield image
    _IMAGE_PAYLOADS.pop(id(image), None)


@pytest.fixture
def image_with_grid_payload(detected_plate, pipeline_file_a: Path, toy_grid_embedder):
    """:func:`detected_plate` with one ``toy-v1/rgb/grid`` payload over the 2 x 2 toy grid."""
    image = _with_cli_payloads(
        detected_plate, pipeline_file_a, {"grid": toy_grid_embedder}
    )
    yield image
    _IMAGE_PAYLOADS.pop(id(image), None)


def save_with_payload(directory: Path, image) -> Path:
    """``save2zarr`` *image* into *directory* with its tables and its payloads.

    The tables are :func:`tables_for`'s; the payloads :func:`payloads_of`'s.

    Returns:
        The promoted store, ``<directory>/payload.ome.zarr``.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    return image.save2zarr(
        Path(directory) / "payload.ome.zarr",
        measurement_table=tables_for(image),
        embeddings=payloads_of(image),
    )


def run_entry(store: Path) -> dict:
    """The latest run's descriptor entry in *store*'s root."""
    from phenotypic.sdk_ import list_table_runs

    return dict(list_table_runs(store)[-1].descriptor)


# ---------------------------------------------------------------------------
# P4 T3 (C18): ``MeasureDinoEmbedding`` without a download (spec E3)
# ---------------------------------------------------------------------------

#: The stub backbone's geometry: DINOv2's patch, a tiny hidden size.
STUB_PATCH_SIZE = 14
STUB_HIDDEN_SIZE = 8
#: What ``physical_batch_for`` reads from the config (#282 P2 I3); the stub
#: has no attention, so any positive count gives a valid ``m``.
STUB_ATTENTION_HEADS = 1
STUB_COMMIT = "0" * 40


#: The value of every register token a :func:`_build_stub_dino` with
#: registers emits: loud, so a pooling that reads one is visibly wrong.
STUB_REGISTER_VALUE = 1000.0


#: Where a stub with ``snapshot_revision`` says it was loaded from: a hub
#: cache snapshot folder, as ``resolved_revision``'s fallback reads one.
STUB_SNAPSHOT_ROOT = "/stub-hf-cache"


def _stub_snapshot_path(dino_size: str, revision: str) -> str:
    """``<root>/models--facebook--dinov2-<size>/snapshots/<revision>``."""
    return f"{STUB_SNAPSHOT_ROOT}/models--facebook--dinov2-{dino_size}/snapshots/{revision}"


class _StubInputs(dict):
    """The stub processor's output: a ``BatchFeature``-like dict with ``.to``."""

    def to(self, device):
        return _StubInputs({key: value.to(device) for key, value in self.items()})


class StubDinoProcessor:
    """The stub backbones' processor: DINO's ImageNet normalisation, torch out.

    Module level, not local to :func:`_build_stub_dino`, so it pickles: Stage
    4 pickles a ``masked_dense`` backbone's processor into its loader's
    dataset for ``spawn`` workers (#282 re-review item 2c).
    """

    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    rescale_factor = 1 / 255

    def __call__(self, images, return_tensors="pt", **kwargs):
        import numpy as np
        import torch

        arr = np.asarray(images, dtype=np.float32) * self.rescale_factor
        arr = (arr - np.asarray(self.image_mean, np.float32)) / np.asarray(
            self.image_std, np.float32
        )
        pixels = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return _StubInputs(pixel_values=pixels[None])


def _build_stub_dino(
    num_register_tokens: int = 0,
    *,
    commit_hash: str | None = STUB_COMMIT,
    name_or_path: str = "",
):
    """A deterministic stand-in for a DINOv2 ``(model, processor)``; needs torch.

    The model average-pools each ``14 x 14`` patch of the normalized pixels
    to its three channel means and spreads them over :data:`STUB_HIDDEN_SIZE`
    dimensions (the means, their squares, the first two again). So a vector
    depends on the pixels it was computed from, and a test can tell regions
    apart without real weights.

    The CLS token is the patch tokens' **maximum**, not their mean, so
    ``crop_cls`` and ``crop_mean`` give different vectors (review T2). With
    *num_register_tokens*, that many tokens of :data:`STUB_REGISTER_VALUE`
    sit between the CLS and the patches, as DINOv2-with-registers lays them
    out, so a pooling that reads one is visibly wrong (review T3). The
    processor is callable (the dense path) and carries the normalization
    constants the batched crop path reads.

    *commit_hash* is the config's ``_commit_hash`` (``None``: unset, as on
    transformers >= 5.18), and *name_or_path* its ``_name_or_path``.
    """
    from types import SimpleNamespace

    import torch

    class _StubDinoModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                patch_size=STUB_PATCH_SIZE,
                hidden_size=STUB_HIDDEN_SIZE,
                num_attention_heads=STUB_ATTENTION_HEADS,
                num_register_tokens=num_register_tokens,
                _commit_hash=commit_hash,
                _name_or_path=name_or_path,
            )

        def forward(self, pixel_values):
            pooled = torch.nn.functional.avg_pool2d(pixel_values, STUB_PATCH_SIZE)
            tokens = pooled.flatten(2).transpose(1, 2)  # (B, Hp*Wp, 3)
            patches = torch.cat([tokens, tokens**2, tokens[..., :2]], dim=-1)
            cls = patches.amax(dim=1, keepdim=True)
            registers = torch.full(
                (patches.shape[0], num_register_tokens, patches.shape[-1]),
                STUB_REGISTER_VALUE,
                dtype=patches.dtype,
            )
            return SimpleNamespace(
                last_hidden_state=torch.cat([cls, registers, patches], dim=1)
            )

    return _StubDinoModel().eval(), StubDinoProcessor()


def _build_param_stub_dino(
    num_register_tokens: int = 0,
    *,
    leaky: bool = False,
    commit_hash: str | None = STUB_COMMIT,
    name_or_path: str = "",
):
    """A parametric stand-in for a DINOv2 ``(model, processor)``; needs torch.

    Where :func:`_build_stub_dino` has no weights, this model has real ones, so
    ``.to(dtype)`` changes what it computes (#282 plan review I14). The pooled
    patch features are laid out as in the plain stub, the CLS (patch maximum)
    and the *num_register_tokens* registers
    (:data:`STUB_REGISTER_VALUE`) are prepended, and every token goes through
    a fixed-seed ``Linear(STUB_HIDDEN_SIZE, STUB_HIDDEN_SIZE)``, a layer norm
    and a softmax mixing **within each sample's tokens**. So patch and
    register tokens reach the CLS output, and a sample's rows never depend on
    its batch neighbours.

    ``leaky=True`` adds the **batch** mean to every row: a model whose rows
    depend on their neighbours, which a batch-isolation check must catch
    (plan review I13). The processor is the plain stub's. *commit_hash* and
    *name_or_path* are as :func:`_build_stub_dino`'s.
    """
    from types import SimpleNamespace

    import torch

    _, processor = _build_stub_dino(num_register_tokens)

    class _ParamStubDinoModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261119)
                self.mix = torch.nn.Linear(STUB_HIDDEN_SIZE, STUB_HIDDEN_SIZE)
            self.config = SimpleNamespace(
                patch_size=STUB_PATCH_SIZE,
                hidden_size=STUB_HIDDEN_SIZE,
                num_attention_heads=STUB_ATTENTION_HEADS,
                num_register_tokens=num_register_tokens,
                _commit_hash=commit_hash,
                _name_or_path=name_or_path,
            )

        def forward(self, pixel_values):
            dtype = self.mix.weight.dtype
            pooled = torch.nn.functional.avg_pool2d(
                pixel_values.to(dtype), STUB_PATCH_SIZE
            )
            tokens = pooled.flatten(2).transpose(1, 2)  # (B, Hp*Wp, 3)
            patches = torch.cat([tokens, tokens**2, tokens[..., :2]], dim=-1)
            cls = patches.amax(dim=1, keepdim=True)
            registers = torch.full(
                (patches.shape[0], num_register_tokens, patches.shape[-1]),
                STUB_REGISTER_VALUE,
                dtype=dtype,
            )
            seq = torch.cat([cls, registers, patches], dim=1)
            hidden = torch.nn.functional.layer_norm(
                self.mix(seq), (STUB_HIDDEN_SIZE,)
            )
            scores = hidden @ hidden.transpose(1, 2) / STUB_HIDDEN_SIZE**0.5
            mixed = torch.softmax(scores, dim=-1) @ hidden  # within each sample
            if leaky:
                mixed = mixed + mixed.mean(dim=0, keepdim=True)
            return SimpleNamespace(last_hidden_state=mixed)

    return _ParamStubDinoModel().eval(), processor


@pytest.fixture
def stub_backbone(monkeypatch):
    """Patch ``MeasureDinoEmbedding``'s backbone loader with a tiny stub.

    No download and no GPU. ``load_dino_backbone`` in the measurer's module
    is replaced by a loader that records each call's arguments in
    ``load_calls`` and returns :func:`_build_stub_dino`'s model and
    processor. torch is imported only when the stub is loaded, so a test
    refused before loading runs without it. The module's backbone cache is
    cleared before and after, so no model leaks between tests.

    Yields:
        A namespace whose ``load_calls`` is a list of
        ``{dino_version, dino_size, device, revision, dtype}`` dicts, one per
        load.
    """
    from types import SimpleNamespace

    from phenotypic.measure import _measure_dino_embedding as module

    calls = install_stub_backbone(monkeypatch)
    yield SimpleNamespace(load_calls=calls)
    module._cached_backbone.cache_clear()


@pytest.fixture
def param_stub_backbone(monkeypatch):
    """:func:`stub_backbone` with the parametric stub (#282 tests use this).

    Same ``load_calls`` namespace; the backbone cache is cleared afterwards.
    """
    from types import SimpleNamespace

    from phenotypic.measure import _measure_dino_embedding as module

    calls = install_stub_backbone(monkeypatch, parametric=True)
    yield SimpleNamespace(load_calls=calls)
    module._cached_backbone.cache_clear()


def inline_golden_cases() -> dict:
    """The #282 P0 golden cases: ``{case: ImagePipeline}``, all on ``device="cpu"``.

    Run each under the parametric stub (``registers_mean`` with
    ``num_register_tokens=4``). Pinned to the CPU so a GPU node cannot move
    the bytes (plan review I13).
    """
    from phenotypic import ImagePipeline
    from phenotypic.grid import ManualGridFinder
    from phenotypic.measure import MeasureDinoEmbedding

    def _pipe(**kwargs):
        return ImagePipeline(meas={"emb": MeasureDinoEmbedding(device="cpu", **kwargs)})

    return {
        "objects_cls": _pipe(),
        "objects_mean_masked": _pipe(
            pooling="crop_mean", mask_background=True, crop_pad_frac=0.5
        ),
        "objects_dense": _pipe(pooling="masked_dense", crop_px=112),
        "grid_cls": _pipe(
            regions="grid",
            grid=ManualGridFinder(row_edges=[0, 100, 200], col_edges=[0, 150, 300]),
        ),
        "gray_objects": _pipe(layer="gray", name="g"),
        "registers_mean": _pipe(pooling="crop_mean"),
    }


def install_stub_backbone(
    monkeypatch,
    *,
    num_register_tokens: int = 0,
    parametric: bool = False,
    leaky: bool = False,
    commit_hash: str | None = STUB_COMMIT,
    snapshot_revision: str | None = None,
    snapshot_from_revision: bool = False,
) -> list[dict]:
    """Patch ``MeasureDinoEmbedding``'s loader with :func:`_build_stub_dino`.

    With ``parametric=True`` the stub is :func:`_build_param_stub_dino` (real
    weights, so ``dtype`` matters: a load's ``dtype=`` casts them), and
    ``leaky=True`` makes its rows depend on the batch. The default is the
    #281 stub, so #281's tests keep it.

    ``commit_hash`` is the loaded config's ``_commit_hash``; ``None`` leaves
    it unset, as transformers >= 5.18 does. ``snapshot_revision`` then lays
    the config's ``_name_or_path`` out as a hub snapshot,
    ``<root>/models--facebook--dinov2-<size>/snapshots/<revision>``, which
    ``resolved_revision``'s fallback reads; ``snapshot_from_revision=True``
    takes that revision from the load's ``revision=`` instead (U5, P5 T2).

    The processor-only loader (``load_dino_processor``, which ``masked_dense``
    preparation uses when given no processor) is patched to the stub's
    processor too, without recording a load.

    Clears the measurer's backbone cache first, so the next measure loads
    this stub. :func:`stub_backbone` is this with no registers; a test that
    compares two stubs calls it twice.

    Returns:
        The list each load appends its ``{dino_version, dino_size, device,
        revision, dtype}`` to.
    """
    from phenotypic.measure import _measure_dino_embedding as module

    calls: list[dict] = []

    def _stub_load(dino_version, dino_size, device, *, revision=None, dtype="float32"):
        calls.append(
            {
                "dino_version": dino_version,
                "dino_size": dino_size,
                "device": device,
                "revision": revision,
                "dtype": dtype,
            }
        )
        snapshot = revision if snapshot_from_revision else snapshot_revision
        identity = {
            "commit_hash": commit_hash,
            "name_or_path": "" if snapshot is None else _stub_snapshot_path(dino_size, snapshot),
        }
        if parametric:
            import torch

            model, processor = _build_param_stub_dino(
                num_register_tokens, leaky=leaky, **identity
            )
            return model.to(getattr(torch, dtype)), processor
        return _build_stub_dino(num_register_tokens, **identity)

    def _stub_processor(dino_version, dino_size, *, revision=None):
        # The processor-only loader (``masked_dense`` tiles, #282 P1): the
        # stub's processor, which every stub shares. Not a model load, so not
        # in ``calls``.
        return _build_stub_dino(num_register_tokens)[1]

    module._cached_backbone.cache_clear()
    monkeypatch.setattr(module, "load_dino_backbone", _stub_load)
    monkeypatch.setattr(module, "load_dino_processor", _stub_processor)
    return calls


# ---------------------------------------------------------------------------
# P4 T4 (C19): CLI wiring and the embedding deliverables (spec E2, E5)
# ---------------------------------------------------------------------------

#: The crop entry a default ``MeasureDinoEmbedding()`` (DINOv2-base) writes.
DINO_OBJECTS_KEY = "dinov2-vitb14/rgb/objects"


def with_dino_objects(directory: Path) -> Path:
    """A pipeline file: :func:`tiny_pipeline`'s ops and ``meas``, plus ``meas["emb"]``.

    ``meas["emb"]`` is ``MeasureDinoEmbedding(device="cpu")``, otherwise the
    defaults: crop embeddings under :data:`DINO_OBJECTS_KEY`. ``device="cpu"``
    keeps the stub backbone off a GPU the test node may have. Run it under
    :func:`stub_backbone`, which needs torch.

    Args:
        directory: Where to write ``dino_objects.pht-pipe.json``.

    Returns:
        The pipeline file.
    """
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureDinoEmbedding, MeasureSize

    pipeline = ImagePipeline(
        name="tiny-pipeline-dino",
        ops=[OtsuDetector()],
        meas={"size": MeasureSize(), "emb": MeasureDinoEmbedding(device="cpu")},
    )
    path = Path(directory) / "dino_objects.pht-pipe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pipeline.to_json() or "", encoding="utf-8")
    return path


#: The grid entry :func:`with_dino_objects_and_grid` adds, and its 2 x 2 grid.
DINO_GRID_KEY = "dinov2-vitb14/rgb/grid"


def with_dino_objects_and_grid(directory: Path) -> Path:
    """:func:`with_dino_objects` plus ``meas["grid"]``: a 2 x 2 grid embedder.

    Two entry keys, ``dinov2-vitb14/rgb/objects`` then ``dinov2-vitb14/rgb/grid``, so a
    finalization publishes two tables. The cells are
    :data:`TOY_GRID_ROW_EDGES` x :data:`TOY_GRID_COL_EDGES`.

    Returns:
        ``<directory>/dino_objects_grid.pht-pipe.json``.
    """
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.grid import ManualGridFinder
    from phenotypic.measure import MeasureDinoEmbedding, MeasureSize

    grid = ManualGridFinder(
        row_edges=list(TOY_GRID_ROW_EDGES), col_edges=list(TOY_GRID_COL_EDGES)
    )
    pipeline = ImagePipeline(
        name="tiny-pipeline-dino-grid",
        ops=[OtsuDetector()],
        meas={
            "size": MeasureSize(),
            "emb": MeasureDinoEmbedding(device="cpu"),
            "grid": MeasureDinoEmbedding(regions="grid", grid=grid, device="cpu"),
        },
    )
    path = Path(directory) / "dino_objects_grid.pht-pipe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pipeline.to_json() or "", encoding="utf-8")
    return path


def differentiate_plate_2(inputs: Path) -> None:
    """Replace ``ds/plate_2.tif`` under *inputs* with the plate mirrored left to right.

    The same colonies at other positions, so plate_2's crops, labels and
    vectors differ from plate_1's and a row attributed to the wrong store is
    visible (review T4).
    """
    import numpy as np
    from PIL import Image as PILImage

    path = Path(inputs) / "ds" / "plate_2.tif"
    pixels = np.asarray(PILImage.open(path))
    PILImage.fromarray(np.ascontiguousarray(pixels[:, ::-1])).save(path)


@pytest.fixture
def finished_tiny_run_with_embeddings(
    tmp_path: Path, tiny_inputs: Path, stub_backbone
) -> Path:
    """A finished local full run of :func:`with_dino_objects` over :func:`tiny_inputs`.

    ``plate_2`` is mirrored (:func:`differentiate_plate_2`), so the two
    stores' vectors differ. The backbone is :func:`stub_backbone`'s, so this
    needs torch. The pipeline file is ``<tmp>/dino_objects.pht-pipe.json``.

    Returns:
        The output root (``<tmp>/out``).
    """
    differentiate_plate_2(tiny_inputs)
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, pipeline=with_dino_objects(tmp_path))
    assert result.exit_code == 0, result.output
    return out


@pytest.fixture
def tree_with_one_stale_store(tmp_path: Path, tiny_inputs: Path, stub_backbone):
    """A finished :func:`with_dino_objects` run over three plates, one crop entry stale.

    ``plate_3``'s store is re-saved table-less with an all-zeros objmap
    (:func:`resave_with_objmap`), so its crop entry is recomputed
    ``stale: true`` (spec D7) while its folder is carried.

    Yields:
        ``(out, runs)``: the output root and each image's certified
        :class:`~phenotypic.sdk_.TableRun`, read from its record through
        :func:`~phenotypic._cli._cli_recompile_recovery.certified_table_run`.
    """
    from phenotypic._cli._cli_recompile_recovery import (
        certified_table_run,
        image_authority_path,
    )

    good = (tiny_inputs / "ds" / "plate_1.tif").read_bytes()
    (tiny_inputs / "ds" / "plate_3.tif").write_bytes(good)
    out = tmp_path / "out"
    result = run_cli(tiny_inputs, out, pipeline=with_dino_objects(tmp_path))
    assert result.exit_code == 0, result.output
    resave_with_objmap(one_store(out, stem="plate_3"), zeros=True)
    runs = []
    for stem in ("plate_1", "plate_2", "plate_3"):
        run = certified_table_run(out, image_authority_path(out, "ds", stem))
        assert run is not None, stem
        runs.append(run)
    yield out, runs


def rows_in(run: "TableRun") -> int:
    """The row count of *run*'s :data:`DINO_OBJECTS_KEY` table, from its footer."""
    import pyarrow.parquet as pq

    return pq.read_metadata(run.embedding_path(DINO_OBJECTS_KEY)).num_rows


# ---------------------------------------------------------------------------
# P5 T1, T2 (C20): process mode runs ``meas`` (spec E4)
# ---------------------------------------------------------------------------


def _write_pipeline_file(directory: Path, stem: str, pipeline: "ImagePipeline") -> Path:
    """Write *pipeline* to ``<directory>/<stem>-<sha8>.pht-pipe.json``.

    The name carries 8 hex of the bytes, so two different pipelines written
    into one directory never overwrite each other, and the same pipeline
    always lands on the same file (the journal records the file's basename,
    so a byte-reproducible store needs both).
    """
    import hashlib

    text = pipeline.to_json() or ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    path = Path(directory) / f"{stem}-{digest}.pht-pipe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def with_named_grids(directory: Path, make) -> Path:
    """A full-run pipeline file: ``OtsuDetector``, ``MeasureSize`` and two named grids.

    ``meas`` is ``size`` then :func:`named_grid_embedders` of *make*, so a
    finalization publishes ``<model>/rgb/grid-wells`` and ``<model>/rgb/grid-agar``
    (#281 named grid regions). The toy embedder runs under
    :func:`registered_toy_embedder`; :func:`grid_embedder` under
    :func:`stub_backbone`.

    Returns:
        ``<directory>/named_grids-<sha8>.pht-pipe.json``.
    """
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize

    pipeline = ImagePipeline(
        name="tiny-pipeline-named-grids",
        ops=[OtsuDetector()],
        meas={"size": MeasureSize(), **named_grid_embedders(make)},
    )
    return _write_pipeline_file(directory, "named_grids", pipeline)


def assert_named_grids_published(out: Path, model: str) -> None:
    """``<model>/rgb/grid-wells`` and ``<model>/rgb/grid-agar``: two store tables per
    image, two deliverables, both manifest keys, and each store's rows are its
    own table of its own grid. For :func:`with_named_grids` runs over
    :func:`tiny_inputs` with plate_2 mirrored."""
    import hashlib
    import json

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from phenotypic.schema import GRID
    from phenotypic.sdk_ import read_embedding_table

    deliverables = out / "deliverables"
    manifest = json.loads(
        (deliverables / "embeddings" / "manifest.json").read_text(encoding="utf-8")
    )
    keys = {f"{model}/rgb/grid-wells": 4, f"{model}/rgb/grid-agar": 1}
    assert set(manifest["entries"]) == set(keys)
    own_vectors = {}
    for key, cells in keys.items():
        entry = manifest["entries"][key]
        assert entry["path"] == f"embeddings/{key}.parquet"
        path = deliverables / entry["path"]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["stores"] == 2 and entry["rows"] == 2 * cells
        published = pq.read_table(path)
        for stem in ("plate_1", "plate_2"):
            store = one_store(out, stem=stem)
            own = read_embedding_table(store, key)
            assert own.num_rows == cells
            rows = published.filter(pc.equal(published["Metadata_ImageName"], stem))
            assert rows.select(own.column_names).equals(own), (key, stem)
            own_vectors[key, stem] = own.column("embedding")
        ends = published.column(str(GRID.ROW_INTERVAL_END)).to_pylist()
        assert set(ends) == ({100, 200} if cells == 4 else {200}), key
    for key in keys:
        assert not own_vectors[key, "plate_1"].equals(own_vectors[key, "plate_2"]), key
    metrics = json.loads((deliverables / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["embeddings"]) == set(keys)


def with_meas(directory: Path, /, **measurers) -> Path:
    """A pipeline file: ``ops=[BlurGauss(sigma=1.0)]`` and *measurers* in ``meas``.

    No detector, so the shape a process run can store: ops that keep the
    pixels' geometry, and measurers keyed by their keyword.

    Args:
        directory: Where to write the file.
        **measurers: ``meas`` entries, ``key=measurer``.

    Returns:
        ``<directory>/process_meas-<sha8>.pht-pipe.json``.
    """
    from phenotypic import ImagePipeline
    from phenotypic.enhance import BlurGauss

    pipeline = ImagePipeline(
        name="tiny-process-meas", ops=[BlurGauss(sigma=1.0)], meas=dict(measurers)
    )
    return _write_pipeline_file(directory, "process_meas", pipeline)


def grid_embedder(**kw):
    """``MeasureDinoEmbedding(regions="grid")`` over the 2 x 2 toy grid, on the CPU.

    The cells are :data:`TOY_GRID_ROW_EDGES` x :data:`TOY_GRID_COL_EDGES`,
    inside the synth plate. ``device="cpu"`` keeps a byte-equality test off a
    GPU the node may have. Building it needs no torch; running it needs
    :func:`stub_backbone`, which does.

    Args:
        **kw: Further ``MeasureDinoEmbedding`` fields, overriding the above.
    """
    from phenotypic.grid import ManualGridFinder
    from phenotypic.measure import MeasureDinoEmbedding

    fields = {
        "regions": "grid",
        "grid": ManualGridFinder(
            row_edges=list(TOY_GRID_ROW_EDGES), col_edges=list(TOY_GRID_COL_EDGES)
        ),
        "device": "cpu",
        **kw,
    }
    return MeasureDinoEmbedding(**fields)


def grid_process_pipeline(directory: Path, /, **kw) -> Path:
    """:func:`with_meas` with ``meas["grid_embedding"] = grid_embedder(**kw)``.

    The AutoConvertRaw-GC shape (spec E6): ops that keep geometry, then one
    grid embedding, stored under :data:`DINO_GRID_KEY`. Needs torch to run.
    """
    return with_meas(directory, grid_embedding=grid_embedder(**kw))


#: The model token of :func:`toy_grid_process_pipeline`'s embedder.
TOY_GRID_KEY = "toy-v1/rgb/grid"


@pytest.fixture
def registered_toy_embedder(monkeypatch):
    """Make :func:`toy_embedder_class` resolvable by ``ImagePipeline.from_json``.

    A pipeline file names an operation by class; one defined outside
    ``phenotypic`` resolves once it is attached to the ``phenotypic``
    namespace (what a ``PHENOTYPIC_PRELOAD_MODULES`` module does). Only an
    in-process CLI run (``run_cli``, ``--njobs 1``) sees the attribute, which
    is every test that uses it. Torch-free.

    Returns:
        The toy embedder class.
    """
    import phenotypic

    cls = toy_embedder_class()
    monkeypatch.setattr(phenotypic, cls.__name__, cls, raising=False)
    return cls


def toy_grid_embedder_of(**kw):
    """The toy embedder over the 2 x 2 toy grid; *kw* overrides its fields (``token``, ``grid``)."""
    from phenotypic.grid import ManualGridFinder

    fields = {
        "regions": "grid",
        "grid": ManualGridFinder(
            row_edges=list(TOY_GRID_ROW_EDGES), col_edges=list(TOY_GRID_COL_EDGES)
        ),
        **kw,
    }
    return toy_embedder_class()(**fields)


#: The one cell of the ``agar`` grid of :func:`named_grid_embedders`: the
#: 2 x 2 toy grid's outer box, as AutoConvertRaw-GC's whole-agar cell is its
#: well lattice's (spec E6).
TOY_AGAR_ROW_EDGES = (TOY_GRID_ROW_EDGES[0], TOY_GRID_ROW_EDGES[-1])
TOY_AGAR_COL_EDGES = (TOY_GRID_COL_EDGES[0], TOY_GRID_COL_EDGES[-1])


def named_grid_embedders(make, **kw) -> dict:
    """``{"wells": ..., "agar": ...}``: two named grid embedders of one model (#281).

    The rig1 shape of spec E6 at the toy scale: ``name="wells"`` over
    the 2 x 2 toy grid and ``name="agar"`` over its one outer cell.

    Args:
        make: :func:`toy_grid_embedder_of` or :func:`grid_embedder`.
        **kw: Further fields for both.
    """
    from phenotypic.grid import ManualGridFinder

    agar = ManualGridFinder(
        row_edges=list(TOY_AGAR_ROW_EDGES), col_edges=list(TOY_AGAR_COL_EDGES)
    )
    return {
        "wells": make(name="wells", **kw),
        "agar": make(name="agar", grid=agar, **kw),
    }


def toy_grid_process_pipeline(directory: Path, /, **kw) -> Path:
    """:func:`grid_process_pipeline` with the toy embedder: torch-free.

    ``meas["grid_embedding"]`` is the toy embedder over the 2 x 2 toy grid,
    stored under :data:`TOY_GRID_KEY` (or ``<token>/rgb/grid`` for another
    ``token=``). Run it under :func:`registered_toy_embedder`.

    Args:
        directory: Where to write the file.
        **kw: Toy embedder fields, e.g. ``token="toy-v2"`` for another
            pipeline with another model key.
    """
    return with_meas(directory, grid_embedding=toy_grid_embedder_of(**kw))


# ---------------------------------------------------------------------------
# P5 T3 (C21): in-place process runs (spec E7)
# ---------------------------------------------------------------------------

#: ``--layer rgb --process-format zarr``: the export an in-place run writes into.
ZARR_RGB = ("--layer", "rgb", "--process-format", "zarr")


def _process_store_tree(tmp_path: Path, tiny_inputs: Path, pipeline: Path) -> Path:
    """A copy process run's output tree, one ``1-<sha>`` folder per store.

    The shape AutoConvertRaw-GC publishes: rgb stores holding one grid
    embedding run. The copy run's machine state is removed, as a published
    tree has none (an AutoConvertRaw-GC store is written by calling the core
    directly).
    """
    import shutil

    out = tmp_path / "stores"
    result = run_cli(tiny_inputs, out, mode="process", pipeline=pipeline, extra=ZARR_RGB)
    assert result.exit_code == 0, result.output
    shutil.rmtree(out / ".phenotypic")
    return out


@pytest.fixture
def process_store_tree(tmp_path: Path, tiny_inputs: Path, stub_backbone) -> Path:
    """Two process stores, each holding :func:`grid_process_pipeline`'s ``1-<sha>``.

    ``<tmp>/stores/ds/plate_{1,2}.ome.zarr``; the DINO stub, so it needs torch.
    """
    return _process_store_tree(tmp_path, tiny_inputs, grid_process_pipeline(tmp_path))


@pytest.fixture
def toy_process_store_tree(tmp_path: Path, tiny_inputs: Path, registered_toy_embedder) -> Path:
    """:func:`process_store_tree` with :func:`toy_grid_process_pipeline`: torch-free."""
    return _process_store_tree(tmp_path, tiny_inputs, toy_grid_process_pipeline(tmp_path))


# ---------------------------------------------------------------------------
# #282 P3 T1 (C7): ``replace_run_embeddings``, the Stage-4 store transaction
# ---------------------------------------------------------------------------


def _metadata_tables_of_a(store: Path, directory: Path):
    """Pipeline A's tables for *store*'s own image, with a metadata table.

    The metadata snapshot shares ``Object_Label`` with the measurements, so
    :func:`~phenotypic._cli._embedded_measurement_tables.prepare_image_tables`
    reaches its joined arm and the run gets a ``pht-metadata.parquet``.
    """
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    frame = _build_tiny_pipeline().measure(_load_store(store))
    frame["Pipeline_Marker"] = "A"
    snapshot = Path(directory) / "metadata.csv"
    snapshot.write_text("Object_Label,Metadata_Strain\n1,WT\n2,MUT\n", encoding="utf-8")
    tables = prepare_image_tables(frame, snapshot)
    assert tables.metadata is not None, "the snapshot must join, or the run gets no metadata"
    return tables


@pytest.fixture
def staged_store_with_tables(tmp_path: Path, measured_image, pipeline_file_a: Path,
                             pipeline_file_b: Path) -> tuple[Path, str]:
    """A promoted v2 store as Stage 3 leaves it, and the run Stage 4 embeds into.

    Built with today's writers, never by hand:

    - ``save2zarr`` of :func:`measured_image` with A's tables and an
      ``upstream`` block (:func:`~phenotypic.sdk_.upstream_entries_for_input`
      of an input store holding A's run), so the run is ``2-<A>``;
    - :func:`resave_with_pipeline` with B, which adds folder ``3-<B>``;
    - :func:`~phenotypic.sdk_.replace_image_tables` of A's run with a
      metadata table.

    No run holds embeddings.

    Returns:
        ``(store, run_id)``: ``run_id`` is A's folder, holding a measurement
        and a metadata table.
    """
    import hashlib

    from phenotypic.sdk_ import list_table_runs, replace_image_tables, upstream_entries_for_input

    _close_open_application(measured_image)
    source = measured_image.save2zarr(
        tmp_path / "in" / "ds" / "source.ome.zarr",
        measurement_table=tables_for(measured_image),
    )
    upstream = upstream_entries_for_input(source, tmp_path / "in")
    assert upstream, "the input store must give the staged store an upstream entry"
    store = measured_image.save2zarr(
        tmp_path / "out" / "staged.ome.zarr",
        measurement_table=tables_for(measured_image),
        upstream=upstream,
    )
    resave_with_pipeline(store, pipeline_file_b)
    sha_a = hashlib.sha256(Path(pipeline_file_a).read_bytes()).hexdigest()
    run = replace_image_tables(
        store, _metadata_tables_of_a(store, tmp_path), pipeline_sha256=sha_a
    )
    runs = list_table_runs(store)
    assert len(runs) == 2 and run.run_id in {r.run_id for r in runs}
    assert run.metadata_path().is_file() and run.measurements_path().is_file()
    assert not any(r.descriptor.get("embeddings") for r in runs)
    return store, run.run_id


def objects_payload_for(store: Path) -> dict:
    """A ``toy-v1/rgb/objects`` payload computed from *store*'s own image.

    Its ``binds.objmap_sha256`` is :func:`~phenotypic.sdk_._measurement_tables.objmap_digest`
    of the store's level-0 objmap, so a write that passes the store's own
    digests records it ``stale: false``.

    Returns:
        ``{"toy-v1/rgb/objects": EmbeddingPayload}``.
    """
    from phenotypic import ImagePipeline

    image = _load_store(store)
    _close_open_application(image)
    payloads = ImagePipeline(meas={"emb": toy_embedder_class()()}).embed(image)
    assert set(payloads) == {"toy-v1/rgb/objects"}, sorted(payloads)
    return payloads


def root_sha256(store: Path) -> str:
    """sha256 of *store*'s root ``zarr.json`` bytes."""
    import hashlib

    return hashlib.sha256((Path(store) / "zarr.json").read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# #282 P4: Stage-4 fixtures (local Stage 4, refusals, faults, crashes)
# ---------------------------------------------------------------------------

#: Where :func:`plant_stage4_fault` writes its triggers; set by :func:`stage4_faults`.
STAGE4_FAULT_DIR_ENV = "PHENOTYPIC_TEST_FAULT_DIR"

#: The self-registering test modules a fresh process imports to read P4's pipelines.
STAGE4_PRELOAD_MODULES = (
    "tests._fakes.register_fake_gpu",
    "tests._fakes.register_faulty_dino_embedder",
)


def with_stage4_dino_objects(
    directory: Path,
    *,
    fake_gpu: bool = False,
    grid: bool = False,
    faulty: bool = False,
    batch_size: int | None = None,
) -> Path:
    """A pipeline file like :func:`with_dino_objects`, with ``device="auto"`` (#282 P4).

    ``meas["emb"]`` is ``MeasureDinoEmbedding()`` with ``device="auto"``, which
    U9 lets resolve to Stage 4 (a ``device="cpu"`` pin resolves inline). Run
    it under :func:`stage4_cpu`, so ``auto`` is the CPU on a GPU node too.

    Args:
        directory: Where to write the file.
        fake_gpu: The detector is ``FakeGpuDetector(threshold=0.3)`` instead of
            ``OtsuDetector``, so the run is staged. Its registering module is
            imported here; a fresh process needs
            ``PHENOTYPIC_PRELOAD_MODULES`` (:data:`STAGE4_PRELOAD_MODULES`,
            which :func:`run_cli_subprocess` sets).
        grid: Add ``meas["grid"]``, a 2 x 2 ``ManualGridFinder`` embedder on
            the same backbone (:data:`DINO_GRID_KEY`).
        faulty: ``meas["emb"]`` is ``FaultyDinoEmbedding``, which fails an
            image planted with :func:`plant_stage4_fault`.
        batch_size: Every embedder's ``batch_size``; :data:`ONE_PASS_BATCH`
            makes each side of an inline/Stage-4 comparison run one pass
            (spec §5). ``None`` keeps the default.

    Returns:
        ``<directory>/stage4_dino[_gpu][_grid][_faulty][_b<n>].pht-pipe.json``.
    """
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.grid import ManualGridFinder
    from phenotypic.measure import MeasureDinoEmbedding, MeasureSize

    if fake_gpu:
        import tests._fakes.register_fake_gpu  # noqa: F401
        from tests._fakes.fake_gpu_detector import FakeGpuDetector

        detector = FakeGpuDetector(threshold=0.3)
    else:
        detector = OtsuDetector()
    if faulty:
        import tests._fakes.register_faulty_dino_embedder  # noqa: F401
        from tests._fakes.faulty_dino_embedder import FaultyDinoEmbedding

        embedder_class = FaultyDinoEmbedding
    else:
        embedder_class = MeasureDinoEmbedding
    sized = {} if batch_size is None else {"batch_size": batch_size}
    meas: dict = {"size": MeasureSize(), "emb": embedder_class(**sized)}
    if grid:
        meas["grid"] = MeasureDinoEmbedding(
            regions="grid",
            grid=ManualGridFinder(
                row_edges=list(TOY_GRID_ROW_EDGES), col_edges=list(TOY_GRID_COL_EDGES)
            ),
            **sized,
        )
    stem = "stage4_dino" + "".join(
        suffix for suffix, on in (("_gpu", fake_gpu), ("_grid", grid), ("_faulty", faulty)) if on
    ) + ("" if batch_size is None else f"_b{batch_size}")
    pipeline = ImagePipeline(name=stem.replace("_", "-"), ops=[detector], meas=meas)
    path = Path(directory) / f"{stem}.pht-pipe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pipeline.to_json() or "", encoding="utf-8")
    return path


@pytest.fixture
def stage4_faults(monkeypatch, tmp_path: Path) -> Path:
    """Point ``$PHENOTYPIC_TEST_FAULT_DIR`` at an empty directory; return it."""
    fault_dir = tmp_path / "stage4_faults"
    fault_dir.mkdir()
    monkeypatch.setenv(STAGE4_FAULT_DIR_ENV, str(fault_dir))
    return fault_dir


def plant_stage4_fault(stem: str) -> None:
    """Make ``FaultyDinoEmbedding.prepare_regions`` fail image *stem* (needs :func:`stage4_faults`)."""
    import os

    (Path(os.environ[STAGE4_FAULT_DIR_ENV]) / stem).write_text("fault", encoding="utf-8")


def clear_stage4_fault(stem: str) -> None:
    """Remove :func:`plant_stage4_fault`'s trigger for *stem*."""
    import os

    (Path(os.environ[STAGE4_FAULT_DIR_ENV]) / stem).unlink(missing_ok=True)


def nth_stem(inputs: Path, index: int) -> str:
    """The stem of the *index*-th input file under *inputs*, in sorted path order."""
    return sorted(path for path in Path(inputs).rglob("*") if path.is_file())[index].stem


@pytest.fixture
def stage4_cpu(monkeypatch, param_stub_backbone):
    """Run Stage 4 and ``device="auto"`` on the CPU, on any node; yield the stub's namespace.

    Sets ``PHENOTYPIC_STAGE4_DEVICE=cpu`` and ``CUDA_VISIBLE_DEVICES=""``, and
    patches ``torch.cuda.is_available`` to ``False``: the variable alone does
    nothing once the process has initialised CUDA (#282 re-review item 5).
    Needs torch; skips without it.
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("PHENOTYPIC_STAGE4_DEVICE", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield param_stub_backbone


#: The worker's four durable writes, in the order one image goes through
#: them (spec §4.2): the signal, the store, the record, the signals consumed.
STAGE4_DURABLE_WRITES = (
    "write_stage4_signal",
    "replace_run_embeddings",
    "publish_image_success",
    "delete_stage4_signals",
)

#: One CLI run in a fresh interpreter on the parametric stub, CUDA hidden.
#: ``argv[1]`` is a JSON object: inputs, out, pipeline, extra, crash_after.
_STAGE4_SUBPROCESS_RUN = """
import json, os, sys, traceback
from pathlib import Path
import pytest
import torch
torch.cuda.is_available = lambda: False
from tests._tiny_run import STAGE4_DURABLE_WRITES, install_stub_backbone, run_cli
args = json.loads(sys.argv[1])
install_stub_backbone(pytest.MonkeyPatch(), parametric=True)
if args.get("spy_stem"):
    from phenotypic._cli import _cli_stage4_worker as worker
    def _stem_of(name, a, k):
        if name == "replace_run_embeddings":
            return Path(a[0]).name.removesuffix(".ome.zarr")
        if name == "publish_image_success":
            return k["image_stem"]
        return a[2]  # write_stage4_signal / delete_stage4_signals(output_dir, dataset, stem)
    def _spying(name, original):
        def _spy(*a, **k):
            result = original(*a, **k)
            if _stem_of(name, a, k) == args["spy_stem"]:
                with open(args["spy_path"], "a", encoding="utf-8") as log:
                    log.write(name + "\\n")
            return result
        return _spy
    for name in STAGE4_DURABLE_WRITES:
        setattr(worker, name, _spying(name, getattr(worker, name)))
if args["crash_after"]:
    from phenotypic._cli import _cli_stage4_worker as worker
    original = getattr(worker, args["crash_after"])
    def _crash_after(*a, **k):
        result = original(*a, **k)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(137)
    setattr(worker, args["crash_after"], _crash_after)
result = run_cli(Path(args["inputs"]), Path(args["out"]), pipeline=Path(args["pipeline"]),
                 extra=args["extra"])
sys.stdout.write(result.output)
if result.exception is not None and not isinstance(result.exception, SystemExit):
    traceback.print_exception(result.exception)
sys.exit(result.exit_code)
"""


def run_cli_subprocess(
    inputs: Path,
    out: Path,
    *,
    pipeline: Path,
    extra: Sequence[str] = (),
    crash_after: str | None = None,
    spy_stem: str | None = None,
    spy_path: Path | None = None,
):
    """One :func:`run_cli` in a fresh interpreter, for crash tests (#282 plan review I5).

    The child installs the parametric stub backbone, sets
    ``PHENOTYPIC_STAGE4_DEVICE=cpu`` and ``CUDA_VISIBLE_DEVICES=""``, and
    patches ``torch.cuda.is_available`` to ``False`` before anything touches
    CUDA; ``PYTHONPATH`` reaches ``src/`` and the repo root, and
    ``PHENOTYPIC_PRELOAD_MODULES`` gains :data:`STAGE4_PRELOAD_MODULES`. The
    caller's ``$PHENOTYPIC_TEST_FAULT_DIR`` is inherited.

    Args:
        inputs: The ``--input`` root.
        out: The ``--output`` root.
        pipeline: The ``--pipeline`` file.
        extra: Further CLI arguments.
        crash_after: A function of ``phenotypic._cli._cli_stage4_worker``.
            Right after its first call returns, the child calls
            ``os._exit(137)``: a real crash between two durable writes, not
            an exception a caller could catch.
        spy_stem: With *spy_path*: append the name of each
            :data:`STAGE4_DURABLE_WRITES` call that concerns this stem to
            *spy_path*, one per line, as it returns
            (:func:`run_cli_subprocess_spying`).
        spy_path: The file the spy appends to.

    Returns:
        The :class:`subprocess.CompletedProcess`, text mode, output captured.
    """
    import json
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root), *filter(None, [env.get("PYTHONPATH")])]
    )
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PHENOTYPIC_STAGE4_DEVICE"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PHENOTYPIC_PRELOAD_MODULES"] = ",".join(
        [*filter(None, env.get("PHENOTYPIC_PRELOAD_MODULES", "").split(",")), *STAGE4_PRELOAD_MODULES]
    )
    args = {
        "inputs": str(inputs),
        "out": str(out),
        "pipeline": str(pipeline),
        "extra": list(extra),
        "crash_after": crash_after,
        "spy_stem": spy_stem,
        "spy_path": None if spy_path is None else str(spy_path),
    }
    return subprocess.run(
        [sys.executable, "-c", _STAGE4_SUBPROCESS_RUN, json.dumps(args)],
        cwd=root, env=env, capture_output=True, text=True, timeout=900,
    )


# ---------------------------------------------------------------------------
# #282 P4 T2: deferral helpers
# ---------------------------------------------------------------------------

#: The modules whose bindings :func:`count_calls` wraps: every caller of the
#: per-image cores and of ``publish_image_success`` on a local run.
_COUNTED_MODULES = (
    "phenotypic.phenotypicCLI",
    "phenotypic._cli._cli_completion",
    "phenotypic._cli._cli_execution_strategies",
    "phenotypic._cli._cli_process_single",
    "phenotypic._cli._cli_staged_slurm_worker",
    "phenotypic._cli._cli_staged_strategy",
    "phenotypic._cli._cli_staged_workers",
)


def count_calls(monkeypatch, names: Sequence[str]) -> dict[str, int]:
    """Count the calls of each named function, wherever a CLI module binds it.

    Each name is wrapped in every module of :data:`_COUNTED_MODULES` that
    binds it (a ``from x import f`` copies the binding, so patching only the
    defining module misses its callers). The returned dict is live: read it
    after the run.

    Args:
        monkeypatch: pytest's ``monkeypatch``; the wrappers are undone with it.
        names: Function names, e.g. ``"stage3_merge_measure_core"``.
    """
    import functools
    import importlib

    counts = {name: 0 for name in names}
    for module_name in _COUNTED_MODULES:
        module = importlib.import_module(module_name)
        for name in names:
            original = module.__dict__.get(name)
            if not callable(original):
                continue

            def _counting(*args, _name=name, _original=original, **kwargs):
                counts[_name] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(module, name, functools.wraps(original)(_counting))
    return counts


def store_root_bytes(out: Path) -> dict[Path, bytes]:
    """Every per-image store's root ``zarr.json`` bytes, by path."""
    return {
        path: path.read_bytes()
        for path in sorted(Path(out).glob("results/*/zarr/*.ome.zarr/zarr.json"))
    }


def dataset_and_stem(store: Path) -> tuple[str, str]:
    """``(dataset, stem)`` of a store at ``results/<ds>/zarr/<stem>.ome.zarr``."""
    return store.parent.parent.name, store.name.removesuffix(".ome.zarr")


class Stage4Item:
    """One image of a run, as the deferral tests address it (#282 P4 T2)."""

    def __init__(self, dataset: str, image_name: str, work_id: str) -> None:
        self.dataset = dataset
        self.image_name = image_name
        self.stem = Path(image_name).stem
        self.work_id = work_id

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Stage4Item({self.dataset}/{self.image_name})"


def run_items(out: Path) -> list[Stage4Item]:
    """Every image of the run at *out*, with its recorded work id, in name order."""
    import json

    from phenotypic.sdk_ import processing_state_path

    state = json.loads(processing_state_path(out).read_text(encoding="utf-8"))
    return [
        Stage4Item(dataset, image_name, work_id)
        for dataset, images in sorted(state["config"]["work_ids"].items())
        for image_name, work_id in sorted(images.items())
    ]


def invalidate_store(out: Path, item: Stage4Item) -> None:
    """Rebind *item*'s store root to another work id, so its core must re-run.

    The record keeps ``stages.stage3`` and the store keeps its table; only
    ``staged_store_matches_work_id`` stops holding, so ``stages_1_to_3_done``
    is false and every continuation re-runs the image's core.
    """
    import json

    root = zarr_store_path_of(out, item) / "zarr.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["attributes"]["phenotypic"]["work_id"] = "invalidated-" + item.work_id
    root.write_text(json.dumps(payload), encoding="utf-8")


def zarr_store_path_of(out: Path, item: Stage4Item) -> Path:
    """*item*'s per-image store."""
    from phenotypic.sdk_ import zarr_store_path

    return zarr_store_path(out, item.dataset, item.stem)


#: Two keys a planted signal goes under: the pipeline's own, and one it no
#: longer has (``delete_stage4_signals`` walks directories, not keys).
PLANTED_STAGE4_KEYS = ("dinov2-vitb14/rgb/objects", "dinov2-vits14/gray/grid")


def plant_stage4_signals(out: Path, item: Stage4Item) -> list[Path]:
    """Write placeholder signal files for *item* under :data:`PLANTED_STAGE4_KEYS`."""
    from phenotypic._cli._cli_stage4_signal import stage4_signal_path

    paths = []
    for key in PLANTED_STAGE4_KEYS:
        path = stage4_signal_path(out, item.dataset, key, item.stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"planted")
        paths.append(path)
    return paths


def signal_files(out: Path, item: Stage4Item) -> list[Path]:
    """*item*'s Stage-4 signal files, under any key."""
    from phenotypic.sdk_ import progress_dir

    base = progress_dir(out) / "stage4_payload" / item.dataset
    return sorted(base.glob(f"*/*/*/{item.stem}.parquet"))


def publish_waiting_records(out: Path) -> int:
    """Publish the record of every image that finished Stages 1-3, as Stage 4 would.

    Stands in for Stage 4's publication (P4 T3) in tests that need a
    published deferral run before Stage 4 exists: the same artifacts as
    ``_publish_local_image_success``, with no embeddings committed.

    Returns:
        How many records were published.
    """
    import json
    from uuid import uuid4

    from phenotypic._cli._cli_completion import (
        image_data_artifact,
        image_measurement_artifact,
        publish_image_success,
    )
    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic._cli._cli_staged_resume import stages_1_to_3_done
    from phenotypic.sdk_ import processing_state_path

    state = json.loads(processing_state_path(out).read_text(encoding="utf-8"))
    manager = OutputManager.from_config(base_dir=out, ext=".tiff")
    published = 0
    for item in run_items(out):
        if not stages_1_to_3_done(out, item.dataset, item.stem, item.work_id):
            continue
        data_key, data_path = image_data_artifact(out, manager, item.dataset, item.stem)
        artifacts = {
            "measurements": image_measurement_artifact(out, item.dataset, item.stem),
            data_key: data_path,
        }
        overlay = manager.get_output_path(item.dataset, "overlays", item.stem)
        if Path(overlay).is_file():
            artifacts["overlay"] = overlay
        publish_image_success(
            out,
            work_id=item.work_id,
            dataset=item.dataset,
            relative_image_path=f"{item.dataset}/{item.image_name}",
            image_stem=item.stem,
            mode="full",
            attempt_id=uuid4().hex,
            lifecycle_epoch=state["config"].get("processing_generation") or "local-unfenced",
            artifacts=artifacts,
        )
        published += 1
    return published


@pytest.fixture
def no_stage4(monkeypatch):
    """Stage 4 does nothing: ``run_local_stage4`` returns ``{}`` (#282 P4 T2).

    Every image of a ``--embed-stage gpu`` run then stops after Stages 1-3,
    waiting on Stage 4. Before P4 T3 nothing calls it; the patch keeps these
    tests' meaning once T3 wires it in.
    """
    from phenotypic._cli import _cli_stage4_worker

    monkeypatch.setattr(_cli_stage4_worker, "run_local_stage4", lambda *a, **k: {})


class Stage4Waiting:
    """A finished-but-for-Stage-4 run: every image passed Stages 1-3 and waits."""

    def __init__(self, inputs: Path, out: Path, pipeline: Path) -> None:
        self.inputs = inputs
        self.out = out
        self.pipeline = pipeline
        self.items = run_items(out)
        self.cli_args = (inputs, out)
        self.cli_kwargs = {"pipeline": pipeline}

    def __iter__(self):
        """``out, item = fixture``: the output root and the first image."""
        return iter((self.out, self.items[0]))

    def rerun(self, *extra: str):
        """The same ``--embed-stage gpu`` command again, with *extra* appended."""
        return run_cli(*self.cli_args, **self.cli_kwargs, extra=["--embed-stage", "gpu", *extra])

    def manifest(self) -> list:
        """The run's images as staged SLURM manifest entries."""
        from phenotypic._cli._cli_staged_orchestration import StagedManifestEntry

        root = Path(self.inputs)
        return [
            StagedManifestEntry(
                dataset=item.dataset,
                image_name=item.image_name,
                stem=item.stem,
                input_path=str((root / item.dataset / item.image_name).absolute()),
                work_id=item.work_id,
                relative_image_path=f"{item.dataset}/{item.image_name}",
            )
            for item in self.items
        ]

    def stage3_kwargs(self, index: int, *, resume: bool = True) -> dict:
        """``run_stage3_step``'s arguments for image *index*, unfenced."""
        return {
            "pipeline_path": self.pipeline,
            "output_dir": self.out,
            "image_type": "Image",
            "manifest": self.manifest(),
            "index": index,
            "resume": resume,
        }


def record_embed_stage_in_job_metadata(out: Path, embed_stage: str) -> None:
    """Record ``EMBED_STAGE`` in *out*'s ``job_metadata.json``, as a staged submitter does."""
    import json

    from phenotypic.sdk_ import JobMetadataKey, atomic_write_json, job_metadata_path

    path = job_metadata_path(out)
    metadata = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    metadata[JobMetadataKey.EMBED_STAGE] = embed_stage
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, metadata)


def _waiting_run(tmp_path: Path, inputs: Path, *, fake_gpu: bool) -> Stage4Waiting:
    out = tmp_path / "out"
    pipeline = with_stage4_dino_objects(tmp_path, fake_gpu=fake_gpu)
    run_cli(inputs, out, pipeline=pipeline, extra=["--embed-stage", "gpu"])
    return Stage4Waiting(inputs, out, pipeline)


@pytest.fixture
def stage4_slurm_waiting(tmp_path: Path, tiny_inputs: Path, stage4_cpu, no_stage4) -> Stage4Waiting:
    """A staged (``FakeGpuDetector``) tree whose images all wait on Stage 4, as a SLURM worker sees it.

    Built by a local ``--embed-stage gpu`` run with Stage 4 stubbed out, then
    ``EMBED_STAGE = "gpu"`` recorded in ``job_metadata.json``, where the
    staged SLURM worker reads deferral.
    """
    waiting = _waiting_run(tmp_path, tiny_inputs, fake_gpu=True)
    record_embed_stage_in_job_metadata(waiting.out, "gpu")
    return waiting


@pytest.fixture
def stage4_pending_with_signals(tmp_path: Path, tiny_inputs: Path, stage4_cpu, no_stage4) -> Stage4Waiting:
    """A single-pass tree whose images all wait on Stage 4, each with planted signals.

    ``out, item = fixture`` gives the output root and the first image;
    ``fixture.items[1]`` is another pending image with signals of its own.
    """
    waiting = _waiting_run(tmp_path, tiny_inputs, fake_gpu=False)
    for item in waiting.items:
        plant_stage4_signals(waiting.out, item)
    return waiting


# ---------------------------------------------------------------------------
# #282 P4 T3: Stage-4 worker fixtures
# ---------------------------------------------------------------------------


def _stage4_pipeline_file(directory: Path, stem: str, meas: dict, *, fake_gpu: bool = True) -> Path:
    """A pipeline file of ``FakeGpuDetector`` (or Otsu) plus ``MeasureSize`` and *meas*."""
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize

    if fake_gpu:
        import tests._fakes.register_fake_gpu  # noqa: F401
        from tests._fakes.fake_gpu_detector import FakeGpuDetector

        detector = FakeGpuDetector(threshold=0.3)
    else:
        detector = OtsuDetector()
    pipeline = ImagePipeline(
        name=stem.replace("_", "-"), ops=[detector], meas={"size": MeasureSize(), **meas}
    )
    return _write_pipeline_file(directory, stem, pipeline)


def _built_staged_tree(tmp_path: Path, inputs: Path, pipeline: Path) -> Stage4Waiting:
    """A local ``--embed-stage gpu`` run of *pipeline* stopped before Stage 4.

    ``run_local_stage4`` is stubbed only while the tree is built, so a test
    can still run the real one over it.
    """
    from phenotypic._cli import _cli_stage4_worker

    out = tmp_path / "out"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_cli_stage4_worker, "run_local_stage4", lambda *a, **k: {})
        result = run_cli(inputs, out, pipeline=pipeline, extra=["--embed-stage", "gpu"])
    assert "Traceback" not in result.output, result.output
    tree = Stage4Waiting(inputs, out, pipeline)
    assert len(tree.items) == 2 and all(
        stage4_status_of(tree, item) == "pending" for item in tree.items
    ), result.output
    return tree


def stage4_status_of(tree: Stage4Waiting, item: Stage4Item) -> str:
    """``stage4_status`` of one of *tree*'s images."""
    from phenotypic._cli._cli_staged_resume import stage4_status

    return stage4_status(tree.out, item.dataset, item.stem, item.work_id)


@pytest.fixture
def staged_tree(tmp_path: Path, tiny_inputs: Path, stage4_cpu) -> Stage4Waiting:
    """A staged (``FakeGpuDetector``) run whose two images wait on Stage 4: one DINOv2-base objects key."""
    return _built_staged_tree(tmp_path, tiny_inputs, with_stage4_dino_objects(tmp_path, fake_gpu=True))


@pytest.fixture
def staged_tree_two_backbones(tmp_path: Path, tiny_inputs: Path, stage4_cpu) -> Stage4Waiting:
    """:func:`staged_tree` with a second, DINOv2-small, objects embedder: two loads."""
    from phenotypic.measure import MeasureDinoEmbedding

    pipeline = _stage4_pipeline_file(
        tmp_path, "stage4_two_backbones",
        {"emb": MeasureDinoEmbedding(), "small": MeasureDinoEmbedding(size="small")},
    )
    return _built_staged_tree(tmp_path, tiny_inputs, pipeline)


@pytest.fixture
def staged_tree_objects_and_grid(tmp_path: Path, tiny_inputs: Path, stage4_cpu) -> Stage4Waiting:
    """:func:`staged_tree` with the 2 x 2 grid embedder on the same backbone: one load."""
    return _built_staged_tree(
        tmp_path, tiny_inputs, with_stage4_dino_objects(tmp_path, fake_gpu=True, grid=True)
    )


@pytest.fixture
def staged_tree_faulty(tmp_path: Path, tiny_inputs: Path, stage4_cpu, stage4_faults) -> Stage4Waiting:
    """:func:`staged_tree` whose objects embedder is ``FaultyDinoEmbedding``."""
    return _built_staged_tree(
        tmp_path, tiny_inputs, with_stage4_dino_objects(tmp_path, fake_gpu=True, faulty=True)
    )


@pytest.fixture
def staged_tree_dense(tmp_path: Path, tiny_inputs: Path, stage4_cpu) -> Stage4Waiting:
    """:func:`staged_tree` with a ``masked_dense`` objects embedder (``crop_px=112``)."""
    from phenotypic.measure import MeasureDinoEmbedding

    pipeline = _stage4_pipeline_file(
        tmp_path, "stage4_dense", {"emb": MeasureDinoEmbedding(pooling="masked_dense", crop_px=112)}
    )
    return _built_staged_tree(tmp_path, tiny_inputs, pipeline)


@pytest.fixture
def staged_tree_one_empty_plate(staged_tree: Stage4Waiting) -> Stage4Waiting:
    """:func:`staged_tree` whose first store's level-0 objmap is all zeros.

    A plate with no colonies fails Stage 3 (``measure()`` raises
    ``NoObjectsError``), so the CLI never hands one to Stage 4. This stands in
    for an objects embedder over an empty objmap at Stage 4 (README Review
    Focus 1): the store still finished Stages 1-3, and its objmap holds no
    object. ``fixture.empty`` is that image.
    """
    import zarr

    from phenotypic.sdk_ import ngff_

    item = staged_tree.items[0]
    store = zarr_store_path_of(staged_tree.out, item)
    labels = ngff_.read_phenotypic_attributes(store)[ngff_.PhenotypicAttr.LABELS]
    array = zarr.open_group(str(store), mode="r+")[f"{labels[ngff_.OBJMAP_LABEL]}/0"]
    array[...] = 0
    staged_tree.empty = item
    return staged_tree


@pytest.fixture
def staged_tree_dinov3(staged_tree: Stage4Waiting, tmp_path: Path) -> Stage4Waiting:
    """:func:`staged_tree` whose Stage 4 reads a DINOv3-base pipeline.

    The licence refusal comes before Stage 4 loads or reads anything, so the
    tree's own stores (built with DINOv2, which needs no licence or weights)
    serve it.
    """
    from phenotypic.measure import MeasureDinoEmbedding

    staged_tree.pipeline = _stage4_pipeline_file(
        tmp_path, "stage4_dinov3", {"emb": MeasureDinoEmbedding(version=3, size="base")}
    )
    return staged_tree


def stage4_epoch(tree: Stage4Waiting) -> str | None:
    """The local lifecycle epoch *tree*'s records are stamped with."""
    import json

    from phenotypic.sdk_ import processing_state_path

    state = json.loads(processing_state_path(tree.out).read_text(encoding="utf-8"))
    return state["config"].get("processing_generation")


def stage4_image_type(tree: Stage4Waiting) -> str:
    """The image class *tree*'s stores were saved as (the run's ``image_type``)."""
    from phenotypic.sdk_ import ngff_

    store = zarr_store_path_of(tree.out, tree.items[0])
    return ngff_.read_phenotypic_attributes(store).get(ngff_.PhenotypicAttr.IMAGE_CLASS, "Image")


def run_stage4_over(tree: Stage4Waiting, *, num_workers: int = 0, **kwargs):
    """``run_stage4`` over every image of *tree*, as a local Stage 4 hands them to it."""
    from phenotypic._cli._cli_output_manager import OutputManager
    from phenotypic._cli._cli_stage4_worker import run_stage4

    return run_stage4(
        tree.pipeline,
        tree.out,
        tree.manifest(),
        image_type=stage4_image_type(tree),
        epoch=stage4_epoch(tree),
        num_workers=num_workers,
        output_manager=OutputManager.from_config(base_dir=tree.out, ext=".tiff"),
        **kwargs,
    )


def stores_of(tree: Stage4Waiting) -> list[Path]:
    """*tree*'s stores, in image order."""
    return [zarr_store_path_of(tree.out, item) for item in tree.items]


def identity_of(tree: Stage4Waiting, index: int) -> str:
    """``StagedManifestEntry.identity`` of *tree*'s image *index*."""
    return tree.manifest()[index].identity


def n_images(tree: Stage4Waiting) -> int:
    """How many images *tree* has."""
    return len(tree.items)


def embedding_path(store: Path, key: str) -> Path:
    """*key*'s embedding table in *store*'s journal run."""
    from phenotypic.sdk_ import journal_table_run

    run = journal_table_run(store)
    assert run is not None
    return run.embedding_path(key)


def valid_record(tree: Stage4Waiting, item: Stage4Item) -> bool:
    """Whether *item*'s record is valid."""
    from phenotypic._cli._cli_completion import valid_image_success

    return valid_image_success(tree.out, dataset=item.dataset, image_stem=item.stem, work_id=item.work_id)


def terminal_failures(tree: Stage4Waiting) -> list:
    """*tree*'s terminal failure records."""
    from phenotypic._cli._cli_failure_tracker import read_terminal_failures

    return read_terminal_failures(tree.out)


def failure_stage(tree: Stage4Waiting, item: Stage4Item) -> str | None:
    """The ``failed_stage`` of *item*'s terminal record under its work id, or ``None``."""
    stages = [r.failed_stage for r in terminal_failures(tree) if r.work_id == item.work_id]
    return stages[-1] if stages else None


def stage4_payloads_of(tree: Stage4Waiting, item: Stage4Item) -> dict:
    """*item*'s payloads for every embedder of *tree*'s pipeline, computed inline on its store."""
    from phenotypic import ImagePipeline

    image = _load_store(zarr_store_path_of(tree.out, item))
    image.name = item.stem
    return ImagePipeline.from_json(tree.pipeline).embed(image)


def write_valid_signals(
    tree: Stage4Waiting,
    *,
    keys: Sequence[str] | None = None,
    root_sha256: str | None = None,
    binds_from_another_objmap: bool = False,
) -> int:
    """Write a Stage-4 signal for each image and key, as an interrupted Stage 4 would have.

    The payloads are computed inline from each store. By default each signal
    is bound to the live root, so it is valid; *root_sha256* binds them to
    another root (a signal from before a reprocess), and
    *binds_from_another_objmap* rebinds each objects payload to an objmap the
    store does not hold.

    Returns:
        How many signals were written.
    """
    from phenotypic._cli._cli_stage4_signal import write_stage4_signal
    from phenotypic.sdk_ import journal_table_run
    from phenotypic.sdk_._embedding_tables import EmbeddingPayload

    written = 0
    for item in tree.items:
        store = zarr_store_path_of(tree.out, item)
        run = journal_table_run(store)
        assert run is not None
        for key, payload in stage4_payloads_of(tree, item).items():
            if keys is not None and key not in keys:
                continue
            if binds_from_another_objmap:
                descriptor = dict(payload.descriptor)
                descriptor["binds"] = {**descriptor["binds"], "objmap_sha256": "f" * 64}
                payload = EmbeddingPayload(key=payload.key, frame=payload.frame, descriptor=descriptor)
            write_stage4_signal(
                tree.out, item.dataset, item.stem, payload,
                work_id=item.work_id, run_id=run.run_id,
                root_sha256=root_sha256 or root_sha256_of_store(store),
            )
            written += 1
    return written


def root_sha256_of_store(store: Path) -> str:
    """sha256 of *store*'s root ``zarr.json`` (alias of :func:`root_sha256`)."""
    return root_sha256(store)


def touch_root_before_commit(monkeypatch, tree: Stage4Waiting | None, *, image: int = 0) -> list[Path]:
    """Change a store's root bytes right before Stage 4 commits it (a concurrent reprocess).

    Wraps the worker's ``replace_run_embeddings``: its first call (*image* 0;
    a higher *image* counts that many calls) touches the store's root, so the
    commit meets ``StoreRootChangedError``. *tree* is unused by the wrapper
    and may be ``None`` (inside a CLI run).

    Returns:
        The stores whose root was touched, filled as it happens.
    """
    from phenotypic._cli import _cli_stage4_worker as worker

    original = worker.replace_run_embeddings
    touched: list[Path] = []
    calls = {"n": 0}

    def _touching(store, *args, **kwargs):
        if calls["n"] == image:
            touch_root(store)
            touched.append(Path(store))
        calls["n"] += 1
        return original(store, *args, **kwargs)

    monkeypatch.setattr(worker, "replace_run_embeddings", _touching)
    return touched


def spy_calls_by_image(monkeypatch, tree: Stage4Waiting, names: Sequence[str]) -> dict[str, list[str]]:
    """Record, per image identity, the order the worker calls each of *names*.

    Each name is wrapped in ``phenotypic._cli._cli_stage4_worker``, where the
    loop looks it up at each call. Keyed per image (plan review I5), so the
    one commit thread's order is compared per image, never flat.
    """
    from phenotypic._cli import _cli_stage4_worker as worker

    identities = {(entry.dataset, entry.stem): entry.identity for entry in tree.manifest()}
    calls: dict[str, list[str]] = {}

    def _image_of(name: str, args, kwargs) -> tuple[str, str]:
        if name == "replace_run_embeddings":
            return dataset_and_stem(Path(args[0]))
        if name == "publish_image_success":
            return kwargs["dataset"], kwargs["image_stem"]
        if name == "write_stage4_signal":
            return args[1], args[2]
        return args[1], args[2]  # delete_stage4_signals(output_dir, dataset, stem)

    for name in names:
        original = getattr(worker, name)

        def _spy(*args, _name=name, _original=original, **kwargs):
            calls.setdefault(identities[_image_of(_name, args, kwargs)], []).append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(worker, name, _spy)
    return calls


def stage4_signal_dir(tree: Stage4Waiting) -> Path:
    """``.phenotypic/progress/stage4_payload`` of *tree*."""
    from phenotypic.sdk_ import progress_dir

    return progress_dir(tree.out) / "stage4_payload"


def fence_after_first_signal(tree: Stage4Waiting, seam: str) -> dict:
    """``run_stage4`` keywords whose *seam* raises once the first signal is on disk.

    *seam* is ``"active_check"`` or ``"commit_guard"``; either raises
    ``SlurmGenerationInactiveError``, as a fenced epoch's does.
    """
    from contextlib import contextmanager

    from phenotypic._cli._cli_slurm_lifecycle import SlurmGenerationInactiveError

    def _fenced() -> bool:
        return any(stage4_signal_dir(tree).rglob("*.parquet"))

    def _check() -> None:
        if _fenced():
            raise SlurmGenerationInactiveError("fenced after the first Stage-4 signal")

    @contextmanager
    def _guard():
        _check()
        yield

    return {"active_check": _check} if seam == "active_check" else {"commit_guard": _guard}


def corrupt_one_store(tree: Stage4Waiting) -> Stage4Item:
    """Overwrite the first store's level-0 objmap chunks with garbage; return its item.

    Not a deletion: zarr reads a missing chunk as the fill value, so a store
    with its chunks deleted still loads. Garbage bytes fail to decode, so
    loading the image raises, as a corrupt store at load does (spec §4.8).
    The root is untouched.
    """
    from phenotypic.sdk_ import ngff_

    item = tree.items[0]
    store = zarr_store_path_of(tree.out, item)
    labels = ngff_.read_phenotypic_attributes(store)[ngff_.PhenotypicAttr.LABELS]
    level0 = store / labels[ngff_.OBJMAP_LABEL] / "0"
    # Chunk keys are "c/..." (a directory) or "c.<i>.<j>" (files), by encoding.
    chunks = [
        path
        for path in level0.rglob("*")
        if path.is_file() and path.name != "zarr.json"
    ]
    assert chunks, sorted(p.name for p in level0.iterdir())
    for chunk in chunks:
        chunk.write_bytes(b"not a zarr chunk")
    return item


def capture_dataloader_kwargs(monkeypatch) -> dict:
    """Record the keywords of every ``torch.utils.data.DataLoader`` built (the last wins)."""
    import torch.utils.data as torch_data

    seen: dict = {}
    original = torch_data.DataLoader

    def _capturing(*args, **kwargs):
        seen.clear()
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch_data, "DataLoader", _capturing)
    return seen


def stage4_dataset_for(tree: Stage4Waiting):
    """The first backbone's ``Stage4ImageDataset`` over every image of *tree*."""
    from phenotypic import ImagePipeline
    from phenotypic._cli._cli_stage4_worker import Stage4ImageDataset, plan_stage4

    ((backbone, group), *_rest) = plan_stage4(ImagePipeline.from_json(tree.pipeline))
    embedders = {embedder.payload_key(): embedder for _, embedder in group}
    dense = any(e.pooling == "masked_dense" for e in embedders.values())
    return Stage4ImageDataset(
        tree.out,
        stage4_image_type(tree),
        [(entry, tuple(embedders)) for entry in tree.manifest()],
        embedders,
        {backbone: StubDinoProcessor()} if dense else {},
        backbone,
    )


def stage4_loader_for(tree: Stage4Waiting, *, num_workers: int):
    """The worker's DataLoader over :func:`stage4_dataset_for`."""
    from phenotypic._cli._cli_stage4_worker import _stage4_loader

    return _stage4_loader(stage4_dataset_for(tree), num_workers)


def inventory_and_work_ids(tree: Stage4Waiting) -> tuple[dict, dict]:
    """``(inventory, work_ids)`` for ``sweep_orphan_stage4_signals``, from *tree*'s state."""
    import json

    from phenotypic.sdk_ import processing_state_path

    work_ids = json.loads(processing_state_path(tree.out).read_text(encoding="utf-8"))["config"]["work_ids"]
    return {ds: list(images) for ds, images in work_ids.items()}, work_ids


# ---------------------------------------------------------------------------
# #282 P4 T4: resume, retry, crash injection
# ---------------------------------------------------------------------------

#: An embedder ``batch_size`` no tiny run's region count reaches, so inline
#: and Stage 4 each run every queue in one pass (spec §5 as amended: byte
#: equality holds only at the same pass shape). A tiny plate has one object
#: under ``FakeGpuDetector(threshold=0.3)`` and a few dozen under Otsu; the
#: 2 x 2 grid adds four cells.
ONE_PASS_BATCH = 512


def embedding_deliverables(out: Path) -> list[Path]:
    """Every published embedding table under ``deliverables/embeddings/``, relative to *out*."""
    base = Path(out) / "deliverables" / "embeddings"
    found = sorted(path.relative_to(out) for path in base.rglob("*.parquet"))
    assert found, f"no embedding deliverable under {base}"
    return found


def manifest_without_compute(out: Path) -> dict:
    """``deliverables/embeddings/manifest.json`` with every ``compute`` block removed.

    ``compute`` (device, torch, ``stage``, physical batch) is the one part a
    Stage-4 run and an inline run may differ in (#282 P2 T1).
    """
    import json

    def _strip(value):
        if isinstance(value, dict):
            return {key: _strip(item) for key, item in value.items() if key != "compute"}
        if isinstance(value, list):
            return [_strip(item) for item in value]
        return value

    path = Path(out) / "deliverables" / "embeddings" / "manifest.json"
    return _strip(json.loads(path.read_text(encoding="utf-8")))


def stage4_signal_leftovers(out: Path) -> list[Path]:
    """Every Stage-4 signal file and signal temp file under *out*."""
    from phenotypic.sdk_ import progress_dir

    base = progress_dir(out) / "stage4_payload"
    return sorted(path for path in base.rglob("*") if path.is_file()) if base.is_dir() else []


def no_stage4_signals(out: Path) -> bool:
    """Whether *out* holds no Stage-4 signal file and no signal temp file."""
    return not stage4_signal_leftovers(out)


def record_is_valid(out: Path, item: Stage4Item) -> bool:
    """Whether *item*'s record under its recorded work id is valid."""
    from phenotypic._cli._cli_completion import valid_image_success

    return valid_image_success(out, dataset=item.dataset, image_stem=item.stem, work_id=item.work_id)


def all_records_valid(out: Path) -> bool:
    """Whether every image of the run at *out* has a valid record."""
    return all(record_is_valid(out, item) for item in run_items(out))


def store_has_embeddings(out: Path, item: Stage4Item) -> bool:
    """Whether *item*'s store journal run carries an ``embeddings`` block."""
    from phenotypic.sdk_ import journal_table_run

    run = journal_table_run(zarr_store_path_of(out, item))
    return bool(run is not None and run.descriptor.get("embeddings"))


def crashed_image(out: Path, crash_after: str) -> Stage4Item:
    """The one image whose state on disk shows a crash right after *crash_after*.

    Read from the disk, never from the test: signals but no commit
    (``write_stage4_signal``); a commit but no record
    (``replace_run_embeddings``); a record with its signals still there
    (``publish_image_success``).
    """
    def _shows(item: Stage4Item) -> bool:
        signals = bool(signal_files(out, item))
        committed = store_has_embeddings(out, item)
        recorded = record_is_valid(out, item)
        if crash_after == "write_stage4_signal":
            return signals and not committed and not recorded
        if crash_after == "replace_run_embeddings":
            return committed and not recorded
        if crash_after == "publish_image_success":
            return recorded and signals
        raise ValueError(crash_after)

    (crashed,) = [item for item in run_items(out) if _shows(item)]
    return crashed


class SpiedRun:
    """A :func:`run_cli_subprocess_spying` result: the process and the spied sequence."""

    def __init__(self, completed, sequence: list[str]) -> None:
        self.completed = completed
        self.returncode = completed.returncode
        self.stdout = completed.stdout
        self.stderr = completed.stderr
        self.sequence = sequence


def run_cli_subprocess_spying(
    inputs: Path, out: Path, *, pipeline: Path, extra: Sequence[str] = (), stem: str
) -> SpiedRun:
    """:func:`run_cli_subprocess`, recording the order of the four durable writes for *stem*."""
    import tempfile

    with tempfile.TemporaryDirectory(dir=Path(out).parent) as scratch:
        spy_path = Path(scratch) / "stage4_calls.txt"
        spy_path.touch()
        completed = run_cli_subprocess(
            inputs, out, pipeline=pipeline, extra=extra, spy_stem=stem, spy_path=spy_path
        )
        sequence = spy_path.read_text(encoding="utf-8").split()
    return SpiedRun(completed, sequence)


def plant_signals(out: Path, *, image: int) -> list[Path]:
    """Planted signal files for the run's image *image* (:func:`plant_stage4_signals`)."""
    return plant_stage4_signals(out, run_items(out)[image])


@contextlib.contextmanager
def stage4_stubbed_out(monkeypatch):
    """While open, ``run_local_stage4`` does nothing and returns ``{}``.

    Every ``--embed-stage gpu`` image then stops after Stages 1-3, so a test
    can look at the disk before Stage 4 would run.
    """
    from phenotypic._cli import _cli_stage4_worker

    with monkeypatch.context() as patch:
        patch.setattr(_cli_stage4_worker, "run_local_stage4", lambda *a, **k: {})
        yield


def other_item(tree: Stage4Waiting) -> Stage4Item:
    """A second pending image of *tree*, besides ``tree.items[0]``."""
    return tree.items[1]


def tail_events(out: Path) -> int:
    """How many lines the run's event log has now (a mark for :func:`new_events`)."""
    from phenotypic.sdk_ import event_log_path

    path = event_log_path(out)
    return len(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else 0


def new_events(out: Path, *, after: int) -> list[list[str]]:
    """The event-log lines appended since mark *after*, split on ``|``."""
    from phenotypic.sdk_ import event_log_path

    lines = event_log_path(out).read_text(encoding="utf-8").splitlines()[after:]
    return [line.split("|") for line in lines if line.strip()]


def stages_of(events: list[list[str]], *, image: str) -> list[str]:
    """The distinct stage tags of *events* for image file *image*, in first-seen order.

    Field 8 (index 7) is the stage; an untagged event reads ``""``.
    """
    seen: list[str] = []
    for fields in events:
        if len(fields) > 2 and fields[2] == image:
            stage = fields[7] if len(fields) > 7 else ""
            if stage not in seen:
                seen.append(stage)
    return seen


@contextlib.contextmanager
def spy_restart_clear(monkeypatch):
    """Wrap the CLI's ``--restart`` clear; yield a namespace whose
    ``saw_no_stage4_payload`` says whether ``progress/stage4_payload`` was gone
    right after it (``None`` if it never ran)."""
    import types

    import phenotypic.phenotypicCLI as cli
    from phenotypic.sdk_ import progress_dir

    seen = types.SimpleNamespace(saw_no_stage4_payload=None)
    original = cli.clear_machine_state

    def _clear(output_dir, *args, **kwargs):
        result = original(output_dir, *args, **kwargs)
        seen.saw_no_stage4_payload = not (progress_dir(output_dir) / "stage4_payload").exists()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(cli, "clear_machine_state", _clear)
        yield seen
