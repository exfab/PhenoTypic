"""Shared test helpers for the #281 plan (DVC outputs, TSV, tables v2, embeddings).

Fixtures here are re-exported by ``tests/conftest.py`` with an explicit import
list, so every test under ``tests/`` can request them by name. ``phenotypic``
is imported inside each fixture rather than at module scope: the top-level
conftest must stay importable in dependency-light pytest runs (the
packaging-integrity CI job), where ``phenotypic`` may be absent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
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
CROP_ENTRY_KEY = "dinov2-vitb14/crops"


def _write_crop_payload(part: Path, run_id: str, labels) -> dict:
    """Write a 4-D float16 crop embedding table into *part* and return its raw entry."""
    import hashlib

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from phenotypic.sdk_ import ngff_
    from phenotypic.sdk_._table_runs import embedding_table_relative_path

    relative = embedding_table_relative_path(run_id, "dinov2-vitb14", "crops")
    payload = part / relative
    payload.parent.mkdir(parents=True, exist_ok=True)
    for group in (payload.parent.parent.parent, payload.parent.parent, payload.parent):
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
        "input": {"series": "rgb", "level": 0},
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

    from phenotypic.sdk_ import latest_table_run
    from phenotypic.sdk_._measurement_tables import (
        _rewrite_store_tables,
        objmap_digest,
        write_run_entry,
    )

    store = measured_image.save2zarr(
        tmp_path / "crops.ome.zarr", measurement_table=tables_for(measured_image)
    )
    run = latest_table_run(store)
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
                embeddings={CROP_ENTRY_KEY: raw},
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


def crop_entry(store: Path) -> dict:
    """The :data:`CROP_ENTRY_KEY` embedding entry of *store*'s first run that has one."""
    from phenotypic.sdk_ import list_table_runs

    for run in list_table_runs(store):
        embeddings = run.descriptor.get("embeddings") or {}
        if CROP_ENTRY_KEY in embeddings:
            return dict(embeddings[CROP_ENTRY_KEY])
    raise KeyError(f"{store} has no {CROP_ENTRY_KEY} entry")


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
