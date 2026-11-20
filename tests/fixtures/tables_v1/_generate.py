"""Generate the v1 flat-``tables/`` fixtures #281 P3 reads.

Run as a script to rebuild both fixtures under this directory::

    uv run python tests/fixtures/tables_v1/_generate.py

**Rebuild only from the commit that added this file** (#281 P3 Task 1; ``git
log --diff-filter=A -- tests/fixtures/tables_v1/_generate.py`` names it). From
P3 Task 2 on, no code writes the v1 flat layout -- a store's tables live in
``tables/<n>-<sha12>/`` -- so a regeneration at any later commit would produce
a v2 tree under a v1 name, and every test that relies on "this is what a
pre-P3 tree looks like" would silently stop testing it.

The fixtures, and what each pins:

==================================  =============================================
``v1_run/``                         a complete one-image full-run output tree
                                    (deliverables, the per-image record, the
                                    run proof), written by ``python -m
                                    phenotypic`` over one
                                    ``load_synth_yeast_plate()`` image with the
                                    ``tiny_pipeline``. Its one store holds the
                                    flat ``tables/measurements/table.parquet``
                                    and ``tables.measurements`` descriptor.
                                    Forward modes must refuse it; ``--mode
                                    migrate`` must convert it and leave it
                                    reading ``complete`` (Review Focus 1).
``v1_store_null_pipeline.ome.zarr`` a copy of ``v1_run``'s store whose journal
                                    holds one ``legacy`` application with
                                    ``pipeline: null`` and which has no figure
                                    run: the store whose table digest is
                                    unknowable, so migrate must name its folder
                                    ``1-unknown`` rather than fabricate one
                                    (spec D2, D10).
==================================  =============================================

The generator copies ``v1_run/`` to a temporary directory and asserts it still
reads ``complete`` there, so no absolute path under ``.phenotypic/`` survives
the move unnoticed. The tree stays well under 5 MB.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FIXTURE_ROOT = Path(__file__).resolve().parent

#: The full-run tree, and the one store inside it.
V1_RUN = "v1_run"
V1_RUN_STORE = Path("results", "ds", "zarr", "plate.ome.zarr")
#: The store edited to carry a ``legacy`` application with no pipeline.
V1_STORE_NULL_PIPELINE = "v1_store_null_pipeline.ome.zarr"

#: Size budget for everything this module writes.
_MAX_BYTES = 5 * 1024 * 1024


def _tiny_pipeline_json() -> str:
    """The ``tiny_pipeline`` of ``tests/_tiny_run.py``, serialized."""
    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize

    return (
        ImagePipeline(ops=[OtsuDetector()], meas={"size": MeasureSize()}).to_json()
        or ""
    )


def _write_input(root: Path) -> Path:
    """Write ``<root>/in/ds/plate.tif`` and return the input root."""
    from PIL import Image as PILImage

    from phenotypic.data import load_synth_yeast_plate

    dataset = root / "in" / "ds"
    dataset.mkdir(parents=True)
    pixels = load_synth_yeast_plate().rgb[:].astype("uint8")
    PILImage.fromarray(pixels).save(dataset / "plate.tif")
    return root / "in"


def write_v1_run(target: Path) -> Path:
    """Run the CLI over one synthetic plate and move its output to *target*."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        inputs = _write_input(work)
        pipeline = work / "tiny_run.pht-pipe.json"
        pipeline.write_text(_tiny_pipeline_json(), encoding="utf-8")
        out = work / "out"
        completed = subprocess.run(
            [
                sys.executable, "-m", "phenotypic",
                "--input", str(inputs),
                "--output", str(out),
                "--pipeline", str(pipeline),
                "--njobs", "1",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"python -m phenotypic exited {completed.returncode}:\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        _scrub_work_dir(out, work)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(out, target, symlinks=True)
    return target


#: What the temporary build directory's absolute path is replaced with, so
#: the committed tree names no host, user or scratch path.
_SCRUBBED_WORK_DIR = "/nonexistent/tables_v1"


def _scrub_work_dir(out: Path, work: Path) -> None:
    """Replace *work*'s absolute path in every text file under *out*.

    Only ``.phenotypic/processing_state.json`` carries it today (the
    ``input_path``/``output_dir``/``pipeline_path`` the run was given). They
    name a directory that is deleted when this function's caller returns, so
    the value is dead either way; :func:`_check_relocatable` proves no reader
    depends on it.
    """
    prefix = str(work)
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".log", ".md", ".html"}:
            continue
        text = path.read_text(encoding="utf-8")
        if prefix in text:
            path.write_text(text.replace(prefix, _SCRUBBED_WORK_DIR), encoding="utf-8")


def write_v1_store_null_pipeline(run: Path, target: Path) -> Path:
    """Copy *run*'s store to *target* and make its journal one null-pipeline ``legacy`` application."""
    from phenotypic.sdk_ import ngff_

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(run / V1_RUN_STORE, target)
    root_path = target / ngff_.STORE_ROOT_JSON
    document = json.loads(root_path.read_text(encoding="utf-8"))
    phenotypic = document["attributes"][ngff_.PhenotypicAttr.ROOT]
    journal = phenotypic[ngff_.PhenotypicAttr.PROVENANCE]
    application = journal["applications"][-1]
    journal["applications"] = [
        {
            **application,
            "sequence": 1,
            "kind": "legacy",
            "phenotypic_version": None,
            "input_filename": None,
            "pipeline": None,
        }
    ]
    journal["original_filename"] = None
    phenotypic.pop(ngff_.PhenotypicAttr.FIGURES, None)
    root_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    from phenotypic._core._provenance import validate_provenance_journal

    validate_provenance_journal(journal)
    return target


def _check_relocatable(run: Path) -> None:
    """Assert a copy of *run* elsewhere still reads ``complete``."""
    from phenotypic.sdk_ import resolve_run_state

    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "moved"
        shutil.copytree(run, copy, symlinks=True)
        state = resolve_run_state(copy, depth="deep")
        if state.completion != "complete":
            raise AssertionError(
                f"v1_run reads {state.completion!r} after a move: {state}"
            )


def _tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def regenerate(root: Path = FIXTURE_ROOT) -> dict[str, Path]:
    """Rebuild every fixture under *root* and return ``{name: path}``."""
    run = write_v1_run(root / V1_RUN)
    _check_relocatable(run)
    store = write_v1_store_null_pipeline(run, root / V1_STORE_NULL_PIPELINE)
    total = _tree_bytes(run) + _tree_bytes(store)
    if total > _MAX_BYTES:
        raise AssertionError(f"tables_v1 fixtures are {total} bytes, over 5 MB")
    return {V1_RUN: run, V1_STORE_NULL_PIPELINE: store}


if __name__ == "__main__":  # pragma: no cover - developer entry point
    for name, target in regenerate().items():
        print(f"{name}: {target}")
