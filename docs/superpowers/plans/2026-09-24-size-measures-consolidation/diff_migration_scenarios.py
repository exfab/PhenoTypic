"""Differential proof for the size measures consolidation (plan amendment A6).

The migration goldens for these four scenarios were already red on main before
this change (plan-review H4), so they cannot show that the change preserved
values. This script compares main with the branch tip directly instead, on the
same frozen inputs and through the same harness
(``tests.migration._runner.run_scenario``):

* ``refine.KeepSectionLargest``
* ``measure.MeasureIntensity``
* ``measure.MeasureShape``
* ``measure.MeasureSize``

It drives shipped code, so it lives beside the plan and not in
``logic_validation_scripts/`` (root CLAUDE.md).

Usage (each ``capture`` runs inside the tree it measures, with that tree's
own ``uv run``; ``compare`` needs only numpy and pandas)::

    uv run --directory <main-tree> python <this-file> capture --out <dir>/main
    uv run --directory <tip-tree>  python <this-file> capture --out <dir>/tip
    uv run python <this-file> compare --main <dir>/main --tip <dir>/tip

Rules applied by ``compare``:

* KeepSectionLargest: ``objmask`` and ``objmap`` are array-equal, dtype included.
* MeasureIntensity: ``assert_frame_equal`` with ``check_dtype=True``,
  ``rtol=1e-10``, ``atol=0``.
* MeasureShape + MeasureSize are pooled per side, because columns move between
  the two measurers. Every main column must either exist unchanged in the tip
  or have its successor (spec section 6, ``RENAMES`` below) present. Values
  must agree at ``rtol=1e-10`` (``atol=0``). The exception is the per-object
  EDT columns (``EDT_SUCCESSORS``), which spec section 4.1 changes on purpose.
  They are compared only on objects that touch neither another label
  (8-connectivity) nor the image border. For an isolated object the per-object
  padded-crop EDT equals main's whole-image EDT exactly. Take any object
  pixel p and its nearest non-object pixel q. The pixel one step from q
  towards p is closer to p, so it belongs to the object. Hence q is
  8-adjacent to the object: it is background, not another label, and it lies
  inside the padded crop. Every excluded object must satisfy tip <= main,
  because the crop only adds background: neighbours, the bounding-box pad and
  the image border. Excluded objects are reported by count and label.
* Tip-only columns (the Size radius family, for example) are listed, not
  compared.

Guards against a comparison that proves nothing: a missing file, scenario or
column fails the run instead of skipping. The two captures must come from
different source trees (the digest of the measured ``src`` files differs).
They must also have run on byte-identical frozen inputs, and their input
object maps must be equal.

Tolerance: rtol 1e-10 is the one derived in
``tests/unit/measure/test_size_consolidation_equivalence.py``. Every value is
a regionprops/Qhull scalar or a reduction over at most ~1e4 pixels, so
summation-order noise is bounded near 1e4 ulp, about 2e-12 relative. That
leaves 50x headroom and stays far below any real behaviour change.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIO_IDS = (
    "refine.KeepSectionLargest",
    "measure.MeasureIntensity",
    "measure.MeasureShape",
    "measure.MeasureSize",
)
LABEL = "Object_Label"
RTOL = 1e-10

# Spec section 6: every retired Shape column and its successor.
RENAMES = {
    "Shape_Area": "Size_Area",
    "Shape_Perimeter": "Size_Perimeter",
    "Shape_ConvexArea": "Size_ConvexArea",
    "Shape_BboxArea": "Size_BboxArea",
    "Shape_MajorAxisLength": "Size_MajorAxisLength",
    "Shape_MinorAxisLength": "Size_MinorAxisLength",
    "Shape_MaxRadius": "Size_InscribedRadius",
    "Shape_MeanRadius": "Shape_MeanBoundaryDist",
    "Shape_MedianRadius": "Shape_MedianBoundaryDist",
}
# Successors computed on a padded per-object crop (spec section 4.1).
EDT_SUCCESSORS = frozenset(
    {"Size_InscribedRadius", "Shape_MeanBoundaryDist", "Shape_MedianBoundaryDist"}
)

# Files whose bytes decide the four scenarios' results.
_SRC_GLOBS = (
    "src/phenotypic/measure/*.py",
    "src/phenotypic/schema/*.py",
    "src/phenotypic/refine/_keep_section_largest.py",
    "src/phenotypic/abc_/_measure_features.py",
)
_INPUT_GLOB = "tests/migration/_inputs/*"
_HARNESS_GLOB = "tests/migration/_*.py"
_MANIFEST = "manifest.json"


class DiffFailure(Exception):
    """A precondition the proof depends on does not hold."""


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_files(tree: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    files: dict[str, str] = {}
    for pattern in patterns:
        for path in sorted(tree.glob(pattern)):
            if path.is_file():
                files[path.relative_to(tree).as_posix()] = _sha256(path)
    if not files:
        raise DiffFailure(f"no files match {patterns} under {tree}")
    return files


def _combined_digest(files: dict[str, str]) -> str:
    joined = "\n".join(f"{name} {digest}" for name, digest in sorted(files.items()))
    return hashlib.sha256(joined.encode()).hexdigest()


def _git(tree: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(tree), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise DiffFailure(f"git {' '.join(args)} failed in {tree}: {done.stderr}")
    return done.stdout.strip()


def _require_inside(module_file: str | None, root: Path, what: str) -> None:
    if module_file is None or not Path(module_file).resolve().is_relative_to(root):
        raise DiffFailure(
            f"{what} was imported from {module_file}, not from {root}. Run "
            "`uv run --directory <tree>` so the tree's own environment is used."
        )


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "absent"


def capture(out: Path, tree: Path) -> None:
    """Run the four scenarios in ``tree`` and save their results under ``out``."""
    tree = tree.resolve()
    out = out.resolve()
    if not (tree / "src" / "phenotypic").is_dir():
        raise DiffFailure(f"{tree} is not a PhenoTypic checkout (no src/phenotypic)")
    if not (tree / "tests" / "migration" / "_runner.py").is_file():
        raise DiffFailure(f"{tree} has no tests/migration/_runner.py")
    if "_goldens" in out.parts:
        raise DiffFailure(f"refusing to write under a _goldens directory: {out}")
    if out.exists() and any(out.iterdir()):
        raise DiffFailure(f"{out} is not empty; give each capture a fresh directory")

    sys.path.insert(0, str(tree))
    import phenotypic
    from tests.migration import _inputs, _runner, _scenarios

    _require_inside(phenotypic.__file__, tree / "src", "phenotypic")
    _require_inside(_runner.__file__, tree, "tests.migration._runner")

    by_id = {s.scenario_id: s for s in _scenarios.build_scenarios()}
    missing = [sid for sid in SCENARIO_IDS if sid not in by_id]
    if missing:
        raise DiffFailure(f"scenarios not found in {tree}: {missing}")

    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for sid in SCENARIO_IDS:
        scenario = by_id[sid]
        result = _runner.run_scenario(scenario)
        if isinstance(result, _runner.ImageGolden):
            file_name = f"{sid}.npz"
        elif isinstance(result, _runner.FrameGolden):
            file_name = f"{sid}.parquet"
        else:
            raise DiffFailure(f"{sid}: unexpected result type {type(result)!r}")
        result.save(out / file_name)

        frozen = _inputs.load_frozen_input(scenario.category)
        objmap = np.ascontiguousarray(frozen.objmap[:])
        objmap_name = f"{sid}.input_objmap.npy"
        np.save(out / objmap_name, objmap)

        results[sid] = {
            "result": file_name,
            "input_objmap": objmap_name,
            "category": scenario.category,
            "kwargs": repr(scenario.resolve_kwargs()),
            "summary": result.summary,
        }
        print(f"[capture] {sid}: {result.summary} -> {file_name}")

    src_files = _digest_files(tree, _SRC_GLOBS)
    manifest = {
        "tree": str(tree),
        "git_head": _git(tree, "rev-parse", "HEAD"),
        "git_dirty": _git(tree, "status", "--porcelain", "--", "src", "tests/migration")
        .splitlines(),
        "phenotypic_version": getattr(phenotypic, "__version__", "unknown"),
        "python": platform.python_version(),
        "libraries": {
            name: _version(name)
            for name in ("numpy", "scipy", "scikit-image", "pandas", "pyarrow")
        },
        "src_files": src_files,
        "src_digest": _combined_digest(src_files),
        "input_files": _digest_files(tree, (_INPUT_GLOB,)),
        "harness_files": _digest_files(tree, (_HARNESS_GLOB,)),
        "scenarios": results,
    }
    (out / _MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(
        f"[capture] tree {tree} HEAD {manifest['git_head'][:12]} "
        f"src digest {manifest['src_digest'][:16]} "
        f"({len(manifest['git_dirty'])} uncommitted paths) -> {out}"
    )


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------


class Report:
    """Collects check outcomes; the run fails if any check failed."""

    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def ok(self, message: str) -> None:
        self.passed += 1
        print(f"  PASS  {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"  FAIL  {message}")

    def note(self, message: str) -> None:
        print(f"  ....  {message}")


def _load_manifest(directory: Path) -> dict:
    path = directory / _MANIFEST
    if not path.is_file():
        raise DiffFailure(f"missing {path}; run `capture` first")
    return json.loads(path.read_text())


def _result_path(directory: Path, manifest: dict, sid: str, key: str) -> Path:
    entry = manifest["scenarios"].get(sid)
    if entry is None:
        raise DiffFailure(f"{directory}: manifest has no scenario {sid}")
    path = directory / entry[key]
    if not path.is_file():
        raise DiffFailure(f"missing {path}")
    return path


def _check_provenance(main: dict, tip: dict, report: Report) -> None:
    print("provenance")
    for side, manifest in (("main", main), ("tip", tip)):
        report.note(
            f"{side}: {manifest['tree']} HEAD {manifest['git_head']} "
            f"src digest {manifest['src_digest'][:16]} "
            f"uncommitted {manifest['git_dirty'] or 'none'}"
        )
    if main["src_digest"] == tip["src_digest"]:
        report.fail(
            "main and tip captures ran byte-identical src files, so this "
            "comparison cannot tell them apart"
        )
    else:
        changed = sorted(
            name
            for name in set(main["src_files"]) | set(tip["src_files"])
            if main["src_files"].get(name) != tip["src_files"].get(name)
        )
        report.ok(f"captures ran different code: {changed}")
    if main["input_files"] == tip["input_files"]:
        report.ok(f"frozen inputs byte-identical ({len(main['input_files'])} files)")
    else:
        report.fail("frozen inputs differ between the two trees")
    if main["harness_files"] != tip["harness_files"]:
        report.note("tests/migration harness modules differ between the trees")
    if main["libraries"] != tip["libraries"] or main["python"] != tip["python"]:
        report.note(
            f"environments differ: main {main['python']} {main['libraries']} / "
            f"tip {tip['python']} {tip['libraries']}"
        )


def _compare_keep_section_largest(
        main_dir: Path, main: dict, tip_dir: Path, tip: dict, report: Report
) -> None:
    sid = "refine.KeepSectionLargest"
    print(sid)
    with np.load(_result_path(main_dir, main, sid, "result")) as data:
        main_arrays = {k: data[k] for k in data.files}
    with np.load(_result_path(tip_dir, tip, sid, "result")) as data:
        tip_arrays = {k: data[k] for k in data.files}
    if set(main_arrays) != set(tip_arrays) or not main_arrays:
        report.fail(f"components differ: main {sorted(main_arrays)} tip {sorted(tip_arrays)}")
        return
    for name in ("objmask", "objmap"):
        if name not in main_arrays:
            report.fail(f"component {name} missing from the capture")
            continue
        a, b = main_arrays[name], tip_arrays[name]
        if a.dtype == b.dtype and np.array_equal(a, b):
            kept = np.unique(a[a > 0]).size if name == "objmap" else int(a.sum())
            report.ok(f"{name} array-equal, dtype {a.dtype} ({kept} kept)")
        else:
            differing = int(np.count_nonzero(a != b)) if a.shape == b.shape else -1
            report.fail(
                f"{name} differs: dtype {a.dtype}/{b.dtype}, shape "
                f"{a.shape}/{b.shape}, {differing} differing pixels"
            )


def _compare_intensity(
        main_dir: Path, main: dict, tip_dir: Path, tip: dict, report: Report
) -> None:
    sid = "measure.MeasureIntensity"
    print(sid)
    a = pd.read_parquet(_result_path(main_dir, main, sid, "result"))
    b = pd.read_parquet(_result_path(tip_dir, tip, sid, "result"))
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=True, rtol=RTOL, atol=0.0)
    except AssertionError as error:
        report.fail(f"frames differ: {error}")
        return
    report.ok(f"assert_frame_equal (dtype checked, rtol {RTOL}): {a.shape}")


def _pool(directory: Path, manifest: dict, side: str) -> pd.DataFrame:
    shape = pd.read_parquet(_result_path(directory, manifest, "measure.MeasureShape", "result"))
    size = pd.read_parquet(_result_path(directory, manifest, "measure.MeasureSize", "result"))
    for frame, name in ((shape, "MeasureShape"), (size, "MeasureSize")):
        if LABEL not in frame.columns:
            raise DiffFailure(f"{side} {name} has no {LABEL} column")
    overlap = (set(shape.columns) & set(size.columns)) - {LABEL}
    if overlap:
        raise DiffFailure(f"{side}: Shape and Size share columns {sorted(overlap)}")
    if set(shape[LABEL]) != set(size[LABEL]):
        raise DiffFailure(f"{side}: Shape and Size cover different objects")
    return shape.merge(size, on=LABEL, validate="one_to_one")


def _contact_labels(objmap: np.ndarray) -> tuple[set[int], set[int]]:
    """Labels on the image border, and labels 8-adjacent to another label."""
    border = np.concatenate(
        [objmap[0, :], objmap[-1, :], objmap[:, 0], objmap[:, -1]]
    )
    on_border = {int(v) for v in np.unique(border) if v > 0}

    touching: set[int] = set()
    # Right, down, down-right and down-left cover all 8 neighbour pairs.
    pairs = (
        (objmap[:, :-1], objmap[:, 1:]),
        (objmap[:-1, :], objmap[1:, :]),
        (objmap[:-1, :-1], objmap[1:, 1:]),
        (objmap[:-1, 1:], objmap[1:, :-1]),
    )
    for a, b in pairs:
        contact = (a > 0) & (b > 0) & (a != b)
        touching.update(int(v) for v in np.unique(a[contact]))
        touching.update(int(v) for v in np.unique(b[contact]))
    return on_border, touching


def _input_objmap(main_dir: Path, main: dict, tip_dir: Path, tip: dict) -> np.ndarray:
    maps = []
    for directory, manifest in ((main_dir, main), (tip_dir, tip)):
        for sid in ("measure.MeasureShape", "measure.MeasureSize"):
            maps.append(np.load(_result_path(directory, manifest, sid, "input_objmap")))
    if not all(m.shape == maps[0].shape and np.array_equal(m, maps[0]) for m in maps):
        raise DiffFailure("the Shape/Size scenarios did not run on one input objmap")
    return maps[0]


def _values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].to_numpy(dtype=float)


def _compare_strict(
        main_pool: pd.DataFrame, tip_pool: pd.DataFrame, old: str, new: str,
        report: Report,
) -> None:
    what = old if old == new else f"{old} -> {new}"
    if old == new and main_pool[old].dtype != tip_pool[new].dtype:
        report.fail(f"{what}: dtype {main_pool[old].dtype} -> {tip_pool[new].dtype}")
        return
    try:
        np.testing.assert_allclose(
            _values(tip_pool, new), _values(main_pool, old),
            rtol=RTOL, atol=0.0, equal_nan=True,
        )
    except AssertionError as error:
        report.fail(f"{what}: {' '.join(str(error).split())[:400]}")
        return
    report.ok(f"{what}: equal at rtol {RTOL} on all {len(main_pool)} objects")


def _compare_edt(
        main_pool: pd.DataFrame, tip_pool: pd.DataFrame, old: str, new: str,
        contact: np.ndarray, report: Report,
) -> None:
    what = f"{old} -> {new}"
    main_values = _values(main_pool, old)
    tip_values = _values(tip_pool, new)
    isolated = ~contact
    if not isolated.any():
        report.fail(f"{what}: no isolated object, so nothing is compared")
        return
    try:
        np.testing.assert_allclose(
            tip_values[isolated], main_values[isolated], rtol=RTOL, atol=0.0,
        )
    except AssertionError as error:
        report.fail(f"{what} (isolated objects): {' '.join(str(error).split())[:400]}")
        return
    report.ok(f"{what}: equal at rtol {RTOL} on {int(isolated.sum())} isolated objects")

    labels = main_pool[LABEL].to_numpy()
    excluded = labels[contact]
    moved = contact & ~np.isclose(tip_values, main_values, rtol=RTOL, atol=0.0)
    report.note(
        f"{what}: {excluded.size} excluded (touching), of which {int(moved.sum())} "
        f"changed; excluded labels {excluded.tolist()}"
    )
    # Per-pixel the crop EDT can only shrink, so mean, median and max can too.
    bound = main_values[contact] * (1.0 + RTOL)
    above = contact.copy()
    above[contact] = ~(tip_values[contact] <= bound)
    if above.any():
        rows = [
            (int(label), float(t), float(m))
            for label, t, m in zip(labels[above], tip_values[above], main_values[above])
        ]
        report.fail(f"{what}: tip > main on excluded objects (label, tip, main) {rows}")
    else:
        report.ok(f"{what}: tip <= main on all {excluded.size} excluded objects")


def _compare_size_shape(
        main_dir: Path, main: dict, tip_dir: Path, tip: dict, report: Report
) -> None:
    print("measure.MeasureShape + measure.MeasureSize (pooled)")
    main_pool = _pool(main_dir, main, "main").sort_values(LABEL)
    tip_pool = _pool(tip_dir, tip, "tip").sort_values(LABEL)
    main_labels = main_pool[LABEL].to_numpy()
    if not np.array_equal(main_labels, tip_pool[LABEL].to_numpy()):
        raise DiffFailure("main and tip measured different objects")
    main_pool = main_pool.reset_index(drop=True)
    tip_pool = tip_pool.reset_index(drop=True)

    objmap = _input_objmap(main_dir, main, tip_dir, tip)
    map_labels = np.unique(objmap[objmap > 0]).astype(main_labels.dtype)
    if not np.array_equal(map_labels, main_labels):
        raise DiffFailure(f"{LABEL} values are not the input objmap's labels")
    on_border, touching = _contact_labels(objmap)
    contact = np.isin(main_labels, sorted(on_border | touching))
    report.note(
        f"{main_labels.size} objects: {len(on_border)} on the border, "
        f"{len(touching)} touching another label, {int(contact.sum())} excluded "
        "from the EDT comparison"
    )

    covered: set[str] = {LABEL}
    for old in main_pool.columns:
        if old == LABEL:
            continue
        targets = [c for c in (old, RENAMES.get(old)) if c is not None and c in tip_pool]
        if not targets:
            report.fail(f"{old}: absent from the tip and no successor present")
            continue
        for new in targets:
            covered.add(new)
            if new in EDT_SUCCESSORS:
                _compare_edt(main_pool, tip_pool, old, new, contact, report)
            else:
                _compare_strict(main_pool, tip_pool, old, new, report)

    new_only = [c for c in tip_pool.columns if c not in covered]
    report.note(f"tip-only columns (listed, not compared): {new_only}")


def compare(main_dir: Path, tip_dir: Path) -> int:
    """Apply the per-scenario rules; return the process exit code."""
    main_dir, tip_dir = main_dir.resolve(), tip_dir.resolve()
    if main_dir == tip_dir:
        raise DiffFailure("--main and --tip name the same directory")
    main = _load_manifest(main_dir)
    tip = _load_manifest(tip_dir)
    report = Report()
    _check_provenance(main, tip, report)
    _compare_keep_section_largest(main_dir, main, tip_dir, tip, report)
    _compare_intensity(main_dir, main, tip_dir, tip, report)
    _compare_size_shape(main_dir, main, tip_dir, tip, report)

    if report.failures:
        print(f"RESULT: FAIL ({len(report.failures)} failed, {report.passed} passed)")
        return 1
    print(f"RESULT: PASS ({report.passed} checks)")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture", help="run the four scenarios in one tree")
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument(
        "--tree", type=Path, default=Path.cwd(),
        help="checkout to measure (default: the working directory)",
    )
    cmp_ = commands.add_parser("compare", help="compare a main and a tip capture")
    cmp_.add_argument("--main", type=Path, required=True)
    cmp_.add_argument("--tip", type=Path, required=True)
    return parser.parse_args(argv)


def run_cli(argv: list[str]) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "capture":
            capture(args.out, args.tree)
            return 0
        return compare(args.main, args.tip)
    except DiffFailure as error:
        print(f"RESULT: FAIL ({error})")
        return 2


if __name__ == "__main__":
    sys.exit(run_cli(sys.argv[1:]))
