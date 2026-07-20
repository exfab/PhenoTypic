"""Generate the A10 FilFinder 1.8 external-oracle fixture.

Run with::

    PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH=docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/filfinder/upstream \
      uv run --with fil-finder==1.8 python \
      docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/filfinder/generate_fixture.py
"""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import pathlib
import platform
import re
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable

import numpy as np
from scipy import ndimage


REFERENCE_DIR = pathlib.Path(__file__).resolve().parent
REPOSITORY_ROOT = REFERENCE_DIR.parents[5]
FIXTURE_PATH = REPOSITORY_ROOT / "tests/fixtures/reconnect/filfinder/oracle.json"
THRESHOLD = 0.5
BEAMWIDTH_PX = 1.0
BRANCH_THRESHOLD_PX: float | None = None
RELATIVE_INTENSITY_THRESHOLD = 0.2
MAX_PRUNE_ITERATIONS = 10
RNG_SEED = 0
EXPECTED_SOURCE = REFERENCE_DIR / "upstream/fil_finder/filfinder2D.py"
EXPECTED_SUPPLIED_MASK_WARNING = "Using inputted mask. Skipping creation of anew mask."
WARNING_POLICY_CONTROL = "A10 warning-policy control must remain visible."
GRAPH_PRUNING_WARNING = "Graph pruning reached max iterations."
EXPECTED_WORKER_STDERR_LINES = [
    "WARNING: AstropyDeprecationWarning: The TestRunner class is deprecated and may be removed in a future version.",
    "        Use pytest instead. [astropy.tests.runner]",
    "WARNING: AstropyDeprecationWarning: The TestRunnerBase class is deprecated and may be removed in a future version.",
    "        Use pytest instead. [astropy.utils.decorators]",
]
EXPECTED_TASK_COUNTS = {
    "straight": 1,
    "y_spur": 1,
    "disconnected": 2,
    "loop_branch": 1,
    "noise": 3,
    "symmetric_tie": 1,
    "threshold_boundary": 2,
}


def warning_records(caught: list[warnings.WarningMessage]) -> list[dict[str, object]]:
    """Serialize and count deterministic warning evidence without absolute paths."""
    counts: dict[tuple[str, str, str, int], int] = {}
    for item in caught:
        key = (
            str(item.message),
            item.category.__name__,
            pathlib.Path(item.filename).name,
            item.lineno,
        )
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            "message": message,
            "category": category,
            "source_file": source_file,
            "line": line,
            "count": count,
        }
        for (message, category, source_file, line), count in sorted(counts.items())
    ]


def execute_with_warning_capture(
    task_index: int,
    function: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[int, str, object, list[dict[str, object]]]:
    """Execute one real process-worker task and return its warning records."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function(*args, **kwargs)
    return task_index, function.__qualname__, result, warning_records(caught)


def initialize_worker_stderr(stderr_path: str) -> None:
    """Redirect one worker's import/runtime stderr to its deterministic evidence file."""
    sys.stderr = open(stderr_path, "w", buffering=1, encoding="utf-8")  # noqa: SIM115


class WarningCapturedFuture:
    """Unwrap a worker result while retaining its warning records in the parent."""

    def __init__(
        self,
        future: object,
        warning_sink: dict[int, dict[str, object]],
    ) -> None:
        self._future = future
        self._warning_sink = warning_sink

    def result(self) -> object:
        """Return the upstream result and persist keyed warning evidence."""
        task_index, function_name, result, records = self._future.result()  # type: ignore[attr-defined]
        self._warning_sink[task_index] = {
            "task_index": task_index,
            "function": function_name,
            "warnings": records,
        }
        return result


class WarningCapturingProcessPoolExecutor(ProcessPoolExecutor):
    """A real one-process executor that transports child warnings to the parent."""

    def __init__(self, case_name: str) -> None:
        self.worker_key = f"{case_name}:worker-0"
        self.stderr_path = pathlib.Path(tempfile.gettempdir()) / (
            f"phenotypic-a10-filfinder-{case_name}-worker-0.stderr"
        )
        self.stderr_path.unlink(missing_ok=True)
        super().__init__(
            max_workers=1,
            initializer=initialize_worker_stderr,
            initargs=(str(self.stderr_path),),
        )
        self._next_task_index = 0
        self.warning_records_by_task: dict[int, dict[str, object]] = {}

    def stderr_records(self) -> list[dict[str, object]]:
        """Return deterministic keyed stderr, or no record if no worker started."""
        if not self.stderr_path.exists():
            return []
        return [
            {
                "worker": self.worker_key,
                "stderr_lines": self.stderr_path.read_text(
                    encoding="utf-8"
                ).splitlines(),
            }
        ]

    def submit(  # type: ignore[override]
        self,
        function: Callable[..., Any],
        /,
        *args: object,
        **kwargs: object,
    ) -> WarningCapturedFuture:
        """Submit a source task through the child warning-capture trampoline."""
        task_index = self._next_task_index
        self._next_task_index += 1
        future = super().submit(
            execute_with_warning_capture,
            task_index,
            function,
            args,
            kwargs,
        )
        return WarningCapturedFuture(future, self.warning_records_by_task)


def paint_square(image: np.ndarray, row: int, col: int, value: float) -> None:
    """Paint a clipped 3-by-3 square around one path coordinate."""
    row_start = max(row - 1, 0)
    row_stop = min(row + 2, image.shape[0])
    col_start = max(col - 1, 0)
    col_stop = min(col + 2, image.shape[1])
    image[row_start:row_stop, col_start:col_stop] = value


def synthetic_cases() -> dict[str, np.ndarray]:
    """Return deterministic masks covering the wrapper-visible topology cases."""
    straight = np.zeros((17, 17), dtype=np.float64)
    straight[3:14, 7:10] = 0.8
    straight[8, 8] = 1.0

    y_spur = np.zeros((21, 21), dtype=np.float64)
    for row in range(4, 17):
        paint_square(y_spur, row, 10, 0.75)
    for offset in range(0, 7):
        paint_square(y_spur, 10 - offset, 10 - offset, 0.9)
        paint_square(y_spur, 10 - offset, 10 + offset, 0.65)

    disconnected = np.zeros((21, 21), dtype=np.float64)
    disconnected[3:16, 3:6] = 0.7
    disconnected[6:19, 15:18] = 0.85

    loop_branch = np.zeros((25, 25), dtype=np.float64)
    loop_branch[5:8, 5:19] = 0.8
    loop_branch[17:20, 5:19] = 0.8
    loop_branch[5:20, 5:8] = 0.8
    loop_branch[5:20, 16:19] = 0.8
    loop_branch[11:14, 18:23] = 0.6

    noise = np.fromfunction(
        lambda row, col: ((row * 17 + col * 11) % 47) / 100.0,
        (19, 23),
        dtype=int,
    ).astype(np.float64)
    noise[2:5, 2:5] = 0.55
    noise[8:11, 11:16] = 0.65
    noise[14:17, 19:22] = 0.75

    symmetric_tie = np.zeros((21, 21), dtype=np.float64)
    for row in range(10, 18):
        paint_square(symmetric_tie, row, 10, 0.8)
    for offset in range(0, 7):
        paint_square(symmetric_tie, 10 - offset, 10 - offset, 0.8)
        paint_square(symmetric_tie, 10 - offset, 10 + offset, 0.8)

    threshold_boundary = np.zeros((7, 9), dtype=np.float32)
    threshold_boundary[2:5, 2] = np.nextafter(
        np.float32(THRESHOLD), np.float32(0.0)
    )
    threshold_boundary[2:5, 4] = THRESHOLD
    threshold_boundary[2:5, 6] = np.nextafter(
        np.float32(THRESHOLD), np.float32(1.0)
    )

    empty = np.full(
        (7, 9),
        np.nextafter(np.float32(THRESHOLD), np.float32(0.0)),
        dtype=np.float32,
    )

    return {
        "straight": straight,
        "y_spur": y_spur,
        "disconnected": disconnected,
        "loop_branch": loop_branch,
        "noise": noise,
        "symmetric_tie": symmetric_tie,
        "threshold_boundary": threshold_boundary,
        "empty": empty,
    }


def encode_array(array: np.ndarray) -> list[object]:
    """Convert an oracle array into JSON-native nested lists."""
    return np.asarray(array).tolist()


def label_eight_connected(mask: np.ndarray) -> np.ndarray:
    """Label a selected raster in deterministic row-major component order."""
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return labels.astype(np.int64, copy=False)


def warning_record_counts(records: list[dict[str, object]]) -> dict[str, int]:
    """Count messages in already serialized warning evidence."""
    counts: dict[str, int] = {}
    for item in records:
        message = str(item["message"])
        counts[message] = counts.get(message, 0) + int(item["count"])
    return dict(sorted(counts.items()))


def validate_warning_evidence(
    record: dict[str, object],
    *,
    empty: bool,
) -> None:
    """Reject a fixture whose process-local warning evidence is incomplete."""
    raw_records = record["create_mask_warning_records"]
    expected_raw = [
        {
            "message": EXPECTED_SUPPLIED_MASK_WARNING,
            "category": "UserWarning",
            "source_file": "filfinder2D.py",
            "line": 313,
            "count": 1,
        }
    ]
    if raw_records != expected_raw:
        raise AssertionError(f"unexpected create_mask warnings: {raw_records!r}")

    visible_records = record["adapter_policy_visible_warning_records"]
    if not isinstance(visible_records, list) or len(visible_records) != 1:
        raise AssertionError(
            f"warning filter hid or added a warning: {visible_records!r}"
        )
    visible = visible_records[0]
    if (
        visible["message"] != WARNING_POLICY_CONTROL
        or visible["category"] != "UserWarning"
        or visible["source_file"] != "generate_fixture.py"
        or visible["count"] != 1
    ):
        raise AssertionError(f"warning policy control changed: {visible!r}")

    if record["analyze_skeleton_parent_warning_records"] != []:
        raise AssertionError("process-worker warnings leaked into the parent channel")

    worker_records = record["analyze_skeleton_worker_warning_records"]
    stderr_records = record["process_worker_stderr_records"]
    if empty:
        if worker_records != [] or stderr_records != []:
            raise AssertionError("empty input unexpectedly started a process worker")
        return

    case_name = str(record["name"])
    expected_task_count = EXPECTED_TASK_COUNTS[case_name]
    if not isinstance(worker_records, list) or len(worker_records) != expected_task_count:
        raise AssertionError(
            f"{case_name}: worker task count changed: {worker_records!r}"
        )
    for task_index, task in enumerate(worker_records):
        if task["task_index"] != task_index:
            raise AssertionError(f"{case_name}: worker task order changed: {task!r}")
        if task["function"] != "Filament2D.skeleton_analysis":
            raise AssertionError(f"{case_name}: worker function changed: {task!r}")
        warning_counts = warning_record_counts(task["warnings"])
        if warning_counts.get(GRAPH_PRUNING_WARNING) != 1:
            raise AssertionError(
                f"{case_name}: graph-pruning warning missing from task {task_index}"
            )
    if stderr_records != [
        {
            "worker": f"{case_name}:worker-0",
            "stderr_lines": EXPECTED_WORKER_STDERR_LINES,
        }
    ]:
        raise AssertionError(
            f"{case_name}: import-time worker stderr changed: {stderr_records!r}"
        )


def analyze_case(name: str, image: np.ndarray) -> dict[str, object]:
    """Capture every wrapper-visible stage from one fresh FilFinder object."""
    import astropy.units as u
    from fil_finder import FilFinder2D

    # ImageData stores detect_mat as float32. The adapter then copies those
    # quantized values into a float64 source buffer before calling FilFinder.
    image = np.asarray(image, dtype=np.float32).astype(np.float64)
    threshold_mask = image >= THRESHOLD
    record: dict[str, object] = {
        "name": name,
        "image": encode_array(image),
        "threshold_mask": encode_array(threshold_mask.astype(np.uint8)),
    }

    with WarningCapturingProcessPoolExecutor(name) as pool:
        filfinder = FilFinder2D(
            image.copy(),
            beamwidth=BEAMWIDTH_PX * u.pix,
            mask=threshold_mask.copy(),
            pool=pool,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            filfinder.create_mask(use_existing_mask=True)
        raw_mask_warnings = warning_records(caught)
        record["create_mask_warning_records"] = raw_mask_warnings
        record["create_mask_warning_counts"] = warning_record_counts(
            raw_mask_warnings
        )
        with warnings.catch_warnings(record=True) as policy_caught:
            warnings.simplefilter("always")
            warnings.filterwarnings(
                "ignore",
                message=f"^{re.escape(EXPECTED_SUPPLIED_MASK_WARNING)}$",
                category=UserWarning,
            )
            filfinder.create_mask(use_existing_mask=True)
            warnings.warn(WARNING_POLICY_CONTROL, UserWarning)
        record["adapter_suppressed_warning"] = EXPECTED_SUPPLIED_MASK_WARNING
        record["adapter_policy_visible_warning_records"] = warning_records(
            policy_caught
        )
        record["filfinder_mask"] = encode_array(
            np.asarray(filfinder.mask, dtype=np.uint8)
        )
        record["mask_labels_8_connected"] = encode_array(
            label_eight_connected(filfinder.mask)
        )

        if not threshold_mask.any():
            record.update(
                {
                    "empty_short_circuit": True,
                    "medial_axis_distance_px": None,
                    "skeleton_pre_prune": None,
                    "skeleton_post_prune": None,
                    "skeleton_longest_path": None,
                    "skeleton_labels_8_connected": None,
                    "longest_path_labels_8_connected": None,
                    "filament_lengths_px": [],
                    "branch_lengths_px": [],
                    "analyze_skeleton_parent_warning_records": [],
                    "analyze_skeleton_warning_counts": {},
                    "analyze_skeleton_worker_warning_records": [],
                    "process_worker_stderr_records": pool.stderr_records(),
                }
            )
            validate_warning_evidence(record, empty=True)
            return record

        filfinder.medskel(rng=RNG_SEED)
        skeleton_pre_prune = np.asarray(filfinder.skeleton, dtype=bool).copy()
        record["empty_short_circuit"] = False
        record["medial_axis_distance_px"] = encode_array(
            np.asarray(filfinder.medial_axis_distance.value, dtype=np.float64)
        )
        record["skeleton_pre_prune"] = encode_array(
            skeleton_pre_prune.astype(np.uint8)
        )
        record["skeleton_labels_8_connected"] = encode_array(
            label_eight_connected(skeleton_pre_prune)
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            branch_threshold = (
                None
                if BRANCH_THRESHOLD_PX is None
                else BRANCH_THRESHOLD_PX * u.pix
            )
            filfinder.analyze_skeletons(
                prune_criteria="all",
                relintens_thresh=RELATIVE_INTENSITY_THRESHOLD,
                skel_thresh=1.0 * u.pix,
                branch_thresh=branch_threshold,
                max_prune_iter=MAX_PRUNE_ITERATIONS,
            )
        parent_warnings = warning_records(caught)
        record["analyze_skeleton_parent_warning_records"] = parent_warnings
        record["analyze_skeleton_warning_counts"] = warning_record_counts(
            parent_warnings
        )
        record["analyze_skeleton_worker_warning_records"] = [
            pool.warning_records_by_task[index]
            for index in sorted(pool.warning_records_by_task)
        ]
        record["skeleton_post_prune"] = encode_array(
            np.asarray(filfinder.skeleton, dtype=np.uint8)
        )
        record["skeleton_longest_path"] = encode_array(
            np.asarray(filfinder.skeleton_longpath, dtype=np.uint8)
        )
        record["longest_path_labels_8_connected"] = encode_array(
            label_eight_connected(filfinder.skeleton_longpath)
        )
        record["filament_lengths_px"] = [
            float(filament.length(u.pix).value) for filament in filfinder.filaments
        ]
        record["branch_lengths_px"] = [
            [float(value.value) for value in lengths]
            for lengths in filfinder.branch_properties["length"]
        ]
        record["effective_skeleton_threshold_px"] = float(
            filfinder.skel_thresh.value
        )
        record["effective_branch_threshold_px"] = int(
            filfinder.branch_thresh.value
        )
        record["process_worker_stderr_records"] = pool.stderr_records()
        validate_warning_evidence(record, empty=False)
        return record


def dependency_versions() -> dict[str, str]:
    """Return the full pinned oracle dependency vector."""
    names = (
        "fil-finder",
        "astropy",
        "numpy",
        "networkx",
        "scipy",
        "matplotlib",
        "scikit-image",
        "h5py",
        "skan",
        "numba",
        "pandas",
    )
    return {name: importlib.metadata.version(name) for name in names}


def verify_authoritative_source() -> None:
    """Fail unless the oracle imports the committed sdist source file."""
    from fil_finder import FilFinder2D

    actual = pathlib.Path(inspect.getfile(FilFinder2D)).resolve()
    if actual != EXPECTED_SOURCE.resolve():
        raise RuntimeError(
            "oracle did not import the committed FilFinder source: "
            f"{actual} != {EXPECTED_SOURCE.resolve()}"
        )


def generate_filfinder_fixture() -> None:
    """Regenerate the deterministic FilFinder 1.8 fixture JSON."""
    verify_authoritative_source()
    fixture = {
        "schema_version": 3,
        "authority": "fil-finder 1.8 sdist and v1.8 tag commit",
        "parameters": {
            "threshold": THRESHOLD,
            "threshold_comparison": ">=",
            "beamwidth_px": BEAMWIDTH_PX,
            "prune_criteria": "all",
            "relative_intensity_threshold": RELATIVE_INTENSITY_THRESHOLD,
            "skeleton_threshold_px": 1.0,
            "branch_threshold_px": BRANCH_THRESHOLD_PX,
            "max_prune_iterations": MAX_PRUNE_ITERATIONS,
            "rng_seed": RNG_SEED,
            "label_connectivity": 8,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": dependency_versions(),
        },
        "cases": [
            analyze_case(name, image) for name, image in synthetic_cases().items()
        ],
    }
    FIXTURE_PATH.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    generate_filfinder_fixture()
