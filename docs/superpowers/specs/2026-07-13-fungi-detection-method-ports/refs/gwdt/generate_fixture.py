"""Build the pinned Vaa3D harness and regenerate the complete GWDT fixture."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np


IMAGE = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 2, 8, 3, 0],
        [0, 4, 1, 7, 0],
        [0, 6, 5, 9, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)
DIAGONAL_IMAGE = np.array([[0, 100], [100, 1]], dtype=np.uint8)
THRESHOLD_IMAGE = np.array([[1, 2, 5]], dtype=np.uint8)
ALL_BACKGROUND_IMAGE = np.array([[1, 2]], dtype=np.uint8)
NO_BACKGROUND_IMAGE = np.array([[1, 2]], dtype=np.uint8)
POST_FRONTIER_DIAGONAL_IMAGE = np.array(
    [[0, 100, 100], [100, 1, 100], [100, 100, 1]],
    dtype=np.uint8,
)


def _parse_harness_output(
    path: Path,
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    parsed: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        label, *values = line.split()
        dtype = np.float32 if label == "DISTANCE" else np.float64
        parsed[label.lower()] = np.asarray(values, dtype=dtype).reshape(shape)
    if "distance" not in parsed:
        raise RuntimeError(f"incomplete harness output: {sorted(parsed)}")
    return parsed


def generate_gwdt_fixture(output: Path) -> None:
    """Compile the source harness and write all observable source outputs.

    Args:
        output: Destination ``.npz`` path.
    """
    reference_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="phenotypic-gwdt-") as temporary:
        temporary_dir = Path(temporary)
        executable = temporary_dir / "source_harness"
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-O0",
                "-I",
                str(reference_dir),
                str(reference_dir / "source_harness.cpp"),
                "-o",
                str(executable),
            ],
            check=True,
        )
        outputs: dict[str, np.ndarray] = {}
        cases = (
            ("standard", IMAGE, 0),
            ("diagonal", DIAGONAL_IMAGE, 0),
            ("threshold", THRESHOLD_IMAGE, 2),
            ("all_background", ALL_BACKGROUND_IMAGE, 2),
            ("no_background", NO_BACKGROUND_IMAGE, 0),
            ("post_frontier_diagonal", POST_FRONTIER_DIAGONAL_IMAGE, 0),
        )
        for case_name, case_image, _ in cases:
            for connectivity, cnn_type in ((4, 1), (8, 2)):
                harness_output = temporary_dir / f"source_{case_name}_{connectivity}.txt"
                subprocess.run(
                    [str(executable), case_name, str(cnn_type), str(harness_output)],
                    check=True,
                )
                for name, values in _parse_harness_output(
                    harness_output,
                    case_image.shape,
                ).items():
                    outputs[f"source_{case_name}_{name}_{connectivity}"] = values

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        image=IMAGE,
        background=IMAGE == 0,
        diagonal_image=DIAGONAL_IMAGE,
        diagonal_background=DIAGONAL_IMAGE == 0,
        threshold_image=THRESHOLD_IMAGE,
        threshold_background=THRESHOLD_IMAGE <= 2,
        all_background_image=ALL_BACKGROUND_IMAGE,
        all_background=ALL_BACKGROUND_IMAGE <= 2,
        no_background_image=NO_BACKGROUND_IMAGE,
        no_background=NO_BACKGROUND_IMAGE <= 0,
        post_frontier_diagonal_image=POST_FRONTIER_DIAGONAL_IMAGE,
        post_frontier_diagonal_background=POST_FRONTIER_DIAGONAL_IMAGE == 0,
        **outputs,
    )


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/reconnect/gwdt/app2_source.npz"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    generate_gwdt_fixture(arguments.output)
