"""RAW inputs carry a decode revision in their work id; nothing else changes.

Spec 2026-09-24-cli-preflight §12 (D8, D11; review R3, R18). The revision lives
inside ``compute_work_id`` so both producers -- ``work_id_for_image`` and the
SLURM worker's ``_worker_work_identity`` -- agree; the worker refuses any
mismatch ("SLURM task work identity does not match worklist").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from phenotypic import ImagePipeline
from phenotypic._cli._cli_failure_tracker import compute_work_id, work_id_for_image
from phenotypic._cli._cli_process_single import _worker_work_identity
from phenotypic.detect import OtsuDetector
from tests.unit.cli._preflight_support import make_config

_FIXED = dict(
    dataset="plate1",
    input_sha256="a" * 64,
    pipeline_fingerprint="b" * 64,
    processing_config_digest="c" * 64,
    mode="full",
)

#: compute_work_id(**_FIXED, relative_image_path=...) at 81d19ec (computed there,
#: pasted here; computing them in the test would be tautological, review R18).
TIFF_DIGEST_AT_81D19EC = "0397ca238f767b76d109685e63db2f55a5cb128dc3f1829a0aa19a1716b7df51"
NEF_DIGEST_AT_81D19EC = "6e81305446a7076d5b13f69c23d320ffcde81bff0b0e2fbdf3844fcff563871c"


def test_a_non_raw_work_id_is_byte_identical_to_before() -> None:
    assert compute_work_id(relative_image_path="plate1/img001.tiff", **_FIXED) == TIFF_DIGEST_AT_81D19EC


def test_a_raw_work_id_changes() -> None:
    assert compute_work_id(relative_image_path="plate1/img001.nef", **_FIXED) != NEF_DIGEST_AT_81D19EC


@pytest.mark.parametrize("name", ["img001.nef", "img001.tiff"])
def test_both_producers_agree(name: str, tmp_path: Path) -> None:
    root = tmp_path / "images"
    image = root / "plate1" / name
    image.parent.mkdir(parents=True)
    if name.endswith(".tiff"):
        tifffile.imwrite(image, np.zeros((4, 4), dtype=np.uint8))
    else:
        image.write_bytes(b"raw bytes; the id hashes them, it never decodes them")
    pipeline = tmp_path / "p.json"
    pipeline.write_text(ImagePipeline(ops={"d": OtsuDetector()}).to_json(), encoding="utf-8")
    config = make_config(pipeline_json=pipeline, input_path=root)

    selected, _ = work_id_for_image(config, "plate1", image)
    worker, _ = _worker_work_identity(
        pipeline=pipeline, image=image, input_root=root, dataset_name="plate1",
        image_type=config.image_type, nrows=config.nrows, ncols=config.ncols,
        bit_depth=config.bit_depth, detect_mode=config.detect_mode, layer=None,
        ext=config.ext, process_format=config.process_format,
        include_dataset_column=config.include_dataset_column,
        overlay_alpha=config.overlay_alpha, save_overlays=config.save_overlays,
        drop_originals=config.drop_originals, mode="full",
    )

    assert selected == worker
