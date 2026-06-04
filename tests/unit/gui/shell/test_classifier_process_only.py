"""Process-only runs are discoverable via their ``.phenotypic`` manifest.

A process-only CLI run writes mirrored layer files + a progress manifest under
``.phenotypic/progress/manifest.json`` but emits NO ``results/`` or
``deliverables/``. The classifier must still surface it (D13) via a dedicated
``is_process_only_output`` capability so the run console lists it, without
flagging a dashboard / results affordance.
"""

from pathlib import Path

import pytest

from phenotypic.gui.shell._classifier import classify, invalidate_cache
from phenotypic.tools_ import manifest_json_path


@pytest.fixture(autouse=True)
def _flush_classifier_cache() -> None:
    invalidate_cache()


def test_process_only_run_is_discoverable(tmp_path: Path) -> None:
    # Process-only run: mirrored layer + .phenotypic/progress/manifest.json,
    # no results/deliverables.
    (tmp_path / "day1").mkdir()
    (tmp_path / "day1" / "plateA_detect_mat.tiff").write_bytes(b"II*\x00")
    mp = manifest_json_path(tmp_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text('{"is_complete": true}', encoding="utf-8")
    caps = classify(tmp_path)
    assert caps.is_process_only_output is True
    assert caps.is_cli_output is False  # not a full forward run
    assert caps.has_dashboard is False


def test_forward_run_not_flagged_process_only(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    deliv = tmp_path / "deliverables"
    deliv.mkdir()
    (deliv / "master_measurements.parquet").write_bytes(b"x")
    caps = classify(tmp_path)
    assert caps.is_cli_output is True
    assert caps.is_process_only_output is False
