"""The results-viewer QC tab needs an ``OperationRegistry`` to add checks.

Regression guard for the bug where the viewer's ``create_app`` never
stashed an ``OperationRegistry`` on ``app.server.config``. Because each
sub-app runs on its own Flask server under ``DispatcherMiddleware``, the
QC tab's ``_get_registry()`` (which reads ``CFG_OPERATION_REGISTRY`` from
the *viewer's* config) returned ``None`` in both standalone and hub
launches. That left the Add-check modal's class picker empty, so no QC
check could ever be added and the QC tab showed no cards.

These tests assert the viewer factory provides a registry whose
``quality_check`` category lists the shipping checks.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from PIL import Image as PILImage

from phenotypic.gui._config import CFG_OPERATION_REGISTRY
from phenotypic.gui._operation_registry import OperationRegistry
from phenotypic.gui.results_viewer._app import create_app
from phenotypic.gui.results_viewer._output_root import OutputRoot

from tests._output_layout import write_master, write_measurements_mirror
from phenotypic.schema import METADATA


@pytest.fixture()
def output_root(tmp_path: Path) -> OutputRoot:
    """A minimal CLI output dir that ``OutputRoot.discover`` accepts."""
    master = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["d1"] * 3,
            str(METADATA.IMAGE_NAME): ["img-1"] * 3,
            "Object_Label": [1, 2, 3],
            "Size_Area": [100.0, 101.0, 102.0],
        }
    )
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)

    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True, exist_ok=True)
    overlays = tmp_path / "deliverables" / "overlays" / "d1"
    overlays.mkdir(parents=True)
    PILImage.new("RGB", (120, 120), (200, 0, 0)).save(overlays / "img-1.png")

    return OutputRoot.discover(tmp_path)


def test_create_app_stashes_operation_registry(output_root) -> None:
    """The viewer factory makes an ``OperationRegistry`` available."""
    app = create_app(output_root=output_root)
    registry = app.server.config.get(CFG_OPERATION_REGISTRY)
    assert isinstance(registry, OperationRegistry)


def test_registry_lists_quality_checks(output_root) -> None:
    """The QC tab's class picker source carries every shipping check."""
    app = create_app(output_root=output_root)
    registry = app.server.config[CFG_OPERATION_REGISTRY]
    names = {info.name for info in registry.get_by_category("quality_check")}
    # The shipping checks (Count, Occupancy, ICC, ZMax, MAD, SE, Tukey).
    expected = {
        "ExpectedVsDetectedCount",
        "GridOccupancy",
        "ICC",
        "MaxModifiedZScore",
        "RelativeMAD",
        "ReplicateAgreement",
        "TukeyOutlierFraction",
    }
    assert expected <= names
