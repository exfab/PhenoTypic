"""The run preflight over a pipeline whose ``filters``/``model`` slots are filled.

``walk_operations`` yields every slot entry, including ``SetAnalyzer`` and
``ModelFitter`` instances, which are not ``BaseOperation`` subclasses and carry
no ``preflight_requirements``. Every requirement-reading check must treat such
a node as declaring nothing rather than crash: seen in a real run on the GUI
tutorial pipeline, where five checks reported ``PF-CHECK-CRASHED`` with
``'TukeyOutlierRemover' object has no attribute 'preflight_requirements'``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from phenotypic import ImagePipeline
from phenotypic._cli._cli_preflight import (
    check_grid_image,
    check_model_licenses,
    check_model_weights_cached,
    check_optional_modules,
    check_rgb_ops_on_gray,
    run_preflight,
)
from phenotypic.abc_ import OperationRequirements, WeightRequirement
from phenotypic.analysis import LogGrowthModel, TukeyOutlierRemover
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize
from phenotypic.refine import RemoveGridOutliers
from tests.unit.cli._preflight_support import make_context, make_datasets

#: The checks that read ``preflight_requirements()`` from every in-scope node.
REQUIREMENT_CHECKS = (
    check_grid_image,
    check_optional_modules,
    check_model_licenses,
    check_model_weights_cached,
    check_rgb_ops_on_gray,
)


class _Needy(OtsuDetector):
    """A detector declaring one of everything, so each check has a real finding."""

    def preflight_requirements(self):
        return OperationRequirements(
            rgb_input=True,
            modules=("phenotypic_no_such_module_for_preflight",),
            weights=(
                WeightRequirement(
                    model="demo:weights",
                    license_key="demo-preflight-license",
                    is_cached=lambda: False,
                ),
            ),
        )


def _analysis_pipeline(**ops) -> ImagePipeline:
    return ImagePipeline(
        ops=ops,
        meas={"s": MeasureSize()},
        filters={"t": TukeyOutlierRemover(on="Size_Area", groupby=["Metadata_Plate"])},
        model=LogGrowthModel(on="Size_Area", groupby=["Metadata_Plate"]),
    )


def _gray_input(tmp_path: Path) -> Path:
    path = tmp_path / "plate1" / "gray.tif"
    path.parent.mkdir(parents=True)
    tifffile.imwrite(path, np.zeros((16, 16), dtype=np.uint8))
    return path


def test_filters_and_model_slots_crash_no_requirement_check(
    tmp_path: Path, monkeypatch
) -> None:
    """The ``ops`` findings still appear; the analysis slots add no crash."""
    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    pipeline = _analysis_pipeline(d=_Needy(), g=RemoveGridOutliers())
    context = make_context(
        pipeline,
        datasets=make_datasets(_gray_input(tmp_path)),
        image_type="Image",
    )

    report = run_preflight(context, checks=REQUIREMENT_CHECKS)

    codes = [finding.code for finding in report.findings]
    assert "PF-CHECK-CRASHED" not in codes, [f.message for f in report.findings]
    assert sorted(codes) == sorted(
        [
            "PF-GRID-IMAGE",
            "PF-MISSING-MODULE",
            "PF-LICENSE",
            "PF-WEIGHTS-UNCACHED",
            "PF-RGB-OP-GRAY",
        ]
    )
    for finding in report.findings:
        assert "filters:t" not in finding.message
        assert "model:LogGrowthModel" not in finding.message


@pytest.mark.parametrize("mode", ["full", "measure"])
def test_analysis_slots_alone_produce_no_requirement_findings(mode: str) -> None:
    """``measure`` mode runs only ``meas``/``post``/``filters``/``model``."""
    context = make_context(_analysis_pipeline(d=OtsuDetector()), mode)

    report = run_preflight(context, checks=REQUIREMENT_CHECKS)

    assert report.findings == ()
