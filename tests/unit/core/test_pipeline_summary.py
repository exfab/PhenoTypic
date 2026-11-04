"""``ImagePipeline.summary()``: one row per step, before any image is loaded.

Plan ``docs/superpowers/plans/2026-10-31-docs-reorganization/03-small-code-fixes.md``
Task 5; design §10 item 10. ``changed`` comes from
``model_dump(exclude_defaults=True)``, so a parameter set to its own default is
not "changed" (``BlurGauss``'s default ``sigma`` is 2.0, K0-02). Paths are the
operation tree's own spelling (``sdk_._operation_tree.walk_operations``), so a
list of ops is keyed by class name (K0-18).
"""

from __future__ import annotations

import numpy as np

from phenotypic import ImagePipeline
from phenotypic.detect import CompositeDetector, FilamentousFungiDetector, OtsuDetector
from phenotypic.enhance import BlurGauss, SetDetectMode
from phenotypic.grid import ManualGridFinder
from phenotypic.measure import MeasureShape, MeasureSize
from phenotypic.refine import SmallObjectRemover

COLUMNS = ["step", "slot", "operation", "module", "writes", "changed"]


def test_summary_lists_steps_and_layers():
    p = ImagePipeline(
        ops=[BlurGauss(sigma=3.0), OtsuDetector(), SmallObjectRemover()],
        meas=[MeasureSize()],
    )
    s = p.summary()
    assert list(s.columns) == COLUMNS
    assert list(s["operation"]) == ["BlurGauss", "OtsuDetector", "SmallObjectRemover", "MeasureSize"]
    assert list(s["slot"]) == ["ops", "ops", "ops", "meas"]
    assert list(s["module"]) == ["enhance", "detect", "refine", "measure"]
    assert list(s["writes"]) == ["detect_mat", "objmask, objmap", "objmask, objmap", "table"]
    assert s.loc[s["operation"] == "BlurGauss", "changed"].item() == "sigma=3.0"  # 2.0 is the default (K0-02)
    assert s.loc[s["operation"] == "OtsuDetector", "changed"].item() == ""


def test_summary_expands_nested_pipelines():
    inner = ImagePipeline(ops=[BlurGauss(sigma=1.0)])
    s = ImagePipeline(ops={"inner": inner, "detect": OtsuDetector()}).summary()
    assert s["step"].iloc[0] == "inner/BlurGauss"  # list ops are keyed by class name (K0-18)
    # The nested pipeline is expanded, never listed as one opaque row.
    assert list(s["step"]) == ["inner/BlurGauss", "detect"]
    assert "ImagePipeline" not in set(s["operation"])
    assert s["changed"].iloc[0] == "sigma=1.0"


def test_a_nested_pipelines_measurers_are_not_listed():
    """A pipeline inside ``ops`` runs only its own ``ops`` (``apply``), so its
    ``meas`` never run and must not appear as steps."""
    inner = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape()])
    s = ImagePipeline(ops={"inner": inner}, meas=[MeasureSize()]).summary()
    assert list(s["step"]) == ["inner/OtsuDetector", "meas:MeasureSize"]
    assert list(s["slot"]) == ["ops", "meas"]


def test_a_composite_is_one_row_with_its_children_in_changed():
    """Only pipelines are expanded; a composite's children are its parameters."""
    s = ImagePipeline(ops=[CompositeDetector(ops=[OtsuDetector(ignore_zeros=True)])]).summary()
    assert list(s["step"]) == ["CompositeDetector"]
    assert s["writes"].item() == "objmask, objmap"
    assert s["changed"].item() == "ops=[OtsuDetector(ignore_zeros=True)]"


def test_set_detect_mode_reports_the_layer_it_rewrites():
    """``SetDetectMode`` subclasses ``ImageOperation`` directly, yet replaces
    ``detect_mat``; a blank ``writes`` would say it writes nothing."""
    s = ImagePipeline(ops=[SetDetectMode(mode="red")]).summary()
    assert s["writes"].item() == "detect_mat"
    assert s["changed"].item() == "mode='red'"


def test_an_empty_pipeline_has_the_columns_and_no_rows():
    s = ImagePipeline().summary()
    assert list(s.columns) == COLUMNS
    assert len(s) == 0


def test_the_grid_finder_a_preset_adds_is_listed():
    """``measure()`` prepends a ``CenteredAutoGridFinder`` when both ``nrows``
    and ``ncols`` are set; it runs, so it is a step (review W2 M1)."""
    p = ImagePipeline(
        ops=[BlurGauss(sigma=3.0), OtsuDetector()], meas=[MeasureSize()], nrows=16, ncols=24
    )
    s = p.summary()
    assert list(s["step"]) == ["BlurGauss", "OtsuDetector", "meas:CenteredAutoGridFinder", "meas:MeasureSize"]
    assert list(s["step"]) == ["BlurGauss", "OtsuDetector"] + [f"meas:{k}" for k in p._build_measurement_run_order()]
    finder = s[s["operation"] == "CenteredAutoGridFinder"]
    assert finder["slot"].item() == "meas"
    assert finder["writes"].item() == "table"
    assert finder["changed"].item() == "nrows=16, ncols=24"
    # No preset, no finder; and a finder already in meas is not doubled.
    assert "CenteredAutoGridFinder" not in set(ImagePipeline(meas=[MeasureSize()]).summary()["operation"])
    assert ImagePipeline(meas=[MeasureSize()], nrows=16).summary()["step"].tolist() == ["meas:MeasureSize"]


def test_values_a_validator_derives_are_not_changes():
    """``FilamentousFungiDetector`` fills eight ``None``-default fields from the
    others; a default-constructed one has changed nothing (review W2 D2)."""
    s = ImagePipeline(ops=[FilamentousFungiDetector()]).summary()
    assert s["changed"].item() == ""
    s = ImagePipeline(ops=[FilamentousFungiDetector(gauss_sigma=50.0)]).summary()
    assert s["changed"].item() == "gauss_sigma=50.0"


def test_a_pipeline_in_a_field_hides_its_generated_name():
    """A pipeline's ``name`` defaults to a fresh uuid4; only a chosen one shows (W2 S2)."""
    held = FilamentousFungiDetector(inoculum_detector=ImagePipeline(ops=[OtsuDetector()]))
    changed = ImagePipeline(ops=[held]).summary()["changed"].item()
    assert changed == "inoculum_detector=ImagePipeline(ops={'OtsuDetector': OtsuDetector()})"
    named = FilamentousFungiDetector(inoculum_detector=ImagePipeline(name="inoculum", ops=[OtsuDetector()]))
    assert "name='inoculum'" in ImagePipeline(ops=[named]).summary()["changed"].item()


def test_a_class_that_needs_arguments_falls_back_to_declared_defaults():
    finder = ManualGridFinder(row_edges=np.array([0, 10, 20]), col_edges=np.array([0, 10, 20, 30]))
    changed = ImagePipeline(meas=[finder]).summary()["changed"].item()
    assert "row_edges=" in changed and "col_edges=" in changed
