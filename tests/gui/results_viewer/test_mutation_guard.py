"""O4 contracts for fail-closed Results and Analysis mutations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import polars as pl
import pytest

from phenotypic._core._image_pipeline import ImagePipeline
from phenotypic.analysis import ReplicateAgreement
from phenotypic.gui._binding_generation import (
    BINDING_GENERATION_PAYLOAD_KEY,
)
from phenotypic.gui.analysis import _ids as analysis_ids
from phenotypic.gui.analysis._layout import (
    build_app_layout as build_analysis_layout,
)
from phenotypic.gui.analysis._recipe_state import RecipeState
from phenotypic.gui.results_viewer import _ids as viewer_ids
from phenotypic.gui.results_viewer._app import create_app as create_results_app
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._error_tab import _ids as error_ids
from phenotypic.gui.results_viewer._layout import (
    build_app_layout as build_results_layout,
)
from phenotypic.gui.results_viewer._mutation_guard import (
    OutputMutationBlocked,
    OutputMutationGuard,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._qc_tab import _ids as qc_ids
from phenotypic.gui.results_viewer._qc_tab.review import _ids as review_ids
from phenotypic.gui.results_viewer.colony_view._grid import build_grid
from phenotypic.gui.results_viewer._viewer_card import (
    layout as build_viewer_card,
)
from phenotypic.schema import METADATA
from phenotypic.plotting import PlotOutput, PlotPage, publish_plot_output
from phenotypic.plotting._writer import PlotPublicationBlocked
from phenotypic.sdk_ import (
    gui_launch_owner_path,
    master_measurements_parquet_path,
    measurements_parquet_path,
    pipeline_json_path,
    resolve_manifest_json_path,
    run_completion_marker_path,
)
from phenotypic.sdk_._qc_recipe import QcRecipeEntry
from phenotypic.sdk_._qc_recipe._runner import (
    QcPublicationBlocked,
    run_qc,
)


def _seed_output(
    root: Path,
    *,
    contradictory: bool,
    overlay_count: int = 2,
    pipeline: ImagePipeline | None = None,
) -> None:
    """Write a small scientific payload with configurable completion evidence."""
    frame = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["plate", "plate"],
            str(METADATA.IMAGE_NAME): ["a", "b"],
            "Object_Label": [1, 2],
            "Centroid": [[5.0, 5.0], [6.0, 6.0]],
            "Metadata_Row": ["A", "A"],
            "Metadata_Column": [1, 2],
            "Size_Area": [10.0, 20.0],
        }
    )
    master = master_measurements_parquet_path(root)
    master.parent.mkdir(parents=True)
    frame.write_parquet(master)
    frame.write_parquet(measurements_parquet_path(root))
    pipeline_json_path(root).write_text(
        (
            pipeline
            if pipeline is not None
            else ImagePipeline(name="mutation-guard")
        ).to_json()
        or "{}",
        encoding="utf-8",
    )
    overlays = root / "deliverables" / "overlays" / "plate"
    overlays.mkdir(parents=True)
    for index in range(overlay_count):
        (overlays / f"image-{index}.png").write_bytes(b"overlay")
    hdf = root / "results" / "plate" / "hdf"
    hdf.mkdir(parents=True)
    (hdf / "a.h5").write_bytes(b"hdf")

    manifest = resolve_manifest_json_path(root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "is_complete": not contradictory,
                "completed": 4 if contradictory else 2,
                "failed": 1 if contradictory else 0,
                "total_images": 2,
            }
        ),
        encoding="utf-8",
    )
    if contradictory:
        run_completion_marker_path(root).write_text(
            json.dumps(
                {
                    "status": "complete",
                    "finalizer_succeeded": True,
                }
            ),
            encoding="utf-8",
        )


def _discover(root: Path) -> OutputRoot:
    return OutputRoot.discover(
        root,
        cache_root=root.parent / f".cache-{root.name}",
    )


def _tree_snapshot(
    root: Path,
) -> tuple[tuple[str, ...], dict[str, tuple[bytes, int]]]:
    """Capture exact file bytes and mtimes for before/after mutation diffs."""
    directories = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    )
    files = {
        path.relative_to(root).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }
    return directories, files


def _walk(component: Any) -> Iterator[Any]:
    if component is None:
        return
    if isinstance(component, (list, tuple)):
        for item in component:
            yield from _walk(item)
        return
    yield component
    children = getattr(component, "children", None)
    if children is not None:
        yield from _walk(children)


def _component(component: Any, component_id: Any) -> Any:
    return next(
        node
        for node in _walk(component)
        if getattr(node, "id", None) == component_id
    )


def test_coherent_guard_issues_fresh_receipt_and_rejects_stale_generation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "coherent"
    _seed_output(source, contradictory=False)
    output = _discover(source)
    guard = OutputMutationGuard(output, "generation-1")

    receipt = guard.authorize(
        "curation",
        presented_generation="generation-1",
    )
    before = _tree_snapshot(source)
    with pytest.raises(OutputMutationBlocked, match="older output binding"):
        guard.authorize(
            "curation",
            presented_generation="generation-0",
        )

    assert output.consistency.state == "coherent"
    assert receipt.binding_generation == "generation-1"
    assert receipt.processing_fingerprint == output.source_fingerprint
    assert receipt.consistency_evidence_fingerprint == (
        output.consistency.evidence_fingerprint
    )
    assert _tree_snapshot(source) == before


def test_contradictory_large_fixture_is_read_only_without_any_repair(
    tmp_path: Path,
) -> None:
    source = tmp_path / "contradictory-large"
    _seed_output(source, contradictory=True, overlay_count=512)
    output = _discover(source)
    before = _tree_snapshot(source)
    guard = OutputMutationGuard(output, "generation-2")

    with pytest.raises(
        OutputMutationBlocked,
        match="completion evidence is contradictory",
    ):
        guard.authorize(
            "QC rebuild",
            presented_generation="generation-2",
        )

    assert output.master_df.height == 2
    assert len(output.overlay_index) == 512
    assert output.consistency.is_read_only
    assert _tree_snapshot(source) == before


def test_guard_detects_processing_change_before_caller_can_write(
    tmp_path: Path,
) -> None:
    source = tmp_path / "changed"
    _seed_output(source, contradictory=False)
    output = _discover(source)
    guard = OutputMutationGuard(output, "generation-3")
    (source / "results" / "plate" / "hdf" / "a.h5").write_bytes(b"changed")
    before_attempt = _tree_snapshot(source)

    with pytest.raises(OutputMutationBlocked, match="artifacts changed"):
        guard.authorize(
            "Error publication",
            presented_generation="generation-3",
        )

    assert _tree_snapshot(source) == before_attempt


def test_guard_detects_completion_evidence_change_before_write(
    tmp_path: Path,
) -> None:
    source = tmp_path / "changed-evidence"
    _seed_output(source, contradictory=False)
    output = _discover(source)
    guard = OutputMutationGuard(output, "generation-4")
    manifest = resolve_manifest_json_path(source)
    manifest.write_text(
        json.dumps(
            {
                "is_complete": False,
                "completed": 1,
                "failed": 0,
                "total_images": 2,
            }
        ),
        encoding="utf-8",
    )
    before_attempt = _tree_snapshot(source)

    with pytest.raises(OutputMutationBlocked, match="is incomplete"):
        guard.authorize(
            "Analysis publication",
            presented_generation="generation-4",
        )

    assert _tree_snapshot(source) == before_attempt


def test_real_qc_writer_rechecks_after_build_and_preserves_all_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "qc-race"
    pipeline = ImagePipeline(name="qc-race")
    pipeline.set_qc(
        [
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={
                    "on": "Size_Area",
                    "groupby": ["MetadataExperiment_Dataset"],
                    "min_replicates": 2,
                },
                instance_id="qc-SE-race0001",
                enabled=True,
            )
        ]
    )
    _seed_output(source, contradictory=False, pipeline=pipeline)
    owner = gui_launch_owner_path(source)
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text('{"status":"complete"}', encoding="utf-8")
    output = _discover(source)
    frame = output.master_df.to_pandas()
    run_qc(
        frame,
        pipeline,
        source,
        qc_output_dir=output.layout.qc_dir,
    )
    output = _discover(source)
    guard = OutputMutationGuard(output, "generation-qc")
    before_dirs, before_files = _tree_snapshot(source)
    checks = 0

    def _late_guard() -> bool:
        nonlocal checks
        checks += 1
        if checks == 3:
            owner.write_text("{malformed", encoding="utf-8")
        try:
            guard.authorize(
                "QC recompute",
                presented_generation="generation-qc",
            )
        except OutputMutationBlocked:
            return False
        return True

    with pytest.raises(QcPublicationBlocked):
        run_qc(
            frame.assign(Size_Area=frame["Size_Area"] * 3),
            pipeline,
            source,
            qc_output_dir=output.layout.qc_dir,
            publication_guard=_late_guard,
        )

    after_dirs, after_files = _tree_snapshot(source)
    owner_key = owner.relative_to(source).as_posix()
    assert checks == 3
    assert after_dirs == before_dirs
    assert after_files.pop(owner_key)[0] == b"{malformed"
    before_files.pop(owner_key)
    assert after_files == before_files


def test_real_plot_writer_rechecks_after_render_and_preserves_generation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plot-race"
    _seed_output(source, contradictory=False)
    owner = gui_launch_owner_path(source)
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text('{"status":"complete"}', encoding="utf-8")
    output = _discover(source)
    plot_dir = output.layout.plots_dir / "guarded"
    publish_plot_output(
        PlotOutput(pages=(PlotPage("default", plt.figure()),)),
        plot_dir,
        plot_id="guarded",
    )
    output = _discover(source)
    guard = OutputMutationGuard(output, "generation-plot")
    before_dirs, before_files = _tree_snapshot(source)
    checks = 0

    def _late_guard() -> bool:
        nonlocal checks
        checks += 1
        if checks == 3:
            owner.write_text('{"status":"future-state"}', encoding="utf-8")
        try:
            guard.authorize(
                "Measurement plot refresh",
                presented_generation="generation-plot",
            )
        except OutputMutationBlocked:
            return False
        return True

    with pytest.raises(PlotPublicationBlocked):
        publish_plot_output(
            PlotOutput(pages=(PlotPage("default", plt.figure()),)),
            plot_dir,
            plot_id="guarded",
            publication_guard=_late_guard,
        )

    after_dirs, after_files = _tree_snapshot(source)
    owner_key = owner.relative_to(source).as_posix()
    assert checks == 3
    assert after_dirs == before_dirs
    assert after_files.pop(owner_key)[0] == b'{"status":"future-state"}'
    before_files.pop(owner_key)
    assert after_files == before_files


def test_inconsistent_results_layout_keeps_views_and_disables_mutations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "layout"
    _seed_output(source, contradictory=True)
    output = _discover(source)
    curation = CurationLabels.load(output.layout, output.master_df)
    page = build_results_layout(output, curation)

    diagnostic = _component(page, viewer_ids.READ_ONLY_DIAGNOSTIC_ID)
    assert diagnostic.is_open is True
    assert "will not repair or resume" in diagnostic.children
    assert _component(page, viewer_ids.TABS_ID) is not None
    disabled_ids = (
        error_ids.ERROR_PUBLISH_BTN_ID,
        qc_ids.QC_ADD_CHECK_BTN_ID,
        qc_ids.QC_MIGRATE_RECIPE_BTN_ID,
        qc_ids.QC_REBUILD_DATABASE_BTN_ID,
        qc_ids.QC_MODAL_SUBMIT_BTN_ID,
        viewer_ids.COLONY_BULK_REMOVE_BTN_ID,
        viewer_ids.COLONY_BULK_RESTORE_BTN_ID,
        viewer_ids.COLONY_BULK_MARK_DROPDOWN_ID,
        review_ids.QC_REVIEW_MARK_REVIEWED_BTN_ID,
        review_ids.QC_REVIEW_BULK_REMOVE_BTN_ID,
        review_ids.QC_REVIEW_BULK_RESTORE_BTN_ID,
        review_ids.QC_REVIEW_BULK_MARK_DROPDOWN_ID,
    )
    for component_id in disabled_ids:
        assert _component(page, component_id).disabled is True

    card = build_viewer_card(
        "card-1",
        output,
        mutations_disabled=True,
    )
    details = _component(
        card,
        {"type": "card-details-table", "index": "card-1"},
    )
    assert details.cell_selectable is False

    grid, _ = build_grid(
        output.master_df,
        "Metadata_Column",
        "Metadata_Row",
        64,
        set(),
        set(),
        output,
        mutations_disabled=True,
    )
    radial = next(
        node
        for node in _walk(grid)
        if isinstance(getattr(node, "id", None), dict)
        and node.id.get("type") == "colony-radial-trigger"
    )
    assert radial.disabled is True


def test_inconsistent_results_app_keeps_read_only_browse_callback_available(
    tmp_path: Path,
) -> None:
    source = tmp_path / "browse"
    _seed_output(source, contradictory=True)
    output = _discover(source)
    before = _tree_snapshot(source)
    app = create_results_app(
        output,
        binding_generation="generation-5",
    )
    output_key = next(
        key
        for key, callback in app.callback_map.items()
        if any(
            item["id"] == viewer_ids.BTN_ADD_CARD
            for item in callback["inputs"]
        )
    )

    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            BINDING_GENERATION_PAYLOAD_KEY: "generation-5",
            "output": output_key,
            "outputs": {
                "id": viewer_ids.STORE_CARD_LIST,
                "property": "data",
            },
            "inputs": [
                {
                    "id": viewer_ids.BTN_ADD_CARD,
                    "property": "n_clicks",
                    "value": 1,
                }
            ],
            "state": [
                {
                    "id": viewer_ids.STORE_CARD_LIST,
                    "property": "data",
                    "value": [],
                }
            ],
            "changedPropIds": [f"{viewer_ids.BTN_ADD_CARD}.n_clicks"],
        },
    )

    assert response.status_code == 200
    payload = response.get_json()["response"][viewer_ids.STORE_CARD_LIST]["data"]
    assert len(payload) == 1
    assert _tree_snapshot(source) == before


def test_inconsistent_analysis_layout_preserves_preview_chrome_read_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "analysis"
    _seed_output(source, contradictory=True)
    output = _discover(source)
    recipe = RecipeState.from_layout(output.layout)
    page = build_analysis_layout(output, recipe)

    diagnostic = _component(
        page,
        analysis_ids.ANALYSIS_READ_ONLY_DIAGNOSTIC,
    )
    assert diagnostic.is_open is True
    assert "Browsing remains available" in diagnostic.children
    assert (
        _component(
            page,
            analysis_ids.ANALYSIS_POST_ADD_DROPDOWN,
        ).disabled
        is True
    )
    assert (
        _component(
            page,
            analysis_ids.ANALYSIS_FILTER_ADD_DROPDOWN,
        ).disabled
        is True
    )
    assert (
        _component(
            page,
            analysis_ids.ANALYSIS_EDGE_ADD_DROPDOWN,
        ).disabled
        is True
    )
    assert (
        _component(
            page,
            analysis_ids.ANALYSIS_MODEL_DROPDOWN,
        ).disabled
        is True
    )
    assert _component(page, analysis_ids.ANALYSIS_RUN_BUTTON).disabled is True
