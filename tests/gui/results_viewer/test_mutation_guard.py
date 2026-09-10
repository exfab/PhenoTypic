"""O4 contracts for fail-closed Results and Analysis mutations."""

from __future__ import annotations

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
from phenotypic.gui._schema_cache import MeasurementSchema
from phenotypic.gui.analysis import _ids as analysis_ids
from phenotypic.gui.analysis._layout import (
    build_app_layout as build_analysis_layout,
)
from phenotypic.gui.analysis._recipe_state import RecipeState
from phenotypic.gui.results_viewer import _ids as viewer_ids
from phenotypic.gui.results_viewer._app import create_app as create_results_app
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._error_tab import (
    _ids as error_ids,
    build_error_tab_body,
)
from phenotypic.gui.results_viewer._layout import (
    build_app_layout as build_results_layout,
)
from phenotypic.gui.results_viewer._mutation_guard import (
    OutputMutationBlocked,
    OutputMutationGuard,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._qc_tab import (
    _ids as qc_ids,
    build_qc_tab_body,
)
from phenotypic.gui.results_viewer._qc_tab._rebuild import (
    QcRebuildError,
    preflight_qc_rebuild,
    qc_publication_lock_path,
    rebuild_qc_database,
)
from phenotypic.gui.results_viewer._qc_tab.review import _ids as review_ids
from phenotypic.gui.results_viewer.colony_view._grid import build_grid
from phenotypic.gui.results_viewer._viewer_card import (
    layout as build_viewer_card,
)
from phenotypic.abc_.plotting import PlotOutput, PlotPage
from phenotypic.plotting._pipeline import (
    PlotPublicationBlocked,
    publish_plot_output,
)
from phenotypic.schema import IMAGE
from phenotypic.sdk_._qc_recipe import QcRecipe
from phenotypic.sdk_ import (
    gui_launch_owner_path,
    resolve_processing_state_path,
    verification_cache_path,
)
from tests._output_layout import bump_scientific_config_digest
from phenotypic.sdk_._qc_recipe import QcRecipeEntry
from phenotypic.sdk_._qc_recipe._runner import (
    QcPublicationBlocked,
    run_qc,
)


def _seed_output(
    root: Path,
    *,
    complete: bool,
    overlay_count: int = 2,
    pipeline: ImagePipeline | None = None,
) -> None:
    """Write a small scientific payload whose run state is chosen by *complete*.

    **The flag inverted and was renamed** (P6 Task 2). It used to be
    ``contradictory``, and it produced its two states by writing a
    ``manifest.json`` whose counts disagreed with the inventory, beside a
    completion marker claiming success. Spec §4.2 demotes both out of the
    evidence set, so that fixture now produces *one* state -- ``incomplete``
    -- for both values, and every test built on it would have gone green
    while asserting nothing.

    The states are now built by the **real publishers**, in the publication
    contract's own order: per-image records, then processing state, then the
    aggregated outputs, then the aggregate proof, then the run proof. An
    incomplete run is a complete one with the second image's record removed,
    which is the state a run killed between promoting a store and publishing
    its proof actually leaves.
    """
    from tests._output_layout import build_complete_viewer_run

    frame = pl.DataFrame(
        {
            "Metadata_Dataset": ["plate", "plate"],
            str(IMAGE.IMAGE_NAME): ["a", "b"],
            "Object_Label": [1, 2],
            # No nested `Centroid` column. The mirror is published as CSV as
            # well as parquet, and CSV cannot hold nested data -- so a frame
            # carrying one describes a tree the CLI cannot produce. It was
            # decoration: no assertion in this file ever read it.
            "Metadata_Row": ["A", "A"],
            "Metadata_Column": [1, 2],
            "Size_Area": [10.0, 20.0],
        }
    )
    build_complete_viewer_run(
        root,
        frame=frame,
        stems=("a", "b"),
        pipeline=(
            pipeline
            if pipeline is not None
            else ImagePipeline(name="mutation-guard")
        ),
        complete=complete,
    )
    overlays = root / "deliverables" / "overlays" / "plate"
    overlays.mkdir(parents=True, exist_ok=True)
    for index in range(overlay_count):
        (overlays / f"image-{index}.png").write_bytes(b"overlay")


def _discover(root: Path) -> OutputRoot:
    return OutputRoot.discover(
        root,
        cache_root=root.parent / f".cache-{root.name}",
    )


def _tree_snapshot(
    root: Path,
    *,
    ignored_files: tuple[Path, ...] = (),
) -> tuple[tuple[str, ...], dict[str, tuple[bytes, int]]]:
    """Capture exact file bytes and mtimes for before/after mutation diffs.

    **The tier-2 verification cache is excluded, and only it.** Resolving a
    run state deep-verifies, and a deep pass rewrites
    ``.phenotypic/verification_cache.json`` -- so every guard call that
    refuses a write still touches that one file. Excluding the whole of
    ``.phenotypic/`` would be easier and wrong: these tests assert on
    ``processing_state.json``, which lives there, and is exactly what the
    publication-boundary races perturb.

    `persist_states` never creates ``.phenotypic/`` and never raises, so a
    tree this package has not written to is still left byte-for-byte alone.
    """
    ignored_keys = {
        path.relative_to(root).as_posix() for path in ignored_files
    }
    ignored_keys.add(
        verification_cache_path(root).relative_to(root).as_posix()
    )
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
        and path.relative_to(root).as_posix() not in ignored_keys
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
    _seed_output(source, complete=True)
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

    assert output.run_completion == "complete"
    assert receipt.binding_generation == "generation-1"
    assert receipt.processing_fingerprint == output.source_fingerprint
    assert output.run_state is not None
    assert receipt.run_identity_digest == (
        output.run_state.identity.digest()
    )
    assert _tree_snapshot(source) == before


def test_an_incomplete_large_fixture_is_read_only_without_any_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "incomplete-large"
    _seed_output(source, complete=False, overlay_count=512)
    output = _discover(source)
    before = _tree_snapshot(source)
    guard = OutputMutationGuard(output, "generation-2")

    def _unexpected_inventory_walk(_self: OutputRoot) -> bool:
        raise AssertionError("read-only mutation attempted inventory validation")

    monkeypatch.setattr(OutputRoot, "snapshot_is_current", _unexpected_inventory_walk)

    with pytest.raises(
        OutputMutationBlocked,
        match="run state is incomplete",
    ):
        guard.authorize(
            "QC rebuild",
            presented_generation="generation-2",
        )

    assert output.master_df.height == 2
    # 512 extras plus the two `_publish_one_image` writes for stems a/b:
    # the fixture publishes real per-image artifacts now, and an overlay
    # is one of them.
    assert len(output.overlay_index) == 514
    assert output.run_is_complete is False
    assert _tree_snapshot(source) == before


def test_guard_detects_processing_change_before_caller_can_write(
    tmp_path: Path,
) -> None:
    source = tmp_path / "changed"
    _seed_output(source, complete=True)
    output = _discover(source)
    guard = OutputMutationGuard(output, "generation-3")
    # A file under ``results/`` that is NOT a declared artifact of any image.
    #
    # This used to rewrite image "a"'s store root, and that reached this
    # branch while completion evidence was manifest-shaped. It no longer
    # does: the store is one of that image's declared artifacts, so rewriting
    # it fails the image's verification and the guard refuses one branch
    # *earlier*, with "run state is incomplete". Catching the same corruption
    # sooner is an improvement -- but it would have left THIS branch, the
    # processing-inventory currency check, with no test at all. A stray file
    # under ``results/`` moves the inventory while every declared artifact
    # still verifies, which is what keeps the branch covered.
    (source / "results" / "plate" / "stray-artifact.bin").write_bytes(b"x")
    before_attempt = _tree_snapshot(source)

    with pytest.raises(OutputMutationBlocked, match="artifacts changed"):
        guard.authorize(
            "Error publication",
            presented_generation="generation-3",
        )

    assert _tree_snapshot(source) == before_attempt


def test_guard_detects_a_run_state_change_before_write(
    tmp_path: Path,
) -> None:
    """Formerly ``test_guard_detects_completion_evidence_change_before_write``.

    It drove the guard by writing an incomplete ``manifest.json``. Spec §4.2
    demotes the manifest out of the evidence set, so that write now moves
    nothing and the test asserted a refusal that can no longer happen. What
    replaces it is the run identity: rewriting ``config.pipeline_sha256``
    re-mints it, and the published run proof no longer matches.
    """
    source = tmp_path / "changed-evidence"
    _seed_output(source, complete=True)
    output = _discover(source)
    guard = OutputMutationGuard(output, "generation-4")
    bump_scientific_config_digest(source)
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
                    "groupby": ["Metadata_Dataset"],
                    "min_replicates": 2,
                },
                instance_id="qc-SE-race0001",
                enabled=True,
            )
        ]
    )
    _seed_output(source, complete=True, pipeline=pipeline)
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
            bump_scientific_config_digest(source)
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
    perturbed = (
        resolve_processing_state_path(source).relative_to(source).as_posix()
    )
    assert checks == 3
    assert after_dirs == before_dirs
    # A REWRITE of one existing file: the tree gains and loses nothing, which
    # is what the surrounding assertions are for.
    assert after_files.pop(perturbed) != before_files.pop(perturbed)
    assert after_files == before_files


def test_real_qc_rebuild_rechecks_with_synced_temp_before_replace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "qc-rebuild-race"
    pipeline = ImagePipeline(name="qc-rebuild-race")
    pipeline.set_qc(
        [
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={
                    "on": "Size_Area",
                    "groupby": ["Metadata_Dataset"],
                    "min_replicates": 2,
                },
                instance_id="qc-SE-rebuild-race",
                enabled=True,
            )
        ]
    )
    _seed_output(source, complete=True, pipeline=pipeline)
    owner = gui_launch_owner_path(source)
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text('{"status":"complete"}', encoding="utf-8")
    output = _discover(source)
    output.layout.qc_dir.mkdir(parents=True, exist_ok=True)
    output.layout.qc_duckdb.write_bytes(b"prior generation")
    owner.with_suffix(".lock").touch()
    qc_lock = qc_publication_lock_path(output.layout.qc_duckdb)
    qc_lock.touch()
    lock_files = (owner.with_suffix(".lock"), qc_lock)
    output = _discover(source)
    guard = OutputMutationGuard(output, "generation-qc-rebuild")
    preflight = preflight_qc_rebuild(output.layout)
    assert preflight.ready
    before_dirs, before_files = _tree_snapshot(
        source,
        ignored_files=lock_files,
    )
    mutation_saw_synced_temp = False

    def _interleaving_guard() -> bool:
        nonlocal mutation_saw_synced_temp
        temps = list(
            output.layout.qc_dir.glob(
                f".{output.layout.qc_duckdb.name}.*.tmp"
            )
        )
        if temps and not mutation_saw_synced_temp:
            assert len(temps) == 1
            assert temps[0].stat().st_size > 0
            mutation_saw_synced_temp = True
            bump_scientific_config_digest(source)
        try:
            guard.authorize(
                "QC rebuild",
                presented_generation="generation-qc-rebuild",
            )
        except OutputMutationBlocked:
            return False
        return True

    with pytest.raises(QcRebuildError, match="snapshot changed"):
        rebuild_qc_database(
            output.layout,
            expected_source_fingerprint=preflight.source_fingerprint,
            publication_guard=_interleaving_guard,
        )

    after_dirs, after_files = _tree_snapshot(
        source,
        ignored_files=lock_files,
    )
    perturbed = (
        resolve_processing_state_path(source).relative_to(source).as_posix()
    )
    assert mutation_saw_synced_temp
    assert after_dirs == before_dirs
    assert after_files.pop(perturbed) != before_files.pop(perturbed)
    assert after_files == before_files
    assert all(path.is_file() for path in lock_files)
    assert not list(source.rglob("*.tmp"))


def test_real_qc_rebuild_rechecks_synced_receipt_before_replace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "qc-rebuild-receipt-race"
    pipeline = ImagePipeline(name="qc-rebuild-receipt-race")
    pipeline.set_qc(
        [
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={
                    "on": "Size_Area",
                    "groupby": ["Metadata_Dataset"],
                    "min_replicates": 2,
                },
                instance_id="qc-SE-receipt-race",
                enabled=True,
            )
        ]
    )
    _seed_output(source, complete=True, pipeline=pipeline)
    owner = gui_launch_owner_path(source)
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text('{"status":"complete"}', encoding="utf-8")
    output = _discover(source)
    output.layout.qc_dir.mkdir(parents=True, exist_ok=True)
    owner.with_suffix(".lock").touch()
    qc_lock = qc_publication_lock_path(output.layout.qc_duckdb)
    qc_lock.touch()
    lock_files = (owner.with_suffix(".lock"), qc_lock)
    output = _discover(source)
    guard = OutputMutationGuard(output, "generation-qc-receipt")
    preflight = preflight_qc_rebuild(output.layout)
    assert preflight.ready
    before_dirs, before_files = _tree_snapshot(
        source,
        ignored_files=lock_files,
    )
    mutation_saw_synced_receipt = False

    def _interleaving_guard() -> bool:
        nonlocal mutation_saw_synced_receipt
        receipt_dir = output.layout.qc_dir / ".rebuild_receipts"
        temps = list(receipt_dir.glob(".*.tmp"))
        if temps and not mutation_saw_synced_receipt:
            assert len(temps) == 1
            assert temps[0].stat().st_size > 0
            mutation_saw_synced_receipt = True
            bump_scientific_config_digest(source)
        try:
            guard.authorize(
                "QC rebuild",
                presented_generation="generation-qc-receipt",
            )
        except OutputMutationBlocked:
            return False
        return True

    with pytest.raises(QcRebuildError, match="snapshot changed"):
        rebuild_qc_database(
            output.layout,
            expected_source_fingerprint=preflight.source_fingerprint,
            publication_guard=_interleaving_guard,
        )

    after_dirs, after_files = _tree_snapshot(
        source,
        ignored_files=lock_files,
    )
    perturbed = (
        resolve_processing_state_path(source).relative_to(source).as_posix()
    )
    assert mutation_saw_synced_receipt
    assert after_dirs == before_dirs
    assert after_files.pop(perturbed) != before_files.pop(perturbed)
    assert after_files == before_files
    assert all(path.is_file() for path in lock_files)
    assert not list(source.rglob("*.tmp"))


def test_real_plot_writer_rechecks_after_render_and_preserves_generation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plot-race"
    _seed_output(source, complete=True)
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
            bump_scientific_config_digest(source)
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
    perturbed = (
        resolve_processing_state_path(source).relative_to(source).as_posix()
    )
    assert checks == 3
    assert after_dirs == before_dirs
    assert after_files.pop(perturbed) != before_files.pop(perturbed)
    assert after_files == before_files


def test_inconsistent_results_layout_keeps_views_and_disables_mutations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "layout"
    _seed_output(source, complete=False)
    output = _discover(source)
    curation = CurationLabels.load(output.layout, output.master_df)
    page = build_results_layout(output, curation)

    diagnostic = _component(page, viewer_ids.READ_ONLY_DIAGNOSTIC_ID)
    assert diagnostic.is_open is True
    assert "will not repair or resume" in diagnostic.children
    assert _component(page, viewer_ids.TABS_ID) is not None
    mounted_disabled_ids = (
        viewer_ids.COLONY_BULK_REMOVE_BTN_ID,
        viewer_ids.COLONY_BULK_RESTORE_BTN_ID,
        viewer_ids.COLONY_BULK_MARK_DROPDOWN_ID,
    )
    for component_id in mounted_disabled_ids:
        assert _component(page, component_id).disabled is True

    # The QC and Error tabs are unmounted by
    # docs/superpowers/specs/2026-08-26-gui-simplification-removals (spec
    # section 3), so their bodies are no longer reachable from the results
    # layout. The packages are retained and still honour
    # ``mutations_disabled``; build them directly so the read-only contract
    # stays covered while they are off the tab bar.
    qc_body = build_qc_tab_body(
        QcRecipe.from_layout(output.layout),
        mutations_disabled=True,
    )
    for component_id in (
        qc_ids.QC_ADD_CHECK_BTN_ID,
        qc_ids.QC_MIGRATE_RECIPE_BTN_ID,
        qc_ids.QC_REBUILD_DATABASE_BTN_ID,
        qc_ids.QC_MODAL_SUBMIT_BTN_ID,
        review_ids.QC_REVIEW_MARK_REVIEWED_BTN_ID,
        review_ids.QC_REVIEW_BULK_REMOVE_BTN_ID,
        review_ids.QC_REVIEW_BULK_RESTORE_BTN_ID,
        review_ids.QC_REVIEW_BULK_MARK_DROPDOWN_ID,
    ):
        assert _component(qc_body, component_id).disabled is True

    error_body = build_error_tab_body(
        output,
        MeasurementSchema.from_layout(output.layout),
        mutations_disabled=True,
    )
    assert _component(error_body, error_ids.ERROR_PUBLISH_BTN_ID).disabled is True

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
    _seed_output(source, complete=False)
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
    _seed_output(source, complete=False)
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
