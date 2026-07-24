"""Integration tests for the QC Review tab's recompute + boot wiring.

These build a real results-viewer app against a synthetic CLI output dir
(master + post-applied mirror + ``qc/`` artifact + ``pipeline.json`` with
a QC entry) and verify the spec §D contracts that span modules:

- ``create_app`` boots with the Review sub-view mounted and the QC crop
  route registered under its own segment;
- the Configure recipe is now pipeline-backed (reads ``pipeline.json``'s
  ``qc`` array, not the legacy sidecar) without mutating legacy source state;
- the in-session per-group recompute (``run_qc`` on the post-applied frame
  anti-joined with removals) **matches** the CLI artifact for identical
  removals — and never wipes ``review_state.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from PIL import Image as PILImage

from phenotypic import ImagePipeline
from phenotypic.analysis import ReplicateAgreement
from phenotypic.gui._config import (
    CFG_FILTERED_STATE,
    CFG_QC_PIPELINE,
    CFG_QC_RECIPE,
    QC_CROPS_URL_SEGMENT,
)
from phenotypic.gui.results_viewer._app import create_app
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.sdk_._qc_recipe import QcRecipeEntry
from phenotypic.sdk_._qc_recipe._runner import run_qc
from phenotypic.sdk_ import (
    BundleLayout,
    measurements_parquet_path,
    pipeline_json_path,
    qc_review_state_path,
)

from tests._output_layout import (
    write_master,
    write_measurements_mirror,
    write_pipeline_json,
)
from phenotypic.schema import METADATA

_INSTANCE_ID = "qc-SE-aaaa1111"


def _layout(tmp_path: Path) -> BundleLayout:
    """Full-run-style layout rooted at ``tmp_path`` (deliverables under it)."""
    return BundleLayout(
        deliverables_base=tmp_path / "deliverables", output_root=tmp_path
    )


class _FakeRoot:
    """Minimal output-root stand-in exposing ``.layout`` for ``_db`` reads."""

    def __init__(self, layout: BundleLayout) -> None:
        self.layout = layout


def _build_pipeline() -> ImagePipeline:
    """A pipeline carrying one ReplicateAgreement QC entry."""
    pipeline = ImagePipeline(name="qc-review-test")
    pipeline.set_qc(
        [
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={
                    "on": "Size_Area",
                    "groupby": [str(METADATA.IMAGE_NAME)],
                    "min_replicates": 2,
                },
                instance_id=_INSTANCE_ID,
                enabled=True,
            )
        ]
    )
    return pipeline


@pytest.fixture()
def output_root(tmp_path: Path) -> OutputRoot:
    """Synthetic output dir with two replicate groups + a QC artifact."""
    master = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["d1"] * 6,
            str(METADATA.IMAGE_NAME): ["img-1"] * 3 + ["img-2"] * 3,
            "Object_Label": [1, 2, 3, 1, 2, 3],
            "Bbox_CenterRR": [50] * 6,
            "Bbox_CenterCC": [50] * 6,
            "Bbox_MinRR": [40] * 6,
            "Bbox_MaxRR": [60] * 6,
            "Bbox_MinCC": [40] * 6,
            "Bbox_MaxCC": [60] * 6,
            # img-1 tight (agree), img-2 has one wild outlier (label 3).
            "Size_Area": [100.0, 101.0, 102.0, 100.0, 101.0, 900.0],
        }
    )
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)

    (tmp_path / "results" / "d1" / "measurements").mkdir(
        parents=True, exist_ok=True
    )
    overlays = tmp_path / "deliverables" / "overlays" / "d1"
    overlays.mkdir(parents=True)
    for stem in ("img-1", "img-2"):
        PILImage.new("RGB", (120, 120), (200, 0, 0)).save(
            overlays / f"{stem}.png"
        )

    pipeline = _build_pipeline()
    write_pipeline_json(tmp_path, pipeline)

    # Seed the qc/ artifact exactly as the CLI would.
    run_qc(master.to_pandas(), pipeline, tmp_path)

    return OutputRoot.discover(tmp_path)


def test_create_app_boots_with_review_and_qc_crop_route(output_root) -> None:
    """App builds; the QC crop route serves a centered PNG under its segment."""
    app = create_app(output_root)
    # Pipeline-backed recipe + pipeline loaded into config for recompute.
    assert app.server.config.get(CFG_QC_PIPELINE) is not None
    recipe = app.server.config.get(CFG_QC_RECIPE)
    assert recipe is not None
    assert [e.instance_id for e in recipe.entries] == [_INSTANCE_ID]

    client = app.server.test_client()
    resp = client.get(f"/{QC_CROPS_URL_SEGMENT}/d1/img-2/3.png?size=48")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"


def test_configure_recipe_is_pipeline_backed(
    output_root, tmp_path: Path
) -> None:
    """The Configure recipe edits pipeline.json's qc array (not the sidecar)."""
    app = create_app(output_root)
    recipe = app.server.config.get(CFG_QC_RECIPE)
    # The recipe's file is pipeline.json, and the entry round-trips.
    assert recipe.path == pipeline_json_path(tmp_path)
    assert recipe.entries[0].cls is ReplicateAgreement


def test_recompute_matches_cli_for_identical_removals(
    output_root, tmp_path: Path
) -> None:
    """GUI in-session recompute == CLI run_qc on the same curated frame.

    Remove the wild outlier (img-2, label 3) and recompute via the GUI's
    data path; the rewritten qc_summary metric for img-2 must equal a
    direct CLI-style run_qc on measurements.parquet minus that key.
    """
    from phenotypic.gui.results_viewer._qc_tab.review import _data, _db

    removed = {("img-2", 3)}

    # GUI path: build the curated frame the Review recompute would feed run_qc.
    gui_frame = _data.build_recompute_frame(output_root, removed)
    run_qc(gui_frame, _build_pipeline(), tmp_path)
    gui_summary = _db.module_summary(output_root, _INSTANCE_ID)

    # CLI-equivalent path: post-applied mirror minus the same key.
    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    cli_frame = mirror.filter(
        ~(
            (pl.col(str(METADATA.IMAGE_NAME)) == "img-2")
            & (pl.col("Object_Label") == 3)
        )
    ).to_pandas()
    # Write to a sibling dir so the two artifacts don't clobber each other.
    cli_dir = tmp_path / "_cli_check"
    cli_dir.mkdir()
    run_qc(cli_frame, _build_pipeline(), cli_dir)
    cli_root = _FakeRoot(
        BundleLayout(
            deliverables_base=cli_dir / "deliverables", output_root=cli_dir
        )
    )
    cli_summary = _db.module_summary(cli_root, _INSTANCE_ID)

    def _img2_metric(summary: pl.DataFrame) -> float:
        row = summary.filter(pl.col(str(METADATA.IMAGE_NAME)) == "img-2")
        return float(row.get_column("metric")[0])

    assert _img2_metric(gui_summary) == pytest.approx(
        _img2_metric(cli_summary)
    )


def test_recompute_does_not_touch_review_state(
    output_root, tmp_path: Path
) -> None:
    """``run_qc`` (the only recompute call) must never clear review_state.json."""
    from phenotypic.gui.results_viewer._qc_tab.review._review_state import (
        ReviewState,
    )

    # GUI marks a group reviewed.
    state = ReviewState.load(_layout(tmp_path))
    state.mark_reviewed(_INSTANCE_ID, ("img-1",))
    assert qc_review_state_path(tmp_path).exists()

    # A recompute runs run_qc only.
    run_qc(
        pl.read_parquet(measurements_parquet_path(tmp_path)).to_pandas(),
        _build_pipeline(),
        tmp_path,
    )

    # review_state.json survives the recompute.
    reloaded = ReviewState.load(_layout(tmp_path))
    assert reloaded.is_reviewed(_INSTANCE_ID, ("img-1",))


def test_legacy_sidecar_is_not_migrated_during_viewer_binding(
    tmp_path: Path,
) -> None:
    """Viewer binding leaves the complete legacy source tree byte-identical."""
    # Minimal output dir WITHOUT a qc entry in pipeline.json, but WITH a
    # legacy sidecar carrying one.
    master = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["d1"] * 2,
            str(METADATA.IMAGE_NAME): ["img-1", "img-1"],
            "Object_Label": [1, 2],
            "Bbox_CenterRR": [50, 50],
            "Bbox_CenterCC": [50, 50],
            "Bbox_MinRR": [40, 40],
            "Bbox_MaxRR": [60, 60],
            "Bbox_MinCC": [40, 40],
            "Bbox_MaxCC": [60, 60],
            "Size_Area": [100.0, 101.0],
        }
    )
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)
    (tmp_path / "results" / "d1" / "measurements").mkdir(
        parents=True, exist_ok=True
    )
    overlay_dir = tmp_path / "deliverables" / "overlays" / "d1"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (120, 120), (200, 0, 0)).save(
        overlay_dir / "img-1.png"
    )
    write_pipeline_json(tmp_path, ImagePipeline(name="no-qc"))
    sidecar_dir = tmp_path / ".viewer_cache"
    sidecar_dir.mkdir()
    sidecar = sidecar_dir / "qc_recipe.json"
    sidecar.write_text(
        json.dumps(
            {
                "version": 1,
                "checks": [
                    {
                        "instance_id": _INSTANCE_ID,
                        "class": "ReplicateAgreement",
                        "enabled": True,
                        "params": {
                            "on": "Size_Area",
                            "groupby": [str(METADATA.IMAGE_NAME)],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    source_before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    root = OutputRoot.discover(tmp_path)
    app = create_app(root)
    recipe = app.server.config.get(CFG_QC_RECIPE)
    # Binding is read-only. Compatibility UI may offer an explicit migration,
    # but app construction cannot fold or retire this source sidecar.
    assert not any(e.instance_id == _INSTANCE_ID for e in recipe.entries)
    source_after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert source_after == source_before


def test_recompute_delta_carries_after_status(
    output_root, tmp_path: Path
) -> None:
    """The in-session recompute delta carries the recomputed ``status_after``.

    Regression guard for the worklist-row-stale-metric/badge fix (spec
    §D.5): the delta the worklist row consumes must include both the new
    metric AND the recomputed status (read straight from the rewritten
    qc_summary), so the in-place row update flips the badge colour — not
    just the number. Removing the wild img-2 outlier tightens the group,
    so its metric moves and the delta reports a concrete ``status_after``.
    """
    from phenotypic.gui.results_viewer._qc_tab.review import _callbacks, _db

    app = create_app(output_root)
    filtered = app.server.config[CFG_FILTERED_STATE]
    filtered.remove_many([("img-2", 3)])

    module = next(
        m
        for m in _db.list_modules(output_root)
        if m.instance_id == _INSTANCE_ID
    )
    groupby_cols = module.groupby_cols
    summary_before = _db.module_summary(output_root, _INSTANCE_ID)
    metric_before = summary_before.filter(
        pl.col(str(METADATA.IMAGE_NAME)) == "img-2"
    ).get_column("metric")[0]

    with app.server.app_context():
        delta = _callbacks._recompute_after_curation(
            _INSTANCE_ID, groupby_cols, ("img-2",), metric_before
        )

    assert delta is not None
    assert delta["moved"] is True
    assert delta["after"] != metric_before
    # The recomputed status is present and is a real QC status label, so the
    # worklist badge can flip to it in place.
    assert delta["status_after"] in {"fail", "warn", "pass", "insufficient"}

    # And the in-place cell update renders that after-metric + status badge.
    cell = _callbacks.worklist_row_metric_update(delta)
    badge = cell[1]
    assert badge.children == delta["status_after"]


def test_review_per_tile_curation_contract(
    output_root, tmp_path: Path
) -> None:
    """Per-tile + bulk curation write the shared removal set correctly.

    Regression guard for the ``mutate_and_payload(action)`` contract — the
    action MUST accept the ``FilteredMeasurements`` instance. A 0-arg
    closure (the original Review bug) 500s the live curation callback;
    this drives the extracted mutation helpers directly so the contract is
    caught without a browser.
    """
    from phenotypic.gui.results_viewer._filtered_state import (
        FilteredMeasurements,
    )
    from phenotypic.gui.results_viewer._qc_tab.review._callbacks import (
        bulk_review_curation,
        toggle_review_tile,
    )

    filtered = FilteredMeasurements.load(
        output_root.root, output_root.master_df
    )

    # Per-tile toggle removes, then restores.
    payload = toggle_review_tile(filtered, "img-2", 3)
    assert ["img-2", 3] in payload
    assert ("img-2", 3) in filtered.removed_keys
    toggle_review_tile(filtered, "img-2", 3)
    assert ("img-2", 3) not in filtered.removed_keys

    # Bulk remove + restore.
    bulk_review_curation(
        filtered, remove=True, selected=[("img-1", 1), ("img-2", 2)]
    )
    assert ("img-1", 1) in filtered.removed_keys
    assert ("img-2", 2) in filtered.removed_keys
    bulk_review_curation(
        filtered, remove=False, selected=[("img-1", 1), ("img-2", 2)]
    )
    assert ("img-1", 1) not in filtered.removed_keys
    assert ("img-2", 2) not in filtered.removed_keys


def _collect_img_srcs(component: object) -> list[str]:
    """Recursively collect every ``html.Img.src`` in a Dash component tree."""
    srcs: list[str] = []

    def _walk(node: object) -> None:
        src = getattr(node, "src", None)
        if isinstance(src, str):
            srcs.append(src)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                _walk(child)
        elif children is not None:
            _walk(children)

    _walk(component)
    return srcs


def test_qc_gallery_threads_dim_alpha_into_tile_urls(output_root) -> None:
    """The Review faceted gallery threads the store alpha onto each ``&dim=``.

    Drives :func:`_render_faceted_gallery` (the unit the detail-render
    callback calls) inside an app context so ``_qc_crop_url`` resolves the
    mount prefix, and asserts every tile ``<img src>`` carries the exact
    store alpha as ``&dim=``.
    """
    from phenotypic.gui.results_viewer._qc_tab.review._callbacks import (
        _render_faceted_gallery,
    )

    app = create_app(output_root)
    alpha = 0.35
    facets = [
        (None, [("d1", "img-1", 1), ("d1", "img-1", 2), ("d1", "img-2", 3)]),
    ]

    with app.server.app_context():
        gallery = _render_faceted_gallery(
            facets,
            removed=set(),
            crop_size=48,
            display_size=120,
            has_image_source=output_root.has_image_source,
            dim_alpha=alpha,
        )

    srcs = _collect_img_srcs(gallery)
    assert len(srcs) == 3
    for src in srcs:
        assert QC_CROPS_URL_SEGMENT in src
        assert "?size=48" in src
        assert f"&dim={alpha}" in src


def test_qc_gallery_default_dim_alpha_is_zero(output_root) -> None:
    """No ``dim_alpha`` degrades the gallery URLs to ``&dim=0.0``."""
    from phenotypic.gui.results_viewer._qc_tab.review._callbacks import (
        _render_faceted_gallery,
    )

    app = create_app(output_root)
    facets = [(None, [("d1", "img-1", 1)])]

    with app.server.app_context():
        gallery = _render_faceted_gallery(
            facets,
            removed=set(),
            crop_size=48,
            display_size=120,
            has_image_source=output_root.has_image_source,
        )

    srcs = _collect_img_srcs(gallery)
    assert srcs
    for src in srcs:
        assert "&dim=0.0" in src


# ---------------------------------------------------------------------------
# T9: durable settings-edit recompute + review-state reconciliation
# ---------------------------------------------------------------------------


def test_settings_edit_durably_rewrites_db(
    output_root, tmp_path: Path
) -> None:
    """A QC settings edit rewrites qc.duckdb so the worklist reflects it.

    The agreeing group (img-1) passes under the default thresholds. Tighten
    the thresholds (a settings edit) and run the durable recompute; the
    rewritten catalog summary must report img-1 as ``fail`` (not the stale
    ``pass``), proving the in-memory pipeline was synced from the recipe
    before the rebuild.
    """
    from phenotypic.gui.results_viewer._qc_tab import (
        _callbacks as qc_callbacks,
    )
    from phenotypic.gui.results_viewer._qc_tab.review import _db

    app = create_app(output_root)
    recipe = app.server.config[CFG_QC_RECIPE]
    pipeline = app.server.config[CFG_QC_PIPELINE]

    before = _db.module_summary(output_root, _INSTANCE_ID)
    img1_before = before.filter(
        pl.col(str(METADATA.IMAGE_NAME)) == "img-1"
    ).get_column("status")[0]
    assert img1_before == "pass"

    # Settings edit: tighten thresholds so the agreeing group now fails.
    assert recipe.update(
        _INSTANCE_ID,
        params={
            "on": "Size_Area",
            "groupby": [str(METADATA.IMAGE_NAME)],
            "min_replicates": 2,
            "warn_threshold": 0.001,
            "fail_threshold": 0.002,
        },
    )

    with app.server.app_context():
        assert qc_callbacks._run_settings_edit_recompute(
            output_root, recipe, pipeline, set()
        )

    after = _db.module_summary(output_root, _INSTANCE_ID)
    img1_after = after.filter(
        pl.col(str(METADATA.IMAGE_NAME)) == "img-1"
    ).get_column("status")[0]
    assert img1_after == "fail"


def test_settings_edit_all_disabled_empties_worklist(
    output_root, tmp_path: Path
) -> None:
    """Disabling every check clears the stale qc.duckdb (empty worklist)."""
    from phenotypic.gui.results_viewer._qc_tab import (
        _callbacks as qc_callbacks,
    )
    from phenotypic.gui.results_viewer._qc_tab.review import _db

    app = create_app(output_root)
    recipe = app.server.config[CFG_QC_RECIPE]
    pipeline = app.server.config[CFG_QC_PIPELINE]

    assert _db.list_modules(output_root)  # seeded module present

    recipe.update(_INSTANCE_ID, enabled=False)
    with app.server.app_context():
        assert qc_callbacks._run_settings_edit_recompute(
            output_root, recipe, pipeline, set()
        )

    # The stale DB is removed → empty module list (worklist degrades empty).
    assert not output_root.layout.qc_duckdb.exists()
    assert _db.list_modules(output_root) == []


def test_reconcile_drops_vanished_reviewed_keys(tmp_path: Path) -> None:
    from phenotypic.gui.results_viewer._qc_tab.review._review_state import (
        ReviewState,
        encode_group_key,
    )

    state = ReviewState(path=tmp_path / "review_state.json")
    state.mark_reviewed("qc-ZMax-1", ("P1",))
    state.mark_reviewed("qc-ZMax-1", ("P_GONE",))
    state.reconcile_to_summary("qc-ZMax-1", {encode_group_key(("P1",))})
    assert state.is_reviewed("qc-ZMax-1", ("P1",))
    assert not state.is_reviewed("qc-ZMax-1", ("P_GONE",))
