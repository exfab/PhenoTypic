"""Integration tests for the QC Review tab's recompute + boot wiring.

These build a real results-viewer app against a synthetic CLI output dir
(master + post-applied mirror + ``qc/`` artifact + ``pipeline.json`` with
a QC entry) and verify the spec §D contracts that span modules:

- ``create_app`` boots with the Review sub-view mounted and the QC crop
  route registered under its own segment;
- the Configure recipe is now pipeline-backed (reads ``pipeline.json``'s
  ``qc`` array, not the legacy sidecar) and a legacy sidecar is migrated;
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
    measurements_parquet_path,
    pipeline_json_path,
    qc_review_state_path,
    qc_summary_parquet_path,
)

from tests._output_layout import write_master, write_measurements_mirror, write_pipeline_json

_INSTANCE_ID = "qc-SE-aaaa1111"


def _build_pipeline() -> ImagePipeline:
    """A pipeline carrying one ReplicateAgreement QC entry."""
    pipeline = ImagePipeline(name="qc-review-test")
    pipeline.set_qc(
        [
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={
                    "on": "Size_Area",
                    "groupby": ["Metadata_ImageFile"],
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
            "Metadata_Dataset": ["d1"] * 6,
            "Metadata_ImageFile": ["img-1"] * 3 + ["img-2"] * 3,
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

    overlays = tmp_path / "results" / "d1" / "overlays"
    overlays.mkdir(parents=True)
    for stem in ("img-1", "img-2"):
        PILImage.new("RGB", (120, 120), (200, 0, 0)).save(overlays / f"{stem}.png")

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


def test_configure_recipe_is_pipeline_backed(output_root, tmp_path: Path) -> None:
    """The Configure recipe edits pipeline.json's qc array (not the sidecar)."""
    app = create_app(output_root)
    recipe = app.server.config.get(CFG_QC_RECIPE)
    # The recipe's file is pipeline.json, and the entry round-trips.
    assert recipe.path == pipeline_json_path(tmp_path)
    assert recipe.entries[0].cls is ReplicateAgreement


def test_recompute_matches_cli_for_identical_removals(output_root, tmp_path: Path) -> None:
    """GUI in-session recompute == CLI run_qc on the same curated frame.

    Remove the wild outlier (img-2, label 3) and recompute via the GUI's
    data path; the rewritten qc_summary metric for img-2 must equal a
    direct CLI-style run_qc on measurements.parquet minus that key.
    """
    from phenotypic.gui.results_viewer._qc_tab.review import _data

    removed = {("img-2", 3)}

    # GUI path: build the curated frame the Review recompute would feed run_qc.
    gui_frame = _data.build_recompute_frame(output_root, removed)
    run_qc(gui_frame, _build_pipeline(), tmp_path)
    gui_summary = pl.read_parquet(qc_summary_parquet_path(tmp_path))

    # CLI-equivalent path: post-applied mirror minus the same key.
    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    cli_frame = mirror.filter(
        ~(
            (pl.col("Metadata_ImageFile") == "img-2")
            & (pl.col("Object_Label") == 3)
        )
    ).to_pandas()
    # Write to a sibling dir so the two artifacts don't clobber each other.
    cli_dir = tmp_path / "_cli_check"
    cli_dir.mkdir()
    run_qc(cli_frame, _build_pipeline(), cli_dir)
    cli_summary = pl.read_parquet(qc_summary_parquet_path(cli_dir))

    def _img2_metric(summary: pl.DataFrame) -> float:
        row = summary.filter(pl.col("Metadata_ImageFile") == "img-2")
        return float(row.get_column("metric")[0])

    assert _img2_metric(gui_summary) == pytest.approx(_img2_metric(cli_summary))


def test_recompute_does_not_touch_review_state(output_root, tmp_path: Path) -> None:
    """``run_qc`` (the only recompute call) must never clear review_state.json."""
    from phenotypic.gui.results_viewer._qc_tab.review._review_state import (
        ReviewState,
    )

    # GUI marks a group reviewed.
    state = ReviewState.load(tmp_path)
    state.mark_reviewed(_INSTANCE_ID, ("img-1",))
    assert qc_review_state_path(tmp_path).exists()

    # A recompute runs run_qc only.
    run_qc(
        pl.read_parquet(measurements_parquet_path(tmp_path)).to_pandas(),
        _build_pipeline(),
        tmp_path,
    )

    # review_state.json survives the recompute.
    reloaded = ReviewState.load(tmp_path)
    assert reloaded.is_reviewed(_INSTANCE_ID, ("img-1",))


def test_legacy_sidecar_is_migrated_into_pipeline(tmp_path: Path) -> None:
    """A legacy .viewer_cache/qc_recipe.json folds into pipeline.json at boot."""
    # Minimal output dir WITHOUT a qc entry in pipeline.json, but WITH a
    # legacy sidecar carrying one.
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"] * 2,
            "Metadata_ImageFile": ["img-1", "img-1"],
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
    (tmp_path / "results" / "d1" / "overlays").mkdir(parents=True)
    PILImage.new("RGB", (120, 120), (200, 0, 0)).save(
        tmp_path / "results" / "d1" / "overlays" / "img-1.png"
    )
    write_pipeline_json(tmp_path, ImagePipeline(name="no-qc"))
    sidecar_dir = tmp_path / ".viewer_cache"
    sidecar_dir.mkdir()
    (sidecar_dir / "qc_recipe.json").write_text(
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
                            "groupby": ["Metadata_ImageFile"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    root = OutputRoot.discover(tmp_path)
    app = create_app(root)
    recipe = app.server.config.get(CFG_QC_RECIPE)
    # The migrated entry now lives in the pipeline-backed recipe.
    assert any(e.instance_id == _INSTANCE_ID for e in recipe.entries)
    # And it landed in pipeline.json's qc array.
    payload = json.loads(pipeline_json_path(tmp_path).read_text(encoding="utf-8"))
    assert any(e["instance_id"] == _INSTANCE_ID for e in payload.get("qc", []))


def test_recompute_delta_carries_after_status(output_root, tmp_path: Path) -> None:
    """The in-session recompute delta carries the recomputed ``status_after``.

    Regression guard for the worklist-row-stale-metric/badge fix (spec
    §D.5): the delta the worklist row consumes must include both the new
    metric AND the recomputed status (read straight from the rewritten
    qc_summary), so the in-place row update flips the badge colour — not
    just the number. Removing the wild img-2 outlier tightens the group,
    so its metric moves and the delta reports a concrete ``status_after``.
    """
    from phenotypic.gui.results_viewer._qc_tab.review import _callbacks, _data

    app = create_app(output_root)
    filtered = app.server.config[CFG_FILTERED_STATE]
    filtered.remove_many([("img-2", 3)])

    groupby_cols = _data.groupby_cols_for(
        _data.load_qc_summary(output_root), _INSTANCE_ID
    )
    summary_before = _data.load_qc_summary(output_root)
    metric_before = _data.group_record(
        summary_before, _INSTANCE_ID, groupby_cols, ("img-2",)
    )["metric"]

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


def test_review_per_tile_curation_contract(output_root, tmp_path: Path) -> None:
    """Per-tile + bulk curation write the shared removal set correctly.

    Regression guard for the ``mutate_and_payload(action)`` contract — the
    action MUST accept the ``FilteredMeasurements`` instance. A 0-arg
    closure (the original Review bug) 500s the live curation callback;
    this drives the extracted mutation helpers directly so the contract is
    caught without a browser.
    """
    from phenotypic.gui.results_viewer._filtered_state import FilteredMeasurements
    from phenotypic.gui.results_viewer._qc_tab.review._callbacks import (
        bulk_review_curation,
        toggle_review_tile,
    )

    filtered = FilteredMeasurements.load(output_root.root, output_root.master_df)

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
            has_overlay=output_root.has_overlay,
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
            has_overlay=output_root.has_overlay,
        )

    srcs = _collect_img_srcs(gallery)
    assert srcs
    for src in srcs:
        assert "&dim=0.0" in src
