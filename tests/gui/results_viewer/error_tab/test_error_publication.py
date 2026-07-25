"""Transactional all-category Error publication regression tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from phenotypic.gui.results_viewer import _ids as viewer_ids
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._error_tab import (
    build_error_tab_body,
    register_error_callbacks,
)
from phenotypic.gui.results_viewer._error_tab import _ids as error_ids
from phenotypic.gui.results_viewer._error_tab import _publication
from phenotypic.gui.results_viewer._error_tab._publication import (
    ErrorPublicationConflict,
    capture_error_source_fingerprints,
    compute_all_category_analysis,
    compute_gui_error_publication,
    error_publication_lock_path,
    publish_error_analysis,
    recover_error_publication,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.schema import ErrorCategory, METADATA
from phenotypic.sdk_ import (
    atomic_write_json,
    gui_launch_owner_path,
)
from phenotypic.sdk_._file_locking import (
    ArtifactLockTimeout,
    exclusive_path_lock,
)
from tests._output_layout import write_master, write_measurements_mirror

KEY_IMAGE_FILE = str(METADATA.IMAGE_NAME)


def _master() -> pl.DataFrame:
    """Return two separable error classes and a sufficiently large baseline."""
    rng = np.random.default_rng(17)
    debris_n = 8
    merged_n = 8
    good_n = 12
    return pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["plate-1"]
            * (debris_n + merged_n + good_n),
            KEY_IMAGE_FILE: ["img-1"] * (debris_n + merged_n + good_n),
            "Object_Label": list(range(1, debris_n + merged_n + good_n + 1)),
            "Size_Area": [
                *rng.normal(10.0, 0.2, debris_n),
                *rng.normal(100.0, 0.2, merged_n),
                *rng.normal(500.0, 0.2, good_n),
            ],
            "Shape_Circularity": rng.normal(
                0.8,
                0.01,
                debris_n + merged_n + good_n,
            ),
        }
    )


@pytest.fixture()
def publication_state(
    tmp_path: Path,
) -> tuple[Path, OutputRoot, CurationLabels]:
    """Seed one stable Results binding with two populated categories."""
    master = _master()
    write_master(tmp_path, master)
    write_measurements_mirror(tmp_path, master)
    from phenotypic.sdk_ import BundleLayout

    layout = BundleLayout.detect(tmp_path)
    labels = CurationLabels.load(layout, master)
    labels.mark_many([("img-1", label) for label in range(1, 9)], "debris")
    labels.mark_many([("img-1", label) for label in range(9, 17)], "merged")
    root = OutputRoot.discover(
        tmp_path,
        cache_root=tmp_path.parent / ".viewer-cache",
    )
    return tmp_path, root, labels


def _compute(root: OutputRoot, labels: CurationLabels):
    return compute_gui_error_publication(
        root,
        filtered_state=labels,
        good_mode="all_unlabeled",
    )


def _canonical_bytes(root: OutputRoot) -> dict[str, bytes]:
    names = (
        "error_analysis.parquet",
        "error_analysis.csv",
        "error_analysis.html",
        "error_analysis.manifest.json",
        "error_analysis.publication.json",
    )
    return {
        name: (root.layout.deliverables_base / name).read_bytes()
        for name in names
        if (root.layout.deliverables_base / name).is_file()
    }


def test_publish_all_categories_records_empty_categories(
    publication_state: tuple[Path, OutputRoot, CurationLabels],
) -> None:
    """Every core category is explicit, even when no object carries it."""
    _output, root, labels = publication_state
    computation = _compute(root, labels)

    published = publish_error_analysis(
        root.layout,
        computation,
        mutation_is_safe=root.mutation_snapshot_is_safe,
    )

    assert published.category_count == len(ErrorCategory.labels())
    assert published.populated_category_count == 2
    assert not published.already_published
    table = pl.read_parquet(root.layout.error_analysis_parquet)
    assert set(table.get_column("category")) == {"debris", "merged"}
    manifest = json.loads(
        (root.layout.deliverables_base / "error_analysis.manifest.json").read_text()
    )
    by_category = {
        item["category"]: item for item in manifest["categories"]
    }
    assert set(by_category) == set(ErrorCategory.labels())
    assert by_category["debris"]["labels"] == 8
    assert by_category["merged"]["labels"] == 8
    assert by_category["other"] == {
        "category": "other",
        "labels": 0,
        "rows": 0,
    }
    assert set(manifest["artifacts"]) == {
        "error_analysis.parquet",
        "error_analysis.csv",
        "error_analysis.html",
    }
    assert not (
        root.layout.deliverables_base / ".error-analysis.generations"
    ).exists()


def test_publish_refuses_fingerprint_conflict_without_targets(
    publication_state: tuple[Path, OutputRoot, CurationLabels],
) -> None:
    """A changed review source invalidates an already-computed generation."""
    _output, root, labels = publication_state
    computation = _compute(root, labels)
    root.layout.qc_review_state_path.write_text('{"changed": true}\n')

    with pytest.raises(ErrorPublicationConflict, match="inputs changed"):
        publish_error_analysis(
            root.layout,
            computation,
            mutation_is_safe=root.mutation_snapshot_is_safe,
        )

    assert not root.layout.error_analysis_parquet.exists()
    assert not root.layout.error_analysis_csv.exists()


def test_publish_refuses_concurrent_writer(
    publication_state: tuple[Path, OutputRoot, CurationLabels],
) -> None:
    """A second publisher cannot enter the shared GUI/CLI lock."""
    _output, root, labels = publication_state
    computation = _compute(root, labels)

    with exclusive_path_lock(error_publication_lock_path(root.layout)):
        with pytest.raises(ArtifactLockTimeout):
            publish_error_analysis(
                root.layout,
                computation,
                mutation_is_safe=root.mutation_snapshot_is_safe,
                lock_timeout=0.01,
            )


def test_mid_publication_failure_restores_previous_generation(
    publication_state: tuple[Path, OutputRoot, CurationLabels],
) -> None:
    """Failure after one replacement rolls every canonical target back."""
    _output, root, labels = publication_state
    first = _compute(root, labels)
    publish_error_analysis(
        root.layout,
        first,
        mutation_is_safe=root.mutation_snapshot_is_safe,
    )
    previous = _canonical_bytes(root)

    labels.mark_many([("img-1", 17)], "debris")
    second = _compute(root, labels)
    replacements = 0

    def _fail_second_replace(source: Path, target: Path) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise OSError("injected replacement failure")
        os.replace(source, target)

    with pytest.raises(OSError, match="injected"):
        publish_error_analysis(
            root.layout,
            second,
            mutation_is_safe=root.mutation_snapshot_is_safe,
            replace_file=_fail_second_replace,
        )

    assert _canonical_bytes(root) == previous
    assert not (
        root.layout.deliverables_base / ".error-analysis.generations"
    ).exists()


def test_committed_recovery_cleans_transaction_backups(
    publication_state: tuple[Path, OutputRoot, CurationLabels],
) -> None:
    """Restart cleanup drops backups only after a valid receipt committed."""
    _output, root, labels = publication_state
    computation = _compute(root, labels)
    publish_error_analysis(
        root.layout,
        computation,
        mutation_is_safe=root.mutation_snapshot_is_safe,
    )
    base = root.layout.deliverables_base
    receipt = json.loads(
        (base / "error_analysis.publication.json").read_text()
    )
    token = "b" * 32
    generation_dir = base / ".error-analysis.generations" / token
    backup_dir = generation_dir / "backup"
    backup_dir.mkdir(parents=True)
    (backup_dir / "error_analysis.csv").write_bytes(b"previous")
    targets = [
        *receipt["artifacts"],
        "error_analysis.manifest.json",
        "error_analysis.publication.json",
    ]
    atomic_write_json(
        base / ".error-analysis.transaction.json",
        {
            "schema_version": 1,
            "token": token,
            "generation": receipt["generation"],
            "targets": targets,
            "existing": ["error_analysis.csv"],
        },
    )

    with exclusive_path_lock(error_publication_lock_path(root.layout)):
        assert recover_error_publication(root.layout) is True

    assert not (base / ".error-analysis.transaction.json").exists()
    assert not (base / ".error-analysis.generations").exists()


@pytest.mark.parametrize("interrupt_after", [1, 2])
def test_restore_is_restart_safe_after_each_replaced_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_after: int,
) -> None:
    """Durable backups survive an interruption after every restore step."""
    base = tmp_path / "deliverables"
    backup = base / ".error-analysis.generations" / ("a" * 32) / "backup"
    backup.mkdir(parents=True)
    targets = ("one", "two", "originally-absent")
    existing = {"one", "two"}
    for name in existing:
        (base / name).write_bytes(f"new-{name}".encode())
        (backup / name).write_bytes(f"old-{name}".encode())
    (base / "originally-absent").write_bytes(b"new-absent")
    replacements = 0

    def _interrupt_after_replace(source: Path, target: Path) -> None:
        nonlocal replacements
        os.replace(source, target)
        replacements += 1
        if replacements == interrupt_after:
            raise OSError("simulated process interruption")

    monkeypatch.setattr(
        _publication,
        "_replace_file",
        _interrupt_after_replace,
    )
    with pytest.raises(OSError, match="simulated"):
        _publication._restore_targets(base, backup, targets, existing)

    assert all((backup / name).is_file() for name in existing)
    monkeypatch.setattr(_publication, "_replace_file", os.replace)
    _publication._restore_targets(base, backup, targets, existing)

    assert (base / "one").read_bytes() == b"old-one"
    assert (base / "two").read_bytes() == b"old-two"
    assert not (base / "originally-absent").exists()


def test_retry_is_idempotent(
    publication_state: tuple[Path, OutputRoot, CurationLabels],
) -> None:
    """Retrying the same complete generation performs no replacement."""
    _output, root, labels = publication_state
    computation = _compute(root, labels)
    first = publish_error_analysis(
        root.layout,
        computation,
        mutation_is_safe=root.mutation_snapshot_is_safe,
    )
    before = _canonical_bytes(root)

    second = publish_error_analysis(
        root.layout,
        computation,
        mutation_is_safe=root.mutation_snapshot_is_safe,
    )

    assert second.generation == first.generation
    assert second.already_published
    assert _canonical_bytes(root) == before


def test_retry_repairs_incomplete_generation(
    publication_state: tuple[Path, OutputRoot, CurationLabels],
) -> None:
    """A matching receipt is not current when its manifest is missing."""
    _output, root, labels = publication_state
    computation = _compute(root, labels)
    publish_error_analysis(
        root.layout,
        computation,
        mutation_is_safe=root.mutation_snapshot_is_safe,
    )
    manifest_path = (
        root.layout.deliverables_base / "error_analysis.manifest.json"
    )
    manifest_path.unlink()

    repaired = publish_error_analysis(
        root.layout,
        computation,
        mutation_is_safe=root.mutation_snapshot_is_safe,
    )

    assert repaired.already_published is False
    assert manifest_path.is_file()


def test_source_change_after_staging_removes_hidden_generation(
    publication_state: tuple[Path, OutputRoot, CurationLabels],
) -> None:
    """A pre-journal conflict leaves neither canonical nor staged artifacts."""
    _output, root, labels = publication_state
    computation = _compute(root, labels)
    checks = 0

    def _guard() -> bool:
        nonlocal checks
        checks += 1
        return checks == 1

    with pytest.raises(ErrorPublicationConflict, match="binding is stale"):
        publish_error_analysis(
            root.layout,
            computation,
            mutation_is_safe=_guard,
        )

    assert _canonical_bytes(root) == {}
    assert not (
        root.layout.deliverables_base / ".error-analysis.generations"
    ).exists()


def test_active_owner_blocks_publication(
    publication_state: tuple[Path, OutputRoot, CurationLabels],
) -> None:
    """A nonterminal launch owner appearing after compute fails closed."""
    output, root, labels = publication_state
    computation = _compute(root, labels)
    atomic_write_json(gui_launch_owner_path(output), {"status": "running"})

    with pytest.raises(ErrorPublicationConflict, match="actively owned"):
        publish_error_analysis(
            root.layout,
            computation,
            mutation_is_safe=root.mutation_snapshot_is_safe,
        )

    assert not root.layout.error_analysis_parquet.exists()


def test_input_change_between_before_snapshot_and_reads_is_rejected(
    publication_state: tuple[Path, OutputRoot, CurationLabels],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deterministic interleaving cannot pair old labels with new sources."""
    _output, root, labels = publication_state
    original = capture_error_source_fingerprints
    calls = 0

    def _capture_then_interleave(layout: Any) -> dict[str, str]:
        nonlocal calls
        calls += 1
        captured = original(layout)
        if calls == 1:
            layout.qc_review_state_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            layout.qc_review_state_path.write_text('{"revision": 2}\n')
        return captured

    monkeypatch.setattr(
        _publication,
        "capture_error_source_fingerprints",
        _capture_then_interleave,
    )

    with pytest.raises(ErrorPublicationConflict, match="during computation"):
        _compute(root, labels)


def test_source_change_during_receipt_commit_rolls_back(
    publication_state: tuple[Path, OutputRoot, CurationLabels],
) -> None:
    """Receipt replacement cannot commit a now-stale generation."""
    _output, root, labels = publication_state
    computation = _compute(root, labels)

    def _replace_then_change_source(source: Path, target: Path) -> None:
        os.replace(source, target)
        if target.name == "error_analysis.publication.json":
            root.layout.qc_review_state_path.write_text(
                '{"interleaved": true}\n',
                encoding="utf-8",
            )

    with pytest.raises(ErrorPublicationConflict, match="inputs changed"):
        publish_error_analysis(
            root.layout,
            computation,
            mutation_is_safe=root.mutation_snapshot_is_safe,
            replace_file=_replace_then_change_source,
        )

    assert _canonical_bytes(root) == {}


def test_pure_computation_has_no_filesystem_side_effects(
    publication_state: tuple[Path, OutputRoot, CurationLabels],
) -> None:
    """The CLI-shared compute seam never creates caches or artifacts."""
    _output, root, labels = publication_state
    before = {
        path.relative_to(root.root).as_posix(): path.read_bytes()
        for path in root.root.rglob("*")
        if path.is_file()
    }
    with labels._lock:
        label_snapshot = dict(labels.labels)
        categories = tuple(labels.categories())
        good = labels.filtered_df(root.clean_master_df).to_pandas()
    computation = compute_all_category_analysis(
        root.clean_master_df,
        labels=label_snapshot,
        categories=categories,
        good_pdf=good,
        good_mode="all_unlabeled",
        source_fingerprints=capture_error_source_fingerprints(root.layout),
    )
    after = {
        path.relative_to(root.root).as_posix(): path.read_bytes()
        for path in root.root.rglob("*")
        if path.is_file()
    }

    assert computation.categories
    assert before == after


def test_dash_activation_and_focus_requests_are_source_read_only(
    publication_state: tuple[Path, OutputRoot, CurationLabels],
) -> None:
    """Real Dash callback transport cannot publish during preview navigation."""
    output, root, labels = publication_state
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(
        [
            dcc.Store(
                id=viewer_ids.STORE_REMOVED_KEYS,
                data=labels.removed_keys_payload(),
            ),
            dbc.Tabs(id=viewer_ids.TABS_ID, active_tab=viewer_ids.TAB_ERROR_ID),
            build_error_tab_body(root, object()),
        ]
    )
    register_error_callbacks(app, root, labels)
    callback_key = next(
        key for key in app.callback_map if error_ids.ERROR_TABLE_ID in key
    )
    callback = app.callback_map[callback_key]
    outputs = [
        {
            "id": output_spec.component_id,
            "property": output_spec.component_property,
        }
        for output_spec in callback["output"]
    ]
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    values = {
        viewer_ids.STORE_REMOVED_KEYS: labels.removed_keys_payload(),
        viewer_ids.TABS_ID: viewer_ids.TAB_ERROR_ID,
        error_ids.ERROR_GOOD_MODE_TOGGLE_ID: "all_unlabeled",
        error_ids.STORE_ERROR_CATEGORY_ID: "debris",
    }
    inputs = [
        {
            **input_spec,
            "value": values[input_spec["id"]],
        }
        for input_spec in callback["inputs"]
    ]
    client = app.server.test_client()
    for changed in (
        f"{viewer_ids.TABS_ID}.active_tab",
        f"{error_ids.STORE_ERROR_CATEGORY_ID}.data",
    ):
        response = client.post(
            "/_dash-update-component",
            json={
                "output": callback_key,
                "outputs": outputs,
                "changedPropIds": [changed],
                "inputs": inputs,
                "state": [],
            },
        )
        assert response.status_code == 200, response.get_data(as_text=True)

    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
