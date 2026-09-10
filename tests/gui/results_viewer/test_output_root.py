"""Unit tests for :mod:`phenotypic._gui.results_viewer._output_root`.

Validates ``OutputRoot.discover`` against tmp-path fixtures that mimic
the on-disk layout produced by ``python -m phenotypic`` and exercises
the small read helpers (``overlay_path``, ``has_overlay``,
``image_pairs``, ``column_value_sets``, ``pipeline_summary``).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from phenotypic._gui.results_viewer import _output_root
from phenotypic._gui.results_viewer._output_root import (
    OutputRoot,
    OutputSnapshotChangedError,
    _all_parse_as_float,
    sandbox_viewer_cache_root,
)
from phenotypic.sdk_ import (
    master_measurements_parquet_path,
    measurements_parquet_path,
)

from tests._output_layout import build_complete_viewer_run
from tests._output_layout import write_complete_manifest, write_pipeline_json
from phenotypic.schema import CULTURE, EXPERIMENT, GENETIC, IMAGE


def _discover(root: Path) -> OutputRoot:
    """Discover with a test-owned cache outside the selected output."""
    source = Path(root).resolve()
    return OutputRoot.discover(
        source,
        cache_root=source.parent / ".test-phenotypic-viewer-cache",
    )


def _write_master_parquet(root: Path, df: pl.DataFrame) -> None:
    """Write ``master_measurements.parquet`` under ``root/deliverables/``."""
    target = master_measurements_parquet_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(target)


def _tree_bytes(root: Path) -> tuple[tuple[str, ...], dict[str, bytes]]:
    """Capture relative directories and exact file bytes."""
    directories = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_dir()
        )
    )
    files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    return directories, files


def _published_output(root: Path) -> Path:
    """A run whose processing inventory is EXHAUSTIVE, not read-only bounded.

    `_make_minimal_output` publishes no proofs, so its run state is
    `incomplete` and discovery binds it with bounded structural anchors --
    five entries, none of them an overlay or a per-image parquet. The three
    tests below are about artifacts being *tracked*, so they need the
    exhaustive path, and the only thing that reaches it now is a real run
    proof over the accepted inventory. A `manifest.json` used to be enough;
    §4.2 demotes it.

    `_make_minimal_output` is left alone deliberately: 25 other tests in this
    file depend on its unpublished shape and pass.
    """
    return build_complete_viewer_run(
        root,
        frame=pl.DataFrame(
            {
                "Metadata_Dataset": ["plate", "plate"],
                str(IMAGE.IMAGE_NAME): ["a", "b"],
                "Size_Area": [10.0, 20.0],
            }
        ),
        stems=("a", "b"),
    )


def _make_minimal_output(
    root: Path,
    dataset: str = "d1",
    *,
    with_overlays: bool = True,
    write_master: bool = True,
) -> pl.DataFrame:
    """Build a minimal CLI-style output directory under ``root``.

    Discovery still enumerates datasets from ``results/`` (per-image
    hdf/measurements live there); overlay PNGs now live under
    ``deliverables/overlays/<dataset>/``.

    Args:
        root: Existing tmp dir to populate.
        dataset: Single dataset name to create under ``results/``.
        with_overlays: If ``True``, touch overlay PNG files for the
            stems used in the master frame.
        write_master: If ``True``, write the master parquet.

    Returns:
        The DataFrame written to ``master_measurements.parquet`` (so
        tests can compare against expected unique sets).
    """

    # discovery still enumerates datasets from results/
    (root / "results" / dataset / "hdf").mkdir(parents=True)
    (root / "results" / dataset / "measurements").mkdir(parents=True)

    df = pl.DataFrame(
        {
            "Metadata_Dataset": [dataset, dataset],
            str(IMAGE.IMAGE_NAME): ["a", "b"],
            "Metadata_Strain": ["s1", "s2"],
            "Size_Area": [100.0, 200.0],
        }
    )
    if write_master:
        _write_master_parquet(root, df)
        write_complete_manifest(root, total_images=2)
    if with_overlays:
        overlays = root / "deliverables" / "overlays" / dataset
        overlays.mkdir(parents=True, exist_ok=True)
        for stem in ("a", "b"):
            (overlays / f"{stem}.png").touch()
    return df


def test_discover_succeeds_on_well_formed_root(tmp_path: Path) -> None:
    """A complete output dir yields a populated ``OutputRoot``."""

    df = _make_minimal_output(tmp_path)
    out = _discover(tmp_path)

    assert out.root == tmp_path.resolve()
    assert out.master_df.height == df.height
    assert str(GENETIC.STRAIN) in out.column_value_sets
    assert not out.cache_dir.is_relative_to(tmp_path.resolve())
    assert out.cache_dir.name == "dzi"
    assert out.source_fingerprint.startswith("sha256:")


def test_external_cache_path_is_pure_and_owned_by_sandbox(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    sandbox = tmp_path / "sandbox"
    _make_minimal_output(source)

    cache_root = sandbox_viewer_cache_root(sandbox)
    out = OutputRoot.discover(
        source,
        cache_root=cache_root,
    )

    assert out.cache_dir.is_relative_to(cache_root)
    assert not out.cache_dir.exists()
    assert not (source / ".viewer_cache").exists()


def test_discover_prefers_post_applied_mirror_over_master(
    tmp_path: Path,
) -> None:
    """When ``measurements.parquet`` exists, viewer reads it (post-applied)."""
    _make_minimal_output(tmp_path)

    # Seed a post-applied mirror that differs from master (extra "post_tag"
    # column simulates what _seed_measurements writes after post runs).
    mirror_df = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1", "d1"],
            str(IMAGE.IMAGE_NAME): ["a", "b"],
            "Metadata_Strain": ["s1", "s2"],
            "Size_Area": [100.0, 200.0],
            "post_tag": ["tagged", "tagged"],
        }
    )
    mirror_path = measurements_parquet_path(tmp_path)
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    mirror_df.write_parquet(mirror_path)

    out = _discover(tmp_path)
    assert "post_tag" in out.master_df.columns
    assert out.master_df["post_tag"].to_list() == ["tagged", "tagged"]


def test_discover_falls_back_to_master_when_mirror_absent(
    tmp_path: Path,
) -> None:
    """Mid-run / legacy outputs without ``measurements.parquet`` use master."""
    df = _make_minimal_output(tmp_path)
    # No measurements.parquet — only master.
    assert not measurements_parquet_path(tmp_path).exists()

    out = _discover(tmp_path)
    # Display frame falls back to the clean master.
    assert out.master_df.height == df.height
    assert "post_tag" not in out.master_df.columns


def test_discover_missing_master_raises(tmp_path: Path) -> None:
    """No ``master_measurements.parquet`` raises ``FileNotFoundError``."""

    _make_minimal_output(tmp_path, write_master=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        _discover(tmp_path)
    msg = str(excinfo.value)
    assert "master_measurements.parquet" in msg
    assert "python -m phenotypic" in msg


def test_discover_without_results_dir_boots_standalone(tmp_path: Path) -> None:
    """A deliverables-only bundle (no ``results/``) now discovers successfully.

    Task 4: ``BundleLayout``-backed discovery boots from a deliverables bundle
    alone — datasets are recovered from the master frame's ``Metadata_Dataset``
    column. ``results/``-backed capabilities (``has_results``/``store_path``)
    simply report unavailable.
    """

    df = pl.DataFrame(
        {"Metadata_Dataset": ["d1"], str(IMAGE.IMAGE_NAME): ["a"]}
    )
    _write_master_parquet(tmp_path, df)

    out = _discover(tmp_path)
    assert out.has_results is False
    assert out.store_path("d1", "a") is None
    assert "d1" in out.master_df[str(EXPERIMENT.DATASET)].to_list()


def test_discover_dataset_from_master_with_empty_results(
    tmp_path: Path,
) -> None:
    """Datasets are data-driven: a master's ``Metadata_Dataset`` wins over an empty ``results/``."""

    (tmp_path / "results").mkdir()
    df = pl.DataFrame(
        {"Metadata_Dataset": ["d1"], str(IMAGE.IMAGE_NAME): ["a"]}
    )
    _write_master_parquet(tmp_path, df)

    out = _discover(tmp_path)
    assert out.image_pairs(out.master_df) == [("d1", "a")]


def test_discover_results_with_no_overlays_succeeds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A dataset dir without ``overlays/`` is allowed; the picker disables those entries."""

    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    df = pl.DataFrame(
        {"Metadata_Dataset": ["d1"], str(IMAGE.IMAGE_NAME): ["a"]}
    )
    _write_master_parquet(tmp_path, df)

    with caplog.at_level("WARNING"):
        out = _discover(tmp_path)
    assert out.master_df.height == 1
    assert any("overlays" in rec.message.lower() for rec in caplog.records)
    assert out.has_overlay("d1", "a") is False


def test_discover_missing_imagefile_column_raises(tmp_path: Path) -> None:
    """Missing the ``Metadata_ImageName`` image-stem column raises."""

    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    _write_master_parquet(
        tmp_path, pl.DataFrame({"Metadata_Dataset": ["d1"], "Other": ["x"]})
    )
    with pytest.raises(ValueError) as excinfo:
        _discover(tmp_path)
    assert str(IMAGE.IMAGE_NAME) in str(excinfo.value)


def test_discover_aliases_imagename_when_imagefile_absent(
    tmp_path: Path,
) -> None:
    """Legacy ``Metadata_ImageName`` alone satisfies the image-stem requirement.

    A pre-flip master carries the legacy ``Metadata_ImageName`` column and lacks
    the canonical ``Metadata_ImageName``; discovery must alias it forward.
    """

    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    (tmp_path / "results" / "d1" / "measurements" / "a.parquet").touch()
    _write_master_parquet(
        tmp_path,
        pl.DataFrame(
            {"Metadata_Dataset": ["d1"], "Metadata_ImageName": ["a"]}
        ),
    )

    out = _discover(tmp_path)
    assert str(IMAGE.IMAGE_NAME) in out.master_df.columns
    assert out.master_df[str(IMAGE.IMAGE_NAME)].to_list() == ["a"]


def test_discover_backfills_dataset_from_filesystem(tmp_path: Path) -> None:
    """When master lacks ``Metadata_Dataset``, it is recovered from the per-image parquets."""

    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    (tmp_path / "results" / "d1" / "measurements" / "a.parquet").touch()
    (tmp_path / "results" / "d2" / "measurements").mkdir(parents=True)
    (tmp_path / "results" / "d2" / "measurements" / "b.parquet").touch()
    _write_master_parquet(
        tmp_path,
        pl.DataFrame(
            {str(IMAGE.IMAGE_NAME): ["a", "b"], "Size_Area": [100.0, 200.0]}
        ),
    )

    out = _discover(tmp_path)
    assert str(EXPERIMENT.DATASET) in out.master_df.columns
    pairs = out.image_pairs(out.master_df)
    assert pairs == [("d1", "a"), ("d2", "b")]


def test_legacy_backfill_parquets_are_part_of_snapshot_revision(
    tmp_path: Path,
) -> None:
    """Every per-image parquet consulted for backfill invalidates the snapshot."""
    _published_output(tmp_path)
    measurements = tmp_path / "results" / "plate" / "measurements"
    measurements.mkdir(parents=True, exist_ok=True)
    legacy_parquet = measurements / "a.parquet"
    legacy_parquet.write_bytes(b"first")

    output = _discover(tmp_path)
    legacy_parquet.write_bytes(b"second")

    assert output.snapshot_is_current() is False
    refreshed = _discover(tmp_path)
    assert refreshed.source_fingerprint != output.source_fingerprint


def test_discover_retries_complete_read_after_snapshot_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre/post mismatch retries from the first source read."""
    _published_output(tmp_path)
    overlay = tmp_path / "deliverables" / "overlays" / "plate" / "a.png"
    real_is_current = _output_root.inventory_is_current
    calls = 0

    def _mutate_before_first_verification(
        inventory,
        *,
        source_root,
        cancellation,
        progress,
    ):
        nonlocal calls
        calls += 1
        if calls == 1:
            overlay.write_bytes(b"new-revision")
        return real_is_current(
            inventory,
            source_root=source_root,
            cancellation=cancellation,
            progress=progress,
        )

    monkeypatch.setattr(
        _output_root,
        "inventory_is_current",
        _mutate_before_first_verification,
    )

    updates = []
    output = OutputRoot.discover(
        tmp_path,
        cache_root=tmp_path.parent / ".test-phenotypic-viewer-cache",
        progress_callback=updates.append,
    )

    assert calls == 2
    assert {update.attempt for update in updates} == {1, 2}
    phase_rank = {
        "classifying": 0,
        "inventory": 1,
        "measurements": 2,
        "indexing": 3,
        "verifying": 4,
        "complete": 5,
    }
    for attempt in (1, 2):
        ranks = [
            phase_rank[update.phase]
            for update in updates
            if update.attempt == attempt
        ]
        assert ranks == sorted(ranks)
    assert output.snapshot_is_current() is True


def test_the_mirrorless_fallback_distinguishes_a_v1_master_from_a_v2_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """§7.3, at the viewer's one master-fallback (P6 Task 7b).

    Both runs bind, and both bind the *same* frame -- the clean master --
    because the mirror is absent in both. What differs is what that frame is
    worth, and therefore what the viewer says about it:

    * **v1** (pre-inversion) joined user metadata per image, so the master
      carries it. The fallback is complete apart from the post ops, and no
      warning is owed.
    * **v2** (post-inversion) moved the join to finalization, so the master
      is intrinsic identity plus measurements. Every metadata-driven surface
      is presence-guarded and therefore shows **nothing rather than raising**
      -- indistinguishable to the user from a run given no ``--metadata``.

    Asserting on the diagnostic rather than on the column sets is deliberate:
    the column sets differ by construction here, which would be true even if
    the branch did not exist.
    """
    v1_root = tmp_path / "v1"
    v2_root = tmp_path / "v2"
    _make_minimal_output(v1_root)  # its master carries Metadata_Strain
    _make_minimal_output(v2_root, write_master=False)
    _write_master_parquet(
        v2_root,
        pl.DataFrame(
            {
                str(EXPERIMENT.DATASET): ["d1", "d1"],
                str(IMAGE.IMAGE_NAME): ["a", "b"],
                "Size_Area": [100.0, 200.0],
            }
        ),
    )
    write_complete_manifest(v2_root, total_images=2)
    assert not measurements_parquet_path(v1_root).exists()
    assert not measurements_parquet_path(v2_root).exists()

    caplog.clear()
    with caplog.at_level("INFO", logger=_output_root.__name__):
        v1 = _discover(v1_root)
    v1_warnings = [
        r.message for r in caplog.records if r.levelname == "WARNING"
    ]

    caplog.clear()
    with caplog.at_level("INFO", logger=_output_root.__name__):
        v2 = _discover(v2_root)
    v2_warnings = [
        r.message for r in caplog.records if r.levelname == "WARNING"
    ]

    # Both took the fallback -- the display frame IS the clean master --
    # so the difference below is the verdict on that frame, not the path.
    assert v1.master_df.equals(v1.clean_master_df)
    assert v2.master_df.equals(v2.clean_master_df)

    assert not any("user metadata" in m for m in v1_warnings), (
        "a v1 master carries its own metadata; the fallback is complete and "
        f"owes no warning, but got: {v1_warnings}"
    )
    assert any("carries no user metadata" in m for m in v2_warnings), (
        "a v2 master without its mirror leaves every metadata surface empty "
        f"with nothing raised; that must be said, but got: {v2_warnings}"
    )


def test_one_currency_check_replaces_two() -> None:
    """§11: ``snapshot_is_current()`` + ``refresh_state_is_current()`` -> one.

    Two overlapping fingerprints with different lifecycles is audit S2, and
    the fix is one owner, not two better-synchronised ones. The surviving
    polled owner is ``snapshot_is_current``. The descriptor's
    ``consumed_state_fingerprint`` stays -- discovery's torn-read guard and
    ``require_session_snapshot_current``'s second question both need it --
    but it is deliberately not re-exposed as a property on ``OutputRoot``,
    because a one-line accessor there is what invited the per-tick
    comparison in the first place.
    """
    assert not hasattr(OutputRoot, "refresh_state_is_current")
    assert not hasattr(OutputRoot, "consumed_state_fingerprint")
    assert hasattr(OutputRoot, "snapshot_is_current")
    descriptor = _output_root.OutputSnapshotDescriptor
    assert "consumed_state_fingerprint" in descriptor.__dataclass_fields__


def test_a_chmod_does_not_report_changed_on_disk(tmp_path: Path) -> None:
    """Audit S3, at the consumer.

    ``_inventory_is_current`` used to compare ``st_ctime_ns``, which moves on
    chmod, chown, hardlink and ``rsync -a`` -- all routine on a shared HPC
    filesystem, and each one made the whole binding report "Changed on disk".
    """
    _make_minimal_output(tmp_path)
    output = _discover(tmp_path)
    assert output.snapshot_is_current() is True

    files = [
        (output.root / entry.relative_path, entry.mtime_ns)
        for entry in output.processing_inventory.entries
        if entry.kind == "file"
    ]
    assert files, "no inventoried files: the chmod below would be a no-op"
    before = {path: path.stat().st_ctime_ns for path, _ in files}
    for path, _ in files:
        path.chmod(0o600)
        path.chmod(0o644)
    # Without this the test could pass on a filesystem where chmod leaves
    # ctime alone, i.e. while exercising nothing at all.
    assert any(
        path.stat().st_ctime_ns != before[path] for path, _ in files
    ), "chmod did not move ctime here, so this test is not exercising S3"
    # ... and it is only ctime that moved, so nothing else explains a pass.
    assert all(path.stat().st_mtime_ns == mtime for path, mtime in files)

    assert output.snapshot_is_current() is True


def test_mutable_viewer_state_does_not_stale_processing_snapshot(
    tmp_path: Path,
) -> None:
    """GUI-owned state is refresh-visible without invalidating image reads.

    Audit S2, at the consumer. Every path this test rewrites is one the GUI
    itself writes -- ``_curation_labels.py`` owns the mirror and the labels,
    ``_qc_tab/_rebuild.py`` owns the resolved pipeline config -- so the
    currency check must stay ``True`` through all of them. Comparing them
    against the frozen binding is what made marking one colony report the
    viewer's own write back to the user as external drift.
    """
    frame = _make_minimal_output(tmp_path)
    mirror = measurements_parquet_path(tmp_path)
    frame.write_parquet(mirror)
    write_pipeline_json(tmp_path, json.dumps({"name": "first"}))

    output = _discover(tmp_path)
    first_source = output.source_fingerprint
    first_consumed = output.snapshot.consumed_state_fingerprint

    frame.with_columns(pl.lit("changed").alias("Mutable_State")).write_parquet(
        mirror
    )
    output.layout.curation_labels_parquet.parent.mkdir(parents=True, exist_ok=True)
    output.layout.curation_labels_parquet.write_bytes(b"labels-revision")
    output.layout.custom_categories_json.write_text(
        '{"categories": ["debris"]}',
        encoding="utf-8",
    )
    output.layout.qc_duckdb.write_bytes(b"qc-revision")
    output.layout.qc_review_state_path.write_text(
        '{"reviewed": ["group-1"]}',
        encoding="utf-8",
    )
    write_pipeline_json(tmp_path, json.dumps({"name": "second"}))

    assert output.snapshot_is_current() is True
    # The construction gate is the one thing that still looks at consumed
    # state, and it is not on any poll -- so a curation click leaves the
    # badge, the tiles and the mutation guard alone while a *rebuild* of the
    # session still refuses to straddle two revisions of the mirror.
    with pytest.raises(OutputSnapshotChangedError):
        output.require_session_snapshot_current(context="Test")

    refreshed = _discover(tmp_path)
    assert refreshed.source_fingerprint == first_source
    assert refreshed.snapshot.consumed_state_fingerprint != first_consumed
    assert refreshed.cache_dir == output.cache_dir
    assert "Mutable_State" in refreshed.master_df.columns
    assert refreshed.pipeline_summary == "second"


def test_discover_retries_when_consumed_state_changes_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refresh never binds Results data to a torn viewer-state revision."""
    _make_minimal_output(tmp_path)
    review_state = (
        tmp_path / "deliverables" / "qc" / "review_state.json"
    )
    review_state.parent.mkdir(parents=True)
    review_state.write_text('{"revision": 1}', encoding="utf-8")
    real_fingerprint = _output_root._cancellable_paths_fingerprint
    calls = 0

    def _mutate_after_first_consumed_fingerprint(
        paths,
        *,
        root,
        cancellation,
    ):
        nonlocal calls
        result = real_fingerprint(
            paths,
            root=root,
            cancellation=cancellation,
        )
        calls += 1
        if calls == 1:
            review_state.write_text('{"revision": 2}', encoding="utf-8")
        return result

    monkeypatch.setattr(
        _output_root,
        "_cancellable_paths_fingerprint",
        _mutate_after_first_consumed_fingerprint,
    )

    output = _discover(tmp_path)

    assert calls == 4
    assert output.snapshot_is_current() is True
    # The torn-read guard survives the currency collapse: the descriptor
    # fingerprint is still captured pre- and post-read within one discovery,
    # which is what forced the retry that made `calls` 4 rather than 2 -- and
    # the binding it produced is one the construction gate accepts, because
    # the retry bound revision 2 rather than straddling both.
    output.require_session_snapshot_current(context="Test")


def test_discover_refuses_continuously_changing_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two unstable pre/post reads fail instead of binding mixed generations.

    **The churned file is deliberately not a declared artifact.** Mutating the
    overlay -- which is one -- makes the run ``incomplete`` on the retry, so
    the second attempt takes the bounded read-only path, does not track the
    overlay, and binds successfully. That degradation is correct (a bounded
    binding claims nothing about the artifacts it excludes, and the pixel
    routes revalidate what they serve), but it is not what this test is
    about. A stray file under ``results/`` keeps every declared artifact
    verifying -- so the run stays ``complete``, the inventory stays
    exhaustive, and both attempts genuinely observe a changing tree.
    """
    _published_output(tmp_path)
    churn = tmp_path / "results" / "plate" / "measurements" / "churn.bin"
    churn.parent.mkdir(parents=True, exist_ok=True)
    churn.write_bytes(b"revision-0")
    real_is_current = _output_root.inventory_is_current
    revision = 0

    def _mutate_before_every_verification(
        inventory,
        *,
        source_root,
        cancellation,
        progress,
    ):
        nonlocal revision
        revision += 1
        churn.write_bytes(f"revision-{revision}".encode())
        return real_is_current(
            inventory,
            source_root=source_root,
            cancellation=cancellation,
            progress=progress,
        )

    monkeypatch.setattr(
        _output_root,
        "inventory_is_current",
        _mutate_before_every_verification,
    )

    with pytest.raises(OutputSnapshotChangedError):
        _discover(tmp_path)


def test_column_value_sets_are_sorted_unique_str(tmp_path: Path) -> None:
    """``column_value_sets`` casts to str, dedupes, sorts, drops nulls."""

    _make_minimal_output(tmp_path)
    out = _discover(tmp_path)

    cvs = out.column_value_sets
    assert cvs[str(GENETIC.STRAIN)] == ["s1", "s2"]
    # Numeric column rendered as string.
    assert cvs["Size_Area"] == sorted({"100.0", "200.0"})
    # Every column on the master frame is represented.
    for column in out.master_df.columns:
        assert column in cvs


def test_column_value_sets_are_empty_for_a_list_valued_column(
    tmp_path: Path,
) -> None:
    """A column polars cannot stringify has no value set -- it does not raise.

    ``_compute`` casts to ``pl.String`` to build a filter value set.
    ``List`` and ``Array`` dtypes reject that cast outright (a type-level
    refusal, so ``strict=False`` does not help). A master carrying one is
    unusual but legal -- nothing stops a post step or a user-added column
    from being list-valued -- and every surface that asks "is this column
    offerable?" reaches this method. Raising here fails **app boot**, not
    just that one column, because building the layout asks every column
    for its value set.

    The empty list is the answer callers already know how to read: it is
    falsy for the axis menus' non-empty guard, and
    ``_all_parse_as_float([])`` is ``False``, so ``is_numeric_column``
    reports the column non-numeric rather than offering a range filter
    over values it cannot render.
    """

    _make_minimal_output(tmp_path)
    df = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    _write_master_parquet(
        tmp_path,
        df.with_columns(pl.Series("Centroid", [[5.0, 5.0], [6.0, 6.0]])),
    )
    out = _discover(tmp_path)

    assert out.master_df.schema["Centroid"] == pl.List(pl.Float64)
    assert out.column_value_sets["Centroid"] == []
    assert out.is_numeric_column("Centroid") is False
    # The stringifiable columns beside it are unaffected.
    assert out.column_value_sets["Size_Area"] == sorted({"100.0", "200.0"})


def test_column_value_sets_still_serve_a_struct_column(tmp_path: Path) -> None:
    """``Struct`` is nested but casts to ``String`` fine -- it keeps its set.

    Guards the fix above against being written as a nested-dtype check.
    ``DataType.is_nested()`` is ``True`` for ``Struct`` as well as for
    ``List``/``Array``, so excluding on nestedness would silently drop a
    column that has a perfectly good value set.
    """

    _make_minimal_output(tmp_path)
    df = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    _write_master_parquet(
        tmp_path,
        df.with_columns(pl.Series("Bounds", [{"a": 1}, {"a": 2}])),
    )
    out = _discover(tmp_path)

    assert out.master_df.schema["Bounds"] == pl.Struct({"a": pl.Int64})
    assert out.column_value_sets["Bounds"] == ["{1}", "{2}"]


def test_column_value_sets_outer_mapping_is_immutable(tmp_path: Path) -> None:
    """The mapping itself rejects ``__setitem__`` (``MappingProxyType``)."""

    _make_minimal_output(tmp_path)
    out = _discover(tmp_path)
    with pytest.raises(TypeError):
        out.column_value_sets["new_column"] = ["x"]  # type: ignore[index]


def test_overlay_path_returns_expected_absolute_path(tmp_path: Path) -> None:
    """``overlay_path`` resolves to ``<root>/deliverables/overlays/<ds>/<stem>.png``."""

    _make_minimal_output(tmp_path)
    out = _discover(tmp_path)
    expected = (
        tmp_path.resolve() / "deliverables" / "overlays" / "d1" / "a.png"
    )
    assert out.overlay_path("d1", "a") == expected


def test_has_overlay_distinguishes_present_and_absent(tmp_path: Path) -> None:
    """``has_overlay`` is True only for files that exist."""

    _make_minimal_output(tmp_path, with_overlays=False)
    # Touch only "a"; leave "b" absent.
    overlays = tmp_path / "deliverables" / "overlays" / "d1"
    overlays.mkdir(parents=True, exist_ok=True)
    (overlays / "a.png").touch()
    out = _discover(tmp_path)

    assert out.has_overlay("d1", "a") is True
    assert out.has_overlay("d1", "b") is False


def test_image_pairs_returns_sorted_unique_tuples(tmp_path: Path) -> None:
    """``image_pairs`` deduplicates and sorts the (dataset, stem) tuples."""

    _make_minimal_output(tmp_path)
    out = _discover(tmp_path)

    # Feed a frame with shuffled order and a duplicate row.
    df = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1", "d1", "d1"],
            str(IMAGE.IMAGE_NAME): ["b", "a", "a"],
        }
    )
    pairs = out.image_pairs(df)
    assert pairs == [("d1", "a"), ("d1", "b")]


def test_pipeline_summary_reads_name_from_pipeline_json(
    tmp_path: Path,
) -> None:
    """A valid ``pipeline.json`` with a ``name`` populates ``pipeline_summary``."""

    _make_minimal_output(tmp_path)
    write_pipeline_json(tmp_path, json.dumps({"name": "test_pipeline"}))
    out = _discover(tmp_path)
    assert out.pipeline_summary == "test_pipeline"


def test_pipeline_summary_is_none_when_missing_or_malformed(
    tmp_path: Path,
) -> None:
    """Missing or malformed ``pipeline.json`` yields ``pipeline_summary=None``."""

    # Case 1: missing → None.
    _make_minimal_output(tmp_path)
    assert _discover(tmp_path).pipeline_summary is None

    # Case 2: malformed JSON → None (does not raise).
    write_pipeline_json(tmp_path, "{not valid json")
    assert _discover(tmp_path).pipeline_summary is None

    # Case 3: parsed JSON dict with no ``name`` or ``class_name`` field → None.
    # Regression: previously returned the literal string ``"pipeline.json"``
    # via the new PIPELINE_JSON constant during the io_constants extraction
    # refactor (the agent substituted the constant where the original code
    # likely had ``return None``). Caught by opus review of PR #78.
    write_pipeline_json(tmp_path, json.dumps({"version": "1.0"}))
    assert _discover(tmp_path).pipeline_summary is None


def test_cache_dir_is_not_created_on_discover(tmp_path: Path) -> None:
    """Discovery computes the external path without writing it."""

    _make_minimal_output(tmp_path)
    out = _discover(tmp_path)
    assert not out.cache_dir.exists()


def test_discover_rejects_cache_root_inside_selected_output(
    tmp_path: Path,
) -> None:
    """A cache-owning caller cannot accidentally mutate the source tree."""
    _make_minimal_output(tmp_path)

    with pytest.raises(ValueError, match="must be external"):
        OutputRoot.discover(
            tmp_path,
            cache_root=tmp_path / ".phenotypic-gui" / "viewer_cache",
        )


def test_discover_leaves_legacy_qc_and_viewer_sidecar_byte_identical(
    tmp_path: Path,
) -> None:
    """Discovery never moves legacy topology or folds a viewer sidecar."""
    source = tmp_path / "run"
    _make_minimal_output(source)
    legacy_qc = source / "qc"
    legacy_qc.mkdir()
    (legacy_qc / "legacy.parquet").write_bytes(b"legacy-qc")
    sidecar = source / ".viewer_cache" / "qc_recipe.json"
    sidecar.parent.mkdir()
    sidecar.write_text('{"version": 1, "checks": []}', encoding="utf-8")
    before = _tree_bytes(source)

    _discover(source)

    assert _tree_bytes(source) == before


def test_all_parse_as_float_true_for_numeric_strings() -> None:
    assert _all_parse_as_float(["2", "10", "1.5"]) is True


def test_all_parse_as_float_false_for_mixed_or_empty() -> None:
    assert _all_parse_as_float(["2", "x", "10"]) is False
    assert _all_parse_as_float([]) is False


def test_column_value_sets_sorts_numeric_columns_numerically(tmp_path) -> None:
    """An all-numeric metadata column sorts 2 < 10, not lexically '10' < '2'."""
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    overlays = tmp_path / "deliverables" / "overlays" / "d1"
    overlays.mkdir(parents=True)
    df = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"] * 3,
            str(IMAGE.IMAGE_NAME): ["a", "b", "c"],
            "Metadata_Time": ["10", "2", "1"],
        }
    )
    _write_master_parquet(tmp_path, df)
    for stem in ("a", "b", "c"):
        (overlays / f"{stem}.png").touch()

    out = _discover(tmp_path)
    assert out.column_value_sets[str(CULTURE.TIME)] == ["1", "2", "10"]


def test_column_value_sets_keeps_lexical_for_text_columns(tmp_path) -> None:
    df = _make_minimal_output(tmp_path)  # has Metadata_Strain = s1, s2
    out = _discover(tmp_path)
    assert out.column_value_sets[str(GENETIC.STRAIN)] == sorted(
        df.get_column("Metadata_Strain").to_list()
    )


def test_is_numeric_column_true_for_float_measurement(tmp_path) -> None:
    _make_minimal_output(tmp_path)  # Size_Area is Float64
    out = _discover(tmp_path)
    assert out.is_numeric_column("Size_Area") is True


def test_is_numeric_column_true_for_numeric_string_metadata(tmp_path) -> None:
    (tmp_path / "results" / "d1" / "measurements").mkdir(parents=True)
    overlays = tmp_path / "deliverables" / "overlays" / "d1"
    overlays.mkdir(parents=True)
    df = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1", "d1"],
            str(IMAGE.IMAGE_NAME): ["a", "b"],
            "Metadata_Time": ["6", "24"],
        }
    )
    _write_master_parquet(tmp_path, df)
    for stem in ("a", "b"):
        (overlays / f"{stem}.png").touch()
    out = _discover(tmp_path)
    assert out.is_numeric_column(str(CULTURE.TIME)) is True


def test_is_numeric_column_false_for_text_and_missing(tmp_path) -> None:
    _make_minimal_output(tmp_path)  # Metadata_Strain = s1, s2
    out = _discover(tmp_path)
    assert out.is_numeric_column(str(GENETIC.STRAIN)) is False
    assert out.is_numeric_column("NoSuchColumn") is False
