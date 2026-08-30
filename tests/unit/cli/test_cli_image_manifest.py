"""Focused contract tests for the CLI-approved ``--image-manifest`` subset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from phenotypic._cli import _cli_directory_scanner as directory_scanner
from phenotypic._cli._cli_failure_tracker import work_id_for_image
from phenotypic._cli._cli_state_management import (
    create_initial_state,
    load_processing_state,
    validate_resume_compatibility,
)
from phenotypic._cli._cli_types import ExecutionConfig


@pytest.fixture
def image_tree(tmp_path: Path) -> Path:
    """Create a nested image tree with repeated capture names."""
    root = tmp_path / "images"
    for dataset in ("plate1", "plate2"):
        for name in ("img001.tiff", "img002.tiff"):
            image = root / dataset / name
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"{dataset}/{name}".encode())
    return root


@pytest.fixture
def pipeline_stub(tmp_path: Path) -> Path:
    path = tmp_path / "pipeline.json"
    path.write_text('{"operations": []}', encoding="utf-8")
    return path


def _manifest(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _config(
    pipeline: Path, input_path: Path, output_dir: Path, manifest: Path | None
) -> ExecutionConfig:
    assert "image_manifest" in ExecutionConfig.__dataclass_fields__
    return ExecutionConfig(
        pipeline_json=pipeline,
        input_path=input_path,
        output_dir=output_dir,
        image_type="GridImage",
        nrows=None,
        ncols=None,
        bit_depth=None,
        n_jobs=1,
        slurm_args={},
        force_local=True,
        wait=False,
        ext=".tiff",
        overlay_alpha=0.3,
        include_dataset_column=True,
        dry_run=False,
        sample=None,
        resume=False,
        retry_failures=False,
        skip_validation=True,
        image_manifest=manifest,
    )


def test_manifest_reader_filters_comments_and_binds_exact_file_bytes(
    tmp_path: Path,
) -> None:
    """Comment-only changes alter the recorded approval digest."""
    reader = getattr(directory_scanner, "read_image_manifest", None)
    digest = getattr(directory_scanner, "image_manifest_digest", None)
    assert callable(reader)
    assert callable(digest)
    manifest = _manifest(
        tmp_path / "approved.images",
        ["# minted list", "", "  plate2/img001.tiff  ", "plate1/img001.tiff"],
    )

    assert reader(manifest) == ["plate2/img001.tiff", "plate1/img001.tiff"]
    assert digest(manifest) == f"sha256:{hashlib.sha256(manifest.read_bytes()).hexdigest()}"


def test_manifest_reader_accepts_a_real_bom_and_refuses_empty_or_invalid_utf8(
    tmp_path: Path,
) -> None:
    """Editor encodings are explicit, while an empty list never means all input."""
    reader = getattr(directory_scanner, "read_image_manifest", None)
    error_type = getattr(directory_scanner, "ImageManifestError", None)
    assert callable(reader)
    assert isinstance(error_type, type)
    bom = tmp_path / "bom.images"
    bom.write_bytes(b"\xef\xbb\xbfplate1/img001.tiff\n")
    empty = _manifest(tmp_path / "empty.images", ["# no approved images", ""])
    invalid = tmp_path / "invalid.images"
    invalid.write_bytes(b"\xff\xfe")

    assert reader(bom) == ["plate1/img001.tiff"]
    with pytest.raises(error_type, match="lists no images"):
        reader(empty)
    with pytest.raises(error_type, match="not valid UTF-8"):
        reader(invalid)


def test_manifest_selection_is_a_checked_subset_with_parent_rooted_work_ids(
    image_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """Unknown/repeated lines cannot silently change the approved workload."""
    apply = getattr(directory_scanner, "apply_image_manifest", None)
    error_type = getattr(directory_scanner, "ImageManifestError", None)
    assert callable(apply)
    assert isinstance(error_type, type)
    scanned = directory_scanner.scan_directory_structure(image_tree)
    manifest = _manifest(
        tmp_path / "approved.images",
        ["plate2/img001.tiff", "plate1/img001.tiff"],
    )

    selected = apply(scanned, manifest, image_tree)

    assert {name: [image.name for image in images] for name, images in selected.items()} == {
        "plate1": ["img001.tiff"],
        "plate2": ["img001.tiff"],
    }
    config = _config(pipeline_stub, image_tree, tmp_path / "out", manifest)
    assert {
        work_id_for_image(config, dataset, image)[1]
        for dataset, images in selected.items()
        for image in images
    } == {"plate1/img001.tiff", "plate2/img001.tiff"}

    duplicate = _manifest(
        tmp_path / "duplicate.images",
        ["plate1/img001.tiff", "plate1/img001.tiff"],
    )
    with pytest.raises(error_type, match="more than once"):
        apply(scanned, duplicate, image_tree)
    unknown = _manifest(tmp_path / "unknown.images", ["plate1/missing.tiff"])
    with pytest.raises(error_type, match="not one of the images"):
        apply(scanned, unknown, image_tree)


def test_manifest_selection_accepts_absolute_paths_and_rejects_alias_duplicates(
    image_tree: Path, tmp_path: Path
) -> None:
    """One real image cannot be approved twice through a symlink spelling."""
    apply = getattr(directory_scanner, "apply_image_manifest", None)
    error_type = getattr(directory_scanner, "ImageManifestError", None)
    assert callable(apply)
    assert isinstance(error_type, type)
    original = image_tree / "plate1" / "img001.tiff"
    alias = image_tree / "plate1" / "alias.tiff"
    alias.symlink_to(original)
    scanned = directory_scanner.scan_directory_structure(image_tree)
    absolute = _manifest(tmp_path / "absolute.images", [str(original)])
    alias_duplicate = _manifest(
        tmp_path / "alias.images", ["plate1/img001.tiff", "plate1/alias.tiff"]
    )

    selected = apply(scanned, absolute, image_tree)
    assert [path.name for path in selected["plate1"]] == ["img001.tiff"]
    with pytest.raises(error_type, match="more than once"):
        apply(scanned, alias_duplicate, image_tree)


def test_manifest_digest_is_saved_and_resume_refuses_an_edited_manifest(
    image_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """A reused output cannot silently exchange one approved list for another."""
    manifest = _manifest(tmp_path / "approved.images", ["plate1/img001.tiff"])
    config = _config(pipeline_stub, image_tree, tmp_path / "out", manifest)
    state = create_initial_state(config, [], tmp_path / "out")

    assert state.config["image_manifest_digest"] == (
        f"sha256:{hashlib.sha256(manifest.read_bytes()).hexdigest()}"
    )
    _manifest(manifest, ["plate2/img001.tiff"])
    compatible, message = validate_resume_compatibility(state, config)
    assert compatible is False
    assert message is not None and "Image manifest mismatch" in message


def test_measure_resume_ignores_manifest_drift_seam(
    image_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """Measure mode reuses stores, rather than revalidating an input subset."""
    manifest = _manifest(tmp_path / "approved.images", ["plate1/img001.tiff"])
    saved = _config(pipeline_stub, image_tree, tmp_path / "out", manifest)
    state = create_initial_state(saved, [], tmp_path / "out")
    current = _config(pipeline_stub, tmp_path / "out", tmp_path / "out", None)
    current.measure_only = True

    assert validate_resume_compatibility(state, current) == (True, None)


def test_manifest_snapshot_remains_bound_after_validation_until_selection(
    image_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """Changing the file after resume validation cannot alter selected images."""
    snapshot_loader = getattr(directory_scanner, "load_image_manifest", None)
    apply = getattr(directory_scanner, "apply_image_manifest", None)
    assert callable(snapshot_loader)
    assert callable(apply)
    manifest = _manifest(tmp_path / "approved.images", ["plate1/img001.tiff"])
    saved = _config(pipeline_stub, image_tree, tmp_path / "out", manifest)
    saved_snapshot = snapshot_loader(manifest)
    saved.image_manifest_digest = saved_snapshot.digest
    state = create_initial_state(saved, [], tmp_path / "out")
    current = _config(pipeline_stub, image_tree, tmp_path / "out", manifest)
    snapshot = snapshot_loader(manifest)
    current.image_manifest_entries = snapshot.entries
    current.image_manifest_digest = snapshot.digest

    assert validate_resume_compatibility(state, current) == (True, None)
    _manifest(manifest, ["plate2/img001.tiff"])
    selected = apply(
        directory_scanner.scan_directory_structure(image_tree),
        manifest,
        image_tree,
        snapshot=snapshot,
    )

    assert {name: [path.name for path in paths] for name, paths in selected.items()} == {
        "plate1": ["img001.tiff"]
    }
    assert state.config["image_manifest_digest"] == snapshot.digest


@pytest.mark.parametrize("mode_args", [[], ["--mode", "process", "--layer", "gray"]])
def test_cli_applies_manifest_in_full_and_process_dry_runs(
    image_tree: Path, pipeline_stub: Path, tmp_path: Path, mode_args: list[str]
) -> None:
    """The Click boundary passes the approved subset into either input mode."""
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    manifest = _manifest(tmp_path / "approved.images", ["plate1/img001.tiff"])
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            *mode_args,
            "--pipeline",
            str(pipeline_stub),
            "--input",
            str(image_tree),
            "--output",
            str(tmp_path / "out"),
            "--image-manifest",
            str(manifest),
            "--dry-run",
            "--skip-validation",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "plate1" in result.output
    assert "plate2" not in result.output


def test_process_cli_manifest_selects_an_ome_zarr_input(
    pipeline_stub: Path, tmp_path: Path
) -> None:
    """A manifest treats an OME-Zarr store directory as one process input."""
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    input_root = tmp_path / "stores"
    store = input_root / "plate.ome.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    manifest = _manifest(tmp_path / "approved.images", [store.name])

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode", "process", "--layer", "gray", "--pipeline", str(pipeline_stub),
            "--input", str(input_root), "--output", str(tmp_path / "out"),
            "--image-manifest", str(manifest), "--dry-run", "--skip-validation",
        ],
    )

    assert result.exit_code == 0, result.output
    assert store.name in result.output


def test_cli_refuses_a_manifest_without_input_or_with_sample(
    image_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """A manifest needs its input coordinate system and cannot be sampled."""
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    manifest = _manifest(tmp_path / "approved.images", ["plate1/img001.tiff"])
    runner = CliRunner()
    missing_input = runner.invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(pipeline_stub),
            "--output",
            str(tmp_path / "out"),
            "--image-manifest",
            str(manifest),
        ],
    )
    sampled = runner.invoke(
        phenotypic_cli,
        [
            "--pipeline", str(pipeline_stub), "--input", str(image_tree),
            "--output", str(tmp_path / "sampled"), "--image-manifest", str(manifest),
            "--sample", "1", "--skip-validation",
        ],
    )

    assert missing_input.exit_code != 0
    assert "--image-manifest requires --input" in missing_input.output
    assert sampled.exit_code != 0
    assert "sample_excludes_manifest" in sampled.output


def test_cli_refuses_a_manifest_in_non_input_modes(tmp_path: Path) -> None:
    """Measure, recompile, and migrate do not have an input-manifest surface."""
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    manifest = _manifest(tmp_path / "approved.images", ["plate1/img001.tiff"])
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode", "recompile", "--output", str(tmp_path / "out"),
            "--image-manifest", str(manifest),
        ],
    )

    assert result.exit_code != 0
    assert "only supported with --mode full or --mode process" in result.output


def test_cli_continuations_preserve_the_manifest_digest(
    monkeypatch: pytest.MonkeyPatch,
    image_tree: Path,
    pipeline_stub: Path,
    tmp_path: Path,
) -> None:
    """Repeated continuations retain approval binding and reject later drift."""
    from click.testing import CliRunner

    import phenotypic.phenotypicCLI as cli

    class StopAfterStateSave:
        def execute(self, datasets, output_dir):
            raise SystemExit(0)

    monkeypatch.setattr(
        cli,
        "create_execution_strategy",
        lambda config, output_manager: StopAfterStateSave(),
    )
    manifest = _manifest(tmp_path / "approved.images", ["plate1/img001.tiff"])
    output_dir = tmp_path / "out"
    command = [
        "--pipeline", str(pipeline_stub), "--input", str(image_tree),
        "--output", str(output_dir), "--image-manifest", str(manifest),
        "--skip-validation",
    ]
    runner = CliRunner()

    first = runner.invoke(cli.phenotypic_cli, command)
    second = runner.invoke(cli.phenotypic_cli, command)
    third = runner.invoke(cli.phenotypic_cli, command)

    assert first.exit_code == second.exit_code == third.exit_code == 0
    state = load_processing_state(output_dir)
    assert state is not None and state.config["image_manifest_digest"]

    without_manifest = runner.invoke(
        cli.phenotypic_cli,
        [
            "--pipeline", str(pipeline_stub), "--input", str(image_tree),
            "--output", str(output_dir), "--skip-validation",
        ],
    )
    assert without_manifest.exit_code != 0
    assert "Image manifest mismatch" in without_manifest.output

    _manifest(manifest, ["plate2/img001.tiff"])
    edited = runner.invoke(cli.phenotypic_cli, command)
    assert edited.exit_code != 0
    assert "Image manifest mismatch" in edited.output
