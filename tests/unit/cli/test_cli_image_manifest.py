"""``--image-manifest``: the approved-subset input for the full CLI.

The manifest sits on the irreversible full-dataset deploy path: the MCP server
resolves ``parent ∩ group_filter`` at plan time, writes the list, and binds its
content digest into a token carrying a human's approval of a specific compute
spend (spec ``05-deploy-and-slurm.md`` §5.4). Two properties therefore matter
more than the parsing:

* a manifest run and a whole-directory run give the *same* image the *same*
  work ID, so continuation, retry, and SLURM reconciliation still line up; and
* a resume cannot quietly substitute a different manifest under the same
  ``--input``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic._cli._cli_directory_scanner import (
    ImageManifestError,
    apply_image_manifest,
    image_manifest_digest,
    read_image_manifest,
    scan_directory_structure,
)
from phenotypic._cli._cli_failure_tracker import work_id_for_image
from phenotypic._cli._cli_state_management import (
    create_initial_state,
    validate_resume_compatibility,
)
from phenotypic._cli._cli_types import ExecutionConfig, ProcessingState


# ---------------------------------------------------------------------------
# Fixtures: a two-dataset tree whose image *names* repeat across datasets.
# ---------------------------------------------------------------------------


@pytest.fixture
def collision_tree(tmp_path: Path) -> Path:
    """``in/plate1/img001.tiff`` and ``in/plate2/img001.tiff``, distinct bytes.

    Repeated basenames across datasets are ordinary in plate phenotyping — a
    scanner names every capture ``img001`` — and they are what makes the
    dataset-qualified relative path visible in a work ID. (They do not
    *collide* under a basename-only path: ``compute_work_id`` hashes
    ``dataset`` separately. See
    ``test_pointing_input_at_the_manifest_would_re_identify_every_image``.)
    """
    root = tmp_path / "in"
    for index, dataset in enumerate(("plate1", "plate2")):
        folder = root / dataset
        folder.mkdir(parents=True)
        for image in ("img001.tiff", "img002.tiff"):
            # Content, not pixels: nothing here decodes the file.
            (folder / image).write_bytes(f"{dataset}/{image}/{index}".encode())
    return root


@pytest.fixture
def pipeline_stub(tmp_path: Path) -> Path:
    """A file for ``file_sha256(config.pipeline_json)`` to digest."""
    path = tmp_path / "pipeline.json"
    path.write_text('{"operations": []}', encoding="utf-8")
    return path


def _config(
    *,
    pipeline: Path,
    input_path: Path,
    output_dir: Path,
    image_manifest: Path | None = None,
) -> ExecutionConfig:
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
        image_manifest=image_manifest,
    )


def _write_manifest(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------


def test_reader_keeps_order_and_ignores_blanks_and_comments(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path / "plan.images",
        [
            "# minted by deploy_plan for pl_7f3a",
            "",
            "  plate2/img001.tiff  ",
            "   # indented comment",
            "plate1/img001.tiff",
        ],
    )

    assert read_image_manifest(manifest) == [
        "plate2/img001.tiff",
        "plate1/img001.tiff",
    ]


def test_reader_refuses_an_empty_manifest(tmp_path: Path) -> None:
    """Empty must not read as "process everything" — that is the whole risk."""
    manifest = _write_manifest(tmp_path / "plan.images", ["# nothing here", ""])

    with pytest.raises(ImageManifestError, match="lists no images"):
        read_image_manifest(manifest)


def test_reader_refuses_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ImageManifestError, match="Cannot read image manifest"):
        read_image_manifest(tmp_path / "absent.images")


def test_digest_is_the_file_bytes(tmp_path: Path) -> None:
    """The digest binds the artifact, so a comment-only edit invalidates it.

    Conservative on purpose: the server binds this same number into a plan
    token, and the human approved the file, not a normalization of it.
    """
    import hashlib

    first = _write_manifest(tmp_path / "a.images", ["plate1/img001.tiff"])
    second = _write_manifest(
        tmp_path / "b.images", ["# same set", "plate1/img001.tiff"]
    )

    assert image_manifest_digest(first) == hashlib.sha256(
        first.read_bytes()
    ).hexdigest()
    assert image_manifest_digest(first) != image_manifest_digest(second)


# ---------------------------------------------------------------------------
# Applying the manifest to a scan
# ---------------------------------------------------------------------------


def test_manifest_selects_a_subset_across_datasets(
    collision_tree: Path, tmp_path: Path
) -> None:
    manifest = _write_manifest(
        tmp_path / "plan.images",
        ["plate2/img002.tiff", "plate1/img001.tiff"],
    )
    scanned = scan_directory_structure(collision_tree)

    selected = apply_image_manifest(scanned, manifest, collision_tree)

    assert {name: [p.name for p in paths] for name, paths in selected.items()} == {
        "plate1": ["img001.tiff"],
        "plate2": ["img002.tiff"],
    }


def test_a_dataset_no_entry_names_disappears(
    collision_tree: Path, tmp_path: Path
) -> None:
    manifest = _write_manifest(tmp_path / "plan.images", ["plate1/img001.tiff"])
    scanned = scan_directory_structure(collision_tree)

    selected = apply_image_manifest(scanned, manifest, collision_tree)

    assert list(selected) == ["plate1"]


def test_absolute_and_relative_entries_name_the_same_image(
    collision_tree: Path, tmp_path: Path
) -> None:
    scanned = scan_directory_structure(collision_tree)
    relative = _write_manifest(
        tmp_path / "rel.images", ["plate1/img001.tiff"]
    )
    absolute = _write_manifest(
        tmp_path / "abs.images",
        [str(collision_tree / "plate1" / "img001.tiff")],
    )

    assert apply_image_manifest(
        scanned, relative, collision_tree
    ) == apply_image_manifest(scanned, absolute, collision_tree)


def test_within_a_dataset_scan_order_wins_over_manifest_order(
    collision_tree: Path, tmp_path: Path
) -> None:
    """The work list must not depend on how the manifest was written."""
    manifest = _write_manifest(
        tmp_path / "plan.images",
        ["plate1/img002.tiff", "plate1/img001.tiff"],
    )
    scanned = scan_directory_structure(collision_tree)

    selected = apply_image_manifest(scanned, manifest, collision_tree)

    assert [p.name for p in selected["plate1"]] == [
        "img001.tiff",
        "img002.tiff",
    ]


def test_an_entry_outside_the_scan_is_an_error(
    collision_tree: Path, tmp_path: Path
) -> None:
    """Silent omission would shrink the set the human approved."""
    outsider = tmp_path / "elsewhere" / "img009.tiff"
    outsider.parent.mkdir()
    outsider.write_bytes(b"outsider")
    manifest = _write_manifest(tmp_path / "plan.images", [str(outsider)])
    scanned = scan_directory_structure(collision_tree)

    with pytest.raises(ImageManifestError, match="not one of the images"):
        apply_image_manifest(scanned, manifest, collision_tree)


def test_a_nonexistent_entry_is_an_error(
    collision_tree: Path, tmp_path: Path
) -> None:
    manifest = _write_manifest(
        tmp_path / "plan.images", ["plate1/img404.tiff"]
    )
    scanned = scan_directory_structure(collision_tree)

    with pytest.raises(ImageManifestError, match="img404.tiff"):
        apply_image_manifest(scanned, manifest, collision_tree)


def test_a_repeated_entry_is_an_error(
    collision_tree: Path, tmp_path: Path
) -> None:
    """Deduplicating would process fewer images than the approved count."""
    manifest = _write_manifest(
        tmp_path / "plan.images",
        ["plate1/img001.tiff", "plate1/img001.tiff"],
    )
    scanned = scan_directory_structure(collision_tree)

    with pytest.raises(ImageManifestError, match="more than"):
        apply_image_manifest(scanned, manifest, collision_tree)


# ---------------------------------------------------------------------------
# Aliasing — one approved line must never become two units of compute
# ---------------------------------------------------------------------------


@pytest.fixture
def aliased_tree(collision_tree: Path) -> Path:
    """``plate1/copy.tiff`` is a symlink to ``plate1/img001.tiff``.

    Symlinked image trees are ordinary staging practice on this cluster — a
    "run" directory of links into an archive — so a scan holding two
    spellings of one real file is not an exotic shape.
    """
    (collision_tree / "plate1" / "copy.tiff").symlink_to(
        collision_tree / "plate1" / "img001.tiff"
    )
    return collision_tree


def test_a_symlink_alias_does_not_multiply_the_selection(
    aliased_tree: Path, tmp_path: Path
) -> None:
    """One entry, one image — even when the scan holds an alias of it.

    Selecting by *resolved-path membership* keeps both ``copy.tiff`` and
    ``img001.tiff``, since they resolve to the same real file. Their relative
    paths differ, so that is two work IDs, two HDFs, two measure passes —
    twice the compute a human approved, with nothing in the run to notice.
    """
    scanned = scan_directory_structure(aliased_tree)
    assert sorted(p.name for p in scanned["plate1"]) == [
        "copy.tiff",
        "img001.tiff",
        "img002.tiff",
    ]
    manifest = _write_manifest(tmp_path / "plan.images", ["plate1/img001.tiff"])

    selected = apply_image_manifest(scanned, manifest, aliased_tree)

    assert {
        name: [p.name for p in paths] for name, paths in selected.items()
    } == {"plate1": ["img001.tiff"]}


def test_an_entry_naming_the_alias_selects_the_alias(
    aliased_tree: Path, tmp_path: Path
) -> None:
    """The spelling the manifest was written as is the image that runs.

    The approved artifact names ``copy.tiff``, and that path is what the run's
    work ID is derived from, so resolving the entry to its target's spelling
    would process something other than what was listed.
    """
    scanned = scan_directory_structure(aliased_tree)
    manifest = _write_manifest(tmp_path / "plan.images", ["plate1/copy.tiff"])

    selected = apply_image_manifest(scanned, manifest, aliased_tree)

    assert {
        name: [p.name for p in paths] for name, paths in selected.items()
    } == {"plate1": ["copy.tiff"]}


def test_naming_both_spellings_of_one_file_is_still_a_duplicate(
    aliased_tree: Path, tmp_path: Path
) -> None:
    """Two lines naming one real file stay an error, not a silent collapse."""
    scanned = scan_directory_structure(aliased_tree)
    manifest = _write_manifest(
        tmp_path / "plan.images", ["plate1/img001.tiff", "plate1/copy.tiff"]
    )

    with pytest.raises(ImageManifestError, match="more than"):
        apply_image_manifest(scanned, manifest, aliased_tree)


def test_a_symlinked_dataset_directory_does_not_multiply_the_selection(
    collision_tree: Path, tmp_path: Path
) -> None:
    """The alias can be a whole dataset, which puts the copy under another key."""
    (collision_tree / "plate1_run2").symlink_to(
        collision_tree / "plate1", target_is_directory=True
    )
    scanned = scan_directory_structure(collision_tree)
    assert "plate1_run2" in scanned
    manifest = _write_manifest(tmp_path / "plan.images", ["plate1/img001.tiff"])

    selected = apply_image_manifest(scanned, manifest, collision_tree)

    assert {
        name: [p.name for p in paths] for name, paths in selected.items()
    } == {"plate1": ["img001.tiff"]}


def test_the_selected_count_must_equal_the_manifest_count(
    collision_tree: Path, tmp_path: Path
) -> None:
    """The backstop invariant, independent of the aliasing it anticipates.

    Fed a scan that lists one image twice — the shape any future aliasing
    would produce — selection is refused rather than run. "The count approved
    is the count that runs" is a checked property here, not one emergent from
    how the lookup happens to be written today.
    """
    duplicated = {
        "plate1": [
            collision_tree / "plate1" / "img001.tiff",
            collision_tree / "plate1" / "img001.tiff",
        ]
    }
    manifest = _write_manifest(tmp_path / "plan.images", ["plate1/img001.tiff"])

    with pytest.raises(ImageManifestError, match="names 1 image"):
        apply_image_manifest(duplicated, manifest, collision_tree)


def test_a_byte_order_mark_does_not_hide_the_first_entry(
    collision_tree: Path, tmp_path: Path
) -> None:
    """A BOM'd manifest reads as its paths, not as one phantom unknown path.

    Read as plain UTF-8 the mark lands inside the first entry and the run
    fails claiming ``'\\ufeffplate1/img001.tiff'`` is not under ``--input`` —
    fail-closed, but it sends a human debugging a server-written file after an
    unknown-path problem instead of an encoding one. The digest is over the
    raw bytes, so tolerating the mark moves nothing the server bound.
    """
    manifest = tmp_path / "plan.images"
    manifest.write_bytes(b"\xef\xbb\xbfplate1/img001.tiff\n")

    assert read_image_manifest(manifest) == ["plate1/img001.tiff"]

    selected = apply_image_manifest(
        scan_directory_structure(collision_tree), manifest, collision_tree
    )
    assert {
        name: [p.name for p in paths] for name, paths in selected.items()
    } == {"plate1": ["img001.tiff"]}


# ---------------------------------------------------------------------------
# Work-ID equivalence — the reason --input stays the parent directory
# ---------------------------------------------------------------------------


def _work_entries(
    config: ExecutionConfig, image_paths_by_dataset: dict[str, list[Path]]
) -> dict[str, tuple[str, str]]:
    """Map ``"<dataset>/<name>" -> (work_id, normalized_relative_path)``."""
    return {
        f"{dataset}/{image.name}": work_id_for_image(config, dataset, image)
        for dataset, images in image_paths_by_dataset.items()
        for image in images
    }


def test_work_ids_are_identical_between_a_manifest_and_a_parent_run(
    collision_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """The manifest narrows the work list; it must not touch image identity.

    If it did, a subset run could not continue, retry, or reconcile against a
    whole-parent run of the same images — and ``EXPECTED_WORK_IDS`` in the
    SLURM array would stop matching.

    The relative paths are asserted alongside the ids because the ids alone
    are a false green: both sides call the same ``work_id_for_image``, so they
    agree even when that function is broken. The dataset-qualified relative
    path is the property that actually distinguishes a parent-rooted identity
    from a basename-only one — measured, not assumed (a mutant that always
    took the basename passed the id-only version of this test).
    """
    output_dir = tmp_path / "out"
    manifest = _write_manifest(
        tmp_path / "plan.images",
        ["plate1/img001.tiff", "plate2/img001.tiff"],
    )
    scanned = scan_directory_structure(collision_tree)

    parent_config = _config(
        pipeline=pipeline_stub,
        input_path=collision_tree,
        output_dir=output_dir,
    )
    manifest_config = _config(
        pipeline=pipeline_stub,
        input_path=collision_tree,
        output_dir=output_dir,
        image_manifest=manifest,
    )
    selected = apply_image_manifest(scanned, manifest, collision_tree)

    parent_entries = _work_entries(parent_config, scanned)
    manifest_entries = _work_entries(manifest_config, selected)

    assert set(manifest_entries) == {
        "plate1/img001.tiff",
        "plate2/img001.tiff",
    }
    assert {
        key: relative for key, (_, relative) in manifest_entries.items()
    } == {
        "plate1/img001.tiff": "plate1/img001.tiff",
        "plate2/img001.tiff": "plate2/img001.tiff",
    }
    assert manifest_entries == {
        key: parent_entries[key] for key in manifest_entries
    }


def test_pointing_input_at_the_manifest_would_re_identify_every_image(
    collision_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """The rejected design, demonstrated rather than described.

    ``work_id_for_image`` takes ``Path(image_path.name)`` when ``input_path``
    is a file and ``relative_to(input_path)`` when it is a directory. Point
    ``--input`` at the manifest and every image's identity changes, so a
    subset run can no longer continue or reconcile against a parent run of the
    same images.

    Note what this is *not*: the two ``img001.tiff`` files do not collide,
    even byte-identical ones, because ``compute_work_id`` hashes ``dataset``
    as its own field. The harm is divergence from the parent run, which is
    disqualifying on its own — a run that cannot resume against the dataset it
    is a subset of.
    """
    manifest = _write_manifest(
        tmp_path / "plan.images",
        ["plate1/img001.tiff", "plate2/img001.tiff"],
    )
    scanned = scan_directory_structure(collision_tree)
    selected = apply_image_manifest(scanned, manifest, collision_tree)
    output_dir = tmp_path / "out"

    parent_rooted = _work_entries(
        _config(
            pipeline=pipeline_stub,
            input_path=collision_tree,
            output_dir=output_dir,
            image_manifest=manifest,
        ),
        selected,
    )
    file_rooted = _work_entries(
        _config(
            pipeline=pipeline_stub,
            input_path=manifest,
            output_dir=output_dir,
        ),
        selected,
    )

    assert {relative for _, relative in file_rooted.values()} == {
        "img001.tiff"
    }
    for key, (work_id, _) in file_rooted.items():
        assert work_id != parent_rooted[key][0], key


# ---------------------------------------------------------------------------
# Resume — the server's own pre-submit drift guard
# ---------------------------------------------------------------------------


def _state_for(config: ExecutionConfig, output_dir: Path) -> ProcessingState:
    return create_initial_state(config, [], output_dir)


def test_resume_rejects_a_different_manifest_under_the_same_input(
    collision_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """``input_path`` equality does not identify the image set."""
    output_dir = tmp_path / "out"
    first = _write_manifest(tmp_path / "a.images", ["plate1/img001.tiff"])
    second = _write_manifest(tmp_path / "b.images", ["plate2/img001.tiff"])

    saved = _state_for(
        _config(
            pipeline=pipeline_stub,
            input_path=collision_tree,
            output_dir=output_dir,
            image_manifest=first,
        ),
        output_dir,
    )
    resumed = _config(
        pipeline=pipeline_stub,
        input_path=collision_tree,
        output_dir=output_dir,
        image_manifest=second,
    )

    compatible, message = validate_resume_compatibility(saved, resumed)

    assert compatible is False
    assert message is not None and "Image manifest mismatch" in message


def test_resume_accepts_the_same_manifest(
    collision_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    manifest = _write_manifest(tmp_path / "a.images", ["plate1/img001.tiff"])
    config = _config(
        pipeline=pipeline_stub,
        input_path=collision_tree,
        output_dir=output_dir,
        image_manifest=manifest,
    )

    assert validate_resume_compatibility(_state_for(config, output_dir), config) == (
        True,
        None,
    )


def test_resume_rejects_a_manifest_whose_contents_changed(
    collision_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """Same path, different bytes — the drift a path comparison cannot see."""
    output_dir = tmp_path / "out"
    manifest = _write_manifest(tmp_path / "a.images", ["plate1/img001.tiff"])
    config = _config(
        pipeline=pipeline_stub,
        input_path=collision_tree,
        output_dir=output_dir,
        image_manifest=manifest,
    )
    saved = _state_for(config, output_dir)

    _write_manifest(manifest, ["plate2/img001.tiff"])

    compatible, message = validate_resume_compatibility(saved, config)

    assert compatible is False
    assert message is not None and "Image manifest mismatch" in message


def test_resume_refuses_when_the_manifest_has_gone_missing(
    collision_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """Reachable, not exotic: the server collects a token's .images file.

    A refusal with a reason, never a traceback out of a function whose
    contract is ``(bool, message)``.
    """
    output_dir = tmp_path / "out"
    manifest = _write_manifest(tmp_path / "a.images", ["plate1/img001.tiff"])
    config = _config(
        pipeline=pipeline_stub,
        input_path=collision_tree,
        output_dir=output_dir,
        image_manifest=manifest,
    )
    saved = _state_for(config, output_dir)
    manifest.unlink()

    compatible, message = validate_resume_compatibility(saved, config)

    assert compatible is False
    assert message is not None and "cannot be read" in message


def test_resume_of_a_whole_parent_run_is_unaffected(
    collision_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    config = _config(
        pipeline=pipeline_stub,
        input_path=collision_tree,
        output_dir=output_dir,
    )

    assert validate_resume_compatibility(_state_for(config, output_dir), config) == (
        True,
        None,
    )


def test_a_state_predating_the_flag_resumes_without_one_and_refuses_with_one(
    collision_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """Legacy states carry no key, and absent must mean "no manifest".

    Reading absent as "unknown, skip the check" would let a manifest be
    introduced on resume against a run that processed the whole parent.
    """
    output_dir = tmp_path / "out"
    no_manifest = _config(
        pipeline=pipeline_stub,
        input_path=collision_tree,
        output_dir=output_dir,
    )
    legacy = _state_for(no_manifest, output_dir)
    del legacy.config["image_manifest_digest"]

    assert validate_resume_compatibility(legacy, no_manifest) == (True, None)

    with_manifest = _config(
        pipeline=pipeline_stub,
        input_path=collision_tree,
        output_dir=output_dir,
        image_manifest=_write_manifest(
            tmp_path / "a.images", ["plate1/img001.tiff"]
        ),
    )
    compatible, message = validate_resume_compatibility(legacy, with_manifest)
    assert compatible is False
    assert message is not None and "Image manifest mismatch" in message


def test_the_recorded_digest_is_the_manifest_content_digest(
    collision_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """The server binds this same number into the plan token (§5.4).

    Asserted against an independently computed SHA-256 of the file's bytes,
    not against ``image_manifest_digest``: calling the same function on both
    sides would agree under *any* definition of the digest, including one over
    the resolved image set, which is exactly the definition the cross-side
    contract rules out.
    """
    import hashlib

    output_dir = tmp_path / "out"
    manifest = _write_manifest(tmp_path / "a.images", ["plate1/img001.tiff"])
    state = _state_for(
        _config(
            pipeline=pipeline_stub,
            input_path=collision_tree,
            output_dir=output_dir,
            image_manifest=manifest,
        ),
        output_dir,
    )

    assert (
        state.config["image_manifest_digest"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert state.config["image_manifest_digest"] == image_manifest_digest(
        manifest
    )


def test_the_manifest_stays_out_of_the_processing_configuration_digest(
    collision_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """Work IDs hash the processing config; adding the manifest would move them."""
    from phenotypic._cli._cli_failure_tracker import (
        processing_configuration_digest,
    )

    output_dir = tmp_path / "out"
    without = _config(
        pipeline=pipeline_stub,
        input_path=collision_tree,
        output_dir=output_dir,
    )
    with_manifest = _config(
        pipeline=pipeline_stub,
        input_path=collision_tree,
        output_dir=output_dir,
        image_manifest=_write_manifest(
            tmp_path / "a.images", ["plate1/img001.tiff"]
        ),
    )

    assert processing_configuration_digest(
        without
    ) == processing_configuration_digest(with_manifest)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_the_flag_requires_input(tmp_path: Path) -> None:
    """A manifest without a parent has no coordinate system to resolve in."""
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    manifest = _write_manifest(tmp_path / "a.images", ["plate1/img001.tiff"])
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "recompile",
            "--output",
            str(tmp_path / "out"),
            "--image-manifest",
            str(manifest),
        ],
    )

    assert result.exit_code != 0
    assert "--image-manifest requires --input" in result.output


def _dry_run_plan(output: str) -> dict[str, list[str]]:
    """Parse the dry-run's "Datasets Discovered" block into {dataset: images}.

    The plan is read back as a *set of (dataset, image) pairs* rather than as
    the ``selected N of M`` echo, because the echo is computed beside the call
    it is meant to police: a CLI that accepted the manifest and then scanned
    the whole parent would still print a plausible count. The plan is what the
    run would actually process.
    """
    plan: dict[str, list[str]] = {}
    dataset: str | None = None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Dataset: "):
            dataset = stripped[len("Dataset: ") :]
            plan[dataset] = []
        elif stripped.startswith("Total images across all datasets"):
            break
        elif stripped.startswith("- ") and dataset is not None:
            plan[dataset].append(stripped[2:])
    return plan


def test_the_cli_runs_exactly_the_images_the_manifest_names(
    collision_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """The wiring, end to end: ``--input`` + ``--image-manifest`` narrows the plan.

    ``apply_image_manifest`` is tested to death in isolation above, but this
    call site is the only place a manifest ever narrows real work. Without a
    test through the command, replacing that call with the identity function —
    manifest accepted, echoed about, and then ignored while the run processes
    the entire parent directory — is a green suite and an approval gate that
    only decorates.

    ``--dry-run`` prints the plan after the manifest applies and before any
    image is opened, so the fixture's byte-content files are enough.
    """
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    manifest = _write_manifest(
        tmp_path / "plan.images",
        ["plate2/img002.tiff", "plate1/img001.tiff"],
    )

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "full",
            "--pipeline",
            str(pipeline_stub),
            "--input",
            str(collision_tree),
            "--output",
            str(tmp_path / "out"),
            "--image-manifest",
            str(manifest),
            "--dry-run",
            "--skip-validation",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _dry_run_plan(result.output) == {
        "plate1": ["img001.tiff"],
        "plate2": ["img002.tiff"],
    }


def test_the_cli_refuses_a_manifest_naming_an_image_outside_input(
    collision_tree: Path, pipeline_stub: Path, tmp_path: Path
) -> None:
    """A manifest the scan cannot account for stops the run, not just the flag.

    The reader's refusals only matter if the command surfaces them: an
    ``ImageManifestError`` swallowed at the call site would fall through to a
    whole-parent run, which is the same failure as ignoring the manifest.
    """
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    outsider = tmp_path / "elsewhere" / "img009.tiff"
    outsider.parent.mkdir()
    outsider.write_bytes(b"outsider")
    manifest = _write_manifest(tmp_path / "plan.images", [str(outsider)])

    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "full",
            "--pipeline",
            str(pipeline_stub),
            "--input",
            str(collision_tree),
            "--output",
            str(tmp_path / "out"),
            "--image-manifest",
            str(manifest),
            "--dry-run",
            "--skip-validation",
        ],
    )

    assert result.exit_code != 0
    assert "not one of the images" in result.output
    assert _dry_run_plan(result.output) == {}
