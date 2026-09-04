"""The run-state schema gate: detect the old shape, refuse it, name the fix.

P7 Task 1, built in P1 (CAN-11). The gate has to exist before P3's clean break,
because on an unconverted tree ``authorized_measurement_sources`` returns an
empty mapping -- a *valid* answer, not a failure -- and ``finalize_run`` would
publish an empty master without raising.

Two invariants drive the shape of this file:

* **INV-DISCHARGEABLE** -- no verdict may be emitted that migrate cannot
  discharge. ``_refuse_unmigrated_output`` fires *before* ``--restart`` is
  handled, so a tree wrongly classified ``CONVERT`` is refused by every writing
  mode forever, escapable only by ``--overwrite``, which deletes the outputs.
  Both shape lists below are single sources of truth, shared by the
  classification tests and the discharge test, so adding a shape to one adds it
  to the other.
* **U-6** -- detection is by shape, never by ``state.version``. There is no
  version floor and no ``BELOW_FLOOR`` verdict.
"""

from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from phenotypic._cli import _cli_schema_gate
from phenotypic.sdk_ import _schema_shape
from phenotypic._cli._cli_schema_gate import (
    STATE_SCHEMA_VERSION,
    ConversionVerdict,
    describe_required_conversion,
    requires_conversion,
)
from phenotypic._cli._cli_staged_resume import stage3_completion_marker_path
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import (
    DIR_IMAGE_COMPLETE,
    DIR_PROGRESS,
    MIGRATION_REMEDY,
    PROCESSING_STATE_JSON,
    deliverables_dir,
    image_completion_marker_path,
    image_record_path,
    master_measurements_parquet_path,
    phenotypic_cache_dir,
    processing_state_path,
)

_WORK_IDS = {"plate": {"a.png": "w-a", "b.png": "w-b"}}

#: A pre-§4.2 ``datasets`` block: the derived per-dataset sets that are a cache
#: of a cache, and are deleted from the file by ``--mode migrate``.
_DATASETS_WITH_COMPLETED = {
    "plate": {
        "completed": ["a.png"],
        "failed": [],
        "errors": {},
        "initial_images": ["a.png", "b.png"],
    }
}

#: What ``datasets`` looks like after migrate strips the three derived keys.
_DATASETS_CONVERTED = {"plate": {"initial_images": ["a.png", "b.png"]}}


# --------------------------------------------------------------------------
# Tree builders. Each returns the output root it built.
# --------------------------------------------------------------------------


def _write_state(
    root: Path,
    *,
    version: str = "3.0.0",
    datasets: Mapping[str, object] | None = None,
    config: Mapping[str, object] | None = None,
    at_output_root: bool = False,
) -> Path:
    """Plant a ``processing_state.json`` with the fields a reader indexes."""
    payload = {
        "version": version,
        "pipeline_path": str(root / "pipeline.json"),
        "input_path": str(root / "input"),
        "output_dir": str(root),
        "timestamp": "2026-09-03T00:00:00",
        "execution_mode": "local",
        "last_updated": "2026-09-03T00:00:00",
        "datasets": dict(datasets or {}),
        "config": dict(config or {}),
    }
    # `at_output_root` is the pre-`.phenotypic/` layout, which has no public
    # path helper because nothing current writes it -- only a test that
    # deliberately builds that era spells it out.
    path = (
        root / PROCESSING_STATE_JSON
        if at_output_root
        else processing_state_path(root)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload)), encoding="utf-8")
    return path


def _legacy_marker_payload(stem: str) -> dict[str, object]:
    return {
        "version": 2,
        "work_id": f"w-{stem}",
        "dataset": "plate",
        "relative_image_path": f"plate/{stem}.png",
        "image_stem": stem,
        "mode": "full",
        "attempt_id": "attempt",
        "lifecycle_epoch": "gen",
        "artifacts": {},
        "completed_at": "2026-09-03T00:00:00.000+00:00",
    }


def _build_markers_era(root: Path) -> Path:
    """Today's shape: image markers, derived dataset sets, no restart epoch."""
    root.mkdir(parents=True, exist_ok=True)
    _write_state(
        root,
        datasets=_DATASETS_WITH_COMPLETED,
        config={"work_ids": _WORK_IDS, "success_markers_required": True},
    )
    _write_json(
        image_completion_marker_path(root, "plate", "a"),
        _legacy_marker_payload("a"),
    )
    return root


def _build_stage3_only(root: Path) -> Path:
    """Staged GPU run interrupted between Stage 3 and the image marker."""
    root.mkdir(parents=True, exist_ok=True)
    _write_state(
        root,
        datasets=_DATASETS_CONVERTED,
        config={"work_ids": _WORK_IDS, "restart_epoch": 0},
    )
    _write_json(
        stage3_completion_marker_path(root, "plate", "a"),
        {"image_stem": "a", "dataset": "plate"},
    )
    return root


def _build_derived_dataset_sets_only(root: Path) -> Path:
    """Signal 3 in isolation: identity is current, ``datasets`` is not.

    This is also the shape a **forward run writes after the whole change
    lands**, because no phase doc changes ``save_processing_state`` -- see the
    finding recorded against this task.
    """
    root.mkdir(parents=True, exist_ok=True)
    _write_state(
        root,
        datasets=_DATASETS_WITH_COMPLETED,
        config={"work_ids": _WORK_IDS, "restart_epoch": 0},
    )
    return root


def _build_no_restart_epoch(root: Path) -> Path:
    """Signal 4 in isolation: accepted inventory present, epoch absent."""
    root.mkdir(parents=True, exist_ok=True)
    _write_state(
        root,
        datasets=_DATASETS_CONVERTED,
        config={"work_ids": _WORK_IDS},
    )
    return root


def _build_pre_markers(root: Path) -> Path:
    """The v0.17.3 shape: schema ``"2.0.0"`` and no ``work_ids`` concept."""
    root.mkdir(parents=True, exist_ok=True)
    _write_state(
        root,
        version="2.0.0",
        datasets=_DATASETS_WITH_COMPLETED,
        config={"image_type": "brightfield", "ext": "png"},
    )
    return root


def _build_pre_markers_process(root: Path) -> Path:
    """MIG-11: a pre-markers ``--mode process`` run.

    No ``image_complete/`` and no ``results/`` tree -- its outputs are process
    layers under the mirrored input tree -- so signal 1 cannot fire and the
    classification rests on the state file alone.
    """
    root.mkdir(parents=True, exist_ok=True)
    _write_state(
        root,
        version="2.0.0",
        datasets=_DATASETS_WITH_COMPLETED,
        config={"process_only_layer": "rgb", "ext": "png"},
    )
    (root / "plate").mkdir(parents=True, exist_ok=True)
    (root / "plate" / "a.tiff").write_bytes(b"")
    return root


def _build_pre_markers_half_converted(root: Path) -> Path:
    """Signal 5 in isolation, and the only shape that isolates it.

    A pre-markers tree between P7's Task 3 and Task 2b: the state file has been
    converted -- ``restart_epoch`` written, the derived dataset sets stripped --
    but the ported promoter has not yet minted ``work_ids``, which is the one
    thing a pre-markers run never had. Signals 1-4 are all silent here, so if
    the absent-``work_ids`` signal were dropped this tree would classify
    ``None`` and a forward run would proceed against a tree with no accepted
    inventory at all.
    """
    root.mkdir(parents=True, exist_ok=True)
    _write_state(
        root,
        version="2.0.0",
        datasets=_DATASETS_CONVERTED,
        config={"restart_epoch": 0, "image_type": "brightfield", "ext": "png"},
    )
    return root


def _build_interrupted_migrate(root: Path) -> Path:
    """Half-converted: a record exists and the legacy marker is still live."""
    _build_markers_era(root)
    _write_json(
        image_record_path(root, "plate", "a"),
        {"work_id": "w-a", "stages": {"measured": {}}, "artifacts": {}},
    )
    return root


def _build_pre_phenotypic_dir(root: Path) -> Path:
    """Machine state at the output root, from before ``.phenotypic/``."""
    root.mkdir(parents=True, exist_ok=True)
    _write_state(
        root,
        datasets=_DATASETS_WITH_COMPLETED,
        config={"work_ids": _WORK_IDS},
        at_output_root=True,
    )
    _write_json(
        root / DIR_PROGRESS / DIR_IMAGE_COMPLETE / "plate" / "a.json",
        _legacy_marker_payload("a"),
    )
    return root


#: Every shape that must classify ``CONVERT``. **This list is the invariant.**
#: ``test_every_convert_verdict_is_dischargeable_by_one_migrate`` iterates the
#: same object, so a shape added here without a migrate arm fails there.
_EVERY_CONVERTIBLE_SHAPE: dict[str, Callable[[Path], Path]] = {
    "markers-era": _build_markers_era,
    "stage3-only": _build_stage3_only,
    "derived-dataset-sets-only": _build_derived_dataset_sets_only,
    "no-restart-epoch": _build_no_restart_epoch,
    "pre-markers": _build_pre_markers,
    "pre-markers-half-converted": _build_pre_markers_half_converted,
    "pre-markers-process": _build_pre_markers_process,
    "interrupted-migrate": _build_interrupted_migrate,
    "pre-phenotypic-dir": _build_pre_phenotypic_dir,
}


def _build_absent(root: Path) -> Path:
    """A path that does not exist. Never touched."""
    return root / "brand-new"


def _build_fresh(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_bundle_only(root: Path) -> Path:
    """A standalone deliverables bundle: no ``.phenotypic/`` at all.

    ``BundleLayout.detect`` supports it explicitly, and it trips none of the
    five signals because signal 5 requires a **present** state file.
    """
    deliverables_dir(root).mkdir(parents=True, exist_ok=True)
    master_measurements_parquet_path(root).write_bytes(b"PAR1")
    return root


def _build_converted(root: Path) -> Path:
    """What migrate leaves behind: records, epoch, no derived dataset sets."""
    root.mkdir(parents=True, exist_ok=True)
    _write_state(
        root,
        datasets=_DATASETS_CONVERTED,
        config={"work_ids": _WORK_IDS, "restart_epoch": 0},
    )
    _write_json(
        image_record_path(root, "plate", "a"),
        {"work_id": "w-a", "stages": {"measured": {}}, "artifacts": {}},
    )
    return root


def _build_converted_with_retained_legacy(root: Path) -> Path:
    """P7 Task 5 Step 1b renames the legacy trees aside rather than deleting.

    The retained copy must not make a converted tree look unconverted.
    """
    _build_converted(root)
    _write_json(
        phenotypic_cache_dir(root)
        / "legacy-v2"
        / DIR_IMAGE_COMPLETE
        / "plate"
        / "a.json",
        _legacy_marker_payload("a"),
    )
    return root


#: Every shape the gate must pass through untouched.
_EVERY_CURRENT_SHAPE: dict[str, Callable[[Path], Path]] = {
    "absent": _build_absent,
    "fresh": _build_fresh,
    "bundle-only": _build_bundle_only,
    "converted": _build_converted,
    "converted-with-retained-legacy": _build_converted_with_retained_legacy,
}

#: Payloads a present ``processing_state.json`` must survive as a verdict.
#: ``'{"config": {}}'`` is object-shaped and still malformed: it carries
#: neither ``version`` nor ``datasets``, so ``load_processing_state`` would
#: ``KeyError`` on it.
_UNREADABLE_STATE_PAYLOADS = (
    "{truncated",
    "null",
    "[]",
    '{"config": {}}',
    "",
)


def _fingerprint(root: Path) -> list[tuple[str, int, int]] | None:
    """Size and mtime of every path below *root*, or ``None`` if absent."""
    if not root.exists():
        return None
    entries = []
    for path in sorted(root.rglob("*")):
        stat = path.stat()
        entries.append(
            (
                path.relative_to(root).as_posix(),
                stat.st_size if path.is_file() else -1,
                stat.st_mtime_ns,
            )
        )
    return entries


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", sorted(_EVERY_CONVERTIBLE_SHAPE))
def test_every_convertible_shape_classifies_convert(
    tmp_path: Path, shape: str
) -> None:
    """Detection is by presence of the OLD shape, not absence of the new."""
    tree = _EVERY_CONVERTIBLE_SHAPE[shape](tmp_path / "run")

    assert requires_conversion(tree) is ConversionVerdict.CONVERT


@pytest.mark.parametrize("shape", sorted(_EVERY_CURRENT_SHAPE))
def test_every_current_shape_is_not_an_unconverted_tree(
    tmp_path: Path, shape: str
) -> None:
    """INV-DISCHARGEABLE's other half.

    A ``CONVERT`` on any of these strands a tree that never needed converting:
    a fresh directory would make every new run start with an error, a bundle
    has no ``.phenotypic/`` for migrate to touch, and a converted tree would be
    refused immediately after the command that was supposed to fix it.
    """
    tree = _EVERY_CURRENT_SHAPE[shape](tmp_path / "run")

    assert requires_conversion(tree) is None


def test_a_fresh_output_directory_is_not_an_unconverted_tree(
    tmp_path: Path,
) -> None:
    """An empty directory has no schema to be wrong about."""
    assert requires_conversion(tmp_path / "brand-new") is None
    assert describe_required_conversion(tmp_path, mode="full") is None


def test_the_pre_markers_shape_is_detected_by_absent_work_ids(
    tmp_path: Path,
) -> None:
    """MIG-14a. ``state.version`` cannot separate the floor.

    ``"2.0.0"`` is the value both at v0.17.3 and immediately before the marker
    commit, so the reliable signal is the absent ``work_ids`` key -- the
    concept did not exist at v0.17.3.

    A v0.17.3 tree trips three signals at once, so it cannot show that *this*
    one is load-bearing: deleting the ``work_ids`` check leaves it classified
    ``CONVERT`` by the derived dataset sets alone. The half-converted shape is
    the one that isolates it, and it is the assertion a mutation must break.
    """
    tree = _build_pre_markers(tmp_path / "run")

    assert requires_conversion(tree) is ConversionVerdict.CONVERT

    isolated = _build_pre_markers_half_converted(tmp_path / "half")
    assert requires_conversion(isolated) is ConversionVerdict.CONVERT

    # Same version string, current identity: the version is not the signal.
    other = tmp_path / "other"
    other.mkdir()
    _write_state(
        other,
        version="2.0.0",
        datasets=_DATASETS_CONVERTED,
        config={"work_ids": _WORK_IDS, "restart_epoch": 0},
    )
    assert requires_conversion(other) is None


def test_stage2_done_is_not_a_conversion_signal(tmp_path: Path) -> None:
    """U-9. ``stage2_done/`` stays a file and stays current.

    Firing on it would classify every modern GPU run ``CONVERT`` and strand
    it -- and unlike the marker trees, nothing converts it away.
    """
    from phenotypic._cli._cli_stage2_token import stage2_token_path

    tree = _build_converted(tmp_path / "run")
    _write_json(
        stage2_token_path(tree, "plate", "a"), {"objmap_shape": [4, 4]}
    )

    assert requires_conversion(tree) is None


@pytest.mark.parametrize("payload", _UNREADABLE_STATE_PAYLOADS)
def test_a_malformed_state_file_yields_a_verdict_not_an_exception(
    tmp_path: Path, payload: str
) -> None:
    """MIG-14b.

    A refusal gate that raises on a malformed tree is worse than the silent
    path it replaces, and one that mutates the tree while deciding is worse
    still -- which is why this reads raw JSON rather than
    ``load_processing_state``.
    """
    tree = tmp_path / "run"
    tree.mkdir()
    state_path = processing_state_path(tree)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(payload, encoding="utf-8")
    before = _fingerprint(tree)

    verdict = requires_conversion(tree)  # must not raise

    assert verdict is ConversionVerdict.UNREADABLE_STATE
    assert _fingerprint(tree) == before, (
        "requires_conversion wrote to the tree while deciding whether to "
        "refuse it -- load_processing_state's migrate_legacy_machine_state "
        "side effect leaked in"
    )


def test_an_unreadable_state_file_never_points_at_migrate(
    tmp_path: Path,
) -> None:
    """INV-DISCHARGEABLE. Migrate cannot repair a truncated state file.

    Sending the user there would strand the tree: every writing mode refuses,
    migrate changes nothing, and the only escape left is ``--overwrite``,
    which deletes the outputs.
    """
    tree = _build_markers_era(tmp_path / "run")
    processing_state_path(tree).write_text("{truncated", encoding="utf-8")

    assert requires_conversion(tree) is ConversionVerdict.UNREADABLE_STATE
    message = describe_required_conversion(tree, mode="full")
    assert message is not None
    assert MIGRATION_REMEDY not in message
    assert PROCESSING_STATE_JSON in message


def test_the_retained_legacy_tree_is_invisible_to_detection(
    tmp_path: Path,
) -> None:
    """CAN-12. Renaming aside preserves revert, not refusal."""
    tree = _build_converted_with_retained_legacy(tmp_path / "run")

    assert (phenotypic_cache_dir(tree) / "legacy-v2").is_dir()
    assert requires_conversion(tree) is None


@pytest.mark.parametrize(
    "shape",
    sorted(_EVERY_CONVERTIBLE_SHAPE) + sorted(_EVERY_CURRENT_SHAPE),
)
def test_requires_conversion_never_writes_to_the_tree(
    tmp_path: Path, shape: str
) -> None:
    """A gate must not mutate the tree it is deciding about.

    ``load_processing_state`` calls ``migrate_legacy_machine_state``
    (``_cli_state_management.py:108``), which relocates ``progress/`` and
    ``processing_state.json`` into ``.phenotypic/``. Reading raw JSON is the
    only way this stays true, and ``pre-phenotypic-dir`` is the shape that
    proves it: a relocating read would move both.
    """
    builders = {**_EVERY_CONVERTIBLE_SHAPE, **_EVERY_CURRENT_SHAPE}
    tree = builders[shape](tmp_path / "run")
    before = _fingerprint(tree)

    requires_conversion(tree)
    describe_required_conversion(tree, mode="full")

    assert _fingerprint(tree) == before


def test_the_gate_never_reaches_the_writing_state_reader() -> None:
    """The rule, pinned at the source rather than by observation.

    ``load_processing_state`` writes and raises; a future edit that reaches for
    it would pass the mutation tests above on any tree that happens to be
    already relocated. Names in docstrings are not references, so this walks
    the AST rather than the text -- both functions are *discussed* in these
    modules and neither may be *called* by either.

    **Both modules, because the detection moved.** The predicate now lives in
    ``sdk_/_schema_shape`` so that ``resolve_run_state`` can emit §4.3's
    reader advisory from it; walking only ``_cli_schema_gate`` would leave
    this guarantee pointing at the eighty lines that stayed behind while the
    two hundred it was written for went unwatched. A guard has to follow the
    code it guards.
    """
    from phenotypic.sdk_ import _schema_shape

    referenced: set[str] = set()
    for module in (_cli_schema_gate, _schema_shape):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                referenced.update(alias.name for alias in node.names)

    assert "load_processing_state" not in referenced
    assert "migrate_legacy_machine_state" not in referenced


# --------------------------------------------------------------------------
# Anchors: the gate's private names against their real writers
# --------------------------------------------------------------------------


def test_the_stage3_directory_name_matches_the_writer() -> None:
    """``_DIR_STAGE3_COMPLETE`` copies a segment its writer hand-joins."""
    written = stage3_completion_marker_path(Path("/out"), "plate", "a")

    assert written.parent.parent.name == _cli_schema_gate._DIR_STAGE3_COMPLETE


def test_the_image_complete_directory_name_matches_the_writer() -> None:
    written = image_completion_marker_path(Path("/out"), "plate", "a")

    assert written.parent.parent.name == DIR_IMAGE_COMPLETE


def test_the_recorded_schema_version_matches_what_the_writer_writes() -> None:
    """A bump of the writer's literal must land back in this module.

    ``STATE_SCHEMA_VERSION`` is never a detection signal (U-6), so nothing else
    would notice.
    """
    from phenotypic._cli._cli_state_management import create_initial_state

    source = inspect.getsource(create_initial_state)

    assert f'version="{STATE_SCHEMA_VERSION}.' in source


# --------------------------------------------------------------------------
# Refusal
# --------------------------------------------------------------------------


@pytest.fixture()
def cli_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """A ``--pipeline`` file and an ``--input`` directory that merely exist.

    The gate fires during argument validation, before either is opened.
    """
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text("{}", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    return pipeline, images


def _invoke(
    mode: str,
    output: Path,
    cli_inputs: tuple[Path, Path],
    *,
    dry_run: bool = False,
) -> Result:
    pipeline, images = cli_inputs
    args = ["--mode", mode, "--output", str(output)]
    if mode in ("full", "measure", "process"):
        args += ["--pipeline", str(pipeline)]
    if mode in ("full", "process"):
        args += ["--input", str(images)]
    if mode == "process":
        args += ["--layer", "rgb"]
    if dry_run:
        args.append("--dry-run")
    return CliRunner().invoke(phenotypic_cli, args)


@pytest.mark.parametrize("mode", ["full", "measure", "recompile", "process"])
def test_every_writing_mode_refuses_an_unconverted_tree(
    tmp_path: Path,
    cli_inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """D1: clean break.

    New code reads only the consolidated schema, so a mode that silently
    half-read a legacy tree would produce a run whose proofs certify nothing.

    Armed explicitly: the gate is inert on this build because at P1 the legacy
    shape and the current shape are the same shape. Arming here is what proves
    the wiring, the message and the mode coverage before P3 flips the constant.
    """
    monkeypatch.setattr(_schema_shape, "SCHEMA_GATE_ARMED", True)
    tree = _build_markers_era(tmp_path / "run")

    result = _invoke(mode, tree, cli_inputs)

    assert result.exit_code != 0, result.output
    assert MIGRATION_REMEDY in result.output, (
        "the refusal must name the command that fixes it; a refusal the user "
        "cannot act on is the bug class this whole change exists to remove"
    )


def test_a_legacy_tree_is_refused_before_the_clean_break_can_empty_it(
    tmp_path: Path,
    cli_inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CAN-11, and the reason this task moved from P7 to P1.

    Without the gate, P3's clean break turns a legacy tree into an empty master
    rather than an error -- because ``{}`` from
    ``authorized_measurement_sources`` is a VALID result, not a failure.
    """
    monkeypatch.setattr(_schema_shape, "SCHEMA_GATE_ARMED", True)
    tree = _build_pre_markers(tmp_path / "run")

    result = _invoke("full", tree, cli_inputs)

    assert result.exit_code != 0, result.output
    assert MIGRATION_REMEDY in result.output


def test_the_refusal_names_the_evidence_and_the_drain_rule(
    tmp_path: Path,
) -> None:
    """CAN-13: a tree migrated under a live old-build array reverts shape."""
    tree = _build_markers_era(tmp_path / "run")

    message = describe_required_conversion(tree, mode="full")

    assert message is not None
    assert DIR_IMAGE_COMPLETE in message
    assert "scancel" in message


def test_migrate_mode_is_never_refused_by_the_gate(
    tmp_path: Path,
    cli_inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MIG-19: guarding the remedy with its own predicate strands every tree.

    Dry-run, because the exemption is an argument-validation fact and a real
    conversion is P7's to exercise.
    """
    monkeypatch.setattr(_schema_shape, "SCHEMA_GATE_ARMED", True)
    tree = _build_markers_era(tmp_path / "run")

    result = _invoke("migrate", tree, cli_inputs, dry_run=True)

    assert "cannot read this output" not in result.output


def test_a_current_tree_is_not_refused_while_the_gate_is_unarmed(
    tmp_path: Path,
) -> None:
    """The inertness is a contract, not an accident.

    On this build the forward path still writes ``image_complete/``, still
    writes ``datasets.<ds>.completed`` and does not yet write
    ``restart_epoch``, so an armed gate would refuse every resume of every
    mode. ``requires_conversion`` is already correct; only the refusal waits.

    Calls the refusal directly rather than through the CLI: with the gate
    inert, a full invocation would run past validation and start a real run.
    """
    from phenotypic.phenotypicCLI import _refuse_unmigrated_output

    tree = _build_markers_era(tmp_path / "run")

    assert requires_conversion(tree) is ConversionVerdict.CONVERT

    _refuse_unmigrated_output(tree, mode="full")  # must not raise


def test_the_gate_is_armed_exactly_when_the_forward_path_stops_writing_markers(
    tmp_path: Path,
) -> None:
    """The arming condition, checked instead of remembered.

    ``SCHEMA_GATE_ARMED`` is ``False`` only because
    ``publish_image_success`` still writes ``image_complete/`` -- signal 1 --
    on this build. The moment P3 makes it write the consolidated record
    instead, that reason is gone and the gate must be armed in the same commit,
    or a legacy tree silently produces an empty master (CAN-11).

    This test is what forces the two to move together: it fails whichever one
    moves first.
    """
    from phenotypic._cli._cli_completion import publish_image_success

    tree = tmp_path / "run"
    artifact = tree / "results" / "plate" / "a.txt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"x")

    publish_image_success(
        tree,
        work_id="w-a",
        dataset="plate",
        relative_image_path="plate/a.png",
        image_stem="a",
        mode="full",
        attempt_id="attempt",
        lifecycle_epoch="gen",
        artifacts={"measurements": artifact},
    )
    writes_legacy_marker = image_completion_marker_path(
        tree, "plate", "a"
    ).is_file()

    assert writes_legacy_marker is not _schema_shape.SCHEMA_GATE_ARMED, (
        "publish_image_success and SCHEMA_GATE_ARMED disagree: arm the gate "
        "in the same commit that moves the publisher onto the consolidated "
        "record (P3 Task 2), or a legacy tree publishes an empty master"
    )


def test_the_gui_reports_rather_than_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4.3: an unconverted tree is an ADVISORY, not a gate.

    The GUI is a reader; refusing to display a legacy tree would be a
    regression from today, where it displays one. This is
    ``resolve_run_state``'s contract, asserted from here because this is where
    the refusal/advisory split is decided.

    **Both halves, because the advisory is gated on the same flag the refusal
    is.** They are two surfacings of one detection, so *"detection is correct
    now; only the surfacing waits"* governs both. It has to: at P1
    ``requires_conversion`` returns ``CONVERT`` for every tree the current
    build writes -- ``publish_image_success`` always creates
    ``image_complete/`` and ``save_processing_state`` always writes
    ``datasets.<ds>.completed`` -- so an ungated advisory would banner
    "run ``--mode migrate``" on every GUI output until P3, advising a
    conversion that does not convert ``.phenotypic/`` until P7.

    The disarmed assertion is deliberately one call and one condition. An
    earlier draft wrote ``advisories == () or not any(...)``, which passes two
    ways: a future change that empties ``advisories`` entirely would satisfy
    the first branch and the test would stop noticing.
    """
    from phenotypic.sdk_ import _schema_shape, resolve_run_state

    tree = _build_markers_era(tmp_path / "run")

    disarmed = resolve_run_state(tree, depth="deep")
    assert not any("migrate" in a for a in disarmed.advisories)

    monkeypatch.setattr(_schema_shape, "SCHEMA_GATE_ARMED", True)
    armed = resolve_run_state(tree, depth="deep")

    assert any("migrate" in advisory for advisory in armed.advisories)
    assert armed.completion in {
        "complete",
        "incomplete",
        "failed",
        "active",
    }, "the reader must return a verdict, never refuse"


def test_the_arming_flag_has_one_source() -> None:
    """One flag, one home, one patch point.

    Both consumers -- ``refuse_unconverted_schema`` here and
    ``resolve_run_state`` in ``sdk_`` -- read
    ``_schema_shape.SCHEMA_GATE_ARMED`` through the module, so P3 arms both
    with a single edit and a test arms both with a single patch.

    **Structural, not an identity comparison.** ``_cli_schema_gate.X is
    _schema_shape.X`` passes the instant someone writes
    ``SCHEMA_GATE_ARMED = False`` beside the import, because both are
    ``False`` today -- it would go green on exactly the divergence it exists
    to catch, and stay green until P3 flips one of them. What has to be
    pinned is that this module **never binds the name at all**: a copy here
    would read correctly while being inert under ``monkeypatch``, which is
    the trap that costs an afternoon because the flag is the last place
    anyone looks.

    Same guard shape, and same reason, as
    ``test_the_marker_schema_constants_have_exactly_one_home``.
    """
    source = Path(inspect.getfile(_cli_schema_gate)).read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            bound |= {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            bound.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            bound |= {alias.asname or alias.name for alias in node.names}

    assert "SCHEMA_GATE_ARMED" not in bound, (
        "_cli_schema_gate binds SCHEMA_GATE_ARMED again. A copy here reads "
        "correctly and is INERT under monkeypatch, so a test patching this "
        "module changes nothing -- read _schema_shape.SCHEMA_GATE_ARMED "
        "through the module instead."
    )
    assert not hasattr(_cli_schema_gate, "SCHEMA_GATE_ARMED"), (
        "the name is reachable on _cli_schema_gate, so someone will patch it"
    )


@pytest.mark.skip(
    reason=(
        "INV-DISCHARGEABLE's migrate half: `--mode migrate` does not yet "
        "convert `.phenotypic/` -- that is P7 Tasks 2, 2b and 3. The gate "
        "ships four phases early (CAN-11), so this assertion cannot hold "
        "until then. P7 Task 5 removes this mark; it is that phase's gate."
    )
)
@pytest.mark.parametrize("shape", sorted(_EVERY_CONVERTIBLE_SHAPE))
def test_every_convert_verdict_is_dischargeable_by_one_migrate(
    tmp_path: Path, cli_inputs: tuple[Path, Path], shape: str
) -> None:
    """INV-DISCHARGEABLE.

    A ``CONVERT`` that migrate cannot discharge strands the tree behind a
    refusal in every writing mode, escapable only by ``--overwrite``.

    This is the test that closes MIG-11, MIG-20, and the next shape nobody
    enumerated -- which is why it is parametrized over the shape list rather
    than written per shape.
    """
    tree = _EVERY_CONVERTIBLE_SHAPE[shape](tmp_path / "run")

    assert requires_conversion(tree) is ConversionVerdict.CONVERT
    result = _invoke("migrate", tree, cli_inputs)
    assert result.exit_code == 0, result.output
    assert requires_conversion(tree) is None, (
        f"{shape}: migrate ran successfully and the gate still refuses it"
    )
