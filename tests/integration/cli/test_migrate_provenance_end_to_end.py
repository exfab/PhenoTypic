"""Local and SLURM ``--mode migrate`` agree on a provenance-only tree.

The full-run topology has had an equivalence test all along, and it earned its
keep immediately: the moment `migrate_machine_state` was wired into the local
driver alone, it caught the SLURM tree keeping the **forward** run's
`processing_generation`.

The provenance-only topology -- store array -> seal -> finalizer -- had no such
test, and carried the identical gap for the same reason: `_run_migrate_owned`
is the local arm for **both** target kinds and converts machine state for each,
while nothing in this chain did. That gap was found by looking for siblings of
the first one, not by anything failing.

**That is what this file is for.** A topology with no equivalence test is not a
gap in the coverage of some behaviour; it is a gap in the ability to detect a
whole class of divergence, and it hid one. The comparison here is exact for the
same reason the full-run one is: the fields that move in this class are
configuration identity, not measurements, and comparing approximately would
hide a missing conversion rather than absorb variance.

It found a second divergence on its first run, and the two tests below now
differ in shape because of what it found. A **process tree** has stores, so
both arms can do the work and the claim is equivalence. A **pre-markers
process tree** has none, so the array topology has neither work nor a
vocabulary for what it would do -- there the claim is the named divergence:
local converts, SLURM refuses and says why.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import shutil

from click.testing import CliRunner
import pytest

from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.sdk_ import (
    image_record_path,
    phenotypic_cache_dir,
    resolve_processing_state_path,
)

#: The record fields that are stable across two migrations of one archive.
#: Deliberately a projection, mirroring the full-run snapshot: a record also
#: carries timestamps, which differ between any two runs and say nothing about
#: whether the two paths agree.
_STABLE_RECORD_KEYS = (
    "version",
    "dataset",
    "image_stem",
    "work_id",
    "provenance",
    "mode",
    "artifacts",
)


def _tree_digest(root: Path) -> str:
    """Return a path-and-content digest for one published artifact tree."""
    digest = hashlib.sha256()
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_process_tree(root: Path, stems: tuple[str, ...]) -> Path:
    """A ``--mode process`` output tree of schema-1 stores, no ``results/``.

    Minimal on purpose, and faithful: provenance migration reads one root
    ``zarr.json`` per store and writes a journal. No pixels are read, so real
    image data would add runtime without adding evidence.
    """
    for stem in stems:
        store = root / "dataset" / f"{stem}.ome.zarr"
        store.mkdir(parents=True)
        (store / "zarr.json").write_text(
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {
                        "ome": {"version": "0.5"},
                        "phenotypic": {
                            "store_schema_version": 3,
                            "provenance": {
                                "schema_version": 1,
                                "status": "complete",
                                "pipeline": None,
                                "retry_base_length": 0,
                                "operations": [],
                            },
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
    return root


def _build_pre_markers_process_tree(root: Path, stems: tuple[str, ...]) -> Path:
    """A pre-markers ``--mode process`` run: flat layers, no stores at all.

    The MIG-11 subject, and the sharper witness of the two: it has nothing to
    provenance-upgrade, so the machine-state conversion is the *only* work the
    chain does for it. A chain that skips the conversion produces an empty
    migration here rather than a subtly different one.
    """
    (root / "plate").mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (root / "plate" / f"{stem}.tiff").write_bytes(b"pixels-" + stem.encode())
    state = resolve_processing_state_path(root)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps(
            {
                "version": "2.0.0",
                "config": {"process_only_layer": "rgb"},
            }
        ),
        encoding="utf-8",
    )
    return root


def _script_indices(script: Path) -> list[int]:
    """Read the concrete work indexes emitted into one array script."""
    text = script.read_text(encoding="utf-8")
    entries = text.split("TASK_INDICES=(\n", 1)[1].split("\n)", 1)[0]
    return [int(entry.strip()) for entry in entries.splitlines()]


def _run_generated_provenance_worker_commands(
    *,
    chunk_scripts: list[Path],
    dispatcher_scripts: list[Path],
    finalizer_script: Path | None = None,
    continuation_dependency_kind: str = "afterany",
    output_dir: Path | None = None,
    generation: str | None = None,
) -> tuple[list[str], None]:
    """Synchronously execute the generated provenance worker commands.

    The provenance mirror of the full-run dispatcher fake: it replaces only
    the scheduler submission primitive, and everything it runs -- the scripts,
    their indexes, the immutable config -- was produced by production code.
    """
    from phenotypic._cli._cli_migrate_provenance_worker import (
        provenance_migration_worker_cli,
    )

    assert finalizer_script is not None
    assert generation is not None
    assert continuation_dependency_kind == "afterany"

    executed: list[str] = []
    for script in (*chunk_scripts, finalizer_script):
        command_line = next(
            line.strip()
            for line in script.read_text(encoding="utf-8").splitlines()
            if "-m phenotypic._cli._cli_migrate_provenance_worker" in line
        )
        parts = shlex.split(command_line)
        config_index = parts.index("--config")
        config = parts[config_index + 1]
        command = parts[config_index + 2]
        indexed = "--index" in parts
        for index in _script_indices(script):
            args = ["--config", config, command]
            if indexed:
                args.extend(["--index", str(index)])
            result = CliRunner().invoke(provenance_migration_worker_cli, args)
            assert result.exit_code == 0, (
                f"{script.name} index {index} failed:\n{result.output}"
            )
            executed.append(command)
    # The topology this file exists for, asserted rather than assumed: the
    # seal is the chain's only singleton, and it runs after the store work.
    assert executed[-2:] == ["seal", "finalize"], executed
    assert set(executed[:-2]) <= {"store"}, executed
    return ["1", "2"], None


def _provenance_snapshot(tree: Path) -> dict[str, object]:
    """Capture converted state, excluding generation-scoped scheduler control.

    Three parts, and each is a place the two paths could disagree:
    the stores (whose journals provenance migration rewrites), the per-image
    records (which the machine-state conversion mints), and the processing
    state's ``config`` (which carries ``processing_generation`` -- the field
    that caught the full-run gap).

    Everything else under ``.phenotypic/`` is generation-named control, and
    the generations differ between local and SLURM by construction.
    """
    cache = phenotypic_cache_dir(tree)
    stores = {
        store.relative_to(tree).as_posix(): _tree_digest(store)
        for store in sorted(tree.rglob("*.ome.zarr"))
        if store.is_dir()
    }

    records: dict[str, object] = {}
    # Derived from the writer's own path helper, never spelled: the directory
    # name is that function's business, and a literal here would agree with a
    # rename by going quietly empty.
    images_root = image_record_path(tree, "_", "_").parent.parent
    if images_root.is_dir():
        for record_path in sorted(images_root.rglob("*.json")):
            payload = json.loads(record_path.read_text(encoding="utf-8"))
            records[record_path.relative_to(images_root).as_posix()] = {
                key: payload[key]
                for key in _STABLE_RECORD_KEYS
                if key in payload
            }

    state_path = resolve_processing_state_path(tree)
    state_config: object = None
    if state_path.is_file():
        state_config = json.loads(state_path.read_text(encoding="utf-8")).get(
            "config"
        )

    assert cache.is_dir(), "migration wrote no machine state at all"
    return {
        "stores": stores,
        "records": records,
        "state_config": state_config,
    }


def _migrate_both_ways(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], dict[str, object]]:
    """Migrate one archive locally and through the synchronous SLURM chain."""
    from phenotypic._cli import _cli_slurm_submission as slurm_submission

    local_tree = tmp_path / "local"
    slurm_tree = tmp_path / "slurm"
    shutil.copytree(source, local_tree)
    shutil.copytree(source, slurm_tree)

    local = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(local_tree)]
    )
    assert local.exit_code == 0, local.output

    monkeypatch.setattr(
        slurm_submission,
        "submit_drip_feed_start",
        _run_generated_provenance_worker_commands,
    )
    slurm = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(slurm_tree),
            "--slurm",
            "slurm_partition=short",
            "--wait",
        ],
    )
    assert slurm.exit_code == 0, slurm.output

    return _provenance_snapshot(local_tree), _provenance_snapshot(slurm_tree)


def test_local_and_slurm_migration_of_a_process_tree_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store-bearing half of the topology.

    **Fires when** the two paths disagree on a converted store, a minted
    record, or the processing state's configuration identity -- the class of
    divergence that a missing conversion produces, and that nothing in this
    topology could detect before this file existed.
    """
    source = _build_process_tree(tmp_path / "archive", ("a", "b"))

    local_snapshot, slurm_snapshot = _migrate_both_ways(
        source, tmp_path, monkeypatch
    )

    assert local_snapshot == slurm_snapshot


def test_a_pre_markers_process_tree_is_migrated_locally_and_refused_on_slurm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The divergence this file found, resolved into a named behaviour.

    **This test previously asserted equivalence and was red.** What it found is
    that the two arms cannot be equivalent for this tree, and the reason is not
    a missing conversion: a pre-markers process tree has **no stores**, so the
    provenance SLURM topology -- store array -> seal -> finalizer -- has no
    work for its array and no vocabulary for what it would do. Its seal
    barriers store statuses and its finalizer reports upgrade counts, so the
    tree would come back "0 upgraded, succeeded" with the machine-state
    conversion, the only real work, invisible in its own terminal report.

    So the claim changed rather than the assertion loosening: the local arm
    does the work, and the SLURM arm **refuses and says why**. A test that
    tolerated the difference by comparing approximately would have hidden
    exactly this.

    **Fires when** the local arm stops converting -- which would return
    MIG-11's minting to being reachable from no execution mode -- or when the
    storeless tree is accepted onto the array topology.
    """
    from phenotypic._cli import _cli_slurm_submission as slurm_submission

    source = _build_pre_markers_process_tree(tmp_path / "archive", ("a", "b"))
    local_tree = tmp_path / "local"
    slurm_tree = tmp_path / "slurm"
    shutil.copytree(source, local_tree)
    shutil.copytree(source, slurm_tree)

    local = CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(local_tree)]
    )
    assert local.exit_code == 0, local.output
    # Exit 0 was already true when the arm did nothing at all, so the records
    # are the assertion and success is not.
    assert _provenance_snapshot(local_tree)["records"], (
        "the local arm minted no records"
    )

    def _refuse_submission(**_kwargs: object) -> tuple[list[str], None]:
        raise AssertionError(
            "a storeless tree reached scheduler submission"
        )

    monkeypatch.setattr(
        slurm_submission, "submit_drip_feed_start", _refuse_submission
    )
    slurm = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "migrate",
            "--output",
            str(slurm_tree),
            "--slurm",
            "slurm_partition=short",
        ],
    )

    assert slurm.exit_code == 2, slurm.output
    assert "nothing to distribute" in slurm.output
    assert "without --slurm" in slurm.output
    # The refusal must come BEFORE anything is written, not after a partial
    # attempt: a refused submission that already converted half a tree is the
    # worst of both arms.
    assert _provenance_snapshot(slurm_tree)["records"] == {}
