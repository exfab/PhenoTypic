"""P4: prove every test in the finalization suite can fail.

``tests/unit/cli/test_finalize_run.py`` is the phase's own suite -- the one
aggregation + join + publish path, INV-INPUTS, INV-PROVEN, the v1/v2
discrimination, and the mixed-authority refusal. It shipped with **no
harness**, and because ``check_mutation_coverage.py`` binds one suite per
harness it did not appear in the coverage report as uncovered: it did not
appear at all.

The P4 implementation-test review found four HIGH items by reading, all of
one shape -- *a guard whose behaviour is proved by a direct unit call, and
whose reachability is proved by nothing*. That shape passes the
establish-then-assert rule while leaving the production behaviour unguarded,
and it is exactly what a harness finds mechanically. Two of the mutations
below (``the authorized arm's mixed-authority check is deleted``, ``the
recompile finalizer's mixed-authority check is deleted``) are the review's
own, and they were green before the tests this harness claims for them
existed.

Run from the worktree root::

    uv run python docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/\
mutation_harnesses/p4_finalize_run.py

**Three gates run before any mutation is applied**, in this order:

0. **Both pytest invocations pass ``-o addopts=``**, and it is load-bearing
   rather than cosmetic -- see ``_PYTEST_FLAGS``. The project's addopts carries
   ``--verbose``, which beats ``-q`` and turns ``--collect-only`` into a tree
   listing with no node ids in it. The first run of this harness aborted on
   exactly that, having mutated nothing.
1. **Name integrity, by pytest ``--collect-only``, never by AST.** A claim may
   name a bare stem or a full parametrized id; both are checked against what
   pytest actually reports. The README's *"two gates have OPPOSITE blind
   spots"* section is why: ``check_mutation_coverage.py`` strips ``[...]`` and
   is blind to a bogus param id by construction, and a harness that also used
   AST would reject every legitimate parametrized claim. This harness claims
   ``test_every_mode_produces_a_byte_identical_master`` by stem, which the
   collector resolves through its three cases.
2. **Coverage.** Every collected test must be claimed by a mutation or
   declared in ``CONTROLS``. Two are declared, and the reason is in the
   ``CONTROLS`` comment.
3. **Anchor ownership.** Every ``old`` must match exactly one of the seven
   targets, exactly once. An ambiguous anchor is refused rather than resolved.

Then the suite must be green, because mutation results against a red suite
are noise.

**Predicted outcomes are written into the table before the run**, per the
README. Where the prediction is a chain of inference rather than a line that
was read, the claim is deliberately left off and the mutation is allowed to
report ``PROVED (broad)`` -- an under-claim costs nothing, an over-claim sends
the next reader to investigate the test, which is the wrong place.

**This suite is unusually integrated**, and several mutations are inherently
broad as a result: it builds real stores through the forward writer, real
records, real state, and drives ``finalize_run`` end to end. A mutation to
the join or to the master frame is *visible to a dozen tests at once*, and
that is a property of the surface, not a weakness in those tests. Every such
mutation says so in its label.

**Known gap, named rather than left implicit** (review finding MEDIUM 6):
``_structural_key_patterns`` calls ``.unique()`` without ``maintain_order``,
so the ragged path's output row order is unspecified. The mutation that
exposes it -- iterating ``reversed(patterns)`` in ``_join_ragged_key_groups``
-- reddens **nothing**, and a mutation that proves nothing must not sit in
this table reading as a proved row. It is recorded here instead. The fix is
``maintain_order=True`` on that ``.unique()``; until then the ragged mirror's
row order has no test.

Safety: every target is copied to a temp directory before anything is
touched, restored in a ``finally``, and its sha256 compared at the end. The
backup keeps the full relative path, never the basename -- a prior harness
clobbered 774 lines of the wrong file that way while reporting a clean
restore. The backup lives outside the repo, so an interrupted run leaves no
stray file in the working tree.

**Do not edit a target while this runs.** It holds the pristine sources in
memory for the whole run and writes them back after every mutation, so an
edit arriving mid-run is silently reverted at the end. Announce start and
finish if anyone else is working in the tree, and remember that a suite run
by someone else mid-mutation is not a suite result.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ENV = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}

#: **``-o addopts=`` is load-bearing for any harness in this repo that parses
#: pytest output**, and it is not a tuning knob. ``pyproject.toml:223`` sets
#: ``addopts = "--verbose --capture=no -m 'not slow'"``, and ``--verbose``
#: **wins over ``-q``** -- so ``--collect-only`` emits the indented ``<Function
#: name>`` tree instead of ``path::nodeid`` lines. This harness's first run
#: aborted on exactly that: it parsed the output correctly and matched zero
#: lines, because the format it parses is decided by configuration it does not
#: control.
#:
#: That is this change's own recurring defect in miniature -- a check that is
#: right about what it read and silent about having read nothing -- which is
#: why ``_collected`` treats an empty id set as an ABORT rather than as "no
#: unknown claims, no uncovered tests". A *partial* set would have been the
#: dangerous one: coverage would have passed with tests missing from the
#: claim check.
#:
#: ``-m "not slow"`` is re-added explicitly rather than left to the neutralised
#: addopts, so the harness selects exactly what the project's default
#: selection does. It deselects nothing in this suite today; it will if a test
#: here is ever marked slow, and a coverage gate that disagreed with the run
#: about which tests exist is the failure this whole directory is about.
#:
#: ``-p no:randomly`` is determinism, not speed. Twenty-three runs are compared
#: against one another, so an ordering that varies between them could move a
#: row between PROVED and PROVED (broad) with no mutation involved.
#:
#: The same three flags, for the same reasons, are in
#: ``docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch``.
_PYTEST_FLAGS = ("-o", "addopts=", "-m", "not slow", "-p", "no:randomly")

TARGETS = (
    "src/phenotypic/_cli/_cli_finalize_run.py",
    "src/phenotypic/_cli/_cli_output_manager.py",
    "src/phenotypic/_cli/_cli_completion.py",
    "src/phenotypic/_cli/_cli_recompile_worker.py",
    "src/phenotypic/_cli/_embedded_measurement_tables.py",
    "src/phenotypic/_cli/_metadata_join.py",
    "src/phenotypic/sdk_/_master_io.py",
)
SUITE = "tests/unit/cli/test_finalize_run.py"

#: Proved by the ABSENCE of a mutation making them fire, so no mutation will
#: ever claim them -- see the harness README's "Controls: declared, never
#: inferred".
#:
#: ``test_master_measurements_csv_is_gone`` fails when the implementation
#: becomes too EAGER: D8 deleted ``master_measurements.csv``, and the test
#: fires the moment anything starts writing one again. A mutation that
#: re-adds the write would be written to satisfy this script rather than to
#: catch a bug.
#:
#: ``test_stores_with_mixed_metadata_snapshots_do_not_abort_finalization`` is
#: the same shape from the other direction: it fails when finalization becomes
#: too STRICT and re-raises on the mixed snapshot generations D-A deliberately
#: manufactures. Its proof is that no mutation here makes it abort.
CONTROLS = (
    "test_master_measurements_csv_is_gone",
    "test_stores_with_mixed_metadata_snapshots_do_not_abort_finalization",
)

# (label, old, new, tests that MUST fail)
MUTATIONS: list[tuple[str, str, str, tuple[str, ...]]] = [
    # -- HIGH 2: the mixed-authority guard, at each of its two call sites ---
    (
        "the authorized arm's mixed-authority check is deleted"
        " [the review's own mutation for HIGH 2 site 1; it was GREEN before"
        " the test claimed below existed]",
        "        refuse_mixed_measurement_authority(list(authorized_sources))\n"
        "        return authorized_sources, True",
        "        return authorized_sources, True",
        ("test_the_authorized_arm_refuses_a_mixed_authority_tree",),
    ),
    (
        "the recompile finalizer's mixed-authority check is deleted"
        " [the review's own mutation for HIGH 2 site 2]",
        "    measurement_sources = task.get(\"measurement_sources\")\n"
        "    if measurement_sources is not None:\n"
        "        refuse_mixed_measurement_authority(\n"
        "            [Path(str(path)) for path in measurement_sources]\n"
        "        )\n",
        "",
        ("test_the_recompile_finalizer_refuses_a_mixed_authority_task",),
    ),
    (
        "the mixture is never a mixture -- the refusal itself is neutered."
        " Claims all three tests of the guard: the direct call, and BOTH"
        " production call sites, which is the point of the pair above",
        "    if embedded and len(embedded) != len(paths):",
        "    if False:",
        (
            "test_mixed_embedded_and_legacy_authority_is_still_refused",
            "test_the_authorized_arm_refuses_a_mixed_authority_tree",
            "test_the_recompile_finalizer_refuses_a_mixed_authority_task",
        ),
    ),
    # -- HIGH 1: the v1/v2 discrimination and its falsifier ----------------
    (
        "master_carries_user_metadata is INVERTED"
        " [the review's mutation 1 for HIGH 1. Before the falsifier was"
        " rebuilt on two PRODUCERS this reddened the two direct tests and"
        " left the falsifier green, which was the finding]",
        "    return bool(user_metadata_headers(frame.columns))",
        "    return not bool(user_metadata_headers(frame.columns))",
        (
            "test_the_master_carries_no_user_metadata",
            "test_master_carries_user_metadata_reads_ownership_not_the_prefix",
            "test_a_v1_metadata_free_master_is_indistinguishable_from_v2"
            "_and_that_is_harmless",
        ),
    ),
    (
        "user_metadata_headers reads the PREFIX instead of ownership -- the"
        " rule the plan stated and the schema cannot support",
        "        and metadata_owner_for_header(column) is not IMAGE\n"
        "        and column not in intrinsic",
        "",
        (
            "test_master_carries_user_metadata_reads_ownership_not_the_prefix",
            "test_the_master_carries_no_user_metadata",
        ),
    ),
    (
        "the PRE-INVERSION producer diverges from prepare_image_tables on a"
        " metadata-free run -- it emits a user-metadata column where the"
        " forward writer emits none. This is what makes the v1/v2 falsifier"
        " falsifiable: the v1 arm is built through this producer, so if the"
        " two ever disagree the masters differ and the no-stamp ruling has to"
        " be revisited",
        "    if inputs.prepared is None:\n"
        "        return PreparedEmbeddedMeasurementTable(\n"
        "            frame=inputs.baseline,",
        "    if inputs.prepared is None:\n"
        "        return PreparedEmbeddedMeasurementTable(\n"
        "            frame=inputs.baseline.assign(Metadata_Strain=None),",
        (
            "test_a_v1_metadata_free_master_is_indistinguishable_from_v2"
            "_and_that_is_harmless",
            "test_every_mode_produces_a_byte_identical_master",
        ),
    ),
    # -- INV-INPUTS (§7.5) -------------------------------------------------
    (
        "step 1 prefers _dataset_aggregated.parquet ON THE AUTHORIZED ARM"
        " TOO -- review finding MEDIUM 7, the phase's ONE prescribed"
        " INV-INPUTS mutation, which the four commit bodies never record as"
        " having been run",
        "        # Schema-3 terminal publication never trusts checkpoint "
        "aggregates or\n"
        "        # an unmarked per-image Parquet merely because it exists.\n"
        "        refuse_mixed_measurement_authority(list(authorized_sources))",
        "        from phenotypic.sdk_ import (\n"
        "            DATASET_AGGREGATED_PARQUET,\n"
        "            DIR_MEASUREMENTS,\n"
        "            DIR_RESULTS,\n"
        "        )\n\n"
        "        _preferred = {\n"
        "            path: str(name)\n"
        "            for name in dataset_names\n"
        "            for path in [\n"
        "                Path(output_dir)\n"
        "                / DIR_RESULTS\n"
        "                / str(name)\n"
        "                / DIR_MEASUREMENTS\n"
        "                / DATASET_AGGREGATED_PARQUET\n"
        "            ]\n"
        "            if path.is_file()\n"
        "        }\n"
        "        if _preferred:\n"
        "            return _preferred, True\n"
        "        refuse_mixed_measurement_authority(list(authorized_sources))",
        ("test_finalize_run_ignores_every_stale_intermediate",),
    ),
    (
        "the LEGACY arm stops preferring its dataset aggregate -- the"
        " \"drop the arm\" option the user ruling rejected",
        "    return (\n"
        "        measurement_sources_by_path(\n"
        "            discover_measurement_sources(output_dir, dataset_names)\n"
        "        ),\n"
        "        False,\n"
        "    )",
        "    from phenotypic.sdk_ import DATASET_AGGREGATED_PARQUET\n\n"
        "    return (\n"
        "        {\n"
        "            path: dataset\n"
        "            for path, dataset in measurement_sources_by_path(\n"
        "                discover_measurement_sources(\n"
        "                    output_dir, dataset_names\n"
        "                )\n"
        "            ).items()\n"
        "            if Path(path).name != DATASET_AGGREGATED_PARQUET\n"
        "        },\n"
        "        False,\n"
        "    )",
        ("test_the_legacy_arm_still_prefers_its_dataset_aggregate",),
    ),
    (
        "the previous finalization's intermediates are never invalidated,"
        " so the next invocation can mistake them for inputs",
        "        _invalidate_finalization_intermediates(output_dir, dataset_names)",
        "        pass",
        ("test_finalize_run_invalidates_the_intermediates_on_success",),
    ),
    # -- INV-PROVEN --------------------------------------------------------
    (
        "finalize_run rewrites every promoted store's root with"
        " BYTE-IDENTICAL content -- the backfill D-A cut, in its most benign"
        " form. It also SIZES review finding LOW 9: if the mtime snapshot"
        " does not catch a same-content rewrite, INV-PROVEN's only gate needs"
        " a content digest instead",
        "    logger.info(\n"
        "        \"Aggregated %d rows x %d cols into %s\",",
        "    for _store in sorted(\n"
        "        Path(output_dir).glob(\"results/*/zarr/*.ome.zarr\")\n"
        "    ):\n"
        "        _root = _store / \"zarr.json\"\n"
        "        if _root.is_file():\n"
        "            _root.write_bytes(_root.read_bytes())\n"
        "    logger.info(\n"
        "        \"Aggregated %d rows x %d cols into %s\",",
        ("test_finalize_run_writes_no_byte_into_a_proven_store",),
    ),
    # -- the master's shape and identity ------------------------------------
    (
        "the master loses Object_Label -- intrinsic identity leaves the"
        " archival set [inherently broad: the object label is the key three"
        " other tests filter on, so they go red too, which is a property of"
        " deleting an identity column rather than a weakness in them]",
        "    return add_metadata_image_name_from_filename(master_df), authorized",
        "    return (\n"
        "        add_metadata_image_name_from_filename(master_df).drop(\n"
        "            \"Object_Label\"\n"
        "        ),\n"
        "        authorized,\n"
        "    )",
        ("test_curation_re_keying_still_works_against_the_intrinsic_master",),
    ),
    (
        "aggregation REORDERS the master's columns, so it is no longer the"
        " exact concatenation of its inputs",
        "    if master_df is None:\n        return None, authorized",
        "    if master_df is None:\n"
        "        return None, authorized\n"
        "    master_df = master_df.select(sorted(master_df.columns))",
        (
            "test_the_master_inherits_its_column_order_from_the_embedded_tables",
        ),
    ),
    (
        "the MASTER is rewritten from the joined mirror -- master and mirror"
        " collapse into one file [inherently broad: the master then carries"
        " user metadata and loses the metadata-unmatched object, which is"
        " several tests at once, and that is the point of keeping them"
        " separate]",
        "    finalize_post_master_outputs(\n"
        "        output_dir,\n"
        "        master_df,\n"
        "        resolved_pipeline,\n"
        "        metadata_csv=metadata_csv,\n"
        "        no_qc=no_qc,\n"
        "        study_config=study_config,\n"
        "        commit_guard=commit_guard,\n"
        "    )",
        "    _mirror_df = finalize_post_master_outputs(\n"
        "        output_dir,\n"
        "        master_df,\n"
        "        resolved_pipeline,\n"
        "        metadata_csv=metadata_csv,\n"
        "        no_qc=no_qc,\n"
        "        study_config=study_config,\n"
        "        commit_guard=commit_guard,\n"
        "    )\n"
        "    atomic_write_with_writer(\n"
        "        master_path,\n"
        "        lambda p: _mirror_df.write_parquet(p, **PARQUET_WRITE_OPTIONS),\n"
        "    )",
        ("test_the_master_keeps_the_object_the_mirror_drops",),
    ),
    # -- the mirror's join --------------------------------------------------
    (
        "the finalizer joins NOTHING -- the retired discriminator's defect,"
        " reintroduced at its source: the common-column set is empty, so"
        " join_metadata returns the frame unjoined and warns"
        " [inherently broad: every test that reads a metadata column out of"
        " the mirror goes red, which is what the defect did in production]",
        "    common = list(prepared.analysis.columns)",
        "    common = []",
        (
            "test_an_authorized_metadata_run_does_not_lose_the_join",
            "test_metadata_added_after_the_stores_still_joins_every_measured_row",
            "test_the_mirror_carries_both_joined_rows_and_phantoms",
        ),
    ),
    (
        "the mirror's join drops metadata-only phantoms (how=left -> inner)"
        " [inherently broad: QC_MetadataOnly stops being emitted at all, so"
        " every test that partitions the mirror on it errors]",
        "            working_df = join_metadata(working_df, metadata_csv, how=\"left\")",
        "            working_df = join_metadata(\n"
        "                working_df, metadata_csv, how=\"inner\"\n"
        "            )",
        (
            "test_the_mirror_carries_both_joined_rows_and_phantoms",
            "test_an_authorized_metadata_run_does_not_lose_the_join",
        ),
    ),
    (
        "the join keeps measurement-unmatched rows too -- the asymmetry the"
        " user ruling pinned is reversed into a full join",
        "        out = metadata_df.join(df, on=common, how=how, maintain_order=\"left\")",
        "        out = metadata_df.join(\n"
        "            df, on=common, how=\"full\", maintain_order=\"left\"\n"
        "        )",
        ("test_a_measured_row_absent_from_metadata_is_dropped_deliberately",),
    ),
    (
        "the mirror is not reordered after the join, so it keeps"
        " join_metadata's metadata-first shape",
        "    post_df = post_df.select(order_measurement_columns(post_df.columns))",
        "    post_df = post_df",
        ("test_the_mirror_keeps_canonical_column_order_after_the_join",),
    ),
    (
        "join keys are coerced to the MEASUREMENT dtype instead of to String"
        " -- the alternative design, which changes the mirror's key dtype"
        " under every reader keyed on the old one",
        "    normalized_measurements = normalized_measurements.with_columns(\n"
        "        pl.col(column).cast(pl.String) for column in common\n"
        "    )\n"
        "    normalized_metadata = normalized_metadata.with_columns(\n"
        "        pl.col(column).cast(pl.String) for column in common\n"
        "    )",
        "    normalized_metadata = normalized_metadata.with_columns(\n"
        "        pl.col(column).cast(normalized_measurements.schema[column])\n"
        "        for column in common\n"
        "    )",
        ("test_the_mirrors_join_key_dtype_is_pinned",),
    ),
    # -- the ragged join ----------------------------------------------------
    (
        "a ragged frame takes the single join again -- every row of every"
        " image that lacked a key column is silently dropped from the mirror",
        "    ragged = len(patterns) > 1",
        "    ragged = False",
        ("test_a_heterogeneous_master_loses_no_measured_row",),
    ),
    (
        "the empty-frame guard in _structural_key_patterns is dropped, so an"
        " empty frame reports ZERO patterns and reads as ragged",
        "    if df.height == 0:\n"
        "        return [tuple(False for _ in common)]\n",
        "",
        ("test_the_ragged_path_is_reached_only_by_a_ragged_frame",),
    ),
    # -- the aggregate <-> run proof binding (U-4) --------------------------
    (
        "the run proof RECOMPUTES its source-set digest instead of copying"
        " the aggregate's -- so a stale aggregate proof and a fresh run proof"
        " each pass independently while disagreeing with each other",
        "        \"source_set_digest\": (\n"
        "            aggregate.get(\"source_set_digest\")\n"
        "            if aggregate is not None\n"
        "            else None\n"
        "        ),",
        "        \"source_set_digest\": \"recomputed-rather-than-copied\",",
        (
            "test_the_run_proof_copies_the_aggregates_source_set_digest",
            "test_the_run_proof_binding_is_checked_end_to_end",
        ),
    ),
    (
        "valid_run_completion stops comparing the run proof's source-set"
        " digest against the aggregate's -- the binding becomes a field"
        " nobody reads",
        "        expected[\"source_set_digest\"] = aggregate.get(\"source_set_digest\")\n",
        "",
        ("test_the_run_proof_binding_is_checked_end_to_end",),
    ),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collected() -> tuple[set[str], set[str]]:
    """Return ``(full pytest ids, their stems)`` for the suite.

    **Collected by pytest, never by AST**, and the README says why: AST
    ``FunctionDef`` names carry no parametrize ids, so an AST precondition
    rejects every legitimate parametrized claim -- which is what aborted the
    first harness to make one, before a single mutation ran. This is the only
    gate that can see a bogus ``pytest.param``; ``check_mutation_coverage.py``
    is blind to it by construction, and that division of labour is deliberate.
    """
    proc = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            SUITE,
            "--collect-only",
            "-q",
            "--no-header",
            *_PYTEST_FLAGS,
        ],
        capture_output=True,
        text=True,
        env={**_ENV},
    )
    ids = {
        line.split("::", 1)[1].strip()
        for line in proc.stdout.splitlines()
        if "::" in line and line.strip().startswith("tests/")
    }
    if not ids:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:])
    return ids, {name.split("[")[0] for name in ids}


def _failed_tests() -> set[str]:
    """Return the failed tests of one suite run, as ids AND as stems.

    Both, so an expectation may be written either way: a mutation that takes
    down every case of a parametrization is honestly claimed by the stem,
    while one that takes down a single case needs the id.
    """
    proc = subprocess.run(
        ["uv", "run", "pytest", SUITE, "-q", "--no-header", "-rf", *_PYTEST_FLAGS],
        capture_output=True,
        text=True,
        env={**_ENV},
    )
    failed: set[str] = set()
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAILED ") or stripped.startswith("ERROR "):
            # "FAILED tests/...::test_name[case] - AssertionError: ..."
            name = stripped.split("::", 1)[-1].split(" ", 1)[0]
            failed.add(name)
            failed.add(name.split("[")[0])
    return failed


def _owner(sources: dict[Path, str], old: str) -> Path | None:
    """Return the one target containing ``old`` exactly once, else ``None``.

    Ambiguity is refused rather than resolved: an anchor matching two targets
    would silently mutate whichever the dict happened to yield first.
    """
    owners = [path for path, text in sources.items() if text.count(old) == 1]
    return owners[0] if len(owners) == 1 else None


def main() -> int:
    targets = [Path(name).resolve() for name in TARGETS]
    missing = [t for t in targets if not t.is_file()]
    if missing:
        print(f"ABORT: run me from the worktree root -- {missing} not found")
        return 4
    if not Path(SUITE).is_file():
        print(f"ABORT: suite {SUITE} not found; run me from the worktree root")
        return 4

    backup_dir = Path(tempfile.mkdtemp(prefix="phenotypic-mutation-"))
    print(f"backup: {backup_dir}")
    sources: dict[Path, str] = {}
    originals: dict[Path, str] = {}
    for name, target in zip(TARGETS, targets):
        # Full relative path, never the basename: a prior harness clobbered
        # 774 lines of the wrong file that way while reporting a clean
        # restore.
        backup = backup_dir / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        sources[target] = target.read_text(encoding="utf-8")
        originals[target] = _sha256(target)

    rows: list[tuple[str, str, str]] = []
    try:
        ids, stems = _collected()
        if not ids:
            print("ABORT: pytest collected nothing from the suite")
            return 4
        known = ids | stems
        named = {name for _l, _o, _n, exp in MUTATIONS for name in exp}
        unknown = (named | set(CONTROLS)) - known
        if unknown:
            print(
                "ABORT: MUTATIONS/CONTROLS name tests pytest does not "
                f"collect: {sorted(unknown)}"
            )
            return 3
        claimed_stems = {name.split("[")[0] for name in named}
        unclaimed = sorted(stems - claimed_stems - set(CONTROLS))
        # BOTH UNITS, each labelled. This harness collects with pytest, so it
        # sees CASES; `check_mutation_coverage.py` reads AST `FunctionDef`
        # names, so it sees FUNCTIONS, and one parametrized test makes the two
        # disagree for good reasons. An unlabelled number here would be read
        # against the checker's output and look like drift. Coverage,
        # claims and controls are all counted in FUNCTIONS -- the checker's
        # unit -- so the two reports can be compared line for line.
        print(
            f"pytest collects   : {len(ids)} case(s)  "
            f"[cases, not functions -- one parametrized test is N of these]"
        )
        print(f"suite functions   : {len(stems)} test function(s)")
        print(f"claimed by a mut  : {len(claimed_stems)} function(s)")
        print(f"declared controls : {sorted(CONTROLS)}")
        if unclaimed:
            print(f"ABORT: NOT COVERED by any mutation: {unclaimed}")
            return 3

        unowned = [
            label
            for label, old, _new, _exp in MUTATIONS
            if _owner(sources, old) is None
        ]
        if unowned:
            print(
                "ABORT: these anchors match no target exactly once: "
                f"{[label[:70] for label in unowned]}"
            )
            return 3

        baseline = _failed_tests()
        if baseline:
            print(f"ABORT: suite is not green to begin with: {sorted(baseline)}")
            return 2
        print("baseline: suite green\n")

        for label, old, new, expected in MUTATIONS:
            target = _owner(sources, old)
            assert target is not None  # pre-validated above
            source = sources[target]
            target.write_text(source.replace(old, new, 1), encoding="utf-8")
            failed = _failed_tests()
            target.write_text(source, encoding="utf-8")

            missing_tests = set(expected) - failed
            extra = {
                name
                for name in failed
                if "[" not in name
                and name not in {e.split("[")[0] for e in expected}
            }
            if missing_tests:
                verdict = "NOT PROVED"
                detail = f"did not fail: {sorted(missing_tests)}"
            elif extra:
                verdict, detail = (
                    "PROVED (broad)",
                    f"also failed: {sorted(extra)}",
                )
            else:
                verdict, detail = "PROVED", f"exactly {sorted(expected)}"
            rows.append((label, verdict, detail))
            print(
                f"{verdict:<14} [{target.name}] {label[:70]}\n"
                f"               {detail}"
            )
    finally:
        for name, target in zip(TARGETS, targets):
            shutil.copy2(backup_dir / name, target)
            restored = _sha256(target)
            status = "OK" if restored == originals[target] else "MISMATCH"
            print(f"restored {name}: {status} ({restored[:12]})")

    print("\n--- summary ---")
    for label, verdict, detail in rows:
        print(f"{verdict:<14} | {label[:70]} | {detail}")
    unproved = [r for r in rows if r[1] not in {"PROVED", "PROVED (broad)"}]
    print(f"\nMUTATIONS_ALL_PROVED={not unproved}")
    return 0 if not unproved else 1


if __name__ == "__main__":
    sys.exit(main())
