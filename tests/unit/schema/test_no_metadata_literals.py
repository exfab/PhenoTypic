"""Grep gate: no stray ``"Metadata_<X>"`` column literals in source (Task B8).

After the metadata-namespace flip, schema-backed columns share the canonical
``Metadata_<Label>`` namespace and should be referenced through their owning
enum. The only legal specific ``"Metadata_"`` column literals left in
``src/phenotypic`` are a handful of deliberate shims and two arbitrary doctest
examples. This gate fails the moment a new ``"Metadata_<X>"`` literal is
introduced anywhere else, indicating a missed conversion or a re-stringly-typed
column.

The allowlist is **per-literal**, not whole-file: each allowed site may carry
*only* its one declared literal, so adding a *different* stray ``"Metadata_<X>"``
to an already-allowed shim file is still caught.
"""
import pathlib
import re
from typing import Final

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "phenotypic"
REPO = SRC.parents[1]

# A quote immediately followed by ``Metadata_<letter>`` -- i.e. a specific
# metadata column literal (not the generic ``"Metadata_"`` prefix constant,
# which has no trailing letter and so never matches).
_PAT = re.compile(r"""["']Metadata_[A-Za-z]""")

# Same, but capturing the whole quoted token so an allowed site can be pinned
# to its exact literal(s).
_LITERAL_PAT = re.compile(r"""["'](Metadata_[A-Za-z][A-Za-z0-9_]*)["']""")

# The only source sites allowed to carry such a literal, keyed by repo-relative
# posix path -> the exact set of literals permitted there (with the reason).
# Any other ``Metadata_<X>`` match -- in these files or elsewhere -- is a missed
# conversion.
_ALLOWED = {
    # generic-fallback + legacy shims (kept literals, load old data)
    "gui/results_viewer/_curation_labels.py": {"Metadata_ImageFile"},  # _LEGACY_IMAGE_FILE curation shim
    "gui/results_viewer/_compatibility.py": {"Metadata_ImageName"},  # explicit output migration alias
    "gui/shell/_metadata_context.py": {  # metadata CSV identity aliases
        "Metadata_ImageFileName",
    },
    "_cli/_cli_recompile_worker.py": {"Metadata_Well"},             # no WELL schema member
    # user metadata column from the --metadata CSV; no MetadataInfo member
    "gui/results_viewer/_scatter_tab/_facets.py": {"Metadata_ImageDatetime"},
    # arbitrary-column doctest examples (demonstrate non-vocabulary columns)
    "abc_/_post_measurement.py": {"Metadata_Flag"},                # AddConstant(column="Metadata_Flag")
    "post/_merge_metadata.py": {"Metadata_Condition"},             # doctest "Metadata_Condition"
}


def test_no_specific_metadata_literals_outside_allowed():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        allowed = _ALLOWED.get(rel)
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not _PAT.search(line):
                continue
            if allowed is None:
                offenders.append(f"{rel}:{i}: {line.strip()}")
                continue
            # Allowed file: only its declared literal(s) may appear. A NEW
            # *different* "Metadata_<X>" on any line here is still an offender.
            unexpected = set(_LITERAL_PAT.findall(line)) - allowed
            if unexpected:
                offenders.append(
                    f"{rel}:{i}: {line.strip()} "
                    f"(unexpected {sorted(unexpected)}; allowed {sorted(allowed)})"
                )
    assert not offenders, (
        "stringly-typed metadata literals remain (use str(ENUM.MEMBER) or "
        "is_metadata_header instead):\n" + "\n".join(offenders)
    )


def test_allowlist_entries_are_not_stale():
    """Every allowlisted site still exists and still carries its declared literal.

    Guards against a rotting allowlist: if a shim is removed or its literal
    renamed, the entry should be dropped rather than silently over-permitting a
    file that no longer needs an exemption.
    """
    stale = []
    for rel, expected in _ALLOWED.items():
        path = SRC / rel
        if not path.is_file():
            stale.append(f"{rel}: file no longer exists")
            continue
        found = set(_LITERAL_PAT.findall(path.read_text(encoding="utf-8")))
        missing = expected - found
        if missing:
            stale.append(f"{rel}: declared but absent {sorted(missing)}")
    assert not stale, (
        "prune stale metadata-literal allowlist entries:\n" + "\n".join(stale)
    )


_LEGACY_NAME_PAT = re.compile(
    r"\b(?:METADATA|GENETIC_METADATA|SAMPLE_METADATA|PLATE_METADATA|"
    r"CONDITION_METADATA|CULTURE_METADATA|EXPERIMENT_METADATA|"
    r"STUDY_METADATA|ACQUISITION_METADATA)\b"
)
_LEGACY_HEADER_PAT = re.compile(
    r"\bMetadata(?:Image|Genetic|Sample|Plate|Condition|Culture|Experiment|"
    r"Study|Acquisition)_[A-Za-z][A-Za-z0-9_]*\b"
)

# Shipping compatibility seams, explicit migration documentation, and tests
# whose subject is compatibility may name the previous public surface. Each
# exception pins the exact permitted tokens so an unrelated stale alias in the
# same file still fails the gate. Frozen superpowers specs/plans remain outside
# this live-code/live-doc/test gate.
_LEGACY_ALLOWED = {
    # OME-Zarr store: `ngff_.py` DEFINES the store-attribute enum member as a
    # bare `METADATA: Final[str] = "metadata"`. That spelling is not covered by
    # `_NOT_THE_BANNED_TOKEN` above, and should not be -- a bare METADATA is
    # exactly what this gate bans, so the definition site is exempted here
    # explicitly rather than by widening the strip. Every USE of it
    # (`PhenotypicAttr.METADATA`) and the NGFF filename (`METADATA.ome.xml`) are
    # handled by context, so no other OME-Zarr file needs an entry.
    "src/phenotypic/sdk_/ngff_.py": {"METADATA"},
    # Permanent read compatibility and one-release import shims.
    "src/phenotypic/_core/_image_parts/_image_io_handler.py": {
        "METADATA",
        "GENETIC_METADATA",
        "SAMPLE_METADATA",
        "PLATE_METADATA",
        "CONDITION_METADATA",
        "CULTURE_METADATA",
        "EXPERIMENT_METADATA",
        "STUDY_METADATA",
        "ACQUISITION_METADATA",
    },
    "src/phenotypic/schema/__init__.py": {
        "METADATA",
        "GENETIC_METADATA",
        "SAMPLE_METADATA",
        "PLATE_METADATA",
        "CONDITION_METADATA",
        "CULTURE_METADATA",
        "EXPERIMENT_METADATA",
        "STUDY_METADATA",
        "ACQUISITION_METADATA",
    },
    "src/phenotypic/schema/_experimental_tags/__init__.py": {
        "GENETIC_METADATA",
        "SAMPLE_METADATA",
        "PLATE_METADATA",
        "CONDITION_METADATA",
        "CULTURE_METADATA",
        "EXPERIMENT_METADATA",
        "STUDY_METADATA",
        "ACQUISITION_METADATA",
    },
    "src/phenotypic/schema/_tiers.py": {"MetadataGenetic_Strain"},
    # Live migration guidance labels historical strings as compatibility data.
    "src/phenotypic/schema/CLAUDE.md": {"MetadataGenetic_Strain"},
    "docs/source/explanation/metadata_namespace.md": {
        "MetadataImage_ImageName",
        "MetadataGenetic_Strain",
        "MetadataCulture_Time",
    },
    "docs/source/how_to/pages/rembi_metadata.md": {"MetadataImage_ImageName"},
    # The capture harness repairs a reusable historical fixture before capture;
    # this changes fixture data only and does not change GUI chrome/screenshots.
    "scripts/capture_gui_tutorial_screenshots.py": {"MetadataImage_ImageName"},
    "tests/unit/gui/test_capture_tutorial_script.py": {
        "MetadataImage_ImageName",
    },
    # Durable migration and external-input immutability coverage.
    "tests/migration/test_metadata_schema_migration.py": {
        "MetadataImage_ImageName",
        "MetadataGenetic_Strain",
    },
    "tests/unit/cli/test_cli_output_manager.py": {
        "MetadataImage_ImageName",
        "MetadataGenetic_Strain",
    },
    # The legacy spelling is the INPUT of a migration assertion in each of
    # these: the test writes it, then asserts the canonical name replaced it.
    # Removing the literal would remove the thing being tested.
    #
    #   test_metadata_migration_journal      LEGACY_STRAIN, paired with
    #                                        CANONICAL_STRAIN = "Metadata_Strain"
    #   test_cli_migrate_image               writes a legacy parquet, then asserts
    #                                        "MetadataGenetic_Strain" not in
    #                                        embedded.columns after the migration
    #   test_windows_metadata_journal        writes a legacy-named aggregate as the
    #                                        precondition preflight_metadata_schema
    #                                        must detect
    "tests/migration/test_metadata_migration_journal.py": {
        "MetadataGenetic_Strain",
    },
    "tests/unit/cli/test_cli_migrate_image.py": {
        "MetadataGenetic_Strain",
    },
    "tests/unit/sdk_/test_windows_metadata_journal.py": {
        "MetadataGenetic_Strain",
    },
    "tests/unit/cli/test_cli_recompile.py": {
        "MetadataImage_ImageName",
        "MetadataSample_Strain",
    },
    # ``--mode migrate`` (Phase 5). Legacy per-topic spellings are the SUBJECT
    # of these files, not incidental usage: the golden fixtures are written
    # with them, and the tests assert they are gone after conversion. A
    # migration suite that could only spell canonical names could not test
    # migration.
    "tests/fixtures/legacy_hdf/_generate.py": {
        "MetadataImage_ImageName",
        "MetadataImage_BitDepth",
        "MetadataGenetic_Strain",
    },
    "tests/unit/sdk_/_migration_fixtures.py": {
        "MetadataImage_ImageFile",
        "MetadataGenetic_Strain",
    },
    "tests/unit/sdk_/test_metadata_canonical_view.py": {
        "MetadataGenetic_Strain",
    },
    "tests/unit/cli/test_cli_migrate_mode.py": {
        "MetadataGenetic_Strain",
    },
    "tests/unit/cli/test_recompile_no_longer_migrates.py": {
        "MetadataGenetic_Strain",
    },
    # Exact legacy/canonical collision and ingress behavior.
    "tests/unit/analysis/abc_/test_quality_check.py": {
        "MetadataGenetic_Strain",
    },
    "tests/unit/core/test_metadata_by_module.py": {
        "MetadataImage_ImageName",
        "MetadataImage_UUID",
    },
    "tests/unit/gui/results_viewer/test_metadata_namespace_compat.py": {
        "MetadataImage_ImageName",
        "MetadataCulture_Time",
    },
    "tests/unit/gui/results_viewer/test_curation_labels.py": {
        "MetadataImage_ImageName",
    },
    "tests/unit/gui/results_viewer/test_metadata_prefix_predicates.py": {
        "MetadataGenetic_Strain",
    },
    "tests/unit/gui/results_viewer/test_qc_db_api.py": {
        "MetadataImage_ImageName",
    },
    "tests/unit/gui/shell/test_metadata_context.py": {
        "MetadataImage_ImageName",
        "MetadataGenetic_Strain",
    },
    "tests/unit/post/test_flat_metadata_ingress.py": {
        "MetadataGenetic_Strain",
        "MetadataSample_SampleID",
    },
    "tests/unit/sdk_/test_metadata_helpers.py": {
        "MetadataGenetic_Strain",
        "MetadataGenetic_NotARealTag",
    },
    # Temporary import-warning and historical pickle/HDF compatibility fixtures.
    "tests/unit/schema/test_metadata_label_uniqueness.py": {
        "METADATA",
        "GENETIC_METADATA",
        "SAMPLE_METADATA",
        "PLATE_METADATA",
        "CONDITION_METADATA",
        "CULTURE_METADATA",
        "EXPERIMENT_METADATA",
        "STUDY_METADATA",
        "ACQUISITION_METADATA",
    },
    "tests/unit/sdk_/test_metadata_io.py": {
        "METADATA",
        "GENETIC_METADATA",
        "SAMPLE_METADATA",
        "PLATE_METADATA",
        "CONDITION_METADATA",
        "CULTURE_METADATA",
        "EXPERIMENT_METADATA",
        "STUDY_METADATA",
        "ACQUISITION_METADATA",
        "MetadataImage_ImageName",
        "MetadataGenetic_Strain",
    },
}


#: Spellings that contain a banned token but are not the banned thing.
#:
#: The gate bans the bare token ``METADATA`` because it names a **deprecated
#: metadata-topic enum**. Two unrelated identifiers introduced by the OME-Zarr
#: store collide with that spelling and are not it:
#:
#: * ``METADATA.ome.xml`` -- the filename NGFF 0.5 §2.2.3 mandates for the
#:   OME-XML document. Not ours to rename.
#: * ``PhenotypicAttr.METADATA`` -- a member of the store-attribute enum,
#:   naming the ``attributes.phenotypic.metadata`` block.
#:
#: These are stripped by CONTEXT rather than allowlisted by file. An allowlist
#: entry would have to be added for every file in every remaining phase that
#: touches the store attributes -- three were needed for Phase 1 and three more
#: for Phase 2 -- and each entry blankets the whole file, so a genuine legacy
#: alias appearing alongside would be waved through. Stripping the exact
#: spelling keeps the gate's teeth: a bare ``METADATA``, or any other qualified
#: use of it, still fails everywhere.
_NOT_THE_BANNED_TOKEN: Final[tuple[str, ...]] = (
    "METADATA.ome.xml",
    "PhenotypicAttr.METADATA",
)


def _legacy_tokens(text: str) -> set[str]:
    """Return exact deprecated enum identifiers and serialized headers."""
    for spelling in _NOT_THE_BANNED_TOKEN:
        text = text.replace(spelling, "")
    return set(_LEGACY_NAME_PAT.findall(text)) | set(_LEGACY_HEADER_PAT.findall(text))


def _live_metadata_files():
    """Yield shipping source, live docs, and normal tests checked by the gate."""
    extensions = {".py", ".md", ".rst", ".csv", ".json", ".yaml", ".yml"}
    for root in (
        REPO / "src" / "phenotypic",
        REPO / "docs" / "source",
        REPO / "scripts",
        REPO / "tests",
    ):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in extensions:
                continue
            if (
                "api_reference/api" in path.relative_to(REPO).as_posix()
                or "measurements_ref" in path.parts
            ):
                continue
            # GUI ledgers are chrome snapshots. This namespace change adds no
            # chrome, so the project workflow explicitly leaves them frozen.
            if path.name in {"FEATURES.md", "WORKFLOWS.md"}:
                continue
            # The gate's own regex vocabulary necessarily spells every legacy
            # name. It is executable policy, not an input/example fixture.
            if path.resolve() == pathlib.Path(__file__).resolve():
                continue
            yield path


def test_legacy_metadata_names_are_confined_to_compatibility_and_migration_docs():
    """Keep previous enum identifiers and topic prefixes out of live examples."""
    offenders = []
    for path in _live_metadata_files():
        rel = path.relative_to(REPO).as_posix()
        allowed = _LEGACY_ALLOWED.get(rel, set())
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            unexpected = _legacy_tokens(line) - allowed
            if unexpected:
                offenders.append(
                    f"{rel}:{lineno}: {line.strip()} "
                    f"(unexpected {sorted(unexpected)}; allowed {sorted(allowed)})"
                )
    assert not offenders, (
        "legacy metadata names remain in live source/docs; use canonical owner "
        "names and Metadata_<Label>, or add an explicitly justified compatibility "
        "seam:\n" + "\n".join(offenders)
    )


def test_legacy_metadata_allowlist_entries_are_not_stale():
    """Prevent removed compatibility exceptions from lingering indefinitely."""
    stale = []
    for rel, expected in sorted(_LEGACY_ALLOWED.items()):
        path = REPO / rel
        if not path.is_file():
            stale.append(f"{rel}: file no longer exists")
            continue
        found = _legacy_tokens(path.read_text(encoding="utf-8"))
        missing = expected - found
        if missing:
            stale.append(f"{rel}: declared but absent {sorted(missing)}")
    assert not stale, (
        "prune stale legacy metadata allowlist entries:\n" + "\n".join(stale)
    )
