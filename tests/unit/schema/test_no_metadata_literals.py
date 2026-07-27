"""Grep gate: no stray ``"Metadata_<X>"`` column literals in source (Task B8).

After the metadata-namespace flip, schema-backed columns are per-topic
(``MetadataGenetic_Strain``), recognized via
:func:`phenotypic.sdk_.is_metadata_header`. The only legal bare ``"Metadata_"``
column literals left in ``src/phenotypic`` are a handful of deliberate shims and
two arbitrary doctest examples. This gate fails the moment a new bare
``"Metadata_<X>"`` literal is introduced anywhere else -- a missed conversion or
a re-stringly-typed column.

The allowlist is **per-literal**, not whole-file: each allowed site may carry
*only* its one declared literal, so adding a *different* stray ``"Metadata_<X>"``
to an already-allowed shim file is still caught.
"""
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "phenotypic"

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
    "gui/_metadata_aliases.py": {
        "Metadata_ImageFileName",
        "Metadata_ImageName",
    },  # centralized historical metadata identity spellings
    "gui/results_viewer/_output_root.py": {"Metadata_ImageName"},   # _IMAGENAME_COL legacy master shim
    "gui/results_viewer/_curation_labels.py": {"Metadata_ImageFile"},  # _LEGACY_IMAGE_FILE curation shim
    "_cli/_cli_recompile_worker.py": {"Metadata_Well"},             # no WELL schema member
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
