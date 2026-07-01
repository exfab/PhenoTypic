"""Grep gate: no stray ``"Metadata_<X>"`` column literals in source (Task B8).

After the metadata-namespace flip, schema-backed columns are per-topic
(``MetadataGenetic_Strain``), recognized via
:func:`phenotypic.sdk_.is_metadata_header`. The only legal bare ``"Metadata_"``
column literals left in ``src/phenotypic`` are a handful of deliberate shims and
two arbitrary doctest examples. This gate fails the moment a new bare
``"Metadata_<X>"`` literal is introduced anywhere else -- a missed conversion or
a re-stringly-typed column.
"""
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "phenotypic"

# A quote immediately followed by ``Metadata_<letter>`` -- i.e. a specific
# metadata column literal (not the generic ``"Metadata_"`` prefix constant,
# which has no trailing letter and so never matches).
_PAT = re.compile(r"""["']Metadata_[A-Za-z]""")

# The only source sites allowed to carry such a literal, keyed by repo-relative
# posix path -> the reason. Every other match is a missed conversion.
_ALLOWED = {
    # generic-fallback + legacy shims (kept literals, load old data)
    "gui/results_viewer/_output_root.py",       # _IMAGENAME_COL legacy master shim
    "gui/results_viewer/_curation_labels.py",   # _LEGACY_IMAGE_FILE curation shim
    "_cli/_cli_recompile_worker.py",            # "Metadata_Well" (no WELL member)
    # arbitrary-column doctest examples (demonstrate non-vocabulary columns)
    "abc_/_post_measurement.py",                # AddConstant(column="Metadata_Flag")
    "post/_merge_metadata.py",                  # doctest "Metadata_Condition"
}


def test_no_specific_metadata_literals_outside_allowed():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel in _ALLOWED:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _PAT.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "stringly-typed metadata literals remain (use str(ENUM.MEMBER) or "
        "is_metadata_header instead):\n" + "\n".join(offenders)
    )
