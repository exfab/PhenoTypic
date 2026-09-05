"""The per-image record's readers and shared vocabulary (spec §6.1).

**Readers here, writers in :mod:`phenotypic._cli._cli_image_record`**, and the
split is forced rather than stylistic. P6 Task 0 moves ``valid_image_success``
into ``sdk_/_run_state.py``, and after P3 that function reads this record --
but INV-LAYER forbids ``sdk_`` importing :mod:`phenotypic._cli` at module
scope *or* inside a function. Readers in ``_cli`` would leave ``sdk_`` unable
to read a record at all without either duplicating this vocabulary or
re-implementing the parse, and both are the duplication CAN-27 and CAN-8 exist
to close. It is the same read/write asymmetry spec §5.2 already declares for
run state.

**``ARTIFACT_KIND_FILE`` / ``ARTIFACT_KIND_STORE`` are deliberately NOT here.**
The plan's Interfaces block placed them in this module, on the reasoning that
the kinds would otherwise be "spelled at each comparison". They are not: they
have lived in :mod:`._io_constants` since the marker schema was written, and
eight modules import them from there. Adding them here would create exactly
the second home that argument exists to prevent -- a correct rule applied to a
tree that had already satisfied it. Import them from ``_io_constants``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from ._io_constants import image_record_path

__all__ = [
    "PROVENANCE_FORWARD",
    "PROVENANCE_MIGRATED",
    "RECORD_VERSION",
    "STAGE_MEASURED",
    "STAGE_STAGE1",
    "STAGE_STAGE2",
    "STAGE_STAGE3",
    "read_image_record",
    "record_provenance",
]

#: The stage names, as shared constants imported by every writer and reader
#: (CAN-27). ``stages`` stays an **open** map (§6.1) -- a future stage is
#: additive -- but the names *this build* writes cannot be misspelled, because
#: there is exactly one place they are spelled. This replaces O-2's
#: ``KNOWN_STAGES`` + advisory, which could not be built without either
#: breaking INV-LAYER (the advisory is emitted from ``sdk_``, which may not
#: import ``_cli``) or duplicating the set.
STAGE_STAGE1: Final[str] = "stage1"
STAGE_STAGE2: Final[str] = "stage2"
STAGE_STAGE3: Final[str] = "stage3"
STAGE_MEASURED: Final[str] = "measured"

#: The record's ``provenance`` values (U-10). Compared in ``sdk_`` by
#: ``valid_image_success`` and written in ``_cli`` by ``--mode migrate``, so
#: the spelling lives here rather than in either one.
PROVENANCE_FORWARD: Final[str] = "forward"
PROVENANCE_MIGRATED: Final[str] = "migrated"

#: A version mismatch **invalidates rather than migrates**, matching the
#: policy `_cli/CLAUDE.md` states for the marker this record replaces.
RECORD_VERSION: Final[int] = 1


def read_image_record(
    output_dir: Path, dataset: str, image_stem: str
) -> dict[str, object] | None:
    """Return one image's record, or ``None`` when it cannot be read.

    Every failure returns ``None`` rather than raising -- INV-VERDICT's
    degrade half: an unreadable record must make an image look *less*
    finished, never make a caller explode. A truncated file, a JSON array
    where an object belongs, and an absent file are all the same answer,
    because a caller that must distinguish them is asking the wrong question
    of this function.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        image_stem: Source image stem.

    Returns:
        The record as a mapping, or ``None``.
    """
    try:
        raw = json.loads(
            image_record_path(output_dir, dataset, image_stem).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError, TypeError):
        return None
    return raw if isinstance(raw, dict) else None


def record_provenance(record: object) -> str:
    """Return a record's provenance, defaulting to ``"forward"`` (U-10).

    **Absent means forward, and that is the strict reading.** A record written
    before this field existed -- or by any writer that forgets it -- must be
    fenced on ``work_id`` like any other, so the default is the value that
    *keeps* the fence. Defaulting to ``"migrated"``, or reading a bare
    ``record["provenance"]`` and treating the ``KeyError`` as "unmarked",
    would strip the fence from every tree written before P3.

    A function rather than the ``record.get("provenance", "forward")`` the
    plan prescribes: the rule is a *default*, and a default restated at each
    call site is one edit away from being two defaults. This is the same
    reason `stage3_markers_required` is a live gate finding.

    Args:
        record: A record mapping, or anything at all -- a non-mapping is
            treated as unmarked rather than raising, so a caller that already
            has :func:`read_image_record`'s ``None`` need not branch twice.

    Returns:
        ``PROVENANCE_MIGRATED`` only when the record says so explicitly;
        ``PROVENANCE_FORWARD`` in every other case.
    """
    if not isinstance(record, dict):
        return PROVENANCE_FORWARD
    value = record.get("provenance", PROVENANCE_FORWARD)
    return value if value == PROVENANCE_MIGRATED else PROVENANCE_FORWARD
