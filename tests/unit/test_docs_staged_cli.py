"""Doc contract: the local staged GPU engine is documented (Spec 1, Plan 2)."""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: The docs that must spell BOTH halves of the Stage-2 signal path in full.
_SIGNAL_DOCS = (
    "CLAUDE.md",
    "src/phenotypic/_cli/CLAUDE.md",
    "docs/source/how_to/pages/gpu_detection_setup.md",
)

#: Every doc whose fully-spelled signal paths must have the real shape. Wider
#: than ``_SIGNAL_DOCS``: the contributor guide spells one worked example and
#: is not asked to spell both, but the example it does spell must be right.
_SHAPE_CHECKED_DOCS = _SIGNAL_DOCS + (
    "docs/source/contrib_guide/gpu_detectors.md",
)

#: A concrete slot id, as ``detector_slot`` renders one: readable segments
#: joined by ``__``, then 8 hex of the exact path.
_CONCRETE_SLOT = re.compile(r"__[0-9a-f]{8}$")

#: A fully-spelled signal path -- ``stage2_raw/.../<stem>.npy`` or the token's
#: ``.json``. It stops at whitespace or a backtick, so prose that merely names
#: the directory ("streams its shard to ``stage2_raw/``") is not matched and is
#: not asked to carry a full path.
_SIGNAL_PATH = re.compile(r"stage2_(?:raw|done)/[^\s`]*?\.(?:npy|json)")


def _collapse(text: str) -> str:
    """Whitespace-insensitive form, so a doc may wrap a quoted message."""
    return " ".join(text.split())


def _real_signal_shapes() -> dict[str, tuple[int, int]]:
    """``{prefix: (segment count, slot segment index)}``, read off the CODE.

    Derived from the path builders rather than restated as a literal, so this
    test cannot itself go stale the way the doc did.
    """
    from phenotypic._cli._cli_stage2_token import (
        stage2_raw_path,
        stage2_token_path,
    )
    from phenotypic.sdk_ import progress_dir

    root = Path("/out")
    shapes: dict[str, tuple[int, int]] = {}
    for prefix, builder in (
        ("stage2_raw", stage2_raw_path),
        ("stage2_done", stage2_token_path),
    ):
        parts = (
            builder(root, "DS", "STEM", "SLOT")
            .relative_to(progress_dir(root))
            .parts
        )
        assert parts[0] == prefix
        shapes[prefix] = (len(parts), parts.index("SLOT"))
    return shapes


def test_the_documented_stage2_signal_paths_have_the_real_shape():
    """A substring check is one directory level too shallow to catch this.

    The previous version of this test asserted only that the token
    ``stage2_raw`` still *appeared* somewhere. It did -- while all three docs
    spelled the pre-slot-keying ``stage2_raw/<ds>/<stem>.npy``, one level short
    of the real ``stage2_raw/<ds>/<slot>/<stem>.npy``. Stale text passed a green
    test, which is the defect worth fixing rather than the sentence.

    So this compares each documented path against the shape the **code** builds:
    the same number of segments, and a segment naming the slot where the code
    puts the slot. Both halves of the signal, because documenting only one is
    how the other rots.

    What shape of input would make this pass on stale text? Only a doc that
    already spells the slot level -- which is the thing being asserted. A doc
    that drops the path entirely is caught by the final assertion, not by
    silence.
    """
    shapes = _real_signal_shapes()
    for relative in _SHAPE_CHECKED_DOCS:
        text = (REPO / relative).read_text(encoding="utf-8")
        found: dict[str, str] = {}
        for match in _SIGNAL_PATH.finditer(text):
            documented = match.group(0)
            prefix, *segments = documented.split("/")
            expected_len, slot_index = shapes[prefix]
            assert len(segments) + 1 == expected_len, (
                f"{relative}: documented {documented!r} has "
                f"{len(segments) + 1} path segments, but {prefix} paths have "
                f"{expected_len} -- the signal is keyed by detector slot"
            )
            slot_segment = segments[slot_index - 1]
            assert (
                "slot" in slot_segment.lower()
                or _CONCRETE_SLOT.search(slot_segment) is not None
            ), (
                f"{relative}: documented {documented!r} does not name the "
                f"detector slot at segment {slot_index} (found "
                f"{slot_segment!r}) -- use a `<slot>` placeholder or a real "
                "slot id ending in 8 hex characters"
            )
            found[prefix] = documented
        if relative in _SIGNAL_DOCS:
            assert set(found) == set(shapes), (
                f"{relative}: must spell BOTH Stage-2 signal paths in full; "
                f"found {sorted(found)}"
            )
        else:
            assert found, f"{relative}: spells no Stage-2 signal path at all"


def test_the_worked_slot_example_is_what_detector_slot_produces():
    """The 8-hex half of a slot id cannot be eyeballed.

    Four docs print ``CompositeDetector__ops-0__<hex>`` as the worked example of
    a detector slot. A hand-written digest is unverifiable prose: it reads
    exactly as convincingly when it is wrong, and a reader who copies it to go
    looking on disk finds nothing and cannot tell which end is at fault. So
    derive it and compare.
    """
    from phenotypic._cli._cli_stage2_token import detector_slot

    slot = detector_slot(("CompositeDetector", "ops[0]"))
    assert _CONCRETE_SLOT.search(slot) is not None, slot
    for relative in (
        "CLAUDE.md",
        "src/phenotypic/_cli/CLAUDE.md",
        "docs/source/how_to/pages/gpu_detection_setup.md",
        "docs/source/contrib_guide/gpu_detectors.md",
    ):
        text = (REPO / relative).read_text(encoding="utf-8")
        assert slot in text, (
            f"{relative}: the worked slot example is not "
            f"detector_slot(('CompositeDetector', 'ops[0]')), which is {slot!r}"
        )


def test_claude_md_documents_local_staged_gpu():
    """The Stage-2 signal is BOTH files; naming only one is how the other rots."""
    txt = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    low = txt.lower()
    assert "GpuDetector" in txt
    assert "stage" in low
    assert "stage2_raw" in low
    assert "token" in low


def test_how_to_documents_local_staged_gpu():
    doc = REPO / "docs" / "source" / "how_to" / "pages" / "gpu_detection_setup.md"
    txt = doc.read_text(encoding="utf-8")
    low = txt.lower()
    assert "stage2_raw" in low
    assert "token" in low
    assert "stage" in low


def test_the_staged_docs_do_not_still_describe_an_objmap_sidecar():
    """The concept is dead. Only the *scheduler* sidecar rule may survive.

    Both files kept the word for an unrelated rule ("do not submit scheduler
    sidecar jobs"), so a bare ``"sidecar" not in txt`` would be wrong -- and a
    bare ``"sidecar" in txt`` (what these tests asserted before) went on passing
    the whole time the object-map sidecar was being removed.
    """
    for relative in (
        "CLAUDE.md",
        "src/phenotypic/_cli/CLAUDE.md",
        "docs/source/how_to/pages/gpu_detection_setup.md",
    ):
        low = (REPO / relative).read_text(encoding="utf-8").lower()
        assert "objmap sidecar" not in low, relative
        assert "objmap **sidecar**" not in low, relative
        assert "results/<dataset>/objmap/" not in low, relative
        for paragraph in low.split("\n\n"):
            if "sidecar" in paragraph:
                assert "scheduler" in paragraph, f"{relative}: {paragraph}"


def test_the_contrib_guide_quotes_the_real_nesting_refusal():
    """A quoted error message is a promise about what the user will see.

    ``gpu_detectors.md`` prints the refusal verbatim in a ``text`` block. If
    someone rewords the raiser, that block becomes a plausible-looking lie and
    nothing else notices -- the refusal's own tests match on a substring
    ("only composition primitives"), which survives almost any rewording.
    """
    from phenotypic._cli._cli_validation import (
        UnstageableGpuDetectorError,
        _child_contract,
    )
    from phenotypic.detect import TwoKFilamentousDetector

    with pytest.raises(UnstageableGpuDetectorError) as excinfo:
        _child_contract(TwoKFilamentousDetector())

    guide = (
        REPO / "docs" / "source" / "contrib_guide" / "gpu_detectors.md"
    ).read_text(encoding="utf-8")
    assert _collapse(str(excinfo.value)) in _collapse(guide)


def test_the_contrib_guide_names_the_real_child_contract_entries():
    """The guide documents a closed table; check it against the table.

    The guide shipped saying the two composites hand children the ``"same"``
    image. The code says ``"parallel"``. Both words describe the same
    behaviour, which is exactly why nothing caught it -- a reader grepping the
    source for the documented literal finds nothing and cannot tell whether the
    doc or their grep is wrong.
    """
    from phenotypic._cli._cli_validation import (
        _CHILD_CONTRACT,
        _populate_child_contract,
    )

    _populate_child_contract()
    guide = (
        REPO / "docs" / "source" / "contrib_guide" / "gpu_detectors.md"
    ).read_text(encoding="utf-8")

    for cls, contract in _CHILD_CONTRACT.items():
        assert cls.__name__ in guide, cls
        assert f'`"{contract}"`' in guide, contract
    # `ImagePipeline` is handled in `_child_contract`, not by a table entry.
    assert '`"sequence"`' in guide


def test_claude_md_documents_gpu_flags():
    txt = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    for flag in (
        "--gpu-slurm",
        "--gpu-shards",
        "--gpu-workers-per-gpu",
    ):
        assert flag in txt
    assert "--gpu-batch-size" not in txt


def test_how_to_documents_slurm_staging():
    doc = REPO / "docs" / "source" / "how_to" / "pages" / "gpu_detection_setup.md"
    low = doc.read_text(encoding="utf-8").lower()
    assert "afterany" in low
    assert "shard" in low
    assert "--gpu-shards" in low
