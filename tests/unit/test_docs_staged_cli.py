"""Doc contract: the local staged GPU engine is documented (Spec 1, Plan 2)."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


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
