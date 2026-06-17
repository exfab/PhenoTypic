"""Doc contract: the local staged GPU engine is documented (Spec 1, Plan 2)."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_claude_md_documents_local_staged_gpu():
    txt = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    assert "GpuDetector" in txt
    assert "stage" in txt.lower() and "sidecar" in txt.lower()


def test_how_to_documents_local_staged_gpu():
    doc = REPO / "docs" / "source" / "how_to" / "pages" / "gpu_detection_setup.md"
    txt = doc.read_text(encoding="utf-8")
    low = txt.lower()
    assert "sidecar" in low
    assert "stage" in low
