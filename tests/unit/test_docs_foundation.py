"""The GPU-setup how-to documents the foundation models + gated install flow."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_HOW_TO = REPO / "docs/source/how_to/pages/gpu_detection_setup.md"


def test_how_to_documents_sam3_and_gated_install():
    txt = _HOW_TO.read_text(encoding="utf-8")
    assert "Sam3Detector" in txt and "DinoSam2Detector" in txt
    assert "hf auth login" in txt and "--extra foundation" in txt


def test_how_to_documents_license_acceptance_and_offline():
    txt = _HOW_TO.read_text(encoding="utf-8")
    assert "PHENOTYPIC_ACCEPT_MODEL_LICENSE" in txt
    assert "HF_HUB_OFFLINE" in txt
    assert "HF_HOME" in txt
    # Per-model license posture is stated.
    assert "SAM License" in txt and "Apache" in txt
