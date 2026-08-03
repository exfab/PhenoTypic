"""The GPU-setup how-to documents the foundation models + gated install flow."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_HOW_TO = REPO / "docs/source/how_to/pages/gpu_detection_setup.md"


def test_how_to_documents_sam3_and_gated_install():
    txt = _HOW_TO.read_text(encoding="utf-8")
    assert "Sam3" in txt and "DinoSam2Detector" in txt
    assert "hf auth login" in txt and "--extra foundation" in txt


def test_how_to_documents_license_acceptance_and_offline():
    txt = _HOW_TO.read_text(encoding="utf-8")
    assert "PHENOTYPIC_ACCEPT_MODEL_LICENSE" in txt
    assert "HF_HUB_OFFLINE" in txt
    assert "HF_HOME" in txt
    # Per-model license posture is stated.
    assert "SAM License" in txt and "Apache" in txt


def test_how_to_documents_semantic_detectors_and_dinov3():
    """Spec 2b: the two semantic detectors + gated DINOv3 handshake + exemplar."""
    txt = _HOW_TO.read_text(encoding="utf-8")
    # Both semantic detectors are documented.
    assert "Insid3Detector" in txt and "FssDinoDetector" in txt
    # Gated DINOv3 handshake (download + license).
    assert "DINOv3 License" in txt
    assert "--model-type dinov3" in txt
    assert "dinov3-vitb16-pretrain-lvd1689m" in txt
    # The exemplar interface (reference / support exemplars).
    assert "reference_image" in txt and "reference_mask" in txt
    assert "support_images" in txt and "support_masks" in txt
    # Semantic → objmask → downstream watershed note.
    assert "output_kind" in txt and "objmask" in txt
    assert "SeparateObjects" in txt
    # Per-model license table covers the new methods.
    assert "CC BY-NC-SA" in txt and "clean-room" in txt
