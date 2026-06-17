"""Licensing scaffolding for the GPU detector components (Spec 1 §12)."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_notice_exists_and_disclaims_weight_redistribution():
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    assert "does not redistribute" in notice.lower()
    assert "SAM2" in notice


def test_license_files_present():
    assert (REPO / "licenses" / "sam2-Apache-2.0.txt").is_file()
    assert (REPO / "licenses" / "micro-sam-LICENSE.txt").is_file()


def test_sam2_license_is_apache_2():
    txt = (REPO / "licenses" / "sam2-Apache-2.0.txt").read_text(encoding="utf-8")
    assert "Apache License" in txt
    assert "Version 2.0" in txt


def test_micro_sam_license_is_mit():
    txt = (REPO / "licenses" / "micro-sam-LICENSE.txt").read_text(encoding="utf-8")
    assert "MIT License" in txt
    assert "Computational Cell Analytics" in txt


def test_sam3_and_dinov2_licenses_present():
    assert (REPO / "licenses" / "sam3-SAM-License.txt").is_file()
    assert (REPO / "licenses" / "dinov2-Apache-2.0.txt").is_file()


def test_dinov2_license_is_apache_2():
    txt = (REPO / "licenses" / "dinov2-Apache-2.0.txt").read_text(encoding="utf-8")
    assert "Apache License" in txt
    assert "Version 2.0" in txt


def test_notice_names_sam3_and_dinov2():
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    assert "SAM3" in notice and "DINOv2" in notice
    assert "does not redistribute" in notice.lower()


# ---------------------------------------------------------------------------
# Spec 2b additions: DINOv3 (gated) + INSID3 (Apache) + FSSDINO (paper)
# ---------------------------------------------------------------------------


def test_dinov3_license_and_insid3_present():
    assert (REPO / "licenses" / "dinov3-License.txt").is_file()
    assert (REPO / "licenses" / "insid3-Apache-2.0.txt").is_file()


def test_dinov3_license_is_the_meta_custom_license():
    txt = (REPO / "licenses" / "dinov3-License.txt").read_text(encoding="utf-8")
    assert "DINOv3 License" in txt
    # Distinguishing clauses of the custom Meta license (not Apache-2.0).
    assert "DINO Materials" in txt
    assert "Built with DINOv3" in txt


def test_insid3_license_is_apache_2():
    txt = (REPO / "licenses" / "insid3-Apache-2.0.txt").read_text(encoding="utf-8")
    assert "Apache License" in txt
    assert "Version 2.0" in txt


def test_notice_names_dinov3_insid3_fssdino():
    n = (REPO / "NOTICE").read_text(encoding="utf-8")
    assert "DINOv3" in n and "INSID3" in n and "FSSDINO" in n
    # FSSDINO must be flagged clean-room (no vendored code).
    assert "clean-room" in n.lower()
