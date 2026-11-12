"""Study-file reads remain best-effort for local and remote submissions."""

from pathlib import Path

from phenotypic._cli._cli_utils import read_study_yaml


def test_study_with_invalid_utf8_is_logged_and_ignored(tmp_path: Path, caplog) -> None:
    study = tmp_path / "study.yaml"
    study.write_bytes(b"Title: caf\xe9\n")

    assert read_study_yaml(study) is None
    assert "Failed to read --study file" in caplog.text
    assert str(study) in caplog.text


def test_utf8_study_text_is_preserved(tmp_path: Path) -> None:
    study = tmp_path / "study.yaml"
    text = "Title: Café growth\nRelease date: 2026-11-06\n"
    study.write_text(text, encoding="utf-8")

    assert read_study_yaml(study) == text
