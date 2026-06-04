from click.testing import CliRunner

from phenotypic._cli._cli_process_single import main


def test_process_only_option_parses(tmp_path, monkeypatch):
    called = {}

    def fake_core(**kwargs):
        called.update(kwargs)
        return True

    monkeypatch.setattr(
        "phenotypic._cli._cli_process_single.process_single_apply_only_core", fake_core
    )
    pipe = tmp_path / "p.json"
    pipe.write_text("{}", encoding="utf-8")
    img = tmp_path / "in" / "a.tif"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    res = CliRunner().invoke(
        main,
        [
            "--pipeline", str(pipe),
            "--image", str(img),
            "--output-dir", str(tmp_path / "out"),
            "--dataset-name", "in",
            "--process-only", "detect_mat",
            "--input-root", str(tmp_path / "in"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert called["layer"] == "detect_mat"
    assert str(called["input_root"]) == str(tmp_path / "in")


def test_process_only_requires_input_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "phenotypic._cli._cli_process_single.process_single_apply_only_core",
        lambda **kwargs: True,
    )
    pipe = tmp_path / "p.json"
    pipe.write_text("{}", encoding="utf-8")
    img = tmp_path / "in" / "a.tif"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"x")
    res = CliRunner().invoke(
        main,
        [
            "--pipeline", str(pipe),
            "--image", str(img),
            "--output-dir", str(tmp_path / "out"),
            "--dataset-name", "in",
            "--process-only", "rgb",
        ],
    )
    assert res.exit_code != 0
    assert "input-root" in res.output.lower()
