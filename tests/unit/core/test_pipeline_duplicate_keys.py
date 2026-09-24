"""Pipeline JSON with a repeated key is refused, not silently reordered.

Spec ``2026-09-24-cli-preflight`` F14 and §10.4. Plain ``json.loads`` keeps the
last duplicate, and the survivor takes the FIRST occurrence's position, so
``det, blur, det`` ran the second detector before ``blur``
(claim-verification report §8).
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector, TriangleDetector
from phenotypic.enhance import BlurGauss
from phenotypic.phenotypicCLI import phenotypic_cli


def _pipeline_text_with_duplicate_op() -> str:
    base = json.loads(
        ImagePipeline(ops={"det": OtsuDetector(), "blur": BlurGauss()}).to_json()
    )
    ops = base["pipe_cfgs"]
    triangle = json.loads(ImagePipeline(ops={"x": TriangleDetector()}).to_json())
    entries = [
        f'"det": {json.dumps(ops["det"])}',
        f'"blur": {json.dumps(ops["blur"])}',
        f'"det": {json.dumps(triangle["pipe_cfgs"]["x"])}',
    ]
    base["pipe_cfgs"] = "__OPS__"
    return json.dumps(base).replace('"__OPS__"', "{" + ", ".join(entries) + "}")


def test_a_duplicate_op_key_is_refused_with_its_path() -> None:
    with pytest.raises(ValueError, match=r"pipe_cfgs\.det"):
        ImagePipeline.from_json(_pipeline_text_with_duplicate_op())


def test_a_duplicate_key_inside_params_is_refused() -> None:
    text = ImagePipeline(ops={"det": OtsuDetector()}).to_json()
    data = json.loads(text)
    params = data["pipe_cfgs"]["det"]["params"]
    first_key = next(iter(params))
    doubled = json.dumps(params)[:-1] + f', "{first_key}": {json.dumps(params[first_key])}}}'
    data["pipe_cfgs"]["det"]["params"] = "__P__"
    text = json.dumps(data).replace('"__P__"', doubled)

    with pytest.raises(ValueError, match=rf"pipe_cfgs\.det\.params\.{first_key}"):
        ImagePipeline.from_json(text)


def test_a_pipeline_without_duplicates_still_round_trips() -> None:
    original = ImagePipeline(ops={"det": OtsuDetector(), "blur": BlurGauss()})

    restored = ImagePipeline.from_json(original.to_json())

    assert list(restored.get_ops()) == ["det", "blur"]


@pytest.mark.parametrize("value", ["12", "0", "32"])
def test_bit_depth_accepts_only_8_or_16(value: str, tmp_path) -> None:
    pipeline = tmp_path / "p.json"
    pipeline.write_text(ImagePipeline(ops={"d": OtsuDetector()}).to_json(), encoding="utf-8")
    result = CliRunner().invoke(
        phenotypic_cli,
        ["--pipeline", str(pipeline), "--input", str(tmp_path), "--output",
         str(tmp_path / "o"), "--bit-depth", value],
    )

    assert result.exit_code == 2, result.output
    assert "--bit-depth" in result.output


def test_bit_depth_16_parses_to_an_integer(monkeypatch, tmp_path) -> None:
    from phenotypic import phenotypicCLI

    seen: dict[str, object] = {}
    real = phenotypicCLI.ExecutionConfig

    def capture(**kwargs):
        seen["bit_depth"] = kwargs["bit_depth"]
        raise SystemExit(0)

    monkeypatch.setattr(phenotypicCLI, "ExecutionConfig", capture, raising=False)
    pipeline = tmp_path / "p.json"
    pipeline.write_text(ImagePipeline(ops={"d": OtsuDetector()}).to_json(), encoding="utf-8")
    (tmp_path / "in").mkdir()
    CliRunner().invoke(
        phenotypic_cli,
        ["--pipeline", str(pipeline), "--input", str(tmp_path / "in"), "--output",
         str(tmp_path / "o"), "--bit-depth", "16"],
    )
    monkeypatch.setattr(phenotypicCLI, "ExecutionConfig", real, raising=False)

    assert seen["bit_depth"] == 16
