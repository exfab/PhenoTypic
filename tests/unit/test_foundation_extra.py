"""Packaging-extras + DINO typing-alias contract for the foundation models."""

import tomllib
from pathlib import Path
from typing import get_args

from phenotypic.tools_.typing_ import DinoSize, DinoVersion

REPO = Path(__file__).resolve().parents[2]


def test_foundation_and_gpu_extras_declared():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert "foundation" in extras and "gpu" in extras
    joined = " ".join(extras["foundation"])
    assert "transformers" in joined and "huggingface_hub" in joined
    # foundation pulls torch; gpu is the umbrella
    assert any("phenotypic[torch" in d for d in extras["foundation"])
    assert any("foundation" in d for d in extras["gpu"])


def test_dino_typing_aliases():
    assert set(get_args(DinoVersion)) == {2, 3}
    assert set(get_args(DinoSize)) == {"small", "base", "large"}
