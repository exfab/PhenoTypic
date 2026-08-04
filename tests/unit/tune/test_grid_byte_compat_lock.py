from __future__ import annotations

import json
from pathlib import Path

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.tune import Categorical, Knob, SearchSpace
from phenotypic.tune._evaluation import build_pipeline
from phenotypic.tune.strategy._enumerate import enumerate_grid

GOLDEN = (
    Path(__file__).resolve().parents[3]
    / "tests/fixtures/tune/grid_golden_manifest.json"
)


def _signature(pipe: ImagePipeline) -> tuple:
    """Order-sensitive op signature, name/uuid-independent."""
    return tuple(
        (type(op).__name__, json.dumps(op.model_dump(mode="json"),
                                       sort_keys=True, default=str))
        for op in pipe.get_ops().values()
    )


def _golden_signatures() -> set:
    manifest = json.loads(GOLDEN.read_text())
    sigs = set()
    for cfg in manifest["configs"].values():
        for pipe_dict in cfg["pipelines"].values():
            pipe = ImagePipeline.from_json(json.dumps(pipe_dict))
            sigs.add(_signature(pipe))
    return sigs


def _tune_space() -> SearchSpace:
    return SearchSpace(knobs=(
        Knob(key="0.BlurGauss.__enabled__",
             domain=Categorical(choices=(True, False)), source="presence_optin"),
        Knob(key="0.sigma", domain=Categorical(choices=(1.0, 2.0)),
             conditional_on=(("0.BlurGauss.__enabled__", True),)),
        Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ))


def test_tune_grid_reproduces_golden_op_combinations():
    base = ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()])
    combos = enumerate_grid(_tune_space())
    assert len(combos) == 6
    tune_sigs = {_signature(build_pipeline(base, c)) for c in combos}
    assert tune_sigs == _golden_signatures()
