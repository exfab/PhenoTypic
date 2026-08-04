from __future__ import annotations

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.tune._search_space._discovery import TunableParam, pipeline_targets


def test_pipeline_targets_catalog():
    pipe = ImagePipeline(ops=[BlurGauss(sigma=2.0), OtsuDetector()])
    cat = pipeline_targets(pipe)
    assert cat and all(isinstance(t, TunableParam) for t in cat)
    sigma = next(t for t in cat if t.target.key == "0.sigma")
    assert sigma.op_class == "BlurGauss"           # always populated
    assert sigma.value_type == "float"
    assert sigma.default == 2.0                       # current value
    assert sigma.suggested_domain is not None
