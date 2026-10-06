from __future__ import annotations

import pytest

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.tune._spec import _refuse_reference_metadata
from tests.unit.abc_.test_ref_metadata import _ReadsStrain


def test_tune_refuses_reference_metadata_pipelines():
    with pytest.raises(ValueError, match="reference metadata"):
        _refuse_reference_metadata(ImagePipeline(ops={"r": _ReadsStrain(), "d": OtsuDetector()}))


def test_tune_accepts_ordinary_pipelines():
    _refuse_reference_metadata(ImagePipeline(ops={"b": BlurGauss(), "d": OtsuDetector()}))
