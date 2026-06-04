from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune import QCScorer
from phenotypic.tune._evaluation import build_pipeline
from phenotypic.tune._strategies._enumerate import enumerate_grid

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from migrate_sweep_manifest import migrate_manifest_to_spec  # noqa: E402

GOLDEN = (
    Path(__file__).resolve().parents[3]
    / "tests/fixtures/tune/grid_golden_manifest.json"
)


def _scorer(tmp_path) -> QCScorer:
    csv = tmp_path / "layout.csv"
    pd.DataFrame({"Metadata_ImageName": ["p"] * 96,
                  "Object_Label": list(range(96))}).to_csv(csv, index=False)
    return QCScorer(check=ExpectedVsDetectedCount(
        metadata=str(csv), groupby=["Metadata_ImageName"]))


def _sig(pipe):
    return tuple((type(o).__name__,
                  json.dumps(o.model_dump(mode="json"), sort_keys=True, default=str))
                 for o in pipe.get_ops().values())


def test_migrated_spec_grid_matches_manifest(tmp_path):
    manifest = json.loads(GOLDEN.read_text())
    spec = migrate_manifest_to_spec(manifest, scorer=_scorer(tmp_path))
    combos = enumerate_grid(spec.search_space)
    migrated = {_sig(build_pipeline(spec.pipeline, c)) for c in combos}
    # the migrated grid reproduces the manifest's op-combinations
    golden = set()
    for cfg in manifest["configs"].values():
        for pd_ in cfg["pipelines"].values():
            golden.add(_sig(ImagePipeline.from_json(json.dumps(pd_))))
    assert migrated == golden
