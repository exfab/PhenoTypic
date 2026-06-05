"""4.5p1 B4 — the held-out config block + group-key inference.

``HeldOutConfig`` carries the conservative robust-eval defaults (all flagged for
a pending fact-check), and ``TuningSpec`` gains a ``held_out`` block that a legacy
``tuning_spec.json`` (with no such block) still validates against. ``infer_group_key``
reads the count scorer's ``check.groupby[0]`` so a run defaults its grouping to
the same unit the QC objective already compares.
"""
from __future__ import annotations

import pandas as pd

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.tune import (
    Budget,
    Categorical,
    Evaluator,
    GridConfig,
    Knob,
    QCScorer,
    SearchSpace,
    TuningSpec,
)
from phenotypic.tune._evaluation._held_out import HeldOutConfig, infer_group_key


def _qc_scorer(group_key="Metadata_ImageName", *, metadata=None):
    if metadata is None:
        # In-memory frame: usable for direct inference, not for JSON round-trip.
        metadata = pd.DataFrame(
            {group_key: ["p1"] * 96, "Object_Label": list(range(96))}
        )
    return QCScorer(
        check=ExpectedVsDetectedCount(metadata=metadata, groupby=[group_key])
    )


def _round_trippable_scorer(tmp_path, group_key="Metadata_ImageName"):
    # A check built from a CSV path round-trips through tuning_spec.json.
    csv = tmp_path / "layout.csv"
    pd.DataFrame(
        {group_key: ["p1"] * 96, "Object_Label": list(range(96))}
    ).to_csv(csv, index=False)
    return _qc_scorer(group_key, metadata=str(csv))


def _spec(scorer=None, **overrides):
    base = dict(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=SearchSpace(knobs=(
            Knob(key="0.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=scorer if scorer is not None else _qc_scorer(),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )
    base.update(overrides)
    return TuningSpec(**base)


def test_evaluator_held_out_defaults():
    cfg = HeldOutConfig()
    assert cfg.held_out_fraction == 0.2
    assert cfg.group_key is None
    assert cfg.min_heldout_plates == 6
    assert cfg.gap_margin_relative == 0.15
    assert cfg.gap_margin_absolute == 0.05


def test_held_out_config_is_frozen():
    cfg = HeldOutConfig()
    try:
        cfg.held_out_fraction = 0.5  # type: ignore[misc]
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "instance" in str(exc).lower()
    else:
        raise AssertionError("HeldOutConfig must be frozen")


def test_tuning_spec_has_default_held_out_block():
    spec = _spec()
    assert isinstance(spec.held_out, HeldOutConfig)
    assert spec.held_out.held_out_fraction == 0.2


def test_tuning_spec_round_trips_with_held_out(tmp_path):
    spec = _spec(
        scorer=_round_trippable_scorer(tmp_path),
        held_out=HeldOutConfig(held_out_fraction=0.3, min_heldout_plates=8),
    )
    payload = spec.model_dump_json()
    back = TuningSpec.model_validate_json(payload)
    assert back.held_out.held_out_fraction == 0.3
    assert back.held_out.min_heldout_plates == 8


def test_legacy_tuning_spec_without_held_out_block_validates(tmp_path):
    # A frozen pre-4.5p1 spec JSON carries no ``held_out`` key → default block.
    spec = _spec(scorer=_round_trippable_scorer(tmp_path))
    payload = spec.model_dump(mode="json")
    payload.pop("held_out", None)
    back = TuningSpec.model_validate(payload)
    assert isinstance(back.held_out, HeldOutConfig)
    assert back.held_out.group_key is None


def test_group_key_auto_infers_from_qc_scorer():
    scorer = _qc_scorer(group_key="Metadata_PlateBatch")
    assert infer_group_key(scorer) == "Metadata_PlateBatch"


def test_infer_group_key_none_when_no_check():
    class _NoCheckScorer:
        check = None

    assert infer_group_key(_NoCheckScorer()) is None
