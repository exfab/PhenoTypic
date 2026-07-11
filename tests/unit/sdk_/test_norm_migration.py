"""The clip -> norm migration, across every class that carried `clip: bool`."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import phenotypic
from phenotypic import ImagePipeline
from phenotypic.correction import BayesShrinkCorrector, ColorDenoise, VisuShrinkCorrector
from phenotypic.enhance import (
    BayesShrinkEnhancer,
    CompositeEnhance,
    EnhanceBlockMatch,
    LocalEdgeDenoise,
    VisuShrinkEnhancer,
)

MIGRATED = [
    LocalEdgeDenoise,
    BayesShrinkEnhancer,
    EnhanceBlockMatch,
    VisuShrinkEnhancer,
    CompositeEnhance,
    ColorDenoise,
    VisuShrinkCorrector,
    BayesShrinkCorrector,
]

_FIXTURE_ROOT = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "tune"
        / "back_compat_pipelines"
)


@pytest.mark.parametrize("cls", MIGRATED, ids=lambda c: c.__name__)
def test_clip_field_is_gone(cls):
    assert "clip" not in cls.model_fields


@pytest.mark.parametrize("cls", MIGRATED, ids=lambda c: c.__name__)
def test_norm_field_exists_and_is_last(cls):
    assert list(cls.model_fields)[-1] == "norm"


@pytest.mark.parametrize("cls", MIGRATED, ids=lambda c: c.__name__)
def test_legacy_clip_kwarg_raises_migration_message(cls):
    with pytest.raises(
            ValidationError, match=r"`clip` was replaced by `norm` in 0\.18\.0"
    ):
        cls(clip=True)


def test_composite_enhance_defaults_to_none():
    """Preserves the old `clip: bool = False` default."""
    assert CompositeEnhance().norm is None


@pytest.mark.parametrize(
        "cls",
        [c for c in MIGRATED if c is not CompositeEnhance],
        ids=lambda c: c.__name__,
)
def test_others_default_to_clip(cls):
    assert cls().norm == "clip"


@pytest.mark.parametrize("cls", MIGRATED, ids=lambda c: c.__name__)
def test_norm_none_round_trips_through_json(cls):
    loaded = cls.from_json(cls(norm=None).to_json())
    assert loaded.norm is None


@pytest.mark.parametrize("cls", MIGRATED, ids=lambda c: c.__name__)
def test_norm_field_carries_a_description(cls):
    """A missing `norm:` line in the Args: block silently empties the JSON schema."""
    desc = cls.model_fields["norm"].description
    assert desc, f"{cls.__name__} omits `norm:` from its docstring Args: block"
    assert cls.model_json_schema()["properties"]["norm"]["description"] == desc


def test_rescale_sigma_untouched():
    assert VisuShrinkEnhancer.model_fields["rescale_sigma"].annotation is bool


def test_version_is_0_18_0():
    assert phenotypic.__version__ == "0.18.0"


#: The only two fixtures that ever serialized a ``clip`` param. Both were
#: regenerated for 0.18.0; the rest of the corpus stays pinned at its original
#: version, which is the whole point of a back-compat lock.
_REGENERATED = ("bm3d_zero_sigma.json", "local_edge_denoise_small_sigma.json")


def test_back_compat_fixtures_carry_norm_not_clip():
    """No fixture may still pin the removed `clip` **key**.

    A substring scan for ``'"clip"'`` would false-positive on the new
    ``"norm": "clip"`` *value*, so inspect the deserialized param names.
    """
    fixtures = sorted(_FIXTURE_ROOT.glob("*.json"))
    assert fixtures, "fixture directory must not be empty"
    for fp in fixtures:
        blob = json.loads(fp.read_text(encoding="utf-8"))
        for op_name, cfg in blob["pipe_cfgs"].items():
            assert "clip" not in cfg["params"], (
                f"{fp.name}:{op_name} still pins the removed `clip` key"
            )


@pytest.mark.parametrize("name", _REGENERATED)
def test_regenerated_fixtures_pin_norm_and_the_new_version(name):
    """Guards the `carry_norm_not_clip` sweep against passing by mere deletion."""
    blob = json.loads((_FIXTURE_ROOT / name).read_text(encoding="utf-8"))
    assert blob["version"] == "0.18.0"
    params = [cfg["params"] for cfg in blob["pipe_cfgs"].values()]
    assert any(p.get("norm") == "clip" for p in params), name


def test_back_compat_fixtures_still_deserialize():
    fixtures = sorted(_FIXTURE_ROOT.glob("*.json"))
    assert fixtures, "fixture directory must not be empty"
    for fp in fixtures:
        assert ImagePipeline.from_json(fp) is not None
