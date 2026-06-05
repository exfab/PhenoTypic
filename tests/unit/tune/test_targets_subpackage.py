from __future__ import annotations


def test_targets_subpackage_surface():
    from phenotypic.tune import targets
    assert set(targets.__all__) == {
        "Param", "Presence", "Nested", "KnobTarget", "parse_key",
        "TunableParam", "pipeline_targets",
    }
    from phenotypic.tune.targets import Param  # importable
    assert Param(op=0, field="sigma").key == "0.sigma"


def test_targets_subpackage_is_optuna_free():
    import sys
    sys.modules.pop("optuna", None)
    import importlib
    importlib.import_module("phenotypic.tune.targets")
    assert "optuna" not in sys.modules


def test_target_symbols_absent_from_top_level():
    import phenotypic.tune as t
    # the param-reference surface lives in the targets subpackage, not the
    # flat top-level __all__ (keeps it lean)
    for name in ("Param", "Presence", "Nested", "KnobTarget",
                 "TunableParam", "pipeline_targets", "parse_key"):
        assert name not in t.__all__
