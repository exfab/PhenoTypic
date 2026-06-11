# tests/unit/tune/test_study_name_cutover.py
"""Phase 2: the study name is bumped to ``tune_cost_v1`` (hard cutover, OQ7).

Bumping the single ``_STUDY_NAME`` constant makes the silent direction-mismatch
load (optuna ``load_if_exists`` keeps the old ``maximize``) impossible by
construction: new code never opens the pre-cutover ``"tune"`` study.
"""
from __future__ import annotations


def test_study_name_is_bumped():
    from phenotypic.tune._tune_cli._run import _STUDY_NAME

    assert _STUDY_NAME == "tune_cost_v1"


def test_gui_default_study_name_matches_cli_constant():
    # The GUI fallback constant must stay in lockstep with the CLI constant so a
    # spec-discovered run resolves the bumped study, not the inert legacy one.
    from phenotypic.gui.tune._run_root import _DEFAULT_STUDY_NAME
    from phenotypic.tune._tune_cli._run import _STUDY_NAME

    assert _DEFAULT_STUDY_NAME == _STUDY_NAME
