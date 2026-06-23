"""Pure cap/over-selection planning for the synced Compare strip (spec §7)."""
from __future__ import annotations

from phenotypic.gui._config import TIMELINE_COMPARE_CAP
from phenotypic.gui._shared.timeline import compare_selection_plan


def test_compare_cap_is_a_small_positive_int_under_webgl_ceiling() -> None:
    # Hard cap on live OSD viewers in the synced Compare strip (spec §7/§9);
    # must stay below the ~16-WebGL-context browser ceiling (spec §12).
    from phenotypic.gui._config import TIMELINE_COMPARE_CAP

    assert isinstance(TIMELINE_COMPARE_CAP, int)
    assert 1 <= TIMELINE_COMPARE_CAP <= 16
    assert TIMELINE_COMPARE_CAP == 12  # spec §7: "~12 live viewers"


def test_under_cap_shows_all_no_notice() -> None:
    plan = compare_selection_plan(["a", "b", "c"], cap=12)
    assert plan.shown == ("a", "b", "c")
    assert plan.total == 3
    assert plan.over_cap is False
    assert plan.notice is None


def test_exactly_cap_shows_all_no_notice() -> None:
    refs = [str(i) for i in range(12)]
    plan = compare_selection_plan(refs, cap=12)
    assert plan.shown == tuple(refs)
    assert plan.over_cap is False
    assert plan.notice is None


def test_over_cap_truncates_to_cap_and_emits_notice() -> None:
    refs = [str(i) for i in range(20)]
    plan = compare_selection_plan(refs, cap=12)
    assert plan.shown == tuple(refs[:12])  # first cap, by selection order
    assert plan.total == 20
    assert plan.over_cap is True
    # EXACT spec §7 wording. The JS controller in browse/_assets/timeline.js
    # (renderOverCapNotice) MUST render this identical string — keep coupled.
    assert plan.notice == "Showing first 12 of 20 — narrow the selection"


def test_default_cap_is_the_config_constant() -> None:
    refs = [str(i) for i in range(TIMELINE_COMPARE_CAP + 1)]
    plan = compare_selection_plan(refs)  # no cap kwarg → uses TIMELINE_COMPARE_CAP
    assert len(plan.shown) == TIMELINE_COMPARE_CAP
    assert plan.over_cap is True


def test_empty_selection_is_a_clean_empty_plan() -> None:
    plan = compare_selection_plan([])
    assert plan.shown == ()
    assert plan.total == 0
    assert plan.over_cap is False
    assert plan.notice is None
