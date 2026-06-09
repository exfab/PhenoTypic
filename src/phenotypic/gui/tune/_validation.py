"""Pure blocked-deploy validation for the Tune Setup surface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from phenotypic.gui.tune._domain_editor import grid_feasibility
from phenotypic.tune._search_space import FloatRange, IntRange, SearchSpace
from phenotypic.tools_ import CONFIG_SUFFIX_TUNING, matches_any_suffix

Blocks = Literal["continue", "deploy", "both"]


@dataclass(frozen=True)
class Issue:
    """One Setup validation problem."""

    section: str
    message: str
    blocks: Blocks = "both"


def validate_setup(
    space: SearchSpace, *, scorer_kind: str, metadata_present: bool
) -> list[Issue]:
    """Return blocking Setup issues."""
    issues: list[Issue] = []
    if len(space.knobs) == 0:
        issues.append(Issue("search_space", "No active knobs to tune."))
    for knob in space.knobs:
        domain = knob.domain
        if isinstance(domain, (FloatRange, IntRange)) and not domain.high > domain.low:
            issues.append(Issue("search_space", f"{knob.key}: low must be < high."))
    if scorer_kind == "qc" and not metadata_present:
            issues.append(Issue("scorer", "QC scorer needs a metadata CSV."))
    return issues


def preflight_issues(space: SearchSpace, *, strategy: str) -> list[Issue]:
    """Return Run/Deploy preflight issues."""
    if strategy != "grid":
        return []
    ok, message = grid_feasibility(space)
    if ok:
        return []
    return [Issue("strategy", message, blocks="deploy")]


def can_deploy(setup_issues: list[Issue], run_issues: list[Issue]) -> bool:
    """Return whether no issue blocks Deploy."""
    for issue in [*setup_issues, *run_issues]:
        if issue.blocks in {"deploy", "both"}:
            return False
    return True


def spec_path_issue(spec_path: str | None) -> Issue | None:
    """Return a Deploy issue when the CLI spec path is not a tuning spec."""
    if not spec_path:
        return Issue("spec", "Choose a pipeline or tuning spec first.", blocks="deploy")
    if matches_any_suffix(spec_path, (CONFIG_SUFFIX_TUNING,)):
        return None
    return Issue(
        "spec",
        "Deploy needs an authored .json.pht-tune spec. Pipeline-only paths "
        "must be converted in Setup before launch.",
        blocks="deploy",
    )
