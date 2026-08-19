"""The boundary that makes `_services` a layer rather than a folder."""

from __future__ import annotations

import pkgutil

import pytest

from ._boundary import forbidden_imports_after_importing, gui_modules_reached


def _service_modules() -> list[str]:
    import phenotypic._services as services

    # walk_packages, not iter_modules: iter_modules is non-recursive, so a
    # _services/<subpkg>/leak.py importing dash was invisible to this gate.
    return [
        m.name
        for m in pkgutil.walk_packages(services.__path__, prefix="phenotypic._services.")
    ]


def test_services_package_exists_and_is_lazy():
    import phenotypic._services as services

    assert services.__path__, "phenotypic._services must be a package"


@pytest.mark.parametrize("module", _service_modules())
def test_service_module_imports_no_dash(module: str) -> None:
    leaked = forbidden_imports_after_importing(module)
    assert not leaked, f"{module} dragged {leaked} into sys.modules"


# The single accepted upward import in this tier, and the reason for it.
# RunRegistry.rehydrate_from_sandbox needs classify(); _classifier.py is itself
# Dash-free after Task 2, but no task promotes it. Anything NOT listed here is a
# test failure rather than a silent precedent.
GUI_IMPORT_ALLOWLIST: dict[str, set[str]] = {
    "phenotypic._services.runs": {"phenotypic.gui.shell._classifier"},
    # TEMPORARY — tracked for removal, not a precedent.
    # tune_spec's setup-authoring half calls resolve_metadata_csv, a five-line
    # compatibility wrapper over resolve_metadata_csv_state in a 596-line
    # browser-payload resolver that transitively reaches
    # gui.shell._source_context -> ._classifier. The other three upward reaches
    # in that half were promoted instead (grid_feasibility -> _services.tune_spec,
    # sandbox_fingerprint -> _services.sandbox, tune_presets_dir ->
    # sdk_._io_constants); this one is not a promotion, it is a design decision
    # about inverting the payload dependency, and it belongs to a phase with room
    # to make it. EXPIRES when that phase promotes or inverts the resolver.
    "phenotypic._services.tune_spec": {
        "phenotypic.gui.shell._metadata_context"
    },
}


@pytest.mark.parametrize("module", _service_modules())
def test_service_module_does_not_import_gui(module: str) -> None:
    """No ``_services`` module may import ``phenotypic.gui`` off-allowlist.

    Parsed imports, never a source substring, and the imported *names* are
    collected alongside the module they are pulled from — ``_boundary`` owns
    the walk and documents why both choices are load-bearing.
    """

    reached = gui_modules_reached(module)
    # An allowlist entry names a *module*, so it covers the names imported from
    # it: `from x._classifier import classify` reaches both "x._classifier" and
    # "x._classifier.classify", and listing every symbol separately would make
    # the allowlist grow on changes that reach nothing new.
    allowed = GUI_IMPORT_ALLOWLIST.get(module, set())
    offenders = sorted(
        name
        for name in reached
        if not any(
            name == entry or name.startswith(f"{entry}.") for entry in allowed
        )
    )
    assert not offenders, (
        f"{module} imports {offenders} from phenotypic.gui; "
        "promote the dependency or add an explicit allowlist entry explaining why"
    )


@pytest.mark.parametrize("module", sorted(GUI_IMPORT_ALLOWLIST))
def test_allowlist_entry_matches_what_the_module_actually_reaches(module: str) -> None:
    """Every allowlist entry is pinned by EQUALITY, not subset.

    The gate above is subset-only: it catches a module reaching something
    un-allowlisted, but not an entry that has been *widened*, gone stale, or
    grown. Without this, editing one line of ``GUI_IMPORT_ALLOWLIST`` — say to
    ``{"phenotypic.gui"}`` — dissolves the boundary for that module and nothing
    fails. An allowlist nobody checks is a comment.

    Equality also makes a stale entry fail: if the import it excused is removed,
    the entry must be removed with it, so the allowlist can only shrink toward
    zero rather than accumulating dead permissions.
    """
    reached = gui_modules_reached(module)
    allowed = GUI_IMPORT_ALLOWLIST[module]

    covered = {
        name
        for name in reached
        if any(name == entry or name.startswith(f"{entry}.") for entry in allowed)
    }
    assert covered == reached, f"{module} reaches un-allowlisted {sorted(reached - covered)}"

    # Each entry must ITSELF be a name the module reaches. This is what forbids
    # a widened entry: allowlisting "phenotypic.gui" would satisfy the coverage
    # check above while dissolving the boundary entirely, but "phenotypic.gui"
    # is not itself imported — only "phenotypic.gui.shell._classifier" is. It
    # also kills a stale entry, whose excused import no longer exists.
    assert allowed <= reached, (
        f"{module}'s allowlist entries {sorted(allowed - reached)} are not "
        "imports this module actually makes. Either the entry is broader than "
        "the real dependency (which would excuse everything beneath it), or it "
        "is stale and the import it covered is gone. Every entry is TEMPORARY "
        "and must name exactly what it excuses."
    )
