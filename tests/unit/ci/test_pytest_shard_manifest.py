"""Coverage checks for the pull-request pytest shard manifest."""

from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import re
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / ".github" / "pytest-shards.json"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "run-pytest.yml"

#: pytest-playwright fixtures that need an installed browser. ``context`` is
#: deliberately absent: two unrelated test modules define a ``context`` fixture
#: of their own, and the bare name would flag them.
PLAYWRIGHT_FIXTURES = frozenset(
    {"page", "browser", "browser_name", "browser_type", "launch_browser", "new_context", "playwright"}
)

#: A browser test module the scan must find, so an empty scan cannot pass.
KNOWN_BROWSER_MODULE = Path("tests/gui/results_viewer/test_splitter_browser.py")


def _is_test_file(path: Path) -> bool:
    """Return whether pytest considers the path a test module by name."""
    return path.name.startswith("test_") or path.name.endswith("_test.py")


def _configured_test_files() -> set[Path]:
    """Return every test module below pytest's configured test roots."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    testpaths = config["tool"]["pytest"]["ini_options"]["testpaths"]
    return {
        path.relative_to(REPO_ROOT)
        for root in testpaths
        for path in (REPO_ROOT / root).rglob("*.py")
        if _is_test_file(path)
    }


def _shards_by_module() -> dict[Path, list[dict]]:
    """Map every test module a manifest entry expands to onto the shards owning it."""
    shards = json.loads(MANIFEST.read_text(encoding="utf-8"))
    owners: dict[Path, list[dict]] = {}

    for shard in shards:
        for entry in shard["paths"]:
            matches = list(REPO_ROOT.glob(entry))
            if not matches:
                raise AssertionError(f"shard path does not exist: {entry}")
            for match in matches:
                if match.is_dir() and not any(match.rglob("*.py")):
                    raise AssertionError(f"shard path has no source files: {entry}")
                candidates = match.rglob("*.py") if match.is_dir() else (match,)
                for path in candidates:
                    if _is_test_file(path):
                        owners.setdefault(path.relative_to(REPO_ROOT), []).append(shard)

    return owners


def _manifest_assignments() -> Counter[Path]:
    """Expand every manifest entry into its assigned test modules."""
    return Counter({path: len(shards) for path, shards in _shards_by_module().items()})


def _requests_a_browser(path: Path) -> bool:
    """Return whether a test module needs an installed Playwright browser."""
    tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parameters = {arg.arg for arg in [*node.args.args, *node.args.kwonlyargs]}
            if parameters & PLAYWRIGHT_FIXTURES:
                return True
        elif isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "playwright" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "playwright":
                return True
    return False


def test_each_configured_test_file_belongs_to_exactly_one_shard() -> None:
    """Prevent tests from being silently omitted or run more than once."""
    configured = _configured_test_files()
    assignments = _manifest_assignments()

    assert set(assignments) == configured
    assert all(count == 1 for count in assignments.values())


def test_pr_workflow_uses_complete_shards_without_testmon() -> None:
    """Keep the PR lane deterministic and independent of selection history."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "fromJSON(needs.check-manual-run.outputs.shards)" in workflow
    assert "join(matrix.shard.paths, ' ')" in workflow
    assert "-n auto" in workflow
    assert "--testmon" not in workflow
    assert ".testmondata" not in workflow


def test_browser_tests_run_only_in_shards_that_install_a_browser() -> None:
    """A browser test in a shard without Chromium errors with a missing executable."""
    owners = _shards_by_module()
    browser_modules = {path for path in owners if _requests_a_browser(path)}

    assert KNOWN_BROWSER_MODULE in browser_modules
    misplaced = sorted(
        str(path)
        for path in browser_modules
        if not all(shard.get("playwright") for shard in owners[path])
    )
    assert misplaced == []


def test_pr_workflow_installs_chromium_for_browser_shards() -> None:
    """The browser install step exists and is gated on the shard's playwright flag."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert re.search(
        r"if: matrix\.shard\.playwright\s*\n\s*run: uv run playwright install --with-deps chromium",
        workflow,
    )
