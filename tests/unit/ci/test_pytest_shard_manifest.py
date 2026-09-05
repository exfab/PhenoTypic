"""Coverage checks for the pull-request pytest shard manifest."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / ".github" / "pytest-shards.json"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "run-pytest.yml"


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


def _manifest_assignments() -> Counter[Path]:
    """Expand every manifest entry into its assigned test modules."""
    shards = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assignments: Counter[Path] = Counter()

    for shard in shards:
        for entry in shard["paths"]:
            matches = list(REPO_ROOT.glob(entry))
            if not matches:
                raise AssertionError(f"shard path does not exist: {entry}")
            for match in matches:
                if match.is_dir() and not any(match.rglob("*.py")):
                    raise AssertionError(f"shard path has no source files: {entry}")
                candidates = match.rglob("*.py") if match.is_dir() else (match,)
                assignments.update(
                    path.relative_to(REPO_ROOT)
                    for path in candidates
                    if _is_test_file(path)
                )

    return assignments


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
