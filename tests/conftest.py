"""Top-level test configuration.

Ensures that calling ``.show()`` on plotly or matplotlib figures during tests
does not spawn browser tabs or GUI windows; loads a repo-root ``.env`` (per-user,
gitignored — keeps the suite user-agnostic, e.g. ``PHENOTYPIC_TEST_PG_URL``); and
autoskips ``@pytest.mark.postgres`` / ``@pytest.mark.slurm`` tests unless a live
Postgres URL / the SLURM client is available.
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

from tests._support.xdist_workers import resolve_xdist_auto_workers

# matplotlib/plotly are guarded so this conftest stays importable in
# dependency-light pytest invocations (e.g. the packaging-integrity CI job,
# which runs the build-artifact tests in a bare ``pytest``-only env). When the
# plotting stack is absent there are no figures to redirect, so skipping the
# headless-backend setup is a safe no-op.
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass

try:
    import plotly.io as pio

    pio.renderers.default = "json"
except ImportError:
    pass


def _load_dotenv() -> None:
    """Load ``<repo-root>/.env`` into ``os.environ`` (no override; dep-free).

    A per-user, gitignored file (see ``.env.example``) so each developer supplies
    their own ``PHENOTYPIC_TEST_PG_URL`` without committing a DB address — the
    test suite stays user-agnostic. Existing environment variables win
    (``setdefault``), so an explicit ``export`` still overrides the file. A
    missing file is a no-op.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

#: Env var carrying a live Postgres URL for the gated tune study-DB tests. When
#: unset (after loading ``.env``), every ``@pytest.mark.postgres`` test is skipped
#: so the default suite needs no database.
PG_URL_ENV = "PHENOTYPIC_TEST_PG_URL"

#: Fixtures only pytest-playwright provides. ``pyproject.toml`` excludes the
#: plugin on Windows, where a test requesting one would otherwise ERROR with
#: ``fixture 'page' not found`` instead of skipping.
PLAYWRIGHT_FIXTURES = frozenset(
    {"page", "browser", "browser_name", "browser_type", "launch_browser", "new_context", "playwright"}
)


def _missing_playwright_fixture(item: pytest.Item) -> str | None:
    """Return a requested Playwright fixture that nothing defines, if any.

    Windows only, deliberately. ``pyproject.toml`` drops pytest-playwright
    solely on ``sys_platform == 'win32'``, so everywhere else the plugin is
    always installed and a missing ``page`` means something is genuinely
    broken -- a dependency-resolution regression, a half-built env. Skipping
    that would report green while every browser test silently vanished,
    including the ones behind the required ``e2e-tests`` gate. Let it ERROR.
    """
    if sys.platform != "win32":
        return None
    fixture_info = getattr(item, "_fixtureinfo", None)
    if fixture_info is None:
        return None
    for name in PLAYWRIGHT_FIXTURES.intersection(fixture_info.names_closure):
        if not fixture_info.name2fixturedefs.get(name):
            return name
    return None


def pytest_collection_modifyitems(config, items):
    """Autoskip tests whose external requirement is absent.

    ``@pytest.mark.postgres`` tests skip unless ``$PHENOTYPIC_TEST_PG_URL`` is set
    (via the environment or ``.env``); ``@pytest.mark.slurm`` tests skip unless the
    SLURM client (``sbatch``) is on ``PATH``; and browser tests skip **on
    Windows only**, where ``pyproject.toml`` omits pytest-playwright -- so CI,
    Windows, and slurm-less local runs never fail on any of them. Off Windows a
    missing Playwright fixture still ERRORs, because there it means the env is
    broken rather than unsupported.

    Args:
        config: The pytest config (unused; required by the hook signature).
        items: The collected test items, mutated in place with skip markers.
    """
    skip_pg = (
        None
        if os.environ.get(PG_URL_ENV)
        else pytest.mark.skip(reason=f"requires a Postgres server via ${PG_URL_ENV}")
    )
    skip_slurm = (
        None
        if shutil.which("sbatch")
        else pytest.mark.skip(reason="requires the SLURM client (sbatch) on PATH")
    )
    for item in items:
        if skip_pg is not None and "postgres" in item.keywords:
            item.add_marker(skip_pg)
        if skip_slurm is not None and "slurm" in item.keywords:
            item.add_marker(skip_slurm)
        missing = _missing_playwright_fixture(item)
        if missing is not None:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"requires pytest-playwright (fixture {missing!r})"
                )
            )


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config) -> int:
    """Use SLURM-allocated CPUs when available, else fall back to affinity mask.

    ``optionalhook=True`` keeps this conftest valid when pytest-xdist is not
    installed (e.g. the packaging-integrity CI job's bare ``pytest``-only env),
    where pluggy would otherwise reject the unknown ``pytest_xdist_*`` hook.
    """
    # Affinity mask respects cgroups/containers; cpu_count() does not
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except AttributeError:
        # sched_getaffinity not available on macOS/Windows
        affinity_count = None
    return resolve_xdist_auto_workers(
        os.environ,
        affinity_count=affinity_count,
        cpu_count=os.cpu_count() or 1,
    )
