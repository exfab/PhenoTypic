"""Top-level test configuration.

Ensures that calling ``.show()`` on plotly or matplotlib figures during tests
does not spawn browser tabs or GUI windows; loads a repo-root ``.env`` (per-user,
gitignored — keeps the suite user-agnostic, e.g. ``PHENOTYPIC_TEST_PG_URL``); and
autoskips ``@pytest.mark.postgres`` / ``@pytest.mark.slurm`` tests unless a live
Postgres URL / the SLURM client is available.
"""

import os
import shutil
from pathlib import Path

import pytest

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


def pytest_collection_modifyitems(config, items):
    """Autoskip ``postgres`` tests without a DB URL and ``slurm`` tests without sbatch.

    ``@pytest.mark.postgres`` tests skip unless ``$PHENOTYPIC_TEST_PG_URL`` is set
    (via the environment or ``.env``); ``@pytest.mark.slurm`` tests skip unless the
    SLURM client (``sbatch``) is on ``PATH`` — so CI and slurm-less local runs
    never fail on either.

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


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_auto_num_workers(config) -> int:
    """Use SLURM-allocated CPUs when available, else fall back to affinity mask.

    ``optionalhook=True`` keeps this conftest valid when pytest-xdist is not
    installed (e.g. the packaging-integrity CI job's bare ``pytest``-only env),
    where pluggy would otherwise reject the unknown ``pytest_xdist_*`` hook.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        return int(slurm_cpus)
    # Affinity mask respects cgroups/containers; cpu_count() does not
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # sched_getaffinity not available on macOS/Windows
        return os.cpu_count() or 1
