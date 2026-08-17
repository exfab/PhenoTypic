"""The boundary that makes `_services` a layer rather than a folder."""

from __future__ import annotations

import pkgutil
import subprocess
import sys

import pytest

FORBIDDEN = ("dash", "dash_bootstrap_components", "flask", "werkzeug")

# One subprocess per module: a single process would let module A's clean import
# be vouched for by module B having already been imported, and vice versa.
_PROBE = """
import importlib, sys
importlib.import_module({module!r})
leaked = sorted(m for m in {forbidden!r} if m in sys.modules)
print(",".join(leaked))
"""


def _service_modules() -> list[str]:
    import phenotypic._services as services

    return [
        f"phenotypic._services.{m.name}"
        for m in pkgutil.iter_modules(services.__path__)
    ]


def test_services_package_exists_and_is_lazy():
    import phenotypic._services as services

    assert services.__path__, "phenotypic._services must be a package"


@pytest.mark.parametrize("module", _service_modules())
def test_service_module_imports_no_dash(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module, forbidden=FORBIDDEN)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"{module} failed to import:\n{proc.stderr}"
    leaked = [name for name in proc.stdout.strip().split(",") if name]
    assert not leaked, f"{module} dragged {leaked} into sys.modules"
