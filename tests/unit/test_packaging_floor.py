"""Guards on the declared dependency universe for the OME-Zarr store.

These are packaging assertions, not behaviour tests: they fail loudly if a
future edit reintroduces Python 3.10, adopts an ome-zarr package, or lets the
NGFF conformance dependency drift back to transitive-only.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_requires_python_floor_is_311(pyproject: dict) -> None:
    assert pyproject["project"]["requires-python"] == ">=3.11, <3.13"


def test_classifiers_drop_310(pyproject: dict) -> None:
    classifiers = pyproject["project"]["classifiers"]
    assert "Programming Language :: Python :: 3.10" not in classifiers
    assert "Programming Language :: Python :: 3.11" in classifiers
    assert "Programming Language :: Python :: 3.12" in classifiers


def test_zarr_is_a_runtime_dependency(pyproject: dict) -> None:
    deps = pyproject["project"]["dependencies"]
    assert any(dep.startswith("zarr") for dep in deps), deps


def test_h5py_is_retained_for_migration(pyproject: dict) -> None:
    deps = pyproject["project"]["dependencies"]
    assert any(dep.split(">")[0].split("=")[0].strip() == "h5py" for dep in deps)


def test_ome_zarr_packages_are_not_adopted_anywhere(pyproject: dict) -> None:
    """`ome-zarr-models` pins pydantic<2.13; uv resolves one universe."""
    banned = {"ome-zarr", "ome-zarr-models"}
    pools: list[list[str]] = [list(pyproject["project"]["dependencies"])]
    for group in pyproject.get("dependency-groups", {}).values():
        pools.append([item for item in group if isinstance(item, str)])
    for extra in pyproject["project"].get("optional-dependencies", {}).values():
        pools.append(list(extra))
    for pool in pools:
        for requirement in pool:
            name = (
                requirement.split(";")[0]
                .split("[")[0]
                .split(">")[0]
                .split("<")[0]
                .split("=")[0]
                .strip()
                .lower()
            )
            assert name not in banned, requirement


def _declared_group_requirements(pyproject: dict) -> dict[str, str]:
    """Map ``name -> full requirement string`` across every dependency group."""
    return {
        requirement.split(";")[0]
        .split(">")[0]
        .split("<")[0]
        .split("=")[0]
        .strip()
        .lower(): requirement
        for group in pyproject.get("dependency-groups", {}).values()
        for requirement in group
        if isinstance(requirement, str)
    }


@pytest.mark.parametrize("package", ["jsonschema", "xmlschema", "referencing"])
def test_conformance_deps_are_declared_not_transitive(
    pyproject: dict, package: str
) -> None:
    """Spec §7: a conformance check may never skip on a missing dependency.

    Parametrized, not three functions: all three gates fail the same way
    (green locally, red in CI) and a new conformance dependency should be one
    list entry, not a copied test. Ledger GEN-24.

    ``referencing`` is here because the harness imports it directly to build
    the ``Registry`` that resolves the vendored ``_version.schema``; reaching
    it only through jsonschema is the same transitive-only hazard.
    """
    assert package in _declared_group_requirements(pyproject)


def test_jsonschema_floor_admits_the_registry_keyword() -> None:
    """A presence-only check would not catch a 4.17-permitting floor.

    ``Draft202012Validator(schema, registry=...)`` is how the conformance
    harness resolves the vendored ``_version.schema`` without network access.
    The ``registry`` keyword arrived in jsonschema 4.18; on 4.17 the call is a
    ``TypeError``, so declaring ``>=4.0`` would let CI resolve a version that
    cannot run the gate at all.
    """
    pyproject_doc = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    requirement = _declared_group_requirements(pyproject_doc)["jsonschema"]
    match = re.search(r">=\s*(\d+)\.(\d+)", requirement)
    assert match is not None, requirement
    assert (int(match.group(1)), int(match.group(2))) >= (4, 18), requirement


def test_conformance_stack_imports_and_registry_keyword_is_real() -> None:
    """The declared floor is only useful if it matches the installed reality."""
    import inspect

    import jsonschema
    import referencing  # noqa: F401
    import xmlschema  # noqa: F401

    signature = inspect.signature(jsonschema.Draft202012Validator)
    assert "registry" in signature.parameters, signature


def test_zarr_v3_is_importable_at_runtime() -> None:
    import zarr

    assert zarr.__version__.startswith("3."), zarr.__version__
