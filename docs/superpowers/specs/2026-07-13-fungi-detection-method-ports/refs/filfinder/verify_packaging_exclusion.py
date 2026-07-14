"""Build release artifacts and prove A10 reference material is excluded."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


REFERENCE_DIR = pathlib.Path(__file__).resolve().parent
REPOSITORY_ROOT = REFERENCE_DIR.parents[5]
FORBIDDEN_FRAGMENTS = (
    "docs/superpowers",
    "refs/filfinder",
    "upstream/fil_finder",
    "tests/fixtures/reconnect/filfinder",
)


def artifact_members(path: pathlib.Path) -> list[str]:
    """List normalized members of a wheel or gzipped source distribution."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, mode="r:gz") as archive:
            return archive.getnames()
    raise ValueError(f"unsupported artifact: {path}")


def verify_packaging_exclusion() -> None:
    """Build wheel and sdist, then reject every forbidden evidence member."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to build release artifacts")
    with tempfile.TemporaryDirectory(prefix="phenotypic-a10-package-") as temporary:
        output = pathlib.Path(temporary)
        completed = subprocess.run(
            [uv, "build", "--out-dir", str(output)],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"uv build failed:\n{completed.stderr[-4000:]}")
        artifacts = sorted(
            path
            for path in output.iterdir()
            if path.suffix == ".whl" or path.name.endswith(".tar.gz")
        )
        if len([path for path in artifacts if path.suffix == ".whl"]) != 1:
            raise AssertionError("build did not produce exactly one wheel")
        if len([path for path in artifacts if path.name.endswith(".tar.gz")]) != 1:
            raise AssertionError("build did not produce exactly one sdist")
        for artifact in artifacts:
            leaked = [
                member
                for member in artifact_members(artifact)
                if any(fragment in member for fragment in FORBIDDEN_FRAGMENTS)
            ]
            if leaked:
                raise AssertionError(
                    f"A10 reference material leaked into {artifact.name}: {leaked}"
                )


if __name__ == "__main__":
    try:
        verify_packaging_exclusion()
    except Exception as error:  # pragma: no cover - command-line failure report
        print(f"A10 packaging exclusion FAILED: {error}", file=sys.stderr)
        raise
    print("A10 packaging exclusion PASSED")
