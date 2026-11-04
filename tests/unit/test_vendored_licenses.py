"""Every license vendored under ``src/`` ships in the wheel and is in NOTICE.

Third-party code copied into the package (OpenSeadragon under the browse and
results-viewer ``_assets``) carries its ``LICENSE`` beside it, but
package-data does not ship those files, and a permissive license such as
BSD-3-Clause still requires its notice in binary redistributions. The repo's
convention for vendored code (see the Viv bundle) is a copy of the text in
``licenses/`` -- which ``[project] license-files`` puts in every wheel and
sdist -- and a ``NOTICE`` entry pointing at it.

The vendored licenses are found from ``git ls-files`` (any ``LICENSE*`` or
``COPYING*`` under ``src/``), so a newly vendored library with its own
license file fails here until it is wired up. Texts are compared after
normalising line endings and trailing whitespace, since vendored copies are
stored byte-for-byte and one OpenSeadragon copy is CRLF.
``tests/integration/packaging/test_package_contents.py`` checks the built
wheel itself.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: A vendored license known to exist, so an empty scan cannot pass.
KNOWN_VENDORED_LICENSE = (
    "src/phenotypic/_gui/browse/_assets/openseadragon/LICENSE"
)


def _normalized_license_text(path: Path) -> str:
    text = path.read_bytes().decode("utf-8")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def _vendored_license_files() -> list[Path]:
    """``LICENSE*``/``COPYING*`` files under ``src/``; each marks vendored code."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "src"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout:
        paths = [
            REPO / name for name in result.stdout.decode().split("\0") if name
        ]
    else:
        paths = [path for path in (REPO / "src").rglob("*") if path.is_file()]
    return sorted(
        path
        for path in paths
        if path.name.upper().startswith(("LICENSE", "COPYING"))
    )


def _shipped_license_files() -> list[Path]:
    """Files the ``[project] license-files`` globs put in every distribution."""
    pyproject = tomllib.loads(
        (REPO / "pyproject.toml").read_text(encoding="utf-8")
    )
    patterns = pyproject["project"]["license-files"]
    return sorted(
        {
            path
            for pattern in patterns
            for path in REPO.glob(pattern)
            if path.is_file()
        }
    )


def test_every_vendored_license_ships_and_is_named_in_notice() -> None:
    vendored = _vendored_license_files()
    rels = [path.relative_to(REPO).as_posix() for path in vendored]
    assert KNOWN_VENDORED_LICENSE in rels, f"scan is mis-wired: {rels}"

    shipped = {
        _normalized_license_text(path): path
        for path in _shipped_license_files()
    }
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    problems = []
    for path, rel in zip(vendored, rels):
        match = shipped.get(_normalized_license_text(path))
        if match is None:
            problems.append(
                f"{rel}: its text is in no file matched by [project] "
                "license-files; copy it to licenses/<name>-<license>.txt"
            )
        elif f"licenses/{match.name}" not in notice:
            problems.append(
                f"{rel}: NOTICE does not name licenses/{match.name}"
            )
    assert not problems, problems
