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


def _license_problems(
    vendored: list[Path], shipped: list[Path], notice: str, root: Path
) -> list[str]:
    """Why each vendored license is not shipped and named in NOTICE.

    Identical texts are common (PhenoTypic's own ``LICENSE`` and several
    ``licenses/*-Apache-2.0.txt`` are byte-identical), so a text maps to
    every shipped file carrying it, and a vendored license passes when any
    of those under ``licenses/`` is named in NOTICE.
    """
    by_text: dict[str, list[Path]] = {}
    for path in shipped:
        by_text.setdefault(_normalized_license_text(path), []).append(path)
    problems = []
    for path in vendored:
        rel = path.relative_to(root).as_posix()
        in_licenses_dir = [
            match.relative_to(root).as_posix()
            for match in by_text.get(_normalized_license_text(path), [])
            if match.parent == root / "licenses"
        ]
        if not in_licenses_dir:
            problems.append(
                f"{rel}: its text is in no licenses/ file matched by "
                "[project] license-files; copy it to "
                "licenses/<name>-<license>.txt"
            )
        elif not any(name in notice for name in in_licenses_dir):
            problems.append(
                f"{rel}: NOTICE names none of {sorted(in_licenses_dir)}"
            )
    return problems


def test_every_vendored_license_ships_and_is_named_in_notice() -> None:
    vendored = _vendored_license_files()
    rels = [path.relative_to(REPO).as_posix() for path in vendored]
    assert KNOWN_VENDORED_LICENSE in rels, f"scan is mis-wired: {rels}"

    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    problems = _license_problems(
        vendored, _shipped_license_files(), notice, REPO
    )
    assert not problems, problems


def test_identical_license_texts_do_not_hide_the_notice_entry(
    tmp_path,
) -> None:
    """A text shared by the root LICENSE and a licenses/ file still passes.

    Covers both orders, since the old one-path-per-text map kept whichever
    file came last and then looked for ``licenses/LICENSE`` in NOTICE.
    """
    apache = "Apache License\nVersion 2.0\n"
    vendored = tmp_path / "src" / "vendor" / "LICENSE"
    vendored.parent.mkdir(parents=True)
    vendored.write_text(apache.replace("\n", "\r\n"), encoding="utf-8")
    (tmp_path / "licenses").mkdir()
    own = tmp_path / "LICENSE"
    third_party = tmp_path / "licenses" / "vendor-Apache-2.0.txt"
    for path in (own, third_party):
        path.write_text(apache, encoding="utf-8")
    notice = "License:   licenses/vendor-Apache-2.0.txt\n"

    for shipped in ([own, third_party], [third_party, own]):
        assert _license_problems([vendored], shipped, notice, tmp_path) == []
    assert _license_problems([vendored], [own], notice, tmp_path) == [
        "src/vendor/LICENSE: its text is in no licenses/ file matched by "
        "[project] license-files; copy it to licenses/<name>-<license>.txt"
    ]
