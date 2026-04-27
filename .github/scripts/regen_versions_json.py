"""Regenerate versions.json + root index.html for the versioned docs site.

Scans a gh-pages checkout for `experimental/` and `v*/` subdirectories whose
names parse as PEP 440 versions, then writes:

- ``<root>/versions.json`` — the switcher feed consumed by pydata-sphinx-theme.
- ``<root>/index.html`` — meta-refresh redirect to the preferred stable
  version (falling back to experimental, then to a placeholder).
- ``<root>/.nojekyll`` — prevents GitHub Pages from running Jekyll on the
  generated HTML.

Usage:
    python regen_versions_json.py <gh_pages_root>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

BASE_URL = "https://exfab.github.io/PhenoTypic"
EXPERIMENTAL_DIR = "experimental"
IGNORED_NAMES = {".git", ".github", ".nojekyll", "index.html", "versions.json"}


def _parse_tag(name: str) -> Version | None:
    """Return the parsed Version for a ``v*`` dir name, or None if unparseable."""
    if not name.startswith("v"):
        return None
    try:
        return Version(name[1:])
    except InvalidVersion:
        return None


def _scan(root: Path) -> tuple[bool, list[tuple[str, Version]]]:
    """Return (has_experimental, sorted_tags) for the gh-pages root.

    ``sorted_tags`` is a list of ``(dir_name, parsed_version)`` pairs, newest
    first. The original dir name is preserved so URLs and JSON entries match
    what's actually on disk (e.g. ``v0.14.0b5`` not a normalized form).
    """
    has_experimental = False
    tags: list[tuple[str, Version]] = []
    for entry in root.iterdir():
        if not entry.is_dir() or entry.name in IGNORED_NAMES:
            continue
        if entry.name == EXPERIMENTAL_DIR:
            has_experimental = True
            continue
        parsed = _parse_tag(entry.name)
        if parsed is not None:
            tags.append((entry.name, parsed))
    tags.sort(key=lambda pair: pair[1], reverse=True)
    return has_experimental, tags


def _build_entries(
    has_experimental: bool,
    sorted_tags: list[tuple[str, Version]],
) -> tuple[list[dict], str | None]:
    """Build the versions.json payload and return (entries, preferred_dir)."""
    preferred_dir: str | None = next(
        (name for name, ver in sorted_tags if not ver.is_prerelease),
        None,
    )

    entries: list[dict] = []
    if has_experimental:
        entries.append(
            {
                "name": "experimental (main)",
                "version": "dev",
                "url": f"{BASE_URL}/{EXPERIMENTAL_DIR}/",
            }
        )
    for name, _ in sorted_tags:
        display = f"{name} (stable)" if name == preferred_dir else name
        entry: dict = {
            "name": display,
            "version": name,
            "url": f"{BASE_URL}/{name}/",
        }
        if name == preferred_dir:
            entry["preferred"] = True
        entries.append(entry)
    return entries, preferred_dir


def _render_index_html(preferred_dir: str | None, has_experimental: bool) -> str:
    """Build the root index.html content for the Pages landing page."""
    if preferred_dir is not None:
        target = f"./{preferred_dir}/"
        label = preferred_dir
    elif has_experimental:
        target = f"./{EXPERIMENTAL_DIR}/"
        label = "experimental"
    else:
        return (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<title>PhenoTypic docs</title></head>"
            "<body><p>No documentation has been published yet.</p></body></html>"
        )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<meta http-equiv=\"refresh\" content=\"0; url={target}\">"
        f"<title>PhenoTypic docs</title></head>"
        f"<body><p>Redirecting to <a href=\"{target}\">{label}</a>.</p></body></html>"
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <gh_pages_root>", file=sys.stderr)
        return 2
    root = Path(argv[1]).resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    has_experimental, sorted_tags = _scan(root)
    entries, preferred_dir = _build_entries(has_experimental, sorted_tags)

    versions_payload = json.dumps(entries, indent=2) + "\n"
    (root / "versions.json").write_text(versions_payload)
    (root / "index.html").write_text(_render_index_html(preferred_dir, has_experimental))
    (root / ".nojekyll").touch()

    print(versions_payload, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
