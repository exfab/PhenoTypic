"""Shell launch helper + ``phenotypic-gui`` console-script entry (Phase 3).

Phase 0 placeholder — implementation lands in Phase 3. See ``GUI_SPEC_V1.md``
section 7 (Entry points). The ``main`` symbol below is referenced by the
``[project.scripts]`` table in ``pyproject.toml``; once wired it must accept
no arguments and return an int exit code.
"""
from __future__ import annotations

import sys


def main() -> int:
    """Console-script stub — replaced in Phase 3.

    Wired into ``[project.scripts]`` so ``uv run phenotypic-gui`` resolves to
    a real callable. Until Phase 3 lands, it exits with a clear error and a
    pointer to the spec.
    """
    print(
        "phenotypic-gui is scaffolded but not yet wired (Phase 3 of GUI_SPEC_V1.md).",
        file=sys.stderr,
    )
    return 1


# TODO(Phase 3): implement ``launch_gui(*, root, port, host, ...)`` and rebuild
# ``main`` to argparse-parse + delegate.

__all__ = ["main"]
