"""Module entry point for the unified PhenoTypic GUI shell.

Boots the Dash hub via :func:`phenotypic.gui.shell._launcher.main`. Two ways
to invoke it:

    * ``python -m phenotypic.gui --root /path/to/sandbox``
    * ``phenotypic-gui --root /path/to/sandbox`` (console script wired in
      Phase 7 via ``[project.scripts]``).

Both paths land in the same launcher; the module-vs-script form is
preserved by the launcher's ``main(argv=None)`` signature.

Note:
    The hyphenated ``phenotypic-gui`` is the only supported entry. There
    is no ``phenotypic gui`` (without a hyphen) subcommand on the existing
    CLI — see ``GUI_SPEC_V1.md`` Section 7 for the design decision.
"""
from __future__ import annotations

from phenotypic.gui.shell._launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
