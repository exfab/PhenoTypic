"""Bundled static assets (logos + dashboard JS) for PhenoTypic.

Canonical home for every image/JS file the runtime or docs build ships.
Resolve assets through this module rather than re-spelling the package path.

Layout::

    _assets/
      logos/                 # logo/brand images (Flask static root for the GUI)
        LogoArtOnly.png      #   CLI dashboard logo
        dashboard_logo.svg   #   GUI logo
        200x150/ 400x150/ 500x500/ 500x500_png/   # brand variant library
      vendor/                # third-party JS bundled by the CLI dashboard
        plotly.min.js
        hyparquet.min.js
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

#: Filesystem path to this package's asset root. ``__init__.py`` lives in the
#: asset dir, so ``__file__``-relative resolution matches the existing on-disk
#: install pattern in :mod:`phenotypic.gui._shared._blueprint`. Wheels install
#: unzipped, so this is a real directory in both editable and built installs.
ASSET_DIR: Path = Path(__file__).resolve().parent


def logos_dir() -> Path:
    """Return the directory holding logo/brand images.

    Used as the Flask static root for the shared GUI logo blueprint.
    """
    return ASSET_DIR / "logos"


#: Filenames of the bundled curated colony exemplar (rendered once from
#: ``load_synth_yeast_plate()``): a reference RGB crop + its 0/255 mask. These
#: are the DEFAULT reference/support for the few-shot semantic detectors
#: (``Insid3Detector``, ``FssDinoDetector``).
COLONY_EXEMPLAR_RGB = "colony_reference_rgb.png"
COLONY_EXEMPLAR_MASK = "colony_reference_mask.png"


def colony_exemplar_paths() -> tuple[Path, Path]:
    """Return the bundled curated colony exemplar ``(rgb_path, mask_path)``.

    The pair is a small reference colony patch and its ground-truth mask,
    rendered once from :func:`phenotypic.data.load_synth_yeast_plate`. The
    few-shot semantic detectors use it as their default
    ``reference_image``/``reference_mask`` (INSID3) and ``support_*`` (FSSDINO),
    so they have a working out-of-the-box exemplar.

    Returns:
        ``(rgb_path, mask_path)`` — absolute paths to the bundled PNGs.
    """
    exemplars = ASSET_DIR / "exemplars"
    return exemplars / COLONY_EXEMPLAR_RGB, exemplars / COLONY_EXEMPLAR_MASK


def asset_bytes(relpath: str) -> bytes:
    """Read an asset's raw bytes by POSIX-style relative path.

    Args:
        relpath: Path relative to the asset root, e.g.
            ``"logos/LogoArtOnly.png"`` or ``"vendor/plotly.min.js"``.

    Returns:
        The file's bytes.

    Raises:
        ValueError: If ``relpath`` is empty (which would otherwise resolve
            to the package directory and raise an opaque read error).
    """
    if not relpath:
        raise ValueError("relpath must be a non-empty asset path")
    return files(__name__).joinpath(*relpath.split("/")).read_bytes()
