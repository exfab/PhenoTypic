"""Every declared MeasurementInfo image resolves to a packaged file."""

from pathlib import Path

import phenotypic
import phenotypic.tools_.constants_  # noqa: F401  (GAMMA_ENCODINGS, PIPE_STATUS)
from phenotypic.schema import MeasurementInfo

_ASSETS = Path(phenotypic.__file__).resolve().parent / "_assets" / "measurements"


def _all_members():
    seen = set()
    stack = list(MeasurementInfo.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        # first-party enums only — excludes test/doctest-defined subclasses whose
        # placeholder image paths (e.g. test_rst_rendering._WithImage) point at
        # files that need not exist on disk.
        if cls.__module__.startswith("phenotypic"):
            yield from list(cls)


def test_declared_images_exist():
    missing = [
        m.image
        for m in _all_members()
        if m.image and not (_ASSETS / m.image).is_file()
    ]
    assert not missing, f"declared images with no file: {missing}"
