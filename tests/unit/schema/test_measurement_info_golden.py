"""The member rewrite must not change any (name, value, label, desc)."""

import json
from pathlib import Path

import phenotypic  # noqa: F401
import phenotypic.sdk_.constants_  # noqa: F401
from phenotypic.schema import MeasurementInfo

_GOLDEN = Path(__file__).parent / "_golden" / "measurement_info_values.json"


def _snapshot() -> dict[str, list[list[str | None]]]:
    seen, snap = set(), {}
    stack = list(MeasurementInfo.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        if not cls.__module__.startswith("phenotypic"):
            continue  # exclude test/doctest-defined subclasses
        members = list(cls)
        if members:
            snap[f"{cls.__module__}.{cls.__name__}"] = [
                [m.name, m.value, m.label, m.desc] for m in members
            ]
    return dict(sorted(snap.items()))


def test_member_values_match_golden():
    assert _GOLDEN.exists(), "golden snapshot missing — generate it first"
    expected = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    assert _snapshot() == expected
