"""Every MeasurementInfo member exposes the universal Entry attribute surface."""

import phenotypic  # noqa: F401  (registers all enum modules)
import phenotypic.tools_.constants_  # noqa: F401  (GAMMA_ENCODINGS, PIPE_STATUS)
from phenotypic.schema import MeasurementInfo


def _all_concrete_info_classes():
    seen, out = set(), []
    stack = list(MeasurementInfo.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        # first-party enums only — excludes test/doctest-defined subclasses that
        # linger in __subclasses__() within a pytest session
        if cls.__module__.startswith("phenotypic") and len(list(cls)) > 0:
            out.append(cls)
    return out


def test_every_member_has_entry_attribute_surface():
    classes = _all_concrete_info_classes()
    assert classes, "no concrete MeasurementInfo subclasses discovered"
    for cls in classes:
        for member in cls:
            assert isinstance(member.label, str) and member.label
            assert isinstance(member.desc, str)
            assert isinstance(member.bio_desc, str)
            assert member.image is None or isinstance(member.image, str)
            assert member.pair == (member.label, member.desc)


def test_discovery_covers_known_enums():
    names = {c.__name__ for c in _all_concrete_info_classes()}
    assert {"SHAPE", "SIZE", "METADATA", "GAMMA_ENCODINGS", "PIPE_STATUS"} <= names
