"""Every public schema renders a README table.

``READMEGenerator._generate_measurement_table`` wraps its body in a broad
``except Exception`` that returns ``""``. A missed rename inside it would
silently drop every measurement table from every run's README, so this guard
asserts each public, member-ful schema renders a non-empty table headed by its
metric family.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import phenotypic.schema as schema
from phenotypic import ImagePipeline
from phenotypic._cli._cli_readme_generator import READMEGenerator
from phenotypic.schema import MeasurementInfo


def _public_info_classes() -> list[type[MeasurementInfo]]:
    seen: list[type[MeasurementInfo]] = []
    for name in schema.__all__:
        value = getattr(schema, name, None)
        if (
            isinstance(value, type)
            and issubclass(value, MeasurementInfo)
            and value is not MeasurementInfo
            and list(value)
            and value not in seen
        ):
            seen.append(value)
    return seen


@pytest.mark.parametrize("info_cls", _public_info_classes(), ids=lambda c: c.__name__)
def test_every_public_schema_renders_a_readme_table(info_cls: type[MeasurementInfo]) -> None:
    generator = READMEGenerator(config=SimpleNamespace(), pipeline=ImagePipeline())
    table = generator._generate_measurement_table(info_cls)
    assert table.startswith(f"\n### {info_cls.metric_family()}\n")
