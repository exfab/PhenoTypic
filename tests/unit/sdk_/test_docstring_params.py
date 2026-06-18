"""Tests for the shared docstring-parameter parsing utilities."""
from __future__ import annotations

from pydantic import BaseModel, Field

from phenotypic.sdk_._docstring_params import (
    apply_docstring_descriptions,
    parse_param_descriptions,
)


class TestParseParamDescriptionsGoogle:
    def test_google_args_block(self) -> None:
        doc = (
            "Detect colonies by Otsu thresholding.\n"
            "\n"
            "Args:\n"
            "    ignore_zeros: Exclude zero-intensity pixels.\n"
            "    ignore_borders: Drop colonies touching the edge.\n"
        )
        result = parse_param_descriptions(doc)
        assert result == {
            "ignore_zeros": "Exclude zero-intensity pixels.",
            "ignore_borders": "Drop colonies touching the edge.",
        }

    def test_google_typed_param_strips_type_annotation(self) -> None:
        doc = (
            "Blur an image.\n"
            "\n"
            "Args:\n"
            "    sigma (float): Standard deviation of the kernel.\n"
        )
        result = parse_param_descriptions(doc)
        assert result == {
            "sigma": "Standard deviation of the kernel."
        }

    def test_google_continuation_lines_joined(self) -> None:
        doc = (
            "Operation.\n"
            "\n"
            "Args:\n"
            "    radius: The radius of the structuring element\n"
            "        used for the morphological pass.\n"
        )
        result = parse_param_descriptions(doc)
        assert result["radius"] == (
            "The radius of the structuring element "
            "used for the morphological pass."
        )

    def test_google_arguments_header_variant(self) -> None:
        doc = (
            "Operation.\n"
            "\n"
            "Arguments:\n"
            "    threshold: Cutoff applied to the detect matrix.\n"
        )
        result = parse_param_descriptions(doc)
        assert result == {
            "threshold": "Cutoff applied to the detect matrix."
        }

    def test_section_ends_at_unindented_prose_line(self) -> None:
        # The end-of-section heuristic fires on an unindented,
        # non-empty line that is not itself a ``name:`` definition.
        # Plain prose after the Args block ends the section so the
        # following text is not captured as a parameter.
        doc = (
            "Operation.\n"
            "\n"
            "Args:\n"
            "    keep: Whether to keep the colony mask.\n"
            "\n"
            "This trailing paragraph is not part of the Args block.\n"
        )
        result = parse_param_descriptions(doc)
        assert result == {"keep": "Whether to keep the colony mask."}


class TestParseParamDescriptionsNumpy:
    def test_numpy_parameters_block_keys_present(self) -> None:
        # NumPy ``name : type`` lines are matched by the Google regex
        # first (the type lands in the description), so the parser
        # prepends the type token to the real description. This quirk
        # is inherited verbatim from the original LazyWidgetMixin
        # parser; the test pins the behaviour rather than an idealized
        # NumPy parse.
        doc = (
            "Detect colonies.\n"
            "\n"
            "Parameters\n"
            "----------\n"
            "ignore_zeros : bool\n"
            "    Exclude zero-intensity pixels from the histogram.\n"
            "ignore_borders : bool\n"
            "    Drop colonies touching the plate edge.\n"
        )
        result = parse_param_descriptions(doc)
        assert set(result) == {"ignore_zeros", "ignore_borders"}
        assert "Exclude zero-intensity pixels" in result[
            "ignore_zeros"
        ]
        assert "Drop colonies touching the plate edge" in result[
            "ignore_borders"
        ]

    def test_numpy_header_without_underline_is_not_a_section(
        self,
    ) -> None:
        # A bare "Parameters" line with no dashed underline must not
        # open a NumPy section (the underline is mandatory).
        doc = (
            "Detect colonies.\n"
            "\n"
            "Parameters\n"
            "ignore_zeros : bool\n"
        )
        assert parse_param_descriptions(doc) == {}


class TestParseParamDescriptionsEdgeCases:
    def test_none_returns_empty_dict(self) -> None:
        assert parse_param_descriptions(None) == {}

    def test_empty_string_returns_empty_dict(self) -> None:
        assert parse_param_descriptions("") == {}

    def test_docstring_without_args_block_returns_empty_dict(
        self,
    ) -> None:
        doc = (
            "Just a summary line describing the operation.\n"
            "\n"
            "No parameter section is present in this docstring.\n"
        )
        assert parse_param_descriptions(doc) == {}


class TestApplyDocstringDescriptions:
    def test_descriptions_reach_model_json_schema(self) -> None:
        class _ThreshOp(BaseModel):
            """Threshold the detect matrix.

            Args:
                cutoff: Intensity cutoff for colony pixels.
                invert: Whether to invert the resulting mask.
            """

            cutoff: float = 0.5
            invert: bool = False

        apply_docstring_descriptions(_ThreshOp)

        schema = _ThreshOp.model_json_schema()
        props = schema["properties"]
        assert (
            props["cutoff"]["description"]
            == "Intensity cutoff for colony pixels."
        )
        assert (
            props["invert"]["description"]
            == "Whether to invert the resulting mask."
        )

    def test_field_description_attribute_populated(self) -> None:
        class _BlurOp(BaseModel):
            """Blur an image.

            Args:
                sigma: Standard deviation of the Gaussian kernel.
            """

            sigma: float = 1.0

        apply_docstring_descriptions(_BlurOp)
        assert (
            _BlurOp.model_fields["sigma"].description
            == "Standard deviation of the Gaussian kernel."
        )

    def test_explicit_description_not_overwritten(self) -> None:
        class _ExplicitOp(BaseModel):
            """Operation with an explicitly described field.

            Args:
                radius: Docstring description that must be ignored.
            """

            radius: int = Field(
                default=3, description="Explicit field description."
            )

        apply_docstring_descriptions(_ExplicitOp)
        assert (
            _ExplicitOp.model_fields["radius"].description
            == "Explicit field description."
        )

    def test_no_docstring_is_a_noop(self) -> None:
        class _NoDocOp(BaseModel):
            value: int = 0

        # Must not raise even though there is no docstring to parse.
        apply_docstring_descriptions(_NoDocOp)
        assert _NoDocOp.model_fields["value"].description is None

    def test_non_model_class_is_a_noop(self) -> None:
        class _PlainClass:
            """A plain class.

            Args:
                x: Some value.
            """

        # No model_fields → silent no-op, no exception.
        apply_docstring_descriptions(_PlainClass)
