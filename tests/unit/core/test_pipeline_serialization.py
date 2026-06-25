import json
import tempfile
from pathlib import Path

import pytest
import pandas as pd
from pydantic import ValidationError

from phenotypic import ImagePipeline, Image
from phenotypic._core._pipeline_parts._serializable_pipeline import SerializablePipeline
from phenotypic.data import load_colony
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur, EnhanceLocalContrast
from phenotypic.measure import MeasureShape, MeasureIntensity, MeasureColor
from phenotypic.refine import SmallObjectRemover, RemoveBorderObjects
from phenotypic.sdk_ import CONFIG_SUFFIX_PIPELINE, ensure_typed_json_suffix


class TestBasicSerialization:
    """Test basic serialization and deserialization functionality."""

    def test_empty_pipeline_serialization(self):
        """Test serialization of an empty pipeline."""
        pipe = ImagePipeline()
        json_str = pipe.to_json()

        # Verify JSON is valid
        config = json.loads(json_str)
        assert "pipe_cfgs" in config
        assert "meas" in config
        assert config["pipe_cfgs"] == {}
        assert config["meas"] == {}

    def test_empty_pipeline_roundtrip(self):
        """Test roundtrip serialization of an empty pipeline."""
        pipe = ImagePipeline()
        json_str = pipe.to_json()

        loaded_pipe = ImagePipeline.from_json(json_str)
        assert len(loaded_pipe._ops) == 0
        assert len(loaded_pipe._meas) == 0

    def test_single_operation_serialization(self):
        """Test serialization with a single operation."""
        pipe = ImagePipeline(ops=[OtsuDetector()])
        json_str = pipe.to_json()

        config = json.loads(json_str)
        assert len(config["pipe_cfgs"]) == 1
        assert "OtsuDetector" in config["pipe_cfgs"]
        assert config["pipe_cfgs"]["OtsuDetector"]["class"] == "OtsuDetector"

    def test_single_measurement_serialization(self):
        """Test serialization with a single measurement."""
        pipe = ImagePipeline(meas=[MeasureShape()])
        json_str = pipe.to_json()

        config = json.loads(json_str)
        assert len(config["meas"]) == 1
        assert "MeasureShape" in config["meas"]
        assert config["meas"]["MeasureShape"]["class"] == "MeasureShape"

    def test_multiple_operations_serialization(self):
        """Test serialization with multiple operations."""
        pipe = ImagePipeline(
                ops=[GaussianBlur(sigma=2), OtsuDetector(),
                     SmallObjectRemover(min_size=50)]
        )
        json_str = pipe.to_json()

        config = json.loads(json_str)
        assert len(config["pipe_cfgs"]) == 3
        assert "GaussianBlur" in config["pipe_cfgs"]
        assert "OtsuDetector" in config["pipe_cfgs"]
        assert "SmallObjectRemover" in config["pipe_cfgs"]

    def test_multiple_measurements_serialization(self):
        """Test serialization with multiple measurements."""
        pipe = ImagePipeline(meas=[MeasureShape(), MeasureIntensity(), MeasureColor()])
        json_str = pipe.to_json()

        config = json.loads(json_str)
        assert len(config["meas"]) == 3
        assert "MeasureShape" in config["meas"]
        assert "MeasureIntensity" in config["meas"]
        assert "MeasureColor" in config["meas"]


class TestParameterSerialization:
    """Test serialization of operations with various parameter types."""

    def test_boolean_parameters(self):
        """Test serialization of boolean parameters."""
        pipe = ImagePipeline(
                ops=[OtsuDetector(ignore_zeros=True, ignore_borders=False)]
        )
        json_str = pipe.to_json()

        loaded_pipe = ImagePipeline.from_json(json_str)
        detector = loaded_pipe._ops["OtsuDetector"]
        assert detector.ignore_zeros is True
        assert detector.ignore_borders is False

    def test_numeric_parameters(self):
        """Test serialization of int and float parameters."""
        pipe = ImagePipeline(
                ops=[
                    GaussianBlur(sigma=3),
                    OtsuDetector(),
                    SmallObjectRemover(min_size=100),
                ],
                meas=[MeasureShape()],
        )
        json_str = pipe.to_json()

        loaded_pipe = ImagePipeline.from_json(json_str)
        blur = loaded_pipe._ops["GaussianBlur"]

        # Test that public attributes are preserved
        assert blur.sigma == 3

        # Test that the loaded pipeline works correctly
        img = Image(load_colony(), name="test")
        result = loaded_pipe.apply_and_measure(img, inplace=False)
        assert result is not None
        assert len(result) > 0

    def test_string_parameters(self):
        """Test serialization of string parameters."""
        # Create a pipeline with an operation that has string parameters
        pipe = ImagePipeline(ops=[EnhanceLocalContrast()])
        json_str = pipe.to_json()

        loaded_pipe = ImagePipeline.from_json(json_str)
        assert "EnhanceLocalContrast" in loaded_pipe._ops

    def test_list_parameters(self):
        """Test serialization of list parameters."""
        # MeasureTexture accepts scale as a list
        from phenotypic.measure import MeasureTexture

        pipe = ImagePipeline(meas=[MeasureTexture(scale=[3, 5, 7], quant_lvl=8)])
        json_str = pipe.to_json()

        loaded_pipe = ImagePipeline.from_json(json_str)
        texture = loaded_pipe._meas["MeasureTexture"]
        assert texture.scale == [3, 5, 7]
        assert texture.quant_lvl == 8

    def test_dict_parameters(self):
        """Test serialization with dict-style operations input."""
        pipe = ImagePipeline(
                ops={"blur": GaussianBlur(sigma=2), "detect": OtsuDetector()}
        )
        json_str = pipe.to_json()

        config = json.loads(json_str)
        assert "blur" in config["pipe_cfgs"]
        assert "detect" in config["pipe_cfgs"]


class TestRoundtripFunctionality:
    """Test that pipelines work correctly after serialization roundtrip."""

    def test_roundtrip_produces_identical_results(self):
        """Test that original and loaded pipelines produce identical results."""
        # Create original pipeline
        original_pipe = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape()])

        # Get results from original
        img = Image(load_colony(), name="test")
        original_results = original_pipe.apply_and_measure(img)

        # Serialize and deserialize
        json_str = original_pipe.to_json()
        loaded_pipe = ImagePipeline.from_json(json_str)

        # Get results from loaded pipeline
        img2 = Image(load_colony(), name="test")
        loaded_results = loaded_pipe.apply_and_measure(img2)

        # Compare results
        pd.testing.assert_frame_equal(original_results, loaded_results)

    def test_complex_pipeline_roundtrip(self):
        """Test roundtrip with a complex pipeline."""
        pipe = ImagePipeline(
                ops=[
                    GaussianBlur(sigma=2),
                    OtsuDetector(ignore_zeros=True),
                    SmallObjectRemover(min_size=25),
                    RemoveBorderObjects(border_size=10),
                ],
                meas=[MeasureShape(), MeasureIntensity(), MeasureColor()],
                benchmark=True,
                verbose=False,
        )

        # Test with actual image
        img = Image(load_colony(), name="test")
        original_results = pipe.apply_and_measure(img)

        # Roundtrip - benchmark and verbose are not serialized, defaults to False
        json_str = pipe.to_json()
        loaded_pipe = ImagePipeline.from_json(json_str)

        # Verify configuration
        assert len(loaded_pipe._ops) == 4
        assert len(loaded_pipe._meas) == 3
        # benchmark and verbose default to False when loading
        assert loaded_pipe._benchmark is False
        assert loaded_pipe._verbose is False

        # Verify results
        img2 = Image(load_colony(), name="test")
        loaded_results = loaded_pipe.apply_and_measure(img2)
        pd.testing.assert_frame_equal(original_results, loaded_results)


class TestFileIO:
    """Test saving to and loading from files."""

    def test_save_to_file(self):
        """Test saving pipeline to a file."""
        pipe = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape()])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "pipeline.json"
            typed_filepath = ensure_typed_json_suffix(filepath, CONFIG_SUFFIX_PIPELINE)
            pipe.to_json(filepath)

            # Verify file exists and contains valid JSON
            assert not filepath.exists()
            assert typed_filepath.exists()
            config = json.loads(typed_filepath.read_text())
            assert "pipe_cfgs" in config
            assert "meas" in config

    def test_save_to_file_without_suffix_appends_typed_suffix(self):
        """A bare save path gets the full typed pipeline suffix."""
        pipe = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape()])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "pipeline"
            typed_filepath = ensure_typed_json_suffix(filepath, CONFIG_SUFFIX_PIPELINE)
            pipe.to_json(filepath)

            assert typed_filepath.exists()
            assert typed_filepath.name == "pipeline.json.pht-pipe"

    def test_load_from_file(self):
        """Test loading pipeline from a file."""
        pipe = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape()])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "pipeline.json"
            typed_filepath = ensure_typed_json_suffix(filepath, CONFIG_SUFFIX_PIPELINE)
            pipe.to_json(filepath)

            # Load from file
            loaded_pipe = ImagePipeline.from_json(typed_filepath)
            assert len(loaded_pipe._ops) == 1
            assert len(loaded_pipe._meas) == 1

    def test_load_explicit_legacy_json_file(self):
        """Explicit legacy ``.json`` files still load."""
        pipe = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape()])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "pipeline.json"
            filepath.write_text(pipe.to_json(), encoding="utf-8")

            loaded_pipe = ImagePipeline.from_json(filepath)
            assert len(loaded_pipe._ops) == 1
            assert len(loaded_pipe._meas) == 1

    def test_load_from_string_path(self):
        """Test loading from a string path (not Path object)."""
        pipe = ImagePipeline(ops=[OtsuDetector()])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "pipeline.json")
            typed_filepath = ensure_typed_json_suffix(filepath, CONFIG_SUFFIX_PIPELINE)
            pipe.to_json(filepath)

            # Load using string path
            loaded_pipe = ImagePipeline.from_json(str(typed_filepath))
            assert len(loaded_pipe._ops) == 1

    def test_roundtrip_through_file(self):
        """Test complete roundtrip through file."""
        original_pipe = ImagePipeline(
                ops=[GaussianBlur(sigma=2), OtsuDetector()],
                meas=[MeasureShape(), MeasureIntensity()],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "pipeline.json"
            typed_filepath = ensure_typed_json_suffix(filepath, CONFIG_SUFFIX_PIPELINE)
            original_pipe.to_json(filepath)
            loaded_pipe = ImagePipeline.from_json(typed_filepath)

            # Test functionality - use same image name to allow direct comparison
            img1 = Image(load_colony(), name="test")
            img2 = Image(load_colony(), name="test")

            results1 = original_pipe.apply_and_measure(img1)
            results2 = loaded_pipe.apply_and_measure(img2)

            pd.testing.assert_frame_equal(results1, results2)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_ops_only_pipeline(self):
        """Test pipeline with only operations, no measurements."""
        pipe = ImagePipeline(ops=[OtsuDetector(), SmallObjectRemover(min_size=50)])
        json_str = pipe.to_json()

        loaded_pipe = ImagePipeline.from_json(json_str)
        assert len(loaded_pipe._ops) == 2
        assert len(loaded_pipe._meas) == 0

    def test_meas_only_pipeline(self):
        """Test pipeline with only measurements, no operations."""
        pipe = ImagePipeline(meas=[MeasureShape(), MeasureIntensity()])
        json_str = pipe.to_json()

        loaded_pipe = ImagePipeline.from_json(json_str)
        assert len(loaded_pipe._ops) == 0
        assert len(loaded_pipe._meas) == 2

    def test_duplicate_operation_names(self):
        """Test handling of duplicate operation names."""
        pipe = ImagePipeline(
                ops=[GaussianBlur(sigma=1), GaussianBlur(sigma=2),
                     GaussianBlur(sigma=3)]
        )
        json_str = pipe.to_json()

        loaded_pipe = ImagePipeline.from_json(json_str)
        assert len(loaded_pipe._ops) == 3

        # Verify all three blurs are present with different parameters
        op_names = list(loaded_pipe._ops.keys())
        assert "GaussianBlur" in op_names
        assert "GaussianBlur_1" in op_names
        assert "GaussianBlur_2" in op_names

    def test_grid_preset_roundtrip(self):
        """nrows/ncols soft preset survives to_json / from_json."""
        pipe = ImagePipeline(nrows=16, ncols=24)
        json_str = pipe.to_json()

        config = json.loads(json_str)
        assert config["nrows"] == 16
        assert config["ncols"] == 24

        loaded = ImagePipeline.from_json(json_str)
        assert loaded.nrows == 16
        assert loaded.ncols == 24

    def test_grid_preset_omitted_when_unset(self):
        """When nrows/ncols are unset, JSON contains no such keys."""
        pipe = ImagePipeline(ops=[OtsuDetector()])
        json_str = pipe.to_json()

        config = json.loads(json_str)
        assert "nrows" not in config
        assert "ncols" not in config

        loaded = ImagePipeline.from_json(json_str)
        assert loaded.nrows is None
        assert loaded.ncols is None

    def test_legacy_json_without_grid_preset_loads(self):
        """Old JSON without nrows/ncols keys still loads cleanly."""
        legacy_json = json.dumps({
            "pipe_cfgs": {},
            "meas"     : {},
            "post"     : {},
            "name"     : "legacy",
            "desc"     : None,
            "reset"    : False,
        })

        loaded = ImagePipeline.from_json(legacy_json)
        assert loaded.nrows is None
        assert loaded.ncols is None

    def test_benchmark_and_verbose_flags_not_serialized(self):
        """Test that benchmark and verbose flags are not serialized but can be passed as params."""
        pipe = ImagePipeline(ops=[OtsuDetector()], benchmark=True, verbose=True)
        json_str = pipe.to_json()

        # Verify benchmark and verbose are NOT in the serialized JSON
        config = json.loads(json_str)
        assert "benchmark" not in config
        assert "verbose" not in config

        # Default load should have False for both
        loaded_pipe = ImagePipeline.from_json(json_str)
        assert loaded_pipe._benchmark is False
        assert loaded_pipe._verbose is False

        # Can override with parameters
        loaded_pipe_with_flags = ImagePipeline.from_json(json_str, benchmark=True,
                                                         verbose=True)
        assert loaded_pipe_with_flags._benchmark is True
        assert loaded_pipe_with_flags._verbose is True

    def test_internal_state_excluded(self):
        """Test that internal state (attributes starting with _) is excluded."""
        pipe = ImagePipeline(ops=[OtsuDetector()])
        json_str = pipe.to_json()

        config = json.loads(json_str)

        # Check that no internal attributes are serialized
        for op_data in config["pipe_cfgs"].values():
            for param_key in op_data["params"].keys():
                assert not param_key.startswith("_"), (
                    f"Internal attribute {param_key} was serialized"
                )

    def test_dataframe_excluded(self):
        """Internal DataFrame state never leaks into serialized params.

        Operations are pydantic models with ``extra="forbid"``: an
        arbitrary attribute (e.g. a stray DataFrame) cannot be attached at
        all, and any legitimate internal state lives in a ``PrivateAttr``
        which ``model_dump`` excludes by construction. Both halves of that
        guarantee are checked here.
        """
        # An arbitrary (DataFrame-valued) attribute is rejected outright.
        op = OtsuDetector()
        with pytest.raises(ValidationError):
            op.test_df = pd.DataFrame({"a": [1, 2, 3]})

        # The serialized params carry only declared fields — no DataFrame.
        pipe = ImagePipeline(ops=[OtsuDetector()])
        config = json.loads(pipe.to_json())
        params = config["pipe_cfgs"]["OtsuDetector"]["params"]
        assert set(params) == {"ignore_zeros", "ignore_borders"}
        assert not any(isinstance(v, dict) and "a" in v for v in params.values())


class TestPreMigrationBackwardCompat:
    """Loading JSON written before the pydantic v2 migration.

    The migration was structure-preserving — same class names, same
    parameter names, same defaults — and kept the on-disk ``{class,
    params}`` envelope. A hand-written pre-migration ``pipeline.json``
    must therefore still load via ``model_validate`` and yield an
    equivalent pipeline.
    """

    def test_loads_pre_migration_pipeline_json(self):
        """A hand-written old-style pipeline.json reconstructs equivalently.

        The fixture mimics what the pre-pydantic ``_serialize_operations``
        emitted: a flat ``params`` dict of public attributes per operation
        plus a measurer entry, under the historical envelope.
        """
        # Hand-written pre-migration payload (a couple of ops + a measurer).
        old_json = json.dumps({
            "version"  : "0.13.0",
            "name"     : "legacy_pipeline",
            "desc"     : "saved before the pydantic migration",
            "reset"    : False,
            "pipe_cfgs": {
                "GaussianBlur"      : {
                    "class" : "GaussianBlur",
                    "params": {"sigma": 3.0, "mode": "reflect"},
                },
                "OtsuDetector"      : {
                    "class" : "OtsuDetector",
                    "params": {"ignore_zeros": True, "ignore_borders": False},
                },
                "SmallObjectRemover": {
                    "class" : "SmallObjectRemover",
                    "params": {"min_size": 40},
                },
            },
            "meas"     : {
                "MeasureShape": {"class": "MeasureShape", "params": {}},
            },
        })

        loaded = ImagePipeline.from_json(old_json)

        # Build the equivalent pipeline directly and compare.
        expected = ImagePipeline(
                ops=[
                    GaussianBlur(sigma=3.0, mode="reflect"),
                    OtsuDetector(ignore_zeros=True, ignore_borders=False),
                    SmallObjectRemover(min_size=40),
                ],
                meas=[MeasureShape()],
                name="legacy_pipeline",
                desc="saved before the pydantic migration",
        )

        assert loaded.name == expected.name == "legacy_pipeline"
        assert loaded.desc == expected.desc
        assert list(loaded._ops.keys()) == list(expected._ops.keys())
        assert list(loaded._meas.keys()) == list(expected._meas.keys())

        loaded_blur = loaded._ops["GaussianBlur"]
        assert isinstance(loaded_blur, GaussianBlur)
        assert loaded_blur.sigma == 3.0
        assert loaded_blur.mode == "reflect"

        loaded_otsu = loaded._ops["OtsuDetector"]
        assert isinstance(loaded_otsu, OtsuDetector)
        assert loaded_otsu.ignore_zeros is True
        assert loaded_otsu.ignore_borders is False

        loaded_remover = loaded._ops["SmallObjectRemover"]
        assert isinstance(loaded_remover, SmallObjectRemover)
        assert loaded_remover.min_size == 40

        assert isinstance(loaded._meas["MeasureShape"], MeasureShape)

        # A re-serialized round-trip of the loaded pipeline is stable.
        assert ImagePipeline.from_json(loaded.to_json()).to_json() == \
               loaded.to_json()

    def test_loads_pre_migration_legacy_nested_operation_list(self):
        """Legacy ``__type__: operation_list`` nesting still reconstructs.

        The hand-rolled ``operation_list`` marker (used before nested ops
        moved to ``OperationField``) must still be translated into live
        operations by ``_deserialize_value`` so old composite pipelines load.
        This pins the *marker* translation under the current field name
        ``ops``; the legacy ``detectors`` *field name* is a hard break, pinned
        separately by
        ``test_pre_migration_detectors_field_name_is_rejected``.
        """
        from phenotypic.detect import CompositeDetector

        old_json = json.dumps({
            "version"  : "0.13.0",
            "name"     : "legacy_composite",
            "desc"     : None,
            "reset"    : False,
            "pipe_cfgs": {
                "CompositeDetector": {
                    "class" : "CompositeDetector",
                    "params": {
                        "mode"             : "union",
                        "min_overlap_ratio": 0.0,
                        "ops"        : {
                            "__type__": "operation_list",
                            "items"   : [
                                {
                                    "class" : "OtsuDetector",
                                    "params": {"ignore_zeros": True},
                                },
                                {
                                    "class" : "CannyDetector",
                                    "params": {"sigma": 2},
                                },
                            ],
                        },
                    },
                },
            },
            "meas"     : {},
        })

        loaded = ImagePipeline.from_json(old_json)
        composite = loaded._ops["CompositeDetector"]
        assert isinstance(composite, CompositeDetector)
        assert composite.mode == "union"
        assert len(composite.ops) == 2
        assert type(composite.ops[0]).__name__ == "OtsuDetector"
        assert composite.ops[0].ignore_zeros is True
        assert type(composite.ops[1]).__name__ == "CannyDetector"
        assert composite.ops[1].sigma == 2

    def test_pre_migration_detectors_field_name_is_rejected(self):
        """The renamed ``detectors``→``ops`` field is a hard break (no alias).

        A pipeline saved before the rename keyed the composite's nested ops
        under ``detectors``. ``BaseOperation`` sets ``extra="forbid"``, so such
        a document no longer loads -- it raises ``ValidationError`` rather than
        silently dropping the legacy ops. This pins the intentional breaking
        change so a future accidental alias/leniency is caught.
        """
        from pydantic import ValidationError

        old_json = json.dumps({
            "version"  : "0.13.0",
            "name"     : "legacy_composite",
            "desc"     : None,
            "reset"    : False,
            "pipe_cfgs": {
                "CompositeDetector": {
                    "class" : "CompositeDetector",
                    "params": {
                        "mode"     : "union",
                        "detectors": {
                            "__type__": "operation_list",
                            "items"   : [
                                {"class": "OtsuDetector", "params": {}},
                            ],
                        },
                    },
                },
            },
            "meas"     : {},
        })

        with pytest.raises(ValidationError):
            ImagePipeline.from_json(old_json)


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_json_string(self):
        """Test loading from invalid JSON string."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            ImagePipeline.from_json("not valid json {]}")

    def test_nonexistent_file(self):
        """Test loading from nonexistent file."""
        # Should treat as JSON string and fail with invalid JSON
        with pytest.raises(ValueError):
            ImagePipeline.from_json("/nonexistent/path/to/file.json")

    def test_missing_class(self):
        """Test error when a class cannot be found."""
        config = {
            "pipe_cfgs": {"fake": {"class": "NonExistentClass", "params": {}}},
            "meas"     : {},
            "benchmark": False,
            "verbose"  : False,
        }
        json_str = json.dumps(config)

        with pytest.raises(AttributeError, match="not found in phenotypic namespace"):
            ImagePipeline.from_json(json_str)

    def test_malformed_config_missing_ops(self):
        """Test handling of config without 'pipe_cfgs' key."""
        config = {"meas": {}, "benchmark": False, "verbose": False}
        json_str = json.dumps(config)

        # Should work with empty pipe_cfgs
        loaded_pipe = ImagePipeline.from_json(json_str)
        assert len(loaded_pipe._ops) == 0

    def test_malformed_config_missing_meas(self):
        """Test handling of config without 'meas' key."""
        config = {"pipe_cfgs": {}, "benchmark": False, "verbose": False}
        json_str = json.dumps(config)

        # Should work with empty meas
        loaded_pipe = ImagePipeline.from_json(json_str)
        assert len(loaded_pipe._meas) == 0


class TestNameAndDescAttributes:
    """Test name and desc attribute functionality."""

    def test_name_auto_generation(self):
        """Test that pipelines get auto-generated UUID4 names."""
        import uuid

        pipe = ImagePipeline(ops=[OtsuDetector()])
        assert pipe.name is not None
        assert isinstance(pipe.name, str)
        # Verify it's a valid UUID4
        uuid.UUID(pipe.name, version=4)

    def test_name_explicit(self):
        """Test explicit name assignment."""
        pipe = ImagePipeline(ops=[OtsuDetector()], name="my_custom_pipeline")
        assert pipe.name == "my_custom_pipeline"

    def test_desc_explicit(self):
        """Test explicit desc assignment."""
        pipe = ImagePipeline(ops=[OtsuDetector()], desc="My custom description")
        assert pipe.desc == "My custom description"

    def test_desc_property_setter(self):
        """Test desc property can be set after instantiation."""
        pipe = ImagePipeline(ops=[OtsuDetector()])
        original_desc = pipe.desc  # Should be docstring

        pipe.desc = "New description"
        assert pipe.desc == "New description"
        assert pipe.desc != original_desc

    def test_name_and_desc_roundtrip(self):
        """Test that both name and desc survive roundtrip."""
        pipe = ImagePipeline(
                ops=[OtsuDetector()],
                name="my_pipeline",
                desc="My pipeline description"
        )
        json_str = pipe.to_json()
        loaded = ImagePipeline.from_json(json_str)
        assert loaded.name == "my_pipeline"
        assert loaded.desc == "My pipeline description"

    def test_version_mismatch_warning(self):
        """Test that version mismatch triggers a warning."""
        import warnings

        # Create JSON with different version
        config = {
            "pipe_cfgs": {},
            "meas"     : {},
            "benchmark": False,
            "verbose"  : False,
            "name"     : "test",
            "desc"     : None,
            "version"  : "0.0.0"  # Different version
        }
        json_str = json.dumps(config)

        # Should trigger warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ImagePipeline.from_json(json_str)
            assert len(w) == 1
            assert "version" in str(w[0].message).lower()
            assert "0.0.0" in str(w[0].message)

    def test_no_version_in_json_no_warning(self):
        """Test that old JSON without version doesn't trigger warning."""
        import warnings

        # Old JSON without version field
        config = {
            "pipe_cfgs": {},
            "meas"     : {},
            "benchmark": False,
            "verbose"  : False
        }
        json_str = json.dumps(config)

        # Should NOT trigger warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ImagePipeline.from_json(json_str)
            # Filter for our specific warning type
            version_warnings = [warn for warn in w if
                                "version" in str(warn.message).lower()]
            assert len(version_warnings) == 0


class TestPipelineAsOperationSerialization:
    """Test serialization of pipelines nested as operations inside other pipelines."""

    def test_plain_pipeline_nested_roundtrip(self):
        """Test that a plain ImagePipeline nested as operation roundtrips correctly."""
        inner = ImagePipeline(ops=[GaussianBlur(sigma=5), OtsuDetector()])
        outer = ImagePipeline(ops=[inner])

        json_str = outer.to_json()
        loaded = ImagePipeline.from_json(json_str)

        inner_loaded = list(loaded._ops.values())[0]
        assert isinstance(inner_loaded, ImagePipeline)
        inner_ops = list(inner_loaded._ops.values())
        assert len(inner_ops) == 2
        assert isinstance(inner_ops[0], GaussianBlur)
        assert inner_ops[0].sigma == 5
        assert isinstance(inner_ops[1], OtsuDetector)

    def test_serialization_format_uses_pipeline_operation_type(self):
        """Test that serialized format uses __type__: pipeline_operation."""
        inner = ImagePipeline(ops=[OtsuDetector()])
        outer = ImagePipeline(ops=[inner])

        config = json.loads(outer.to_json())
        inner_data = list(config["pipe_cfgs"].values())[0]
        assert inner_data["__type__"] == "pipeline_operation"
        assert "config" in inner_data
        assert "pipe_cfgs" in inner_data["config"]

    def test_nested_pipeline_with_measurements_preserved(self):
        """Test that measurements inside a nested pipeline are preserved."""
        inner = ImagePipeline(
                ops=[OtsuDetector()],
                meas=[MeasureShape(), MeasureIntensity()],
        )
        outer = ImagePipeline(ops=[inner])

        loaded = ImagePipeline.from_json(outer.to_json())
        inner_loaded = list(loaded._ops.values())[0]
        assert len(inner_loaded._meas) == 2
        meas_classes = {type(m).__name__ for m in inner_loaded._meas.values()}
        assert meas_classes == {"MeasureShape", "MeasureIntensity"}

    def test_deeply_nested_pipelines(self):
        """Test three levels of pipeline nesting."""
        level1 = ImagePipeline(ops=[GaussianBlur(sigma=3)])
        level2 = ImagePipeline(ops=[level1, OtsuDetector()])
        level3 = ImagePipeline(ops=[level2])

        loaded = ImagePipeline.from_json(level3.to_json())
        l2_loaded = list(loaded._ops.values())[0]
        assert isinstance(l2_loaded, ImagePipeline)

        l2_ops = list(l2_loaded._ops.values())
        assert len(l2_ops) == 2
        l1_loaded = l2_ops[0]
        assert isinstance(l1_loaded, ImagePipeline)

        l1_ops = list(l1_loaded._ops.values())
        assert isinstance(l1_ops[0], GaussianBlur)
        assert l1_ops[0].sigma == 3

    def test_mixed_regular_ops_and_pipeline_ops(self):
        """Test pipeline containing both regular operations and nested pipelines."""
        inner = ImagePipeline(ops=[OtsuDetector()])
        outer = ImagePipeline(
                ops=[GaussianBlur(sigma=2), inner, SmallObjectRemover(min_size=50)]
        )

        loaded = ImagePipeline.from_json(outer.to_json())
        ops = list(loaded._ops.values())
        assert len(ops) == 3
        assert isinstance(ops[0], GaussianBlur)
        assert isinstance(ops[1], ImagePipeline)
        assert isinstance(ops[2], SmallObjectRemover)

    def test_inner_pipeline_name_and_desc_preserved(self):
        """Test that inner pipeline name and desc are preserved."""
        inner = ImagePipeline(
                ops=[OtsuDetector()],
                name="inner_pipe",
                desc="Inner pipeline description",
        )
        outer = ImagePipeline(ops=[inner])

        loaded = ImagePipeline.from_json(outer.to_json())
        inner_loaded = list(loaded._ops.values())[0]
        assert inner_loaded.name == "inner_pipe"
        assert inner_loaded.desc == "Inner pipeline description"

    def test_find_class_discovers_prefab_classes(self):
        """Test that _find_class_in_phenotypic discovers prefab classes."""
        from phenotypic.prefab import FilamentousFungiPipeline

        found = SerializablePipeline._find_class_in_phenotypic(
                "FilamentousFungiPipeline"
        )
        assert found is FilamentousFungiPipeline

    def test_prefab_pipeline_nested_roundtrip_preserves_class(self):
        """Test that a PrefabPipeline nested as operation is re-tagged on roundtrip."""
        from phenotypic.prefab import HeavyOtsuPipeline

        inner = HeavyOtsuPipeline()
        outer = ImagePipeline(ops=[inner])
        loaded = ImagePipeline.from_json(outer.to_json())
        inner_loaded = list(loaded._ops.values())[0]
        assert type(inner_loaded).__name__ == "HeavyOtsuPipeline"

    def test_find_class_discovers_all_prefab_pipelines(self):
        """Test that all prefab pipeline classes are discoverable."""
        prefab_names = [
            "HeavyWatershedPipeline",
            "HeavyOtsuPipeline",
            "GridSectionPipeline",
            "HeavyRoundPeaksPipeline",
            "RoundPeaksPipeline",
            "FilamentousFungiPipeline",
        ]
        for name in prefab_names:
            found = SerializablePipeline._find_class_in_phenotypic(name)
            assert found is not None, f"{name} not found in phenotypic namespace"
