import json
import tempfile
from pathlib import Path

import pytest
import pandas as pd

from phenotypic import ImagePipeline, Image
from phenotypic.data import load_colony, load_synth_yeast_plate
from phenotypic.detect import OtsuDetector, CannyDetector
from phenotypic.enhance import GaussianBlur, CLAHE
from phenotypic.measure import MeasureShape, MeasureIntensity, MeasureColor
from phenotypic.refine import SmallObjectRemover, BorderObjectRemover


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
        pipe = ImagePipeline(ops=[CLAHE()])
        json_str = pipe.to_json()

        loaded_pipe = ImagePipeline.from_json(json_str)
        assert "CLAHE" in loaded_pipe._ops

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
                    BorderObjectRemover(border_size=10),
                ],
                meas=[MeasureShape(), MeasureIntensity(), MeasureColor()],
                benchmark=True,
                verbose=False,
        )

        # Test with actual image
        img = Image(load_colony(), name="test")
        original_results = pipe.apply_and_measure(img)

        # Roundtrip
        json_str = pipe.to_json()
        loaded_pipe = ImagePipeline.from_json(json_str)

        # Verify configuration
        assert len(loaded_pipe._ops) == 4
        assert len(loaded_pipe._meas) == 3
        assert loaded_pipe._benchmark is True
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
            pipe.to_json(filepath)

            # Verify file exists and contains valid JSON
            assert filepath.exists()
            config = json.loads(filepath.read_text())
            assert "pipe_cfgs" in config
            assert "meas" in config

    def test_load_from_file(self):
        """Test loading pipeline from a file."""
        pipe = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape()])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "pipeline.json"
            pipe.to_json(filepath)

            # Load from file
            loaded_pipe = ImagePipeline.from_json(filepath)
            assert len(loaded_pipe._ops) == 1
            assert len(loaded_pipe._meas) == 1

    def test_load_from_string_path(self):
        """Test loading from a string path (not Path object)."""
        pipe = ImagePipeline(ops=[OtsuDetector()])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "pipeline.json")
            pipe.to_json(filepath)

            # Load using string path
            loaded_pipe = ImagePipeline.from_json(filepath)
            assert len(loaded_pipe._ops) == 1

    def test_roundtrip_through_file(self):
        """Test complete roundtrip through file."""
        original_pipe = ImagePipeline(
                ops=[GaussianBlur(sigma=2), OtsuDetector()],
                meas=[MeasureShape(), MeasureIntensity()],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "pipeline.json"
            original_pipe.to_json(filepath)
            loaded_pipe = ImagePipeline.from_json(filepath)

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

    def test_benchmark_and_verbose_flags(self):
        """Test that benchmark and verbose flags are preserved."""
        pipe = ImagePipeline(ops=[OtsuDetector()], benchmark=True, verbose=True)
        json_str = pipe.to_json()

        loaded_pipe = ImagePipeline.from_json(json_str)
        assert loaded_pipe._benchmark is True
        assert loaded_pipe._verbose is True

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
        """Test that pandas DataFrames are excluded from serialization."""
        pipe = ImagePipeline(ops=[OtsuDetector()])

        # Manually add a DataFrame to an operation (simulating internal state)
        pipe._ops["OtsuDetector"].test_df = pd.DataFrame({"a": [1, 2, 3]})

        json_str = pipe.to_json()
        config = json.loads(json_str)

        # Verify DataFrame is not in the serialized data
        assert "test_df" not in config["pipe_cfgs"]["OtsuDetector"]["params"]


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

    def test_name_serialization(self):
        """Test that name is included in JSON."""
        pipe = ImagePipeline(ops=[OtsuDetector()], name="test_pipeline")
        json_str = pipe.to_json()
        config = json.loads(json_str)
        assert "name" in config
        assert config["name"] == "test_pipeline"

    def test_name_deserialization(self):
        """Test that name is restored from JSON."""
        pipe = ImagePipeline(ops=[OtsuDetector()], name="test_pipeline")
        json_str = pipe.to_json()
        loaded = ImagePipeline.from_json(json_str)
        assert loaded.name == "test_pipeline"

    def test_desc_default_returns_docstring(self):
        """Test that desc returns class docstring when not set."""
        pipe = ImagePipeline(ops=[OtsuDetector()])
        assert pipe.desc is not None
        assert isinstance(pipe.desc, str)
        # Should be the ImagePipeline docstring
        assert "comprehensive class" in pipe.desc.lower()

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

    def test_desc_serialization(self):
        """Test that desc is included in JSON."""
        pipe = ImagePipeline(ops=[OtsuDetector()], desc="Test description")
        json_str = pipe.to_json()
        config = json.loads(json_str)
        assert "desc" in config
        assert config["desc"] == "Test description"

    def test_desc_serialization_none(self):
        """Test that desc can be null in JSON when not set."""
        pipe = ImagePipeline(ops=[OtsuDetector()])
        json_str = pipe.to_json()
        config = json.loads(json_str)
        assert "desc" in config
        assert config["desc"] is None

    def test_desc_deserialization(self):
        """Test that desc is restored from JSON."""
        pipe = ImagePipeline(ops=[OtsuDetector()], desc="Test description")
        json_str = pipe.to_json()
        loaded = ImagePipeline.from_json(json_str)
        assert loaded.desc == "Test description"

    def test_desc_deserialization_none_returns_docstring(self):
        """Test that desc returns docstring when deserialized as None."""
        pipe = ImagePipeline(ops=[OtsuDetector()])  # desc not set
        json_str = pipe.to_json()
        loaded = ImagePipeline.from_json(json_str)
        assert loaded.desc is not None
        assert "comprehensive class" in loaded.desc.lower()

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

    def test_version_included_in_serialization(self):
        """Test that phenotypic version is included in JSON."""
        import phenotypic

        pipe = ImagePipeline(ops=[OtsuDetector()])
        json_str = pipe.to_json()
        config = json.loads(json_str)
        assert "version" in config
        assert config["version"] == phenotypic.__version__

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
            loaded = ImagePipeline.from_json(json_str)
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
            loaded = ImagePipeline.from_json(json_str)
            # Filter for our specific warning type
            version_warnings = [warn for warn in w if
                                "version" in str(warn.message).lower()]
            assert len(version_warnings) == 0


class TestNestedOperationsSerialization:
    """Test serialization of operations containing other operations."""

    def test_nested_operations_serialization(self):
        """Test that operations containing other operations serialize correctly."""
        from phenotypic.detect import CompositeDetector, CannyDetector

        # Create composite detector with nested detectors
        composite = CompositeDetector(
                detectors=[OtsuDetector(), CannyDetector(sigma=2.0)],
                mode='overlap',
                min_overlap_ratio=0.6
        )

        pipeline = ImagePipeline([composite])

        # Serialize
        json_str = pipeline.to_json()

        # Verify JSON structure contains nested operations
        config = json.loads(json_str)
        pipe_cfgs = config['pipe_cfgs']

        # Should have CompositeDetector
        assert any('CompositeDetector' in key for key in pipe_cfgs.keys())

        # Find the composite detector entry
        composite_key = [k for k in pipe_cfgs.keys() if 'CompositeDetector' in k][0]
        composite_data = pipe_cfgs[composite_key]

        # Verify nested detectors are serialized
        assert 'detectors' in composite_data['params']
        detectors_data = composite_data['params']['detectors']
        assert detectors_data['__type__'] == 'operation_list'
        assert len(detectors_data['items']) == 2

        # Verify first detector (OtsuDetector)
        otsu_data = detectors_data['items'][0]
        assert otsu_data['class'] == 'OtsuDetector'

        # Verify second detector (CannyDetector)
        canny_data = detectors_data['items'][1]
        assert canny_data['class'] == 'CannyDetector'
        assert canny_data['params']['sigma'] == 2.0

        # Verify mode and min_overlap_ratio
        assert composite_data['params']['mode'] == 'overlap'
        assert composite_data['params']['min_overlap_ratio'] == 0.6

    def test_nested_operations_deserialization(self):
        """Test that nested operations are correctly deserialized."""
        from phenotypic.detect import CompositeDetector

        # Create composite detector
        composite = CompositeDetector(
                detectors=[
                    OtsuDetector(ignore_zeros=True),
                    CannyDetector(sigma=2)
                ],
                mode='union'
        )

        pipeline = ImagePipeline([composite])

        # Serialize and deserialize
        json_str = pipeline.to_json()
        restored_pipeline = ImagePipeline.from_json(json_str)

        # Verify structure
        assert len(restored_pipeline._ops) == 1
        restored_composite = list(restored_pipeline._ops.values())[0]
        assert isinstance(restored_composite, CompositeDetector)
        assert len(restored_composite.detectors) == 2
        assert isinstance(restored_composite.detectors[0], OtsuDetector)
        assert isinstance(restored_composite.detectors[1], CannyDetector)
        assert restored_composite.mode == 'union'

        # Verify nested detector parameters
        assert restored_composite.detectors[0].ignore_zeros == True
        assert restored_composite.detectors[1].sigma == 2

    def test_nested_operations_functional_equivalence(self):
        """Test that serialized/deserialized nested operations work identically."""
        from phenotypic.detect import CompositeDetector

        image = load_synth_yeast_plate()

        # Original detector
        composite = CompositeDetector(
                detectors=[OtsuDetector(), CannyDetector(sigma=2)],
                mode='intersection'
        )
        original_result = composite.apply(image)

        # Serialize and deserialize
        pipeline = ImagePipeline([composite])
        json_str = pipeline.to_json()
        restored_pipeline = ImagePipeline.from_json(json_str)

        # Apply restored detector
        restored_result = restored_pipeline.apply(image.copy(), inplace=False)

        # Results should be identical
        import numpy as np

        np.testing.assert_array_equal(
                original_result.objmask[:],
                restored_result.objmask[:]
        )
        np.testing.assert_array_equal(
                original_result.objmap[:],
                restored_result.objmap[:]
        )

    def test_single_nested_operation(self):
        """Test serialization of operation containing single nested operation."""
        from phenotypic.detect import CompositeDetector

        # Single detector in CompositeDetector
        composite = CompositeDetector(
                detectors=[OtsuDetector()],
                mode='union'
        )

        pipeline = ImagePipeline([composite])
        json_str = pipeline.to_json()

        # Deserialize
        restored = ImagePipeline.from_json(json_str)
        restored_composite = list(restored._ops.values())[0]

        assert len(restored_composite.detectors) == 1
        assert isinstance(restored_composite.detectors[0], OtsuDetector)
