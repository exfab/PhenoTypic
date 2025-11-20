import json
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from phenotypic import ImagePipeline, Image
from phenotypic.data import load_colony
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur, CLAHE, MedianEnhancer
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
        assert 'ops' in config
        assert 'meas' in config
        assert config['ops'] == {}
        assert config['meas'] == {}

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
        assert len(config['ops']) == 1
        assert 'OtsuDetector' in config['ops']
        assert config['ops']['OtsuDetector']['class'] == 'OtsuDetector'

    def test_single_measurement_serialization(self):
        """Test serialization with a single measurement."""
        pipe = ImagePipeline(meas=[MeasureShape()])
        json_str = pipe.to_json()
        
        config = json.loads(json_str)
        assert len(config['meas']) == 1
        assert 'MeasureShape' in config['meas']
        assert config['meas']['MeasureShape']['class'] == 'MeasureShape'

    def test_multiple_operations_serialization(self):
        """Test serialization with multiple operations."""
        pipe = ImagePipeline(ops=[
            GaussianBlur(sigma=2),
            OtsuDetector(),
            SmallObjectRemover(min_size=50)
        ])
        json_str = pipe.to_json()
        
        config = json.loads(json_str)
        assert len(config['ops']) == 3
        assert 'GaussianBlur' in config['ops']
        assert 'OtsuDetector' in config['ops']
        assert 'SmallObjectRemover' in config['ops']

    def test_multiple_measurements_serialization(self):
        """Test serialization with multiple measurements."""
        pipe = ImagePipeline(meas=[
            MeasureShape(),
            MeasureIntensity(),
            MeasureColor()
        ])
        json_str = pipe.to_json()
        
        config = json.loads(json_str)
        assert len(config['meas']) == 3
        assert 'MeasureShape' in config['meas']
        assert 'MeasureIntensity' in config['meas']
        assert 'MeasureColor' in config['meas']


class TestParameterSerialization:
    """Test serialization of operations with various parameter types."""

    def test_boolean_parameters(self):
        """Test serialization of boolean parameters."""
        pipe = ImagePipeline(ops=[OtsuDetector(ignore_zeros=True, ignore_borders=False)])
        json_str = pipe.to_json()
        
        loaded_pipe = ImagePipeline.from_json(json_str)
        detector = loaded_pipe._ops['OtsuDetector']
        assert detector.ignore_zeros is True
        assert detector.ignore_borders is False

    def test_numeric_parameters(self):
        """Test serialization of int and float parameters."""
        pipe = ImagePipeline(
            ops=[
                GaussianBlur(sigma=3),
                OtsuDetector(),
                SmallObjectRemover(min_size=100)
            ],
            meas=[MeasureShape()]
        )
        json_str = pipe.to_json()
        
        loaded_pipe = ImagePipeline.from_json(json_str)
        blur = loaded_pipe._ops['GaussianBlur']
        
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
        assert 'CLAHE' in loaded_pipe._ops

    def test_list_parameters(self):
        """Test serialization of list parameters."""
        # MeasureTexture accepts scale as a list
        from phenotypic.measure import MeasureTexture
        pipe = ImagePipeline(meas=[MeasureTexture(scale=[3, 5, 7], quant_lvl=8)])
        json_str = pipe.to_json()
        
        loaded_pipe = ImagePipeline.from_json(json_str)
        texture = loaded_pipe._meas['MeasureTexture']
        assert texture.scale == [3, 5, 7]
        assert texture.quant_lvl == 8

    def test_dict_parameters(self):
        """Test serialization with dict-style operations input."""
        pipe = ImagePipeline(ops={
            'blur': GaussianBlur(sigma=2),
            'detect': OtsuDetector()
        })
        json_str = pipe.to_json()
        
        config = json.loads(json_str)
        assert 'blur' in config['ops']
        assert 'detect' in config['ops']


class TestRoundtripFunctionality:
    """Test that pipelines work correctly after serialization roundtrip."""

    def test_roundtrip_produces_identical_results(self):
        """Test that original and loaded pipelines produce identical results."""
        # Create original pipeline
        original_pipe = ImagePipeline(
            ops=[OtsuDetector()],
            meas=[MeasureShape()]
        )
        
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
                BorderObjectRemover(border_size=10)
            ],
            meas=[
                MeasureShape(),
                MeasureIntensity(),
                MeasureColor()
            ],
            benchmark=True,
            verbose=False
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
            assert 'ops' in config
            assert 'meas' in config

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
            meas=[MeasureShape(), MeasureIntensity()]
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
        pipe = ImagePipeline(ops=[
            GaussianBlur(sigma=1),
            GaussianBlur(sigma=2),
            GaussianBlur(sigma=3)
        ])
        json_str = pipe.to_json()
        
        loaded_pipe = ImagePipeline.from_json(json_str)
        assert len(loaded_pipe._ops) == 3
        
        # Verify all three blurs are present with different parameters
        op_names = list(loaded_pipe._ops.keys())
        assert 'GaussianBlur' in op_names
        assert 'GaussianBlur_1' in op_names
        assert 'GaussianBlur_2' in op_names

    def test_benchmark_and_verbose_flags(self):
        """Test that benchmark and verbose flags are preserved."""
        pipe = ImagePipeline(
            ops=[OtsuDetector()],
            benchmark=True,
            verbose=True
        )
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
        for op_data in config['ops'].values():
            for param_key in op_data['params'].keys():
                assert not param_key.startswith('_'), f"Internal attribute {param_key} was serialized"

    def test_dataframe_excluded(self):
        """Test that pandas DataFrames are excluded from serialization."""
        pipe = ImagePipeline(ops=[OtsuDetector()])
        
        # Manually add a DataFrame to an operation (simulating internal state)
        pipe._ops['OtsuDetector'].test_df = pd.DataFrame({'a': [1, 2, 3]})
        
        json_str = pipe.to_json()
        config = json.loads(json_str)
        
        # Verify DataFrame is not in the serialized data
        assert 'test_df' not in config['ops']['OtsuDetector']['params']


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
            'ops': {
                'fake': {
                    'class': 'NonExistentClass',
                    'params': {}
                }
            },
            'meas': {},
            'benchmark': False,
            'verbose': False
        }
        json_str = json.dumps(config)
        
        with pytest.raises(AttributeError, match="not found in phenotypic namespace"):
            ImagePipeline.from_json(json_str)

    def test_malformed_config_missing_ops(self):
        """Test handling of config without 'ops' key."""
        config = {
            'meas': {},
            'benchmark': False,
            'verbose': False
        }
        json_str = json.dumps(config)
        
        # Should work with empty ops
        loaded_pipe = ImagePipeline.from_json(json_str)
        assert len(loaded_pipe._ops) == 0

    def test_malformed_config_missing_meas(self):
        """Test handling of config without 'meas' key."""
        config = {
            'ops': {},
            'benchmark': False,
            'verbose': False
        }
        json_str = json.dumps(config)
        
        # Should work with empty meas
        loaded_pipe = ImagePipeline.from_json(json_str)
        assert len(loaded_pipe._meas) == 0


class TestBatchPipelineSerialization:
    """Test serialization with ImagePipelineBatch."""

    def test_batch_pipeline_serialization(self):
        """Test that ImagePipelineBatch can also be serialized."""
        from phenotypic.core._pipeline_parts._image_pipeline_batch import ImagePipelineBatch
        
        pipe = ImagePipelineBatch(
            ops=[OtsuDetector()],
            meas=[MeasureShape()],
            njobs=2
        )
        
        # Should only serialize core parameters, not batch-specific ones
        json_str = pipe.to_json()
        config = json.loads(json_str)
        
        assert 'ops' in config
        assert 'meas' in config
        assert 'benchmark' in config
        assert 'verbose' in config
        # njobs should not be in the serialization
        assert 'njobs' not in config

    def test_batch_pipeline_roundtrip(self):
        """Test roundtrip of ImagePipelineBatch preserves core functionality."""
        from phenotypic.core._pipeline_parts._image_pipeline_batch import ImagePipelineBatch
        
        pipe = ImagePipelineBatch(
            ops=[OtsuDetector()],
            meas=[MeasureShape()],
            njobs=2,
            benchmark=True
        )
        
        json_str = pipe.to_json()
        
        # Load back as ImagePipelineBatch
        loaded_pipe = ImagePipelineBatch.from_json(json_str)
        
        assert len(loaded_pipe._ops) == 1
        assert len(loaded_pipe._meas) == 1
        assert loaded_pipe._benchmark is True

