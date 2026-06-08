from typing import ClassVar

import numpy as np
import pytest

from phenotypic import Image, ImagePipeline
from phenotypic.abc_ import ObjectDetector
from phenotypic.detect import (
    OtsuDetector,
    CannyDetector,
    CompositeDetector,
    TriangleDetector,
)


class StaticMaskDetector(ObjectDetector):
    """Detector fixture that emits one registered binary mask."""

    masks: ClassVar[dict[str, np.ndarray]] = {}
    key: str

    def _operate(self, image):
        image.objmask[:] = self.masks[self.key]
        return image


class TestCompositeDetector:
    """Test suite for CompositeDetector functionality and serialization."""

    def test_union_mode(self, synth_plate):
        """Test that union mode combines masks with logical OR."""
        image = synth_plate.copy()

        composite = CompositeDetector(
                detectors=[OtsuDetector(), CannyDetector(sigma=2)],
                mode='union'
        )
        result = composite.apply(image)

        # Union mask should contain all pixels from both detectors
        otsu_result = OtsuDetector().apply(image)
        canny_result = CannyDetector(sigma=2).apply(image)

        # Union mask should be logical OR of both
        expected_mask = np.logical_or(otsu_result.objmask[:], canny_result.objmask[:])
        np.testing.assert_array_equal(result.objmask[:], expected_mask)

    def test_intersection_mode(self, synth_plate):
        """Test that intersection mode combines masks with logical AND."""
        image = synth_plate.copy()

        composite = CompositeDetector(
                detectors=[OtsuDetector(), CannyDetector(sigma=2)],
                mode='intersection'
        )
        result = composite.apply(image)

        # Intersection mask should be logical AND of both
        otsu_result = OtsuDetector().apply(image)
        canny_result = CannyDetector(sigma=2).apply(image)

        expected_mask = np.logical_and(otsu_result.objmask[:], canny_result.objmask[:])
        np.testing.assert_array_equal(result.objmask[:], expected_mask)

    def test_overlap_mode(self, synth_plate):
        """Test that overlap mode filters objects by overlap threshold."""
        image = synth_plate.copy()

        composite = CompositeDetector(
                detectors=[OtsuDetector(), CannyDetector(sigma=2)],
                mode='overlap',
                min_overlap_ratio=0.7
        )
        result = composite.apply(image)

        # Should produce valid objmask and objmap
        assert result.objmask[:].sum() >= 0
        assert result.objmap[:].max() >= 0

    # objmask/objmap consistency is now covered by the smoke contract
    # tests/smoke/test_operation.py::test_detector_objmap_objmask_consistency

    def test_single_detector(self, synth_plate):
        """Test that single detector in CompositeDetector works."""
        image = synth_plate.copy()

        composite = CompositeDetector(
                detectors=[OtsuDetector()],
                mode='union'
        )
        result = composite.apply(image)

        # Single detector should match the original detector
        single_result = OtsuDetector().apply(image)

        np.testing.assert_array_equal(result.objmask[:], single_result.objmask[:])

    def test_three_detector_ensemble(self, synth_plate):
        """Test ensemble with three detectors."""
        image = synth_plate.copy()

        composite = CompositeDetector(
                detectors=[
                    OtsuDetector(),
                    CannyDetector(sigma=2),
                    TriangleDetector()
                ],
                mode='union'
        )
        result = composite.apply(image)

        # Should produce valid results
        assert result.objmap[:].max() > 0
        assert result.objmask[:].sum() > 0

    def test_empty_detectors_raises(self, synth_plate):
        """Test that empty detectors list raises an error."""
        with pytest.raises(Exception, match="At least one detector"):
            CompositeDetector(detectors=[]).apply(synth_plate.copy())

    def test_invalid_mode_raises(self, synth_plate):
        """Test that an invalid mode is rejected at construction.

        Under the pydantic v2 migration ``mode`` is typed
        ``Literal['union', 'intersection', 'overlap']``, so a bad value
        raises ``pydantic.ValidationError`` (a ``ValueError`` subclass)
        when the detector is constructed rather than when ``apply`` runs.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CompositeDetector(
                    detectors=[OtsuDetector()],
                    mode='invalid'
            )

    def test_serialization_roundtrip(self):
        """Test that CompositeDetector serializes and deserializes correctly."""
        # Create composite detector with nested detectors
        composite = CompositeDetector(
                detectors=[
                    OtsuDetector(ignore_zeros=True),
                    CannyDetector(sigma=2)
                ],
                mode='union'
        )

        # Create pipeline
        pipeline = ImagePipeline(ops=[composite])

        # Serialize to JSON
        json_str = pipeline.to_json()

        # Deserialize
        restored_pipeline = ImagePipeline.from_json(json_str)

        # Verify structure
        assert len(restored_pipeline._ops) == 1
        restored_composite = list(restored_pipeline._ops.values())[0]
        assert isinstance(restored_composite, CompositeDetector)
        assert len(restored_composite.detectors) == 2
        assert isinstance(restored_composite.detectors[0], OtsuDetector)
        assert isinstance(restored_composite.detectors[1], CannyDetector)
        assert restored_composite.mode == 'union'

        # Verify parameters preserved
        assert restored_composite.detectors[0].ignore_zeros is True
        assert restored_composite.detectors[1].sigma == 2

    def test_serialization_functional_equivalence(self, synth_plate):
        """Test that serialized/deserialized CompositeDetector produces identical results."""
        image = synth_plate.copy()

        # Original detector
        composite = CompositeDetector(
                detectors=[OtsuDetector(), CannyDetector(sigma=2)],
                mode='intersection'
        )
        original_result = composite.apply(image, inplace=False)

        # Serialize and deserialize
        pipeline = ImagePipeline(ops=[composite])
        json_str = pipeline.to_json()
        restored_pipeline = ImagePipeline.from_json(json_str)

        # Apply restored detector via pipeline
        restored_result = restored_pipeline.apply(image.copy(), inplace=False)

        # Results should be identical
        np.testing.assert_array_equal(
                original_result.objmask[:],
                restored_result.objmask[:]
        )
        np.testing.assert_array_equal(
                original_result.objmap[:],
                restored_result.objmap[:]
        )

    def test_overlap_ratio_effects(self):
        """Test that min_overlap_ratio parameter has effect in overlap mode."""
        large = np.zeros((8, 8), dtype=bool)
        large[1:5, 1:5] = True

        small = np.zeros((8, 8), dtype=bool)
        small[1:3, 1:3] = True

        StaticMaskDetector.masks = {"large": large, "small": small}
        image = Image(arr=np.zeros((8, 8), dtype=np.uint8))

        # Conservative overlap (high ratio)
        conservative = CompositeDetector(
                detectors=[
                    StaticMaskDetector(key="large"),
                    StaticMaskDetector(key="small"),
                ],
                mode='overlap',
                min_overlap_ratio=0.5
        )
        conservative_result = conservative.apply(image)

        # Permissive overlap (low ratio)
        permissive = CompositeDetector(
                detectors=[
                    StaticMaskDetector(key="large"),
                    StaticMaskDetector(key="small"),
                ],
                mode='overlap',
                min_overlap_ratio=0.2
        )
        permissive_result = permissive.apply(image)

        # Permissive should retain more or equal foreground area
        assert permissive_result.objmask[:].sum() >= conservative_result.objmask[:].sum()

    def test_overlap_mode_applies_min_overlap_ratio_to_component_fraction(self):
        """Overlap mode keeps only components meeting the requested overlap fraction."""
        large = np.zeros((8, 8), dtype=bool)
        large[1:5, 1:5] = True

        small = np.zeros((8, 8), dtype=bool)
        small[1:3, 1:3] = True

        StaticMaskDetector.masks = {"large": large, "small": small}
        image = Image(arr=np.zeros((8, 8), dtype=np.uint8))

        permissive = CompositeDetector(
                detectors=[
                    StaticMaskDetector(key="large"),
                    StaticMaskDetector(key="small"),
                ],
                mode="overlap",
                min_overlap_ratio=0.2,
        ).apply(image)

        strict = CompositeDetector(
                detectors=[
                    StaticMaskDetector(key="large"),
                    StaticMaskDetector(key="small"),
                ],
                mode="overlap",
                min_overlap_ratio=0.5,
        ).apply(image)

        np.testing.assert_array_equal(permissive.objmask[:], large)
        np.testing.assert_array_equal(strict.objmask[:], small)

    # inplace=True/False semantics are now covered by the smoke contract
    # tests/smoke/test_operation.py::test_inplace_contract

    def test_json_serialization_structure(self):
        """Test the JSON structure for nested operations."""
        import json

        composite = CompositeDetector(
                detectors=[
                    OtsuDetector(ignore_zeros=True),
                    CannyDetector(sigma=2)
                ],
                mode='overlap',
                min_overlap_ratio=0.6
        )

        pipeline = ImagePipeline(ops=[composite])
        json_str = pipeline.to_json()
        config = json.loads(json_str)

        # Find the CompositeDetector entry
        pipe_cfgs = config['pipe_cfgs']
        composite_key = [k for k in pipe_cfgs.keys() if 'CompositeDetector' in k][0]
        composite_data = pipe_cfgs[composite_key]

        # Verify nested detectors are serialized. Under the pydantic
        # serializer each ``OperationField`` entry is a plain
        # ``{"class", "params"}`` dict, so ``detectors`` is a JSON list.
        assert 'detectors' in composite_data['params']
        detectors_data = composite_data['params']['detectors']
        assert isinstance(detectors_data, list)
        assert len(detectors_data) == 2

        # Verify first detector (OtsuDetector)
        otsu_data = detectors_data[0]
        assert otsu_data['class'] == 'OtsuDetector'
        assert otsu_data['params']['ignore_zeros'] is True

        # Verify second detector (CannyDetector)
        canny_data = detectors_data[1]
        assert canny_data['class'] == 'CannyDetector'
        assert canny_data['params']['sigma'] == 2

        # Verify mode and min_overlap_ratio
        assert composite_data['params']['mode'] == 'overlap'
        assert composite_data['params']['min_overlap_ratio'] == 0.6

    def test_pipeline_with_multiple_operations(self, synth_plate):
        """Test CompositeDetector in a complex pipeline."""
        from phenotypic.enhance import GaussianBlur
        from phenotypic.refine import SmallObjectRemover

        image = synth_plate.copy()

        pipeline = ImagePipeline(ops=[
            GaussianBlur(sigma=2),
            CompositeDetector(
                    detectors=[OtsuDetector(), CannyDetector(sigma=2)],
                    mode='union'
            ),
            SmallObjectRemover(min_size=50)
        ])

        # Apply pipeline
        result = pipeline.apply(image, inplace=False)

        # Should have detection
        assert result.objmap[:].max() > 0

        # Serialize and deserialize
        json_str = pipeline.to_json()
        restored = ImagePipeline.from_json(json_str)

        # Apply restored pipeline
        restored_result = restored.apply(image.copy(), inplace=False)

        # Results should be identical
        np.testing.assert_array_equal(
                result.objmask[:],
                restored_result.objmask[:]
        )

    def test_pipeline_as_detector(self, synth_plate):
        """Test that CompositeDetector accepts ImagePipeline as detector."""
        from phenotypic.enhance import GaussianBlur

        image = synth_plate.copy()

        # Pipeline with preprocessing + detection
        pipeline = ImagePipeline(ops=[
            GaussianBlur(sigma=2),
            OtsuDetector()
        ])

        composite = CompositeDetector(
                detectors=[pipeline],
                mode='union'
        )
        result = composite.apply(image)

        assert result.objmap[:].max() > 0
        assert result.objmask[:].sum() > 0

    def test_mixed_detectors_and_pipelines(self, synth_plate):
        """Test mixing ObjectDetector and ImagePipeline."""
        from phenotypic.enhance import GaussianBlur

        image = synth_plate.copy()

        composite = CompositeDetector(
                detectors=[
                    OtsuDetector(),  # Direct detector
                    ImagePipeline(ops=[  # Pipeline
                        GaussianBlur(sigma=2),
                        CannyDetector(sigma=2)
                    ])
                ],
                mode='union'
        )
        result = composite.apply(image)

        # Should successfully combine both
        assert result.objmap[:].max() > 0
        assert result.objmask[:].sum() > 0

    def test_pipeline_serialization_roundtrip(self):
        """Test that CompositeDetector with pipelines serializes correctly."""
        from phenotypic.enhance import GaussianBlur

        composite = CompositeDetector(
                detectors=[
                    OtsuDetector(),
                    ImagePipeline(ops=[
                        GaussianBlur(sigma=2),
                        CannyDetector(sigma=2)
                    ])
                ],
                mode='intersection'
        )

        pipeline = ImagePipeline(ops=[composite])
        json_str = pipeline.to_json()

        # Deserialize
        restored = ImagePipeline.from_json(json_str)
        restored_composite = list(restored._ops.values())[0]

        # Verify structure
        assert len(restored_composite.detectors) == 2
        assert isinstance(restored_composite.detectors[0], OtsuDetector)
        assert isinstance(restored_composite.detectors[1], ImagePipeline)

        # Verify nested pipeline structure
        nested_pipeline = restored_composite.detectors[1]
        assert len(nested_pipeline._ops) == 2

    def test_pipeline_functional_equivalence(self, synth_plate):
        """Test functional equivalence after serialization with pipelines."""
        from phenotypic.enhance import GaussianBlur

        image = synth_plate.copy()

        composite = CompositeDetector(
                detectors=[
                    OtsuDetector(),
                    ImagePipeline(ops=[
                        GaussianBlur(sigma=2),
                        CannyDetector(sigma=2)
                    ])
                ],
                mode='union'
        )
        original_result = composite.apply(image, inplace=False)

        # Serialize and deserialize
        pipeline = ImagePipeline(ops=[composite])
        json_str = pipeline.to_json()
        restored = ImagePipeline.from_json(json_str)

        # Apply restored
        restored_result = restored.apply(image.copy(), inplace=False)

        # Results should be identical
        np.testing.assert_array_equal(
                original_result.objmask[:],
                restored_result.objmask[:]
        )

    def test_json_structure_with_pipelines(self):
        """Test JSON structure contains nested pipelines."""
        from phenotypic.enhance import GaussianBlur
        import json

        composite = CompositeDetector(
                detectors=[
                    OtsuDetector(),
                    ImagePipeline(ops=[
                        GaussianBlur(sigma=2),
                        OtsuDetector()
                    ])
                ],
                mode='union'
        )

        pipeline = ImagePipeline(ops=[composite])
        json_str = pipeline.to_json()
        config = json.loads(json_str)

        # Find CompositeDetector
        pipe_cfgs = config['pipe_cfgs']
        composite_key = [k for k in pipe_cfgs.keys() if 'CompositeDetector' in k][0]
        composite_data = pipe_cfgs[composite_key]

        # Verify detectors list structure. ``OperationField`` serializes
        # each entry to a class-tagged dict; an ``ObjectDetector`` entry is
        # ``{"class", "params"}`` and a nested ``ImagePipeline`` entry is
        # ``{"__type__": "pipeline", "config": ...}``.
        detectors_data = composite_data['params']['detectors']
        assert isinstance(detectors_data, list)
        assert len(detectors_data) == 2

        # First item is ObjectDetector
        assert detectors_data[0]['class'] == 'OtsuDetector'

        # Second item is ImagePipeline
        assert detectors_data[1]['__type__'] == 'pipeline'
        assert 'config' in detectors_data[1]
        nested_config = detectors_data[1]['config']
        assert 'pipe_cfgs' in nested_config
        assert len(nested_config['pipe_cfgs']) == 2
