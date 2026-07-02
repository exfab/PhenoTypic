"""Tests for post-measurement integration in ImagePipeline."""

import pytest

from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureShape
from phenotypic.post import ExpandMetadata, MergeMetadata
from phenotypic.abc_._post_measurement import PostMeasurement


class TestPipelinePostParameter:
    """Test that ImagePipeline accepts and stores post transforms."""

    def test_accepts_post_list(self):
        """Pipeline accepts a list of PostMeasurement objects."""

        class DummyPost(PostMeasurement):
            def _operate(self, df):
                return df

        pipe = ImagePipeline(post=[DummyPost()])
        assert len(pipe._post) == 1

    def test_accepts_post_dict(self):
        """Pipeline accepts a dict of PostMeasurement objects."""

        class DummyPost(PostMeasurement):
            def _operate(self, df):
                return df

        pipe = ImagePipeline(post={"my_post": DummyPost()})
        assert "my_post" in pipe._post

    def test_default_post_is_empty(self):
        """Pipeline with no post parameter has empty _post dict."""
        pipe = ImagePipeline()
        assert len(pipe._post) == 0


class TestPipelinePostExecution:
    """Test that post transforms run after measurements."""

    @pytest.fixture(scope="class")
    def sample_image(self):
        """Load and detect colonies on synth yeast plate."""
        image = load_synth_yeast_plate()
        return OtsuDetector().apply(image)

    def test_expand_metadata_in_pipeline(self, sample_image):
        """ExpandMetadata runs as part of pipeline.measure()."""
        sample_image.metadata["Condition"] = "WT_30C"
        pipe = ImagePipeline(
            meas=[MeasureShape()],
            post=[ExpandMetadata(column="Condition", labels=["Strain", "Temp"], delimiter="_")],
        )
        df = pipe.measure(sample_image)
        assert "MetadataGenetic_Strain" in df.columns
        assert "Metadata_Temp" in df.columns

    def test_post_transforms_run_in_order(self, sample_image):
        """Multiple post transforms execute sequentially."""
        sample_image.metadata["Condition"] = "WT_30C"
        pipe = ImagePipeline(
            meas=[MeasureShape()],
            post=[
                ExpandMetadata(column="Condition", labels=["Strain", "Temp"], delimiter="_"),
                MergeMetadata(columns=["Strain", "Temp"], label="Recombined", delimiter="-"),
            ],
        )
        df = pipe.measure(sample_image)
        assert "Metadata_Recombined" in df.columns
        assert df["Metadata_Recombined"].iloc[0] == "WT-30C"

    def test_apply_and_measure_runs_post(self, sample_image):
        """Post transforms also run via apply_and_measure()."""
        sample_image.metadata["Condition"] = "WT_30C"
        pipe = ImagePipeline(
            ops=[OtsuDetector()],
            meas=[MeasureShape()],
            post=[ExpandMetadata(column="Condition", labels=["Strain", "Temp"], delimiter="_")],
        )
        df = pipe.apply_and_measure(sample_image)
        assert "MetadataGenetic_Strain" in df.columns

    def test_measure_apply_post_false_skips_post(self, sample_image):
        """measure(apply_post=False) returns the merged frame before post runs."""
        sample_image.metadata["Condition"] = "WT_30C"
        pipe = ImagePipeline(
            meas=[MeasureShape()],
            post=[ExpandMetadata(column="Condition", labels=["Strain", "Temp"], delimiter="_")],
        )
        df_clean = pipe.measure(sample_image, apply_post=False)
        df_post = pipe.measure(sample_image)

        # Post added the split columns; the clean frame is missing them.
        assert "MetadataGenetic_Strain" not in df_clean.columns
        assert "Metadata_Temp" not in df_clean.columns
        assert "MetadataGenetic_Strain" in df_post.columns
        assert "Metadata_Temp" in df_post.columns
        # Original Condition column survives in both frames.
        assert "Metadata_Condition" in df_clean.columns

    def test_apply_and_measure_apply_post_false_skips_post(self, sample_image):
        """apply_and_measure(apply_post=False) forwards the flag to measure()."""
        sample_image.metadata["Condition"] = "WT_30C"
        pipe = ImagePipeline(
            ops=[OtsuDetector()],
            meas=[MeasureShape()],
            post=[ExpandMetadata(column="Condition", labels=["Strain", "Temp"], delimiter="_")],
        )
        df = pipe.apply_and_measure(sample_image, apply_post=False)
        assert "MetadataGenetic_Strain" not in df.columns

    def test_measure_apply_post_default_true(self, sample_image):
        """Default measure() still applies post (no behavior change for callers)."""
        sample_image.metadata["Condition"] = "WT_30C"
        pipe = ImagePipeline(
            meas=[MeasureShape()],
            post=[ExpandMetadata(column="Condition", labels=["Strain", "Temp"], delimiter="_")],
        )
        df = pipe.measure(sample_image)
        assert "MetadataGenetic_Strain" in df.columns
