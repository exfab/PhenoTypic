"""Tests for post-measurement serialization in ImagePipeline."""

import json


from phenotypic import ImagePipeline
from phenotypic.post import ExpandMetadata, MergeMetadata


class TestPostSerialization:
    """Test that post transforms serialize and deserialize."""

    def test_post_appears_in_json(self):
        """Post transforms are included in JSON output."""
        pipe = ImagePipeline(
            post=[ExpandMetadata(column="ImageName", labels=["A", "B"], delimiter="_")],
        )
        json_str = pipe.to_json()
        config = json.loads(json_str)
        assert "post" in config
        assert "ExpandMetadata" in config["post"]

    def test_roundtrip_expand_metadata(self):
        """ExpandMetadata survives JSON roundtrip."""
        pipe = ImagePipeline(
            post=[ExpandMetadata(column="ImageName", labels=["A", "B"], delimiter="_")],
        )
        json_str = pipe.to_json()
        loaded = ImagePipeline.from_json(json_str)
        assert len(loaded._post) == 1
        post_op = list(loaded._post.values())[0]
        assert isinstance(post_op, ExpandMetadata)
        assert post_op.column == "MetadataImage_ImageName"
        assert post_op.labels == ["Metadata_A", "Metadata_B"]
        assert post_op.delimiter == "_"

    def test_roundtrip_merge_metadata(self):
        """MergeMetadata survives JSON roundtrip."""
        pipe = ImagePipeline(
            post=[MergeMetadata(columns=["A", "B"], label="AB", delimiter="-")],
        )
        json_str = pipe.to_json()
        loaded = ImagePipeline.from_json(json_str)
        assert len(loaded._post) == 1
        post_op = list(loaded._post.values())[0]
        assert isinstance(post_op, MergeMetadata)
        assert post_op.columns == ["Metadata_A", "Metadata_B"]
        assert post_op.label == "Metadata_AB"
        assert post_op.delimiter == "-"

    def test_empty_post_roundtrip(self):
        """Pipeline with no post transforms roundtrips correctly."""
        pipe = ImagePipeline()
        json_str = pipe.to_json()
        loaded = ImagePipeline.from_json(json_str)
        assert len(loaded._post) == 0

    def test_backward_compatible_json_without_post(self):
        """Loading JSON from older version without 'post' key works."""
        old_json = json.dumps({
            "version": "0.0.0",
            "name": "test",
            "desc": None,
            "reset": False,
            "pipe_cfgs": {},
            "meas": {},
        })
        loaded = ImagePipeline.from_json(old_json)
        assert len(loaded._post) == 0
