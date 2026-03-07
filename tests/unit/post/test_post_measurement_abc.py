"""Tests for PostMeasurement abstract base class."""

import pytest
import pandas as pd

from phenotypic.abc_ import PostMeasurement


class TestPostMeasurementABC:
    """Tests for PostMeasurement ABC."""

    def test_cannot_instantiate_directly(self):
        """PostMeasurement is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            PostMeasurement()

    def test_concrete_subclass_works(self):
        """A concrete subclass with _operate implemented can be instantiated."""

        class DummyPost(PostMeasurement):
            def _operate(self, df: pd.DataFrame) -> pd.DataFrame:
                return df

        post = DummyPost()
        assert post is not None

    def test_apply_calls_operate(self):
        """apply() delegates to _operate()."""

        class DummyPost(PostMeasurement):
            def _operate(self, df: pd.DataFrame) -> pd.DataFrame:
                df["added"] = 1
                return df

        post = DummyPost()
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = post.apply(df)
        assert "added" in result.columns

    def test_has_logger(self):
        """PostMeasurement inherits BaseOperation logging."""

        class DummyPost(PostMeasurement):
            def _operate(self, df: pd.DataFrame) -> pd.DataFrame:
                return df

        post = DummyPost()
        assert hasattr(post, "_logger")
