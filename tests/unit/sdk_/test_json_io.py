"""Unit tests for ``phenotypic.sdk_._json_io.read_json_source``."""

from __future__ import annotations

import json

import pytest

from phenotypic.sdk_._json_io import read_json_source


def test_dict_passthrough():
    """A dict is returned unchanged (identity)."""
    payload = {"class": "OtsuDetector", "params": {"ignore_zeros": True}}
    assert read_json_source(payload) is payload


def test_parses_json_string():
    """A JSON string is parsed to a Python object."""
    assert read_json_source('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_reads_existing_file(tmp_path):
    """An existing file path is read and parsed."""
    filepath = tmp_path / "data.json"
    filepath.write_text(json.dumps({"degree": 3}))
    assert read_json_source(filepath) == {"degree": 3}


def test_reads_existing_file_from_str_path(tmp_path):
    """A str pointing at an existing file is read and parsed."""
    filepath = tmp_path / "data.json"
    filepath.write_text(json.dumps({"degree": 4}))
    assert read_json_source(str(filepath)) == {"degree": 4}


def test_invalid_json_raises_value_error():
    """A non-JSON, non-path string raises ValueError."""
    with pytest.raises(ValueError, match="Invalid JSON data"):
        read_json_source("not valid json {")


def test_nonexistent_path_string_treated_as_json():
    """A short string that is not an existing file is parsed as JSON, not stat-looped."""
    # Valid JSON scalar that also is not a file → parsed as JSON.
    assert read_json_source("42") == 42
