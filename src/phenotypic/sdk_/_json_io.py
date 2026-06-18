from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union


def read_json_source(json_data: Union[str, Path, dict]) -> Any:
    """Coerce a JSON source to a Python object.

    Shared front-door for the ``from_json`` classmethods on operations,
    pipelines, and profiles so they accept the same inputs. Mirrors
    :meth:`SerializablePipeline.from_json` input handling: a ``dict`` passes
    through unchanged; a ``str``/``Path`` is read as a file when it looks like a
    path and exists on disk (the ``< 256`` length guard avoids ``stat``-ing long
    JSON strings); otherwise the value is parsed as a JSON string.

    Args:
        json_data: A pre-parsed dict, a path to a JSON file, or a JSON string.

    Returns:
        The parsed Python object (typically a dict).

    Raises:
        ValueError: If the value is not a dict and cannot be parsed as JSON.

    Example:
        >>> read_json_source('{"a": 1}')
        {'a': 1}
        >>> read_json_source({"a": 1})
        {'a': 1}
    """
    if isinstance(json_data, dict):
        return json_data
    text = str(json_data)
    try:
        path = Path(json_data)
        # Only read as a file if it looks like a path and exists. The length
        # guard prevents stat-ing very long JSON strings.
        if len(text) < 256 and path.exists() and path.is_file():
            text = path.read_text()
    except (OSError, ValueError):
        # If Path operations fail, fall through and treat as a JSON string.
        pass
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {e}")
