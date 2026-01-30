"""Parameter sweep specification for pipeline exploration.

Supports multiple sweep types:
- Numeric ranges (start, stop, step)
- Categorical lists (discrete values)
- Nested operation sweeps (swap entire operations)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
import itertools
import json

import numpy as np


@dataclass
class SweepSpec:
    """Parameter sweep specification supporting multiple data types.

    Args:
        param: Name of the parameter to sweep. Use '__operation__' to swap
            the entire operation instance.
        values: List of values to sweep over. Can be numeric, string,
            or ImageOperation instances.

    Examples:
        Numeric range sweep:

        >>> spec = SweepSpec.from_range('sigma', 1.0, 3.0, 0.5)
        >>> spec.values
        [1.0, 1.5, 2.0, 2.5, 3.0]

        Categorical list:

        >>> spec = SweepSpec('shape', ['disk', 'square', 'diamond'])
        >>> spec.count
        3

        Nested operation sweep:

        >>> from phenotypic.enhance import GaussianBlur, MedianFilter
        >>> spec = SweepSpec('__operation__', [
        ...     GaussianBlur(sigma=1.0),
        ...     MedianFilter(size=3),
        ... ])
    """

    param: str
    values: List[Any]

    @classmethod
    def from_range(
        cls,
        param: str,
        start: float,
        stop: float,
        step: float,
    ) -> "SweepSpec":
        """Create sweep from numeric range.

        Args:
            param: Parameter name to sweep.
            start: Start value (inclusive).
            stop: Stop value (inclusive).
            step: Step size between values.

        Returns:
            SweepSpec with generated range values.

        Examples:
            >>> spec = SweepSpec.from_range('sigma', 1.0, 2.0, 0.5)
            >>> spec.values
            [1.0, 1.5, 2.0]
        """
        # Use linspace-like approach to include stop value
        n_steps = int(round((stop - start) / step)) + 1
        values = [start + i * step for i in range(n_steps)]
        # Round to avoid floating point artifacts
        values = [round(v, 10) for v in values]
        return cls(param=param, values=values)

    @classmethod
    def from_linspace(
        cls,
        param: str,
        start: float,
        stop: float,
        num: int,
    ) -> "SweepSpec":
        """Create sweep from linear spacing.

        Args:
            param: Parameter name to sweep.
            start: Start value (inclusive).
            stop: Stop value (inclusive).
            num: Number of values to generate.

        Returns:
            SweepSpec with linearly spaced values.

        Examples:
            >>> spec = SweepSpec.from_linspace('sigma', 1.0, 3.0, 5)
            >>> spec.values
            [1.0, 1.5, 2.0, 2.5, 3.0]
        """
        values = list(np.linspace(start, stop, num))
        return cls(param=param, values=values)

    @classmethod
    def from_logspace(
        cls,
        param: str,
        start: float,
        stop: float,
        num: int,
        base: float = 10.0,
    ) -> "SweepSpec":
        """Create sweep from logarithmic spacing.

        Args:
            param: Parameter name to sweep.
            start: Start exponent (10^start).
            stop: Stop exponent (10^stop).
            num: Number of values to generate.
            base: Base of the logarithm.

        Returns:
            SweepSpec with logarithmically spaced values.

        Examples:
            >>> spec = SweepSpec.from_logspace('threshold', 1, 3, 3)
            >>> spec.values  # [10, 100, 1000]
            [10.0, 100.0, 1000.0]
        """
        values = list(np.logspace(start, stop, num, base=base))
        return cls(param=param, values=values)

    @property
    def count(self) -> int:
        """Number of values in sweep."""
        return len(self.values)

    @property
    def is_operation_sweep(self) -> bool:
        """Check if this sweeps entire operations."""
        return self.param == "__operation__"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.

        Note:
            Operation sweeps cannot be fully serialized - only their
            class names and parameters are stored.
        """
        if self.is_operation_sweep:
            # Serialize operations as class + params
            serialized_values = []
            for op in self.values:
                op_dict = {
                    "__class__": f"{op.__class__.__module__}.{op.__class__.__name__}",
                    "__params__": self._get_operation_params(op),
                }
                serialized_values.append(op_dict)
            return {"param": self.param, "values": serialized_values}
        return {"param": self.param, "values": self.values}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SweepSpec":
        """Create from dictionary.

        Args:
            data: Dictionary with 'param' and 'values' keys.

        Returns:
            Reconstructed SweepSpec.

        Note:
            Operation sweeps will have their operations reconstructed
            if the classes are importable.
        """
        param = data["param"]
        values = data["values"]

        if param == "__operation__" and values and isinstance(values[0], dict):
            # Reconstruct operations
            reconstructed = []
            for v in values:
                if "__class__" in v:
                    op = cls._reconstruct_operation(v)
                    reconstructed.append(op)
                else:
                    reconstructed.append(v)
            values = reconstructed

        return cls(param=param, values=values)

    @staticmethod
    def _get_operation_params(op: Any) -> Dict[str, Any]:
        """Extract serializable parameters from an operation."""
        # Get params from __init__ signature that are set as attributes
        params = {}
        if hasattr(op, "__dict__"):
            for key, value in op.__dict__.items():
                if not key.startswith("_"):
                    # Only include JSON-serializable values
                    try:
                        json.dumps(value)
                        params[key] = value
                    except (TypeError, ValueError):
                        pass
        return params

    @staticmethod
    def _reconstruct_operation(data: Dict[str, Any]) -> Any:
        """Reconstruct an operation from serialized data."""
        import importlib

        class_path = data["__class__"]
        params = data.get("__params__", {})

        # Import the class
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        op_class = getattr(module, class_name)

        return op_class(**params)

    def __repr__(self) -> str:
        if self.count <= 5:
            return f"SweepSpec({self.param!r}, {self.values!r})"
        return f"SweepSpec({self.param!r}, [{self.values[0]!r}, ..., {self.values[-1]!r}] ({self.count} values))"


def expand_sweep_combinations(
    sweeps: List[SweepSpec],
) -> List[Dict[str, Any]]:
    """Expand multiple sweeps into all combinations.

    Args:
        sweeps: List of SweepSpec objects to combine.

    Returns:
        List of dictionaries, each containing one combination of
        parameter values.

    Examples:
        >>> sweeps = [
        ...     SweepSpec('sigma', [1.0, 2.0]),
        ...     SweepSpec('threshold', [50, 100]),
        ... ]
        >>> combos = expand_sweep_combinations(sweeps)
        >>> len(combos)
        4
        >>> combos[0]
        {'sigma': 1.0, 'threshold': 50}
    """
    if not sweeps:
        return [{}]

    # Build lists for cartesian product
    param_names = [s.param for s in sweeps]
    value_lists = [s.values for s in sweeps]

    # Generate all combinations
    combinations = []
    for combo in itertools.product(*value_lists):
        combinations.append(dict(zip(param_names, combo)))

    return combinations


def count_sweep_combinations(sweeps: List[SweepSpec]) -> int:
    """Count total combinations without generating them.

    Args:
        sweeps: List of SweepSpec objects.

    Returns:
        Total number of combinations.
    """
    if not sweeps:
        return 1
    total = 1
    for sweep in sweeps:
        total *= sweep.count
    return total
