"""Sweep and Fixed classes for parameter sweep specification."""

from __future__ import annotations

import inspect
from typing import Any, Dict


class Fixed:
    """Wrapper to explicitly mark a parameter value as fixed (not swept).

    Use ``Fixed`` when you need to pass a tuple as a literal fixed value
    rather than having it interpreted as a set of values to sweep over.

    Args:
        value: The parameter value to hold fixed.

    Examples:
        >>> from phenotypic.sweep import Sweep, Fixed
        >>> from phenotypic.enhance import GaussianBlur
        >>> # Without Fixed, a tuple is swept:
        >>> Sweep(GaussianBlur, sigma=(1.0, 2.0))  # sweeps sigma over 1.0 and 2.0
        >>> # With Fixed, the tuple is passed as-is:
        >>> Sweep(GaussianBlur, sigma=Fixed((1.0, 2.0)))  # sigma = (1.0, 2.0)
    """

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"Fixed({self.value!r})"


class Sweep:
    """Specification for an operation in a parameter sweep.

    Defines which operation class to instantiate and which parameters to vary
    (sweep) vs. hold constant (fix) across a cartesian product of combinations.

    Args:
        operation_class: The operation **class** (not an instance). Must be a
            class with an ``__init__`` accepting the provided parameter names.
        **params: Keyword arguments specifying parameter values.

            - **Tuple** values are swept (each element becomes one combination).
            - **Scalar** values (int, float, str, bool, None) are fixed.
            - **List** values are fixed (passed as-is, not swept).
            - **``Fixed(value)``** explicitly marks any value as fixed.

    Raises:
        TypeError: If ``operation_class`` is an instance rather than a class.
        ValueError: If a parameter name does not exist in the operation
            class constructor.

    Examples:
        >>> from phenotypic.sweep import Sweep, Fixed
        >>> from phenotypic.enhance import GaussianBlur
        >>> from phenotypic.detect import OtsuDetector
        >>> # Sweep sigma over 3 values, fix truncate
        >>> s = Sweep(GaussianBlur, sigma=(1.0, 1.5, 2.0), truncate=4.0)
        >>> s.sweep_params
        {'sigma': [1.0, 1.5, 2.0]}
        >>> s.fixed_params
        {'truncate': 4.0}
    """

    def __init__(self, operation_class: type, **params: Any) -> None:
        if not isinstance(operation_class, type):
            raise TypeError(
                f"operation_class must be a class, not an instance. "
                f"Got {type(operation_class).__name__}: {operation_class!r}. "
                f"Pass the class itself (e.g., GaussianBlur) not an instance "
                f"(e.g., GaussianBlur())."
            )

        self.operation_class: type = operation_class

        # Validate parameter names against the constructor signature
        self._validate_param_names(params)

        # Classify parameters
        self.sweep_params: Dict[str, list] = {}
        self.fixed_params: Dict[str, Any] = {}

        for key, value in params.items():
            if isinstance(value, Fixed):
                self.fixed_params[key] = value.value
            elif isinstance(value, tuple):
                self.sweep_params[key] = list(value)
            else:
                # scalar, list, or any other type → fixed
                self.fixed_params[key] = value

    def _validate_param_names(self, params: Dict[str, Any]) -> None:
        """Check that all param names exist in the operation class constructor.

        Args:
            params: Parameter names to validate.

        Raises:
            ValueError: If a parameter name is not accepted by the constructor.
        """
        try:
            sig = inspect.signature(self.operation_class.__init__)
        except (ValueError, TypeError):
            # If we can't inspect the signature, skip validation
            return

        valid_params = set(sig.parameters.keys()) - {"self"}
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )

        if has_var_keyword:
            # Constructor accepts **kwargs, so any param name is valid
            return

        for name in params:
            if name not in valid_params:
                raise ValueError(
                    f"{self.operation_class.__name__}.__init__() has no "
                    f"parameter '{name}'. Valid parameters: "
                    f"{sorted(valid_params)}"
                )

    def __repr__(self) -> str:
        parts = [self.operation_class.__name__]
        for k, v in self.sweep_params.items():
            parts.append(f"{k}={tuple(v)!r}")
        for k, v in self.fixed_params.items():
            parts.append(f"{k}={v!r}")
        return f"Sweep({', '.join(parts)})"
